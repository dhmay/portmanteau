"""Finding pairs of equal-length sequences within a small Hamming distance.

The core routine buckets sequences by a masked copy of themselves: for every combination of
`distance` positions, the sequence with those positions removed is a key, and any two sequences
sharing a key differ only within those positions. Checking that they differ at *all* of them gives
exactly-`distance` pairs, and means each pair surfaces from exactly one bucket, so no
deduplication state is needed.

**Pair counts grow quadratically with the number of sequences.** Doubling the input roughly
quadruples the output, which is easy to underestimate: a dataset ten times larger yields a hundred
times the pairs. `group_cols` is the main lever against this, because finding pairs within each of
k groups costs `k * (n/k)^2` rather than `n^2`. For sequences that carry a natural partition - one
subject, one sample, one repertoire - grouping on it is both cheaper and usually the right
question. Where the output is still too large to hold, `iter_hamming_pairs` streams it in chunks.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Iterator, Sequence

import pandas as pd

DEFAULT_CHUNK_SIZE = 1_000_000


def find_hamming_pairs(
    pdf: pd.DataFrame,
    seq_col: str,
    *,
    distance: int = 1,
    exact: bool = True,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Find all pairs of rows whose sequences are within `distance` of each other.

    Rows whose sequences differ in length are never paired. Rows with identical sequences are not
    paired with each other, but each is paired with everything either of them is close to.

    Args:
        pdf: Input dataframe.
        seq_col: Column holding the sequences. Required: there is no safe default, and passing a
            column that concatenates a sequence with other fields silently compares those fields
            too.
        distance: Hamming distance to look for.
        exact: If True (the default), return only pairs at exactly `distance`. If False, also
            return closer pairs, and include a ``hamming_distance`` column. At distance 1 the two
            are the same, since identical sequences are never paired with each other.
        group_cols: Columns that must match for two rows to be paired. None compares every row
            against every other.
        annotation_columns: Columns of `pdf` to carry onto both sides of each pair, as by
            annotate_pairs. None adds nothing.
        max_pairs: Raise rather than return more than this many pairs. A guard against the
            quadratic growth described in the module docstring; None means no limit.

    Returns:
        One row per pair, with columns:

        - ``row_i``, ``row_j``: index labels of the two rows in `pdf`
        - ``{seq_col}_i``, ``{seq_col}_j``: their sequences
        - ``seq_length``: the length the two sequences share
        - ``differing_positions``: tuple of the 0-based positions at which they differ
        - ``hamming_distance``: only when `exact` is False, where it varies
        - ``{c}_i``, ``{c}_j`` for each column in `annotation_columns`

    Raises:
        ValueError: if `distance` is below 1, if `seq_col` is missing or has null values, or if
            `max_pairs` is exceeded.
    """
    chunks = []
    total = 0
    for chunk in iter_hamming_pairs(
        pdf,
        seq_col,
        distance=distance,
        exact=exact,
        group_cols=group_cols,
        annotation_columns=annotation_columns,
    ):
        total += len(chunk)
        if max_pairs is not None and total > max_pairs:
            raise ValueError(
                f"more than max_pairs={max_pairs:,} pairs; pair counts grow quadratically with "
                "input size. Narrow the search with group_cols, or stream the result with "
                "iter_hamming_pairs."
            )
        chunks.append(chunk)
    return chunks[0] if len(chunks) == 1 else pd.concat(chunks, ignore_index=True)


def find_hamming1_pairs(
    pdf: pd.DataFrame,
    seq_col: str,
    *,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Find pairs of rows whose sequences differ at exactly one position.

    A convenience wrapper over find_hamming_pairs for the common case, differing only in that the
    single differing position is reported as a scalar ``differing_position`` rather than as a
    one-element ``differing_positions`` tuple.
    """
    pdf_pairs = find_hamming_pairs(
        pdf,
        seq_col,
        distance=1,
        group_cols=group_cols,
        annotation_columns=annotation_columns,
        max_pairs=max_pairs,
    )
    # replace in place rather than appending, so the column keeps its position in the frame
    pdf_pairs["differing_positions"] = [positions[0] for positions in pdf_pairs.differing_positions]
    return pdf_pairs.rename(columns={"differing_positions": "differing_position"})


def annotate_pairs(
    pdf_pairs: pd.DataFrame,
    pdf_source: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Add columns from the source dataframe to each side of a pairs dataframe.

    For each column ``c``, adds ``c_i`` and ``c_j``: the values for the two rows of the pair.
    Existing ``c_i``/``c_j`` columns, such as the sequence columns, are replaced.

    Args:
        pdf_pairs: Pairs with ``row_i`` and ``row_j`` columns holding index labels of pdf_source,
            as returned by find_hamming_pairs.
        pdf_source: The dataframe the pairs were found in. Its index must be unique.
        columns: Columns of pdf_source to add.

    Returns:
        pdf_pairs with the additional columns, reindexed from 0.

    Raises:
        ValueError: if pdf_source's index is not unique, so row labels cannot be looked up.
    """
    if not pdf_source.index.is_unique:
        raise ValueError("pdf_source must have a unique index to look up row_i/row_j")

    columns = list(columns)
    pdf_annotated = pdf_pairs.drop(
        columns=[f"{c}_{side}" for c in columns for side in ("i", "j")], errors="ignore"
    ).reset_index(drop=True)
    for side in ("i", "j"):
        pdf_side = pdf_source.loc[pdf_annotated[f"row_{side}"], columns].reset_index(drop=True)
        pdf_side.columns = [f"{c}_{side}" for c in columns]
        pdf_annotated = pd.concat([pdf_annotated, pdf_side], axis=1)
    return pdf_annotated


def iter_hamming_pairs(
    pdf: pd.DataFrame,
    seq_col: str,
    *,
    distance: int = 1,
    exact: bool = True,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield pairs of rows whose sequences are within `distance`, in chunks.

    Use this instead of find_hamming_pairs when the full result may not fit in memory; see the
    module docstring on how quickly pair counts grow. Arguments are as for find_hamming_pairs,
    except that chunk_size sets the approximate number of pairs per yielded frame. Annotation is
    applied per chunk, which costs the same in total as annotating once at the end.

    Yields:
        DataFrames with the columns described in find_hamming_pairs. At least one frame is always
        yielded, so the schema is available even when there are no pairs.
    """
    if distance < 1:
        raise ValueError(f"distance must be at least 1, got {distance}")

    columns = _pair_columns(seq_col, exact=exact)
    buffer: list[tuple] = []
    yielded = False

    def to_frame(rows: list[tuple]) -> pd.DataFrame:
        pdf_chunk = pd.DataFrame(rows, columns=columns)
        if annotation_columns:
            pdf_chunk = annotate_pairs(pdf_chunk, pdf, annotation_columns)
        return pdf_chunk

    groups: Iterable[pd.DataFrame]
    if group_cols is None:
        groups = [pdf]
    else:
        groups = (group for _, group in pdf.groupby(list(group_cols), observed=True, sort=False))

    for group in groups:
        for row in _group_pairs(group, seq_col, distance=distance, exact=exact):
            buffer.append(row)
            if len(buffer) >= chunk_size:
                yield to_frame(buffer)
                buffer, yielded = [], True
    if buffer or not yielded:
        yield to_frame(buffer)


def _pair_columns(seq_col: str, *, exact: bool) -> list[str]:
    columns = ["row_i", "row_j", f"{seq_col}_i", f"{seq_col}_j", "seq_length", "differing_positions"]
    return columns if exact else columns + ["hamming_distance"]


def _group_pairs(
    pdf: pd.DataFrame, seq_col: str, *, distance: int, exact: bool
) -> Iterator[tuple]:
    """Yield pair rows within one group, one sequence length at a time."""
    if seq_col not in pdf.columns:
        raise ValueError(f"Input dataframe must contain a '{seq_col}' column.")
    if pdf.empty:
        return
    sequences = pdf[seq_col]
    if sequences.isna().any():
        raise ValueError(f"Input dataframe contains null values in '{seq_col}'.")

    sequences = sequences.astype(str)
    row_labels = pdf.index.to_list()
    # distinct sequences are what we mask; rows sharing a sequence are expanded at the end
    rows_by_length: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, sequence in enumerate(sequences.to_list()):
        rows_by_length[len(sequence)][sequence].append(position)

    for seq_length, rows_by_sequence in rows_by_length.items():
        if seq_length <= distance or len(rows_by_sequence) < 2:
            continue
        unique_sequences = list(rows_by_sequence)
        for id_i, id_j, positions in _iter_close_sequence_pairs(
            unique_sequences, distance=distance, exact=exact
        ):
            seq_i, seq_j = unique_sequences[id_i], unique_sequences[id_j]
            for row_i in rows_by_sequence[seq_i]:
                for row_j in rows_by_sequence[seq_j]:
                    row = (row_labels[row_i], row_labels[row_j], seq_i, seq_j, seq_length, positions)
                    yield row if exact else row + (len(positions),)


def _iter_close_sequence_pairs(
    sequences: list[str], *, distance: int, exact: bool
) -> Iterable[tuple[int, int, tuple[int, ...]]]:
    """Yield (id_i, id_j, differing positions) for distinct equal-length sequences.

    Buckets sequences on every combination of `distance` masked positions. Two sequences sharing a
    bucket differ only within those positions, so their true differing positions are found by
    comparing just those.

    Each pair is yielded once. When `exact`, a pair is emitted from the single bucket whose mask is
    its differing positions. Otherwise a pair differing at fewer positions would appear in several
    buckets, so it is emitted only from the canonical one: its differing positions padded with the
    lowest unused positions. That keeps the pass stateless either way - no set of seen pairs, which
    at these counts would dominate memory.
    """
    if not sequences:
        return
    seq_len = len(sequences[0])

    masked_buckets: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(list)
    for seq_id, sequence in enumerate(sequences):
        for positions in combinations(range(seq_len), distance):
            masked = "".join(c for i, c in enumerate(sequence) if i not in positions)
            masked_buckets[(positions, masked)].append(seq_id)

    for (positions, _masked), seq_ids in masked_buckets.items():
        if len(seq_ids) < 2:
            continue
        for id_i, id_j in combinations(seq_ids, 2):
            seq_i, seq_j = sequences[id_i], sequences[id_j]
            differing = tuple(p for p in positions if seq_i[p] != seq_j[p])
            if exact:
                if len(differing) == distance:
                    yield id_i, id_j, differing
            elif positions == _canonical_mask(differing, distance, seq_len):
                yield id_i, id_j, differing


def _canonical_mask(differing: tuple[int, ...], distance: int, seq_len: int) -> tuple[int, ...]:
    """The one mask a pair is emitted from: its differing positions, padded with the lowest others."""
    if len(differing) == distance:
        return differing
    padding = (p for p in range(seq_len) if p not in differing)
    mask = set(differing)
    while len(mask) < distance:
        mask.add(next(padding))
    return tuple(sorted(mask))
