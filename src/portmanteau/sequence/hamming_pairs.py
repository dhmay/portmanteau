"""Finding pairs of equal-length sequences within a small Hamming distance.

Pairs can be sought **within one frame** (find_hamming_pairs, find_hamming1_pairs) or **across two**
(find_hamming_pairs_between, find_hamming1_pairs_between) - two repertoires, a query set against a
reference, a database against a cohort. There is one implementation: the one-frame search is a
cover over the two-frame routine with both sides the same list, which costs nothing because a
bucket is then enumerated with combinations() rather than a full product.

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
from itertools import combinations, product
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
    return _collect(
        iter_hamming_pairs(pdf, seq_col, distance=distance, exact=exact, group_cols=group_cols,
                           annotation_columns=annotation_columns),
        max_pairs,
    )


def _collect(chunks: Iterable[pd.DataFrame], max_pairs: int | None) -> pd.DataFrame:
    """Concatenate streamed chunks, refusing to build a result larger than max_pairs."""
    collected = []
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if max_pairs is not None and total > max_pairs:
            raise ValueError(
                f"more than max_pairs={max_pairs:,} pairs; pair counts grow quadratically with "
                "input size. Narrow the search with group_cols, or stream the result with "
                "iter_hamming_pairs."
            )
        collected.append(chunk)
    return collected[0] if len(collected) == 1 else pd.concat(collected, ignore_index=True)


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
    return _scalar_position(pdf_pairs)


def _scalar_position(pdf_pairs: pd.DataFrame) -> pd.DataFrame:
    """Report the one differing position as a scalar, keeping the column in place."""
    # replace in place rather than appending, so the column keeps its position in the frame
    pdf_pairs["differing_positions"] = [positions[0] for positions in pdf_pairs.differing_positions]
    return pdf_pairs.rename(columns={"differing_positions": "differing_position"})


def find_hamming_pairs_between(
    pdf_a: pd.DataFrame,
    pdf_b: pd.DataFrame,
    seq_col: str,
    *,
    distance: int = 1,
    exact: bool = True,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Find pairs within `distance` drawn one from each of two dataframes.

    Use this when the question is "which sequences in A are close to one in B" - two repertoires,
    a query set against a reference, a database against a cohort. Pairing within either frame is
    not of interest and is never computed, which is the point: concatenating the two and
    discarding the same-frame pairs afterwards costs about three times as much.

    Rows whose sequences differ in length are never paired. ⚠️ Unlike the one-frame search,
    **identical sequences in the two frames are a legitimate pair** - they are the same sequence
    seen in both sources, at distance 0. With `exact` (the default) they are excluded like any
    other pair below `distance`; with ``exact=False`` they appear with ``hamming_distance`` 0.

    Args:
        pdf_a: The frame ``row_i`` comes from.
        pdf_b: The frame ``row_j`` comes from.
        seq_col: Column holding the sequences, in both frames.
        distance: Hamming distance to look for.
        exact: If True (the default), return only pairs at exactly `distance`. If False, also
            return closer pairs, and include a ``hamming_distance`` column.
        group_cols: Columns that must match for two rows to be paired. A group present in only one
            frame contributes nothing, having nothing to pair against.
        annotation_columns: Columns to carry onto both sides of each pair, taken from whichever
            frame that side came from. Must be present in both.
        max_pairs: Raise rather than return more than this many pairs; None means no limit.

    Returns:
        One row per pair, with the columns described in find_hamming_pairs. ``row_i`` and
        ``{seq_col}_i`` are always from `pdf_a`, ``row_j`` and ``{seq_col}_j`` from `pdf_b`.

    Raises:
        ValueError: if `distance` is below 1, if `seq_col` is missing from either frame or has
            null values, or if `max_pairs` is exceeded.
    """
    return _collect(
        iter_hamming_pairs_between(pdf_a, pdf_b, seq_col, distance=distance, exact=exact,
                                   group_cols=group_cols, annotation_columns=annotation_columns),
        max_pairs,
    )


def find_hamming1_pairs_between(
    pdf_a: pd.DataFrame,
    pdf_b: pd.DataFrame,
    seq_col: str,
    *,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Find cross-frame pairs differing at exactly one position.

    A convenience wrapper over find_hamming_pairs_between for the common case, differing only in
    that the single differing position is reported as a scalar ``differing_position`` rather than
    as a one-element ``differing_positions`` tuple.
    """
    return _scalar_position(
        find_hamming_pairs_between(pdf_a, pdf_b, seq_col, distance=1, group_cols=group_cols,
                                   annotation_columns=annotation_columns, max_pairs=max_pairs)
    )


def annotate_pairs(
    pdf_pairs: pd.DataFrame,
    pdf_source: pd.DataFrame,
    columns: Sequence[str],
    pdf_source_b: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add columns from the source dataframe to each side of a pairs dataframe.

    For each column ``c``, adds ``c_i`` and ``c_j``: the values for the two rows of the pair.
    Existing ``c_i``/``c_j`` columns, such as the sequence columns, are replaced.

    Args:
        pdf_pairs: Pairs with ``row_i`` and ``row_j`` columns holding index labels of the source
            frames, as returned by find_hamming_pairs.
        pdf_source: The dataframe ``row_i`` labels refer to. Its index must be unique.
        columns: Columns to add. Must be present in both source frames.
        pdf_source_b: Where ``row_j`` labels are looked up, for pairs from
            find_hamming_pairs_between. None means both sides come from pdf_source.

    Returns:
        pdf_pairs with the additional columns, reindexed from 0.

    Raises:
        ValueError: if either source's index is not unique, so row labels cannot be looked up.
    """
    sources = {"i": pdf_source, "j": pdf_source if pdf_source_b is None else pdf_source_b}
    for side, source in sources.items():
        if not source.index.is_unique:
            raise ValueError(
                f"the source frame for row_{side} must have a unique index to look up its labels"
            )

    columns = list(columns)
    pdf_annotated = pdf_pairs.drop(
        columns=[f"{c}_{side}" for c in columns for side in ("i", "j")], errors="ignore"
    ).reset_index(drop=True)
    for side, source in sources.items():
        pdf_side = source.loc[pdf_annotated[f"row_{side}"], columns].reset_index(drop=True)
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
    yield from _iter_pairs(pdf, None, seq_col, distance=distance, exact=exact,
                           group_cols=group_cols, annotation_columns=annotation_columns,
                           chunk_size=chunk_size)


def iter_hamming_pairs_between(
    pdf_a: pd.DataFrame,
    pdf_b: pd.DataFrame,
    seq_col: str,
    *,
    distance: int = 1,
    exact: bool = True,
    group_cols: Sequence[str] | None = None,
    annotation_columns: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield cross-frame pairs within `distance`, in chunks. See find_hamming_pairs_between."""
    yield from _iter_pairs(pdf_a, pdf_b, seq_col, distance=distance, exact=exact,
                           group_cols=group_cols, annotation_columns=annotation_columns,
                           chunk_size=chunk_size)


def _iter_pairs(
    pdf_a: pd.DataFrame,
    pdf_b: pd.DataFrame | None,
    seq_col: str,
    *,
    distance: int,
    exact: bool,
    group_cols: Sequence[str] | None,
    annotation_columns: Sequence[str] | None,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """The one implementation. `pdf_b` None means pair within `pdf_a`."""
    if distance < 1:
        raise ValueError(f"distance must be at least 1, got {distance}")

    columns = _pair_columns(seq_col, exact=exact)
    buffer: list[tuple] = []
    yielded = False

    def to_frame(rows: list[tuple]) -> pd.DataFrame:
        pdf_chunk = pd.DataFrame(rows, columns=columns)
        if annotation_columns:
            pdf_chunk = annotate_pairs(pdf_chunk, pdf_a, annotation_columns, pdf_source_b=pdf_b)
        return pdf_chunk

    for group_a, group_b in _iter_groups(pdf_a, pdf_b, group_cols):
        for row in _group_pairs(group_a, group_b, seq_col, distance=distance, exact=exact):
            buffer.append(row)
            if len(buffer) >= chunk_size:
                yield to_frame(buffer)
                buffer, yielded = [], True
    if buffer or not yielded:
        yield to_frame(buffer)


def _iter_groups(
    pdf_a: pd.DataFrame, pdf_b: pd.DataFrame | None, group_cols: Sequence[str] | None
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]:
    """Yield the (group_a, group_b) slices to search. Groups must match on `group_cols`.

    With two frames, a group key present in only one of them yields nothing - there is no row on
    the other side to pair with.
    """
    if group_cols is None:
        yield pdf_a, pdf_b
        return
    grouped_a = pdf_a.groupby(list(group_cols), observed=True, sort=False)
    if pdf_b is None:
        for _key, group in grouped_a:
            yield group, None
        return
    groups_b = dict(list(pdf_b.groupby(list(group_cols), observed=True, sort=False)))
    for key, group in grouped_a:
        if key in groups_b:
            yield group, groups_b[key]


def _pair_columns(seq_col: str, *, exact: bool) -> list[str]:
    columns = ["row_i", "row_j", f"{seq_col}_i", f"{seq_col}_j", "seq_length", "differing_positions"]
    return columns if exact else columns + ["hamming_distance"]


def _group_pairs(
    pdf_a: pd.DataFrame, pdf_b: pd.DataFrame | None, seq_col: str, *, distance: int, exact: bool
) -> Iterator[tuple]:
    """Yield pair rows within one group, one sequence length at a time.

    With `pdf_b`, pairs are drawn one from each frame; without it, from `pdf_a` alone.
    """
    same_frame = pdf_b is None
    rows_a, labels_a = _rows_by_length(pdf_a, seq_col)
    rows_b, labels_b = (rows_a, labels_a) if same_frame else _rows_by_length(pdf_b, seq_col)

    for seq_length, rows_by_sequence_a in rows_a.items():
        if seq_length <= distance:
            continue
        rows_by_sequence_b = rows_b.get(seq_length)
        if rows_by_sequence_b is None:
            continue
        if same_frame and len(rows_by_sequence_a) < 2:
            continue
        unique_a = list(rows_by_sequence_a)
        unique_b = unique_a if same_frame else list(rows_by_sequence_b)
        for id_i, id_j, positions in _iter_close_sequence_pairs(
            unique_a, None if same_frame else unique_b, distance=distance, exact=exact
        ):
            seq_i, seq_j = unique_a[id_i], unique_b[id_j]
            for row_i in rows_by_sequence_a[seq_i]:
                for row_j in rows_by_sequence_b[seq_j]:
                    row = (labels_a[row_i], labels_b[row_j], seq_i, seq_j, seq_length, positions)
                    yield row if exact else row + (len(positions),)


def _rows_by_length(pdf: pd.DataFrame, seq_col: str) -> tuple[dict, list]:
    """Group a frame's row positions by sequence length, then by sequence.

    Distinct sequences are what gets masked; rows sharing a sequence are expanded afterwards, so
    a repeated sequence costs one trip through the bucketing rather than one per row.
    """
    if seq_col not in pdf.columns:
        raise ValueError(f"Input dataframe must contain a '{seq_col}' column.")
    rows_by_length: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    if pdf.empty:
        return rows_by_length, []
    sequences = pdf[seq_col]
    if sequences.isna().any():
        raise ValueError(f"Input dataframe contains null values in '{seq_col}'.")
    for position, sequence in enumerate(sequences.astype(str).to_list()):
        rows_by_length[len(sequence)][sequence].append(position)
    return rows_by_length, pdf.index.to_list()


def _iter_close_sequence_pairs(
    sequences_a: list[str],
    sequences_b: list[str] | None = None,
    *,
    distance: int,
    exact: bool,
) -> Iterable[tuple[int, int, tuple[int, ...]]]:
    """Yield (id_i, id_j, differing positions) for close equal-length sequences.

    Buckets sequences on every combination of `distance` masked positions. Two sequences sharing a
    bucket differ only within those positions, so their true differing positions are found by
    comparing just those.

    With `sequences_b`, ids on each side index their own list and every pair is drawn one from
    each. Without it, both ids index `sequences_a` and each unordered pair is yielded once, with
    ``id_i < id_j``.

    Each pair is yielded once. When `exact`, a pair is emitted from the single bucket whose mask is
    its differing positions. Otherwise a pair differing at fewer positions would appear in several
    buckets, so it is emitted only from the canonical one: its differing positions padded with the
    lowest unused positions. That keeps the pass stateless either way - no set of seen pairs, which
    at these counts would dominate memory.
    """
    same_list = sequences_b is None
    if not sequences_a or (not same_list and not sequences_b):
        return
    seq_len = len(sequences_a[0])

    # One bucket per masked key, holding the ids from each side that landed in it. Building the
    # keys dominates the cost, so the two sides share one pass and one dict.
    masked_buckets: dict[tuple[tuple[int, ...], str], tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    sides = ((0, sequences_a),) if same_list else ((0, sequences_a), (1, sequences_b))
    # Masking is where nearly all the time goes - every sequence is rewritten once per mask - so
    # distance 1, by far the common case, gets a loop that slices instead of rebuilding the string
    # character by character. Measured on 40k CDR3s: 7.8x faster to build the keys, which is ~88%
    # of the work, so ~4x end to end. The branch is hoisted out of the sequence loop deliberately.
    masks = None if distance == 1 else list(combinations(range(seq_len), distance))
    for side, sequences in sides:
        bucket_side = side
        if masks is None:
            for seq_id, sequence in enumerate(sequences):
                for i in range(seq_len):
                    masked_buckets[((i,), sequence[:i] + sequence[i + 1 :])][bucket_side].append(
                        seq_id
                    )
        else:
            for seq_id, sequence in enumerate(sequences):
                for positions in masks:
                    masked = "".join(c for i, c in enumerate(sequence) if i not in positions)
                    masked_buckets[(positions, masked)][bucket_side].append(seq_id)

    for (positions, _masked), (ids_a, ids_b) in masked_buckets.items():
        # combinations() rather than product() when both sides are the same list: it is what makes
        # covering the one-list case with this routine free rather than twice the work, because a
        # singleton bucket then costs no comparison at all instead of comparing a sequence with
        # itself. Measured on 112k CDR3s: product 2.00x the one-list cost, combinations 1.00x.
        if same_list:
            if len(ids_a) < 2:
                continue
            pairs: Iterable[tuple[int, int]] = combinations(ids_a, 2)
        else:
            if not ids_a or not ids_b:
                continue
            pairs = product(ids_a, ids_b)
        for id_i, id_j in pairs:
            seq_i = sequences_a[id_i]
            seq_j = sequences_a[id_j] if same_list else sequences_b[id_j]
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
