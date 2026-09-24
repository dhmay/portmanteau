"""Utilities for constructing Hamming-1 CDR3 sequence pairs."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import pandas as pd


def find_hamming1_pairs_same_vj(
    pdf: pd.DataFrame,
    seq_col: str = "tcr",
    vgene_col: str = "vgene",
    jgene_col: str = "jgene",
) -> pd.DataFrame:
    """Find Hamming-1 pairs of `seq_col` among rows sharing the same V and J genes.

    Output columns are as for find_cdr3_hamming1_pairs.
    """
    output_dfs = [
        find_cdr3_hamming1_pairs(group, seq_col=seq_col)
        for _, group in pdf.groupby([vgene_col, jgene_col])
    ]
    if not output_dfs:
        return _empty_pairs(seq_col)
    return pd.concat(output_dfs, ignore_index=True)


def find_cdr3_hamming1_pairs(pdf: pd.DataFrame, seq_col: str = "cdr3") -> pd.DataFrame:
    """Build all Hamming-1 pairs from the specified column
    in an input dataframe.

    If multiple rows have the same CDR3 sequence, all combinations of
    pairs involving those rows are included in the output.

    Args:
        pdf: Input dataframe containing the specified CDR3 column.
        seq_col: Name of the column containing CDR3 sequences.

    Returns:
        DataFrame with one row per pair of input rows whose
        sequences are the same length and Hamming distance 1.

        Output columns:
        - ``row_i``: original index label of first row in pair
        - ``row_j``: original index label of second row in pair
        - ``{seq_col}_i``: first row's CDR3 sequence
        - ``{seq_col}_j``: second row's CDR3 sequence
        - ``differing_position``: 0-based index where the sequences differ
        - ``seq_length``: sequence length shared by the pair
    """
    if seq_col not in pdf.columns:
        raise ValueError(f"Input dataframe must contain a '{seq_col}' column.")

    if pdf.empty:
        return _empty_pairs(seq_col)

    seq = pdf[seq_col]
    if seq.isna().any():
        raise ValueError(f"Input dataframe contains null values in '{seq_col}'.")

    seq_str = seq.astype(str)
    lengths = seq_str.str.len()

    row_index = pdf.index.to_list()
    seq_to_row_positions_by_length: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for pos, (seq, seq_len) in enumerate(zip(seq_str.to_list(), lengths.to_list())):
        seq_to_row_positions_by_length[seq_len][seq].append(pos)

    output_rows: list[dict[str, object]] = []

    for seq_len, seq_to_positions in seq_to_row_positions_by_length.items():
        unique_sequences = list(seq_to_positions.keys())
        if len(unique_sequences) < 2:
            continue

        for seq_id_i, seq_id_j, differing_position in _iter_hamming1_unique_sequence_pairs(unique_sequences):
            seq_i = unique_sequences[seq_id_i]
            seq_j = unique_sequences[seq_id_j]
            pos_i = seq_to_positions[seq_i]
            pos_j = seq_to_positions[seq_j]

            for row_pos_i in pos_i:
                for row_pos_j in pos_j:
                    output_rows.append(
                        {
                            "row_i": row_index[row_pos_i],
                            "row_j": row_index[row_pos_j],
                            f"{seq_col}_i": seq_i,
                            f"{seq_col}_j": seq_j,
                            "differing_position": differing_position,
                            "seq_length": seq_len,
                        }
                    )

    if not output_rows:
        return _empty_pairs(seq_col)

    return pd.DataFrame(output_rows)


def populate_hamming1_pairs_othercols(
    pdf_all_ham1_pairs: pd.DataFrame,
    pdf_original: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add columns from the original dataframe to a dataframe of Hamming-1 pairs.

    For each column ``c`` in `columns`, adds ``c_i`` and ``c_j`` (the values for each
    row of the pair) and ``same_c`` (whether they're equal). Existing ``c_i``/``c_j``
    columns, e.g. the sequence columns, are replaced.

    Args:
        pdf_all_ham1_pairs: DataFrame containing Hamming-1 pairs with columns
            ``row_i`` and ``row_j`` that are index labels of the original dataframe.
        pdf_original: Original dataframe from which the pairs were derived. Its index
            must be unique.
        columns: Columns of pdf_original to add.

    Returns:
        DataFrame with additional columns populated from the original dataframe.
    """
    if not pdf_original.index.is_unique:
        raise ValueError("pdf_original must have a unique index to look up row_i/row_j")

    pdf_pairs = pdf_all_ham1_pairs.drop(
        columns=[f"{c}_{side}" for c in columns for side in ("i", "j")], errors="ignore"
    ).reset_index(drop=True)
    for side in ("i", "j"):
        pdf_side = pdf_original.loc[pdf_pairs[f"row_{side}"], columns].reset_index(drop=True)
        pdf_side.columns = [f"{c}_{side}" for c in columns]
        pdf_pairs = pd.concat([pdf_pairs, pdf_side], axis=1)
    for c in columns:
        pdf_pairs[f"same_{c}"] = pdf_pairs[f"{c}_i"] == pdf_pairs[f"{c}_j"]
    return pdf_pairs


def _empty_pairs(seq_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["row_i", "row_j", f"{seq_col}_i", f"{seq_col}_j", "differing_position", "seq_length"]
    )


def _iter_hamming1_unique_sequence_pairs(sequences: list[str]) -> Iterable[tuple[int, int, int]]:
    """Yield unique sequence-id pairs that are Hamming-1 apart.

    The input sequences must all be the same length and unique.

    Yields:
        Tuples of ``(left_id, right_id, differing_position)``.
    """
    if not sequences:
        return

    seq_len = len(sequences[0])
    masked_buckets: dict[tuple[int, str], list[int]] = defaultdict(list)

    for seq_id, seq in enumerate(sequences):
        for pos in range(seq_len):
            masked = seq[:pos] + "*" + seq[pos + 1 :]
            masked_buckets[(pos, masked)].append(seq_id)

    seen_pairs: set[tuple[int, int]] = set()
    for (pos, _masked), seq_ids in masked_buckets.items():
        if len(seq_ids) < 2:
            continue
        for left, right in combinations(seq_ids, 2):
            pair = (left, right) if left < right else (right, left)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            yield pair[0], pair[1], pos
