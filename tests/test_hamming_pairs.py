import pandas as pd
import pytest

from portmanteau.sequence.hamming_pairs import (
    annotate_pairs,
    find_hamming1_pairs,
    find_hamming_pairs,
    iter_hamming_pairs,
)


@pytest.fixture
def cdr3s():
    # non-default index, so index labels and positions differ; "CASS" appears twice
    return pd.DataFrame(
        {"cdr3": ["CASS", "CATS", "CATT", "CAXXX", "CASS"], "hla": ["A1", "A1", "A2", "A1", "A2"]},
        index=[10, 20, 30, 40, 50],
    )


@pytest.fixture
def tcrs():
    return pd.DataFrame(
        {
            "cdr3": ["CASS", "CATS", "CATT", "CATG"],
            "vgene": ["TRBV5-1", "TRBV5-1", "TRBV6-2", "TRBV6-2"],
            "jgene": ["TRBJ2-1", "TRBJ2-1", "TRBJ1-1", "TRBJ1-1"],
        }
    )


def pair_set(pdf_pairs, position_col="differing_positions"):
    """Order-independent pair identity, so row_i/row_j ordering is not asserted on."""
    return {
        (frozenset((row_i, row_j)), positions)
        for row_i, row_j, positions in zip(pdf_pairs.row_i, pdf_pairs.row_j, pdf_pairs[position_col])
    }


def test_finds_hamming1_pairs(cdr3s):
    pairs = find_hamming1_pairs(cdr3s, "cdr3")
    assert pair_set(pairs, "differing_position") == {
        (frozenset((10, 20)), 2),
        (frozenset((50, 20)), 2),
        (frozenset((20, 30)), 3),
    }


def test_hamming1_cover_reports_a_scalar_position(cdr3s):
    pairs = find_hamming1_pairs(cdr3s, "cdr3")
    assert "differing_positions" not in pairs.columns
    assert pairs.differing_position.tolist() == [2, 2, 3] or set(pairs.differing_position) == {2, 3}
    assert pairs.seq_length.unique().tolist() == [4]


def test_identical_sequences_are_not_paired_with_each_other(cdr3s):
    # rows 10 and 50 are both "CASS"
    pairs = find_hamming1_pairs(cdr3s, "cdr3")
    assert frozenset((10, 50)) not in {frozenset((i, j)) for i, j in zip(pairs.row_i, pairs.row_j)}


def test_different_lengths_are_never_paired(cdr3s):
    # "CAXXX" (len 5) is one edit from nothing here, and must not pair with any 4-mer
    pairs = find_hamming1_pairs(cdr3s, "cdr3")
    assert 40 not in set(pairs.row_i) | set(pairs.row_j)


def test_distance_2_excludes_distance_1_pairs(cdr3s):
    pairs = find_hamming_pairs(cdr3s, "cdr3", distance=2)
    # CASS/CATT differ at 2 and 3; CASS/CATS and CATS/CATT differ at one position only
    assert pair_set(pairs) == {(frozenset((10, 30)), (2, 3)), (frozenset((50, 30)), (2, 3))}
    assert all(len(positions) == 2 for positions in pairs.differing_positions)


def test_inexact_distance_2_also_returns_closer_pairs(cdr3s):
    exact = find_hamming_pairs(cdr3s, "cdr3", distance=2)
    inexact = find_hamming_pairs(cdr3s, "cdr3", distance=2, exact=False)
    assert pair_set(exact) <= pair_set(inexact)
    # every Hamming-1 pair appears exactly once, not once per mask containing its position
    at_distance_1 = inexact[inexact.hamming_distance == 1]
    assert len(at_distance_1) == len(find_hamming1_pairs(cdr3s, "cdr3"))
    assert len(inexact) == len(exact) + len(at_distance_1)


def test_exact_is_a_no_op_at_distance_1(cdr3s):
    assert pair_set(find_hamming_pairs(cdr3s, "cdr3", distance=1)) == pair_set(
        find_hamming_pairs(cdr3s, "cdr3", distance=1, exact=False)
    )


def test_group_cols_restricts_pairing(tcrs):
    # ungrouped, CATS is one edit from all three others, so there are four pairs
    ungrouped = find_hamming1_pairs(tcrs, "cdr3")
    assert len(ungrouped) == 4
    # grouping leaves only the two pairs that share a V and J
    grouped = find_hamming1_pairs(tcrs, "cdr3", group_cols=["vgene", "jgene"])
    assert len(grouped) < len(ungrouped)
    assert pair_set(grouped, "differing_position") == {
        (frozenset((0, 1)), 2),  # CASS / CATS
        (frozenset((2, 3)), 3),  # CATT / CATG
    }


def test_group_cols_accepts_a_single_column(tcrs):
    assert len(find_hamming1_pairs(tcrs, "cdr3", group_cols=["vgene"])) == 2


def test_chunked_iteration_matches_the_whole_frame(cdr3s):
    whole = find_hamming_pairs(cdr3s, "cdr3", distance=1)
    chunks = list(iter_hamming_pairs(cdr3s, "cdr3", distance=1, chunk_size=1))
    assert len(chunks) == len(whole)
    assert pair_set(pd.concat(chunks, ignore_index=True)) == pair_set(whole)


def test_iteration_always_yields_the_schema(cdr3s):
    chunks = list(iter_hamming_pairs(cdr3s.iloc[[3]], "cdr3"))
    assert len(chunks) == 1 and chunks[0].empty
    assert list(chunks[0].columns) == list(find_hamming_pairs(cdr3s, "cdr3").columns)


def test_max_pairs_raises(cdr3s):
    with pytest.raises(ValueError, match="max_pairs"):
        find_hamming_pairs(cdr3s, "cdr3", max_pairs=1)
    assert len(find_hamming_pairs(cdr3s, "cdr3", max_pairs=3)) == 3


def test_empty_and_pairless_input_keep_the_columns(cdr3s):
    columns = list(find_hamming_pairs(cdr3s, "cdr3").columns)
    assert list(find_hamming_pairs(cdr3s.iloc[[3]], "cdr3").columns) == columns
    assert list(find_hamming_pairs(cdr3s.iloc[:0], "cdr3").columns) == columns


def test_sequence_column_name_is_used_in_the_output(cdr3s):
    renamed = cdr3s.rename(columns={"cdr3": "junction"})
    assert {"junction_i", "junction_j"} <= set(find_hamming_pairs(renamed, "junction").columns)


def test_rejects_bad_input(cdr3s):
    with pytest.raises(ValueError, match="distance"):
        find_hamming_pairs(cdr3s, "cdr3", distance=0)
    with pytest.raises(ValueError, match="missing"):
        find_hamming_pairs(cdr3s, "missing")
    with pytest.raises(ValueError, match="null"):
        find_hamming_pairs(cdr3s.assign(cdr3=[None] * 5), "cdr3")


def test_annotate_pairs_adds_both_sides_and_equality(cdr3s):
    annotated = annotate_pairs(find_hamming1_pairs(cdr3s, "cdr3"), cdr3s, ["cdr3", "hla"])
    assert {"cdr3_i", "cdr3_j", "same_cdr3", "hla_i", "hla_j", "same_hla"} <= set(annotated.columns)
    assert not annotated.same_cdr3.any()  # Hamming-1 pairs never share a sequence
    assert annotated.same_hla.tolist() == [
        hla_i == hla_j for hla_i, hla_j in zip(annotated.hla_i, annotated.hla_j)
    ]


def test_annotate_pairs_requires_a_unique_index(cdr3s):
    pairs = find_hamming1_pairs(cdr3s, "cdr3")
    with pytest.raises(ValueError, match="unique index"):
        annotate_pairs(pairs, cdr3s.set_index(pd.Index([1, 1, 2, 3, 4])), ["hla"])
