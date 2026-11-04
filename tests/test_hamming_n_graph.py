import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
pytest.importorskip("networkx")

from portmanteau.sequence import hamming1_graph as hg1  # noqa: E402
from portmanteau.sequence import hamming_n_graph as hgn  # noqa: E402


@pytest.fixture
def ladder():
    """A chain one substitution apart, so distance 2 adds the skip-a-rung pairs."""
    return pd.DataFrame(
        {
            "cdr3": ["CASS", "CATS", "CATT", "QQQQ"],
            "vgene": ["TRBV5-1", "TRBV5-1", "TRBV5-1", "TRBV7-1"],
            "jgene": ["TRBJ2-1", "TRBJ2-1", "TRBJ2-1", "TRBJ1-1"],
        }
    )


def edge_distances(G):
    return sorted(d["hamming_distance"] for _, _, d in G.edges(data=True))


def test_distance_2_includes_the_closer_pairs_by_default(ladder):
    at_1 = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=1)
    at_2 = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2)
    assert edge_distances(at_1) == [1, 1]
    assert edge_distances(at_2) == [1, 1, 2]        # CASS-CATT is the added one
    assert set(at_1.edges) < set(at_2.edges)


def test_exact_keeps_only_the_far_pairs(ladder):
    G = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2, exact=True)
    assert edge_distances(G) == [2]
    assert {tuple(sorted(e)) for e in G.edges} == {("CASS", "CATT")}


def test_every_edge_carries_its_distance_whatever_the_weight(ladder):
    G = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2)
    assert all("hamming_distance" in d for _, _, d in G.edges(data=True))
    assert all(d["weight"] == 1 for _, _, d in G.edges(data=True))


@pytest.mark.parametrize(
    "edge_weight, expected",
    [
        (None, [1, 1, 1]),
        ("distance", [1.0, 1.0, 2.0]),
        ("similarity", [0.5, 1.0, 1.0]),
        (lambda d: 10 * d, [10.0, 10.0, 20.0]),
    ],
)
def test_edge_weight_schemes(ladder, edge_weight, expected):
    G = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2,
                                    edge_weight=edge_weight)
    assert sorted(d["weight"] for _, _, d in G.edges(data=True)) == pytest.approx(expected)


def test_similarity_is_the_inverse_of_distance(ladder):
    """The direction that matters: a closer pair must weigh more, not less."""
    G = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2,
                                    edge_weight="similarity")
    by_distance = {d["hamming_distance"]: d["weight"] for _, _, d in G.edges(data=True)}
    assert by_distance[1] > by_distance[2]


def test_unknown_edge_weight_raises(ladder):
    with pytest.raises(ValueError, match="edge_weight must be"):
        hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", edge_weight="closeness")


def test_max_pairs_refuses_to_build(ladder):
    with pytest.raises(ValueError, match="max_pairs"):
        hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2, max_pairs=1)


def test_distance_and_grouping_compose(ladder):
    G = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=2, group_cols=["vgene"])
    assert G.graph["group_cols"] == ["vgene"]
    assert edge_distances(G) == [1, 1, 2]           # QQQQ is alone in its V gene anyway
    assert all(isinstance(n, tuple) for n in G.nodes)


def test_ccs_round_trip_at_distance_2(ladder):
    pdf = hgn.build_seq_hamming_graph_and_extract_ccs(ladder, seq_column="cdr3", distance=2)
    assert len(pdf) == len(ladder)
    assert sorted(pdf.groupby("connected_component").size()) == [1, 3]


def test_ham1_cover_is_the_distance_1_graph(ladder):
    """The claim the cover module makes, pinned."""
    cover = hg1.build_seq_ham1_graph(ladder, seq_column="cdr3")
    direct = hgn.build_seq_hamming_graph(ladder, seq_column="cdr3", distance=1, exact=True)
    assert set(cover.nodes) == set(direct.nodes)
    assert set(cover.edges) == set(direct.edges)
    assert edge_distances(cover) == [1, 1]


def test_plotter_is_reachable_under_both_names(ladder):
    assert hg1.plot_seq_ham1_graph is hgn.plot_seq_graph


def test_edges_follow_index_labels_not_positions():
    """⚠️ Regression: row_i/row_j are index labels. Treating them as positions is invisible on a
    RangeIndex and wires edges to the wrong nodes on any other index."""
    shuffled = pd.DataFrame(
        {"cdr3": ["QQQQ", "CASS", "CATS"]},         # positions 0,1,2
        index=[2, 0, 1],                            # labels deliberately a permutation
    )
    G = hgn.build_seq_hamming_graph(shuffled, seq_column="cdr3")
    assert {tuple(sorted(e)) for e in G.edges} == {("CASS", "CATS")}


def test_non_range_index_is_handled(ladder):
    """A frame carved out of a bigger one keeps that frame's labels."""
    carved = ladder.iloc[1:].copy()                 # index 1,2,3
    G = hgn.build_seq_hamming_graph(carved, seq_column="cdr3")
    assert {tuple(sorted(e)) for e in G.edges} == {("CATS", "CATT")}


def test_duplicate_index_raises(ladder):
    doubled = pd.concat([ladder, ladder])           # labels repeat
    with pytest.raises(ValueError, match="non-unique index"):
        hgn.build_seq_hamming_graph(doubled, seq_column="cdr3")
