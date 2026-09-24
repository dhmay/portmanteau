import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
pytest.importorskip("networkx")

from portmanteau.sequence import hamming1_graph as hg  # noqa: E402
from portmanteau.sequence.hamming1_pairs import (  # noqa: E402
    find_cdr3_hamming1_pairs,
    find_hamming1_pairs_same_vj,
    populate_hamming1_pairs_othercols,
)


@pytest.fixture
def cdr3s():
    # non-default index, so index labels and positions differ
    return pd.DataFrame(
        {"cdr3": ["CASS", "CATS", "CATT", "CAXXX", "CASS"], "hla": ["A1", "A1", "A2", "A1", "A2"]},
        index=[10, 20, 30, 40, 50],
    )


@pytest.fixture
def tcrs():
    return pd.DataFrame(
        {
            "junc": ["CASS+TRBV5-1+TRBJ2-1", "CATS+TRBV5-1+TRBJ2-1", "CATT+TRBV6-2+TRBJ1-1", "QQQQ+TRBV7-1+TRBJ1-1"],
            "vgene": ["TRBV5-1", "TRBV5-1", "TRBV6-2", "TRBV7-1"],
            "jgene": ["TRBJ2-1", "TRBJ2-1", "TRBJ1-1", "TRBJ1-1"],
            "hla": ["A1", "A2", "A1", "A2"],
        }
    )


def test_find_pairs(cdr3s):
    pairs = find_cdr3_hamming1_pairs(cdr3s)
    got = set(zip(pairs.row_i, pairs.row_j, pairs.differing_position))
    assert got == {(10, 20, 2), (50, 20, 2), (20, 30, 3)}
    assert (pairs.seq_length == 4).all()


def test_empty_pairs_have_same_columns(cdr3s):
    cols = list(find_cdr3_hamming1_pairs(cdr3s).columns)
    assert list(find_cdr3_hamming1_pairs(cdr3s.iloc[[3]]).columns) == cols  # no pairs
    assert list(find_cdr3_hamming1_pairs(cdr3s.iloc[:0]).columns) == cols  # no rows
    alt = cdr3s.rename(columns={"cdr3": "junc"}).iloc[[3]]
    assert list(find_cdr3_hamming1_pairs(alt, seq_col="junc").columns)[2:4] == ["junc_i", "junc_j"]


def test_same_vj():
    vj = pd.DataFrame({"tcr": ["CASS", "CATS", "CASS2"], "vgene": ["V1", "V1", "V2"], "jgene": ["J1"] * 3})
    assert len(find_hamming1_pairs_same_vj(vj)) == 1
    assert list(find_hamming1_pairs_same_vj(vj.iloc[[0]]).columns)[2:4] == ["tcr_i", "tcr_j"]


def test_populate_othercols_uses_index_labels(cdr3s):
    full = populate_hamming1_pairs_othercols(find_cdr3_hamming1_pairs(cdr3s), cdr3s, ["cdr3", "hla"])
    for _, r in full.iterrows():
        assert r.cdr3_i == cdr3s.loc[r.row_i, "cdr3"]
        assert r.hla_j == cdr3s.loc[r.row_j, "hla"]
        assert r.same_hla == (r.hla_i == r.hla_j)


def test_populate_othercols_requires_unique_index(cdr3s):
    pairs = find_cdr3_hamming1_pairs(cdr3s)
    with pytest.raises(ValueError):
        populate_hamming1_pairs_othercols(pairs, cdr3s.set_index(pd.Index([1, 1, 2, 3, 4])), ["hla"])


def test_build_graph_with_attributes(tcrs):
    G = hg.build_seq_ham1_graph(tcrs, seq_column="junc", attribute_columns=["hla", "vfamily"])
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 1
    assert dict(G.nodes["CATS+TRBV5-1+TRBJ2-1"]) == {"vfamily": "TRBV5", "jfamily": "TRBJ2", "hla": "A2"}


def test_build_graph_same_vj(tcrs):
    G = hg.build_seq_ham1_graph(tcrs, seq_column="junc", same_vj=True)
    assert {tuple(sorted(e)) for e in G.edges} == {("CASS+TRBV5-1+TRBJ2-1", "CATS+TRBV5-1+TRBJ2-1")}


def test_extract_ccs(tcrs):
    ccs = hg.build_seq_ham1_graph_and_extract_ccs(tcrs, seq_column="junc", same_vj=True)
    assert sorted(ccs.groupby("connected_component").size()) == [1, 1, 2]
    with pytest.raises(ValueError):
        hg.build_seq_ham1_graph_and_extract_ccs(pd.concat([tcrs, tcrs]), seq_column="junc")


def test_plot_graph(tcrs):
    _, f1, ax1 = hg.build_and_plot_seq_ham1_graph(
        tcrs, seq_column="junc", attribute_columns=["hla"], node_color_attribute="hla", show_legend=True
    )
    _, f2, _ = hg.build_and_plot_seq_ham1_graph(tcrs, seq_column="junc")
    assert f1 is not f2
    assert [t.get_text() for t in ax1.get_legend().get_texts()] == ["A1", "A2"]
