import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
pytest.importorskip("networkx")

from portmanteau.sequence import hamming1_graph as hg  # noqa: E402


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
