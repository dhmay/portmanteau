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


def _node_collection(ax):
    """The PathCollection holding the nodes, which is drawn before the edges."""
    from matplotlib.collections import PathCollection

    return next(c for c in ax.collections if isinstance(c, PathCollection))


def test_plot_graph_default_sizes_unchanged(tcrs):
    """The golden case: without the new arguments every node keeps the scalar size."""
    _, _, ax = hg.build_and_plot_seq_ham1_graph(tcrs, seq_column="junc", node_size=25)
    assert set(_node_collection(ax).get_sizes()) == {25}


def test_plot_graph_color_map_sends_unmapped_to_other(tcrs):
    _, _, ax = hg.build_and_plot_seq_ham1_graph(
        tcrs,
        seq_column="junc",
        attribute_columns=["hla"],
        node_color_attribute="hla",
        node_color_map={"A1": "#112233"},
        show_legend=True,
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A1", "other"]


def test_plot_graph_max_color_values_keeps_the_commonest(tcrs):
    # Three A1 nodes against one A2, so A1 is the value that keeps a color of its own.
    skewed = tcrs.copy()
    skewed["hla"] = ["A1", "A1", "A1", "A2"]
    _, _, ax = hg.build_and_plot_seq_ham1_graph(
        skewed,
        seq_column="junc",
        attribute_columns=["hla"],
        node_color_attribute="hla",
        max_color_values=1,
        show_legend=True,
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A1", "other"]


def test_plot_graph_node_size_attribute(tcrs):
    sized = tcrs.assign(cells=[1, 4, 9, 16])
    _, _, ax = hg.build_and_plot_seq_ham1_graph(
        sized,
        seq_column="junc",
        attribute_columns=["cells"],
        node_size_attribute="cells",
        node_size_range=(10.0, 100.0),
        node_size_scale="sqrt",
    )
    sizes = sorted(_node_collection(ax).get_sizes())
    # sqrt(1,4,9,16) = 1,2,3,4 spread evenly onto (10, 100)
    assert sizes == pytest.approx([10.0, 40.0, 70.0, 100.0])


def test_plot_graph_node_size_constant_attribute_is_mid_range(tcrs):
    flat = tcrs.assign(cells=[3, 3, 3, 3])
    _, _, ax = hg.build_and_plot_seq_ham1_graph(
        flat,
        seq_column="junc",
        attribute_columns=["cells"],
        node_size_attribute="cells",
        node_size_range=(10.0, 100.0),
    )
    assert set(_node_collection(ax).get_sizes()) == {55.0}


def test_plot_graph_node_size_errors(tcrs):
    with pytest.raises(ValueError, match="have no 'cells' attribute"):
        hg.build_and_plot_seq_ham1_graph(tcrs, seq_column="junc", node_size_attribute="cells")
    sized = tcrs.assign(cells=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="node_size_scale"):
        hg.build_and_plot_seq_ham1_graph(
            sized, seq_column="junc", attribute_columns=["cells"],
            node_size_attribute="cells", node_size_scale="cubic",
        )
    negative = tcrs.assign(cells=[-1, 2, 3, 4])
    with pytest.raises(ValueError, match="non-negative"):
        hg.build_and_plot_seq_ham1_graph(
            negative, seq_column="junc", attribute_columns=["cells"],
            node_size_attribute="cells", node_size_scale="sqrt",
        )


def test_plot_graph_layout_seed_is_reproducible(tcrs):
    G = hg.build_seq_ham1_graph(tcrs, seq_column="junc")
    _, ax1 = hg.plot_seq_ham1_graph(G, layout_seed=0)
    _, ax2 = hg.plot_seq_ham1_graph(G, layout_seed=0)
    assert _node_collection(ax1).get_offsets().data == pytest.approx(
        _node_collection(ax2).get_offsets().data
    )


def test_build_and_plot_forwards_legend_title(tcrs):
    _, _, ax = hg.build_and_plot_seq_ham1_graph(
        tcrs, seq_column="junc", attribute_columns=["hla"], node_color_attribute="hla",
        show_legend=True, legend_title="HLA",
    )
    assert ax.get_legend().get_title().get_text() == "HLA"
