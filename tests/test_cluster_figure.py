"""The cluster-field figure: shelf layout and palette."""
import networkx as nx
import numpy as np
from portmanteau.plots.cluster_figure import extended_palette, shelf_layout
from portmanteau.plots.style import CATEGORICAL_PALETTE


def test_blobs_do_not_overlap_and_every_node_is_placed():
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("x", "y"), ("p", "q"), ("q", "r"), ("r", "p")])
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    positions = shelf_layout(components, width=4.0)
    assert set(positions) == set(graph)
    by_component = [np.array([positions[n] for n in sub]) for sub in components]
    for i, first in enumerate(by_component):
        for second in by_component[i + 1:]:
            apart = (first[:, 0].min() > second[:, 0].max()
                     or second[:, 0].min() > first[:, 0].max()
                     or first[:, 1].min() > second[:, 1].max()
                     or second[:, 1].min() > first[:, 1].max())
            assert apart, "two clusters were drawn on top of each other"


def test_the_palette_gives_distinct_colours_past_its_own_length():
    colors = extended_palette(len(CATEGORICAL_PALETTE) + 3)
    assert len(set(colors)) == len(colors)
