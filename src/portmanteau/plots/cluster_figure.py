"""Drawing a field of sequence clusters: each laid out on its own, then packed onto a shelf.

A Hamming-1 component is small and there are many of them, so the useful picture is not one
graph layout but many little ones side by side. `networkx` lays a disconnected graph out by
piling its components on top of each other, which is why this exists.

⭐ **Kamada-Kawai per component, scaled uniformly** - and both halves of that matter. These
components are usually chain-like, so a spring layout leaves one a near-straight line whose tiny
off-axis wobble a per-axis rescale amplifies into a zigzag that **looks like structure and is
numerical noise**. A long thin blob here is a long thin component.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from matplotlib.colors import to_rgb
from portmanteau.plots.style import CATEGORICAL_PALETTE

#: Grey, for a node with no colour of its own.
UNANNOTATED_COLOR = "0.78"


def draw_cluster_field(graph, components, color_of_node, ax, *, width: float = 20.0,
                       pad: float = 0.7, node_size: int = 30,
                       unannotated_color: str = UNANNOTATED_COLOR) -> dict:
    """Draw each component as its own blob, packed left to right, largest first.

    Args:
        graph: the whole graph, used for its edges.
        components: the subgraphs to draw. Anything not in one is left out entirely.
        color_of_node: {node: colour}; a node missing from it gets `unannotated_color`.
        ax: the axes to draw on. Its frame is turned off.
        width: how wide a shelf row may get before it wraps, in layout units.
        pad: the gap between blobs.
        node_size: matplotlib node area.
        unannotated_color: the colour for a node `color_of_node` says nothing about.

    Returns:
        The {node: (x, y)} layout, so a caller can annotate individual blobs.
    """
    positions = shelf_layout(components, width=width, pad=pad)
    drawn = graph.subgraph(positions).copy()
    nx.draw_networkx_edges(drawn, positions, ax=ax, edge_color="0.8", width=0.8)
    nx.draw_networkx_nodes(
        drawn, positions, ax=ax, node_size=node_size, linewidths=0,
        node_color=[color_of_node.get(node, unannotated_color) for node in drawn])
    ax.set_axis_off()
    return positions


def extended_palette(n: int, base=CATEGORICAL_PALETTE) -> list:
    """`n` distinct colours: the base palette, then the same hues darkened.

    Cycling `base` would give two categories the same colour. Darkening keeps every colour inside
    the base palette's hues while separating the 9th onwards from the 1st onwards.
    """
    colors = []
    for step in range(n // len(base) + 1):
        shade = 1.0 - 0.45 * step
        colors += [tuple(channel * shade for channel in to_rgb(color)) for color in base]
    return colors[:n]


def shelf_layout(components, *, width: float = 20.0, pad: float = 0.7) -> dict:
    """Lay each component out alone, then pack the blobs into rows, largest first.

    A blob's radius goes as the square root of its size, so area tracks membership rather than
    every component getting the same room.
    """
    positions, x, y, row_height = {}, 0.0, 0.0, 0.0
    for sub in sorted(components, key=len, reverse=True):
        local = nx.kamada_kawai_layout(sub)
        coords = np.array(list(local.values()))
        coords = coords - coords.mean(axis=0)
        unit = coords / (np.abs(coords).max() or 1.0)
        radius = np.sqrt(len(sub)) * 0.55
        if x > 0 and x + 2 * radius > width:
            x, y, row_height = 0.0, y - row_height - pad, 0.0
        for node, (ux, uy) in zip(local, unit):
            positions[node] = (x + radius + ux * radius, y - radius - uy * radius)
        x += 2 * radius + pad
        row_height = max(row_height, 2 * radius)
    return positions
