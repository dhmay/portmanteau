# This file contains functions to build and visualize a networkx graph of
# Hamming-1 connections between TCRs
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.patches import Patch  # type: ignore
from networkx.classes.graph import Graph as NxGraph

from .hamming_pairs import find_hamming1_pairs


def build_and_plot_seq_ham1_graph(
    pdf_tcrs: pd.DataFrame,
    node_color_attribute: Optional[str] = None,
    width: int = 10,
    height: int = 10,
    edge_width: float = 0.6,
    edge_alpha: float = 0.5,
    node_size: int = 25,
    show_legend: bool = False,
    legend_bbox_to_anchor: Tuple[float, float] = (1.03, 1.0),
    legend_loc: str = "upper left",
    seq_column: str = "cdr3",
    attribute_columns: Optional[List[str]] = None,
    title: str = "Hamming-1 connection graph",
    legend_title: Optional[str] = None,
    node_color_map: Optional[Dict[Any, Any]] = None,
    max_color_values: Optional[int] = None,
    other_color: Any = "0.72",
    other_label: str = "other",
    node_size_attribute: Optional[str] = None,
    node_size_range: Tuple[float, float] = (12.0, 200.0),
    node_size_scale: str = "sqrt",
    pos: Optional[Dict[Any, Any]] = None,
    layout_seed: Optional[int] = None,
) -> Tuple[NxGraph, plt.Figure, plt.Axes]:
    """Builds and plots a networkx graph of hamming-1 connections between tcrs.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        node_color_attribute (str, optional): attribute for node color. Defaults to None.
        width (int, optional): plot width. Defaults to 10.
        height (int, optional): plot height. Defaults to 10.
        edge_width (float, optional): edge width. Defaults to 0.6.
        edge_alpha (float, optional): edge alpha. Defaults to 0.5.
        node_size (int, optional): node size. Defaults to 25.
        show_legend (bool, optional): Show legend? Defaults to False.
        legend_bbox_to_anchor (Tuple[float, float], optional): legend location. Defaults to (1.03, 1.0).
        legend_loc (str, optional): legend anchor. Defaults to 'upper left'.
        seq_column (str, optional): sequence column. Defaults to "cdr3".
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to None.
        title (str, optional): title. Defaults to "Hamming-1 connection graph".
        legend_title (str, optional): heading for the legend. Defaults to None.

    Returns:
        Tuple[NxGraph, plt.Figure, plt.Axes]: Graph, Figure and Axes
    """
    G = build_seq_ham1_graph(
        pdf_tcrs, seq_column=seq_column, attribute_columns=attribute_columns
    )
    f, ax = plot_seq_ham1_graph(
        G,
        node_color_attribute=node_color_attribute,
        width=width,
        height=height,
        edge_width=edge_width,
        edge_alpha=edge_alpha,
        node_size=node_size,
        show_legend=show_legend,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_loc=legend_loc,
        title=title,
        legend_title=legend_title,
        node_color_map=node_color_map,
        max_color_values=max_color_values,
        other_color=other_color,
        other_label=other_label,
        node_size_attribute=node_size_attribute,
        node_size_range=node_size_range,
        node_size_scale=node_size_scale,
        pos=pos,
        layout_seed=layout_seed,
    )
    return G, f, ax


def _color_scheme(values, node_color_map, max_color_values, other_color, other_label):
    """The ordered values that get their own color, and the colors for them.

    An explicit map wins; otherwise the most common values are kept when max_color_values is set,
    and everything else falls through to other_color at draw time.
    """
    if node_color_map is not None:
        ordered = [v for v in node_color_map if v in set(values)]
        colors = [node_color_map[v] for v in ordered]
    elif max_color_values is not None:
        counts = pd.Series(values).value_counts()
        ordered = list(counts.head(max_color_values).index)
        colors = list(sns.color_palette(n_colors=len(ordered)))
    else:
        # Historical behaviour: every distinct value gets a color.
        ordered = sorted(set(values))
        return ordered, list(sns.color_palette(n_colors=len(ordered)))
    if set(values) - set(ordered):
        ordered = ordered + [other_label]
        colors = colors + [other_color]
    return ordered, colors


def _node_sizes(G, attribute, size_range, scale):
    """Per-node point areas from a node attribute, in G's own node order."""
    missing = [n for n in G if G.nodes[n].get(attribute) is None]
    if missing:
        raise ValueError(
            f"{len(missing):,} of {G.number_of_nodes():,} nodes have no {attribute!r} attribute; "
            "pass it in build_seq_ham1_graph's attribute_columns"
        )
    values = np.asarray([G.nodes[n][attribute] for n in G], dtype=float)
    if scale not in ("linear", "sqrt", "log"):
        raise ValueError(f"node_size_scale must be 'linear', 'sqrt' or 'log'; got {scale!r}")
    if scale in ("sqrt", "log") and (values < 0).any():
        raise ValueError(f"{scale!r} scaling needs non-negative values; got a negative one")
    transformed = {"linear": values, "sqrt": np.sqrt(values), "log": np.log1p(values)}[scale]
    smallest, largest = size_range
    span = float(np.ptp(transformed))
    if span == 0:
        # Nothing to compare, so don't imply a comparison: every node gets the middle size.
        return np.full(len(values), (smallest + largest) / 2)
    return smallest + (largest - smallest) * (transformed - transformed.min()) / span


def build_seq_ham1_graph_and_extract_ccs(
    pdf_tcrs: pd.DataFrame,
    min_seqs: int = 1,
    seq_column: str = "cdr3",
    same_vj: bool = False,
) -> pd.DataFrame:
    """Builds a networkx graph of hamming-1 connections between sequences and extracts connected
    components with at least `min_seqs` sequences.

    Takes a Pandas DataFrame of TCRs and returns the same dataframe with connected component as a new
    Raises ValueError if pdf_tcrs has duplicate sequences in the seq_column.

    Args:
        df_tcrs (DataFrame): DataFrame of TCRs
        min_seqs (int, optional): minimum sequences per component. Defaults to 1.
        seq_column (str, optional): sequence column. Defaults to "cdr3". If you want to use something
            else, like a preprocessed sequence column, you can specify it here.

    Returns:
        pd.DataFrame: pdf_tcrs with connected_component column
    """
    if len(set(pdf_tcrs[seq_column])) != len(pdf_tcrs):
        raise ValueError(f"pdf_tcrs has duplicate sequences in column '{seq_column}'")
    else:
        G = build_seq_ham1_graph(pdf_tcrs, seq_column=seq_column, same_vj=same_vj)
    pdf_ccs = extract_connected_components(G, min_seqs, seq_column=seq_column)
    pdf_tcrs_with_ccs = pdf_tcrs.merge(pdf_ccs, on=seq_column)
    return pdf_tcrs_with_ccs

def build_seq_ham1_graph(
    pdf_tcrs: pd.DataFrame,  # type: ignore
    seq_column: str = "cdr3",
    attribute_columns: Optional[List[str]] = None,
    same_vj: bool = False,
) -> NxGraph:
    """Builds a networkx graph from a DataFrame of TCRs, where nodes are seqs and edges are
    hamming-1 connections between sequences.

    Args:
        pdf_tcrs (DataFrame): DataFrame of TCRs
        seq_column (str, optional): sequence column. Defaults to "cdr3".
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to None.
            Regardless of what's specified here, vfamily and jfamily will always be added as node
            attributes, derived from the sequence column. That's because they're so often useful,
            but it's probably a bad idea.

    Returns:
        NxGraph: Networkx graph of hamming-1 connections between sequences
    """
    seq_column_i = seq_column + "_i"
    seq_column_j = seq_column + "_j"

    if same_vj:
        pdf_ham1pairs = find_hamming1_pairs(pdf_tcrs, seq_column, group_cols=["vgene", "jgene"])
    else:
        pdf_ham1pairs = find_hamming1_pairs(pdf_tcrs, seq_column)

    all_seqs = set(pdf_tcrs[seq_column])

    G = nx.Graph()

    for seq in all_seqs:
        chunks = seq.split("+")
        if len(chunks) == 3:
            _, vgene, jgene = chunks
        else:
            vgene = jgene = "unknown"
        vfamily = vgene.split("-")[0]
        jfamily = jgene.split("-")[0]
        G.add_node(seq, vfamily=vfamily, jfamily=jfamily)

    # if there are other attributes to add, we have to get a little clever, because we haven't explicitly
    # verified that sequence is unique. We'll just take the first value for each sequence
    if attribute_columns is not None:
        attrs = [a for a in attribute_columns if a not in ("vfamily", "jfamily")]
        pdf_first = pdf_tcrs.drop_duplicates(seq_column).set_index(seq_column)[attrs]
        nx.set_node_attributes(G, pdf_first.to_dict(orient="index"))

    for _, row in pdf_ham1pairs.iterrows():
        G.add_edge(row[seq_column_i], row[seq_column_j], weight=1)

    return G


def plot_seq_ham1_graph(
    G: NxGraph,
    node_color_attribute: Optional[str] = None,  # type: ignore
    width: int = 10,
    height: int = 10,
    edge_width: float = 0.6,
    edge_alpha: float = 0.5,
    node_size: int = 25,
    show_legend: bool = False,  # type: ignore
    legend_bbox_to_anchor: Tuple[float, float] = (1.03, 1.0),
    legend_loc: str = "upper left",
    title: str = "Hamming-1 connection graph",
    legend_title: Optional[str] = None,
    node_color_map: Optional[Dict[Any, Any]] = None,
    max_color_values: Optional[int] = None,
    other_color: Any = "0.72",
    other_label: str = "other",
    node_size_attribute: Optional[str] = None,
    node_size_range: Tuple[float, float] = (12.0, 200.0),
    node_size_scale: str = "sqrt",
    pos: Optional[Dict[Any, Any]] = None,
    layout_seed: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a networkx graph of hamming-1 connections between sequences.

    Args:
        G (NxGraph): graph
        node_color_attribute (str, optional): attribute for node color. Defaults to None.
        width (int, optional): plot width. Defaults to 10.
        height (int, optional): plot height. Defaults to 10.
        edge_width (float, optional): edge width. Defaults to 0.6.
        edge_alpha (float, optional): edge alpha. Defaults to 0.5.
        node_size (int, optional): node size. Defaults to 25.
        show_legend (bool, optional): Show legend? Defaults to False.
        legend_bbox_to_anchor (Tuple[float, float], optional): legend location. Defaults to (1.03, 1.0).
        legend_loc (str, optional): legend anchor. Defaults to 'upper left'.
        title (str, optional): title. Defaults to "Hamming-1 connection graph".
        node_color_map (dict, optional): explicit value -> color. Any attribute value not in it is
            drawn in other_color and collected under other_label in the legend. Defaults to None,
            which colors every distinct value from the seaborn palette.
        max_color_values (int, optional): color only the this-many most common attribute values and
            send the rest to other_color. ⚠️ Without it a palette of N colors is stretched over
            however many values there are, and seaborn recycles colors past the end of the palette,
            so two values can get the same swatch with nothing saying so. Defaults to None.
        other_color: color for unmapped values. Defaults to "0.72".
        other_label (str, optional): legend label for them. Defaults to "other".
        node_size_attribute (str, optional): node attribute to size nodes by, in place of the
            node_size scalar. ⚠️ Sizing is keyed on the attribute rather than taking a sequence,
            because a sequence would have to match G's node order and build_seq_ham1_graph builds
            its nodes from a set, whose order a caller cannot predict. Defaults to None.
        node_size_range (tuple, optional): (smallest, largest) point area to map values onto.
            Defaults to (12.0, 200.0).
        node_size_scale (str, optional): "linear", "sqrt" or "log", applied to the values before
            they are mapped onto node_size_range. Defaults to "sqrt".
        pos (dict, optional): a precomputed layout, so a caller can draw twice against one layout
            or annotate afterwards. Defaults to None, which computes a spring layout.
        layout_seed (int, optional): seed for that spring layout. Defaults to None, which is
            nondeterministic - the historical behaviour.
    """
    node_colors = "b"
    if node_color_attribute is not None:
        # Define the node colors based on the attribute, in G's node order (which is what
        # draw_networkx_nodes uses)
        node_color_vals = [G.nodes[n].get(node_color_attribute) for n in G]
        ordered_unique_color_vals, colors_inorder = _color_scheme(
            node_color_vals, node_color_map, max_color_values, other_color, other_label
        )
        val_color_map = dict(zip(*[ordered_unique_color_vals, colors_inorder]))
        node_colors = [  # type: ignore
            val_color_map.get(x, other_color) for x in node_color_vals
        ]

    node_sizes: Any = node_size
    if node_size_attribute is not None:
        node_sizes = _node_sizes(G, node_size_attribute, node_size_range, node_size_scale)

    # Initialize the plot
    f, ax = plt.subplots(figsize=(width, height))

    # Draw the graph
    if pos is None:
        pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=edge_width, ax=ax)
    ax.set_title(title)

    # Draw the legend
    if show_legend:
        if node_color_attribute is None:
            legend_elements = [Patch(facecolor="b", label="Nodes")]
        else:
            legend_elements = [
                Patch(facecolor=colors_inorder[i], label=ordered_unique_color_vals[i])
                for i in range(len(colors_inorder))
            ]
        ax.legend(handles=legend_elements, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc,
                  title=legend_title)  # type: ignore

    return f, ax


def extract_connected_components(
    G: NxGraph, min_seqs: int = 1, seq_column: str = "cdr3",
) -> pd.DataFrame:
    """Extracts connected components from a networkx graph into a Pandas dataframe.
    Component identifiers aren't meaningful, but are unique integers.

    Will fail if there are no components with at least `min_seqs` sequences

    Args:
        G (NxGraph): graph
        min_seqs (int, optional): minimum sequences per component. Defaults to 1.
        seq_column (str, optional): sequence column. Defaults to "cdr3"

    Returns:
        pd.DataFrame: Pandas dataframe with columns seq_column and `connected_component`
    """
    pdfs = []
    nodes_in_ccs = set()
    for idx, seqs in enumerate(nx.connected_components(G)):
        if len(seqs) >= min_seqs:
            pdf_one_cc = pd.DataFrame({seq_column: list(seqs)})
            pdf_one_cc["connected_component"] = idx
            pdfs.append(pdf_one_cc)
            nodes_in_ccs.update(seqs)

    if len(pdfs) == 0:
        return pd.DataFrame(columns=[seq_column, "connected_component"])
    else:
        pdf_ccs = pd.concat(pdfs)
    return pdf_ccs


def prepend_other_column_to_seq(
    pdf_tcrs: pd.DataFrame, seq_column, other_column: str, sep: str = "|"
) -> pd.DataFrame:
    """Utility function to prepend another column to the sequence column, in a new column, with
    a separator between. This is useful for e.g. forcing Hamming-1
    connections between sequences to only be between members of the same cluster.

    Args:
        pdf_tcrs (pd.DataFrame): DataFrame of TCRs
        other_column (str): column to prepend to the sequence column
        sep (str, optional): separator. Defaults to "|".
    """
    return pdf_tcrs[other_column] + sep + pdf_tcrs[seq_column]  # type: ignore

