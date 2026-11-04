"""Graphs of sequences joined when they sit within a Hamming distance of each other.

⚠️ `exact` defaults to **False** here, unlike `find_hamming_pairs`, whose default is
True. A graph of pairs at *exactly* distance 2, with the distance-1 pairs deliberately
left out, is almost never what someone means, and its connected components are chains
of exactly-two steps with the closer relatives missing. `hamming1_graph` pins
`exact=True`, where at distance 1 the two agree anyway.

`hamming1_graph` holds thin Hamming-1 covers over everything here.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.patches import Patch  # type: ignore
from networkx.classes.graph import Graph as NxGraph

from .hamming_pairs import find_hamming_pairs


def build_and_plot_seq_hamming_graph(
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
    same_vj: bool = False,
    group_cols: Optional[Sequence[str]] = None,
    same_v: bool = False,
    same_j: bool = False,
    node_color_map: Optional[Dict[Any, Any]] = None,
    max_color_values: Optional[int] = None,
    other_color: Any = "0.72",
    other_label: str = "other",
    node_size_attribute: Optional[str] = None,
    node_size_range: Tuple[float, float] = (12.0, 200.0),
    node_size_scale: str = "sqrt",
    pos: Optional[Dict[Any, Any]] = None,
    layout_seed: Optional[int] = None,
    distance: int = 1,
    exact: bool = False,
    edge_weight: Union[None, str, Callable[[int], float]] = None,
    max_pairs: Optional[int] = None,
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
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to
            None, which carries exactly the attributes the plot needs - whichever of
            node_color_attribute and node_size_attribute were given. Pass a list to put more on
            the returned graph for your own later use.
        title (str, optional): title. Defaults to "Hamming-1 connection graph".
        legend_title (str, optional): heading for the legend. Defaults to None.

    Returns:
        Tuple[NxGraph, plt.Figure, plt.Axes]: Graph, Figure and Axes
    """
    if attribute_columns is None:
        # Nothing to carry but what the plot consumes, so the caller does not have to name the
        # same attribute twice. ⚠️ Forgetting it used to be silent: every node took None, so the
        # whole graph came out one colour with a blank legend entry.
        inferred = [a for a in (node_color_attribute, node_size_attribute) if a is not None]
        attribute_columns = inferred or None
    G = build_seq_hamming_graph(
        pdf_tcrs, seq_column=seq_column, attribute_columns=attribute_columns,
        same_vj=same_vj, group_cols=group_cols, same_v=same_v, same_j=same_j,
        distance=distance, exact=exact, edge_weight=edge_weight, max_pairs=max_pairs,
    )
    f, ax = plot_seq_graph(
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
            "pass it in build_seq_hamming_graph's attribute_columns"
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


def build_seq_hamming_graph_and_extract_ccs(
    pdf_tcrs: pd.DataFrame,
    min_seqs: int = 1,
    seq_column: str = "cdr3",
    same_vj: bool = False,
    group_cols: Optional[Sequence[str]] = None,
    same_v: bool = False,
    same_j: bool = False,
    distance: int = 1,
    exact: bool = False,
    edge_weight: Union[None, str, Callable[[int], float]] = None,
    max_pairs: Optional[int] = None,
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
    resolved_group_cols = _resolve_group_cols(group_cols, same_vj, same_v, same_j)
    # ⚠️ Uniqueness is per group, not per sequence: under grouping one sequence may legitimately
    # appear once under each group, and those are different nodes.
    key_columns = [seq_column] + list(resolved_group_cols or [])
    if pdf_tcrs.duplicated(key_columns).any():
        raise ValueError(
            f"pdf_tcrs has duplicate rows over {key_columns}, so a component cannot be merged "
            "back onto one row"
        )
    G = build_seq_hamming_graph(
        pdf_tcrs, seq_column=seq_column, same_vj=same_vj, group_cols=group_cols,
        same_v=same_v, same_j=same_j, distance=distance, exact=exact,
        edge_weight=edge_weight, max_pairs=max_pairs,
    )
    pdf_ccs = extract_connected_components(G, min_seqs, seq_column=seq_column)
    pdf_tcrs_with_ccs = pdf_tcrs.merge(pdf_ccs, on=key_columns)
    return pdf_tcrs_with_ccs

def build_seq_hamming_graph(
    pdf_tcrs: pd.DataFrame,  # type: ignore
    seq_column: str = "cdr3",
    attribute_columns: Optional[List[str]] = None,
    same_vj: bool = False,
    group_cols: Optional[Sequence[str]] = None,
    same_v: bool = False,
    same_j: bool = False,
    distance: int = 1,
    exact: bool = False,
    edge_weight: Union[None, str, Callable[[int], float]] = None,
    max_pairs: Optional[int] = None,
) -> NxGraph:
    """Builds a networkx graph from a DataFrame of TCRs, where nodes are seqs and edges are
    hamming-1 connections between sequences.

    Args:
        pdf_tcrs (DataFrame): DataFrame of TCRs
        seq_column (str, optional): sequence column. Defaults to "cdr3".
        attribute_columns (List[str], optional): columns to use as node attributes. Defaults to None.
            Regardless of what's specified here, vfamily and jfamily are added as node attributes,
            taken from pdf_tcrs where it has a column of that name and otherwise derived from the
            sequence column. ⚠️ The derivation reads a "seq+vgene+jgene" sequence and gives
            "unknown" for any other shape, which is why a column the caller supplies wins.
        same_vj (bool, optional): only join sequences sharing a vgene and a jgene. Sugar for
            group_cols=["vgene", "jgene"]. Defaults to False.
        group_cols (Sequence[str], optional): only join sequences agreeing on these columns. Any
            columns, so "same V family" or "same donor" are as expressible as "same V gene".
            ⚠️ **Grouping changes what a node is**: a bare sequence would collect edges from
            every group it appears in, defeating the condition, so with group_cols a node is the
            tuple (sequence, *group values). Without it, nodes are bare sequences as before.
            Also the lever for size - pair counts grow quadratically, and grouping cuts the
            comparisons. Defaults to None.
        distance (int, optional): join sequences within this Hamming distance. Defaults to 1.
        exact (bool, optional): join only pairs at exactly `distance`, rather than within it.
            ⚠️ Defaults to **False**, unlike find_hamming_pairs. Defaults to False.
        edge_weight: what to put on each edge's `weight`. None (the default) writes 1, as before.
            "similarity" writes 1/d, which is what layout and community detection want - closer
            is a stronger tie. "distance" writes d, which is what shortest path and minimum
            spanning tree want - closer is a shorter hop. A callable takes the distance and
            returns the weight. ⚠️ `hamming_distance` is on every edge regardless.
        max_pairs (int, optional): refuse to build if there are more pairs than this. Pair counts
            grow quadratically with input size and sharply with distance, so a distance-2 graph
            over a large frame is the case that needs a ceiling. Defaults to None.
        same_v (bool, optional): sugar for group_cols=["vgene"]. Defaults to False.
        same_j (bool, optional): sugar for group_cols=["jgene"]. Defaults to False.
            ⚠️ The three same_* flags assume columns literally named "vgene" and "jgene"; a frame
            that calls them something else wants group_cols. Passing group_cols together with any
            same_* flag is an error rather than a silent precedence rule, as is same_vj with
            same_v or same_j.

    Returns:
        NxGraph: Networkx graph of hamming-1 connections between sequences. `G.graph["group_cols"]`
            records the grouping, so a consumer can tell what a node is without being told.
    """
    resolved_group_cols = _resolve_group_cols(group_cols, same_vj, same_v, same_j)
    weigh = _edge_weigher(edge_weight)
    pdf_pairs = find_hamming_pairs(pdf_tcrs, seq_column, distance=distance, exact=exact,
                                   group_cols=resolved_group_cols, max_pairs=max_pairs)

    # The node key carries the grouping, or the condition would be undone at the node: one
    # sequence under two V genes would be a single node wearing both sets of edges.
    key_columns = [seq_column] + list(resolved_group_cols or [])
    pdf_keys = pdf_tcrs[key_columns].drop_duplicates()
    node_of_row = {
        i: (row[0] if len(key_columns) == 1 else tuple(row))
        for i, row in zip(pdf_tcrs.index, pdf_tcrs[key_columns].itertuples(index=False, name=None))
    }
    position_of_row = {position: i for position, i in enumerate(pdf_tcrs.index)}

    G = nx.Graph(group_cols=list(resolved_group_cols) if resolved_group_cols else None)

    # vfamily and jfamily are parsed out of the sequence, but only for the one the caller has not
    # supplied. ⚠️ The parse reads a "seq+vgene+jgene" sequence and yields "unknown" for anything
    # else, so overriding a real column with it replaced data with a guess.
    parsed = [c for c in ("vfamily", "jfamily") if c not in pdf_tcrs.columns]
    supplied = [c for c in ("vfamily", "jfamily") if c in pdf_tcrs.columns]

    for key_row in pdf_keys.itertuples(index=False, name=None):
        seq = key_row[0]
        chunks = seq.split("+")
        if len(chunks) == 3:
            _, vgene, jgene = chunks
        else:
            vgene = jgene = "unknown"
        families = {"vfamily": vgene.split("-")[0], "jfamily": jgene.split("-")[0]}
        node = key_row[0] if len(key_columns) == 1 else key_row
        G.add_node(node, **{c: families[c] for c in parsed})

    # if there are other attributes to add, we have to get a little clever, because we haven't explicitly
    # verified that sequence is unique. We'll just take the first value for each sequence
    attrs = [a for a in (attribute_columns or []) if a not in parsed]
    attrs += [c for c in supplied if c not in attrs]
    if attrs:
        pdf_first = pdf_tcrs.drop_duplicates(key_columns)
        keys = [row[0] if len(key_columns) == 1 else tuple(row)
                for row in pdf_first[key_columns].itertuples(index=False, name=None)]
        nx.set_node_attributes(
            G, dict(zip(keys, pdf_first[attrs].to_dict(orient="records")))
        )

    # Endpoints come from the pair rows' positions rather than their sequence values: under
    # grouping a sequence alone no longer names a node.
    # Every edge carries hamming_distance, whatever the weight scheme, so a caller who wants a
    # different rescaling can compute it without rebuilding.
    distances = (pdf_pairs.hamming_distance if "hamming_distance" in pdf_pairs.columns
                 else [distance] * len(pdf_pairs))
    for row_i, row_j, edge_distance in zip(pdf_pairs.row_i, pdf_pairs.row_j, distances):
        G.add_edge(node_of_row[position_of_row[row_i]], node_of_row[position_of_row[row_j]],
                   weight=weigh(int(edge_distance)), hamming_distance=int(edge_distance))

    return G


def _edge_weigher(edge_weight) -> Callable[[int], float]:
    """Turn the edge_weight choice into a function of the Hamming distance.

    ⚠️ networkx reads `weight` two incompatible ways. Layout (`spring_layout`) and community
    detection treat it as a **strength**, where closer should mean larger; shortest path and
    minimum spanning tree treat it as a **cost**, where closer should mean smaller. There is no
    direction that is right for both, which is why the scheme is named at the call site rather
    than implied by a flag.
    """
    if edge_weight is None:
        return lambda edge_distance: 1
    if callable(edge_weight):
        return edge_weight
    if edge_weight == "distance":
        return lambda edge_distance: float(edge_distance)
    if edge_weight == "similarity":
        def similarity(edge_distance: int) -> float:
            if edge_distance == 0:
                raise ValueError(
                    "a pair at distance 0 has no similarity weight under 1/d; identical "
                    "sequences should be one node, not an edge"
                )
            return 1.0 / edge_distance
        return similarity
    raise ValueError(
        f"edge_weight must be None, 'similarity', 'distance' or a callable; got {edge_weight!r}"
    )


def _resolve_group_cols(group_cols, same_vj, same_v, same_j) -> Optional[List[str]]:
    """One grouping from the general parameter and the three sugar flags, or a loud conflict."""
    sugar = [name for name, on in (("same_v", same_v), ("same_j", same_j)) if on]
    if same_vj and sugar:
        raise ValueError(
            f"same_vj already means same_v and same_j; passing it with {' and '.join(sugar)} "
            "says the same thing twice and could say two different things. Pass one or the other."
        )
    if group_cols is not None and (same_vj or sugar):
        named = ", ".join(["same_vj"] if same_vj else sugar)
        raise ValueError(f"pass group_cols or {named}, not both")
    if group_cols is not None:
        return list(group_cols)
    if same_vj:
        return ["vgene", "jgene"]
    resolved = (["vgene"] if same_v else []) + (["jgene"] if same_j else [])
    return resolved or None


def plot_seq_graph(
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
        node_color_attribute raises if no node carries it at all. Nodes that individually lack a
            value are drawn in other_color rather than rejected, since a missing value can be
            real data.
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
            because a sequence would have to match G's node order and build_seq_hamming_graph builds
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
        if all(v is None for v in node_color_vals):
            raise ValueError(
                f"no node carries a {node_color_attribute!r} attribute, so every node would be "
                "drawn in one colour; pass it in build_seq_hamming_graph's attribute_columns"
            )
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
    # A grouped graph's nodes are (sequence, *group values); the frame gets a column each, so it
    # still merges back onto the rows it came from. G records the grouping it was built with.
    group_cols = list(G.graph.get("group_cols") or [])
    columns = [seq_column] + group_cols

    pdfs = []
    nodes_in_ccs = set()
    for idx, seqs in enumerate(nx.connected_components(G)):
        if len(seqs) >= min_seqs:
            rows = [node if group_cols else (node,) for node in seqs]
            pdf_one_cc = pd.DataFrame(rows, columns=columns)
            pdf_one_cc["connected_component"] = idx
            pdfs.append(pdf_one_cc)
            nodes_in_ccs.update(seqs)

    if len(pdfs) == 0:
        return pd.DataFrame(columns=columns + ["connected_component"])
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

