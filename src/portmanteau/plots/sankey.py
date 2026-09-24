import html as _html
from typing import Tuple

import pandas as pd
import plotly.io as pio
from plotly import graph_objects as go  # type: ignore


def use_jupyterlab_renderer() -> None:
    """Set plotly's default renderer to "jupyterlab", which is more compatible with Jupyter
    notebooks. Call this in a notebook before creating any plotly figures.
    """
    pio.renderers.default = "jupyterlab"


def plot_sankey(  # type: ignore
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
    *,
    min_count: int | None = 10,
    top_n_links: int | None = None,
    title: str | None = None,
    display_html: bool = True,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Build a Sankey diagram from a row-wise mapping table.

    Parameters
    ----------
    df : DataFrame
        Input table with at least [source_col, target_col] columns.
        Each row represents one item (e.g., one TCR).
    source_col, target_col : str
        Column names for left and right group labels.
    weight_col : str | None
        If None, each row contributes weight 1. Otherwise, sum weight_col.
    min_count : int | None
        Drop links with aggregated value < min_count. Use None to keep all.
    top_n_links : int | None
        Keep only the top N links by value after filtering/sorting.
    title : str | None
        Plot title.
    display_html : bool
        Whether to display the plot as HTML in the notebook. If False, just return the figure

    Returns
    -------
    fig : plotly.graph_objects.Figure
    links : DataFrame with columns [source_col, target_col, value]
    """

    # 1) Aggregate links
    if weight_col is None:
        links = df.groupby([source_col, target_col]).size().reset_index(name="value")  # type: ignore
    else:
        links = df.groupby([source_col, target_col])[weight_col].sum().reset_index(name="value")  # type: ignore

    # 2) Filter + sort
    if min_count is not None:
        links = links[links["value"] >= min_count]
    links = links.sort_values("value", ascending=False)

    if top_n_links is not None:
        links = links.head(top_n_links)

    # 3) Build node list (namespace labels so left/right don't collide)
    src_raw = links[source_col].astype(str)
    tgt_raw = links[target_col].astype(str)

    src_labels = source_col + ":" + src_raw
    tgt_labels = target_col + ":" + tgt_raw

    node_labels = pd.Index(pd.concat([src_labels, tgt_labels]).unique())  # type: ignore
    node_index = {lab: i for i, lab in enumerate(node_labels)}

    link_source = src_labels.map(node_index)
    link_target = tgt_labels.map(node_index)

    # 4) Plotly Sankey
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
                    thickness=14,
                    label=node_labels.tolist(),
                ),
                link=dict(
                    source=link_source.tolist(),
                    target=link_target.tolist(),
                    value=links["value"].tolist(),
                    customdata=list(zip(src_raw.tolist(), tgt_raw.tolist())),
                    hovertemplate=(
                        f"{source_col} %{{customdata[0]}} → {target_col} %{{customdata[1]}}"
                        "<br>count %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        title=title or f"Sankey: {source_col} → {target_col}",
        height=800,
    )
    if display_html:
        display_plotly_html(fig)

    return fig, links


def display_plotly_html(fig: go.Figure) -> None:  # type: ignore
    """
    Display a Plotly figure as an HTML iframe. This is a workaround for some rendering
    issues with Plotly in Jupyter notebooks.
    Args:
        fig: A Plotly figure object.
    """
    from IPython.display import HTML, display

    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)

    # Escape so it can live inside an iframe srcdoc attribute
    srcdoc = _html.escape(html_str, quote=True)

    display(
        HTML(
            f"""
    <iframe
    srcdoc="{srcdoc}"
    style="width: 100%; height: 800px; border: 0;"
    ></iframe>
    """
        )
    )

