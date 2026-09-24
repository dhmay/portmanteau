"""ROC curve plots."""

from typing import Any, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..metrics.roc import compute_roc, labeled_rows


def plot_roc(
    pdf: pd.DataFrame,
    score_col: str,
    *,
    label_col: str = "label",
    pos_label: Any = 1,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    add_auroc_to_label: bool = False,
    title: Optional[str] = None,
    add_auroc_to_title: bool = False,
    plot_1to1: bool = True,
    diag_kws: Optional[dict] = None,
    **line_kws,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the ROC curve for a score column. Rows with null labels are dropped.

    Args:
        pdf: DataFrame with scores and true class labels.
        score_col: column of scores; higher means more likely positive.
        label_col: column of true binary class labels. Defaults to "label".
        pos_label: value of label_col for the positive class. Defaults to 1.
        ax: axes to plot on. Defaults to a new figure.
        label: legend text for the ROC line (not the class labels; see label_col). If
            given, a legend is drawn. Defaults to None.
        add_auroc_to_label: append the AUROC to the legend text. Defaults to False.
        title: title for the axes. Defaults to None.
        add_auroc_to_title: append the AUROC to the title. Defaults to False.
        plot_1to1: plot the 1:1 line, dashed and gray. Defaults to True.
        diag_kws: properties for the 1:1 line, overriding the default dashed gray,
            e.g. {"color": "black", "lw": 0.5}. Defaults to None.
        **line_kws: properties for the ROC line, passed to Axes.plot, e.g. color,
            linestyle/ls, linewidth/lw, alpha, marker, zorder.

    Returns:
        figure and axes
    """
    roc = compute_roc(pdf, score_col, label_col=label_col, pos_label=pos_label)
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()

    if add_auroc_to_label:
        label = _with_auroc(label, roc.auroc)
    ax.plot(roc.fpr, roc.tpr, label=label, **line_kws)
    if label is not None:
        ax.legend()
    if plot_1to1:
        ax.plot([0, 1], [0, 1], **{"linestyle": "--", "color": "gray", **(diag_kws or {})})
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    if add_auroc_to_title:
        title = _with_auroc(title, roc.auroc)
    if title is not None:
        ax.set_title(title)
    return f, ax


def boxplot_and_roc(
    pdf: pd.DataFrame,
    score_col: str,
    *,
    label_col: str = "label",
    pos_label: Any = 1,
    use_violins: bool = False,
    boxplot_title: Optional[str] = None,
    boxplot_xlabel: str = "Label",
    boxplot_ylabel: Optional[str] = None,
    box_kws: Optional[dict] = None,
    roc_kws: Optional[dict] = None,
    suptitle: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot score distributions by class (left) and the ROC curve (right).

    Rows with null labels are dropped.

    Args:
        pdf: DataFrame with scores and true class labels.
        score_col: column of scores; higher means more likely positive.
        label_col: column of true binary class labels. Defaults to "label".
        pos_label: value of label_col for the positive class. Defaults to 1.
        use_violins: violin plots instead of boxplots. Defaults to False.
        boxplot_title: title for the left panel. Defaults to "<score_col> by label".
        boxplot_xlabel: x-axis label for the left panel. Defaults to "Label".
        boxplot_ylabel: y-axis label for the left panel. Defaults to score_col.
        box_kws: extra keyword arguments for sns.boxplot (or sns.violinplot), e.g.
            showfliers or palette. Can't include data, x, y or ax. Defaults to None.
        roc_kws: extra keyword arguments for plot_roc, e.g. label, title, diag_kws, or
            line properties like color and lw. Can't include ax, label_col or pos_label.
            Defaults to {"title": "ROC", "add_auroc_to_title": True}; entries given here
            override those.
        suptitle: title for the figure. Defaults to None.
        figsize: figure size in inches. Defaults to (12, 6).

    Returns:
        figure, and (distribution axes, ROC axes)
    """
    pdf = labeled_rows(pdf, score_col, label_col=label_col, pos_label=pos_label)
    labels = pdf[label_col]
    # null labels make the column float; show 0/1 rather than 0.0/1.0 on the axis
    if pd.api.types.is_float_dtype(labels) and (labels == labels.round()).all():
        pdf = pdf.assign(**{label_col: labels.astype(int)})

    f, (box_ax, roc_ax) = plt.subplots(1, 2, figsize=figsize)

    if use_violins:
        sns.violinplot(data=pdf, x=label_col, y=score_col, ax=box_ax, **{"cut": 0, **(box_kws or {})})
    else:
        sns.boxplot(data=pdf, x=label_col, y=score_col, ax=box_ax, **(box_kws or {}))
    box_ax.set_title(boxplot_title if boxplot_title is not None else f"{score_col} by label")
    box_ax.set_xlabel(boxplot_xlabel)
    box_ax.set_ylabel(boxplot_ylabel if boxplot_ylabel is not None else score_col)

    plot_roc(
        pdf,
        score_col,
        ax=roc_ax,
        label_col=label_col,
        pos_label=pos_label,
        **{"title": "ROC", "add_auroc_to_title": True, **(roc_kws or {})},
    )

    if suptitle is not None:
        f.suptitle(suptitle)
    f.tight_layout()
    return f, (box_ax, roc_ax)


def _with_auroc(text: Optional[str], auroc: float) -> str:
    return f"AUROC: {auroc:.3f}" if text is None else f"{text} (AUROC: {auroc:.3f})"
