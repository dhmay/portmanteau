import logging
from bisect import bisect_left
from typing import Any, List, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve

DEFAULT_ROC_SUMMARY_SPECIFICITIES = (0.9, 0.95, 0.99)

logger = logging.getLogger(__name__)


def calc_roc_fpr_tpr_thresh(pdf_predictions, column, label_col: str = "label") -> Tuple:
    """Calculate ROC FPR, TPR, and thresholds for a given measurand column
    in a Pandas DataFrame. Just a convenience method around sklearn.metrics.roc_curve.
    Args:
        pdf_predictions: DataFrame with the predictions and the labels
        column: column with the predictions
        label_col: label column. Defaults to "label".
    Returns:
        fpr, tpr, thresholds: as returned by sklearn.metrics.roc_curve
    """
    pdf_predictions_forplot = pdf_predictions[~(pdf_predictions[label_col].isnull())]
    fpr, tpr, thresh = roc_curve(
        pdf_predictions_forplot[label_col], pdf_predictions_forplot[column]
    )
    return fpr, tpr, thresh


def calc_auroc(pdf_predictions, column, label_col: str = "label") -> float:
    """Calculate AUROC for a given measurand column in a Pandas DataFrame.
    Just a convenience method around sklearn.metrics.roc_auc_score that drops
    rows with missing labels.
    Args:
        pdf_predictions: DataFrame with the predictions and the labels
        column: column with the predictions
        label_col: label column. Defaults to "label".
    Returns:
        AUROC as a float
    """
    pdf_labeled = pdf_predictions[pdf_predictions[label_col].notnull()]
    return roc_auc_score(pdf_labeled[label_col], pdf_labeled[column])


def calc_roc_summary(
    pdf_predictions,
    column,
    label_col: str = "label",
    specificities=DEFAULT_ROC_SUMMARY_SPECIFICITIES,
) -> dict:
    """Calculate a dict of ROC summary data for a given measurand and label column
    in a Pandas DataFrame.
    Args:
        pdf_predictions: DataFrame with the predictions and the labels
        column: column with the predictions
        label_col: label column. Defaults to "label".
        specificities: list or tuple of specificities at which to calculate sensitivities.
            Defaults to (0.9, 0.95, 0.99).
    Returns:
        dict with keys:
            "fpr": array of false positive rates
            "tpr": array of true positive rates
            "auroc": area under the ROC curve
            "senses_at_specs": list of sensitivities at the specified specificities
    """
    fpr, tpr, _ = calc_roc_fpr_tpr_thresh(pdf_predictions, column, label_col=label_col)
    auroc = auc(fpr, tpr)
    pdf_predictions_withlabel = pdf_predictions[~(pdf_predictions[label_col].isnull())]
    predictions = pdf_predictions_withlabel[column]
    labels = pdf_predictions_withlabel[label_col]
    senses_at_specs = [
        sensitivity_at_specificity(spec, labels, predictions) for spec in specificities
    ]
    roc_summary_dict = {
        "fpr": fpr,
        "tpr": tpr,
        "auroc": auroc,
        "senses_at_specs": senses_at_specs,
    }
    logger.debug(
        f"ROC summary: AUROC={auroc}, Sensitivities at {specificities}={senses_at_specs}"
    )
    return roc_summary_dict


def plot_roc(
    pdf_predictions: pd.DataFrame,
    measurand_col: str,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    label_col: str = "label",
    title: Optional[str] = None,
    plot_1to1: bool = True,
    add_auc_to_label: bool = False,
    add_auc_to_title: bool = False,
    diag_kws: Optional[dict] = None,
    **line_kws,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the ROC curve for a given measurand column in a Pandas DataFrame.

    If ax is provided, plot on that axes. Otherwise, create a new figure.

    Args:
        pdf_predictions (pd.DataFrame): dataframe with the predictions and the labels
        measurand_col (str): column with the predictions
        ax (plt.Axes, optional): axes. Defaults to None.
        label (str, optional): legend text for the ROC line (not the class labels; see
            label_col). If given, a legend is drawn. Defaults to None.
        label_col (str, optional): column of true binary class labels. Defaults to "label".
        title (str, optional): title for the axes. Defaults to None.
        plot_1to1 (bool, optional): plot the 1:1 line, dashed and grey. Defaults to True.
        add_auc_to_label (bool, optional): append the AUROC to the line label. Defaults to False.
        add_auc_to_title (bool, optional): append the AUROC to the title. Defaults to False.
        diag_kws (dict, optional): properties for the 1:1 line, overriding the default
            dashed gray, e.g. {"color": "black", "lw": 0.5}. Defaults to None.
        **line_kws: properties for the ROC line, passed to Axes.plot, e.g. color,
            linestyle/ls, linewidth/lw, alpha, marker, zorder.

    Returns:
        Tuple[plt.Figure, plt.Axes]: figure and axes
    """

    # ensure that the label column is an integer
    pdf_predictions = pdf_predictions[pdf_predictions[label_col].notnull()].copy()
    pdf_predictions[label_col] = pdf_predictions[label_col].astype(int)

    logger.debug(
        f"Label counts: 0={sum(pdf_predictions[label_col] == 0)}, 1={sum(pdf_predictions[label_col] == 1)}"
    )
    roc_summary_dict = calc_roc_summary(pdf_predictions, measurand_col, label_col=label_col)
    fprs = roc_summary_dict["fpr"]
    tprs = roc_summary_dict["tpr"]
    auroc = roc_summary_dict["auroc"]
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
    if add_auc_to_label:
        label = f"AUROC {auroc:.3f}" if label is None else f"{label} ({auroc:.3f})"
    ax.plot(fprs, tprs, label=label, **line_kws)
    if label is not None:
        ax.legend()
    if plot_1to1:
        ax.plot([0, 1], [0, 1], **{"linestyle": "--", "color": "gray", **(diag_kws or {})})
    ax.set_ylabel("TPR")
    ax.set_xlabel("FPR")
    if title is not None:
        if add_auc_to_title:
            title += f" (AUROC: {auroc:.3f})"
        ax.set_title(title)
    return f, ax


def boxplot_and_roc(
    pdf_predictions: pd.DataFrame,
    measurand_col: str,
    label_col: str = "label",
    boxplot_ylabel: Optional[str] = None,
    boxplot_xlabel: str = "Label",
    add_auroc_to_title: bool = True,
    boxplot_title: Optional[str] = None,
    suptitle: Optional[str] = None,
    use_violins: bool = False,
    plot_roc_1to1: bool = True,
    roc_kws: Optional[dict] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot a boxplot of the measurand and the ROC curve for a given measurand column in a Pandas DataFrame.

    Args:
        pdf_predictions (pd.DataFrame): dataframe with the predictions and the labels
        measurand_col (str): column with the predictions
        label_col (str, optional): column of true binary class labels. Defaults to "label".
        boxplot_ylabel (str, optional): y-axis label for the boxplot. Defaults to None.
        boxplot_xlabel (str, optional): x-axis label for the boxplot. Defaults to "Label".
        add_auroc_to_title (bool, optional): add AUROC to the ROC title. Defaults to True.
        boxplot_title (str, optional): title for the boxplot. Defaults to "<measurand_col> by label".
        suptitle (str, optional): title for the figure. Defaults to None.
        use_violins (bool, optional): use violins instead of boxplots. Defaults to False.
        plot_roc_1to1 (bool, optional): plot the 1:1 line on the ROC plot. Defaults to True.
        roc_kws (dict, optional): extra keyword arguments for plot_roc, e.g. label, diag_kws,
            or line properties like color and lw. Can't include ax, label_col or
            plot_1to1. Defaults to None.

    Returns:
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]: figure and axes
    """

    # ensure that the label column is an integer
    pdf_predictions = pdf_predictions[pdf_predictions[label_col].notnull()].copy()
    pdf_predictions[label_col] = pdf_predictions[label_col].astype(int)

    logger.debug(
        f"Label counts: 0={sum(pdf_predictions[label_col] == 0)}, 1={sum(pdf_predictions[label_col] == 1)}"
    )
    f, axes = plt.subplots(1, 2)

    ax = axes[0]
    if use_violins:
        sns.violinplot(x=label_col, y=measurand_col, data=pdf_predictions, ax=ax, cut=0)
    else:
        sns.boxplot(x=label_col, y=measurand_col, data=pdf_predictions, ax=ax)
    if boxplot_ylabel is None:
        boxplot_ylabel = measurand_col
    ax.set_ylabel(boxplot_ylabel)
    ax.set_xlabel(boxplot_xlabel)
    if boxplot_title is None:
        boxplot_title = f"{measurand_col} by label"
    ax.set_title(boxplot_title)

    ax = axes[1]
    plot_roc(
        pdf_predictions,
        measurand_col,
        ax=ax,
        label_col=label_col,
        plot_1to1=plot_roc_1to1,
        **(roc_kws or {}),
    )
    roc_title = "ROC"
    if add_auroc_to_title:
        roc_title += f" (AUROC: {calc_auroc(pdf_predictions, measurand_col, label_col):.3f})"
    ax.set_title(roc_title)

    if suptitle is not None:
        f.suptitle(suptitle)
    f.set_size_inches(12, 6)
    f.tight_layout()
    return f, axes


def sensitivity_at_specificity(spec_req: float, labels: List[Any], scores: List[Any]):
    """Calculate sensitivity at a given specificity using ROC curve.

    This function uses roc_curve() to calculate fpr and tpr, then uses
    bisect_left on (1 - specificity) to find the sensitivity at the required specificity.
    bisect_left is conservative in that it'll return the smallest index where fpr >= fpr_req.
    I.e., it may return a sensitivity slightly higher than the exact value.

    Args:
        spec_req (float): required specificity
        labels (List[Any]): true binary labels
        scores (List[Any]): predicted scores
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    # Calculate the required false positive rate
    fpr_req = 1.0 - spec_req

    # Find the index of the closest false positive rate, not going over
    idx = max(0, min(len(fpr) - 1, bisect_left(fpr, fpr_req)))
    # if the fpr at that index is greater than the required fpr,
    # step back one index
    if fpr[idx] > fpr_req and idx > 0:
        idx -= 1  # step back to the last fpr below fpr_req

    # Return the corresponding sensitivity (true positive rate)
    return tpr[idx]

