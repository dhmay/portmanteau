"""ROC curves, AUROC, and sensitivity at fixed specificity."""

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_SPECIFICITIES = (0.9, 0.95, 0.99)

# An FPR within this of the target counts as meeting it, so that float error in
# 1 - specificity (1 - 0.9 == 0.09999999999999998) doesn't exclude a threshold that
# meets the target exactly (e.g. 1 false positive out of 10 negatives).
_FPR_TOLERANCE = 1e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class RocResult:
    """ROC curve and summary statistics for one score column.

    fpr, tpr and thresholds include every distinct threshold (roc_curve with
    drop_intermediate=False), so sensitivity_at() can find the best one.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float
    n_pos: int
    n_neg: int

    def sensitivity_at(self, specificity: float) -> float:
        """Highest sensitivity at any threshold whose specificity is at least `specificity`."""
        return _sensitivity_at(self.fpr, self.tpr, specificity)

    def sensitivities_at(
        self, specificities: Iterable[float] = DEFAULT_SPECIFICITIES
    ) -> dict[float, float]:
        """sensitivity_at() for each specificity, keyed by specificity."""
        return {spec: self.sensitivity_at(spec) for spec in specificities}


def labeled_rows(
    pdf: pd.DataFrame, score_col: str, *, label_col: str = "label", pos_label: Any = 1
) -> pd.DataFrame:
    """Rows of pdf that have a label, after checking that they're usable for ROC analysis.

    Raises:
        ValueError: if any labeled row has a null score, or the labels aren't exactly two
            classes, one of them pos_label.
    """
    pdf = pdf[pdf[label_col].notnull()]
    if pdf[score_col].isnull().any():
        raise ValueError(
            f"'{score_col}' has null scores in labeled rows; drop or fill them first"
        )
    classes = set(pdf[label_col].unique())
    if len(classes) != 2 or pos_label not in classes:
        raise ValueError(
            f"'{label_col}' must have exactly two classes, one of them "
            f"pos_label={pos_label!r}; found {sorted(classes, key=str)}"
        )
    return pdf


def compute_roc(
    pdf: pd.DataFrame, score_col: str, *, label_col: str = "label", pos_label: Any = 1
) -> RocResult:
    """Compute the ROC curve and AUROC for a score column. Rows with null labels are dropped.

    Args:
        pdf: DataFrame with scores and true class labels.
        score_col: column of scores; higher means more likely positive.
        label_col: column of true binary class labels. Defaults to "label".
        pos_label: value of label_col for the positive class. Defaults to 1.

    Returns:
        RocResult
    """
    pdf = labeled_rows(pdf, score_col, label_col=label_col, pos_label=pos_label)
    y_true = (pdf[label_col] == pos_label).to_numpy()
    scores = pdf[score_col].to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    result = RocResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auroc=float(roc_auc_score(y_true, scores)),
        n_pos=int(y_true.sum()),
        n_neg=int((~y_true).sum()),
    )
    logger.debug(
        "ROC for %s: n_pos=%d, n_neg=%d, AUROC=%.4f",
        score_col, result.n_pos, result.n_neg, result.auroc,
    )
    return result


def calc_auroc(
    pdf: pd.DataFrame, score_col: str, *, label_col: str = "label", pos_label: Any = 1
) -> float:
    """AUROC for a score column. Rows with null labels are dropped."""
    return compute_roc(pdf, score_col, label_col=label_col, pos_label=pos_label).auroc


def sensitivity_at_specificity(
    labels: Iterable[Any], scores: Iterable[float], specificity: float, *, pos_label: Any = 1
) -> float:
    """Highest sensitivity at any threshold whose specificity is at least `specificity`.

    Args:
        labels: true binary class labels.
        scores: scores; higher means more likely positive.
        specificity: required specificity, in [0, 1].
        pos_label: label of the positive class. Defaults to 1.
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=pos_label, drop_intermediate=False)
    return _sensitivity_at(fpr, tpr, specificity)


def _sensitivity_at(fpr: np.ndarray, tpr: np.ndarray, specificity: float) -> float:
    if not 0 <= specificity <= 1:
        raise ValueError(f"specificity must be in [0, 1], got {specificity}")
    # fpr and tpr are non-decreasing, so the last point with fpr <= target has the highest tpr.
    # fpr[0] is always 0, so there's always such a point.
    idx = bisect_right(fpr, 1.0 - specificity + _FPR_TOLERANCE) - 1
    return float(tpr[idx])
