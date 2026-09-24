import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from portmanteau.metrics.roc import (  # noqa: E402
    calc_auroc,
    compute_roc,
    labeled_rows,
    sensitivity_at_specificity,
)


def brute_force_sensitivity(labels, scores, specificity):
    """Highest sensitivity over every threshold meeting the specificity."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    neg, pos = labels == 0, labels == 1
    return max(
        (scores >= t)[pos].mean()
        for t in np.unique(np.r_[scores, np.inf])
        if (scores < t)[neg].mean() >= specificity - 1e-12
    )


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame({"label": rng.integers(0, 2, 200).astype(float), "score": rng.normal(size=200)})
    pdf.loc[pdf.label == 1, "score"] += 1
    pdf.loc[::17, "label"] = np.nan
    return pdf


def test_compute_roc(predictions):
    labeled = predictions.dropna()
    roc = compute_roc(predictions, "score")
    assert roc.auroc == pytest.approx(roc_auc_score(labeled.label, labeled.score))
    assert (roc.n_pos, roc.n_neg) == ((labeled.label == 1).sum(), (labeled.label == 0).sum())
    assert len(roc.fpr) == len(roc.tpr) == len(roc.thresholds)
    assert calc_auroc(predictions, "score") == roc.auroc


def test_pos_label_with_string_labels(predictions):
    labeled = predictions.dropna()
    strs = labeled.assign(label=labeled.label.map({0.0: "control", 1.0: "case"}))
    assert calc_auroc(strs, "score", pos_label="case") == pytest.approx(calc_auroc(labeled, "score"))
    assert calc_auroc(strs, "score", pos_label="control") == pytest.approx(1 - calc_auroc(labeled, "score"))


def test_bool_labels(predictions):
    labeled = predictions.dropna()
    bools = labeled.assign(label=labeled.label == 1)
    assert calc_auroc(bools, "score", pos_label=True) == pytest.approx(calc_auroc(labeled, "score"))


def test_labeled_rows_validation(predictions):
    assert len(labeled_rows(predictions, "score")) == predictions.label.notnull().sum()
    with pytest.raises(ValueError, match="null scores"):
        labeled_rows(predictions.assign(score=np.nan), "score")
    with pytest.raises(ValueError, match="two classes"):
        labeled_rows(predictions, "score", pos_label="case")
    with pytest.raises(ValueError, match="two classes"):
        labeled_rows(predictions.assign(label=predictions.label.fillna(2)), "score")
    with pytest.raises(ValueError, match="two classes"):
        labeled_rows(predictions[predictions.label == 1], "score")
    # null score in an unlabeled row is fine
    unlabeled_null = predictions.copy()
    unlabeled_null.loc[predictions.label.isnull(), "score"] = np.nan
    labeled_rows(unlabeled_null, "score")


def test_sensitivity_at_exact_specificity():
    # 1 false positive out of 10 negatives is exactly specificity 0.9, despite 1 - 0.9 < 0.1
    labels = [0] * 10 + [1] * 10
    scores = [0] * 9 + [5] + [4] * 5 + [6] * 5
    assert sensitivity_at_specificity(labels, scores, 0.9) == 1.0


def test_sensitivity_at_specificity_uses_collinear_points():
    # (fpr 1/3, tpr 0.6) lies on the line between its neighbors, so roc_curve drops it by default
    labels = np.array([0] * 6 + [1] * 5)
    scores = np.array([0, 0, 2, 3, 4, 6, 1, 3, 4, 5, 6])
    assert 1 / 3 not in roc_curve(labels, scores)[0]
    assert sensitivity_at_specificity(labels, scores, 2 / 3) == pytest.approx(0.6)


def test_sensitivity_at_specificity_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = rng.integers(5, 40)
        labels = rng.integers(0, 2, n)
        labels[:2] = [0, 1]
        scores = rng.integers(0, 8, n).astype(float) if rng.random() < 0.5 else rng.normal(size=n)
        for spec in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
            assert sensitivity_at_specificity(labels, scores, spec) == brute_force_sensitivity(labels, scores, spec)


def test_sensitivities_at_matches_function(predictions):
    labeled = predictions.dropna()
    roc = compute_roc(predictions, "score")
    sens = roc.sensitivities_at()
    assert list(sens) == [0.9, 0.95, 0.99]
    for spec, value in sens.items():
        assert value == sensitivity_at_specificity(labeled.label, labeled.score, spec)


def test_specificity_out_of_range():
    with pytest.raises(ValueError):
        sensitivity_at_specificity([0, 1], [0, 1], 1.5)
