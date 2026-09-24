import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
pytest.importorskip("sklearn")
pytest.importorskip("seaborn")

from portmanteau.metrics.roc import calc_auroc  # noqa: E402
from portmanteau.plots.roc import boxplot_and_roc, plot_roc  # noqa: E402


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame({"label": rng.integers(0, 2, 200).astype(float), "score": rng.normal(size=200)})
    pdf.loc[pdf.label == 1, "score"] += 1
    pdf.loc[::17, "label"] = np.nan
    return pdf


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_options_are_keyword_only(predictions):
    with pytest.raises(TypeError):
        plot_roc(predictions, "score", "label")
    with pytest.raises(TypeError):
        boxplot_and_roc(predictions, "score", "label")


def test_plot_roc_labels_and_title(predictions):
    auroc = calc_auroc(predictions, "score")
    _, ax = plot_roc(predictions, "score", add_auroc_to_label=True)
    assert legend_texts(ax) == [f"AUROC: {auroc:.3f}"]
    assert ax.get_title() == ""

    _, ax = plot_roc(predictions, "score", label="m", title="T", add_auroc_to_label=True, add_auroc_to_title=True)
    assert ax.get_title() == f"T (AUROC: {auroc:.3f})"
    assert legend_texts(ax) == [f"m (AUROC: {auroc:.3f})"]

    _, ax = plot_roc(predictions, "score", add_auroc_to_title=True)
    assert ax.get_title() == f"AUROC: {auroc:.3f}"


def test_plot_roc_line_kws(predictions):
    _, ax = plot_roc(predictions, "score", ls=":", lw=3, alpha=0.5, color="red")
    roc_line, diag = ax.get_lines()
    assert roc_line.get_linestyle() == ":"
    assert roc_line.get_linewidth() == 3
    assert roc_line.get_alpha() == 0.5
    assert roc_line.get_color() == "red"
    assert ax.get_legend() is None  # no label, no legend
    assert (diag.get_linestyle(), diag.get_color()) == ("--", "gray")


def test_plot_roc_diag_kws_override_defaults_partially(predictions):
    _, ax = plot_roc(predictions, "score", diag_kws={"color": "black", "lw": 0.5})
    diag = ax.get_lines()[1]
    assert (diag.get_linestyle(), diag.get_color(), diag.get_linewidth()) == ("--", "black", 0.5)


def test_plot_roc_multiple_curves_share_legend(predictions):
    _, ax = plot_roc(predictions, "score", label="a", plot_1to1=False)
    plot_roc(predictions, "score", ax=ax, label="b", ls="--")
    assert legend_texts(ax) == ["a", "b"]


def test_plot_roc_pos_label(predictions):
    labeled = predictions.dropna()
    strs = labeled.assign(label=labeled.label.map({0.0: "control", 1.0: "case"}))
    _, ax = plot_roc(strs, "score", pos_label="case", add_auroc_to_title=True)
    assert ax.get_title() == f"AUROC: {calc_auroc(labeled, 'score'):.3f}"


def test_plot_roc_does_not_modify_input(predictions):
    before = predictions.copy()
    plot_roc(predictions, "score")
    boxplot_and_roc(predictions, "score")
    pd.testing.assert_frame_equal(predictions, before)


def test_boxplot_and_roc_defaults(predictions):
    f, (box_ax, roc_ax) = boxplot_and_roc(predictions, "score")
    assert box_ax.get_title() == "score by label"
    assert box_ax.get_ylabel() == "score"
    assert [t.get_text() for t in box_ax.get_xticklabels()] == ["0", "1"]
    assert roc_ax.get_title() == f"ROC (AUROC: {calc_auroc(predictions, 'score'):.3f})"
    assert tuple(f.get_size_inches()) == (12, 6)


def test_boxplot_and_roc_options(predictions):
    f, (box_ax, roc_ax) = boxplot_and_roc(
        predictions,
        "score",
        boxplot_title="custom",
        box_kws={"showfliers": False},
        roc_kws={"title": "Mine", "add_auroc_to_title": False, "label": "m", "lw": 4,
                 "diag_kws": {"color": "black"}},
        figsize=(8, 4),
    )
    assert box_ax.get_title() == "custom"
    assert roc_ax.get_title() == "Mine"
    roc_line, diag = roc_ax.get_lines()
    assert roc_line.get_linewidth() == 4
    assert diag.get_color() == "black"
    assert legend_texts(roc_ax) == ["m"]
    assert tuple(f.get_size_inches()) == (8, 4)


def test_boxplot_and_roc_violins(predictions):
    _, (box_ax, _) = boxplot_and_roc(predictions, "score", use_violins=True, box_kws={"inner": None})
    assert box_ax.collections  # violins drawn
