import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
pio = pytest.importorskip("plotly.io")
pytest.importorskip("sklearn")

from portmanteau.plots import roc_utils as ru  # noqa: E402


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    pdf = pd.DataFrame({"label": rng.integers(0, 2, 200).astype(float), "score": rng.normal(size=200)})
    pdf.loc[pdf.label == 1, "score"] += 1
    pdf.loc[::17, "label"] = np.nan
    return pdf


def test_sankey_import_has_no_side_effects(monkeypatch):
    monkeypatch.delitem(sys.modules, "portmanteau.plots.sankey", raising=False)
    monkeypatch.setitem(sys.modules, "IPython", None)  # importing sankey shouldn't need IPython
    monkeypatch.setattr(pio.renderers, "default", "browser")
    from portmanteau.plots import sankey

    assert pio.renderers.default == "browser"
    sankey.use_jupyterlab_renderer()
    assert pio.renderers.default == "jupyterlab"


def test_sankey_hovertemplate():
    from portmanteau.plots import sankey

    df = pd.DataFrame({"a": list("xxy"), "b": list("ppq")})
    fig, links = sankey.plot_sankey(df, "a", "b", min_count=1, display_html=False)
    assert fig.data[0].link.hovertemplate.startswith("a %{customdata[0]} → b %{customdata[1]}")
    assert links["value"].tolist() == [2, 1]


def test_set_style_does_not_mutate_plotly_white():
    from portmanteau.plots import style

    white_before = pio.templates["plotly_white"].layout.colorway
    style.set_style()
    assert pio.templates["plotly_white"].layout.colorway == white_before
    assert pio.templates.default == "portmanteau"
    assert list(pio.templates["portmanteau"].layout.colorway) == style.CATEGORICAL_PALETTE
    assert matplotlib.rcParams["image.cmap"] == "portmanteau_sequential"


def test_calc_auroc_matches_roc_curve(predictions):
    from sklearn.metrics import auc

    fpr, tpr, _ = ru.calc_roc_fpr_tpr_thresh(predictions, "score")
    assert ru.calc_auroc(predictions, "score") == pytest.approx(auc(fpr, tpr))


def test_plot_roc_labels_and_title(predictions):
    auroc = ru.calc_auroc(predictions, "score")
    _, ax = ru.plot_roc(predictions, "score", add_auc_to_label=True)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [f"AUROC {auroc:.3f}"]

    _, ax = ru.plot_roc(predictions, "score", label="m", title="T", add_auc_to_label=True, add_auc_to_title=True)
    assert ax.get_title() == f"T (AUROC: {auroc:.3f})"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [f"m ({auroc:.3f})"]


def test_plot_roc_line_kws(predictions):
    _, ax = ru.plot_roc(predictions, "score", ls=":", lw=3, alpha=0.5, color="red")
    roc_line, diag = ax.get_lines()
    assert roc_line.get_linestyle() == ":"
    assert roc_line.get_linewidth() == 3
    assert roc_line.get_alpha() == 0.5
    assert roc_line.get_color() == "red"
    assert ax.get_legend() is None  # no label, no legend
    assert (diag.get_linestyle(), diag.get_color()) == ("--", "gray")


def test_plot_roc_diag_kws_override_defaults_partially(predictions):
    _, ax = ru.plot_roc(predictions, "score", diag_kws={"color": "black", "lw": 0.5})
    diag = ax.get_lines()[1]
    assert (diag.get_linestyle(), diag.get_color(), diag.get_linewidth()) == ("--", "black", 0.5)


def test_plot_roc_multiple_curves_share_legend(predictions):
    _, ax = ru.plot_roc(predictions, "score", label="a", plot_1to1=False)
    ru.plot_roc(predictions, "score", ax=ax, label="b", ls="--")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


def test_boxplot_and_roc_roc_kws(predictions):
    _, (_, roc_ax) = ru.boxplot_and_roc(
        predictions, "score", roc_kws={"label": "m", "lw": 4, "diag_kws": {"color": "black"}}
    )
    roc_line, diag = roc_ax.get_lines()
    assert roc_line.get_linewidth() == 4
    assert diag.get_color() == "black"
    assert [t.get_text() for t in roc_ax.get_legend().get_texts()] == ["m"]


def test_plot_roc_does_not_modify_input(predictions):
    before = predictions.copy()
    ru.plot_roc(predictions, "score")
    pd.testing.assert_frame_equal(predictions, before)


def test_boxplot_and_roc_titles(predictions):
    f, _ = ru.boxplot_and_roc(predictions, "score", boxplot_title="custom")
    assert f.axes[0].get_title() == "custom"
    f, _ = ru.boxplot_and_roc(predictions, "score")
    assert f.axes[0].get_title() == "score by label"


def test_sensitivity_at_specificity_is_conservative():
    labels = [0] * 10 + [1] * 10
    scores = list(range(20))
    assert ru.sensitivity_at_specificity(0.9, labels, scores) == 1.0
