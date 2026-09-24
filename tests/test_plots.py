import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
pio = pytest.importorskip("plotly.io")


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
