"""
Default plotting palette for portmanteau, and a helper to make it the
default for seaborn/matplotlib and plotly.

The categorical palette is colorblind-safe: adjacent slots pass a
CVD-simulated separation check (OKLab deltaE >= 8) and a normal-vision
separation floor (>= 15), so series remain distinguishable to both
colorblind and full-color readers. Assign colors to series in this
fixed order -- never reorder or cycle past slot 8; fold additional
series into "Other" or facet instead.
"""

import inspect

import matplotlib as mpl
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import matplotlib_venn
    _HAS_VENN = True
except ImportError:
    _HAS_VENN = False

# Categorical: identity encoding (one color per series/group).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Sequential: magnitude encoding (heatmaps, continuous scales), light -> dark.
SEQUENTIAL_PALETTE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#1c5cab", "#104281", "#0d366b",
]

# Diverging: polarity encoding, blue <-> red around a neutral gray midpoint.
DIVERGING_PALETTE = [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb",
    "#f0efec",
    "#f8d6d5", "#ec8f8e", "#e34948", "#8a1f1e",
]

# venn2/venn3 have exactly two/three circles, so they use the first two/three
# categorical slots -- the only slots validated for *all-pairs* separation
# (not just adjacent), which matters since every circle overlaps every other.
VENN2_COLORS = tuple(CATEGORICAL_PALETTE[:2])
VENN3_COLORS = tuple(CATEGORICAL_PALETTE[:3])

SEQUENTIAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "portmanteau_sequential", SEQUENTIAL_PALETTE)
DIVERGING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "portmanteau_diverging", DIVERGING_PALETTE)

# Register by name so rcParams (e.g. "image.cmap") and cmap="..." lookups
# resolve; force=True makes re-running set_style() in a notebook idempotent.
mpl.colormaps.register(cmap=SEQUENTIAL_CMAP, force=True)
mpl.colormaps.register(cmap=DIVERGING_CMAP, force=True)


def _patch_default(func, **overrides) -> None:
    """
    Replace one or more of a function's default argument values in
    place, so the new default applies even where the function was
    imported (`from module import func`) before this call runs.
    """
    defaulted_params = [
        p for p in inspect.signature(func).parameters.values()
        if p.default is not inspect.Parameter.empty
    ]
    defaults = list(func.__defaults__)
    for name, value in overrides.items():
        defaults[[p.name for p in defaulted_params].index(name)] = value
    func.__defaults__ = tuple(defaults)


def set_style() -> None:
    """
    Make the portmanteau palette the default for seaborn/matplotlib
    (and plotly, if installed) for the rest of the session.
    """
    sns.set_theme(style="white", palette=CATEGORICAL_PALETTE)
    sns.set_context("talk")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CATEGORICAL_PALETTE)
    mpl.rcParams["image.cmap"] = "portmanteau_sequential"

    if _HAS_PLOTLY:
        # Copy, so plotly's built-in plotly_white template isn't modified
        template = go.layout.Template(pio.templates["plotly_white"])
        template.layout.colorway = CATEGORICAL_PALETTE
        pio.templates["portmanteau"] = template
        pio.templates.default = "portmanteau"
        px.defaults.color_discrete_sequence = CATEGORICAL_PALETTE
        px.defaults.color_continuous_scale = SEQUENTIAL_PALETTE

    if _HAS_VENN:
        _patch_default(matplotlib_venn.venn2, set_colors=VENN2_COLORS)
        _patch_default(matplotlib_venn.venn3, set_colors=VENN3_COLORS)
        _patch_default(matplotlib_venn.venn2_unweighted, set_colors=VENN2_COLORS)
        _patch_default(matplotlib_venn.venn3_unweighted, set_colors=VENN3_COLORS)
