# portmanteau

Small, portable Python utilities for data analysis: cached downloads, ROC metrics and
plots, Hamming-1 sequence graphs, and styling for plots.

## Install

Install from a tagged release, choosing the extras you need:

```bash
uv add "portmanteau[download,plots] @ git+https://github.com/dhmay/portmanteau@v0.1.1"
```

Or, if you're feeling lucky, install from main:

```bash
uv add "portmanteau[plots] @ git+https://github.com/dhmay/portmanteau" --branch main
```

If you do that, to grab the latest:

```bash
uv lock --upgrade-package portmanteau && uv sync
```

## Dependencies

The only required dependency is pandas. Importing individual modules incurs additional dependencies.

| Extra      | Modules                                         | Adds                               |
|------------|-------------------------------------------------|------------------------------------|
| `download` | `portmanteau.data.download`                     | pooch                              |
| `metrics`  | `portmanteau.metrics.roc`                       | scikit-learn                       |
| `plots`    | `portmanteau.plots.*`                           | matplotlib, seaborn, plotly, scikit-learn |
| `sequence` | `portmanteau.sequence.*`                        | networkx, matplotlib, seaborn      |
| `notebook` | `plots.sankey.display_plotly_html`              | ipython                            |


## Usage

**Downloads** are cached, so repeated calls return the local file without using the
network. Pass `known_hash` to verify the contents.

```python
from portmanteau.data.download import fetch, fetch_zip

path = fetch("https://example.org/data.csv", known_hash="sha256:...")  # -> Path
files = fetch_zip("https://example.org/archive.zip")                    # -> list[Path]
```

Files go to the OS cache directory (`~/Library/Caches/portmanteau` on macOS), or to
`$PORTMANTEAU_DATA_DIR` or `dest_dir=` if set.

**ROC metrics and plots:**

```python
from portmanteau.metrics.roc import compute_roc
from portmanteau.plots.roc import plot_roc, boxplot_and_roc

roc = compute_roc(df, "score", label_col="label", pos_label=1)
roc.auroc, roc.sensitivity_at(0.95), roc.sensitivities_at([0.9, 0.99])

fig, ax = plot_roc(df, "score_a", label="A", add_auroc_to_label=True)
plot_roc(df, "score_b", ax=ax, label="B", ls="--", lw=2)  # extra kwargs style the line
fig, (box_ax, roc_ax) = boxplot_and_roc(df, "score")
```

**Hamming-1 sequence graphs:**

```python
from portmanteau.sequence.hamming1_pairs import find_cdr3_hamming1_pairs
from portmanteau.sequence.hamming1_graph import build_seq_ham1_graph_and_extract_ccs

pairs = find_cdr3_hamming1_pairs(df, seq_col="cdr3")
df_with_ccs = build_seq_ham1_graph_and_extract_ccs(df, seq_column="cdr3", min_seqs=2)
```

**Plot style:** `portmanteau.plots.style.set_style()` makes the colorblind-safe palette the
default for matplotlib, seaborn and plotly.

## Development

```bash
uv sync --all-extras
uv run pytest
```

## License

MIT
