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

**Close sequence pairs.** `find_hamming_pairs` returns one row per pair of rows whose sequences
are a given Hamming distance apart, with the differing positions:

```python
from portmanteau.sequence.hamming_pairs import find_hamming_pairs, find_hamming1_pairs, annotate_pairs

# every pair of CDR3s differing at exactly one position
pairs = find_hamming1_pairs(df, "cdr3")

# two positions, and only between rows sharing a V and J gene
pairs = find_hamming_pairs(df, "cdr3", distance=2, group_cols=["vgene", "jgene"])

# at most two positions, adding a hamming_distance column
pairs = find_hamming_pairs(df, "cdr3", distance=2, exact=False)

# carry other columns onto both sides, either afterwards or inline
pairs = annotate_pairs(pairs, df, columns=["peptide", "subject"])          # adds _i and _j columns
pairs = find_hamming1_pairs(df, "cdr3", annotation_columns=["peptide"])    # same, in one call
```

Pair counts grow **quadratically**: ten times the sequences gives roughly a hundred times the
pairs. `group_cols` is the lever against that, since pairing within each of k groups costs
`k * (n/k)^2` rather than `n^2`. Where the result is still too large to hold, `iter_hamming_pairs`
yields it in chunks, and `find_hamming_pairs(..., max_pairs=...)` raises instead of exhausting
memory.

**Hamming-1 sequence graphs:**

```python
from portmanteau.sequence.hamming1_graph import build_seq_ham1_graph_and_extract_ccs

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
