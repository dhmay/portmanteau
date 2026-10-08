# portmanteau

Small, portable Python utilities for data analysis: cached downloads, ROC metrics and
plots, Hamming-1 sequence graphs, styling for plots, and slide decks built from notebooks.

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
| `slides`   | `portmanteau.slides.notebook_deck`              | python-pptx, pillow, lxml, nbformat |


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

**Decks from notebooks.** `portmanteau.slides.notebook_deck` renders an executed notebook as a
pptx, so every number on a slide is produced by code sitting beside the prose describing it, and a
slide is edited by editing a cell. The notebook's headings are the format:

```
# Deck title       the opening section slide (that line only)
## Slide title     one slide
### prep           code whose output is never rendered
### text           markdown, which becomes the bullets
### images         code whose outputs become pictures and tables
### notes          markdown, which becomes the speaker notes
```

In the notebook's first code cell:

```python
from portmanteau.slides.notebook_deck import init_slide_nb
init_slide_nb()
```

and then, from a script:

```python
from portmanteau.slides.notebook_deck import build_deck

build_deck("chapter1.ipynb", out="chapter1.pptx", template="house_style.pptx")
```

`init_slide_nb` exists because a deck is built from what the notebook *displayed*: pandas truncates
a table cell past 50 characters by default, and the abbreviation then appears on the slide looking
deliberate. It also warns if matplotlib is on a backend that will not display figures.

**Several notebooks make one deck**, in the order given, each opening with its own `# ` title as a
section slide - which is how a talk written in chunks is stitched together:

```python
build_deck(["part1.ipynb", "part2.ipynb"], out="talk.pptx")
```

Building the combined deck from the notebooks, rather than merging the separate decks, means every
guard runs over the result and there is no pptx surgery to go wrong.

Stored outputs are used as they are; nothing is re-executed, so a build takes about a second. The
cost is trusting the last kernel run, and the builder refuses a notebook that cannot have been run
cleanly - any error output, any unexecuted cell, any `### images` cell that produced nothing.

Layout is fixed rather than configurable: bullets left, figures right (one centred, two stacked), a
table instead going full width beneath the bullets. Three figures, a table and a figure together,
text that overflows its box, or a table running off the slide all raise `DeckError` naming the slide
and section - a generated deck fails silently otherwise, and the errors say which cell to change.

[`examples/minimal_deck.ipynb`](examples/minimal_deck.ipynb) is a four-slide notebook demonstrating
every layout and markdown rule. It is committed executed, so it builds without being re-run:

```bash
uv run python -c "from portmanteau.slides.notebook_deck import build_deck; \
    build_deck('examples/minimal_deck.ipynb', out='minimal_deck.pptx')"
```

One gotcha it documents: figures are picked up from a cell's *displayed output*, which under a
Jupyter kernel is the inline backend's doing. Forcing a non-interactive backend such as Agg silences
that, and the build then fails with "cell produced no output".

## Development

```bash
uv sync --all-extras
uv run pytest
```

## License

MIT
