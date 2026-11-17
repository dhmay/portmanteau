"""Normalizing TCR V and J gene names to allele-free IMGT.

The same gene reaches us under several names: the alpha/delta genes have older names without
their "/DVn" part (TRAV14 is TRAV14/DV4), beta J genes are often written family-dot-member
(TRBJ2.1 for TRBJ2-1), and databases add their own typos on top. None of those join to the
"TRAV14/DV4" scirpy gives us for the scRNA data.

**tidytcells does the resolving.** It ships IMGT's human gene list — 250 genes — which a regex
cannot replicate: it knows TRBV21 is not a gene but TRBV21-1 is, and that TRBV17-1 is not a gene
but TRBV17 is. We pin it below 3 because its own docs say the `species` default changes there,
and because the bundled gene list is what we are relying on.

Around it sit two corrections, both measured against the 248 distinct gene tokens in VDJdb
2026-06-03 and locked by tests:

- **unicode dashes are repaired first.** Handed "TRBJ2–3" (en-dash) tidytcells returns *TRBJ2-1* —
  a different, real gene, with no error. A silently wrong answer is worse than a rejection, so the
  dash never reaches it. This guards against the engine, not against any one database, which is
  why it lives here.
- **TRAV14-1 is remapped afterwards.** tidytcells maps TRAV23-1, TRAV29-1, TRAV36-1 and TRAV38-2
  to their /DV names but leaves TRAV14-1 alone, and `tt.tr.get_aa_sequence("TRAV14-1")` raises
  "No data found" — so it is a spurious entry in its list, not a gene. Join-critical for us,
  because the scRNA side spells it TRAV14/DV4.

One known quirk we accept rather than fight: tidytcells maps the bare "TRBV8" to TRBV8-1, though
TRBV8-1 and TRBV8-2 both exist and nothing in the name chooses between them. It affects 20 VDJdb
rows of a pseudogene that will not match functional scRNA data.

gene_family() is this module's own — tidytcells standardizes to allele or gene, with no notion
of a family, and a family is what you have to match on when one side of a join carries only
family-level calls.
"""
import re
from collections.abc import Callable

import pandas as pd
import tidytcells as tt

#: Passed to tidytcells on every call. The species default changes in v3, and we want pseudogenes
#: and ORFs kept (enforce_functional is False by default): we are matching names, not asserting
#: that a gene is expressed.
TIDYTCELLS_SPECIES = "homosapiens"

# A V gene's "/DVn" or "/ORn-n" composite suffix, stripped before the family's "-n" suffix.
COMPOSITE_SUFFIX_PATTERN = r"/(?:DV|OR)\d+(?:-\d+)?$"
# What's left of a gene name once its family is removed, e.g. the "-1" of TRBV20-1.
FAMILY_SUFFIX_PATTERN = r"-.*$"
# Separators seen between the candidate genes of one ambiguous call. tidytcells silently returns
# only the first gene of such a call, so we split before it sees them.
CALL_SEPARATOR_PATTERN = r"[,;]"

# Unicode dashes that must become an ASCII hyphen before tidytcells sees them; see the module
# docstring for what happens otherwise.
UNICODE_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})

# Applied to tidytcells' output. Only for names it resolves to something that is not an IMGT gene.
GENE_NAME_OVERRIDES = {"TRAV14-1": "TRAV14/DV4"}


def normalize_gene(name: str | None, clean: Callable[[str], str] | None = None) -> str | None:
    """Return the allele-free IMGT name for one V or J gene call, or None if it can't be resolved.

    Every human TR locus tidytcells knows is recognized, gamma and delta included. Our sources
    are filtered to alpha/beta upstream, so in practice only those appear.

    Args:
        name: a single gene name, e.g. "TRAV12-2*01". Not a multi-gene call — split those on
            CALL_SEPARATOR_PATTERN first, or use explode_gene_calls().
        clean: optional source-specific repair applied to the raw name before anything else, for
            a defect this module does not know about. VDJdb needs none: tidytcells already
            handles its internal spaces, trailing underscores, dotted beta J names and
            zero-padded TRAJ names. Kept for the databases still to come.

    Returns:
        The canonical allele-free IMGT name, e.g. "TRAV12-2"; None for a missing, malformed or
        non-existent gene.
    """
    if name is None or pd.isna(name):
        return None
    name = str(name)
    if clean is not None:
        name = clean(name)
    standardized = tt.tr.standardize(
        name.translate(UNICODE_DASHES),
        species=TIDYTCELLS_SPECIES,
        precision="gene",
        on_fail="reject",
        log_failures=False,
    )
    return GENE_NAME_OVERRIDES.get(standardized, standardized)


def gene_family(name: str | None) -> str | None:
    """Return the IMGT family of one normalized gene name, or None.

    The composite suffix goes first, so TRAV38-2/DV8 and TRAV14/DV4 both reduce to their IMGT
    family (TRAV38, TRAV14) rather than keeping or losing the /DV part depending on whether the
    gene happens to have a "-n". TRAJ genes have no families and return None.

    Args:
        name: an allele-free IMGT gene name, as normalize_gene() returns.

    Returns:
        The family, e.g. "TRBV20" for TRBV20-1; None if name is None or has no family.
    """
    if name is None or pd.isna(name):
        return None
    if str(name).startswith("TRAJ"):
        return None
    return re.sub(FAMILY_SUFFIX_PATTERN, "", re.sub(COMPOSITE_SUFFIX_PATTERN, "", str(name)))


def normalize_genes(calls: pd.Series, clean: Callable[[str], str] | None = None) -> pd.Series:
    """Vectorized normalize_gene() over a Series of single gene calls, with pd.NA for the failures.

    Resolves each distinct call once: a tidytcells lookup per row would cost half a million calls
    on VDJdb for 248 distinct answers.
    """
    distinct = calls.dropna().unique()
    resolved = {call: normalize_gene(call, clean) for call in distinct}
    return calls.map(resolved).astype("string")


def gene_families(genes: pd.Series) -> pd.Series:
    """Vectorized gene_family() over a Series of normalized gene names."""
    return genes.map(gene_family).astype("string")


def explode_gene_calls(
    pdf: pd.DataFrame, col: str, out_col: str, clean: Callable[[str], str] | None = None
) -> pd.DataFrame:
    """Return one row per candidate gene of an ambiguous call, normalized, in a new column.

    A call naming several genes ("TRAV1-1*01,TRAV1-2*01") becomes one row per gene. Rows whose
    call can't be normalized are kept with out_col set to NA, so a row is never lost for having a
    bad call in one of its chains.

    Args:
        pdf: any frame with a gene-call column.
        col: the column holding the calls, which may name several genes separated by "," or ";".
        out_col: name of the normalized single-gene column to add.
        clean: optional source-specific repair, passed through to normalize_gene().

    Returns:
        A copy of pdf with out_col added, one row per candidate gene. The index repeats for
        multi-gene calls.
    """
    split = pdf[col].astype(object).map(
        lambda call: [None] if call is None or pd.isna(call) else re.split(CALL_SEPARATOR_PATTERN, call)
    )
    exploded = pdf.assign(**{out_col: split}).explode(out_col)
    return exploded.assign(**{out_col: normalize_genes(exploded[out_col], clean)})
