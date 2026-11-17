"""VDJdb-specific handling.

`load()` is the entry point: it ensures the release is downloaded, reshapes vdjdb_full.txt
(the per-clonotype table) into the columns the rest of the project uses, and caches the result, so
a notebook can call it every run. The chain columns are renamed to the short AIRR names so the
two sides join directly. `DROPPED_COLUMNS` records what we leave behind and why.

**There is no gene-call cleaner here, and that is a measured result.** Across the 248 distinct
gene tokens in vdjdb/2026-06-03, portmanteau.tcr.genes resolves every one VDJdb mangles —
internal spaces ("TRBJ 2-1"), trailing underscores ("TRAJ33_"), dotted beta J names ("TRBJ2.1"),
zero-padded TRAJ names ("TRAJ01-1") and the old /DV spellings — without any VDJdb-specific help.
The one defect that needed special handling, the en-dash in "TRBJ2–3", is corrected inside
normalize_gene() because it defeats tidytcells itself rather than being VDJdb's problem. The
`clean` hook stays on those functions for the databases still to come.
"""
from pathlib import Path

import pandas as pd

from portmanteau.tcr.publicdb.common import (
    ANTIGEN_COL,
    DEFAULT_TRB_KEY,
    LIST_SEPARATOR,
    TRB_KEY_CDR3,
    TRB_KEY_V_FAMILY,
    TRB_KEY_V_GENE,
    deduplicate_trb_antigen,
    trb_key_slug,
)
from portmanteau.tcr.publicdb.cache import derived_frame
from portmanteau.tcr.publicdb.species import SpeciesPreset
from portmanteau.tcr.genes import explode_gene_calls, gene_families
from portmanteau.tcr.hla import CLASS_I as HLA_CLASS_I
from portmanteau.tcr.hla import CLASS_II as HLA_CLASS_II
from portmanteau.tcr.hla import hla_classes, hla_field_counts, normalize_hlas

# vdjdb_full.txt's chain columns -> the short AIRR names, so a VDJdb row and a single-cell row
# join on the same column names. Checked against a single-cell AIRR table whose tra_v is
# allele-free and spells the composites TRAV14/DV4 — exactly what normalize_gene() produces.
# `d.beta` is deliberately absent; see DROPPED_COLUMNS.
CHAIN_COLUMNS = {
    "cdr3.alpha": "tra_cdr3", "v.alpha": "tra_v", "j.alpha": "tra_j",
    "cdr3.beta": "trb_cdr3", "v.beta": "trb_v", "j.beta": "trb_j",
}

# VDJdb's own columns we keep, under their original names: a renamed column is one more thing to
# look up when tracing a row back upstream, and only the chain columns need to match the scRNA side.
KEEP_COLUMNS = [
    "antigen.epitope", "antigen.gene", "antigen.species",
    "mhc.a", "mhc.b", "mhc.class",
    "reference.id", "vdjdb.score",
    "method.identification", "method.singlecell", "method.verification",
    "meta.subject.id", "meta.donor.MHC", "meta.cell.subset", "meta.tissue",
]

# Columns vdjdb_full.txt has that we deliberately do not carry, and why. Recorded so the next
# person does not have to rediscover that each one is a dead end.
DROPPED_COLUMNS = {
    "d.beta": "14 spellings for 2 genes, including free text ('TRBD1 or TCRBD2') and "
              "comma lists; 23% filled, allele on only a third. The D segment is ~12 nt, so the "
              "calls are unreliable in principle and we have no use for them.",
    "meta.structure.id": "1.5% filled and not a structure id: only 381 of 2,586 values look like "
                         "a PDB accession, the rest are 'Fig 9', 'Supp Table 2'. Use PDB or "
                         "STCRDab if we ever want structures.",
    "tra_vfamily": "derivable in one call and nothing consumes it. trb_vfamily is kept because "
                   "beta-chain joins commonly match on it; alpha families are not used that way.",
    "cdr3fix.alpha / cdr3fix.beta": "VDJdb's correction blobs.",
    "TCR_hash": "VDJdb's internal key.",
    "method.frequency / method.sequencing / meta.study.id / meta.replica.id / meta.clone.id "
    "/ meta.epitope.id / meta.subject.cohort / meta.donor.MHC.method": "no use found.",
}

# V gene column -> the family column derived from it. J genes get no family: TRAJ is numbered
# straight through, and nothing of ours matches on TRBJ family.
GENE_COLUMNS = {"tra_v": None, "tra_j": None, "trb_v": "trb_vfamily", "trb_j": None}

# mhc.b is the invariant light chain on every class I row, so it carries no information there.
SOURCE = "vdjdb"

CLASS_I = "MHCI"
#: VDJdb's class column -> the hla_class tags, so the two compare directly.
MHC_CLASS_TO_TAG = {"MHCI": HLA_CLASS_I, "MHCII": HLA_CLASS_II}
INVARIANT_LIGHT_CHAIN = "B2M"


def trim_full(pdf_raw: pd.DataFrame) -> pd.DataFrame:
    """Trim and reshape vdjdb_full.txt into the project's per-clonotype table.

    vdjdb_full.txt is already one row per clonotype with separate alpha and beta columns, so there
    is no pivot to do — this renames the chain columns to the scRNA names, drops the columns in
    DROPPED_COLUMNS, canonicalizes the gene calls, and tidies three columns that are not usable as
    they ship:

        - `method.singlecell` becomes a real boolean. VDJdb stores "yes" or nothing, so True means
          VDJdb recorded single-cell data and False means it recorded nothing — not that the study
          is known to be bulk.
        - `method.identification` is stripped; 103 rows carry a trailing space, which splits
          "tetramer-sort" into two apparent categories.
        - `mhc.b` is nulled on class I rows, where it is always B2M. It stays on class II, where
          the beta chain is half the restriction.

    It also adds `mhc_a_std` / `mhc_b_std`, the standard IMGT names from `portmanteau.tcr.hla`,
    and `mhc_a_fields`, the resolution those reached.

    Gene calls go through portmanteau.tcr.genes, giving allele-free IMGT names. This table carries no comma-separated multi-gene calls (the slim table has ~1,800), but
    it does carry one semicolon call, so the calls are still exploded to one row per candidate
    gene; expect a handful of extra rows, not a cross product. An unresolvable call nulls its gene
    and keeps the row, because a row with a bad TRA call usually still has a usable TRB chain.

    Args:
        pdf_raw: vdjdb_full.txt as read, all species.

    Returns:
        Human rows only, with tra_cdr3/tra_v/tra_j/trb_cdr3/trb_v/trb_j, the trb_vfamily V family,
        and the KEEP_COLUMNS under their VDJdb names. One row per clonotype, except where an
        ambiguous gene call was exploded.
    """
    pdf = (
        pdf_raw.loc[pdf_raw["species"] == "HomoSapiens", [*CHAIN_COLUMNS, *KEEP_COLUMNS]]
        .rename(columns=CHAIN_COLUMNS)
        .reset_index(drop=True)
    )
    for gene_col, family_col in GENE_COLUMNS.items():
        pdf = explode_gene_calls(pdf, gene_col, gene_col)
        if family_col is not None:
            pdf[family_col] = gene_families(pdf[gene_col])
    pdf["method.singlecell"] = pdf["method.singlecell"].eq("yes")
    pdf["method.identification"] = pdf["method.identification"].str.strip()
    pdf["mhc.b"] = pdf["mhc.b"].mask(pdf["mhc.class"] == CLASS_I)
    # Standard IMGT names beside the originals, so a row can still be traced to what VDJdb wrote.
    # mhc_a_fields records the resolution reached: 16% of CMV Class I rows are one-field
    # (HLA-A*02) and cannot be refined, which is a join policy question, not a trim one.
    pdf["mhc_a_std"] = normalize_hlas(pdf["mhc.a"])
    pdf["mhc_b_std"] = normalize_hlas(pdf["mhc.b"])
    pdf["mhc_a_fields"] = hla_field_counts(pdf["mhc_a_std"])
    # VDJdb states the class itself, so that is authoritative; the derived value only fills gaps.
    pdf["hla_class"] = (pdf["mhc.class"].map(MHC_CLASS_TO_TAG)
                        .fillna(hla_classes(pdf["mhc.a"])).astype("string"))
    pdf["source"] = SOURCE
    return pdf.reset_index(drop=True)


#: The release `load` uses unless told otherwise. A bare "vdjdb" would take the newest
#: version already downloaded, which is convenient but makes a notebook's output depend on what
#: happens to be on disk; pin it here and bump it deliberately.
DEFAULT_RELEASE = "vdjdb/2026-06-03"
DERIVED_FILENAME = "vdjdb.parquet"


def read_full(path: Path) -> pd.DataFrame:
    """Read vdjdb_full.txt as it ships. ~119 MB, so prefer load() in a notebook."""
    return pd.read_csv(path, sep="\t", low_memory=False)


def load(spec: str = DEFAULT_RELEASE, *, refresh: bool = False) -> pd.DataFrame:
    """Return VDJdb as one row per clonotype, downloading and deriving it only when needed.

    A row carries whatever chains VDJdb observed: about half have both, a third are beta-only and
    a sixth alpha-only. It is *not* a table of alpha-beta pairs, and nothing here filters for
    pairing — a TRB with an epitope is most of what we want from this database.

    Safe to call every run: the download is cached by `ensure_release` and the trimmed table by
    `derived_frame`, so after the first call this reads a parquet of a few megabytes.

    Args:
        spec: a release, e.g. "vdjdb/2026-06-03"; defaults to DEFAULT_RELEASE.
        refresh: rebuild the derived table even if a current one is cached. Pass this after
            changing trim_full(), which the sidecar cannot detect on its own.

    Returns:
        The table trim_full() describes: human rows, chain columns under the scRNA names, gene
        calls canonicalized to allele-free IMGT.
    """
    return derived_frame(
        spec,
        DERIVED_FILENAME,
        lambda path: trim_full(read_full(path)),
        refresh=refresh,
        notes={
            "filters": "species == HomoSapiens",
            "shape": "one row per clonotype, as vdjdb_full.txt ships it; chain columns renamed to "
                     "the scRNA names by portmanteau.tcr.publicdb.vdjdb.trim_full",
            "genes": "allele-free IMGT via portmanteau.tcr.genes (tidytcells); "
                     "ambiguous calls exploded to one row per candidate gene",
            "dropped": DROPPED_COLUMNS,
            "licence": "GNU AGPL-3.0 (from the VDJdb archive)",
        },
    )


# --- One species, deduplicated to one row per (TRB, antigen) ---------------------------------
#
# This is for assigning putative antigen specificity to a set of beta chains. It is NOT for
# finding alpha-beta pair candidates: that wants load(), undeduplicated, with the alphas intact.
# Deduplicating costs nothing when the receiving side has no alpha to join to.

TRB_FILENAME = "vdjdb_{species}_trb_antigen__{slug}.parquet"


def filter_species(pdf: pd.DataFrame, preset: SpeciesPreset) -> pd.DataFrame:
    """Return the rows for one antigen species, with antigen.gene spellings folded together.

    `antigen.species` is a short controlled value in VDJdb, so the match is exact rather than a
    pattern.

    Args:
        pdf: VDJdb as load() returns it.
        preset: the species to select; `preset.vdjdb` is the exact `antigen.species` value.

    Returns:
        The matching rows, with `preset.gene_aliases` applied to antigen.gene. Empty, with the
        same columns, where the preset selects nothing from VDJdb.
    """
    if preset.vdjdb is None:
        return pdf.iloc[:0].copy()
    out = pdf[pdf["antigen.species"] == preset.vdjdb].copy()
    out["antigen.gene"] = out["antigen.gene"].replace(dict(preset.gene_aliases))
    return out.reset_index(drop=True)




def load_trb_antigens(
    preset: SpeciesPreset, spec: str = DEFAULT_RELEASE, *,
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, refresh: bool = False
) -> pd.DataFrame:
    """Return one species' rows as one row per (TRB, antigen), built and cached on first use.

    Built from load(), so the expensive read is already cached. Each species and each `trb_key`
    gets its own file: the derived-table sidecar keys on the release, not on how this was
    computed, so two settings must never share a name.

    Args:
        preset: the antigen species to select.
        spec: a release, e.g. "vdjdb/2026-06-03"; defaults to DEFAULT_RELEASE.
        trb_key: the columns that define one TRB; see deduplicate_trb_antigen().
        refresh: rebuild even if a current cached table exists. Needed after changing the code
            here, which the sidecar cannot detect.

    Returns:
        The table deduplicate_trb_antigen() describes, for this species only.
    """
    return derived_frame(
        spec,
        TRB_FILENAME.format(species=preset.slug, slug=trb_key_slug(trb_key)),
        lambda _path: deduplicate_trb_antigen(filter_species(load(spec), preset), trb_key=trb_key),
        refresh=refresh,
        notes={
            "filters": f"antigen.species == {preset.vdjdb!r}, rows with a TRB only",
            "shape": f"one row per ({' + '.join(trb_key)}, {ANTIGEN_COL})",
            "purpose": "assigning putative antigen specificity to a set of beta chains; "
                       "use load() for alpha-beta pair candidates",
            "caveat": "mhc_a is standardized but not reconciled against any particular "
                      "dataset's HLA naming",
            "licence": "GNU AGPL-3.0 (from the VDJdb archive)",
        },
    )
