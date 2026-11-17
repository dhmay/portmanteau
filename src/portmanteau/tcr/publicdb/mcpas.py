"""McPAS-TCR: a manually curated catalogue of pathology-associated TCRs.

`load()` is the entry point, mirroring `ingest.vdjdb.load()`: it ensures the release, reshapes the
table into the canonical columns of `ingest.common`, and caches the result.

McPAS differs from VDJdb in ways that shape this module:

- **It is organized by pathology, not antigen.** `Pathology` is filled on every row, but
  `Epitope.peptide` on only 41%. So most of McPAS cannot say what a TCR recognizes, only what
  disease it came from. CMV is the exception: every human CMV row has an epitope.
- ⚠️ **`Pathology` must be matched exactly.** A `cmv|cytomegalo` pattern also catches
  `Lymphocytic choriomeningitis virus (LCMV)`, an unrelated virus, and `mCMV` (murine).
- **It has no MHC class column**; the class is derived from the allele name.
- **It is mostly beta.** 94% of rows have a beta CDR3, 36% an alpha — so keeping chain-only rows
  matters even more here than in VDJdb.
- **The file carries a BOM**, so it is read as utf-8-sig.

The CDR3s need no transformation: 99.9% of beta CDR3s already start with the conserved cysteine,
and prepending one where it is missing changes the VDJdb overlap from 7,331 to 7,330 — noise, not
a convention. Checked rather than assumed, because the first row of the file has no leading C.
"""
import re
from pathlib import Path

import pandas as pd
import tidytcells as tt

from portmanteau.tcr.publicdb.common import (
    ANTIGEN_COL,
    DEFAULT_TRB_KEY,
    deduplicate_trb_antigen,
    trb_key_slug,
)
from portmanteau.tcr.publicdb.cache import derived_frame
from portmanteau.tcr.publicdb.species import SpeciesPreset
from portmanteau.tcr.genes import explode_gene_calls, gene_families
from portmanteau.tcr.hla import hla_classes, hla_field_counts, normalize_hlas

SOURCE = "mcpas"
#: McPAS is updated in place with no version string, so this is the date we retrieved it and the
#: pinned hash is the release's only real identity.
DEFAULT_RELEASE = "mcpas/2026-10-23"
DERIVED_FILENAME = "mcpas.parquet"
TRB_FILENAME = "mcpas_{species}_trb_antigen__{slug}.parquet"
#: The file begins with a BOM; without this the first column is named "﻿CDR3.alpha.aa".
ENCODING = "utf-8-sig"

HUMAN = "Human"

#: McPAS column -> the short AIRR chain names.
CHAIN_COLUMNS = {
    "CDR3.alpha.aa": "tra_cdr3", "TRAV": "tra_v", "TRAJ": "tra_j",
    "CDR3.beta.aa": "trb_cdr3", "TRBV": "trb_v", "TRBJ": "trb_j",
}
#: McPAS column -> the canonical antigen/MHC/provenance names, which follow VDJdb.
RENAMED_COLUMNS = {
    "Epitope.peptide": "antigen.epitope",
    "Antigen.protein": "antigen.gene",
    "MHC": "mhc.a",
    "PubMed.ID": "reference.id",
}
#: McPAS's own columns worth keeping, under their original names.
KEEP_COLUMNS = [
    "Pathology", "Category", "Antigen.identification.method", "Single.cell",  # Single.cell becomes method.singlecell
    "Tissue", "T.Cell.Type", "Protein.ID", "Epitope.ID",
]
GENE_COLUMNS = {"tra_v": None, "tra_j": None, "trb_v": "trb_vfamily", "trb_j": None}

#: Trailing numeric junk on a gene call, e.g. "TRAJ11 59 0".
TRAILING_JUNK_PATTERN = r"\s+[\d\s]+$"
#: A slash-shorthand multi-gene call: "TRBV6-1/6-5/6-6" means three genes, each needing the locus
#: and family of the first put back. The alternatives must be numeric so that the /DV and /OR
#: composites, whose suffixes contain letters, are left alone.
SLASH_SHORTHAND_PATTERN = re.compile(r"^(TR[AB][VJ])(\d+(?:-\d+)?)((?:/\d+(?:-\d+)?)+)$")


def clean_gene_call(call: str) -> str:
    """Repair McPAS's gene-call defects. Passed to `tcr.genes` as its `clean` hook.

    Only the trailing numeric junk is repaired here; slash shorthand is expanded earlier, by
    `expand_slash_shorthand`, because it turns one call into several.

    ⚠️ **Locus mislabelling is deliberately not repaired.** `TRBJ9:01` and `TRBJ33:01` look like
    TRAJ genes filed under TRBJ, and `TRBV23/DV6` is a TRAV gene's name on a TRBV column. Guessing
    the intended locus would invent data, so these fail normalization and become NA, keeping their
    rows. Well under 1% of rows.
    """
    return re.sub(TRAILING_JUNK_PATTERN, "", call).strip()


def expand_slash_shorthand(calls: pd.Series) -> pd.Series:
    """Rewrite "TRBV6-1/6-5/6-6" as "TRBV6-1,TRBV6-5,TRBV6-6" so the usual explode can split it.

    Applied before `explode_gene_calls`, which splits on "," and ";" but must not split on "/" —
    that would break the `TRAV29/DV5` composites.
    """
    def expand(call: object) -> object:
        if call is None or pd.isna(call):
            return call
        match = SLASH_SHORTHAND_PATTERN.match(str(call).strip())
        if match is None:
            return call
        locus, first, rest = match.groups()
        return ",".join([f"{locus}{first}"] + [f"{locus}{part}" for part in rest.split("/") if part])
    return calls.map(expand)


def read_raw(path: Path) -> pd.DataFrame:
    """Read McPAS-TCR.csv as it ships, BOM and all."""
    return pd.read_csv(path, encoding=ENCODING, low_memory=False)


def trim(pdf_raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape McPAS into the canonical columns of `ingest.common`.

    Keeps human rows, renames the chains and the antigen/MHC/provenance columns, canonicalizes the
    gene calls and HLA names, and derives the MHC class McPAS does not record. Rows missing an
    epitope are kept: most of McPAS has none, and a TCR with a pathology is still a TCR.

    Args:
        pdf_raw: McPAS-TCR.csv as read, all species.

    Returns:
        Human rows with the canonical columns, plus KEEP_COLUMNS under their McPAS names. One row
        per McPAS record, except where an ambiguous gene call was exploded.
    """
    pdf = (
        pdf_raw.loc[pdf_raw["Species"] == HUMAN, [*CHAIN_COLUMNS, *RENAMED_COLUMNS, *KEEP_COLUMNS]]
        .rename(columns={**CHAIN_COLUMNS, **RENAMED_COLUMNS})
        .reset_index(drop=True)
    )
    for gene_col, family_col in GENE_COLUMNS.items():
        pdf[gene_col] = expand_slash_shorthand(pdf[gene_col])
        pdf = explode_gene_calls(pdf, gene_col, gene_col, clean=clean_gene_call)
        if family_col is not None:
            pdf[family_col] = gene_families(pdf[gene_col])
    # VDJdb writes "PMID:28629751"; McPAS writes the bare number. Match VDJdb so the two sources
    # can be deduplicated against each other on provenance.
    pdf["reference.id"] = "PMID:" + pdf["reference.id"].astype("string").str.strip()
    pdf["mhc_a_std"] = normalize_hlas(pdf["mhc.a"])
    pdf["mhc_a_fields"] = hla_field_counts(pdf["mhc_a_std"])
    # McPAS has no class column, and no second MHC chain either.
    pdf["mhc.class"] = pdf["mhc_a_std"].map(
        lambda h: None if h is None or pd.isna(h) else tt.mh.get_class(h, log_failures=False)
    ).map({1: "MHCI", 2: "MHCII"}).astype("string")
    # McPAS writes Yes/No; VDJdb writes "yes" or nothing. Both become the same bool.
    pdf["method.singlecell"] = pdf.pop("Single.cell").eq("Yes")
    pdf["mhc.b"] = pd.NA
    pdf["mhc_b_std"] = pd.NA
    # Derived from the raw name, not the standardized one: the coarse and malformed values that
    # defeat normalize_hla (HLA-DQ2, "HLA class II") still have a legible class.
    pdf["hla_class"] = hla_classes(pdf["mhc.a"])
    pdf["source"] = SOURCE
    return pdf.reset_index(drop=True)


def filter_species(pdf: pd.DataFrame, preset: SpeciesPreset) -> pd.DataFrame:
    """Return the rows for one species: `Pathology` exactly `preset.mcpas`.

    Exact, not a pattern: see the module docstring for the trap.
    """
    if preset.mcpas is None:
        return pdf.iloc[:0].copy()
    return pdf[pdf["Pathology"] == preset.mcpas].reset_index(drop=True)


def load(spec: str = DEFAULT_RELEASE, *, refresh: bool = False) -> pd.DataFrame:
    """Return McPAS-TCR in the canonical columns, downloading and deriving it only when needed."""
    return derived_frame(
        spec, DERIVED_FILENAME, lambda path: trim(read_raw(path)), refresh=refresh,
        notes={
            "filters": f"Species == {HUMAN}",
            "shape": "one row per McPAS record, reshaped by portmanteau.tcr.publicdb.mcpas.trim",
            "caveat": "Epitope.peptide is filled on only ~41% of rows; McPAS is organized by "
                      "pathology, not antigen",
            "licence": "see friedmanlab.weizmann.ac.il/McPAS-TCR for terms; cite Tickotsky et al. 2017",
        },
    )


def load_trb_antigens(
    preset: SpeciesPreset, spec: str = DEFAULT_RELEASE, *,
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, refresh: bool = False
) -> pd.DataFrame:
    """Return one species' rows as one row per (TRB, antigen), built and cached on first use."""
    return derived_frame(
        spec, TRB_FILENAME.format(species=preset.slug, slug=trb_key_slug(trb_key)),
        lambda _path: deduplicate_trb_antigen(filter_species(load(spec), preset), trb_key=trb_key),
        refresh=refresh,
        notes={
            "filters": f"Pathology == {preset.mcpas!r}, rows with a TRB only",
            "shape": f"one row per ({' + '.join(trb_key)}, {ANTIGEN_COL})",
        },
    )
