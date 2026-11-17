"""IEDB: the receptor export, reshaped into the canonical columns.

`load()` is the entry point, mirroring `ingest.vdjdb.load()` and `ingest.mcpas.load()`.

IEDB is the awkward one of the three, in four ways:

- **It has a two-row header.** Row 1 groups the columns (`Receptor`, `Epitope`, `Chain 1`,
  `Chain 2`), row 2 names the fields, and field names repeat across the two chains. They are
  flattened to `"Chain 2 | Junction Calculated"` on read.
- **Chains are numbered, not named.** `Chain 1` is alpha (or gamma), `Chain 2` is beta (or
  delta); each carries its own `Type`, which is what we check rather than trusting the number.
- ⚠️ **Its `CDR3` column is the IMGT CDR3, not the junction.** `ASSQDRDTQY`, where VDJdb and
  McPAS both write `CASSQDRDTQYF`. Using it would join to nothing. We take
  `Junction Calculated` instead, which is the junction convention on 99.9% of rows.
- **Each gene has a Curated and a Calculated call**, filled at different rates; we prefer the
  curated one and fall back.

It is also the biggest: 301,598 rows over a 167 MB CSV, so only the columns we use are read.
"""
import csv
from pathlib import Path

import pandas as pd
import tidytcells as tt

from portmanteau.tcr.publicdb.common import (
    ANTIGEN_COL,
    DEFAULT_TRB_KEY,
    LIST_SEPARATOR,
    deduplicate_trb_antigen,
    trb_key_slug,
)
from portmanteau.tcr.publicdb.cache import derived_frame
from portmanteau.tcr.publicdb.species import SpeciesPreset
from portmanteau.tcr.genes import explode_gene_calls, gene_families
from portmanteau.tcr.hla import hla_class, hla_classes, hla_field_count, normalize_hla

SOURCE = "iedb"
#: IEDB is rebuilt in place with no version string; the pinned hash is the release's identity.
DEFAULT_RELEASE = "iedb/2026-10-23"
DERIVED_FILENAME = "iedb.parquet"
TRB_FILENAME = "iedb_{species}_trb_antigen__{slug}.parquet"

#: Only alpha/beta receptors; the export also holds gamma/delta and a few constructs.
ALPHABETA = "alphabeta"
HUMAN_TAXON_SUFFIX = "9606"
ALPHA, BETA = "alpha", "beta"
#: IEDB separates several possible MHC alleles with a pipe, which is also our LIST_SEPARATOR.
IEDB_MHC_SEPARATOR = "|"

#: Flattened "<group> | <field>" names we read. Everything else in the 100+ column file is
#: sequence, coordinate or IRI detail we have no use for.
SHARED_COLUMNS = {
    "Receptor | Type": "receptor_type",
    "Receptor | IEDB Receptor ID": "iedb_receptor_id",
    "Reference | IEDB IRI": "reference.id",
    "Epitope | Name": "antigen.epitope",
    "Epitope | Source Molecule": "antigen.gene",
    "Epitope | Source Organism": "antigen.species",
    "Assay | MHC Allele Names": "mhc.a",
}
#: Per chain: the IEDB field -> our suffix. Curated is preferred over Calculated where both exist.
CHAIN_FIELDS = {
    "Type": "type", "Organism IRI": "organism",
    "Junction Calculated": "cdr3",
    "Curated V Gene": "v_curated", "Calculated V Gene": "v_calculated",
    "Curated J Gene": "j_curated", "Calculated J Gene": "j_calculated",
}
CHAIN_GROUPS = {"Chain 1": "c1", "Chain 2": "c2"}
GENE_COLUMNS = {"tra_v": None, "tra_j": None, "trb_v": "trb_vfamily", "trb_j": None}


def flat_column_names(path: Path) -> list[str]:
    """Read IEDB's two header rows and flatten them to "<group> | <field>"."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        groups, fields = next(reader), next(reader)
    return [f"{group} | {field}".strip(" |") for group, field in zip(groups, fields)]


def read_raw(path: Path) -> pd.DataFrame:
    """Read the columns we use from tcr_full_v3.csv, with the two-row header flattened.

    ⚠️ `usecols` cannot be combined with a multi-index header, and when given indices pandas
    applies `names` in *sorted index* order — so the indices are sorted before being paired with
    their names. Getting this wrong silently misaligns every column.
    """
    flat = flat_column_names(path)
    wanted = list(SHARED_COLUMNS) + [
        f"{group} | {field}" for group in CHAIN_GROUPS for field in CHAIN_FIELDS
    ]
    indices = sorted(flat.index(name) for name in wanted)
    return pd.read_csv(
        path, header=None, skiprows=2, usecols=indices,
        names=[flat[i] for i in indices], low_memory=False,
    )


def _prefer_curated(pdf: pd.DataFrame, group: str, field: str) -> pd.Series:
    """IEDB gives a curated and a calculated call for each gene; take the curated one first."""
    return pdf[f"{group} | Curated {field} Gene"].fillna(pdf[f"{group} | Calculated {field} Gene"])


def _normalize_mhc(value: object) -> tuple[object, object]:
    """Standardize a pipe-separated IEDB MHC call, returning (standard names, lowest resolution)."""
    if value is None or pd.isna(value):
        return pd.NA, pd.NA
    parts = [normalize_hla(part) for part in str(value).split(IEDB_MHC_SEPARATOR)]
    resolved = sorted({part for part in parts if part})
    if not resolved:
        return pd.NA, pd.NA
    fields = [hla_field_count(part) for part in resolved]
    present = [f for f in fields if f is not None]
    return LIST_SEPARATOR.join(resolved), (min(present) if present else pd.NA)


MHC_CLASS_NAMES = {1: "MHCI", 2: "MHCII"}


def _mhc_class(standardized: object) -> object:
    """MHCI/MHCII for a standardized (possibly multi-allele) call, or NA if its parts disagree."""
    if standardized is None or pd.isna(standardized):
        return pd.NA
    classes = {tt.mh.get_class(part, log_failures=False)
               for part in str(standardized).split(LIST_SEPARATOR)}
    classes.discard(None)
    return MHC_CLASS_NAMES.get(classes.pop()) if len(classes) == 1 else pd.NA


def trim(pdf_raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape IEDB into the canonical columns of `ingest.common`.

    Keeps human alpha/beta receptors, resolves Chain 1/Chain 2 to alpha/beta by each chain's own
    `Type`, takes the junction rather than the IMGT CDR3, and canonicalizes genes and HLA.

    Args:
        pdf_raw: tcr_full_v3.csv as read by `read_raw`.

    Returns:
        Human rows in the canonical columns, plus `iedb_receptor_id` and `receptor_type`.
    """
    is_ab = pdf_raw["Receptor | Type"] == ALPHABETA
    is_human = pd.concat(
        [pdf_raw[f"{group} | Organism IRI"].astype("string").str.endswith(HUMAN_TAXON_SUFFIX)
         for group in CHAIN_GROUPS], axis=1
    ).fillna(False).any(axis=1)
    raw = pdf_raw[is_ab & is_human].reset_index(drop=True)

    pdf = raw[list(SHARED_COLUMNS)].rename(columns=SHARED_COLUMNS)
    # Always present, even when nothing survives the filters: callers and the derived-table
    # schema depend on the canonical columns existing, empty frame or not.
    for prefix in ("tra", "trb"):
        for suffix in ("cdr3", "v", "j"):
            pdf[f"{prefix}_{suffix}"] = pd.Series(pd.NA, index=pdf.index, dtype="object")
    # Chain 1 is alpha and Chain 2 beta in practice, but each chain states its own type, so use
    # that: a row whose chain is gamma or delta contributes nothing to the alpha/beta columns.
    for group in CHAIN_GROUPS:
        for locus, prefix in ((ALPHA, "tra"), (BETA, "trb")):
            matches = (raw[f"{group} | Type"] == locus).to_numpy()
            if not matches.any():
                continue
            pdf.loc[matches, f"{prefix}_cdr3"] = raw.loc[matches, f"{group} | Junction Calculated"]
            pdf.loc[matches, f"{prefix}_v"] = _prefer_curated(raw, group, "V")[matches]
            pdf.loc[matches, f"{prefix}_j"] = _prefer_curated(raw, group, "J")[matches]
    for gene_col, family_col in GENE_COLUMNS.items():
        pdf = explode_gene_calls(pdf, gene_col, gene_col)
        if family_col is not None:
            pdf[family_col] = gene_families(pdf[gene_col])
    standardized = pdf["mhc.a"].map(_normalize_mhc)
    pdf["mhc_a_std"] = [value for value, _ in standardized]
    pdf["mhc_a_fields"] = pd.array([fields for _, fields in standardized], dtype="Int8")
    # IEDB records no second MHC chain and no class column, so the class is derived from the
    # allele. An ambiguous call whose alleles disagree on class gets no class rather than a guess.
    pdf["mhc.b"] = pd.NA
    pdf["mhc_b_std"] = pd.NA
    pdf["mhc.class"] = pdf["mhc_a_std"].map(_mhc_class).astype("string")
    pdf["iedb_receptor_id"] = raw["Receptor | IEDB Receptor ID"]
    # Derived from the raw name, not the standardized one: the coarse and malformed values that
    # defeat normalize_hla (HLA-DQ2, "HLA class II") still have a legible class.
    pdf["hla_class"] = hla_classes(pdf["mhc.a"])
    pdf["source"] = SOURCE
    return pdf.reset_index(drop=True)


def filter_species(pdf: pd.DataFrame, preset: SpeciesPreset) -> pd.DataFrame:
    """Return the rows for one species, by `antigen.species` **prefix**.

    A prefix, because IEDB appends strain and host detail to the organism name, so one species
    appears under several full spellings and an exact match would drop most of them.
    """
    if preset.iedb is None:
        return pdf.iloc[:0].copy()
    organism = pdf["antigen.species"].astype("string")
    return pdf[organism.str.startswith(preset.iedb).fillna(False)].reset_index(drop=True)


def load(spec: str = DEFAULT_RELEASE, *, refresh: bool = False) -> pd.DataFrame:
    """Return IEDB's TCR export in the canonical columns, downloading and deriving when needed."""
    return derived_frame(
        spec, DERIVED_FILENAME, lambda path: trim(read_raw(path)), refresh=refresh,
        notes={
            "filters": f"Receptor Type == {ALPHABETA}, human by either chain's organism",
            "shape": "one row per IEDB receptor record, reshaped by portmanteau.tcr.publicdb.iedb.trim",
            "caveat": "CDR3s are 'Junction Calculated', not IEDB's 'CDR3' column, which holds the "
                      "IMGT CDR3 and would join to nothing",
            "licence": "free to use; see iedb.org for the citation policy",
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
        notes={"filters": f"antigen.species starts with {preset.iedb!r}, rows with a TRB only",
               "shape": f"one row per ({' + '.join(trb_key)}, {ANTIGEN_COL})"},
    )
