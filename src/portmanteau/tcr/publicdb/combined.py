"""VDJdb, McPAS-TCR and IEDB as one annotation set for a chosen antigen species.

Three artifacts, each cached, each built from the one above. All three take a `SpeciesPreset`:

- **`load_rows()`** - every matching record from every source, one row as the source wrote it,
  with the antigen family attached. The only one that keeps **gene-level pairing** - which alpha
  chain, with which V and J, sits beside which beta - so it is what anything scoring a whole
  receptor reads.
- **`load_annotations()`** - the long table, one row per **(TRB, antigen, database)**. This is
  the primitive: it keeps every peptide, every reference and every paired alpha, so a question
  that was not anticipated can still be answered without rebuilding.
- **`load_trb_antigens()`** - the flattened view, one row per **(TRB, MHC class)** with its
  best-supported antigen and its runner-up, each carrying the HLA alleles it was recorded on.
  Lossy by design, and what an equality join against a beta-chain table wants. ⚠️ **Class is part
  of the key, not an aggregate**: in one build 2 TRBs of ~31,900 were reported under both
  classes, and collapsing them would leave a cell no equality join could match.

Each source's own `filter_species` decides what matches, because none of them encodes a species
the same way - see `publicdb.species` for why one match is exact and another is a prefix.

Two things here cost real time to learn and are kept separate on purpose:

- ⚠️ **`pmids` is not `references`.** IEDB stores its own IRIs, so unioning its reference set with
  VDJdb's PMIDs counts the same paper twice. Only PMIDs are comparable across sources; IEDB
  contributes none, and its support shows up in `databases` instead.
- ⚠️ **`antigen` is a family, `peptides` are the members.** The family name is one representative
  of a group merged by `portmanteau.tcr.peptides`; anything fed to a model wants the real
  peptide.
"""
from __future__ import annotations

import pandas as pd

from portmanteau.tcr.publicdb import iedb, mcpas, vdjdb
from portmanteau.tcr.publicdb.common import DEFAULT_TRB_KEY, LIST_SEPARATOR, trb_key_slug
from portmanteau.tcr.publicdb.cache import derived_frame
from portmanteau.tcr.publicdb.species import SpeciesPreset
from portmanteau.tcr.peptides import antigen_family_map

#: Columns every source carries, so the row-level union has one schema. Source-specific columns
#: are dropped: keeping them would make the frame's shape depend on which sources were included.
ROW_COLUMNS = [
    "trb_vfamily", "trb_cdr3", "trb_v", "trb_j",
    "tra_cdr3", "tra_v", "tra_j",
    "antigen.epitope", "antigen.gene",
    "mhc.a", "mhc_a_std", "mhc_a_fields", "mhc.class", "hla_class",
    "reference.id",
]

#: Database name -> its ingest module. The order fixes column order in the flattened view.
SOURCE_MODULE_MAP = {"vdjdb": vdjdb, "mcpas": mcpas, "iedb": iedb}
ROWS_FILENAME = "{species}_rows.parquet"
ANNOTATIONS_FILENAME = "{species}_annotations__{slug}.parquet"
TRB_ANTIGENS_FILENAME = "{species}_trb_antigens__{slug}.parquet"
EPITOPE_COLUMN = "antigen.epitope"
#: Part of the flattened table's key, so a caller can filter on it.
CLASS_COLUMN = "hla_class"
PMID_PATTERN = r"PMID:(\d+)"
#: ⚠️ The 10x Genomics highly-multiplexed dextramer application note. In one build it was cited
#: by 14,041 of ~41,200 rows and by **no** other reference - one URL, VDJdb only. A multiplexed
#: dextramer screen is the classic source of spurious specificity calls, and such rows land
#: disproportionately on HLA-disagreeing matches, so a caller needs to stand them down without
#: reaching back to the row-level table. Matched on the domain rather than the full URL, which
#: carries a trailing fragment.
DEXTRAMER_SCREEN_PATTERN = "10xgenomics.com"

def load_rows(preset: SpeciesPreset, *, refresh: bool = False) -> pd.DataFrame:
    """Return every CMV record from every source, one row each, building and caching on first use.

    Unlike the aggregated tables this keeps **one row per source record**, so the alpha chain and
    its V and J genes stay attached to the beta they were observed with. That pairing is lost the
    moment rows are grouped, and it is what a model scoring a whole receptor needs.

    Args:
        refresh: rebuild even if a current cached table exists.

    Returns:
        ROW_COLUMNS plus `database` (which source) and `antigen` (the family the row's peptide
        belongs to). No TRB key is applied, so this is not slugged.
    """
    return derived_frame(
        [module.DEFAULT_RELEASE for module in SOURCE_MODULE_MAP.values()],
        ROWS_FILENAME.format(species=preset.slug),
        lambda _paths: species_rows(preset),
        refresh=refresh,
        notes={
            "shape": "one row per source record, union of the three databases",
            "filters": "each source's own filter_species(); rows without a TRB CDR3 are kept",
            "antigen": "a family from portmanteau.tcr.peptides; antigen.epitope is the real peptide",
        },
    )


def load_annotations(
    preset: SpeciesPreset, trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, *, refresh: bool = False
) -> pd.DataFrame:
    """Return the long CMV annotation table, building and caching it on first use.

    Args:
        trb_key: the columns defining one TRB. A dataset carrying family-only V calls - Adaptive
            output typically does - wants `TRB_KEY_V_FAMILY`.
        refresh: rebuild even if a current cached table exists. Needed after changing the build,
            which the sidecar cannot detect.

    Returns:
        One row per (TRB, antigen, database); see `species_annotations`.
    """
    return derived_frame(
        [module.DEFAULT_RELEASE for module in SOURCE_MODULE_MAP.values()],
        ANNOTATIONS_FILENAME.format(species=preset.slug, slug=trb_key_slug(trb_key)),
        lambda _paths: species_annotations(preset, trb_key=trb_key),
        refresh=refresh,
        notes={
            "shape": "one row per (TRB, antigen, database)",
            "trb_key": list(trb_key),
            "antigen": "a family from portmanteau.tcr.peptides; `peptides` holds the members",
            "evidence": "`pmids` counts only PMIDs, which are comparable across sources; IEDB "
                        "stores its own IRIs and contributes none, so read its support from "
                        "`database`",
        },
    )


def load_trb_antigens(
    preset: SpeciesPreset, trb_key: tuple[str, ...] = DEFAULT_TRB_KEY, *, refresh: bool = False
) -> pd.DataFrame:
    """Return the flattened per-TRB view, building and caching it on first use.

    Built from `load_annotations`, so refreshing this does not rebuild that.

    Args:
        trb_key: the columns defining one TRB.
        refresh: rebuild even if a current cached table exists.

    Returns:
        One row per (TRB, MHC class); see `top_antigens`.
    """
    return derived_frame(
        [module.DEFAULT_RELEASE for module in SOURCE_MODULE_MAP.values()],
        TRB_ANTIGENS_FILENAME.format(species=preset.slug, slug=trb_key_slug(trb_key)),
        lambda _paths: top_antigens(load_annotations(preset, trb_key), trb_key=trb_key),
        refresh=refresh,
        notes={
            "shape": "one row per (TRB, hla_class), with its top two antigens by supporting "
                     "PMID count",
            "trb_key": list(trb_key),
            "derived_from": ANNOTATIONS_FILENAME.format(species=preset.slug,
                                                        slug=trb_key_slug(trb_key)),
        },
    )


def species_rows(preset: SpeciesPreset) -> pd.DataFrame:
    """Build the row-level union of the three sources' records for one species.

    Returns:
        ROW_COLUMNS plus `database` and `antigen`. Every source's own `filter_species` decides
        what matches.
    """
    frames = [module.filter_species(module.load(), preset)[ROW_COLUMNS].assign(database=name)
              for name, module in SOURCE_MODULE_MAP.items()]
    pdf_rows = pd.concat(frames, ignore_index=True)
    peptides = pdf_rows[EPITOPE_COLUMN].dropna()
    families = antigen_family_map(peptides, peptides.value_counts())
    return pdf_rows.assign(antigen=pdf_rows[EPITOPE_COLUMN].map(families))


def species_annotations(preset: SpeciesPreset,
                        trb_key: tuple[str, ...] = DEFAULT_TRB_KEY) -> pd.DataFrame:
    """Build the long table: one row per (TRB, antigen, database).

    Args:
        trb_key: the columns defining one TRB.

    Returns:
        A frame with the key columns plus:
            database: which source this row's evidence comes from.
            antigen: the antigen family name; `peptides` the members seen in this source.
            pmids / n_pmids: distinct PubMed IDs behind this (TRB, antigen) in this source.
            references: the source's own reference ids, PMIDs or IEDB IRIs.
            tras: alpha-chain CDR3s paired with this TRB for this antigen, where recorded.
            hla_class, mhc_a_std, antigen_gene, n_rows: as the source gives them.
            min_mhc_a_fields: the *lowest* HLA resolution in the group - 1 for a bare `HLA-A*02`,
                2 for `HLA-A*02:01`. An allele join needs it to tell "these disagree" from
                "one side never resolved far enough to say".
    """
    pdf_rows = load_rows(preset)
    pdf_rows = pdf_rows[pdf_rows["trb_cdr3"].notna()].dropna(subset=list(trb_key))
    return (
        pdf_rows.groupby([*trb_key, "antigen", "database"], dropna=False)
        .agg(
            peptides=(EPITOPE_COLUMN, _joined),
            pmids=("reference.id", _joined_pmids),
            references=("reference.id", _joined),
            tras=("tra_cdr3", _joined),
            hla_class=("hla_class", _joined),
            mhc_a_std=("mhc_a_std", _joined),
            min_mhc_a_fields=("mhc_a_fields", "min"),
            antigen_gene=("antigen.gene", _joined),
            n_rows=("trb_cdr3", "size"),
        )
        .reset_index()
        .assign(n_pmids=lambda pdf: pdf.pmids.map(lambda v: 0 if not v else v.count(LIST_SEPARATOR) + 1))
    )


def top_antigens(
    pdf_annotations: pd.DataFrame, trb_key: tuple[str, ...] = DEFAULT_TRB_KEY
) -> pd.DataFrame:
    """Flatten the long table to one row per (TRB, MHC class), with its top two antigens.

    Antigens are ranked by how many distinct PMIDs back them across all sources, then by row count
    so the order is deterministic. Ties at one PMID are common and expected.

    `hla_class` joins the key rather than being aggregated. A TRB reported under both classes is
    two receptors that happen to share a CDR3, and the antigen that best explains it differs
    between them; `"ci|cii"` in one cell would also defeat any equality filter on the class.
    Blank classes become NA so such a filter can drop them.

    **The HLA allele rides with the antigen, not with the TRB.** `<prefix>_antigen_mhc_a_std`
    holds the alleles that antigen was recorded on for this TRB, so an allele filter asks the
    question that matters - *was this peptide ever seen on this allele?* - rather than the weaker
    *does this TRB have any evidence on this allele, under any antigen?* A TRB-level union over
    every antigen would let a minor antigen's allele vouch for the top antigen's call.

    Args:
        pdf_annotations: the long table, as `species_annotations` returns it.
        trb_key: the columns defining one TRB.

    Returns:
        One row per (TRB, `hla_class`) with, for each of `top` and `second`: the antigen, the
        databases backing it, its distinct PMID count, the paired alpha CDR3s, the alleles it was
        recorded on (`_mhc_a_std`, LIST_SEPARATOR-joined) and the lowest resolution any of those
        reached (`_min_mhc_a_fields`, 1 or 2, NA where no allele was recorded). `n_antigens`
        counts all its antigen families, not just the two shown.

        Provenance, so a caller can stand an annotation down without the row-level table:
        `_references` (every reference id behind it, LIST_SEPARATOR-joined - ⚠️ IEDB contributes
        IRIs, so this is not comparable to `_pmids`) and `_dextramer_only`, true when **every**
        one of them is the multiplexed dextramer screen (`DEXTRAMER_SCREEN_PATTERN`).
    """
    pdf_annotations = pdf_annotations.assign(
        **{CLASS_COLUMN: pdf_annotations[CLASS_COLUMN].replace("", pd.NA)})
    group_key = [*trb_key, CLASS_COLUMN]
    per_antigen = (
        pdf_annotations.groupby([*group_key, "antigen"], dropna=False)
        .agg(databases=("database", _joined), pmids=("pmids", _joined_unique),
             tras=("tras", _joined_unique), n_rows=("n_rows", "sum"),
             mhc_a_std=("mhc_a_std", _joined_unique),
             min_mhc_a_fields=("min_mhc_a_fields", "min"),
             references=("references", _joined_unique))
        .reset_index()
    )
    # Derived here rather than left to the caller so it cannot disagree with `references`.
    per_antigen["dextramer_only"] = [
        bool(value) and all(DEXTRAMER_SCREEN_PATTERN in part
                            for part in value.split(LIST_SEPARATOR))
        for value in per_antigen.references]
    per_antigen["n_pmids"] = per_antigen.pmids.map(
        lambda v: 0 if not v else v.count(LIST_SEPARATOR) + 1)

    ranked = per_antigen.sort_values(["n_pmids", "n_rows"], ascending=False)
    ranked = ranked.groupby(group_key, dropna=False).head(2).copy()
    ranked["rank"] = ranked.groupby(group_key, dropna=False).cumcount()

    def side(rank: int, prefix: str) -> pd.DataFrame:
        columns = {"antigen": f"{prefix}_antigen", "databases": f"{prefix}_antigen_databases",
                   "n_pmids": f"{prefix}_antigen_pmids", "tras": f"{prefix}_antigen_tras",
                   "mhc_a_std": f"{prefix}_antigen_mhc_a_std",
                   "min_mhc_a_fields": f"{prefix}_antigen_min_mhc_a_fields",
                   "references": f"{prefix}_antigen_references",
                   "dextramer_only": f"{prefix}_antigen_dextramer_only"}
        return (ranked[ranked["rank"] == rank]
                .set_index(group_key)[list(columns)].rename(columns=columns))

    pdf_trb = side(0, "top").join(side(1, "second"), how="left").reset_index()
    pdf_trb["second_antigen_pmids"] = pdf_trb.second_antigen_pmids.fillna(0).astype(int)
    # A missing second antigen, or an antigen no source gave an allele for, must stay NA rather
    # than become 0 fields: "no resolution recorded" is not "resolved to nothing".
    for prefix in ("top", "second"):
        pdf_trb[f"{prefix}_antigen_min_mhc_a_fields"] = (
            pdf_trb[f"{prefix}_antigen_min_mhc_a_fields"].astype("Int64"))
        # Nullable rather than object: a missing second antigen is "unknown", not "not a screen".
        pdf_trb[f"{prefix}_antigen_dextramer_only"] = (
            pdf_trb[f"{prefix}_antigen_dextramer_only"].astype("boolean"))
    counts = per_antigen.groupby(group_key, dropna=False).size().rename("n_antigens")
    return pdf_trb.merge(counts.reset_index(), on=group_key, how="left")


def _joined(values: pd.Series) -> str:
    """Distinct values, sorted and LIST_SEPARATOR-joined. Empty rather than NA when there are none."""
    return LIST_SEPARATOR.join(sorted({str(v) for v in values.dropna()}))


def _joined_unique(values: pd.Series) -> str:
    """As _joined, but flattening cells that are themselves LIST_SEPARATOR-joined sets."""
    parts = {p for value in values.dropna() for p in str(value).split(LIST_SEPARATOR) if p}
    return LIST_SEPARATOR.join(sorted(parts))


def _joined_pmids(values: pd.Series) -> str:
    """The distinct PubMed IDs in a reference column. IEDB's IRIs yield none, deliberately."""
    found = set(re.findall(PMID_PATTERN, " ".join(values.dropna().astype(str))))
    return LIST_SEPARATOR.join(sorted(found))
