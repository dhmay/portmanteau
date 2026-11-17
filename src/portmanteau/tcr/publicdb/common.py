"""What every public TCR-pMHC database is reshaped into, and the operations shared across them.

Each source gets a module in this package that knows its own quirks, and all of them
emit the same **canonical columns**, so the tables can be concatenated and one piece of join code
works on any of them:

    tra_cdr3, tra_v, tra_j                 the alpha chain, allele-free IMGT
    trb_cdr3, trb_v, trb_j, trb_vfamily    the beta chain, plus its V family
    antigen.epitope, antigen.gene          what it recognizes
    mhc.a, mhc.b, mhc.class                the restriction, as the source wrote it
    mhc_a_std, mhc_b_std, mhc_a_fields     the same in standard IMGT form, and its resolution
    hla_class                              "ci"/"cii"; see portmanteau.tcr.hla
    reference.id                           where it came from
    source                                 which database the row is from

The chain names are the short `tra_*` / `trb_*` forms that single-cell AIRR tables commonly
use, so a database row and a cell row join on the same column names without renaming either.
The antigen and MHC names follow VDJdb, the most widely used of the three. A source's own extra
columns are kept beside these under their original names.
"""
import pandas as pd

ANTIGEN_COL = "antigen.epitope"
#: What counts as "the same TRB". V family is derivable from the V gene, so keying on the gene
#: keeps both join modes open; keying on the family is irreversible. In one build the family key
#: merged only 137 more TRBs than the gene key, so the gene key is the default and a caller whose
#: other side carries family-only V calls opts into TRB_KEY_V_FAMILY.
TRB_KEY_V_GENE = ("trb_v", "trb_cdr3")
TRB_KEY_V_FAMILY = ("trb_vfamily", "trb_cdr3")
TRB_KEY_CDR3 = ("trb_cdr3",)
DEFAULT_TRB_KEY = TRB_KEY_V_GENE

CMV_TRB_FILENAME = "vdjdb_cmv_trb_antigen__{slug}.parquet"

#: Separates the distinct values collapsed into one cell. Not a comma: VDJdb uses commas inside
#: its own multi-value fields, and a list column cannot round-trip through parquet back into
#: pandas (the arrow list dtype has no pandas equivalent), so these are delimited strings.
LIST_SEPARATOR = "|"


def _sorted_unique(values: pd.Series) -> object:
    """Join the distinct values, or NA when there are none — an empty string reads as a value."""
    distinct = sorted(set(values.dropna()))
    return LIST_SEPARATOR.join(distinct) if distinct else pd.NA

#: Aggregated only when the source has the column. Every database records its own idea of
#: evidence quality, and none of them records the same one.
OPTIONAL_AGGREGATIONS = {
    "n_subjects": ("meta.subject.id", "nunique"),
    "max_vdjdb_score": ("vdjdb.score", "max"),
    "any_singlecell": ("method.singlecell", "any"),
}


def deduplicate_trb_antigen(
    pdf: pd.DataFrame,
    *,
    trb_key: tuple[str, ...] = DEFAULT_TRB_KEY,
    antigen_col: str = ANTIGEN_COL,
    extra_aggregations: dict | None = None,
) -> pd.DataFrame:
    """Collapse to one row per (TRB, antigen), aggregating the evidence rather than discarding it.

    Rows with no TRB, or with an unresolvable gene in the key, are dropped: they cannot take part
    in a TRB join. Use load() if you want them.

    Nothing here filters out promiscuous TRBs. `n_epitopes` counts how many antigens the TRB
    carries so a caller can require an unambiguous assignment — about 1% of CMV TRBs carry more
    than one, with a tail of seven carrying 10 to 85. Note `vdjdb.score` does **not** separate
    them: 0.95% of score-0 TRBs are multi-epitope against 2.46% of score-2 TRBs, because a
    better-attested TRB has simply been seen in more studies.

    Args:
        pdf: VDJdb rows, usually from filter_cmv().
        trb_key: the columns that define one TRB. TRB_KEY_V_GENE (the default), TRB_KEY_V_FAMILY
            or TRB_KEY_CDR3, or any tuple of columns.
        antigen_col: the antigen identity to group by; the epitope by default.
        extra_aggregations: further named aggregations, as pandas named-agg tuples, for columns
            only this source has.

    Returns:
        One row per (TRB, antigen), with the key columns, `trb_vfamily`, `mhc.class` and
        `antigen.gene` (each constant within a group), and:
            n_rows / n_references: how much independent support there is; n_subjects too where
                the source records a subject.
            max_vdjdb_score, any_singlecell: quality flags, present only for sources that have
                them (see OPTIONAL_AGGREGATIONS).
            mhc_a / mhc_a_std / trb_j / references / antigen.gene / hla_class / source: the
                distinct values seen, sorted and joined with LIST_SEPARATOR; n_mhc_a counts them so the common case needs no splitting.
            min_mhc_a_fields: the lowest HLA resolution in the group, so a join can decide
                whether a one-field allele may match a two-field one.
            n_epitopes: how many distinct antigens this TRB has across the whole table.
    """
    key = [*trb_key, antigen_col]
    pdf = pdf[pdf["trb_cdr3"].notna()].dropna(subset=list(trb_key))
    grouped = pdf.groupby(key, dropna=False)
    aggregations = {
        "n_rows": ("trb_cdr3", "size"),
        "n_references": ("reference.id", "nunique"),
        "mhc_a": ("mhc.a", _sorted_unique),
        "mhc_a_std": ("mhc_a_std", _sorted_unique),
        "min_mhc_a_fields": ("mhc_a_fields", "min"),
        "trb_j": ("trb_j", _sorted_unique),
        "references": ("reference.id", _sorted_unique),
        "source": ("source", _sorted_unique),
        # Only some sources carry these, so they are added below if present.
        **{name: spec for name, spec in OPTIONAL_AGGREGATIONS.items() if spec[0] in pdf.columns},
        **(extra_aggregations or {}),
        # mhc.class is determined by the epitope's restriction, which the key fixes, so "first"
        # is not a choice — the assertion below makes that explicit. antigen.gene is *not*
        # determined: VDJdb uses a controlled vocabulary but McPAS's Antigen.protein is free text,
        # so one epitope can carry several spellings. It is aggregated rather than asserted.
        "mhc.class": ("mhc.class", "first"),
        "hla_class": ("hla_class", _sorted_unique),
        "antigen.gene": ("antigen.gene", _sorted_unique),
    }
    # The V family is only fixed by the key when the V gene is in it. Under TRB_KEY_CDR3 one CDR3
    # really can carry several families, so it is aggregated like any other multi-valued column.
    if "trb_vfamily" not in key:
        aggregations["trb_vfamily"] = ("trb_vfamily", "first" if "trb_v" in key else _sorted_unique)
    out = grouped.agg(**aggregations).reset_index()
    varies = grouped["mhc.class"].nunique() > 1
    assert not varies.any(), f"mhc.class varies within {varies.sum()} (TRB, antigen) groups"
    out["n_mhc_a"] = grouped["mhc.a"].nunique().to_numpy()
    out["n_epitopes"] = out.groupby(list(trb_key))[antigen_col].transform("nunique")
    return out.sort_values([*trb_key, antigen_col], ignore_index=True)


def trb_key_slug(trb_key: tuple[str, ...]) -> str:
    """A filename-safe name for a TRB key, so two definitions cannot share a cached file."""
    return "_".join(column.removeprefix("trb_") for column in trb_key)

