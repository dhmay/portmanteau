"""Tests for normalizing TCR gene names to allele-free IMGT.

Only source-neutral nomenclature rules belong here; VDJdb's own defects are covered by
tests/test_ingest_vdjdb.py.
"""
import pandas as pd
import pytest

from portmanteau.tcr.genes import (
    explode_gene_calls,
    gene_families,
    gene_family,
    normalize_gene,
    normalize_genes,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Already canonical, with and without an allele.
        ("TRAV12-2*01", "TRAV12-2"),
        ("TRBV20-1", "TRBV20-1"),
        ("TRAV14/DV4*01", "TRAV14/DV4"),
        ("TRBV20/OR9-2*01", "TRBV20/OR9-2"),
        ("TRBJ2-7*01", "TRBJ2-7"),
        ("TRAJ16*01", "TRAJ16"),
        # The /DV genes under their non-canonical spellings.
        ("TRAV14", "TRAV14/DV4"),
        ("TRAV14-1", "TRAV14/DV4"),
        ("TRAV14DV4", "TRAV14/DV4"),
        ("TRAV29*01", "TRAV29/DV5"),
        ("TRAV36DV7", "TRAV36/DV7"),
        ("TRAV38-2", "TRAV38-2/DV8"),
        # Family-dot-member beta J names.
        ("TRBJ2.1", "TRBJ2-1"),
        ("TRBJ1.2*01", "TRBJ1-2"),
    ],
)
def test_normalize_gene_resolves_known_spellings(raw, expected):
    assert normalize_gene(raw) == expected


@pytest.mark.parametrize("raw", ["TRAJ", "TRAJ16.5", "TRBJ10-3", "TRBV11/2", "TRBVIS1", "", None])
def test_normalize_gene_returns_none_for_unresolvable(raw):
    assert normalize_gene(raw) is None


@pytest.mark.parametrize(
    "gene,expected",
    [
        ("TRBV20-1", "TRBV20"),
        ("TRBV19", "TRBV19"),
        # Both /DV forms reduce to the IMGT family; the old "-.*$" strip got TRAV14/DV4 wrong.
        ("TRAV14/DV4", "TRAV14"),
        ("TRAV38-2/DV8", "TRAV38"),
        ("TRBV20/OR9-2", "TRBV20"),
        ("TRBJ2-1", "TRBJ2"),
        # TRAJ genes are numbered straight through, so they have no family.
        ("TRAJ16", None),
        (None, None),
    ],
)
def test_gene_family(gene, expected):
    assert gene_family(gene) == expected


def test_normalize_and_family_series_keep_na():
    genes = normalize_genes(pd.Series(["TRAV14*01", "TRAJ", None, "TRBV20-1*02"]))
    assert genes.tolist()[:1] == ["TRAV14/DV4"]
    assert genes.isna().tolist() == [False, True, True, False]
    assert gene_families(genes).tolist() == ["TRAV14", pd.NA, pd.NA, "TRBV20"]


def test_explode_gene_calls_one_row_per_candidate():
    pdf = pd.DataFrame(
        {
            "complex.id": ["1", "2", "3", "4"],
            "v.segm": ["TRAV1-1*01,TRAV1-2*01", "TRBV20-1*01", "TRAJ27;TRAJ28", "TRBV11/2"],
        }
    )
    out = explode_gene_calls(pdf, "v.segm", "v_gene")
    assert out["v_gene"].tolist() == ["TRAV1-1", "TRAV1-2", "TRBV20-1", "TRAJ27", "TRAJ28", pd.NA]
    # A call nothing can be made of nulls the gene but keeps its row, so a usable partner chain survives.
    assert out["complex.id"].tolist() == ["1", "1", "2", "3", "3", "4"]
    assert out.loc[out["complex.id"] == "4", "v_gene"].isna().all()


def test_explode_gene_calls_keeps_other_columns_and_index():
    pdf = pd.DataFrame({"cdr3": ["CASSX", "CASSY"], "j.segm": ["TRBJ1-2*01", "TRBJ2.1*01,TRBJ2-3*01"]})
    out = explode_gene_calls(pdf, "j.segm", "j_gene")
    assert out["j_gene"].tolist() == ["TRBJ1-2", "TRBJ2-1", "TRBJ2-3"]
    assert out["cdr3"].tolist() == ["CASSX", "CASSY", "CASSY"]
    assert out.index.tolist() == [0, 1, 1]
    # The original call is left verbatim for forensics.
    assert out["j.segm"].tolist() == ["TRBJ1-2*01", "TRBJ2.1*01,TRBJ2-3*01", "TRBJ2.1*01,TRBJ2-3*01"]


def test_clean_hook_runs_before_everything_else():
    """No source needs it today; it is the seam for the next database's defects."""
    assert normalize_gene("xxTRBV20-1") is None
    assert normalize_gene("xxTRBV20-1", clean=lambda call: call.removeprefix("xx")) == "TRBV20-1"


def test_clean_hook_reaches_each_candidate_of_an_exploded_call():
    pdf = pd.DataFrame({"j.segm": ["xxTRBJ1-2,xxTRBJ2-1"]})
    out = explode_gene_calls(pdf, "j.segm", "j_gene", clean=lambda call: call.removeprefix("xx"))
    assert out["j_gene"].tolist() == ["TRBJ1-2", "TRBJ2-1"]


# --- what we rely on tidytcells for, and where we override it -------------------------------
#
# These lock behaviour the pinned tidytcells gives us. If a version bump breaks one, that is the
# signal to re-run the token comparison that justified the pin before accepting the upgrade.


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Defects VDJdb carries that tidytcells resolves unaided, so we ship no cleaner for it.
        ("TRBJ 2-1", "TRBJ2-1"),
        ("TRAJ33_", "TRAJ33"),
        ("TRAJ01-1", "TRAJ1"),
        ("TRAJ16-1", "TRAJ16"),
        # Names a structural regex cannot judge: these need IMGT's actual gene list.
        ("TRBV21", "TRBV21-1"),
        ("TRBV22", "TRBV22-1"),
        ("TRBV17-1", "TRBV17"),
    ],
)
def test_tidytcells_resolves_what_we_depend_on_it_for(raw, expected):
    assert normalize_gene(raw) == expected


@pytest.mark.parametrize("raw", ["TRBJ2\u20133", "TRBJ2\u20133*01", "TRBV20\u20111"])
def test_unicode_dashes_are_repaired_before_tidytcells_sees_them(raw):
    """Handed an en-dash, tidytcells returns a different real gene with no error (TRBJ2-1 for
    TRBJ2-3). A silently wrong gene is worse than a rejection, so the dash never reaches it."""
    assert normalize_gene(raw) in {"TRBJ2-3", "TRBV20-1"}
    assert normalize_gene("TRBJ2\u20133") == "TRBJ2-3"


def test_trav14_1_is_overridden_to_the_imgt_name():
    """tidytcells maps TRAV23-1/TRAV29-1/TRAV36-1/TRAV38-2 to their /DV names but leaves
    TRAV14-1, which has no sequence data in its own tables. The scRNA side spells it TRAV14/DV4."""
    assert normalize_gene("TRAV14-1") == "TRAV14/DV4"
    assert normalize_gene("TRAV14-1*01") == "TRAV14/DV4"
    for sibling, expected in [("TRAV23-1", "TRAV23/DV6"), ("TRAV29-1", "TRAV29/DV5"),
                              ("TRAV36-1", "TRAV36/DV7"), ("TRAV38-2", "TRAV38-2/DV8")]:
        assert normalize_gene(sibling) == expected


def test_multi_gene_calls_are_split_before_tidytcells_sees_them():
    """tidytcells silently returns only the first gene of "TRAJ27;TRAJ28"."""
    out = explode_gene_calls(pd.DataFrame({"j": ["TRAJ27;TRAJ28"]}), "j", "gene")
    assert out["gene"].tolist() == ["TRAJ27", "TRAJ28"]
