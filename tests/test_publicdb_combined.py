"""Aggregating the three public databases into one CMV annotation set."""
import pandas as pd
import pytest

from portmanteau.tcr.publicdb.common import TRB_KEY_V_FAMILY
from portmanteau.tcr.publicdb import combined
from portmanteau.tcr.publicdb.combined import DEXTRAMER_SCREEN_PATTERN, top_antigens

TRB_KEY = TRB_KEY_V_FAMILY


def annotation(vfamily, cdr3, antigen, database, pmids="", tras="", n_rows=1, hla_class="ci",
               mhc_a_std="HLA-A*02:01", min_mhc_a_fields=2, references="PMID:1"):
    return {"trb_vfamily": vfamily, "trb_cdr3": cdr3, "antigen": antigen, "database": database,
            "pmids": pmids, "tras": tras, "n_rows": n_rows, "hla_class": hla_class,
            "mhc_a_std": mhc_a_std, "min_mhc_a_fields": min_mhc_a_fields,
            "references": references}


def test_ranks_antigens_by_distinct_pmids():
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="1|2|3"),
        annotation("TRBV4", "CASSX", "TPRVTGGGAM", "vdjdb", pmids="4"),
    ])
    out = top_antigens(pdf, trb_key=TRB_KEY)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.top_antigen == "RPHERNGFTVL" and row.top_antigen_pmids == 3
    assert row.second_antigen == "TPRVTGGGAM" and row.second_antigen_pmids == 1
    assert row.n_antigens == 2


def test_pmids_are_unioned_across_databases_not_counted_twice():
    """Two sources citing the same paper is one PMID, not two."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="28636589"),
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "mcpas", pmids="28636589"),
    ])
    out = top_antigens(pdf, trb_key=TRB_KEY)
    assert out.iloc[0].top_antigen_pmids == 1
    assert out.iloc[0].top_antigen_databases == "mcpas|vdjdb"


def test_a_single_antigen_leaves_the_second_side_empty():
    pdf = pd.DataFrame([annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", pmids="1")])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert pd.isna(row.second_antigen)
    assert row.second_antigen_pmids == 0, "absent means zero PMIDs, not NA"
    assert row.n_antigens == 1


def test_iedb_only_antigen_has_no_pmids_but_still_ranks():
    """IEDB stores IRIs, so it contributes no PMIDs; the database column is where its support is."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="1"),
        annotation("TRBV4", "CASSX", "TPRVTGGGAM", "iedb", pmids=""),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen == "RPHERNGFTVL" and row.second_antigen == "TPRVTGGGAM"
    assert row.second_antigen_pmids == 0
    assert row.second_antigen_databases == "iedb"


def test_ties_are_broken_deterministically_by_row_count():
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "AAA", "vdjdb", pmids="1", n_rows=1),
        annotation("TRBV4", "CASSX", "BBB", "vdjdb", pmids="2", n_rows=9),
    ])
    assert top_antigens(pdf, trb_key=TRB_KEY).iloc[0].top_antigen == "BBB"


def test_only_the_top_two_are_shown_but_all_are_counted():
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", name, "vdjdb", pmids="1") for name in ("AAA", "BBB", "CCC")
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.n_antigens == 3
    assert {row.top_antigen, row.second_antigen} < {"AAA", "BBB", "CCC"}


def test_alphas_are_carried_through_per_antigen():
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="1|2", tras="CAAA|CABB"),
        annotation("TRBV4", "CASSX", "TPRVTGGGAM", "iedb", tras="CACC"),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen_tras == "CAAA|CABB" and row.second_antigen_tras == "CACC"


def test_mhc_class_is_part_of_the_key_not_an_aggregate():
    """One CDR3 under both classes is two receptors, and each keeps its own best antigen."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", pmids="1", hla_class="ci"),
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="2", hla_class="cii"),
    ])
    out = top_antigens(pdf, trb_key=TRB_KEY).set_index("hla_class")
    assert len(out) == 2
    assert out.loc["ci"].top_antigen == "NLVPMVATV"
    assert out.loc["cii"].top_antigen == "RPHERNGFTVL"
    assert out.loc["ci"].n_antigens == 1, "the other class must not be counted here"


def test_blank_class_becomes_na_so_an_equality_join_can_drop_it():
    pdf = pd.DataFrame([annotation("TRBV4", "CASSX", "NLVPMVATV", "mcpas", hla_class="")])
    assert pd.isna(top_antigens(pdf, trb_key=TRB_KEY).iloc[0].hla_class)


def test_alleles_ride_with_the_antigen_not_with_the_trb():
    """The top antigen's alleles must not be vouched for by a different antigen's."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "RPHERNGFTVL", "vdjdb", pmids="1|2",
                   mhc_a_std="HLA-B*07:02"),
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", pmids="3",
                   mhc_a_std="HLA-A*02:01"),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen == "RPHERNGFTVL"
    assert row.top_antigen_mhc_a_std == "HLA-B*07:02"
    assert row.second_antigen_mhc_a_std == "HLA-A*02:01"


def test_alleles_are_unioned_across_the_sources_backing_one_antigen():
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", mhc_a_std="HLA-A*02:01"),
        annotation("TRBV4", "CASSX", "NLVPMVATV", "mcpas", mhc_a_std="HLA-A*02:06"),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen_mhc_a_std == "HLA-A*02:01|HLA-A*02:06"


def test_a_cell_that_is_already_joined_is_flattened_not_nested():
    """species_annotations joins its own alleles, so the second pass must split before unioning."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb",
                   mhc_a_std="HLA-A*02:01|HLA-A*02:06"),
        annotation("TRBV4", "CASSX", "NLVPMVATV", "mcpas", mhc_a_std="HLA-A*02:01"),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen_mhc_a_std == "HLA-A*02:01|HLA-A*02:06"


def test_field_count_is_the_weakest_resolution_behind_the_antigen():
    """One unrefinable source makes the whole antigen unrefinable, so min is the right summary."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", mhc_a_std="HLA-A*02:01",
                   min_mhc_a_fields=2),
        annotation("TRBV4", "CASSX", "NLVPMVATV", "mcpas", mhc_a_std="HLA-A*02",
                   min_mhc_a_fields=1),
    ])
    assert top_antigens(pdf, trb_key=TRB_KEY).iloc[0].top_antigen_min_mhc_a_fields == 1


def test_an_absent_allele_is_na_not_zero_fields():
    """"No resolution recorded" is not "resolved to nothing"; and a missing second side stays NA."""
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", mhc_a_std="",
                   min_mhc_a_fields=None),
    ])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert row.top_antigen_mhc_a_std == ""
    assert pd.isna(row.top_antigen_min_mhc_a_fields)
    assert pd.isna(row.second_antigen_min_mhc_a_fields)


def test_a_dextramer_only_antigen_is_flagged_and_a_mixed_one_is_not():
    """One ordinary reference is enough to stop an annotation being screen-only."""
    screen = f"https://www.{DEXTRAMER_SCREEN_PATTERN}/resources/application-notes/x/#"
    pdf = pd.DataFrame([
        annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb", references=screen),
        annotation("TRBV9", "CASSY", "NLVPMVATV", "vdjdb", references=f"{screen}|PMID:123"),
    ])
    out = top_antigens(pdf, trb_key=TRB_KEY).set_index("trb_cdr3")
    assert out.loc["CASSX", "top_antigen_dextramer_only"]
    assert not out.loc["CASSY", "top_antigen_dextramer_only"]
    assert out.loc["CASSY", "top_antigen_references"] == f"PMID:123|{screen}"


def test_an_absent_second_antigen_has_an_unknown_screen_flag_not_a_false_one():
    pdf = pd.DataFrame([annotation("TRBV4", "CASSX", "NLVPMVATV", "vdjdb")])
    row = top_antigens(pdf, trb_key=TRB_KEY).iloc[0]
    assert pd.isna(row.second_antigen_dextramer_only)
    assert row.top_antigen_dextramer_only is False or row.top_antigen_dextramer_only == False


# --- the aggregation helpers themselves ------------------------------------------------------
#
# ⚠️ These are reached only when a derived table is actually BUILT. A cached parquet skips them
# entirely, so a break here stays invisible until someone clears the cache - which is exactly how
# a missing `import re` survived a full green suite. Test them directly.


def test_joined_pmids_extracts_only_pubmed_ids():
    """IEDB stores its own IRIs; counting those as references double-counts a paper."""
    values = pd.Series(["PMID:28629751", "PMID:28629751", "http://www.iedb.org/reference/1038466",
                        None, "PMID:31324695"])
    assert combined._joined_pmids(values) == "28629751|31324695"


def test_joined_pmids_on_references_with_no_pubmed_id_is_empty():
    assert combined._joined_pmids(pd.Series(["http://www.iedb.org/reference/1", None])) == ""


def test_every_aggregation_helper_is_importable_and_callable():
    """A NameError inside one of these only surfaces on a cache miss. Touch each one."""
    series = pd.Series(["a", "b", None, "a"])
    for name in ("_joined", "_joined_pmids"):
        helper = getattr(combined, name)
        assert isinstance(helper(series), str), name
