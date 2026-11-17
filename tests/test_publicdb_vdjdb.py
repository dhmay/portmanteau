"""Tests for the VDJdb ingest: the trim, the renames, and the columns we decided against."""
import pandas as pd
import pytest

from portmanteau.tcr.publicdb.species import SpeciesPreset
from portmanteau.tcr.publicdb.vdjdb import (
    TRB_FILENAME,
    TRB_KEY_CDR3,
    TRB_KEY_V_FAMILY,
    TRB_KEY_V_GENE,
    deduplicate_trb_antigen,
    filter_species,
    trb_key_slug,
    trim_full,
)

#: A preset for the tests. The strings are public NCBI organism names; the point of the fixture
#: is that the module holds no species of its own, so every filter test has to supply one.
SPECIES = SpeciesPreset(slug="hhv5", vdjdb="CMV", gene_aliases={"IE-1": "IE1"})

# --- trim_full -----------------------------------------------------------------------------

RAW_ROW = {
    "species": "HomoSapiens",
    "cdr3.alpha": "CAVRDSNYQLIW", "v.alpha": "TRAV14*01", "j.alpha": "TRAJ33*01",
    "cdr3.beta": "CASSLGQAYEQYF", "v.beta": "TRBV20-1*01", "j.beta": "TRBJ2.7*01",
    "d.beta": "TRBD1 or TCRBD2",
    "antigen.epitope": "NLVPMVATV", "antigen.gene": "pp65", "antigen.species": "CMV",
    "mhc.a": "HLA-A*02:01", "mhc.b": "B2M", "mhc.class": "MHCI",
    "reference.id": "PMID:1", "vdjdb.score": 1,
    "method.identification": "tetramer-sort ", "method.singlecell": "yes",
    "method.verification": "direct",
    "meta.subject.id": "donor1", "meta.donor.MHC": "A02", "meta.cell.subset": "CD8+",
    "meta.tissue": "PBMC", "meta.structure.id": "Fig 9",
    "cdr3fix.alpha": "{}", "TCR_hash": "abc", "method.frequency": "1/100",
}


def _raw(**overrides):
    return pd.DataFrame([{**RAW_ROW, **overrides}])


def test_trim_full_renames_the_chain_columns_to_the_scrna_names():
    out = trim_full(_raw())
    assert out.loc[0, "tra_cdr3"] == "CAVRDSNYQLIW"
    assert out.loc[0, "trb_cdr3"] == "CASSLGQAYEQYF"
    # Allele-free IMGT, with the composite spelled the way scirpy spells it for the scRNA data.
    assert out.loc[0, "tra_v"] == "TRAV14/DV4"
    assert out.loc[0, "trb_v"] == "TRBV20-1"
    assert out.loc[0, "trb_j"] == "TRBJ2-7"
    assert out.loc[0, "trb_vfamily"] == "TRBV20"


@pytest.mark.parametrize("column", ["d.beta", "trb_d", "meta.structure.id", "tra_vfamily",
                                    "cdr3fix.alpha", "TCR_hash", "method.frequency", "species"])
def test_trim_full_drops_the_columns_we_decided_against(column):
    assert column not in trim_full(_raw()).columns


def test_trim_full_keeps_only_human_rows():
    pdf = pd.concat([_raw(), _raw(species="MusMusculus")], ignore_index=True)
    out = trim_full(pdf)
    assert len(out) == 1 and out.loc[0, "antigen.epitope"] == "NLVPMVATV"


def test_trim_full_makes_singlecell_a_real_boolean():
    out = trim_full(pd.concat([_raw(), _raw(**{"method.singlecell": None})], ignore_index=True))
    assert out["method.singlecell"].dtype == bool
    assert out["method.singlecell"].tolist() == [True, False]


def test_trim_full_strips_method_identification():
    """A trailing space splits one category in two; 103 real rows carry one."""
    out = trim_full(pd.concat([_raw(), _raw(**{"method.identification": "tetramer-sort"})],
                              ignore_index=True))
    assert out["method.identification"].nunique() == 1


def test_trim_full_nulls_mhc_b_on_class_i_only():
    """B2M is the invariant light chain, so it says nothing on class I; class II keeps its beta."""
    pdf = pd.concat([_raw(), _raw(**{"mhc.class": "MHCII", "mhc.b": "HLA-DRB1*03:01"})],
                    ignore_index=True)
    out = trim_full(pdf)
    assert pd.isna(out.loc[0, "mhc.b"])
    assert out.loc[1, "mhc.b"] == "HLA-DRB1*03:01"


def test_trim_full_keeps_a_beta_only_row():
    """A TRB with an epitope is most of what we want; it must not need a partner alpha."""
    out = trim_full(_raw(**{"cdr3.alpha": None, "v.alpha": None, "j.alpha": None}))
    assert len(out) == 1
    assert pd.isna(out.loc[0, "tra_cdr3"]) and out.loc[0, "trb_cdr3"] == "CASSLGQAYEQYF"


def test_trim_full_keeps_a_row_whose_gene_call_is_unresolvable():
    out = trim_full(_raw(**{"v.beta": "TRBVIS1"}))
    assert len(out) == 1
    assert pd.isna(out.loc[0, "trb_v"]) and pd.isna(out.loc[0, "trb_vfamily"])
    assert out.loc[0, "trb_cdr3"] == "CASSLGQAYEQYF", "the usable chain survives a bad call"


def test_trim_full_explodes_an_ambiguous_call():
    out = trim_full(_raw(**{"j.alpha": "TRAJ27;TRAJ28"}))
    assert out["tra_j"].tolist() == ["TRAJ27", "TRAJ28"]
    assert out["trb_cdr3"].tolist() == ["CASSLGQAYEQYF"] * 2


# --- the CMV subset, deduplicated to (TRB, antigen) -----------------------------------------

CMV_ROW = {**RAW_ROW, "antigen.species": "CMV"}


def _cmv(**overrides):
    return trim_full(pd.DataFrame([{**CMV_ROW, **overrides}]))


def test_filter_species_keeps_only_the_preset_species_and_folds_the_gene_alias():
    pdf = pd.concat([_cmv(), _cmv(**{"antigen.species": "EBV"}),
                     _cmv(**{"antigen.gene": "IE-1"})], ignore_index=True)
    out = filter_species(pdf, SPECIES)
    assert len(out) == 2
    assert set(out["antigen.species"]) == {"CMV"}
    assert sorted(out["antigen.gene"]) == ["IE1", "pp65"]


def test_deduplicate_collapses_duplicate_rows_and_counts_the_support():
    pdf = pd.concat([
        _cmv(**{"reference.id": "PMID:1", "meta.subject.id": "d1", "vdjdb.score": 1}),
        _cmv(**{"reference.id": "PMID:2", "meta.subject.id": "d2", "vdjdb.score": 3,
                "mhc.a": "HLA-A*02"}),
    ], ignore_index=True)
    out = deduplicate_trb_antigen(filter_species(pdf, SPECIES))
    assert len(out) == 1
    row = out.iloc[0]
    assert row.n_rows == 2 and row.n_references == 2 and row.n_subjects == 2
    assert row.max_vdjdb_score == 3
    # The evidence is aggregated, not thrown away.
    assert row.mhc_a == "HLA-A*02|HLA-A*02:01" and row.n_mhc_a == 2
    assert row.references == "PMID:1|PMID:2"


def test_deduplicate_keeps_one_row_per_antigen_and_counts_them():
    """A TRB against two epitopes stays two rows; n_epitopes flags it on both."""
    pdf = pd.concat([_cmv(), _cmv(**{"antigen.epitope": "KLGGALQAK"})], ignore_index=True)
    out = deduplicate_trb_antigen(filter_species(pdf, SPECIES))
    assert len(out) == 2
    assert out["n_epitopes"].tolist() == [2, 2]


def test_deduplicate_drops_rows_with_no_trb():
    """Alpha-only rows cannot take part in a TRB join; load() is where they live."""
    pdf = pd.concat([_cmv(), _cmv(**{"cdr3.beta": None, "v.beta": None, "j.beta": None})],
                    ignore_index=True)
    out = deduplicate_trb_antigen(filter_species(pdf, SPECIES))
    assert len(out) == 1 and out.loc[0, "trb_cdr3"] == "CASSLGQAYEQYF"


def test_the_trb_key_is_a_parameter():
    """Two V genes, one CDR3: separate TRBs by gene, one TRB by family."""
    pdf = pd.concat([_cmv(), _cmv(**{"v.beta": "TRBV20-1*02"}),
                     _cmv(**{"v.beta": "TRBV19*01"})], ignore_index=True)
    assert len(deduplicate_trb_antigen(filter_species(pdf, SPECIES), trb_key=TRB_KEY_V_GENE)) == 2
    assert len(deduplicate_trb_antigen(filter_species(pdf, SPECIES), trb_key=TRB_KEY_CDR3)) == 1
    by_family = deduplicate_trb_antigen(filter_species(pdf, SPECIES), trb_key=TRB_KEY_V_FAMILY)
    assert sorted(by_family.trb_vfamily) == ["TRBV19", "TRBV20"]


def test_each_trb_key_gets_its_own_cache_filename():
    """derived_frame keys on the release, not on how the table was computed, so two keys must
    not share a file."""
    slugs = {trb_key_slug(k) for k in (TRB_KEY_V_GENE, TRB_KEY_V_FAMILY, TRB_KEY_CDR3)}
    assert slugs == {"v_cdr3", "vfamily_cdr3", "cdr3"}
    assert len({TRB_FILENAME.format(species=SPECIES.slug, slug=s) for s in slugs}) == 3


def test_each_species_gets_its_own_cache_filename():
    """Same reason: the sidecar keys on the release, so two species must not share a file."""
    other = SpeciesPreset(slug="ebv", vdjdb="EBV")
    names = {TRB_FILENAME.format(species=p.slug, slug="v_cdr3") for p in (SPECIES, other)}
    assert len(names) == 2


def test_aggregated_columns_survive_a_parquet_round_trip(tmp_path):
    """A list column cannot be read back out of parquet by pandas; these are delimited strings."""
    out = deduplicate_trb_antigen(filter_species(_cmv(), SPECIES))
    path = tmp_path / "t.parquet"
    out.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(out, pd.read_parquet(path))


def test_trim_full_adds_standard_hla_beside_the_original():
    out = trim_full(pd.concat([_raw(), _raw(**{"mhc.a": "HLA-A*02"})], ignore_index=True))
    assert out["mhc.a"].tolist() == ["HLA-A*02:01", "HLA-A*02"], "the original is untouched"
    assert out["mhc_a_std"].tolist() == ["HLA-A*02:01", "HLA-A*02"]
    assert out["mhc_a_fields"].tolist() == [2, 1], "resolution reached is recorded, not forced"


def test_deduplicate_carries_the_standard_hla_and_its_resolution():
    pdf = pd.concat([_cmv(), _cmv(**{"mhc.a": "HLA-A*02", "reference.id": "PMID:2"})],
                    ignore_index=True)
    out = deduplicate_trb_antigen(filter_species(pdf, SPECIES))
    assert out.loc[0, "mhc_a_std"] == "HLA-A*02|HLA-A*02:01"
    assert out.loc[0, "min_mhc_a_fields"] == 1, "a join needs the weakest resolution in the group"
