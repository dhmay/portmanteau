"""The McPAS-TCR ingest: its own defects, and the canonical shape it maps onto."""
import pandas as pd
import pytest

from portmanteau.tcr.publicdb.mcpas import (
    clean_gene_call,
    expand_slash_shorthand,
    filter_species,
    trim,
)
from portmanteau.tcr.publicdb.species import SpeciesPreset

#: A preset for the tests. The strings are public NCBI organism names; the point of the fixture
#: is that the module holds no species of its own, so every filter test has to supply one.
SPECIES = SpeciesPreset(slug="hhv5", mcpas="Cytomegalovirus (CMV)")
CMV_PATHOLOGY = SPECIES.mcpas

RAW_ROW = {
    "Species": "Human",
    "CDR3.alpha.aa": "CAVRDSNYQLIW", "TRAV": "TRAV14", "TRAJ": "TRAJ33",
    "CDR3.beta.aa": "CASSLGQAYEQYF", "TRBV": "TRBV20-1", "TRBJ": "TRBJ2-7",
    "Epitope.peptide": "NLVPMVATV", "Antigen.protein": "pp65", "MHC": "HLA-A*02:01",
    "PubMed.ID": 28629751,
    "Pathology": CMV_PATHOLOGY, "Category": "Pathogens",
    "Antigen.identification.method": 1.0, "Single.cell": "Yes",
    "Tissue": "PBMC", "T.Cell.Type": "CD8", "Protein.ID": "P06725", "Epitope.ID": 1,
}


def _raw(**overrides):
    return pd.DataFrame([{**RAW_ROW, **overrides}])


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Trailing numeric junk, e.g. "TRAJ11 59 0".
        ("TRAJ11 59 0", "TRAJ11"),
        ("TRAJ58 60 0", "TRAJ58"),
        ("TRBV7-2", "TRBV7-2"),
        # Not our business: tidytcells already handles the colon allele and zero padding.
        ("TRAV8-4:03", "TRAV8-4:03"),
        ("TRBV10-03", "TRBV10-03"),
    ],
)
def test_clean_gene_call(raw, expected):
    assert clean_gene_call(raw) == expected


def test_slash_shorthand_expands_to_full_gene_names():
    """"TRBV6-1/6-5/6-6" names three genes, each needing the locus and family put back."""
    out = expand_slash_shorthand(pd.Series(["TRBV6-1/6-5/6-6", "TRAV8-2/8-4"]))
    assert out.tolist() == ["TRBV6-1,TRBV6-5,TRBV6-6", "TRAV8-2,TRAV8-4"]


@pytest.mark.parametrize("call", ["TRAV29/DV5", "TRAV38-2/DV8", "TRBV20/OR9-2", "TRBV7-2", None])
def test_slash_shorthand_leaves_the_composites_alone(call):
    """The /DV and /OR composites use a slash too and must survive untouched."""
    assert expand_slash_shorthand(pd.Series([call])).tolist() == [call]


def test_trim_maps_onto_the_canonical_columns():
    out = trim(_raw())
    assert out.loc[0, "tra_cdr3"] == "CAVRDSNYQLIW"
    assert out.loc[0, "tra_v"] == "TRAV14/DV4", "gene names go through tcr.genes"
    assert out.loc[0, "trb_vfamily"] == "TRBV20"
    assert out.loc[0, "antigen.epitope"] == "NLVPMVATV"
    assert out.loc[0, "mhc_a_std"] == "HLA-A*02:01"
    assert out.loc[0, "source"] == "mcpas"


def test_trim_keeps_only_human():
    out = trim(pd.concat([_raw(), _raw(Species="Mouse")], ignore_index=True))
    assert len(out) == 1


def test_trim_derives_the_mhc_class_mcpas_does_not_record():
    out = trim(pd.concat([_raw(), _raw(MHC="HLA-DRB1*15:01")], ignore_index=True))
    assert out["mhc.class"].tolist() == ["MHCI", "MHCII"]


def test_trim_matches_vdjdbs_reference_format():
    """McPAS writes a bare PubMed number; VDJdb writes PMID:28629751."""
    assert trim(_raw()).loc[0, "reference.id"] == "PMID:28629751"


def test_trim_keeps_a_row_with_no_epitope():
    """Most of McPAS has no epitope — it is organized by pathology — and those rows still carry
    a TCR worth having."""
    out = trim(_raw(**{"Epitope.peptide": None}))
    assert len(out) == 1 and pd.isna(out.loc[0, "antigen.epitope"])


def test_serological_hla_keeps_the_resolution_it_has():
    out = trim(_raw(MHC="HLA-A2"))
    assert out.loc[0, "mhc_a_std"] == "HLA-A*02" and out.loc[0, "mhc_a_fields"] == 1


def test_unresolvable_hla_nulls_the_standard_form_and_keeps_the_original():
    out = trim(_raw(MHC="HLA-DQ2"))
    assert out.loc[0, "mhc.a"] == "HLA-DQ2" and pd.isna(out.loc[0, "mhc_a_std"])


def test_filter_species_matches_the_pathology_exactly():
    """⚠️ A cmv/cytomegalo pattern also catches LCMV, a different virus, and murine mCMV."""
    pdf = trim(pd.concat([
        _raw(),
        _raw(Pathology="Lymphocytic choriomeningitis virus (LCMV)"),
        _raw(Pathology="mCMV"),
    ], ignore_index=True))
    out = filter_species(pdf, SPECIES)
    assert len(out) == 1 and out.loc[0, "Pathology"] == CMV_PATHOLOGY
