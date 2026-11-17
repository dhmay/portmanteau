"""The IEDB ingest: its two-row header, numbered chains, and junction-vs-CDR3 trap."""
import pandas as pd
import pytest

from portmanteau.tcr.publicdb.iedb import (
    SHARED_COLUMNS,
    filter_species,
    flat_column_names,
    read_raw,
    trim,
)
from portmanteau.tcr.publicdb.species import SpeciesPreset

#: A preset for the tests. The strings are public NCBI organism names; the point of the fixture
#: is that the module holds no species of its own, so every filter test has to supply one.
SPECIES = SpeciesPreset(slug="hhv5", iedb="Human herpesvirus 5")
CMV_ORGANISM_PREFIX = SPECIES.iedb

CHAIN_FIELD_NAMES = ["Type", "Organism IRI", "Junction Calculated", "Curated V Gene",
                     "Calculated V Gene", "Curated J Gene", "Calculated J Gene"]
HUMAN = "http://purl.obolibrary.org/obo/NCBITaxon_9606"

ROW = {
    "Receptor | Type": "alphabeta",
    "Receptor | IEDB Receptor ID": 57,
    "Reference | IEDB IRI": "https://www.iedb.org/reference/1004539",
    "Epitope | Name": "NLVPMVATV",
    "Epitope | Source Molecule": "pp65",
    "Epitope | Source Organism": "Human herpesvirus 5 (Human cytomegalovirus)",
    "Assay | MHC Allele Names": "HLA-A*02:01",
    "Chain 1 | Type": "alpha", "Chain 1 | Organism IRI": HUMAN,
    "Chain 1 | Junction Calculated": "CIVVRSSNTGKLIF",
    "Chain 1 | Curated V Gene": "TRAV26-1*01", "Chain 1 | Calculated V Gene": "TRAV26-1*01",
    "Chain 1 | Curated J Gene": "TRAJ37*01", "Chain 1 | Calculated J Gene": "TRAJ37*01",
    "Chain 2 | Type": "beta", "Chain 2 | Organism IRI": HUMAN,
    "Chain 2 | Junction Calculated": "CASSQDRDTQYF",
    "Chain 2 | Curated V Gene": "TRBV14*01", "Chain 2 | Calculated V Gene": "TRBV14*01",
    "Chain 2 | Curated J Gene": "TRBJ2-3*01", "Chain 2 | Calculated J Gene": "TRBJ2-3*01",
}


def _raw(**overrides):
    return pd.DataFrame([{**ROW, **overrides}])


def _write_two_row_header_csv(path, rows):
    """IEDB's real layout: a group row, then a field row, then data."""
    columns = list(ROW)
    groups = [c.split(" | ")[0] for c in columns]
    fields = [c.split(" | ")[1] for c in columns]
    frame = pd.DataFrame([groups, fields] + [[r[c] for c in columns] for r in rows])
    frame.to_csv(path, index=False, header=False)


def test_flat_column_names_joins_the_two_header_rows(tmp_path):
    path = tmp_path / "tcr.csv"
    _write_two_row_header_csv(path, [ROW])
    names = flat_column_names(path)
    assert "Chain 2 | Junction Calculated" in names
    assert "Epitope | Name" in names


def test_read_raw_keeps_the_columns_aligned(tmp_path):
    """⚠️ pandas applies `names` in sorted-index order when `usecols` holds indices; getting
    that wrong silently misaligns every column, which is exactly what happened in development."""
    path = tmp_path / "tcr.csv"
    _write_two_row_header_csv(path, [ROW])
    out = read_raw(path)
    assert out.loc[0, "Chain 2 | Junction Calculated"] == "CASSQDRDTQYF"
    assert out.loc[0, "Epitope | Name"] == "NLVPMVATV"
    assert out.loc[0, "Chain 1 | Curated V Gene"] == "TRAV26-1*01"


def test_trim_takes_the_junction_not_the_imgt_cdr3():
    """IEDB's CDR3 column holds ASSQDRDTQY; everything we join to writes CASSQDRDTQYF."""
    out = trim(_raw())
    assert out.loc[0, "trb_cdr3"] == "CASSQDRDTQYF"
    assert out.loc[0, "tra_cdr3"] == "CIVVRSSNTGKLIF"


def test_trim_maps_chains_by_their_own_type_not_their_number():
    out = trim(_raw())
    assert out.loc[0, "tra_v"] == "TRAV26-1" and out.loc[0, "trb_v"] == "TRBV14"
    assert out.loc[0, "trb_vfamily"] == "TRBV14"
    assert out.loc[0, "source"] == "iedb"


def test_trim_prefers_the_curated_gene_call_and_falls_back():
    out = trim(_raw(**{"Chain 2 | Curated V Gene": None, "Chain 2 | Calculated V Gene": "TRBV20-1*01"}))
    assert out.loc[0, "trb_v"] == "TRBV20-1"


@pytest.mark.parametrize("receptor_type", ["gammadelta", "TscFv", "construct"])
def test_trim_keeps_only_alphabeta_receptors(receptor_type):
    assert len(trim(_raw(**{"Receptor | Type": receptor_type}))) == 0


def test_trim_keeps_only_human():
    mouse = "http://purl.obolibrary.org/obo/NCBITaxon_10090"
    out = trim(_raw(**{"Chain 1 | Organism IRI": mouse, "Chain 2 | Organism IRI": mouse})) 
    assert len(out) == 0
    # Human on either chain is enough; one chain is often blank.
    assert len(trim(_raw(**{"Chain 1 | Organism IRI": None}))) == 1


def test_trim_standardizes_a_pipe_separated_mhc_call():
    out = trim(_raw(**{"Assay | MHC Allele Names": "HLA-E*01:01|HLA-E*01:03"}))
    assert out.loc[0, "mhc_a_std"] == "HLA-E*01:01|HLA-E*01:03"
    assert out.loc[0, "mhc_a_fields"] == 2


def test_trim_derives_the_mhc_class_iedb_does_not_record():
    assert trim(_raw()).loc[0, "mhc.class"] == "MHCI"
    assert trim(_raw(**{"Assay | MHC Allele Names": "HLA-DRB1*01:01"})).loc[0, "mhc.class"] == "MHCII"


def test_filter_species_prefix_excludes_a_different_host_species():
    """⚠️ A substring match on "cytomegalo" would also catch the murid virus, which is why the
    IEDB filter anchors at the start of the organism name."""
    pdf = trim(pd.concat([
        _raw(),
        _raw(**{"Epitope | Source Organism": "Murid betaherpesvirus 1 (Mouse cytomegalovirus 1)"}),
        _raw(**{"Epitope | Source Organism": "Influenza A virus"}),
    ], ignore_index=True))
    out = filter_species(pdf, SPECIES)
    assert len(out) == 1
    assert out.loc[0, "antigen.species"].startswith(CMV_ORGANISM_PREFIX)
