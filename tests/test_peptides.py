"""Grouping peptides that are the same antigen written differently."""
import pytest

from portmanteau.tcr.peptides import antigen_family_map, same_antigen


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("NLVPMVATV", "NLVPMVATV", True),          # identical
        ("NLVPMVATV", "ALVPMVATV", True),          # one substitution
        ("NLVPMVATV", "ALVPMVATA", False),         # two substitutions
        ("NLVPMVATV", "ARNLVPMVATVQGQN", True),    # the short one sits inside the long one
        ("NLVPMVATV", "KLGGALQAK", False),         # unrelated, same length
        ("NLVPMVATV", "TPRVTGGGAM", False),        # unrelated, different length
    ],
)
def test_same_antigen(left, right, expected):
    assert same_antigen(left, right) is expected
    assert same_antigen(right, left) is expected, "the rule must be symmetric"


def test_scan_variants_collapse_into_one_family():
    """The case this exists for: VDJdb records a substitution scan as one epitope per variant."""
    scan = ["NLVPMVATV"] + [f"{a}LVPMVATV" for a in "ACDEFGHIK"]
    families = antigen_family_map(scan + ["KLGGALQAK"])
    assert len(set(families.values())) == 2
    assert families["KLGGALQAK"] == "KLGGALQAK"


def test_transitive_grouping():
    """A and C differ at two positions but both are one away from B, so all three are one family."""
    families = antigen_family_map(["AAAAAAAAA", "AAAAAAAAB", "AAAAAAACB"])
    assert len(set(families.values())) == 1


def test_counts_choose_the_representative():
    families = antigen_family_map(["NLVPMVATV", "ALVPMVATV"], {"ALVPMVATV": 99, "NLVPMVATV": 1})
    assert set(families.values()) == {"ALVPMVATV"}


def test_deterministic_without_counts():
    """No counts: the lexicographically first member names the family, so runs agree."""
    peptides = ["NLVPMVATV", "ALVPMVATV", "CLVPMVATV"]
    assert antigen_family_map(peptides) == antigen_family_map(list(reversed(peptides)))
    assert set(antigen_family_map(peptides).values()) == {"ALVPMVATV"}


def test_every_peptide_is_mapped_and_singletons_map_to_themselves():
    families = antigen_family_map(["NLVPMVATV", "KLGGALQAK", "NLVPMVATV"])
    assert families == {"NLVPMVATV": "NLVPMVATV", "KLGGALQAK": "KLGGALQAK"}


def test_empty_input():
    assert antigen_family_map([]) == {}
