"""Normalizing HLA names to the IMGT/WHO standard form.

Locks the behaviour we rely on from the pinned tidytcells, and the two things it does not do:
heterodimers and P groups.
"""
import pandas as pd
import pytest

from portmanteau.tcr.hla import (
    CLASS_I,
    CLASS_II,
    hla_class,
    hla_classes,
    expand_p_group,
    hla_field_count,
    hla_field_counts,
    is_p_group,
    normalize_hla,
    normalize_hlas,
)

P_GROUPS = {"HLA-A*23:01P": ["HLA-A*23:01", "HLA-A*23:17"]}


@pytest.mark.parametrize(
    "raw,expected",
    [
        # VDJdb already writes the standard form.
        ("HLA-A*02:01", "HLA-A*02:01"),
        # Some datasets drop the prefix; adding it back is the whole change on that side.
        ("A*01:01", "HLA-A*01:01"),
        ("DRB1*07:01", "HLA-DRB1*07:01"),
        # Some pipelines compress it to the punctuation-free form.
        ("A0201", "HLA-A*02:01"),
        # Deeper fields truncate to protein resolution.
        ("HLA-B*44:03:08", "HLA-B*44:03"),
        ("HLA-E*01:01:01:03", "HLA-E*01:01"),
        # One field is kept as one field: it cannot be refined, and inventing a refinement is worse.
        ("HLA-A*02", "HLA-A*02"),
        # Not an allele, but a real value of VDJdb's mhc.b on class I rows.
        ("B2M", "B2M"),
    ],
)
def test_normalize_hla(raw, expected):
    assert normalize_hla(raw) == expected


def test_heterodimers_keep_both_chains():
    """tidytcells alone returns only the alpha half, silently dropping the beta chain."""
    assert normalize_hla("DQA1*05:01+DQB1*02:01") == "HLA-DQA1*05:01+HLA-DQB1*02:01"


@pytest.mark.parametrize("raw", ["HLA-B*12", "", None, "not-an-allele"])
def test_unresolvable_names_return_none(raw):
    assert normalize_hla(raw) is None


@pytest.mark.parametrize(
    "name,expected",
    [("HLA-A*02:01", 2), ("HLA-A*02", 1), ("B2M", None), (None, None),
     # A heterodimer is limited by its lowest-resolution chain.
     ("HLA-DQA1*05:01+HLA-DQB1*02", 1), ("HLA-DQA1*05:01+HLA-DQB1*02:01", 2)],
)
def test_hla_field_count(name, expected):
    assert hla_field_count(name) == expected


def test_p_groups_are_recognized_and_expanded():
    """A P group names a set of alleles, so it can never equal an allele name in a join."""
    assert normalize_hla("A*23:01P") == "HLA-A*23:01P"
    assert is_p_group("HLA-A*23:01P") and not is_p_group("HLA-A*23:01")
    assert expand_p_group("HLA-A*23:01P", P_GROUPS) == ["HLA-A*23:01", "HLA-A*23:17"]
    # Anything that is not a P group expands to itself, so a caller can map unconditionally.
    assert expand_p_group("HLA-A*02:01", P_GROUPS) == ["HLA-A*02:01"]
    assert expand_p_group(None, P_GROUPS) == []


def test_series_helpers_keep_na():
    names = normalize_hlas(pd.Series(["A*01:01", "HLA-B*12", None, "HLA-A*02"]))
    assert names.tolist() == ["HLA-A*01:01", pd.NA, pd.NA, "HLA-A*02"]
    assert hla_field_counts(names).tolist() == [2, pd.NA, pd.NA, 1]


# --- the coarse class filter ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        # From a resolvable allele.
        ("HLA-A*02:01", "ci"), ("A*01:01", "ci"), ("HLA-DRB1*07:01", "cii"),
        # From a name that states the class instead of an allele. IEDB does this on 147,555 rows.
        ("HLA class I", "ci"), ("HLA class II", "cii"), ("MHCI", "ci"), ("MHCII", "cii"),
        # From the gene letter alone, where no allele can be read: these defeat normalize_hla.
        ("HLA-DR11", "cii"), ("HLA-DQ2", "cii"), ("HLA-DQw1", "cii"), ("DR3*02:02", "cii"),
        ("HLA- DQ5", "cii"),
        # Multi-allele calls and heterodimers resolve when their parts agree.
        ("HLA-E*01:01|HLA-E*01:03", "ci"), ("DQA1*05:01+DQB1*02:01", "cii"),
        # Class I-like but not HLA: a peptide-HLA library must not treat these as class I.
        ("human MR1", None), ("human CD1b", None), ("cattle MR1", None),
        # Nothing legible.
        ("Complete HLA typing is provided in the paper for each patient", None),
        ("nonsense", None), (None, None),
    ],
)
def test_hla_class(name, expected):
    assert hla_class(name) == expected


def test_hla_class_tags_are_the_short_lowercase_tokens():
    """These values reach filenames, column names and join keys, so they are pinned."""
    assert (CLASS_I, CLASS_II) == ("ci", "cii")


def test_hla_classes_series_keeps_na():
    out = hla_classes(pd.Series(["HLA-A*02:01", "human MR1", None, "HLA class II"]))
    assert out.tolist() == ["ci", pd.NA, pd.NA, "cii"]
