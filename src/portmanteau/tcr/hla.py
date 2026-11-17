"""Normalizing HLA allele names to the IMGT/WHO standard form.

The same allele is written at least three ways across the public databases and the pipelines
that read them: `HLA-A*02:01`, the prefix dropped (`A*01:01`), and the punctuation dropped with
it (`A0201`). The first is the WHO Nomenclature Committee / IMGT-HLA standard, so that is what
this module consolidates on; the second differs from it by a pure prefix addition.

`tidytcells.mh` does the work. Standardization is at **protein level** (two fields): it is the
resolution HLA typing is usually reported at, and the deepest that most datasets carry.
Higher-resolution VDJdb values truncate to it (`HLA-B*44:03:08` -> `HLA-B*44:03`).

Two things it will not do:

- **Heterodimers.** Handed `DQA1*05:01+DQB1*02:01` it returns only the alpha half, silently
  dropping the beta chain. We split on "+" and standardize each part. Class II only.
- **P groups.** `A*23:01P` names a group of alleles with identical peptide-binding domains, not an
  allele; tidytcells passes the `P` through without understanding it, so `HLA-A*23:01P` would
  never match VDJdb's `HLA-A*23:01`. Stripping the P would silently discard a deliberate choice
  made by whatever produced the name, so instead `expand_p_group` turns a P group into its
  members at join time.

`hla_field_count` records the resolution actually achieved, because some values cannot reach two
fields: in one VDJdb subset, 16% of Class I rows are one-field (`HLA-A*02`) and cannot be
refined.
Whether a one-field value may match a two-field one is a join policy, not a normalization
decision, so it is left to the caller.
"""
import re

import pandas as pd
import tidytcells as tt

#: Two fields: the allele group and the protein, e.g. HLA-A*02:01. Deeper fields are synonymous or
#: non-coding differences that no typing we join against resolves.
PRECISION = "protein"
#: Separates the chains of a class II heterodimer, e.g. `DQA1*05:01+DQB1*02:01`.
HETERODIMER_SEPARATOR = "+"
#: Marks a P group rather than an allele, e.g. A*23:01P.
P_GROUP_SUFFIX = "P"

#: MHC class tags. Short lowercase tokens rather than "I"/"II", so a value is safe to use in a
#: filename, a column name and a join key without quoting or case rules. A caller comparing
#: against a dataset that spells the class differently maps at the boundary.
CLASS_I = "ci"
CLASS_II = "cii"
_CLASS_BY_NUMBER = {1: CLASS_I, 2: CLASS_II}
#: Some sources name the class instead of an allele; IEDB does this on 147,555 rows.
EXPLICIT_CLASS = {"hla class i": CLASS_I, "hla class ii": CLASS_II, "mhci": CLASS_I,
                  "mhcii": CLASS_II, "class i": CLASS_I, "class ii": CLASS_II}
#: Last resort, for names too coarse or too malformed to standardize but whose gene is legible:
#: HLA-DR11, HLA-DQw1, DR3*02:02. The classical class I genes are A, B, C, E, F and G.
CLASS_BY_GENE_PATTERN = (
    (re.compile(r"^(?:HLA-\s*)?D[RQP]", re.IGNORECASE), CLASS_II),
    (re.compile(r"^(?:HLA-\s*)?[ABCEFG][\*\d]", re.IGNORECASE), CLASS_I),
)
#: Antigen-presenting molecules that are class I-*like* but not HLA: MR1 presents metabolites,
#: CD1 presents lipids. Deliberately left unclassified — calling them class I would be wrong for
#: a peptide-HLA library.
NON_CLASSICAL_PATTERN = re.compile(r"\b(MR1|CD1[abcd]?)\b", re.IGNORECASE)


def normalize_hla(name: str | None) -> str | None:
    """Return one HLA name in standard IMGT form at protein resolution, or None if unresolvable.

    Args:
        name: an allele name in any of the forms above, optionally a "+"-joined class II
            heterodimer.

    Returns:
        e.g. "HLA-A*02:01"; a heterodimer comes back with both chains standardized and still
        joined by "+". None if no part could be resolved.
    """
    if name is None or pd.isna(name):
        return None
    parts = [
        tt.mh.standardize(part, precision=PRECISION, on_fail="reject", log_failures=False)
        for part in str(name).split(HETERODIMER_SEPARATOR)
    ]
    return HETERODIMER_SEPARATOR.join(parts) if all(parts) else None


def hla_field_count(name: str | None) -> int | None:
    """How many fields the normalized name actually resolves to: 1 (HLA-A*02) or 2 (HLA-A*02:01).

    For a heterodimer this is the *lowest* resolution of its chains, since that is what limits a
    match. Returns None for an unresolvable or non-allele name (B2M has no fields).
    """
    if name is None or pd.isna(name):
        return None
    counts = [part.count(":") + 1 for part in str(name).split(HETERODIMER_SEPARATOR) if "*" in part]
    return min(counts) if counts else None


def is_p_group(name: str | None) -> bool:
    """Whether a normalized name is a P group (a set of alleles) rather than a single allele."""
    if name is None or pd.isna(name):
        return False
    return str(name).endswith(P_GROUP_SUFFIX)


def expand_p_group(name: str | None, p_groups: dict[str, list[str]]) -> list[str]:
    """Return the alleles a P group stands for, or just the name itself if it is not one.

    A P group cannot match an allele name directly, so a join expands it to its members and
    matches on any of them.

    Args:
        name: a normalized HLA name.
        p_groups: normalized P group name -> its normalized member alleles.

    Returns:
        A list of normalized allele names; empty if name is missing.
    """
    if name is None or pd.isna(name):
        return []
    return list(p_groups.get(str(name), [str(name)]))


def normalize_hlas(names: pd.Series) -> pd.Series:
    """Vectorized normalize_hla(), resolving each distinct value once."""
    resolved = {name: normalize_hla(name) for name in names.dropna().unique()}
    return names.map(resolved).astype("string")


def hla_field_counts(names: pd.Series) -> pd.Series:
    """Vectorized hla_field_count() over already-normalized names."""
    return names.map(hla_field_count).astype("Int8")


def hla_class(name: str | None) -> str | None:
    """Return "ci" or "cii" for an HLA name, or None if the class cannot be told.

    Spelled `ci`/`cii`; see `CLASS_I`. This is the
    coarsest useful filter, and it survives names that `normalize_hla` cannot resolve — the point
    is that a coarse match is still better than no match when the allele itself is ambiguous.

    Three sources of truth, in order: a name that states the class outright ("HLA class II", which
    IEDB uses on 147,555 rows); the standardized allele, via tidytcells; and failing both, the
    gene letter in the raw name (`HLA-DQw1` is class II even though no allele can be read from it).

    A multi-allele call ("|"-separated) or a heterodimer ("+") resolves only if its parts agree.
    MR1 and CD1 return None: they are class I-like but not HLA, and a peptide-HLA library should
    not treat them as class I.

    Args:
        name: an HLA name in any form, standardized or not.

    Returns:
        CLASS_I, CLASS_II, or None.
    """
    if name is None or pd.isna(name):
        return None
    text = str(name).strip()
    if text.lower() in EXPLICIT_CLASS:
        return EXPLICIT_CLASS[text.lower()]
    if NON_CLASSICAL_PATTERN.search(text):
        return None
    found = set()
    for part in re.split(r"[|+]", text):
        part = part.strip()
        if not part:
            continue
        standardized = normalize_hla(part)
        number = tt.mh.get_class(standardized, log_failures=False) if standardized else None
        if number in _CLASS_BY_NUMBER:
            found.add(_CLASS_BY_NUMBER[number])
            continue
        for pattern, tag in CLASS_BY_GENE_PATTERN:
            if pattern.match(part):
                found.add(tag)
                break
    return found.pop() if len(found) == 1 else None


def hla_classes(names: pd.Series) -> pd.Series:
    """Vectorized hla_class(), resolving each distinct value once."""
    resolved = {name: hla_class(name) for name in names.dropna().unique()}
    return names.map(resolved).astype("string")
