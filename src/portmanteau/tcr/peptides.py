"""Grouping antigen peptides that are the same antigen written differently.

The public databases record one epitope per *assayed peptide*, which is not the same as one per
antigen. Mutational-scanning studies land in VDJdb as a separate epitope per variant, so a single
TCR can appear against 85 "epitopes" that are all NLVPMVATV with one residue changed; and the same
epitope turns up inside longer constructs (`NLVPMVATV` within `ARNLVPMVATVQGQN`). Counting those
as distinct antigens makes every TCR look promiscuous.

Two peptides are treated as one antigen when **one contains the other**, or they are **equal
length and differ at a single position**. On the CMV subsets of vdjdb/2026-06-03,
mcpas/2026-10-23 and iedb/2026-10-23 that reduces 164 peptides to 34 families, with 126 of the
164 falling into the NLVPMVATV family alone.

⚠️ This is a deliberately loose rule and it is not a claim about cross-reactivity. It says two
*records* are describing the same antigen, not that a receptor binds both. Where the question is
whether a TCR really binds two antigens, the families are the unit to count.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def same_antigen(left: str, right: str) -> bool:
    """Whether two peptides are the same antigen written differently.

    Args:
        left: a peptide sequence.
        right: another peptide sequence.

    Returns:
        True when one contains the other, or they are equal length and differ at no more than one
        position. Identical peptides are the same antigen.
    """
    if len(left) != len(right):
        return left in right or right in left
    return sum(a != b for a, b in zip(left, right)) <= 1


def antigen_family_map(
    peptides: Sequence[str], counts: Mapping[str, int] | None = None
) -> dict[str, str]:
    """Map each peptide to the name of its antigen family.

    Args:
        peptides: the peptides to group. Duplicates and order do not matter.
        counts: how much support each peptide has, used to pick the family's representative. The
            best-supported member names the family. Without it the lexicographically first member
            does, so the result is deterministic either way.

    Returns:
        peptide -> family name, covering every input peptide. A peptide with no relatives maps to
        itself.
    """
    unique = sorted(set(peptides))
    parent = {peptide: peptide for peptide in unique}

    def root(peptide: str) -> str:
        while parent[peptide] != peptide:
            parent[peptide] = parent[parent[peptide]]  # path compression
            peptide = parent[peptide]
        return peptide

    # Union-find rather than a graph library: this is one connected-components pass over a few
    # hundred nodes, and it is not worth a dependency.
    for i, left in enumerate(unique):
        for right in unique[i + 1:]:
            if same_antigen(left, right):
                left_root, right_root = root(left), root(right)
                if left_root != right_root:
                    parent[left_root] = right_root

    groups: dict[str, list[str]] = defaultdict(list)
    for peptide in unique:
        groups[root(peptide)].append(peptide)

    def rank(peptide: str) -> tuple[int, str]:
        support = 0 if counts is None else int(counts.get(peptide, 0) or 0)
        return (-support, peptide)

    return {peptide: min(group, key=rank)
            for group in groups.values() for peptide in group}
