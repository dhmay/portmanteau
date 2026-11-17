"""Naming one antigen species across three databases that do not agree on how to name it.

Selecting "the rows about this pathogen" is a different operation in each database, and none of
them encodes it obviously. A `SpeciesPreset` carries the three spellings together so a caller
names the species once.

⚠️ **The match modes are not interchangeable, and getting one wrong is silent.** Each was chosen
against the data:

- **VDJdb** matches `antigen.species` exactly, because the column is a short controlled value.
- **McPAS** matches `Pathology` exactly for the same reason — and a *pattern* here is a known
  trap: a `cmv|cytomegalo` regex also catches LCMV, which is a different virus entirely.
- **IEDB** matches `antigen.species` by **prefix**, because it records full organism names with
  strain and host detail appended, so an exact match would drop most rows.

A preset with `None` for a database selects nothing from it, which is how a species absent from
one source is handled: no rows, rather than an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeciesPreset:
    """How to recognise one antigen species in each of the three public databases.

    Attributes:
        slug: short identifier used in derived-table filenames, so two species cached under the
            same key can never be confused. Keep it filesystem-safe.
        vdjdb: exact `antigen.species` value, or None to select nothing from VDJdb.
        mcpas: exact `Pathology` value, or None to select nothing from McPAS.
        iedb: `antigen.species` prefix, or None to select nothing from IEDB.
        gene_aliases: `antigen.gene` spellings to fold together, applied after filtering. One
            database writing a gene two ways otherwise splits a single antigen in two.
    """

    slug: str
    vdjdb: str | None = None
    mcpas: str | None = None
    iedb: str | None = None
    gene_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slug or any(c in self.slug for c in r'/\ .'):
            raise ValueError(f"slug must be a non-empty filesystem-safe token, got {self.slug!r}")
        if self.vdjdb is None and self.mcpas is None and self.iedb is None:
            raise ValueError(f"preset {self.slug!r} selects nothing from any database")
