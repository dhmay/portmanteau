"""The public TCR-antigen databases - VDJdb, McPAS-TCR and IEDB - as one table.

Each database is pinned to a dated release, downloaded once, hash-checked, and reshaped into a
common schema so a row from one can sit beside a row from another. What each module does:

- **`releases`** - the registry: which release of which database, with its URL and sha256.
- **`cache`** - the `ReleaseCache` over that registry. Downloads land in the OS cache directory
  unless `PORTMANTEAU_CACHE` says otherwise.
- **`vdjdb`, `mcpas`, `iedb`** - one module per database, each knowing only its own quirks, each
  exposing `load()` and a species filter.
- **`common`** - the shared column vocabulary the three reshape into.
- **`combined`** - the three as one: row-level union, a long per-(receptor, antigen) table, and a
  flattened per-receptor view.

Selecting an antigen species is a caller's choice, not this package's: the modules take a species
preset rather than hardcoding one, so a project studying one virus supplies its own spellings.

Needs the `publicdb` extra (tidytcells, pooch, pyarrow).
"""
