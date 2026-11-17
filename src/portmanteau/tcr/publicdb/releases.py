"""Pinned releases of the public TCR-antigen databases.

The `Release` dataclass and everything that acts on one live in `portmanteau.data`; this module
is only the registry of which releases are known. Read `portmanteau.data.releases` for what a
release is and why it is a release rather than a database, and `publicdb.cache` for the cache
over them.

A new upstream release is a new entry here, never a silent change under an existing name.

`sha256` is the hash of whatever the URL returns - the archive, for a zipped release. It starts as
None for an entry nobody has fetched; `ensure_release(..., allow_unpinned=True)` downloads it once
and prints the value to paste back in. After that an upstream change fails loudly.
"""

from __future__ import annotations

from portmanteau.data.releases import Release

__all__ = ["RELEASES", "Release"]


def _entry(release: Release) -> tuple[str, Release]:
    return release.name, release


RELEASES: dict[str, Release] = dict([
    _entry(Release(
        database="vdjdb",
        version="2026-06-03",
        url="https://github.com/antigenomics/vdjdb-db/releases/download/"
            "2026-06-03-ZENODO/vdjdb-2026-06-03.zip",
        sha256="9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1",
        member="vdjdb-2026-06-03/vdjdb_full.txt",
        expect_header="antigen.epitope",
        released="2026-08-26",
        retrieved="2026-10-23",
        citation="Goncharov et al., VDJdb; see vdjdb.com for the current citation. "
                 "The archive ships GNU AGPL-3.0.",
        notes="GitHub release 2026-06-03-ZENODO, 41,017,409 bytes. `member` is vdjdb_full.txt: "
              "192,753 rows, one per clonotype, with separate alpha and beta columns, so it needs "
              "no pivot and carries no comma-separated multi-gene calls. It also has method.* and "
              "meta.* (subject id, donor MHC, tissue) that the other members drop. The archive "
              "also holds vdjdb.slim.txt (197,729 rows, one per chain, complex.id linking pairs — "
              "multi-valued where one chain appears in several pairs) and vdjdb.txt (284,546 rows, "
              "one per chain-and-complex, adds cdr3fix blobs); read those from the cached archive "
              "when needed.",
    )),
    _entry(Release(
        database="mcpas",
        version="2026-10-23",
        url="https://friedmanlab.weizmann.ac.il/McPAS-TCR/McPAS-TCR.csv",
        sha256="ccf7ae8068c787c0bc365f89e52e016a9944cf1775d1980fbf06243e598164d3",
        expect_header="CDR3.beta.aa",
        released=None,
        retrieved="2026-10-23",
        citation="Tickotsky, Sagiv, Prilusky, Shifrut & Friedman, McPAS-TCR: a manually curated "
                 "catalogue of pathology-associated T cell receptor sequences. Bioinformatics "
                 "2017. See friedmanlab.weizmann.ac.il/McPAS-TCR for terms.",
        notes="The site is a Shiny app whose Download button is a session-scoped link, but "
              "McPAS-TCR.csv is served statically at the app root and is the same file - that is "
              "what we pin. ⚠️ The database is updated in place with no version string, so "
              "`version` is the date we retrieved it and the hash is the only identity it has; a "
              "changed hash means upstream changed, not that we did something wrong.",
    )),
    _entry(Release(
        database="iedb",
        version="2026-10-23",
        url="https://www.iedb.org/downloader.php?file_name=doc/receptor_full_v3.zip",
        sha256="566627511a30d72e3d60cb63943ad1c771e814aaa9239f2686517da82671e893",
        member="tcr_full_v3.csv",
        filename="receptor_full_v3.zip",
        expect_header="Receptor",
        released="2026-10-20",
        retrieved="2026-10-23",
        citation="Vita et al., The Immune Epitope Database (IEDB). Nucleic Acids Research. "
                 "IEDB is free to use; see iedb.org for the citation policy.",
        notes="The receptor export, not the assay export: `tcr_full_v3.csv` (167 MB, 301,598 "
              "rows) is receptor-centric and carries CDR3/junction sequences, where "
              "tcell_full_v3.zip is assay-centric. The archive also holds bcr_full_v3.csv. "
              "⚠️ The file has a TWO-ROW header (a group row then a field row) and names chains "
              "'Chain 1'/'Chain 2' rather than alpha/beta. ⚠️ Like McPAS, IEDB is rebuilt in "
              "place with no version string, so `version` is the retrieval date and the hash is "
              "the release's only identity; the inner CSV was dated 2026-10-20.",
    )),
])
