"""A pinned, downloadable release of an external dataset.

A **release**, not a dataset. VDJdb 2026-06-03 and VDJdb 2022 are different objects, and an
analysis that names one states which it used. A new upstream release is a new `Release`, never a
silent change under an existing name.

A release is identified by `<database>/<version>` — `vdjdb/2026-06-03` — which is also its layout
on disk, so versions of the same database sit side by side and pinning a new one never overwrites
the one an earlier result came from.

`sha256` is the hash of whatever the URL returns — the archive, for a zipped release. It starts as
None for an entry nobody has fetched; `ReleaseCache.ensure_release(..., allow_unpinned=True)`
downloads it once and prints the value to paste back in.

⚠️ **What that hash then *means* depends on `rolling`**, because not every upstream lets you ask
for a version. A URL that names one (`.../releases/download/2026-06-03/...`) can be re-fetched
forever, so a hash mismatch is a real error. A URL that serves "the current build"
(`downloader.php?file_name=x.zip`) cannot: the moment upstream rebuilds, the old bytes are gone
and a mismatch means only that time has passed. Marking the second kind `rolling=True` says the
hash records which build is in use rather than demanding one.

This module holds only the description of a release. The registry of which releases exist belongs
to the project that uses them; hand it to `portmanteau.data.release_cache.ReleaseCache`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Release"]


@dataclass(frozen=True)
class Release:
    """One downloadable release of an external dataset.

    Attributes:
        database: the resource, e.g. "vdjdb". Shared by all its versions.
        version: the upstream's own version string, e.g. "2026-06-03".
        url: direct download URL.
        sha256: hash of the downloaded file, or None while unpinned.
        member: for a zip, the member to extract and return. None for a plain file.
        filename: local name for the download. Defaults to the URL's basename.
        expect_header: text that must appear near the start of the returned file. The cheapest
            guard against a 404 that returns an HTML page, which a hash cannot catch on the first
            fetch because that fetch is the one doing the pinning.
        min_bytes: refuse anything smaller.
        rolling: the URL serves whatever upstream currently has, rather than naming a version.
            Then `sha256` is a **record** of the build in use, not a contract: a cold fetch
            accepts what arrives and says so, because the pinned bytes are unobtainable. Leave
            False — the default — wherever the URL names a version, so a mismatch stays an error.
            ⚠️ This is consulted only on a cache miss. A cached release is served without
            re-hashing and without contacting the network either way.
        citation: what to cite when the data is used.
        released: upstream release date, ISO. Used to order versions when the version string does
            not sort chronologically by itself.
        retrieved: date the hash was pinned, as YYYY-MM-DD.
        notes: anything a reader needs before trusting it.
    """

    database: str
    version: str
    url: str
    sha256: str | None = None
    member: str | None = None
    filename: str | None = None
    expect_header: str | None = None
    min_bytes: int = 1024
    rolling: bool = False
    citation: str = ""
    released: str | None = None
    retrieved: str | None = None
    notes: str = ""

    @property
    def name(self) -> str:
        """`<database>/<version>`: the registry key and the path under the cache root."""
        return f"{self.database}/{self.version}"

    @property
    def order(self) -> str:
        """Sort key for "newest". The release date if we have one, else the version string."""
        return self.released or self.version
