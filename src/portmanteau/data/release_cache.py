"""Ensure a pinned `Release` is on disk, cheaply enough to call on every run.

`portmanteau.data.download` already does the mechanical work — pooch-backed caching, hash
verification, retry with backoff. A `ReleaseCache` adds the bookkeeping an analysis needs on top
of it:

- **a registry of pinned releases**, supplied by the caller, so `<database>/<version>` is the only
  name anything downstream has to know.
- **a content check.** A 404 can return a perfectly well-formed HTML page; a hash will not catch
  that on the fetch that pins the hash.
- **a provenance sidecar**, which makes the fast path possible and feeds `download_log_entry`.
- **a derived-table cache.** A release is raw input; what an analysis actually wants is the
  trimmed table built from it. `derived_frame` builds that once, writes it beside the download
  with a sidecar naming the release it came from, and reads the parquet on every later call.
- **no re-hashing on the happy path.** Pooch verifies the hash on every call when one is given,
  which on a multi-gigabyte file costs half a minute per run. Once a release has been downloaded
  and verified, a matching size against the sidecar is taken as good enough; call `verify_hash` to
  re-check deliberately.

Typical use: build one module-level cache in your project and re-export its methods, so call sites
say `ensure_release("vdjdb")` and never see the instance.

    from portmanteau.data.release_cache import ReleaseCache

    RELEASES = {r.name: r for r in (Release(database="vdjdb", ...),)}
    _CACHE = ReleaseCache(RELEASES, root_env="MYPROJECT_CACHE")

    ensure_release = _CACHE.ensure_release
    derived_frame = _CACHE.derived_frame
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from portmanteau.data.download import cache_dir, fetch, fetch_zip

from .releases import Release

__all__ = ["ReleaseCache"]

SIDECAR_NAME = "_provenance.json"
DERIVED_DIRNAME = "derived"
HTML_PREFIXES = (b"<!doctype", b"<!DOCTYPE", b"<html", b"<HTML", b"<?xml")
SNIFF_BYTES = 64 * 1024

#: What `download_log_entry` writes on its "Command" line when the caller names no other.
DEFAULT_LOG_COMMAND = (
    "python -c \"from portmanteau.data.release_cache import ReleaseCache; "
    "...ensure_release('{release}')\""
)


class ReleaseCache:
    """A registry of pinned releases plus the local cache they are downloaded into.

    Args:
        releases: the registry, as a `{name: Release}` mapping or any iterable of `Release`. A
            mapping is held by reference, not copied, so a project that registers a release later
            — or a test that registers a fake one — is seen here without rebuilding the cache.
        root: where downloads go. A `Path`, or a callable returning one so that nothing is
            resolved at import time. None means `portmanteau.data.download.cache_dir()`, the OS
            cache directory, which works on any machine.
        root_env: name of an environment variable that overrides `root` when set. None means the
            root cannot be overridden that way.
        log_command: template for the "Command" line of `download_log_entry`, with a `{release}`
            placeholder. Name the function your project actually exposes, so the log says
            something a reader can run.
    """

    def __init__(
        self,
        releases: Mapping[str, Release] | Sequence[Release],
        *,
        root: Path | Callable[[], Path] | None = None,
        root_env: str | None = None,
        log_command: str = DEFAULT_LOG_COMMAND,
    ) -> None:
        self.releases: Mapping[str, Release] = (
            releases if isinstance(releases, Mapping) else {r.name: r for r in releases}
        )
        self._root = root
        self.root_env = root_env
        self.log_command = log_command

    # --- the registry ---------------------------------------------------------------------

    def databases(self) -> list[str]:
        """Every database with at least one registered release."""
        return sorted({release.database for release in self.releases.values()})

    def versions(self, database: str) -> list[Release]:
        """Registered releases of one database, oldest first."""
        found = [r for r in self.releases.values() if r.database == database]
        if not found:
            raise ValueError(f"no releases registered for {database!r}; have {self.databases()}")
        return sorted(found, key=lambda r: r.order)

    def get_release(self, name: str) -> Release:
        """The release with this exact `<database>/<version>` key."""
        try:
            return self.releases[name]
        except KeyError:
            raise ValueError(
                f"no release named {name!r}; registered: {sorted(self.releases)}"
            ) from None

    def resolve_release(self, spec: str) -> Release:
        """The release a spec refers to. `<database>/<version>` is exact; a bare name is resolved.

        A bare database name means **the newest version already downloaded**, so asking for
        "vdjdb" keeps giving the same answer until a newer one is fetched, and never triggers a
        surprise download of a release you have not seen. If none is downloaded, it falls back to
        the newest registered version, which `ensure_release` will then fetch.

        ⚠️ That convenience costs reproducibility: the answer depends on what is in *this* cache,
        so the same analysis can resolve differently on another machine. `ensure_release` prints
        the version it resolved for exactly this reason — name the release explicitly in anything
        whose result has to be reproducible.
        """
        if "/" in spec:
            return self.get_release(spec)
        registered = self.versions(spec)
        cached = [release for release in registered if self._cached_path(release) is not None]
        return (cached or registered)[-1]

    # --- where things live ----------------------------------------------------------------

    def cache_root(self) -> Path:
        """Where downloads live: $`root_env` if set, else `root`, else the OS cache directory."""
        override = os.environ.get(self.root_env) if self.root_env else None
        if override:
            return Path(override)
        if self._root is None:
            return cache_dir()
        return Path(self._root() if callable(self._root) else self._root)

    def release_dir(self, spec: str) -> Path:
        """This release's own directory, `<cache root>/<database>/<version>`.

        One directory per release: versions sit side by side, and downloads are untrusted so they
        are kept apart.
        """
        return self.cache_root() / self.resolve_release(spec).name

    def derived_root(self) -> Path:
        """Where derived tables live: a sibling of the download cache, so the two share a root."""
        return self.cache_root().parent / DERIVED_DIRNAME

    def derived_dir(self, spec: str) -> Path:
        """The directory holding tables derived from one release, e.g. .../derived/vdjdb/2026-06-03."""
        return self.derived_root() / self.resolve_release(spec).name

    # --- fetching -------------------------------------------------------------------------

    def release_provenance(self, spec: str) -> dict | None:
        """What the last successful fetch recorded, or None if it has never been fetched here."""
        return self._provenance(self.resolve_release(spec))

    def cached_path(self, spec: str) -> Path | None:
        """The local file for this release if it is already present and intact, else None.

        Never touches the network and never hashes. Use it to ask "do we have this yet?".
        """
        return self._cached_path(self.resolve_release(spec))

    def latest_cached(self, database: str) -> Release | None:
        """The newest version of this database already downloaded here, or None."""
        downloaded = [r for r in self.versions(database) if self._cached_path(r) is not None]
        return downloaded[-1] if downloaded else None

    def ensure_release(self, spec: str, *, allow_unpinned: bool = False,
                       progressbar: bool = False) -> Path:
        """Return the local path to a release, downloading it only if it is not already here.

        Args:
            spec: `<database>/<version>`, or a bare database name — see `resolve_release`. A bare
                name is reported on stdout, so a run's output records which version it used.
            allow_unpinned: permit a release whose `sha256` is still None. The download then
                cannot be verified; the hash it produces is printed, to be pasted into the
                registry.
            progressbar: pass through to the downloader. Needs tqdm.

        Returns:
            Path to the file this release yields — the extracted member for a zip, the download
            itself otherwise.

        Raises:
            ValueError: if the release is unpinned and `allow_unpinned` is False, if the hash does
                not match, or if the content fails its check.
        """
        release = self.resolve_release(spec)
        if "/" not in spec:
            print(f"{spec} resolved to {release.name}")
        existing = self._cached_path(release)
        if existing is not None:
            # Re-run the content check even on the cached path: it costs one 64 KiB read, and it
            # means a release whose declared expectations have changed fails rather than being
            # silently served from the cache.
            _check_content(existing, release)
            return existing

        if release.sha256 is None and not allow_unpinned:
            raise ValueError(
                f"release {release.name!r} has no pinned sha256, so a download cannot be "
                "verified. Fetch it once with allow_unpinned=True, then paste the hash it prints "
                "into the registry so later runs are reproducible and an upstream change fails "
                "loudly."
            )

        directory = self.cache_root() / release.name
        directory.mkdir(parents=True, exist_ok=True)
        filename = release.filename or Path(urlsplit(release.url).path).name
        known_hash = f"sha256:{release.sha256}" if release.sha256 else None

        if release.member:
            extracted = fetch_zip(release.url, members=[release.member], known_hash=known_hash,
                                  fname=filename, dest_dir=directory, progressbar=progressbar)
            matches = [p for p in extracted if p.name == Path(release.member).name]
            if not matches:
                raise ValueError(f"{release.name}: {release.member!r} not found in the archive; "
                                 f"extracted {[p.name for p in extracted]}")
            result = matches[0]
        else:
            result = fetch(release.url, known_hash=known_hash, fname=filename,
                           dest_dir=directory, progressbar=progressbar)

        _check_content(result, release)
        digest = _sha256(directory / filename)
        self._sidecar(release).write_text(json.dumps({
            "release": release.name,
            "url": release.url,
            "sha256": digest,
            "archive": filename,
            "member": release.member,
            "path": str(result),
            "bytes": result.stat().st_size,
            "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n")

        if release.sha256 is None:
            print(f"{release.name} is unpinned. Add to the registry:\n    sha256=\"{digest}\",")
        return result

    def verify_hash(self, spec: str) -> str:
        """Re-hash the downloaded archive and check it against the registry. Returns the hash.

        The happy path of `ensure_release` trusts the sidecar rather than re-hashing, so this is
        how you deliberately confirm a cached copy is still what it claims to be.
        """
        release = self.resolve_release(spec)
        record = self._provenance(release)
        if record is None:
            raise ValueError(
                f"{release.name!r} has not been fetched into {self.cache_root() / release.name}")
        digest = _sha256(self.cache_root() / release.name / record["archive"])
        if release.sha256 is not None and digest != release.sha256:
            raise ValueError(f"{release.name}: sha256 is {digest}, registry says {release.sha256}")
        return digest

    def download_log_entry(self, spec: str) -> str:
        """A stanza for a download log, to paste in by hand.

        Deliberately returned rather than appended: a library function that writes to a tracked
        file would fire on every rerun.
        """
        release = self.resolve_release(spec)
        record = self._provenance(release)
        if record is None:
            raise ValueError(
                f"{release.name!r} has not been fetched into {self.cache_root() / release.name}")
        date = record["retrieved"][:10]
        member = f" (member `{record['member']}`)" if record["member"] else ""
        return (
            f"## {date} — {release.name}\n"
            f"- Source: {release.url}\n"
            f"- Command: `{self.log_command.format(release=release.name)}`\n"
            f"- Destination: {record['path']}{member}\n"
            f"- Size: {record['bytes']:,} bytes; sha256 of the download {record['sha256']}\n"
            f"- Contents: {release.notes or 'TODO'}\n"
        )

    # --- derived tables -------------------------------------------------------------------

    def derived_frame(
        self,
        spec: str | Sequence[str],
        filename: str,
        build: Callable[[Any], pd.DataFrame],
        *,
        refresh: bool = False,
        notes: dict | None = None,
    ) -> pd.DataFrame:
        """Return a table derived from one or more releases, building and caching it on first use.

        The point is that an analysis can call this every run. The expensive part - reading a
        large source and reshaping it - happens once; afterwards this reads a parquet.

        The cached table is rebuilt when any contributing release's hash or member changes, so
        re-pinning a release cannot leave a stale derived table behind. It is *not* rebuilt when
        `build` changes, which no sidecar can detect: pass refresh=True after editing the build.

        Args:
            spec: one release, as `ensure_release` takes it, or several. Several are allowed
                because some tables aggregate sources, and such a table is invalidated by a change
                in any of them.
            filename: the parquet to write, e.g. "vdjdb.parquet".
            build: called with the path `ensure_release` returned for a single spec, or with a
                {spec: path} mapping for several; returns the table to cache.
            refresh: rebuild even if a current cached table exists.
            notes: extra fields to record in the sidecar, e.g. what the build did.

        Returns:
            The derived table, read back from the parquet it was written to.
        """
        specs = [spec] if isinstance(spec, str) else list(spec)
        if not specs:
            raise ValueError("derived_frame needs at least one release spec")
        releases = [self.resolve_release(one) for one in specs]

        # One release keeps its own directory; an aggregate belongs to none of them.
        directory = (self.derived_dir(releases[0].name) if len(releases) == 1
                     else self.derived_root() / "combined")
        path = directory / filename
        sources = {}
        for one, release in zip(specs, releases):
            source_path = self.ensure_release(release.name)
            record = self._provenance(release) or {}
            sources[release.name] = {
                "requested": one, "path": str(source_path), "sha256": record.get("sha256"),
                "member": release.member, "citation": release.citation,
            }
        fingerprint = sorted((name, source["sha256"], source["member"])
                             for name, source in sources.items())

        sidecar = path.with_suffix(".source.json")
        if not refresh and path.exists() and sidecar.exists():
            cached = json.loads(sidecar.read_text())
            if [list(entry) for entry in fingerprint] == cached.get("fingerprint"):
                return pd.read_parquet(path)

        argument = (Path(sources[releases[0].name]["path"]) if len(releases) == 1
                    else {name: Path(source["path"]) for name, source in sources.items()})
        pdf = build(argument)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(path, index=False)
        sidecar.write_text(json.dumps({
            "releases": list(sources),
            "fingerprint": [list(entry) for entry in fingerprint],
            "sources": sources,
            "rows": len(pdf),
            "columns": list(pdf.columns),
            "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(notes or {}),
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"built {path} ({len(pdf):,} rows x {len(pdf.columns)} columns)")
        return pd.read_parquet(path)

    # --- internals ------------------------------------------------------------------------

    def _sidecar(self, release: Release) -> Path:
        return self.cache_root() / release.name / SIDECAR_NAME

    def _provenance(self, release: Release) -> dict | None:
        path = self._sidecar(release)
        return json.loads(path.read_text()) if path.is_file() else None

    def _cached_path(self, release: Release) -> Path | None:
        record = self._provenance(release)
        if not record or record.get("url") != release.url:
            return None
        # `member` decides *which file* is returned, so a change to it invalidates the cache just
        # as surely as a change to the url. Without this, setting `member` on a release already
        # fetched whole keeps handing back the archive.
        if record.get("member") != release.member:
            return None
        if release.sha256 is not None and record.get("sha256") != release.sha256:
            return None
        path = Path(record["path"])
        return path if path.is_file() and path.stat().st_size == record["bytes"] else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_content(path: Path, release: Release) -> None:
    """Reject an HTML error page or a truncated file before anything downstream sees it."""
    size = path.stat().st_size
    if size < release.min_bytes:
        raise ValueError(f"{release.name}: {path} is {size:,} bytes, under the "
                         f"{release.min_bytes:,} expected. Truncated, or an error page?")
    head = path.read_bytes()[:SNIFF_BYTES]
    if head.lstrip()[:16].startswith(HTML_PREFIXES):
        raise ValueError(f"{release.name}: {path} starts like an HTML or XML document, which is "
                         "what a 404 page looks like. Check the URL.")
    if release.expect_header and release.expect_header not in head.decode("utf-8", "replace"):
        raise ValueError(f"{release.name}: {release.expect_header!r} not in the first "
                         f"{SNIFF_BYTES:,} bytes of {path}; wrong file, or the format changed.")
