"""The download cache for the public TCR-antigen databases.

A `portmanteau.data.release_cache.ReleaseCache` over `publicdb.releases`, holding the hash
pinning, the content check that catches a 404 served as HTML, the provenance sidecar, the
no-rehash fast path and `derived_frame`.

**Where the files land.** `$PORTMANTEAU_CACHE` if it is set, otherwise the OS cache directory
that `portmanteau.data.download.cache_dir()` picks. The environment variable is the hook for a
machine where the OS cache directory is the wrong filesystem - a small home quota, a scratch
volume, a shared read-only mount - and it is the only thing a consuming project needs to set.

Every name below is a bound method of the module-level cache, so a caller writes
`ensure_release("vdjdb")` or `derived_frame(...)` without touching `CACHE` itself. Reach for
`CACHE` only to get at something this module does not re-export.
"""

from __future__ import annotations

from portmanteau.data.release_cache import ReleaseCache

from .releases import RELEASES

__all__ = ["CACHE", "CACHE_ENV_VAR", "cache_root", "cached_path", "databases", "derived_dir",
           "derived_frame", "derived_root", "download_log_entry", "ensure_release",
           "get_release", "latest_cached", "release_dir", "release_provenance",
           "resolve_release", "verify_hash", "versions"]

#: Override the cache location, e.g. on a machine whose OS cache directory is the wrong volume.
CACHE_ENV_VAR = "PORTMANTEAU_CACHE"

#: What `download_log_entry` writes on its "Command" line. It has to name something a reader can
#: actually run, so it names this module's own entry point.
LOG_COMMAND = ("python -c \"from portmanteau.tcr.publicdb.cache import ensure_release; "
               "ensure_release('{release}')\"")

CACHE = ReleaseCache(RELEASES, root_env=CACHE_ENV_VAR, log_command=LOG_COMMAND)

cache_root = CACHE.cache_root
cached_path = CACHE.cached_path
databases = CACHE.databases
derived_dir = CACHE.derived_dir
derived_frame = CACHE.derived_frame
derived_root = CACHE.derived_root
download_log_entry = CACHE.download_log_entry
ensure_release = CACHE.ensure_release
get_release = CACHE.get_release
latest_cached = CACHE.latest_cached
release_dir = CACHE.release_dir
release_provenance = CACHE.release_provenance
resolve_release = CACHE.resolve_release
verify_hash = CACHE.verify_hash
versions = CACHE.versions
