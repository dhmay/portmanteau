"""The public-database registry and the cache wired over it.

The cache *machinery* is tested in `test_release_cache.py` - pinning, the no-rehash fast path,
the content checks, the provenance record, `derived_frame`. What is left here is the wiring:
which releases are registered, that each is pinned and guarded, that the root is portable by
default, and that the download-log command names something a reader can run.
"""
import re
from pathlib import Path

import pytest

from portmanteau.data import download
from portmanteau.tcr.publicdb import cache as cache_module
from portmanteau.tcr.publicdb.cache import (
    CACHE,
    CACHE_ENV_VAR,
    cache_root,
    databases,
    derived_root,
    get_release,
    release_dir,
)
from portmanteau.tcr.publicdb.releases import RELEASES


def test_all_three_public_databases_are_registered():
    assert databases() == ["iedb", "mcpas", "vdjdb"]
    assert set(RELEASES) == {"vdjdb/2026-06-03", "mcpas/2026-10-23", "iedb/2026-10-23"}


@pytest.mark.parametrize("name", sorted(RELEASES))
def test_every_registered_release_is_pinned_and_guarded(name):
    """An unpinned or unguarded entry would be fetched without either check that catches a 404."""
    release = get_release(name)
    assert len(release.sha256) == 64, f"{name} is unpinned"
    assert release.expect_header, f"{name} has no content guard"
    assert release.citation and release.retrieved, f"{name} has no provenance"


def test_the_vdjdb_entry_is_the_full_member_not_the_slim_pivot():
    """`vdjdb_full.txt` is one row per clonotype, so it needs no pivot and carries no
    comma-separated multi-gene calls."""
    release = get_release("vdjdb/2026-06-03")
    assert release.url.startswith("https://github.com/antigenomics/vdjdb-db/releases/download/")
    assert release.member.endswith("vdjdb_full.txt")
    assert release_dir(release.name) == release_dir("vdjdb"), "the only version registered"


def test_the_default_root_is_the_os_cache_dir_so_it_works_anywhere(monkeypatch):
    """No hardcoded path: the default has to resolve on a laptop and in a container alike."""
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    root = cache_root()
    assert root == download.cache_dir()
    assert derived_root() == root.parent / "derived"


def test_the_env_var_overrides_the_root(monkeypatch, tmp_path):
    """The one hook a consuming project needs when the OS cache dir is the wrong volume."""
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "elsewhere"))
    assert cache_root() == tmp_path / "elsewhere"


def test_the_env_var_is_read_per_call_not_captured_at_import(monkeypatch, tmp_path):
    """A project that sets the variable after importing this module still gets its root."""
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "first"))
    assert cache_root() == tmp_path / "first"
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "second"))
    assert cache_root() == tmp_path / "second"


def test_the_download_log_command_names_this_modules_entry_point():
    """The command is quoted into a provenance record and has to be runnable from there."""
    command = cache_module.LOG_COMMAND.format(release="vdjdb/2026-06-03")
    assert "from portmanteau.tcr.publicdb.cache import ensure_release" in command
    assert "ensure_release('vdjdb/2026-06-03')" in command


def test_the_module_re_exports_the_caches_methods():
    """Call sites say `ensure_release(...)`, not `CACHE.ensure_release(...)`."""
    for name in cache_module.__all__:
        if name in {"CACHE", "CACHE_ENV_VAR"}:
            continue
        assert getattr(cache_module, name).__self__ is CACHE, f"{name} is not bound to CACHE"


def test_the_registry_is_held_by_reference_so_late_entries_are_seen():
    """`ReleaseCache` must not copy the mapping: a project may register its own release."""
    extra = type(next(iter(RELEASES.values())))(
        database="zzz_test", version="1", url="https://example.invalid/x.csv")
    RELEASES[extra.name] = extra
    try:
        assert "zzz_test" in databases()
    finally:
        del RELEASES[extra.name]
    assert "zzz_test" not in databases()


def test_the_registry_carries_no_local_filesystem_path():
    """This package is public. A release is identified by a URL and a hash; an absolute path in
    any of its prose would be a local detail shipped to everyone who installs it."""
    local_path = re.compile(r"(?<![\w:])/(?!/)[\w.-]+/[\w.-]+")
    for name, release in RELEASES.items():
        blob = f"{release.notes} {release.citation}"
        assert not local_path.search(blob), f"{name} names a filesystem path"
        assert release.url.startswith("https://"), name
