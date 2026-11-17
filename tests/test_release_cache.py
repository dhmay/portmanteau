"""The registry/cache layer, tested without a network.

The downloader and pooch underneath it are tested in test_download.py, so the stub here stands in
for them and these tests cover only what ReleaseCache adds: pinning, the no-rehash fast path, the
content checks, the provenance record and the derived-table cache.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from portmanteau.data import release_cache as cache_module
from portmanteau.data.release_cache import ReleaseCache
from portmanteau.data.releases import Release


def _sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


PAYLOAD = b"complex.id\tcdr3\tantigen.epitope\n1\tCASSLF\tNLVPMVATV\n" + b"x" * 2000


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A cache over a temp directory, a way to register test releases, and a stub downloader."""
    registry: dict[str, Release] = {}
    cache = ReleaseCache(registry, root=tmp_path / "cache")
    calls = []

    def stub_fetch(url, *, known_hash=None, fname=None, dest_dir=None, progressbar=False):
        calls.append(url)
        target = dest_dir / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload["bytes"])
        return target

    monkeypatch.setattr(cache_module, "fetch", stub_fetch)
    payload = {"bytes": PAYLOAD}

    def register(version="v1", **overrides):
        release = Release(database="testdb", version=version,
                          url=f"https://example.invalid/{version}.tsv", **overrides)
        registry[release.name] = release
        return release

    return cache, register, calls, payload


def test_unpinned_release_is_refused_by_default(registered):
    cache, register, calls, _ = registered
    register()
    with pytest.raises(ValueError, match="no pinned sha256"):
        cache.ensure_release("testdb/v1")
    assert not calls, "nothing should have been downloaded"


def test_allow_unpinned_downloads_and_records_provenance(registered, capsys):
    cache, register, calls, _ = registered
    register()
    path = cache.ensure_release("testdb/v1", allow_unpinned=True)
    assert path.read_bytes() == PAYLOAD and len(calls) == 1
    record = cache.release_provenance("testdb/v1")
    assert record["url"] == "https://example.invalid/v1.tsv"
    assert record["bytes"] == len(PAYLOAD)
    assert record["sha256"] in capsys.readouterr().out, "the hash to pin should be printed"


def test_second_call_does_not_download_or_rehash(registered):
    cache, register, calls, _ = registered
    register()
    first = cache.ensure_release("testdb/v1", allow_unpinned=True)
    digest = cache.release_provenance("testdb/v1")["sha256"]
    register(sha256=digest)
    assert cache.ensure_release("testdb/v1") == first
    assert len(calls) == 1, "the pinned, cached release should not be fetched again"


def test_a_changed_pin_invalidates_the_cache(registered):
    cache, register, calls, _ = registered
    register()
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    register(sha256="deadbeef")
    assert cache.cached_path("testdb/v1") is None
    cache.ensure_release("testdb/v1")            # re-fetches under the new pin
    assert len(calls) == 2


def test_a_changed_url_invalidates_the_cache(registered):
    cache, register, _, _ = registered
    register()
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    cache.releases["testdb/v1"] = Release(database="testdb", version="v1",
                                          url="https://example.invalid/other.tsv")
    assert cache.cached_path("testdb/v1") is None


def test_html_error_page_is_rejected(registered):
    cache, register, _, payload = registered
    register()
    payload["bytes"] = b"<!DOCTYPE html>\n<html><body>404 Not Found</body></html>" + b" " * 2000
    with pytest.raises(ValueError, match="HTML or XML"):
        cache.ensure_release("testdb/v1", allow_unpinned=True)


def test_a_truncated_file_is_rejected(registered):
    cache, register, _, payload = registered
    register()
    payload["bytes"] = b"oops"
    with pytest.raises(ValueError, match="under the"):
        cache.ensure_release("testdb/v1", allow_unpinned=True)


def test_expected_header_must_be_present(registered):
    cache, register, _, _ = registered
    register(expect_header="complex.id")
    cache.ensure_release("testdb/v1", allow_unpinned=True)        # present, so fine

    register(expect_header="not-in-this-file")
    with pytest.raises(ValueError, match="not in the first"):
        cache.ensure_release("testdb/v1", allow_unpinned=True)


def test_verify_hash_checks_a_cached_copy(registered):
    cache, register, _, _ = registered
    register()
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    digest = cache.release_provenance("testdb/v1")["sha256"]
    register(sha256=digest)
    assert cache.verify_hash("testdb/v1") == digest

    register(sha256="nope")
    with pytest.raises(ValueError, match="registry says"):
        cache.verify_hash("testdb/v1")


def test_download_log_entry_has_the_fields_the_log_wants(registered):
    cache, register, _, _ = registered
    register(notes="a test file")
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    entry = cache.download_log_entry("testdb/v1")
    for expected in ("## ", "- Source: https://example.invalid/v1.tsv", "- Destination:",
                     f"- Size: {len(PAYLOAD):,} bytes", "sha256", "- Contents: a test file"):
        assert expected in entry, f"{expected!r} missing from:\n{entry}"


def test_download_log_entry_names_the_caller_s_own_function(tmp_path, monkeypatch):
    """The log line has to be runnable by whoever reads it, so the project names its own entry
    point rather than inheriting portmanteau's."""
    registry = {}
    cache = ReleaseCache(registry, root=tmp_path,
                         log_command="from myproj.io import ensure; ensure('{release}')")
    monkeypatch.setattr(cache_module, "fetch", lambda url, **kw: _write(kw["dest_dir"],
                                                                       kw["fname"], PAYLOAD))
    registry["testdb/v1"] = Release(database="testdb", version="v1",
                                    url="https://example.invalid/v1.tsv")
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    assert "from myproj.io import ensure; ensure('testdb/v1')" in \
        cache.download_log_entry("testdb/v1")


def _write(dest_dir, fname, payload):
    target = dest_dir / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def test_unknown_release_lists_what_exists(registered):
    cache, _, _, _ = registered
    with pytest.raises(ValueError, match="registered:"):
        cache.get_release("no-such-release")


def test_an_unregistered_database_says_what_is_registered(registered):
    cache, register, _, _ = registered
    register()
    with pytest.raises(ValueError, match="have \\['testdb'\\]"):
        cache.versions("nosuchdb")


def test_the_root_env_var_overrides_the_default_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MYPROJ_CACHE", str(tmp_path / "from-env"))
    cache = ReleaseCache({}, root=tmp_path / "default", root_env="MYPROJ_CACHE")
    assert cache.cache_root() == tmp_path / "from-env"
    monkeypatch.delenv("MYPROJ_CACHE")
    assert cache.cache_root() == tmp_path / "default"


def test_the_default_root_may_be_a_callable_so_nothing_resolves_at_import(tmp_path):
    calls = []

    def root():
        calls.append(1)
        return tmp_path / "lazy"

    cache = ReleaseCache({}, root=root)
    assert not calls, "constructing the cache must not resolve the root"
    assert cache.cache_root() == tmp_path / "lazy" and len(calls) == 1


def test_with_no_root_the_cache_falls_back_to_the_os_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTMANTEAU_DATA_DIR", str(tmp_path / "os-cache"))
    assert ReleaseCache({}).cache_root() == tmp_path / "os-cache"


def test_a_bare_database_name_resolves_to_the_newest_cached_version(registered, capsys):
    cache, register, calls, _ = registered
    register("2026-01-01")
    register("2026-06-03")

    # Nothing downloaded: fall back to the newest *registered* version
    assert cache.resolve_release("testdb").version == "2026-06-03"
    assert cache.latest_cached("testdb") is None

    # Download only the older one; the bare name should then give that, not the newer
    cache.ensure_release("testdb/2026-01-01", allow_unpinned=True)
    assert cache.latest_cached("testdb").version == "2026-01-01"
    assert cache.resolve_release("testdb").version == "2026-01-01"
    assert cache.ensure_release("testdb", allow_unpinned=True) == \
        cache.cached_path("testdb/2026-01-01")
    assert len(calls) == 1, "resolving a bare name must not trigger a surprise download"
    assert "testdb resolved to testdb/2026-01-01" in capsys.readouterr().out

    # Fetch the newer one and it supersedes
    cache.ensure_release("testdb/2026-06-03", allow_unpinned=True)
    assert cache.resolve_release("testdb").version == "2026-06-03"


def test_versions_are_ordered_by_release_date_when_given(registered):
    cache, register, _, _ = registered
    register("beta", released="2026-09-01")
    register("alpha", released="2026-10-01")   # sorts later by date, earlier by name
    assert [r.version for r in cache.versions("testdb")] == ["beta", "alpha"]
    assert cache.resolve_release("testdb").version == "alpha"


def test_versions_live_in_separate_directories(registered):
    cache, register, _, _ = registered
    register("2026-01-01")
    register("2026-06-03")
    first = cache.release_dir("testdb/2026-01-01")
    second = cache.release_dir("testdb/2026-06-03")
    assert first != second and first.parent == second.parent
    assert first.parent.name == "testdb" and first.name == "2026-01-01"


def test_setting_a_member_invalidates_a_cache_of_the_whole_archive(registered):
    """`member` decides which file comes back, so changing it must not reuse the old record."""
    cache, register, _, _ = registered
    register()
    cache.ensure_release("testdb/v1", allow_unpinned=True)
    assert cache.cached_path("testdb/v1") is not None

    register(member="inner.txt")
    assert cache.cached_path("testdb/v1") is None, \
        "a new member must invalidate the cached archive"


# --- derived_frame -------------------------------------------------------------------------


@pytest.fixture
def built(registered):
    """A registered release plus a counter for how many times the build callable ran."""
    cache, register, calls, payload = registered
    register(sha256=_sha256_of(PAYLOAD))
    builds = []

    def build(source):
        # a Path for one release, a {spec: Path} mapping for several
        builds.append(source)
        label = source.name if hasattr(source, "name") else ",".join(sorted(source))
        return pd.DataFrame({"n": [1, 2, 3], "src": [label] * 3})

    return cache, register, build, builds, payload


def test_derived_frame_builds_once_then_reads_the_parquet(built):
    cache, _, build, builds, _ = built
    first = cache.derived_frame("testdb/v1", "t.parquet", build)
    second = cache.derived_frame("testdb/v1", "t.parquet", build)
    assert len(builds) == 1, "the second call must not rebuild"
    pd.testing.assert_frame_equal(first, second)
    assert first["n"].tolist() == [1, 2, 3]


def test_derived_frame_writes_a_sidecar_naming_its_source(built):
    cache, _, build, _, _ = built
    cache.derived_frame("testdb/v1", "t.parquet", build, notes={"filters": "none"})
    sidecar = json.loads((cache.derived_dir("testdb/v1") / "t.source.json").read_text())
    assert sidecar["releases"] == ["testdb/v1"]
    assert sidecar["rows"] == 3 and sidecar["columns"] == ["n", "src"]
    assert sidecar["filters"] == "none", "notes are merged into the sidecar"
    source = sidecar["sources"]["testdb/v1"]
    assert source["sha256"] == _sha256_of(PAYLOAD) and source["member"] is None


def test_derived_frame_rebuilds_when_refresh_is_asked_for(built):
    cache, _, build, builds, _ = built
    cache.derived_frame("testdb/v1", "t.parquet", build)
    cache.derived_frame("testdb/v1", "t.parquet", build, refresh=True)
    assert len(builds) == 2


def test_derived_frame_rebuilds_when_the_release_hash_changes(built):
    cache, register, build, builds, payload = built
    cache.derived_frame("testdb/v1", "t.parquet", build)
    # Re-pin the same release on different upstream content: the cached table is now stale.
    payload["bytes"] = PAYLOAD + b"more"
    register(sha256=_sha256_of(payload["bytes"]))
    cache.derived_frame("testdb/v1", "t.parquet", build)
    assert len(builds) == 2, "a changed source hash must invalidate the derived table"


def test_derived_frame_keeps_tables_of_different_releases_apart(built):
    cache, register, build, builds, _ = built
    register(version="v2", sha256=_sha256_of(PAYLOAD))
    cache.derived_frame("testdb/v1", "t.parquet", build)
    cache.derived_frame("testdb/v2", "t.parquet", build)
    assert len(builds) == 2
    assert cache.derived_dir("testdb/v1") != cache.derived_dir("testdb/v2")
    assert (cache.derived_dir("testdb/v2") / "t.parquet").exists()


def test_derived_root_sits_beside_the_downloads(built):
    cache, _, _, _, _ = built
    assert cache.derived_root().parent == cache.cache_root().parent
    assert cache.derived_root().name == "derived"


def test_derived_frame_accepts_several_releases(built):
    """A table aggregating sources belongs to none of their directories, and is invalidated by
    a change in any of them."""
    cache, register, build, builds, _ = built
    register(version="v2", sha256=_sha256_of(PAYLOAD))

    first = cache.derived_frame(["testdb/v1", "testdb/v2"], "combo.parquet", build)
    assert len(first) == 3
    assert (cache.derived_root() / "combined" / "combo.parquet").exists()
    assert isinstance(builds[-1], dict) and set(builds[-1]) == {"testdb/v1", "testdb/v2"}

    cache.derived_frame(["testdb/v1", "testdb/v2"], "combo.parquet", build)
    assert len(builds) == 1, "the second call must not rebuild"

    sidecar = json.loads((cache.derived_root() / "combined" / "combo.source.json").read_text())
    assert sidecar["releases"] == ["testdb/v1", "testdb/v2"]


def test_derived_frame_rebuilds_when_any_contributing_release_changes(built):
    cache, register, build, builds, payload = built
    register(version="v2", sha256=_sha256_of(PAYLOAD))
    cache.derived_frame(["testdb/v1", "testdb/v2"], "combo.parquet", build)

    payload["bytes"] = PAYLOAD + b"changed"
    register(version="v2", sha256=_sha256_of(payload["bytes"]))
    cache.derived_frame(["testdb/v1", "testdb/v2"], "combo.parquet", build)
    assert len(builds) == 2, "a change in the second release must invalidate the table"


def test_derived_frame_refuses_an_empty_spec_list(built):
    cache, _, build, _, _ = built
    with pytest.raises(ValueError, match="at least one"):
        cache.derived_frame([], "t.parquet", build)


def test_a_registry_given_as_a_sequence_is_keyed_by_name(tmp_path):
    cache = ReleaseCache([Release(database="d", version="1", url="https://example.invalid/a")],
                         root=tmp_path)
    assert cache.get_release("d/1").url.endswith("/a")
    assert cache.databases() == ["d"]


def _writer(body):
    """A stand-in for `fetch` that writes fixed bytes and returns the path."""
    def _fetch(url, known_hash=None, fname=None, dest_dir=None, progressbar=False):
        target = Path(dest_dir) / fname
        target.write_bytes(body)
        return target
    return _fetch

# --- rolling releases, update checks and expiry ----------------------------------------------
#
# A rolling release is one whose URL serves "the current build" rather than naming a version, so
# its pinned bytes stop existing the moment upstream rebuilds.


def _rolling_registry(tmp_path, body=b"header\nrow\n" + b"x" * 2048, sha="__match__"):
    """A rolling release whose recorded hash matches `body` unless `sha` says otherwise."""
    if sha == "__match__":
        sha = hashlib.sha256(body).hexdigest()
    release = Release(database="roll", version="v1",
                      url="https://example.invalid/current.csv",
                      sha256=sha, expect_header="header", rolling=True)
    return {release.name: release}, body


def test_a_cache_hit_never_touches_the_network_or_rehashes(tmp_path, monkeypatch):
    """The whole point of the cache: reading from it must not depend on upstream being up."""
    registry, body = _rolling_registry(tmp_path)
    cache = ReleaseCache(registry, root=tmp_path)
    calls = []

    def _fetch(url, known_hash=None, fname=None, dest_dir=None, progressbar=False):
        calls.append(url)
        target = Path(dest_dir) / fname
        target.write_bytes(body)
        return target

    monkeypatch.setattr(cache_module, "fetch", _fetch)
    cache.ensure_release("roll/v1")
    assert len(calls) == 1
    for _ in range(5):
        cache.ensure_release("roll/v1")
    assert len(calls) == 1, "a cache hit went back to the network"


def test_a_rolling_release_accepts_a_rebuilt_upstream_on_a_cold_fetch(tmp_path, monkeypatch, capsys):
    """The recorded hash is a record, not a contract: the pinned bytes no longer exist."""
    registry, body = _rolling_registry(tmp_path, sha="0" * 64)
    cache = ReleaseCache(registry, root=tmp_path)
    monkeypatch.setattr(cache_module, "fetch", _writer(body))

    path = cache.ensure_release("roll/v1")
    assert path.read_bytes() == body
    out = capsys.readouterr().out
    assert "rolling" in out and "rebuilt" in out, "the swap must be announced"
    record = cache.release_provenance("roll/v1")
    assert record["registry_sha256"] == "0" * 64
    assert record["sha256"] != "0" * 64


@pytest.mark.parametrize("rolling,expect_known_hash", [(False, True), (True, False)])
def test_only_a_versioned_release_hands_its_hash_to_the_downloader(
        tmp_path, monkeypatch, rolling, expect_known_hash):
    """Rolling is opt-in, and this is the whole of what it changes on a fetch.

    The downloader verifies `known_hash` (pooch's job, covered in test_download.py). Passing it
    makes a mismatch fatal; withholding it makes the fetch accept whatever upstream now serves.
    """
    body = b"header\n" + b"x" * 2048
    release = Release(database="db", version="v1", url="https://example.invalid/v1.csv",
                      sha256=hashlib.sha256(body).hexdigest(), expect_header="header",
                      rolling=rolling)
    cache = ReleaseCache({release.name: release}, root=tmp_path)
    seen = {}

    def _fetch(url, known_hash=None, fname=None, dest_dir=None, progressbar=False):
        seen["known_hash"] = known_hash
        target = Path(dest_dir) / fname
        target.write_bytes(body)
        return target

    monkeypatch.setattr(cache_module, "fetch", _fetch)
    cache.ensure_release("db/v1")
    assert (seen["known_hash"] is not None) is expect_known_hash


def test_check_for_update_reports_staleness_without_touching_the_cache(tmp_path, monkeypatch):
    registry, body = _rolling_registry(tmp_path)
    cache = ReleaseCache(registry, root=tmp_path)
    monkeypatch.setattr(cache_module, "fetch", _writer(body))
    cache.ensure_release("roll/v1")
    before = cache.release_provenance("roll/v1")

    assert cache.check_for_update("roll/v1").status == "current"

    rebuilt = b"header\nrow\n" + b"y" * 4096
    monkeypatch.setattr(cache_module, "fetch", _writer(rebuilt))
    result = cache.check_for_update("roll/v1")
    assert result.stale and result.status == "stale"
    assert result.upstream_bytes == len(rebuilt)
    assert result.cached_sha256 != result.upstream_sha256
    assert cache.release_provenance("roll/v1") == before, "the check mutated the cache"
    assert cache.cached_path("roll/v1").read_bytes() == body


def test_expire_removes_the_release_so_the_next_call_refetches(tmp_path, monkeypatch):
    registry, body = _rolling_registry(tmp_path)
    cache = ReleaseCache(registry, root=tmp_path)
    monkeypatch.setattr(cache_module, "fetch", _writer(body))
    cache.ensure_release("roll/v1")

    removed = cache.expire("roll/v1")
    assert removed is not None and not removed.exists()
    assert cache.cached_path("roll/v1") is None
    assert cache.expire("roll/v1") is None, "expiring twice is not an error"

    rebuilt = b"header\nrow\n" + b"z" * 4096
    monkeypatch.setattr(cache_module, "fetch", _writer(rebuilt))
    assert cache.ensure_release("roll/v1").read_bytes() == rebuilt


def test_the_cache_directory_can_be_copied_elsewhere(tmp_path, monkeypatch):
    """The sidecar records a relative path, so a cache is not welded to where it was filled."""
    registry, body = _rolling_registry(tmp_path)
    first = ReleaseCache(registry, root=tmp_path / "one")
    monkeypatch.setattr(cache_module, "fetch", _writer(body))
    first.ensure_release("roll/v1")

    shutil.copytree(tmp_path / "one", tmp_path / "two")
    second = ReleaseCache(registry, root=tmp_path / "two")

    def _explode(*a, **k):
        raise AssertionError("a copied cache went back to the network")

    monkeypatch.setattr(cache_module, "fetch", _explode)
    served = second.ensure_release("roll/v1")
    assert served.read_bytes() == body
    assert (tmp_path / "two") in served.parents
