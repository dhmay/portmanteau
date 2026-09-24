import hashlib
import http.server
import io
import threading
import zipfile

import pytest

pytest.importorskip("pooch")

from portmanteau.data import download  # noqa: E402

HELLO = b"hello\n"


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "A")
        z.writestr("sub/b.txt", "B")
    return buf.getvalue()


@pytest.fixture(scope="module")
def server():
    """Local HTTP server. Yields (base_url, hits per path, User-Agent of each request)."""
    files = {"/hello.txt": HELLO, "/data.zip": _zip_bytes(), "/notzip.zip": b"nope"}
    hits: dict[str, int] = {}
    user_agents: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits[self.path] = hits.get(self.path, 0) + 1
            user_agents.append(self.headers["User-Agent"])
            if self.path == "/boom":
                self.send_response(500)
                self.end_headers()
                return
            body = files.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", hits, user_agents
    srv.shutdown()


@pytest.fixture
def dest(tmp_path):
    # resolve() so paths compare equal on macOS, where /var is a symlink to /private/var
    return tmp_path.resolve()


def test_fetch_returns_path_and_caches(server, dest):
    base, hits, user_agents = server
    path = download.fetch(f"{base}/hello.txt", dest_dir=dest)
    assert path.parent == dest
    assert path.name.endswith("-hello.txt")
    assert path.read_bytes() == HELLO

    n_hits = hits["/hello.txt"]
    assert download.fetch(f"{base}/hello.txt", dest_dir=dest) == path
    assert hits["/hello.txt"] == n_hits
    assert user_agents[-1] == download.USER_AGENT


def test_fetch_accepts_str_dest_and_fname(server, dest):
    base, _, _ = server
    path = download.fetch(f"{base}/hello.txt", dest_dir=str(dest), fname="greeting.txt")
    assert path == dest / "greeting.txt"


def test_known_hash(server, dest):
    base, _, _ = server
    good = "sha256:" + hashlib.sha256(HELLO).hexdigest()
    assert download.fetch(f"{base}/hello.txt", dest_dir=dest, known_hash=good).exists()

    bad_dir = dest / "bad"
    with pytest.raises(ValueError):
        download.fetch(f"{base}/hello.txt", dest_dir=bad_dir, known_hash="sha256:" + "0" * 64)
    assert list(bad_dir.glob("*")) == []


def test_fetch_zip(server, dest):
    base, _, _ = server
    paths = download.fetch_zip(f"{base}/data.zip", dest_dir=dest)
    root = paths[0].parent if paths[0].name == "a.txt" else paths[0].parent.parent
    assert sorted(p.relative_to(root).as_posix() for p in paths) == ["a.txt", "sub/b.txt"]
    assert sorted(p.read_text() for p in paths) == ["A", "B"]


def test_fetch_zip_members(server, dest):
    base, _, _ = server
    paths = download.fetch_zip(f"{base}/data.zip", dest_dir=dest, members=["a.txt"])
    assert [p.name for p in paths] == ["a.txt"]


def test_fetch_zip_not_a_zip(server, dest):
    base, _, _ = server
    with pytest.raises(zipfile.BadZipFile):
        download.fetch_zip(f"{base}/notzip.zip", dest_dir=dest)


def test_404_is_not_retried(server, dest):
    base, hits, _ = server
    with pytest.raises(Exception):
        download.fetch(f"{base}/missing", dest_dir=dest, backoff_seconds=0)
    assert hits["/missing"] == 1


def test_500_is_retried(server, dest):
    base, hits, _ = server
    with pytest.raises(Exception):
        download.fetch(f"{base}/boom", dest_dir=dest, max_retries=3, backoff_seconds=0)
    assert hits["/boom"] == 3


def test_cache_dir_env_override(monkeypatch, dest):
    monkeypatch.setenv(download.CACHE_ENV_VAR, str(dest))
    assert download.cache_dir() == dest
    monkeypatch.delenv(download.CACHE_ENV_VAR)
    assert download.cache_dir().name == "portmanteau"
