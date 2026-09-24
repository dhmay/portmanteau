"""Download files from the internet into a local cache, via pooch.

Files are cached, so repeated calls return the local copy without touching the
network. Pass ``known_hash`` (e.g. ``"sha256:..."``) to verify contents; with
``known_hash=None``, pooch logs the downloaded file's hash so you can pin it.

The cache lives in the OS cache dir (e.g. ``~/Library/Caches/portmanteau``),
overridable with the ``PORTMANTEAU_DATA_DIR`` environment variable or a
per-call ``dest_dir``.
"""

import logging
import os
import time
from os import PathLike
from pathlib import Path

import pooch
import requests

logger = logging.getLogger(__name__)

StrPath = str | PathLike[str]

CACHE_ENV_VAR = "PORTMANTEAU_DATA_DIR"
USER_AGENT = "portmanteau"


def cache_dir() -> Path:
    """Default download directory: $PORTMANTEAU_DATA_DIR, else the OS cache dir."""
    return Path(os.environ.get(CACHE_ENV_VAR) or pooch.os_cache("portmanteau"))


def fetch(
    url: str,
    *,
    known_hash: str | None = None,
    fname: str | None = None,
    dest_dir: StrPath | None = None,
    timeout_seconds: float = 60,
    max_retries: int = 3,
    backoff_seconds: float = 5,
    progressbar: bool = False,
) -> Path:
    """Download a file (or reuse the cached copy) and return its local path.

    Args:
        url: URL of the file.
        known_hash: Expected hash, e.g. "sha256:abc...". If given, a mismatch raises
            ValueError. If None, the file isn't verified and pooch logs its hash.
        fname: Local filename. Defaults to the URL's basename prefixed with a hash of
            the URL, so different URLs with the same basename don't collide.
        dest_dir: Directory to download into. Defaults to cache_dir().
        timeout_seconds: Per-request timeout.
        max_retries: Attempts before giving up on transient errors (timeouts,
            connection errors, 5xx responses). Other errors, like 404, raise immediately.
        backoff_seconds: Wait before the first retry, doubling after each failure.
        progressbar: Show a progress bar (requires tqdm).

    Returns:
        Path to the downloaded file.
    """
    return Path(
        _retrieve(
            url,
            known_hash=known_hash,
            fname=fname,
            dest_dir=dest_dir,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            progressbar=progressbar,
        )
    )


def fetch_zip(
    url: str,
    *,
    members: list[str] | None = None,
    known_hash: str | None = None,
    fname: str | None = None,
    dest_dir: StrPath | None = None,
    timeout_seconds: float = 60,
    max_retries: int = 3,
    backoff_seconds: float = 5,
    progressbar: bool = False,
) -> list[Path]:
    """Download a zip archive (or reuse the cached copy) and extract it.

    Extracted files go in "<archive name>.unzip/" next to the archive, and are
    also reused on later calls.

    Args:
        url: URL of the zip archive.
        members: Archive members to extract. Defaults to all.
        (remaining args): As for fetch(); known_hash applies to the archive itself.

    Returns:
        Paths to the extracted files.

    Raises:
        zipfile.BadZipFile: If the download isn't a valid zip archive.
    """
    paths = _retrieve(
        url,
        known_hash=known_hash,
        fname=fname,
        dest_dir=dest_dir,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        progressbar=progressbar,
        processor=pooch.Unzip(members=members),
    )
    return [Path(p) for p in paths]


def _retrieve(
    url: str,
    *,
    known_hash: str | None,
    fname: str | None,
    dest_dir: StrPath | None,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    progressbar: bool,
    processor: pooch.processors.ExtractorProcessor | None = None,
):
    """pooch.retrieve with retries on transient errors.

    Retrying is safe because pooch downloads to a temp file and only moves it into
    place once complete.
    """
    downloader = pooch.HTTPDownloader(
        progressbar=progressbar,
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(1, max_retries + 1):
        try:
            return pooch.retrieve(
                url,
                known_hash=known_hash,
                fname=fname,
                path=dest_dir if dest_dir is not None else cache_dir(),
                processor=processor,
                downloader=downloader,
            )
        except requests.RequestException as e:
            if attempt == max_retries or not _is_transient(e):
                raise
            logger.warning(
                "Download attempt %d/%d for %s failed (%s); retrying in %.0fs...",
                attempt, max_retries, url, e, backoff_seconds,
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2


def _is_transient(e: requests.RequestException) -> bool:
    if isinstance(e, (requests.ConnectionError, requests.Timeout)):
        return True
    return e.response is not None and e.response.status_code >= 500
