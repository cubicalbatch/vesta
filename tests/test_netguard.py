"""The egress guard (AUDIT_0824 A1) — rejection classes, allow paths, and the
request-controlled fetch sites it protects.

Policy under test (``vesta/config/netguard.py``):

* URLs from REQUEST parameters (manual download URL, refresh-catalog feed
  override) or from fetched REMOTE CONTENT (metalink mirrors) must be public
  internet http(s) — scheme checked AND every resolved address globally
  routable — with redirects re-validated hop by hop.
* Owner-configured endpoints (``catalog.opds_url`` setting, inference LLM) are
  deliberately unrestricted: pointing the appliance at a LAN host is the
  product.

No network is touched: rejection targets are numeric literals (resolved
locally by ``getaddrinfo``, never dialed because the guard fires first), and
allow paths run against ``httpx.MockTransport`` / a closed loopback port.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from vesta import config
from vesta.catalog import bind_runtime
from vesta.catalog.download import DownloadError, DownloadZimJob
from vesta.catalog.opds import fetch_opds_feed
from vesta.config import netguard
from vesta.config.netguard import EgressBlocked, assert_public_http_url, guarded_request
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.inference.download import DownloadModelError, DownloadModelJob
from vesta.jobs.handle import JobHandleImpl

# ── the guard itself ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/1",
        "ftp://example.com/x.zim",
        "javascript:alert(1)",
    ],
)
def test_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(EgressBlocked):
        assert_public_http_url(url)


def test_rejects_url_without_host() -> None:
    with pytest.raises(EgressBlocked):
        assert_public_http_url("http://")


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "[::1]",
        "169.254.169.254",  # link-local: the classic cloud-metadata target
        "10.1.2.3",
        "192.168.0.20",
        "172.16.5.5",
        "0.0.0.0",
    ],
)
def test_rejects_private_and_local_targets(host: str) -> None:
    with pytest.raises(EgressBlocked):
        assert_public_http_url(f"http://{host}/x")


def test_rejects_when_any_resolved_address_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name resolving to one private address (rebinding-style mix) is blocked."""

    def mixed(_host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return [ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(netguard, "_resolve_host", mixed)
    with pytest.raises(EgressBlocked):
        assert_public_http_url("http://mixed.example/x")


def test_rejects_unresolvable_host(monkeypatch: pytest.MonkeyPatch) -> None:

    def fail(_host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        raise OSError("no dns")

    monkeypatch.setattr(netguard, "_resolve_host", fail)
    with pytest.raises(EgressBlocked, match="cannot resolve"):
        assert_public_http_url("http://nope.invalid/x")


@pytest.mark.parametrize("url", ["https://8.8.8.8/x.zim", "http://93.184.216.34/f"])
def test_allows_public_internet_targets(url: str) -> None:
    assert_public_http_url(url)  # does not raise


# ── redirects cannot escape ─────────────────────────────────────────────────


def _redirect_transport(targets: list[str]) -> tuple[httpx.MockTransport, list[int]]:
    """A transport serving 302 → next target, finally 200 ``done``."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        idx = len(calls)
        calls.append(idx)
        if idx < len(targets):
            return httpx.Response(302, headers={"Location": targets[idx]})
        return httpx.Response(200, text="done")

    return httpx.MockTransport(handler), calls


async def test_guarded_request_blocks_redirect_to_internal() -> None:
    transport, calls = _redirect_transport(["http://169.254.169.254/latest/meta-data/"])
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(EgressBlocked):
            await guarded_request(client, "GET", "http://93.184.216.34/feed")
    assert len(calls) == 1  # the redirect hop was validated, never dialed


async def test_guarded_request_follows_public_redirect() -> None:
    transport, calls = _redirect_transport(["http://93.184.216.34/final"])
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        resp = await guarded_request(client, "GET", "http://93.184.216.34/feed")
    assert resp.status_code == 200
    assert resp.text == "done"
    assert len(calls) == 2  # both hops dialed


# ── OPDS fetch site (refresh_catalog job) ──────────────────────────────────


async def test_fetch_opds_feed_guards_only_when_asked() -> None:
    """egress_guard=True blocks a loopback feed; without it (the
    settings-resolved path) the same URL is fetched — the owner's LAN catalog
    is legitimate."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, text="<feed/>"))
    ) as client:
        with pytest.raises(EgressBlocked):
            await fetch_opds_feed("http://127.0.0.1:1/opds", egress_guard=True)
        # Unguarded (settings-resolved): the loopback fetch is attempted.
        assert await fetch_opds_feed("http://127.0.0.1:1/opds", client=client) == "<feed/>"
        # Guarded allow-path: a public IP passes and the body comes back.
        assert (
            await fetch_opds_feed("http://93.184.216.34/opds", client=client, egress_guard=True)
            == "<feed/>"
        )


# ── download_zim job site ──────────────────────────────────────────────────


async def test_zim_download_job_rejects_metadata_target(tmp_path: Path) -> None:
    """A request-supplied acquisition URL pointing at link-local metadata space
    fails the job without any bytes written."""
    bind_runtime(db=None, zims_dir=str(tmp_path), register_archive=_noop_register)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        with pytest.raises(DownloadError, match="mirror"):
            await DownloadZimJob().run(
                handle,
                {"url": "http://169.254.169.254/latest/x.zim.meta4", "name": "x"},
            )
        assert list(tmp_path.iterdir()) == []
    finally:
        config.reset_for_test()


# ── download_model job site ────────────────────────────────────────────────


async def test_model_download_job_rejects_loopback_target() -> None:
    """A raw GGUF URL aimed at the appliance's own loopback (where the
    llama-server router lives) is refused before anything is bound or dialed."""
    handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
    with pytest.raises(DownloadModelError, match="blocked"):
        await DownloadModelJob().run(
            handle, {"url": "http://127.0.0.1:9999/v1/m.gguf", "filename": "m.gguf"}
        )


# ── refresh_catalog job site ───────────────────────────────────────────────


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await d.start()
    async with d.write() as conn:
        await run_migrations(conn)
    yield d
    await d.stop()


async def test_refresh_catalog_override_is_egress_guarded(db: Database) -> None:
    """The job guards ONLY the request-supplied ``url`` override; the
    settings-resolved feed (here pointed at the appliance's own loopback, as an
    owner LAN catalog legitimately could be) is attempted unguarded."""
    bind_runtime(db=db, zims_dir=None, register_archive=None)

    class Runner:
        async def _publish_progress(self, *_a: object) -> None:
            pass

        async def _write_progress(self, *_a: object, **_k: object) -> None:
            pass

        async def _write_checkpoint(self, *_a: object) -> None:
            pass

        def _is_cancelling(self, _job_id: int) -> bool:
            return False

    handle = JobHandleImpl(Runner(), job_id=1)  # type: ignore[arg-type]

    from vesta.catalog.refresh import RefreshCatalogJob

    # Request-controlled override → blocked before any dial.
    with pytest.raises(RuntimeError, match="EgressBlocked"):
        await RefreshCatalogJob().run(handle, {"url": "file:///etc/passwd"})

    # Settings-owner override (env → catalog.opds_url) → attempted, not blocked:
    # connecting to a closed loopback port fails as ConnectError, proving the
    # guard did not fire.
    os.environ["catalog.opds_url"] = "http://127.0.0.1:1/opds"
    config.configure()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await RefreshCatalogJob().run(handle, {})
        assert "EgressBlocked" not in str(exc_info.value)
    finally:
        os.environ.pop("catalog.opds_url", None)
        config.reset_for_test()
        bind_runtime(db=None, zims_dir=None, register_archive=None)


# ── shared fakes ───────────────────────────────────────────────────────────


class _FakeRunner:
    async def _publish_progress(self, _job_id: int, _d: int, _t: int, _m: str) -> None:
        pass

    async def _write_progress(self, _job_id: int, _d: int, _t: int, _m: str) -> None:
        pass

    async def _write_checkpoint(self, _job_id: int, _blob: object) -> None:
        pass

    def _is_cancelling(self, _job_id: int) -> bool:
        return False


async def _noop_register(path: Any) -> None:
    pass
