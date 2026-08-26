"""Egress guard for request-controlled outbound HTTP fetches (AUDIT_0824 A1).

Policy
------
Vesta is a trusted-LAN, no-auth appliance: endpoints the OWNER configures —
``catalog.opds_url``, ``inference.llm.*``, ``eval.judge.*``, the supervised
local llama-server router — are fetched without restriction. Pointing the
appliance at a machine on your own LAN is the product; a guard there would
only break legitimate operation.

But URLs that arrive from a REQUEST (a manual download URL on
``POST /api/zims/download``, a ``refresh_catalog`` feed override) or from
fetched REMOTE CONTENT (metalink mirror lists) are untrusted input to a
server-side fetch. Those must be public-internet http(s) only: the scheme must
be http/https AND every address the host resolves to globally routable — no
loopback, link-local (169.254.x, cloud-metadata), private, reserved, or
multicast targets. Resolution-based checking also defuses decimal/hex/octal
IP-literal obfuscation, since ``getaddrinfo`` normalizes them.

Redirects cannot escape the check: create the client with :func:`safe_client`
(redirects off) and fetch through :func:`guarded_request` /
:func:`guarded_stream`, which re-validate EVERY redirect hop before following
it.

Accepted residual risk (documented, boring): DNS is resolved at validation
time and again at connect time, so a rebinding resolver could theoretically
slip a private address past the check. Single-user trusted-LAN posture accepts
this; pinning connections to validated addresses would mean owning the
transport, which is not worth the complexity here.

Every internal package may import ``config``, so this module adds no new
package dependency for its callers (``catalog/``, ``inference/``).
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

import httpx

#: Maximum number of redirect hops a guarded fetch may follow.
_MAX_HOPS = 5


class EgressBlocked(RuntimeError):
    """A URL was refused by the egress guard (bad scheme or non-global host)."""


def _resolve_host(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``host`` to unique IP addresses. Module-level so tests can
    monkeypatch it (the default touches the real resolver)."""
    addrs: dict[ipaddress.IPv4Address | ipaddress.IPv6Address, None] = {}
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
        if family in (socket.AF_INET, socket.AF_INET6):
            addrs[ipaddress.ip_address(sockaddr[0])] = None
    return list(addrs)


def assert_public_http_url(url: str) -> None:
    """Refuse any URL that is not public-internet http(s).

    Raises :class:`EgressBlocked` unless the scheme is http/https and EVERY
    address ``url``'s host resolves to is globally routable. Apply this ONLY
    to request-controlled / remote-content URLs; owner-configured endpoints
    are unrestricted by policy (see module docstring).
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise EgressBlocked(f"blocked non-http(s) egress target: {url!r}")
    host = parts.hostname  # lowercased, IPv6 brackets stripped
    if not host:
        raise EgressBlocked(f"blocked URL with no host: {url!r}")
    try:
        addrs = _resolve_host(host)
    except OSError as exc:
        raise EgressBlocked(f"cannot resolve host {host!r}: {exc}") from exc
    if not addrs:
        raise EgressBlocked(f"host {host!r} resolved to no addresses")
    bad = next((a for a in addrs if not a.is_global), None)
    if bad is not None:
        raise EgressBlocked(f"blocked private/non-global egress target {url!r} (resolves to {bad})")


def safe_client(**kwargs: Any) -> httpx.AsyncClient:
    """AsyncClient factory for guarded fetches: redirects OFF.

    :func:`guarded_request`/:func:`guarded_stream` follow redirects manually,
    validating each hop; httpx's own auto-follow would bypass that.
    """
    kwargs["follow_redirects"] = False
    return httpx.AsyncClient(**kwargs)


def _next_hop(current: str, resp: httpx.Response) -> str | None:
    """The next redirect target, or ``None`` when ``resp`` is final."""
    if not resp.is_redirect:
        return None
    return str(urllib.parse.urljoin(current, resp.headers.get("location", "")))


async def guarded_request(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """Perform ``method url`` following up to ``_MAX_HOPS`` redirects, every
    hop validated by :func:`assert_public_http_url`. The client MUST have been
    created with redirects off (:func:`safe_client`)."""
    kwargs.pop("follow_redirects", None)  # never let auto-follow back in
    current = url
    for hop in range(_MAX_HOPS + 1):
        assert_public_http_url(current)
        resp = await client.request(method, current, **kwargs)
        nxt = _next_hop(current, resp)
        if nxt is None:
            return resp
        if hop == _MAX_HOPS:
            break
        current = nxt
    raise EgressBlocked(f"too many redirects fetching {url!r}")


@contextlib.asynccontextmanager
async def guarded_stream(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> AsyncIterator[httpx.Response]:
    """Stream ``method url`` like :func:`guarded_request`, yielding the final
    response inside an open stream context. Redirect hops are closed unread
    before following; every hop is validated."""
    kwargs.pop("follow_redirects", None)
    current = url
    for hop in range(_MAX_HOPS + 1):
        assert_public_http_url(current)
        async with client.stream(method, current, **kwargs) as resp:
            nxt = _next_hop(current, resp)
            if nxt is None:
                yield resp
                return
        if hop == _MAX_HOPS:
            break
        current = nxt
    raise EgressBlocked(f"too many redirects streaming {url!r}")


__all__ = [
    "EgressBlocked",
    "assert_public_http_url",
    "guarded_request",
    "guarded_stream",
    "safe_client",
]
