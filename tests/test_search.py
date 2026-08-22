"""Integration test for ``GET /api/search``.

Runs the real retrieval pipeline over the tiny-ZIM fixture end to end and
asserts the fast path returns source cards, a complete trace (every stage,
component, params, timing, counts), a profile content hash, and confidence
signals. No LLM is configured — this is the "search works with nothing
else" path.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_search_returns_cards_trace_and_profile_hash(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, _zim_id = app_client_with_zim

    resp = await client.get("/api/search", params={"q": "Einstein", "profile": "lexical"})
    assert resp.status_code == 200
    body = resp.json()

    # Cards: the fast path returns at least the title-suggest hit ("Albert
    # Einstein"); title_suggest needs no capability so it always runs.
    assert isinstance(body["cards"], list)
    assert len(body["cards"]) >= 1
    card = body["cards"][0]
    assert card["path"]
    assert card["source"]
    assert card["zim_id"] == _zim_id

    # Trace: versioned, with every pipeline stage recorded, plus the profile
    # name and content hash.
    trace = body["trace"]
    assert trace["version"] == 1
    stage_names = {s["name"] for s in trace["stages"]}
    assert {"preparer", "candidate_source", "fuser", "context_assembler"} <= stage_names
    for stage in trace["stages"]:
        # Timing is always recorded; duration_ms may be ~0 but not absent.
        assert "duration_ms" in stage
        assert "params" in stage
    assert trace["profile"] == "lexical"
    assert trace["profile_hash"]
    assert len(trace["profile_hash"]) == 64

    # Confidence signals are present (recorded, not acted on, this phase).
    assert "confidence" in body
    for key in ("top_score", "score_dropoff", "density", "agreement"):
        assert key in body["confidence"]

    # Response echoes the resolved profile + hash.
    assert body["profile"] == "lexical"
    assert body["profile_hash"] == trace["profile_hash"]


@pytest.mark.asyncio
async def test_search_profile_override_switches_result_set(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """Switching ?profile= changes results with no code change
    and no restart. ``standard`` and ``lexical`` resolve to different profiles
    with different content hashes."""
    client, _zim_id = app_client_with_zim

    lexical = await client.get("/api/search", params={"q": "Einstein", "profile": "lexical"})
    other = await client.get("/api/search", params={"q": "Einstein", "profile": "standard"})
    assert lexical.status_code == 200
    assert other.status_code == 200
    assert lexical.json()["profile_hash"] != other.json()["profile_hash"]
    assert lexical.json()["profile"] == "lexical"
    assert other.json()["profile"] == "standard"


@pytest.mark.asyncio
async def test_search_no_candidates_returns_empty_not_500(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """When the pipeline finds nothing it surfaces an empty result with the
    trace preserved (NoCandidatesError), not a 500."""
    client, _zim_id = app_client_with_zim
    resp = await client.get("/api/search", params={"q": "zzznothingzzz", "profile": "lexical"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cards"] == []
    # Trace is still present and well-formed even on the empty path.
    assert body["trace"]["version"] == 1
    assert body["profile_hash"]
