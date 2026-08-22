"""Tests for the ``encoders/`` package.

Fast, dependency-free tests use hand-rolled fake ONNX sessions/tokenizers so
CI never needs network access or a downloaded model (unit
tests run in ms). A handful of integration tests are gated on the real default
models being present under ``data/models/`` (``vesta models`` downloads them);
they skip cleanly otherwise, matching the project's existing pattern for the
gitignored pinned eval archive.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vesta.encoders.manager import EncoderManager
from vesta.encoders.onnx_cross_encoder import OnnxCrossEncoder
from vesta.encoders.onnx_encoder import OnnxBiEncoder, _l2_normalize, _mean_pool
from vesta.encoders.registry import MODEL_SPECS, ModelSpec, resolve_spec

# ── Fakes: no real tokenizer/ONNX weights needed for mechanics tests ────────


class FakeEncoding:
    def __init__(self, ids: list[int], type_ids: list[int] | None = None) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = type_ids if type_ids is not None else [0] * len(ids)


class FakeTokenizer:
    """Whitespace tokenizer: word -> a small deterministic id. No real vocab —
    exercises the padding/truncation/offset *mechanics* the encoders rely on,
    not real subword tokenization (covered by the integration tests)."""

    def __init__(self) -> None:
        self._padding = False
        self._max_length: int | None = None
        self.seen_texts: list[Any] = []

    def enable_padding(self) -> None:
        self._padding = True

    def no_padding(self) -> None:
        self._padding = False

    def enable_truncation(self, max_length: int) -> None:
        self._max_length = max_length

    def no_truncation(self) -> None:
        self._max_length = None

    def encode_batch(self, texts: list[Any], add_special_tokens: bool = True) -> list[FakeEncoding]:
        self.seen_texts = list(texts)
        encs: list[FakeEncoding] = []
        for t in texts:
            if isinstance(t, tuple):
                a, b = t
                a_words, b_words = a.split(), b.split()
                words = a_words + b_words
                type_ids = [0] * len(a_words) + [1] * len(b_words)
            else:
                words = str(t).split()
                type_ids = [0] * len(words)
            ids = [(abs(hash(w)) % 997) + 1 for w in words]
            if self._max_length is not None:
                ids = ids[: self._max_length]
                type_ids = type_ids[: self._max_length]
            encs.append(FakeEncoding(ids, type_ids=type_ids))
        if self._padding and encs:
            maxlen = max(len(e.ids) for e in encs)
            for e in encs:
                pad = maxlen - len(e.ids)
                e.ids = e.ids + [0] * pad
                e.attention_mask = [1] * (len(e.ids) - pad) + [0] * pad
                e.type_ids = e.type_ids + [0] * pad
        return encs


class FakeSession:
    def __init__(self, output_names: list[str], run_fn: Any) -> None:
        self._outputs = [SimpleNamespace(name=n) for n in output_names]
        self._run_fn = run_fn
        self.last_feed: dict[str, Any] = {}

    def get_outputs(self) -> list[Any]:
        return self._outputs

    def run(self, output_names: list[str] | None, feed: dict[str, Any]) -> list[Any]:
        self.last_feed = feed
        return self._run_fn(feed)


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "repo_id": "test/model",
        "role": "embed",
        "kind": "bi_encoder_padded",
        "onnx_file": "model.onnx",
        "onnx_data_file": None,
        "dim": 4,
        "max_tokens": 16,
        "query_prefix": "",
        "passage_prefix": "",
        "normalize": True,
        "pooling": "mean",
        "license": "mit",
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


# ── Pure pooling/normalization math ──────────────────────────────────────────


def test_mean_pool_ignores_padded_positions() -> None:
    last_hidden = np.array([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)  # third token is padding
    pooled = _mean_pool(last_hidden, mask)
    assert pooled.shape == (1, 2)
    assert np.allclose(pooled[0], [2.0, 2.0])  # mean of first two rows only


def test_l2_normalize_produces_unit_vectors() -> None:
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalize(vecs)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.linalg.norm(out[1]) == 0.0  # zero vector stays zero, no div-by-zero


# ── OnnxBiEncoder mechanics ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bi_encoder_uses_sentence_embedding_output_when_present() -> None:
    spec = _spec(dim=3)
    fixed = np.array([[1.0, 2.0, 2.0]], dtype=np.float32)  # norm = 3

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        batch = feed["input_ids"].shape[0]
        return [np.zeros((batch, 1, 3), dtype=np.float32), np.repeat(fixed, batch, axis=0)]

    session = FakeSession(["last_hidden_state", "sentence_embedding"], run_fn)
    enc = OnnxBiEncoder(spec, session, FakeTokenizer(), semaphore=asyncio.Semaphore(2))
    out = await enc.embed(["hello world"], kind="passage")
    assert out.shape == (1, 3)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)  # normalize=True applied


@pytest.mark.asyncio
async def test_bi_encoder_falls_back_to_mean_pool_without_named_output() -> None:
    spec = _spec(dim=2, normalize=False)

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        mask = feed["attention_mask"]
        batch, seqlen = mask.shape
        # last_hidden[i, j] = j+1 in both dims, so mean-pool is checkable by hand.
        vals = np.arange(1, seqlen + 1, dtype=np.float32)
        last_hidden = np.tile(vals[None, :, None], (batch, 1, 2))
        return [last_hidden]

    session = FakeSession(["last_hidden_state"], run_fn)
    tok = FakeTokenizer()
    enc = OnnxBiEncoder(spec, session, tok, semaphore=asyncio.Semaphore(2))
    out = await enc.embed(["a b c"], kind="passage")
    assert out.shape == (1, 2)
    # 3 real tokens (no padding, single item): mean of [1,2,3] = 2.0
    assert np.allclose(out[0], [2.0, 2.0])


@pytest.mark.asyncio
async def test_bi_encoder_applies_role_prefix() -> None:
    spec = _spec(query_prefix="query: ", passage_prefix="passage: ")

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        batch = feed["input_ids"].shape[0]
        return [np.zeros((batch, 1, 4), dtype=np.float32), np.ones((batch, 4), dtype=np.float32)]

    session = FakeSession(["last_hidden_state", "sentence_embedding"], run_fn)
    tok = FakeTokenizer()
    enc = OnnxBiEncoder(spec, session, tok, semaphore=asyncio.Semaphore(2))
    await enc.embed(["capital of france"], kind="query")
    assert tok.seen_texts == ["query: capital of france"]
    await enc.embed(["Paris is a city."], kind="passage")
    assert tok.seen_texts == ["passage: Paris is a city."]


@pytest.mark.asyncio
async def test_bi_encoder_ragged_builds_offsets_from_item_lengths() -> None:
    spec = _spec(kind="bi_encoder_ragged", dim=2)
    captured: dict[str, Any] = {}

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        captured.update(feed)
        batch = feed["offsets"].shape[0]
        return [np.ones((batch, 2), dtype=np.float32)]

    session = FakeSession(["embeddings"], run_fn)
    enc = OnnxBiEncoder(spec, session, FakeTokenizer(), semaphore=asyncio.Semaphore(2))
    out = await enc.embed(["a b", "c d e"], kind="passage")
    assert out.shape == (2, 2)
    # first item is 2 tokens, second is 3 -> offsets [0, 2]
    assert list(captured["offsets"]) == [0, 2]
    assert captured["input_ids"].shape == (5,)


@pytest.mark.asyncio
async def test_bi_encoder_embed_empty_texts_returns_empty_array() -> None:
    spec = _spec(dim=5)
    enc = OnnxBiEncoder(
        spec, FakeSession([], lambda feed: []), FakeTokenizer(), semaphore=asyncio.Semaphore(1)
    )
    out = await enc.embed([], kind="query")
    assert out.shape == (0, 5)


@pytest.mark.asyncio
async def test_bi_encoder_sub_batches_large_padded_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch that would tile a huge attention mask (08's real Tile-overflow
    crash) must be split into several ``session.run`` calls rather than one
    unbounded call — regardless of caller batch size (index.batch_size only
    bounds articles, not the chunks flattened into one ``embed()`` call)."""
    import vesta.encoders.onnx_encoder as onnx_encoder_module

    monkeypatch.setattr(onnx_encoder_module, "_MAX_PADDED_ELEMENTS", 1)  # force sub_batch=1
    spec = _spec(dim=1, normalize=False)
    calls: list[int] = []

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        batch = feed["input_ids"].shape[0]
        assert batch == 1  # sub-batching actually kicked in
        calls.append(int(feed["input_ids"][0, 0]))
        return [np.zeros((1, 1, 1), dtype=np.float32), feed["input_ids"][:, :1].astype(np.float32)]

    session = FakeSession(["last_hidden_state", "sentence_embedding"], run_fn)
    enc = OnnxBiEncoder(spec, session, FakeTokenizer(), semaphore=asyncio.Semaphore(2))
    out = await enc.embed(["aaa", "bbb", "ccc"], kind="passage")
    assert len(calls) == 3  # three separate session.run calls, not one batch of 3
    # concatenation preserves input order across sub-batches
    assert out.shape == (3, 1)
    assert calls == [hash_id("aaa"), hash_id("bbb"), hash_id("ccc")]


def hash_id(word: str) -> int:
    return (abs(hash(word)) % 997) + 1


# ── Length bucketing (padded path) ───────────────────────────────────────────
#
# The padded path sorts texts by token length, pads each bucket to its OWN
# width, and scatters the results back to input positions. The scatter is the
# dangerous part: ``index/job.py::_materialize_batch`` pairs ``vecs[i]`` with
# ``pending_rows[i]`` positionally, so a permutation bug would attach every
# vector to the wrong chunk id and fail SILENTLY — search would return passages
# that don't match, with nothing raised. These tests make each output row
# self-identifying so a permutation shows up as a value mismatch.


class LengthTokenizer:
    """Tokenizer whose output identifies its input.

    Text ``"<tag>:<n>"`` encodes to ``n`` copies of token id ``tag``. So the
    first token of any output row names the text that produced it, and the row
    count names its length — exactly what an ordering assertion needs.
    """

    def __init__(self) -> None:
        self._max_length: int | None = None
        self.padding_enabled = False

    def enable_padding(self) -> None:
        self.padding_enabled = True

    def no_padding(self) -> None:
        self.padding_enabled = False

    def enable_truncation(self, max_length: int) -> None:
        self._max_length = max_length

    def no_truncation(self) -> None:
        self._max_length = None

    def encode_batch(self, texts: list[Any], add_special_tokens: bool = True) -> list[FakeEncoding]:
        encs: list[FakeEncoding] = []
        for text in texts:
            tag, count = str(text).split(":")
            ids = [int(tag)] * int(count)
            if self._max_length is not None:
                ids = ids[: self._max_length]
            encs.append(FakeEncoding(ids))
        return encs


def _identity_session() -> tuple[FakeSession, list[dict[str, Any]]]:
    """A session that echoes each row's first token id as its 1-d embedding."""
    seen: list[dict[str, Any]] = []

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        ids = feed["input_ids"]
        seen.append({"width": int(ids.shape[1]), "count": int(ids.shape[0])})
        return [
            np.zeros((ids.shape[0], 1, 1), dtype=np.float32),
            ids[:, :1].astype(np.float32),
        ]

    return FakeSession(["last_hidden_state", "sentence_embedding"], run_fn), seen


def _bucketing_encoder(session: FakeSession) -> OnnxBiEncoder:
    spec = _spec(dim=1, normalize=False, max_tokens=512)
    return OnnxBiEncoder(spec, session, LengthTokenizer(), semaphore=asyncio.Semaphore(2))


@pytest.mark.asyncio
async def test_length_bucketing_preserves_input_order() -> None:
    """Every output row must correspond to the input at the SAME index, even
    though bucketing reorders the work internally."""
    session, _ = _identity_session()
    enc = _bucketing_encoder(session)
    # Deliberately non-monotonic lengths so sorting genuinely permutes.
    lengths = [5, 400, 1, 90, 400, 2, 250, 3]
    texts = [f"{i + 1}:{n}" for i, n in enumerate(lengths)]

    out = await enc.embed(texts, kind="passage")

    assert out.shape == (len(texts), 1)
    # Row i must carry tag i+1 — i.e. the embedding of texts[i], not of
    # whichever text happened to sort into position i.
    assert [int(v) for v in out[:, 0]] == [i + 1 for i in range(len(texts))]


@pytest.mark.asyncio
async def test_indexer_shaped_call_splits_into_narrow_length_buckets() -> None:
    """At DEFAULT settings, an indexer-shaped call (hundreds of chunks, lengths
    from a title line to the 512-token ceiling) must split into passes that are
    each padded near their own contents — not one pass padded to 512."""
    import random

    rng = random.Random(7)
    lengths = [rng.choice([rng.randint(20, 120), rng.randint(120, 512)]) for _ in range(300)]
    texts = [f"{i + 1}:{n}" for i, n in enumerate(lengths)]
    session, seen = _identity_session()
    enc = _bucketing_encoder(session)

    out = await enc.embed(texts, kind="passage")

    assert [int(v) for v in out[:, 0]] == [i + 1 for i in range(300)]
    assert len(seen) >= 8  # actually split, not one 512-wide pass
    # The real payoff: total padded token-slots vs the single-batch behaviour
    # this replaced (every chunk padded to the longest in the call).
    bucketed = sum(p["width"] * p["count"] for p in seen)
    single_batch = max(lengths) * len(lengths)
    assert bucketed < single_batch * 0.6


@pytest.mark.asyncio
async def test_length_bucketing_pads_each_bucket_to_its_own_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the sort: short texts must NOT be padded to the longest
    text in the call. Two short + two long, two per bucket."""
    import vesta.encoders.onnx_encoder as onnx_encoder_module

    monkeypatch.setattr(onnx_encoder_module, "_MAX_BUCKET_ITEMS", 2)
    session, seen = _identity_session()
    enc = _bucketing_encoder(session)

    out = await enc.embed(["1:400", "2:3", "3:400", "4:3"], kind="passage")

    assert [int(v) for v in out[:, 0]] == [1, 2, 3, 4]
    # Short texts share a 3-wide pass; long texts share a 400-wide pass. Before
    # bucketing every one of these was padded to 400.
    assert sorted(s["width"] for s in seen) == [3, 400]
    assert all(s["count"] == 2 for s in seen)


@pytest.mark.asyncio
async def test_length_bucketing_masks_padding_positions() -> None:
    """Padded positions must carry attention_mask 0 so the pooled output is
    identical to what an unpadded single-item pass would produce."""
    captured: dict[str, Any] = {}

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        captured["mask"] = feed["attention_mask"].copy()
        captured["ids"] = feed["input_ids"].copy()
        batch = feed["input_ids"].shape[0]
        return [np.zeros((batch, 1, 1), dtype=np.float32), np.ones((batch, 1), dtype=np.float32)]

    session = FakeSession(["last_hidden_state", "sentence_embedding"], run_fn)
    enc = _bucketing_encoder(session)
    await enc.embed(["1:2", "2:5"], kind="passage")

    mask, ids = captured["mask"], captured["ids"]
    assert mask.shape == (2, 5)
    # Sorted ascending by length: the 2-token text is row 0, padded to 5.
    assert list(mask[0]) == [1, 1, 0, 0, 0]
    assert list(mask[1]) == [1, 1, 1, 1, 1]
    assert list(ids[0]) == [1, 1, 0, 0, 0]  # pad id fills the tail


@pytest.mark.asyncio
async def test_length_bucketing_order_survives_randomized_lengths() -> None:
    """Permutation bugs are order-sensitive — sweep deterministic corner cases."""
    import random

    rng = random.Random(20260805)
    for _ in range(20):
        n = rng.randint(1, 40)
        lengths = [rng.randint(1, 512) for _ in range(n)]
        texts = [f"{i + 1}:{ln}" for i, ln in enumerate(lengths)]
        session, _ = _identity_session()
        enc = _bucketing_encoder(session)
        out = await enc.embed(texts, kind="passage")
        assert [int(v) for v in out[:, 0]] == [i + 1 for i in range(n)], (
            f"order broken for lengths={lengths}"
        )


@pytest.mark.asyncio
async def test_length_bucketing_respects_element_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No forward pass may exceed the count * width**2 budget — that bound is
    what keeps ModernBERT's attention-mask Tile under ONNX's 4 GiB ceiling."""
    import vesta.encoders.onnx_encoder as onnx_encoder_module

    budget = 4096
    monkeypatch.setattr(onnx_encoder_module, "_MAX_PADDED_ELEMENTS", budget)
    session, seen = _identity_session()
    enc = _bucketing_encoder(session)

    texts = [f"{i + 1}:{(i % 60) + 1}" for i in range(60)]
    out = await enc.embed(texts, kind="passage")

    assert [int(v) for v in out[:, 0]] == [i + 1 for i in range(60)]
    for pass_ in seen:
        # A single item may exceed the budget on its own (irreducible); more
        # than one may not.
        assert pass_["count"] == 1 or pass_["count"] * pass_["width"] ** 2 <= budget


@pytest.mark.asyncio
async def test_single_oversized_item_runs_alone_rather_than_stalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk too large for the budget on its own is irreducible; it must
    still run (as a bucket of one), not loop forever or be dropped."""
    import vesta.encoders.onnx_encoder as onnx_encoder_module

    monkeypatch.setattr(onnx_encoder_module, "_MAX_PADDED_ELEMENTS", 1)
    session, seen = _identity_session()
    enc = _bucketing_encoder(session)

    out = await enc.embed(["1:500", "2:1"], kind="passage")

    assert [int(v) for v in out[:, 0]] == [1, 2]
    assert all(p["count"] == 1 for p in seen)


# ── OnnxCrossEncoder mechanics ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_encoder_applies_sigmoid_and_preserves_order() -> None:
    spec = _spec(role="rerank", kind="cross_encoder_pair", dim=0)

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        batch = feed["input_ids"].shape[0]
        # logits: first pair strongly relevant, second strongly irrelevant.
        logits = np.array([[8.0], [-8.0]][:batch], dtype=np.float32)
        return [logits]

    session = FakeSession(["logits"], run_fn)
    enc = OnnxCrossEncoder(
        spec, session, FakeTokenizer(), max_tokens=16, semaphore=asyncio.Semaphore(2)
    )
    scores = await enc.score("q", ["relevant passage", "irrelevant passage"])
    assert len(scores) == 2
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0
    assert scores[0] > 0.9
    assert scores[1] < 0.1


@pytest.mark.asyncio
async def test_cross_encoder_score_empty_passages_returns_empty() -> None:
    spec = _spec(role="rerank", kind="cross_encoder_pair", dim=0)
    enc = OnnxCrossEncoder(
        spec,
        FakeSession([], lambda feed: []),
        FakeTokenizer(),
        max_tokens=16,
        semaphore=asyncio.Semaphore(1),
    )
    assert await enc.score("q", []) == []


@pytest.mark.asyncio
async def test_cross_encoder_truncates_to_configured_max_tokens() -> None:
    spec = _spec(role="rerank", kind="cross_encoder_pair", dim=0)
    captured: dict[str, Any] = {}

    def run_fn(feed: dict[str, Any]) -> list[Any]:
        captured.update(feed)
        batch = feed["input_ids"].shape[0]
        return [np.zeros((batch, 1), dtype=np.float32)]

    session = FakeSession(["logits"], run_fn)
    enc = OnnxCrossEncoder(
        spec, session, FakeTokenizer(), max_tokens=3, semaphore=asyncio.Semaphore(2)
    )
    long_passage = " ".join(f"word{i}" for i in range(50))
    await enc.score("q", [long_passage])
    assert captured["input_ids"].shape[1] <= 3


# ── registry.resolve_spec ────────────────────────────────────────────────────


def test_resolve_spec_returns_none_for_unknown_repo() -> None:
    assert resolve_spec("nonexistent/repo", "embed") is None


def test_resolve_spec_returns_none_on_role_mismatch() -> None:
    static_repo = next(k for k, v in MODEL_SPECS.items() if v.role == "static")
    assert resolve_spec(static_repo, "rerank") is None


def test_resolve_spec_returns_spec_on_match() -> None:
    embed_repo = next(k for k, v in MODEL_SPECS.items() if v.role == "embed")
    spec = resolve_spec(embed_repo, "embed")
    assert spec is not None
    assert spec.role == "embed"


# ── EncoderManager: readiness + lazy caching (no real model files needed) ───


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _manager(model_dir: Path) -> EncoderManager:
    return EncoderManager(
        model_dir=model_dir,
        static_model="minishlab/potion-retrieval-32M",
        embed_model="onnx-community/granite-embedding-small-english-r2-ONNX",
        rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
        intra_op_threads=1,
        spinning=False,
        cpu_mem_arena=False,
        pool_size=1,
        rerank_truncate_tokens=256,
    )


def test_manager_not_ready_when_files_absent(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    assert mgr.static_ready() is False
    assert mgr.embed_ready() is False
    assert mgr.rerank_ready() is False


def test_manager_ready_when_required_files_present(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    spec = MODEL_SPECS["minishlab/potion-retrieval-32M"]
    base = tmp_path / spec.repo_id
    _touch(base / spec.onnx_file)
    _touch(base / "tokenizer.json")
    assert mgr.static_ready() is True
    assert mgr.embed_ready() is False


def test_manager_ready_requires_onnx_data_sidecar_when_declared(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    spec = MODEL_SPECS["onnx-community/granite-embedding-small-english-r2-ONNX"]
    base = tmp_path / spec.repo_id
    _touch(base / spec.onnx_file)
    _touch(base / "tokenizer.json")
    assert mgr.embed_ready() is False  # .onnx_data sidecar still missing
    assert spec.onnx_data_file is not None
    _touch(base / spec.onnx_data_file)
    assert mgr.embed_ready() is True


@pytest.mark.asyncio
async def test_manager_get_returns_none_when_not_ready(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    assert await mgr.get_static() is None
    assert await mgr.get_embed() is None
    assert await mgr.get_rerank() is None


@pytest.mark.asyncio
async def test_manager_caches_load_result_including_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = _manager(tmp_path)
    calls = {"n": 0}

    def fake_load(role: str) -> object | None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(mgr, "_load_sync", fake_load)
    r1 = await mgr.get_static()
    r2 = await mgr.get_static()
    assert r1 is None
    assert r2 is None
    assert calls["n"] == 1  # second call hit the cache, not _load_sync again


@pytest.mark.asyncio
async def test_manager_concurrent_get_loads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = _manager(tmp_path)
    calls = {"n": 0}

    def fake_load(role: str) -> object | None:
        calls["n"] += 1
        return object()

    monkeypatch.setattr(mgr, "_load_sync", fake_load)
    results = await asyncio.gather(mgr.get_static(), mgr.get_static(), mgr.get_static())
    assert calls["n"] == 1
    assert results[0] is results[1] is results[2]


# ── build_session: CPU memory arena toggle (RSS-bounding fix) ─────────────────


def test_build_session_sets_enable_cpu_mem_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_session`` must thread ``cpu_mem_arena`` onto the ONNX
    ``SessionOptions`` so the arena can be disabled to bound RSS (the default-
    on arena caches peak tensor allocations for the session lifetime and
    fragments under varying input shapes)."""
    import onnxruntime as ort

    from vesta.encoders.runtime import build_session

    captured: dict[str, Any] = {}

    class _FakeInferenceSession:
        def __init__(self, path: str, sess_options: Any, providers: list[str]) -> None:
            captured["options"] = sess_options

    monkeypatch.setattr(ort, "InferenceSession", _FakeInferenceSession)

    build_session(tmp_path / "model.onnx", intra_op_threads=1, spinning=False, cpu_mem_arena=False)
    assert captured["options"].enable_cpu_mem_arena is False

    build_session(tmp_path / "model.onnx", intra_op_threads=1, spinning=False, cpu_mem_arena=True)
    assert captured["options"].enable_cpu_mem_arena is True


@pytest.mark.asyncio
async def test_manager_threads_cpu_mem_arena_into_build_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manager must pass its ``cpu_mem_arena`` constructor arg through to
    every ``build_session`` call (both load paths), so the setting actually
    reaches the ONNX session."""
    mgr = EncoderManager(
        model_dir=tmp_path,
        static_model="minishlab/potion-retrieval-32M",
        embed_model="onnx-community/granite-embedding-small-english-r2-ONNX",
        rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
        intra_op_threads=1,
        spinning=False,
        cpu_mem_arena=False,
        pool_size=1,
        rerank_truncate_tokens=256,
    )
    spec = MODEL_SPECS["minishlab/potion-retrieval-32M"]
    base = tmp_path / spec.repo_id
    _touch(base / spec.onnx_file)
    _touch(base / "tokenizer.json")

    captured: dict[str, Any] = {}

    class _FakeSession:
        def get_outputs(self) -> list[Any]:
            return []

    def fake_build_session(
        path: Path, *, intra_op_threads: int, spinning: bool, cpu_mem_arena: bool
    ) -> Any:
        captured["cpu_mem_arena"] = cpu_mem_arena
        captured["spinning"] = spinning
        return _FakeSession()

    monkeypatch.setattr("vesta.encoders.manager.build_session", fake_build_session)
    monkeypatch.setattr("vesta.encoders.manager.load_tokenizer", lambda p: object())

    assert await mgr.get_static() is not None
    assert captured["cpu_mem_arena"] is False


# ── Capability probe wiring ───────────────────────────────────────────────────


def test_capability_probe_reflects_manager_readiness(tmp_path: Path) -> None:
    from vesta.config.capabilities import Capability
    from vesta.encoders import _capability_probe, bind_manager

    try:
        mgr = _manager(tmp_path)
        spec = MODEL_SPECS["minishlab/potion-retrieval-32M"]
        base = tmp_path / spec.repo_id
        _touch(base / spec.onnx_file)
        _touch(base / "tokenizer.json")
        bind_manager(mgr)
        caps = _capability_probe()
        assert Capability.STATIC_ENCODER in caps
        assert Capability.CROSS_ENCODER not in caps
    finally:
        bind_manager(None)


def test_capability_probe_empty_when_no_manager_bound() -> None:
    from vesta.encoders import _capability_probe, bind_manager

    bind_manager(None)
    assert _capability_probe() == frozenset()


# ── Integration: real downloaded models (skips if `vesta models` wasn't run) ─

_REAL_MODEL_DIR = Path("data/models")


def _real_manager() -> EncoderManager:
    return EncoderManager(
        model_dir=_REAL_MODEL_DIR,
        static_model="minishlab/potion-retrieval-32M",
        embed_model="onnx-community/granite-embedding-small-english-r2-ONNX",
        rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
        intra_op_threads=2,
        spinning=False,
        cpu_mem_arena=False,
        pool_size=2,
        rerank_truncate_tokens=256,
    )


_real_models_present = _real_manager().static_ready() and _real_manager().rerank_ready()


@pytest.mark.skipif(
    not _real_models_present, reason="run `vesta models` to fetch real models first"
)
@pytest.mark.asyncio
async def test_real_static_encoder_ranks_relevant_passage_higher() -> None:
    mgr = _real_manager()
    enc = await mgr.get_static()
    assert enc is not None
    qv = await enc.embed(["capital of france"], kind="query")
    pv = await enc.embed(
        ["Paris is the capital of France.", "Bananas are a good source of potassium."],
        kind="passage",
    )
    sims = pv @ qv[0]
    assert sims[0] > sims[1]


@pytest.mark.skipif(
    not _real_models_present, reason="run `vesta models` to fetch real models first"
)
@pytest.mark.asyncio
async def test_real_cross_encoder_ranks_relevant_passage_higher() -> None:
    mgr = _real_manager()
    enc = await mgr.get_rerank()
    assert enc is not None
    scores = await enc.score(
        "capital of france",
        ["Paris is the capital of France.", "Bananas are a good source of potassium."],
    )
    assert scores[0] > scores[1]
