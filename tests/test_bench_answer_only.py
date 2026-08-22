"""Answer-only mode + context-snapshot round-trip tests.

Offline: the LLM gateway is a stub and the archive registry is never touched —
``answer_only`` in snapshot mode reads passages straight from the snapshot file,
and in oracle mode goes through a stubbed ``build_oracle_context``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vesta.api.bench import (
    AnswerOnlySystem,
    RetrievalOnlySystem,
    load_context_snapshot,
    make_system,
    snapshot_missing_ids,
    write_context_snapshot,
)
from vesta.eval.bench_dataset import BenchDataset, BenchQuestion, BenchSource


def _q(qid: str, *, behavior: str = "answer") -> BenchQuestion:
    srcs = (
        (BenchSource(zim="z", article_title="T", article_path="Article", required=True),)
        if behavior == "answer"
        else ()
    )
    return BenchQuestion(
        id=qid,
        question=f"{qid}?",
        capability="buried_fact",
        difficulty="medium",
        slice="core",
        expected_behavior=behavior,
        answer="42",
        sources=srcs,
    )


def _ds() -> BenchDataset:
    return BenchDataset(name="test", version=1, questions=(_q("q1"), _q("q2")))


class _FakeGateway:
    """Stub gateway: returns a canned answer and records the prompt it saw."""

    def __init__(self, answer: str = "the answer") -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.calls = 0

    async def chat_once(self, messages: list[Any], **kwargs: Any) -> Any:
        self.calls += 1
        self.prompts.append(str(messages[0].content))
        return _FakeRes(self.answer)


class _FakeRes:
    def __init__(self, text: str) -> None:
        self.text = text
        self.input_tokens = 10
        self.output_tokens = 5


class _FakeState:
    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.db = None
        self.registry = None


def _snapshot_file(tmp_path: Path, *, qid: str = "q1") -> Path:
    p = tmp_path / "snap.json"
    p.write_text(
        json.dumps(
            {
                "format": "vesta-bench-context-snapshot/1",
                "generated": "2026-08-15T00:00:00+00:00",
                "system": "retrieval_only",
                "profile": "hybrid",
                "level": 1,
                "dataset": {"name": "test", "hash": "h", "subset_hash": "s"},
                "questions": {
                    qid: {
                        "paths": ["Article"],
                        "passages": [
                            {
                                "path": "Article",
                                "title": "Article",
                                "breadcrumb": "Lead",
                                "score": 0.9,
                                "text": "The answer is 42.",
                            }
                        ],
                    }
                },
            }
        )
    )
    return p


def _multi_snapshot_file(tmp_path: Path, k: int = 3) -> Path:
    """A snapshot with k rank-ordered passages (path/text distinguishable)."""
    p = tmp_path / "multi-snap.json"
    passages = [
        {
            "path": f"Art{i}",
            "title": f"Article {i}",
            "breadcrumb": f"Lead {i}",
            "score": 1.0 - i * 0.1,
            "text": f"PASSAGE-{i} holds the fact.",
        }
        for i in range(1, k + 1)
    ]
    p.write_text(
        json.dumps(
            {
                "format": "vesta-bench-context-snapshot/1",
                "generated": "2026-08-16T00:00:00+00:00",
                "system": "retrieval_only",
                "profile": "hybrid",
                "level": 1,
                "dataset": {"name": "test", "hash": "h", "subset_hash": "s"},
                "questions": {
                    "q1": {
                        "paths": [f"Art{i}" for i in range(1, k + 1)],
                        "passages": passages,
                    }
                },
            }
        )
    )
    return p


# ── Snapshot I/O ─────────────────────────────────────────────────────────────


def test_snapshot_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "out" / "snap.json"
    write_context_snapshot(
        p,
        questions={"q1": {"paths": ["A"], "passages": []}},
        dataset=_ds(),
        subset_hash_val="s",
        system="retrieval_only",
        profile="hybrid",
        level=1,
    )
    data = load_context_snapshot(p)
    assert data["format"] == "vesta-bench-context-snapshot/1"
    assert data["questions"]["q1"]["paths"] == ["A"]
    assert data["dataset"]["subset_hash"] == "s"


def test_load_rejects_non_snapshot(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError, match="not a context snapshot"):
        load_context_snapshot(p)


def test_snapshot_missing_ids() -> None:
    snap = {"questions": {"q1": {}, "q2": {}}}
    assert snapshot_missing_ids(snap, ["q1", "q2", "q3"]) == ("q3",)
    assert snapshot_missing_ids(snap, ["q1"]) == ()


# ── AnswerOnlySystem ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_answer_only_snapshot_replay(tmp_path: Path) -> None:
    gw = _FakeGateway()
    state = _FakeState(gw)
    sut = AnswerOnlySystem(state, model_id="m", context_path=str(_snapshot_file(tmp_path)))
    out = await sut.run_one(_q("q1"))
    assert out.answer_text == "the answer"
    assert out.retrieved_paths == ("Article",)  # frozen snapshot order
    assert out.abstained is False
    assert out.input_tokens == 10 and out.output_tokens == 5
    assert "The answer is 42." in gw.prompts[0]  # snapshot passage fed verbatim
    assert "Article" in gw.prompts[0]


@pytest.mark.asyncio
async def test_answer_only_detects_abstention(tmp_path: Path) -> None:
    from vesta.answer.abstention import ABSTENTION_NO_MATCH

    gw = _FakeGateway(answer=ABSTENTION_NO_MATCH)
    state = _FakeState(gw)
    sut = AnswerOnlySystem(state, model_id="m", context_path=str(_snapshot_file(tmp_path)))
    out = await sut.run_one(_q("q1"))
    assert out.abstained is True


@pytest.mark.asyncio
async def test_answer_only_oracle_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _FakeGateway()
    state = _FakeState(gw)
    sut = AnswerOnlySystem(state, model_id="m", oracle_context=True)

    async def _fake_oracle(state: Any, q: Any, find: Any) -> tuple[str, tuple[str, ...]]:
        return "=== Gold ===\ngold text", ("GoldPath",)

    monkeypatch.setattr("vesta.api.bench.build_oracle_context", _fake_oracle)
    out = await sut.run_one(_q("q1"))
    assert out.answer_text == "the answer"
    assert out.retrieved_paths == ("GoldPath",)
    assert "gold text" in gw.prompts[0]


# ── --context-passages (pre-seed sensitivity) ───────────────────────────────


@pytest.mark.asyncio
async def test_answer_only_context_passages_truncates_rank_order(tmp_path: Path) -> None:
    """N=2 feeds only the top-2 passages (rank order); the tail is dropped."""
    gw = _FakeGateway()
    state = _FakeState(gw)
    sut = AnswerOnlySystem(
        state, model_id="m", context_path=str(_multi_snapshot_file(tmp_path)), context_passages=2
    )
    out = await sut.run_one(_q("q1"))
    prompt = gw.prompts[0]
    assert "PASSAGE-1" in prompt and "PASSAGE-2" in prompt
    assert "PASSAGE-3" not in prompt
    # Retrieval stays the frozen axis: paths replay the FULL snapshot card
    # list so source metrics do not move with N.
    assert out.retrieved_paths == ("Art1", "Art2", "Art3")
    assert out.trace["context_passages"] == 2
    assert out.trace["passages_used"] == 2


@pytest.mark.asyncio
async def test_answer_only_without_knob_feeds_every_passage(tmp_path: Path) -> None:
    """Knob absent → today's behaviour: every snapshot passage, no trace key."""
    gw = _FakeGateway()
    state = _FakeState(gw)
    sut = AnswerOnlySystem(state, model_id="m", context_path=str(_multi_snapshot_file(tmp_path)))
    out = await sut.run_one(_q("q1"))
    prompt = gw.prompts[0]
    for i in (1, 2, 3):
        assert f"PASSAGE-{i}" in prompt
    assert "context_passages" not in out.trace
    assert out.trace["passages_used"] == 3


def test_answer_only_context_passages_validation(tmp_path: Path) -> None:
    """N<1 and oracle mode fail loudly at construction, not mid-run."""
    state = _FakeState(_FakeGateway())
    snap = str(_multi_snapshot_file(tmp_path))
    with pytest.raises(ValueError, match="must be >= 1"):
        AnswerOnlySystem(state, model_id="m", context_path=snap, context_passages=0)
    with pytest.raises(ValueError, match="snapshot replay only"):
        AnswerOnlySystem(state, model_id="m", oracle_context=True, context_passages=2)


def test_make_system_routes_context_passages(tmp_path: Path) -> None:
    """The CLI seam forwards the knob; the public pin is readable for config_json."""
    state = _FakeState(_FakeGateway())
    sut = make_system(
        "answer_only",
        state,
        model_id="m",
        context_path=str(_multi_snapshot_file(tmp_path)),
        context_passages=2,
    )
    assert isinstance(sut, AnswerOnlySystem)
    assert sut.context_passages == 2
    default = make_system(
        "answer_only", state, model_id="m", context_path=str(_snapshot_file(tmp_path))
    )
    assert default.context_passages is None


@pytest.mark.asyncio
async def test_run_benchmark_records_context_passages_in_config(tmp_path: Path) -> None:
    """`config_json.context_passages` appears iff the knob was set — the
    persisted pin that makes per-N replay runs comparable."""
    from vesta.api.bench import InMemoryBenchStore
    from vesta.eval.bench_runner import run_benchmark

    async def scenario(context_passages: int | None) -> dict[str, object]:
        sut = AnswerOnlySystem(
            _FakeState(_FakeGateway()),
            model_id="m",
            context_path=str(_multi_snapshot_file(tmp_path)),
            context_passages=context_passages,
        )
        ds = _ds()
        records = await run_benchmark(
            dataset=ds,
            questions=[_q("q1")],
            systems=[sut],
            store=InMemoryBenchStore(),
            judge=None,
            judge_model="",
        )
        return records[0].config_json

    with_knob = await scenario(2)
    assert with_knob["context_passages"] == 2
    without = await scenario(None)
    assert "context_passages" not in without


def test_answer_only_requires_exactly_one_context_source(tmp_path: Path) -> None:
    state = _FakeState(_FakeGateway())
    with pytest.raises(ValueError, match="exactly one context source"):
        AnswerOnlySystem(state, model_id="m")  # neither
    with pytest.raises(ValueError, match="exactly one context source"):
        AnswerOnlySystem(
            state, model_id="m", context_path=str(_snapshot_file(tmp_path)), oracle_context=True
        )


def test_answer_only_loads_snapshot_eagerly(tmp_path: Path) -> None:
    state = _FakeState(_FakeGateway())
    with pytest.raises(ValueError, match="could not read context snapshot"):
        AnswerOnlySystem(state, model_id="m", context_path=str(tmp_path / "nope.json"))


# ── make_system routing ─────────────────────────────────────────────────────


def test_make_system_routes_answer_only(tmp_path: Path) -> None:
    state = _FakeState(_FakeGateway())
    sut = make_system(
        "answer_only", state, model_id="m", context_path=str(_snapshot_file(tmp_path))
    )
    assert isinstance(sut, AnswerOnlySystem)
    assert sut.name == "answer_only"


def test_make_system_retrieval_only_collect(tmp_path: Path) -> None:
    state = _FakeState(_FakeGateway())
    sut = make_system("retrieval_only", state, collect_context=True)
    assert isinstance(sut, RetrievalOnlySystem)
    assert sut.generates_answers is False
    assert sut.context_snapshot() == {}
