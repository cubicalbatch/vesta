"""CLI tests for the unified benchmark umbrella.

Offline: the heavy pieces (LLM gateway, archive registry, the real
``run_benchmark`` orchestration) are monkeypatched with fakes; the DB is a real
(temp-file) migrated database so persistence paths (``bench_runs`` /
``bench_question_results``) are exercised for real.

Covers: ``vesta bench run --limit 2 --no-persist`` bootstraps, ``bench list``
lists, ``bench compare`` buckets, and the ``vesta eval`` deprecation pointer.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from vesta import cli, config
from vesta.answer import ANSWER_AGENT_ECONOMY, ANSWER_AGENT_PRESEED_ORDER
from vesta.api.bench import InMemoryBenchStore, SqliteBenchStore
from vesta.api.state import AppState
from vesta.cli import CLIPipelineRunner, _open_runtime
from vesta.config.capabilities import Capability, compute_capabilities
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.db.settings_store import upsert_setting
from vesta.encoders import ENCODERS_EMBED_MODEL
from vesta.eval.bench_dataset import (
    BenchDataset,
    BenchQuestion,
    BenchSource,
    dataset_hash,
)
from vesta.eval.bench_runner import BenchRunRecord, QuestionOutput, run_benchmark
from vesta.eval.golden import EVAL_ARCHIVE_PATH
from vesta.retrieval.profiles import load_profile
from vesta.vectors import get_store


class StubSUT:
    """A SystemUnderTest that reports fixed token counts per answer."""

    name = "stub"
    answer_model = "model-a"
    profile_name = ""
    profile_hash = ""

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        return QuestionOutput(
            answer_text="42",
            retrieved_paths=("A",),
            abstained=False,
            error=None,
            trace={},
            resolved_strategy="stub",
            rounds=1,
            input_tokens=11,
            output_tokens=7,
        )


def _q(qid: str, *, capability: str = "lookup", answer: str = "42") -> BenchQuestion:
    return BenchQuestion(
        id=qid,
        question=f"Question {qid}?",
        capability=capability,
        difficulty="easy",
        slice="core",
        expected_behavior="answer",
        answer=answer,
        sources=(
            BenchSource(
                zim="wikipedia_en_top_nopic_2026-06.zim",
                article_title="A",
                article_path="A",
            ),
        ),
    )


def _dataset() -> BenchDataset:
    qs = tuple(_q(f"q{i}") for i in range(3))
    return BenchDataset(name="vesta_test", version=1, questions=qs, hash=dataset_hash(qs))


def _run_record(rid: int, *, strict: float = 0.5) -> BenchRunRecord:
    return BenchRunRecord(
        run_group="grp",
        label="test",
        started_at="2026-08-08T00:00:00+00:00",
        status="complete",
        dataset_name="vesta_test",
        dataset_hash="deadbeef",
        subset_hash="beef",
        system="sources_only",
        profile_name="",
        profile_hash="",
        answer_model="model-a",
        judge_model="judge-b",
        metrics_json={"strict_accuracy": strict, "weighted_accuracy": strict},
        id=rid,
        finished_at="2026-08-08T00:00:01+00:00",
    )


@pytest.fixture
async def cli_db(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "cli.db"), busy_timeout_ms=1000)
    await database.start()
    async with database.write() as conn:
        await run_migrations(conn)
    yield database
    await database.stop()


def _fake_state(db: Database) -> AppState:
    return AppState(db=db, runner=None, registry=None, gateway=None, supervisor=None)


@pytest.mark.asyncio
async def test_eval_deprecation_pointer(capsys: pytest.CaptureFixture[str]) -> None:
    """`vesta eval` prints the one-line pointer to `vesta bench retrieval`."""
    from vesta.cli import main

    code = main(["eval"])
    out = capsys.readouterr().out
    assert code == 0
    assert "bench retrieval" in out
    assert "deprecated" in out


@pytest.mark.asyncio
async def test_bench_list_empty(cli_db: Database, capsys: pytest.CaptureFixture[str]) -> None:
    """`vesta bench list` on an empty DB reports no runs."""
    from vesta import cli

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        yield _fake_state(cli_db)

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(cli, "_open_runtime", _fake_open)
    try:
        from vesta.cli import _build_parser

        args = _build_parser().parse_args(["bench", "list"])
        code = await cli._cmd_bench_list(args)
        assert code == 0
        assert "no bench runs yet" in capsys.readouterr().out
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_bench_run_no_persist_bootstraps(
    cli_db: Database, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`vesta bench run --limit 2 --no-persist` runs the matrix path and reports."""
    from vesta import cli
    from vesta import config as app_config

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        app_config.configure()
        yield _fake_state(cli_db)

    async def _fake_run_benchmark(**_kwargs):
        return [_run_record(1, strict=0.5), _run_record(2, strict=1.0)]

    def _fake_make_judge(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)
    monkeypatch.setattr("vesta.eval.bench_runner.run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr("vesta.api.bench.make_judge_llm", _fake_make_judge)

    args = cli._build_parser().parse_args(
        ["bench", "run", "--limit", "2", "--no-persist", "--model", "stub-model"]
    )
    code = await cli._cmd_bench_run(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "run 1" in out
    assert "run 2" in out
    # Default system set resolves; report is written (md default).
    assert "wrote " in out or "run 1" in out


@pytest.mark.asyncio
async def test_bench_run_save_context_requires_retrieval_only(
    cli_db: Database, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`vesta bench run --save-context ...` without `--system retrieval_only` fails with exit code 1."""
    from vesta import cli
    from vesta import config as app_config

    @asynccontextmanager
    async def _fake_open(*_args: Any, **_kwargs: Any) -> Any:
        app_config.configure()
        yield _fake_state(cli_db)

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)

    # Case 1: default systems (not retrieval_only)
    args = cli._build_parser().parse_args(
        [
            "bench",
            "run",
            "--limit",
            "2",
            "--save-context",
            "snap.json",
            "--no-persist",
            "--model",
            "stub-model",
        ]
    )
    code = await cli._cmd_bench_run(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "--save-context needs --system retrieval_only" in out

    # Case 2: explicit non-retrieval_only system
    args_sources = cli._build_parser().parse_args(
        [
            "bench",
            "run",
            "--limit",
            "2",
            "--save-context",
            "snap.json",
            "--system",
            "sources_only",
            "--no-persist",
            "--model",
            "stub-model",
        ]
    )
    code_sources = await cli._cmd_bench_run(args_sources)
    out_sources = capsys.readouterr().out
    assert code_sources == 1
    assert "--save-context needs --system retrieval_only" in out_sources


@pytest.mark.asyncio
async def test_bench_compare_buckets(
    cli_db: Database, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`vesta bench compare A B` prints the four buckets incl. BROKEN."""
    from vesta import cli
    from vesta.api.bench import SqliteBenchStore

    store = SqliteBenchStore(cli_db)
    rid_a = await store.insert_run(_run_record(1, strict=0.5))
    rid_b = await store.insert_run(_run_record(2, strict=0.5))
    # A: q1 correct, q2 wrong. B: q1 wrong (broken), q2 correct (fixed).
    from vesta.eval.bench_runner import BenchQuestionResult
    from vesta.eval.bench_scoring import Verdict

    qs = (_q("q1"), _q("q2"))
    # A: q1 correct, q2 wrong. B: q1 wrong (broken), q2 correct (fixed).
    verdicts_a = {"q1": Verdict.CORRECT.value, "q2": Verdict.INCORRECT.value}
    verdicts_b = {"q1": Verdict.INCORRECT.value, "q2": Verdict.CORRECT.value}
    for run_id, verdicts in ((rid_a, verdicts_a), (rid_b, verdicts_b)):
        for q in qs:
            await store.insert_question_result(
                run_id,
                BenchQuestionResult(
                    run_id=run_id,
                    question_id=q.id,
                    capability=q.capability,
                    difficulty=q.difficulty,
                    question_text=q.question,
                    expected_answer=q.answer,
                    answer_text="x",
                    abstained=False,
                    verdict=verdicts[q.id],
                ),
            )

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        yield _fake_state(cli_db)

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)

    args = cli._build_parser().parse_args(["bench", "compare", str(rid_a), str(rid_b)])
    code = await cli._cmd_bench_compare_cli(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "BROKEN" in out
    assert "q1" in out
    assert "fixed" in out


@pytest.mark.asyncio
async def test_bench_compare_refuses_dataset_mismatch(
    cli_db: Database, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`vesta bench compare` refuses runs over different datasets (exit 1)."""
    from vesta import cli
    from vesta.api.bench import SqliteBenchStore

    store = SqliteBenchStore(cli_db)
    rid_a = await store.insert_run(_run_record(1))
    rec_b = replace(_run_record(2), dataset_hash="other")
    rid_b = await store.insert_run(rec_b)

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        yield _fake_state(cli_db)

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)

    args = cli._build_parser().parse_args(["bench", "compare", str(rid_a), str(rid_b)])
    code = await cli._cmd_bench_compare_cli(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "dataset mismatch" in out


@pytest.mark.asyncio
async def test_open_runtime_logs_registry_start_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AUDIT_0824 M12: a registry-start failure is logged, not swallowed —
    a silent dead registry is what let poisoned all-zero eval rows persist."""

    async def _boom(self: Any) -> Any:
        raise RuntimeError("zim dir exploded")

    monkeypatch.setattr(cli.ArchiveRegistry, "start", _boom)
    try:
        async with _open_runtime(str(tmp_path)):
            pass
    finally:
        monkeypatch.undo()
    assert "zim.scan_failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_open_runtime_logs_gateway_build_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AUDIT_0824 M12: a gateway that fails to build is logged (mirrors
    main.py's lifespan) instead of silently yielding gateway=None."""
    import vesta.inference as inference_mod

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no llama-server binary")

    monkeypatch.setattr(inference_mod, "build_gateway_from_settings", _boom)
    try:
        async with _open_runtime(str(tmp_path), with_gateway=True) as state:
            assert state.gateway is None
    finally:
        monkeypatch.undo()
    assert "inference.gateway_failed" in capsys.readouterr().out


def test_bench_umbrella_surface(capsys: pytest.CaptureFixture[str]) -> None:
    """`vesta bench` exposes the 8 documented subcommands; `run` its flags.

    Pure argparse construction — no DB, no network. Guards the documented CLI
    surface against accidental removal/renames.
    """
    from vesta import cli

    parser = cli._build_parser()

    def _help_text(*argv: str) -> str:
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, "--help"])
        return capsys.readouterr().out

    bench_help = _help_text("bench")
    for name in ("run", "retrieval", "hardware", "rejudge", "compare", "verify", "list", "show"):
        assert name in bench_help, f"missing bench subcommand {name}"

    run_help = _help_text("bench", "run")
    for flag in (
        "--system",
        "--profile",
        "--model",
        "--endpoint",
        "--api-key",
        "--judge-model",
        "--judge-endpoint",
        "--judge-api-key",
        "--dataset",
        "--slice",
        "--capability",
        "--difficulty",
        "--limit",
        "--repeats",
        "--concurrency",
        "--judge-concurrency",
        "--economy",
        "--scope",
        "--label",
        "--no-persist",
        "--report",
        "--baseline",
        "--data-dir",
    ):
        assert flag in run_help, f"missing run flag {flag}"


def test_bench_report_renders_peak_context_section(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The markdown report renders the measured peak-context
    distribution (p50/p90/p95/max, >8k/>16k shares, requests, overflow fb)
    and degrades to n/a for runs whose traces predate the meter."""
    from dataclasses import replace

    from vesta.cli import _write_bench_run_report

    rec = replace(
        _run_record(7),
        metrics_json={
            "tokens": {
                "answer": {
                    "total_input": 1000,
                    "total_output": 100,
                    "total": 1100,
                    "p50": 1100,
                    "p50_input": 1000,
                    "p50_output": 100,
                },
                "peak_context": {
                    "n": 2,
                    "p50": 4_044,
                    "p90": 20_748,
                    "p95": 20_748,
                    "max": 20_748,
                    "over_8192": 1,
                    "over_16384": 1,
                    "requests_total": 5,
                    "overflow_fallbacks": 0,
                },
            }
        },
    )
    monkeypatch.chdir(tmp_path)
    _write_bench_run_report("md", [rec], _dataset(), "peaktest")
    md = next((tmp_path / "benchmarks" / "results").glob("*-peaktest.md")).read_text()
    assert "## Peak context (largest single request per question)" in md
    assert "| 7 | sources_only | 2 | 4,044 | 20,748 | 20,748 | 20,748 |" in md
    assert "1 (50.0%) | 1 (50.0%) | 5 | 0 |" in md


def test_bench_report_peak_context_n_a_for_pre_meter_runs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A run whose metrics carry no peak_context renders the n/a row, not a
    crash or fabricated zeros."""
    from vesta.cli import _write_bench_run_report

    monkeypatch.chdir(tmp_path)
    _write_bench_run_report("md", [_run_record(8)], _dataset(), "premeter")
    md = next((tmp_path / "benchmarks" / "results").glob("*-premeter.md")).read_text()
    assert "## Peak context (largest single request per question)" in md
    assert "| 8 | sources_only | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in md


@pytest.mark.asyncio
async def test_cli_composition_root_wires_vectors_and_seeds_capability(tmp_path: Path) -> None:
    """Composition root wires vectors and seeds Capability.VECTORS from DB state."""
    db_path = tmp_path / "vesta.db"
    seed_db = Database(str(db_path), busy_timeout_ms=2000)
    await seed_db.start()
    async with seed_db.write() as conn:
        await run_migrations(conn)
        await conn.execute(
            "INSERT INTO zims(id, path, index_depth, index_status, enabled) "
            "VALUES (1, '/fake.zim', 1, 'complete', 1)"
        )
    await seed_db.stop()

    async with _open_runtime(str(tmp_path)) as state:
        store = get_store()
        assert store is not None, "CLIPipelineRunner's Deps.vectors depends on a bound store"
        assert Capability.VECTORS in compute_capabilities(), (
            "an archive with index_depth>=1/index_status='complete' existed before "
            "_open_runtime ran, so its capability-seeding query should have found it"
        )

        runner = CLIPipelineRunner(state)
        profile = load_profile("hybrid")
        assert profile is not None
        _paths, trace = await runner.run(profile, "one")
        degradations = trace.get("degradations", [])
        assert isinstance(degradations, list)
        assert not any(
            isinstance(d, dict) and d.get("missing") == "vectors" for d in degradations
        ), (
            f"vector_knn was capability-dropped even with a bound store + seeded "
            f"index state: {degradations!r}"
        )


@pytest.mark.parametrize(
    ("args", "attr", "expected"),
    [
        (["bench", "run", "--economy", "on"], "economy", "on"),
        (["bench", "run", "--economy", "off"], "economy", "off"),
        (["bench", "run", "--economy", "auto"], "economy", "auto"),
        (["bench", "run"], "economy", None),
        (
            [
                "bench",
                "run",
                "--set",
                "answer.agent.preseed_order=idf",
                "--set",
                "answer.agent.preseed_show_archive_id=false",
            ],
            "set",
            ["answer.agent.preseed_order=idf", "answer.agent.preseed_show_archive_id=false"],
        ),
        (["bench", "run"], "set", None),
    ],
)
def test_bench_flags_parsing(args: list[str], attr: str, expected: object) -> None:
    parser = cli._build_parser()
    parsed = parser.parse_args(args)
    assert getattr(parsed, attr) == expected


@pytest.mark.parametrize(
    ("args", "error_match"),
    [
        (["bench", "run", "--economy", "turbo"], "invalid choice"),
        (["bench", "run", "--set", "no.such.key=1"], "unknown settings key 'no.such.key'"),
        (
            ["bench", "run", "--set", "answer.agent.preseed_order=sideways"],
            "answer.agent.preseed_order",
        ),
        (["bench", "run", "--set", "answer.agent.economy"], "KEY=VALUE"),
    ],
)
def test_bench_flags_invalid_rejected(args: list[str], error_match: str) -> None:
    parser = cli._build_parser()
    if "--economy" in args:
        with pytest.raises(SystemExit):
            parser.parse_args(args)
    else:
        parsed = parser.parse_args(args)
        with pytest.raises(SystemExit) as exc:
            cli._build_apply_overrides(parsed)
        assert error_match in str(exc.value)


def test_build_apply_overrides_mapping() -> None:
    """`--economy` and `--set` map onto setting overrides; later repeats win."""
    # economy flag
    assert cli._build_apply_overrides(argparse.Namespace(economy="on")) == {
        "answer.agent.economy": "on"
    }
    assert "answer.agent.economy" not in cli._build_apply_overrides(
        argparse.Namespace(economy=None)
    )
    assert "answer.agent.economy" not in cli._build_apply_overrides(argparse.Namespace())

    # set flag
    mapped = cli._build_apply_overrides(
        argparse.Namespace(
            set=[
                "answer.agent.preseed_order=idf",
                "answer.agent.preseed_show_archive_id=false",
            ]
        )
    )
    assert mapped == {
        "answer.agent.preseed_order": "idf",
        "answer.agent.preseed_show_archive_id": "false",
    }

    # Later repeat wins
    twice = cli._build_apply_overrides(
        argparse.Namespace(
            set=["answer.agent.preseed_order=idf", "answer.agent.preseed_order=rank"]
        )
    )
    assert twice == {"answer.agent.preseed_order": "rank"}
    assert cli._build_apply_overrides(argparse.Namespace(set=None)) == {}


def test_settings_overrides_reach_snapshot() -> None:
    """The override merged by `_open_runtime` (set_db_values) becomes the snapshot's effective value."""
    config.configure()
    try:
        for forced in ("on", "off", "auto"):
            config.set_db_values({ANSWER_AGENT_ECONOMY.key: forced})
            snap = config.snapshot()
            assert snap.get(ANSWER_AGENT_ECONOMY) == forced
            assert snap.values[ANSWER_AGENT_ECONOMY.key] == forced

        config.set_db_values({ANSWER_AGENT_PRESEED_ORDER.key: "rank"})
        snap = config.snapshot()
        assert snap.get(ANSWER_AGENT_PRESEED_ORDER) == "rank"
    finally:
        config.reset_for_test()


@pytest.mark.asyncio
async def test_run_benchmark_records_economy_settings_set_and_tokens(cli_db: Database) -> None:
    """A forced run records economy and settings_set in config_json and per-question tokens."""
    store = SqliteBenchStore(cli_db)
    dataset = _dataset()

    records = await run_benchmark(
        dataset=dataset,
        questions=list(dataset.questions),
        systems=[StubSUT()],
        store=store,
        judge=None,
        judge_model="",
        config_snapshot={"answer.agent.economy": "on"},
        economy="on",
        settings_set={
            "answer.agent.preseed_order": "idf",
            "answer.agent.preseed_show_archive_id": "false",
        },
    )
    rec = records[0]
    snapshot_json = rec.config_json["settings_snapshot"]
    assert isinstance(snapshot_json, dict)
    assert snapshot_json["answer.agent.economy"] == "on"
    assert rec.config_json["settings_set"] == {
        "answer.agent.preseed_order": "idf",
        "answer.agent.preseed_show_archive_id": "false",
    }

    # Round-trips through store
    persisted = await store.get_run(rec.id)
    assert persisted is not None
    assert persisted.config_json["economy"] == "on"
    assert persisted.config_json["settings_set"] == {
        "answer.agent.preseed_order": "idf",
        "answer.agent.preseed_show_archive_id": "false",
    }

    # Token accounting flows into bench_question_results
    qrows = await store.list_question_results(rec.id)
    assert len(qrows) == len(dataset.questions)
    assert all(r.input_tokens == 11 and r.output_tokens == 7 for r in qrows)


@pytest.mark.asyncio
async def test_run_benchmark_defaults_leave_forced_keys_untouched() -> None:
    """Default (no forcing) does not invent economy or settings_set keys in config_json."""
    records = await run_benchmark(
        dataset=_dataset(),
        questions=list(_dataset().questions),
        systems=[StubSUT()],
        store=InMemoryBenchStore(),
        judge=None,
        judge_model="",
    )
    snapshot_json = records[0].config_json["settings_snapshot"]
    assert isinstance(snapshot_json, dict)
    assert "economy" not in snapshot_json
    assert "settings_set" not in records[0].config_json


# ── `vesta bench verify` bounded-concurrency passes (AUDIT_0822 P4) ─────────


def _out(text: str) -> QuestionOutput:
    return QuestionOutput(
        answer_text=text,
        retrieved_paths=(),
        abstained=False,
        error=None,
        trace={},
        resolved_strategy="stub",
    )


def _vq(
    qid: str,
    *,
    capability: str = "lookup",
    answer: str = "alpha beta",
    paths: tuple[str, ...] = ("a",),
) -> BenchQuestion:
    """A verify fixture question; article texts live in _VERIFY_EXTRACTS."""
    return BenchQuestion(
        id=qid,
        question=f"Question {qid}?",
        capability=capability,
        difficulty="easy",
        slice="core",
        expected_behavior="answer",
        answer=answer,
        sources=tuple(
            BenchSource(zim="z.zim", article_title=p.upper(), article_path=p) for p in paths
        ),
    )


# (qid, article_path) → extracted text. q2/q3's first required source MISMATCHES
# the answer tokens, so the serial short-circuit never extracts their second
# source — concurrency must preserve that exactly.
_VERIFY_EXTRACTS: dict[tuple[str, str], str] = {
    ("q0", "a"): "alpha story",  # token overlap → support PASS
    ("q1", "b"): "epsilon zeta",  # no overlap → support FAIL
    ("q2", "c1"): "nothing here",  # mismatch → short-circuit before "c2"
    ("q2", "c2"): "omega here",
    ("q3", "d1"): "nope",  # mismatch → short-circuit before "d2"
    ("q3", "d2"): "omega here",
}


def _verify_qs() -> tuple[BenchQuestion, ...]:
    return (
        _vq("q0", answer="alpha beta"),
        _vq("q1", answer="gamma delta", paths=("b",)),
        # fact capability → counts toward the floor (non-lookup denominator).
        _vq("q2", capability="fact", answer="omega psi", paths=("c1", "c2")),
        _vq("q3", capability="fact", answer="omega psi", paths=("d1", "d2")),
    )


_CB_ANSWERS = {"q0": "yes cb0", "q1": "cb1", "q2": "yes cb2", "q3": "cb3"}
_OR_ANSWERS = {f"q{i}": f"yes or-q{i}" for i in range(4)}


class _ScriptedSUT:
    """closed_book/oracle stand-in answering from a fixed table."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        return _out(self._answers[q.id])


async def _stub_judge_verdict(
    *,
    question: BenchQuestion,
    model_answer: str,
    abstained: bool,
    judge: object,
    judge_model: str,
) -> Any:
    from vesta.eval.bench_scoring import Verdict

    return SimpleNamespace(
        verdict=Verdict.CORRECT if model_answer.startswith("yes") else Verdict.INCORRECT,
        reason="stub-rule",
    )


def _serial_reference_stdout(qs: Sequence[BenchQuestion]) -> str:
    """The pre-P4 serial algorithm's printed lines, computed independently."""
    support: dict[str, bool] = {}
    for q in qs:
        at = {w for w in cli._tokenize(q.answer) if len(w) >= 4}
        ok = True
        for src in q.sources:
            if not src.required:
                continue
            text = _VERIFY_EXTRACTS[(q.id, src.article_path)]
            if at and not (at & {w for w in cli._tokenize(text) if len(w) >= 4}):
                ok = False
                break
        support[q.id] = ok

    def _correct(answer: str) -> bool:
        return answer.startswith("yes")

    active = [q for q in qs if q.status == "active"]
    ceiling = sum(1 for q in active if _correct(_OR_ANSWERS[q.id])) / len(active)
    non_lookup = [q for q in active if q.capability != "lookup"]
    floor = sum(1 for q in non_lookup if _correct(_CB_ANSWERS[q.id])) / len(non_lookup)
    return (
        f"ceiling (oracle) ≥ 85%? {ceiling * 100:.1f}% "
        f"{'PASS' if ceiling >= 0.85 else 'FAIL'}\n"
        f"floor (closed-book, excl lookup) ≤ 20%? {floor * 100:.1f}% "
        f"{'PASS' if floor <= 0.20 else 'FAIL'}\n"
        "wrote benchmarks/verification review files (.md + .json).\n"
    )


async def _run_verify_once(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
) -> tuple[int, str, bytes, dict[str, Any]]:
    """One full `bench verify` run against the offline fixture; returns exit
    code + stdout + the review md bytes + the review json (minus timestamp)."""
    import json as _json

    from vesta.eval.bench_dataset import dataset_hash

    qs = _verify_qs()
    dataset = BenchDataset(name="vesta_test", version=1, questions=qs, hash=dataset_hash(qs))

    @asynccontextmanager
    async def _fake_open(*_args: object, **_kwargs: object):
        config.configure()
        yield _fake_state(Database(":memory:"))

    async def _noop_extract(state: object, q: Any, src: Any) -> str:
        await asyncio.sleep(0)
        return _VERIFY_EXTRACTS[(q.id, src.article_path)]

    def _make_system(system: str, *_args: object, **_kwargs: object) -> _ScriptedSUT:
        return _ScriptedSUT(_CB_ANSWERS if system == "closed_book" else _OR_ANSWERS)

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)
    monkeypatch.setattr("vesta.eval.bench_dataset.load_bench_dataset", lambda _path=None: dataset)
    monkeypatch.setattr("vesta.api.bench.make_judge_llm", lambda *_a, **_k: (None, None))
    monkeypatch.setattr("vesta.api.bench.make_system", _make_system)
    monkeypatch.setattr("vesta.eval.bench_scoring.judge_verdict", _stub_judge_verdict)
    monkeypatch.setattr(cli, "_extract_article", _noop_extract)

    run_dir = tmp_path / f"run-{'-'.join(extra_args) or 'defaults'}"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    args = cli._build_parser().parse_args(["bench", "verify", "--model", "stub-model", *extra_args])
    code = await cli._cmd_bench_verify(args)
    stdout = capsys.readouterr().out
    md = next(iter((run_dir / "benchmarks" / "verification").glob("*-review.md"))).read_bytes()
    payload = _json.loads(
        next(iter((run_dir / "benchmarks" / "verification").glob("*-review.json"))).read_text()
    )
    payload.pop("generated")  # wall-clock stamp — compared modulo it
    assert isinstance(payload, dict)
    return code, stdout, md, payload


@pytest.mark.asyncio
async def test_bench_verify_output_identical_serial_vs_concurrent(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Golden: default (answers=1/judges=4) and wide (--concurrency 4) runs both
    print exactly what the old fully-serial implementation would print, write
    byte-identical review md, and agree on every judged field."""
    expected_stdout = _serial_reference_stdout(_verify_qs())

    serial = await _run_verify_once(tmp_path, monkeypatch, capsys, ["--concurrency", "1"])
    wide = await _run_verify_once(tmp_path, monkeypatch, capsys, ["--concurrency", "4"])
    defaults = await _run_verify_once(tmp_path, monkeypatch, capsys, [])

    for code, stdout, _md, _payload in (serial, wide, defaults):
        assert code == 0
        assert stdout == expected_stdout  # golden vs the old serial algorithm
    assert serial[2] == wide[2] == defaults[2], "review md must be byte-identical"
    assert serial[3] == wide[3] == defaults[3], "review json must agree (modulo stamp)"

    # The fixture itself must exercise a MIXED outcome matrix, else the golden
    # comparison is vacuous.
    payload = serial[3]
    questions = cast("dict[str, dict[str, Any]]", payload["questions"])
    assert [questions[q]["support"] for q in ("q0", "q1", "q2", "q3")] == [
        True,
        False,
        False,
        False,
    ]
    assert [questions[q]["closed_book"]["verdict"] for q in ("q0", "q1", "q2", "q3")] == [
        "correct",
        "incorrect",
        "correct",
        "incorrect",
    ]
    assert all(questions[q]["oracle"]["verdict"] == "correct" for q in questions)


def test_bench_verify_flags_parsing() -> None:
    """`verify` honors --concurrency/--judge-concurrency; both default to None."""
    parser = cli._build_parser()
    explicit = parser.parse_args(
        ["bench", "verify", "--concurrency", "4", "--judge-concurrency", "2"]
    )
    assert explicit.concurrency == 4
    assert explicit.judge_concurrency == 2
    bare = parser.parse_args(["bench", "verify"])
    assert bare.concurrency is None
    assert bare.judge_concurrency is None


@pytest.mark.asyncio
async def test_verify_pass_preserves_order_under_variable_delays(
    monkeypatch: MonkeyPatch,
) -> None:
    """Stubbed SUT with reversed per-question delays still returns answers keyed
    in input order (gather preserves argument order), while proving overlap."""
    delays = {"q0": 0.30, "q1": 0.05, "q2": 0.01}
    done: list[str] = []

    class DelayedSUT:
        async def run_one(self, q: BenchQuestion) -> QuestionOutput:
            await asyncio.sleep(delays[q.id])
            done.append(q.id)
            return _out(f"ans-{q.id}")

    monkeypatch.setattr("vesta.api.bench.make_system", lambda *_a, **_k: DelayedSUT())
    qs = [_q(f"q{i}") for i in range(3)]
    out = await cli._verify_pass(object(), qs, "closed_book", "m", max_concurrent=3)
    assert list(out) == ["q0", "q1", "q2"]
    assert out == {"q0": "ans-q0", "q1": "ans-q1", "q2": "ans-q2"}
    assert done != ["q0", "q1", "q2"], "delays should interleave under concurrency"
    assert done[-1] == "q0"


@pytest.mark.asyncio
async def test_verify_pass_respects_concurrency_bound(monkeypatch: MonkeyPatch) -> None:
    """Max in-flight SUT calls never exceeds the bound (and exceeds 1)."""
    inflight = 0
    peak = 0

    class CountingSUT:
        async def run_one(self, q: BenchQuestion) -> QuestionOutput:
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            try:
                await asyncio.sleep(0.05)
                return _out("x")
            finally:
                inflight -= 1

    monkeypatch.setattr("vesta.api.bench.make_system", lambda *_a, **_k: CountingSUT())
    out = await cli._verify_pass(
        object(), [_q(f"q{i}") for i in range(6)], "oracle", "m", max_concurrent=2
    )
    assert list(out) == [f"q{i}" for i in range(6)]
    assert peak <= 2, f"SUT ran {peak} wide, bound is 2"
    assert peak > 1, "bound=2 should actually overlap"


@pytest.mark.asyncio
async def test_verify_judge_respects_bound_and_order(monkeypatch: MonkeyPatch) -> None:
    """Judgments stay input-ordered and bounded under a slow fake judge."""
    from vesta.eval.bench_scoring import Verdict

    inflight = 0
    peak = 0
    graded: list[str] = []

    async def _slow_judge(**kwargs: Any) -> Any:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0.05)
            graded.append(kwargs["question"].id)
            return SimpleNamespace(verdict=Verdict.CORRECT, reason="r")
        finally:
            inflight -= 1

    monkeypatch.setattr("vesta.eval.bench_scoring.judge_verdict", _slow_judge)
    qs = [_q(f"q{i}") for i in range(5)]
    judged = await cli._verify_judge(None, "", qs, {}, {}, max_concurrent=2)
    assert list(judged) == [f"q{i}" for i in range(5)]
    assert all(o.verdict == Verdict.CORRECT for o in judged.values())
    assert peak <= 2, f"judge ran {peak} wide, bound is 2"
    assert peak > 1
    assert sorted(graded) == [f"q{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_verify_support_respects_bound_and_short_circuit(
    monkeypatch: MonkeyPatch,
) -> None:
    """Extraction stays within the given bound, keeps the serial short-circuit
    (mismatched first source ⇒ later sources of that question never extracted),
    and extracts exactly the same articles the serial pass did."""
    inflight = 0
    peak = 0
    extracted: list[tuple[str, str]] = []

    async def _counting_extract(state: object, q: Any, src: Any) -> str:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0.02)
            extracted.append((q.id, src.article_path))
            return _VERIFY_EXTRACTS[(q.id, src.article_path)]
        finally:
            inflight -= 1

    monkeypatch.setattr(cli, "_extract_article", _counting_extract)
    qs = list(_verify_qs())
    support = await cli._verify_support(object(), qs, max_concurrent=2)
    assert support == {"q0": True, "q1": False, "q2": False, "q3": False}
    assert peak <= 2, f"extraction ran {peak} wide, bound is 2"
    assert peak > 1
    # Same extraction set+count as the serial implementation: second sources of
    # q2/q3 skipped by the short-circuit.
    assert sorted(extracted) == [("q0", "a"), ("q1", "b"), ("q2", "c1"), ("q3", "d1")]


def test_resolve_profile_unknown_exits_cleanly(cli_db: Database) -> None:
    """AUDIT_0824 B5: an explicit unknown --profile exits with a clean message
    instead of silently running the lexical profile; known names resolve."""
    state = _fake_state(cli_db)
    p = cli._resolve_profile(state, "lexical")
    assert p.name == "lexical"
    with pytest.raises(SystemExit, match="no_such_profile"):
        cli._resolve_profile(state, "no_such_profile")


# ── `vesta models` / `bench hardware` DB-aware settings (AUDIT_0824 N20) ────

_GTE_REPO = "onnx-community/gte-modernbert-base-ONNX"
_GRANITE_REPO = "onnx-community/granite-embedding-small-english-r2-ONNX"


@pytest.fixture
async def settings_data_dir(tmp_path: Path) -> Path:
    """A data root whose vesta.db pins a non-default embed repo."""
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        await upsert_setting(conn, "encoders.embed.model", _GTE_REPO, "2026-08-26T00:00:00Z")
    await db.stop()
    return tmp_path


def _stub_hardware_rows(monkeypatch: MonkeyPatch) -> None:
    """Replace the physical measurements with instant stub rows."""
    monkeypatch.setattr(cli.hardware, "measure_cpu_info", lambda: {"machine_id": "test-machine"})
    monkeypatch.setattr(
        cli.hardware,
        "measure_gemm_ceiling",
        lambda: SimpleNamespace(to_row=lambda: {"name": "gemm"}),
    )
    monkeypatch.setattr(
        cli.hardware,
        "measure_memory_bandwidth",
        lambda: SimpleNamespace(to_row=lambda: {"name": "mem"}),
    )
    monkeypatch.setattr(
        cli.encoder,
        "embedder_throughput",
        lambda enc: SimpleNamespace(to_row=lambda: {"name": "embed"}),
    )
    monkeypatch.setattr(
        cli.encoder,
        "reranker_latency",
        lambda enc: SimpleNamespace(to_row=lambda: {"name": "rerank"}),
    )


@pytest.mark.asyncio
async def test_cmd_models_fetches_db_configured_repos(
    settings_data_dir: Path, monkeypatch: MonkeyPatch
) -> None:
    """`vesta models` fetches the repos stored in the settings table, not the
    built-in defaults (N20)."""
    config.reset_for_test()
    fetched: list[tuple[str, str, Path]] = []  # (role, repo_id, model_dir)

    def _fake_ensure_model(spec: Any, model_dir: Path) -> Path:
        fetched.append((str(spec.role), str(spec.repo_id), model_dir))
        return model_dir / str(spec.repo_id)

    monkeypatch.setattr(cli, "ensure_model", _fake_ensure_model)
    try:
        rc = await cli._cmd_models(argparse.Namespace(role=None, data_dir=str(settings_data_dir)))
    finally:
        config.reset_for_test()
    assert rc == 0
    by_role = {role: repo for role, repo, _ in fetched}
    assert by_role["embed"] == _GTE_REPO
    assert by_role["embed"] != _GRANITE_REPO
    assert all(model_dir == settings_data_dir / "models" for _, _, model_dir in fetched)


@pytest.mark.asyncio
async def test_cmd_models_without_db_keeps_defaults(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """No DB yet (fresh install) → default+env behavior is unchanged."""
    config.reset_for_test()
    monkeypatch.delenv("VESTA_ENCODERS_EMBED_MODEL", raising=False)
    fetched: list[str] = []
    monkeypatch.setattr(
        cli, "ensure_model", lambda spec, model_dir: fetched.append(str(spec.repo_id))
    )
    try:
        rc = await cli._cmd_models(argparse.Namespace(role=["embed"], data_dir=str(tmp_path)))
    finally:
        config.reset_for_test()
    assert rc == 0
    assert fetched == [_GRANITE_REPO]


@pytest.mark.asyncio
async def test_bench_hardware_builds_encoders_from_db_settings(
    settings_data_dir: Path, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """`bench hardware` measures the DB-configured encoder repos (N20)."""
    config.reset_for_test()
    _stub_hardware_rows(monkeypatch)
    seen: dict[str, Any] = {}

    class _FakeMgr:
        async def get_embed(self) -> object:
            return object()

        async def get_rerank(self) -> object:
            return object()

    def _fake_build(snapshot: Any, *, model_dir: Path | None = None) -> _FakeMgr:
        seen["embed"] = str(snapshot.get(ENCODERS_EMBED_MODEL))
        seen["model_dir"] = model_dir
        return _FakeMgr()

    monkeypatch.setattr(cli, "build_manager_from_settings", _fake_build)
    args = argparse.Namespace(
        skip_extraction=True,
        archive=None,
        out_dir=str(tmp_path / "bench_results"),
        data_dir=str(settings_data_dir),
    )
    try:
        rc = await cli._cmd_bench(args)
    finally:
        config.reset_for_test()
    assert rc == 0
    assert seen["embed"] == _GTE_REPO
    assert seen["model_dir"] == settings_data_dir / "models"


@pytest.mark.asyncio
async def test_bench_hardware_archive_default_anchored_to_data_root(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """With no --archive, the pinned archive resolves under <data-dir>/zims —
    independent of the process working directory (N20)."""
    config.reset_for_test()
    data_root = tmp_path / "elsewhere"
    (data_root / "zims").mkdir(parents=True)
    archive = data_root / "zims" / EVAL_ARCHIVE_PATH
    archive.write_bytes(b"not a real zim")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    captured: list[str] = []
    monkeypatch.setattr(cli, "_run_extraction_bench", lambda path: captured.append(path) or [])
    _stub_hardware_rows(monkeypatch)
    monkeypatch.setattr(
        cli.encoder, "onnx_int8_speedup", lambda: SimpleNamespace(to_row=lambda: {"name": "int8"})
    )
    args = argparse.Namespace(
        skip_extraction=False,
        archive=None,
        out_dir=str(tmp_path / "bench_results"),
        data_dir=str(data_root),
    )
    try:
        rc = await cli._cmd_bench(args)
    finally:
        config.reset_for_test()
    assert rc == 0
    assert captured == [str(archive)]
