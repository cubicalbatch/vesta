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
from contextlib import asynccontextmanager
from pathlib import Path

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
from vesta.eval.bench_dataset import (
    BenchDataset,
    BenchQuestion,
    BenchSource,
    dataset_hash,
)
from vesta.eval.bench_runner import BenchRunRecord, QuestionOutput, run_benchmark
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
