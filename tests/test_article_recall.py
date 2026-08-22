"""Article-recall arms (``vesta bench retrieval --dataset``).

Two layers, both offline:

* **Recall/diff computation** — :mod:`vesta.eval.article_recall` over a scripted
  ``PipelineRunner``: rank semantics (any required source), aggregate @1/@5/@10
  math, per-arm rescued/lost ids, the degradation guard, render + artifact shape.
* **CLI wiring** — the ``--dataset`` mode boots, selects source-eligible
  questions, prints the arm table, writes the JSON artifact, and never touches
  ``eval_runs``/``bench_runs``; golden-set flags are rejected in dataset mode.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from vesta.api.state import AppState
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.eval.article_recall import (
    ARMS,
    evaluate_article_recall,
    gold_source,
    select_recall_questions,
)
from vesta.eval.bench_dataset import BenchQuestion, BenchSource
from vesta.retrieval.profiles import RetrievalProfile, load_profile

# ── Fakes ────────────────────────────────────────────────────────────────────


class ArmScriptRunner:
    """Returns scripted paths keyed on ``(profile.hash, query)``; records calls."""

    def __init__(
        self,
        script: Mapping[tuple[str, str], Sequence[str]],
        degraded_profiles: frozenset[str] = frozenset(),
        degrade_missing: str = "vectors",
    ) -> None:
        self._script = script
        self._degraded = degraded_profiles
        self._missing = degrade_missing
        self.calls: list[tuple[str, str]] = []

    async def run(
        self, profile: RetrievalProfile, query: str
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        self.calls.append((profile.hash, query))
        paths = tuple(self._script.get((profile.hash, query), ()))
        trace: dict[str, object] = {"stages": [], "degradations": []}
        if profile.hash in self._degraded:
            trace["degradations"] = [
                {"component": "preparer/conversational_rewrite", "missing": self._missing}
            ]
        return paths, trace


def _question(
    qid: str,
    question: str,
    *,
    sources: Sequence[tuple[str, str]] = (("Some_Gold", "Some Gold"),),
    behavior: str = "answer",
    required: bool = True,
) -> BenchQuestion:
    return BenchQuestion(
        id=qid,
        question=question,
        capability="buried_fact",
        difficulty="easy",
        slice="core",
        expected_behavior=behavior,
        answer="x",
        sources=tuple(
            BenchSource(
                zim="wikipedia.zim", article_title=title, article_path=path, required=required
            )
            for path, title in sources
        ),
    )


def _profiles() -> dict[str, RetrievalProfile]:
    out: dict[str, RetrievalProfile] = {}
    for name in {arm.profile for arm in ARMS}:
        p = load_profile(name)
        assert p is not None
        out[name] = p
    return out


def _scripted_three_question_run() -> tuple[ArmScriptRunner, list[BenchQuestion]]:
    """The 3-question scenario: exact ranks per arm (see assertions below)."""
    prof = _profiles()
    std, hyb = prof["standard"].hash, prof["hybrid"].hash
    q1 = _question("q1", "q1?", sources=(("Olfactory_bulb", "Olfactory bulb"),))
    q2 = _question("q2", "q2?", sources=(("Telephone", "Telephone"),))
    q3 = _question("q3", "q3?", sources=(("Elizabeth_II", "Elizabeth II"),))
    script = {
        (std, "q1?"): ("X1", "Olfactory_bulb"),
        (hyb, "q1?"): ("X1", "X2", "Olfactory_bulb"),
        (std, "Olfactory bulb"): ("Olfactory_bulb",),
        (std, "q2?"): ("A", "B", "C", "D", "E", "F"),
        (hyb, "q2?"): ("A", "B", "C", "D", "E", "F", "Telephone"),
        (std, "Telephone"): ("Telephone",),
        (std, "q3?"): ("Elizabeth_II",),
        (hyb, "q3?"): ("X",),
        (std, "Elizabeth II"): ("X",),
    }
    return ArmScriptRunner(script), [q1, q2, q3]


# ── Selection ────────────────────────────────────────────────────────────────


def test_selection_keeps_answer_questions_with_a_required_source() -> None:
    eligible = _question("a", "a?")
    optional_only = _question("b", "b?", required=False)
    abstain = _question("c", "c?", behavior="abstain")
    qs = [eligible, optional_only, abstain]
    assert [q.id for q in select_recall_questions(qs)] == ["a"]


def test_gold_source_is_the_first_required_one() -> None:
    q = _question("a", "a?", sources=(("First_A", "First A"), ("Second_B", "Second B")))
    assert gold_source(q).article_path == "First_A"


# ── Ranks, aggregates, diffs ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_math_and_arm_diffs() -> None:
    runner, qs = _scripted_three_question_run()
    report = await evaluate_article_recall(qs, runner, _profiles())
    by_arm = {a.arm: a for a in report.arms}

    # Ranks: A = (2, miss, 1); D = (3, 7, miss); B = (1, 1, miss).
    assert [r.arms["A"].rank for r in report.questions] == [2, None, 1]
    assert [r.arms["D"].rank for r in report.questions] == [3, 7, None]
    assert [r.arms["B"].rank for r in report.questions] == [1, 1, None]

    a, d, b = by_arm["A"], by_arm["D"], by_arm["B"]
    assert (a.n, a.recall_at_1, a.recall_at_5, a.recall_at_10, a.recall_any) == (
        3,
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    assert (d.recall_at_1, d.recall_at_5, d.recall_at_10, d.recall_any) == (
        pytest.approx(0.0),
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )
    assert (b.recall_at_1, b.recall_at_5, b.recall_at_10, b.recall_any) == (
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    )

    # The diff names exactly which questions moved.
    d_diff, b_diff = report.diffs
    assert (d_diff.baseline, d_diff.arm, d_diff.rescued, d_diff.lost) == (
        "A",
        "D",
        ("q2",),
        ("q3",),
    )
    assert (b_diff.baseline, b_diff.arm, b_diff.rescued, b_diff.lost) == (
        "A",
        "B",
        ("q2",),
        ("q3",),
    )

    # Arm queries: A/D searched the NL question, B searched the gold title.
    prof = _profiles()
    std, hyb = prof["standard"].hash, prof["hybrid"].hash
    assert (std, "q1?") in runner.calls  # A = NL question on standard
    assert (hyb, "q1?") in runner.calls  # D = NL question on hybrid
    assert (std, "Olfactory bulb") in runner.calls  # B = gold title on standard
    assert not any(query == "Olfactory bulb" and ph != std for ph, query in runner.calls)

    # The artifact carries per-question per-arm ranks.
    payload = report.to_dict()
    assert payload["metric"] == "article_recall"
    q1 = payload["questions"][0]
    assert q1["id"] == "q1"
    assert q1["oracle_title"] == "Olfactory bulb"
    assert q1["gold_paths"] == ["Olfactory_bulb"]
    assert q1["arms"]["A"]["rank"] == 2
    assert q1["arms"]["D"]["paths"] == ["X1", "X2", "Olfactory_bulb"]


@pytest.mark.asyncio
async def test_rank_counts_any_required_source_and_ignores_optional() -> None:
    """Multi-hop semantics: the SECOND required source hitting still counts."""
    q = _question(
        "mh",
        "mh?",
        sources=(("First_A", "First A"), ("Second_B", "Second B")),
    )
    optional = _question("opt", "opt?", sources=(("Opt_C", "Opt C"),), required=False)
    # "Opt_C" is the only source of `optional`, so that question is not selected.
    prof = _profiles()
    std = prof["standard"].hash
    runner = ArmScriptRunner(
        {
            (std, "mh?"): ("Second_B", "First_A"),  # second required first
            (std, "First A"): ("Opt_C", "Second_B", "First_A"),  # optional never counts
        }
    )
    report = await evaluate_article_recall([q, optional], runner, prof)
    assert [r.question_id for r in report.questions] == ["mh"]
    # NL query: the second required source at rank 1 still counts as a hit.
    assert report.questions[0].arms["A"].rank == 1
    # Oracle arm: an optional (non-required) source ahead of gold does not count.
    assert report.questions[0].arms["B"].rank == 2


@pytest.mark.asyncio
async def test_llm_capability_drop_is_not_degradation() -> None:
    """The zero-LLM arms drop the conversational rewriter by design."""
    prof = _profiles()
    runner = ArmScriptRunner(
        {(p.hash, "a?"): ("Gold",) for p in prof.values()},
        degraded_profiles=frozenset(p.hash for p in prof.values()),
        degrade_missing="llm",
    )
    report = await evaluate_article_recall([_question("a", "a?")], runner, prof)
    assert report.degraded == ()
    assert not report.questions[0].arms["D"].degraded


@pytest.mark.asyncio
async def test_degraded_arm_is_flagged() -> None:
    prof = _profiles()
    runner = ArmScriptRunner(
        {(p, "a?"): ("Gold",) for p in (prof["standard"].hash, prof["hybrid"].hash)},
        degraded_profiles=frozenset({prof["hybrid"].hash}),
    )
    report = await evaluate_article_recall([_question("a", "a?")], runner, prof)
    assert report.degraded == ("D",)
    assert report.questions[0].arms["D"].degraded
    assert not report.questions[0].arms["A"].degraded


@pytest.mark.asyncio
async def test_missing_arm_profile_raises() -> None:
    runner = ArmScriptRunner({})
    with pytest.raises(ValueError, match="hybrid"):
        await evaluate_article_recall(
            [_question("a", "a?")], runner, {"standard": _profiles()["standard"]}
        )


@pytest.mark.asyncio
async def test_empty_selection_raises() -> None:
    """A non-source question set cannot produce a recall denominator."""
    runner = ArmScriptRunner({})
    with pytest.raises(ValueError, match="no source-eligible questions"):
        await evaluate_article_recall(
            [_question("a", "a?", behavior="abstain")], runner, _profiles()
        )


@pytest.mark.asyncio
async def test_render_names_moved_questions() -> None:
    runner, qs = _scripted_three_question_run()
    report = await evaluate_article_recall(qs, runner, _profiles())
    text = report.render()
    assert "article recall" in text
    assert "A  NL question / standard (today)" in text
    assert "D  NL question / hybrid (dense)" in text
    assert "B  gold article title / standard (oracle)" in text
    assert "D vs A: rescues 1/3, loses 1/3" in text
    assert "rescued: q2" in text
    assert "lost: q3" in text


# ── CLI wiring ───────────────────────────────────────────────────────────────


@pytest.fixture
async def cli_db(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "cli.db"), busy_timeout_ms=1000)
    await database.start()
    async with database.write() as conn:
        await run_migrations(conn)
    yield database
    await database.stop()


def _write_dataset(path: Path) -> None:
    payload = {
        "name": "vesta_test",
        "version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "Where is the gold?",
                "capability": "buried_fact",
                "difficulty": "easy",
                "slice": "core",
                "expected_behavior": "answer",
                "answer": "There.",
                "sources": [
                    {
                        "zim": "wikipedia.zim",
                        "article_title": "Gold A",
                        "article_path": "Gold_A",
                        "required": True,
                    }
                ],
            },
            {
                "id": "q2",
                "question": "Unanswerable?",
                "capability": "adversarial",
                "difficulty": "hard",
                "slice": "core",
                "expected_behavior": "abstain",
                "answer": "No answer.",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_cli_dataset_mode_runs_arms_and_writes_artifact(
    cli_db: Database,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--dataset` runs the arms, prints the table, writes the artifact only."""
    from vesta import cli
    from vesta import config as app_config

    ds_path = tmp_path / "ds.json"
    out_path = tmp_path / "recall.json"
    _write_dataset(ds_path)

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        app_config.configure()
        yield AppState(db=cli_db, runner=None, registry=None, gateway=None, supervisor=None)

    class _FakeRunner:
        def __init__(self, _state):
            pass

        async def run(self, profile, query):
            return ("Gold_A", "Other"), {"stages": [], "degradations": []}

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)
    monkeypatch.setattr(cli, "CLIPipelineRunner", _FakeRunner)

    args = cli._build_parser().parse_args(
        ["bench", "retrieval", "--dataset", str(ds_path), "--out", str(out_path)]
    )
    code = await cli._cmd_eval(args)
    out = capsys.readouterr().out
    assert code == 0
    # The abstain question is excluded from the denominator.
    assert "1 source-eligible questions" in out
    assert "[1/1] q1" in out
    assert "A  NL question / standard (today)" in out
    assert "wrote " in out
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["metric"] == "article_recall"
    assert payload["dataset"]["questions_selected"] == 1
    assert payload["questions"][0]["arms"]["A"]["rank"] == 1
    assert payload["arms"][0]["recall_at_1"] == pytest.approx(1.0)

    # Dataset mode never writes the bench tables.
    async with cli_db.read() as conn:
        for table in ("eval_runs", "bench_runs"):
            cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = (await cur.fetchone())[0]
            assert count == 0, f"{table} must stay empty in dataset mode"


@pytest.mark.asyncio
async def test_cli_dataset_mode_rejects_golden_set_flags(
    cli_db: Database,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--profile/--sweep and sub-actions are golden-set only; dataset pins arms."""
    from vesta import cli

    ds_path = tmp_path / "ds.json"
    _write_dataset(ds_path)

    @asynccontextmanager
    async def _fake_open(*_args, **_kwargs):
        yield AppState(db=cli_db, runner=None, registry=None, gateway=None, supervisor=None)

    monkeypatch.setattr(cli, "_open_runtime", _fake_open)

    parser = cli._build_parser()
    for argv, needle in (
        (["bench", "retrieval", "--dataset", str(ds_path), "--profile", "standard"], "--profile"),
        (["bench", "retrieval", "--dataset", str(ds_path), "--sweep", "rrf.k=10"], "--sweep"),
        (["bench", "retrieval", "--dataset", str(ds_path), "calibrate"], "sub-action"),
    ):
        code = await cli._cmd_eval(parser.parse_args(argv))
        assert code == 2
        assert needle in capsys.readouterr().out
