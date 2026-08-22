"""Round-0 article-recall measurement over the bench dataset.

The round-0 "before" number. For every dataset question that names at least
one ``required`` gold article, run the retrieval pipeline under three fixed,
zero-LLM arms and record the 1-based rank of a gold article among the returned
source cards:

* **A** — the natural-language question, ``standard`` profile. Exactly what
  Round-0 ``search_exact`` fires today.
* **D** — the natural-language question, ``hybrid`` profile (the dense rescue).
* **B** — the *gold article's title*, ``standard`` profile. The oracle ceiling
  for any query-shaping step, LLM or otherwise.

Per-question ranks reuse :func:`vesta.eval.bench_scoring.source_hit_rank`
(first retrieved path naming ANY required source — the same semantics as
``bench run``'s source metrics, so one number means one thing across both
eval surfaces). Arm B's query is the first required source's title, matching
the throwaway probe this eval supersedes. Aggregates report
recall@1/@5/@10/any; the per-arm diff names every question that moved so a
later phase can point at exactly the questions it won or lost.

Arms pin their profiles by built-in name — never the ``retrieval.active_profile``
setting, which a user's DB may have flipped away from the code default.

Boundary: imports only ``vesta.eval`` siblings (same
package, not counted) + ``vesta.retrieval``. No db/zim/api. The pipeline runs
through the injected :class:`vesta.eval.runner.PipelineRunner`; persistence is
the caller's concern (this mode writes a JSON artifact, never the bench tables).
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from vesta.eval.bench_dataset import BenchQuestion, BenchSource
from vesta.eval.bench_scoring import source_hit_rank
from vesta.eval.runner import PipelineRunner
from vesta.retrieval.profiles import RetrievalProfile

__all__ = [
    "ARMS",
    "ArmDiff",
    "ArmOutcome",
    "ArmRecall",
    "ArticleRecallReport",
    "RecallArm",
    "evaluate_article_recall",
    "gold_source",
    "select_recall_questions",
]


# ── The fixed arms ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecallArm:
    """One measurement arm: profile pin + which string becomes the query."""

    key: str  # "A" | "D" | "B"
    label: str
    profile: str  # built-in profile NAME, pinned — never the active setting
    oracle_title: bool  # query = the gold article's title, not the question


#: The arms, in table order. Zero LLM in every arm.
ARMS: tuple[RecallArm, ...] = (
    RecallArm("A", "NL question / standard (today)", "standard", False),
    RecallArm("D", "NL question / hybrid (dense)", "hybrid", False),
    RecallArm("B", "gold article title / standard (oracle)", "standard", True),
)


# ── Question selection ───────────────────────────────────────────────────────


def _required_sources(sources: Sequence[BenchSource]) -> tuple[BenchSource, ...]:
    return tuple(s for s in sources if s.required)


def select_recall_questions(questions: Sequence[BenchQuestion]) -> tuple[BenchQuestion, ...]:
    """The article-recall denominator: answer questions naming a gold article.

    ``expected_behavior == "answer"`` AND ≥1 ``required`` source — the probe
    criterion. Abstain questions score abstention, not recall;
    a question with no required source has no gold article to rank.
    """
    return tuple(
        q for q in questions if q.expected_behavior == "answer" and _required_sources(q.sources)
    )


def gold_source(q: BenchQuestion) -> BenchSource:
    """The first required source — arm B's oracle title (probe semantics)."""
    return _required_sources(q.sources)[0]


# ── Records ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArmOutcome:
    """One arm's outcome for one question."""

    rank: int | None  # 1-based rank of a gold article; None = miss
    n_cards: int
    paths: tuple[str, ...]
    degraded: bool  # retrieval-relevant capability drop (the llm drop is expected)


@dataclass(frozen=True)
class QuestionRecall:
    """Per-question detail — the unit a later phase diffs against."""

    question_id: str
    capability: str
    question: str
    gold_paths: tuple[str, ...]  # every required source path
    oracle_title: str  # arm B's query
    arms: Mapping[str, ArmOutcome]


@dataclass(frozen=True)
class ArmRecall:
    """Set-level aggregate for one arm."""

    arm: str
    label: str
    profile: str
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_any: float


@dataclass(frozen=True)
class ArmDiff:
    """Per-question movement of one arm against the baseline arm."""

    baseline: str
    arm: str
    rescued: tuple[str, ...]  # question ids the baseline missed, this arm found
    lost: tuple[str, ...]  # ids the baseline found, this arm missed


@dataclass(frozen=True)
class ArticleRecallReport:
    """The full article-recall measurement: aggregates + per-question movement."""

    arms: tuple[ArmRecall, ...]
    diffs: tuple[ArmDiff, ...]  # D vs A, then B vs A
    questions: tuple[QuestionRecall, ...]
    degraded: tuple[str, ...]  # arm keys with any degraded run (honesty guard)

    def to_dict(self) -> dict[str, object]:
        """The JSON-artifact shape: aggregates, moved ids, per-question ranks."""
        return {
            "metric": "article_recall",
            "n": len(self.questions),
            "degraded": list(self.degraded),
            "arms": [
                {
                    "arm": a.arm,
                    "label": a.label,
                    "profile": a.profile,
                    "n": a.n,
                    "recall_at_1": a.recall_at_1,
                    "recall_at_5": a.recall_at_5,
                    "recall_at_10": a.recall_at_10,
                    "recall_any": a.recall_any,
                }
                for a in self.arms
            ],
            "diffs": [
                {
                    "baseline": d.baseline,
                    "arm": d.arm,
                    "rescued": list(d.rescued),
                    "lost": list(d.lost),
                }
                for d in self.diffs
            ],
            "questions": [
                {
                    "id": q.question_id,
                    "capability": q.capability,
                    "question": q.question,
                    "gold_paths": list(q.gold_paths),
                    "oracle_title": q.oracle_title,
                    "arms": {
                        key: {
                            "rank": o.rank,
                            "n_cards": o.n_cards,
                            "paths": list(o.paths),
                            "degraded": o.degraded,
                        }
                        for key, o in q.arms.items()
                    },
                }
                for q in self.questions
            ],
        }

    def render(self) -> str:
        """The recall table + the moved-question lists, as terminal text."""
        lines = [
            f"article recall — gold article among source cards (n={len(self.questions)})",
            f"{'arm':<40} {'@1':>6} {'@5':>6} {'@10':>6} {'any':>6}",
        ]
        for a in self.arms:
            lines.append(
                f"{f'{a.arm}  {a.label}':<40} {a.recall_at_1:>6.2f} {a.recall_at_5:>6.2f} "
                f"{a.recall_at_10:>6.2f} {a.recall_any:>6.2f}"
            )
        n = len(self.questions)
        for d in self.diffs:
            lines.append(
                f"\n{d.arm} vs {d.baseline}: rescues {len(d.rescued)}/{n}, loses {len(d.lost)}/{n}"
            )
            for label, ids in (("rescued", d.rescued), ("lost", d.lost)):
                if ids:
                    wrapped = textwrap.fill(
                        " ".join(ids),
                        width=96,
                        initial_indent=f"  {label}: ",
                        subsequent_indent="    ",
                    )
                    lines.append(wrapped)
        return "\n".join(lines)


# ── Orchestration ────────────────────────────────────────────────────────────


def _trace_degraded(trace: Mapping[str, object]) -> bool:
    """A RETRIEVAL-relevant capability drop was recorded.

    The ``llm`` drop is expected here — the arms are zero-LLM by design and the
    conversational rewriter is a no-op on turn 1 regardless. Any OTHER drop
    (vectors, static/cross encoder) means an arm is not measuring what it
    claims — e.g. arm D silently equal to arm A when ``VECTORS`` is unmet.
    """
    raw = trace.get("degradations")
    degs: Sequence[object] = raw if isinstance(raw, list) else ()
    return any(isinstance(d, Mapping) and str(d.get("missing")) != "llm" for d in degs)


async def evaluate_article_recall(
    questions: Sequence[BenchQuestion],
    runner: PipelineRunner,
    profiles: Mapping[str, RetrievalProfile],
    *,
    progress: Callable[[int, int, QuestionRecall], None] | None = None,
) -> ArticleRecallReport:
    """Run every arm over every question; rank gold among the source cards.

    Sequential and deterministic: one question at a time, arms in
    :data:`ARMS` order, through the injected ``runner``. ``profiles`` maps each
    arm's pinned profile name to the resolved profile (the composition root
    resolves built-ins; this module never reads the active-profile setting).

    The denominator is always :func:`select_recall_questions` of the input —
    non-source questions are dropped here, not trusted from the caller.
    """
    missing = sorted({a.profile for a in ARMS} - set(profiles))
    if missing:
        raise ValueError(f"missing pinned arm profile(s): {', '.join(missing)}")
    questions = select_recall_questions(questions)
    if not questions:
        raise ValueError("no source-eligible questions (answer behavior + >=1 required source)")

    rows: list[QuestionRecall] = []
    total = len(questions)
    for q in questions:
        gold = gold_source(q)
        oracle_title = gold.article_title or gold.article_path
        outcomes: dict[str, ArmOutcome] = {}
        for arm in ARMS:
            query = oracle_title if arm.oracle_title else q.question
            paths, trace = await runner.run(profiles[arm.profile], query)
            outcomes[arm.key] = ArmOutcome(
                rank=source_hit_rank(paths, q.sources),
                n_cards=len(paths),
                paths=tuple(paths),
                degraded=_trace_degraded(trace),
            )
        row = QuestionRecall(
            question_id=q.id,
            capability=q.capability,
            question=q.question,
            gold_paths=tuple(s.article_path for s in _required_sources(q.sources)),
            oracle_title=oracle_title,
            arms=outcomes,
        )
        rows.append(row)
        if progress is not None:
            progress(len(rows), total, row)

    n = len(rows)
    by_id = {r.question_id: r for r in rows}

    def _aggregate(arm: RecallArm) -> ArmRecall:
        ranks = [r.arms[arm.key].rank for r in rows]

        def frac(limit: int | None) -> float:
            return sum(1 for rk in ranks if rk is not None and (limit is None or rk <= limit)) / n

        return ArmRecall(
            arm=arm.key,
            label=arm.label,
            profile=arm.profile,
            n=n,
            recall_at_1=frac(1),
            recall_at_5=frac(5),
            recall_at_10=frac(10),
            recall_any=frac(None),
        )

    def _diff(baseline: str, arm: str) -> ArmDiff:
        ids = [r.question_id for r in rows]
        rescued = tuple(
            qid
            for qid in ids
            if by_id[qid].arms[baseline].rank is None and by_id[qid].arms[arm].rank is not None
        )
        lost = tuple(
            qid
            for qid in ids
            if by_id[qid].arms[baseline].rank is not None and by_id[qid].arms[arm].rank is None
        )
        return ArmDiff(baseline=baseline, arm=arm, rescued=rescued, lost=lost)

    degraded = tuple(arm.key for arm in ARMS if any(r.arms[arm.key].degraded for r in rows))
    return ArticleRecallReport(
        arms=tuple(_aggregate(arm) for arm in ARMS),
        diffs=(_diff("A", "D"), _diff("A", "B")),
        questions=tuple(rows),
        degraded=degraded,
    )
