"""The eval runner: run a profile over the golden set, persist, compare, sweep.

Three jobs, one module:

1. **Run** a profile over every golden query, collecting retrieved paths + the
   trace per query, then reduce to :class:`RunMetrics`.
2. **Persist** each run to ``eval_runs`` with the five pins:
   profile content hash, settings snapshot, archive checksum, git SHA, machine
   id — plus the golden-set hash. A run without all five is not comparable.
3. **Compare** against a baseline (``--baseline``): a delta table with per-slice
   movement and a per-query win/loss list (the win/loss list is what
   actually explains a regression; the aggregate rarely does).

The parameter sweep (``--sweep rrf.k=10,20,40,60``) is driven off the registry's
component schemas with **zero per-component harness code**: the sweep
names a component by its ``impl`` id and a param on that component's ``Params``,
clones the profile with the override, and re-runs. Adding a new sweepable param
requires no change here.

Boundary: ``eval`` imports only ``retrieval`` and ``config``. The pipeline runs
through the injected :class:`PipelineRunner` and persistence through the
injected :class:`EvalStore` Protocol — both defined here so this module never
imports ``db`` or ``zim`` (the ≤2 dependency cap).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from vesta.eval.golden import SLICES, GoldenEntry, GoldenSet, load_set
from vesta.eval.metrics import (
    LatencyPercentiles,
    RunMetrics,
    SliceMetrics,
    aggregate,
    degradations_from_traces,
    latency_from_traces,
    path_matches,
    run_metrics_from_dict,
)
from vesta.retrieval.profiles import (
    ProfileComponent,
    RetrievalProfile,
    profile_to_yaml,
)

# ── Injected seams (keep eval free of db/zim imports) ────────────────────────


class PipelineRunner(Protocol):
    """Run one retrieval query and return ``(retrieved_paths, trace_dict)``.

    The CLI wires this to the real :func:`run_pipeline` over an
    :class:`ArchiveRegistry`; tests wire it to a fake. ``retrieved_paths`` is the
    ordered list of candidate article paths (source cards) — what metrics rank.
    """

    async def run(
        self, profile: RetrievalProfile, query: str
    ) -> tuple[Sequence[str], Mapping[str, object]]: ...


class EvalStore(Protocol):
    """Persistence backend for eval runs (wired to ``eval_runs`` by the CLI/API).

    Defining the Protocol here keeps ``eval`` from importing ``db``: the runner
    hands the store a plain dict and does not care where it lands.
    """

    async def insert_run(self, record: RunRecord) -> int: ...
    async def update_run(self, run_id: int, record: RunRecord) -> bool: ...
    async def get_run(self, run_id: int) -> RunRecord | None: ...
    async def list_runs(self, limit: int = 50) -> list[RunRecord]: ...


# ── Run record (the persisted row) ───────────────────────────────────────────


@dataclass(frozen=True)
class RunRecord:
    """One persisted eval run: identity + the five pins + metrics.

    The pins (profile_hash, settings_snapshot, archive_checksum, git_sha,
    machine_id) make a run comparable to another — without all five, two numbers
    are not comparable. ``golden_hash`` additionally pins which
    version of the golden set produced the metrics.
    """

    id: int
    started_at: str
    profile_name: str
    profile_hash: str
    profile_yaml: str
    golden_hash: str
    archive_path: str
    archive_checksum: str
    settings_snapshot: Mapping[str, object]
    git_sha: str
    machine_id: str
    metrics: RunMetrics
    per_query: tuple[dict[str, object], ...]
    notes: str = ""

    def to_config_json(self) -> dict[str, object]:
        """The ``config_json`` column: everything that identifies this run."""
        return {
            "profile_name": self.profile_name,
            "profile_hash": self.profile_hash,
            "profile_yaml": self.profile_yaml,
            "golden_hash": self.golden_hash,
            "archive_path": self.archive_path,
            "archive_checksum": self.archive_checksum,
            "settings_snapshot": dict(self.settings_snapshot),
            "git_sha": self.git_sha,
            "machine_id": self.machine_id,
            "notes": self.notes,
        }

    def to_metrics_json(self) -> dict[str, object]:
        """The ``metrics_json`` column: aggregate metrics + per-query detail."""
        return {
            "metrics": self.metrics.to_dict(),
            "per_query": list(self.per_query),
        }


def record_from_row(
    row_id: int,
    config: Mapping[str, object],
    metrics_blob: Mapping[str, object],
    started_at: str,
    *,
    profile_name: str | None = None,
    profile_hash: str | None = None,
    golden_hash: str | None = None,
    archive_checksum: str | None = None,
    git_sha: str | None = None,
    machine_id: str | None = None,
) -> RunRecord:
    """Reconstruct a :class:`RunRecord` from its persisted columns + JSON blobs.

    The columns (added in migration 0003) carry the indexed facets; the JSON
    blobs carry the full detail. Columns take precedence when present (they are
    authoritative over the blob for the facets the runner queries by).
    """
    raw_metrics = metrics_blob.get("metrics")
    metrics = (
        run_metrics_from_dict(raw_metrics)
        if isinstance(raw_metrics, dict)
        else RunMetrics({}, LatencyPercentiles(), False, (), 0)
    )
    raw_pq = metrics_blob.get("per_query")
    per_query = tuple(
        pq if isinstance(pq, dict) else {} for pq in (raw_pq if isinstance(raw_pq, list) else [])
    )
    raw_snapshot = config.get("settings_snapshot")
    snapshot = raw_snapshot if isinstance(raw_snapshot, dict) else {}
    return RunRecord(
        id=row_id,
        started_at=started_at,
        profile_name=str(profile_name or config.get("profile_name") or ""),
        profile_hash=str(profile_hash or config.get("profile_hash") or ""),
        profile_yaml=str(config.get("profile_yaml") or ""),
        golden_hash=str(golden_hash or config.get("golden_hash") or ""),
        archive_path=str(config.get("archive_path") or ""),
        archive_checksum=str(archive_checksum or config.get("archive_checksum") or ""),
        settings_snapshot=snapshot,
        git_sha=str(git_sha or config.get("git_sha") or ""),
        machine_id=str(machine_id or config.get("machine_id") or ""),
        metrics=metrics,
        per_query=per_query,
        notes=str(config.get("notes") or ""),
    )


# ── Per-query result ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryResult:
    """One query's outcome: retrieved paths, hit rank, and the trace."""

    entry: GoldenEntry
    retrieved_paths: tuple[str, ...]
    hit_rank: int | None
    trace: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.entry.id,
            "query": self.entry.query,
            "slice": self.entry.slice,
            "expected_paths": list(self.entry.expected_paths),
            "retrieved_paths": list(self.retrieved_paths),
            "hit_rank": self.hit_rank,
            "expected_fact": self.entry.expected_fact,
            "provenance": self.entry.provenance,
        }


# ── Machine / git identity (pure helpers, no internal imports) ───────────────


def machine_id() -> str:
    """A stable per-machine identifier for benchmark namespacing.

    Uses the platform node name plus the CPU count; not a hardware fingerprint —
    its only job is to namespace committed benchmark files so a laptop's numbers
    are never mistaken for the reference. Read from the real environment here
    only because this is the one place identity is gathered (mirrors how
    ``config/resolution`` is the single env reader; this is not a setting).
    """
    import os
    import platform

    node = platform.node() or "unknown"
    ncpu = os.cpu_count() or 0
    return f"{node}-{ncpu}cpu"


def git_sha() -> str:
    """Current git commit short SHA, or ``unknown`` outside a worktree.

    Recorded with every run so a metric is traceable to the exact code that
    produced it. Best-effort: a missing git binary or worktree is not fatal.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


# ── Core run ─────────────────────────────────────────────────────────────────


async def evaluate_profile(
    profile: RetrievalProfile,
    runner: PipelineRunner,
    golden: GoldenSet | str = "full",
) -> tuple[RunMetrics, tuple[QueryResult, ...]]:
    """Run ``profile`` over the golden set; return metrics + per-query results.

    ``golden`` may be a loaded :class:`GoldenSet` or a name (``"full"`` /
    ``"fixture_subset"``). Each query runs through the injected ``runner``; the
    retrieved paths are ranked against the expected set, the trace is kept for
    latency/degradation reduction. ``out_of_corpus`` queries contribute to
    abstention accuracy (handled by answer metrics) and to latency, not recall.
    """
    gs = golden if isinstance(golden, GoldenSet) else load_set(golden)
    results: list[QueryResult] = []
    for entry in gs.entries:
        retrieved, trace = await runner.run(profile, entry.query)
        retrieved_tuple = tuple(retrieved)
        hit_rank: int | None = None
        if entry.expected_paths:
            for i, p in enumerate(retrieved_tuple):
                if any(path_matches(p, exp) for exp in entry.expected_paths):
                    hit_rank = i + 1
                    break
        results.append(
            QueryResult(
                entry=entry,
                retrieved_paths=retrieved_tuple,
                hit_rank=hit_rank,
                trace=trace,
            )
        )
    metrics = _reduce_metrics(results)
    return metrics, tuple(results)


def _reduce_metrics(results: Sequence[QueryResult]) -> RunMetrics:
    """Aggregate per-query results into per-slice + overall metrics + latency."""
    slices: dict[str, SliceMetrics] = {}
    # Group by slice; the 'all' aggregate spans every non-abstention query.
    by_slice: dict[str, list[tuple[Sequence[str], Sequence[str]]]] = {s: [] for s in SLICES}
    all_pairs: list[tuple[Sequence[str], Sequence[str]]] = []
    for r in results:
        pair = (list(r.retrieved_paths), list(r.entry.expected_paths))
        by_slice.setdefault(r.entry.slice, []).append(pair)
        if r.entry.expected_paths:  # out_of_corpus excluded from recall aggregates
            all_pairs.append(pair)
    for name, pairs in by_slice.items():
        slices[name] = aggregate(pairs)
    slices["all"] = aggregate(all_pairs)
    traces = [r.trace for r in results]
    latency = latency_from_traces(traces)
    degraded_components = degradations_from_traces(traces)
    return RunMetrics(
        slices=slices,
        latency_ms=latency,
        degraded=len(degraded_components) > 0,
        degraded_components=degraded_components,
        query_count=len(results),
    )


async def persist_run(
    store: EvalStore,
    *,
    profile: RetrievalProfile,
    golden: GoldenSet,
    metrics: RunMetrics,
    results: Sequence[QueryResult],
    settings_snapshot: Mapping[str, object],
    archive_path: str,
    archive_checksum: str,
    notes: str = "",
) -> int:
    """Insert a fully-pinned run record and return its id."""
    record = RunRecord(
        id=0,
        started_at=_now_iso(),
        profile_name=profile.name,
        profile_hash=profile.hash,
        profile_yaml=profile_to_yaml(profile),
        golden_hash=golden.hash,
        archive_path=archive_path,
        archive_checksum=archive_checksum,
        settings_snapshot=settings_snapshot,
        git_sha=git_sha(),
        machine_id=machine_id(),
        metrics=metrics,
        per_query=tuple(r.to_dict() for r in results),
        notes=notes,
    )
    return await store.insert_run(record)


# ── Delta + win/loss (per-query win/loss is the useful artifact) ────────


@dataclass(frozen=True)
class DeltaRow:
    """One metric's movement between two runs, per slice."""

    metric: str
    overall: float
    slices: Mapping[str, float]  # slice -> delta (signed)

    def to_dict(self) -> dict[str, object]:
        return {"metric": self.metric, "overall": self.overall, "slices": dict(self.slices)}


@dataclass(frozen=True)
class QueryDelta:
    """Per-query win/loss/unchanged vs the baseline."""

    entry_id: str
    query: str
    slice: str
    status: str  # win | loss | unchanged
    baseline_rank: int | None
    candidate_rank: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.entry_id,
            "query": self.query,
            "slice": self.slice,
            "status": self.status,
            "baseline_rank": self.baseline_rank,
            "candidate_rank": self.candidate_rank,
        }


@dataclass(frozen=True)
class Comparison:
    """A candidate run vs a baseline: metric deltas + per-query win/loss."""

    baseline: RunRecord
    candidate: RunRecord
    metric_deltas: tuple[DeltaRow, ...]
    query_deltas: tuple[QueryDelta, ...]
    wins: int
    losses: int
    unchanged: int
    degraded_guard: str | None  # non-None when a degradation mismatch needs --force

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_id": self.baseline.id,
            "candidate_id": self.candidate.id,
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
            "query_deltas": [d.to_dict() for d in self.query_deltas],
            "wins": self.wins,
            "losses": self.losses,
            "unchanged": self.unchanged,
            "degraded_guard": self.degraded_guard,
        }


def _rank_field(pq: Mapping[str, object]) -> int | None:
    raw = pq.get("hit_rank")
    if isinstance(raw, int | float):
        return int(raw)
    return None


def compare(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    force: bool = False,
) -> Comparison:
    """Diff two runs: per-slice metric movement + per-query win/loss list.

    The **degradation guard**: if one run is degraded (a component
    was capability-dropped) and the other is not, the comparison is flagged and
    normally refused — "the reranker helped" is meaningless when the reranker
    wasn't loaded. ``force=True`` overrides the guard and records the reason.
    """
    degraded_guard: str | None = None
    if baseline.metrics.degraded != candidate.metrics.degraded and not force:
        degraded_guard = (
            "degradation mismatch: baseline.degraded="
            f"{baseline.metrics.degraded} candidate.degraded={candidate.metrics.degraded} "
            "(components were capability-dropped in one run; pass --force to compare anyway)"
        )
    metric_names = ("recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr")
    deltas: list[DeltaRow] = []
    for name in metric_names:
        b_all = _slice_metric(baseline.metrics, "all", name)
        c_all = _slice_metric(candidate.metrics, "all", name)
        per_slice: dict[str, float] = {}
        for sl in (*SLICES, "all"):
            per_slice[sl] = _slice_metric(candidate.metrics, sl, name) - _slice_metric(
                baseline.metrics, sl, name
            )
        deltas.append(DeltaRow(metric=name, overall=c_all - b_all, slices=per_slice))

    # Per-query win/loss keyed by entry id.
    base_by_id = {str(pq.get("id")): pq for pq in baseline.per_query}
    wins = losses = unchanged = 0
    qdeltas: list[QueryDelta] = []
    for cq in candidate.per_query:
        qid = str(cq.get("id"))
        bq = base_by_id.get(qid)
        b_rank = _rank_field(bq) if bq else None
        c_rank = _rank_field(cq)
        # Higher = better rank (1 = best). A hit where there was none is a win.
        if not cq.get("expected_paths"):
            continue  # out_of_corpus: scored by abstention, not rank
        status = _win_loss(b_rank, c_rank)
        if status == "win":
            wins += 1
        elif status == "loss":
            losses += 1
        else:
            unchanged += 1
        qdeltas.append(
            QueryDelta(
                entry_id=qid,
                query=str(cq.get("query") or ""),
                slice=str(cq.get("slice") or ""),
                status=status,
                baseline_rank=b_rank,
                candidate_rank=c_rank,
            )
        )
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        metric_deltas=tuple(deltas),
        query_deltas=tuple(qdeltas),
        wins=wins,
        losses=losses,
        unchanged=unchanged,
        degraded_guard=degraded_guard,
    )


def _slice_metric(metrics: RunMetrics, slice_name: str, metric_name: str) -> float:
    sm = metrics.slice(slice_name)
    d = sm.to_dict()
    val = d.get(metric_name)
    return float(val) if isinstance(val, int | float) else 0.0


def _win_loss(b: int | None, c: int | None) -> str:
    """Classify a per-query rank change as win/loss/unchanged.

    Rank 1 is best; a hit where there was no hit (None→rank) is a win, and a
    lost hit (rank→None) is a loss. Equal ranks or both-miss are unchanged.
    """
    if b == c:
        return "unchanged"
    if b is None and c is not None:
        return "win"
    if c is None and b is not None:
        return "loss"
    assert b is not None and c is not None
    return "win" if c < b else "loss"


# ── Generic parameter sweep (zero per-component harness code) ───────


@dataclass(frozen=True)
class SweepPoint:
    """One sweep value + the cloned profile that carries it."""

    value: str
    profile: RetrievalProfile

    @property
    def label(self) -> str:
        return self.value


def parse_sweep(spec: str, base: RetrievalProfile) -> list[SweepPoint]:
    """Parse ``<impl>.<param>=v1,v2,...`` into cloned profiles per value.

    The sweep is generic over the registry: it finds the component in
    ``base`` whose ``impl`` matches and overrides one param, with no per-
    component code. ``rrf.k=10,20,40`` finds the fuser component with
    ``impl == "rrf"`` and produces four profiles with ``k`` ∈ {10,20,40}. A
    component may appear multiple times in a list-kind stage; the override
    applies to every match (sweeping one knob that several sources share is
    rare but well-defined).
    """
    if "=" not in spec:
        raise ValueError(f"sweep spec must be 'impl.param=v1,v2,...', got {spec!r}")
    left, _, right = spec.partition("=")
    if "." not in left:
        raise ValueError(f"sweep spec left side must be 'impl.param', got {left!r}")
    impl, _, param = left.partition(".")
    values = [v.strip() for v in right.split(",") if v.strip()]
    if not values:
        raise ValueError(f"sweep spec {spec!r}: no values")
    points: list[SweepPoint] = []
    for v in values:
        cloned = _clone_with_override(base, impl, param, v)
        points.append(SweepPoint(value=v, profile=cloned))
    return points


def _coerce_value(raw: str) -> Any:
    """Coerce a sweep value string to int/float/bool/str (best-effort)."""
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _clone_with_override(
    base: RetrievalProfile, impl: str, param: str, raw_value: str
) -> RetrievalProfile:
    """Clone ``base``, overriding ``param`` on every component whose impl matches."""
    value = _coerce_value(raw_value)

    def override_list(
        comps: tuple[ProfileComponent, ...],
    ) -> tuple[ProfileComponent, ...]:
        out: list[ProfileComponent] = []
        for c in comps:
            if c.impl == impl:
                new_params = {**c.params, param: value}
                out.append(ProfileComponent(impl=c.impl, params=new_params))
            else:
                out.append(c)
        return tuple(out)

    def override_single(c: ProfileComponent) -> ProfileComponent:
        if c.impl == impl:
            return ProfileComponent(impl=c.impl, params={**c.params, param: value})
        return c

    cloned = RetrievalProfile(
        name=base.name,
        description=base.description,
        hash="",
        preparers=override_list(base.preparers),
        sources=override_list(base.sources),
        fusion=override_single(base.fusion),
        passages=override_single(base.passages),
        scorers=override_list(base.scorers),
        assembler=override_single(base.assembler),
    )
    # Rehash from canonical YAML so the clone is content-addressed like any profile.
    yaml_text = profile_to_yaml(cloned)
    return replace(cloned, hash=_hash_yaml_local(yaml_text))


def _hash_yaml_local(yaml_text: str) -> str:
    """Recompute the profile content hash (mirrors profiles._hash_yaml)."""
    import hashlib

    import yaml

    data = yaml.safe_load(yaml_text)
    canonical = yaml.dump(data, sort_keys=True, default_flow_style=False, allow_unicode=True)
    return hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()


__all__ = [
    "Comparison",
    "DeltaRow",
    "EvalStore",
    "PipelineRunner",
    "QueryDelta",
    "QueryResult",
    "RunRecord",
    "SweepPoint",
    "compare",
    "evaluate_profile",
    "git_sha",
    "machine_id",
    "parse_sweep",
    "persist_run",
    "record_from_row",
]
