"""Unified benchmark dataset — schema, loader, content hash.

One frozen, capability-tagged dataset replaces the four overlapping question
files (``gap_questions``, ``user_gap_questions``, ``multi_fact_questions``,
``gap_questions_spare``). Every question carries a stable slug ``id`` (never an
ordinal — retiring one must not renumber the rest), a ``sources[]`` list (the
flat ``article_title``/``article_path`` pair is gone), and per-question
``oracle``/``closed_book`` reference points.

Boundary: this module imports ONLY
``vesta.config`` + stdlib. The loader is pure data — no DB, no ZIM, no inference.
The dataset hash deliberately EXCLUDES ``oracle``/``closed_book``/``provenance``
/``tags`` so a re-verification pass does not invalidate the comparability of
pipeline runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from vesta.config.settings import setting

# An immutable empty mapping, shared safely as a dataclass default. The contract
# names it ``mapping_proxy({})``; MappingProxyType({}) is immutable, so a plain
# default (not default_factory) is correct and shares one object.
_EMPTY_MAP: Mapping[str, object] = MappingProxyType({})

# ── Settings ────────────────────────────────────────────────────────────────

BENCH_DATASET = setting(
    "bench.dataset",
    str,
    "benchmarks/vesta_bench_v2.json",
    group="Benchmark",
    help="Path to the unified benchmark dataset. Replaces the four "
    "overlapping gap/multi-fact/spare question files with one capability-tagged "
    "set whose questions carry stable slug ids.",
    hot=False,
)
BENCH_SLICE = setting(
    "bench.slice",
    str,
    "core",
    group="Benchmark",
    help="Default question slice. The benchmark is Wikipedia-only ('core' "
    "is the whole set; the legacy non-Wikipedia 'cross' slice was retired). "
    "A --slice/limit filter records both the full-set hash and a subset_hash "
    "so filtered runs are never silently compared to full runs.",
    choices=("core",),
    hot=False,
)


# ── Domain objects (frozen — the shapes shared across modules) ──────────────


@dataclass(frozen=True)
class BenchSource:
    """One expected source article for a question.

    ``required`` distinguishes a source the answer *must* come from (counted in
    source-recall/coverage) from an acceptable alternative. Multi-hop and
    comparison questions carry several required sources; coverage says "found
    them all", recall@k says "found one".
    """

    zim: str
    article_title: str
    article_path: str
    fact_location: str = ""
    required: bool = True


@dataclass(frozen=True)
class SubFact:
    """One discrete fact a compositional question must assemble.

    ``source_index`` points into the question's ``sources[]`` so a missing
    sub-fact can be attributed to a specific unretrieved article. Sub-facts are
    JUDGED (the structured rubric reports ``sub_facts_present[]``), never
    substring-matched.
    """

    fact: str
    source_index: int = 0


@dataclass(frozen=True)
class BenchQuestion:
    """One benchmark question with its ground truth + reference points."""

    id: str  # stable slug, NOT an ordinal
    question: str
    capability: str
    difficulty: str  # easy|medium|hard
    slice: str  # core
    expected_behavior: str  # answer|abstain
    answer: str
    answer_detail: str = ""
    sources: tuple[BenchSource, ...] = ()
    sub_facts: tuple[SubFact, ...] = ()
    tags: tuple[str, ...] = ()
    level: int = 3  # 1|2|3 tier (smoke / standard / release); default = deepest
    closed_book: Mapping[str, object] = _EMPTY_MAP  # {model,verdict,answer,checked_at} — the FLOOR
    oracle: Mapping[str, object] = (
        _EMPTY_MAP  # {model,verdict,answer,adjudicated_by,checked_at} — the CEILING
    )
    provenance: Mapping[str, object] = _EMPTY_MAP
    status: str = "active"  # active|quarantined|retired


@dataclass(frozen=True)
class BenchDataset:
    """A loaded dataset: questions + archive pins + content hash.

    ``hash`` is the FULL-set hash (every question, in id order); a slice/limit
    filter records it alongside a separate ``subset_hash`` so a filtered run is
    never compared to a full run without a marker.
    """

    name: str
    version: int
    questions: tuple[BenchQuestion, ...]
    archives: tuple[Mapping[str, str], ...] = ()
    generated: str = ""
    hash: str = ""  # full-set hash, set by the loader

    def __len__(self) -> int:
        return len(self.questions)


# ── Content hash ────────────────────────────────────────────────────────────


def _hash_record(q: BenchQuestion) -> str:
    """The per-question fields that define retrieval + answer identity.

    Deliberately EXCLUDES ``oracle``/``closed_book``/``provenance``/``tags``: a
    re-verification pass changes those, not the question's identity, and must
    not invalidate comparability of pipeline runs. Source
    article_paths and sub-fact texts are sorted so source/sub_fact ORDER (a
    presentation detail) does not perturb the hash.
    """
    paths = ",".join(sorted(s.article_path for s in q.sources))
    facts = ",".join(sorted(sf.fact for sf in q.sub_facts))
    return f"{q.id}\x1f{q.question}\x1f{q.answer}\x1f{q.expected_behavior}\x1f{paths}\x1f{facts}"


def _compute_hash(questions: Sequence[BenchQuestion]) -> str:
    """sha256 over per-question identity fields in id order, truncated to 16 hex."""
    h = hashlib.sha256(usedforsecurity=False)
    for q in sorted(questions, key=lambda q: q.id):
        h.update(_hash_record(q).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def dataset_hash(questions: Sequence[BenchQuestion]) -> str:
    """Content hash of a question set (the full set). Stable; GT-edit-insensitive."""
    return _compute_hash(questions)


def subset_hash(questions: Sequence[BenchQuestion]) -> str:
    """Content hash of a filtered subset (slice/limit). Same algorithm, fewer rows.

    Recorded alongside the full-set hash on every filtered run so two runs over
    different subsets are never compared as equal.
    """
    return _compute_hash(questions)


# ── Filtering ───────────────────────────────────────────────────────────────


def filter(
    questions: Sequence[BenchQuestion],
    *,
    slice: str | None = None,
    capabilities: Sequence[str] = (),
    difficulties: Sequence[str] = (),
    level: int | None = None,
) -> tuple[BenchQuestion, ...]:
    """Select questions by slice / capability / difficulty / level tier.

    ``level`` is cumulative: ``level=L`` keeps every question with
    ``q.level <= L`` (a higher tier is a superset, so runs on the same
    tier are directly comparable and a level-1 smoke tests every capability).

    Returns a tuple (frozen shape). Status filtering (active/quarantined/retired)
    is the runner's concern — the dataset retains every status so a quarantined
    question keeps its history; only ``active`` questions are scored by default.
    """
    out: list[BenchQuestion] = list(questions)
    if slice is not None:
        out = [q for q in out if q.slice == slice]
    if capabilities:
        wanted = set(capabilities)
        out = [q for q in out if q.capability in wanted]
    if difficulties:
        wanted = set(difficulties)
        out = [q for q in out if q.difficulty in wanted]
    if level is not None:
        out = [q for q in out if q.level <= level]
    return tuple(out)


# ── Loader ──────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "question",
    "capability",
    "difficulty",
    "slice",
    "expected_behavior",
    "answer",
)
_VALID_BEHAVIORS = frozenset({"answer", "abstain"})
_VALID_STATUS = frozenset({"active", "quarantined", "retired"})


def _src(d: Mapping[str, object], key: str, ctx: str) -> str:
    """Fetch a required string field, naming the offending slug on failure."""
    if key not in d:
        raise ValueError(f"{ctx}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str):
        raise ValueError(f"{ctx}: field {key!r} must be a string")
    return v


def _parse_source(s: object, slug: str, idx: int) -> BenchSource:
    if not isinstance(s, Mapping):
        raise ValueError(f"question {slug!r}: source[{idx}] must be an object")
    ctx = f"question {slug!r} source[{idx}]"
    return BenchSource(
        zim=_src(s, "zim", ctx),
        article_title=_src(s, "article_title", ctx),
        article_path=_src(s, "article_path", ctx),
        fact_location=str(s.get("fact_location") or ""),
        required=bool(s.get("required", True)),
    )


def _parse_question(q: object) -> BenchQuestion:
    if not isinstance(q, Mapping):
        raise ValueError("question entries must be objects")
    # The slug is needed before any other validation so every error names it.
    slug = str(q.get("id") or "")
    if not slug:
        raise ValueError("question missing required field 'id' (stable slug)")
    ctx = f"question {slug!r}"
    fields = {k: _src(q, k, ctx) for k in _REQUIRED_FIELDS}
    if fields["expected_behavior"] not in _VALID_BEHAVIORS:
        raise ValueError(
            f"{ctx}: expected_behavior must be one of "
            f"{sorted(_VALID_BEHAVIORS)}, got {fields['expected_behavior']!r}"
        )
    status = str(q.get("status") or "active")
    if status not in _VALID_STATUS:
        raise ValueError(f"{ctx}: status must be one of {sorted(_VALID_STATUS)}, got {status!r}")
    raw_level = q.get("level", 3)
    try:
        level = int(raw_level)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ctx}: level must be an int 1..3, got {raw_level!r}") from exc
    if level not in (1, 2, 3):
        raise ValueError(f"{ctx}: level must be 1, 2, or 3, got {level}")
    sources_raw = q.get("sources") or []
    if not isinstance(sources_raw, list):
        raise ValueError(f"{ctx}: 'sources' must be a list")
    sources = tuple(_parse_source(s, slug, i) for i, s in enumerate(sources_raw))
    # An answerable question needs at least one source to retrieve from; an
    # abstain (out_of_corpus) question has no gold source by construction.
    if fields["expected_behavior"] == "answer" and not sources:
        raise ValueError(f"{ctx}: expected_behavior 'answer' requires >=1 source")

    sub_facts_raw = q.get("sub_facts") or []
    if not isinstance(sub_facts_raw, list):
        raise ValueError(f"{ctx}: 'sub_facts' must be a list")
    sub_facts: list[SubFact] = []
    for sf in sub_facts_raw:
        if not isinstance(sf, Mapping) or not sf.get("fact"):
            continue
        sub_facts.append(
            SubFact(fact=str(sf["fact"]), source_index=int(sf.get("source_index") or 0))
        )

    tags_raw = q.get("tags") or []
    tags = tuple(str(t) for t in tags_raw) if isinstance(tags_raw, list) else ()

    return BenchQuestion(
        id=fields["id"],
        question=fields["question"],
        capability=fields["capability"],
        difficulty=fields["difficulty"],
        slice=fields["slice"],
        expected_behavior=fields["expected_behavior"],
        answer=fields["answer"],
        answer_detail=str(q.get("answer_detail") or ""),
        sources=sources,
        sub_facts=tuple(sub_facts),
        tags=tags,
        level=level,
        closed_book=_as_mapping(q.get("closed_book")),
        oracle=_as_mapping(q.get("oracle")),
        provenance=_as_mapping(q.get("provenance")),
        status=status,
    )


def _as_mapping(v: object) -> Mapping[str, object]:
    """Coerce a JSON object to an immutable mapping (empty when absent/non-dict)."""
    if isinstance(v, Mapping):
        return MappingProxyType(dict(v))
    return MappingProxyType({})


def load_bench_dataset(path: str | Path = str(BENCH_DATASET.default)) -> BenchDataset:
    """Load the benchmark JSON into frozen dataclasses + content-hash it.

    TOLERANT of extra/unknown fields (the dataset carries ``tags``,
    ``provenance``, audit fields — kept, not scored). STRICT about required ones;
    every validation error names the offending question slug.
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top-level JSON must be an object")
    name = str(raw.get("name") or p.stem)
    version = int(raw.get("version") or 1)

    archives_raw = raw.get("archives") or []
    archives: tuple[Mapping[str, str], ...] = ()
    if isinstance(archives_raw, list):
        archives = tuple(
            {k: str(v) for k, v in a.items()} if isinstance(a, Mapping) else {}
            for a in archives_raw
        )

    questions_raw = raw.get("questions") or []
    if not isinstance(questions_raw, list):
        raise ValueError(f"{path}: 'questions' must be a list")
    questions = tuple(_parse_question(q) for q in questions_raw)

    # Detect duplicate ids — a slug collision silently merges per-question
    # history, the same hazard as ordinals.
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id!r} in {path}")
        seen.add(q.id)

    ds = BenchDataset(
        name=name,
        version=version,
        questions=questions,
        archives=archives,
        generated=str(raw.get("generated") or ""),
    )
    return replace(ds, hash=dataset_hash(ds.questions))


__all__ = [
    "BENCH_DATASET",
    "BENCH_SLICE",
    "BenchDataset",
    "BenchQuestion",
    "BenchSource",
    "SubFact",
    "dataset_hash",
    "filter",
    "load_bench_dataset",
    "subset_hash",
]
