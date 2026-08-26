"""Golden set loading and validation.

The golden set is the ground truth every retrieval change is measured against.
Two honesty rules make it trustworthy rather than aspirational:

* **Every expected path is verified present in the pinned archive** before the
  set ships. A 60-query set with verified paths beats 150 aspirational ones.
  ``verify_against_archive`` performs this check; the CLI exposes it
  (``vesta eval verify-golden``) and a unit test asserts it for the fixture set.
* **Provenance is recorded** (hand-written vs derived) so retrieval bias is
  auditable: queries written *after* seeing results bias toward what works.

The full set targets the pinned Wikipedia archive
(``wikipedia_en_top_nopic_2026-06.zim``); the ``fixture_subset`` targets the
tiny fixture ZIM and is what the CI regression gate actually runs (the pinned
archive is gitignored, so the full gate is on-demand/nightly only).

``eval/`` imports only ``retrieval`` and ``config`` (the ≤2
dependency cap). Archive access is therefore *injected*: callers hand
``verify_against_archive`` a callable that resolves a path to extracted text, so
the loader never imports ``zim`` or ``db``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vesta.config.settings import setting

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

#: The loadable set names. Anything else is a typo (e.g. ``"fixture_subsets"``)
#: and must be rejected, not silently expanded to the full pinned-archive run.
GOLDEN_SET_NAMES: tuple[str, ...] = ("full", "fixture_subset")

#: The slices. ``out_of_corpus`` is the abstention slice.
#: ``reformulation`` is a seventh slice:
#: direct terms fail, a synonym / broader term / successor title succeeds.
SLICES: tuple[str, ...] = (
    "entity",
    "paraphrase",
    "multi_hop",
    "deep_content",
    "out_of_corpus",
    "keyword",
    "reformulation",
)


@dataclass(frozen=True)
class GoldenEntry:
    """One golden query with its ground truth.

    ``expected_paths`` is the set of acceptable article paths (any one present
    in the retrieved candidates counts as a hit; empty for ``out_of_corpus``).
    ``expected_fact`` is a short string verified present in the article text —
    it documents *what* the query is after and lets the harness sanity-check
    that a hit is a real one, not a path collision.
    """

    id: str
    query: str
    slice: str
    expected_paths: tuple[str, ...]
    expected_fact: str
    provenance: str
    notes: str = ""


@dataclass(frozen=True)
class GoldenSet:
    """A loaded golden set: archive pin + entries, content-hashed for runs."""

    name: str
    archive_path: str
    archive_checksum: str
    entries: tuple[GoldenEntry, ...]
    hash: str = ""

    def by_slice(self) -> dict[str, list[GoldenEntry]]:
        """Entries grouped by slice (missing slices default to empty)."""
        out: dict[str, list[GoldenEntry]] = {s: [] for s in SLICES}
        for e in self.entries:
            out.setdefault(e.slice, []).append(e)
        return out


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _entry_from_dict(raw: dict[str, Any], default_slice: str) -> GoldenEntry:
    paths = _coerce_str_tuple(raw.get("expected_paths"))
    return GoldenEntry(
        id=str(raw["id"]),
        query=str(raw["query"]),
        slice=str(raw.get("slice") or default_slice),
        expected_paths=paths,
        expected_fact=str(raw.get("expected_fact") or ""),
        provenance=str(raw.get("provenance") or "hand-written"),
        notes=str(raw.get("notes") or ""),
    )


def load_slice(path: Path) -> tuple[str, list[GoldenEntry]]:
    """Load one slice YAML. The slice name comes from the ``name`` key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"golden slice {path}: YAML must map to a dict")
    name = str(data.get("name") or path.stem)
    entries_raw = data.get("entries") or []
    if not isinstance(entries_raw, list):
        raise ValueError(f"golden slice {path}: 'entries' must be a list")
    entries = [_entry_from_dict(e, default_slice=name) for e in entries_raw]
    return name, entries


# ── Archive pin (eval config) ────────────────────────────────────────────────
# These two settings pin the eval reference archive: results are meaningless
# across archive versions. The
# filename AND the sha256 are pinned so a re-release cannot masquerade as the
# same set.

EVAL_ARCHIVE_PATH = setting(
    "eval.archive.path",
    str,
    "wikipedia_en_top_nopic_2026-06.zim",
    group="Eval / Golden set",
    help="Filename of the pinned reference ZIM the full golden set runs against. "
    "The ~2.1 GiB wikipedia_en_top_nopic_2026-06 archive; results are pinned to it.",
    hot=False,
)
EVAL_ARCHIVE_CHECKSUM = setting(
    "eval.archive.checksum",
    str,
    "b2806831e14690cbcafeb1b6e7bd4439fd59b3e5fbeaeb300a5792dece510ee0",
    group="Eval / Golden set",
    help="sha256 of the pinned eval archive. A mismatch means the reference set "
    "is stale and every recorded number is no longer comparable.",
    hot=False,
)

EVAL_JUDGE_ENDPOINT_URL = setting(
    "eval.judge.endpoint_url",
    str,
    "https://bifrost.loki.onoz.cc/v1",
    group="Judge Inference / LLM",
    help="OpenAI-compatible base URL for the judge LLM (e.g. "
    "'http://host:1234/v1' or 'https://api.openai.com/v1'). Empty = reuse the "
    "main inference gateway (inference.llm.*) for judging.",
    hot=True,
)
EVAL_JUDGE_API_KEY = setting(
    "eval.judge.api_key",
    str,
    "",
    group="Judge Inference / LLM",
    help="API key for the judge endpoint. Empty for local or unauthenticated "
    "endpoints. Leave blank or unchanged when saving to keep the stored key.",
    hot=True,
    secret=True,
)
EVAL_JUDGE_MODEL = setting(
    "eval.judge.model",
    str,
    "cline/cline-pass/deepseek-v4-flash",
    group="Judge Inference / LLM",
    help="Judge model id (OpenAI-compatible) used as the LLM-judge for the "
    "answer benchmark + answer metrics. Empty = no judge; lexical-only "
    "scoring, the run is inconclusive.",
    hot=True,
)
EVAL_REGRESSION_EPSILON = setting(
    "eval.regression.epsilon",
    float,
    0.02,
    group="Eval / Regression gate",
    help="Max allowed drop in recall@10 vs the recorded baseline before the "
    "regression gate fails a run. A change that drops the metric more "
    "than this does not ship.",
    min=0.0,
    max=1.0,
    hot=True,
)


def _content_hash(entries: Sequence[GoldenEntry], archive_checksum: str) -> str:
    """Stable hash of the set's content (queries+paths+facts) + archive pin.

    Recorded with every run so a drift in the golden set is detectable: two runs
    with the same ``golden_hash`` are directly comparable on retrieval metrics.
    """
    import hashlib

    h = hashlib.sha256(usedforsecurity=False)
    h.update(archive_checksum.encode("utf-8"))
    for e in entries:
        h.update(b"|")
        h.update(e.id.encode("utf-8"))
        h.update(e.query.encode("utf-8"))
        h.update(e.slice.encode("utf-8"))
        h.update(";".join(e.expected_paths).encode("utf-8"))
        h.update(e.expected_fact.encode("utf-8"))
    return h.hexdigest()[:16]


def load_full_set() -> GoldenSet:
    """Load the 6-slice full golden set (the pinned-archive set).

    Concatenates every slice YAML under ``golden/`` except ``fixture_subset``.
    The archive pin comes from the resolved eval settings.
    """
    entries: list[GoldenEntry] = []
    for path in sorted(GOLDEN_DIR.glob("*.yaml")):
        if path.stem == "fixture_subset":
            continue
        _, slice_entries = load_slice(path)
        entries.extend(slice_entries)
    archive_path = str(EVAL_ARCHIVE_PATH.default)
    archive_checksum = str(EVAL_ARCHIVE_CHECKSUM.default)
    gs = GoldenSet(
        name="full",
        archive_path=archive_path,
        archive_checksum=archive_checksum,
        entries=tuple(entries),
    )
    from dataclasses import replace

    return replace(gs, hash=_content_hash(gs.entries, archive_checksum))


def load_fixture_set() -> GoldenSet:
    """Load the CI-runnable fixture subset (targets the tiny ZIM)."""
    path = GOLDEN_DIR / "fixture_subset.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    archive = data.get("archive") or {}
    entries_raw = data.get("entries") or []
    entries = tuple(_entry_from_dict(e, default_slice="entity") for e in entries_raw)
    gs = GoldenSet(
        name="fixture_subset",
        archive_path=str(archive.get("path", "fixture")),
        archive_checksum=str(archive.get("checksum", "fixture")),
        entries=entries,
    )
    from dataclasses import replace

    return replace(gs, hash=_content_hash(gs.entries, gs.archive_checksum))


def load_set(name: str = "full") -> GoldenSet:
    """Load a named golden set: ``full`` (default) or ``fixture_subset``."""
    if name not in GOLDEN_SET_NAMES:
        known = ", ".join(repr(n) for n in GOLDEN_SET_NAMES)
        raise ValueError(f"unknown golden set {name!r}; valid sets: {known}")
    if name == "fixture_subset":
        return load_fixture_set()
    return load_full_set()


# ── Verification (injected archive access — eval never imports zim) ──────────

#: A resolver from an article path to its extracted text (or ``None`` if the path
#: does not resolve). The CLI/API wires this to the open archive registry; tests
#: wire it to a fake. Defined here so the loader stays free of ``zim``/``db``
#: imports (eval depends on retrieval + config only).
TextResolver = Callable[[str], str | None]


def verify_against_archive(gs: GoldenSet, resolve_text: TextResolver) -> list[str]:
    """Confirm every expected path resolves and every fact is present.

    Returns a list of human-readable failures (empty = fully verified). This is
    the honesty check: a golden set whose paths don't resolve is
    measuring noise. ``out_of_corpus`` entries have no expected path and are
    skipped. The resolver is injected so this module stays archive-agnostic.
    """
    failures: list[str] = []
    for e in gs.entries:
        if not e.expected_paths:
            continue
        texts: list[str] = []
        missing: list[str] = []
        for p in e.expected_paths:
            t = resolve_text(p)
            if t is None:
                missing.append(p)
            else:
                texts.append(t)
        if missing and not texts:
            failures.append(f"{e.id}: no expected path resolves ({missing})")
            continue
        if missing:
            failures.append(f"{e.id}: some expected paths missing ({missing})")
        if (
            e.expected_fact
            and texts
            and not any(e.expected_fact.lower() in t.lower() for t in texts)
        ):
            failures.append(f"{e.id}: fact {e.expected_fact!r} not in any article")
    return failures


__all__ = [
    "EVAL_ARCHIVE_CHECKSUM",
    "EVAL_ARCHIVE_PATH",
    "EVAL_JUDGE_MODEL",
    "EVAL_REGRESSION_EPSILON",
    "GOLDEN_DIR",
    "GOLDEN_SET_NAMES",
    "SLICES",
    "GoldenEntry",
    "GoldenSet",
    "TextResolver",
    "load_fixture_set",
    "load_full_set",
    "load_set",
    "load_slice",
    "verify_against_archive",
]
