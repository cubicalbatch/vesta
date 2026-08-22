"""Retrieval package root.

This package must remain dependency-free of ``answer``/``api``/``index`` —
enforced by ``tests/test_boundaries.py``. It depends on ``zim``, ``config``, and
``encoders``.

Imports here register components via ``impls/``, ``scorers/`` and
``assemblers/`` so that ``resolve(kind, name)`` finds them. Built-in profiles
are loaded at the same time from the YAML files in ``profiles/``.
"""

from __future__ import annotations

from vesta.config.settings import setting

# Re-export the trace types.
from vesta.retrieval.trace import TRACE_VERSION, DegradationRecord, StageCtx, Trace

# ── Retrieval settings ──────────────────────────────────────────────────────

RETRIEVAL_ACTIVE_PROFILE = setting(
    "retrieval.active_profile",
    str,
    "hybrid",
    group="Retrieval / General",
    help="Default retrieval profile. ?profile= query param overrides this "
    "(gated by api.allow_profile_override). Built-in: hybrid, standard, "
    "lexical. ``hybrid`` is the default: ``standard``'s lexical "
    "Stage A + two-pass Stage B scoring (static shortlist + cross-encoder "
    "rerank) + lead_boost assembly, plus the dense ``vector_knn`` source. "
    "With ``VECTORS`` unmet (depth-0 box) ``vector_knn`` is capability-dropped "
    "and ``hybrid`` is byte-equivalent to ``standard``. ``standard`` is the "
    "pure lexical+Stage-B profile; ``lexical`` is the depth-0 baseline and "
    "the fallback for unknown profile names. All degrade gracefully.",
    hot=True,
)
RETRIEVAL_PROFILES = setting(
    "retrieval.profiles",
    str,
    "{}",
    group="Retrieval / General",
    help="User-defined retrieval profiles as JSON blob. Cloned from built-ins "
    "with param overrides (the UI edits this).",
    hot=True,
)
RETRIEVAL_MAX_ARCHIVES_CONCURRENT = setting(
    "retrieval.max_archives_concurrent",
    int,
    8,
    group="Retrieval / Concurrency",
    help="Semaphore bound for concurrent archive searches. "
    "Prevents libzim pool exhaustion on multi-archive corpora.",
    min=1,
    max=64,
    hot=True,
)
RETRIEVAL_CANDIDATES_MAX_ARTICLES = setting(
    "retrieval.candidates.max_articles",
    int,
    20,
    group="Retrieval / Candidates",
    help="Maximum articles read and split for passage generation. Caps the "
    "dominant Stage B latency cost.",
    min=1,
    max=100,
    hot=True,
)
RETRIEVAL_CONTEXT_BUDGET_TOKENS = setting(
    "retrieval.context.budget_tokens",
    int,
    12000,
    group="Retrieval / Context",
    help="Evidence-token budget for the answer prompt. This is a latency "
    "budget on CPU: ~4 s of prefill per 1000 tokens. "
    "Raised from 2400: at 2400, a strongly-matching but long "
    "article (multiple candidate passages) could only ever contribute 2 "
    "passages total, which is not enough headroom for the article's actual "
    "answer to reliably be among them.",
    min=256,
    max=32000,
    hot=True,
)
API_ALLOW_PROFILE_OVERRIDE = setting(
    "api.allow_profile_override",
    bool,
    True,
    group="API",
    help="Allow ?profile= query param to override the active retrieval profile "
    "(dev console and eval harness use this; disable for production).",
    hot=True,
)
RETRIEVAL_CONFIDENCE_TOP_SCORE = setting(
    "retrieval.confidence.top_score",
    float,
    0.3,
    group="Retrieval / Confidence",
    help="Minimum top passage score before the abstention gate will consider "
    "answering (abstention gate). Calibrated against the golden set.",
    min=0.0,
    max=1.0,
    hot=True,
)
RETRIEVAL_CONFIDENCE_SCORE_DROPOFF = setting(
    "retrieval.confidence.score_dropoff",
    float,
    0.5,
    group="Retrieval / Confidence",
    help="Maximum score dropoff ratio before the abstention gate triggers "
    "(abstention gate). A low ratio means the top passage stands alone (good signal).",
    min=0.0,
    max=1.0,
    hot=True,
)
RETRIEVAL_CONFIDENCE_DENSITY = setting(
    "retrieval.confidence.density",
    float,
    0.5,
    group="Retrieval / Confidence",
    help="Minimum density (fraction of passages from top article) below which "
    "abstention is considered (abstention gate). Calibrated against the golden set.",
    min=0.0,
    max=1.0,
    hot=True,
)
RETRIEVAL_STAGE_B_CANDIDATES_MAX = setting(
    "retrieval.stage_b.candidates_max",
    int,
    200,
    group="Retrieval / Stage B",
    help="Defensive cap on passages entering Stage B1's static-embedder pass. "
    "Mirrored as static_pass.Params.candidates_max in the standard profile.",
    min=1,
    max=2000,
    hot=True,
)
RETRIEVAL_STAGE_B_SHORTLIST = setting(
    "retrieval.stage_b.shortlist",
    int,
    20,
    group="Retrieval / Stage B",
    help="Stage B1's output shortlist size, fed to Stage B2. Mirrored "
    "as static_pass.Params.shortlist in the standard profile.",
    min=1,
    max=200,
    hot=True,
)
RETRIEVAL_RERANK_ENABLED = setting(
    "retrieval.rerank.enabled",
    bool,
    True,
    group="Retrieval / Stage B",
    help="Stage B2 cross-encoder rerank A/B toggle. "
    "Ships on by default; if "
    "the golden-set A/B measures negative, a rerank-disabled profile becomes "
    "the default and this flips. Mirrored as cross_encoder.Params.enabled.",
    hot=True,
)
RETRIEVAL_CONTEXT_MAX_PER_ARTICLE = setting(
    "retrieval.context.max_per_article",
    int,
    4,
    group="Retrieval / Context",
    help="Default per-article passage cap for context assembly. Mirrored as "
    "each assembler's Params.max_per_article in the built-in profiles. "
    "Raised from 2 alongside the budget_tokens increase — see "
    "that setting's help text.",
    min=1,
    max=50,
    hot=True,
)
RETRIEVAL_CONTEXT_ORDERING = setting(
    "retrieval.context.ordering",
    str,
    "score_desc",
    group="Retrieval / Context",
    help="Where the strongest evidence lands in the assembled prompt. "
    "'score_desc': rank order. 'edges': strongest "
    "passages at both ends, weakest in the middle. Mirrored as every "
    "assembler's Params.ordering.",
    choices=("score_desc", "edges"),
    hot=True,
)
RETRIEVAL_CONFIDENCE_AGREEMENT = setting(
    "retrieval.confidence.agreement",
    float,
    0.0,
    group="Retrieval / Confidence",
    help="Minimum agreement (Jaccard overlap between source candidate sets) "
    "below which abstention is considered.",
    min=0.0,
    max=1.0,
    hot=True,
)
RETRIEVAL_DENSE_K = setting(
    "retrieval.dense.k",
    int,
    40,
    group="Retrieval / Dense",
    help="Top-k dense (vector kNN) candidates the Stage A3 ``vector_knn`` source "
    "returns per query. Same order of magnitude as the lexical "
    "sources' limit so RRF fusion is balanced. Mirrored as vector_knn.Params.k.",
    min=1,
    max=200,
    hot=True,
)
RETRIEVAL_DENSE_ENABLED = setting(
    "retrieval.dense.enabled",
    bool,
    True,
    group="Retrieval / Dense",
    help="Stage A3 dense source A/B toggle (mirrors retrieval.rerank.enabled's "
    "shape). Ships on by default; on a depth-0 box the VECTORS capability is "
    "unmet so the profile drops vector_knn regardless (degrade-don't-fail).",
    hot=True,
)

# Imports deferred to avoid circular imports — settings must be registered
# before impls/scorers/assemblers and profiles are loaded (which reference the
# registry). Order matters: profiles.py validates every component reference
# against the registry at import time, so it must load LAST. Using
# ``import_module`` (not plain ``import`` statements) is deliberate — isort/
# ruff's I001 alphabetizes contiguous import blocks, which would silently
# reorder ``profiles`` before ``scorers``/``assemblers`` and break this.
from importlib import import_module as _import_module  # noqa: E402

for _module_name in (
    "vesta.retrieval.assemblers",
    "vesta.retrieval.impls",
    "vesta.retrieval.scorers",
    "vesta.retrieval.profiles",
):
    _import_module(_module_name)

__all__ = [
    "API_ALLOW_PROFILE_OVERRIDE",
    "RETRIEVAL_ACTIVE_PROFILE",
    "RETRIEVAL_CANDIDATES_MAX_ARTICLES",
    "RETRIEVAL_CONFIDENCE_AGREEMENT",
    "RETRIEVAL_CONFIDENCE_DENSITY",
    "RETRIEVAL_CONFIDENCE_SCORE_DROPOFF",
    "RETRIEVAL_CONFIDENCE_TOP_SCORE",
    "RETRIEVAL_CONTEXT_BUDGET_TOKENS",
    "RETRIEVAL_CONTEXT_MAX_PER_ARTICLE",
    "RETRIEVAL_CONTEXT_ORDERING",
    "RETRIEVAL_DENSE_ENABLED",
    "RETRIEVAL_DENSE_K",
    "RETRIEVAL_MAX_ARCHIVES_CONCURRENT",
    "RETRIEVAL_PROFILES",
    "RETRIEVAL_RERANK_ENABLED",
    "RETRIEVAL_STAGE_B_CANDIDATES_MAX",
    "RETRIEVAL_STAGE_B_SHORTLIST",
    "TRACE_VERSION",
    "DegradationRecord",
    "StageCtx",
    "Trace",
]
