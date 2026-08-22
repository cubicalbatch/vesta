"""ONNX embedder + cross-encoder runtime.

In-process ONNX Runtime, int8, torch-free. Three model roles share this package:
``static`` (Stage B1 shortlister), ``embed`` (the indexing/query bi-encoder),
``rerank`` (Stage B2 cross-encoder).

Capabilities: on import this package registers probes that
turn on ``STATIC_ENCODER``/``CROSS_ENCODER`` when the configured model's files
are present under ``encoders.model_dir``. The composition root binds the live
:class:`~vesta.encoders.manager.EncoderManager` via :func:`bind_manager`; probes
re-evaluate on every capability computation (search.py calls
``compute_capabilities()`` per request), so they must stay a cheap filesystem
stat — never a session load (see ``manager.py``'s module docstring).

``encoders/`` depends only on ``config`` — no
``zim``, no ``retrieval``. It knows nothing about passages or archives; it only
turns text into numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vesta.config.capabilities import Capability, CapabilitySet, register_probe
from vesta.config.settings import setting
from vesta.encoders.registry import DEFAULT_MODEL

if TYPE_CHECKING:
    from vesta.encoders.manager import EncoderManager

# ── Settings ─────────────────────────────────────────────────────────────
# Model choice / session shape are ``hot=False``: they select which ONNX
# session is constructed, and the manager caches sessions for the process
# lifetime (manager.py), so changing them takes effect on restart, same as
# ``zim.read_pool_size``.

ENCODERS_MODEL_DIR = setting(
    "encoders.model_dir",
    str,
    "./data/models",
    group="Encoders",
    help="Directory holding downloaded ONNX models, laid out as "
    "<model_dir>/<repo_id>/... (mirrors the HF repo tree).",
    hot=False,
)
ENCODERS_STATIC_MODEL = setting(
    "encoders.static.model",
    str,
    DEFAULT_MODEL["static"],
    group="Encoders",
    help="Stage B1 shortlister model (repo id). Fast tier: scores ~200 "
    "candidate passages in ~50 ms. Not the same role as encoders.embed.model.",
    hot=False,
)
ENCODERS_EMBED_MODEL = setting(
    "encoders.embed.model",
    str,
    DEFAULT_MODEL["embed"],
    group="Encoders",
    help="Balanced-tier bi-encoder (repo id). Also reused for bulk "
    "indexing; here it is available for query-time use.",
    hot=False,
)
ENCODERS_RERANK_MODEL = setting(
    "encoders.rerank.model",
    str,
    DEFAULT_MODEL["rerank"],
    group="Encoders",
    help="Stage B2 cross-encoder (repo id). ms-marco-MiniLM-L-6-v2 int8 is the "
    "only model in budget on CPU; a 278M+ reranker costs 4-7s.",
    hot=False,
)
ENCODERS_INTRA_OP_THREADS = setting(
    "encoders.intra_op_threads",
    int,
    2,
    group="Encoders",
    help="ONNX Runtime intra-op thread count, from hardware measurements, not a guess.",
    min=1,
    max=32,
    hot=False,
)
ENCODERS_INDEX_INTRA_OP_THREADS = setting(
    "encoders.index_intra_op_threads",
    int,
    4,
    group="Encoders",
    help="ONNX Runtime intra-op thread count for the BULK INDEXING embedder "
    "only (the session ``get_embed_for`` builds for index.embedder). Separate "
    "from encoders.intra_op_threads, which is tuned for interactive query "
    "latency on a box that is also serving search: indexing is a throughput "
    "job, usually in a dedicated `vesta index` process, and wants the box's "
    "PHYSICAL core count. Do not exceed physical cores — embedding is pure "
    "GEMM and SMT siblings contend for the same FMA port.",
    min=1,
    max=32,
    hot=False,
)
ENCODERS_SPINNING = setting(
    "encoders.spinning",
    bool,
    False,
    group="Encoders",
    help="ONNX Runtime intra-op spin-wait. Disabled by default to reduce "
    "CPU usage and cache pressure on a local appliance.",
    hot=False,
)
ENCODERS_CPU_MEM_ARENA = setting(
    "encoders.cpu_mem_arena",
    bool,
    False,
    group="Encoders",
    help="ONNX Runtime CPU memory arena. When enabled (ONNX's default) the arena "
    "caches peak tensor allocations for the session lifetime and fragments under "
    "varying input shapes (a depth-1 index run's leads vary 30→512 tokens after "
    "padding), so RSS climbs monotonically until OOM. Disabled by default: ONNX "
    "malloc/free's each tensor so RSS tracks the live working set, at a ~5-15% "
    "inference-throughput cost that is acceptable on a single-user appliance "
    "where memory stability matters more than peak throughput.",
    hot=False,
)
ENCODERS_POOL_SIZE = setting(
    "encoders.pool_size",
    int,
    2,
    group="Encoders",
    help="Bounded concurrency for ONNX inference across all three model roles "
    "combined (two concurrent 200-passage searches is a real CPU spike "
    "on a 4-core box).",
    min=1,
    max=16,
    hot=False,
)
RETRIEVAL_RERANK_TRUNCATE_TOKENS = setting(
    "retrieval.rerank.truncate_tokens",
    int,
    256,
    group="Retrieval / Stage B",
    help="Cross-encoder input truncation length (256, not 512 — the "
    "~90-180ms latency figure assumes it).",
    min=32,
    max=512,
    hot=True,
)

#: The live manager, bound by the composition root (``main`` lifespan). A
#: module-level reference the capability probes read — matching ``zim``'s
#: ``bind_registry`` precedent (a configured singleton, not
#: a per-call import).
_MANAGER: EncoderManager | None = None


def _capability_probe() -> CapabilitySet:
    """``STATIC_ENCODER``/``CROSS_ENCODER`` are on iff the configured model's
    files are present on disk — a filesystem stat, never a session load."""
    if _MANAGER is None:
        return frozenset()
    caps: set[Capability] = set()
    if _MANAGER.static_ready():
        caps.add(Capability.STATIC_ENCODER)
    if _MANAGER.rerank_ready():
        caps.add(Capability.CROSS_ENCODER)
    return frozenset(caps)


register_probe(_capability_probe)


def bind_manager(manager: EncoderManager | None) -> None:
    """Attach (or detach, with ``None``) the live manager for the capability probes."""
    global _MANAGER
    _MANAGER = manager


def get_manager() -> EncoderManager | None:
    """The live manager, or ``None`` if the composition root hasn't bound one
    (e.g. a component running outside the FastAPI lifespan, such as a unit test)."""
    return _MANAGER


def build_manager_from_settings(
    snapshot: object, *, model_dir: Path | None = None
) -> EncoderManager:
    """Construct an :class:`~vesta.encoders.manager.EncoderManager` from a
    resolved :class:`~vesta.config.SettingsSnapshot`.

    ``model_dir`` should normally be passed explicitly by the composition root
    as ``<data.dir>/models`` — the same "derive from ``data.dir``, don't trust
    the standalone default" pattern ``main.py``/``cli.py`` already use for
    ``zims_dir`` (``zim.dir``'s default is similarly vestigial). Falls back to
    resolving ``encoders.model_dir`` from the snapshot when no override is
    given, so the function stays usable standalone (e.g. in a unit test).

    Kept out of ``main.py``'s import surface (which only needs ``bind_manager``
    conceptually) but colocated with the settings it reads, so a new
    ``encoders.*`` setting can't be added without updating the one place that
    constructs the manager.
    """
    from vesta.config.settings import SettingsSnapshot
    from vesta.encoders.manager import EncoderManager

    assert isinstance(snapshot, SettingsSnapshot)
    return EncoderManager(
        model_dir=model_dir
        if model_dir is not None
        else Path(str(snapshot.get(ENCODERS_MODEL_DIR))),
        static_model=str(snapshot.get(ENCODERS_STATIC_MODEL)),
        embed_model=str(snapshot.get(ENCODERS_EMBED_MODEL)),
        rerank_model=str(snapshot.get(ENCODERS_RERANK_MODEL)),
        intra_op_threads=int(snapshot.get(ENCODERS_INTRA_OP_THREADS)),
        index_intra_op_threads=int(snapshot.get(ENCODERS_INDEX_INTRA_OP_THREADS)),
        spinning=bool(snapshot.get(ENCODERS_SPINNING)),
        cpu_mem_arena=bool(snapshot.get(ENCODERS_CPU_MEM_ARENA)),
        pool_size=int(snapshot.get(ENCODERS_POOL_SIZE)),
        rerank_truncate_tokens=int(snapshot.get(RETRIEVAL_RERANK_TRUNCATE_TOKENS)),
    )


__all__ = [
    "ENCODERS_CPU_MEM_ARENA",
    "ENCODERS_EMBED_MODEL",
    "ENCODERS_INTRA_OP_THREADS",
    "ENCODERS_MODEL_DIR",
    "ENCODERS_POOL_SIZE",
    "ENCODERS_RERANK_MODEL",
    "ENCODERS_SPINNING",
    "ENCODERS_STATIC_MODEL",
    "RETRIEVAL_RERANK_TRUNCATE_TOKENS",
    "bind_manager",
    "build_manager_from_settings",
    "get_manager",
]
