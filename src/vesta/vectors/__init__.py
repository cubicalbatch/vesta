"""Vector store package — interface, sqlite-vec implementation, and the
composition-root binding point.

Three things live here:

* **Settings** (``vectors.*``): declared at import so ``GET /api/settings/schema``
  surfaces them with no frontend change. All ``hot=False``:
  they describe the storage/session shape (which vec0 DDL variant, which table
  layout), and the store is constructed once in the lifespan, so a change takes
  effect on restart — the same precedent as ``zim.read_pool_size`` and
  ``encoders.intra_op_threads``.
* **The store singleton** (``_STORE``): bound by ``main``'s lifespan via
  :func:`bind_store`, read by the dense candidate source via
  :func:`get_store`. Mirrors the ``encoders``/``inference`` ``bind_*`` precedent
  (a configured singleton, not a per-call import).
* **NOT here**: the ``Capability.VECTORS`` probe. "At least one archive at depth
  ≥1" is *index state*, so ``index/__init__.py`` owns the probe — it
  can stat the ``zims.index_depth``/``index_status`` columns, whereas this
  package knows only that a *store* exists, not that anything has been indexed.

``vectors/`` depends only on ``db`` + ``config`` (module map).
It must not import ``retrieval``/``answer``/``api``/``index``; the dense source
receives the store through DI (:func:`get_store`), never by importing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vesta.config.settings import setting

if TYPE_CHECKING:
    from vesta.vectors.contracts import VectorStore

# ── Settings ─────────────────────────────────────────────────────────────
# These describe the vec0 table layout. ``hot=False``: the store is built once
# at startup (main lifespan) and vec0 tables are created then, so a runtime
# change cannot retrofit an existing table — restart to apply (same shape as
# zim.read_pool_size / encoders.intra_op_threads).

VECTORS_BACKEND = setting(
    "vectors.backend",
    str,
    "sqlite_vec",
    group="Vectors",
    help="Vector store backend id. Only 'sqlite_vec' is currently supported.",
    hot=False,
)
VECTORS_QUANTIZER = setting(
    "vectors.quantizer",
    str,
    "bit",
    group="Vectors",
    help="vec0 rescore quantizer ('bit' | 'int8'). 'bit' is the load-bearing "
    "default: it keeps the working set resident past 10 M vectors, where flat f32 "
    "spills out of page cache. Applied when the vec0 build supports the rescore table option; "
    "stock wheels fall back to a flat index.",
    hot=False,
)
VECTORS_OVERSAMPLE = setting(
    "vectors.oversample",
    int,
    8,
    group="Vectors",
    help="vec0 rescore oversample factor. k x oversample bit-quantized candidates "
    "are scanned, then the top-k are re-ranked from full f32 vectors. "
    "Default 8 buys ~0.988 recall@10 with significant speedup.",
    min=1,
    max=64,
    hot=False,
)

#: The live store, bound by the composition root (``main`` lifespan). A
#: module-level reference the dense source reads — matching ``encoders``'s
#: ``bind_manager`` and ``inference``'s ``bind_gateway`` precedent (a
#: configured singleton, not a per-call import).
_STORE: VectorStore | None = None


def bind_store(store: VectorStore | None) -> None:
    """Attach (or detach, with ``None``) the live store. The dense candidate
    source reads it via :func:`get_store`."""
    global _STORE
    _STORE = store


def get_store() -> VectorStore | None:
    """The live store, or ``None`` if the composition root hasn't bound one (e.g.
    vec0 unavailable on this build, or a component running outside the FastAPI
    lifespan such as a unit test)."""
    return _STORE


__all__ = [
    "VECTORS_BACKEND",
    "VECTORS_OVERSAMPLE",
    "VECTORS_QUANTIZER",
    "bind_store",
    "get_store",
]
