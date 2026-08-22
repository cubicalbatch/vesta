#!/usr/bin/env python
"""Per-candidate latency harness for the ``candidate_articles`` stage.

DIAGNOSTIC ONLY.
This script reproduces the stage-latency evidence table (one
7-document PDF-only ZIM was 99.2 % of the stage's wall time). It is NOT product code:

- It **monkeypatches** ``CandidateArticles.build``, ``LocalArchive.extract``
  and ``zim.passages.split_passages`` at startup, then hands argv to
  ``vesta.cli.main`` unchanged.
- **No production code imports it.** Nothing under ``src/vesta`` knows this
  file exists; it lives in ``scripts/`` precisely so it can never sit on a
  product import path (same stance as ``scripts/bench_vector_scale.py``).
- Its timing overhead is ~7 ms per 40 candidates (the ``run_in_executor``
  hop) — below the table's own 0.1 ms row resolution,
  so the numbers it prints are the pipeline's, not the harness's.

Usage — identical arguments to ``vesta bench run``, prefixed by the literal
``bench run`` subcommand (anything else fails fast):

    uv run python scripts/bench_stage_latency.py bench run \
        --system retrieval_only --profile hybrid --level 3 \
        --limit 3 --repeats 1 --concurrency 1 --no-persist

Output (stderr, so bench's own report stays clean on stdout):

- one ``### build() total=… n_cands=… passages=…`` line per pipeline call;
- at exit, the process-wide per-candidate table (sorted by extract ms, desc)
  and the per-archive rollup (sorted by summed stage ms, desc).
"""

from __future__ import annotations

import atexit
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vesta.retrieval.contracts import Candidate, PreparedQuery
    from vesta.retrieval.impls.candidate_articles import CandidateArticles
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import LocalArchive
    from vesta.zim.types import EntryPath, ExtractedArticle, Passage

#: Per-candidate rows, accumulated for the whole process:
#: ``[zim_id, path, extract_ms, split_ms, chars]``. ``split_ms`` starts at
#: 0.0 and is filled in by the split wrapper when build() reaches it;
#: empty-body / title-fallback candidates never call ``split_passages`` and
#: keep 0.0.
_ROWS: list[list[Any]] = []

#: ``id(article)`` → index into ``_ROWS`` for the extract whose split has not
#: run yet. Keyed by ``id()`` and cleared at each build boundary, so a CPython
#: id reused after garbage collection can never attach a split time to a row
#: from an earlier build.
_PENDING: dict[int, int] = {}

#: > 0 while an instrumented ``CandidateArticles.build`` is on the stack.
#: Extracts by every other caller (oracle context, reader API, golden-set
#: arms) pass through timed-but-unrecorded — only the candidate loop is
#: recorded.
_IN_BUILD = 0

#: Row format of the report table; the header below is the same fields.
_ROW_FMT = "%4d %11.1f %9.1f %8d  %s"


def _short(path: str, width: int = 52) -> str:
    """Truncate long entry paths to one table line."""
    return path if len(path) <= width else f"{path[: width - 3]}..."


def _print_report() -> None:
    """The exit-time table + per-archive rollup."""
    if not _ROWS:
        return
    print(
        f"\n{'zim':>4} {'extract ms':>11} {'split ms':>9} {'chars':>8}  path",
        file=sys.stderr,
        flush=True,
    )
    for zim_id, path, extract_ms, split_ms, chars in sorted(_ROWS, key=lambda r: -r[2]):
        print(
            _ROW_FMT % (zim_id, extract_ms, split_ms, chars, _short(str(path))),
            file=sys.stderr,
            flush=True,
        )
    per_archive: dict[int, list[float]] = {}
    for zim_id, _path, extract_ms, split_ms, _chars in _ROWS:
        per_archive.setdefault(zim_id, []).append(extract_ms + split_ms)
    cells = [
        f"zim{zim_id}:n={len(ms)},sum={sum(ms):.0f}ms"
        for zim_id, ms in sorted(per_archive.items(), key=lambda kv: -sum(kv[1]))
    ]
    print(" per-archive: " + "  ".join(cells[:3]), file=sys.stderr, flush=True)
    for i in range(3, len(cells), 3):
        print(" " * 14 + "  ".join(cells[i : i + 3]), file=sys.stderr, flush=True)


def _install_patches() -> None:
    """Wrap the three symbols the candidate loop calls, then delegate.

    This delegates to the *real* ``build()`` instead of reimplementing it
    with timers inline (the /tmp scratchpad this replaces did the latter and
    had already drifted from the live method — it dropped the
    ``title_fallback`` branch). Timing the called symbols keeps the real
    loop authoritative and this harness drift-free.

    ``split_passages`` timing choice: ``build()`` imports it *inside the
    method body* (first line of ``build`` in ``candidate_articles.py``) and
    re-binds from ``vesta.zim.passages`` on every call, so there is no
    module-level name in ``candidate_articles``' namespace to patch. The
    least-intrusive interception matching the real code is wrapping the
    attribute on its defining module, ``vesta.zim.passages``.
    """
    # Local by design: the argv fail-fast guard above must not pay vesta's
    # import cost, so nothing from the app imports at module top.
    from vesta.retrieval.impls.candidate_articles import CandidateArticles  # noqa: PLC0415
    from vesta.zim import passages as passages_mod  # noqa: PLC0415
    from vesta.zim.registry import LocalArchive  # noqa: PLC0415

    orig_extract = LocalArchive.extract
    orig_split = passages_mod.split_passages
    orig_build = CandidateArticles.build

    async def timed_extract(self: LocalArchive, path: EntryPath) -> ExtractedArticle:
        t0 = time.perf_counter()
        article = await orig_extract(self, path)
        if _IN_BUILD:
            # Failed extracts propagate to build()'s own
            # ``except Exception: continue`` and leave no row — the
            # table records extractions, not attempts.
            _PENDING[id(article)] = len(_ROWS)
            _ROWS.append(
                [
                    self.id,
                    str(path),
                    (time.perf_counter() - t0) * 1000.0,
                    0.0,
                    len(article.text),
                ]
            )
        return article

    def timed_split(article: ExtractedArticle, *args: Any, **kwargs: Any) -> list[Passage]:
        t0 = time.perf_counter()
        parts = orig_split(article, *args, **kwargs)
        idx = _PENDING.pop(id(article), None)
        if idx is not None:
            _ROWS[idx][3] = (time.perf_counter() - t0) * 1000.0
        return parts

    async def timed_build(
        self: CandidateArticles,
        cands: list[Candidate],
        q: PreparedQuery,
        tr: Trace,
    ) -> list[Passage]:
        global _IN_BUILD
        n_before = len(_ROWS)
        _IN_BUILD += 1
        t0 = time.perf_counter()
        try:
            result = await orig_build(self, cands, q, tr)
        finally:
            _IN_BUILD -= 1
            _PENDING.clear()  # drop ids of articles this build released
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # n_cands = rows this call added. With --concurrency 1 (the only sane
        # way to run latency benches — cli.py warns on the flag) builds never
        # interleave, so the delta is exact.
        print(
            f"### build() total={dt_ms:.0f}ms"
            f"  n_cands={len(_ROWS) - n_before}  passages={len(result)}",
            file=sys.stderr,
            flush=True,
        )
        return result

    LocalArchive.extract = timed_extract  # type: ignore[method-assign]
    passages_mod.split_passages = timed_split
    CandidateArticles.build = timed_build  # type: ignore[method-assign]


def run() -> int:
    argv = sys.argv[1:]
    if argv[:2] != ["bench", "run"]:
        print(
            "bench_stage_latency.py: this harness wraps `vesta bench run` only —\n"
            "the first two arguments must be the literal `bench run` subcommand.\n"
            "example: uv run python scripts/bench_stage_latency.py bench run \\\n"
            "             --system retrieval_only --profile hybrid --level 3 \\\n"
            "             --limit 1 --repeats 1 --concurrency 1 --no-persist",
            file=sys.stderr,
        )
        return 2
    _install_patches()
    atexit.register(_print_report)
    from vesta.cli import main  # noqa: PLC0415  # local so the guard above stays free

    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
