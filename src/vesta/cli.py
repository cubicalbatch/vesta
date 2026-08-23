"""``vesta`` command-line interface — the eval/bench composition root.

A top-level module (like ``main.py``): exempt from the ≤2 internal-dependency cap
because it is the composition root that wires the DB, archive registry, and eval
harness together. The committed, tested eval surface for humans; the API is the
same surface for the SPA.

Commands:

* ``vesta bench run`` the unified benchmark (matrix: systems x profiles x models)
  over ``benchmarks/vesta_bench_v2.json``, persisted to ``bench_runs``, with
  optional markdown/JSON reports to ``benchmarks/results/``.
* ``vesta bench verify`` oracle + closed-book dataset verification → review file.
* ``vesta bench rejudge ID`` re-grade a stored run's ``pending`` answers.
* ``vesta bench compare A B`` per-question diff (fixed/broken/both_correct/both_wrong).
* ``vesta bench list|show`` list runs / show one run's scorecard.
* ``vesta bench retrieval [--profile P] [--baseline B] [--sweep k=a,b] [--explain]``
  run a profile over the golden set, print the
  reporting-format table, and persist to ``eval_runs``.
* ``vesta bench retrieval --dataset PATH`` the article-recall arms:
  fixed zero-LLM A/D/B arms (NL/standard, NL/hybrid, gold-title/standard) over
  the bench dataset — recall@1/@5/@10/any table + per-question JSON artifact.
* ``vesta bench hardware`` the hardware encoder/extraction/latency
  harness, writing ``bench_results/<machine>-<date>.md``.
* ``vesta eval`` deprecated alias for ``vesta bench retrieval`` (prints a pointer).
* ``vesta index [--depth N] [--zim ID|NAME]`` index a ZIM archive (semantic)
  in THIS process, detached from the web server. The web UI enqueues an
  ``index_zim`` job on the server's ``JobRunner``, which a uvicorn reload
  interrupts; this runs the same job directly here so code edits can't kill it.

Sync where fine (CLI); the API path is the job-shaped one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog

# Top-level composition root: may import broadly (composition roots hold wires, not logic).
from vesta import config
from vesta.api.eval import LivePipelineRunner, SqliteEvalStore
from vesta.api.state import AppState
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.db.settings_store import load_settings
from vesta.encoders import bind_manager, build_manager_from_settings
from vesta.encoders.download import ensure_model
from vesta.encoders.registry import MODEL_SPECS
from vesta.eval.article_recall import ArticleRecallReport, QuestionRecall
from vesta.eval.bench import encoder, extraction, hardware
from vesta.eval.bench_dataset import BenchDataset
from vesta.eval.bench_runner import BENCH_SYSTEMS, resolve_matrix_axes
from vesta.eval.bench_scoring import metric_lookup
from vesta.eval.calibrate import ConfidenceSample, fit_thresholds
from vesta.eval.golden import (
    EVAL_ARCHIVE_CHECKSUM,
    EVAL_ARCHIVE_PATH,
    EVAL_REGRESSION_EPSILON,
    GoldenSet,
    load_set,
    verify_against_archive,
)
from vesta.eval.regression import evaluate as gate_evaluate
from vesta.eval.runner import (
    Comparison,
    PipelineRunner,
    RunRecord,
    SweepPoint,
    compare,
    evaluate_profile,
    parse_sweep,
    persist_run,
)
from vesta.index import INDEX_EMBEDDER, reseed_indexed_state, set_indexed_state
from vesta.index import bind_runtime as bind_index_runtime
from vesta.retrieval.profiles import RetrievalProfile, load_profile, resolve_profile
from vesta.vectors import VECTORS_OVERSAMPLE, VECTORS_QUANTIZER, bind_store
from vesta.vectors.sqlite_vec_store import SqliteVecStore
from vesta.zim import bind_registry
from vesta.zim.registry import ArchiveRegistry


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``vesta`` console script."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv[:1] == ["eval"]:
        # Deprecated alias (formerly `vesta eval`). One-line pointer;
        # swallow any former flags so `vesta eval --profile X` still hints.
        print(
            "`vesta eval` is deprecated; the golden-set retrieval gate is now "
            "`uv run vesta bench retrieval` (same flags, same eval_runs)."
        )
        return 0
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "bench":
        return asyncio.run(_cmd_bench_root(args))
    if command == "index":
        return asyncio.run(_cmd_index(args))
    if command == "models":
        return _cmd_models(args)
    parser.error(f"unknown command {command!r}")
    return 2  # unreachable


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 — one parser for every subcommand
    parser = argparse.ArgumentParser(prog="vesta", description="Vesta CLI (bench + eval + index).")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── `vesta bench <subcommand>` — the ONE benchmark umbrella ─────────────
    p_bench = sub.add_parser("bench", help="The unified benchmark umbrella.")
    bsub = p_bench.add_subparsers(dest="bench_command", required=True)

    p_run = bsub.add_parser("run", help="Run the unified benchmark (matrix-capable).")
    p_run.add_argument(
        "--system",
        action="append",
        default=None,
        help="Repeatable; matrix axis (6 systems).",
    )
    p_run.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Repeatable; matrix axis (retrieval profile).",
    )
    p_run.add_argument(
        "--model", action="append", default=None, help="Repeatable; matrix axis (answer model)."
    )
    p_run.add_argument("--endpoint", default=None, help="Per-run answer LLM endpoint override.")
    p_run.add_argument("--api-key", default=None, help="Per-run answer LLM API key override.")
    p_run.add_argument(
        "--economy",
        default=None,
        choices=("auto", "on", "off"),
        help="Force the agent token-economy mode (answer.agent.economy) for this run, "
        "regardless of stored settings and hardware; recorded in the run's config_json.",
    )
    p_run.add_argument(
        "--context-profile",
        default=None,
        choices=("auto", "8k", "8k-fullprompt", "8k-fullprompt-wide", "16k", "full"),
        help="Force the agent context-window profile (answer.agent.context_profile) "
        "for this run, regardless of stored settings and runtime; recorded in the "
        "run's config_json. '8k'/'16k' activate window budgeting even on remote "
        "endpoints; '8k' uses full system prompt + 2400-char passage cap; "
        "'8k-fullprompt' is the 1800-char-cap arm; '8k-fullprompt-wide' is "
        "values-identical to '8k'.",
    )
    p_run.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help="Force the agent tool-round cap (answer.agent.max_tool_rounds) for "
        "this run. -1 = no extra cap (the shipped default); 0 = derive from "
        "the context-window ledger when a profile is active; an explicit N "
        "wins on any profile.",
    )
    p_run.add_argument(
        "--compact-reask",
        default=None,
        choices=("auto", "on", "off"),
        help="Force compact-and-re-ask (answer.agent.compact_reask) "
        "for this run. 'off' is the shipped default; 'auto' = on for windowed "
        "profiles, off at full; 'on' forces it on any profile.",
    )
    p_run.add_argument(
        "--age-tool-chars",
        type=int,
        default=None,
        help="Force the context-aging budget (answer.agent.age_tool_chars) "
        "for this run. -1 = aging off (the shipped default); 0 = derive from "
        "the window ledger when a profile is active; N > 0 binds.",
    )
    p_run.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Repeatable; force any registered settings key for this run "
        "(e.g. --set answer.agent.preseed_order=idf). Keys are validated "
        "against the settings registry; the pairs merge over stored settings "
        "before the resolver seeds — a forced value wins over plan defaults "
        "and over the dedicated flags above — and are recorded in the run's "
        "config_json.",
    )
    p_run.add_argument(
        "--judge-model", default=None, help="Judge model id (default: eval.judge.model)."
    )
    p_run.add_argument(
        "--judge-endpoint", default=None, help="Judge endpoint override (eval.judge.endpoint_url)."
    )
    p_run.add_argument(
        "--judge-api-key", default=None, help="Judge API key override (eval.judge.api_key)."
    )
    p_run.add_argument(
        "--dataset",
        default=None,
        help="Path to the dataset (default: bench.dataset → benchmarks/vesta_bench_v2.json).",
    )
    p_run.add_argument(
        "--slice",
        default="core",
        choices=("core", "all"),
        help="Dataset slice (default core; 'all' = same set, Wikipedia-only).",
    )
    p_run.add_argument(
        "--level",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Question tier: cumulative level<=L subset — 1=50 smoke, "
        "2=100 standard (default), 3=200 release.",
    )
    p_run.add_argument(
        "--capability", action="append", default=None, help="Repeatable; filter by capability."
    )
    p_run.add_argument(
        "--save-context",
        default=None,
        metavar="PATH",
        help="With --system retrieval_only: dump the retrieved passages to a JSON "
        "context snapshot for later answer-only replay (--from-context).",
    )
    p_run.add_argument(
        "--from-context",
        default=None,
        metavar="PATH",
        help="Run --system answer_only against a saved context snapshot "
        "(zero retrieval confounding).",
    )
    p_run.add_argument(
        "--context-passages",
        type=int,
        default=None,
        metavar="N",
        help="With --from-context: feed only the top N snapshot passages in rank "
        "order (pre-seed sensitivity); recorded in config_json.",
    )
    p_run.add_argument(
        "--oracle-context",
        action="store_true",
        help="Run --system answer_only against gold oracle articles (perfect context).",
    )
    p_run.add_argument(
        "--difficulty",
        action="append",
        default=None,
        help="Repeatable; filter easy|medium|hard.",
    )
    p_run.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    p_run.add_argument(
        "--repeats", type=int, default=None, help="Run each cell N times (default 1)."
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Pipeline questions in flight (default 1 — raising it contaminates latency; "
        "use only for a dedicated endpoint).",
    )
    p_run.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="Judge calls in flight (default 4; clamped to 1 when the judge shares the answer endpoint).",
    )
    p_run.add_argument("--scope", default=None, help="Restrict retrieval scope (ZIM id/name).")
    p_run.add_argument("--label", default=None, help="Human label for the run group.")
    p_run.add_argument(
        "--no-persist", action="store_true", help="Report only; do not write bench_runs."
    )
    p_run.add_argument(
        "--report",
        default="md",
        choices=("md", "json", "both"),
        help="Write a report to benchmarks/results/ (default md).",
    )
    p_run.add_argument(
        "--baseline",
        default=None,
        help="RUN_ID of a prior run; print the compare table at the end.",
    )
    p_run.add_argument(
        "--import-old",
        action="store_true",
        help="One-shot: import historical answer_runs rows into bench_runs (idempotent), then exit.",
    )
    p_run.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_ret = bsub.add_parser("retrieval", help="Former `vesta eval` — golden-set retrieval gate.")
    p_ret.add_argument(
        "--profile",
        default=None,
        help="Retrieval profile name (golden-set mode; default lexical). Dataset "
        "mode pins its own A/D/B profiles.",
    )
    p_ret.add_argument(
        "--golden",
        default="full",
        choices=("full", "fixture_subset"),
        help="Which golden set (full needs the pinned archive).",
    )
    p_ret.add_argument("--baseline", default=None, help="Baseline run id or profile name.")
    p_ret.add_argument("--sweep", default=None, help="Param sweep, e.g. rrf.k=10,20,40,60.")
    p_ret.add_argument(
        "--force", action="store_true", help="Allow comparing degraded vs clean runs."
    )
    p_ret.add_argument("--explain", action="store_true", help="Print per-query win/loss detail.")
    p_ret.add_argument("--data-dir", default=None, help="Override data.dir (else settings/env).")
    p_ret.add_argument(
        "--no-persist", action="store_true", help="Do not write the run to eval_runs."
    )
    p_ret.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="Dataset mode: run the fixed zero-LLM A/D/B article-recall "
        "arms over the bench dataset instead of the golden set. Writes a JSON "
        "artifact; never touches eval_runs/bench_runs.",
    )
    p_ret.add_argument(
        "--level",
        type=int,
        default=None,
        choices=(1, 2, 3),
        help="Dataset mode: cumulative level<=L tier filter.",
    )
    p_ret.add_argument(
        "--capability",
        action="append",
        default=None,
        help="Dataset mode: repeatable capability filter.",
    )
    p_ret.add_argument(
        "--limit", type=int, default=None, help="Dataset mode: only the first N questions."
    )
    p_ret.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Dataset mode: JSON artifact path (default benchmarks/results/"
        "<ts>-round0-article-recall.json).",
    )
    p_ret.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=("run", "verify-golden", "calibrate", "regression"),
        help="Eval sub-action.",
    )
    p_hw = bsub.add_parser("hardware", help="Former `vesta bench` — encoder/extraction/latency.")
    p_hw.add_argument(
        "--archive",
        default=None,
        help="Path to a ZIM for extraction throughput (defaults to the pinned archive).",
    )
    p_hw.add_argument("--out-dir", default="bench_results", help="Where to write the report.")
    p_hw.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_hw.add_argument(
        "--skip-extraction", action="store_true", help="Skip the extraction throughput benches."
    )
    p_rejudge = bsub.add_parser("rejudge", help="Re-grade a stored run's pending answers.")
    p_rejudge.add_argument("run_id", type=int, help="bench_runs id to re-judge.")
    p_rejudge.add_argument(
        "--dataset",
        default=None,
        help="Path to the dataset (default: bench.dataset) — needed to render "
        "rubrics for judge-cache misses on a stored run.",
    )
    p_rejudge.add_argument(
        "--judge-model", default=None, help="Judge model id (default: eval.judge.model)."
    )
    p_rejudge.add_argument("--judge-endpoint", default=None, help="Judge endpoint override.")
    p_rejudge.add_argument("--judge-api-key", default=None, help="Judge API key override.")
    p_rejudge.add_argument(
        "--judge-concurrency", type=int, default=None, help="Judge calls in flight."
    )
    p_rejudge.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_compare = bsub.add_parser("compare", help="Per-question diff across two runs.")
    p_compare.add_argument("run_a", type=int, help="First run id.")
    p_compare.add_argument("run_b", type=int, help="Second run id.")
    p_compare.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_verify = bsub.add_parser("verify", help="Oracle + closed-book dataset verification.")
    p_verify.add_argument(
        "--dataset", default=None, help="Path to the dataset (default: bench.dataset)."
    )
    p_verify.add_argument(
        "--model", default=None, help="Answer model (default: inference.llm.model)."
    )
    p_verify.add_argument("--endpoint", default=None, help="Answer endpoint override.")
    p_verify.add_argument("--api-key", default=None, help="Answer API key override.")
    p_verify.add_argument(
        "--judge-model", default=None, help="Judge model id (default: eval.judge.model)."
    )
    p_verify.add_argument("--judge-endpoint", default=None, help="Judge endpoint override.")
    p_verify.add_argument("--judge-api-key", default=None, help="Judge API key override.")
    p_verify.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    p_verify.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Answer-pass questions in flight (default: bench.max_concurrent). Verify "
        "reports no latency, so raising it cannot contaminate a reported p50/p95.",
    )
    p_verify.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="Judge calls in flight (default 4; clamped to 1 when the judge shares the answer endpoint).",
    )
    p_verify.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_list = bsub.add_parser("list", help="List persisted bench runs.")
    p_list.add_argument("--limit", type=int, default=50, help="Max runs to list (default 50).")
    p_list.add_argument("--data-dir", default=None, help="Override data.dir.")
    p_show = bsub.add_parser("show", help="Show one run's scorecard + per-question verdicts.")
    p_show.add_argument("run_id", type=int, help="bench_runs id.")
    p_show.add_argument("--data-dir", default=None, help="Override data.dir.")

    p_models = sub.add_parser(
        "models", help="Fetch encoder model files (dev/install-time only, needs network)."
    )
    p_models.add_argument(
        "--role",
        action="append",
        choices=("static", "embed", "rerank"),
        help="Which role(s) to fetch (default: all three, using their configured repo).",
    )
    p_models.add_argument("--data-dir", default=None, help="Override data.dir.")

    p_index = sub.add_parser(
        "index",
        help="Index a ZIM archive (semantic) in this process — detached from the web server.",
    )
    p_index.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Index depth 1..3 (1=article, 2=+H2 sections, 3=~400-token passages). Default 1.",
    )
    p_index.add_argument(
        "--zim",
        action="append",
        default=None,
        help="Archive id or filename/name substring. Repeatable / comma-separated to queue "
        "several archives; omit for every registered archive (default: the only archive).",
    )
    p_index.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the resume checkpoint and rebuild from scratch.",
    )
    p_index.add_argument("--data-dir", default=None, help="Override data.dir.")
    return parser


# ── `vesta bench <subcommand>` — the ONE benchmark umbrella ──────────────────


async def _cmd_bench_root(args: argparse.Namespace) -> int:  # noqa: PLR0911 — one branch per subcommand
    """Dispatch ``vesta bench run|retrieval|hardware|rejudge|compare|verify|list|show``."""
    cmd = args.bench_command
    if cmd == "run":
        return await _cmd_bench_run(args)
    if cmd == "retrieval":
        return await _cmd_eval(args)  # former `vesta eval` — golden-set gate
    if cmd == "hardware":
        return await _cmd_bench(args)  # former `vesta bench` — hardware harness
    if cmd == "rejudge":
        return await _cmd_bench_rejudge(args)
    if cmd == "compare":
        return await _cmd_bench_compare_cli(args)
    if cmd == "verify":
        return await _cmd_bench_verify(args)
    if cmd == "list":
        return await _cmd_bench_list(args)
    if cmd == "show":
        return await _cmd_bench_show(args)
    print(f"unknown bench command {cmd!r}")
    return 2


def _build_apply_overrides(args: argparse.Namespace) -> dict[str, str]:
    """Map ``--endpoint/--api-key/--judge-*`` flags onto settings keys.

    The runner does NOT read endpoints itself (boundary); the CLI overrides the
    settings before the gateway + judge are built (mirrors the former matrix runner).
    """
    from vesta.answer import (
        ANSWER_AGENT_AGE_TOOL_CHARS,
        ANSWER_AGENT_COMPACT_REASK,
        ANSWER_AGENT_CONTEXT_PROFILE,
        ANSWER_AGENT_ECONOMY,
        ANSWER_AGENT_MAX_TOOL_ROUNDS,
    )
    from vesta.eval.golden import EVAL_JUDGE_API_KEY, EVAL_JUDGE_ENDPOINT_URL
    from vesta.inference import (
        INFERENCE_LLM_API_KEY,
        INFERENCE_LLM_ENDPOINT_URL,
        INFERENCE_LLM_MODEL,
        INFERENCE_LLM_SOURCE,
    )

    overrides: dict[str, str] = {}
    if getattr(args, "model", None):
        # `--model` is a repeatable matrix axis on `bench run` (so `args.model`
        # is a list there) but a scalar on `verify`/`rejudge`. The settings-
        # backed gateway holds ONE model — generation reads
        # `inference.llm.model` from the live snapshot, so the first selected
        # model is the one the gateway actually calls. A true multi-model
        # matrix needs per-cell settings rebinding (a follow-up); the list
        # repr must NEVER become the model id (it 400s the endpoint).
        model = args.model[0] if isinstance(args.model, list) else args.model
        overrides[INFERENCE_LLM_MODEL.key] = str(model)
    if getattr(args, "endpoint", None):
        overrides[INFERENCE_LLM_ENDPOINT_URL.key] = args.endpoint
        overrides[INFERENCE_LLM_SOURCE.key] = "remote"
    if getattr(args, "api_key", None) is not None:
        overrides[INFERENCE_LLM_API_KEY.key] = args.api_key
    if getattr(args, "judge_endpoint", None):
        overrides[EVAL_JUDGE_ENDPOINT_URL.key] = args.judge_endpoint
    if getattr(args, "judge_api_key", None) is not None:
        overrides[EVAL_JUDGE_API_KEY.key] = args.judge_api_key
    if getattr(args, "economy", None):
        # `--economy` (bench run) forces the agent token-economy mode for the
        # whole SUT run regardless of stored settings and hardware; merged into
        # the resolver here so every snapshot read (incl. agent_chat) sees it.
        overrides[ANSWER_AGENT_ECONOMY.key] = args.economy
    if getattr(args, "context_profile", None):
        # `--context-profile` (bench run) forces the window plan the answer
        # path resolves — same mechanic as --economy: merged
        # into the resolver so every snapshot read (incl. agent_chat's
        # _runtime_window_tokens + resolve_budget) sees it for the whole run.
        overrides[ANSWER_AGENT_CONTEXT_PROFILE.key] = args.context_profile
    if getattr(args, "max_tool_rounds", None) is not None:
        # `--max-tool-rounds` (bench run): the tool-round cap,
        # same mechanic as --economy — merged into the resolver so the agent
        # path's per-turn lever resolution sees it for the whole run.
        overrides[ANSWER_AGENT_MAX_TOOL_ROUNDS.key] = str(args.max_tool_rounds)
    if getattr(args, "compact_reask", None):
        # `--compact-reask` (bench run): auto|on|off.
        overrides[ANSWER_AGENT_COMPACT_REASK.key] = args.compact_reask
    if getattr(args, "age_tool_chars", None) is not None:
        # `--age-tool-chars` (bench run): N>0 binds, -1 = off,
        # 0 = derive under a windowed profile.
        overrides[ANSWER_AGENT_AGE_TOOL_CHARS.key] = str(args.age_tool_chars)
    set_pairs = _parse_set_pairs(args)
    if set_pairs:
        # `--set KEY=VALUE` (bench run, repeatable): generic settings forcing,
        # the same mechanic as --economy but for ANY registered key.
        # Merged LAST so the explicit generic pairs win over
        # the dedicated flags when both set the same key.
        overrides.update(set_pairs)
    return overrides


def _parse_set_pairs(args: argparse.Namespace) -> dict[str, str]:
    """Parse + validate ``--set KEY=VALUE`` (repeatable; ``bench run`` only).

    KEY must exist in the settings registry (an unknown key is a clean
    :class:`SystemExit` naming the valid groups, not a traceback mid-run);
    VALUE is coerced/validated through :func:`resolve_value` so a bad value
    (wrong type, out of bounds, off the choices list) also fails here, at the
    flag, instead of when the resolver first seeds. Returns the applied pairs
    in flag order; later repeats of the same key win.
    """
    from vesta.config.settings import all_settings, resolve_value

    pairs: dict[str, str] = {}
    for raw in getattr(args, "set", None) or ():
        key, sep, value = raw.partition("=")
        if not sep or not key:
            raise SystemExit(f"vesta bench run --set: expected KEY=VALUE, got {raw!r}")
        descriptor = all_settings().get(key)
        if descriptor is None:
            groups = ", ".join(sorted({s.group for s in all_settings().values()}))
            raise SystemExit(
                f"vesta bench run --set: unknown settings key {key!r}. "
                f"Keys must be declared in the settings registry; valid groups: {groups}"
            )
        try:
            resolve_value(descriptor, db_values={key: value}, env={})
        except ValueError as exc:
            raise SystemExit(f"vesta bench run --set: invalid value for {key}: {exc}") from None
        pairs[key] = value
    return pairs


def _metric(metrics: dict[str, object], path: str) -> object:
    """Look a metric up in ``metrics_json`` by dotted path (``answer.strict_accuracy``);
    falls back to a top-level key for the retired flat layout (old eval_runs rows)."""
    return metric_lookup(metrics, path, flat_fallback=True)


def _resolve_matrix(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Resolve the systems / profiles / models matrix axes."""
    from vesta.inference import INFERENCE_LLM_MODEL

    systems, profiles, models = resolve_matrix_axes(
        args.system,
        args.profile,
        args.model,
        default_systems=str(config.get(BENCH_SYSTEMS)),
        default_model=str(config.get(INFERENCE_LLM_MODEL)),
    )
    if not models and any(s != "retrieval_only" for s in systems):
        # The model default is inert on a fresh install ("") — a
        # clear CLI error instead of a silent run against an empty model id.
        # retrieval_only makes no LLM calls and needs none.
        raise SystemExit("no model configured; pass --model or set inference.llm.model")
    if not models:
        models = [""]
    return systems, profiles, models


def _resolve_questions(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    """Load the dataset, apply slice/level/capability/difficulty/limit filters.

    Shared by ``bench run`` and ``bench retrieval --dataset``; the getattr
    defaults let the lighter dataset-mode flag set omit slice/difficulty.
    """
    from vesta.eval.bench_dataset import BENCH_DATASET, apply_flag_filters, load_bench_dataset

    path = args.dataset or str(config.get(BENCH_DATASET))
    dataset = load_bench_dataset(path)
    qs = apply_flag_filters(
        dataset.questions,
        slice=getattr(args, "slice", None),
        level=getattr(args, "level", None),
        capabilities=getattr(args, "capability", None),
        difficulties=getattr(args, "difficulty", None),
        limit=args.limit,
    )
    return dataset, qs


def _question_map(qs: Sequence[Any]) -> dict[str, Any]:
    """Map question id → BenchQuestion (for rejudge's rubric rendering)."""
    return {q.id: q for q in qs}


def _print_progress(update: Any) -> None:
    """Render a progress tick to the terminal (one line per cell state change)."""
    tag = {"pipeline": "run", "judging": "judge", "complete": "done", "aborted": "ABORT"}.get(
        update.stage, update.stage
    )
    print(
        f"  [{update.system}] {tag} {update.done}/{update.total}{' ' + update.stage if update.run_id else ''}",
        flush=True,
    )


async def _cmd_bench_run(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912, PLR0915 — one coherent matrix flow
    """``vesta bench run`` — the unified, matrix-capable benchmark."""
    from vesta.api.bench import (
        SYSTEM_CLASSES,
        InMemoryBenchStore,
        SqliteBenchStore,
        import_answer_runs,
        make_judge_llm,
        make_system,
        reconcile_stale_bench_runs,
    )
    from vesta.eval.bench_runner import (
        BENCH_JUDGE_CONCURRENCY,
        BENCH_MAX_CONCURRENT,
        BENCH_REPEATS,
        resolve_judge_concurrency,
        run_benchmark,
    )
    from vesta.eval.golden import EVAL_JUDGE_ENDPOINT_URL, EVAL_JUDGE_MODEL
    from vesta.eval.runner import git_sha as _git_sha
    from vesta.inference import (
        INFERENCE_LLM_API_KEY,
        INFERENCE_LLM_ENDPOINT_URL,
    )

    if args.import_old:
        # One-shot answer_runs import (idempotent — imported rows are labeled
        # `imported:<id>`; re-running re-imports the same source rows).
        async with _open_runtime(args.data_dir) as state:
            n = await import_answer_runs(state.db)
            print(f"imported {n} historical answer_runs rows into bench_runs.")
        return 0

    overrides = _build_apply_overrides(args)
    settings_set = _parse_set_pairs(args)  # the --set pairs, for config_json
    async with _open_runtime(
        args.data_dir,
        with_gateway=True,
        settings_overrides=overrides or None,
    ) as state:
        # Reconcile stale runs at open (mirrors main.py lifespan).
        try:
            stale = await reconcile_stale_bench_runs(state.db)
            if stale:
                print(f"reconciled {stale} stale bench run(s) left 'running' by a dead process.")
        except Exception as exc:
            print(f"warning: bench reconcile failed: {exc!r}")

        dataset, qs = _resolve_questions(args)
        if not qs:
            print(
                "no questions selected (check --slice/--level/--capability/--difficulty/--limit)."
            )
            return 1
        systems, profiles, models = _resolve_matrix(args)

        # ── Context mode flags: --from-context / --oracle-context / --save-context ──
        snapshot: dict[str, object] | None = None
        context_path: str | None = args.from_context
        oracle_context = bool(args.oracle_context)
        if context_path and oracle_context:
            print("--from-context and --oracle-context are mutually exclusive.")
            return 1
        if (context_path or oracle_context) and args.system and "answer_only" not in args.system:
            print("--from-context/--oracle-context run --system answer_only (implied if unset).")
            return 1
        if context_path or oracle_context:
            systems = ["answer_only"]
        if context_path is not None:
            from vesta.api.bench import load_context_snapshot, snapshot_missing_ids

            try:
                snapshot = load_context_snapshot(context_path)
            except ValueError as exc:
                print(str(exc))
                return 1
            missing = snapshot_missing_ids(snapshot, [q.id for q in qs])
            if missing:
                # Fail loudly, not with silent empty contexts: the snapshot's
                # level/filter must cover the selected question set.
                print(
                    f"snapshot {context_path} is missing {len(missing)} of {len(qs)} selected "
                    f"questions (first: {missing[0]}). Re-run --save-context with a level "
                    f">= the replay's --level and the same filters."
                )
                return 1
        if args.save_context and systems != ["retrieval_only"]:
            print("--save-context needs --system retrieval_only (it snapshots the pipeline).")
        if args.context_passages is not None and context_path is None:
            print("--context-passages requires --from-context (it truncates a snapshot replay).")
            return 1
        if args.context_passages is not None and args.context_passages < 1:
            print("--context-passages must be >= 1.")
            return 1

        needs_judge = any(s != "retrieval_only" for s in systems)
        judge_model = (args.judge_model or str(config.get(EVAL_JUDGE_MODEL))) if needs_judge else ""
        answer_endpoint = str(config.get(INFERENCE_LLM_ENDPOINT_URL))
        judge_endpoint = str(config.get(EVAL_JUDGE_ENDPOINT_URL)) if args.judge_endpoint else ""

        # Per-run LLM overrides handled by settings; the judge needs the model
        # explicit and the endpoint for the concurrency clamp.
        judge, judge_gateway = make_judge_llm(state, judge_model) if needs_judge else (None, None)

        # Judge endpoint override must be reflected in the clamp too.
        effective_judge_endpoint = args.judge_endpoint or judge_endpoint
        judge_concurrency, shares = resolve_judge_concurrency(
            args.judge_concurrency or int(BENCH_JUDGE_CONCURRENCY.default),
            answer_endpoint=answer_endpoint,
            judge_endpoint=effective_judge_endpoint,
        )
        # Build the flat system list = systems x profiles x models.
        sut_list: list[Any] = []
        for sys_name in systems:
            if sys_name not in SYSTEM_CLASSES:
                print(f"unknown system {sys_name!r}; choose from {sorted(SYSTEM_CLASSES)}.")
                return 1
            for prof in profiles:
                profile_obj = _resolve_profile(state, prof) if prof else None
                profile_hash = profile_obj.hash if profile_obj is not None else ""
                for model in models:
                    sut_list.append(
                        make_system(
                            sys_name,
                            state,
                            profile=prof or None,
                            scope=args.scope,
                            model_id=model,
                            endpoint=answer_endpoint,
                            api_key=str(config.get(INFERENCE_LLM_API_KEY)),
                            profile_hash=profile_hash,
                            context_path=context_path,
                            oracle_context=oracle_context,
                            collect_context=bool(args.save_context),
                            context_passages=args.context_passages,
                        )
                    )
        label = args.label or f"cli matrix {len(systems)}x{len(profiles)}x{len(models)}"
        run_group = str(uuid.uuid4())
        print(
            f"dataset {dataset.name}#{dataset.hash[:8]}  {len(qs)} questions (level {args.level})  "
            f"git {_git_sha()[:8]}  judge {judge_model or '(none)'}"
        )
        print(
            f"matrix {len(systems)} systems x {len(profiles)} profiles x {len(models)} models "
            f"= {len(sut_list)} cells x {args.repeats or BENCH_REPEATS.default} repeat(s)"
        )
        print(
            f"answer endpoint {answer_endpoint}  judge concurrency {judge_concurrency}"
            + (" (shared endpoint → clamped to 1)" if judge_concurrency == 1 and shares else "")
            + (f"  economy {args.economy} (forced)" if args.economy else "")
            + (f"  context-profile {args.context_profile} (forced)" if args.context_profile else "")
            + (
                f"  max-tool-rounds {args.max_tool_rounds} (forced)"
                if args.max_tool_rounds is not None
                else ""
            )
            + (f"  compact-reask {args.compact_reask} (forced)" if args.compact_reask else "")
            + (
                f"  age-tool-chars {args.age_tool_chars} (forced)"
                if args.age_tool_chars is not None
                else ""
            )
            + (f"  --set {len(settings_set)} key(s) forced" if settings_set else "")
        )
        if context_path is not None:
            print(
                f"context replay: {context_path}"
                + (
                    f" (top {args.context_passages} passages)"
                    if args.context_passages is not None
                    else ""
                )
                + " (zero retrieval confounding)"
            )
        if oracle_context:
            print("context replay: gold oracle articles")
        if args.save_context:
            print(f"context snapshot: will write {args.save_context}")
        try:
            records = await run_benchmark(
                dataset=dataset,
                questions=qs,
                systems=sut_list,
                store=(SqliteBenchStore(state.db) if not args.no_persist else InMemoryBenchStore()),
                judge=judge,
                judge_model=judge_model,
                run_group=run_group,
                label=label,
                scope=args.scope or "",
                config_snapshot=dict(config.snapshot().values),
                economy=args.economy,
                context_profile=args.context_profile,
                settings_set=(settings_set or None),
                judge_concurrency=judge_concurrency,
                judge_shares_endpoint=shares,
                repeats=args.repeats or int(BENCH_REPEATS.default),
                max_concurrent=args.concurrency or int(BENCH_MAX_CONCURRENT.default),
                progress=_print_progress,
                level=args.level,
            )
        finally:
            if judge_gateway is not None:
                with contextlib.suppress(Exception):
                    await judge_gateway.aclose()

        for rec in records:
            _print_run_scorecard(rec)

        if args.save_context:
            from vesta.api.bench import write_context_snapshot

            retriever = next((s for s in sut_list if hasattr(s, "context_snapshot")), None)
            if retriever is not None:
                sub = next((r.subset_hash for r in records), "")
                prof = next((r.profile_name for r in records), "")
                out = write_context_snapshot(
                    args.save_context,
                    questions=retriever.context_snapshot(),
                    dataset=dataset,
                    subset_hash_val=sub,
                    system="retrieval_only",
                    profile=prof,
                    level=args.level,
                )
                print(
                    f"wrote context snapshot {out} ({len(retriever.context_snapshot())} questions)"
                )
        if not args.no_persist:
            print(f"\n persisted run_group={run_group} ({len(records)} runs)")

        if args.report in ("md", "both"):
            _write_bench_run_report("md", records, dataset, label)
        if args.report in ("json", "both"):
            _write_bench_run_report("json", records, dataset, label)

        if args.baseline:
            try:
                base_id = int(args.baseline)
                await _cmd_bench_compare(state, base_id, records[0].id)
            except (ValueError, IndexError):
                print(f"warning: could not compare against baseline {args.baseline!r}")
        return 0


def _print_run_scorecard(rec: Any) -> None:
    """One line per run — source metrics for retrieval runs, strict for answer runs."""
    m = rec.metrics_json
    head = f"  run {rec.id} [{rec.system}/{rec.profile_name or 'default'}/{rec.answer_model}]"
    if rec.system == "retrieval_only":
        src = m.get("source", {}) if isinstance(m, dict) else {}
        lat = src.get("latency", {}) if isinstance(src, dict) else {}
        p50 = lat.get("p50", 0) if isinstance(lat, dict) else 0
        print(
            f"{head} n={src.get('n', 0)} recall@1={_pct(src.get('recall_at_1'))} "
            f"recall@5={_pct(src.get('recall_at_5'))} recall@10={_pct(src.get('recall_at_10'))} "
            f"coverage={_pct(src.get('source_coverage'))} mrr={src.get('source_mrr'):.3f} "
            f"precision={_pct(src.get('retrieved_precision'))} p50={p50:.0f}ms "
            f"status={rec.status}"
        )
        return
    strict = _metric(m, "answer.strict_accuracy")
    strict_pct = f"{float(strict) * 100:.1f}%" if isinstance(strict, int | float) else "n/a"
    print(f"{head} strict={strict_pct} status={rec.status}")


def _pct(v: object) -> str:
    return f"{float(v) * 100:.1f}%" if isinstance(v, int | float) else "n/a"


def _slug_text(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower())
        elif out and out[-1] != "-":
            out.append("-")
    slug = "".join(out)
    return slug.strip("-") or "run"


def _write_bench_run_report(fmt: str, records: Sequence[Any], dataset: Any, label: str) -> None:
    """Write a markdown/JSON compare-style report to benchmarks/results/."""
    import datetime as _dt

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = _slug_text(label or "bench")
    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    if fmt == "json":
        payload = {
            "label": label,
            "generated": now,
            "dataset": {"name": dataset.name, "hash": dataset.hash},
            "runs": [r.to_dict() for r in records],
        }
        path = out_dir / f"{stamp}-{slug}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        lines = [
            f"# Benchmark report — {label}",
            "",
            f"generated: {now}",
            f"dataset: {dataset.name} #{dataset.hash}",
            "",
            "| run | system | profile | model | strict | weighted | status |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in records:
            m = r.metrics_json
            strict = _metric(m, "answer.strict_accuracy")
            weighted = _metric(m, "answer.weighted_accuracy")
            strict_txt = f"{float(strict) * 100:.1f}%" if isinstance(strict, int | float) else "n/a"
            weighted_txt = (
                f"{float(weighted) * 100:.1f}%" if isinstance(weighted, int | float) else "n/a"
            )
            lines.append(
                f"| {r.id} | {r.system} | {r.profile_name or 'default'} | "
                f"{r.answer_model} | {strict_txt} | {weighted_txt} | {r.status} |"
            )
        # Source metrics table (retrieval runs carry these; answer runs too).
        lines.extend(["", "## Source metrics", ""])
        lines.append(
            "| run | system | n | recall@1 | recall@5 | recall@10 | coverage | mrr | p50 ms |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in records:
            m = r.metrics_json
            src = m.get("source", {}) if isinstance(m, dict) else {}
            lat = src.get("latency", {}) if isinstance(src, dict) else {}
            p50 = lat.get("p50", 0) if isinstance(lat, dict) else 0
            lines.append(
                f"| {r.id} | {r.system} | {src.get('n', 0)} | "
                f"{_pct(src.get('recall_at_1'))} | {_pct(src.get('recall_at_5'))} | "
                f"{_pct(src.get('recall_at_10'))} | {_pct(src.get('source_coverage'))} | "
                f"{src.get('source_mrr', 0):.3f} | {p50:.0f} |"
            )
        # Token usage summary (answer-LLM only).
        lines.extend(["", "## Token usage (answer LLM)", ""])
        lines.append("| run | system | total in | total out | total | p50/question |")
        lines.append("|---|---|---|---|---|---|")
        for r in records:
            m = r.metrics_json
            t = _metric(m, "tokens.answer")
            if isinstance(t, dict):
                ti, to, tt, p50 = (
                    t.get("total_input", 0),
                    t.get("total_output", 0),
                    t.get("total", 0),
                    t.get("p50", 0),
                )
                lines.append(f"| {r.id} | {r.system} | {ti:,} | {to:,} | {tt:,} | {p50:,} |")
            else:
                lines.append(f"| {r.id} | {r.system} | n/a | n/a | n/a | n/a |")
        lines.extend(_peak_context_report_lines(records))
        path = out_dir / f"{stamp}-{slug}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _peak_context_report_lines(records: Sequence[Any]) -> list[str]:
    """Peak-context report section: the largest single request per
    question (p50/p90/p95/max, >8k/>16k shares, request totals, overflow
    fallbacks). n=0 rows (pre-meter runs, no-LLM systems) render as n/a."""
    lines = ["", "## Peak context (largest single request per question)", ""]
    lines.append(
        "| run | system | n | p50 | p90 | p95 | max | >8k | >16k | requests | overflow fb |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        m = r.metrics_json
        pc = _metric(m, "tokens.peak_context")
        if isinstance(pc, dict) and pc.get("n", 0):
            n = int(pc.get("n", 0))
            over8, over16 = int(pc.get("over_8192", 0)), int(pc.get("over_16384", 0))
            lines.append(
                f"| {r.id} | {r.system} | {n} | "
                f"{int(pc.get('p50', 0)):,} | {int(pc.get('p90', 0)):,} | "
                f"{int(pc.get('p95', 0)):,} | {int(pc.get('max', 0)):,} | "
                f"{over8} ({over8 / n:.1%}) | {over16} ({over16 / n:.1%}) | "
                f"{int(pc.get('requests_total', 0)):,} | "
                f"{int(pc.get('overflow_fallbacks', 0)):,} |"
            )
        else:
            lines.append(
                f"| {r.id} | {r.system} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
    return lines


# ── `vesta bench rejudge` ───────────────────────────────────────────────────


async def _cmd_bench_rejudge(args: argparse.Namespace) -> int:
    """Re-grade a stored run's pending answers (no pipeline work)."""
    from vesta.api.bench import SqliteBenchStore, make_judge_llm
    from vesta.eval.bench_dataset import BENCH_DATASET, load_bench_dataset
    from vesta.eval.bench_runner import BENCH_JUDGE_CONCURRENCY, rejudge_run
    from vesta.eval.golden import EVAL_JUDGE_MODEL

    overrides = _build_apply_overrides(args)
    async with _open_runtime(
        args.data_dir, with_gateway=True, settings_overrides=overrides or None
    ) as state:
        store = SqliteBenchStore(state.db)
        record = await store.get_run(args.run_id)
        if record is None:
            print(f"no bench run with id={args.run_id}.")
            return 1
        # Rejudge needs the dataset questions map to
        # render rubrics for cache misses — load the run's dataset by its hash.
        dataset_path = args.dataset or str(config.get(BENCH_DATASET))
        dataset = None
        try:
            dataset = load_bench_dataset(dataset_path)
        except Exception as exc:
            print(f"warning: could not load dataset {dataset_path!r}: {exc}")
        questions = _question_map(dataset.questions) if dataset is not None else None
        judge_model = args.judge_model or str(config.get(EVAL_JUDGE_MODEL))
        judge, judge_gateway = make_judge_llm(state, judge_model)
        try:
            graded = await rejudge_run(
                store,
                judge,
                judge_model,
                args.run_id,
                questions=questions,
                judge_concurrency=args.judge_concurrency or int(BENCH_JUDGE_CONCURRENCY.default),
            )
        finally:
            if judge_gateway is not None:
                with contextlib.suppress(Exception):
                    await judge_gateway.aclose()
        print(f"rejudged {args.run_id}: {graded} pending row(s) graded.")
        return 0


# ── `vesta bench compare` ───────────────────────────────────────────────────


async def _cmd_bench_compare(state: Any, run_a: int, run_b: int) -> int:
    """Print the per-question diff + four buckets for two runs (state = AppState)."""
    from vesta.api.bench import SqliteBenchStore
    from vesta.eval.bench_runner import compare_runs

    store = SqliteBenchStore(state.db)
    ra = await store.get_run(run_a)
    rb = await store.get_run(run_b)
    if ra is None or rb is None:
        print(f"compare: run {run_a if ra is None else run_b} not found.")
        return 1
    comp = await compare_runs(store, run_a, run_b)
    print(f"compare run {run_a} [{ra.system}] vs run {run_b} [{rb.system}]")
    print(
        f"  shared questions: {comp.shared_denominator}  "
        f"(only A: {len(comp.only_a)}, only B: {len(comp.only_b)})"
    )
    print(
        f"  fixed (A wrong→B correct): {len(comp.fixed)}   "
        f"broken (A correct→B wrong): {len(comp.broken)}   "
        f"both correct: {len(comp.both_correct)}   both wrong: {len(comp.both_wrong)}"
    )
    if comp.deltas:
        print("  aggregate deltas (B - A):")
        for k, v in comp.deltas.items():
            print(f"    {k}: {v:+.3f}")
    if comp.broken:
        print("\n  BROKEN (regressions — a mean hides these):")
        for qid in comp.broken:
            print(f"    - {qid}")
    if comp.fixed:
        print("\n  fixed:")
        for qid in comp.fixed:
            print(f"    + {qid}")
    return 0


async def _cmd_bench_compare_cli(args: argparse.Namespace) -> int:
    async with _open_runtime(args.data_dir) as state:
        return await _cmd_bench_compare(state, args.run_a, args.run_b)


# ── `vesta bench list` / `show` ─────────────────────────────────────────────


async def _cmd_bench_list(args: argparse.Namespace) -> int:
    from vesta.api.bench import SqliteBenchStore

    async with _open_runtime(args.data_dir) as state:
        store = SqliteBenchStore(state.db)
        runs = await store.list_runs(limit=args.limit or 50)
        if not runs:
            print("no bench runs yet.")
            return 0
        print(f"{'id':>4}  {'status':<9} {'system':<20} {'profile':<12} {'model':<22} strict")
        for r in runs:
            strict = _metric(r.metrics_json, "answer.strict_accuracy")
            strict_pct = f"{float(strict) * 100:.1f}%" if isinstance(strict, int | float) else "n/a"
            print(
                f"{r.id:>4}  {r.status:<9} {r.system:<20} {r.profile_name or 'default':<12} "
                f"{r.answer_model:<22} {strict_pct}"
            )
    return 0


async def _cmd_bench_show(args: argparse.Namespace) -> int:
    from vesta.api.bench import SqliteBenchStore

    async with _open_runtime(args.data_dir) as state:
        store = SqliteBenchStore(state.db)
        record = await store.get_run(args.run_id)
        if record is None:
            print(f"no bench run with id={args.run_id}.")
            return 1
        print(f"run {record.id}  group {record.run_group}")
        print(
            f"  system={record.system}  profile={record.profile_name or 'default'}  "
            f"model={record.answer_model}  judge={record.judge_model}"
        )
        print(
            f"  status={record.status}  dataset={record.dataset_name}#{record.dataset_hash[:8]}  "
            f"subset={record.subset_hash[:8] if record.subset_hash else ''}"
        )
        print(f"  started={record.started_at}  finished={record.finished_at or ''}")
        if record.abort_reason:
            print(f"  abort_reason={record.abort_reason}")
        m = record.metrics_json
        for key in (
            "answer.strict_accuracy",
            "answer.weighted_accuracy",
            "source.recall_at_1",
            "source.source_coverage",
        ):
            val = _metric(m, key)
            if val is not None:
                print(f"  {key}: {val}")
        tok = _metric(m, "tokens.answer")
        if isinstance(tok, dict) and tok.get("total", 0) > 0:
            print(
                f"  tokens (answer LLM): total={tok.get('total', 0):,} "
                f"(in={tok.get('total_input', 0):,}, out={tok.get('total_output', 0):,})  "
                f"p50/question={tok.get('p50', 0):,}"
            )
        pc = _metric(m, "tokens.peak_context")
        if isinstance(pc, dict) and pc.get("n", 0):
            n = int(pc.get("n", 0))
            over8 = int(pc.get("over_8192", 0))
            over16 = int(pc.get("over_16384", 0))
            print(
                f"  peak context (per-question max request): n={n}  "
                f"p50={int(pc.get('p50', 0)):,}  p90={int(pc.get('p90', 0)):,}  "
                f"p95={int(pc.get('p95', 0)):,}  max={int(pc.get('max', 0)):,}  "
                f">8k={over8} ({over8 / n:.1%})  >16k={over16} ({over16 / n:.1%})  "
                f"requests={int(pc.get('requests_total', 0)):,}  "
                f"overflow_fb={int(pc.get('overflow_fallbacks', 0)):,}"
            )
        rows = await store.list_question_results(args.run_id)
        if rows:
            print(f"  questions: {len(rows)}")
        for row in rows[:20]:
            toks = row.input_tokens + row.output_tokens
            print(
                f"    [{row.question_id}] {row.verdict:<10} "
                f"source={row.source_hit_rank if row.source_hit_rank is not None else 'miss'}  "
                f"tokens={toks:,}"
            )
    return 0


# ── `vesta bench verify` — oracle + closed-book dataset verification ────────


async def _cmd_bench_verify(args: argparse.Namespace) -> int:
    """Three passes: support check (no LLM), closed-book, oracle → review file."""
    from vesta.api.bench import make_judge_llm
    from vesta.eval.bench_dataset import BENCH_DATASET, load_bench_dataset
    from vesta.eval.bench_runner import (
        BENCH_JUDGE_CONCURRENCY,
        BENCH_MAX_CONCURRENT,
        resolve_judge_concurrency,
    )
    from vesta.eval.bench_scoring import Verdict
    from vesta.eval.golden import EVAL_JUDGE_ENDPOINT_URL, EVAL_JUDGE_MODEL
    from vesta.inference import INFERENCE_LLM_ENDPOINT_URL, INFERENCE_LLM_MODEL

    overrides = _build_apply_overrides(args)
    async with _open_runtime(
        args.data_dir, with_gateway=True, settings_overrides=overrides or None
    ) as state:
        answer_model = args.model or str(config.get(INFERENCE_LLM_MODEL))
        if not answer_model:
            # Bench verify needs an answer model; the fresh-install default is empty.
            raise SystemExit("no model configured; pass --model or set inference.llm.model")
        dataset = load_bench_dataset(args.dataset or str(config.get(BENCH_DATASET)))
        qs = list(dataset.questions)
        if args.limit is not None:
            qs = qs[: args.limit]
        judge_model = args.judge_model or str(config.get(EVAL_JUDGE_MODEL))
        judge, judge_gateway = make_judge_llm(state, judge_model)
        # Stage bounds — same machinery as `bench run` (semaphore + ordered
        # gather). Defaults mirror each stage's sibling: pipeline questions
        # bench.max_concurrent (verify reports no latency, so raise freely via
        # --concurrency), judges bench.judge.concurrency clamped to 1 when the
        # judge shares the answer endpoint, archive ops the retrieval fan-out.
        answer_concurrency = max(1, args.concurrency or int(BENCH_MAX_CONCURRENT.default))
        judge_concurrency, _shares = resolve_judge_concurrency(
            args.judge_concurrency or int(BENCH_JUDGE_CONCURRENCY.default),
            answer_endpoint=str(config.get(INFERENCE_LLM_ENDPOINT_URL)),
            judge_endpoint=str(config.get(EVAL_JUDGE_ENDPOINT_URL)),
        )
        try:
            # Pass 1: support check (no LLM) — answer tokens present in each
            # required source's extracted article body.
            support = await _verify_support(state, qs)
            # Pass 2: closed-book (floor).
            closed = await _verify_pass(
                state, qs, "closed_book", answer_model, max_concurrent=answer_concurrency
            )
            # Pass 3: oracle (ceiling).
            oracle = await _verify_pass(
                state, qs, "oracle", answer_model, max_concurrent=answer_concurrency
            )
            # Judge both passes.
            closed_judged = await _verify_judge(
                judge, judge_model, qs, closed, support, max_concurrent=judge_concurrency
            )
            oracle_judged = await _verify_judge(
                judge, judge_model, qs, oracle, support, max_concurrent=judge_concurrency
            )
        finally:
            if judge_gateway is not None:
                with contextlib.suppress(Exception):
                    await judge_gateway.aclose()

        _write_verification_review(
            dataset, qs, support, closed, closed_judged, oracle, oracle_judged, answer_model
        )

        # Derived checks over active questions.
        active = [q for q in qs if q.status == "active"]
        if active:
            oracle_correct = sum(
                1 for q in active if oracle_judged[q.id].verdict == Verdict.CORRECT
            )
            ceiling = oracle_correct / len(active)
            non_lookup = [q for q in active if q.capability != "lookup"]
            cb_correct = sum(
                1 for q in non_lookup if closed_judged[q.id].verdict == Verdict.CORRECT
            )
            floor = cb_correct / len(non_lookup) if non_lookup else 0.0
            print(
                f"ceiling (oracle) ≥ 85%? {ceiling * 100:.1f}% "
                f"{'PASS' if ceiling >= 0.85 else 'FAIL'}"
            )
            print(
                f"floor (closed-book, excl lookup) ≤ 20%? {floor * 100:.1f}% "
                f"{'PASS' if floor <= 0.20 else 'FAIL'}"
            )
        print("wrote benchmarks/verification review files (.md + .json).")
        return 0


async def _verify_support(
    state: Any, qs: Sequence[Any], *, max_concurrent: int | None = None
) -> dict[str, bool]:
    """Pass 1: the answer's distinctive tokens appear in each required source.

    Questions run concurrently bounded by ``max_concurrent`` (default:
    ``retrieval.max_archives_concurrent`` — the pipeline's own archive fan-out,
    so libzim never sees more concurrent readers than a normal query allows).
    Sources stay sequential *within* a question so the token short-circuit
    still skips extraction exactly like the serial implementation did.
    """
    from vesta.retrieval import RETRIEVAL_MAX_ARCHIVES_CONCURRENT

    def _tokens(text: str) -> set[str]:
        return {w for w in _tokenize(text) if len(w) >= 4}

    answer_tokens: dict[str, set[str]] = {q.id: _tokens(q.answer) for q in qs}
    bound = max(1, int(max_concurrent or int(config.get(RETRIEVAL_MAX_ARCHIVES_CONCURRENT))))
    sem = asyncio.Semaphore(bound)

    async def _one(q: Any) -> tuple[str, bool]:
        async with sem:
            ok = True
            for src in q.sources:
                if not src.required:
                    continue
                text = await _extract_article(state, q, src)
                at = answer_tokens[q.id]
                if at and not (at & _tokens(text)):
                    ok = False
                    break
            return q.id, ok

    return dict(await asyncio.gather(*(_one(q) for q in qs)))


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch.lower())
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


async def _extract_article(state: Any, q: Any, src: Any) -> str:
    """Extract a source article's text via the archive registry."""
    if state.registry is None:
        return ""
    try:
        async with state.db.read() as conn:
            cur = await conn.execute(
                "SELECT id FROM zims WHERE filename=? OR name=? LIMIT 1", (src.zim, src.zim)
            )
            row = await cur.fetchone()
        if row is None:
            return ""
        arc = state.registry.get(int(row[0]))
        article = await arc.extract(src.article_path)
        return str(article.text)
    except Exception:
        return ""


async def _verify_pass(
    state: Any,
    qs: Sequence[Any],
    system: str,
    model: str,
    *,
    max_concurrent: int | None = None,
) -> dict[str, str]:
    """Run the closed-book or oracle system over each question → answer text.

    Questions run concurrently bounded by ``max_concurrent`` (default:
    ``bench.max_concurrent``, the same knob ``bench run`` uses for pipeline
    questions); gather keeps results keyed in input order either way.
    """
    from vesta.api.bench import make_system
    from vesta.eval.bench_runner import BENCH_MAX_CONCURRENT

    sut = make_system(system, state, model_id=model)
    sem = asyncio.Semaphore(max(1, int(max_concurrent or int(BENCH_MAX_CONCURRENT.default))))

    async def _one(q: Any) -> tuple[str, str]:
        async with sem:
            try:
                output = await sut.run_one(q)
                return q.id, output.answer_text
            except Exception:
                return q.id, ""

    return dict(await asyncio.gather(*(_one(q) for q in qs)))


async def _verify_judge(
    judge: Any,
    judge_model: str,
    qs: Sequence[Any],
    answers: dict[str, str],
    support: dict[str, bool],
    *,
    max_concurrent: int | None = None,
) -> dict[str, Any]:
    """Grade each pass's answers via the structured rubric.

    Judgments run concurrently bounded by ``max_concurrent`` (default:
    ``bench.judge.concurrency``; `_cmd_bench_verify` pre-clamps it to 1 when
    the judge shares the answer endpoint, via ``resolve_judge_concurrency``).
    """
    from vesta.eval.bench_runner import BENCH_JUDGE_CONCURRENCY
    from vesta.eval.bench_scoring import judge_verdict

    sem = asyncio.Semaphore(max(1, int(max_concurrent or int(BENCH_JUDGE_CONCURRENCY.default))))

    async def _one(q: Any) -> tuple[str, Any]:
        ans = answers.get(q.id, "")
        async with sem:
            outcome = await judge_verdict(
                question=q,
                model_answer=ans,
                abstained=not bool(ans.strip()),
                judge=judge,
                judge_model=judge_model,
            )
        return q.id, outcome

    return dict(await asyncio.gather(*(_one(q) for q in qs)))


def _write_verification_review(
    dataset: Any,
    qs: Sequence[Any],
    support: dict[str, bool],
    closed_answers: dict[str, str],
    closed_judged: dict[str, Any],
    oracle_answers: dict[str, str],
    oracle_judged: dict[str, Any],
    model: str,
) -> None:
    """Write benchmarks/verification/<date>-review.{md,json} — one section per question."""
    import datetime as _dt

    out_dir = Path("benchmarks/verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    date = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    path = out_dir / f"{date}-review.md"
    lines = [
        f"# Bench verification review — {date}",
        "",
        f"dataset: {dataset.name} #{dataset.hash}  model: {model}",
        "",
    ]
    # Machine-readable twin: per-question verdicts + answers, consumed by
    # scripts/bench_authoring/bake_reference.py to bake oracle/closed_book
    # blocks into the dataset (hash-neutral by design — see bench_dataset.py).
    json_qs: dict[str, object] = {}
    for q in qs:
        cb_ans = closed_answers.get(q.id, "")
        ob_ans = oracle_answers.get(q.id, "")
        cb = closed_judged.get(q.id)
        ob = oracle_judged.get(q.id)
        lines += [
            f"## {q.id} [{q.capability}/{q.difficulty}]",
            f"- question: {q.question}",
            f"- ground truth: {q.answer}",
            f"- support check: {'PASS' if support.get(q.id) else 'FAIL'}",
            f"- closed-book answer: {cb_ans or '(none)'}",
            f"- closed-book verdict: {cb.verdict.value} — {cb.reason}"
            if cb
            else "- closed-book verdict: (unjudged)",
            f"- oracle answer: {ob_ans or '(none)'}",
            f"- oracle verdict: {ob.verdict.value} — {ob.reason}"
            if ob
            else "- oracle verdict: (unjudged)",
            "",
        ]
        json_qs[q.id] = {
            "capability": q.capability,
            "support": bool(support.get(q.id)),
            "closed_book": {
                "verdict": cb.verdict.value if cb else "unjudged",
                "answer": cb_ans,
                "reason": cb.reason if cb else "",
            },
            "oracle": {
                "verdict": ob.verdict.value if ob else "unjudged",
                "answer": ob_ans,
                "reason": ob.reason if ob else "",
            },
        }
    path.write_text("\n".join(lines), encoding="utf-8")
    json_path = out_dir / f"{date}-review.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": {"name": dataset.name, "hash": dataset.hash},
                "model": model,
                "generated": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
                "questions": json_qs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ── Shared lifespan: open DB + registry the way main.py does ────────────────


@asynccontextmanager
async def _open_runtime(  # noqa: PLR0915
    data_dir: str | None,
    *,
    with_gateway: bool = False,
    with_index_runtime: bool = False,
    settings_overrides: Mapping[str, str] | None = None,
) -> AsyncIterator[AppState]:
    """Boot the DB + settings + archive registry for a CLI run, then tear down.

    Mirrors ``main.lifespan`` minus the HTTP server: the CLI needs the same
    wired state (DB, settings, open archives) but is short-lived. ``with``
    guarantees archives close even on error.

    ``with_gateway`` additionally constructs the inference gateway
    and binds it — needed by ``bench run`` (the answer pipeline requires a
    gateway). Both default off so ``bench retrieval`` /
    ``bench hardware`` are unaffected.

    ``with_index_runtime`` (used by ``vesta index``) binds the index job's
    runtime singletons (db, registry, embedder provider) so ``IndexZimJob`` can
    run in-process. No preemption coordinator is bound — a CLI run has no
    interactive requests to yield to, and ``job.py`` treats a ``None``
    coordinator as "run flat out".

    ``settings_overrides`` (used by ``vesta bench run``) is merged on top of the
    DB-stored settings before the resolver is seeded, so the gateway + judge
    endpoint pick up CLI-flag overrides (answer model/endpoint, judge
    endpoint/api-key). Resolution order is unchanged: DB values (now including
    the overrides) still win over code defaults.
    """
    config.configure()
    # The --data-dir override applies directly as the on-disk path (NOT via
    # os.environ — forbids env reads outside config/). The
    # settings resolver keeps its own view; only the DB/archive location follows.
    ddir = Path(data_dir or str(config.get(config.DATA_DIR)))
    ddir.mkdir(parents=True, exist_ok=True)
    db = Database(
        str(ddir / "vesta.db"), busy_timeout_ms=int(config.get(config.DB_BUSY_TIMEOUT_MS))
    )
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    async with db.read() as conn:
        stored = await load_settings(conn)
    if settings_overrides:
        stored = {**stored, **settings_overrides}
    config.set_db_values(stored)

    registry = ArchiveRegistry(
        db=db,
        zims_dir=ddir / "zims",
        read_pool_size=int(config.get(config.ZIM_READ_POOL_SIZE)),
        cluster_cache_mb=int(config.get(config.ZIM_CLUSTER_CACHE_MB)),
    )
    with contextlib.suppress(Exception):
        await registry.start()
    bind_registry(registry)
    encoders = build_manager_from_settings(config.snapshot(), model_dir=ddir / "models")
    bind_manager(encoders)
    # The CLI is a composition root too — without this, ``vesta eval``/
    # ``vesta bench`` always capability-drop ``vector_knn``. Mirrors main.py's lifespan wiring.
    vector_store = SqliteVecStore(
        db,
        quantizer=str(config.get(VECTORS_QUANTIZER)),
        oversample=int(config.get(VECTORS_OVERSAMPLE)),
    )
    await vector_store.ensure_default_table()
    bind_store(vector_store)
    # Capability.VECTORS is gated on a separate "any archive indexed" flag,
    # not merely on the store being bound (index/__init__.py) — main.py seeds
    # it at startup; the CLI needs the same seeding or vector_knn always
    # capability-drops regardless of the store wiring above.
    try:
        await reseed_indexed_state(db)
    except Exception:
        set_indexed_state(False)

    if with_index_runtime:
        # Wire the index job's runtime singletons (db, registry, embedder
        # provider) so ``IndexZimJob`` can run in-process — mirrors main's
        # ``bind_runtime``. Importing the job module also registers the
        # ``index_zim`` job type (import side effect), matching main.py.
        from vesta.index import job as _index_job  # noqa: F401

        _index_embedder_repo = str(config.get(INDEX_EMBEDDER))

        async def _cli_index_embedder_provider() -> Any:
            return await encoders.get_embed_for(_index_embedder_repo)

        bind_index_runtime(db, registry, _cli_index_embedder_provider)

    # Optionally build the inference gateway so the answer benchmark
    # can actually run the agentic_pydantic system.
    gateway: Any = None
    supervisor: Any = None
    if with_gateway:
        from vesta.inference import bind_gateway, build_gateway_from_settings

        try:
            gateway, supervisor = build_gateway_from_settings(config.snapshot(), data_dir=ddir)
            bind_gateway(gateway, supervisor)
        except Exception:
            gateway = None

    try:
        yield AppState(
            db=db,
            runner=None,  # type: ignore[arg-type]
            registry=registry,
            encoders=encoders,
            gateway=gateway,
            supervisor=supervisor,
        )
    finally:
        bind_registry(None)
        bind_manager(None)
        bind_store(None)
        set_indexed_state(False)
        if gateway is not None:
            from vesta.inference import bind_gateway

            bind_gateway(None, None)
            with contextlib.suppress(Exception):
                await gateway.aclose()
        await registry.stop()
        await db.stop()


def _resolve_profile(state: AppState, name: str) -> RetrievalProfile:
    """Resolve a profile name (user-saved shadow builtins), fall back to lexical."""
    del state  # registry not needed for profile resolution
    p = resolve_profile(name)
    if p is not None:
        return p
    return load_profile("lexical")  # type: ignore[return-value]


# The API runner is the single pipeline-runner implementation (the CLI's
# former near-verbatim clone could silently drift and break the
# CLI↔API comparability of bench numbers). ``profile`` is accepted-and-ignored
# so both historical call shapes work.
CLIPipelineRunner = LivePipelineRunner


# ── `vesta bench retrieval` ─────────────────────────────────────────────────────────


async def _cmd_eval(args: argparse.Namespace) -> int:
    action: str = args.action
    async with _open_runtime(args.data_dir) as state:
        if args.dataset is not None:
            # Dataset mode — the A/D/B article-recall arms.
            return await _dataset_recall_run(state, args)
        if args.profile is None:
            # Golden-set default. The flag's argparse default is None so
            # dataset mode can reject an explicit --profile (the arms pin
            # their own profiles; see _dataset_recall_run).
            args.profile = "lexical"
        if action == "verify-golden":
            failures = await _verify_golden(state, golden=args.golden)
            if failures:
                print(f"FAIL: {len(failures)} golden entries did not verify:")
                for f in failures:
                    print(f"  - {f}")
                return 1
            print(
                f"OK: {len(load_set(args.golden).entries)} golden entries verified against the archive."
            )
            return 0
        if action == "calibrate":
            return await _calibrate(state, args)
        if action == "regression":
            return await _regression(state, args)
        return await _eval_run(state, args)


def _verify_golden(state: AppState, *, golden: str) -> Any:
    """Async: confirm every golden path resolves + fact present (honesty check).

    Pre-fetches the expected articles' text (archive reads are async), then
    hands a sync lookup to ``verify_against_archive``. The lookup maps any
    expected path to its extracted text across every enabled archive.
    """
    gs = load_set(golden)

    async def _fetch() -> dict[str, str]:
        out: dict[str, str] = {}
        if state.registry is None:
            return out
        for arc in state.registry.enabled():
            for entry in gs.entries:
                for path in entry.expected_paths:
                    if path in out:
                        continue
                    if not _archive_has_path(arc, path):
                        continue
                    try:
                        art = await arc.extract(path)
                        out[path] = art.text
                    except Exception:
                        pass
        return out

    async def _run() -> list[str]:
        texts = await _fetch()
        return verify_against_archive(gs, texts.get)

    return _run()


def _archive_has_path(arc: Any, path: str) -> bool:
    """Probe whether an open archive resolves ``path`` (handles namespace schemes)."""
    # eval never imports zim; the CLI may touch the archive object's methods.
    with contextlib.suppress(Exception):
        # The registry's archives expose a low-level libzim archive via _lz.
        lz = getattr(arc, "_lz", None)
        if lz is not None and lz.has_entry_by_path(path):
            return True
    return False


async def _eval_run(state: AppState, args: argparse.Namespace) -> int:
    profile = _resolve_profile(state, args.profile)
    golden = load_set(args.golden)
    runner = CLIPipelineRunner(state)

    if args.sweep:
        return await _run_sweep(state, args, profile, golden, runner)

    metrics, results = await evaluate_profile(profile, runner, golden)
    _print_run_report(profile, golden, metrics)
    run_id: int | None = None
    if not args.no_persist:
        store = SqliteEvalStore(state.db)
        run_id = await persist_run(
            store,
            profile=profile,
            golden=golden,
            metrics=metrics,
            results=results,
            settings_snapshot=dict(config.snapshot().values),
            archive_path=str(EVAL_ARCHIVE_PATH.default),
            archive_checksum=str(EVAL_ARCHIVE_CHECKSUM.default),
            notes="cli",
        )
        print(f"\n persisted run id={run_id}")
    if args.baseline:
        await _print_baseline_delta(
            state, args, profile, golden, runner, metrics=metrics, results=results
        )
    return 0


# ── `vesta bench retrieval --dataset` (article-recall arms) ───────


def _print_recall_progress(done: int, total: int, row: QuestionRecall) -> None:
    """One line per question: the three arm ranks (like the probe it replaces)."""
    ranks = "  ".join(
        f"{key}={row.arms[key].rank if row.arms[key].rank is not None else '-'}"
        for key in ("A", "D", "B")
    )
    print(f"[{done}/{total}] {row.question_id}  {ranks}  {row.question[:64]}", flush=True)


def _write_article_recall_artifact(
    path_str: str | None,
    report: ArticleRecallReport,
    dataset: BenchDataset,
    n_selected: int,
) -> Path:
    """Write the per-question JSON artifact (benchmarks/results/ by default)."""
    import datetime as _dt

    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    payload: dict[str, object] = {
        "generated": now,
        "dataset": {
            "name": dataset.name,
            "hash": dataset.hash,
            "questions_selected": n_selected,
        },
        **report.to_dict(),
    }
    if path_str is None:
        out_dir = Path("benchmarks/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{stamp}-round0-article-recall.json"
    else:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


async def _dataset_recall_run(state: AppState, args: argparse.Namespace) -> int:
    """The fixed A/D/B article-recall arms over the bench dataset.

    Zero LLM, zero persistence — the output is the printed table + a JSON
    artifact carrying per-question per-arm ranks.
    """
    from vesta.eval.article_recall import ARMS, evaluate_article_recall, select_recall_questions

    if args.action != "run":
        print(f"--dataset mode has no {args.action!r} sub-action; use plain --dataset PATH.")
        return 2
    for flag, value in (
        ("--profile", args.profile),
        ("--sweep", args.sweep),
        ("--baseline", args.baseline),
    ):
        if value:
            print(f"--dataset mode runs the fixed A/D/B arms; {flag} is a golden-set flag.")
            return 2

    dataset, qs = _resolve_questions(args)
    questions = select_recall_questions(qs)
    if not questions:
        print(
            "no source-eligible questions (need expected_behavior=answer with >=1 "
            "required source; check --level/--capability/--limit)."
        )
        return 1

    profiles: dict[str, RetrievalProfile] = {}
    for name in sorted({arm.profile for arm in ARMS}):
        p = load_profile(name)
        if p is None:
            print(f"recall arm profile {name!r} is not a built-in profile.")
            return 1
        profiles[name] = p

    arm_pins = "  ".join(
        f"{arm.key}={arm.profile}#{profiles[arm.profile].hash[:8]}" for arm in ARMS
    )
    print(
        f"dataset {dataset.name}#{dataset.hash[:8]}  {len(questions)} source-eligible "
        f"questions\narms {arm_pins}"
    )
    report = await evaluate_article_recall(
        questions, CLIPipelineRunner(state), profiles, progress=_print_recall_progress
    )
    print()
    print(report.render())
    if report.degraded:
        # With VECTORS unmet, arm D silently degrades to arm A.
        print(
            f"warning: degraded pipeline runs in arm(s) {', '.join(report.degraded)} — "
            "arm D is not measuring the dense path."
        )
    out = _write_article_recall_artifact(args.out, report, dataset, len(questions))
    print(f"\nwrote {out}  (dataset mode never writes eval_runs/bench_runs)")
    return 0


async def _run_sweep(
    state: AppState,
    args: argparse.Namespace,
    profile: RetrievalProfile,
    golden: GoldenSet,
    runner: PipelineRunner,
) -> int:
    """Run ``--sweep impl.k=v1,v2,...`` and report one row per value."""
    points = parse_sweep(args.sweep, profile)
    print(f"sweep {args.sweep}: {len(points)} values\n")
    print(f"{'value':>10}  recall@10  ndcg@10    mrr")
    rows: list[tuple[SweepPoint, float, float, float]] = []
    for pt in points:
        metrics, _ = await evaluate_profile(pt.profile, runner, golden)
        sm = metrics.slice("all")
        d = sm.to_dict()
        r10 = float(d["recall@10"])
        nd = float(d["ndcg@10"])
        mrr = float(d["mrr"])
        rows.append((pt, r10, nd, mrr))
        print(f"{pt.value:>10}  {r10:.3f}     {nd:.3f}    {mrr:.3f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest recall@10: {best[0].label} ({best[1]:.3f})")
    return 0


async def _print_baseline_delta(
    state: AppState,
    args: argparse.Namespace,
    profile: RetrievalProfile,
    golden: GoldenSet,
    runner: PipelineRunner,
    *,
    metrics: Any,
    results: Any,
) -> int:
    """Resolve the baseline (run id or profile), run if a profile name, diff."""
    store = SqliteEvalStore(state.db)
    baseline = await _resolve_baseline(state, args, store, golden, runner)
    if baseline is None:
        print(f"baseline {args.baseline!r} not found.")
        return 1
    # Build a candidate RunRecord in-memory from the just-computed run.
    from vesta.eval.runner import RunRecord
    from vesta.retrieval.profiles import profile_to_yaml

    candidate = RunRecord(
        id=0,
        started_at="",
        profile_name=profile.name,
        profile_hash=profile.hash,
        profile_yaml=profile_to_yaml(profile),
        golden_hash=golden.hash,
        archive_path=str(EVAL_ARCHIVE_PATH.default),
        archive_checksum=str(EVAL_ARCHIVE_CHECKSUM.default),
        settings_snapshot={},
        git_sha="",
        machine_id="",
        metrics=metrics,
        per_query=tuple(r.to_dict() for r in results),
        notes="cli candidate",
    )
    comp = compare(baseline, candidate, force=args.force)
    _print_comparison(comp, explain=args.explain)
    return 0


async def _resolve_baseline(
    state: AppState,
    args: argparse.Namespace,
    store: SqliteEvalStore,
    golden: GoldenSet,
    runner: PipelineRunner,
) -> RunRecord | None:
    """A baseline may be a numeric run id or a profile name (run on the fly)."""
    baseline_spec = args.baseline or ""
    if baseline_spec.isdigit():
        return await store.get_run(int(baseline_spec))
    base_profile = _resolve_profile(state, baseline_spec)
    bm, br = await evaluate_profile(base_profile, runner, golden)
    from vesta.eval.runner import RunRecord
    from vesta.retrieval.profiles import profile_to_yaml

    return RunRecord(
        id=-1,
        started_at="(on-the-fly)",
        profile_name=base_profile.name,
        profile_hash=base_profile.hash,
        profile_yaml=profile_to_yaml(base_profile),
        golden_hash=golden.hash,
        archive_path=str(EVAL_ARCHIVE_PATH.default),
        archive_checksum=str(EVAL_ARCHIVE_CHECKSUM.default),
        settings_snapshot={},
        git_sha="",
        machine_id="",
        metrics=bm,
        per_query=tuple(r.to_dict() for r in br),
        notes="on-the-fly baseline",
    )


def _print_run_report(profile: RetrievalProfile, golden: GoldenSet, metrics: Any) -> None:
    """The reporting format: header + metric table + latency + degradation."""
    print(f"run: {profile.name}#{profile.hash[:6]}  golden={golden.name}#{golden.hash[:8]}")
    print(f"archive={golden.archive_path}  checksum={golden.archive_checksum[:12]}...")
    slices = [
        "all",
        "entity",
        "paraphrase",
        "multi_hop",
        "deep_content",
        "keyword",
        "out_of_corpus",
        "reformulation",
    ]
    header = f"{'metric':<16} " + " ".join(f"{s:>13}" for s in slices)
    print(header)
    for metric_name in ("recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr"):
        row = f"{metric_name:<16} "
        for sl in slices:
            sm = metrics.slice(sl)
            val = sm.to_dict().get(metric_name, 0.0)
            row += f" {float(val):>12.3f}"
        print(row)
    lat = metrics.latency_ms
    p50 = lat.stage_p50
    stage_str = "  ".join(f"{n}:{v:.0f}" for n, v in sorted(p50.items()))
    print(f"stage p50 (ms)   {stage_str}   total:{lat.total_p50:.0f}")
    deg = "none" if not metrics.degraded else ", ".join(metrics.degraded_components)
    print(f"degraded: {deg}")
    counts = "  ".join(f"{s}={len(golden.by_slice().get(s, []))}" for s in slices[1:])
    print(f"counts: {counts}")


def _print_comparison(comp: Comparison, *, explain: bool) -> None:
    """Delta table + win/loss summary + optional per-query list."""
    if comp.degraded_guard:
        print(f"\n⚠ degradation guard: {comp.degraded_guard}")
        return
    print(f"\nvs baseline {comp.baseline.profile_name}#{comp.baseline.profile_hash[:6]}:")
    print(
        f"{'metric':<16} {'overall':>10} "
        + " ".join(
            f"{s:>11}"
            for s in (
                "entity",
                "paraphrase",
                "multi_hop",
                "deep_content",
                "keyword",
                "reformulation",
            )
        )
    )
    for d in comp.metric_deltas:
        sl = " ".join(
            f"{d.slices.get(s, 0.0):>+10.3f}"
            for s in (
                "entity",
                "paraphrase",
                "multi_hop",
                "deep_content",
                "keyword",
                "reformulation",
            )
        )
        print(f"{d.metric:<16} {d.overall:>+10.3f} {sl}")
    print(f"  wins {comp.wins}  losses {comp.losses}  unchanged {comp.unchanged}")
    if explain:
        losses = [q for q in comp.query_deltas if q.status == "loss"]
        wins = [q for q in comp.query_deltas if q.status == "win"]
        if losses:
            print("\nlosses:")
            for q in losses:
                print(
                    f"  [{q.slice}] {q.query}  (baseline rank {q.baseline_rank} -> {q.candidate_rank})"
                )
        if wins:
            print("\nwins:")
            for q in wins:
                print(
                    f"  [{q.slice}] {q.query}  (baseline rank {q.baseline_rank} -> {q.candidate_rank})"
                )


# ── `vesta bench retrieval calibrate` ──────────────────────────────────────────────────


async def _calibrate(state: AppState, args: argparse.Namespace) -> int:
    """Fit confidence thresholds against the golden set; report achieved rho."""
    profile = _resolve_profile(state, args.profile)
    golden = load_set(args.golden)
    runner = CLIPipelineRunner(state)
    metrics, results = await evaluate_profile(profile, runner, golden)
    samples: list[ConfidenceSample] = []
    # Lift the confidence signals off each result; here we read the per-query
    # hit + slice.
    for r in results:
        # The full ConfidenceSignals live on the RetrievalResult; the runner kept
        # only paths+trace. Re-derive a coarse density from the retrieved set as
        # the depth-0 signal (scores absent). This is honest: calibration at
        # depth 0 is density-driven.
        retrieved = r.retrieved_paths
        from collections import Counter

        c = Counter(retrieved)
        density = (c.most_common(1)[0][1] / len(retrieved)) if retrieved else 0.0
        samples.append(
            ConfidenceSample(
                slice=r.entry.slice,
                top_score=None,
                score_dropoff=None,
                density=density,
                agreement=0.0,
                hit=r.hit_rank is not None,
            )
        )
    del metrics
    result = fit_thresholds(samples)
    print("Confidence-gate calibration (target rho ~= 0.25):")
    print(json.dumps(result.to_dict(), indent=2))
    # The fitted thresholds are *written to settings*, not just printed.
    # The four retrieval.confidence.* settings are hot=True, so the running app
    # picks them up immediately. Caller holds the write transaction.
    import datetime as _dt

    from vesta.db.settings_store import upsert_setting

    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    fitted = result.to_dict()["thresholds"]
    assert isinstance(fitted, dict)  # CalibrationResult.to_dict guarantees this
    async with state.db.write() as conn:
        for key, value in fitted.items():
            await upsert_setting(conn, str(key), str(value), now)
    async with state.db.read() as conn:
        from vesta.db.settings_store import load_settings

        config.set_db_values(await load_settings(conn))
    print("fitted thresholds written to settings:")
    print(json.dumps(fitted, indent=2))
    return 0


# ── `vesta bench retrieval regression` ─────────────────────────────────────────────────


async def _regression(state: AppState, args: argparse.Namespace) -> int:
    """Run the regression gate: active vs baseline, fail on a drop > epsilon.

    The CI-runnable gate uses the ``fixture_subset`` (tiny ZIM) so it executes
    without the gitignored pinned archive; pass ``--golden full`` for the
    nightly/on-demand gate where the pinned archive is present.
    """
    golden = load_set(args.golden)
    runner = CLIPipelineRunner(state)
    baseline_profile = _resolve_profile(state, args.baseline or "lexical")
    candidate_profile = _resolve_profile(state, args.profile)
    bm, br = await evaluate_profile(baseline_profile, runner, golden)
    cm, cr = await evaluate_profile(candidate_profile, runner, golden)
    from vesta.eval.runner import RunRecord

    baseline_rec = RunRecord(
        id=-1,
        started_at="",
        profile_name=baseline_profile.name,
        profile_hash=baseline_profile.hash,
        profile_yaml="",
        golden_hash=golden.hash,
        archive_path=golden.archive_path,
        archive_checksum=golden.archive_checksum,
        settings_snapshot={},
        git_sha="",
        machine_id="",
        metrics=bm,
        per_query=tuple(r.to_dict() for r in br),
        notes="baseline",
    )
    candidate_rec = RunRecord(
        id=-2,
        started_at="",
        profile_name=candidate_profile.name,
        profile_hash=candidate_profile.hash,
        profile_yaml="",
        golden_hash=golden.hash,
        archive_path=golden.archive_path,
        archive_checksum=golden.archive_checksum,
        settings_snapshot={},
        git_sha="",
        machine_id="",
        metrics=cm,
        per_query=tuple(r.to_dict() for r in cr),
        notes="candidate",
    )
    epsilon = float(EVAL_REGRESSION_EPSILON.default)
    decision = gate_evaluate(baseline_rec, candidate_rec, epsilon=epsilon)
    tag = "fixture" if golden.name == "fixture_subset" else "full"
    print(
        f"regression gate [{tag} set]: baseline={baseline_profile.name} candidate={candidate_profile.name}"
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0 if decision.passed else 1


# ── `vesta models` ──────────────────────────────────────────────────────────


def _cmd_models(args: argparse.Namespace) -> int:
    """Download the configured static/embed/rerank models into encoders.model_dir.

    Dev/install-time only — never called from the request path (encoders/
    module docstring). Reads settings default+env only (no DB layer needed for
    a one-shot fetch); pass --data-dir to control where models land.
    """
    config.configure()
    ddir = Path(args.data_dir or str(config.get(config.DATA_DIR)))
    model_dir = ddir / "models"
    from vesta.encoders import (
        ENCODERS_EMBED_MODEL,
        ENCODERS_RERANK_MODEL,
        ENCODERS_STATIC_MODEL,
    )

    role_settings = {
        "static": ENCODERS_STATIC_MODEL,
        "embed": ENCODERS_EMBED_MODEL,
        "rerank": ENCODERS_RERANK_MODEL,
    }
    roles = args.role or ["static", "embed", "rerank"]
    for role in roles:
        repo_id = str(config.get(role_settings[role]))
        spec = MODEL_SPECS.get(repo_id)
        if spec is None:
            print(f"skip {role}: {repo_id!r} not in the model registry")
            continue
        print(f"fetching {role}: {repo_id} -> {model_dir / repo_id}")
        dest = ensure_model(spec, model_dir)
        print(f"  done: {dest}")
    return 0


# ── `vesta bench` ───────────────────────────────────────────────────────────


async def _run_encoder_bench(args: argparse.Namespace) -> list[dict[str, object]]:
    """Real encoder rows when models are on disk (``vesta models``); each row
    falls back to a ``DeferredRow`` on its own if that particular
    role's model is absent — a missing reranker doesn't block the embedder row.

    Dispatched via ``asyncio.to_thread`` because ``eval.bench.encoder``'s
    functions call ``asyncio.run()`` internally (that module
    must not import ``vesta.encoders``, so it can't ``await`` a real ``Encoder``
    directly) — and ``_cmd_bench`` is itself already inside a running loop.
    """
    config.configure()
    ddir = Path(args.data_dir or str(config.get(config.DATA_DIR)))
    mgr = build_manager_from_settings(config.snapshot(), model_dir=ddir / "models")
    embed_enc = await mgr.get_embed()
    rerank_enc = await mgr.get_rerank()
    rows: list[dict[str, object]] = []
    rows.append((await asyncio.to_thread(encoder.embedder_throughput, embed_enc)).to_row())
    rows.append((await asyncio.to_thread(encoder.reranker_latency, rerank_enc)).to_row())
    # int8-vs-fp32 needs a second (fp32) session over the same graph, which
    # `vesta models` does not fetch by default (the fp32 export is 4x the
    # download for a number this project only needs once) — stays deferred.
    rows.append(encoder.onnx_int8_speedup().to_row())
    return rows


async def _cmd_bench(args: argparse.Namespace) -> int:
    """Run the hardware benchmark harness and write bench_results/<machine>-<date>.md."""
    import datetime as _dt

    rows: list[dict[str, object]] = []
    cpu = hardware.measure_cpu_info()
    rows.append(hardware.measure_gemm_ceiling().to_row())
    rows.append(hardware.measure_memory_bandwidth().to_row())
    rows.extend(await _run_encoder_bench(args))

    archive_path = args.archive
    if archive_path is None:
        # Default to the pinned archive if present.
        default = Path("data/zims") / str(EVAL_ARCHIVE_PATH.default)
        archive_path = str(default) if default.exists() else None

    extraction_rows: list[dict[str, object]] = []
    if not args.skip_extraction and archive_path and Path(archive_path).exists():
        extraction_rows = _run_extraction_bench(archive_path)
        rows.extend(extraction_rows)
    elif not args.skip_extraction:
        rows.append(
            {
                "name": "Extraction (threads/processes)",
                "value": None,
                "unit": "MB/s",
                "projection": "threads scale negatively",
                "projection_source": "measured 8-thread datapoint",
                "verdict": "skipped — no archive path provided",
                "notes": "Pass --archive <path.zim> to run extraction throughput.",
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    mid = cpu["machine_id"]
    out_path = out_dir / f"{mid}-{today}.md"
    _write_bench_md(out_path, rows, cpu, include_extraction=bool(extraction_rows))
    print(f"wrote {out_path}")
    return 0


def _run_extraction_bench(archive_path: str) -> list[dict[str, object]]:
    """Fetch a sample of article HTML from the archive, then time threads/processes."""
    import time

    from libzim.reader import Archive as LibzimArchive

    a = LibzimArchive(archive_path)
    # Gather ~200 real article paths by sampling random entries.
    sample_paths: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(sample_paths) < 200 and tries < 4000:
        tries += 1
        try:
            e = a.get_random_entry()
            p = str(e.path)
            if p in seen or p.startswith("I/"):
                continue
            seen.add(p)
            sample_paths.append(p)
        except Exception:
            continue
    # Pre-fetch raw HTML for the thread benchmark (processes open their own copy).
    from vesta.zim.extract import extract_article
    from vesta.zim.reader import read_entry_sync

    htmls: list[bytes] = []
    for p in sample_paths[:200]:
        try:
            raw = read_entry_sync(a, p)
            if raw.is_redirect or not raw.content:
                continue
            htmls.append(raw.content)
        except Exception:
            continue

    # The extract callable the thread bench times (returns text length only).
    def _extract_one(html: bytes, path: str) -> int:
        art = extract_article(html, path=path, title=path)
        return len(art.text)

    rows: list[dict[str, object]] = []
    rows.append(extraction.measure_extraction_threads(htmls, _extract_one, workers=1).to_row())
    rows.append(extraction.measure_extraction_threads(htmls, _extract_one, workers=4).to_row())

    # Process pool: a real multi-process run (extract_many opens the archive per
    # worker). Timed here because it needs the zim dep the eval package can't import.
    proc_paths = sample_paths[:120]
    text_bytes = 0
    if proc_paths:
        from vesta.zim.extract import extract_many

        start = time.perf_counter()
        articles = extract_many(archive_path, proc_paths, processes=4)
        elapsed = time.perf_counter() - start
        text_bytes = sum(len(x.text.encode("utf-8")) for x in articles)
        rows.append(
            extraction.measure_extraction_process_pool(
                text_bytes, len(articles), elapsed, processes=4
            ).to_row()
        )
    return rows


def _write_bench_md(
    path: Path,
    rows: list[dict[str, object]],
    cpu: Mapping[str, object],
    *,
    include_extraction: bool,
) -> None:
    """Render the committed bench file: one verdict table + provenance header."""
    lines: list[str] = []
    lines.append("# Vesta hardware benchmark\n")
    lines.append("> One file per machine+date. Every number is annotated")
    lines.append("> ``confirms`` or ``replaces`` against the estimated projection.\n")
    lines.append("## Machine\n")
    for k, v in cpu.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Results\n")
    lines.append("| measurement | value | unit | projection | verdict | notes |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        val = r.get("value")
        val_s = f"{val}" if val is not None else "—"
        # Encoder rows carry 'projected' (the value to confirm);
        # measured rows carry 'projection' (the plan anchor).
        proj = r.get("projected", r.get("projection"))
        lines.append(
            f"| {r.get('name')} | {val_s} | {r.get('unit')} | {proj} "
            f"| {r.get('verdict')} | {r.get('notes')} |"
        )
    lines.append("")
    encoder_names = {
        "ONNX int8 vs fp32 speedup",
        "Embedder throughput",
        "Reranker latency @256 tokens",
    }
    still_deferred: set[str] = {
        str(r["name"])
        for r in rows
        if r["name"] in encoder_names and str(r.get("verdict", "")).startswith("deferred")
    }
    if still_deferred:
        lines.append(
            f"Encoder rows still deferred: {', '.join(sorted(still_deferred))} "
            "(model not found under encoders.model_dir — run `vesta models` first). "
            "The rest are real measurements through the ONNX runtime "
            "(intra_op/spinning per encoders.* settings)."
        )
    else:
        lines.append(
            "Encoder rows (embedder throughput, reranker latency) are real "
            "measurements through the ONNX runtime (encoders.intra_op_threads/"
            "encoders.spinning settings applied). ONNX int8-vs-fp32 speedup is measured "
            "manually (fp32 export not fetched by `vesta models` by default)."
        )
    if not include_extraction:
        lines.append("\n_Extraction throughput not run (no archive)._\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── `vesta index` ────────────────────────────────────────────────────────────


async def _cmd_index(args: argparse.Namespace) -> int:
    """Index one or more ZIM archives at a semantic depth, running jobs sequentially.

    The jobs run in THIS process — so web-server restarts (uvicorn reload) can't
    kill them mid-flight. ``--zim`` may be repeated or comma-separated to queue
    several archives back-to-back; with no ``--zim`` it indexes every registered
    archive. No ``jobs`` row is created (see ``_run_index``): status/progress
    land on the ``zims`` row, and resume is driven by the sidecar checkpoints.

    Resume: each archive picks up from its high-water checkpoint file;
    ``--fresh`` wipes and starts over.
    """
    await _validate_data_dir(args.data_dir)
    async with _open_runtime(args.data_dir, with_index_runtime=True) as state:
        return await _run_index_many(state, args)


async def _run_index_many(state: AppState, args: argparse.Namespace) -> int:
    specs: list[str] = []
    z = args.zim
    if z:
        raw = z if isinstance(z, list) else [str(z)]
        for item in raw:
            specs.extend(s.strip() for s in str(item).replace(",", " ").split() if s.strip())
    if not specs:
        specs = await _all_zim_specs(state.db)
    if not specs:
        print("no archives registered. Start the app once (./start.sh) so it scans data/zims/.")
        return 1
    ids = await _resolve_zims(state.db, specs)
    if not ids:
        print("no matching archives found for:", ", ".join(specs))
        return 1
    if len(ids) == 1:
        return await _run_index(
            state,
            argparse.Namespace(
                depth=args.depth, zim=str(ids[0]), fresh=args.fresh, data_dir=args.data_dir
            ),
        )
    code = 0
    # Queue each archive back-to-back; the first failure stops the queue.
    for i, zim_id in enumerate(ids, 1):
        sub = argparse.Namespace(
            depth=args.depth, zim=str(zim_id), fresh=args.fresh, data_dir=args.data_dir
        )
        print(f"\n\u2501\u2501\u2501 [{i}/{len(ids)}] \u2501\u2501\u2501\n")
        rc = await _run_index(state, sub)
        if rc != 0:
            code = rc
            break
    return code


async def _all_zim_specs(db: Database) -> list[str]:
    async with db.read() as conn:
        cur = await conn.execute("SELECT id FROM zims ORDER BY id")
        rows = await cur.fetchall()
    return [str(r["id"]) for r in rows]


async def _validate_data_dir(data_dir: str | None) -> None:
    """Best-effort: ensure the configured data dir exists (helpful before long jobs).

    Runs before ``_open_runtime`` (which would otherwise silently ``mkdir`` a
    missing dir rather than fail fast), so the resolver isn't configured yet —
    call ``configure()`` here too; it's idempotent and ``_open_runtime`` calls
    it again right after.
    """
    config.configure()
    ddir = Path(data_dir or str(config.get(config.DATA_DIR)))
    if not ddir.exists():
        raise SystemExit(f"data dir not found: {ddir}")


async def _resolve_zims(db: Database, specs: list[str]) -> list[int]:
    """Resolve each spec to an archive id, skipping ones that don't match exactly.

    ``_resolve_zim`` prints the candidate list and raises ``SystemExit`` on an
    ambiguous or missing spec; a queued run should keep going rather than abort
    the rest of the queue, so those specs are skipped (the message already told
    the user what the problem is).
    """
    resolved: list[int] = []
    for spec in specs:
        try:
            zim_id = await _resolve_zim(db, spec)
        except SystemExit:
            continue
        if zim_id is not None and zim_id not in resolved:
            resolved.append(zim_id)
    return resolved


async def _run_index(state: AppState, args: argparse.Namespace) -> int:  # noqa: PLR0915, PLR0912 — one coherent flow: resolve → refuse-if-held → supersede → resume → run → report
    from vesta.index.job import IndexZimJob
    from vesta.index.leases import IndexLeaseHeld, active_holder
    from vesta.jobs.types import RESUME_CHECKPOINT_KEY

    db = state.db
    depth = int(args.depth)
    if depth < 1 or depth > 3:
        print(f"depth must be 1..3, got {depth}")
        return 2

    zim_id = await _resolve_zim(db, args.zim)

    # Cross-process exclusion (AUDIT_0822 M7): a live server-side build holds
    # an index lease even though we're about to cancel its job row below.
    # Refuse BEFORE touching anything — no stranded-row cancellation, no
    # sidecar deletion — so we never strand the build that's beating us. The
    # hard gate is the job's own claim inside IndexZimJob.run; this just fails
    # faster and keeps the cleanup below honest about what it may cancel.
    holder = await active_holder(db, zim_id)
    if holder is not None:
        print(f"another index build is already running for this archive ({holder}).")
        print("wait for it to finish (or stop it), then retry.")
        return 1

    # Supersede any half-finished server-side index job for this archive so the
    # web server's JobRunner.start() can't (re)launch it concurrently on reload.
    stranded = await _cancel_pending_index_jobs(db, zim_id)
    if stranded:
        print(f"cancelled {stranded} stranded server-side index job(s) for this archive.")

    ddir = Path(args.data_dir or str(config.get(config.DATA_DIR)))
    cp_path = ddir / f".index_progress_{zim_id}.json"
    if args.fresh and cp_path.exists():
        # A stale resume sidecar must die BEFORE the job starts: --fresh reads
        # none of it, and the run's first checkpoint only lands after its first
        # batch materializes — dying in that window would leave done_count=N on
        # disk and the next plain `vesta index` would "resume" into the wiped
        # index (AUDIT_0822 M8).
        with contextlib.suppress(OSError):
            cp_path.unlink()
    params: dict[str, Any] = {"zim_id": zim_id, "depth": depth, "owner": "cli"}
    if not args.fresh and cp_path.exists():
        try:
            blob = json.loads(cp_path.read_text())
        except (OSError, ValueError):
            blob = None
        if (
            isinstance(blob, dict)
            and int(blob.get("depth", -1)) == depth
            and int(blob.get("done_count", 0)) > 0
        ):
            params[RESUME_CHECKPOINT_KEY] = blob
            print(f"resuming at ~{blob.get('done_count')} articles (from {cp_path.name}).")

    handle = _CLIIndexHandle(cp_path)

    import signal

    def _on_signal(signum: int, _frame: Any) -> None:
        if handle.cancel():
            print("\n[interrupt] pausing after the current batch (press again to force-quit)…")
        else:
            raise KeyboardInterrupt

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    label = await _zim_label(db, zim_id)
    print(f"indexing {label} (id={zim_id}) at depth {depth}")
    print(f"embedder: {config.get(INDEX_EMBEDDER)}")
    print("runs in this process; web server restarts will not interrupt it.\n")

    code = 0
    try:
        await IndexZimJob().run(handle, params)
    except KeyboardInterrupt:
        print("\nforced quit. Index state left as-is; re-run to resume.")
        code = 130
    except IndexLeaseHeld as exc:
        # Lost the race: a build claimed the archive between our pre-check and
        # the job's own lease claim. Same refusal as above, minus the framing.
        print(f"\n{exc}")
        code = 1
    except Exception as exc:
        print(f"\nindex failed: {exc!r}")
        code = 1
    finally:
        signal.signal(signal.SIGINT, prev_int or signal.SIG_DFL)
        signal.signal(signal.SIGTERM, prev_term or signal.SIG_DFL)
    if code != 0:
        return code

    status = await _zim_index_status(db, zim_id)
    if status == "paused":
        print(f"\npaused. Resume with: uv run vesta index --depth {depth} --zim {zim_id}")
        return 0
    with contextlib.suppress(OSError):
        cp_path.unlink()
    print(f"\ndone: {label} indexed at depth {depth}.")
    return 0


async def _resolve_zim(db: Database, spec: str | None) -> int:
    """Resolve a ``--zim`` argument to a zims.id.

    Numeric ids match directly; anything else is a case-insensitive substring
    match against filename / name / title. With no ``--zim`` and exactly one
    registered archive, that archive is used. Ambiguous / missing specs list the
    available archives and exit.
    """
    async with db.read() as conn:
        if spec is not None and spec.strip().isdigit():
            cur = await conn.execute("SELECT id FROM zims WHERE id=?", (int(spec),))
            row = await cur.fetchone()
            if row is not None:
                return int(row["id"])
        cur = await conn.execute("SELECT id, filename, name, title FROM zims ORDER BY id")
        rows = [dict(r) for r in await cur.fetchall()]

    if not rows:
        print("no archives registered. Start the app once (./start.sh) so it scans data/zims/.")
        raise SystemExit(1)

    def _label(r: dict[str, Any]) -> str:
        return str(r.get("filename") or r.get("name") or r.get("title") or f"zim#{r['id']}")

    if spec is None:
        if len(rows) == 1:
            return int(rows[0]["id"])
        print("multiple archives registered; specify one with --zim.")
    elif spec.strip().isdigit():
        print(f"no archive with id={spec!r}.")
    else:
        needle = spec.lower()
        matches = [
            r
            for r in rows
            if needle in (r.get("filename") or "").lower()
            or needle in (r.get("name") or "").lower()
            or needle in (r.get("title") or "").lower()
        ]
        if len(matches) == 1:
            return int(matches[0]["id"])
        if not matches:
            print(f"no archive matches {spec!r}.")
        else:
            print(f"{spec!r} matches several archives; narrow it down:")
    print("available archives:")
    for r in rows:
        print(f"  id={r['id']:<4} {_label(r)}")
    raise SystemExit(2)


async def _cancel_pending_index_jobs(db: Database, zim_id: int) -> int:
    """Mark any non-terminal ``index_zim`` job rows for this archive cancelled.

    The server's ``JobRunner.start()`` resumes every job left ``running``, which
    would double-execute this index on the next reload. The API stores the job
    ``target`` as ``str(zim_id)`` (api/zims.py); matching that here neutralises
    stranded rows without touching the CLI's own (it creates none)."""
    import datetime as _dt

    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    async with db.write() as conn:
        cur = await conn.execute(
            "UPDATE jobs SET status='cancelled', error='superseded by `vesta index`', "
            "updated_at=?, finished_at=? "
            "WHERE type='index_zim' AND target=? "
            "AND status NOT IN ('done','error','cancelled')",
            (now, now, str(zim_id)),
        )
        return int(cur.rowcount or 0)


async def _zim_index_status(db: Database, zim_id: int) -> str:
    async with (
        db.read() as conn,
        conn.execute("SELECT index_status FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    return str(row["index_status"]) if row is not None else "none"


async def _zim_label(db: Database, zim_id: int) -> str:
    async with (
        db.read() as conn,
        conn.execute("SELECT filename, name, title FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    if row is None:
        return f"zim#{zim_id}"
    return str(row["filename"] or row["name"] or row["title"] or f"zim#{zim_id}")


class _CLIIndexHandle:
    """A :class:`JobHandle` for the CLI index run.

    Prints throttled progress, persists the resume checkpoint to a sidecar file,
    and cooperates with Ctrl+C (first signal → pause after the current batch;
    second → force-quit). Does NOT touch the ``jobs`` table: the index job
    already mirrors status/progress onto the ``zims`` row, and a ``jobs`` row
    would let the web server's ``JobRunner`` re-execute this index on reload.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self._cp = checkpoint_path
        self._cancelled = False
        self._last_print = 0.0

    def cancel(self) -> bool:
        """Request a graceful pause. Returns True the first time, False after."""
        if self._cancelled:
            return False
        self._cancelled = True
        return True

    async def progress(self, done: int, total: int, message: str) -> None:
        now = time.monotonic()
        is_final = total > 0 and done >= total
        if not (is_final or (now - self._last_print) >= 1.0):
            return
        self._last_print = now
        pct = (100.0 * done / total) if total else 0.0
        print(f"  {done:>9}/{total} ({pct:5.1f}%)  {message}", flush=True)

    async def checkpoint(self, blob: Mapping[str, object]) -> None:
        tmp = self._cp.with_name(self._cp.name + ".tmp")
        try:
            tmp.write_text(json.dumps(dict(blob)))
            tmp.replace(self._cp)
        except OSError:
            pass

    def cancelled(self) -> bool:
        return self._cancelled


if __name__ == "__main__":
    sys.exit(main())


# Keep structlog quiet on the CLI path (it would print startup JSON otherwise).
structlog.configure(processors=[structlog.processors.JSONRenderer()])
