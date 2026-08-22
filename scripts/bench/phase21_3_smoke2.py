#!/usr/bin/env python3
"""One-question lever smokes (the smoke2 replacement).

Runs 65-67 (``phase21-3-smoke-{a,b,c}``) all used wiki-0001, whose turn shape
is the 8-request abstention-retry loop that ends in the crash fallback — no
lever ever fired there (``round_cap_fires`` did not even exist in the trace
payload when they ran), so they demonstrate nothing. This driver re-runs the
smoke on ONE question hand-picked from run 64's (``phase21-2-8k-fpw``) trace
shapes, where the lever in question demonstrably bites, through the REAL bench
machinery (``run_benchmark`` + ``SqliteBenchStore``) so the smoke lands in
``bench_runs`` with full provenance (settings_snapshot with the forced lever
values, ``config_json.context_profile``), exactly like a ``--limit 1`` CLI run
but on a chosen question instead of the dataset's first.

Arms (all at the pinned models, ``agentic_pydantic``, ``8k-fullprompt-wide``,
level-3 core — run 64's configuration):
* ``a`` — mfs-0025 (run 64: 3 requests, 2 tool calls): the DERIVED round cap
  (ledger 1963 // est(4500) = 1) steers the second call. Levers (b)/(c)
  pinned off. Evidence: ``trace.budget.max_tool_rounds=1``,
  ``trace.round_cap_fires >= 1``.
* ``b`` — mfs-0027 (run 64: 3 requests, 1 tool call, abstained-with-evidence,
  converged): the compact re-ask fires on the ``abstain`` trigger. Cap pinned
  open, aging off. Evidence: ``trace.compact_reask.fired=true`` with
  ``trigger="abstain"``, the fresh request's measured ``input_tokens`` and the
  ``steered_est_tokens`` it replaced.
* ``c`` — wiki-0010 (run 64: 4 requests, 3 tool calls): the DERIVED aging
  budget (3·ledger//6 ≈ 981) truncates the oldest round on the turn's later
  requests. Cap pinned open, re-ask off. Evidence:
  ``trace.aged_requests >= 1``, ``trace.age_saved_chars > 0``, and the
  request_log's growth stall.

Usage:
    .venv/bin/python scripts/bench/phase21_3_smoke2.py a
    .venv/bin/python scripts/bench/phase21_3_smoke2.py b c
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from vesta import cli, config
from vesta.api.bench import (
    SqliteBenchStore,
    make_judge_llm,
    make_system,
    reconcile_stale_bench_runs,
)
from vesta.eval.bench_dataset import filter, load_bench_dataset
from vesta.eval.bench_runner import (
    BENCH_JUDGE_CONCURRENCY,
    BENCH_MAX_CONCURRENT,
    resolve_judge_concurrency,
    run_benchmark,
)
from vesta.eval.golden import EVAL_JUDGE_ENDPOINT_URL, EVAL_JUDGE_MODEL
from vesta.inference import (
    INFERENCE_LLM_API_KEY,
    INFERENCE_LLM_ENDPOINT_URL,
    INFERENCE_LLM_MODEL,
)

#: arm → (question_id, forced lever values, the trace contract expected)
_ARMS: dict[str, dict[str, object]] = {
    "a": {
        "question": "mfs-0025",
        "max_tool_rounds": 0,  # derive → 1 at 8k-fpw; the lever under test
        "compact_reask": "off",
        "age_tool_chars": -1,
        "expect": "budget.max_tool_rounds >= 1, round_cap_fires >= 1",
    },
    "b": {
        "question": "mfs-0027",
        "max_tool_rounds": 6,  # pinned open so the trigger is 'abstain'
        "compact_reask": "auto",  # on under a window; the lever under test
        "age_tool_chars": -1,
        "expect": "compact_reask.fired=true trigger=abstain + pricing",
    },
    "c": {
        "question": "wiki-0010",
        "max_tool_rounds": 6,  # pinned open so all rounds execute
        "compact_reask": "off",
        "age_tool_chars": 0,  # derive → ~981; the lever under test
        "expect": "aged_requests >= 1, age_saved_chars > 0",
    },
}


async def _run_arm(arm: str, data_dir: str | None) -> int:
    spec = _ARMS[arm]
    # The same CLI-flag mapping `vesta bench run` applies (provenance: the
    # forced values land in settings_snapshot exactly as runs 65-67 show).
    overrides = cli._build_apply_overrides(
        argparse.Namespace(
            context_profile="8k-fullprompt-wide",
            max_tool_rounds=spec["max_tool_rounds"],
            compact_reask=spec["compact_reask"],
            age_tool_chars=spec["age_tool_chars"],
        )
    )
    async with cli._open_runtime(
        data_dir, with_gateway=True, settings_overrides=overrides or None
    ) as state:
        with contextlib.suppress(Exception):
            await reconcile_stale_bench_runs(state.db)
        dataset = load_bench_dataset()
        qs = [
            q
            for q in filter(filter(dataset.questions, slice="core"), level=3)
            if q.id == spec["question"]
        ]
        if len(qs) != 1:
            print(f"arm {arm}: question {spec['question']!r} not found once ({len(qs)})")
            return 1
        judge_model = str(config.get(EVAL_JUDGE_MODEL))
        answer_endpoint = str(config.get(INFERENCE_LLM_ENDPOINT_URL))
        judge, judge_gateway = make_judge_llm(state, judge_model)
        judge_concurrency, shares = resolve_judge_concurrency(
            int(BENCH_JUDGE_CONCURRENCY.default),
            answer_endpoint=answer_endpoint,
            judge_endpoint=str(config.get(EVAL_JUDGE_ENDPOINT_URL)),
        )
        sut = make_system(
            "agentic_pydantic",
            state,
            model_id=str(config.get(INFERENCE_LLM_MODEL)),
            endpoint=answer_endpoint,
            api_key=str(config.get(INFERENCE_LLM_API_KEY)),
        )
        print(
            f"arm {arm}: {spec['question']}  profile 8k-fullprompt-wide  "
            f"max_tool_rounds={spec['max_tool_rounds']} "
            f"compact_reask={spec['compact_reask']} age_tool_chars={spec['age_tool_chars']}"
        )
        print(f"expect: {spec['expect']}")
        try:
            records = await run_benchmark(
                dataset=dataset,
                questions=qs,
                systems=[sut],
                store=SqliteBenchStore(state.db),
                judge=judge,
                judge_model=judge_model,
                label=f"phase21-3-smoke2-{arm}",
                config_snapshot=dict(config.snapshot().values),
                context_profile="8k-fullprompt-wide",
                judge_concurrency=judge_concurrency,
                judge_shares_endpoint=shares,
                repeats=1,
                max_concurrent=int(BENCH_MAX_CONCURRENT.default),
                level=3,
            )
        finally:
            if judge_gateway is not None:
                with contextlib.suppress(Exception):
                    await judge_gateway.aclose()
        for rec in records:
            print(f"run {rec.id} label={rec.label} status={rec.status}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", choices=sorted(_ARMS))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    for arm in args.arms:
        rc = asyncio.run(_run_arm(arm, args.data_dir))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
