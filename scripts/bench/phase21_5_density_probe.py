#!/usr/bin/env python3
"""Pre-flight probe — is the 3.0 chars/token floor safe on lfm2.5?

The window plans (D4 pre-seed fit, D5 ledger) enforce with
``answer.tokens.estimate_tokens`` calibrated on **qwen3.5-4b** (min 3.038
chars/token over 90 one-shot requests → floor 3.0). A second answer model,
``lfm2.5-1.2b-instruct@q4_k_m``, whose tokenizer is
not that calibration's subject. If lfm's real density on bench-shaped text
is **below** 3.0 chars/token, every windowed run on lfm is mismeasured:
the ledger would admit requests the real window rejects (hard 400s) — a
calibration stop, not a bench result.

Probe: rebuild Round-0-shaped requests exactly like the original calibration
(full ``SYSTEM_PROMPT`` + question + top-6 snapshot passages, the pre-seed's
content) for a spread of dataset questions, send each to the endpoint with
``max_tokens=1`` and read ``usage.prompt_tokens``; chars are counted the
meter's way (sum of message content — chat-template overhead excluded, the
same safe-side bias the original calibration documented). The same request is sent to the
pinned qwen model as the method control. A second arm samples real
completions (``max_tokens=1200``) to measure output-side chars/token and
verbosity — the output-reserve assumption (qwen: mean 151, max 1305) is
also model-specific.

Usage:
    uv run python scripts/bench/phase21_5_density_probe.py
"""

from __future__ import annotations

import json
import statistics
import urllib.request

from vesta.api.agent_chat import SYSTEM_PROMPT

ENDPOINT = "http://desktop.onoz.cc:1234/v1/chat/completions"
QWEN = "qwen3.5-4b@q4_k_s"
LFM = "lfm2.5-1.2b-instruct@q4_k_m"
SNAPSHOT = "benchmarks/context/phase21-1a-snapshot.json"
DATASET = "benchmarks/vesta_bench_v2.json"
FLOOR = 3.0  # answer/tokens.py CHARS_PER_TOKEN

# Spread across capabilities + question ids (every 13th -> mixed capabilities).
PROMPT_SAMPLES = 12
COMPLETION_SAMPLES = 6


def chat(model: str, messages: list[dict[str, str]], max_tokens: int) -> dict:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def build_user(qid: str, question: str, passages: list[dict]) -> str:
    parts = [f"Question: {question}", "", "Initial source material:"]
    for p in passages[:6]:
        crumb = p.get("breadcrumb") or ""
        head = f"### {p.get('title', '')}" + (f" ({crumb})" if crumb else "")
        parts.append(f"{head}\n{p.get('text', '')}")
    return "\n\n".join(parts)


def main() -> int:
    with open(SNAPSHOT) as f:
        snap = json.load(f)["questions"]
    with open(DATASET) as f:
        ds = {q["id"]: q for q in json.load(f)["questions"]}
    qids = sorted(snap)[:: max(1, len(snap) // PROMPT_SAMPLES)][:PROMPT_SAMPLES]

    print(f"== prompt-side density ({PROMPT_SAMPLES} Round-0-shaped requests) ==")
    ratios: dict[str, list[float]] = {QWEN: [], LFM: []}
    for qid in qids:
        user = build_user(qid, ds[qid]["question"], snap[qid]["passages"])
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        chars = sum(len(m["content"]) for m in msgs)
        for model in (QWEN, LFM):
            pt = chat(model, msgs, 1)["usage"]["prompt_tokens"]
            r = chars / pt
            ratios[model].append(r)
            print(
                f"  {qid} {model.split('@')[0]:<22} chars={chars:>6} "
                f"prompt_tokens={pt:>6} chars/token={r:.3f}"
            )

    print("\n== summary ==")
    for model, rs in ratios.items():
        print(
            f"  {model}: min {min(rs):.3f} / p50 {statistics.median(rs):.3f} "
            f"/ max {max(rs):.3f}  (floor {FLOOR})"
        )

    print(f"\n== completion-side ({COMPLETION_SAMPLES} real answers, max_tokens=1200) ==")
    ctoks, crats = [], []
    for qid in qids[:COMPLETION_SAMPLES]:
        user = build_user(qid, ds[qid]["question"], snap[qid]["passages"])
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        out = chat(LFM, msgs, 1200)
        u = out["usage"]
        text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        ctoks.append(u["completion_tokens"])
        if u["completion_tokens"]:
            crats.append(len(text) / u["completion_tokens"])
        print(
            f"  {qid} completion_tokens={u['completion_tokens']:>5} "
            f"chars={len(text):>5} chars/token="
            f"{len(text) / max(1, u['completion_tokens']):.2f}"
        )

    lfm_min = min(ratios[LFM])
    print(
        f"\nlfm prompt-side min chars/token = {lfm_min:.3f} "
        f"-> floor 3.0 is {'SAFE' if lfm_min >= FLOOR else 'UNSAFE — STOP'}"
    )
    if ctoks:
        print(
            f"lfm completion tokens: min {min(ctoks)} / median "
            f"{statistics.median(ctoks)} / max {max(ctoks)} "
            f"(qwen bench: mean 151, max 1305; reserve at 8k = 1280)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
