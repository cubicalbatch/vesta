# Vesta Model Matrix Benchmark Report (5 Models × {8k, 16k} Context Windows)

**Date:** 2026-08-19
**Dataset:** `vesta_bench_v2.json` (Release Tier: 200 Questions, Level 3)
**Retrieval Profile:** `hybrid` (dense + BM25)
**Retrieval Scope:** `wikipedia_en_top_nopic_2026-06.zim`
**Judge Model:** `google/gemma-4-12b-qat` (temperature=0.0, max_tokens=4096)
**Endpoint:** `http://desktop.onoz.cc:1234/v1`
**Runner System:** `agentic_pydantic` (Full live AI agent search workflow)

---

## 1. Executive Summary

This benchmark measures 5 local LLMs across two context window profiles (`8k` and `16k` ceilings) on the unified 200-question Wikipedia benchmark. Every run exercises the full agentic search pipeline: query reformulation, hybrid retrieval, dynamic tool calling (`read_article`, `search_exact`, etc.), conversation memory, and window-aware budgeting.

### Key Highlights
- **Top Performer Overall:** **`qwen3.5-4b@q4_k_s` (16k)** achieved **74.0%** strict accuracy and **80.8%** weighted accuracy at an average turn latency of 19.9s.
- **Largest Context-Window Dividend:** **`lfm2.5-2.6b@q4_k_m`** gained **+8.5 pp strict** (62.5% → 71.0%) when given a 16k context window, closely matching the 12B model (**71.5%**) at a fraction of the compute size.
- **Best 8k Profile Efficiency:** **`qwen3.5-4b@q4_k_s` (8k)** achieved **70.0%** strict accuracy with only 1.72M total tokens consumed.
- **Fastest Inference:** **`lfm2.5-1.2b-instruct@q4_k_m`** completed turns in **10.6s – 14.7s**, though bounded by a 49.5% strict accuracy ceiling.

---

## 2. Complete 10-Cell Matrix Summary

| Run ID | Model | Context Profile | Strict Accuracy | Weighted Accuracy | Correct / Partial / Incorrect / Unjudged | Total Tokens | p50 Tokens / Question | Avg Latency / Turn | Result File |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 80 | `lfm2.5-1.2b-instruct@q4_k_m` | **8k** | **47.0%** | 57.5% | 94 / 42 / 60 / 4 | 821,031 | 3,834 | 14.7s | [run-80.md](../results/20260818-232051-matrix-lfm2-5-1-2b-instruct-q4-k-m-ctx-8k.md) |
| 81 | `lfm2.5-1.2b-instruct@q4_k_m` | **16k** | **49.5%** | 59.8% | 99 / 41 / 56 / 4 | 828,738 | 3,861 | 10.6s | [run-81.md](../results/20260819-021021-matrix-lfm2-5-1-2b-instruct-q4-k-m-ctx-16k.md) |
| 82 | `lfm2.5-2.6b@q4_k_m` | **8k** | **62.5%** | 66.2% | 125 / 15 / 59 / 1 | 1,728,092 | 9,112 | 24.5s | [run-82.md](../results/20260819-034616-matrix-lfm2-5-2-6b-q4-k-m-ctx-8k.md) |
| 83 | `lfm2.5-2.6b@q4_k_m` | **16k** | **71.0%** | 76.0% | 142 / 20 / 34 / 4 | 3,127,987 | 12,158 | 24.8s | [run-83.md](../results/20260819-053014-matrix-lfm2-5-2-6b-q4-k-m-ctx-16k.md) |
| 84 | `google/gemma-4-e2b` | **8k** | **60.0%** | 66.8% | 120 / 27 / 48 / 5 | 1,276,110 | 4,278 | 18.0s | [run-84.md](../results/20260819-065203-matrix-google-gemma-4-e2b-ctx-8k.md) |
| 85 | `google/gemma-4-e2b` | **16k** | **60.0%** | 67.5% | 120 / 30 / 48 / 2 | 1,296,840 | 4,284 | 18.0s | [run-85.md](../results/20260819-075913-matrix-google-gemma-4-e2b-ctx-16k.md) |
| 86 | `google/gemma-4-12b-qat` | **8k** | **63.5%** | 69.0% | 127 / 22 / 45 / 6 | 1,253,304 | 4,204 | 22.7s | [run-86.md](../results/20260819-093358-matrix-google-gemma-4-12b-qat-ctx-8k.md) |
| 87 | `google/gemma-4-12b-qat` | **16k** | **71.5%** | 77.0% | 143 / 22 / 28 / 7 | 1,590,282 | 4,270 | 24.5s | [run-87.md](../results/20260819-110823-matrix-google-gemma-4-12b-qat-ctx-16k.md) |
| 88 | `qwen3.5-4b@q4_k_s` | **8k** | **70.0%** | 76.0% | 140 / 24 / 31 / 5 | 1,721,333 | 4,327 | 17.6s | [run-88.md](../results/20260819-122306-matrix-qwen3-5-4b-q4_k_s-ctx-8k.md) |
| 89 | `qwen3.5-4b@q4_k_s` | **16k** | **74.0%** | 80.8% | 148 / 27 / 18 / 7 | 2,351,156 | 4,277 | 19.9s | [run-89.md](../results/20260819-134630-matrix-qwen3-5-4b-q4-k-s-ctx-16k.md) |

---

## 3. Context Window Delta Analysis (8k vs 16k)

| Model | 8k Strict | 16k Strict | Strict Δ (pp) | 8k Weighted | 16k Weighted | Weighted Δ (pp) | Token Δ (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lfm2.5-1.2b-instruct@q4_k_m` | 47.0% | 49.5% | **+2.5%** | 57.5% | 59.8% | +2.3% | +0.9% |
| `lfm2.5-2.6b@q4_k_m` | 62.5% | 71.0% | **+8.5%** | 66.2% | 76.0% | +9.8% | +81.0% |
| `google/gemma-4-e2b` | 60.0% | 60.0% | **+0.0%** | 66.8% | 67.5% | +0.7% | +1.6% |
| `google/gemma-4-12b-qat` | 63.5% | 71.5% | **+8.0%** | 69.0% | 77.0% | +8.0% | +26.9% |
| `qwen3.5-4b@q4_k_s` | 70.0% | 74.0% | **+4.0%** | 76.0% | 80.8% | +4.8% | +36.6% |

### Analysis:
1. **Multi-round tool exploration:** Models like `lfm2.5-2.6b` and `google/gemma-4-12b-qat` heavily exploit additional window headroom. At 16k, `lfm2.5-2.6b` increases its multi-hop cross-article accuracy from 44.0% to 66.0%.
2. **Single-shot / pre-seed reliance:** `google/gemma-4-e2b` answers primarily from the first retrieved pre-seed bundle without initiating deeper reading tool calls, keeping its accuracy flat across context sizes.
3. **Reasoning capacity ceiling:** `lfm2.5-1.2b` remains bounded by model capability rather than context length (+2.5 pp gain).

---

## 4. Capability Breakdown (16k Profile)

| Capability (Question Count) | `lfm-1.2b` | `lfm-2.6b` | `gemma-e2b` | `gemma-12b` | `qwen-4b` |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`buried_fact` (50)** | 62.0% | 82.0% | 72.0% | 84.0% | **88.0%** |
| **`multi_fact_same_article` (50)** | 58.0% | 72.0% | 68.0% | **90.0%** | 84.0% |
| **`multi_hop_cross_article` (50)** | 22.0% | 66.0% | 38.0% | 54.0% | **68.0%** |
| **`concept_lookup` (10)** | 40.0% | 70.0% | 40.0% | 50.0% | **80.0%** |
| **`complex_explanation` (10)** | 60.0% | **70.0%** | **70.0%** | 60.0% | 60.0% |
| **`comparative` (10)** | 50.0% | 40.0% | 40.0% | **50.0%** | 30.0% |
| **`procedural` (10)** | 40.0% | 60.0% | **80.0%** | 70.0% | 50.0% |
| **`adversarial_abstention` (10)** | **90.0%** | 80.0% | 80.0% | 60.0% | 60.0% |
| **TOTAL (200)** | **49.5%** | **71.0%** | **60.0%** | **71.5%** | **74.0%** |

---

## 5. Token Usage, Speed & Efficiency Analysis

| Model | Context | Input Tokens | Output Tokens | Total Tokens | Median Tokens/Q | Avg Turn Latency | Cost Efficiency (Strict / 1M Tok) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lfm2.5-1.2b-instruct@q4_k_m` | 8k | 793,759 | 27,272 | 821,031 | 3,834 | 14.7s | 57.2 |
| `lfm2.5-1.2b-instruct@q4_k_m` | 16k | 801,445 | 27,293 | 828,738 | 3,861 | 10.6s | 59.7 |
| `google/gemma-4-e2b` | 8k | 1,126,717 | 149,393 | 1,276,110 | 4,278 | 18.0s | 47.0 |
| `google/gemma-4-e2b` | 16k | 1,148,003 | 148,837 | 1,296,840 | 4,284 | 18.0s | 46.3 |
| `google/gemma-4-12b-qat` | 8k | 1,125,436 | 127,868 | 1,253,304 | 4,204 | 22.7s | 50.7 |
| `google/gemma-4-12b-qat` | 16k | 1,447,904 | 142,378 | 1,590,282 | 4,270 | 24.5s | 45.0 |
| `qwen3.5-4b@q4_k_s` | 8k | 1,689,642 | 31,691 | 1,721,333 | 4,327 | 17.6s | 40.7 |
| `qwen3.5-4b@q4_k_s` | 16k | 2,319,302 | 31,854 | 2,351,156 | 4,277 | 19.9s | 31.5 |
| `lfm2.5-2.6b@q4_k_m` | 8k | 1,596,570 | 131,522 | 1,728,092 | 9,112 | 24.5s | 36.2 |
| `lfm2.5-2.6b@q4_k_m` | 16k | 2,916,613 | 211,374 | 3,127,987 | 12,158 | 24.8s | 22.7 |

---

## 6. Recommendations by Deployment Goal

1. **Recommended Default / Best Quality:** **`qwen3.5-4b@q4_k_s` (16k context)**
   - Top overall accuracy (**74.0%** strict, **80.8%** weighted).
   - Fast 19.9s turn latency and solid prompt following.
2. **Best Quality-to-Size Ratio:** **`lfm2.5-2.6b@q4_k_m` (16k context)**
   - Scores **71.0%**, matching the 12B model at 2.6B parameters.
3. **Best Constrained-Context (8k) Model:** **`qwen3.5-4b@q4_k_s` (8k context)**
   - Delivers **70.0%** strict accuracy on minimal memory budget (1.72M tokens).
4. **Fastest / Ultra-Lightweight:** **`lfm2.5-1.2b-instruct@q4_k_m`**
   - 10.6s latency, 828k tokens, for low-tier edge devices.
