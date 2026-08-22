#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="http://desktop.onoz.cc:1234/v1"
API_KEY="dummy"
JUDGE_MODEL="google/gemma-4-12b-qat"
CONTEXT_FILE="snapshots/retrieval_l3_wikipedia.json"

MODELS=(
  "lfm2.5-1.2b-instruct@q4_k_m"
  "qwen3.5-4b@iq2_xxs"
  "lfm2.5-2.6b@q4_k_m"
  "qwen3.5-4b@q4_k_s"
  "google/gemma-4-12b-qat"
)

echo "================================================================="
echo " Starting Level 3 Baseline Evaluation across ${#MODELS[@]} Models "
echo " Context: ${CONTEXT_FILE}"
echo " Endpoint: ${ENDPOINT}"
echo " Judge: ${JUDGE_MODEL}"
echo "================================================================="

for i in "${!MODELS[@]}"; do
  idx=$((i + 1))
  model="${MODELS[$i]}"
  echo ""
  echo "================================================================="
  echo " [${idx}/${#MODELS[@]}] Evaluating model: ${model}"
  echo "================================================================="

  uv run vesta bench run \
    --system answer_only \
    --from-context "${CONTEXT_FILE}" \
    --level 3 \
    --model "${model}" \
    --endpoint "${ENDPOINT}" \
    --api-key "${API_KEY}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-endpoint "${ENDPOINT}" \
    --judge-api-key "${API_KEY}" \
    --label "baseline-l3-${model}" \
    --report both

  echo " [${idx}/${#MODELS[@]}] Completed evaluation for: ${model}"
done

echo ""
echo "================================================================="
echo " All ${#MODELS[@]} model evaluations completed successfully! "
echo "================================================================="
