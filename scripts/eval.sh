#!/usr/bin/env bash
# WRAPPER=cpr DOMAIN=math MODEL_ARGS="base_model_path=...,expert_model_path=...,router_model_path=..." bash scripts/eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."

WRAPPER="${WRAPPER:?set WRAPPER}"
MODEL_ARGS="${MODEL_ARGS:?set MODEL_ARGS}"
DOMAIN="${DOMAIN:-math}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${WRAPPER}}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"

TASK_DIR=lm-evaluation-harness/lm_eval/tasks

# task  num_fewshot  batch_size
MATH_TASKS=(
  "${TASK_DIR}/gsm8k/gsm8k-cot-zeroshot.yaml 0 2"
  "${TASK_DIR}/asdiv/asdiv-cot-manual.yaml 0 2"
  "${TASK_DIR}/manual_svamp/svamp-cot-manual.yaml 0 2"
)
MEDICAL_TASKS=(
  "${TASK_DIR}/pubmedqa/pubmedqa-cot-manual.yaml 2 8"
  "${TASK_DIR}/medqa/medqa_cot.yaml 2 8"
  "${TASK_DIR}/careqa/careqa_en_cot.yaml 2 8"
)
GENERAL_TASKS=(
  "mmlu_flan_cot_fewshot 2 8"
  "${TASK_DIR}/commonsense_qa/cot.yaml 2 8"
  "${TASK_DIR}/arc/flan_cot/arc_challenge_cot.yaml 2 8"
)

case "${DOMAIN}" in
  math)    TASKS=("${MATH_TASKS[@]}" "${GENERAL_TASKS[@]}") ;;
  medical) TASKS=("${MEDICAL_TASKS[@]}" "${GENERAL_TASKS[@]}") ;;
  *) echo "Unknown DOMAIN: ${DOMAIN}" >&2; exit 1 ;;
esac

for ENTRY in "${TASKS[@]}"; do
  read -r TASK NUM_FEWSHOT BATCH_SIZE <<< "${ENTRY}"
  TASK_NAME=$(basename "${TASK}" .yaml)
  EXTRA=()
  [[ -n "${LIMIT}" ]] && EXTRA+=(--limit "${LIMIT}")

  lm_eval \
    --model hf-wrapper \
    --model_args "wrapper=${WRAPPER},${MODEL_ARGS}" \
    --tasks "${TASK}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --log_samples \
    --output_path "${OUTPUT_DIR}/${TASK_NAME}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
done
