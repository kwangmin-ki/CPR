#!/usr/bin/env bash
# BASE_MODEL=google/gemma-3-4b-pt TAG=gemma3-4b DATASET=gsm bash scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-pt}"
TAG="${TAG:-gemma3-4b}"
DATASET="${DATASET:-gsm}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
EXPERT_DIR="${CKPT_DIR}/${TAG}-${DATASET}-sft"

python train_expert.py --method sft --model_path "${BASE_MODEL}" \
    --dataset "${DATASET}" --output_dir "${EXPERT_DIR}"

python train_router.py --base_model "${BASE_MODEL}" --expert_model "${EXPERT_DIR}" \
    --dataset "${DATASET}" --output_dir "${CKPT_DIR}" --tag "${TAG}"

if [[ "${BASELINES:-0}" == "1" ]]; then
    for METHOD in dft eaft; do
        python train_expert.py --method "${METHOD}" --model_path "${BASE_MODEL}" \
            --dataset "${DATASET}" --output_dir "${CKPT_DIR}/${TAG}-${DATASET}-${METHOD}"
    done
fi
