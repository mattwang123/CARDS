#!/bin/bash
set -euo pipefail

# Run the complete Exp 15 pipeline for one model:
#   1. strip conclusion + inject wait tokens + generate extended CoT
#   2. evaluate outputs into Q1/Q2/Q3/Q4
#   3. update analysis CSVs/plots for that model
#
# Usage:
#   MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct" bash run_exp15_single_model_pipeline.sh
#   bash run_exp15_single_model_pipeline.sh "Qwen/Qwen2.5-Math-1.5B-Instruct"

REPO_ROOT="${REPO_ROOT:-/home/xwang397/CARDS}"
CONDA_ENV="${CONDA_ENV:-torch}"
MODEL_NAME="${1:-${MODEL_NAME:-}}"
DATASETS="${DATASETS:-umwp treecut}"
WAIT_COUNTS="${WAIT_COUNTS:-1 3 5}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

EXP15_OUTPUT_BASE="${EXP15_OUTPUT_BASE:-/export/fs06/xwang397/CARDS/results_new}"
EXP2_GENERATION_DIR="${EXP2_GENERATION_DIR:-/export/fs06/hwang302/CARDS/experiments/dynamic_tracking_test}"
EXP2_EVAL_DIR="${EXP2_EVAL_DIR:-/export/fs06/hwang302/CARDS/experiments/dynamic_tracking_test_evaluation}"
EXP11_RESULTS_DIR="${EXP11_RESULTS_DIR:-/export/fs06/hwang302/CARDS/exp_temporal_new/results}"

GENERATE_EXTRA_ARGS="${GENERATE_EXTRA_ARGS:-}"
EVALUATE_EXTRA_ARGS="${EVALUATE_EXTRA_ARGS:-}"
ANALYSIS_EXTRA_ARGS="${ANALYSIS_EXTRA_ARGS:-}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: Set MODEL_NAME or pass the HuggingFace model name as the first argument." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH=.

if command -v module >/dev/null 2>&1; then
  module load cuda/12.1 || true
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}" || true
fi

echo "======================================================================"
echo "Exp 15 pipeline for ${MODEL_NAME}"
echo "Datasets:    ${DATASETS}"
echo "Wait counts: ${WAIT_COUNTS}"
echo "Output base: ${EXP15_OUTPUT_BASE}"
echo "======================================================================"

mkdir -p "${EXP15_OUTPUT_BASE}"

echo "[1/3] Stripping conclusions + injecting wait tokens + generating"
python src/exp15_wait_extension_generate.py \
  --model "${MODEL_NAME}" \
  --generation_dir "${EXP2_GENERATION_DIR}" \
  --output_dir "${EXP15_OUTPUT_BASE}/experiments/wait_extension" \
  --datasets ${DATASETS} \
  --wait_counts ${WAIT_COUNTS} \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  ${GENERATE_EXTRA_ARGS}

JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}")
if [[ -n "${JUDGE_BASE_URL}" ]]; then
  JUDGE_ARGS+=(--judge_base_url "${JUDGE_BASE_URL}")
fi

echo "[2/3] Evaluating wait-extended outputs"
python src/exp15_wait_extension_evaluate.py \
  --model "${MODEL_NAME}" \
  --input_dir "${EXP15_OUTPUT_BASE}/experiments/wait_extension" \
  --output_dir "${EXP15_OUTPUT_BASE}/experiments/wait_extension_evaluation" \
  --datasets ${DATASETS} \
  --wait_counts ${WAIT_COUNTS} \
  "${JUDGE_ARGS[@]}" \
  ${EVALUATE_EXTRA_ARGS}

echo "[3/3] Updating Exp 15 analysis"
python src/exp15_wait_extension_analysis.py \
  --model "${MODEL_NAME}" \
  --eval_dir "${EXP15_OUTPUT_BASE}/experiments/wait_extension_evaluation" \
  --exp2_eval_dir "${EXP2_EVAL_DIR}" \
  --results_dir "${EXP15_OUTPUT_BASE}/results" \
  --exp11_results_dir "${EXP11_RESULTS_DIR}" \
  --plot_dir "${EXP15_OUTPUT_BASE}/paper_plots/exp15_wait" \
  --datasets ${DATASETS} \
  --wait_counts ${WAIT_COUNTS} \
  ${ANALYSIS_EXTRA_ARGS}

echo "Completed Exp 15 pipeline for ${MODEL_NAME}"
