#!/bin/bash
set -euo pipefail

# Run the complete Exp 14 pipeline for one model:
#   1. generate early-cutoff force-decode outputs
#   2. evaluate outputs into Q1/Q2/Q3/Q4
#   3. update analysis CSVs/plots for that model
#
# Usage:
#   MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct" bash run_exp14_single_model_pipeline.sh
#   bash run_exp14_single_model_pipeline.sh "Qwen/Qwen2.5-Math-1.5B-Instruct"

REPO_ROOT="${REPO_ROOT:-/home/xwang397/CARDS}"
CONDA_ENV="${CONDA_ENV:-torch}"
MODEL_NAME="${1:-${MODEL_NAME:-}}"
DATASETS="${DATASETS:-umwp treecut}"
# 0 = no CoT before force decode (prompt + boxed answer only).
CUTOFFS="${CUTOFFS:-0 0.2 0.4 0.6 1.0}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
EXP14_OUTPUT_BASE="${EXP14_OUTPUT_BASE:-/export/fs06/xwang397/CARDS/results_new}"
EXP2_GENERATION_DIR="${EXP2_GENERATION_DIR:-/export/fs06/hwang302/CARDS/experiments/dynamic_tracking_test}"
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
echo "Exp 14 pipeline for ${MODEL_NAME}"
echo "Datasets: ${DATASETS}"
echo "Cutoffs: ${CUTOFFS}"
echo "Output base: ${EXP14_OUTPUT_BASE}"
echo "======================================================================"

mkdir -p "${EXP14_OUTPUT_BASE}"

echo "[1/3] Generating early-cutoff force-decode outputs"
python src/exp14_early_cutoff_generate.py \
  --model "${MODEL_NAME}" \
  --generation_dir "${EXP2_GENERATION_DIR}" \
  --output_dir "${EXP14_OUTPUT_BASE}/experiments/early_cutoff" \
  --datasets ${DATASETS} \
  --cutoffs ${CUTOFFS} \
  ${GENERATE_EXTRA_ARGS}

JUDGE_ARGS=(--judge_model "${JUDGE_MODEL}")
if [[ -n "${JUDGE_BASE_URL}" ]]; then
  JUDGE_ARGS+=(--judge_base_url "${JUDGE_BASE_URL}")
fi

echo "[2/3] Evaluating cutoff outputs"
python src/exp14_early_cutoff_evaluate.py \
  --model "${MODEL_NAME}" \
  --input_dir "${EXP14_OUTPUT_BASE}/experiments/early_cutoff" \
  --output_dir "${EXP14_OUTPUT_BASE}/experiments/early_cutoff_evaluation" \
  --datasets ${DATASETS} \
  --cutoffs ${CUTOFFS} \
  "${JUDGE_ARGS[@]}" \
  ${EVALUATE_EXTRA_ARGS}

echo "[3/3] Updating Exp 14 analysis"
python src/exp14_early_cutoff_analysis.py \
  --model "${MODEL_NAME}" \
  --eval_dir "${EXP14_OUTPUT_BASE}/experiments/early_cutoff_evaluation" \
  --results_dir "${EXP14_OUTPUT_BASE}/results" \
  --exp11_results_dir "${EXP11_RESULTS_DIR}" \
  --plot_dir "${EXP14_OUTPUT_BASE}/paper_plots/exp14_cutoff" \
  --datasets ${DATASETS} \
  --cutoffs ${CUTOFFS} \
  ${ANALYSIS_EXTRA_ARGS}

echo "Completed Exp 14 pipeline for ${MODEL_NAME}"
