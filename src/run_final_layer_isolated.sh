#!/usr/bin/env bash
# ============================================================================
# run_final_layer_isolated.sh
# ============================================================================
# Runs final_layer_probe.py with subprocess isolation: each (model, dataset)
# pair runs in its own python invocation. If any pair hits a sticky CUDA
# error (cudaErrorLaunchFailure poisoning the context), only that pair dies
# — the next pair starts a fresh process with a clean CUDA context.
#
# The _COMPLETE marker in each pair's output dir means re-running the
# wrapper is safe: completed pairs print [skip] and exit quickly.
#
# Usage:
#   bash run_final_layer_isolated.sh                  # all models, both datasets
#   bash run_final_layer_isolated.sh umwp             # all models, umwp only
#   bash run_final_layer_isolated.sh treecut          # all models, treecut only
#
# Logs per pair: logs/final_layer_probe/{slug}__{dataset}.log
# ============================================================================

set -u   # error on unset vars, but DON'T -e (we want to continue past failures)

CARDS_ROOT="/home/hwang302/.local/nlp/CARDS"
SRC="${CARDS_ROOT}/src/final_layer_probe.py"
LOGDIR="${CARDS_ROOT}/logs/final_layer_probe"
mkdir -p "${LOGDIR}"

# Datasets to run
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
else
    DATASETS=("umwp" "treecut")
fi

# Models — should match FULL_MODELS in final_layer_probe.py
MODELS=(
    "Qwen/Qwen2.5-Math-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"
    "google/gemma-3-4b-it"
    "Qwen/Qwen2.5-Math-7B"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "google/gemma-3-12b-it"
    "allenai/Olmo-3-7B-Think"
    "allenai/Olmo-3-7B-Instruct"
    "deepseek-ai/deepseek-math-7b-instruct"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-14B-Instruct"
    "google/gemma-3-27b-it"
    "allenai/Olmo-3-32B-Think"
    "openai/gpt-oss-20b"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    "Qwen/Qwen2.5-72B-Instruct"
)

SUCCESS=()
FAILED=()
SKIPPED=()

cd "${CARDS_ROOT}"

for ds in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        slug=$(basename "$m")
        logfile="${LOGDIR}/${slug}__${ds}.log"
        echo "==[ ${m} / ${ds} ]=="
        echo "  log: ${logfile}"

        # Each pair in its own python process — CUDA context is fresh.
        # Output goes to both stdout and the log file.
        python -u "${SRC}" --model "$m" --dataset "$ds" 2>&1 | tee "${logfile}"
        rc=${PIPESTATUS[0]}

        if [[ $rc -eq 0 ]]; then
            if grep -q '_COMPLETE marker present' "${logfile}"; then
                SKIPPED+=("${slug}/${ds}")
            else
                SUCCESS+=("${slug}/${ds}")
            fi
        else
            FAILED+=("${slug}/${ds} (rc=$rc)")
            echo "  [FAIL] ${slug}/${ds} returned $rc"
        fi
        echo
    done
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Skipped (already complete): ${#SKIPPED[@]}"
for x in "${SKIPPED[@]}"; do echo "  $x"; done
echo
echo "Succeeded this run: ${#SUCCESS[@]}"
for x in "${SUCCESS[@]}"; do echo "  $x"; done
echo
echo "Failed: ${#FAILED[@]}"
for x in "${FAILED[@]}"; do echo "  $x"; done

# Rebuild master CSV from whatever's complete
echo
echo "Rebuilding master CSV..."
python -u "${SRC}" --aggregate_only