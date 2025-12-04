#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

# SLURM script to run full pipeline: probe training + workflow experiments
# This script:
# 1. Trains and saves all probes using run_all_probes.py
# 2. Waits for probe training to complete
# 3. Runs workflow experiments with all user simulators

#SBATCH --job-name=CARDS-FULL-PIPELINE
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="xwang397@jhu.edu"
#SBATCH --time=6:00:00
#SBATCH --output=slurm_logs/full_pipeline_%j.out
#SBATCH --error=slurm_logs/full_pipeline_%j.err

# Configuration variables
PROBE_EXPERIMENT_DIR="experiments/all_probes_linear_max"
PROBE_TYPE="linear"
POOLING="mean"
MODEL_NAME="qwen2.5-math-1.5b"
DATASET_PATH="data/processed/insufficient_dataset_umwp/umwp_test.json"
MAX_EXAMPLES=100
OUTPUT_DIR="experiments/workflow_results"
LINEAR_C=1.0

source /home/xwang397/.bashrc
module load cuda/12.1

conda activate cards

cd /home/xwang397/CARDS/src

export PYTHONPATH=.

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p slurm_logs

echo "=================================================================================="
echo "STEP 1: TRAINING PROBES"
echo "=================================================================================="
echo "Output directory: ${PROBE_EXPERIMENT_DIR}"
echo "Probe type: ${PROBE_TYPE}"
echo "Pooling: ${POOLING}"
echo "=================================================================================="

# Step 1: Train and save all probes
python run_all_probes.py \
    --probe_type ${PROBE_TYPE} \
    --pooling ${POOLING} \
    --device cuda \
    --linear_C ${LINEAR_C} \
    --output_dir ${PROBE_EXPERIMENT_DIR}

PROBE_TRAIN_EXIT_CODE=$?

if [ $PROBE_TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Probe training failed with exit code $PROBE_TRAIN_EXIT_CODE"
    exit $PROBE_TRAIN_EXIT_CODE
fi

echo ""
echo "=================================================================================="
echo "STEP 2: VERIFYING PROBES WERE SAVED"
echo "=================================================================================="

# Verify probes were saved
PROBE_CHECK_PATH="${PROBE_EXPERIMENT_DIR}/${MODEL_NAME}/train_on_ALL/probes_${PROBE_TYPE}/all_metrics.json"

if [ ! -f "${PROBE_CHECK_PATH}" ]; then
    echo "ERROR: Probe metrics file not found at ${PROBE_CHECK_PATH}"
    echo "Probe training may have failed or probes were not saved correctly."
    exit 1
fi

echo "✓ Found probe metrics at: ${PROBE_CHECK_PATH}"

# Check for at least one probe file
PROBE_FILE_PATTERN="${PROBE_EXPERIMENT_DIR}/${MODEL_NAME}/train_on_ALL/probes_${PROBE_TYPE}/layer_*_probe.*"
PROBE_FILES=$(ls ${PROBE_FILE_PATTERN} 2>/dev/null | wc -l)

if [ $PROBE_FILES -eq 0 ]; then
    echo "ERROR: No probe files found matching pattern: ${PROBE_FILE_PATTERN}"
    exit 1
fi

echo "✓ Found ${PROBE_FILES} probe files"
echo "=================================================================================="

echo ""
echo "=================================================================================="
echo "STEP 3: RUNNING WORKFLOW EXPERIMENTS"
echo "=================================================================================="

# Array of user simulators to test
USER_SIMULATORS=("rag" "gpt5" "gpt5_recall")

# Determine probe path (experiment directory - load_best_probe will find train_on_ALL)
PROBE_PATH="${PROBE_EXPERIMENT_DIR}"

# Run workflow experiments for each simulator
for USER_SIMULATOR in "${USER_SIMULATORS[@]}"; do
    echo ""
    echo "--- Running workflow experiment with ${USER_SIMULATOR} simulator ---"
    
    OUTPUT_FILE="${OUTPUT_DIR}/workflow_${USER_SIMULATOR}_results.json"
    
    python workflow/run_experiment.py \
        --probe_path ${PROBE_PATH} \
        --probe_type ${PROBE_TYPE} \
        --model_name ${MODEL_NAME} \
        --dataset_path ${DATASET_PATH} \
        --user_simulator ${USER_SIMULATOR} \
        --baselines just_answer always_prompt \
        --device cuda \
        --max_examples ${MAX_EXAMPLES} \
        --output_path ${OUTPUT_FILE} \
        --train_config train_on_ALL
    
    WORKFLOW_EXIT_CODE=$?
    
    if [ $WORKFLOW_EXIT_CODE -ne 0 ]; then
        echo "WARNING: Workflow experiment with ${USER_SIMULATOR} failed with exit code $WORKFLOW_EXIT_CODE"
        echo "Continuing with next simulator..."
    else
        echo "✓ Completed workflow experiment with ${USER_SIMULATOR}"
        echo "  Results saved to: ${OUTPUT_FILE}"
    fi
done

echo ""
echo "=================================================================================="
echo "FULL PIPELINE COMPLETE!"
echo "=================================================================================="
echo "Probe training results: ${PROBE_EXPERIMENT_DIR}"
echo "Workflow experiment results: ${OUTPUT_DIR}"
echo "=================================================================================="

