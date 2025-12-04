#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

# SLURM script to run workflow experiments with all user simulators
# This script runs three separate jobs: one for each user simulator type

#SBATCH --job-name=CARDS-WORKFLOW-ALL
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="xwang397@jhu.edu"
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/workflow_all_%A_%a.out
#SBATCH --error=slurm_logs/workflow_all_%A_%a.err

# Configuration variables
PROBE_EXPERIMENT_DIR="experiments/all_probes_linear_max"
PROBE_TYPE="linear"
MODEL_NAME="qwen2.5-math-1.5b"
DATASET_PATH="data/processed/insufficient_dataset_umwp/umwp_test.json"
MAX_EXAMPLES=100
OUTPUT_DIR="experiments/workflow_results"

# Array of user simulators to test
USER_SIMULATORS=("rag" "gpt5" "gpt5_recall")

# Select simulator based on array index
USER_SIMULATOR=${USER_SIMULATORS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_FILE="${OUTPUT_DIR}/workflow_${USER_SIMULATOR}_results.json"

source /home/xwang397/.bashrc
module load cuda/12.1

conda activate cards

cd /home/xwang397/CARDS/src

export PYTHONPATH=.

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p slurm_logs

# Determine probe path
# New structure: {experiment_dir}/{model_name}/train_on_{dataset}/probes_{type}/
# Can use experiment directory and let it auto-detect, or specify full path
# For auto-detection, use experiment directory path
PROBE_PATH="${PROBE_EXPERIMENT_DIR}/${MODEL_NAME}/train_on_ALL"

echo "Running workflow experiment with ${USER_SIMULATOR} simulator (array task ${SLURM_ARRAY_TASK_ID})"

# Run workflow experiment
# Use experiment directory path - will auto-detect best layer
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

echo "Experiment completed. Results saved to: ${OUTPUT_FILE}"

