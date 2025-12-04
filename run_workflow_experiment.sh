#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CARDS-WORKFLOW
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="xwang397@jhu.edu"
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/workflow_gpt5_%j.out
#SBATCH --error=slurm_logs/workflow_gpt5_%j.err

# Configuration variables
PROBE_EXPERIMENT_DIR="experiments/all_probes_linear_max"
PROBE_TYPE="linear"
MODEL_NAME="qwen2.5-math-1.5b"
DATASET_PATH="data/processed/insufficient_dataset_umwp/umwp_test.json"
USER_SIMULATOR="gpt5"  # Options: gpt5, rag, gpt5_recall
MAX_EXAMPLES=100
OUTPUT_DIR="experiments/workflow_results"
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
PROBE_PATH="${PROBE_EXPERIMENT_DIR}/${MODEL_NAME}/train_on_ALL"

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

