#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CARDS-INTERVENTION
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

cd /export/fs06/psingh54/CARDS/src/workflow

export PYTHONPATH=/export/fs06/psingh54/CARDS/src

# Create logs directory if it doesn't exist
mkdir -p ../../logs

# Run intervention pipeline for all models and datasets
# 4 models × 2 datasets = 8 experiments
echo "Starting Training-Free Intervention Pipeline"
echo "=============================================="
echo "Models: qwen2.5-math-1.5b, qwen2.5-1.5b, llama-3.2-3b-instruct, qwen2.5-math-7b"
echo "Datasets: umwp, gsm8k"
echo "Output: /export/fs06/psingh54/CARDS/src/experiments/intervention"
echo ""

python intervention_pipeline.py \
    --device cuda \
    --output_dir ../experiments/intervention

echo ""
echo "=============================================="
echo "Intervention Pipeline Complete!"
echo "Results saved in: /export/fs06/psingh54/CARDS/src/experiments/intervention"
echo "=============================================="
