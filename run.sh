#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CARDS-FULL-RUN
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

cd /export/fs06/psingh54/CARDS/src

export PYTHONPATH=.

python run_all_probes.py --probe_type linear --pooling mean --device cuda --linear_C 1.0 --output_dir experiments/all_probes_linear_max
