#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CARDS
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

cd /export/fs06/psingh54/COURSEWORK/CARDS

export PYTHONPATH=.

python main.py \
    --train_embeddings data/embeddings/umwp_train_llama-3.2-3b-instruct_last_token.npy \
    --train_labels data/insufficient_dataset_umwp/umwp_train.json \
    --test_embeddings data/embeddings/umwp_test_llama-3.2-3b-instruct_last_token.npy \
    --test_labels data/insufficient_dataset_umwp/umwp_test.json \
    --probe_type mlp \
    --visualize_all

python run_advanced_interp.py --train_labels data/insufficient_dataset_umwp/umwp_train.json --train_embeddings data/embeddings/umwp_train_llama-3.2-3b-instruct_last_token.npy \
 --test_embeddings data/embeddings/umwp_test_llama-3.2-3b-instruct_last_token.npy --test_labels data/insufficient_dataset_umwp/umwp_test.json

python main_analyze.py --model_name llama-3.2-3b-instruct --data_path data/insufficient_dataset_umwp/umwp_train.json --device cuda \
 --max_examples 500 --max_new_tokens 1024 