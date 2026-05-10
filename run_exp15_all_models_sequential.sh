#!/bin/bash
set -euo pipefail

# Run Exp 15 model-by-model. Each line completes:
# generate -> evaluate -> analysis
# before moving to the next model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Representative first pass across size/model families.
MODEL_NAME='Qwen/Qwen2.5-Math-1.5B-Instruct' bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='allenai/Olmo-3-7B-Think'          bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='google/gemma-3-27b-it'             bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Llama-70B' bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"

# Then fill in the remaining models.
MODEL_NAME='Qwen/Qwen2.5-Math-1.5B'                  bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-3B'                          bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'                 bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='google/gemma-3-4b-it'                     bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-Math-7B'                     bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-Math-7B-Instruct'            bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='meta-llama/Meta-Llama-3.1-8B'             bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='meta-llama/Meta-Llama-3.1-8B-Instruct'    bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='google/gemma-3-12b-it'                    bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='allenai/Olmo-3-7B-Instruct'               bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='deepseek-ai/deepseek-math-7b-instruct'    bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-14B'                         bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-14B-Instruct'                bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='allenai/Olmo-3-32B-Think'                 bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='openai/gpt-oss-20b'                       bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
MODEL_NAME='Qwen/Qwen2.5-72B-Instruct'               bash "${SCRIPT_DIR}/run_exp15_single_model_pipeline.sh"
