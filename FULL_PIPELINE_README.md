# Full Pipeline Script - Probe Training + Workflow Experiments

## Overview

`run_full_pipeline.sh` is a comprehensive SLURM script that runs the complete pipeline:
1. **Probe Training**: Trains and saves all probes using `run_all_probes.py`
2. **Verification**: Checks that probes were saved correctly
3. **Workflow Experiments**: Runs workflow experiments with all user simulators

## Usage

```bash
sbatch run_full_pipeline.sh
```

## What It Does

### Step 1: Probe Training
- Runs `run_all_probes.py` to train probes for all models and datasets
- Saves probes to: `experiments/all_probes_linear_max/{model_name}/train_on_{dataset}/probes_{type}/`
- Exits if probe training fails

### Step 2: Verification
- Checks for `all_metrics.json` file
- Verifies probe files exist (`.pkl` for linear, `.pt` for MLP)
- Exits if probes are not found

### Step 3: Workflow Experiments
- Runs workflow experiments for each user simulator:
  - `rag` (no API needed)
  - `gpt5` (requires OpenAI API key)
  - `gpt5_recall` (requires OpenAI API key)
- Saves results to: `experiments/workflow_results/workflow_{simulator}_results.json`
- Continues even if one simulator fails

## Configuration

Edit the configuration variables at the top of the script:

```bash
PROBE_EXPERIMENT_DIR="experiments/all_probes_linear_max"  # Where probes are saved
PROBE_TYPE="linear"                                        # linear or mlp
POOLING="mean"                                             # last_token, mean, or max
MODEL_NAME="qwen2.5-math-1.5b"                            # Model for workflow
DATASET_PATH="data/processed/insufficient_dataset_umwp/umwp_test.json"
MAX_EXAMPLES=100                                           # Examples per workflow run
OUTPUT_DIR="experiments/workflow_results"                  # Workflow results directory
LINEAR_C=1.0                                              # Regularization for linear probes
```

## Directory Structure

After running, you'll have:

```
experiments/
├── all_probes_linear_max/              # Probe training results
│   ├── qwen2.5-math-1.5b/
│   │   ├── train_on_umwp/
│   │   │   └── probes_linear/
│   │   │       ├── layer_0_probe.pkl
│   │   │       ├── ...
│   │   │       └── all_metrics.json
│   │   ├── train_on_gsm8k/
│   │   ├── train_on_treecut/
│   │   └── train_on_ALL/               # Used by workflow
│   │       └── probes_linear/
│   │           ├── layer_0_probe.pkl
│   │           └── all_metrics.json
│   └── ...
└── workflow_results/                    # Workflow experiment results
    ├── workflow_rag_results.json
    ├── workflow_gpt5_results.json
    └── workflow_gpt5_recall_results.json
```

## Prerequisites

1. **Trained Probes**: Script will train them automatically
2. **Dataset**: Ensure test dataset exists at specified path
3. **OpenAI API Key** (for GPT-5 simulators): Set in `~/.bashrc`:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

## Time Requirements

- **Probe Training**: ~24-48 hours (depends on number of models/datasets)
- **Workflow Experiments**: ~2-4 hours per simulator
- **Total**: ~72 hours (script sets 72:00:00 time limit)

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f slurm_logs/full_pipeline_<job_id>.out
```

## Error Handling

- **Probe Training Failure**: Script exits immediately
- **Probe Verification Failure**: Script exits with error message
- **Workflow Experiment Failure**: Script logs warning and continues with next simulator

## Output Files

### Probe Training
- `experiments/all_probes_linear_max/all_layers_linear.json` - Full results (all layers)
- `experiments/all_probes_linear_max/best_layers_linear.json` - Best layer results
- Probe files in `{model_name}/train_on_{dataset}/probes_{type}/`

### Workflow Experiments
- `experiments/workflow_results/workflow_rag_results.json`
- `experiments/workflow_results/workflow_gpt5_results.json`
- `experiments/workflow_results/workflow_gpt5_recall_results.json`

Each workflow result contains:
```json
{
  "workflow": {
    "accuracy": 0.85,
    "avg_tokens": 450,
    "total_tokens": 45000,
    "num_examples": 100
  },
  "baseline_just_answer": {...},
  "baseline_always_prompt": {...}
}
```

## Troubleshooting

1. **Probe training takes too long**: Reduce number of models or use `--test` mode first
2. **Probes not found**: Check that `save_probes=True` (default) in `run_all_probes.py`
3. **API key errors**: Ensure `OPENAI_API_KEY` is set in environment
4. **Out of memory**: Increase `--mem-per-cpu` in SLURM directives

## Alternative: Run Steps Separately

If you prefer to run steps separately:

1. **Train probes only**:
   ```bash
   sbatch run.sh
   ```

2. **Run workflow experiments** (after probes are trained):
   ```bash
   sbatch run_workflow_experiment_all.sh
   ```


