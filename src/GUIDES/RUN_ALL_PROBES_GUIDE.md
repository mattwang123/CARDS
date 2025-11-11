# Run All Probes - Comprehensive Guide

## What This Script Does

`run_all_probes.py` is a comprehensive pipeline that:

1. **Extracts embeddings** for all model-dataset-split combinations (caches them for reuse)
2. **Trains probes** with sophisticated generalization testing
3. **Outputs structured results** in two JSON files

## The Generalization Matrix

For each of 4 models, we create a **4×3 generalization matrix**:

### Structure

**Rows** (4 training configurations):
1. Train on UMWP → Test on [UMWP, GSM8K, TreeCut]
2. Train on GSM8K → Test on [UMWP, GSM8K, TreeCut]
3. Train on TreeCut → Test on [UMWP, GSM8K, TreeCut]
4. **Train on ALL (combined)** → Test on [UMWP, GSM8K, TreeCut]

**Columns** (3 test datasets):
- UMWP
- GSM8K
- TreeCut

### Example Cell Interpretation

Cell `[Train on UMWP, Test on GSM8K]` tells us:
- How well a probe trained **only** on UMWP data
- Generalizes to the **unseen** GSM8K dataset
- This measures **cross-dataset generalization**

The "ALL" row is special:
- Train on **combined train sets** of all three datasets
- Test on **each dataset's test set individually**
- This measures performance with maximum training diversity

## Usage

### Basic Usage (Linear Probes with LR)

```bash
cd /Users/prabhavsingh/Documents/CLASSES/Fall2025/CARDS/src

python run_all_probes.py \
    --probe_type linear \
    --pooling last_token \
    --device cuda \
    --output_dir experiments/all_probes_linear
```

### MLP Probes (Neural Network)

```bash
python run_all_probes.py \
    --probe_type mlp \
    --pooling last_token \
    --device cuda \
    --mlp_hidden_dim 128 \
    --mlp_epochs 50 \
    --mlp_lr 0.001 \
    --output_dir experiments/all_probes_mlp
```

### All Arguments

```
--probe_type {linear,mlp}      # Type of probe (default: linear)
--pooling {last_token,mean,max}  # Pooling strategy (default: last_token)
--device {cpu,cuda}            # Computation device (default: cpu)
--output_dir PATH              # Where to save results (default: experiments/all_probes)

# Linear probe specific
--linear_C FLOAT               # L2 regularization (default: 1.0)

# MLP probe specific
--mlp_hidden_dim INT           # Hidden layer size (default: 128)
--mlp_epochs INT               # Training epochs (default: 50)
--mlp_lr FLOAT                 # Learning rate (default: 0.001)
```

## Output Files

### 1. `all_layers_{probe_type}.json`

Full results with **all layer performance**.

**Structure:**
```json
{
  "qwen2.5-math-1.5b": {
    "train_on_umwp": {
      "layer_0": {
        "test_on_umwp": {"accuracy": 0.85, "f1": 0.84, ...},
        "test_on_gsm8k": {"accuracy": 0.78, "f1": 0.77, ...},
        "test_on_treecut": {"accuracy": 0.72, "f1": 0.71, ...}
      },
      "layer_1": { ... },
      ...
      "layer_27": { ... }
    },
    "train_on_gsm8k": { ... },
    "train_on_treecut": { ... },
    "train_on_ALL": { ... }
  },
  "qwen2.5-1.5b": { ... },
  "llama-3.2-3b-instruct": { ... },
  "llama-3.1-8b-instruct": { ... }
}
```

**Use this for:**
- Layer-wise analysis
- Finding where in the model insufficiency is represented
- Plotting layer performance curves

### 2. `best_layers_{probe_type}.json`

Simplified results with **only the best layer** per model (based on average F1).

**Structure:**
```json
{
  "qwen2.5-math-1.5b": {
    "train_on_umwp": {
      "best_layer": 15,
      "avg_f1": 0.82,
      "results": {
        "test_on_umwp": {"accuracy": 0.85, "f1": 0.84, ...},
        "test_on_gsm8k": {"accuracy": 0.78, "f1": 0.77, ...},
        "test_on_treecut": {"accuracy": 0.72, "f1": 0.71, ...}
      }
    },
    "train_on_gsm8k": { ... },
    "train_on_treecut": { ... },
    "train_on_ALL": { ... }
  },
  ...
}
```

**Use this for:**
- Quick performance summary
- Creating generalization tables/plots
- Comparing model performance at optimal layers

## Workflow Example

### Step 1: Ensure datasets are preprocessed

```bash
# UMWP already done (copied from PROPOSAL_1)

# GSM8K
python scripts/preprocess_gsm8k.py \
    --output_dir data/processed/gsm8k \
    --train_samples 4160 \
    --test_samples 1040

# TreeCut
python scripts/preprocess_treecut.py \
    --output_dir data/processed/treecut \
    --train_samples 4160 \
    --test_samples 1040
```

### Step 2: Run linear probes

```bash
python run_all_probes.py \
    --probe_type linear \
    --pooling last_token \
    --device cuda \
    --linear_C 1.0 \
    --output_dir experiments/all_probes_linear
```

### Step 3: Run MLP probes

```bash
python run_all_probes.py \
    --probe_type mlp \
    --pooling last_token \
    --device cuda \
    --mlp_hidden_dim 128 \
    --mlp_epochs 50 \
    --mlp_lr 0.001 \
    --output_dir experiments/all_probes_mlp
```

### Step 4: Analyze results

Results are in:
- `experiments/all_probes_linear/all_layers_linear.json`
- `experiments/all_probes_linear/best_layers_linear.json`
- `experiments/all_probes_mlp/all_layers_mlp.json`
- `experiments/all_probes_mlp/best_layers_mlp.json`

## Creating Tables and Plots

### Example: Extract generalization table for Qwen-Math

```python
import json

with open('experiments/all_probes_linear/best_layers_linear.json', 'r') as f:
    results = json.load(f)

model_results = results['qwen2.5-math-1.5b']

# Create table
print("Train\\Test | UMWP  | GSM8K | TreeCut")
print("-" * 50)

for train_config in ['train_on_umwp', 'train_on_gsm8k', 'train_on_treecut', 'train_on_ALL']:
    train_name = train_config.replace('train_on_', '').upper()
    row = [train_name]

    test_results = model_results[train_config]['results']
    for test_dataset in ['umwp', 'gsm8k', 'treecut']:
        f1 = test_results[f'test_on_{test_dataset}']['f1']
        row.append(f"{f1:.3f}")

    print(" | ".join(row))
```

### Example: Plot layer performance

```python
import json
import matplotlib.pyplot as plt

with open('experiments/all_probes_linear/all_layers_linear.json', 'r') as f:
    results = json.load(f)

model_name = 'qwen2.5-math-1.5b'
train_config = 'train_on_umwp'

layer_results = results[model_name][train_config]
num_layers = len(layer_results)

layers = list(range(num_layers))
f1_scores = [layer_results[f'layer_{i}']['test_on_umwp']['f1'] for i in range(num_layers)]

plt.plot(layers, f1_scores)
plt.xlabel('Layer')
plt.ylabel('F1 Score')
plt.title(f'{model_name}: F1 vs Layer (trained on UMWP)')
plt.savefig('layer_performance.png')
```

## Embeddings Caching

The script automatically:
1. Checks if embeddings exist in `data/embeddings/`
2. Loads them if available
3. Extracts and saves them if not

**Embedding filename format:**
```
{dataset}_{split}_{model}_{pooling}.npy
```

**Example:**
```
umwp_train_qwen2.5-math-1.5b_last_token.npy
umwp_test_qwen2.5-math-1.5b_last_token.npy
```

## Time Estimates

On a GPU (RTX 3090 equivalent):
- **Embedding extraction**: ~2-5 min per model-dataset combination
- **Linear probe training**: ~30 sec per model-dataset combination
- **MLP probe training**: ~5 min per model-dataset combination

**Total time estimate:**
- Linear probes: ~2-3 hours (with embedding extraction)
- MLP probes: ~3-4 hours (with embedding extraction)
- If embeddings already extracted: ~30 min (linear), ~2 hours (MLP)

## Troubleshooting

### CUDA Out of Memory

Use smaller model first or CPU:
```bash
python run_all_probes.py --device cpu --probe_type linear
```

### Dataset not found

Make sure datasets are preprocessed:
```bash
ls -la data/processed/*/
```

Should show:
- `data/processed/insufficient_dataset_umwp/umwp_train.json`
- `data/processed/gsm8k/gsm8k_train.json`
- `data/processed/treecut/treecut_train.json`

### Import errors

Make sure you're in the src directory:
```bash
cd /Users/prabhavsingh/Documents/CLASSES/Fall2025/CARDS/src
python run_all_probes.py --help
```

## What Makes This "Comprehensive"?

1. **All models**: 4 models (Qwen-Math, Qwen-Base, Llama-3.2, Llama-3.1)
2. **All datasets**: 3 datasets (UMWP, GSM8K, TreeCut)
3. **All layers**: Every layer tested individually
4. **Cross-dataset generalization**: Train on one, test on all
5. **Combined training**: Special "ALL" configuration
6. **Both probe types**: Linear (LR) and non-linear (MLP)
7. **Automatic caching**: Embeddings reused across experiments
8. **Structured output**: Easy to parse and visualize

This gives us **4 models × 4 train configs × 3 test datasets × N layers** worth of data!
