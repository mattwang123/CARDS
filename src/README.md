# CARDS Project - Source Code

This directory contains the reorganized and cleaned codebase for the CARDS project.

## Directory Structure

```
src/
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   └── processed/              # Processed datasets in unified JSON format
│       ├── insufficient_dataset_umwp/
│       ├── gsm8k/
│       └── treecut/
├── models/
│   ├── config.py               # Model configuration (4 core models)
│   ├── extractor.py            # Embedding extraction utilities
│   ├── inference.py            # Model inference utilities
│   ├── answer_parser.py        # Answer parsing utilities
│   └── probe.py                # Probe architecture definitions
├── probes/
│   └── train_probes.py         # Probe training script (linear & MLP)
└── scripts/
    ├── preprocess_umwp.py      # UMWP dataset preprocessing
    ├── preprocess_gsm8k.py     # GSM8K dataset preprocessing
    └── preprocess_treecut.py   # TreeCut dataset preprocessing
```

## Core Models

We use 4 models for experiments (defined in `models/config.py`):

1. **qwen2.5-math-1.5b**: Math-specialized 1.5B model
2. **qwen2.5-1.5b**: General-purpose 1.5B model (non-math)
3. **llama-3.2-3b-instruct**: General instruction-tuned 3B model
4. **llama-3.1-8b-instruct**: Larger 8B model for scale testing

## Datasets

All datasets follow a unified JSON format:

### Sufficient Example:
```json
{
  "question": "Jake has 9 fewer peaches than Steven...",
  "answer": "#### 27",
  "is_sufficient": true
}
```

### Insufficient Example:
```json
{
  "question": "Ali is collecting bottle caps. He has red ones and green ones...",
  "answer": "N/A",
  "is_sufficient": false,
  "original_question": "Ali is collecting bottle caps. He has 125 bottle caps...",
  "removed_value": "125",
  "removed_description": "Missing critical information: total count"
}
```

## Usage

### 1. Preprocess Datasets

#### UMWP (already processed - copied from PROPOSAL_1)
```bash
# Already available in src/data/processed/insufficient_dataset_umwp/
```

#### GSM8K
```bash
cd src
python scripts/preprocess_gsm8k.py \
    --output_dir data/processed/gsm8k \
    --train_samples 4160 \
    --test_samples 1040 \
    --insufficient_ratio 0.5
```

#### TreeCut
```bash
cd src
python scripts/preprocess_treecut.py \
    --output_dir data/processed/treecut \
    --train_samples 4160 \
    --test_samples 1040 \
    --insufficient_ratio 0.5
```

### 2. Extract Embeddings

```bash
cd src
python -c "
from models.extractor import extract_embeddings
from models.config import get_model_config

# Extract embeddings for a model and dataset
model_name = 'qwen2.5-math-1.5b'
dataset_path = 'data/processed/insufficient_dataset_umwp/umwp_train.json'
output_path = 'data/embeddings/umwp_train_qwen_embeddings.npy'

extract_embeddings(
    model_name=model_name,
    data_path=dataset_path,
    output_path=output_path,
    pooling='last_token',
    device='cuda'  # or 'cpu'
)
"
```

### 3. Train Probes

#### Train Linear Probes
```bash
cd src
python probes/train_probes.py \
    --train_embeddings data/embeddings/umwp_train_qwen_embeddings.npy \
    --train_labels data/processed/insufficient_dataset_umwp/umwp_train.json \
    --test_embeddings data/embeddings/umwp_test_qwen_embeddings.npy \
    --test_labels data/processed/insufficient_dataset_umwp/umwp_test.json \
    --output_dir experiments/qwen_umwp_linear \
    --probe_type linear \
    --C 1.0
```

#### Train MLP Probes
```bash
cd src
python probes/train_probes.py \
    --train_embeddings data/embeddings/umwp_train_qwen_embeddings.npy \
    --train_labels data/processed/insufficient_dataset_umwp/umwp_train.json \
    --test_embeddings data/embeddings/umwp_test_qwen_embeddings.npy \
    --test_labels data/processed/insufficient_dataset_umwp/umwp_test.json \
    --output_dir experiments/qwen_umwp_mlp \
    --probe_type mlp \
    --hidden_dim 128 \
    --num_epochs 50 \
    --lr 0.001
```

#### Train Both
```bash
cd src
python probes/train_probes.py \
    --train_embeddings data/embeddings/umwp_train_qwen_embeddings.npy \
    --train_labels data/processed/insufficient_dataset_umwp/umwp_train.json \
    --test_embeddings data/embeddings/umwp_test_qwen_embeddings.npy \
    --test_labels data/processed/insufficient_dataset_umwp/umwp_test.json \
    --output_dir experiments/qwen_umwp_both \
    --probe_type both
```

## Model Configuration

To list available models:
```bash
cd src
python -c "from models.config import list_models; list_models()"
```

To get a specific model config:
```python
from models.config import get_model_config

config = get_model_config('qwen2.5-math-1.5b')
print(f"Model: {config['name']}")
print(f"Layers: {config['num_layers']}")
print(f"Hidden size: {config['hidden_size']}")
```

## Dataset Format Requirements

All preprocessing scripts produce datasets with:
- Balanced sufficient/insufficient examples (50/50 by default)
- Train/test split (80/20 by default)
- Matching sizes across datasets (4160 train, 1040 test by default)
- Unified JSON structure for compatibility

## Notes

- All embeddings use **last-token pooling** by default
- Probes are trained independently for each layer
- Linear probes use L2-regularized logistic regression
- MLP probes use 2-layer architecture with ReLU
- All scripts support `--help` for detailed options

## Dependencies

```bash
pip install torch transformers datasets scikit-learn numpy tqdm
```

## Next Steps

After running the above:
1. Run cross-dataset experiments (train on UMWP, test on GSM8K/TreeCut)
2. Run cross-model experiments (train on Qwen, test on Llama)
3. Analyze results with visualization scripts
4. Proceed to Stage II (activation steering & clarification)
