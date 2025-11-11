# QUICKSTART GUIDE

## Step-by-Step: From Raw Data to Trained Probes

### Step 1: Preprocess All Datasets

```bash
cd /Users/prabhavsingh/Documents/CLASSES/Fall2025/CARDS/src

# UMWP (already done - data copied from PROPOSAL_1)
echo "✓ UMWP data already processed"

# GSM8K
echo "Processing GSM8K..."
python scripts/preprocess_gsm8k.py \
    --output_dir data/processed/gsm8k \
    --train_samples 4160 \
    --test_samples 1040 \
    --insufficient_ratio 0.5 \
    --random_seed 42

# TreeCut
echo "Processing TreeCut..."
python scripts/preprocess_treecut.py \
    --output_dir data/processed/treecut \
    --train_samples 4160 \
    --test_samples 1040 \
    --insufficient_ratio 0.5 \
    --random_seed 42
```

### Step 2: Extract Embeddings (Example for one model + dataset)

You'll need to create an embedding extraction script. Here's a template:

```python
# save as: extract_embeddings_example.py
import sys
sys.path.append('.')
from models.extractor import HiddenStateExtractor
from models.config import get_model_config
import json

def extract_for_dataset(model_name, dataset_name, split='train'):
    """Extract embeddings for a model-dataset-split combination"""

    # Paths
    data_path = f'data/processed/{dataset_name}/{dataset_name}_{split}.json'
    output_path = f'data/embeddings/{dataset_name}_{split}_{model_name}_embeddings.npy'
    metadata_path = output_path.replace('.npy', '_metadata.json')

    print(f"Extracting: {model_name} + {dataset_name} ({split})")
    print(f"  Input: {data_path}")
    print(f"  Output: {output_path}")

    # Get model config
    config = get_model_config(model_name)

    # Extract
    extractor = HiddenStateExtractor(
        model_name=config['name'],
        device='cuda'  # or 'cpu'
    )

    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)

    questions = [item['question'] for item in data]
    embeddings = extractor.extract_batch(questions, pooling='last_token')

    # Save
    import numpy as np
    np.save(output_path, embeddings)

    with open(metadata_path, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'split': split,
            'shape': list(embeddings.shape),
            'num_samples': len(data),
            'num_layers': embeddings.shape[1],
            'hidden_dim': embeddings.shape[2]
        }, f, indent=2)

    print(f"✓ Saved embeddings: {embeddings.shape}")
    print(f"✓ Saved metadata: {metadata_path}")

if __name__ == '__main__':
    # Example: Extract for Qwen-Math on UMWP
    extract_for_dataset('qwen2.5-math-1.5b', 'insufficient_dataset_umwp', 'train')
    extract_for_dataset('qwen2.5-math-1.5b', 'insufficient_dataset_umwp', 'test')
```

### Step 3: Train Probes

```bash
cd /Users/prabhavsingh/Documents/CLASSES/Fall2025/CARDS/src

# Train linear probes for Qwen-Math on UMWP
python probes/train_probes.py \
    --train_embeddings data/embeddings/insufficient_dataset_umwp_train_qwen2.5-math-1.5b_embeddings.npy \
    --train_labels data/processed/insufficient_dataset_umwp/umwp_train.json \
    --test_embeddings data/embeddings/insufficient_dataset_umwp_test_qwen2.5-math-1.5b_embeddings.npy \
    --test_labels data/processed/insufficient_dataset_umwp/umwp_test.json \
    --output_dir experiments/qwen_math_umwp_probes \
    --probe_type both \
    --device cuda
```

## What's Been Set Up

### ✓ Directory Structure
- `/src/data/` - Data storage (raw + processed)
- `/src/models/` - Model configs and utilities
- `/src/probes/` - Probe training scripts
- `/src/scripts/` - Data preprocessing scripts

### ✓ Model Configuration
4 models ready to use:
1. `qwen2.5-math-1.5b` - Math-specialized 1.5B
2. `qwen2.5-1.5b` - General 1.5B
3. `llama-3.2-3b-instruct` - General 3B
4. `llama-3.1-8b-instruct` - Large 8B

### ✓ Data Processing Scripts
- `scripts/preprocess_umwp.py` - UMWP preprocessing
- `scripts/preprocess_gsm8k.py` - GSM8K with synthetic insufficiency
- `scripts/preprocess_treecut.py` - TreeCut download and format

### ✓ Probe Training
- `probes/train_probes.py` - Train linear/MLP probes
- Supports both probe types
- Trains on all layers
- Saves metrics and weights

### ✓ UMWP Data
Already copied from PROPOSAL_1:
- `data/processed/insufficient_dataset_umwp/umwp_train.json`
- `data/processed/insufficient_dataset_umwp/umwp_test.json`

## Next Steps

1. **Run GSM8K and TreeCut preprocessing** (Step 1 above)
2. **Create embedding extraction script** (see template in Step 2)
3. **Extract embeddings for all model-dataset combinations** you want to test
4. **Train probes** (Step 3 above)
5. **Analyze results** - Create visualization scripts as needed

## Experiment Matrix

For full generalization testing, you'll want:

**Models (4):**
- qwen2.5-math-1.5b
- qwen2.5-1.5b
- llama-3.2-3b-instruct
- llama-3.1-8b-instruct

**Datasets (3):**
- UMWP
- GSM8K
- TreeCut

**Total combinations:** 4 × 3 = 12 (train + test for each)

## Troubleshooting

### If preprocessing fails:
- Check internet connection (for dataset downloads)
- Try with smaller `--train_samples` and `--test_samples`
- Check disk space for cache

### If embedding extraction fails:
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Try with `device='cpu'` (slower but should work)
- Check HuggingFace token for Llama models

### If probe training fails:
- Verify embedding files exist: `ls -lh data/embeddings/`
- Check embedding shape matches labels: shapes should be (N, L, D)
- Try with `--layers 0,5,10` to test specific layers first
