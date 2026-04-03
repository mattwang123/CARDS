"""
================================================================================
EXPERIMENT 1A: Comprehensive Probe Training & Evaluation (Internal Representation)
================================================================================

This script:
1. Extracts hidden state embeddings for all model-dataset combinations.
2. Trains Linear Probes (Logistic Regression) on each dataset's training split.
3. Tests the probes on all datasets to evaluate cross-dataset generalization.
4. Includes a special "ALL" configuration (trains on combined datasets).
5. Incrementally saves progress (checkpointing) so interrupted runs don't lose data.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.extractor import HiddenStateExtractor

# --- CONFIGURATION ---
# Aligned with the verbalization script: Math domain only.
DATASETS = {
    'umwp': {
        'train': 'src/data/processed/insufficient_dataset_umwp/umwp_train.json',
        'test': 'src/data/processed/insufficient_dataset_umwp/umwp_test.json'
    },
    'treecut': {
        'train': 'src/data/processed/treecut/treecut_train.json',
        'test': 'src/data/processed/treecut/treecut_test.json'
    }
}

# The Top-Tier Conference Model Matrix
MODELS = [
    # --- SMALL/MEDIUM SCALE (~1.5B - 3B) ---
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    
    # --- OLMo FAMILY (Valid HF IDs) ---
    'allenai/OLMo-7B-0724-hf',               # Base
    'allenai/OLMo-7B-Instruct-hf',           # Instruct
    'allenai/OLMoE-1B-7B-0924-Instruct',     # MoE Instruct
    
    # --- MEDIUM/LARGE SCALE (~7B - 8B) ---
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    
    # --- LARGE SCALE (14B+) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
]

def get_embedding_path(dataset_name, split, model_name, pooling):
    """Generate embedding file path. Slugifies model name to prevent directory errors."""
    model_slug = model_name.replace('/', '_')
    filename = f"{dataset_name}_{split}_{model_slug}_{pooling}.npy"
    return f"src/data/embeddings/{filename}"

def extract_embeddings_if_needed(dataset_name, split, model_name, pooling, device):
    """
    Extract embeddings if they don't already exist.
    Returns: numpy array: (num_samples, num_layers, hidden_dim)
    """
    embedding_path = get_embedding_path(dataset_name, split, model_name, pooling)
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

    if os.path.exists(embedding_path):
        print(f"  ✓ Loading existing embeddings: {embedding_path}")
        return np.load(embedding_path)

    print(f"  → Extracting embeddings: {dataset_name}/{split} with {model_name}")

    # Load dataset
    data_path = DATASETS[dataset_name][split]
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Create extractor
    extractor = HiddenStateExtractor(model_name, device=device)

    # Extract
    embeddings = extractor.extract_dataset(data, layers='all', pooling=pooling)

    # Save
    np.save(embedding_path, embeddings)

    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'split': split,
        'model': model_name,
        'pooling': pooling,
        'shape': list(embeddings.shape)
    }
    metadata_path = embedding_path.replace('.npy', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    del extractor
    torch.cuda.empty_cache()

    print(f"  ✓ Saved embeddings: {embedding_path}")
    return embeddings

def load_labels(dataset_name, split):
    """Load labels from dataset"""
    data_path = DATASETS[dataset_name][split]
    with open(data_path, 'r') as f:
        data = json.load(f)
    labels = np.array([1 if item.get('is_sufficient', False) else 0 for item in data])
    return labels

def train_linear_probe(train_embeddings_layer, train_labels, C=1.0):
    """Train linear probe for a single layer"""
    probe = LogisticRegression(C=C, max_iter=5000, random_state=42, class_weight='balanced')
    probe.fit(train_embeddings_layer, train_labels)
    return probe

def evaluate_probe(probe, test_embeddings_layer, test_labels):
    """Evaluate linear probe on test data"""
    predictions = probe.predict(test_embeddings_layer)

    metrics = {
        'accuracy': float(accuracy_score(test_labels, predictions)),
        'f1': float(f1_score(test_labels, predictions, average='binary')),
        'precision': float(precision_score(test_labels, predictions, average='binary', zero_division=0)),
        'recall': float(recall_score(test_labels, predictions, average='binary', zero_division=0))
    }
    return metrics

def train_and_evaluate_all_layers(
    train_embeddings, train_labels,
    test_embeddings_dict, test_labels_dict,
    linear_C=1.0
):
    """Train probes on all layers and evaluate on all test sets"""
    num_layers = train_embeddings.shape[1]
    results = {}

    print(f"    Training Linear probes on {num_layers} layers...")

    for layer_idx in tqdm(range(num_layers), desc=f"    Layers"):
        train_emb_layer = train_embeddings[:, layer_idx, :]

        probe = train_linear_probe(train_emb_layer, train_labels, C=linear_C)

        layer_results = {}
        for test_dataset_name, test_embeddings in test_embeddings_dict.items():
            test_emb_layer = test_embeddings[:, layer_idx, :]
            test_labels = test_labels_dict[test_dataset_name]
            metrics = evaluate_probe(probe, test_emb_layer, test_labels)
            layer_results[f"test_on_{test_dataset_name}"] = metrics

        results[f"layer_{layer_idx}"] = layer_results

    return results

def combine_datasets(dataset_names, split, pooling, model_name, device):
    """Combine multiple datasets"""
    all_embeddings = []
    all_labels = []

    for dataset_name in dataset_names:
        emb = extract_embeddings_if_needed(dataset_name, split, model_name, pooling, device)
        labels = load_labels(dataset_name, split)
        all_embeddings.append(emb)
        all_labels.append(labels)

    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    return combined_embeddings, combined_labels

def run_all_experiments(args):
    """Run all probe experiments across models and datasets with checkpointing"""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_file = os.path.join(args.output_dir, 'checkpoint_all_layers_linear.json')
    
    all_results = {}
    if os.path.exists(checkpoint_file) and not args.test:
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'r') as f:
            all_results = json.load(f)
            
    dataset_names = list(DATASETS.keys())

    print("="*80)
    print("RUNNING ALL PROBE EXPERIMENTS (LINEAR)")
    print("="*80)

    for model_name in MODELS:
        if model_name in all_results and not args.test:
            print(f"\n✓ Skipping {model_name} (Already in checkpoint)")
            continue
            
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        model_results = {}

        # Step 1: Extract Embeddings
        print(f"\n[Step 1/2] Extracting embeddings for {model_name}...")
        embeddings_cache = {}
        labels_cache = {}

        try:
            for dataset_name in dataset_names:
                for split in ['train', 'test']:
                    emb = extract_embeddings_if_needed(dataset_name, split, model_name, args.pooling, args.device)
                    labels = load_labels(dataset_name, split)
                    embeddings_cache[f"{dataset_name}_{split}"] = emb
                    labels_cache[f"{dataset_name}_{split}"] = labels
        except Exception as e:
            print(f"  ! Failed to extract embeddings for {model_name}: {e}")
            continue # Skip to next model if extraction fails (e.g. OOM)

        # Step 2: Train and evaluate probes
        print(f"\n[Step 2/2] Training and evaluating probes...")

        for train_dataset_name in dataset_names:
            print(f"\n  → Training on {train_dataset_name.upper()}...")
            train_embeddings = embeddings_cache[f"{train_dataset_name}_train"]
            train_labels = labels_cache[f"{train_dataset_name}_train"]

            test_embeddings_dict = {name: embeddings_cache[f"{name}_test"] for name in dataset_names}
            test_labels_dict = {name: labels_cache[f"{name}_test"] for name in dataset_names}

            results = train_and_evaluate_all_layers(
                train_embeddings, train_labels,
                test_embeddings_dict, test_labels_dict,
                linear_C=args.linear_C
            )
            model_results[f"train_on_{train_dataset_name}"] = results

        # Train on ALL combined datasets
        print(f"\n  → Training on ALL (combined datasets)...")
        train_embeddings_all, train_labels_all = combine_datasets(dataset_names, 'train', args.pooling, model_name, args.device)

        results_all = train_and_evaluate_all_layers(
            train_embeddings_all, train_labels_all,
            test_embeddings_dict, test_labels_dict,
            linear_C=args.linear_C
        )
        model_results["train_on_ALL"] = results_all

        # Save Checkpoint
        all_results[model_name] = model_results
        if not args.test:
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ Checkpoint updated for {model_name}.")

    return all_results

def compute_best_layers(full_results):
    """Compute best layer for each model based on average F1 across all test sets"""
    best_results = {}
    for model_name, model_results in full_results.items():
        best_model_results = {}
        for train_config, layer_results in model_results.items():
            layer_scores = {}
            for layer_name, test_results in layer_results.items():
                f1_scores = [metrics['f1'] for metrics in test_results.values()]
                layer_scores[layer_name] = np.mean(f1_scores)

            best_layer = max(layer_scores, key=layer_scores.get)
            best_layer_idx = int(best_layer.split('_')[1])
            best_model_results[train_config] = {
                'best_layer': best_layer_idx,
                'avg_f1': layer_scores[best_layer],
                'results': layer_results[best_layer]
            }
        best_results[model_name] = best_model_results
    return best_results

def run_test_mode(args):
    """Quick test mode"""
    model_name = 'Qwen/Qwen2.5-Math-1.5B'
    dataset_name = 'umwp'
    print("="*80 + "\nTEST MODE - Quick Probe Test on UMWP\n" + "="*80)
    
    train_embeddings = extract_embeddings_if_needed(dataset_name, 'train', model_name, args.pooling, args.device)
    test_embeddings = extract_embeddings_if_needed(dataset_name, 'test', model_name, args.pooling, args.device)
    
    train_labels = load_labels(dataset_name, 'train')
    test_labels = load_labels(dataset_name, 'test')
    
    num_layers = train_embeddings.shape[1]
    layer_results = []

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        train_emb_layer = train_embeddings[:, layer_idx, :]
        test_emb_layer = test_embeddings[:, layer_idx, :]
        probe = train_linear_probe(train_emb_layer, train_labels, C=args.linear_C)
        metrics = evaluate_probe(probe, test_emb_layer, test_labels)
        layer_results.append((layer_idx, metrics))

    best_layer_idx = max(range(len(layer_results)), key=lambda i: layer_results[i][1]['f1'])
    best_metrics = layer_results[best_layer_idx][1]
    print(f"\nBest layer: {best_layer_idx} | F1: {best_metrics['f1']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive probe experiments')
    parser.add_argument('--pooling', type=str, default='last_token', choices=['last_token', 'mean', 'max'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--output_dir', type=str, default='experiments/probing')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--linear_C', type=float, default=1.0)
    args = parser.parse_args()

    if args.test:
        run_test_mode(args)
        return

    full_results = run_all_experiments(args)
    best_results = compute_best_layers(full_results)

    # Save final results
    full_path = os.path.join(args.output_dir, 'all_layers_linear.json')
    with open(full_path, 'w') as f: json.dump(full_results, f, indent=2)
    
    best_path = os.path.join(args.output_dir, 'best_layers_linear.json')
    with open(best_path, 'w') as f: json.dump(best_results, f, indent=2)
    
    print(f"\n✓ Saved final linear probe results to {args.output_dir}")

if __name__ == '__main__':
    main()