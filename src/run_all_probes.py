"""
Comprehensive probe training and evaluation across all models and datasets

This script:
1. Extracts embeddings for all model-dataset combinations
2. Trains probes on each dataset and tests on all datasets (generalization)
3. Includes special "ALL_TRAIN" configuration (train on combined datasets)
4. Outputs two JSONs: full results (all layers) and best layer results
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
import torch.nn as nn
import torch.optim as optim

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.config import get_model_config, list_models
from models.extractor import HiddenStateExtractor
from models.probe import MLPProbe


# Dataset configurations
DATASETS = {
    'umwp': {
        'train': 'data/processed/insufficient_dataset_umwp/umwp_train.json',
        'test': 'data/processed/insufficient_dataset_umwp/umwp_test.json'
    },
    'gsm8k': {
        'train': 'data/processed/gsm8k/gsm8k_train.json',
        'test': 'data/processed/gsm8k/gsm8k_test.json'
    },
    'treecut': {
        'train': 'data/processed/treecut/treecut_train.json',
        'test': 'data/processed/treecut/treecut_test.json'
    }
}

# Model names from config
MODELS = ['qwen2.5-math-1.5b', 'qwen2.5-1.5b', 'llama-3.2-3b-instruct', 'qwen2.5-math-7b']


def get_embedding_path(dataset_name, split, model_name, pooling):
    """Generate embedding file path"""
    filename = f"{dataset_name}_{split}_{model_name}_{pooling}.npy"
    return f"data/embeddings/{filename}"


def extract_embeddings_if_needed(dataset_name, split, model_name, pooling, device):
    """
    Extract embeddings if they don't already exist

    Returns:
        numpy array: (num_samples, num_layers, hidden_dim)
    """
    embedding_path = get_embedding_path(dataset_name, split, model_name, pooling)

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
    os.makedirs('data/embeddings', exist_ok=True)
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

    print(f"  ✓ Saved embeddings: {embedding_path}")
    return embeddings


def load_labels(dataset_name, split):
    """Load labels from dataset"""
    data_path = DATASETS[dataset_name][split]
    with open(data_path, 'r') as f:
        data = json.load(f)

    labels = np.array([1 if item['is_sufficient'] else 0 for item in data])
    return labels


def train_linear_probe(train_embeddings_layer, train_labels, C=1.0):
    """
    Train linear probe for a single layer

    Args:
        train_embeddings_layer: (num_samples, hidden_dim)
        train_labels: (num_samples,)
        C: Regularization parameter

    Returns:
        LogisticRegression model
    """
    probe = LogisticRegression(C=C, max_iter=3000, random_state=42, class_weight='balanced')
    probe.fit(train_embeddings_layer, train_labels)
    return probe


def train_mlp_probe(train_embeddings_layer, train_labels, hidden_dim=128, num_epochs=50, lr=0.001, device='cpu'):
    """
    Train MLP probe for a single layer

    Args:
        train_embeddings_layer: (num_samples, hidden_dim)
        train_labels: (num_samples,)
        hidden_dim: MLP hidden dimension
        num_epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        MLPProbe model
    """
    input_dim = train_embeddings_layer.shape[1]

    probe = MLPProbe(input_dim, hidden_dim, num_classes=2)
    probe.to(device)

    # Convert to tensors
    X_train = torch.FloatTensor(train_embeddings_layer).to(device)
    y_train = torch.LongTensor(train_labels).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    # Train
    probe.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(probe, test_embeddings_layer, test_labels, probe_type='linear', device='cpu'):
    """
    Evaluate probe on test data

    Returns:
        dict: {accuracy, f1, precision, recall}
    """
    if probe_type == 'linear':
        predictions = probe.predict(test_embeddings_layer)
    else:  # mlp
        X_test = torch.FloatTensor(test_embeddings_layer).to(device)
        with torch.no_grad():
            outputs = probe(X_test)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

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
    probe_type='linear',
    device='cpu',
    mlp_hidden_dim=128,
    mlp_epochs=50,
    mlp_lr=0.001,
    linear_C=1.0
):
    """
    Train probes on all layers and evaluate on all test sets

    Args:
        train_embeddings: (num_train, num_layers, hidden_dim)
        train_labels: (num_train,)
        test_embeddings_dict: {dataset_name: (num_test, num_layers, hidden_dim)}
        test_labels_dict: {dataset_name: (num_test,)}
        probe_type: 'linear' or 'mlp'

    Returns:
        dict: {layer_idx: {test_dataset: metrics}}
    """
    num_layers = train_embeddings.shape[1]
    results = {}

    print(f"    Training {probe_type} probes on {num_layers} layers...")

    for layer_idx in tqdm(range(num_layers), desc=f"    Layers"):
        # Get layer embeddings
        train_emb_layer = train_embeddings[:, layer_idx, :]

        # Train probe
        if probe_type == 'linear':
            probe = train_linear_probe(train_emb_layer, train_labels, C=linear_C)
        else:
            probe = train_mlp_probe(
                train_emb_layer, train_labels,
                hidden_dim=mlp_hidden_dim,
                num_epochs=mlp_epochs,
                lr=mlp_lr,
                device=device
            )

        # Evaluate on all test sets
        layer_results = {}
        for test_dataset_name, test_embeddings in test_embeddings_dict.items():
            test_emb_layer = test_embeddings[:, layer_idx, :]
            test_labels = test_labels_dict[test_dataset_name]

            metrics = evaluate_probe(probe, test_emb_layer, test_labels, probe_type, device)
            layer_results[f"test_on_{test_dataset_name}"] = metrics

        results[f"layer_{layer_idx}"] = layer_results

    return results


def combine_datasets(dataset_names, split, pooling, model_name, device):
    """
    Combine multiple datasets

    Returns:
        embeddings: (total_samples, num_layers, hidden_dim)
        labels: (total_samples,)
    """
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


def run_all_experiments(probe_type='linear', pooling='last_token', device='cpu',
                        mlp_hidden_dim=128, mlp_epochs=50, mlp_lr=0.001, linear_C=1.0):
    """
    Run all probe experiments across models and datasets

    Returns:
        dict: Full results structure
    """
    all_results = {}
    dataset_names = list(DATASETS.keys())

    print("="*80)
    print(f"RUNNING ALL PROBE EXPERIMENTS ({probe_type.upper()})")
    print("="*80)
    print(f"Models: {MODELS}")
    print(f"Datasets: {dataset_names}")
    print(f"Pooling: {pooling}")
    print(f"Device: {device}")
    print("="*80)

    for model_name in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        model_results = {}

        # First, extract all embeddings for this model
        print(f"\n[Step 1/2] Extracting embeddings for {model_name}...")
        embeddings_cache = {}
        labels_cache = {}

        for dataset_name in dataset_names:
            for split in ['train', 'test']:
                print(f"  {dataset_name}/{split}:")
                emb = extract_embeddings_if_needed(dataset_name, split, model_name, pooling, device)
                labels = load_labels(dataset_name, split)

                embeddings_cache[f"{dataset_name}_{split}"] = emb
                labels_cache[f"{dataset_name}_{split}"] = labels

        # Train on each individual dataset
        print(f"\n[Step 2/2] Training and evaluating probes...")

        for train_dataset_name in dataset_names:
            print(f"\n  → Training on {train_dataset_name.upper()}...")

            # Get training data
            train_embeddings = embeddings_cache[f"{train_dataset_name}_train"]
            train_labels = labels_cache[f"{train_dataset_name}_train"]

            # Prepare test data (all datasets)
            test_embeddings_dict = {}
            test_labels_dict = {}
            for test_dataset_name in dataset_names:
                test_embeddings_dict[test_dataset_name] = embeddings_cache[f"{test_dataset_name}_test"]
                test_labels_dict[test_dataset_name] = labels_cache[f"{test_dataset_name}_test"]

            # Train and evaluate
            results = train_and_evaluate_all_layers(
                train_embeddings, train_labels,
                test_embeddings_dict, test_labels_dict,
                probe_type=probe_type,
                device=device,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_epochs=mlp_epochs,
                mlp_lr=mlp_lr,
                linear_C=linear_C
            )

            model_results[f"train_on_{train_dataset_name}"] = results

        # Train on ALL combined datasets
        print(f"\n  → Training on ALL (combined datasets)...")

        train_embeddings_all, train_labels_all = combine_datasets(
            dataset_names, 'train', pooling, model_name, device
        )

        # Prepare test data (all datasets, tested individually)
        test_embeddings_dict = {}
        test_labels_dict = {}
        for test_dataset_name in dataset_names:
            test_embeddings_dict[test_dataset_name] = embeddings_cache[f"{test_dataset_name}_test"]
            test_labels_dict[test_dataset_name] = labels_cache[f"{test_dataset_name}_test"]

        # Train and evaluate on ALL
        results_all = train_and_evaluate_all_layers(
            train_embeddings_all, train_labels_all,
            test_embeddings_dict, test_labels_dict,
            probe_type=probe_type,
            device=device,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_epochs=mlp_epochs,
            mlp_lr=mlp_lr,
            linear_C=linear_C
        )

        model_results["train_on_ALL"] = results_all

        all_results[model_name] = model_results

    return all_results


def compute_best_layers(full_results):
    """
    Compute best layer for each model based on average F1 across all test sets

    Returns:
        dict: Same structure as full_results but only best layer per model
    """
    best_results = {}

    for model_name, model_results in full_results.items():
        best_model_results = {}

        for train_config, layer_results in model_results.items():
            # Compute average F1 for each layer across all test sets
            layer_scores = {}

            for layer_name, test_results in layer_results.items():
                f1_scores = [metrics['f1'] for metrics in test_results.values()]
                avg_f1 = np.mean(f1_scores)
                layer_scores[layer_name] = avg_f1

            # Find best layer
            best_layer = max(layer_scores, key=layer_scores.get)
            best_layer_idx = int(best_layer.split('_')[1])

            # Store only best layer results
            best_model_results[train_config] = {
                'best_layer': best_layer_idx,
                'avg_f1': layer_scores[best_layer],
                'results': layer_results[best_layer]
            }

        best_results[model_name] = best_model_results

    return best_results


def run_test_mode(probe_type='linear', pooling='last_token', device='cpu',
                  mlp_hidden_dim=128, mlp_epochs=50, mlp_lr=0.001, linear_C=1.0):
    """
    Quick test mode: Train and test on UMWP only with one model
    No saving, just print results
    """
    model_name = 'qwen2.5-math-1.5b'
    dataset_name = 'umwp'

    print("="*80)
    print("TEST MODE - Quick Probe Test on UMWP")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Probe type: {probe_type}")
    print(f"Pooling: {pooling}")
    print(f"Device: {device}")
    print("="*80)

    # Extract embeddings
    print("\n[1/3] Extracting embeddings...")
    train_embeddings = extract_embeddings_if_needed(dataset_name, 'train', model_name, pooling, device)
    test_embeddings = extract_embeddings_if_needed(dataset_name, 'test', model_name, pooling, device)

    # Load labels
    print("\n[2/3] Loading labels...")
    train_labels = load_labels(dataset_name, 'train')
    test_labels = load_labels(dataset_name, 'test')

    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Embedding shape: {train_embeddings.shape}")

    # Train and evaluate
    print(f"\n[3/3] Training {probe_type} probes on all layers...")
    num_layers = train_embeddings.shape[1]

    layer_results = []

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        # Get layer embeddings
        train_emb_layer = train_embeddings[:, layer_idx, :]
        test_emb_layer = test_embeddings[:, layer_idx, :]

        # Train probe
        if probe_type == 'linear':
            probe = train_linear_probe(train_emb_layer, train_labels, C=linear_C)
        else:
            probe = train_mlp_probe(
                train_emb_layer, train_labels,
                hidden_dim=mlp_hidden_dim,
                num_epochs=mlp_epochs,
                lr=mlp_lr,
                device=device
            )

        # Evaluate
        metrics = evaluate_probe(probe, test_emb_layer, test_labels, probe_type, device)
        layer_results.append((layer_idx, metrics))

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'Layer':<8} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)

    for layer_idx, metrics in layer_results:
        print(f"{layer_idx:<8} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

    # Find best layer
    best_layer_idx = max(range(len(layer_results)), key=lambda i: layer_results[i][1]['f1'])
    best_metrics = layer_results[best_layer_idx][1]

    print("\n" + "="*80)
    print("BEST LAYER SUMMARY")
    print("="*80)
    print(f"Best layer: {best_layer_idx}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive probe experiments across all models and datasets'
    )
    parser.add_argument('--probe_type', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='Type of probe: linear (LR) or mlp (neural network)')
    parser.add_argument('--pooling', type=str, default='last_token',
                        choices=['last_token', 'mean', 'max'],
                        help='Pooling strategy for embeddings')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for computation')
    parser.add_argument('--output_dir', type=str, default='experiments/all_probes',
                        help='Directory to save results')

    # Test mode
    parser.add_argument('--test', action='store_true',
                        help='Test mode: quick run on UMWP only, no saving')

    # MLP-specific arguments
    parser.add_argument('--mlp_hidden_dim', type=int, default=128,
                        help='Hidden dimension for MLP probe')
    parser.add_argument('--mlp_epochs', type=int, default=50,
                        help='Number of epochs for MLP training')
    parser.add_argument('--mlp_lr', type=float, default=0.001,
                        help='Learning rate for MLP training')

    # Linear probe arguments
    parser.add_argument('--linear_C', type=float, default=1.0,
                        help='Regularization parameter for linear probe')

    args = parser.parse_args()

    # Test mode
    if args.test:
        run_test_mode(
            probe_type=args.probe_type,
            pooling=args.pooling,
            device=args.device,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_epochs=args.mlp_epochs,
            mlp_lr=args.mlp_lr,
            linear_C=args.linear_C
        )
        return

    # Run experiments
    full_results = run_all_experiments(
        probe_type=args.probe_type,
        pooling=args.pooling,
        device=args.device,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_epochs=args.mlp_epochs,
        mlp_lr=args.mlp_lr,
        linear_C=args.linear_C
    )

    # Compute best layers
    best_results = compute_best_layers(full_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save full results
    full_path = os.path.join(args.output_dir, f'all_layers_{args.probe_type}.json')
    with open(full_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✓ Saved full results (all layers): {full_path}")

    # Save best layer results
    best_path = os.path.join(args.output_dir, f'best_layers_{args.probe_type}.json')
    with open(best_path, 'w') as f:
        json.dump(best_results, f, indent=2)
    print(f"✓ Saved best layer results: {best_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - BEST LAYERS")
    print("="*80)
    for model_name in MODELS:
        print(f"\n{model_name}:")
        for train_config, results in best_results[model_name].items():
            print(f"  {train_config}:")
            print(f"    Best layer: {results['best_layer']}")
            print(f"    Avg F1: {results['avg_f1']:.4f}")
            print(f"    Per-dataset F1:")
            for test_name, metrics in results['results'].items():
                print(f"      {test_name}: {metrics['f1']:.4f}")

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
