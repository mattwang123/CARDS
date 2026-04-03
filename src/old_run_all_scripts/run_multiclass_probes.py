"""
Multi-class probe training for UMWP 6-class insufficiency classification

This script:
1. Extracts embeddings for UMWP multiclass dataset
2. Trains 6-class linear probes (0=answerable, 1-5=insufficient types) 
3. Evaluates across all models and layers
4. Focuses on macro F1 and per-class performance
"""
import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.config import get_model_config
from models.extractor import HiddenStateExtractor


# Dataset configuration
DATASET_CONFIG = {
    'train': 'data/processed/umwp_multiclass/umwp_multiclass_train.json',
    'test': 'data/processed/umwp_multiclass/umwp_multiclass_test.json'
}

# Model names
MODELS = ['qwen2.5-math-1.5b', 'qwen2.5-1.5b', 'qwen2.5-math-7b']  # Removed problematic llama model

# Class names for interpretability
CLASS_NAMES = {
    0: "Answerable",
    1: "Missing Critical Info", 
    2: "Incomplete Constraint",
    3: "Contradictory Info",
    4: "Irrelevant Question",
    5: "Other Unanswerability"
}


def get_embedding_path(split, model_name, pooling):
    """Generate embedding file path"""
    filename = f"umwp_multiclass_{split}_{model_name}_{pooling}.npy"
    return f"data/embeddings/{filename}"


def extract_embeddings_if_needed(split, model_name, pooling, device):
    """Extract embeddings if they don't exist"""
    embedding_path = get_embedding_path(split, model_name, pooling)

    if os.path.exists(embedding_path):
        print(f"  ✓ Loading existing embeddings: {embedding_path}")
        return np.load(embedding_path)

    print(f"  → Extracting embeddings: {split} with {model_name}")

    # Load dataset
    data_path = DATASET_CONFIG[split]
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Create extractor
    extractor = HiddenStateExtractor(model_name, device=device)

    # Extract
    embeddings = extractor.extract_dataset(data, layers='all', pooling=pooling)

    # Save
    os.makedirs('data/embeddings', exist_ok=True)
    np.save(embedding_path, embeddings)

    print(f"  ✓ Saved embeddings: {embedding_path}")
    return embeddings


def load_multiclass_labels(split):
    """Load 6-class labels from dataset"""
    data_path = DATASET_CONFIG[split]
    with open(data_path, 'r') as f:
        data = json.load(f)

    labels = np.array([item['multiclass_label'] for item in data])
    return labels


def train_linear_probe(train_embeddings_layer, train_labels, C=1.0):
    """Train multiclass linear probe"""
    probe = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=5000, random_state=42, class_weight='balanced')
    )
    probe.fit(train_embeddings_layer, train_labels)
    return probe


def evaluate_linear_probe(probe, test_embeddings_layer, test_labels):
    """Evaluate multiclass linear probe"""
    predictions = probe.predict(test_embeddings_layer)

    # Multi-class metrics
    metrics = {
        'accuracy': float(accuracy_score(test_labels, predictions)),
        'f1_macro': float(f1_score(test_labels, predictions, average='macro')),
        'f1_weighted': float(f1_score(test_labels, predictions, average='weighted')),
        'precision_macro': float(precision_score(test_labels, predictions, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(test_labels, predictions, average='macro', zero_division=0))
    }
    
    # Add per-class F1 scores
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    for class_id in sorted([k for k in report.keys() if k.isdigit()]):
        metrics[f'f1_class_{class_id}'] = report[class_id]['f1-score']
        
    # Add confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def save_checkpoint(all_results, output_dir, completed_models, current_model=None, current_layer=None):
    """Save checkpoint with current progress"""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'completed_models': completed_models,
        'current_model': current_model,
        'current_layer': current_layer,
        'total_models': len(MODELS),
        'results': all_results
    }
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'checkpoint_multiclass_linear.json')
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(output_dir):
    """Load existing checkpoint if available"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_multiclass_linear.json')
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"  ✓ Loaded checkpoint from: {checkpoint_path}")
        return checkpoint
    
    return None


def train_all_layers(train_embeddings, train_labels, test_embeddings, test_labels, 
                    linear_C=1.0, output_dir=None, model_name=None, existing_results=None):
    """Train linear probes on all layers with checkpointing"""
    num_layers = train_embeddings.shape[1]
    results = existing_results if existing_results else {}
    
    # Determine starting layer
    start_layer = len(results)
    
    if start_layer > 0:
        print(f"Resuming from layer {start_layer} (found {start_layer} existing results)")
    
    print(f"Training linear probes on layers {start_layer}-{num_layers-1}...")

    for layer_idx in tqdm(range(start_layer, num_layers), desc="Layers"):
        # Get layer embeddings
        train_emb_layer = train_embeddings[:, layer_idx, :]
        test_emb_layer = test_embeddings[:, layer_idx, :]

        # Train probe
        probe = train_linear_probe(train_emb_layer, train_labels, C=linear_C)

        # Evaluate
        metrics = evaluate_linear_probe(probe, test_emb_layer, test_labels)
        results[f"layer_{layer_idx}"] = metrics
        
        # Save checkpoint every 5 layers
        if output_dir and model_name and (layer_idx + 1) % 5 == 0:
            # Create temporary checkpoint for this model's progress
            temp_results = {model_name: results}
            save_checkpoint(temp_results, output_dir, [], model_name, layer_idx)

    return results


def run_multiclass_experiment(pooling='last_token', device='cpu', linear_C=1.0, output_dir='experiments/multiclass_probes'):
    """Run multiclass experiment on all models with checkpointing"""
    
    # Try to load existing checkpoint
    checkpoint = load_checkpoint(output_dir)
    
    if checkpoint:
        all_results = checkpoint.get('results', {})
        completed_models = checkpoint.get('completed_models', [])
        print(f"Resuming experiment. Completed models: {completed_models}")
    else:
        all_results = {}
        completed_models = []
    
    print("="*80)
    print("MULTICLASS LINEAR PROBE EXPERIMENT (6 CLASSES)")
    print("="*80)
    print(f"Dataset: UMWP Multiclass")
    print(f"Models: {MODELS}")
    print(f"Pooling: {pooling}")
    print(f"Device: {device}")
    print(f"Completed models: {completed_models}")
    print("="*80)

    for model_name in MODELS:
        # Skip if already completed
        if model_name in completed_models:
            print(f"\n✓ Skipping {model_name} (already completed)")
            continue
            
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            # Extract embeddings
            print("Extracting embeddings...")
            train_embeddings = extract_embeddings_if_needed('train', model_name, pooling, device)
            test_embeddings = extract_embeddings_if_needed('test', model_name, pooling, device)

            # Load labels
            train_labels = load_multiclass_labels('train')
            test_labels = load_multiclass_labels('test')
            
            print(f"Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
            print(f"Embedding shape: {train_embeddings.shape}")
            
            # Print class distribution
            unique_train, counts_train = np.unique(train_labels, return_counts=True)
            print("Train class distribution:")
            for label, count in zip(unique_train, counts_train):
                class_name = CLASS_NAMES.get(label, f"Class_{label}")
                print(f"  {label} ({class_name}): {count} ({count/len(train_labels)*100:.1f}%)")

            # Check if this model has partial results
            existing_model_results = all_results.get(model_name, {})

            # Train and evaluate
            results = train_all_layers(
                train_embeddings, train_labels,
                test_embeddings, test_labels,
                linear_C=linear_C,
                output_dir=output_dir,
                model_name=model_name,
                existing_results=existing_model_results
            )

            all_results[model_name] = results
            completed_models.append(model_name)
            
            # Save checkpoint after each model
            save_checkpoint(all_results, output_dir, completed_models)
            
            print(f"✓ Completed {model_name}")
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            print("Saving checkpoint and continuing...")
            save_checkpoint(all_results, output_dir, completed_models, model_name, "ERROR")
            continue

    return all_results


def run_test_mode(pooling='last_token', device='cpu', linear_C=1.0):
    """Quick test mode: one model, few layers"""
    model_name = 'qwen2.5-math-1.5b'
    
    print("="*80)
    print("TEST MODE - Multiclass Linear Probe Test")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Pooling: {pooling}")
    print(f"Device: {device}")
    print("="*80)

    # Extract embeddings
    print("\n[1/3] Extracting embeddings...")
    train_embeddings = extract_embeddings_if_needed('train', model_name, pooling, device)
    test_embeddings = extract_embeddings_if_needed('test', model_name, pooling, device)

    # Load labels
    print("\n[2/3] Loading labels...")
    train_labels = load_multiclass_labels('train')
    test_labels = load_multiclass_labels('test')

    print(f"Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
    print(f"Embedding shape: {train_embeddings.shape}")

    # Print class distribution
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print("Class distribution:")
    for label, count in zip(unique_train, counts_train):
        class_name = CLASS_NAMES.get(label, f"Class_{label}")
        print(f"  {label} ({class_name}): {count}")

    # Test on first 5 layers only
    print(f"\n[3/3] Testing linear probes on first 5 layers...")
    
    for layer_idx in range(min(5, train_embeddings.shape[1])):
        print(f"\nLayer {layer_idx}:")
        
        # Get layer embeddings
        train_emb_layer = train_embeddings[:, layer_idx, :]
        test_emb_layer = test_embeddings[:, layer_idx, :]

        # Train probe
        probe = train_linear_probe(train_emb_layer, train_labels, C=linear_C)

        # Evaluate
        metrics = evaluate_linear_probe(probe, test_emb_layer, test_labels)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['f1_macro']:.4f}")
        print(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
        print(f"  Per-class F1:")
        for i in range(6):
            if f'f1_class_{i}' in metrics:
                class_name = CLASS_NAMES.get(i, f"Class_{i}")
                print(f"    {i} ({class_name}): {metrics[f'f1_class_{i}']:.3f}")

    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiclass linear probe experiments on UMWP (6 classes)'
    )
    parser.add_argument('--pooling', type=str, default='last_token',
                        choices=['last_token', 'mean', 'max'],
                        help='Pooling strategy for embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='experiments/multiclass_probes',
                        help='Directory to save results')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: one model, few layers, no saving')
    parser.add_argument('--linear_C', type=float, default=1.0,
                        help='Regularization parameter for linear probe')

    args = parser.parse_args()

    # Test mode
    if args.test:
        run_test_mode(
            pooling=args.pooling,
            device=args.device,
            linear_C=args.linear_C
        )
        return

    # Run full experiment
    results = run_multiclass_experiment(
        pooling=args.pooling,
        device=args.device,
        linear_C=args.linear_C,
        output_dir=args.output_dir
    )
    
    # Save final results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'multiclass_linear.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved final results: {results_path}")
    
    # Clean up checkpoint file
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_multiclass_linear.json')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Cleaned up checkpoint file")
    
    # Print summary
    print("\n" + "="*80)
    print("MULTICLASS SUMMARY - BEST LAYERS")
    print("="*80)
    
    for model_name in MODELS:
        if model_name in results:
            model_results = results[model_name]
            
            # Find best layer by macro F1
            best_layer = max(model_results.keys(), 
                           key=lambda k: model_results[k]['f1_macro'])
            best_layer_idx = int(best_layer.split('_')[1])
            best_metrics = model_results[best_layer]
            
            print(f"\n{model_name}:")
            print(f"  Best layer: {best_layer_idx}")
            print(f"  Macro F1: {best_metrics['f1_macro']:.4f}")
            print(f"  Weighted F1: {best_metrics['f1_weighted']:.4f}")
            print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            print(f"  Per-class F1:")
            for i in range(6):
                if f'f1_class_{i}' in best_metrics:
                    class_name = CLASS_NAMES.get(i, f"Class_{i}")
                    print(f"    {i} ({class_name}): {best_metrics[f'f1_class_{i}']:.3f}")

    print("\n" + "="*80)
    print("MULTICLASS EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()