"""
Probe Evaluation for Three-Channel Correlation Analysis

This script:
1. Extracts embeddings for all model-dataset combinations
2. Trains probes on best layers only (for efficiency)
3. Saves sample-level predictions for correlation analysis
4. Outputs both aggregate metrics and detailed predictions
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

from models.config import get_model_config
from models.extractor import HiddenStateExtractor

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
MODELS = ['qwen2.5-math-1.5b', 'qwen2.5-1.5b', 'qwen2.5-math-7b']


def get_embedding_path(dataset_name, split, model_name, pooling):
    """Generate embedding file path"""
    filename = f"{dataset_name}_{split}_{model_name}_{pooling}.npy"
    return f"data/embeddings/{filename}"


def extract_embeddings_if_needed(dataset_name, split, model_name, pooling, device):
    """Extract embeddings if they don't already exist"""
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


def load_dataset_with_questions(dataset_name, split):
    """Load dataset with questions and labels"""
    data_path = DATASETS[dataset_name][split]
    with open(data_path, 'r') as f:
        data = json.load(f)

    labels = np.array([1 if item['is_sufficient'] else 0 for item in data])
    return data, labels


def train_linear_probe(train_embeddings_layer, train_labels, C=1.0):
    """Train linear probe for a single layer"""
    probe = LogisticRegression(C=C, max_iter=5000, random_state=42, class_weight='balanced')
    probe.fit(train_embeddings_layer, train_labels)
    return probe


def evaluate_probe_with_predictions(probe, test_embeddings_layer, test_labels, test_questions):
    """
    Evaluate probe and return both metrics and individual predictions
    
    Returns:
        tuple: (metrics, detailed_predictions)
    """
    predictions = probe.predict(test_embeddings_layer)
    probabilities = probe.predict_proba(test_embeddings_layer)[:, 1]  # Probability of class 1 (sufficient)

    # Compute aggregate metrics
    metrics = {
        'accuracy': float(accuracy_score(test_labels, predictions)),
        'f1': float(f1_score(test_labels, predictions, average='binary')),
        'precision': float(precision_score(test_labels, predictions, average='binary', zero_division=0)),
        'recall': float(recall_score(test_labels, predictions, average='binary', zero_division=0))
    }
    
    # Create detailed predictions for correlation analysis
    detailed_predictions = []
    for i in range(len(test_labels)):
        detailed_predictions.append({
            'question_idx': i,
            'question': test_questions[i]['question'],
            'is_sufficient': bool(test_labels[i]),
            'probe_prediction': int(predictions[i]),  # 0=insufficient, 1=sufficient
            'probe_confidence': float(probabilities[i]),  # Confidence in sufficient prediction
            'probe_correct': bool(predictions[i] == test_labels[i])
        })

    return metrics, detailed_predictions


def find_best_layer_from_existing_results(model_name, dataset_name, probe_results_dir):
    """Find best layer from existing comprehensive probe results"""
    best_results_path = os.path.join(probe_results_dir, 'best_layers_linear.json')
    
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)
        
        # Get best layer for this model-dataset combination
        if model_name in best_results:
            train_config = f"train_on_{dataset_name}"
            if train_config in best_results[model_name]:
                return best_results[model_name][train_config]['best_layer']
    
    # Default fallback - use middle layer
    return 15  # Reasonable default for most models


def evaluate_model_on_dataset(model_name, dataset_name, device, pooling, output_dir, 
                               probe_results_dir, test_mode=False, test_samples=3):
    """Evaluate probe on best layer with sample-level predictions"""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} | DATASET: {dataset_name} | PROBE CORRELATION")
    print(f"{'='*80}")

    # Find best layer from existing results
    best_layer = find_best_layer_from_existing_results(model_name, dataset_name, probe_results_dir)
    print(f"Using best layer: {best_layer}")

    # Load datasets with questions
    train_questions, train_labels = load_dataset_with_questions(dataset_name, 'train')
    test_questions, test_labels = load_dataset_with_questions(dataset_name, 'test')
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")

    if test_mode:
        test_questions = test_questions[:test_samples]
        test_labels = test_labels[:test_samples]
        print(f"TEST MODE: Using only {len(test_questions)} samples")

    # Extract embeddings
    print(f"\n[1/3] Extracting embeddings...")
    train_embeddings = extract_embeddings_if_needed(dataset_name, 'train', model_name, pooling, device)
    test_embeddings = extract_embeddings_if_needed(dataset_name, 'test', model_name, pooling, device)

    # Get best layer embeddings
    print(f"\n[2/3] Training probe on layer {best_layer}...")
    train_emb_layer = train_embeddings[:, best_layer, :]
    test_emb_layer = test_embeddings[:, best_layer, :]
    
    if test_mode:
        test_emb_layer = test_emb_layer[:test_samples]

    # Train probe
    probe = train_linear_probe(train_emb_layer, train_labels, C=1.0)

    # Evaluate with detailed predictions
    print(f"\n[3/3] Evaluating and generating predictions...")
    metrics, detailed_predictions = evaluate_probe_with_predictions(
        probe, test_emb_layer, test_labels, test_questions
    )

    # Create final results
    final_results = {
        'model': model_name,
        'dataset': dataset_name,
        'mode': 'probe_correlation',
        'best_layer': best_layer,
        'total_questions': len(detailed_predictions),
        'metrics': metrics,
        'detailed_predictions': detailed_predictions
    }

    # Save results
    if not test_mode:
        results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_probe_predictions.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✓ Saved results: {results_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} on {dataset_name} (Probe Correlation)")
    print(f"{'='*80}")
    print(f"Best layer: {best_layer}")
    print(f"Total questions: {len(detailed_predictions)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Show sample predictions in test mode
    if test_mode:
        print(f"\nSample Predictions:")
        for i, pred in enumerate(detailed_predictions[:3]):
            print(f"  Q{i+1}: Ground={pred['is_sufficient']}, Pred={bool(pred['probe_prediction'])}, "
                  f"Conf={pred['probe_confidence']:.3f}, Correct={pred['probe_correct']}")
    
    print(f"{'='*80}")

    return final_results


def run_test_mode(device, pooling, probe_results_dir, test_samples=3):
    """Quick test mode: 1 model, 1 dataset, few samples with detailed output"""
    model_name = 'qwen2.5-math-1.5b'
    dataset_name = 'umwp'

    print("="*80)
    print("TEST MODE - Probe Correlation Analysis")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {test_samples}")
    print("="*80)

    result = evaluate_model_on_dataset(
        model_name, dataset_name, device, pooling, 
        output_dir=None, probe_results_dir=probe_results_dir,
        test_mode=True, test_samples=test_samples
    )

    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Best layer used: {result['best_layer']}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {result['metrics']['f1']:.4f}")
    print(f"Sample predictions saved: {len(result['detailed_predictions'])}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate probe predictions for three-channel correlation analysis'
    )
    parser.add_argument('--pooling', type=str, default='last_token',
                        choices=['last_token', 'mean', 'max'],
                        help='Pooling strategy for embeddings')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for computation')
    parser.add_argument('--output_dir', type=str, default='experiments/probe_correlation',
                        help='Directory to save correlation results')
    parser.add_argument('--probe_results_dir', type=str, default='experiments/all_probes',
                        help='Directory containing existing probe results (for best layers)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: quick run on UMWP only, no saving')
    parser.add_argument('--test_samples', type=int, default=3,
                        help='Number of samples in test mode')

    args = parser.parse_args()

    # Test mode
    if args.test:
        run_test_mode(args.device, args.pooling, args.probe_results_dir, args.test_samples)
        return

    # Full evaluation mode
    print("="*80)
    print("PROBE CORRELATION ANALYSIS - ALL MODELS ON ALL DATASETS")
    print("="*80)
    print(f"Models: {MODELS}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Device: {args.device}")
    print(f"Pooling: {args.pooling}")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all evaluations
    all_results = {}

    for model_name in MODELS:
        model_results = {}

        for dataset_name in DATASETS.keys():
            result = evaluate_model_on_dataset(
                model_name, dataset_name, args.device, args.pooling,
                args.output_dir, args.probe_results_dir
            )
            model_results[dataset_name] = {
                'best_layer': result['best_layer'],
                'accuracy': result['metrics']['accuracy'],
                'f1': result['metrics']['f1'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'total_predictions': result['total_questions']
            }

        all_results[model_name] = model_results

    # Save summary
    summary_path = os.path.join(args.output_dir, 'probe_correlation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("PROBE CORRELATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            acc = all_results[model_name][dataset_name]['accuracy']
            row.append(f"{acc*100:>5.1f}%")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n{'='*80}")
    print("BEST LAYERS USED")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            layer = all_results[model_name][dataset_name]['best_layer']
            row.append(f"{layer:>8}")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ Individual predictions saved in: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()