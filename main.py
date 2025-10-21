"""
Main script to train linear probes on hidden states for sufficiency classification
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.probe import MLPProbe
from viz.plot_embeddings import plot_embedding_pca_3d, plot_all_layers_pca
from viz.plot_probe_results import plot_layer_performance, plot_confusion_matrices
from viz.mech_interp import (
    analyze_activation_statistics,
    analyze_cosine_similarity,
    plot_weight_importance,
    plot_activation_stats,
    plot_cosine_similarity_trends
)
from sklearn.linear_model import LogisticRegression


def load_embeddings_and_labels(embeddings_path, labels_path):
    """
    Load embeddings and extract labels from JSON

    Args:
        embeddings_path: Path to .npy file with shape (num_samples, num_layers, hidden_dim)
        labels_path: Path to JSON file with 'is_sufficient' field

    Returns:
        embeddings: numpy array
        labels: numpy array of 0/1 (0=insufficient, 1=sufficient)
    """
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading labels from {labels_path}")
    with open(labels_path, 'r') as f:
        data = json.load(f)

    # Extract is_sufficient field (True=1, False=0)
    labels = np.array([1 if item['is_sufficient'] else 0 for item in data])
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sufficient: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Insufficient: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return embeddings, labels


def train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels):
    """Train logistic regression probe"""
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(train_embeddings, train_labels)

    # Predictions
    train_preds = probe.predict(train_embeddings)
    test_preds = probe.predict(test_embeddings)

    # Metrics
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_cm = confusion_matrix(test_labels, test_preds)

    metrics = {
        'train_accuracy': float(train_acc),
        'train_f1': float(train_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'confusion_matrix': test_cm.tolist()
    }

    # Extract weights for visualization
    weights = probe.coef_[0]  # Shape: (input_dim,)

    return probe, metrics, weights


def train_mlp_probe(train_embeddings, train_labels, test_embeddings, test_labels,
                   input_dim, hidden_dim=128, num_epochs=50, lr=0.001, device='cpu'):
    """Train MLP probe"""
    probe = MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    X_train = torch.FloatTensor(train_embeddings).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_test = torch.FloatTensor(test_embeddings).to(device)
    y_test = torch.LongTensor(test_labels).to(device)

    # Training loop
    probe.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    probe.eval()
    with torch.no_grad():
        train_preds = probe.predict(X_train).cpu().numpy()
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        test_preds = probe.predict(X_test).cpu().numpy()
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds)
        test_cm = confusion_matrix(test_labels, test_preds)

    metrics = {
        'train_accuracy': float(train_acc),
        'train_f1': float(train_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'confusion_matrix': test_cm.tolist()
    }

    return probe, metrics, None


def train_probe_for_layer(layer_idx, train_embeddings, train_labels, test_embeddings, test_labels,
                          probe_type='mlp', hidden_dim=128, num_epochs=50, lr=0.001, device='cpu'):
    """
    Train a probe for a specific layer

    Args:
        layer_idx: Which layer to train on
        train_embeddings: Training embeddings for this layer (num_samples, hidden_dim)
        train_labels: Training labels (num_samples,)
        test_embeddings: Test embeddings for this layer
        test_labels: Test labels
        probe_type: 'linear' or 'mlp'
        hidden_dim: Hidden dimension for MLP probe
        num_epochs: Number of training epochs
        lr: Learning rate
        device: cpu or cuda

    Returns:
        probe: Trained probe
        dict: Metrics
        weights: Linear probe weights (None for MLP)
    """
    input_dim = train_embeddings.shape[1]

    if probe_type == 'linear':
        probe, metrics, weights = train_linear_probe(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
    else:  # mlp
        probe, metrics, weights = train_mlp_probe(
            train_embeddings, train_labels, test_embeddings, test_labels,
            input_dim, hidden_dim, num_epochs, lr, device
        )

    metrics['layer'] = layer_idx
    return probe, metrics, weights


def main():
    parser = argparse.ArgumentParser(description='Train linear probes on hidden states')
    parser.add_argument('--train_embeddings', type=str, required=True,
                        help='Path to train embeddings .npy file')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Path to train labels JSON file')
    parser.add_argument('--test_embeddings', type=str, required=True,
                        help='Path to test embeddings .npy file')
    parser.add_argument('--test_labels', type=str, required=True,
                        help='Path to test labels JSON file')
    parser.add_argument('--output_dir', type=str, default='experiments/probe_results',
                        help='Directory to save results')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for MLP probe (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--probe_type', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='Probe type: linear (logistic regression) or mlp (default: linear)')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Create visualizations for ALL layers (3D PCA, etc.)')
    parser.add_argument('--skip_pca', action='store_true',
                        help='Skip PCA visualization (faster)')

    args = parser.parse_args()

    print("="*80)
    print("LINEAR PROBE TRAINING PIPELINE")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_embeddings, train_labels = load_embeddings_and_labels(
        args.train_embeddings, args.train_labels
    )
    test_embeddings, test_labels = load_embeddings_and_labels(
        args.test_embeddings, args.test_labels
    )

    num_samples, num_layers, hidden_size = train_embeddings.shape

    print(f"\nProbe type: {args.probe_type.upper()}")

    # Create subfolder structure
    pca_dir = os.path.join(args.output_dir, 'pca_3d')
    mech_interp_dir = os.path.join(args.output_dir, 'mech_interp')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(pca_dir, exist_ok=True)
    os.makedirs(mech_interp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Mechanistic interpretability: Activation statistics and cosine similarity
    print("\n" + "="*80)
    print("MECHANISTIC INTERPRETABILITY ANALYSIS")
    print("="*80)

    activation_stats = analyze_activation_statistics(train_embeddings, train_labels)
    cosine_sims = analyze_cosine_similarity(train_embeddings, train_labels)

    plot_activation_stats(
        activation_stats,
        os.path.join(mech_interp_dir, 'activation_statistics.png')
    )

    plot_cosine_similarity_trends(
        cosine_sims,
        os.path.join(mech_interp_dir, 'cosine_similarity.png')
    )

    # 3D PCA visualization for all layers (optional, can be slow)
    if args.visualize_all and not args.skip_pca:
        print("\n" + "="*80)
        print("GENERATING 3D PCA FOR ALL LAYERS")
        print("="*80)

        plot_all_layers_pca(
            train_embeddings,
            train_labels,
            pca_dir
        )

    # Train probes for all layers
    print("\n" + "="*80)
    print(f"TRAINING {args.probe_type.upper()} PROBES FOR ALL LAYERS")
    print("="*80)

    all_metrics = []
    all_weights = []  # For linear probes

    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        # Extract embeddings for this layer
        train_layer_emb = train_embeddings[:, layer_idx, :]
        test_layer_emb = test_embeddings[:, layer_idx, :]

        # Train probe
        probe, metrics, weights = train_probe_for_layer(
            layer_idx,
            train_layer_emb,
            train_labels,
            test_layer_emb,
            test_labels,
            probe_type=args.probe_type,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device
        )

        all_metrics.append(metrics)
        if weights is not None:
            all_weights.append(weights)

        # Save probe
        probe_path = os.path.join(results_dir, f'layer_{layer_idx}_probe.pt')
        if args.probe_type == 'linear':
            # Save sklearn model with pickle
            import pickle
            with open(probe_path.replace('.pt', '.pkl'), 'wb') as f:
                pickle.dump(probe, f)
        else:
            torch.save(probe.state_dict(), probe_path)

    # Save all metrics
    metrics_path = os.path.join(args.output_dir, 'all_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    best_f1_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_f1'])
    best_acc_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_accuracy'])

    print(f"\nBest F1 Score: Layer {best_f1_idx}")
    print(f"  F1: {all_metrics[best_f1_idx]['test_f1']:.4f}")
    print(f"  Accuracy: {all_metrics[best_f1_idx]['test_accuracy']:.4f}")
    print(f"  Precision: {all_metrics[best_f1_idx]['test_precision']:.4f}")
    print(f"  Recall: {all_metrics[best_f1_idx]['test_recall']:.4f}")

    print(f"\nBest Accuracy: Layer {best_acc_idx}")
    print(f"  Accuracy: {all_metrics[best_acc_idx]['test_accuracy']:.4f}")
    print(f"  F1: {all_metrics[best_acc_idx]['test_f1']:.4f}")

    # Plot results
    print("\n" + "="*80)
    print("GENERATING RESULT VISUALIZATIONS")
    print("="*80)

    # Layer performance plot
    plot_layer_performance(
        all_metrics,
        os.path.join(results_dir, 'layer_performance.png')
    )

    # Confusion matrices for best layers
    top_layers = sorted(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_f1'], reverse=True)[:3]
    plot_confusion_matrices(
        all_metrics,
        top_layers,
        os.path.join(results_dir, 'confusion_matrices_top3.png')
    )

    # Weight importance visualization (for linear probes only)
    if args.probe_type == 'linear' and len(all_weights) > 0:
        print("\nGenerating weight importance visualizations...")
        plot_weight_importance(
            all_weights,
            all_metrics,
            os.path.join(mech_interp_dir, 'weight_importance.png')
        )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Probe models: {results_dir}")
    print(f"  - 3D PCA plots: {pca_dir}")
    print(f"  - Mech interp: {mech_interp_dir}")
    print(f"  - Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
