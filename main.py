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
from viz.plot_embeddings import plot_embedding_visualization
from viz.plot_probe_results import plot_layer_performance, plot_confusion_matrices


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


def train_probe_for_layer(layer_idx, train_embeddings, train_labels, test_embeddings, test_labels,
                          hidden_dim=128, num_epochs=50, lr=0.001, device='cpu'):
    """
    Train a probe for a specific layer

    Args:
        layer_idx: Which layer to train on
        train_embeddings: Training embeddings for this layer (num_samples, hidden_dim)
        train_labels: Training labels (num_samples,)
        test_embeddings: Test embeddings for this layer
        test_labels: Test labels
        hidden_dim: Hidden dimension for MLP probe
        num_epochs: Number of training epochs
        lr: Learning rate
        device: cpu or cuda

    Returns:
        dict: Metrics (accuracy, f1, precision, recall, confusion_matrix)
    """
    input_dim = train_embeddings.shape[1]

    # Create probe
    probe = MLPProbe(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    # Convert to tensors
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
        # Train set
        train_preds = probe.predict(X_train).cpu().numpy()
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # Test set
        test_preds = probe.predict(X_test).cpu().numpy()
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds)
        test_cm = confusion_matrix(test_labels, test_preds)

    metrics = {
        'layer': layer_idx,
        'train_accuracy': float(train_acc),
        'train_f1': float(train_f1),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'confusion_matrix': test_cm.tolist()
    }

    return probe, metrics


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
    parser.add_argument('--visualize_embeddings', action='store_true',
                        help='Create embedding visualizations before training')
    parser.add_argument('--layers_to_visualize', type=str, default='0,13,27',
                        help='Comma-separated layer indices to visualize (default: 0,13,27)')

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

    # Visualize embeddings (optional)
    if args.visualize_embeddings:
        print("\n" + "="*80)
        print("VISUALIZING EMBEDDINGS")
        print("="*80)

        layers_to_viz = [int(x) for x in args.layers_to_visualize.split(',')]
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        for layer_idx in layers_to_viz:
            if layer_idx < num_layers:
                print(f"\nVisualizing layer {layer_idx}...")
                plot_embedding_visualization(
                    train_embeddings[:, layer_idx, :],
                    train_labels,
                    layer_idx,
                    os.path.join(viz_dir, f'layer_{layer_idx}_embeddings.png')
                )

    # Train probes for all layers
    print("\n" + "="*80)
    print("TRAINING PROBES")
    print("="*80)

    all_metrics = []

    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        # Extract embeddings for this layer
        train_layer_emb = train_embeddings[:, layer_idx, :]
        test_layer_emb = test_embeddings[:, layer_idx, :]

        # Train probe
        probe, metrics = train_probe_for_layer(
            layer_idx,
            train_layer_emb,
            train_labels,
            test_layer_emb,
            test_labels,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device
        )

        all_metrics.append(metrics)

        # Save probe
        probe_path = os.path.join(args.output_dir, f'layer_{layer_idx}_probe.pt')
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
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Layer performance plot
    plot_layer_performance(
        all_metrics,
        os.path.join(viz_dir, 'layer_performance.png')
    )

    # Confusion matrices for best layers
    plot_confusion_matrices(
        all_metrics,
        [best_f1_idx, best_acc_idx],
        os.path.join(viz_dir, 'confusion_matrices.png')
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Visualizations saved to: {viz_dir}")


if __name__ == '__main__':
    main()
