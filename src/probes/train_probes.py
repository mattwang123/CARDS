"""
Train linear and MLP probes on frozen model embeddings

This script trains probes for binary classification: sufficient vs insufficient
Supports both logistic regression (linear) and MLP probes
"""
import argparse
import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm


class MLPProbe(nn.Module):
    """Two-layer MLP probe for binary classification"""

    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        """Get predicted class labels"""
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1)


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


def train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels, C=1.0):
    """
    Train logistic regression probe

    Args:
        train_embeddings: Training embeddings (N, D)
        train_labels: Training labels (N,)
        test_embeddings: Test embeddings (M, D)
        test_labels: Test labels (M,)
        C: Regularization strength (default: 1.0)

    Returns:
        probe: Trained LogisticRegression model
        metrics: Dict of performance metrics
        weights: Probe weight vector
    """
    probe = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='lbfgs')
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
    """
    Train MLP probe

    Args:
        train_embeddings: Training embeddings (N, D)
        train_labels: Training labels (N,)
        test_embeddings: Test embeddings (M, D)
        test_labels: Test labels (M,)
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to use ('cpu' or 'cuda')

    Returns:
        probe: Trained MLPProbe model
        metrics: Dict of performance metrics
    """
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

    return probe, metrics


def train_probe_for_layer(layer_idx, train_embeddings, train_labels, test_embeddings, test_labels,
                          probe_type='linear', hidden_dim=128, num_epochs=50, lr=0.001, device='cpu', C=1.0):
    """
    Train a probe for a specific layer

    Args:
        layer_idx: Which layer to train on
        train_embeddings: Training embeddings for this layer (N, D)
        train_labels: Training labels (N,)
        test_embeddings: Test embeddings for this layer (M, D)
        test_labels: Test labels (M,)
        probe_type: 'linear' or 'mlp'
        hidden_dim: Hidden dimension for MLP probe
        num_epochs: Number of training epochs for MLP
        lr: Learning rate for MLP
        device: Device to use
        C: Regularization strength for linear probe

    Returns:
        probe: Trained probe
        metrics: Dict of performance metrics
        weights: Linear probe weights (None for MLP)
    """
    input_dim = train_embeddings.shape[1]

    if probe_type == 'linear':
        probe, metrics, weights = train_linear_probe(
            train_embeddings, train_labels, test_embeddings, test_labels, C=C
        )
    else:  # mlp
        probe, metrics = train_mlp_probe(
            train_embeddings, train_labels, test_embeddings, test_labels,
            input_dim, hidden_dim, num_epochs, lr, device
        )
        weights = None

    metrics['layer'] = layer_idx
    metrics['probe_type'] = probe_type
    return probe, metrics, weights


def main():
    parser = argparse.ArgumentParser(description='Train probes on frozen embeddings')
    parser.add_argument('--train_embeddings', type=str, required=True,
                        help='Path to train embeddings .npy file')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Path to train labels JSON file')
    parser.add_argument('--test_embeddings', type=str, required=True,
                        help='Path to test embeddings .npy file')
    parser.add_argument('--test_labels', type=str, required=True,
                        help='Path to test labels JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save probe results')
    parser.add_argument('--probe_type', type=str, default='linear',
                        choices=['linear', 'mlp', 'both'],
                        help='Probe type: linear, mlp, or both (default: linear)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for MLP probe (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs for MLP (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for MLP (default: 0.001)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization strength for linear probe (default: 1.0)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--layers', type=str, default='all',
                        help='Layers to train on: "all" or comma-separated list like "0,5,10" (default: all)')

    args = parser.parse_args()

    print("="*80)
    print("PROBE TRAINING PIPELINE")
    print("="*80)
    print(f"Probe type: {args.probe_type}")
    print(f"Output directory: {args.output_dir}")

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

    # Determine which layers to train on
    if args.layers == 'all':
        layers_to_train = list(range(num_layers))
    else:
        layers_to_train = [int(x.strip()) for x in args.layers.split(',')]

    print(f"\nTraining on layers: {layers_to_train}")

    # Determine probe types to train
    if args.probe_type == 'both':
        probe_types = ['linear', 'mlp']
    else:
        probe_types = [args.probe_type]

    # Train probes
    for probe_type in probe_types:
        print("\n" + "="*80)
        print(f"TRAINING {probe_type.upper()} PROBES")
        print("="*80)

        all_metrics = []
        all_weights = [] if probe_type == 'linear' else None

        results_dir = os.path.join(args.output_dir, f'probes_{probe_type}')
        os.makedirs(results_dir, exist_ok=True)

        for layer_idx in tqdm(layers_to_train, desc=f"Training {probe_type} probes"):
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
                probe_type=probe_type,
                hidden_dim=args.hidden_dim,
                num_epochs=args.num_epochs,
                lr=args.lr,
                device=args.device,
                C=args.C
            )

            all_metrics.append(metrics)
            if weights is not None:
                all_weights.append(weights)

            # Save probe
            probe_path = os.path.join(results_dir, f'layer_{layer_idx}_probe')
            if probe_type == 'linear':
                with open(probe_path + '.pkl', 'wb') as f:
                    pickle.dump(probe, f)
            else:
                torch.save(probe.state_dict(), probe_path + '.pt')

        # Save all metrics
        metrics_path = os.path.join(results_dir, 'all_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\nSaved {probe_type} metrics to: {metrics_path}")

        # Print summary
        best_f1_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_f1'])

        print("\n" + "="*80)
        print(f"{probe_type.upper()} RESULTS SUMMARY")
        print("="*80)
        print(f"\nBest F1 Score: Layer {all_metrics[best_f1_idx]['layer']}")
        print(f"  F1: {all_metrics[best_f1_idx]['test_f1']:.4f}")
        print(f"  Accuracy: {all_metrics[best_f1_idx]['test_accuracy']:.4f}")
        print(f"  Precision: {all_metrics[best_f1_idx]['test_precision']:.4f}")
        print(f"  Recall: {all_metrics[best_f1_idx]['test_recall']:.4f}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
