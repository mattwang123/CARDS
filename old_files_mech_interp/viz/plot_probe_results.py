"""
Visualize probe training results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_layer_performance(all_metrics, output_path):
    """
    Plot F1 score and accuracy vs layer number

    Args:
        all_metrics: List of dicts with metrics for each layer
        output_path: Where to save the plot
    """
    layers = [m['layer'] for m in all_metrics]
    train_f1 = [m['train_f1'] for m in all_metrics]
    test_f1 = [m['test_f1'] for m in all_metrics]
    train_acc = [m['train_accuracy'] for m in all_metrics]
    test_acc = [m['test_accuracy'] for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 Score plot
    axes[0].plot(layers, train_f1, 'o-', label='Train F1', color='blue', alpha=0.7)
    axes[0].plot(layers, test_f1, 's-', label='Test F1', color='red', linewidth=2)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline', alpha=0.5)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Find best layer
    best_layer = max(range(len(test_f1)), key=lambda i: test_f1[i])
    axes[0].axvline(x=best_layer, color='green', linestyle=':', alpha=0.5,
                   label=f'Best Layer: {best_layer}')

    # Accuracy plot
    axes[1].plot(layers, train_acc, 'o-', label='Train Acc', color='blue', alpha=0.7)
    axes[1].plot(layers, test_acc, 's-', label='Test Acc', color='red', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline', alpha=0.5)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Layer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nLayer performance plot saved to: {output_path}")


def plot_confusion_matrices(all_metrics, layer_indices, output_path):
    """
    Plot confusion matrices for selected layers

    Args:
        all_metrics: List of dicts with metrics for each layer
        layer_indices: List of layer indices to plot
        output_path: Where to save the plot
    """
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))

    if num_layers == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layer_indices):
        cm = np.array(all_metrics[layer_idx]['confusion_matrix'])
        f1 = all_metrics[layer_idx]['test_f1']
        acc = all_metrics[layer_idx]['test_accuracy']

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Insufficient', 'Sufficient'],
                   yticklabels=['Insufficient', 'Sufficient'],
                   ax=axes[idx], cbar=True, vmin=0, vmax=1)

        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_title(f'Layer {layer_idx}\nF1: {f1:.3f}, Acc: {acc:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrices saved to: {output_path}")


def analyze_best_layers(all_metrics, top_k=5):
    """
    Print analysis of best performing layers

    Args:
        all_metrics: List of dicts with metrics for each layer
        top_k: Number of top layers to show
    """
    print("\n" + "="*80)
    print(f"TOP {top_k} LAYERS BY F1 SCORE")
    print("="*80)

    # Sort by test F1
    sorted_metrics = sorted(all_metrics, key=lambda x: x['test_f1'], reverse=True)

    for i, metrics in enumerate(sorted_metrics[:top_k], 1):
        print(f"\n{i}. Layer {metrics['layer']}")
        print(f"   Test F1: {metrics['test_f1']:.4f}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   Test Precision: {metrics['test_precision']:.4f}")
        print(f"   Test Recall: {metrics['test_recall']:.4f}")

        # Analyze confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        tn, fp, fn, tp = cm.ravel()
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # False positive rate (insufficient predicted as sufficient)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False negative rate (sufficient predicted as insufficient)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"   False Positive Rate: {fpr:.4f} (insufficient wrongly predicted as sufficient)")
        print(f"   False Negative Rate: {fnr:.4f} (sufficient wrongly predicted as insufficient)")

    print("\n" + "="*80)


if __name__ == '__main__':
    # Test with dummy data
    print("Testing probe results visualization...")

    # Create dummy metrics
    num_layers = 28
    all_metrics = []

    for layer in range(num_layers):
        # Simulate performance that improves in middle layers
        performance = 0.5 + 0.3 * np.sin((layer - 10) / 5)
        performance = max(0.5, min(0.95, performance))

        all_metrics.append({
            'layer': layer,
            'train_f1': min(0.99, performance + 0.1),
            'test_f1': performance,
            'train_accuracy': min(0.99, performance + 0.08),
            'test_accuracy': performance - 0.02,
            'test_precision': performance,
            'test_recall': performance,
            'confusion_matrix': [
                [int(100 * performance), int(100 * (1 - performance))],
                [int(100 * (1 - performance)), int(100 * performance)]
            ]
        })

    plot_layer_performance(all_metrics, 'test_layer_performance.png')
    plot_confusion_matrices(all_metrics, [10, 15, 20], 'test_confusion_matrices.png')
    analyze_best_layers(all_metrics, top_k=3)

    print("\nTest complete! Check test_*.png files")
