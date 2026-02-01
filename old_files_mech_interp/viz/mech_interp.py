"""
Mechanistic interpretability visualizations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_activation_statistics(embeddings, labels):
    """
    Compute activation statistics (mean, std, norm) for each class and layer

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,) with 0/1 labels

    Returns:
        dict: Statistics for each layer
    """
    num_layers = embeddings.shape[1]
    stats = []

    for layer_idx in range(num_layers):
        layer_emb = embeddings[:, layer_idx, :]

        # Split by class
        insufficient_emb = layer_emb[labels == 0]
        sufficient_emb = layer_emb[labels == 1]

        # Compute statistics
        stats.append({
            'layer': layer_idx,
            'insufficient_mean_norm': float(np.linalg.norm(insufficient_emb.mean(axis=0))),
            'sufficient_mean_norm': float(np.linalg.norm(sufficient_emb.mean(axis=0))),
            'insufficient_std': float(insufficient_emb.std()),
            'sufficient_std': float(sufficient_emb.std()),
            'insufficient_avg_norm': float(np.linalg.norm(insufficient_emb, axis=1).mean()),
            'sufficient_avg_norm': float(np.linalg.norm(sufficient_emb, axis=1).mean()),
        })

    return stats


def analyze_cosine_similarity(embeddings, labels):
    """
    Compute cosine similarity between class centroids for each layer

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)

    Returns:
        list: Cosine similarities for each layer
    """
    num_layers = embeddings.shape[1]
    cosine_sims = []

    for layer_idx in range(num_layers):
        layer_emb = embeddings[:, layer_idx, :]

        # Compute centroids
        insufficient_centroid = layer_emb[labels == 0].mean(axis=0)
        sufficient_centroid = layer_emb[labels == 1].mean(axis=0)

        # Cosine similarity
        cos_sim = np.dot(insufficient_centroid, sufficient_centroid) / (
            np.linalg.norm(insufficient_centroid) * np.linalg.norm(sufficient_centroid)
        )

        cosine_sims.append({
            'layer': layer_idx,
            'cosine_similarity': float(cos_sim),
            'centroid_distance': float(np.linalg.norm(insufficient_centroid - sufficient_centroid))
        })

    return cosine_sims


def plot_activation_stats(stats, output_path):
    """
    Plot activation statistics across layers

    Args:
        stats: List of stats dicts from analyze_activation_statistics
        output_path: Where to save the plot
    """
    layers = [s['layer'] for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean norm
    axes[0, 0].plot(layers, [s['insufficient_mean_norm'] for s in stats],
                   'o-', label='Insufficient', color='red')
    axes[0, 0].plot(layers, [s['sufficient_mean_norm'] for s in stats],
                   's-', label='Sufficient', color='blue')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Norm of Mean Activation')
    axes[0, 0].set_title('Centroid Magnitude Across Layers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Standard deviation
    axes[0, 1].plot(layers, [s['insufficient_std'] for s in stats],
                   'o-', label='Insufficient', color='red')
    axes[0, 1].plot(layers, [s['sufficient_std'] for s in stats],
                   's-', label='Sufficient', color='blue')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Activation Spread Across Layers')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Average norm
    axes[1, 0].plot(layers, [s['insufficient_avg_norm'] for s in stats],
                   'o-', label='Insufficient', color='red')
    axes[1, 0].plot(layers, [s['sufficient_avg_norm'] for s in stats],
                   's-', label='Sufficient', color='blue')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Average Sample Norm')
    axes[1, 0].set_title('Average Activation Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Difference in norms
    norm_diff = [s['sufficient_avg_norm'] - s['insufficient_avg_norm'] for s in stats]
    axes[1, 1].plot(layers, norm_diff, 'o-', color='green')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Sufficient - Insufficient Norm')
    axes[1, 1].set_title('Relative Activation Strength')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Activation statistics saved to: {output_path}")


def plot_cosine_similarity_trends(cosine_sims, output_path):
    """
    Plot cosine similarity between class centroids across layers

    Args:
        cosine_sims: List of dicts from analyze_cosine_similarity
        output_path: Where to save the plot
    """
    layers = [c['layer'] for c in cosine_sims]
    cos_sim_values = [c['cosine_similarity'] for c in cosine_sims]
    distances = [c['centroid_distance'] for c in cosine_sims]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cosine similarity
    axes[0].plot(layers, cos_sim_values, 'o-', color='purple', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Cosine Similarity Between Class Centroids')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-1, 1])

    # Add interpretation text
    axes[0].text(0.02, 0.98, 'Lower = More Separable',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Euclidean distance
    axes[1].plot(layers, distances, 'o-', color='orange', linewidth=2)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Euclidean Distance')
    axes[1].set_title('Distance Between Class Centroids')
    axes[1].grid(True, alpha=0.3)

    axes[1].text(0.02, 0.98, 'Higher = More Separable',
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Cosine similarity trends saved to: {output_path}")


def plot_weight_importance(all_weights, all_metrics, output_path, top_layers=5):
    """
    Visualize which embedding dimensions are most important for linear probes

    Args:
        all_weights: List of weight arrays (input_dim,) for each layer
        all_metrics: List of metrics dicts
        output_path: Where to save the plot
        top_layers: Number of top-performing layers to show
    """
    # Get top layers by F1 score
    sorted_indices = sorted(range(len(all_metrics)),
                          key=lambda i: all_metrics[i]['test_f1'],
                          reverse=True)[:top_layers]

    fig, axes = plt.subplots(top_layers, 1, figsize=(12, 3*top_layers))
    if top_layers == 1:
        axes = [axes]

    for plot_idx, layer_idx in enumerate(sorted_indices):
        weights = all_weights[layer_idx]
        f1 = all_metrics[layer_idx]['test_f1']

        # Sort weights by absolute value
        abs_weights = np.abs(weights)
        sorted_dims = np.argsort(abs_weights)[::-1]

        # Plot top features
        num_features_to_show = min(50, len(weights))
        top_dims = sorted_dims[:num_features_to_show]

        axes[plot_idx].bar(range(num_features_to_show),
                          weights[top_dims],
                          color=['red' if w < 0 else 'blue' for w in weights[top_dims]])
        axes[plot_idx].set_xlabel('Feature Index (sorted by importance)')
        axes[plot_idx].set_ylabel('Weight Value')
        axes[plot_idx].set_title(f'Layer {layer_idx} (F1: {f1:.3f}) - Top {num_features_to_show} Features')
        axes[plot_idx].grid(True, alpha=0.3, axis='y')
        axes[plot_idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Weight importance visualization saved to: {output_path}")
