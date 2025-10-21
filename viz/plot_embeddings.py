"""
Visualize embeddings using PCA and t-SNE
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embedding_visualization(embeddings, labels, layer_idx, output_path):
    """
    Create PCA and t-SNE visualizations of embeddings

    Args:
        embeddings: numpy array of shape (num_samples, hidden_dim)
        labels: numpy array of shape (num_samples,) with 0/1 labels
        layer_idx: Which layer these embeddings are from
        output_path: Where to save the plot
    """
    print(f"  Reducing dimensionality for layer {layer_idx}...")

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    pca_var = pca.explained_variance_ratio_

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    colors = ['red', 'blue']
    class_names = ['Insufficient', 'Sufficient']

    # PCA plot
    for label_idx, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels == label_idx
        axes[0].scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            s=20
        )
    axes[0].set_xlabel(f'PC1 ({pca_var[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca_var[1]*100:.1f}% variance)')
    axes[0].set_title(f'PCA - Layer {layer_idx}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE plot
    for label_idx, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels == label_idx
        axes[1].scatter(
            embeddings_tsne[mask, 0],
            embeddings_tsne[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            s=20
        )
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title(f't-SNE - Layer {layer_idx}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    # Test with dummy data
    print("Testing embedding visualization...")

    # Create dummy embeddings and labels
    np.random.seed(42)
    num_samples = 500
    hidden_dim = 128

    # Create two clusters
    embeddings_class_0 = np.random.randn(num_samples // 2, hidden_dim) - 1
    embeddings_class_1 = np.random.randn(num_samples // 2, hidden_dim) + 1
    embeddings = np.vstack([embeddings_class_0, embeddings_class_1])

    labels = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

    plot_embedding_visualization(embeddings, labels, 0, 'test_embedding_viz.png')
    print("Test complete! Check test_embedding_viz.png")
