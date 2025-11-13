"""
Visualize embeddings using 3D PCA
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from tqdm import tqdm


def plot_embedding_pca_3d(embeddings, labels, layer_idx, output_path):
    """
    Create 3D PCA visualization of embeddings

    Args:
        embeddings: numpy array of shape (num_samples, hidden_dim)
        labels: numpy array of shape (num_samples,) with 0/1 labels
        layer_idx: Which layer these embeddings are from
        output_path: Where to save the plot
    """
    # Apply PCA
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings)
    pca_var = pca.explained_variance_ratio_

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colors
    colors = ['red', 'blue']
    class_names = ['Insufficient', 'Sufficient']

    # Plot
    for label_idx, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels == label_idx
        ax.scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            embeddings_pca[mask, 2],
            c=color,
            label=name,
            alpha=0.6,
            s=20
        )

    ax.set_xlabel(f'PC1 ({pca_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_var[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca_var[2]*100:.1f}%)')
    ax.set_title(f'3D PCA - Layer {layer_idx}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_layers_pca(train_embeddings, train_labels, output_dir):
    """
    Generate 3D PCA plots for all layers

    Args:
        train_embeddings: numpy array of shape (num_samples, num_layers, hidden_dim)
        train_labels: numpy array of shape (num_samples,)
        output_dir: Directory to save plots
    """
    num_layers = train_embeddings.shape[1]

    print(f"Generating 3D PCA for {num_layers} layers...")

    for layer_idx in tqdm(range(num_layers), desc="PCA visualization"):
        output_path = f"{output_dir}/layer_{layer_idx:02d}_pca3d.png"
        plot_embedding_pca_3d(
            train_embeddings[:, layer_idx, :],
            train_labels,
            layer_idx,
            output_path
        )

    print(f"Saved {num_layers} PCA plots to {output_dir}")
