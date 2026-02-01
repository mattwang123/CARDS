"""
Advanced mechanistic interpretability analyses for embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from tqdm import tqdm
import warnings
import json
import os
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Transformers not available for token attribution")


def selectivity_analysis(embeddings, labels, output_path, top_k=20):
    """
    Find which dimensions are most selective for sufficiency

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)
        output_path: Where to save plots
        top_k: Number of top selective dimensions to show
    """
    num_layers = embeddings.shape[1]
    hidden_dim = embeddings.shape[2]

    print(f"\n{'='*80}")
    print("SELECTIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Finding dimensions most correlated with sufficiency...")

    # Analyze middle layers (most likely to have structure)
    layers_to_analyze = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    fig, axes = plt.subplots(len(layers_to_analyze), 2, figsize=(16, 5*len(layers_to_analyze)))
    if len(layers_to_analyze) == 1:
        axes = axes.reshape(1, -1)

    for plot_idx, layer_idx in enumerate(layers_to_analyze):
        layer_emb = embeddings[:, layer_idx, :]

        # Compute correlation for each dimension
        correlations = np.array([pearsonr(layer_emb[:, dim], labels)[0]
                                for dim in range(hidden_dim)])

        # Get top dimensions by absolute correlation
        top_dims = np.argsort(np.abs(correlations))[-top_k:][::-1]

        # Plot 1: Correlation values
        axes[plot_idx, 0].barh(range(top_k), correlations[top_dims],
                               color=['red' if c < 0 else 'blue' for c in correlations[top_dims]])
        axes[plot_idx, 0].set_xlabel('Correlation with Sufficiency')
        axes[plot_idx, 0].set_ylabel('Dimension Index')
        axes[plot_idx, 0].set_yticks(range(top_k))
        axes[plot_idx, 0].set_yticklabels([f'Dim {d}' for d in top_dims])
        axes[plot_idx, 0].set_title(f'Layer {layer_idx}: Top {top_k} Selective Dimensions')
        axes[plot_idx, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[plot_idx, 0].grid(True, alpha=0.3, axis='x')

        # Plot 2: Distribution of most selective dimension
        most_selective_dim = top_dims[0]
        dim_values = layer_emb[:, most_selective_dim]

        axes[plot_idx, 1].hist(dim_values[labels == 0], bins=50, alpha=0.6,
                               color='red', label='Insufficient', density=True)
        axes[plot_idx, 1].hist(dim_values[labels == 1], bins=50, alpha=0.6,
                               color='blue', label='Sufficient', density=True)
        axes[plot_idx, 1].set_xlabel('Activation Value')
        axes[plot_idx, 1].set_ylabel('Density')
        axes[plot_idx, 1].set_title(f'Layer {layer_idx}, Dim {most_selective_dim} '
                                    f'(r={correlations[most_selective_dim]:.3f})')
        axes[plot_idx, 1].legend()
        axes[plot_idx, 1].grid(True, alpha=0.3)

        print(f"  Layer {layer_idx}: Most selective dim {most_selective_dim}, "
              f"correlation = {correlations[most_selective_dim]:.3f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved selectivity analysis to: {output_path}")


def control_task_probing(train_embeddings, train_labels, test_embeddings, test_labels, output_path):
    """
    Validate probe results with control tasks (random labels, shuffled labels)

    Args:
        train_embeddings: (num_samples, num_layers, hidden_dim)
        train_labels: (num_samples,)
        test_embeddings: (num_samples, num_layers, hidden_dim)
        test_labels: (num_samples,)
        output_path: Where to save plot
    """
    print(f"\n{'='*80}")
    print("CONTROL TASK PROBING")
    print(f"{'='*80}")
    print("Validating probe results aren't spurious...")

    num_layers = train_embeddings.shape[1]

    real_f1_scores = []
    random_f1_scores = []
    shuffled_f1_scores = []

    # Create random and shuffled labels
    np.random.seed(42)
    random_labels = np.random.randint(0, 2, size=len(train_labels))
    shuffled_labels = train_labels.copy()
    np.random.shuffle(shuffled_labels)

    for layer_idx in tqdm(range(num_layers), desc="Control probing"):
        train_layer = train_embeddings[:, layer_idx, :]
        test_layer = test_embeddings[:, layer_idx, :]

        # Real labels
        probe_real = LogisticRegression(max_iter=1000, random_state=42)
        probe_real.fit(train_layer, train_labels)
        pred_real = probe_real.predict(test_layer)
        real_f1_scores.append(f1_score(test_labels, pred_real))

        # Random labels
        probe_random = LogisticRegression(max_iter=1000, random_state=42)
        probe_random.fit(train_layer, random_labels)
        pred_random = probe_random.predict(test_layer)
        random_f1_scores.append(f1_score(test_labels, pred_random))

        # Shuffled labels
        probe_shuffled = LogisticRegression(max_iter=1000, random_state=42)
        probe_shuffled.fit(train_layer, shuffled_labels)
        pred_shuffled = probe_shuffled.predict(test_layer)
        shuffled_f1_scores.append(f1_score(test_labels, pred_shuffled))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = range(num_layers)

    ax.plot(layers, real_f1_scores, 'o-', label='Real Labels', color='green', linewidth=2)
    ax.plot(layers, random_f1_scores, 's-', label='Random Labels', color='gray', linewidth=2, alpha=0.7)
    ax.plot(layers, shuffled_f1_scores, '^-', label='Shuffled Labels', color='orange', linewidth=2, alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')

    ax.set_xlabel('Layer')
    ax.set_ylabel('F1 Score')
    ax.set_title('Control Task Validation: Real vs Random/Shuffled Labels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    max_real_f1 = max(real_f1_scores)
    max_random_f1 = max(random_f1_scores)
    ax.text(0.02, 0.98, f'Max Real F1: {max_real_f1:.3f}\nMax Random F1: {max_random_f1:.3f}\n'
                        f'Gap: {max_real_f1 - max_random_f1:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Real labels max F1: {max_real_f1:.3f}")
    print(f"  Random labels max F1: {max_random_f1:.3f}")
    print(f"  Gap: {max_real_f1 - max_random_f1:.3f} (larger is better)")
    print(f"Saved control task analysis to: {output_path}")


def subspace_dimensionality_analysis(train_embeddings, train_labels,
                                     test_embeddings, test_labels, output_path):
    """
    Determine minimum dimensions needed for good performance

    Args:
        train_embeddings: (num_samples, num_layers, hidden_dim)
        train_labels: (num_samples,)
        test_embeddings: (num_samples, num_layers, hidden_dim)
        test_labels: (num_samples,)
        output_path: Where to save plot
    """
    print(f"\n{'='*80}")
    print("SUBSPACE DIMENSIONALITY ANALYSIS")
    print(f"{'='*80}")
    print("Finding minimal sufficiency subspace...")

    num_layers = train_embeddings.shape[1]
    dim_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, train_embeddings.shape[2]]

    # Analyze a few key layers
    layers_to_analyze = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    fig, ax = plt.subplots(figsize=(12, 6))

    for layer_idx in layers_to_analyze:
        train_layer = train_embeddings[:, layer_idx, :]
        test_layer = test_embeddings[:, layer_idx, :]

        f1_scores = []

        for k in dim_values:
            if k > train_layer.shape[1]:
                k = train_layer.shape[1]

            # Use PCA to select top k dimensions
            pca = PCA(n_components=k)
            train_pca = pca.fit_transform(train_layer)
            test_pca = pca.transform(test_layer)

            # Train probe
            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(train_pca, train_labels)
            pred = probe.predict(test_pca)
            f1_scores.append(f1_score(test_labels, pred))

        ax.plot(dim_values[:len(f1_scores)], f1_scores, 'o-',
               label=f'Layer {layer_idx}', linewidth=2, markersize=8)

        print(f"  Layer {layer_idx}: 10 dims → F1={f1_scores[dim_values.index(10)]:.3f}, "
              f"Full → F1={f1_scores[-1]:.3f}")

    ax.set_xlabel('Number of Dimensions (PCA)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance vs Subspace Dimensionality')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved subspace dimensionality analysis to: {output_path}")


def nearest_neighbor_analysis(train_embeddings, train_labels,
                              test_embeddings, test_labels, output_path, k=5):
    """
    Analyze local structure via k-nearest neighbors

    Args:
        train_embeddings: (num_samples, num_layers, hidden_dim)
        train_labels: (num_samples,)
        test_embeddings: (num_samples, num_layers, hidden_dim)
        test_labels: (num_samples,)
        output_path: Where to save plot
        k: Number of neighbors
    """
    print(f"\n{'='*80}")
    print("NEAREST NEIGHBOR ANALYSIS")
    print(f"{'='*80}")
    print(f"Analyzing {k}-NN label consistency...")

    num_layers = train_embeddings.shape[1]
    nn_accuracies = []

    for layer_idx in tqdm(range(num_layers), desc="k-NN analysis"):
        train_layer = train_embeddings[:, layer_idx, :]
        test_layer = test_embeddings[:, layer_idx, :]

        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(train_layer)

        # Find neighbors for test samples
        distances, indices = knn.kneighbors(test_layer)

        # Predict by majority vote
        neighbor_labels = train_labels[indices]
        predictions = (neighbor_labels.mean(axis=1) > 0.5).astype(int)

        accuracy = accuracy_score(test_labels, predictions)
        nn_accuracies.append(accuracy)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(num_layers), nn_accuracies, 'o-', color='purple', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel(f'{k}-NN Accuracy')
    ax.set_title(f'{k}-Nearest Neighbor Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_layer = np.argmax(nn_accuracies)
    ax.text(0.02, 0.98, f'Best Layer: {best_layer}\nAccuracy: {nn_accuracies[best_layer]:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Best k-NN layer: {best_layer}, accuracy: {nn_accuracies[best_layer]:.3f}")
    print(f"Saved k-NN analysis to: {output_path}")


def centered_kernel_alignment(embeddings, labels, output_path):
    """
    Compute CKA similarity between all layer pairs

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)
        output_path: Where to save plot
    """
    print(f"\n{'='*80}")
    print("CENTERED KERNEL ALIGNMENT (CKA)")
    print(f"{'='*80}")
    print("Computing layer similarity matrix...")

    def linear_cka(X, Y):
        """Compute linear CKA between two matrices"""
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        hsic = np.trace(X.T @ Y @ Y.T @ X)
        normX = np.linalg.norm(X.T @ X, 'fro')
        normY = np.linalg.norm(Y.T @ Y, 'fro')

        return hsic / (normX * normY + 1e-10)

    num_layers = embeddings.shape[1]
    cka_matrix = np.zeros((num_layers, num_layers))

    for i in tqdm(range(num_layers), desc="Computing CKA"):
        for j in range(num_layers):
            cka_matrix[i, j] = linear_cka(embeddings[:, i, :], embeddings[:, j, :])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_title('CKA Similarity Between Layers')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CKA Similarity')

    # Add gridlines
    ax.set_xticks(range(0, num_layers, max(1, num_layers // 10)))
    ax.set_yticks(range(0, num_layers, max(1, num_layers // 10)))
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CKA analysis to: {output_path}")


def lda_visualization(embeddings, labels, output_path):
    """
    LDA-based supervised dimensionality reduction (better than PCA for classification)

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)
        output_path: Where to save plot
    """
    print(f"\n{'='*80}")
    print("LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print(f"{'='*80}")
    print("Computing supervised projections...")

    num_layers = embeddings.shape[1]
    layers_to_plot = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(6*len(layers_to_plot), 5))
    if len(layers_to_plot) == 1:
        axes = [axes]

    for plot_idx, layer_idx in enumerate(layers_to_plot):
        layer_emb = embeddings[:, layer_idx, :]

        # Apply LDA (max 1 component for binary classification)
        lda = LinearDiscriminantAnalysis(n_components=1)
        embeddings_lda = lda.fit_transform(layer_emb, labels)

        # Plot histograms
        axes[plot_idx].hist(embeddings_lda[labels == 0], bins=50, alpha=0.6,
                           color='red', label='Insufficient', density=True)
        axes[plot_idx].hist(embeddings_lda[labels == 1], bins=50, alpha=0.6,
                           color='blue', label='Sufficient', density=True)
        axes[plot_idx].set_xlabel('LDA Component 1')
        axes[plot_idx].set_ylabel('Density')
        axes[plot_idx].set_title(f'Layer {layer_idx} - LDA Projection')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

        # Compute separation
        mean_0 = embeddings_lda[labels == 0].mean()
        mean_1 = embeddings_lda[labels == 1].mean()
        std_pooled = np.sqrt((embeddings_lda[labels == 0].var() +
                             embeddings_lda[labels == 1].var()) / 2)
        cohen_d = abs(mean_1 - mean_0) / std_pooled

        axes[plot_idx].text(0.02, 0.98, f"Cohen's d: {cohen_d:.2f}",
                           transform=axes[plot_idx].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        print(f"  Layer {layer_idx}: Cohen's d = {cohen_d:.2f} (>0.8 is good separation)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved LDA visualization to: {output_path}")


def umap_visualization(embeddings, labels, output_path):
    """
    UMAP-based non-linear dimensionality reduction

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)
        output_path: Where to save plot
    """
    if not UMAP_AVAILABLE:
        print("\nSkipping UMAP (not installed)")
        return

    print(f"\n{'='*80}")
    print("UMAP VISUALIZATION")
    print(f"{'='*80}")
    print("Computing non-linear embeddings...")

    num_layers = embeddings.shape[1]
    layers_to_plot = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(6*len(layers_to_plot), 5))
    if len(layers_to_plot) == 1:
        axes = [axes]

    for plot_idx, layer_idx in enumerate(layers_to_plot):
        layer_emb = embeddings[:, layer_idx, :]

        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_umap = reducer.fit_transform(layer_emb)

        # Plot
        for label_val, color, name in [(0, 'red', 'Insufficient'), (1, 'blue', 'Sufficient')]:
            mask = labels == label_val
            axes[plot_idx].scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                                  c=color, label=name, alpha=0.6, s=20)

        axes[plot_idx].set_xlabel('UMAP 1')
        axes[plot_idx].set_ylabel('UMAP 2')
        axes[plot_idx].set_title(f'Layer {layer_idx} - UMAP')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP visualization to: {output_path}")


def intrinsic_dimensionality(embeddings, labels, output_path, k=10):
    """
    Estimate intrinsic dimensionality using MLE method

    Args:
        embeddings: (num_samples, num_layers, hidden_dim)
        labels: (num_samples,)
        output_path: Where to save plot
        k: Number of nearest neighbors for estimation
    """
    print(f"\n{'='*80}")
    print("INTRINSIC DIMENSIONALITY ESTIMATION")
    print(f"{'='*80}")
    print("Estimating local dimensionality...")

    def mle_id(X, k=10):
        """Maximum Likelihood Estimation of intrinsic dimension"""
        n = X.shape[0]
        # Compute pairwise distances
        dists = cdist(X, X, metric='euclidean')

        # For each point, get k+1 nearest neighbors (including itself)
        nearest_dists = np.partition(dists, k+1, axis=1)[:, 1:k+2]

        # MLE formula
        r_k = nearest_dists[:, -1]
        r_i = nearest_dists[:, :-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratios = np.log(r_k[:, None] / r_i)
            log_ratios = log_ratios[np.isfinite(log_ratios)]

        if len(log_ratios) == 0:
            return 0

        return k / np.mean(log_ratios)

    num_layers = embeddings.shape[1]

    overall_dims = []
    insufficient_dims = []
    sufficient_dims = []

    for layer_idx in tqdm(range(num_layers), desc="ID estimation"):
        layer_emb = embeddings[:, layer_idx, :]

        # Subsample for speed (ID estimation is slow)
        max_samples = min(500, len(layer_emb))
        indices = np.random.choice(len(layer_emb), max_samples, replace=False)
        layer_sample = layer_emb[indices]
        labels_sample = labels[indices]

        overall_dims.append(mle_id(layer_sample, k))
        insufficient_dims.append(mle_id(layer_sample[labels_sample == 0], k))
        sufficient_dims.append(mle_id(layer_sample[labels_sample == 1], k))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = range(num_layers)

    ax.plot(layers, overall_dims, 'o-', label='Overall', color='purple', linewidth=2)
    ax.plot(layers, insufficient_dims, 's-', label='Insufficient', color='red', linewidth=2, alpha=0.7)
    ax.plot(layers, sufficient_dims, '^-', label='Sufficient', color='blue', linewidth=2, alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Estimated Intrinsic Dimension')
    ax.set_title('Intrinsic Dimensionality Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved intrinsic dimensionality analysis to: {output_path}")


def token_attribution_analysis(data_path, probe_path, model_name, layer_idx,
                                output_path, num_examples=6, device='cpu'):
    """
    Attribute probe decisions to individual tokens in the input

    Args:
        data_path: Path to JSON data file
        probe_path: Path to trained probe (.pkl file)
        model_name: Model identifier (e.g., 'qwen2.5-math-1.5b')
        layer_idx: Which layer the probe was trained on
        output_path: Where to save visualization
        num_examples: Number of examples to visualize
        device: cpu or cuda
    """
    if not TORCH_AVAILABLE:
        print("\nSkipping token attribution (PyTorch/Transformers not available)")
        return

    print(f"\n{'='*80}")
    print("TOKEN ATTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Analyzing which tokens trigger sufficiency detectors...")

    # Import model config
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.config import get_model_config

    # Load probe
    import pickle
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)

    probe_weights = probe.coef_[0]  # Shape: (hidden_dim,)
    print(f"  Loaded probe from layer {layer_idx}, weights shape: {probe_weights.shape}")

    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Sample examples (3 sufficient, 3 insufficient)
    sufficient_examples = [item for item in data if item['is_sufficient']][:num_examples//2]
    insufficient_examples = [item for item in data if not item['is_sufficient']][:num_examples//2]
    examples = sufficient_examples + insufficient_examples

    # Load model and tokenizer
    model_config = get_model_config(model_name)
    model_path = model_config['name']  # HuggingFace model path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_path,
                                       output_hidden_states=True,
                                       trust_remote_code=True).to(device)
    model.eval()

    print(f"  Loaded model: {model_name}")
    print(f"  Analyzing {len(examples)} examples...")

    # Create figure - one plot per example
    for idx, example in enumerate(tqdm(examples, desc="Token attribution")):
        question = example['question']
        label = example['is_sufficient']

        # Tokenize
        inputs = tokenizer(question, return_tensors='pt', truncation=True, max_length=512).to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
            layer_hidden = hidden_states[layer_idx][0]  # Shape: (seq_len, hidden_dim)

        # Project each token onto probe direction
        layer_hidden_np = layer_hidden.cpu().numpy()
        token_scores = layer_hidden_np @ probe_weights  # Shape: (seq_len,)

        # Get top tokens by absolute score
        top_k = min(15, len(tokens))  # Show top 15 most important tokens
        abs_scores = np.abs(token_scores)
        top_indices = np.argsort(abs_scores)[-top_k:][::-1]

        # Create plot for this example
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot top tokens
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = token_scores[top_indices]

        # Normalize for visualization
        max_abs = np.abs(top_scores).max()
        if max_abs > 0:
            top_scores_norm = top_scores / max_abs
        else:
            top_scores_norm = top_scores

        colors = ['red' if s < 0 else 'blue' for s in top_scores_norm]
        y_pos = np.arange(len(top_tokens))

        ax.barh(y_pos, top_scores_norm, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_tokens, fontsize=12, family='monospace')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Attribution Score (Red → Insufficient, Blue → Sufficient)', fontsize=11)
        ax.set_ylabel('Token', fontsize=11)

        # Title with prediction
        probe_pred = 1 if token_scores.mean() > 0 else 0
        true_label_str = "Sufficient" if label else "Insufficient"
        pred_label_str = "Sufficient" if probe_pred else "Insufficient"
        correct = "✓" if probe_pred == label else "✗"

        ax.set_title(f'Example {idx+1}: True={true_label_str}, Pred={pred_label_str} {correct}\n'
                     f'Question: {question}\n'
                     f'Top {top_k} Most Important Tokens (Layer {layer_idx})',
                     fontsize=11, loc='left', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([-1.2, 1.2])

        # Add legend
        red_patch = mpatches.Patch(color='red', label='Pushes toward "Insufficient"', alpha=0.7)
        blue_patch = mpatches.Patch(color='blue', label='Pushes toward "Sufficient"', alpha=0.7)
        ax.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=10)

        # Invert y-axis so most important is at top
        ax.invert_yaxis()

        plt.tight_layout()

        # Save individual example
        example_output_path = output_path.replace('.png', f'_example{idx+1}.png')
        plt.savefig(example_output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(examples)} token attribution plots to: {os.path.dirname(output_path)}")
    print(f"\nInterpretation:")
    print(f"  - Blue bars: Tokens that push toward 'Sufficient' classification")
    print(f"  - Red bars: Tokens that push toward 'Insufficient' classification")
    print(f"  - Bar length: Strength of the contribution")
