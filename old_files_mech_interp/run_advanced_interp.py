"""
Run comprehensive mechanistic interpretability analyses on embeddings

Usage:
    python run_advanced_interp.py \
        --train_embeddings data/embeddings/umwp_train_qwen2.5-math-1.5b_last_token.npy \
        --train_labels data/insufficient_dataset_umwp/umwp_train.json \
        --test_embeddings data/embeddings/umwp_test_qwen2.5-math-1.5b_last_token.npy \
        --test_labels data/insufficient_dataset_umwp/umwp_test.json \
        --output_dir experiments/probe_results
"""
import argparse
import json
import os
import numpy as np

from viz.advanced_mech_interp import (
    selectivity_analysis,
    control_task_probing,
    subspace_dimensionality_analysis,
    nearest_neighbor_analysis,
    centered_kernel_alignment,
    lda_visualization,
    umap_visualization,
    intrinsic_dimensionality
)


def load_embeddings_and_labels(embeddings_path, labels_path):
    """
    Load embeddings and extract labels from JSON

    Args:
        embeddings_path: Path to .npy file
        labels_path: Path to JSON file with 'is_sufficient' field

    Returns:
        embeddings: numpy array
        labels: numpy array of 0/1
    """
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading labels from {labels_path}")
    with open(labels_path, 'r') as f:
        data = json.load(f)

    labels = np.array([1 if item['is_sufficient'] else 0 for item in data])
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sufficient: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Insufficient: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description='Advanced mechanistic interpretability analysis')
    parser.add_argument('--train_embeddings', type=str, required=True,
                        help='Path to train embeddings .npy file')
    parser.add_argument('--train_labels', type=str, required=True,
                        help='Path to train labels JSON file')
    parser.add_argument('--test_embeddings', type=str, required=True,
                        help='Path to test embeddings .npy file')
    parser.add_argument('--test_labels', type=str, required=True,
                        help='Path to test labels JSON file')
    parser.add_argument('--output_dir', type=str, default='experiments/probe_results',
                        help='Base output directory (will create mech_interp subfolder)')
    parser.add_argument('--skip_analyses', nargs='+', default=[],
                        choices=['selectivity', 'control', 'subspace', 'knn', 'cka', 'lda', 'umap', 'intrinsic'],
                        help='Analyses to skip (optional)')

    args = parser.parse_args()

    print("="*80)
    print("ADVANCED MECHANISTIC INTERPRETABILITY ANALYSIS")
    print("="*80)

    # Create output directory
    mech_interp_dir = os.path.join(args.output_dir, 'mech_interp')
    os.makedirs(mech_interp_dir, exist_ok=True)

    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    train_embeddings, train_labels = load_embeddings_and_labels(
        args.train_embeddings, args.train_labels
    )
    test_embeddings, test_labels = load_embeddings_and_labels(
        args.test_embeddings, args.test_labels
    )

    print(f"\nOutput directory: {mech_interp_dir}")

    # Run analyses
    analyses = {
        'selectivity': {
            'name': 'Selectivity Analysis',
            'func': lambda: selectivity_analysis(
                train_embeddings, train_labels,
                os.path.join(mech_interp_dir, 'selectivity_analysis.png')
            )
        },
        'control': {
            'name': 'Control Task Probing',
            'func': lambda: control_task_probing(
                train_embeddings, train_labels, test_embeddings, test_labels,
                os.path.join(mech_interp_dir, 'control_task_probing.png')
            )
        },
        'subspace': {
            'name': 'Subspace Dimensionality',
            'func': lambda: subspace_dimensionality_analysis(
                train_embeddings, train_labels, test_embeddings, test_labels,
                os.path.join(mech_interp_dir, 'subspace_dimensionality.png')
            )
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'func': lambda: nearest_neighbor_analysis(
                train_embeddings, train_labels, test_embeddings, test_labels,
                os.path.join(mech_interp_dir, 'knn_analysis.png')
            )
        },
        'cka': {
            'name': 'Centered Kernel Alignment',
            'func': lambda: centered_kernel_alignment(
                train_embeddings, train_labels,
                os.path.join(mech_interp_dir, 'cka_similarity.png')
            )
        },
        'lda': {
            'name': 'Linear Discriminant Analysis',
            'func': lambda: lda_visualization(
                train_embeddings, train_labels,
                os.path.join(mech_interp_dir, 'lda_visualization.png')
            )
        },
        'umap': {
            'name': 'UMAP Visualization',
            'func': lambda: umap_visualization(
                train_embeddings, train_labels,
                os.path.join(mech_interp_dir, 'umap_visualization.png')
            )
        },
        'intrinsic': {
            'name': 'Intrinsic Dimensionality',
            'func': lambda: intrinsic_dimensionality(
                train_embeddings, train_labels,
                os.path.join(mech_interp_dir, 'intrinsic_dimensionality.png')
            )
        }
    }

    for key, analysis in analyses.items():
        if key in args.skip_analyses:
            print(f"\nSkipping {analysis['name']}...")
            continue

        try:
            analysis['func']()
        except Exception as e:
            print(f"\nERROR in {analysis['name']}: {str(e)}")
            print("Continuing with other analyses...")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {mech_interp_dir}")
    print("\nGenerated plots:")
    for key, analysis in analyses.items():
        if key not in args.skip_analyses:
            filename = key + '_' if key == 'control' else ''
            if key == 'selectivity':
                filename = 'selectivity_analysis.png'
            elif key == 'control':
                filename = 'control_task_probing.png'
            elif key == 'subspace':
                filename = 'subspace_dimensionality.png'
            elif key == 'knn':
                filename = 'knn_analysis.png'
            elif key == 'cka':
                filename = 'cka_similarity.png'
            elif key == 'lda':
                filename = 'lda_visualization.png'
            elif key == 'umap':
                filename = 'umap_visualization.png'
            elif key == 'intrinsic':
                filename = 'intrinsic_dimensionality.png'
            print(f"  - {filename}")


if __name__ == '__main__':
    main()
