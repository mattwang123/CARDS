"""
Visualize probe results comparing last_token vs mean pooling strategies

Creates multiple plots analyzing:
1. Generalization heatmaps (train vs test)
2. Pooling strategy comparison
3. Cross-dataset generalization
4. Best configurations
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(last_token_path, mean_path):
    """Load both pooling strategy results"""
    with open(last_token_path, 'r') as f:
        last_token = json.load(f)

    with open(mean_path, 'r') as f:
        mean_pool = json.load(f)

    return last_token, mean_pool


def create_generalization_heatmap(model_name, last_token_data, mean_data, metric='f1', output_path=None):
    """
    Create side-by-side heatmaps comparing pooling strategies

    Rows: Training datasets (UMWP, GSM8K, TreeCut, ALL)
    Cols: Test datasets (UMWP, GSM8K, TreeCut)
    """
    train_configs = ['train_on_umwp', 'train_on_gsm8k', 'train_on_treecut', 'train_on_ALL']
    test_datasets = ['test_on_umwp', 'test_on_gsm8k', 'test_on_treecut']

    train_labels = ['UMWP', 'GSM8K', 'TreeCut', 'ALL']
    test_labels = ['UMWP', 'GSM8K', 'TreeCut']

    # Extract data
    last_token_matrix = np.zeros((4, 3))
    mean_matrix = np.zeros((4, 3))

    for i, train_config in enumerate(train_configs):
        for j, test_dataset in enumerate(test_datasets):
            last_token_matrix[i, j] = last_token_data[model_name][train_config]['results'][test_dataset][metric]
            mean_matrix[i, j] = mean_data[model_name][train_config]['results'][test_dataset][metric]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Last token heatmap
    sns.heatmap(last_token_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'F1 Score'},
                xticklabels=test_labels, yticklabels=train_labels, ax=axes[0])
    axes[0].set_title(f'{model_name}\nLast Token Pooling', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Test Dataset', fontsize=11)
    axes[0].set_ylabel('Training Dataset', fontsize=11)

    # Mean pooling heatmap
    sns.heatmap(mean_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'F1 Score'},
                xticklabels=test_labels, yticklabels=train_labels, ax=axes[1])
    axes[1].set_title(f'{model_name}\nMean Pooling', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Test Dataset', fontsize=11)
    axes[1].set_ylabel('Training Dataset', fontsize=11)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_pooling_comparison(last_token_data, mean_data, metric='f1', output_path=None):
    """
    Compare pooling strategies across all models

    Bar plot showing average F1 scores for each model with both pooling strategies
    """
    models = list(last_token_data.keys())
    train_configs = ['train_on_umwp', 'train_on_gsm8k', 'train_on_treecut', 'train_on_ALL']
    test_datasets = ['test_on_umwp', 'test_on_gsm8k', 'test_on_treecut']

    last_token_avgs = []
    mean_avgs = []

    for model in models:
        # Compute average F1 across all train/test combinations
        last_scores = []
        mean_scores = []

        for train_config in train_configs:
            for test_dataset in test_datasets:
                last_scores.append(last_token_data[model][train_config]['results'][test_dataset][metric])
                mean_scores.append(mean_data[model][train_config]['results'][test_dataset][metric])

        last_token_avgs.append(np.mean(last_scores))
        mean_avgs.append(np.mean(mean_scores))

    # Create bar plot
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, last_token_avgs, width, label='Last Token',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, mean_avgs, width, label='Mean Pooling',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Pooling Strategy Comparison\n(Average across all train/test combinations)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '-\n') for m in models], fontsize=10)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_cross_dataset_analysis(last_token_data, mean_data, metric='f1', output_path=None):
    """
    Analyze cross-dataset generalization (train on X, test on Y where X≠Y)

    Shows how well models generalize to unseen datasets
    """
    models = list(last_token_data.keys())

    # Define cross-dataset pairs (train ≠ test)
    cross_pairs = [
        ('train_on_umwp', 'test_on_gsm8k'),
        ('train_on_umwp', 'test_on_treecut'),
        ('train_on_gsm8k', 'test_on_umwp'),
        ('train_on_gsm8k', 'test_on_treecut'),
        ('train_on_treecut', 'test_on_umwp'),
        ('train_on_treecut', 'test_on_gsm8k'),
    ]

    pair_labels = [
        'UMWP→GSM8K',
        'UMWP→TreeCut',
        'GSM8K→UMWP',
        'GSM8K→TreeCut',
        'TreeCut→UMWP',
        'TreeCut→GSM8K'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (pair, label) in enumerate(zip(cross_pairs, pair_labels)):
        train_config, test_dataset = pair

        last_scores = [last_token_data[m][train_config]['results'][test_dataset][metric] for m in models]
        mean_scores = [mean_data[m][train_config]['results'][test_dataset][metric] for m in models]

        x = np.arange(len(models))
        width = 0.35

        axes[idx].bar(x - width/2, last_scores, width, label='Last Token',
                     color='#3498db', alpha=0.8, edgecolor='black')
        axes[idx].bar(x + width/2, mean_scores, width, label='Mean',
                     color='#e74c3c', alpha=0.8, edgecolor='black')

        axes[idx].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('F1 Score', fontsize=10)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([m.split('-')[0] for m in models], fontsize=9, rotation=15)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        axes[idx].set_ylim([0.5, 1.0])

    plt.suptitle('Cross-Dataset Generalization Analysis\n(Training on one dataset, testing on another)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_best_config_comparison(last_token_data, mean_data, metric='f1', output_path=None):
    """
    Show best configuration (train_on_ALL) across all models
    """
    models = list(last_token_data.keys())
    test_datasets = ['test_on_umwp', 'test_on_gsm8k', 'test_on_treecut']
    test_labels = ['UMWP', 'GSM8K', 'TreeCut']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(test_datasets))
    width = 0.12

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, model in enumerate(models):
        last_scores = [last_token_data[model]['train_on_ALL']['results'][f'test_on_{ds.lower()}'][metric]
                      for ds in test_labels]
        mean_scores = [mean_data[model]['train_on_ALL']['results'][f'test_on_{ds.lower()}'][metric]
                      for ds in test_labels]

        # Plot last_token (solid) and mean (hatched)
        offset = (i - 1.5) * width
        ax.bar(x + offset, last_scores, width, label=f'{model} (Last)',
               color=colors[i], alpha=0.8, edgecolor='black')
        ax.bar(x + offset + 4*width, mean_scores, width, label=f'{model} (Mean)',
               color=colors[i], alpha=0.5, edgecolor='black', hatch='//')

    ax.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Configuration: Train on ALL datasets\n(Comparing Last Token vs Mean Pooling)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks([x[0] + 1.5*width, x[1] + 1.5*width, x[2] + 1.5*width])
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_summary_table(last_token_data, mean_data, output_path):
    """Create a summary table comparing pooling strategies"""
    models = list(last_token_data.keys())

    summary = []
    summary.append("="*100)
    summary.append("POOLING STRATEGY COMPARISON SUMMARY")
    summary.append("="*100)
    summary.append("")

    for model in models:
        summary.append(f"\n{model.upper()}")
        summary.append("-"*100)

        # Average across all configs
        all_last = []
        all_mean = []

        for train_config in ['train_on_umwp', 'train_on_gsm8k', 'train_on_treecut', 'train_on_ALL']:
            for test_dataset in ['test_on_umwp', 'test_on_gsm8k', 'test_on_treecut']:
                all_last.append(last_token_data[model][train_config]['results'][test_dataset]['f1'])
                all_mean.append(mean_data[model][train_config]['results'][test_dataset]['f1'])

        avg_last = np.mean(all_last)
        avg_mean = np.mean(all_mean)

        summary.append(f"  Average F1 (Last Token): {avg_last:.4f}")
        summary.append(f"  Average F1 (Mean Pool):  {avg_mean:.4f}")
        summary.append(f"  Difference:              {avg_mean - avg_last:+.4f} {'✓ Mean better' if avg_mean > avg_last else '✓ Last better'}")

        # Best configuration
        best_last = last_token_data[model]['train_on_ALL']
        best_mean = mean_data[model]['train_on_ALL']

        summary.append(f"\n  Best Config (train_on_ALL):")
        summary.append(f"    Last Token - Best Layer: {best_last['best_layer']}, Avg F1: {best_last['avg_f1']:.4f}")
        summary.append(f"    Mean Pool  - Best Layer: {best_mean['best_layer']}, Avg F1: {best_mean['avg_f1']:.4f}")

    summary.append("\n" + "="*100)

    summary_text = "\n".join(summary)

    with open(output_path, 'w') as f:
        f.write(summary_text)

    print(f"  ✓ Saved: {output_path}")
    print("\n" + summary_text)


def load_accuracy_results(accuracy_path):
    """Load accuracy evaluation results"""
    if not os.path.exists(accuracy_path):
        return None

    with open(accuracy_path, 'r') as f:
        return json.load(f)


def create_accuracy_heatmap(accuracy_data, output_path=None):
    """
    Create heatmap showing model accuracy on sufficient questions

    Rows: Models
    Cols: Datasets
    """
    if accuracy_data is None:
        print("  ⚠ Skipping accuracy heatmap - no data found")
        return

    models = list(accuracy_data.keys())
    datasets = ['umwp', 'gsm8k', 'treecut']
    dataset_labels = ['UMWP', 'GSM8K', 'TreeCut']

    # Extract accuracy matrix
    accuracy_matrix = np.zeros((len(models), len(datasets)))

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            accuracy_matrix[i, j] = accuracy_data[model][dataset]['accuracy']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Better model labels for heatmap
    model_labels = [
        'Qwen2.5-Math-1.5B',
        'Qwen2.5-1.5B',
        'Llama-3.2-3B',
        'Qwen2.5-Math-7B'
    ]

    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.0, vmax=1.0, cbar_kws={'label': 'Accuracy'},
                xticklabels=dataset_labels,
                yticklabels=model_labels,
                ax=ax, linewidths=1, linecolor='gray')

    ax.set_title('Model Accuracy on Sufficient Questions', fontsize=14, pad=20)
    ax.set_xlabel('Test Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_accuracy_comparison(accuracy_data, output_path=None):
    """
    Bar chart comparing model accuracy across datasets
    """
    if accuracy_data is None:
        print("  ⚠ Skipping accuracy comparison - no data found")
        return

    models = list(accuracy_data.keys())
    datasets = ['umwp', 'gsm8k', 'treecut']
    dataset_labels = ['UMWP', 'GSM8K', 'TreeCut']

    x = np.arange(len(datasets))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Better legend labels
    legend_labels = {
        'qwen2.5-math-1.5b': 'Qwen2.5-Math-1.5B',
        'qwen2.5-1.5b': 'Qwen2.5-1.5B',
        'llama-3.2-3b-instruct': 'Llama-3.2-3B',
        'qwen2.5-math-7b': 'Qwen2.5-Math-7B'
    }

    for i, model in enumerate(models):
        accuracies = [accuracy_data[model][ds]['accuracy'] for ds in datasets]
        offset = (i - 1.5) * width

        bars = ax.bar(x + offset, accuracies, width, label=legend_labels.get(model, model),
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy on Sufficient Questions', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.legend(fontsize=10, title='Model', frameon=True, shadow=True,
             loc='upper right', ncol=1)  # Vertical legend
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def create_combined_probe_accuracy(last_token_data, accuracy_data, output_path=None):
    """
    Combined plot showing:
    - Top: Probe F1 scores (detection of insufficient questions)
    - Bottom: Accuracy on sufficient questions (solving ability)

    This shows the full picture: Can models detect AND solve?
    """
    if accuracy_data is None:
        print("  ⚠ Skipping combined plot - no accuracy data found")
        return

    models = list(last_token_data.keys())
    datasets = ['umwp', 'gsm8k', 'treecut']
    dataset_labels = ['UMWP', 'GSM8K', 'TreeCut']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: Probe F1 scores (train_on_ALL configuration)
    x = np.arange(len(datasets))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # Better legend labels
    legend_labels = {
        'qwen2.5-math-1.5b': 'Qwen2.5-Math-1.5B',
        'qwen2.5-1.5b': 'Qwen2.5-1.5B',
        'llama-3.2-3b-instruct': 'Llama-3.2-3B',
        'qwen2.5-math-7b': 'Qwen2.5-Math-7B'
    }

    for i, model in enumerate(models):
        f1_scores = [last_token_data[model]['train_on_ALL']['results'][f'test_on_{ds}']['f1']
                    for ds in datasets]
        offset = (i - 1.5) * width

        bars = axes[0].bar(x + offset, f1_scores, width, label=legend_labels.get(model, model),
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    axes[0].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Insufficiency Detection (Probe F1)', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dataset_labels, fontsize=10)
    axes[0].legend(fontsize=9, title='Model', frameon=True, shadow=True, ncol=1, loc='upper right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([0.5, 1.0])

    # Bottom: Accuracy on sufficient questions
    for i, model in enumerate(models):
        accuracies = [accuracy_data[model][ds]['accuracy'] for ds in datasets]
        offset = (i - 1.5) * width

        bars = axes[1].bar(x + offset, accuracies, width, label=legend_labels.get(model, model),
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    axes[1].set_xlabel('Dataset', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('Problem Solving Accuracy (Sufficient Questions)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset_labels, fontsize=10)
    axes[1].legend(fontsize=9, title='Model', frameon=True, shadow=True, ncol=1, loc='upper right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.05])

    plt.suptitle('Model Evaluation: Detection and Solving', fontsize=16, y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot probe results comparing pooling strategies'
    )
    parser.add_argument('--last_token_dir', type=str,
                       default='experiments/all_probes_linear',
                       help='Directory with last_token results')
    parser.add_argument('--mean_dir', type=str,
                       default='experiments/all_probes_linear_mean',
                       help='Directory with mean pooling results')
    parser.add_argument('--accuracy_path', type=str,
                       default='experiments/accuracy_eval/accuracy_summaries.json',
                       help='Path to accuracy evaluation results')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    print("="*80)
    print("PLOTTING PROBE RESULTS")
    print("="*80)

    # Load results
    last_token_path = os.path.join(args.last_token_dir, 'best_layers_linear.json')
    mean_path = os.path.join(args.mean_dir, 'best_layers_linear.json')

    print(f"\nLoading results...")
    print(f"  Last Token: {last_token_path}")
    print(f"  Mean Pool:  {mean_path}")
    print(f"  Accuracy:   {args.accuracy_path}")

    last_token_data, mean_data = load_results(last_token_path, mean_path)
    accuracy_data = load_accuracy_results(args.accuracy_path)

    models = list(last_token_data.keys())

    print(f"\n✓ Loaded results for {len(models)} models")
    if accuracy_data:
        print(f"✓ Loaded accuracy evaluation results")
    else:
        print(f"⚠ No accuracy results found - skipping accuracy plots")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")

    # 1. Generalization heatmaps for each model
    print("[1/8] Creating generalization heatmaps...")
    for model in models:
        output_path = os.path.join(args.output_dir, f'heatmap_{model}.png')
        create_generalization_heatmap(model, last_token_data, mean_data,
                                     metric='f1', output_path=output_path)

    # 2. Pooling comparison across models
    print("\n[2/8] Creating pooling comparison...")
    output_path = os.path.join(args.output_dir, 'pooling_comparison.png')
    create_pooling_comparison(last_token_data, mean_data,
                             metric='f1', output_path=output_path)

    # 3. Cross-dataset generalization
    print("\n[3/8] Creating cross-dataset analysis...")
    output_path = os.path.join(args.output_dir, 'cross_dataset_generalization.png')
    create_cross_dataset_analysis(last_token_data, mean_data,
                                 metric='f1', output_path=output_path)

    # 4. Best configuration comparison
    print("\n[4/8] Creating best config comparison...")
    output_path = os.path.join(args.output_dir, 'best_config_comparison.png')
    create_best_config_comparison(last_token_data, mean_data,
                                 metric='f1', output_path=output_path)

    # 5. Summary table
    print("\n[5/8] Creating summary table...")
    output_path = os.path.join(args.output_dir, 'summary.txt')
    create_summary_table(last_token_data, mean_data, output_path)

    # 6. Accuracy heatmap
    print("\n[6/8] Creating accuracy heatmap...")
    output_path = os.path.join(args.output_dir, 'accuracy_heatmap.png')
    create_accuracy_heatmap(accuracy_data, output_path=output_path)

    # 7. Accuracy comparison
    print("\n[7/8] Creating accuracy comparison...")
    output_path = os.path.join(args.output_dir, 'accuracy_comparison.png')
    create_accuracy_comparison(accuracy_data, output_path=output_path)

    # 8. Combined probe + accuracy plot
    print("\n[8/8] Creating combined detection + solving plot...")
    output_path = os.path.join(args.output_dir, 'combined_detection_solving.png')
    create_combined_probe_accuracy(last_token_data, accuracy_data, output_path=output_path)

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  PROBE RESULTS:")
    print("    - heatmap_<model>.png (4 files)")
    print("    - pooling_comparison.png")
    print("    - cross_dataset_generalization.png")
    print("    - best_config_comparison.png")
    print("    - summary.txt")
    if accuracy_data:
        print("  ACCURACY EVALUATION:")
        print("    - accuracy_heatmap.png")
        print("    - accuracy_comparison.png")
        print("    - combined_detection_solving.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
