"""
Visualize multiclass probe training results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Class names for interpretability
CLASS_NAMES = {
    0: "Answerable",
    1: "Missing Critical Info", 
    2: "Incomplete Constraint",
    3: "Contradictory Info",
    4: "Irrelevant Question",
    5: "Other Unanswerability"
}

# Colors for each class (consistent across plots)
CLASS_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']


def plot_multiclass_layer_performance(results, output_path):
    """
    Plot macro F1, weighted F1, and accuracy vs layer for all models
    
    Args:
        results: Dict from multiclass experiment {model_name: {layer_X: metrics}}
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for model_name, model_results in results.items():
        layers = [int(k.split('_')[1]) for k in model_results.keys()]
        layers.sort()
        
        # Extract metrics
        macro_f1 = [model_results[f'layer_{l}']['f1_macro'] for l in layers]
        weighted_f1 = [model_results[f'layer_{l}']['f1_weighted'] for l in layers]
        accuracy = [model_results[f'layer_{l}']['accuracy'] for l in layers]
        
        # Plot macro F1 (most important for multiclass)
        axes[0, 0].plot(layers, macro_f1, 'o-', label=model_name, linewidth=2, markersize=4)
        
        # Plot weighted F1
        axes[0, 1].plot(layers, weighted_f1, 's-', label=model_name, linewidth=2, markersize=4)
        
        # Plot accuracy
        axes[1, 0].plot(layers, accuracy, '^-', label=model_name, linewidth=2, markersize=4)
    
    # Configure macro F1 plot
    axes[0, 0].axhline(y=1/6, color='gray', linestyle='--', label='Random (6-class)', alpha=0.5)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Macro F1 Score')
    axes[0, 0].set_title('Macro F1 vs Layer (Most Important)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Configure weighted F1 plot
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Weighted F1 Score')
    axes[0, 1].set_title('Weighted F1 vs Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Configure accuracy plot
    axes[1, 0].axhline(y=1/6, color='gray', linestyle='--', label='Random (6-class)', alpha=0.5)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Layer')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Per-class F1 heatmap for best model
    best_model = max(results.keys(), key=lambda m: max(
        results[m][layer]['f1_macro'] for layer in results[m].keys()
    ))
    
    # Create per-class F1 matrix
    layers = [int(k.split('_')[1]) for k in results[best_model].keys()]
    layers.sort()
    
    per_class_matrix = []
    for layer in layers:
        layer_metrics = results[best_model][f'layer_{layer}']
        row = [layer_metrics.get(f'f1_class_{i}', 0) for i in range(6)]
        per_class_matrix.append(row)
    
    per_class_matrix = np.array(per_class_matrix)
    
    # Plot heatmap
    im = axes[1, 1].imshow(per_class_matrix.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Class')
    axes[1, 1].set_title(f'Per-Class F1 Heatmap ({best_model})')
    
    # Set ticks
    axes[1, 1].set_xticks(range(0, len(layers), 5))
    axes[1, 1].set_xticklabels([str(layers[i]) for i in range(0, len(layers), 5)])
    axes[1, 1].set_yticks(range(6))
    axes[1, 1].set_yticklabels([f"{i}: {CLASS_NAMES[i][:15]}" for i in range(6)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Multiclass layer performance plot saved to: {output_path}")


def plot_model_comparison(results, output_path):
    """
    Compare best layer performance across models
    
    Args:
        results: Dict from multiclass experiment
        output_path: Where to save the plot
    """
    model_names = []
    best_macro_f1 = []
    best_layers = []
    per_class_f1 = {i: [] for i in range(6)}
    
    for model_name, model_results in results.items():
        # Find best layer by macro F1
        best_layer_key = max(model_results.keys(), 
                           key=lambda k: model_results[k]['f1_macro'])
        best_metrics = model_results[best_layer_key]
        best_layer_idx = int(best_layer_key.split('_')[1])
        
        model_names.append(model_name.replace('-', '\n'))  # Line break for readability
        best_macro_f1.append(best_metrics['f1_macro'])
        best_layers.append(best_layer_idx)
        
        # Collect per-class F1
        for i in range(6):
            per_class_f1[i].append(best_metrics.get(f'f1_class_{i}', 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overall performance comparison
    x_pos = np.arange(len(model_names))
    bars = axes[0].bar(x_pos, best_macro_f1, color='steelblue', alpha=0.7)
    
    # Add best layer annotations
    for i, (bar, layer) in enumerate(zip(bars, best_layers)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'Layer {layer}', ha='center', va='bottom', fontsize=9)
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Best Macro F1 Score')
    axes[0].set_title('Model Comparison (Best Layer Performance)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, max(best_macro_f1) * 1.1])
    
    # Plot 2: Per-class F1 comparison
    per_class_matrix = np.array([per_class_f1[i] for i in range(6)])
    
    im = axes[1].imshow(per_class_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Class')
    axes[1].set_title('Per-Class F1 Comparison (Best Layers)')
    
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels([name.replace('\n', '-') for name in model_names], rotation=45)
    axes[1].set_yticks(range(6))
    axes[1].set_yticklabels([f"{i}: {CLASS_NAMES[i][:15]}" for i in range(6)])
    
    # Add text annotations
    for i in range(6):
        for j in range(len(model_names)):
            text = axes[1].text(j, i, f'{per_class_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {output_path}")


def plot_confusion_matrix(results, model_name, output_path):
    """
    Plot confusion matrix for best performing model/layer
    
    Args:
        results: Dict from multiclass experiment
        model_name: Which model to plot
        output_path: Where to save the plot
    """
    model_results = results[model_name]
    
    # Find best layer
    best_layer_key = max(model_results.keys(), 
                        key=lambda k: model_results[k]['f1_macro'])
    best_metrics = model_results[best_layer_key]
    best_layer_idx = int(best_layer_key.split('_')[1])
    
    # Get confusion matrix
    cm = np.array(best_metrics['confusion_matrix'])
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f"{i}: {CLASS_NAMES[i][:10]}" for i in range(6)],
               yticklabels=[f"{i}: {CLASS_NAMES[i][:10]}" for i in range(6)],
               ax=axes[0], cbar=True)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title(f'Confusion Matrix (Counts)\n{model_name} - Layer {best_layer_idx}')
    
    # Plot normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=[f"{i}: {CLASS_NAMES[i][:10]}" for i in range(6)],
               yticklabels=[f"{i}: {CLASS_NAMES[i][:10]}" for i in range(6)],
               ax=axes[1], cbar=True, vmin=0, vmax=1)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title(f'Confusion Matrix (Normalized)\n{model_name} - Layer {best_layer_idx}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def analyze_multiclass_results(results):
    """
    Print detailed analysis of multiclass results
    
    Args:
        results: Dict from multiclass experiment
    """
    print("\n" + "="*80)
    print("MULTICLASS ANALYSIS SUMMARY")
    print("="*80)
    
    for model_name, model_results in results.items():
        # Find best layer
        best_layer_key = max(model_results.keys(), 
                           key=lambda k: model_results[k]['f1_macro'])
        best_metrics = model_results[best_layer_key]
        best_layer_idx = int(best_layer_key.split('_')[1])
        
        print(f"\n{model_name.upper()}:")
        print(f"  Best Layer: {best_layer_idx}")
        print(f"  Macro F1: {best_metrics['f1_macro']:.4f}")
        print(f"  Weighted F1: {best_metrics['f1_weighted']:.4f}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        
        print(f"  Per-class F1:")
        for i in range(6):
            f1 = best_metrics.get(f'f1_class_{i}', 0)
            class_name = CLASS_NAMES[i]
            print(f"    {i} ({class_name}): {f1:.3f}")
        
        # Identify problematic classes
        problematic_classes = []
        good_classes = []
        for i in range(6):
            f1 = best_metrics.get(f'f1_class_{i}', 0)
            if f1 < 0.3:
                problematic_classes.append((i, CLASS_NAMES[i], f1))
            elif f1 > 0.6:
                good_classes.append((i, CLASS_NAMES[i], f1))
        
        if problematic_classes:
            print(f"  ⚠️  Problematic classes (F1 < 0.3):")
            for i, name, f1 in problematic_classes:
                print(f"    {i} ({name}): {f1:.3f}")
        
        if good_classes:
            print(f"  ✅ Well-learned classes (F1 > 0.6):")
            for i, name, f1 in good_classes:
                print(f"    {i} ({name}): {f1:.3f}")
    
    print("\n" + "="*80)


def create_multiclass_plots(results_path, output_dir):
    """
    Create all multiclass visualization plots
    
    Args:
        results_path: Path to multiclass_linear.json
        output_dir: Directory to save plots
    """
    import json
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating multiclass visualization plots...")
    
    # 1. Layer performance across models
    plot_multiclass_layer_performance(
        results, 
        f"{output_dir}/multiclass_layer_performance.png"
    )
    
    # 2. Model comparison
    plot_model_comparison(
        results,
        f"{output_dir}/multiclass_model_comparison.png"
    )
    
    # 3. Confusion matrices for each model
    for model_name in results.keys():
        safe_model_name = model_name.replace('/', '_').replace('.', '_')
        plot_confusion_matrix(
            results,
            model_name,
            f"{output_dir}/confusion_matrix_{safe_model_name}.png"
        )
    
    # 4. Print analysis
    analyze_multiclass_results(results)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    # Test with dummy data or load real results
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "plots/multiclass"
        create_multiclass_plots(results_path, output_dir)
    else:
        print("Usage: python plot_multiclass_results.py <results_path> [output_dir]")
        print("Example: python viz/plot_multiclass_results.py experiments/multiclass_probes/multiclass_linear.json plots/multiclass")