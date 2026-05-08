"""
================================================================================
VISUALIZATION: Epistemic Drift & Generative Momentum (Nature Style)
================================================================================
This script generates publication-ready visualizations and data tables to prove 
the "Overthinking Trap" hypothesis:
1. Layer-Time Waterfall: Shows the structural collapse of latent awareness.
2. Generative Drift (UMAP): Maps the physical trajectory into the Hallucination Zone.
3. Epistemic Decay Table: Exports CSV and LaTeX tracking exact F1 collapse deltas.
================================================================================
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# --- NATURE-STYLE AESTHETICS ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- CONFIGURATION ---
DATASET = 'umwp' # Change to 'treecut' as needed
BASE_DIR = '/export/fs06/hwang302/CARDS/exp_temporal_new'
RESULTS_PATH = os.path.join(BASE_DIR, 'results', f'final_momentum_{DATASET}.json')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings', DATASET)
PROBE_DIR = os.path.join(BASE_DIR, 'probes', DATASET)

TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]

def load_data():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}. Please run Exp 3 first.")
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)

def plot_layer_time_waterfall(model_slug):
    """
    Produces an evolving 2D heatmap tracking awareness 'hotspots' across all layers and timesteps.
    Applies a survival-rate alpha mask so dissolved signals fade into the background.
    """
    print(f"Generating Layer-Time Waterfall for {model_slug}...")
    data = load_data()
    model_key = next((k for k in data.keys() if model_slug in k), None)
    if not model_key: 
        print(f"  ! No data found for {model_slug}")
        return

    model_data = data[model_key]
    if "t_0" not in model_data:
        return
        
    num_layers = len(model_data["t_0"]["all_layers"])
    
    f1_matrix = np.zeros((num_layers, len(TIMESTEPS)))
    survival_mask = np.zeros((num_layers, len(TIMESTEPS)))
    
    for j, t in enumerate(TIMESTEPS):
        t_key = f"t_{t}"
        if t_key not in model_data: continue
            
        survival = model_data[t_key].get("test_survival_rate_pct", 0) / 100.0
        
        for i in range(num_layers):
            f1 = model_data[t_key]["all_layers"][f"layer_{i}"]["test_f1"]
            f1_matrix[i, j] = f1
            survival_mask[i, j] = survival

    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Custom colormap: Navy -> Teal -> Yellow (Nature aesthetic)
    cmap = sns.color_palette("mako", as_cmap=True)
    
    # Apply survival rate as an alpha mask (fade out dead sequences)
    ax.set_facecolor('#ffffff')
    
    heatmap = ax.imshow(
        f1_matrix, 
        aspect='auto', 
        cmap=cmap, 
        alpha=np.clip(survival_mask + 0.1, 0, 1), # Ensure it doesn't vanish entirely
        origin='lower',
        vmin=0.0, vmax=1.0
    )
    
    ax.set_xticks(np.arange(len(TIMESTEPS)))
    ax.set_xticklabels(TIMESTEPS)
    ax.set_ylabel('Transformer Layer')
    ax.set_xlabel('Generative Timestep ($t$)')
    ax.set_title(f'Epistemic Signal Decay ({model_slug})', pad=15, fontweight='bold')
    
    cbar = plt.colorbar(heatmap, ax=ax, label='Latent F1 Score (Insufficiency)')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f"figures/waterfall_{DATASET}_{model_slug}.pdf")
    plt.close()
    print(f"  -> Saved figures/waterfall_{DATASET}_{model_slug}.pdf")

def plot_generative_drift_umap(model_slug, t_start=0, t_end=32):
    """
    Visualizes epistemic drift by tracking how latent states are pulled across 
    the decision boundary by autoregressive momentum.
    """
    print(f"Generating UMAP Trajectory for {model_slug} (t={t_start} -> t={t_end})...")
    
    data = load_data()
    model_key = next((k for k in data.keys() if model_slug in k), None)
    if not model_key: return

    try:
        layer_start = data[model_key][f"t_{t_start}"]["best_layer"]
        layer_end = data[model_key][f"t_{t_end}"]["best_layer"]
    except KeyError:
        print(f"  ! Missing layer data for t={t_start} or t={t_end}. Skipping UMAP.")
        return
        
    emb_start_path = os.path.join(EMB_DIR, model_slug, f"t_{t_start}_test.npy")
    emb_end_path = os.path.join(EMB_DIR, model_slug, f"t_{t_end}_test.npy")
    
    if not (os.path.exists(emb_start_path) and os.path.exists(emb_end_path)):
        print(f"  ! Missing embeddings for {model_slug}. Skipping UMAP.")
        return

    X_start = np.load(emb_start_path)[:, layer_start, :]
    X_end = np.load(emb_end_path)[:, layer_end, :]
    
    # Dynamically route the ground truth JSON path
    if DATASET == 'umwp':
        test_json_path = 'src/data/processed/insufficient_dataset_umwp/umwp_test.json'
    else:
        test_json_path = 'src/data/processed/treecut/treecut_test.json'
        
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in test_data])
    
    # Track only 'Insufficient' problems to see if they drift into hallucination
    insuf_mask = (labels == 1)
    X_start_insuf = X_start[insuf_mask]
    X_end_insuf = X_end[insuf_mask]
    
    if len(X_start_insuf) == 0:
        return
        
    # Fit UMAP on the starting state to define the topological space
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    U_start = reducer.fit_transform(X_start_insuf)
    U_end = reducer.transform(X_end_insuf)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.scatter(U_start[:, 0], U_start[:, 1], c='#2b83ba', s=10, alpha=0.6, label=f'Initial State ($t={t_start}$)')
    ax.scatter(U_end[:, 0], U_end[:, 1], c='#d7191c', s=10, alpha=0.6, label=f'Drift State ($t={t_end}$)')
    
    # Draw arrows for a random subset to show the vector field of Generative Momentum
    n_arrows = min(50, len(U_start))
    idx_sample = np.random.choice(len(U_start), size=n_arrows, replace=False)
    for i in idx_sample:
        ax.arrow(U_start[i, 0], U_start[i, 1], 
                 U_end[i, 0] - U_start[i, 0], U_end[i, 1] - U_start[i, 1], 
                 color='gray', alpha=0.3, head_width=0.1, length_includes_head=True)

    ax.set_title(f'Generative Drift (UMAP Trajectory) - {model_slug}', fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"figures/umap_drift_{DATASET}_{model_slug}.pdf")
    plt.close()
    print(f"  -> Saved figures/umap_drift_{DATASET}_{model_slug}.pdf")

def generate_academic_table():
    """Generates a CSV for analysis and a LaTeX table for the paper."""
    print(f"\nGenerating Academic Summary Tables for {DATASET.upper()}...")
    data = load_data()
    
    table_data = []
    
    for model_key, results in data.items():
        slug = model_key.split('/')[-1]
        
        try:
            f1_0 = results["t_0"]["max_test_f1"]
            l_0 = results["t_0"]["best_layer"]
            
            f1_64 = results.get("t_64", {}).get("max_test_f1", 0.0)
            l_64 = results.get("t_64", {}).get("best_layer", "-")
            
            delta = f1_0 - f1_64
            layer_shift = f"L{l_0} -> L{l_64}"
            latex_layer_shift = f"L{l_0} $\\rightarrow$ L{l_64}"
            
            table_data.append({
                "model": slug,
                "f1_t0": f1_0,
                "f1_t64": f1_64,
                "delta": delta,
                "layer_shift": layer_shift,
                "latex_layer_shift": latex_layer_shift
            })
            
        except KeyError:
            continue
            
    # Sort data alphabetically by model name
    table_data = sorted(table_data, key=lambda x: x["model"])

    os.makedirs('figures', exist_ok=True)

    # 1. GENERATE CSV
    csv_path = f'figures/momentum_collapse_{DATASET}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Initial_F1_t0', 'Collapse_F1_t64', 'Delta_F1', 'Layer_Shift'])
        for row in table_data:
            writer.writerow([row['model'], f"{row['f1_t0']:.3f}", f"{row['f1_t64']:.3f}", f"-{row['delta']:.3f}", row['layer_shift']])
    print(f"  -> Saved analytical data to {csv_path}")

    # 2. GENERATE LATEX & MARKDOWN
    md_table = "| Model | Initial F1 ($t=0$) | Collapse F1 ($t=64$) | $\\Delta$ F1 | Optimal Layer Shift |\n"
    md_table += "|---|---|---|---|---|\n"
    
    latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{lcccc}\n\\toprule\n"
    latex_table += "\\textbf{Model} & \\textbf{Initial F1 ($t=0$)} & \\textbf{Collapse F1 ($t=64$)} & \\textbf{$\\Delta$ F1} & \\textbf{Layer Shift} \\\\\n\\midrule\n"
    
    for row in table_data:
        # Markdown is safe
        md_table += f"| {row['model']} | {row['f1_t0']:.3f} | {row['f1_t64']:.3f} | -{row['delta']:.3f} | {row['latex_layer_shift']} |\n"
        
        # FIX: Process the string replacement outside the f-string first
        safe_model_name = row['model'].replace('_', '\\_')
        
        # Then inject the safe variable into the f-string
        latex_table += f"{safe_model_name} & {row['f1_t0']:.3f} & {row['f1_t64']:.3f} & -{row['delta']:.3f} & {row['latex_layer_shift']} \\\\\n"
        
    latex_table += "\\bottomrule\n\\end{tabular}\n"
    latex_table += f"\\caption{{Generative Momentum ({DATASET.upper()}): Collapse of Latent F1 across sequence length.}}\n"
    latex_table += "\\label{tab:momentum_collapse}\n\\end{table}"
    
    latex_path = f'figures/table_collapse_{DATASET}.md'
    with open(latex_path, 'w') as f:
        f.write("### Markdown Format\n" + md_table + "\n\n### LaTeX Format\n" + latex_table)
    print(f"  -> Saved publication tables to {latex_path}")

if __name__ == '__main__':
    # Add or remove specific models you want to visualize
    target_models = [
        'Meta-Llama-3.1-8B-Instruct', 
        'Qwen2.5-Math-7B-Instruct',
        'gemma-3-12b-it',
        'Olmo-3-7B-Instruct'
    ]
    
    generate_academic_table()
    
    for model in target_models:
        plot_layer_time_waterfall(model)
        plot_generative_drift_umap(model, t_start=0, t_end=32)
        
    print("\n[SUCCESS] Visualizations ready for paper integration.")