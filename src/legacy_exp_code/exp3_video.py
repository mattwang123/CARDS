"""
================================================================================
ANIMATION: Generative Drift (UMAP Trajectory)
================================================================================
Generates an animated GIF showing the token-by-token collapse of the model's 
latent awareness into the Q1 Hallucination zone.
================================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# --- NATURE-STYLE AESTHETICS ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150, # Slightly lower DPI for GIF speed/size optimization
})

# --- CONFIGURATION ---
DATASET = 'umwp' # Change to 'treecut' as needed
BASE_DIR = '/export/fs06/hwang302/CARDS/exp_temporal_new'
RESULTS_PATH = os.path.join(BASE_DIR, 'results', f'final_momentum_{DATASET}.json')
EMB_DIR = os.path.join(BASE_DIR, 'embeddings', DATASET)

TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]

def load_data():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r') as f:
        return json.load(f)

def generate_umap_animation(model_slug):
    print(f"\n[{model_slug}] Initializing UMAP Animation generation...")
    
    data = load_data()
    model_key = next((k for k in data.keys() if model_slug in k), None)
    if not model_key: 
        print(f"  ! Model {model_slug} not found in {RESULTS_PATH}")
        return

    # Load Ground Truth to isolate 'Insufficient' problems
    if DATASET == 'umwp':
        test_json_path = 'src/data/processed/insufficient_dataset_umwp/umwp_test.json'
    else:
        test_json_path = 'src/data/processed/treecut/treecut_test.json'
        
    if not os.path.exists(test_json_path):
        print(f"  ! Missing ground truth test file: {test_json_path}")
        return
        
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
        
    labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in test_data])
    insuf_mask = (labels == 1)

    X_dict = {}
    valid_timesteps = []
    
    # 1. Load all available embeddings across time
    print("  -> Loading hidden states from disk...")
    for t in TIMESTEPS:
        try:
            best_layer = data[model_key][f"t_{t}"]["best_layer"]
            emb_path = os.path.join(EMB_DIR, model_slug, f"t_{t}_test.npy")
            if os.path.exists(emb_path):
                X = np.load(emb_path)[:, best_layer, :]
                X_dict[t] = X[insuf_mask] # Track only the broken problems
                valid_timesteps.append(t)
        except Exception:
            continue

    if not valid_timesteps: 
        print("  ! No valid embeddings found to animate.")
        return

    # 2. Fit Global Topological Space
    print("  -> Fitting global UMAP projection (this may take a minute)...")
    X_all = np.vstack(list(X_dict.values()))
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine').fit(X_all)
    
    # Define locked camera bounds based on the global space
    U_all = reducer.transform(X_all)
    x_min, x_max = U_all[:, 0].min() - 1, U_all[:, 0].max() + 1
    y_min, y_max = U_all[:, 1].min() - 1, U_all[:, 1].max() + 1

    os.makedirs('figures', exist_ok=True)
    filenames = []
    
    # 3. Generate Individual Frames
    print("  -> Rendering frames...")
    for t in valid_timesteps:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('#f8f9fa') # Light gray background for contrast
        
        U_t = reducer.transform(X_dict[t])
        
        # Color transition from Aware (Blue) to Hallucination (Red)
        progress = valid_timesteps.index(t) / max(1, (len(valid_timesteps) - 1))
        color = plt.cm.coolwarm(progress)
        
        ax.scatter(U_t[:, 0], U_t[:, 1], c=[color], s=25, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f"Generative Momentum Collapse\n{model_slug} (Timestep $t={t}$)", fontweight='bold', pad=15)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP Dimension 1", fontweight='bold')
        ax.set_ylabel("UMAP Dimension 2", fontweight='bold')
        
        filename = f"figures/temp_frame_{model_slug}_{t}.png"
        filenames.append(filename)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    # 4. Compile GIF
    print("  -> Compiling GIF...")
    try:
        import imageio
        gif_path = f"figures/umap_animation_{DATASET}_{model_slug}.gif"
        
        # Duration: time (in ms) between frames. 800ms gives a nice readable pace.
        with imageio.get_writer(gif_path, mode='I', duration=800) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
            # Optional: Append the last frame 2 extra times so it pauses at the end
            for _ in range(2):
                writer.append_data(image)
                
        print(f"  [SUCCESS] Animation saved to {gif_path}")
    except ImportError:
        print("  ! 'imageio' library not found. Run: pip install imageio")
        
    # 5. Cleanup temporary frames
    for filename in filenames:
        if os.path.exists(filename): 
            os.remove(filename)

if __name__ == '__main__':
    # Add the models you want to animate
    target_models = [
        'Meta-Llama-3.1-8B-Instruct', 
        'Qwen2.5-Math-7B-Instruct',
        'gemma-3-12b-it'
    ]
    
    for model in target_models:
        generate_umap_animation(model)