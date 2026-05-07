"""
================================================================================
EXPERIMENT 7: Relative Epistemic Awakening (Scatter Plot)
================================================================================
Maps fixed-timestep embeddings (t=0, 2, 4...) to relative generation proportions 
(t / total_length) and plots the Unified Probe's predicted probability over time.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from transformers import AutoTokenizer
from tqdm import tqdm

# =============================================================================
# NATURE-STYLE PLOT CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,   
    'axes.spines.right': False, 
    'figure.dpi': 300
})

# --- CONFIGURATION ---
MODELS = [
    'Qwen/Qwen2.5-72B-Instruct',  # Can change to any flagship model you ran
    'Qwen/Qwen2.5-Math-7B'
]
DATASET = 'umwp'
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]

def run_relative_scatter():
    os.makedirs('paper_plots/exp7_awakening', exist_ok=True)
    
    # Load Exp 5 data for Unified Layers
    exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{DATASET}.json')
    if not os.path.exists(exp5_path):
        print("Missing Exp 5 json.")
        return
    with open(exp5_path, 'r') as f:
        exp5_data = json.load(f)

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\nProcessing {model_slug}...")

        if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]:
            continue
            
        unified_layer = exp5_data[model_name]["unified_layer"]
        probe_path = os.path.join(BASE_DIR, 'probes', DATASET, model_slug, f"unified_probe_layer{unified_layer}.joblib")
        
        if not os.path.exists(probe_path):
            continue
            
        unified_probe = joblib.load(probe_path)

        # 1. Load Traces to calculate exact CoT token lengths
        eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{DATASET}_evaluated_traces.json")
        with open(eval_path, 'r') as f:
            eval_data = json.load(f).get("data", [])

        # Load tokenizer to calculate actual generation length
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            continue

        lengths = []
        quadrants = []
        
        print("Calculating CoT lengths...")
        for item in tqdm(eval_data):
            text = item.get('full_cot_text', item.get('extracted_raw_text', ''))
            # Approximate or exact token count of the generated reasoning
            lengths.append(len(tokenizer.encode(text)))
            quadrants.append(item['epistemic_quadrant'])
            
        lengths = np.array(lengths)
        quadrants = np.array(quadrants)

        records = []

        # 2. Extract Active Probabilities for fixed timesteps
        print("Mapping temporal embeddings to relative proportions...")
        for t in TIMESTEPS:
            emb_path = os.path.join(BASE_DIR, 'embeddings', DATASET, model_slug, f"t_{t}_test.npy")
            if not os.path.exists(emb_path): continue
                
            X_t = np.load(emb_path)[:, unified_layer, :]
            probs_t = unified_probe.predict_proba(X_t)[:, 1] # Probability of Insufficiency
            
            for i in range(len(lengths)):
                # ONLY include active tokens (t must be less than total generated length)
                if lengths[i] > 0 and t <= lengths[i]:
                    proportion = t / lengths[i]
                    records.append({
                        'Proportion': proportion,
                        'Probability': probs_t[i],
                        'Quadrant': quadrants[i]
                    })

        # 3. Extract EOS Probabilities (Proportion = 1.0)
        eos_path = os.path.join(BASE_DIR, 'embeddings', DATASET, model_slug, "t_eos_test.npy")
        if os.path.exists(eos_path):
            X_eos = np.load(eos_path)[:, unified_layer, :]
            probs_eos = unified_probe.predict_proba(X_eos)[:, 1]
            for i in range(len(lengths)):
                records.append({
                    'Proportion': 1.0,
                    'Probability': probs_eos[i],
                    'Quadrant': quadrants[i]
                })

        df = pd.DataFrame(records)
        
        # 4. PLOTTING: Focus on Q1 (Hallucinations) to show the Tragic Trap
        # You can change to 'Q2_Correct_Rejection' to see the difference
        df_q1 = df[df['Quadrant'] == 'Q1_Hallucination'].copy()
        
        if df_q1.empty: continue

        fig, ax = plt.subplots(figsize=(9, 6))

        # A. The Scatter Plot (Jittered slightly on X-axis to see density)
        sns.scatterplot(
            data=df_q1, 
            x='Proportion', 
            y='Probability', 
            alpha=0.15,       # High transparency to show density
            color='#d62728',  # Red for Q1 Hallucination
            edgecolor=None,
            s=20,
            ax=ax
        )

        # B. The Binned Trend Line (The Awakening Curve)
        # Create 10 bins across the proportion [0, 1]
        df_q1['Bin'] = pd.cut(df_q1['Proportion'], bins=np.linspace(0, 1.01, 15), right=False)
        # Get the mean proportion and probability for each bin
        bin_means = df_q1.groupby('Bin').mean(numeric_only=True).dropna()

        ax.plot(
            bin_means['Proportion'], 
            bin_means['Probability'], 
            color='#000000', 
            linewidth=3.5, 
            marker='D',
            markersize=8,
            zorder=10,
            label='Average Trend (Awakening Curve)'
        )

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Formatting X-axis to percentages
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0% (Start)', '25%', '50%', '75%', '100% (EOS)'])
        
        ax.set_xlabel('Reasoning Progression (Relative to Total CoT Length)')
        ax.set_ylabel('Latent Probability of "Insufficient"')
        ax.set_title(f'Epistemic Wandering and Awakening in Hallucinations (Q1)\nModel: {model_slug}', pad=15)
        
        # Add a decision boundary line
        ax.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)
        ax.text(0.02, 0.52, 'Decision Boundary', color='gray', fontsize=10)

        ax.legend(frameon=False, loc='lower center')
        
        plt.tight_layout()
        out_path = f"paper_plots/exp7_awakening/Fig_Scatter_Q1_{model_slug.replace('/', '_')}.pdf"
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    run_relative_scatter()