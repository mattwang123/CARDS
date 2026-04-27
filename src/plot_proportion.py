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

# ANSI colors for terminal debug readability
COLOR_RESET = "\033[0m"
COLOR_HEADER = "\033[95m"
COLOR_META = "\033[93m"
COLOR_COT = "\033[96m"

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
    # --- SMALL/MEDIUM SCALE (~1.5B - 4B) ---
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',

    # --- MEDIUM/LARGE SCALE (~7B - 9B) ---
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think',
    'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',

    # --- LARGE SCALE (14B - 32B) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it',
    'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',

    # --- MASSIVE SCALE (70B+) ---
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]
DATASET = 'umwp'
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]
QUADRANT_STYLES = {
    'Q1_Hallucination': {
        'color': '#d62728',
        'title': 'Epistemic Wandering and Awakening in Hallucinations (Q1)'
    },
    'Q2_Correct_Rejection': {
        'color': '#2ca02c',
        'title': 'Epistemic Dynamics in Correct Rejections (Q2)'
    },
    'Q3_True_Positive': {
        'color': '#1f77b4',
        'title': 'Epistemic Dynamics in True Positives (Q3)'
    },
    'Q4_True_Negative': {
        'color': '#9467bd',
        'title': 'Epistemic Dynamics in True Negatives (Q4)'
    },
}

def get_available_model_slugs(dataset):
    """
    Discover available model slugs from embedding folder names.
    Prefer EXPORT_BASE, but also support local in-repo experiment_result path.
    """
    candidates = [
        os.path.join(BASE_DIR, 'embeddings', dataset),
        os.path.join('experiment_result', 'exp_temporal_new', 'embeddings', dataset),
    ]
    for emb_dir in candidates:
        if os.path.isdir(emb_dir):
            return sorted(
                d for d in os.listdir(emb_dir)
                if os.path.isdir(os.path.join(emb_dir, d))
            )
    return []

def run_relative_scatter():
    os.makedirs('paper_plots/exp7_awakening', exist_ok=True)
    
    # Load Exp 5 data for Unified Layers
    exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{DATASET}.json')
    if not os.path.exists(exp5_path):
        print("Missing Exp 5 json.")
        return
    with open(exp5_path, 'r') as f:
        exp5_data = json.load(f)

    available_slugs = set(get_available_model_slugs(DATASET))
    if available_slugs:
        models_to_run = [m for m in MODELS if m.split('/')[-1] in available_slugs]
        skipped = [m for m in MODELS if m.split('/')[-1] not in available_slugs]
        print(f"Found {len(available_slugs)} embedding model folders for dataset '{DATASET}'.")
        if skipped:
            print(f"Skipping {len(skipped)} models not present as embedding folders.")
    else:
        print(f"Could not find embedding folders for dataset '{DATASET}'. Using MODELS as-is.")
        models_to_run = MODELS

    for model_name in models_to_run:
        model_slug = model_name.split('/')[-1]
        print(f"\nProcessing {model_slug}...")

        if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]:
            continue
            
        unified_layer = exp5_data[model_name]["unified_layer"]
        probe_path = os.path.join(BASE_DIR, 'probes', DATASET, model_slug, f"unified_probe_layer{unified_layer}.joblib")
        
        if not os.path.exists(probe_path):
            continue
            
        unified_probe = joblib.load(probe_path)

        # 1. Load evaluated traces for quadrant labels
        eval_path = os.path.join(
            EXPORT_BASE,
            f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{DATASET}_evaluated_traces.json"
        )
        with open(eval_path, 'r') as f:
            eval_data = json.load(f).get("data", [])

        # 2. Load original generation traces for CoT length (must match embedding provenance)
        gen_path = os.path.join(
            EXPORT_BASE,
            f"experiments/dynamic_tracking_test/math/{model_slug}/{DATASET}_cot_generations.json"
        )
        if not os.path.exists(gen_path):
            print(f"Missing generation trace file: {gen_path}")
            continue
        with open(gen_path, 'r') as f:
            gen_data = json.load(f)

        # Load tokenizer to calculate actual generation length
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            continue

        lengths = []
        quadrants = []

        if len(gen_data) != len(eval_data):
            print(f"Length mismatch: gen={len(gen_data)} vs eval={len(eval_data)}; skipping model.")
            continue

        print("Calculating CoT lengths from generated_response...")
        for i in tqdm(range(len(gen_data)), desc=f"Lengths ({model_slug})", leave=False):
            gen_item = gen_data[i]
            eval_item = eval_data[i]
            text = gen_item.get('generated_response', '')
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
            quadrants.append(eval_item['epistemic_quadrant'])
            
        lengths = np.array(lengths)
        quadrants = np.array(quadrants)

        # Debug: show concrete examples from the exact CoT text used for normalization.
        print(f"{COLOR_HEADER}Sample CoT examples used for length normalization:{COLOR_RESET}")
        sample_indices = [0, len(gen_data) // 2, len(gen_data) - 1]
        for idx in sample_indices:
            text = gen_data[idx].get('generated_response', '')
            print(
                f"{COLOR_META}  idx={idx:<4} len={lengths[idx]:<4} "
                f"quadrant={quadrants[idx]}{COLOR_RESET}"
            )
            print(f"{COLOR_COT}  full_cot={text!r}{COLOR_RESET}")

        records = []

        # 3. Extract Active Probabilities for fixed timesteps
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
                        'Timestep': t,
                        'Probability': probs_t[i],
                        'Quadrant': quadrants[i],
                        'IsEOS': False
                    })

        # 4. Extract EOS Probabilities (Proportion = 1.0)
        eos_path = os.path.join(BASE_DIR, 'embeddings', DATASET, model_slug, "t_eos_test.npy")
        if os.path.exists(eos_path):
            X_eos = np.load(eos_path)[:, unified_layer, :]
            probs_eos = unified_probe.predict_proba(X_eos)[:, 1]
            for i in range(len(lengths)):
                records.append({
                    'Proportion': 1.0,
                    'Timestep': np.nan,
                    'Probability': probs_eos[i],
                    'Quadrant': quadrants[i],
                    'IsEOS': True
                })

        df = pd.DataFrame(records)
        
        # 4. PLOTTING: Generate one plot per quadrant (Q1-Q4)
        for quadrant, style in QUADRANT_STYLES.items():
            df_q = df[df['Quadrant'] == quadrant].copy()
            if df_q.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 6))

            # A. Scatter points
            sns.scatterplot(
                data=df_q,
                x='Proportion',
                y='Probability',
                alpha=0.15,
                color=style['color'],
                edgecolor=None,
                s=20,
                ax=ax
            )

            # B. Binned trend line
            df_q['Bin'] = pd.cut(df_q['Proportion'], bins=np.linspace(0, 1.01, 15), right=False)
            bin_means = df_q.groupby('Bin').mean(numeric_only=True).dropna()

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

            # C. Faint cumulative count curve on right-side y-axis
            cdf_x = np.sort(df_q['Proportion'].to_numpy())
            cdf_y = np.arange(1, len(cdf_x) + 1)
            ax2 = ax.twinx()
            ax2.plot(
                cdf_x,
                cdf_y,
                color='#1f77b4',
                linewidth=2.0,
                alpha=0.35,
                linestyle='-',
                zorder=5,
                label='Cumulative Count (points left of x)'
            )
            ax2.set_ylabel('Cumulative Number of Points', color='#1f77b4')
            ax2.tick_params(axis='y', colors='#1f77b4')
            ax2.set_ylim(0, len(cdf_x) * 1.02)

            ax.set_xlim(-0.02, 1.05)
            ax.set_ylim(-0.05, 1.05)

            # Formatting X-axis to percentages
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0% (Start)', '25%', '50%', '75%', '100% (EOS)'])

            ax.set_xlabel('Reasoning Progression (Relative to Total CoT Length)')
            ax.set_ylabel('Latent Probability of "Insufficient"')
            ax.set_title(f"{style['title']}\nModel: {model_slug}", pad=15)

            # Add a decision boundary line
            ax.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)
            ax.text(0.02, 0.52, 'Decision Boundary', color='gray', fontsize=10)

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc='lower center')

            plt.tight_layout()
            quadrant_short = quadrant.split('_')[0]
            out_path = f"paper_plots/exp7_awakening/Fig_Scatter_{quadrant_short}_{model_slug.replace('/', '_')}.pdf"
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {out_path}")

            # 5. Absolute-timestep view (exclude EOS), with mean evaluated at each timestep.
            df_q_abs = df_q[df_q['IsEOS'] == False].copy()
            if df_q_abs.empty:
                continue

            fig_abs, ax_abs = plt.subplots(figsize=(9, 6))

            sns.scatterplot(
                data=df_q_abs,
                x='Timestep',
                y='Probability',
                alpha=0.15,
                color=style['color'],
                edgecolor=None,
                s=20,
                ax=ax_abs
            )

            timestep_means = (
                df_q_abs.groupby('Timestep', as_index=False)['Probability']
                .mean()
                .sort_values('Timestep')
            )
            ax_abs.plot(
                timestep_means['Timestep'],
                timestep_means['Probability'],
                color='#000000',
                linewidth=3.5,
                marker='D',
                markersize=8,
                zorder=10,
                label='Average at each timestep'
            )

            ax_abs.set_xlim(min(TIMESTEPS) - 2, max(TIMESTEPS) + 8)
            ax_abs.set_ylim(-0.05, 1.05)
            ax_abs.set_xticks(TIMESTEPS)
            ax_abs.set_xlabel('Reasoning Progression (Absolute Timestep)')
            ax_abs.set_ylabel('Latent Probability of "Insufficient"')
            ax_abs.set_title(
                f"{style['title']} (Absolute Timesteps, EOS Removed)\nModel: {model_slug}",
                pad=15
            )
            ax_abs.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)
            ax_abs.text(min(TIMESTEPS) + 2, 0.52, 'Decision Boundary', color='gray', fontsize=10)
            ax_abs.legend(frameon=False, loc='lower center')

            plt.tight_layout()
            out_path_abs = (
                f"paper_plots/exp7_awakening/"
                f"Fig_Scatter_{quadrant_short}_{model_slug.replace('/', '_')}_absolute_t.pdf"
            )
            plt.savefig(out_path_abs, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {out_path_abs}")

if __name__ == '__main__':
    run_relative_scatter()