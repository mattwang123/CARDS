"""
================================================================================
EXPERIMENT 12: Combined Quadrant Dynamics & Microscopic Trajectories
================================================================================
Unifies macroscopic quadrant analysis with microscopic variance.
- The Mean Trend (thick line) is calculated using ALL valid samples in the quadrant.
- The Spaghetti Lines (thin lines) are randomly sampled (N=20) for visual clarity.
- Saves the raw 11-point trajectory probabilities to CSV for future statistical tests.

Uses the frozen Proportional Unified Probe from Exp 10 to ensure the probing 
mechanism remains objective and unaffected.
================================================================================
"""

import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- CONFIGURATION ---
# The 4 representative models spanning different scales and paradigms
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'allenai/Olmo-3-7B-Think',
    'google/gemma-3-27b-it',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
]

DATASETS = ['umwp', 'treecut']

NUM_SAMPLES_FOR_LINES = 20  # Only for the background spaghetti lines
PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
PLOT_DIR = os.path.join(BASE_DIR, 'paper_plots')

# Styling dictionary for the 4 quadrants
QUAD_STYLES = {
    'Q1_Hallucination':     {'title': 'Q1: Hallucination (Output=Num, Latent=Insuff)', 'color': '#d62728'}, # Red
    'Q2_Correct_Rejection': {'title': 'Q2: Correct Reject (Output=Reject, Latent=Insuff)', 'color': '#1f77b4'}, # Blue
    'Q3_Solved_Correctly':  {'title': 'Q3: Correct Math (Output=Num, Latent=Suff)', 'color': '#2ca02c'}, # Green
    'Q4_Competence_Failure':{'title': 'Q4: Math Error (Output=Wrong Num, Latent=Suff)', 'color': '#ff7f0e'}  # Orange
}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300
})

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    np.random.seed(42) # Ensure reproducible random sampling for the spaghetti lines

    for dataset in DATASETS:
        print(f"\n{'#'*60}\nPROCESSING DATASET: {dataset.upper()}\n{'#'*60}")
        
        # Load Exp 10 metadata to find the Frozen Proportional Unified Probe
        exp10_path = os.path.join(BASE_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
        if not os.path.exists(exp10_path):
            print(f"  ! Missing {exp10_path}. Ensure Exp 10 is completed.")
            continue
            
        exp10_df = pd.read_csv(exp10_path)

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            out_plot_path = os.path.join(PLOT_DIR, f'Fig_QuadSpaghetti_{dataset}_{model_slug}.pdf')
            csv_out_path = os.path.join(BASE_DIR, 'results', f'exp12_trajectories_{dataset}_{model_slug}.csv')
            
            if os.path.exists(out_plot_path) and os.path.exists(csv_out_path):
                print(f"  [SKIP] Plot and CSV already exist for {model_slug}.")
                continue
                
            print(f"\n{'='*50}\nGenerating Plot & CSV for {model_slug}\n{'='*50}")

            model_exp10_data = exp10_df[exp10_df['Model'] == model_slug]
            if model_exp10_data.empty:
                print(f"  ! Missing unified layer metadata for {model_slug}.")
                continue
                
            unified_layer = int(model_exp10_data['Optimal_Layer'].iloc[0])
            probe_path = os.path.join(BASE_DIR, 'probes_proportional', dataset, model_slug, f"unified_probe_layer{unified_layer}.joblib")
            
            if not os.path.exists(probe_path):
                print(f"  ! Missing global unified probe for {model_slug}.")
                continue
            unified_probe = joblib.load(probe_path)

            # Load Generations and Eval Data
            eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
            gen_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json")
            if not os.path.exists(eval_path) or not os.path.exists(gen_path): 
                print(f"  ! Missing evaluation traces for {model_slug}.")
                continue
                
            with open(eval_path, 'r') as f: eval_data = json.load(f).get("data", [])
            with open(gen_path, 'r') as f: gen_data = json.load(f)

            # Gather ALL valid samples for extraction
            target_samples = []
            for g_item, e_item in zip(gen_data, eval_data):
                quad = e_item.get('epistemic_quadrant', '')
                cot_text = g_item.get('generated_response', '')
                
                # Filter out sequences that are too short to bin into 11 percentages meaningfully
                if quad in QUAD_STYLES and len(cot_text) > 50:
                    target_samples.append({
                        'sample_id': g_item.get('sample_id', 'unknown'),
                        'quadrant': quad,
                        'prompt': g_item['prompt'],
                        'cot_text': cot_text
                    })

            if not target_samples:
                print(f"  ! No valid samples found for {model_slug}. Skipping.")
                continue

            # Load Model into VRAM
            print("  -> Loading Model into VRAM...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                num_gpus = torch.cuda.device_count()
                memory_map = {0: "65GB"} if num_gpus > 0 else None
                if num_gpus > 1:
                    for i in range(1, num_gpus): memory_map[i] = "78GB"

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", max_memory=memory_map, 
                    torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                model.eval()
            except Exception as e:
                print(f"  ! Failed to load model: {e}")
                continue

            # Extraction: Run Forward Pass on ALL valid samples
            trajectories = []
            for item in tqdm(target_samples, desc="Extracting 11-Point Trajectories"):
                prompt_ids = tokenizer(item['prompt'], return_tensors="pt")['input_ids'][0]
                full_ids = tokenizer(item['prompt'] + item['cot_text'], return_tensors="pt")['input_ids'][0]
                
                p_len = prompt_ids.shape[0]
                cot_len = full_ids.shape[0] - p_len
                
                if cot_len < 10: continue
                
                target_indices = [min(p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0), full_ids.shape[0]-1) for pct in PERCENTAGES]
                inputs = tokenizer(item['prompt'] + item['cot_text'], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Extract hidden states at the exact optimal layer
                layer_states = outputs.hidden_states[unified_layer][0].to(torch.float32).cpu().numpy()
                target_states = layer_states[target_indices, :]
                
                # Probe the probabilities
                probs = unified_probe.predict_proba(target_states)[:, 1]
                
                trajectories.append({'quadrant': item['quadrant'], 'probs': probs})
                
                del outputs, layer_states, inputs
                torch.cuda.empty_cache()

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            # ==========================================
            # Save raw trajectories to CSV
            # ==========================================
            print("  -> Saving raw trajectory data to CSV...")
            csv_data = []
            for item, traj in zip(target_samples, trajectories):
                row_dict = {
                    'Dataset': dataset,
                    'Model': model_slug,
                    'Sample_ID': item['sample_id'],
                    'Quadrant': traj['quadrant']
                }
                # Unpack the 11 probabilities into separate columns
                for idx, pct in enumerate(PERCENTAGES):
                    row_dict[f'Prob_{int(pct*100)}%'] = round(float(traj['probs'][idx]), 4)
                csv_data.append(row_dict)
            
            pd.DataFrame(csv_data).to_csv(csv_out_path, index=False)
            print(f"  [SUCCESS] Saved trajectory CSV to {csv_out_path}")

            # ==========================================
            # Plotting the 2x2 Grid
            # ==========================================
            print("  -> Generating final 2x2 plots...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, sharex=True)
            axes = axes.flatten()

            for i, quad in enumerate(QUAD_STYLES.keys()):
                ax = axes[i]
                
                # Fetch ALL extracted data for this quadrant
                quad_data = [t['probs'] for t in trajectories if t['quadrant'] == quad]
                
                if not quad_data:
                    ax.set_title(f"{QUAD_STYLES[quad]['title']} (No Data)")
                    continue
                
                # 1. Plot Spaghetti Lines (Randomly sample N lines for visual clarity)
                num_to_sample = min(NUM_SAMPLES_FOR_LINES, len(quad_data))
                quad_matrix = np.array(quad_data) 
                
                # Randomly select row indices
                sampled_indices = np.random.choice(len(quad_matrix), num_to_sample, replace=False)
                lines_to_plot = quad_matrix[sampled_indices]
                
                for trace in lines_to_plot:
                    ax.plot(PERCENTAGES, trace, marker='.', alpha=0.15, linewidth=1.0, color=QUAD_STYLES[quad]['color'])
                
                # 2. Plot Mean Trend Line (Calculated using 100% of ALL valid samples)
                mean_trace = np.mean(quad_matrix, axis=0)
                ax.plot(PERCENTAGES, mean_trace, marker='s', linewidth=4.0, color='black', label=f"Mean Trend (All N={len(quad_matrix)})")
                
                # Formatting
                ax.set_title(QUAD_STYLES[quad]['title'])
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axvspan(0.9, 1.0, color='gray', alpha=0.1) # Highlight Context Aggregation zone
                ax.legend(loc='lower left' if i < 2 else 'upper left')
                
                if i >= 2: ax.set_xlabel('Reasoning Progress (% of Generated CoT)')
                if i % 2 == 0: ax.set_ylabel('Latent Prob of Insufficiency')

            plt.suptitle(f"Microscopic vs. Macroscopic Epistemic Dynamics\nModel: {model_slug} | Dataset: {dataset.upper()}", fontsize=18, y=1.02)
            plt.tight_layout()
            
            plt.savefig(out_plot_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"  [SUCCESS] Saved 2x2 Quadrant Plot to {out_plot_path}")

if __name__ == '__main__':
    main()