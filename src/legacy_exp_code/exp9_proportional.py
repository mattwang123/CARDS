"""
================================================================================
EXPERIMENT 9: Proportional Quadrant Dynamics & Visualization
================================================================================
Evaluates the model's latent awareness at strictly normalized sequence percentages 
(0%, 10%, ..., 100% of CoT length). 
Eliminates length-bias and integrates Epistemic Quadrants to reveal the 
exact trajectory of the Overthinking Trap and the EOS Awakening.
================================================================================
"""

import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    
    # --- LARGE SCALE (14B - 32B) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    
    # --- MASSIVE SCALE (70B+) ---
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]

DATASETS = ['umwp', 'treecut']
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'proportional_curves')
PLOT_DIR = os.path.join(BASE_DIR, 'paper_plots')
PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# --- PLOTTING STYLE ---
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300
})

def run_extraction():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for dataset in DATASETS:
        print(f"\n{'#'*60}\nEXTRACTING: {dataset.upper()}\n{'#'*60}")
        exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{dataset}.json')
        if not os.path.exists(exp5_path):
            print(f"Skipping {dataset}: exp5_global_dynamics_{dataset}.json not found.")
            continue
            
        with open(exp5_path, 'r') as f: 
            exp5_data = json.load(f)

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            out_csv = os.path.join(RESULTS_DIR, f'exp9_{dataset}_{model_slug}.csv')
            
            if os.path.exists(out_csv):
                print(f"  [SKIP] {model_slug} extraction already done.")
                continue
                
            print(f"\n{'='*50}\nExtracting Proportional Dynamics: {model_slug}\n{'='*50}")

            # 1. Zip Prompt and Evaluated Data
            eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
            gen_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json")
            if not os.path.exists(eval_path) or not os.path.exists(gen_path): 
                print(f"  ! Missing traces/generations for {model_slug}. Skipping.")
                continue
                
            with open(eval_path, 'r') as f: eval_data = json.load(f).get("data", [])
            with open(gen_path, 'r') as f: gen_data = json.load(f)
                
            target_samples = []
            for g_item, e_item in zip(gen_data, eval_data):
                quadrant = e_item.get('epistemic_quadrant', '')
                if quadrant: # Keep all quadrants
                    target_samples.append({
                        'sample_id': g_item.get('sample_id', 0),
                        'prompt': g_item['prompt'],
                        'cot_text': g_item.get('generated_response', ''),
                        'epistemic_quadrant': quadrant
                    })

            if not target_samples:
                continue

            # 2. Load the Frozen Unified Probe
            if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]: 
                print(f"  ! Missing unified layer info for {model_slug}. Skipping.")
                continue
                
            unified_layer = exp5_data[model_name]["unified_layer"]
            probe_path = os.path.join(BASE_DIR, 'probes', dataset, model_slug, f"unified_probe_layer{unified_layer}.joblib")
            
            if not os.path.exists(probe_path): 
                print(f"  ! Missing unified probe file for {model_slug}. Skipping.")
                continue
                
            unified_probe = joblib.load(probe_path)

            # 3. Load Model and Tokenizer securely
            print("  -> Loading weights into VRAM...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                num_gpus = torch.cuda.device_count()
                memory_map = None
                if num_gpus > 0:
                    memory_map = {0: "65GB"} # Leave headroom on GPU0 for hidden states
                    for i in range(1, num_gpus): memory_map[i] = "78GB"

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", max_memory=memory_map,
                    torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                model.eval()
            except Exception as e:
                print(f"  ! Failed to load model {model_slug}: {e}")
                continue

            model_results = []

            # 4. Extract Percentages
            for sample_idx, item in enumerate(tqdm(target_samples, desc=f"Evaluating Percentages for {model_slug}")):
                prompt_text = item['prompt']
                cot_text = item['cot_text']
                quadrant = item['epistemic_quadrant']
                
                # Check lengths
                prompt_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'][0]
                full_ids = tokenizer(prompt_text + cot_text, return_tensors="pt")['input_ids'][0]
                
                p_len = prompt_ids.shape[0]
                total_len = full_ids.shape[0]
                cot_len = total_len - p_len
                
                # If sequence is too short to bin meaningfully, skip
                if cot_len < 10: 
                    continue
                
                # Calculate exact indices
                target_indices = []
                for pct in PERCENTAGES:
                    idx = p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0)
                    target_indices.append(min(idx, total_len - 1))
                
                inputs = tokenizer(prompt_text + cot_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    
                layer_states = outputs.hidden_states[unified_layer][0].to(torch.float32).cpu().numpy()
                target_states = layer_states[target_indices, :]
                probs = unified_probe.predict_proba(target_states)[:, 1]
                
                for pct, prob in zip(PERCENTAGES, probs):
                    model_results.append({
                        "Model": model_slug,
                        "Sample_ID": sample_idx,
                        "Quadrant": quadrant,
                        "Sequence_Percentage": int(pct * 100),
                        "Insufficiency_Prob": round(float(prob), 4)
                    })
                    
                # Aggressive Memory Cleanup
                del outputs, layer_states, target_states, inputs
                torch.cuda.empty_cache()

            # Clean VRAM per model
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            # Save Model CSV
            if model_results:
                df_model = pd.DataFrame(model_results)
                df_model.to_csv(out_csv, index=False)
                print(f"  -> Saved proportional trace to {out_csv}")
            else:
                print(f"  -> No valid trajectories found for {model_slug}")

def plot_proportional_dynamics():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("\n==================================================")
    print("PLOTTING PROPORTIONAL DYNAMICS")
    print("==================================================")
    
    for dataset in DATASETS:
        all_dfs = []
        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            out_csv = os.path.join(RESULTS_DIR, f'exp9_{dataset}_{model_slug}.csv')
            if os.path.exists(out_csv): 
                all_dfs.append(pd.read_csv(out_csv))
                
        if not all_dfs: 
            print(f"No proportional data found for {dataset}. Run extraction first.")
            continue
            
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save the master compilation for safety
        master_csv = os.path.join(BASE_DIR, 'results', f'exp9_master_proportional_{dataset}.csv')
        master_df.to_csv(master_csv, index=False)
        
        # Aggregate across all models
        agg_df = master_df.groupby(['Quadrant', 'Sequence_Percentage'])['Insufficiency_Prob'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {
            'Q1_Hallucination': '#d62728',       # Red
            'Q2_Correct_Rejection': '#1f77b4',   # Blue
            'Q3_Solved_Correctly': '#2ca02c',    # Green
            'Q4_Competence_Failure': '#ff7f0e'   # Orange
        }
        
        labels = {
            'Q1_Hallucination': 'Q1: Hallucination (Insufficient)',
            'Q2_Correct_Rejection': 'Q2: Correct Reject (Insufficient)',
            'Q3_Solved_Correctly': 'Q3: Correct Math (Sufficient)',
            'Q4_Competence_Failure': 'Q4: Math Error (Sufficient)'
        }

        for quad in ['Q1_Hallucination', 'Q2_Correct_Rejection', 'Q3_Solved_Correctly', 'Q4_Competence_Failure']:
            q_data = agg_df[agg_df['Quadrant'] == quad]
            if not q_data.empty:
                ax.plot(q_data['Sequence_Percentage'], q_data['Insufficiency_Prob'], 
                        marker='o', linewidth=3.0, color=colors[quad], label=labels[quad])

        ax.set_xlabel('Reasoning Progress (% of Generated CoT Length)')
        ax.set_ylabel('Latent Probability of Insufficiency')
        ax.set_title(f'The Epistemic Trajectory across 21 Models ({dataset.upper()})')
        ax.set_xticks(range(0, 110, 10))
        ax.set_xticklabels([f"{x}%" for x in range(0, 110, 10)])
        
        # Shading the "Awakening Zone"
        ax.axvspan(90, 100, color='gray', alpha=0.1)
        ax.text(95, ax.get_ylim()[0] + 0.1, 'Context\nAggregation\n(EOS)', ha='center', va='center', rotation=90, color='#666666')

        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOT_DIR, f'Fig_Proportional_Quadrants_{dataset}.pdf')
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Saved master plot to {plot_path}")

if __name__ == '__main__':
    run_extraction()
    plot_proportional_dynamics()