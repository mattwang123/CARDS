"""
================================================================================
EXPERIMENT 11: 11-Point Trajectory Extraction & Quadrant Aggregation
================================================================================
This script fixes the incomplete trajectory extraction by explicitly executing
a stable 11-point (0% to 100%, step 10%) latent probing pass.

Key Features:
1. Loads the frozen Unified Proportional Probe from Exp 10.
2. Extracts probabilities for all 11 points during CoT generation.
3. Saves raw trajectories into a dedicated 'sample_wise' folder per model.
4. Aggregates the averages split by Epistemic Quadrants (Q1-Q4).
5. Continuously updates a master 'exp11_average_trajectories.csv'.
================================================================================
"""

import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

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
PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SAMPLE_WISE_DIR = os.path.join(RESULTS_DIR, 'exp11_sample_wise')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_WISE_DIR, exist_ok=True)

MASTER_AVG_CSV = os.path.join(RESULTS_DIR, 'exp11_average_trajectories.csv')

# Only consider these 4 epistemic quadrants
VALID_QUADRANTS = ['Q1_Hallucination', 'Q2_Correct_Rejection', 'Q3_Solved_Correctly', 'Q4_Competence_Failure']

def update_master_averages(all_averages_list):
    """Safely updates the master average CSV."""
    if not all_averages_list: return
    df = pd.DataFrame(all_averages_list)
    df.to_csv(MASTER_AVG_CSV, index=False)

def main():
    print(f"\n{'='*80}\nSTARTING EXP 11: 11-POINT TRAJECTORIES & AVERAGES\n{'='*80}")

    all_averages = []

    for dataset in DATASETS:
        # Load Exp 10 metadata to find the optimal layer for the probe
        exp10_path = os.path.join(RESULTS_DIR, f'exp10_ultimate_proportional_{dataset}.csv')
        if not os.path.exists(exp10_path):
            print(f"Skipping {dataset}: Cannot find exp10 metadata.")
            continue
        exp10_df = pd.read_csv(exp10_path)

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            sample_csv_path = os.path.join(SAMPLE_WISE_DIR, f'traj_{dataset}_{model_slug}.csv')
            
            # --- RESUME LOGIC ---
            # If the sample-wise CSV already exists, just load it, compute averages, and append!
            if os.path.exists(sample_csv_path):
                print(f"  [SKIP/LOAD] {model_slug} on {dataset} already extracted. Computing averages...")
                df_sample = pd.read_csv(sample_csv_path)
                
                # Compute quadrant averages
                for quad in VALID_QUADRANTS:
                    quad_data = df_sample[df_sample['Quadrant'] == quad]
                    if quad_data.empty: continue
                    
                    avg_dict = {
                        'Dataset': dataset,
                        'Model': model_slug,
                        'Quadrant': quad,
                        'Sample_Count': len(quad_data)
                    }
                    for pct in PERCENTAGES:
                        col_name = f'Prob_{int(pct*100)}%'
                        avg_dict[col_name] = round(quad_data[col_name].mean(), 4)
                    all_averages.append(avg_dict)
                
                update_master_averages(all_averages)
                continue
            
            print(f"\n{'#'*60}\nModel: {model_slug} | Dataset: {dataset}\n{'#'*60}")
            
            # 1. Load Probe and Optimal Layer
            model_exp10_data = exp10_df[exp10_df['Model'] == model_slug]
            if model_exp10_data.empty:
                print(f"  ! Missing unified layer metadata. Skipping.")
                continue
                
            unified_layer = int(model_exp10_data['Optimal_Layer'].iloc[0])
            probe_path = os.path.join(BASE_DIR, 'probes_proportional', dataset, model_slug, f"unified_probe_layer{unified_layer}.joblib")
            
            if not os.path.exists(probe_path):
                print(f"  ! Missing unified probe for {model_slug}. Skipping.")
                continue
            unified_probe = joblib.load(probe_path)

            # 2. Load Evaluation and Generation Traces
            eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
            gen_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json")
            if not os.path.exists(eval_path) or not os.path.exists(gen_path): 
                print(f"  ! Missing eval/gen traces. Skipping.")
                continue
                
            with open(eval_path, 'r') as f: eval_data = json.load(f).get("data", [])
            with open(gen_path, 'r') as f: gen_data = json.load(f)

            # Filter valid samples
            target_samples = []
            for g_item, e_item in zip(gen_data, eval_data):
                quad = e_item.get('epistemic_quadrant', '')
                cot_text = g_item.get('generated_response', '')
                
                if quad in VALID_QUADRANTS and len(cot_text) > 20: # Ensure some generation exists
                    target_samples.append({
                        'sample_id': g_item.get('sample_id', 'unknown'),
                        'quadrant': quad,
                        'prompt': g_item['prompt'],
                        'cot_text': cot_text
                    })

            if not target_samples:
                print("  ! No valid quadrant samples found. Skipping.")
                continue

            # 3. Load Model into VRAM
            print("  -> Loading Model...")
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

            # 4. Extraction Loop
            extracted_rows = []
            for item in tqdm(target_samples, desc="Extracting 11-Pt Trajectories"):
                prompt_ids = tokenizer(item['prompt'], return_tensors="pt")['input_ids'][0]
                full_ids = tokenizer(item['prompt'] + item['cot_text'], return_tensors="pt")['input_ids'][0]
                
                p_len = prompt_ids.shape[0]
                cot_len = full_ids.shape[0] - p_len
                
                if cot_len < 10: continue
                
                target_indices = [min(p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0), full_ids.shape[0]-1) for pct in PERCENTAGES]
                inputs = tokenizer(item['prompt'] + item['cot_text'], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                layer_states = outputs.hidden_states[unified_layer][0].to(torch.float32).cpu().numpy()
                target_states = layer_states[target_indices, :]
                
                # Predict Probabilities P(Insufficient)
                probs = unified_probe.predict_proba(target_states)[:, 1]
                
                row_dict = {
                    'Dataset': dataset,
                    'Model': model_slug,
                    'Sample_ID': item['sample_id'],
                    'Quadrant': item['quadrant']
                }
                for idx, pct in enumerate(PERCENTAGES):
                    row_dict[f'Prob_{int(pct*100)}%'] = round(float(probs[idx]), 4)
                    
                extracted_rows.append(row_dict)
                
                del outputs, layer_states, inputs
                torch.cuda.empty_cache()

            # Clean memory
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            # 5. Save Sample-Wise CSV
            df_extracted = pd.DataFrame(extracted_rows)
            df_extracted.to_csv(sample_csv_path, index=False)
            print(f"  -> Saved {len(extracted_rows)} trajectories to sample_wise directory.")

            # 6. Compute Quadrant Averages & Update Master
            for quad in VALID_QUADRANTS:
                quad_data = df_extracted[df_extracted['Quadrant'] == quad]
                if quad_data.empty: continue
                
                avg_dict = {
                    'Dataset': dataset,
                    'Model': model_slug,
                    'Quadrant': quad,
                    'Sample_Count': len(quad_data)
                }
                for pct in PERCENTAGES:
                    col_name = f'Prob_{int(pct*100)}%'
                    avg_dict[col_name] = round(quad_data[col_name].mean(), 4)
                all_averages.append(avg_dict)
            
            update_master_averages(all_averages)
            print("  -> Updated master average CSV.")

    print(f"\n[SUCCESS] Exp 11 Finished! Master Averages saved to: {MASTER_AVG_CSV}")

if __name__ == '__main__':
    main()