"""
================================================================================
EXPERIMENT 7: The Awakening Curve (Granular Backwards Relative Probing)
================================================================================
Tracks the exact token-by-token epistemic awakening during the conclusion phase.
Aligns natural language tokens (e.g., "Therefore", "answer", "\boxed") with the 
Unified Probe's probability of "Insufficiency" for the last K tokens.

Production Upgrades:
- Full 21-Model Support & Both Datasets.
- Zips Original Prompts with Evaluated Traces to preserve perfect context.
- Atomic Saves & Resume Capability (Skip completed models).
- Aggressive VRAM Management for 70B+ models.
================================================================================
"""

import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
import gc
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'awakening_curves')
K_TOKENS = 30  # Look at the last 30 tokens before EOS

def run_awakening_curve():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for dataset in DATASETS:
        print(f"\n{'#'*60}\nPROCESSING DATASET: {dataset.upper()}\n{'#'*60}")
        
        # Load Exp 5 metrics to find the Unified Probe layer
        exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{dataset}.json')
        if not os.path.exists(exp5_path):
            print(f"Skipping {dataset}, missing Unified Probe data ({exp5_path})")
            continue
            
        with open(exp5_path, 'r') as f:
            exp5_data = json.load(f)

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            
            # ATOMIC RESUME: Check if this model is already processed
            out_csv = os.path.join(RESULTS_DIR, f'exp7_{dataset}_{model_slug}.csv')
            if os.path.exists(out_csv):
                print(f"  [SKIP] {model_slug} already completed for {dataset}.")
                continue
                
            print(f"\n{'='*50}\nTracking Awakening Curve: {model_slug}\n{'='*50}")

            # 1. Load BOTH Evaluated Traces and Original Generations
            eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
            gen_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json")
            
            if not os.path.exists(eval_path) or not os.path.exists(gen_path):
                print(f"  ! Missing traces or generation data for {model_slug}.")
                continue
                
            with open(eval_path, 'r') as f:
                eval_data = json.load(f).get("data", [])
                
            with open(gen_path, 'r') as f:
                gen_data = json.load(f)
                
            # Zip them together and filter for Q1 / Q2
            target_samples = []
            for g_item, e_item in zip(gen_data, eval_data):
                quadrant = e_item.get('epistemic_quadrant', '')
                if quadrant in ['Q1_Hallucination', 'Q2_Correct_Rejection']:
                    target_samples.append({
                        'prompt': g_item['prompt'],
                        'cot_text': g_item.get('generated_response', ''),
                        'epistemic_quadrant': quadrant
                    })
            
            if not target_samples:
                print(f"  ! No Q1/Q2 samples found for {model_slug}.")
                continue

            # 2. Load the Frozen Unified Probe
            if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]:
                print(f"  ! Unified Probe missing in Exp5 JSON for {model_slug}.")
                continue
                
            unified_layer = exp5_data[model_name]["unified_layer"]
            probe_path = os.path.join(BASE_DIR, 'probes', dataset, model_slug, f"unified_probe_layer{unified_layer}.joblib")
            
            if not os.path.exists(probe_path):
                print(f"  ! Probe file missing: {probe_path}")
                continue
                
            unified_probe = joblib.load(probe_path)

            # 3. Load Model and Tokenizer (With strict memory mapping for 72B models)
            print("  -> Loading weights into VRAM...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                num_gpus = torch.cuda.device_count()
                memory_map = None
                if num_gpus > 0:
                    memory_map = {0: "65GB"} # Leave headroom on GPU0 for hidden states
                    for i in range(1, num_gpus): memory_map[i] = "78GB"

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto", 
                    max_memory=memory_map,
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
                model.eval()
            except Exception as e:
                print(f"  ! Failed to load model {model_slug}: {e}")
                continue

            model_trajectories = []

            # 4. Process Samples Granularly
            for sample_idx, item in enumerate(tqdm(target_samples, desc=f"Extracting K={K_TOKENS} for {model_slug}")):
                prompt_text = item['prompt']
                cot_text = item['cot_text']
                quadrant = item['epistemic_quadrant']
                
                full_text = prompt_text + cot_text
                inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                input_ids = inputs['input_ids'][0]
                seq_len = input_ids.shape[0]
                
                # Dynamic Start Index (safeguard for short sequences)
                start_idx = max(0, seq_len - K_TOKENS)
                actual_k = seq_len - start_idx
                
                if actual_k == 0:
                    continue
                    
                # Run Forward Pass
                with torch.no_grad():
                    # CAUTION: output_hidden_states=True is very VRAM heavy for long sequences.
                    outputs = model(**inputs, output_hidden_states=True)
                    
                # Extract the specific unified layer
                layer_states = outputs.hidden_states[unified_layer][0].to(torch.float32).cpu().numpy()
                target_states = layer_states[start_idx:seq_len, :]
                
                # Predict probability of Insufficiency
                probs = unified_probe.predict_proba(target_states)[:, 1]
                
                # Decode exact tokens
                for relative_pos, (token_id, prob) in enumerate(zip(input_ids[start_idx:seq_len], probs)):
                    token_str = tokenizer.decode([token_id])
                    pos_from_end = actual_k - relative_pos # e.g., 30... 2, 1 (EOS)
                    
                    model_trajectories.append({
                        "Dataset": dataset,
                        "Model": model_slug,
                        "Sample_ID": sample_idx,
                        "Quadrant": quadrant,
                        "Pos_From_End": pos_from_end,
                        "Exact_Token": repr(token_str), # repr() captures \n spaces cleanly
                        "Insufficiency_Prob": round(float(prob), 4)
                    })
                    
                # Aggressive memory cleanup after EVERY sample
                del outputs, layer_states, target_states, inputs
                torch.cuda.empty_cache()

            # Clean up Model VRAM before moving to the next model
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            # 5. Save Model-Specific Results Atomically
            if model_trajectories:
                df_model = pd.DataFrame(model_trajectories)
                df_model.to_csv(out_csv, index=False)
                print(f"  -> Saved {len(model_trajectories)} token states to {out_csv}")
            else:
                print("  -> No valid trajectories found.")

    # 6. Master Compilation
    print("\n==================================================")
    print("COMPILING MASTER AWAKENING CURVES")
    print("==================================================")
    for dataset in DATASETS:
        all_dfs = []
        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            out_csv = os.path.join(RESULTS_DIR, f'exp7_{dataset}_{model_slug}.csv')
            if os.path.exists(out_csv):
                all_dfs.append(pd.read_csv(out_csv))
                
        if all_dfs:
            master_df = pd.concat(all_dfs, ignore_index=True)
            master_csv = os.path.join(BASE_DIR, 'results', f'exp7_master_awakening_{dataset}.csv')
            master_df.to_csv(master_csv, index=False)
            print(f"[SUCCESS] Compiled master file for {dataset.upper()}: {master_csv}")

if __name__ == '__main__':
    run_awakening_curve()