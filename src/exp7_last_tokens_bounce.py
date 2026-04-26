"""
================================================================================
EXPERIMENT 7: The Awakening Curve (Granular Backwards Relative Probing)
================================================================================
Tracks the exact token-by-token epistemic awakening during the conclusion phase.
Aligns natural language tokens (e.g., "Therefore", "answer", "\boxed") with the 
Unified Probe's probability of "Insufficiency" for the last K tokens.
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

# --- CONFIGURATION ---
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'google/gemma-3-4b-it'
] # Start with smaller models for rapid testing
DATASET = 'umwp'
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
K_TOKENS = 25  # Look at the last 25 tokens before EOS

def run_awakening_curve():
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    
    # Load Exp 5 metrics to find the Unified Probe layer
    exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{DATASET}.json')
    with open(exp5_path, 'r') as f:
        exp5_data = json.load(f)

    all_trajectories = []

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\n{'='*50}\nTracking Awakening Curve for: {model_slug}\n{'='*50}")

        # 1. Load the Evaluated Traces (We only want Q1 and Q2)
        eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{DATASET}_evaluated_traces.json")
        if not os.path.exists(eval_path): continue
            
        with open(eval_path, 'r') as f:
            eval_data = json.load(f).get("data", [])
            
        # Filter for Q1 (Hallucination) and Q2 (Correct Rejection)
        target_samples = [d for d in eval_data if d['epistemic_quadrant'] in ['Q1_Hallucination', 'Q2_Correct_Rejection']]
        
        # 2. Load the Frozen Unified Probe
        if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]:
            print(f"Skipping {model_slug}, Unified Probe data missing.")
            continue
            
        unified_layer = exp5_data[model_name]["unified_layer"]
        probe_path = os.path.join(BASE_DIR, 'probes', DATASET, model_slug, f"unified_probe_layer{unified_layer}.joblib")
        
        if not os.path.exists(probe_path):
            print(f"Probe file missing: {probe_path}")
            continue
            
        unified_probe = joblib.load(probe_path)

        # 3. Load Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        model.eval()

        # 4. Process Samples Granularly
        for sample_idx, item in enumerate(tqdm(target_samples[:100], desc="Extracting Last K Tokens")): # Limit to 100 for speed
            prompt_text = item['prompt']
            cot_text = item.get('full_cot_text', item.get('extracted_raw_text', ''))
            quadrant = item['epistemic_quadrant']
            
            full_text = prompt_text + cot_text
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            input_ids = inputs['input_ids'][0]
            seq_len = input_ids.shape[0]
            
            # Ensure sequence is long enough
            if seq_len < K_TOKENS + 10:
                continue
                
            # Run Forward Pass to get Hidden States
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            # Extract the specific unified layer
            # Shape: [batch, seq_len, hidden_dim]
            layer_states = outputs.hidden_states[unified_layer][0].cpu().numpy()
            
            # 5. Token-by-Token Alignment (The Core Granularity)
            # We look at indices from (seq_len - K_TOKENS) up to seq_len - 1
            start_idx = seq_len - K_TOKENS
            
            # Get the exact sub-sequence of hidden states
            target_states = layer_states[start_idx:seq_len, :]
            
            # Predict probability of Insufficiency using Unified Probe
            probs = unified_probe.predict_proba(target_states)[:, 1]
            
            # Decode the exact text tokens to see WHAT triggered the awakening
            decoded_tokens = [tokenizer.decode(t) for t in input_ids[start_idx:seq_len]]
            
            # Store the trajectory
            for relative_pos, (token_str, prob) in enumerate(zip(decoded_tokens, probs)):
                pos_from_end = K_TOKENS - relative_pos # e.g., 25, 24, 23 ... 1 (EOS)
                
                all_trajectories.append({
                    "Model": model_slug,
                    "Sample_ID": sample_idx,
                    "Quadrant": quadrant,
                    "Pos_From_End": pos_from_end,
                    "Exact_Token": token_str.replace('\n', '\\n'), # Clean up formatting
                    "Insufficiency_Prob": round(float(prob), 4)
                })
                
        # Clean up VRAM
        del model
        torch.cuda.empty_cache()

    # 6. Save Granular Results
    df = pd.DataFrame(all_trajectories)
    csv_out = os.path.join(BASE_DIR, 'results', f'exp7_awakening_curve_{DATASET}.csv')
    df.to_csv(csv_out, index=False)
    print(f"\n[SUCCESS] Granular awakening trajectories saved to: {csv_out}")

if __name__ == '__main__':
    run_awakening_curve()