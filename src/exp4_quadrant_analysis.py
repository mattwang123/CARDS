"""
================================================================================
EXPERIMENT 4: Quadrant Dynamics & Compute Waste (Rigorous & Exportable)
================================================================================
Leverages frozen, pre-trained probes to track Signal Death per Epistemic Quadrant.
- Calculates exact Token Compute Waste for Q1 (Hallucinations).
- Calculates Active-Only Mean Probability of Insufficiency to prevent EOS leakage.
- Integrates the absolute 'EOS' timestep from Experiment 5.
- Exports a structured CSV mapping Layer, Probabilities, and Survival Rates.
================================================================================
"""

import json
import os
import numpy as np
import pandas as pd
import argparse
import joblib
from transformers import AutoTokenizer
from tqdm import tqdm

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

TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 'eos']

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def load_evaluated_data(model_name, dataset):
    """Loads the quadrants and text from Exp 2 Evaluated Traces (Test Set)."""
    model_slug = model_name.split('/')[-1]
    path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
    
    if not os.path.exists(path):
        return None, None, None, None
        
    with open(path, 'r') as f:
        payload = json.load(f)
        data = payload.get("data", [])
        
    labels = []
    quadrants = []
    texts = []
    prompts = []
    
    for item in data:
        labels.append(1 if not item['is_sufficient'] else 0)
        quadrants.append(item['epistemic_quadrant'])
        texts.append(item.get('full_cot_text', item.get('extracted_raw_text', '')))
        prompts.append(item.get('prompt', ''))
        
    return data, np.array(labels), np.array(quadrants), texts, prompts

def calculate_compute_waste(model_name, quadrants, texts):
    """Calculates the exact token waste for Q1 hallucinations."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except:
        return 0, 0, 0
        
    wasted_tokens = 0
    q1_count = 0
    
    for q, text in zip(quadrants, texts):
        if q == 'Q1_Hallucination':
            q1_count += 1
            if text:
                tokens = tokenizer.encode(text)
                wasted_tokens += len(tokens)
            
    avg_waste = (wasted_tokens / q1_count) if q1_count > 0 else 0
    if q1_count > 0:
        print(f"   -> Wasted Compute (Q1): {wasted_tokens:,} tokens across {q1_count} hallucinations.")
        print(f"   -> Average Waste per Hallucination: {avg_waste:.1f} tokens.")
        
    return wasted_tokens, q1_count, avg_waste

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='treecut')
    args = parser.parse_args()

    print(f"=== Running Quadrant Dynamics for {args.dataset.upper()} ===\n")
    
    master_results_path = os.path.join(RESULTS_DIR, f"final_momentum_{args.dataset}.json")
    exp5_results_path = os.path.join(RESULTS_DIR, f"exp5_global_dynamics_{args.dataset}.json")
    
    if not os.path.exists(master_results_path):
        print(f"Error: Run Exp 3 first. Missing {master_results_path}")
        return
        
    with open(master_results_path, 'r') as f:
        master_results = json.load(f)
        
    exp5_results = {}
    if os.path.exists(exp5_results_path):
        with open(exp5_results_path, 'r') as f:
            exp5_results = json.load(f)
    
    all_export_data = []

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        # 1. Load the Test Set behavioral data
        data, labels, quadrants, texts, prompts = load_evaluated_data(model_name, args.dataset)
        if data is None: continue
            
        # 2. Compute Sequence Lengths for Survival Masking
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except:
            continue
            
        lengths = []
        for prompt, text in zip(prompts, texts):
            prompt_len = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
            full_text = prompt + text
            full_len = tokenizer(full_text, return_tensors="pt")['input_ids'].shape[1]
            lengths.append(full_len - prompt_len)
        lengths = np.array(lengths)
            
        # 3. Compute Waste Analysis
        print("[1] Compute Waste Analysis")
        total_waste, q1_count, avg_waste = calculate_compute_waste(model_name, quadrants, texts)
        
        # 4. Quadrant-Specific Temporal Tracking
        print("\n[2] Quadrant-Specific Temporal Tracking (Active-Only Mean Probability)")
        
        model_emb_dir = os.path.join(BASE_DIR, 'embeddings', args.dataset, model_slug)
        probes_dir = os.path.join(BASE_DIR, 'probes', args.dataset, model_slug)
        
        for t in TIMESTEPS:
            try:
                if t == 'eos':
                    emb_path_test = os.path.join(model_emb_dir, "t_eos_test.npy")
                    best_layer = exp5_results[model_name]["eos_layer"]
                    probe_path = os.path.join(probes_dir, f"eos_probe_layer{best_layer}.joblib")
                    # For EOS, survival is 100% (all sequences have an end)
                    surviving_mask = np.ones(len(lengths), dtype=bool) 
                else:
                    emb_path_test = os.path.join(model_emb_dir, f"t_{t}_test.npy")
                    best_layer = master_results[model_name][f"t_{t}"]["best_layer"]
                    probe_path = os.path.join(probes_dir, f"best_probe_t{t}_layer{best_layer}.joblib")
                    # Active-only survival mask
                    surviving_mask = (lengths >= t)
            except KeyError:
                continue

            if not (os.path.exists(emb_path_test) and os.path.exists(probe_path)):
                continue
                
            X_test_all = np.load(emb_path_test)
            probe = joblib.load(probe_path)
            
            X_test_layer = X_test_all[:, best_layer, :]
            
            if len(X_test_layer) != len(quadrants):
                print(f"   -> Data mismatch! Embeddings: {len(X_test_layer)}, Quadrants: {len(quadrants)}")
                break
            
            # Predict probability for the whole set, then we mask
            probs = probe.predict_proba(X_test_layer)[:, 1]
            
            # Define masks
            q1_mask = (quadrants == 'Q1_Hallucination')
            q2_mask = (quadrants == 'Q2_Correct_Rejection')
            q3_mask = (quadrants == 'Q3_Solved_Correctly')
            q4_mask = (quadrants == 'Q4_Competence_Failure')
            
            def get_active_stats(q_mask, surv_mask):
                active_q_mask = q_mask & surv_mask
                active_count = np.sum(active_q_mask)
                mean_prob = np.mean(probs[active_q_mask]) if active_count > 0 else None
                return mean_prob, active_count

            p_q1, n_q1 = get_active_stats(q1_mask, surviving_mask)
            p_q2, n_q2 = get_active_stats(q2_mask, surviving_mask)
            p_q3, n_q3 = get_active_stats(q3_mask, surviving_mask)
            p_q4, n_q4 = get_active_stats(q4_mask, surviving_mask)
            
            # Formatting for print
            def fmt(val): return f"{val:.3f}" if val is not None else " N/A "
            print(f"   t={str(t):<3} | Lyr: {best_layer:<2} | Q1: {fmt(p_q1)} (n={n_q1:<3}) | Q2: {fmt(p_q2)} (n={n_q2:<3}) | Q3: {fmt(p_q3)} (n={n_q3:<3}) | Q4: {fmt(p_q4)} (n={n_q4:<3})")

            # Collect for Export
            all_export_data.append({
                "Dataset": args.dataset,
                "Model": model_slug,
                "Timestep": str(t),
                "Best_Layer": int(best_layer),
                "Q1_Total_Compute_Waste": int(total_waste),
                "Q1_Avg_Waste_Per_Hallucination": float(round(avg_waste, 1)),
                "Q1_Active_Prob": float(p_q1) if p_q1 is not None else None,
                "Q1_Surviving_N": int(n_q1),
                "Q2_Active_Prob": float(p_q2) if p_q2 is not None else None,
                "Q2_Surviving_N": int(n_q2),
                "Q3_Active_Prob": float(p_q3) if p_q3 is not None else None,
                "Q3_Surviving_N": int(n_q3),
                "Q4_Active_Prob": float(p_q4) if p_q4 is not None else None,
                "Q4_Surviving_N": int(n_q4)
            })

    # Save outputs
    df = pd.DataFrame(all_export_data)
    csv_out = os.path.join(RESULTS_DIR, f'exp4_quadrant_dynamics_{args.dataset}.csv')
    df.to_csv(csv_out, index=False)
    
    # Save a nested JSON
    json_dict = {}
    for row in all_export_data:
        mod = row['Model']
        if mod not in json_dict: json_dict[mod] = {}
        json_dict[mod][f"t_{row['Timestep']}"] = row
        
    json_out = os.path.join(RESULTS_DIR, f'exp4_quadrant_dynamics_{args.dataset}.json')
    with open(json_out, 'w') as f:
        json.dump(json_dict, f, indent=2)
        
    print(f"\n[SUCCESS] Quadrant Data saved to:\n -> {csv_out}\n -> {json_out}")

if __name__ == '__main__':
    main()