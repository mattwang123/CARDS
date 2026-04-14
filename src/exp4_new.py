"""
================================================================================
EXPERIMENT 4: Quadrant Dynamics & Compute Waste
================================================================================
Leverages frozen, pre-trained probes from Exp 3 to track Signal Death per Epistemic Quadrant.
- Calculates exact Token Compute Waste for Q1 (Hallucinations) on the Test Set.
- Loads the optimal semantic probe (.joblib) trained on the isolated Train Set.
- Tracks Mean Latent Probability of Insufficiency over time across the 4 
  behavioral quadrants on the completely unseen Test Set.
================================================================================
"""

import json
import os
import numpy as np
import argparse
import joblib
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct'
]

TIMESTEPS = [0, 5, 10, 20, 50, 100, 200]
BASE_DIR = 'exp_temporal'

def load_evaluated_data(model_name, dataset):
    """Loads the quadrants and text from Exp 2 Evaluated Traces (Test Set)."""
    model_slug = model_name.split('/')[-1]
    
    # Check test-specific path
    path = f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json"
    
    if not os.path.exists(path):
        print(f"Warning: Evaluated test traces not found at {path}")
        return None, None, None, None
        
    with open(path, 'r') as f:
        payload = json.load(f)
        data = payload.get("data", [])
        
    labels = []
    quadrants = []
    texts = []
    
    for item in data:
        labels.append(1 if not item['is_sufficient'] else 0)
        quadrants.append(item['epistemic_quadrant'])
        
        # We need the text to calculate token waste. 
        texts.append(item.get('full_cot_text', item.get('extracted_raw_text', '')))
        
    return data, np.array(labels), np.array(quadrants), texts

def calculate_compute_waste(model_name, quadrants, texts):
    """Calculates the exact token waste for Q1 hallucinations."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wasted_tokens = 0
    q1_count = 0
    
    for q, text in zip(quadrants, texts):
        if q == 'Q1_Hallucination':
            q1_count += 1
            if text:
                tokens = tokenizer.encode(text)
                wasted_tokens += len(tokens)
            
    if q1_count > 0:
        print(f"   -> Wasted Compute (Q1): {wasted_tokens:,} tokens across {q1_count} hallucinations.")
        print(f"   -> Average Waste per Hallucination: {wasted_tokens / q1_count:.1f} tokens.")
    else:
        print("   -> No Q1 hallucinations found.")
    return wasted_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='treecut')
    args = parser.parse_args()

    print(f"=== Running Quadrant Dynamics for {args.dataset.upper()} ===\n")
    
    # Path to Exp 3 master results to find the best layer
    results_dir = os.path.join(BASE_DIR, 'results')
    master_results_path = os.path.join(results_dir, f"final_momentum_{args.dataset}.json")
    
    if not os.path.exists(master_results_path):
        print(f"Error: Run Exp 3 first. Missing {master_results_path}")
        return
        
    with open(master_results_path, 'r') as f:
        master_results = json.load(f)
    
    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        # 1. Load the Test Set behavioral data
        data, labels, quadrants, texts = load_evaluated_data(model_name, args.dataset)
        if data is None: continue
            
        # 2. Compute Waste Analysis
        print("[1] Compute Waste Analysis")
        calculate_compute_waste(model_name, quadrants, texts)
        
        # 3. Quadrant-Specific Temporal Tracking
        print("\n[2] Quadrant-Specific Temporal Tracking (Mean Probability over Time)")
        model_emb_dir = os.path.join(BASE_DIR, 'embeddings', model_slug)
        probes_dir = os.path.join(BASE_DIR, 'probes', model_slug)
        
        if model_name not in master_results:
            print(f"   -> Missing Exp 3 tracking data for {model_name}. Skipping.")
            continue
            
        for t in TIMESTEPS:
            # A. Load the Test Set hidden states for this timestep
            emb_path_test = os.path.join(model_emb_dir, f"t_{t}_test.npy")
            if not os.path.exists(emb_path_test):
                continue
                
            X_test_all = np.load(emb_path_test)
            
            # B. Identify the best layer and load the frozen probe from Exp 3
            try:
                best_layer = master_results[model_name][f"t_{t}"]["best_layer"]
            except KeyError:
                continue
                
            probe_path = os.path.join(probes_dir, f"best_probe_t{t}_layer{best_layer}.joblib")
            if not os.path.exists(probe_path):
                print(f"   -> Probe missing at {probe_path}")
                continue
                
            probe = joblib.load(probe_path)
            
            # Isolate the exact layer we are testing
            X_test_layer = X_test_all[:, best_layer, :]
            
            # Ensure indices align (X_test length should match quadrants length)
            if len(X_test_layer) != len(quadrants):
                print(f"   -> Data mismatch! Embeddings: {len(X_test_layer)}, Quadrants: {len(quadrants)}")
                break
            
            # C. Evaluate the frozen probe's PROBABILITY
            # predict_proba returns [prob_class_0, prob_class_1]. We want prob_class_1 (Insufficiency)
            probs = probe.predict_proba(X_test_layer)[:, 1]
            
            # D. Map probabilities to their behavioral quadrants
            q1_mask = (quadrants == 'Q1_Hallucination')
            q2_mask = (quadrants == 'Q2_Correct_Rejection')
            q3_mask = (quadrants == 'Q3_Solved_Correctly')
            q4_mask = (quadrants == 'Q4_Competence_Failure')
            
            def get_mean_prob(mask):
                return np.mean(probs[mask]) if np.sum(mask) > 0 else 0.0
                
            p_q1 = get_mean_prob(q1_mask)
            p_q2 = get_mean_prob(q2_mask)
            p_q3 = get_mean_prob(q3_mask)
            p_q4 = get_mean_prob(q4_mask)
            
            print(f"   t={t:<3} | Best Layer: {best_layer:<2} | Q1(Fail): {p_q1:.3f} | Q2(Safe): {p_q2:.3f} | Q3(Solve): {p_q3:.3f} | Q4(MathErr): {p_q4:.3f}")

if __name__ == '__main__':
    main()