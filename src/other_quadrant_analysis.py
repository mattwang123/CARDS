"""
================================================================================
EXPERIMENT 4: Quadrant Dynamics & Compute Waste
================================================================================
Leverages cached embeddings to track Signal Death per Epistemic Quadrant.
- Calculates exact Token Compute Waste for Q1 (Hallucinations).
- Tracks Mean Latent Probability of Insufficiency over time across all 4 
  quadrants to prove that Signal Maintenance correlates with behavioral safety, 
  while Signal Death causes hallucination.
"""

import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    """Loads the quadrants and text from Exp 2 Evaluated Traces."""
    model_slug = model_name.split('/')[-1]
    path = f"experiments/dynamic_tracking_evaluation/{dataset}/{model_slug}/{dataset}_evaluated_traces.json"
    
    # Fallback to math domain path if structured that way
    if not os.path.exists(path):
        path = f"experiments/dynamic_tracking_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json"
        
    if not os.path.exists(path):
        print(f"Warning: Evaluated traces not found at {path}")
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
        # Fallback to extracted text if full cot was dropped to save space.
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
    
    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        data, labels, quadrants, texts = load_evaluated_data(model_name, args.dataset)
        if data is None: continue
            
        # 1. Compute Waste Analysis
        print("[1] Compute Waste Analysis")
        calculate_compute_waste(model_name, quadrants, texts)
        
        # 2. Quadrant-Specific Temporal Tracking (Probability)
        print("\n[2] Quadrant-Specific Temporal Tracking (Mean Probability over Time)")
        model_emb_dir = os.path.join(BASE_DIR, 'embeddings', model_slug)
        
        for t in TIMESTEPS:
            emb_path = os.path.join(model_emb_dir, f"t_{t}.npy")
            if not os.path.exists(emb_path):
                continue
                
            X_all = np.load(emb_path)
            num_layers = X_all.shape[1]
            
            best_f1 = 0.0
            best_layer = -1
            best_probe = None
            
            # Step A: Find the best general layer for this timestep (using full dataset)
            # We stratify by quadrant to ensure perfectly balanced train/test splits
            X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
                X_all, labels, quadrants, test_size=0.3, random_state=42, stratify=quadrants
            )
            
            for layer_idx in range(num_layers):
                probe = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='liblinear')
                probe.fit(X_train[:, layer_idx, :], y_train)
                
                # Standard F1 to pick the best semantic layer
                test_f1 = f1_score(y_test, probe.predict(X_test[:, layer_idx, :]))
                
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_layer = layer_idx
                    best_probe = probe
                    
            # Step B: Evaluate the best probe's PROBABILITY on specific quadrants
            # probe.predict_proba returns [prob_0, prob_1]. We want prob_1 (Insufficiency)
            probs = best_probe.predict_proba(X_test[:, best_layer, :])[:, 1]
            
            # Create masks for all 4 quadrants
            q1_mask = (q_test == 'Q1_Hallucination')
            q2_mask = (q_test == 'Q2_Correct_Rejection')
            q3_mask = (q_test == 'Q3_Solved_Correctly')
            q4_mask = (q_test == 'Q4_Competence_Failure')
            
            def get_mean_prob(mask):
                return np.mean(probs[mask]) if np.sum(mask) > 0 else 0.0
                
            p_q1 = get_mean_prob(q1_mask)
            p_q2 = get_mean_prob(q2_mask)
            p_q3 = get_mean_prob(q3_mask)
            p_q4 = get_mean_prob(q4_mask)
            
            print(f"   t={t:<3} | Q1(Fail): {p_q1:.3f} | Q2(Safe): {p_q2:.3f} | Q3(Solve): {p_q3:.3f} | Q4(MathErr): {p_q4:.3f}")

if __name__ == '__main__':
    main()