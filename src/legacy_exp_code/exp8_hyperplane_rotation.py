"""
================================================================================
EXPERIMENT 8: Hyperplane Rotation Diagnostic
================================================================================
Calculates the Cosine Similarity between the optimal separating hyperplane 
at t=0 and subsequent timesteps, locked to the Unified Layer.
This mathematically proves whether "Insufficiency" is geometrically 
stationary or if it undergoes Latent Feature Rotation during complex CoT.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

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
TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 'eos']
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')

def get_active_mask(model_name, dataset, target_t):
    """Calculates which samples are still actively generating at timestep target_t."""
    if target_t == 'eos':
        return None # Return None to indicate 'all true'
        
    model_slug = model_name.split('/')[-1]
    eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
    if not os.path.exists(eval_path): return None
    
    with open(eval_path, 'r') as f:
        data = json.load(f).get("data", [])
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except:
        return None
        
    lengths = []
    for item in data:
        prompt = item.get('prompt', '')
        cot = item.get('full_cot_text', item.get('extracted_raw_text', ''))
        prompt_len = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
        full_len = tokenizer(prompt + cot, return_tensors="pt")['input_ids'].shape[1]
        lengths.append(full_len - prompt_len)
        
    lengths = np.array(lengths)
    return (lengths >= int(target_t))

def load_labels(model_name, dataset):
    model_slug = model_name.split('/')[-1]
    eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
    with open(eval_path, 'r') as f:
        data = json.load(f).get("data", [])
    return np.array([1 if not item['is_sufficient'] else 0 for item in data])

def run_rotation_diagnostic():
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    results = []

    for dataset in DATASETS:
        print(f"\n{'='*60}\nDATASET: {dataset.upper()}\n{'='*60}")
        
        exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{dataset}.json')
        if not os.path.exists(exp5_path): 
            print(f"Skipping {dataset}, missing Exp 5 results.")
            continue
            
        with open(exp5_path, 'r') as f: 
            exp5_data = json.load(f)

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]: 
                continue
            
            unified_layer = exp5_data[model_name]["unified_layer"]
            labels = load_labels(model_name, dataset)
            
            # 1. Fit the reference Hyperplane at t=0
            emb_t0_path = os.path.join(BASE_DIR, 'embeddings', dataset, model_slug, "t_0_test.npy")
            if not os.path.exists(emb_t0_path): 
                continue
            
            X_t0 = np.load(emb_t0_path)[:, unified_layer, :]
            clf_t0 = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            clf_t0.fit(X_t0, labels)
            vec_t0 = clf_t0.coef_ # Shape (1, hidden_dim)

            # 2. Compare against subsequent timesteps
            for t in TIMESTEPS:
                if t == 0: 
                    # Cosine sim of t=0 to itself is 1.0, just append it for completeness
                    results.append({
                        "Dataset": dataset,
                        "Model": model_slug,
                        "Timestep": str(t),
                        "Unified_Layer": unified_layer,
                        "Cosine_Similarity": 1.0,
                        "Active_N": len(labels)
                    })
                    continue
                
                emb_path = os.path.join(BASE_DIR, 'embeddings', dataset, model_slug, f"t_{t}_test.npy")
                if not os.path.exists(emb_path): 
                    continue
                
                X_t = np.load(emb_path)[:, unified_layer, :]
                
                # Filter for active sequences only
                mask = get_active_mask(model_name, dataset, t)
                if mask is None or t == 'eos':
                    mask = np.ones(len(labels), dtype=bool)
                    
                active_X = X_t[mask]
                active_y = labels[mask]
                
                # Need at least 2 classes and decent sample size to draw a valid line
                if len(np.unique(active_y)) < 2 or len(active_y) < 20:
                    continue
                    
                clf_t = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                clf_t.fit(active_X, active_y)
                vec_t = clf_t.coef_
                
                # Calculate the Angle (Cosine Similarity) between the t=0 truth axis and the t=K truth axis
                cos_sim = cosine_similarity(vec_t0, vec_t)[0][0]
                
                results.append({
                    "Dataset": dataset,
                    "Model": model_slug,
                    "Timestep": str(t),
                    "Unified_Layer": unified_layer,
                    "Cosine_Similarity": round(float(cos_sim), 4),
                    "Active_N": len(active_y)
                })
                
                print(f"[{dataset.upper()}] {model_slug} | t={str(t):<3} | Cosine Sim to t=0: {cos_sim:.4f} (N={len(active_y)})")

    df = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, 'results', 'exp8_hyperplane_rotation.csv')
    df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved rotation diagnostic to {out_path}")

if __name__ == '__main__':
    run_rotation_diagnostic()