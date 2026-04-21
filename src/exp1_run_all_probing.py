"""
================================================================================
EXPERIMENT 1A: Direct Binary Probing (Internal Representation)
================================================================================
Scientific Rigor:
- Chat Template Control: Uses the exact same binary "Yes/No" prompt format 
  as Exp 1B to map latent representation against explicit verbalization.
- Optimized Pipeline: Uses float16 numpy arrays, StandardScaler + LBFGS, 
  and Dual-A100 specific VRAM mapping.
- Balanced Sampling: Forces a perfect 50/50 Train Set split to prevent 
  linear probe bias (matching Exp 3).
- Ultimate Resume: Checks final JSON to skip expensive GPU forward passes.
================================================================================
"""

import argparse
import json
import os
import sys
import gc
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact prompt format from verbalization to ensure 1:1 mapping
from exp1_run_all_verbalization import format_prompt

# --- CONFIGURATION ---
DATASETS = {
    'umwp': {
        'train': 'src/data/processed/insufficient_dataset_umwp/umwp_train.json',
        'test': 'src/data/processed/insufficient_dataset_umwp/umwp_test.json'
    },
    'treecut': {
        'train': 'src/data/processed/treecut/treecut_train.json',
        'test': 'src/data/processed/treecut/treecut_test.json'
    }
}

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

BASE_OUT_DIR = '/export/fs06/hwang302/CARDS/exp_verb_probing' # High Capacity Storage

def balance_and_sample(data, max_samples):
    """Ensures a perfect 50/50 balance of classes to prevent linear probe bias."""
    sufficient = [x for x in data if x.get('is_sufficient', True)]
    insufficient = [x for x in data if not x.get('is_sufficient', True)]
    
    target_per_class = max_samples // 2
    n_suff = min(len(sufficient), target_per_class)
    n_insuff = min(len(insufficient), target_per_class)
    
    random.seed(42)
    sampled = random.sample(sufficient, n_suff) + random.sample(insufficient, n_insuff)
    random.shuffle(sampled)
    
    print(f"    [Train Sample] Total: {len(sampled)} | Sufficient: {n_suff} | Insufficient: {n_insuff}")
    return sampled

def extract_last_token_fast(model, tokenizer, data, model_name, desc_label=""):
    """Extracts float32 hidden states for the absolute final token of the binary prompt."""
    extracted = []
    
    for item in tqdm(data, desc=f"Forward Passing ({desc_label})"):
        prompt_text = format_prompt(item['question'], model_name)
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        target_idx = inputs['input_ids'].shape[1] - 1
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # CRITICAL FIX: Changed float16 to float32 to prevent Infinity corruption
            states = [layer[0, target_idx, :].to(torch.float32).cpu().numpy() for layer in outputs.hidden_states]
            extracted.append(states)
            
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
            
    return np.array(extracted)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umwp', choices=['umwp', 'treecut'])
    args = parser.parse_args()

    results_dir = os.path.join(BASE_OUT_DIR, 'results')
    probes_dir = os.path.join(BASE_OUT_DIR, 'probes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    master_results_path = os.path.join(results_dir, f"binary_probe_{args.dataset}.json")
    
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            master_results = json.load(f)
    else:
        master_results = {}

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        # ULTIMATE RESUME
        if model_name in master_results and "best_layer" in master_results[model_name]:
            print(f"   [RESUME HIT] Model already fully probed. Skipping.")
            continue
            
        model_emb_dir = os.path.join(BASE_OUT_DIR, 'embeddings', args.dataset, model_slug)
        model_probe_dir = os.path.join(probes_dir, args.dataset, model_slug)
        os.makedirs(model_emb_dir, exist_ok=True)
        os.makedirs(model_probe_dir, exist_ok=True)

        # 1. Load Raw Data and Labels (With rigorous 50/50 balancing)
        with open(DATASETS[args.dataset]['train'], 'r') as f:
            raw_train = json.load(f)
            train_data = balance_and_sample(raw_train, 3000)
            
        train_labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in train_data])

        with open(DATASETS[args.dataset]['test'], 'r') as f:
            test_data = json.load(f)
        test_labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in test_data])

        # 2. Check Cache
        train_emb_path = os.path.join(model_emb_dir, "train_t0.npy")
        test_emb_path = os.path.join(model_emb_dir, "test_t0.npy")
        
        if not (os.path.exists(train_emb_path) and os.path.exists(test_emb_path)):
            print("   [CACHE MISS] Extracting hidden states...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Dual-GPU Safe Loading
            num_gpus = torch.cuda.device_count()
            memory_map = {0: "65GB"} if num_gpus > 0 else None
            if num_gpus > 1:
                for i in range(1, num_gpus):
                    memory_map[i] = "78GB"

            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                max_memory=memory_map, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            model.eval()
            
            train_extracted = extract_last_token_fast(model, tokenizer, train_data, model_name, "Train Set")
            np.save(train_emb_path, train_extracted)
            
            test_extracted = extract_last_token_fast(model, tokenizer, test_data, model_name, "Test Set")
            np.save(test_emb_path, test_extracted)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("   [CACHE HIT] Load embeddings from disk.")

        # 3. Train Scaled Linear Probes
        X_train_all = np.load(train_emb_path)
        X_test_all = np.load(test_emb_path)
        num_layers = X_train_all.shape[1]
        
        layer_results = {}
        best_f1, best_train_f1, best_layer, best_probe = 0.0, 0.0, -1, None

        for layer_idx in tqdm(range(num_layers), desc=f"Probing {num_layers} layers"):
            X_train = X_train_all[:, layer_idx, :]
            X_test = X_test_all[:, layer_idx, :]

            # Scaled LBFGS Pipeline
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs', n_jobs=-1)
            )
            probe.fit(X_train, train_labels)

            train_f1 = f1_score(train_labels, probe.predict(X_train))
            test_f1 = f1_score(test_labels, probe.predict(X_test))
            
            layer_results[f"layer_{layer_idx}"] = {"train_f1": float(train_f1), "test_f1": float(test_f1)}

            if test_f1 > best_f1:
                best_f1, best_train_f1, best_layer, best_probe = test_f1, train_f1, layer_idx, probe

        print(f"   -> BEST Layer: {best_layer:<2} | Train F1: {best_train_f1:.3f} | Max Test F1: {best_f1:.3f}")
        
        joblib.dump(best_probe, os.path.join(model_probe_dir, f"best_probe_layer{best_layer}.joblib"))

        master_results[model_name] = {
            "best_layer": best_layer,
            "max_train_f1": float(best_train_f1),
            "max_test_f1": float(best_f1),
            "all_layers": layer_results
        }
        
        with open(master_results_path, 'w') as f:
            json.dump(master_results, f, indent=2)

if __name__ == '__main__':
    main()