"""
================================================================================
EXPERIMENT 5: Global Dynamics & Representational Stationarity
================================================================================
Scientific Objectives:
1. EOS State Tracking: Probes the absolute final token to determine the 
   model's definitive belief state at the exact moment of conclusion.
2. Unified Probe Training: Stacks all intermediate timesteps to train a 
   single global classifier. Proves whether the representational "direction" 
   of Insufficiency is stationary across the generative sequence.

Architecture inherits the identical caching and memory safety rules from Exp 3.
================================================================================
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import gc

# Add current directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_OUT_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
TRAIN_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_train')
TEST_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_test')

def load_exp2_data(model_name, dataset, split_dir):
    model_slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{model_slug}/{dataset}_cot_generations.json"
    
    if not os.path.exists(path):
        return None, None
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    labels = [1 if not d.get('is_sufficient', True) else 0 for d in data]
    return data, np.array(labels)

def extract_eos_embeddings_fast(model, tokenizer, data, desc_label=""):
    """
    Performs a single forward pass on the FULL generated sequence, 
    but only extracts the absolute final token (-1 index).
    """
    extracted = []
    
    for item in tqdm(data, desc=f"EOS Forward Pass ({desc_label})"):
        prompt_text = item['prompt']
        generated_text = item.get('generated_response', '')
        
        full_text = prompt_text + generated_text
        full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**full_inputs, output_hidden_states=True)
            
            # The Magic Index: -1 grabs the exact conclusion token safely in float32
            states = [layer[0, -1, :].to(torch.float32).cpu().numpy() for layer in outputs.hidden_states]
            extracted.append(states)
            
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

    return np.array(extracted)

def train_unified_probe(model_slug, dataset, train_labels, model_emb_dir, model_probe_dir):
    """
    Stacks all intermediate timesteps to train a single global classifier.
    """
    print("   [1] Stacking Train Embeddings for Unified Probe...")
    X_train_unified = []
    y_train_unified = []
    
    for t in TIMESTEPS:
        emb_path = os.path.join(model_emb_dir, f"t_{t}_train.npy")
        if os.path.exists(emb_path):
            X_train_unified.append(np.load(emb_path))
            y_train_unified.extend(train_labels)
            
    if not X_train_unified:
        print("       ! No training embeddings found. Run Exp 3 first.")
        return -1, None
        
    X_train_unified = np.vstack(X_train_unified)
    y_train_unified = np.array(y_train_unified)
    num_layers = X_train_unified.shape[1]
    
    best_f1, best_layer, best_probe = 0.0, -1, None
    
    for layer_idx in tqdm(range(num_layers), desc="Training Unified Layer Probes"):
        X_layer = X_train_unified[:, layer_idx, :]
        
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight='balanced', C=1.0, solver='lbfgs', n_jobs=-1)
        )
        probe.fit(X_layer, y_train_unified)
        
        f1 = f1_score(y_train_unified, probe.predict(X_layer))
        if f1 > best_f1:
            best_f1, best_layer, best_probe = f1, layer_idx, probe
            
    print(f"       -> Global Unified Layer: {best_layer} (Train F1: {best_f1:.3f})")
    joblib.dump(best_probe, os.path.join(model_probe_dir, f"unified_probe_layer{best_layer}.joblib"))
    return best_layer, best_probe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umwp', choices=['umwp', 'treecut'])
    args = parser.parse_args()

    results_dir = os.path.join(BASE_OUT_DIR, 'results')
    probes_dir = os.path.join(BASE_OUT_DIR, 'probes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    master_results_path = os.path.join(results_dir, f"exp5_global_dynamics_{args.dataset}.json")
    
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            master_results = json.load(f)
    else:
        master_results = {}

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        if model_name in master_results and "unified_f1_t0" in master_results[model_name] and "eos_f1" in master_results[model_name]:
            print(f"   [RESUME HIT] Exp 5 fully complete for {model_slug}. Skipping.")
            continue
            
        model_emb_dir = os.path.join(BASE_OUT_DIR, 'embeddings', args.dataset, model_slug)
        model_probe_dir = os.path.join(probes_dir, args.dataset, model_slug)
        os.makedirs(model_emb_dir, exist_ok=True)
        os.makedirs(model_probe_dir, exist_ok=True)

        if model_name not in master_results:
            master_results[model_name] = {}

        train_data, train_labels = load_exp2_data(model_name, args.dataset, TRAIN_DIR)
        test_data, test_labels = load_exp2_data(model_name, args.dataset, TEST_DIR)
        
        if train_data is None or test_data is None: 
            print("   ! Missing required train or test splits. Skipping model.")
            continue

        # ==========================================
        # PHASE 1: UNIFIED STATIONARITY PROBE
        # ==========================================
        if "unified_layer" not in master_results[model_name]:
            unified_layer, unified_probe = train_unified_probe(model_slug, args.dataset, train_labels, model_emb_dir, model_probe_dir)
            if unified_probe is None: continue
            
            print("   [2] Evaluating Unified Probe Across Time (Stationarity Test)...")
            master_results[model_name]["unified_layer"] = unified_layer
            
            for t in TIMESTEPS:
                test_emb_path = os.path.join(model_emb_dir, f"t_{t}_test.npy")
                if not os.path.exists(test_emb_path): continue
                    
                X_test = np.load(test_emb_path)[:, unified_layer, :]
                test_f1 = f1_score(test_labels, unified_probe.predict(X_test))
                master_results[model_name][f"unified_f1_t{t}"] = float(test_f1)
                print(f"       t={t:<3} | Unified F1: {test_f1:.3f}")

        # ==========================================
        # PHASE 2: EOS (END-OF-SEQUENCE) PROBE
        # ==========================================
        eos_train_path = os.path.join(model_emb_dir, "t_eos_train.npy")
        eos_test_path = os.path.join(model_emb_dir, "t_eos_test.npy")
        
        if not (os.path.exists(eos_train_path) and os.path.exists(eos_test_path)):
            print("   [3] Extracting Absolute EOS Conclusion States...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            num_gpus = torch.cuda.device_count()
            memory_map = {0: "65GB"} if num_gpus > 0 else None
            if num_gpus > 1:
                for i in range(1, num_gpus):
                    memory_map[i] = "78GB"

            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory=memory_map, 
                torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            model.eval()
            
            train_eos = extract_eos_embeddings_fast(model, tokenizer, train_data, "Train EOS")
            np.save(eos_train_path, train_eos)
            
            test_eos = extract_eos_embeddings_fast(model, tokenizer, test_data, "Test EOS")
            np.save(eos_test_path, test_eos)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("   [3] [CACHE HIT] Loaded EOS embeddings from disk.")

        print("   [4] Training Probe on EOS States...")
        X_train_eos = np.load(eos_train_path)
        X_test_eos = np.load(eos_test_path)
        num_layers = X_train_eos.shape[1]
        
        best_eos_f1, best_eos_layer, best_eos_probe = 0.0, -1, None
        
        for layer_idx in tqdm(range(num_layers), desc="Probing EOS layers"):
            X_train = X_train_eos[:, layer_idx, :]
            X_test = X_test_eos[:, layer_idx, :]

            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs', n_jobs=-1))
            probe.fit(X_train, train_labels)

            test_f1 = f1_score(test_labels, probe.predict(X_test))
            if test_f1 > best_eos_f1:
                best_eos_f1, best_eos_layer, best_eos_probe = test_f1, layer_idx, probe

        print(f"       -> EOS BEST Layer: {best_eos_layer:<2} | Test F1: {best_eos_f1:.3f}")
        joblib.dump(best_eos_probe, os.path.join(model_probe_dir, f"eos_probe_layer{best_eos_layer}.joblib"))
        
        master_results[model_name]["eos_layer"] = best_eos_layer
        master_results[model_name]["eos_f1"] = float(best_eos_f1)

        with open(master_results_path, 'w') as f:
            json.dump(master_results, f, indent=2)

if __name__ == '__main__':
    main()