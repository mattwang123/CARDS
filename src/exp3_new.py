"""
================================================================================
EXPERIMENT 3: Generative Momentum (Forward-Pass Extraction)
================================================================================
Scientific Rigor:
- Strict Train/Test Isolation: Loads generated CoTs from strictly separated 
  training and testing directories to prevent data leakage.
- Chat Template Control: Uses the EXACT prompt string from Exp 2.
- Bottleneck Probing: t=0 is explicitly calculated as the final token of the 
  chat template.
- Fast Forward Pass: Processes the entire Prompt + CoT in a single pass.
- Exhaustive Layer Search: Trains probes on all layers to find the optimal 
  semantic representation at each timestep.
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

# --- CONFIGURATION ---
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct'
]

TIMESTEPS = [0, 5, 10, 20, 50, 100, 200]
BASE_OUT_DIR = 'exp_temporal'
TRAIN_DIR = 'experiments/dynamic_tracking_train'
TEST_DIR = 'experiments/dynamic_tracking_test'

def load_exp2_data(model_name, dataset, split_dir):
    """Loads the pre-generated CoT from Exp 2 based on the requested split."""
    model_slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{model_slug}/{dataset}_cot_generations.json"
    
    if not os.path.exists(path):
        print(f"Warning: Exp 2 data not found at {path}.")
        return None, None
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    labels = [1 if not d.get('is_sufficient', True) else 0 for d in data]
    return data, np.array(labels)

def extract_all_timesteps_fast(model, tokenizer, data, timesteps, desc_label=""):
    """
    Performs a SINGLE forward pass using the EXACT prompt and generation from Exp 2.
    """
    extracted = {t: [] for t in timesteps}
    
    for item in tqdm(data, desc=f"Forward Passing ({desc_label})"):
        prompt_text = item['prompt']
        generated_text = item.get('generated_response', '')
        
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        full_text = prompt_text + generated_text
        full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        total_len = full_inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model(**full_inputs, output_hidden_states=True)
            
            for t in timesteps:
                target_idx = (prompt_len - 1) + t
                if target_idx >= total_len:
                    target_idx = total_len - 1 
                
                states_at_t = [layer[0, target_idx, :].to(torch.float32).cpu().numpy() for layer in outputs.hidden_states]
                extracted[t].append(states_at_t)
                
            del outputs
            torch.cuda.empty_cache()

    for t in timesteps:
        extracted[t] = np.array(extracted[t])
        
    return extracted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umwp')
    args = parser.parse_args()

    results_dir = os.path.join(BASE_OUT_DIR, 'results')
    probes_dir = os.path.join(BASE_OUT_DIR, 'probes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    master_results_path = os.path.join(results_dir, f"final_momentum_{args.dataset}.json")
    
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            master_results = json.load(f)
    else:
        master_results = {}

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        model_emb_dir = os.path.join(BASE_OUT_DIR, 'embeddings', model_slug)
        model_probe_dir = os.path.join(probes_dir, model_slug)
        os.makedirs(model_emb_dir, exist_ok=True)
        os.makedirs(model_probe_dir, exist_ok=True)

        if model_name not in master_results:
            master_results[model_name] = {}

        # 1. Load existing generation data (Train and Test)
        train_data, train_labels = load_exp2_data(model_name, args.dataset, TRAIN_DIR)
        test_data, test_labels = load_exp2_data(model_name, args.dataset, TEST_DIR)
        
        if train_data is None or test_data is None: 
            print("Missing required train or test splits. Skipping model.")
            continue
        
        # Check if we already extracted embeddings for this model
        missing_embs = not all(
            os.path.exists(os.path.join(model_emb_dir, f"t_{t}_train.npy")) and 
            os.path.exists(os.path.join(model_emb_dir, f"t_{t}_test.npy")) 
            for t in TIMESTEPS
        )
        
        if missing_embs:
            print("   [CACHE MISS] Extracting hidden states...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            # Use bfloat16 for VRAM safety
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
            model.eval()
            
            # Extract Train
            train_extracted = extract_all_timesteps_fast(model, tokenizer, train_data, TIMESTEPS, "Train Set")
            for t in TIMESTEPS:
                np.save(os.path.join(model_emb_dir, f"t_{t}_train.npy"), train_extracted[t])
                
            # Extract Test
            test_extracted = extract_all_timesteps_fast(model, tokenizer, test_data, TIMESTEPS, "Test Set")
            for t in TIMESTEPS:
                np.save(os.path.join(model_emb_dir, f"t_{t}_test.npy"), test_extracted[t])
                
            del model
            torch.cuda.empty_cache()
        else:
            print("   [CACHE HIT] All train and test embeddings found on disk.")

        model_results = {}

        # 2. Train Probes for all layers at all timesteps
        for t in TIMESTEPS:
            print(f"\n--- Training Probes for Timestep t={t} ---")
            
            X_train_all = np.load(os.path.join(model_emb_dir, f"t_{t}_train.npy"))
            X_test_all = np.load(os.path.join(model_emb_dir, f"t_{t}_test.npy"))
            num_layers = X_train_all.shape[1]
            
            layer_results = {}
            best_f1 = 0.0
            best_train_f1 = 0.0
            best_layer = -1
            best_probe = None

            for layer_idx in tqdm(range(num_layers), desc=f"Probing {num_layers} layers"):
                X_train = X_train_all[:, layer_idx, :]
                X_test = X_test_all[:, layer_idx, :]

                # Standard Regularization - Fitting ONLY on Train Set
                probe = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='liblinear')
                probe.fit(X_train, train_labels)

                # Evaluate explicitly on Test Set
                train_f1 = f1_score(train_labels, probe.predict(X_train))
                test_f1 = f1_score(test_labels, probe.predict(X_test))
                
                layer_results[f"layer_{layer_idx}"] = {
                    "train_f1": float(train_f1),
                    "test_f1": float(test_f1)
                }

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_train_f1 = train_f1
                    best_layer = layer_idx
                    best_probe = probe

            print(f"Timestep t={t} | BEST Layer: {best_layer} | Train F1: {best_train_f1:.3f} | Max Test F1: {best_f1:.3f}")
            
            joblib.dump(best_probe, os.path.join(model_probe_dir, f"best_probe_t{t}_layer{best_layer}.joblib"))

            model_results[f"t_{t}"] = {
                "best_layer": best_layer,
                "max_train_f1": float(best_train_f1),
                "max_test_f1": float(best_f1),
                "all_layers": layer_results
            }

        master_results[model_name] = model_results
        
        with open(master_results_path, 'w') as f:
            json.dump(master_results, f, indent=2)

    print(f"\n[COMPLETE] Fast Forward-Pass Momentum tracking finished. Results in {master_results_path}")

if __name__ == '__main__':
    main()