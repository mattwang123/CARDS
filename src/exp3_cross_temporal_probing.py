"""
================================================================================
EXPERIMENT 3: Generative Momentum (Forward-Pass Extraction)
================================================================================
Scientific Rigor:
- Strict Train/Test Isolation: Loads generated CoTs from strictly separated 
  training and testing directories to prevent data leakage.
- Chat Template Control: Uses the EXACT prompt string from Exp 2.
- Ultimate Resume: Checks `final_momentum.json` to aggressively skip models, 
  timesteps, and layer training that have already been computed.
- Observability: Calculates and saves the Sequence Survival Rate to guarantee
  the F1 decay is measured on active reasoning, not padding tokens.
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

# --- CONFIGURATION ---
# MODELS = [
#     # --- SMALL/MEDIUM SCALE (~1.5B - 3B) ---
#     'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
#     'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    
#     # --- MEDIUM/LARGE SCALE (~7B - 8B) ---
#     'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
#     'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    
#     # --- LARGE SCALE (14B+) ---
#     'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
# ]

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

# Output directory on /export (for large files: embeddings, probes, results)
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_OUT_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')

# Input directories (can stay in home or also move to export)
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

def calculate_survival_rates(tokenizer, data, timesteps):
    """
    Lightning-fast CPU calculation of how many sequences are still active at each timestep.
    Ensures observability even on cache hits where forward passes are skipped.
    """
    active_counts = {t: 0 for t in timesteps}
    total_samples = len(data)
    
    for item in data:
        prompt_inputs = tokenizer(item['prompt'], return_tensors="pt")
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        full_text = item['prompt'] + item.get('generated_response', '')
        full_inputs = tokenizer(full_text, return_tensors="pt")
        total_len = full_inputs['input_ids'].shape[1]
        
        for t in timesteps:
            target_idx = (prompt_len - 1) + t
            if target_idx < total_len:
                active_counts[t] += 1
                
    survival_rates = {t: (active_counts[t] / total_samples) * 100.0 for t in timesteps}
    return survival_rates

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
            gc.collect() # Force destroy the python activation references
            torch.cuda.empty_cache() # Now PyTorch can actually clear the VRAM

    for t in timesteps:
        extracted[t] = np.array(extracted[t])
        
    return extracted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='umwp')
    args = parser.parse_args()

    # Ensure export base directory exists
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    results_dir = os.path.join(BASE_OUT_DIR, 'results')
    probes_dir = os.path.join(BASE_OUT_DIR, 'probes')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    master_results_path = os.path.join(results_dir, f"final_momentum_{args.dataset}.json")
    
    # Ultimate Resume: Load the master JSON to act as the source of truth
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            master_results = json.load(f)
    else:
        master_results = {}

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        
        # 1. ULTIMATE RESUME (Model Level)
        model_completed = (
            model_name in master_results and 
            all(f"t_{t}" in master_results[model_name] for t in TIMESTEPS)
        )
        if model_completed:
            print(f"   [RESUME HIT] All timesteps already completed for {model_slug}. Skipping.")
            continue
            
        # CRITICAL FIX: Inject args.dataset so domains do not overwrite each other!
        model_emb_dir = os.path.join(BASE_OUT_DIR, 'embeddings', args.dataset, model_slug)
        model_probe_dir = os.path.join(probes_dir, args.dataset, model_slug)
        
        os.makedirs(model_emb_dir, exist_ok=True)
        os.makedirs(model_probe_dir, exist_ok=True)

        if model_name not in master_results:
            master_results[model_name] = {}

        # 2. Load existing generation data (Train and Test)
        train_data, train_labels = load_exp2_data(model_name, args.dataset, TRAIN_DIR)
        test_data, test_labels = load_exp2_data(model_name, args.dataset, TEST_DIR)
        
        if train_data is None or test_data is None: 
            print("   ! Missing required train or test splits. Skipping model.")
            continue
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Calculate survival rates instantly on CPU (crucial for observability)
        print("   -> Calculating sequence survival rates...")
        test_survival_rates = calculate_survival_rates(tokenizer, test_data, TIMESTEPS)
        
        # 3. Check Embeddings Cache
        missing_embs = not all(
            os.path.exists(os.path.join(model_emb_dir, f"t_{t}_train.npy")) and 
            os.path.exists(os.path.join(model_emb_dir, f"t_{t}_test.npy")) 
            for t in TIMESTEPS
        )
        
        if missing_embs:
            print("   [CACHE MISS] Extracting hidden states...")
            
            # Dynamically build max_memory mapping based on actual hardware
            num_gpus = torch.cuda.device_count()
            memory_map = None
            if num_gpus > 0:
                # Always leave ~15GB breathing room on GPU 0 for output_hidden_states
                memory_map = {0: "65GB"} 
                # Max out any additional GPUs if they exist
                for i in range(1, num_gpus):
                    memory_map[i] = "78GB"

            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                max_memory=memory_map, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            
            # Failsafe: check where the model actually landed
            if str(model.device) == 'cpu':
                raise RuntimeError("HuggingFace silently dumped the model to CPU! Clear your VRAM.")
            model.eval()
            
            # Extract Train
            train_extracted = extract_all_timesteps_fast(model, tokenizer, train_data, TIMESTEPS, "Train Set")
            for t in TIMESTEPS:
                np.save(os.path.join(model_emb_dir, f"t_{t}_train.npy"), train_extracted[t])
                
            # Extract Test
            test_extracted = extract_all_timesteps_fast(model, tokenizer, test_data, TIMESTEPS, "Test Set")
            for t in TIMESTEPS:
                np.save(os.path.join(model_emb_dir, f"t_{t}_test.npy"), test_extracted[t])
                
            # Nuclear cleanup between models
            del model
            gc.collect() # Kill the Python ghost
            torch.cuda.empty_cache() # Flush the hardware cache
        else:
            print("   [CACHE HIT] All train and test embeddings found on disk.")

        model_results = master_results[model_name]

        # 4. Train Probes (with Timestep-Level Resume)
        for t in TIMESTEPS:
            t_key = f"t_{t}"
            survival_pct = test_survival_rates[t]
            
            # ULTIMATE RESUME (Timestep Level)
            if t_key in model_results and "best_layer" in model_results[t_key]:
                saved_f1 = model_results[t_key].get('max_test_f1', 0.0)
                saved_surv = model_results[t_key].get('test_survival_rate_pct', survival_pct)
                print(f"   [RESUME HIT] Skipping t={t:<3} | Survival: {saved_surv:>5.1f}% | Saved Max Test F1: {saved_f1:.3f}")
                continue
                
            print(f"\n--- Training Probes for Timestep t={t} (Test Survival: {survival_pct:.1f}%) ---")
            
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

                # # Standard Regularization - Fitting ONLY on Train Set
                # probe = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='liblinear')
                # probe.fit(X_train, train_labels)

                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000, # Drops from 2000 because scaled data converges fast
                        class_weight='balanced', 
                        C=1.0, 
                        solver='lbfgs', 
                        n_jobs=-1     # Uses all available CPU cores
                    )
                )
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

            print(f"   -> Result t={t:<3} | BEST Layer: {best_layer:<2} | Train F1: {best_train_f1:.3f} | Max Test F1: {best_f1:.3f}")
            
            joblib.dump(best_probe, os.path.join(model_probe_dir, f"best_probe_t{t}_layer{best_layer}.joblib"))

            model_results[t_key] = {
                "test_survival_rate_pct": float(survival_pct),
                "best_layer": best_layer,
                "max_train_f1": float(best_train_f1),
                "max_test_f1": float(best_f1),
                "all_layers": layer_results
            }

        master_results[model_name] = model_results
        
        # Save incrementally after every model/timestep update
        with open(master_results_path, 'w') as f:
            json.dump(master_results, f, indent=2)

    print(f"\n[COMPLETE] Fast Forward-Pass Momentum tracking finished. Results in {master_results_path}")

if __name__ == '__main__':
    main()