"""
================================================================================
EXPERIMENT 10: Ultimate Proportional Dynamics & Smart Storage
================================================================================
Extracts features at normalized reasoning percentages [0%, 20%, 40%, 60%, 80%, 100%].
At the optimal layer, it computes four critical metrics per position:
1. Unified Train F1
2. Unified Test F1
3. Separate Train F1 (The absolute theoretical information capacity)
4. Separate Test F1

Finally, safely slices and stores ONLY the embeddings from the optimal layer to 
the robust NFS storage (/export/fs06) to bypass I/O bottlenecks while securing 
data for future t-SNE visualizations.
================================================================================
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
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

PERCENTAGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_OUT_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
TRAIN_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_train')
TEST_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_test')

def load_exp2_data(model_name, dataset, split_dir):
    model_slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{model_slug}/{dataset}_cot_generations.json"
    if not os.path.exists(path): return None, None
    with open(path, 'r') as f: data = json.load(f)
    labels = [1 if not d.get('is_sufficient', True) else 0 for d in data]
    return data, np.array(labels)

def extract_proportional_features(model, tokenizer, data, labels, desc_label=""):
    """
    Returns:
        X: shape (num_samples, len(PERCENTAGES), num_layers, hidden_dim)
        y: shape (num_samples,)
    """
    extracted_features = []
    valid_labels = []

    for item, label in tqdm(zip(data, labels), total=len(data), desc=desc_label):
        prompt_text = item['prompt']
        generated_text = item.get('generated_response', '')
        
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'][0]
        full_ids = tokenizer(prompt_text + generated_text, return_tensors="pt")['input_ids'][0]
        
        p_len = prompt_ids.shape[0]
        total_len = full_ids.shape[0]
        cot_len = total_len - p_len
        
        # Skip anomalous short generations
        if cot_len < 10: continue 
            
        target_indices = []
        for pct in PERCENTAGES:
            idx = p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0)
            target_indices.append(min(idx, total_len - 1))
            
        inputs = tokenizer(prompt_text + generated_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # layer_states shape: [num_layers, seq_len, hidden_dim]
            layer_states = torch.stack([layer[0].to(torch.float32).cpu() for layer in outputs.hidden_states])
            
            # Extract target indices. Result: [len(PERCENTAGES), num_layers, hidden_dim]
            target_states = layer_states[:, target_indices, :].transpose(0, 1).numpy()
            
            extracted_features.append(target_states)
            valid_labels.append(label)
                
            del outputs, layer_states, inputs
            torch.cuda.empty_cache()

    return np.array(extracted_features), np.array(valid_labels)

def train_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs', n_jobs=-1)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='treecut', choices=['umwp', 'treecut'])
    args = parser.parse_args()

    results_dir = os.path.join(BASE_OUT_DIR, 'results')
    emb_base_dir = os.path.join(BASE_OUT_DIR, 'embeddings_proportional', args.dataset)
    probe_base_dir = os.path.join(BASE_OUT_DIR, 'probes_proportional', args.dataset)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(emb_base_dir, exist_ok=True)
    os.makedirs(probe_base_dir, exist_ok=True)
    
    csv_out_path = os.path.join(results_dir, f"exp10_ultimate_proportional_{args.dataset}.csv")
    
    all_results = []
    # Support atomic resume
    if os.path.exists(csv_out_path):
        all_results = pd.read_csv(csv_out_path).to_dict('records')
        processed_models = set([r['Model'] for r in all_results])
    else:
        processed_models = set()

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        
        if model_slug in processed_models:
            print(f"   [SKIP] Ultimate Proportional already done for {model_slug}.")
            continue
            
        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")

        train_data, train_labels = load_exp2_data(model_name, args.dataset, TRAIN_DIR)
        test_data, test_labels = load_exp2_data(model_name, args.dataset, TEST_DIR)
        if train_data is None or test_data is None: 
            print(f"   ! Missing data for {model_slug}. Skipping.")
            continue

        # ==========================================
        # PHASE 1: EXTRACTION
        # ==========================================
        print("   [1] Extracting Proportional States into VRAM...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            num_gpus = torch.cuda.device_count()
            memory_map = {0: "65GB"} if num_gpus > 0 else None
            if num_gpus > 1:
                for i in range(1, num_gpus): memory_map[i] = "78GB"

            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory=memory_map, 
                torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            model.eval()
            
            # Shapes: [num_samples, num_pct, num_layers, hidden_dim]
            X_train, y_train = extract_proportional_features(model, tokenizer, train_data, train_labels, "Train")
            X_test, y_test = extract_proportional_features(model, tokenizer, test_data, test_labels, "Test")
            
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ! Extraction failed: {e}")
            continue

        # ==========================================
        # PHASE 2: FIND OPTIMAL GLOBAL LAYER
        # ==========================================
        print("   [2] Finding Optimal Unified Layer...")
        num_layers = X_train.shape[2]
        
        # Flatten across percentages for unified training
        X_train_flat = X_train.reshape(-1, num_layers, X_train.shape[-1])
        y_train_flat = np.repeat(y_train, len(PERCENTAGES))
        X_test_flat = X_test.reshape(-1, num_layers, X_test.shape[-1])
        y_test_flat = np.repeat(y_test, len(PERCENTAGES))
        
        best_unified_f1 = 0.0
        best_layer = -1
        best_unified_probe = None
        
        for layer_idx in tqdm(range(num_layers), desc="Scanning Layers"):
            probe = train_probe()
            probe.fit(X_train_flat[:, layer_idx, :], y_train_flat)
            f1 = f1_score(y_test_flat, probe.predict(X_test_flat[:, layer_idx, :]))
            
            if f1 > best_unified_f1:
                best_unified_f1 = f1
                best_layer = layer_idx
                best_unified_probe = probe
                
        print(f"       -> Best Unified Layer: {best_layer} (Overall Test F1: {best_unified_f1:.4f})")

        # Save the global unified probe
        model_probe_dir = os.path.join(probe_base_dir, model_slug)
        os.makedirs(model_probe_dir, exist_ok=True)
        joblib.dump(best_unified_probe, os.path.join(model_probe_dir, f"unified_probe_layer{best_layer}.joblib"))

        # ==========================================
        # PHASE 3: EVALUATE PER POSITION
        # ==========================================
        print("   [3] Evaluating Cognitive Dynamics per Position...")
        
        for pct_idx, pct in enumerate(PERCENTAGES):
            X_train_pct = X_train[:, pct_idx, best_layer, :]
            X_test_pct = X_test[:, pct_idx, best_layer, :]
            
            # 1. Unified Probe Performance on this specific position
            unified_train_f1 = f1_score(y_train, best_unified_probe.predict(X_train_pct))
            unified_test_f1 = f1_score(y_test, best_unified_probe.predict(X_test_pct))
            
            # 2. Separate Probe Performance strictly isolated to this position
            sep_probe = train_probe()
            sep_probe.fit(X_train_pct, y_train)
            sep_train_f1 = f1_score(y_train, sep_probe.predict(X_train_pct))
            sep_test_f1 = f1_score(y_test, sep_probe.predict(X_test_pct))
            
            # Save the separate probe for this specific percentage 
            joblib.dump(sep_probe, os.path.join(model_probe_dir, f"separate_probe_pct{int(pct*100)}_layer{best_layer}.joblib"))

            # Log Results
            all_results.append({
                "Dataset": args.dataset,
                "Model": model_slug,
                "Percentage": f"{int(pct*100)}%",
                "Optimal_Layer": int(best_layer),
                "Unified_Train_F1": round(float(unified_train_f1), 4),
                "Unified_Test_F1": round(float(unified_test_f1), 4),
                "Separate_Train_F1": round(float(sep_train_f1), 4),
                "Separate_Test_F1": round(float(sep_test_f1), 4)
            })
            
            print(f"       Pct: {int(pct*100):>3}% | Uni Train: {unified_train_f1:.4f} | Sep Train: {sep_train_f1:.4f}")

        # Atomic Save per model
        pd.DataFrame(all_results).to_csv(csv_out_path, index=False)

        # ==========================================
        # PHASE 4: SMART NFS STORAGE (EMBEDDINGS)
        # ==========================================
        print(f"   [4] Smart Saving Layer {best_layer} Embeddings to fs06...")
        model_emb_dir = os.path.join(emb_base_dir, model_slug)
        os.makedirs(model_emb_dir, exist_ok=True)
        
        # Slice only the best layer: shape becomes [num_samples, num_pct, hidden_dim]
        optimal_X_train = X_train[:, :, best_layer, :]
        optimal_X_test = X_test[:, :, best_layer, :]
        
        np.save(os.path.join(model_emb_dir, f"X_train_layer{best_layer}.npy"), optimal_X_train)
        np.save(os.path.join(model_emb_dir, f"X_test_layer{best_layer}.npy"), optimal_X_test)
        np.save(os.path.join(model_emb_dir, f"y_train.npy"), y_train)
        np.save(os.path.join(model_emb_dir, f"y_test.npy"), y_test)
        
        mb_saved = (optimal_X_train.nbytes + optimal_X_test.nbytes) / (1024 * 1024)
        print(f"       -> Sliced and saved approx {mb_saved:.1f} MB to disk. Bypassed NFS bottleneck.")

    print(f"\n[SUCCESS] Master CSV saved to {csv_out_path}")

if __name__ == '__main__':
    main()