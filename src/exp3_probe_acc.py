import os
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATASETS = ['umwp', 'treecut']
MODELS = [
    'Qwen2.5-Math-1.5B', 'Qwen2.5-Math-1.5B-Instruct',
    'Qwen2.5-3B', 'Qwen2.5-3B-Instruct', 'gemma-3-4b-it',
    'Qwen2.5-Math-7B', 'Qwen2.5-Math-7B-Instruct',
    'Meta-Llama-3.1-8B', 'Meta-Llama-3.1-8B-Instruct', 'gemma-3-12b-it',
    'Olmo-3-7B-Think', 'Olmo-3-7B-Instruct', 'deepseek-math-7b-instruct',
    'Qwen2.5-14B', 'Qwen2.5-14B-Instruct', 'gemma-3-27b-it',
    'Olmo-3-32B-Think', 'gpt-oss-20b', 'DeepSeek-R1-Distill-Qwen-32B',
    'DeepSeek-R1-Distill-Llama-70B', 'Qwen2.5-72B-Instruct'
]
TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]
BASE_DIR = 'exp_temporal_new'
TEST_DIR = 'experiments/dynamic_tracking_test'

def get_test_labels(dataset, model_slug):
    """Loads the true labels from the test JSON."""
    path = f"{TEST_DIR}/math/{model_slug}/{dataset}_cot_generations.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array([1 if not d.get('is_sufficient', True) else 0 for d in data])

def main():
    print("=== Extracting Accuracies from Saved Probes ===\n")
    results_dir = os.path.join(BASE_DIR, 'results')
    
    for dataset in DATASETS:
        master_path = os.path.join(results_dir, f"final_momentum_{dataset}.json")
        if not os.path.exists(master_path):
            continue
            
        with open(master_path, 'r') as f:
            master_results = json.load(f)
            
        accuracy_results = {}
            
        for model_slug in MODELS:
            # Need the full model name as it appears in your JSON keys
            # A quick hack to find the matching key since the JSON stores full paths:
            model_key = next((k for k in master_results.keys() if model_slug in k), None)
            
            if not model_key or model_key not in master_results:
                continue
                
            y_test = get_test_labels(dataset, model_slug)
            if y_test is None:
                continue
                
            accuracy_results[model_key] = {}
            emb_dir = os.path.join(BASE_DIR, 'embeddings', model_slug)
            probe_dir = os.path.join(BASE_DIR, 'probes', model_slug)
            
            print(f"Processing: {model_key}")
            
            for t in TIMESTEPS:
                t_key = f"t_{t}"
                if t_key not in master_results[model_key]:
                    continue
                    
                best_layer = master_results[model_key][t_key]["best_layer"]
                probe_path = os.path.join(probe_dir, f"best_probe_t{t}_layer{best_layer}.joblib")
                emb_path = os.path.join(emb_dir, f"t_{t}_test.npy")
                
                if not os.path.exists(probe_path) or not os.path.exists(emb_path):
                    continue
                    
                # Load pre-extracted test embeddings and frozen probe
                X_test_all = np.load(emb_path)
                X_test_layer = X_test_all[:, best_layer, :]
                probe = joblib.load(probe_path)
                
                # Predict and score instantly
                y_pred = probe.predict(X_test_layer)
                acc = accuracy_score(y_test, y_pred)
                
                # Store the result
                accuracy_results[model_key][t_key] = float(acc)
                print(f"  t={t:<3} | F1: {master_results[model_key][t_key]['max_test_f1']:.3f} | Accuracy: {acc:.3f}")

        # Save the new accuracies to a separate clean file
        out_path = os.path.join(results_dir, f"accuracies_{dataset}.json")
        with open(out_path, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        print(f"\nSaved accuracies to {out_path}\n")

if __name__ == '__main__':
    main()