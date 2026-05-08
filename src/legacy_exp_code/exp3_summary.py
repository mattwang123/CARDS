import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import joblib
from tqdm import tqdm

# --- CONFIGURATION ---
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'Qwen/Qwen2.5-72B-Instruct'
]

TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]
DATASETS = ['umwp', 'treecut']

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
RESULTS_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new', 'results')
TEST_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_test')
EMB_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new', 'embeddings')
PROBES_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new', 'probes')

def compile_master_summary():
    all_results = []
    
    for dataset in DATASETS:
        momentum_path = os.path.join(RESULTS_DIR, f'final_momentum_{dataset}.json')
        exp5_path = os.path.join(RESULTS_DIR, f'exp5_global_dynamics_{dataset}.json')
        
        if not os.path.exists(momentum_path):
            continue
            
        with open(momentum_path, 'r') as f:
            momentum_data = json.load(f)
            
        exp5_data = {}
        if os.path.exists(exp5_path):
            with open(exp5_path, 'r') as f:
                exp5_data = json.load(f)
                
        print(f"\n{'='*50}\nProcessing Dataset: {dataset.upper()}\n{'='*50}")
            
        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            if model_name not in momentum_data: continue
                
            cot_path = os.path.join(TEST_DIR, 'math', model_slug, f"{dataset}_cot_generations.json")
            if not os.path.exists(cot_path): continue
                
            print(f"Analyzing {model_slug}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except:
                continue
                
            with open(cot_path, 'r') as f:
                cot_data = json.load(f)
                
            lengths = []
            labels = [] 
            
            for item in tqdm(cot_data, desc="  Tokenizing", leave=False):
                labels.append(1 if not item.get('is_sufficient', True) else 0)
                prompt_len = tokenizer(item['prompt'], return_tensors="pt")['input_ids'].shape[1]
                full_text = item['prompt'] + item.get('generated_response', '')
                full_len = tokenizer(full_text, return_tensors="pt")['input_ids'].shape[1]
                lengths.append(full_len - prompt_len)
                
            lengths = np.array(lengths)
            labels = np.array(labels)
            
            model_res = momentum_data[model_name]
            e5_res = exp5_data.get(model_name, {})
            eos_f1 = e5_res.get('eos_f1', None)
            eos_layer = e5_res.get('eos_layer', None)
            
            for t in TIMESTEPS:
                t_key = f"t_{t}"
                if t_key not in model_res: continue
                
                best_layer = model_res[t_key].get('best_layer', -1)
                global_test_f1 = model_res[t_key].get('max_test_f1', 0.0)
                
                surviving_mask = lengths >= t
                surviving_insuff = np.sum((surviving_mask) & (labels == 1))
                surviving_suff = np.sum((surviving_mask) & (labels == 0))
                total_surviving = surviving_insuff + surviving_suff
                pool_insuff_pct = (surviving_insuff / total_surviving * 100) if total_surviving > 0 else 0
                
                # --- CALCULATE ACTIVE-ONLY F1 ---
                active_f1 = None
                if total_surviving > 0:
                    emb_path = os.path.join(EMB_DIR, dataset, model_slug, f"t_{t}_test.npy")
                    probe_path = os.path.join(PROBES_DIR, dataset, model_slug, f"best_probe_t{t}_layer{best_layer}.joblib")
                    
                    if os.path.exists(emb_path) and os.path.exists(probe_path):
                        try:
                            # Load test embeddings and the trained probe
                            X_test = np.load(emb_path)
                            probe = joblib.load(probe_path)
                            
                            # Filter to only sequences that are still actively reasoning
                            X_active = X_test[surviving_mask, best_layer, :]
                            y_active = labels[surviving_mask]
                            
                            # Only calculate F1 if we have at least some representation of both classes
                            # (Otherwise F1 is ill-defined)
                            if len(np.unique(y_active)) > 1:
                                active_f1 = f1_score(y_active, probe.predict(X_active))
                        except Exception as e:
                            pass
                
                all_results.append({
                    "Dataset": dataset,
                    "Model": model_slug,
                    "Timestep": int(t),
                    "Best_Layer": int(best_layer),
                    "Global_Test_F1": float(round(global_test_f1, 4)),
                    "Active_Only_F1": float(round(active_f1, 4)) if active_f1 is not None else None,
                    "EOS_F1": float(round(eos_f1, 4)) if eos_f1 is not None else None,
                    "EOS_Layer": int(eos_layer) if eos_layer is not None else None,
                    "Surviving_Total_N": int(total_surviving),
                    "Surviving_Insuff_N": int(surviving_insuff),
                    "Surviving_Suff_N": int(surviving_suff),
                    "Surviving_Pool_Insuff_%": float(round(pool_insuff_pct, 1))
                })

    df = pd.DataFrame(all_results)
    output_csv = os.path.join(RESULTS_DIR, 'exp3_master_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Master summary saved to: {output_csv}")

if __name__ == '__main__':
    compile_master_summary()