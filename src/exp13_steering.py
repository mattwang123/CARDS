import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_NAME = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
DATASET = 'umwp'
ALPHAS = [10, 30, 60, 100, 150]

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
OUT_JSON = 'debug_alpha_sweep.json'

class SteeringHook:
    def __init__(self, v_steer, alpha=1.0):
        self.v_steer = v_steer
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        target_device = hidden_states.device
        current_v_steer = self.v_steer.to(target_device)

        modified_hs = hidden_states.clone()
        last_token_hs = modified_hs[..., -1:, :]
        modified_hs[..., -1:, :] = last_token_hs + self.alpha * current_v_steer
        
        if isinstance(output, tuple):
            return (modified_hs,) + output[1:]
        else:
            return modified_hs

def main():
    model_slug = MODEL_NAME.split('/')[-1]
    
    exp10_df = pd.read_csv(os.path.join(BASE_DIR, 'results', f'exp10_ultimate_proportional_{DATASET}.csv'))
    best_layer = int(exp10_df[exp10_df['Model'] == model_slug]['Optimal_Layer'].iloc[0])
    
    sk_probe = joblib.load(os.path.join(BASE_DIR, 'probes_proportional', DATASET, model_slug, f"unified_probe_layer{best_layer}.joblib"))
    scaler = sk_probe.named_steps['standardscaler']
    clf = sk_probe.named_steps['logisticregression']
    
    W_eff = clf.coef_[0] / scaler.scale_
    v_steer_np = W_eff / np.linalg.norm(W_eff)
    
    eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{DATASET}_evaluated_traces.json")
    gen_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test/math/{model_slug}/{DATASET}_cot_generations.json")
    
    with open(eval_path, 'r') as f: eval_data = json.load(f).get("data", [])
    with open(gen_path, 'r') as f: gen_data = json.load(f)
        
    q1_samples = [g['prompt'] for g, e in zip(gen_data, eval_data) if e.get('epistemic_quadrant') == 'Q1_Hallucination'][:3]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", dtype=torch.bfloat16)
    
    v_steer = torch.tensor(v_steer_np, dtype=model.dtype, device="cpu")
    target_module = model.model.layers[best_layer] if hasattr(model, 'model') else model.layers[best_layer]

    print("\n" + "="*50)
    print(f"STARTING ALPHA SWEEP DIAGNOSTIC -> Saving to {OUT_JSON}")
    print("="*50)

    all_results = []

    for i, prompt in enumerate(q1_samples):
        print(f"\n--- Sample {i+1} ---")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[1]
        
        sample_dict = {"prompt": prompt, "generations": {}}
        
        # Baseline
        with torch.no_grad():
            base_out = model.generate(**inputs, max_new_tokens=600, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_len:], skip_special_tokens=True).strip()
        sample_dict["generations"]["BASELINE"] = base_text
        print(f"BASELINE Ends with:\n...{base_text[-100:]}\n")
        
        # Sweep
        for alpha in ALPHAS:
            hook_obj = SteeringHook(v_steer=v_steer, alpha=alpha)
            handle = target_module.register_forward_hook(hook_obj)
            try:
                with torch.no_grad():
                    steer_out = model.generate(**inputs, max_new_tokens=600, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                res = tokenizer.decode(steer_out[0][input_len:], skip_special_tokens=True).strip()
                sample_dict["generations"][f"ALPHA_{alpha}"] = res
                
                print(f"ALPHA={alpha} Ends with:\n...{res[-100:]}\n")
            finally:
                handle.remove()
                
        all_results.append(sample_dict)

    # Save to JSON
    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ All results saved completely to {OUT_JSON}. Open it to read the full text!")

if __name__ == '__main__':
    main()