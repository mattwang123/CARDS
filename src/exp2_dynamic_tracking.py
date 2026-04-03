"""
================================================================================
EXPERIMENT 2B: Dynamic Temporal Tracking (Latent Awareness Extraction)
================================================================================

This script:
1. Loads the generated CoT responses from Experiment 2A.
2. Identifies the optimal probe layer from Experiment 1 (`best_layers_linear.json`).
3. Re-trains the linear probe on the training embeddings for that specific layer.
4. Performs a SINGLE forward pass on the [Prompt + Response] string.
5. Extracts hidden states for all generated tokens simultaneously.
6. Applies the probe to compute the "Insufficiency Probability" per token.
7. Saves the token-level trace to `[dataset]_temporal_traces.json`.
================================================================================
"""

import argparse
import json
import os
import sys
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    # --- SMALL/MEDIUM SCALE (~1.5B - 3B) ---
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    
    # --- MEDIUM/LARGE SCALE (~7B - 8B) ---
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    
    # --- LARGE SCALE (14B+) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
]

def load_labels(dataset_name, split):
    """Load binary labels from dataset. 1 = Sufficient, 0 = Insufficient."""
    data_path = DATASETS[dataset_name][split]
    with open(data_path, 'r') as f:
        data = json.load(f)
    labels = np.array([1 if item.get('is_sufficient', False) else 0 for item in data])
    return labels

def get_embedding_path(dataset_name, split, model_name, pooling='last_token'):
    """Generate embedding file path consistent with Exp 1."""
    model_slug = model_name.replace('/', '_')
    filename = f"{dataset_name}_{split}_{model_slug}_{pooling}.npy"
    return f"src/data/embeddings/{filename}"

def train_best_layer_probe(model_name, dataset_name, best_layer, device_info="cpu"):
    """Loads Exp 1 embeddings and quickly re-trains the probe for the target layer."""
    labels = load_labels(dataset_name, 'train')
    emb_path = get_embedding_path(dataset_name, 'train', model_name)
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing Exp 1 embeddings: {emb_path}. Run Exp 1 first.")
        
    print(f"    Loading Exp 1 training embeddings from {emb_path}")
    # Shape: (num_samples, num_layers, hidden_dim)
    embeddings = np.load(emb_path)
    
    layer_emb = embeddings[:, best_layer, :]
    
    print(f"    Training Logistic Regression probe on Layer {best_layer}...")
    probe = LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight='balanced')
    probe.fit(layer_emb, labels)
    
    return probe

def process_and_probe_generation(item, model, tokenizer, probe, best_layer, device, max_len=8192):
    """
    Runs a single forward pass on the prompt+response, extracts hidden states for
    the response tokens, and applies the trained probe.
    """
    prompt = item['prompt']
    response = item['generated_response']
    
    # Strictly separate tokenization to map tokens cleanly
    prompt_ids = tokenizer.encode(prompt)
    # add_special_tokens=False ensures we don't accidentally add BOS tags in the middle
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    
    # Safety truncation to avoid OOM or position embedding errors
    if len(prompt_ids) + len(response_ids) > max_len:
        response_ids = response_ids[:(max_len - len(prompt_ids))]
        
    input_ids = torch.tensor([prompt_ids + response_ids]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # hidden_states tuple: (layer_0, layer_1, ..., layer_L)
        # Shape of specific layer: (1, seq_len, hidden_dim)
        layer_hidden_states = outputs.hidden_states[best_layer][0]
    
    # Slice to isolate only the generated response tokens
    prompt_len = len(prompt_ids)
    response_hidden_states = layer_hidden_states[prompt_len:, :]
    
    # Convert to numpy for sklearn probe
    hidden_np = response_hidden_states.to(torch.float32).cpu().numpy()
    
    # probe.predict_proba returns shape (seq_len, 2). 
    # Class 0 = Insufficient, Class 1 = Sufficient.
    # We want the probability that the model thinks it is INSUFFICIENT.
    probs = probe.predict_proba(hidden_np)[:, 0] 
    
    # Map tokens to probabilities
    trace = []
    for tok_id, prob in zip(response_ids, probs):
        trace.append({
            "token_id": int(tok_id),
            "token": tokenizer.decode([tok_id]),
            "insufficiency_prob": round(float(prob), 4)
        })
        
    return trace

def run_dynamic_tracking(args):
    print("="*80)
    print("EXPERIMENT 2B: DYNAMIC TEMPORAL TRACKING")
    print("="*80)

    # 1. Load the optimal layers dict
    best_layers_file = os.path.join(args.probing_dir, 'best_layers_linear.json')
    if not os.path.exists(best_layers_file):
        print(f"Error: Could not find {best_layers_file}. Run run_all_probes.py first.")
        sys.exit(1)
        
    with open(best_layers_file, 'r') as f:
        best_layers_dict = json.load(f)

    # Allow model to use max context if needed (aligns with vLLM config)
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        model_dir = Path(args.gen_dir) / args.domain / model_slug
        
        if not model_dir.exists():
            print(f"Skipping {model_slug} - No generation directory found.")
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*60}")
        
        # Check if we actually need to process any datasets for this model
        datasets_to_process = []
        for ds_name in DATASETS:
            gen_file = model_dir / f"{ds_name}_cot_generations.json"
            trace_file = model_dir / f"{ds_name}_temporal_traces.json"
            if gen_file.exists() and not trace_file.exists():
                datasets_to_process.append(ds_name)
                
        if not datasets_to_process and not args.test:
            print(f"  ✓ All temporal traces already generated for {model_slug}.")
            continue
            
        # Load Model & Tokenizer lazily
        print(f"  → Loading HF Model & Tokenizer to {args.device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=args.device, 
                dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            model.eval()
        except Exception as e:
            print(f"  ! Failed to load model {model_name}: {e}")
            continue

        for ds_name in datasets_to_process:
            gen_file = model_dir / f"{ds_name}_cot_generations.json"
            trace_file = model_dir / f"{ds_name}_temporal_traces.json"
            
            # Identify Best Layer from Exp 1
            # We assume probe was trained on the in-domain train set (e.g. train_on_umwp)
            train_config = f"train_on_{ds_name}"
            try:
                best_layer = best_layers_dict[model_name][train_config]['best_layer']
            except KeyError:
                print(f"  ! Warning: Best layer not found for {model_name} on {train_config}. Falling back to layer 16.")
                best_layer = 16
                
            print(f"\n  → Dataset: {ds_name.upper()} | Best Layer: {best_layer}")
            
            # Train Probe
            try:
                probe = train_best_layer_probe(model_name, ds_name, best_layer, args.device)
            except Exception as e:
                print(f"  ! Probe training failed: {e}")
                continue

            # Load Generations
            with open(gen_file, 'r') as f:
                generations = json.load(f)
                
            if args.test: generations = generations[:args.test_samples]
                
            all_traces = []
            
            # Process Temporal Traces
            for item in tqdm(generations, desc=f"    Extracting Traces"):
                try:
                    trace = process_and_probe_generation(
                        item=item,
                        model=model,
                        tokenizer=tokenizer,
                        probe=probe,
                        best_layer=best_layer,
                        device=args.device
                    )
                    
                    all_traces.append({
                        "question_idx": item.get('question_idx', item.get('id', -1)),
                        "is_sufficient": item.get('is_sufficient', None),
                        "trace": trace
                    })
                except Exception as e:
                    print(f"    ! Failed to process item {item.get('question_idx')}: {e}")
                    all_traces.append({
                        "question_idx": item.get('question_idx', item.get('id', -1)),
                        "error": str(e)
                    })

            # Save Output
            with open(trace_file, 'w') as f:
                json.dump(all_traces, f, indent=2)
            print(f"  ✓ Saved traces to {trace_file}")

        # Aggressive memory cleanup before loading next model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='math')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--probing_dir', type=str, default='experiments/probing', help="Path to best_layers_linear.json")
    parser.add_argument('--gen_dir', type=str, default='experiments/dynamic_tracking', help="Path to CoT generations")
    parser.add_argument('--test', action='store_true', help="Run on a tiny subset for testing")
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    run_dynamic_tracking(args)