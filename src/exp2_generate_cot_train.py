"""
================================================================================
EXPERIMENT 2A: Dynamic CoT Generation (Training Set)
================================================================================

THE TASK:
Generate step-by-step Chain-of-Thought (CoT) traces exclusively for the Train set 
to build a gold-standard, unbiased probe training pipeline.

PIPELINE ARCHITECTURE:
* Uses vLLM for high-throughput batch generation.
* Performs balanced downsampling (50/50 Valid/Broken) on the train set.
* Consistently caps at 3,000 samples across all datasets to ensure probe 
  generalizability and comparability across domains.
* Outputs cleanly to `[dataset]_cot_generations.json` inside the isolated
  train directory to ensure seamless loading in downstream scripts.
================================================================================
"""

import argparse
import json
import os
import sys
import gc
import random
from pathlib import Path
from tqdm import tqdm

# Allow vLLM to override the model's native context length constraints
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- CONFIGURATION ---
DATASETS = {
    'math': {
        'umwp': 'src/data/processed/insufficient_dataset_umwp/umwp_train.json',
        'treecut': 'src/data/processed/treecut/treecut_train.json'
    }
}

# Applied consistently across all datasets for methodological rigor
MAX_TRAIN_SAMPLES = 3000 

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

def format_cot_prompt(question, model_name):
    """
    Formats the prompt neutrally, requesting step-by-step reasoning and providing
    a structured escape hatch (\boxed{Insufficient}).
    """
    model_lower = model_name.lower()
    
    instruction = (
        "Solve the following math problem step-by-step. "
        "If the problem can be solved, enclose your final numerical answer in \\boxed{}. "
        "If the problem lacks sufficient information to reach a definitive answer, "
        "enclose the word 'Insufficient' in \\boxed{Insufficient}."
    )
    
    if "think" in model_lower:
        return f"<|user|>\n{instruction}\n\nProblem: {question}\n<|assistant|>\n<|thought|>\n"
        
    is_inst = "instruct" in model_lower or "chat" in model_lower
    if is_inst:
        return (
            f"<|im_start|>user\n{instruction}\n\nProblem: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
    return f"{instruction}\n\nProblem: {question}\n\nSolution:"

def balance_and_sample(data, max_samples):
    """
    Ensures the training set has a mathematically perfect 50/50 balance of 
    Sufficient and Insufficient problems to prevent linear probe bias.
    """
    sufficient = [x for x in data if x.get('is_sufficient', True)]
    insufficient = [x for x in data if not x.get('is_sufficient', True)]
    
    # Calculate target per class (half of max_samples)
    target_per_class = max_samples // 2
    
    # We can only sample up to what actually exists
    n_suff = min(len(sufficient), target_per_class)
    n_insuff = min(len(insufficient), target_per_class)
    
    # Deterministic sampling for reproducibility
    random.seed(42)
    sampled = random.sample(sufficient, n_suff) + random.sample(insufficient, n_insuff)
    
    # Shuffle the final merged list to prevent sequential bias during extraction
    random.shuffle(sampled)
    
    print(f"    [Train Sample] Total: {len(sampled)} | Sufficient: {n_suff} | Insufficient: {n_insuff}")
    return sampled

def run_cot_generation(args):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vLLM is required. Please install it or run on the correct node.")
        sys.exit(1)

    print(f"\n[PHASE: CoT GENERATION (TRAIN SET)] Loading models for {args.domain}...")
    save_interval = 50 

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        model_dir = Path(args.output_dir) / args.domain / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
        try:
            llm = LLM(model=model_name, tensor_parallel_size=args.tp, trust_remote_code=True, max_model_len=8192)
        except Exception as e:
            print(f"    ! Failed to load {model_name}: {e}")
            continue
        
        for ds_name, ds_path in DATASETS[args.domain].items():
            
            # Keep standard naming since it is physically isolated in the _train directory
            ds_save_path = model_dir / f"{ds_name}_cot_generations.json"
            
            if not os.path.exists(ds_path):
                print(f"    ! Source file missing: {ds_path}")
                continue
                
            with open(ds_path, 'r') as f:
                raw_data = json.load(f)
            
            if args.test: 
                data = raw_data[:args.test_samples]
            else:
                data = balance_and_sample(raw_data, MAX_TRAIN_SAMPLES)

            gen_results = []
            start_idx = 0

            # Load partial progress to allow safe resuming
            if ds_save_path.exists() and not args.test:
                try:
                    with open(ds_save_path, 'r') as f:
                        gen_results = json.load(f)
                    start_idx = len(gen_results)
                    if start_idx > 0 and start_idx < len(data):
                        print(f"    Resuming {ds_name} from query {start_idx}/{len(data)}")
                except json.JSONDecodeError:
                    print(f"    Warning: {ds_save_path} is corrupted. Starting from scratch.")

            if start_idx >= len(data):
                print(f"    - {ds_name} already completed. Skipping.")
                continue

            max_tokens = 1024 
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            for i in tqdm(range(start_idx, len(data), save_interval), desc=f"Generating CoT for {ds_name}"):
                chunk = data[i:i + save_interval]
                prompts = [format_cot_prompt(item['question'], model_name) for item in chunk]
                
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                for j, (item, prompt_text, out) in enumerate(zip(chunk, prompts, outputs)):
                    actual_idx = item.get('idx', item.get('id', i + j))
                    gen_results.append({
                        "question_idx": actual_idx, 
                        "question": item['question'],
                        "is_sufficient": item.get('is_sufficient', None),
                        "prompt": prompt_text,
                        "generated_response": out.outputs[0].text
                    })
                
                with open(ds_save_path, 'w') as f:
                    json.dump(gen_results, f, indent=2)

        # Free GPU memory heavily before loading the next 14B model
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='math')
    parser.add_argument('--tp', type=int, default=1) # Tensor Parallel size
    parser.add_argument('--output_dir', type=str, default='experiments/dynamic_tracking_train')
    parser.add_argument('--test', action='store_true', help="Run a quick 5-sample debug test")
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    run_cot_generation(args)