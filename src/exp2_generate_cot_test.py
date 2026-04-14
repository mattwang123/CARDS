"""
================================================================================
EXPERIMENT 2A: Dynamic CoT Generation (Actual Solving)
================================================================================

THE TASK:
Unlike the binary "Can this be solved?" prompt, here we force the model to 
actually solve the problem step-by-step. We provide a neutral formatting rule
to output \boxed{Insufficient} if they realize information is missing.

PIPELINE ARCHITECTURE:
* Uses vLLM for high-throughput batch generation.
* Saves full textual responses to `[dataset]_cot_generations.json`.
* Incremental checkpointing prevents data loss on large models.
================================================================================
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from tqdm import tqdm

# Allow vLLM to override the model's native context length constraints
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- CONFIGURATION ---
DATASETS = {
    'math': {
        'umwp': 'src/data/processed/insufficient_dataset_umwp/umwp_test.json',
        'treecut': 'src/data/processed/treecut/treecut_test.json'
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

def format_cot_prompt(question, model_name):
    """
    Formats the prompt neutrally, requesting step-by-step reasoning and providing
    a structured escape hatch (\boxed{Insufficient}) without over-priming suspicion.
    """
    model_lower = model_name.lower()
    
    instruction = (
        "Solve the following math problem step-by-step. "
        "If the problem can be solved, enclose your final numerical answer in \\boxed{}. "
        "If the problem lacks sufficient information to reach a definitive answer, "
        "enclose the word 'Insufficient' in \\boxed{Insufficient}."
    )
    
    # Special handling for reasoning/think models (e.g., OLMo-3 Think)
    if "think" in model_lower:
        return f"<|user|>\n{instruction}\n\nProblem: {question}\n<|assistant|>\n<|thought|>\n"
        
    # General Instruct/Chat handling (Qwen, Llama)
    is_inst = "instruct" in model_lower or "chat" in model_lower
    if is_inst:
        return (
            f"<|im_start|>user\n{instruction}\n\nProblem: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
    # Base Model Completion Template
    return f"{instruction}\n\nProblem: {question}\n\nSolution:"

def run_cot_generation(args):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vLLM is required for generation. Please install it or run on the correct node.")
        sys.exit(1)

    print(f"\n[PHASE: CoT GENERATION] Loading models for {args.domain}...")
    
    save_interval = 20 # Save progress every 20 queries

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        model_dir = Path(args.output_dir) / args.domain / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        # Check completion status
        all_done = True
        for ds_name in DATASETS[args.domain]:
            ds_save_path = model_dir / f"{ds_name}_cot_generations.json"
            if not ds_save_path.exists():
                all_done = False
                break
            else:
                with open(DATASETS[args.domain][ds_name], 'r') as f:
                    total_expected = len(json.load(f)) if not args.test else args.test_samples
                try:
                    with open(ds_save_path, 'r') as f:
                        total_generated = len(json.load(f))
                except json.JSONDecodeError:
                    total_generated = 0
                if total_generated < total_expected:
                    all_done = False
                    break

        if all_done and not args.test:
            print(f"  - Skipping {model_slug}, all CoT datasets fully generated.")
            continue

        print(f"  - Loading {model_name}...")
        try:
            # Note: max_model_len restricted to save KV cache memory as seen in previous logs
            llm = LLM(model=model_name, tensor_parallel_size=args.tp, trust_remote_code=True, max_model_len=8192)
        except Exception as e:
            print(f"    ! Failed to load {model_name}: {e}")
            continue
        
        for ds_name, ds_path in DATASETS[args.domain].items():
            ds_save_path = model_dir / f"{ds_name}_cot_generations.json"
            
            with open(ds_path, 'r') as f:
                data = json.load(f)
            if args.test: data = data[:args.test_samples]

            gen_results = []
            start_idx = 0

            # Load partial progress
            if ds_save_path.exists() and not args.test:
                try:
                    with open(ds_save_path, 'r') as f:
                        gen_results = json.load(f)
                    start_idx = len(gen_results)
                    if start_idx > 0:
                        print(f"    Resuming {ds_name} from query {start_idx}/{len(data)}")
                except json.JSONDecodeError:
                    print(f"    Warning: {ds_save_path} is corrupted. Starting from scratch.")

            if start_idx >= len(data):
                continue

            # CRITICAL: We need a much longer context for CoT generation than binary QA
            max_tokens = 1024 
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            for i in tqdm(range(start_idx, len(data), save_interval), desc=f"Generating CoT for {ds_name}"):
                chunk = data[i:i + save_interval]
                prompts = [format_cot_prompt(item['question'], model_name) for item in chunk]
                
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                for j, (item, prompt_text, out) in enumerate(zip(chunk, prompts, outputs)):
                    actual_idx = i + j
                    gen_results.append({
                        "question_idx": item.get('idx', item.get('id', actual_idx)), 
                        "question": item['question'],
                        "is_sufficient": item.get('is_sufficient', None),
                        "prompt": prompt_text,
                        "generated_response": out.outputs[0].text
                    })
                
                with open(ds_save_path, 'w') as f:
                    json.dump(gen_results, f, indent=2)

        # Better garbage collection to ensure GPU memory is freed between loop iterations
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='math')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='experiments/dynamic_tracking_test')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    run_cot_generation(args)