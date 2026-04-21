"""
================================================================================
EXPERIMENT 1B: Direct Binary Assessment (Scale & Generalization Revision)
================================================================================

THE TASK:
Models are given a prompt containing a problem and asked a direct question: 
"Can this problem be solved with the given information? Answer Yes or No."

PIPELINE ARCHITECTURE (Decoupled for reliability & scalability):
* PHASE 1 (Generation): Uses vLLM (for speed) or standard HF to generate answers. 
  Generations are saved incrementally to prevent data loss.
* PHASE 2 (Evaluation): 
  - Primary Judge: Strict Regex parsing (looks for boxed answers or explicit Yes/No).
  - Fallback Judge: LLM-as-a-judge. If the model rambles, an evaluator LLM 
    (GPT-4o-mini, or a local open-source model like Llama-3) parses the semantic intent.
  - Checkpointing: Saves progress every 20 items. Safe to interrupt/resume.

FEATURES:
- Cross-Domain: Evaluates Math (UMWP/TreeCut) and Non-Math datasets.
- Cross-Scale: Compares Base vs Instruct pairs from 1.5B to 14B+.
- Flexible Judge: Supports OpenAI or Local vLLM servers for the evaluator.
================================================================================
"""

import argparse
import json
import os
import sys
import re
import gc
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

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

# --- UTILS & PROMPTING ---
def format_prompt(question, model_name):
    """Native format for Instruct models, completion format for Base models, and specialized formatting for OLMo-Think."""
    model_lower = model_name.lower()
    
    # Special handling for reasoning/think models
    if "think" in model_lower:
        return (
            f"<|user|>\nProblem: {question}\n"
            f"Think step by step and then decide: Can this problem be solved with the given information? "
            f"End your response with 'Conclusion: Yes' or 'Conclusion: No'.\n<|assistant|>\n<|thought|>\n"
        )
    
    # OLMo Instruct handling
    if "olmo" in model_lower and "instruct" in model_lower:
        return (
            f"<|user|>\nProblem: {question}\n"
            f"Answer exactly 'Yes' or 'No'. Can this problem be solved with the given information?\n"
            f"<|assistant|>\n"
        )
        
    # General Instruct/Chat handling (Qwen, Llama)
    is_inst = "instruct" in model_lower or "chat" in model_lower
    if is_inst:
        return f"<|im_start|>system\nYou are a logical assistant. Answer exactly 'Yes' or 'No'.<|im_end|>\n<|im_start|>user\nProblem: {question}\n\nCan this problem be solved with the given information?<|im_end|>\n<|im_start|>assistant\n"
        
    # Base Model Completion Template (OLMo Base, Qwen Base, Llama Base)
    return f"Problem: {question}\nCan this problem be solved with the given information? Answer exactly Yes or No.\nAnswer:"

def extract_regex(output):
    """Primary intent extraction."""
    output = output.lower().strip()
    
    # Look for boxed or final answers
    match = re.search(r'(?:answer|final answer|conclusion|solvable) is[:\s]*\\*boxed\{([yn]o?)\}', output)
    
    # Support for reasoning conclusion format
    if not match:
        matches = re.findall(r'conclusion:\s*(yes|no)', output)
        if matches:
            return "Yes" if matches[-1].startswith('y') else "No"
            
    # Fallback to standard yes/no detection
    if not match:
        match = re.search(r'\b(yes|no)\b', output)
    
    if match:
        val = match.group(1) if hasattr(match, 'group') else match
        if isinstance(val, tuple): val = val[0] # Handle findall vs search edge cases
        return "Yes" if val.startswith('y') else "No"
    return None

# --- PHASE 1: GENERATION (With Granular Checkpointing) ---
def run_generation(args):
    from vllm import LLM, SamplingParams
    print(f"\n[PHASE: GENERATION] Loading models for {args.domain}...")
    
    save_interval = 20

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        model_dir = Path(args.output_dir) / args.domain / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        all_done = True
        for ds_name in DATASETS[args.domain]:
            ds_save_path = model_dir / f"{ds_name}_generations.json"
            if not ds_save_path.exists():
                all_done = False
                break
            else:
                with open(DATASETS[args.domain][ds_name], 'r') as f:
                    total_expected = len(json.load(f)) if not args.test else args.test_samples
                with open(ds_save_path, 'r') as f:
                    try:
                        total_generated = len(json.load(f))
                    except json.JSONDecodeError:
                        total_generated = 0
                if total_generated < total_expected:
                    all_done = False
                    break

        if all_done and not args.test:
            print(f"  - [RESUME HIT] Skipping {model_slug}, all datasets fully generated.")
            continue

        print(f"  - Loading {model_name}...")
        try:
            # Added max_model_len to safely fit massive models on dual A100s
            llm = LLM(model=model_name, tensor_parallel_size=args.tp, trust_remote_code=True, max_model_len=4096)
        except Exception as e:
            print(f"    ! Failed to load {model_name}: {e}")
            continue
        
        for ds_name, ds_path in DATASETS[args.domain].items():
            ds_save_path = model_dir / f"{ds_name}_generations.json"
            
            with open(ds_path, 'r') as f:
                data = json.load(f)
            if args.test: data = data[:args.test_samples]

            gen_results = []
            start_idx = 0

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

            max_tokens = 512 if "think" in model_name.lower() else 128
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            for i in tqdm(range(start_idx, len(data), save_interval), desc=f"Generating {ds_name}"):
                chunk = data[i:i + save_interval]
                prompts = [format_prompt(item['question'], model_name) for item in chunk]
                
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                for item, out in zip(chunk, outputs):
                    gen_results.append({
                        "question": item['question'],
                        "is_sufficient": item['is_sufficient'],
                        "model_output": out.outputs[0].text
                    })
                
                with open(ds_save_path, 'w') as f:
                    json.dump(gen_results, f, indent=2)

        # Nuclear Cleanup for 70B models
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

# --- PHASE 2: EVALUATION ---
def run_evaluation(args):
    print(f"\n[PHASE: EVALUATION] Judging existing results...")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=args.judge_base_url)

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        model_dir = Path(args.output_dir) / args.domain / model_slug
        
        for ds_name in DATASETS[args.domain].keys():
            gen_file = model_dir / f"{ds_name}_generations.json"
            final_file = model_dir / f"{ds_name}_results.json"
            checkpoint_file = model_dir / f"{ds_name}_results_checkpoint.json"
            
            if not gen_file.exists(): 
                continue
                
            if final_file.exists() and not args.test:
                print(f"  - Skipping {model_slug}/{ds_name}, evaluation already complete.")
                continue
            
            with open(gen_file, 'r') as f:
                generations = json.load(f)
            
            results = []
            stats = {"regex": 0, "llm": 0, "error": 0}
            
            # Load Checkpoint if it exists
            if checkpoint_file.exists() and not args.test:
                try:
                    with open(checkpoint_file, 'r') as f:
                        ckpt = json.load(f)
                    results = ckpt.get("data", [])
                    stats = ckpt.get("stats", {"regex": 0, "llm": 0, "error": 0})
                    print(f"    Resuming {model_slug}/{ds_name} from checkpoint: {len(results)}/{len(generations)} items processed.")
                except Exception as e:
                    print(f"    Warning: Corrupted checkpoint for {model_slug}/{ds_name}, starting fresh. ({e})")
                    results = []
            
            start_idx = len(results)
            
            if start_idx < len(generations):
                for idx in tqdm(range(start_idx, len(generations)), desc=f"Evaluating {model_slug}/{ds_name}"):
                    item = generations[idx]
                    gt = "Yes" if item['is_sufficient'] else "No"
                    pred = extract_regex(item['model_output'])
                    
                    judge_type = "regex"
                    if pred is None:
                        # Fallback to LLM Judge
                        judge_type = "llm"
                        try:
                            resp = client.chat.completions.create(
                                model=args.judge_model,
                                messages=[{"role": "user", "content": f"Does the following model response mean Yes or No? Output only JSON with 'verdict': 'Yes' or 'No'.\nResponse: {item['model_output']}"}],
                                response_format={"type": "json_object"} if "gpt" in args.judge_model.lower() else None
                            )
                            # Clean output for local models that might wrap JSON in markdown
                            content = resp.choices[0].message.content.strip()
                            if content.startswith("```json"): content = content[7:]
                            if content.endswith("```"): content = content[:-3]
                            
                            pred = json.loads(content).get('verdict', "Error")
                        except Exception as e:
                            pred = "Error"

                    if pred == "Error": stats["error"] += 1
                    else: stats[judge_type] += 1
                    
                    item['judgment'] = {"pred": pred, "gt": gt, "correct": pred == gt, "judge_type": judge_type}
                    results.append(item)
                    
                    # Save checkpoint every 20 iterations
                    if (idx + 1) % 20 == 0 or (idx + 1) == len(generations):
                        with open(checkpoint_file, 'w') as f:
                            json.dump({"stats": stats, "data": results}, f, indent=2)

            # Final Score Calculation
            correct = sum(1 for r in results if r['judgment']['correct'])
            acc = correct / len(results) if len(results) > 0 else 0
            
            with open(final_file, 'w') as f:
                json.dump({"accuracy": acc, "stats": stats, "data": results}, f, indent=2)
            
            # Clean up checkpoint on success
            if checkpoint_file.exists() and not args.test:
                checkpoint_file.unlink()
                
            print(f"  - {model_slug}/{ds_name}: {acc*100:.1f}% (Regex: {stats['regex']}, LLM: {stats['llm']})")

# --- MAIN ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['gen_only', 'eval_only', 'all'], required=True)
    parser.add_argument('--domain', type=str, default='math')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='/export/fs06/hwang302/CARDS/exp_verbalization')
    parser.add_argument('--judge_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--judge_base_url', type=str, default='http://e02:8000/v1')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=2)
    args = parser.parse_args()

    if args.mode in ['gen_only', 'all']:
        run_generation(args)
    
    if args.mode in ['eval_only', 'all']:
        run_evaluation(args)