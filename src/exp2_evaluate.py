"""
================================================================================
EXPERIMENT 2C: Comprehensive Evaluation & Trace Analysis
================================================================================

THE TASK:
Evaluate the Chain-of-Thought (CoT) responses generated in Stage A and correlate 
them with the temporal logic traces extracted in Stage B. 

PIPELINE ARCHITECTURE:
1. Instant Resume: Checks for completion before executing heavy I/O JSON loads.
2. Data Merging: Loads Ground Truth, CoT Generations, and Temporal Traces.
3. Regex Extraction: Naively extracts the \\boxed{} content (NO evaluation).
4. LLM Judging: A 70B judge determines if the extracted text is mathematically
   equivalent to the ground truth (if Sufficient) or correctly states the 
   missing info (if Insufficient).
5. Trace Analytics: Computes 12+ temporal metrics (mean, max, end_prob, 
   collapse_delta, etc.) for mechanistic interpretation.
6. Checkpointing: Saves progress aggressively to avoid losing LLM API calls.
================================================================================
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
import numpy as np
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

def extract_regex_naive(text):
    """
    Pure extraction phase (Regex Only). 
    Dumbly extracts \\boxed{} content or fallback final sentences. No judgment here.
    """
    if not text:
        return ""
        
    # Standard boxed extraction
    box_match = re.search(r'\\boxed\{([^{}]+)\}', text)
    if box_match:
        return box_match.group(1).strip()
        
    # Nested boxed extraction fallback (e.g., \boxed{\text{Insufficient}})
    nested_box = re.search(r'\\boxed\{(.*?)\}', text)
    if nested_box:
        return nested_box.group(1).strip()
        
    # Common fallback phrases
    fallback_match = re.search(r'(?:[Tt]he final answer is|[Cc]onclusion:?|[Aa]nswer:?)\s*\*?([^\.\n]+)', text)
    if fallback_match:
        return fallback_match.group(1).strip()
        
    # Ultimate fallback: return the last 200 characters to let the LLM figure it out
    return text[-200:].strip()

def calculate_trace_metrics(trace):
    """
    Calculates 12+ temporal metrics for mechanistic interpretation of token-by-token latent states.
    Handles noise in autoregressive token generation.
    """
    if not trace or len(trace) == 0:
        return {
            "trace_mean": None, "trace_median": None, "trace_max": None, "trace_min": None,
            "trace_variance": None, "trace_start": None, "trace_end": None, "collapse_delta": None,
            "trace_slope": None, "volatility_index": None, "suspicion_duration_ratio": None,
            "auc_insuf": None, "peak_location_relative": None
        }
        
    if isinstance(trace[0], dict):
        trace_values = [float(step.get('insufficiency_prob', 0.0)) for step in trace]
    else:
        # Fallback if trace is already a list of floats
        trace_values = trace
        
    t = np.array(trace_values, dtype=float)
    
    n_tokens = len(t)
    
    # 1. Global Aggregates
    trace_mean = np.mean(t)
    trace_median = np.median(t)
    trace_max = np.max(t)
    trace_min = np.min(t)
    trace_variance = np.var(t)
    
    # 2. Positional Anchors
    N = min(3, n_tokens)
    trace_start = np.mean(t[:N])
    trace_end = np.mean(t[-N:])
    collapse_delta = trace_max - trace_end
    
    # 3. Dynamic & Trend Metrics
    if n_tokens > 1:
        x = np.arange(n_tokens)
        slope, _ = np.polyfit(x, t, 1)
        volatility_index = np.mean(np.abs(np.diff(t)))
    else:
        slope = 0.0
        volatility_index = 0.0
        
    # 4. Threshold & Area Metrics
    suspicion_duration_ratio = np.mean(t > 0.5)
    auc_insuf = np.trapz(t) / max(1, n_tokens - 1) if n_tokens > 1 else t[0]
    peak_location_relative = np.argmax(t) / n_tokens
    
    return {
        "trace_mean": float(trace_mean),
        "trace_median": float(trace_median),
        "trace_max": float(trace_max),
        "trace_min": float(trace_min),
        "trace_variance": float(trace_variance),
        "trace_start": float(trace_start),
        "trace_end": float(trace_end),
        "collapse_delta": float(collapse_delta),
        "trace_slope": float(slope),
        "volatility_index": float(volatility_index),
        "suspicion_duration_ratio": float(suspicion_duration_ratio),
        "auc_insuf": float(auc_insuf),
        "peak_location_relative": float(peak_location_relative)
    }

def get_llm_judgment(client, model_name, extracted_text, is_sufficient, ground_truth=None):
    """
    Queries the evaluator LLM to semantically judge the extracted text against the ground truth.
    """
    if is_sufficient:
        sys_msg = "You are a strict mathematical evaluator. Answer only with a JSON object: {\"correct\": true} or {\"correct\": false}."
        prompt = (
            f"Ground Truth: {ground_truth}\n"
            f"Model Extracted Answer: {extracted_text}\n\n"
            f"Are these two mathematical answers perfectly equivalent? Consider floating-point tolerances, "
            f"fractions vs decimals, and minor unit variations as equivalent. If the model says 'Insufficient' or "
            f"fails to provide a number, it is false. Output only JSON."
        )
    else:
        sys_msg = "You are a strict logical evaluator. Answer only with a JSON object: {\"correct\": true} or {\"correct\": false}."
        prompt = (
            f"Model Extracted Answer: {extracted_text}\n\n"
            f"Does this extracted text explicitly state that the problem lacks sufficient information to be solved "
            f"(e.g., using words like 'Insufficient', 'Not enough info', 'Cannot be determined')? "
            f"If it attempts to provide a final numerical answer or a hallucinated value, output false. "
            f"Output only JSON."
        )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=64,
            response_format={"type": "json_object"} if "gpt" in model_name.lower() else None
        )
        content = resp.choices[0].message.content.strip()
        
        # Clean markdown if present
        if content.startswith("```json"): 
            content = content[7:]
        elif content.startswith("```"): 
            content = content[3:]
            
        if content.endswith("```"): 
            content = content[:-3]
            
        verdict = json.loads(content.strip()).get('correct', False)
        return bool(verdict), "success"
    except Exception as e:
        return False, f"error: {str(e)}"

def run_evaluation(args):
    print(f"\n[PHASE: EVALUATION & TRACE ANALYSIS] Initializing Judge: {args.judge_model}")
    load_dotenv()
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), 
        base_url=args.judge_base_url
    )

    for model_name in MODELS:
        model_slug = model_name.split('/')[-1]
        
        gen_model_dir = Path(args.generation_dir) / args.domain / model_slug
        eval_model_dir = Path(args.output_dir) / args.domain / model_slug
        gen_model_dir.mkdir(parents=True, exist_ok=True)
        eval_model_dir.mkdir(parents=True, exist_ok=True)
        
        for ds_name, ds_path in DATASETS[args.domain].items():
            cot_file = gen_model_dir / f"{ds_name}_cot_generations.json"
            trace_file = gen_model_dir / f"{ds_name}_temporal_traces.json"
            final_file = eval_model_dir / f"{ds_name}_evaluated_traces.json"
            ckpt_file = eval_model_dir / f"{ds_name}_eval_checkpoint.json"
            
            # --- ULTIMATE RESUME: Check completion BEFORE heavy I/O ---
            if final_file.exists() and not args.test:
                print(f"  - [RESUME HIT] Skipping {model_slug}/{ds_name}, evaluation already complete.")
                continue

            if not cot_file.exists():
                print(f"  ! Missing CoT generations for {model_slug}/{ds_name} at {cot_file}")
                continue
                
            # Load Datasets
            with open(ds_path, 'r') as f:
                raw_gt = json.load(f)
                gt_data = {str(item.get('idx', item.get('id', i))): item for i, item in enumerate(raw_gt)}
                
            with open(cot_file, 'r') as f:
                cot_generations = json.load(f)
                if args.test: 
                    cot_generations = cot_generations[:args.test_samples]
                
            # Load Traces if available, else gracefully proceed with empty traces
            traces = {}
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                    # Support multiple formats of trace saving
                    if isinstance(trace_data, dict):
                        traces = {str(k): v for k, v in trace_data.items()}
                    elif isinstance(trace_data, list):
                        traces = {str(item.get('question_idx', item.get('id', i))): item.get('trace', []) for i, item in enumerate(trace_data)}
            else:
                print(f"  ~ Warning: Traces file missing for {model_slug}/{ds_name}. Temporal metrics will be null.")

            results = []
            start_idx = 0
            stats = {"q1": 0, "q2": 0, "q3": 0, "q4": 0, "errors": 0}

            if ckpt_file.exists() and not args.test:
                try:
                    with open(ckpt_file, 'r') as f:
                        ckpt = json.load(f)
                    results = ckpt.get("data", [])
                    stats = ckpt.get("stats", stats)
                    start_idx = len(results)
                    if start_idx > 0:
                        print(f"    Resuming {model_slug}/{ds_name} from checkpoint: {start_idx}/{len(cot_generations)}")
                except Exception as e:
                    print(f"    Warning: Corrupted checkpoint, starting fresh. ({e})")
                    results = []

            for i in tqdm(range(start_idx, len(cot_generations)), desc=f"Eval {model_slug}/{ds_name}"):
                gen_item = cot_generations[i]
                q_idx = str(gen_item.get('question_idx', gen_item.get('id', i)))
                
                # 1. Merge Data
                gt_item = gt_data.get(q_idx, {})
                
                # Fallbacks for keys depending on dataset schema
                is_sufficient = gt_item.get('is_sufficient', gen_item.get('is_sufficient'))
                if is_sufficient is None:
                    is_sufficient = True # Default assumption if missing
                    
                ground_truth = gt_item.get('answer', gt_item.get('ground_truth', ''))
                cot_text = gen_item.get('generated_response', gen_item.get('model_output', ''))
                trace = traces.get(q_idx, [])
                
                # 2. Extract Regex
                extracted_raw_text = extract_regex_naive(cot_text)
                
                # 3. LLM Judgment
                judge_correct, status = get_llm_judgment(
                    client=client, 
                    model_name=args.judge_model,
                    extracted_text=extracted_raw_text,
                    is_sufficient=is_sufficient,
                    ground_truth=ground_truth
                )
                
                if status != "success":
                    stats["errors"] += 1
                
                # 4. Epistemic Quadrant Mapping
                quadrant = "Unknown"
                if not is_sufficient and not judge_correct:
                    quadrant = "Q1_Hallucination"
                    stats["q1"] += 1
                elif not is_sufficient and judge_correct:
                    quadrant = "Q2_Correct_Rejection"
                    stats["q2"] += 1
                elif is_sufficient and judge_correct:
                    quadrant = "Q3_Solved_Correctly"
                    stats["q3"] += 1
                elif is_sufficient and not judge_correct:
                    quadrant = "Q4_Competence_Failure"
                    stats["q4"] += 1
                    
                # 5. Trace Analytics
                trace_metrics = calculate_trace_metrics(trace)
                
                # Construct final row
                result_row = {
                    "question_idx": q_idx,
                    "is_sufficient": is_sufficient,
                    "ground_truth": ground_truth,
                    "extracted_raw_text": extracted_raw_text,
                    "judge_correct": judge_correct,
                    "epistemic_quadrant": quadrant,
                    "judge_status": status,
                    **trace_metrics
                }
                
                # Keep payload light, optionally drop full CoT string to save disk space 
                if args.keep_cot:
                    result_row["full_cot_text"] = cot_text
                    
                results.append(result_row)
                
                # Checkpoint every 20 iterations
                if (i + 1) % 20 == 0 or (i + 1) == len(cot_generations):
                    with open(ckpt_file, 'w') as f:
                        json.dump({"stats": stats, "data": results}, f, indent=2)

            # Final Save & Cleanup
            total = len(results)
            if total > 0:
                acc_suff = stats["q3"] / (stats["q3"] + stats["q4"]) if (stats["q3"] + stats["q4"]) > 0 else 0.0
                acc_insuff = stats["q2"] / (stats["q1"] + stats["q2"]) if (stats["q1"] + stats["q2"]) > 0 else 0.0
                
                summary = {
                    "accuracy_sufficient": float(acc_suff),
                    "accuracy_insufficient": float(acc_insuff),
                    "quadrant_counts": stats,
                    "data": results
                }
                with open(final_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                    
            if ckpt_file.exists() and not args.test:
                ckpt_file.unlink()
                
            print(f"  - {model_slug}/{ds_name} Complete | Q1(Hallucinate):{stats['q1']}, Q2(Reject):{stats['q2']}, Q3(Solve):{stats['q3']}, Q4(Fail):{stats['q4']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='math')
    # Note: Using the directory you provided in the code, but remember if you saved
    # test generations in 'experiments/dynamic_tracking_test', pass that flag!
    parser.add_argument('--generation_dir', type=str, default='experiments/dynamic_tracking_test')
    parser.add_argument('--output_dir', type=str, default='experiments/dynamic_tracking_test_evaluation')
    parser.add_argument('--judge_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--judge_base_url', type=str, default='http://e03:8000/v1')
    parser.add_argument('--keep_cot', action='store_true', help="Retain full CoT text in output JSON")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    run_evaluation(args)