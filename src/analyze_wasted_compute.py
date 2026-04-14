"""
================================================================================
COMPUTE WASTE ANALYSIS (The "So What?" for Test-Time Compute)
================================================================================
Parses the output of exp2_evaluate.py to calculate the exact number of tokens
wasted on confident hallucinations (Quadrant 1), proving the efficiency value 
of the Latent Watchdog.
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def analyze_wasted_compute(model_name, eval_file_path):
    print(f"\n{'='*60}")
    print(f"COMPUTE WASTE ANALYSIS: {model_name}")
    print(f"{'='*60}")

    with open(eval_file_path, 'r') as f:
        data = json.load(f).get("data", [])

    if not data:
        print("No evaluation data found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    stats = {
        "Q1_Hallucination": {"count": 0, "tokens": 0},
        "Q2_Correct_Rejection": {"count": 0, "tokens": 0},
        "Q3_Solved_Correctly": {"count": 0, "tokens": 0},
        "Q4_False_Rejection": {"count": 0, "tokens": 0}
    }

    for item in data:
        quadrant = item.get("epistemic_quadrant", "Unknown")
        generated_text = item.get("generated_cot", "")
        
        # Count actual tokens generated during test-time compute
        token_count = len(tokenizer.encode(generated_text))
        
        if quadrant in stats:
            stats[quadrant]["count"] += 1
            stats[quadrant]["tokens"] += token_count

    # Calculate Impact Metrics
    total_insufficient = stats["Q1_Hallucination"]["count"] + stats["Q2_Correct_Rejection"]["count"]
    wasted_tokens = stats["Q1_Hallucination"]["tokens"]
    
    print(f"[BASELINE GENERATION COSTS]")
    print(f"Total Insufficient Problems   : {total_insufficient}")
    print(f"Hallucination Rate (Q1)       : {100 * stats['Q1_Hallucination']['count'] / max(1, total_insufficient):.1f}%")
    print(f"Total Tokens Wasted on Q1     : {wasted_tokens:,} tokens")
    print(f"Average Wasted Tokens per Q1  : {wasted_tokens / max(1, stats['Q1_Hallucination']['count']):.1f} tokens/query\n")

    print(f"[LATENT WATCHDOG INTERVENTION (Assumes t=0 Probe with 90% F1)]")
    # If our static probe from Exp 1 catches 90% of these at Token 0:
    saved_tokens = int(wasted_tokens * 0.90)
    print(f"Intervention Trigger Point    : Token 0 (Prompt Phase)")
    print(f"Compute Saved                 : {saved_tokens:,} tokens")
    print(f"Efficiency Gain               : {100 * saved_tokens / max(1, wasted_tokens):.1f}% reduction in wasted FLOPs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-7B-Instruct')
    parser.add_argument('--eval_file', type=str, required=True, help="Path to *_evaluated_traces.json")
    args = parser.parse_args()
    
    analyze_wasted_compute(args.model, args.eval_file)