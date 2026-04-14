"""
================================================================================
EXPERIMENT 2D: Impact Analysis & Compute Waste Quantification
================================================================================
This script analyzes the generative momentum hypothesis and calculates
the exact compute savings of the "Latent Watchdog" early-stopping intervention.

It generates a publication-ready multi-panel figure containing:
1. Q1 Trace Heatmap (Generative Momentum)
2. Quadrant Probability Distributions
3. Compute Efficiency Matrix (Standard CoT vs Latent Watchdog)
================================================================================
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from pathlib import Path

def load_and_merge(eval_path, trace_path):
    """Merges the evaluation verdicts with the raw token-by-token traces."""
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)["data"]
    
    with open(trace_path, 'r') as f:
        trace_data = json.load(f)

    # Convert traces to a lookup dictionary
    if isinstance(trace_data, dict):
        trace_lookup = {str(k): v for k, v in trace_data.items()}
    else:
        trace_lookup = {str(item.get('question_idx', item.get('id'))): item.get('trace', []) for item in trace_data}

    merged = []
    for e in eval_data:
        q_idx = str(e["question_idx"])
        raw_trace = trace_lookup.get(q_idx, [])
        
        # Extract probabilities
        probs = [float(step.get("insufficiency_prob", 0)) for step in raw_trace if isinstance(step, dict)]
        
        if len(probs) > 5: # Only keep valid traces
            merged.append({
                "question_idx": q_idx,
                "quadrant": e["epistemic_quadrant"],
                "is_sufficient": e["is_sufficient"],
                "probs": probs,
                "total_tokens": len(probs)
            })
    return merged

def smooth_and_interpolate(probs, target_length=100, window_size=5):
    """Applies a moving average and scales the trace to a 0-100% relative timeline."""
    if len(probs) >= window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(probs, kernel, mode='valid')
    else:
        smoothed = np.array(probs)

    x_old = np.linspace(0, 1, num=len(smoothed))
    x_new = np.linspace(0, 1, num=target_length)
    
    interpolator = interp1d(x_old, smoothed, kind='linear', bounds_error=False, fill_value="extrapolate")
    return interpolator(x_new)

def simulate_intervention(merged_data, k_window=20, threshold=0.5):
    """
    Simulates the Latent Watchdog: If the probe fires > threshold within the 
    first k_window tokens, we halt generation.
    """
    results = {
        "total_wasted_baseline": 0,
        "total_wasted_intervention": 0,
        "q1_prevented": 0,
        "q1_total": 0,
        "q3_false_positives": 0, # Solvable problems we accidentally stopped
        "q3_total": 0
    }

    for item in merged_data:
        trace = item["probs"]
        quad = item["quadrant"]
        tokens = item["total_tokens"]

        # Track base populations
        if quad == "Q1_Hallucination":
            results["q1_total"] += 1
            results["total_wasted_baseline"] += tokens
        elif quad == "Q3_Solved_Correctly":
            results["q3_total"] += 1

        # Simulate early stopping
        triggered = False
        stop_token = tokens
        
        # Check first K tokens
        for idx, p in enumerate(trace[:k_window]):
            if p >= threshold:
                triggered = True
                stop_token = idx + 1
                break

        if quad == "Q1_Hallucination":
            if triggered:
                results["q1_prevented"] += 1
                results["total_wasted_intervention"] += stop_token # Only wasted up to the stop point
            else:
                results["total_wasted_intervention"] += tokens # Failed to catch, wasted all
                
        elif quad == "Q3_Solved_Correctly" and triggered:
            # We accidentally stopped a correct answer!
            results["q3_false_positives"] += 1

    # Calculate ratios safely
    results["compute_saved_pct"] = 100 * (1 - (results["total_wasted_intervention"] / max(1, results["total_wasted_baseline"])))
    results["hallucination_catch_rate"] = 100 * (results["q1_prevented"] / max(1, results["q1_total"]))
    results["false_positive_rate"] = 100 * (results["q3_false_positives"] / max(1, results["q3_total"]))

    return results

def run_impact_analysis(model_name, eval_file, trace_file, k_window=15, threshold=0.6):
    print(f"\n{'='*60}")
    print(f"RUNNING IMPACT ANALYSIS: {model_name}")
    print(f"{'='*60}")

    merged_data = load_and_merge(eval_file, trace_file)
    if not merged_data:
        print("Error: No valid traces found or merge failed.")
        return

    # --- 1. Compute Analytics ---
    sim_results = simulate_intervention(merged_data, k_window=k_window, threshold=threshold)
    
    print("\n[LATENT WATCHDOG INTERVENTION RESULTS]")
    print(f"Intervention Window : First {k_window} tokens")
    print(f"Activation Threshold: Probe Prob >= {threshold}")
    print("-" * 40)
    print(f"Baseline Wasted Tokens (Q1) : {sim_results['total_wasted_baseline']:,}")
    print(f"Intervention Wasted Tokens  : {sim_results['total_wasted_intervention']:,}")
    print(f"-> COMPUTE SAVED            : {sim_results['compute_saved_pct']:.2f}%")
    print("-" * 40)
    print(f"Hallucinations Prevented    : {sim_results['hallucination_catch_rate']:.1f}% ({sim_results['q1_prevented']}/{sim_results['q1_total']})")
    print(f"False Positive Rate (on Q3) : {sim_results['false_positive_rate']:.1f}% ({sim_results['q3_false_positives']}/{sim_results['q3_total']})")
    
    # --- 2. Visualization Setup ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.2])

    # PANEL A: The Heatmap (Generative Momentum)
    ax0 = fig.add_subplot(gs[0])
    q1_traces = [smooth_and_interpolate(item["probs"]) for item in merged_data if item["quadrant"] == "Q1_Hallucination"]
    
    if q1_traces:
        # Sort traces by their initial suspicion (first 10%) so the hottest are at the top
        q1_traces.sort(key=lambda x: np.mean(x[:10]), reverse=True)
        heatmap_data = np.vstack(q1_traces)
        
        sns.heatmap(heatmap_data, cmap="coolwarm", cbar_kws={'label': 'Probe Probability'}, ax=ax0, vmin=0, vmax=1)
        ax0.set_title("Generative Momentum: Q1 Hallucination Traces", fontweight='bold')
        ax0.set_ylabel("Individual Generation Samples (Sorted)")
        ax0.set_xlabel("Normalized Generation Length (%)")
        ax0.set_yticks([]) # Hide y-ticks as they are just sample IDs
    else:
        ax0.text(0.5, 0.5, "No Q1 Traces Found", ha='center', va='center')

    # PANEL B: Initial State Distribution (They knew all along)
    ax1 = fig.add_subplot(gs[1])
    q1_initials = [np.mean(item["probs"][:5]) for item in merged_data if item["quadrant"] == "Q1_Hallucination"]
    q3_initials = [np.mean(item["probs"][:5]) for item in merged_data if item["quadrant"] == "Q3_Solved_Correctly"]
    
    sns.kdeplot(q3_initials, fill=True, color="blue", label="Q3: Solved (Sufficient)", ax=ax1, alpha=0.5)
    sns.kdeplot(q1_initials, fill=True, color="red", label="Q1: Hallucinated (Insufficient)", ax=ax1, alpha=0.5)
    
    ax1.axvline(threshold, color='black', linestyle='--', label=f'Intervention Threshold ({threshold})')
    ax1.set_title("Latent State at Tokens 0-5", fontweight='bold')
    ax1.set_xlabel("Mean Probe Probability (Initial)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 1)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))

    # PANEL C: The Efficiency Matrix (Compute vs Safety)
    ax2 = fig.add_subplot(gs[2])
    categories = ['Standard CoT', 'Latent Watchdog']
    wasted_tokens = [sim_results['total_wasted_baseline'], sim_results['total_wasted_intervention']]
    
    bars = ax2.bar(categories, wasted_tokens, color=['#e74c3c', '#2ecc71'], width=0.5)
    ax2.set_title("Test-Time Compute Wasted on Unsolvable Problems", fontweight='bold')
    ax2.set_ylabel("Total Tokens Wasted")
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(wasted_tokens)*0.02), 
                 f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')

    # Save the composite figure
    plt.tight_layout()
    output_slug = model_name.replace('/', '_')
    out_path = f"experiments/impact_matrix_{output_slug}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Publication figure saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='treecut')
    parser.add_argument('--k_window', type=int, default=15, help='Number of initial tokens to monitor')
    parser.add_argument('--threshold', type=float, default=0.6, help='Probability threshold to trigger halt')
    args = parser.parse_args()

    model_slug = args.model.split('/')[-1]
    
    # Construct standard paths based on your architecture
    eval_file = Path(f"experiments/dynamic_tracking_evaluation/math/{model_slug}/{args.dataset}_evaluated_traces.json")
    trace_file = Path(f"experiments/dynamic_tracking/math/{model_slug}/{args.dataset}_temporal_traces.json")
    
    if eval_file.exists() and trace_file.exists():
        run_impact_analysis(args.model, eval_file, trace_file, k_window=args.k_window, threshold=args.threshold)
    else:
        print(f"Error: Could not find required JSON files for {model_slug}.")
        print(f"Checked:\n - {eval_file}\n - {trace_file}")