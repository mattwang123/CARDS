import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

def load_and_merge(eval_path, trace_path):
    """Merges the Q1-Q4 evaluation verdicts with the raw token-by-token traces."""
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)["data"]
    
    with open(trace_path, 'r') as f:
        trace_data = json.load(f)

    # Convert traces to a lookup dictionary
    trace_lookup = {str(item.get('question_idx')): item.get('trace', []) for item in trace_data}

    merged = []
    for e in eval_data:
        q_idx = str(e["question_idx"])
        raw_trace = trace_lookup.get(q_idx, [])
        
        # Extract just the probabilities
        probs = [float(step.get("insufficiency_prob", 0)) for step in raw_trace if isinstance(step, dict)]
        
        if len(probs) > 5: # Only keep valid traces
            merged.append({
                "question_idx": q_idx,
                "quadrant": e["epistemic_quadrant"],
                "probs": probs
            })
    return merged

def smooth_and_interpolate(probs, target_length=100, window_size=5):
    """Applies a moving average and scales the trace to a 0-100% relative timeline."""
    # 1. Moving Average Smoothing
    if len(probs) >= window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(probs, kernel, mode='valid')
    else:
        smoothed = np.array(probs)

    # 2. Interpolation to normalized length (0 to 100)
    x_old = np.linspace(0, 1, num=len(smoothed))
    x_new = np.linspace(0, 1, num=target_length)
    
    interpolator = interp1d(x_old, smoothed, kind='linear', bounds_error=False, fill_value="extrapolate")
    return interpolator(x_new)

def analyze_and_plot(model_name, eval_file, trace_file):
    print(f"\nAnalyzing {model_name}...")
    merged_data = load_and_merge(eval_file, trace_file)
    
    quadrant_traces = {"Q1_Hallucination": [], "Q2_Correct_Rejection": [], "Q3_Solved_Correctly": [], "Q4_Competence_Failure": []}
    
    # Process all traces
    for item in merged_data:
        q = item["quadrant"]
        if q in quadrant_traces:
            normalized_trace = smooth_and_interpolate(item["probs"])
            quadrant_traces[q].append(normalized_trace)

    

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = {"Q1_Hallucination": "red", "Q2_Correct_Rejection": "green", "Q3_Solved_Correctly": "blue", "Q4_Competence_Failure": "orange"}
    labels = {"Q1_Hallucination": "Q1: Hallucination (Self-Deception)", "Q2_Correct_Rejection": "Q2: Correct Rejection", "Q3_Solved_Correctly": "Q3: Solved Correctly", "Q4_Competence_Failure": "Q4: Math Failure"}

    for quad, traces in quadrant_traces.items():
        if not traces:
            continue
            
        trace_matrix = np.vstack(traces)
        mean_trace = np.mean(trace_matrix, axis=0)
        std_trace = np.std(trace_matrix, axis=0)
        
        x_axis = np.linspace(0, 100, 100)
        plt.plot(x_axis, mean_trace, label=f"{labels[quad]} (n={len(traces)})", color=colors[quad], linewidth=2.5)
        plt.fill_between(x_axis, mean_trace - (std_trace*0.2), mean_trace + (std_trace*0.2), color=colors[quad], alpha=0.1)

        # Calculate Delta P for Q1 (Self-Deception Score)
        if quad == "Q1_Hallucination":
            initial_p = np.mean(mean_trace[:10]) # First 10%
            terminal_p = np.mean(mean_trace[-10:]) # Last 10%
            delta_p = initial_p - terminal_p
            print(f"  -> Q1 Self-Deception Delta P: {delta_p:.3f}")

    plt.title(f"Epistemic Trajectory during Generation: {model_name}")
    plt.xlabel("Generation Progress (%)")
    plt.ylabel("Internal Insufficiency Probability (Probe)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = f"experiments/{model_name.replace('/', '_')}_trajectory.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  -> Saved plot to {output_path}")

if __name__ == "__main__":
    # Example usage for one model/dataset combination
    # Modify these paths to loop through all your models as needed
    eval_file = "experiments/dynamic_tracking_evaluation/math/Qwen2.5-Math-1.5B/umwp_evaluated_traces.json"
    trace_file = "experiments/dynamic_tracking/math/Qwen2.5-Math-1.5B/umwp_temporal_traces.json"
    
    if Path(eval_file).exists() and Path(trace_file).exists():
        analyze_and_plot("Qwen 2.5 Math 1.5B (UMWP)", eval_file, trace_file)
    else:
        print("Files not found. Check your paths.")