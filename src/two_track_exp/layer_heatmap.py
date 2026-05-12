"""
================================================================================
E1: Layer x Time Probe F1 Heatmap (THE most important mechanistic experiment)
================================================================================

WHY THIS IS THE CORE EXPERIMENT FOR THE TWO-TRACK HYPOTHESIS:

  Our central claim: there are TWO latent tracks in the transformer:
    X = the causal "decision" track that drives output
    Y = the "reportable" track that the probe reads cleanly

  exp10 picked the SINGLE BEST LAYER per (model, dataset). That layer
  could be anywhere; we never asked WHICH layer is which track.
  
  HYPOTHESIS PREDICTION:
    - EARLY/MIDDLE layers (closer to comprehension) -> may be the X-track:
      these layers represent the actual cognitive content.
    - LATE layers (closer to unembedding) -> may be the Y-track:
      these layers prepare the output and reflect a "report" of state.
  
  The probe's high F1 at the best layer might be a Y-track artifact. The
  X-track might be visible at a different layer, possibly with a
  different temporal pattern.

WHAT WE TEST:
  For each model:
    1. Extract hidden states at ALL layers (not just best layer) at ALL
       11 percentages of CoT generation (already have for best layer from
       exp10, just need to repeat for other layers).
    2. Train a separate probe at each (layer, percentage) cell.
    3. Plot F1 as a 2D heatmap: x = percentage, y = layer.

  KEY OBSERVATIONS TO LOOK FOR:
  
    Pattern A: ALL layers show the U-shape.
      -> Y-track artifact hypothesis FALSE. U-shape is a property of the
         whole network. Two-track hypothesis weakens but doesn't break.
    
    Pattern B: Only LATE layers show U-shape; early layers are flat-high.
      -> STRONG EVIDENCE for two-track hypothesis. Early = X (stable),
         Late = Y (gets overwritten by generation, then recovers).
    
    Pattern C: Early layers PEAK in middle, late layers DIP in middle.
      -> Two-track with opposite temporal phase. Very interesting -
         the X-track engages WITH reasoning, the Y-track gets disrupted.
    
    Pattern D: One specific layer band consistently high regardless of t.
      -> That band is the X-track candidate. Use this band for E3 steering.

  KEY DECISION OUTPUT:
  We identify, per model:
    - "stable_layer": the layer with smallest std of F1 across time
    - "best_layer_t0": best probing layer at t=0 only
    - "best_layer_t100": best probing layer at t=100% only
  These give CANDIDATE X-track and Y-track layers for E3 (steering).

COST:
  This is the most expensive experiment in the series.
  - Need to extract hidden states at ALL layers (not just best).
  - Reuse cached embeddings from exp10 if possible:
    exp10 saves embeddings/proportional/{dataset}/{model_slug}/X_train_layer{N}.npy
    but only for the BEST layer. We need ALL layers.
  
  Strategy:
    - For PILOT (3-5 representative models): re-extract all layers.
    - For full sweep: use a subset of 6 strategic layers
      (early, early-mid, mid, mid-late, late, last).

OUTPUTS:
  experiments/layer_time/{model_slug}/{dataset}_layer_time_f1.npy
  experiments/layer_time/{model_slug}/{dataset}_heatmap.png
  experiments/layer_time/layer_time_summary.csv
    columns: model, dataset, layer, percentage, train_f1, test_f1

USAGE:
  # Pilot first (3 models, all layers):
  python E1_layer_time_heatmap.py --pilot
  
  # Subset of layers across all models:
  python E1_layer_time_heatmap.py --strategy strategic
================================================================================
"""

import argparse
import json
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

# ---- CONFIG ----
PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PILOT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
]

FULL_MODELS = [
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
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]

DATASETS = ['umwp', 'treecut']

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
TRAIN_DIR = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_train')
TEST_DIR  = os.path.join(EXPORT_BASE, 'experiments/dynamic_tracking_test')
OUT_DIR   = os.path.join(EXPORT_BASE, 'experiments/layer_time')
os.makedirs(OUT_DIR, exist_ok=True)


def load_cot_data(model_name, dataset, split_dir):
    """Load CoT generations + labels (same as exp10)."""
    model_slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{model_slug}/{dataset}_cot_generations.json"
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in data])
    return data, labels


def extract_all_layers_proportional(model, tokenizer, data, labels, layer_subset=None):
    """
    Extract hidden states at ALL layers (or a subset) at all 11 percentages of CoT.

    Returns:
      X: shape (n_valid_samples, n_percentages, n_selected_layers, hidden_dim)
      y: shape (n_valid_samples,)
      selected_layers: which layer indices were extracted
    """
    extracted = []
    valid_y = []
    selected = None

    for item, label in tqdm(zip(data, labels), total=len(data),
                            desc="extracting all-layer states"):
        prompt_text = item['prompt']
        gen_text = item.get('generated_response', '')

        prompt_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'][0]
        full_ids = tokenizer(prompt_text + gen_text, return_tensors="pt")['input_ids'][0]
        p_len = prompt_ids.shape[0]
        cot_len = full_ids.shape[0] - p_len
        if cot_len < 10:
            continue

        target_indices = []
        for pct in PERCENTAGES:
            idx = p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0)
            target_indices.append(min(idx, full_ids.shape[0] - 1))

        inputs = tokenizer(prompt_text + gen_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            n_layers = len(outputs.hidden_states)
            if selected is None:
                if layer_subset is None:
                    selected = list(range(n_layers))
                else:
                    # Subset: convert proportional indices to layer indices
                    # e.g., layer_subset=[0.0, 0.2, ...] -> picks layer indices
                    if all(0.0 <= x <= 1.0 for x in layer_subset):
                        selected = sorted(set(min(int(p * (n_layers - 1)), n_layers - 1)
                                              for p in layer_subset))
                    else:
                        selected = sorted(set(min(int(l), n_layers - 1) for l in layer_subset))

            # Stack only selected layers
            stack = torch.stack([outputs.hidden_states[l][0] for l in selected])
            # shape: (n_selected_layers, seq_len, hidden_dim)
            stack = stack.to(torch.float32).cpu().numpy()

            target_states = stack[:, target_indices, :].transpose(1, 0, 2)
            # shape: (n_percentages, n_selected_layers, hidden_dim)

            extracted.append(target_states)
            valid_y.append(label)

            del outputs, stack, inputs
            torch.cuda.empty_cache()

    return np.array(extracted), np.array(valid_y), selected


def train_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           C=1.0, solver='lbfgs', n_jobs=-1)
    )


def plot_heatmap(f1_matrix, layer_indices, save_path, model_slug, dataset):
    """f1_matrix: shape (n_layers, n_percentages)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(f1_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0.5, vmax=1.0, origin='lower')

    ax.set_xticks(range(len(PERCENTAGES)))
    ax.set_xticklabels([f"{int(p*100)}%" for p in PERCENTAGES])
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([f"L{l}" for l in layer_indices])
    ax.set_xlabel('CoT Progress (%)')
    ax.set_ylabel('Layer')
    ax.set_title(f'{model_slug} on {dataset}\nProbe Test F1 by (Layer, Time)')

    # Annotate each cell
    for i in range(f1_matrix.shape[0]):
        for j in range(f1_matrix.shape[1]):
            ax.text(j, i, f'{f1_matrix[i,j]:.2f}',
                    ha='center', va='center',
                    color='black' if f1_matrix[i,j] > 0.7 else 'white', fontsize=7)

    plt.colorbar(im, ax=ax, label='Test F1')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def run_for_model(model_name, dataset, layer_subset, all_rows):
    model_slug = model_name.split('/')[-1]
    out_dir = Path(OUT_DIR) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / f"{dataset}_layer_time_f1.npy"
    layers_path = out_dir / f"{dataset}_layers.npy"

    # Resume: if cached F1 matrix exists, just plot + add to summary
    if npy_path.exists() and layers_path.exists():
        print(f"  [CACHE HIT] {model_slug}/{dataset}, plotting from cache")
        f1_matrix = np.load(npy_path)
        layers = np.load(layers_path).tolist()
        plot_heatmap(f1_matrix, layers, out_dir / f"{dataset}_heatmap.png",
                     model_slug, dataset)
        for li, l in enumerate(layers):
            for pi, p in enumerate(PERCENTAGES):
                all_rows.append({
                    'model': model_slug, 'dataset': dataset,
                    'layer': int(l), 'percentage': f"{int(p*100)}%",
                    'test_f1': float(f1_matrix[li, pi])
                })
        return

    train_data, y_train = load_cot_data(model_name, dataset, TRAIN_DIR)
    test_data,  y_test  = load_cot_data(model_name, dataset, TEST_DIR)
    if train_data is None or test_data is None:
        print(f"  ! Missing CoT data for {model_slug}/{dataset}")
        return

    print(f"  Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        num_gpus = torch.cuda.device_count()
        memory_map = {0: "65GB"} if num_gpus > 0 else None
        if num_gpus > 1:
            for i in range(1, num_gpus): memory_map[i] = "78GB"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", max_memory=memory_map,
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"    ! Load failed: {e}")
        return

    try:
        X_tr, y_tr_v, layers_used = extract_all_layers_proportional(
            model, tokenizer, train_data, y_train, layer_subset)
        X_te, y_te_v, _ = extract_all_layers_proportional(
            model, tokenizer, test_data, y_test, layer_subset)

        # shapes: (n_samples, n_pct, n_layers, hidden_dim)
        n_pct = X_tr.shape[1]
        n_lay = X_tr.shape[2]

        f1_matrix = np.zeros((n_lay, n_pct))
        for li in tqdm(range(n_lay), desc="probing each layer"):
            for pi in range(n_pct):
                probe = train_probe()
                probe.fit(X_tr[:, pi, li, :], y_tr_v)
                f1_matrix[li, pi] = f1_score(y_te_v, probe.predict(X_te[:, pi, li, :]))

                all_rows.append({
                    'model': model_slug, 'dataset': dataset,
                    'layer': int(layers_used[li]),
                    'percentage': f"{int(PERCENTAGES[pi]*100)}%",
                    'test_f1': float(f1_matrix[li, pi])
                })

        np.save(npy_path, f1_matrix)
        np.save(layers_path, np.array(layers_used))
        plot_heatmap(f1_matrix, layers_used, out_dir / f"{dataset}_heatmap.png",
                     model_slug, dataset)
        print(f"  Saved heatmap + matrix to {out_dir}")

    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true',
                        help="Run only on pilot models (all layers)")
    parser.add_argument('--strategy', choices=['all', 'strategic'], default='all',
                        help="all = every layer; strategic = 6 layers at 0/0.2/0.4/0.6/0.8/1.0 of depth")
    args = parser.parse_args()

    models = PILOT_MODELS if args.pilot else FULL_MODELS

    # For strategic mode, pick 6 layers proportionally across model depth
    layer_subset = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if args.strategy == 'strategic' else None

    all_rows = []
    for model_name in models:
        print(f"\n{'='*70}\nMODEL: {model_name}\n{'='*70}")
        for dataset in DATASETS:
            try:
                run_for_model(model_name, dataset, layer_subset, all_rows)
            except Exception as e:
                print(f"  ! Error on {model_name}/{dataset}: {e}")
                import traceback; traceback.print_exc()

        # Save summary CSV incrementally
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(OUT_DIR, 'layer_time_summary.csv'), index=False)

    print(f"\nDone. Summary at {os.path.join(OUT_DIR, 'layer_time_summary.csv')}")


if __name__ == '__main__':
    main()