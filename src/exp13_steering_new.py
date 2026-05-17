"""
================================================================================
exp13 (FULL v2): Sample-Major Layer Steering Experiment
================================================================================

WHAT'S NEW vs v1:
  - Multi-model and multi-dataset sweep in one script
  - Sample-major output format: per-prompt records with all
    (condition, alpha) variants side-by-side for easy comparison
  - Strict flip detection (only \\boxed{Insufficient}-style, no loose substring)
  - Batched generation within a condition (much faster than 1-at-a-time)
  - Probe-normal vector now optionally scaled to match DIM magnitude, so we
    don't confound 'direction' with 'magnitude' (NEW: probe_normal_add_scaled)

PIPELINE:
  For each (model, dataset):
    1. Load Q1 samples (hallucination) + Q3 samples (correctly solved, for
       Q3 false-abstention sanity check)
    2. Extract t=0 activations from balanced train prompts to compute DIM
    3. Run all (condition, alpha) configurations with batched generation
    4. Save sample-major JSON: one record per prompt with all variants nested

INTERPRETATION (the honest one):
  - Probe normal and DIM are BOTH "observer" directions (statistical
    descriptions of how the two classes differ in activation space).
    Neither is guaranteed to be the causal X direction.
  - If DIM works substantially better than probe normal at LOW alpha
    AND with Q3 stays clean, that's evidence the probe found a less
    causal direction within the same "observer family".
  - If both fail at all alphas without breaking Q3, that's evidence
    that linear interventions cannot reach X. This SUPPORTS X being
    non-linear / distributed, which is still consistent with two-track.
  - If DIM only works at alphas where Q3 also collapses, the apparent
    flip rate is a vocabulary-level shortcut, not real epistemic control.

CONDITIONS:
  - baseline                  -- no intervention
  - probe_normal_add          -- v_probe (unit norm), alpha in sweep
  - probe_normal_add_scaled   -- v_probe scaled to ||v_DIM||, alpha in sweep
                                 (controls for the magnitude confound)
  - diff_in_means_add         -- v_DIM (unit norm), alpha in sweep
  - diff_in_means_ablate      -- project out v_DIM from every layer
  - random_add                -- random unit vector, alpha in sweep

OUTPUTS PER (model, dataset):
  results_sample_major.json:
    [
      {"prompt": "...",
       "quadrant": "Q1",
       "variants": {
          "baseline":                {"text": "...", "boxed_insufficient": false},
          "probe_normal_add_a5":     {"text": "...", "boxed_insufficient": true},
          "diff_in_means_add_a5":    {"text": "...", "boxed_insufficient": false},
          ...
       }},
      ...
    ]
  summary.csv:
    one row per (condition, alpha, quadrant) with aggregate metrics
  curves.png:
    flip-rate curves for Q1 and false-abstention curves for Q3

USAGE:
  # Single model
  python E3_layer_steering_v2.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
      --dataset umwp --n_q1 50 --n_q3 20

  # All scales, both datasets
  python E3_layer_steering_v2.py --all_models --all_datasets
================================================================================
"""

import argparse
import json
import os
import re
import gc
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


# ============================================================================
# CONFIG
# ============================================================================

ALPHAS = [0.5, 1, 2, 5, 10, 20, 50]
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
EXP_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
OUT_DIR = os.path.join(EXPORT_BASE, 'experiments/steering_full_v2')
os.makedirs(OUT_DIR, exist_ok=True)

# Models spanning scales: small / medium / large within each family.
# Pick instruct variants since our story is about the alignment-induced gap.
DEFAULT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',          # small
    'Qwen/Qwen2.5-Math-7B-Instruct',            # medium
    'meta-llama/Meta-Llama-3.1-8B-Instruct',    # medium (different family)
    'Qwen/Qwen2.5-14B-Instruct',                # large
]


# ============================================================================
# STRICT FLIP DETECTION
# ============================================================================
# v1 used loose substring matching which had false positives (e.g. matching
# "missing information" inside a side comment about a typo). v2 requires that
# the model explicitly placed an insufficiency claim in a \boxed{} expression,
# which is what the solving prompt asks for.

BOXED_INSUFF_REGEX = re.compile(
    r'\\boxed\s*\{\s*('
    r'insufficient|not enough(?: information)?|cannot be determined|'
    r'undetermined|missing(?: information)?|unknown|unsolvable|'
    r'no unique answer|cannot determine|unable to determine'
    r')\s*\}',
    re.IGNORECASE
)


def strict_boxed_insufficient(text):
    """Detect explicit \\boxed{Insufficient}-style claims. Strict."""
    return bool(BOXED_INSUFF_REGEX.search(text))


def is_coherent(text, min_chars=20, max_repeat_ratio=0.4):
    text = text.strip()
    if len(text) < min_chars:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    counts = Counter(words)
    return max(counts.values()) / len(words) < max_repeat_ratio


# ============================================================================
# STEERING HOOKS
# ============================================================================

class AdditionHook:
    """Add alpha * v to every token position at the hooked layer.
    Persists across generation steps (this hook stays registered while
    model.generate() runs)."""
    def __init__(self, v, alpha):
        self.v = v
        self.alpha = alpha
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = self.v.to(device=hs.device, dtype=hs.dtype)
        modified = hs + self.alpha * self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


class AblationHook:
    """Project out v from residual stream at this layer. Apply to every layer
    for full directional ablation."""
    def __init__(self, v):
        self.v = v   # unit norm
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = self.v.to(device=hs.device, dtype=hs.dtype)
        proj = (hs * self._cached).sum(dim=-1, keepdim=True)
        modified = hs - proj * self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


# ============================================================================
# DIRECTION COMPUTATION
# ============================================================================

def probe_normal_direction(probe):
    """Effective probe weight on raw (un-scaled) activations, unit-normed."""
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def diff_in_means_direction(X_insuff, X_suff):
    """Centroid difference, unit-normed. Also returns the raw magnitude
    (for the probe_normal_scaled control)."""
    v = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
    raw_mag = float(np.linalg.norm(v))
    return (v / raw_mag).astype(np.float32), raw_mag


def random_direction(hidden_dim, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(hidden_dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ============================================================================
# HIDDEN STATE EXTRACTION (for DIM computation)
# ============================================================================

def extract_t0_batch(model, tokenizer, prompts, layer, batch_size=4):
    """Batched extraction of hidden state at final prompt token."""
    states = []
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=f"  t=0 extraction layer={layer}", leave=False):
        batch = prompts[i:i + batch_size]
        # Pad to the longest in batch
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # Find the last non-pad position for each sequence
        attn = enc['attention_mask']
        last_idx = attn.sum(dim=1) - 1
        h = out.hidden_states[layer]  # (B, T, D)
        for b, idx in enumerate(last_idx.tolist()):
            states.append(h[b, idx, :].to(torch.float32).cpu().numpy())
        del out
        torch.cuda.empty_cache()
    return np.array(states)


# ============================================================================
# BATCHED GENERATION
# ============================================================================

def generate_batch(model, tokenizer, prompts, max_new_tokens=400, batch_size=8):
    """Batched greedy generation. Returns list of generated strings (no prompt)."""
    generated = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_lens = enc['attention_mask'].sum(dim=1).tolist()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for b, in_len in enumerate(input_lens):
            text = tokenizer.decode(out[b, in_len:], skip_special_tokens=True).strip()
            generated.append(text)
    return generated


# ============================================================================
# I/O HELPERS
# ============================================================================

def load_q_samples(model_slug, dataset, quadrant, max_n):
    eval_path = os.path.join(
        EXPORT_BASE,
        f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json"
    )
    gen_path = os.path.join(
        EXPORT_BASE,
        f"experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json"
    )
    with open(eval_path) as f:
        eval_data = json.load(f).get("data", [])
    with open(gen_path) as f:
        gen_data = json.load(f)

    return [g['prompt'] for g, e in zip(gen_data, eval_data)
            if e.get('epistemic_quadrant') == quadrant][:max_n]


def load_train_prompts_balanced(model_slug, dataset, n_per_class):
    """Get balanced sufficient + insufficient prompts from exp10 train data
    for DIM computation."""
    train_path = os.path.join(
        EXPORT_BASE,
        f"experiments/dynamic_tracking_train/math/{model_slug}/{dataset}_cot_generations.json"
    )
    with open(train_path) as f:
        train = json.load(f)
    suff   = [g['prompt'] for g in train if g.get('is_sufficient', True)][:n_per_class]
    insuff = [g['prompt'] for g in train if not g.get('is_sufficient', True)][:n_per_class]
    return suff, insuff


def get_layer_modules(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Cannot locate transformer layers")


# ============================================================================
# CONDITION DEFINITION
# ============================================================================
# Each condition is a tuple (key, alpha, hook_factory).
# hook_factory(model) registers the hooks and returns a list of handles.
# Key is the string used in the sample-major output.

def build_conditions(target_module, layer_modules, v_probe_unit, v_probe_scaled,
                     v_dim_unit, v_rand_unit):
    """Build the list of conditions to run."""
    conditions = []

    # Baseline
    conditions.append(('baseline', 0, lambda m: []))

    # Probe normal (unit norm) with addition
    for a in ALPHAS:
        v = v_probe_unit
        conditions.append((
            f'probe_normal_add_a{a}',
            a,
            (lambda v_=v, a_=a: (lambda m: [target_module.register_forward_hook(AdditionHook(v_, a_))]))()
        ))

    # Probe normal scaled to ||v_DIM||
    for a in ALPHAS:
        v = v_probe_scaled
        conditions.append((
            f'probe_normal_scaled_add_a{a}',
            a,
            (lambda v_=v, a_=a: (lambda m: [target_module.register_forward_hook(AdditionHook(v_, a_))]))()
        ))

    # DIM addition
    for a in ALPHAS:
        v = v_dim_unit
        conditions.append((
            f'diff_in_means_add_a{a}',
            a,
            (lambda v_=v, a_=a: (lambda m: [target_module.register_forward_hook(AdditionHook(v_, a_))]))()
        ))

    # DIM ablation across all layers (alpha is N/A, we tag it 1)
    def ablate_factory(m):
        handles = []
        for lm in layer_modules:
            handles.append(lm.register_forward_hook(AblationHook(v_dim_unit)))
        return handles
    conditions.append(('diff_in_means_ablate', 1, ablate_factory))

    # Random direction addition
    for a in ALPHAS:
        v = v_rand_unit
        conditions.append((
            f'random_add_a{a}',
            a,
            (lambda v_=v, a_=a: (lambda m: [target_module.register_forward_hook(AdditionHook(v_, a_))]))()
        ))

    return conditions


# ============================================================================
# MAIN RUN PER (model, dataset)
# ============================================================================

def run_model_dataset(model_name, dataset, args):
    model_slug = model_name.split('/')[-1]
    out_dir = Path(OUT_DIR) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"{dataset}_results_sample_major.json"
    summary_path = out_dir / f"{dataset}_summary.csv"
    meta_path = out_dir / f"{dataset}_meta.json"

    # 1) Probe + layer info
    csv_path = os.path.join(EXP_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(csv_path):
        print(f"  ! Missing exp10 csv: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    row = df[df['Model'] == model_slug]
    if row.empty:
        print(f"  ! No exp10 entry for {model_slug}")
        return
    best_layer = int(row['Optimal_Layer'].iloc[0])

    probe_path = os.path.join(EXP_DIR, 'probes_proportional', dataset, model_slug,
                              f"unified_probe_layer{best_layer}.joblib")
    if not os.path.exists(probe_path):
        print(f"  ! No probe at {probe_path}")
        return
    probe = joblib.load(probe_path)
    v_probe_unit = probe_normal_direction(probe)
    hidden_dim = v_probe_unit.shape[0]

    print(f"\n[{model_slug} / {dataset}]")
    print(f"  best_layer={best_layer}, hidden_dim={hidden_dim}")

    # 2) Load test samples
    q1_prompts = load_q_samples(model_slug, dataset, 'Q1_Hallucination', args.n_q1)
    q3_prompts = load_q_samples(model_slug, dataset, 'Q3_Solved_Correctly', args.n_q3)
    print(f"  Q1 samples: {len(q1_prompts)}, Q3 samples: {len(q3_prompts)}")
    if not q1_prompts:
        print(f"  ! No Q1 samples available")
        return

    # 3) Load model
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # required for generation

    num_gpus = torch.cuda.device_count()
    mem = {0: "65GB"}
    for i in range(1, num_gpus):
        mem[i] = "78GB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", max_memory=mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    layer_modules = get_layer_modules(model)
    target_module = layer_modules[best_layer]

    # 4) Compute DIM direction from balanced train prompts
    print(f"  Computing DIM at layer {best_layer}...")
    suff_prompts, insuff_prompts = load_train_prompts_balanced(model_slug, dataset, args.n_dim)
    X_suff = extract_t0_batch(model, tokenizer, suff_prompts, best_layer, batch_size=args.batch_size)
    X_insuff = extract_t0_batch(model, tokenizer, insuff_prompts, best_layer, batch_size=args.batch_size)
    v_dim_unit, dim_magnitude = diff_in_means_direction(X_insuff, X_suff)

    # probe_normal scaled to match DIM magnitude (controls for magnitude confound)
    v_probe_scaled = v_probe_unit * dim_magnitude
    # Note: v_probe_scaled is no longer unit norm; alpha is interpreted differently
    # for it. Save magnitude for honest reporting.

    v_rand_unit = random_direction(hidden_dim, seed=42)

    cos_probe_dim  = float(np.dot(v_probe_unit, v_dim_unit))
    cos_probe_rand = float(np.dot(v_probe_unit, v_rand_unit))
    cos_dim_rand   = float(np.dot(v_dim_unit, v_rand_unit))
    print(f"  cos(probe, DIM)={cos_probe_dim:.3f}, cos(probe, rand)={cos_probe_rand:.3f}, "
          f"cos(DIM, rand)={cos_dim_rand:.3f}")
    print(f"  ||DIM||={dim_magnitude:.3f}")

    # Save metadata
    meta = {
        'model': model_slug, 'dataset': dataset,
        'best_layer': best_layer, 'hidden_dim': hidden_dim,
        'dim_magnitude': dim_magnitude,
        'cos_probe_dim': cos_probe_dim,
        'cos_probe_rand': cos_probe_rand,
        'cos_dim_rand': cos_dim_rand,
        'n_q1': len(q1_prompts), 'n_q3': len(q3_prompts),
        'alphas': ALPHAS,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Move direction tensors to CPU torch
    v_probe_unit_t   = torch.tensor(v_probe_unit, dtype=torch.float32)
    v_probe_scaled_t = torch.tensor(v_probe_scaled, dtype=torch.float32)
    v_dim_unit_t     = torch.tensor(v_dim_unit, dtype=torch.float32)
    v_rand_unit_t    = torch.tensor(v_rand_unit, dtype=torch.float32)

    # 5) Build conditions
    conditions = build_conditions(
        target_module, layer_modules,
        v_probe_unit_t, v_probe_scaled_t, v_dim_unit_t, v_rand_unit_t
    )

    # 6) Initialize sample-major storage
    # Resume if exists
    if results_path.exists() and not args.force:
        with open(results_path) as f:
            sample_records = json.load(f)
        print(f"  [RESUME] loaded {len(sample_records)} existing records")
    else:
        sample_records = []
        for p in q1_prompts:
            sample_records.append({'prompt': p, 'quadrant': 'Q1', 'variants': {}})
        for p in q3_prompts:
            sample_records.append({'prompt': p, 'quadrant': 'Q3', 'variants': {}})

    # Index by prompt for fast updates
    by_prompt = {r['prompt']: r for r in sample_records}
    q1_set = set(q1_prompts)
    q3_set = set(q3_prompts)

    # 7) Run each condition on all relevant prompts.
    # Q3 only needs baseline and the alphas we care about for sanity. To keep
    # it cheap, run Q3 only on a SUBSET of conditions: baseline + every alpha
    # for each condition (so we always have the matched Q3 false-abstention
    # check at each alpha).
    for cond_key, alpha, hook_factory in tqdm(conditions, desc=f"  conditions"):
        # Skip if all prompts already have this variant
        needs_q1 = [p for p in q1_prompts if cond_key not in by_prompt[p]['variants']]
        needs_q3 = [p for p in q3_prompts if cond_key not in by_prompt[p]['variants']]
        if not needs_q1 and not needs_q3:
            continue

        for prompts_batch, prompt_set in [(needs_q1, q1_set), (needs_q3, q3_set)]:
            if not prompts_batch:
                continue
            # Register hooks once for this condition
            handles = hook_factory(model)
            try:
                gens = generate_batch(model, tokenizer, prompts_batch,
                                      max_new_tokens=args.max_new_tokens,
                                      batch_size=args.batch_size)
            finally:
                for h in handles:
                    h.remove()

            for p, gen in zip(prompts_batch, gens):
                rec = {
                    'text': gen,
                    'boxed_insufficient': strict_boxed_insufficient(gen),
                    'coherent': is_coherent(gen),
                    'tail_200': gen[-200:],
                }
                by_prompt[p]['variants'][cond_key] = rec

        # Save incrementally
        with open(results_path, 'w') as f:
            json.dump(sample_records, f, indent=2)

    # 8) Build summary
    summary_rows = []
    all_cond_keys = set()
    for r in sample_records:
        all_cond_keys.update(r['variants'].keys())

    for cond_key in sorted(all_cond_keys):
        # Parse alpha from key
        if cond_key == 'baseline':
            alpha = 0.0
            base = 'baseline'
        elif cond_key == 'diff_in_means_ablate':
            alpha = float('nan')
            base = 'diff_in_means_ablate'
        else:
            m = re.match(r'(.+)_a([\d.]+)$', cond_key)
            base, alpha = m.group(1), float(m.group(2))

        for quad in ('Q1', 'Q3'):
            recs = [r['variants'][cond_key] for r in sample_records
                    if r['quadrant'] == quad and cond_key in r['variants']]
            n = len(recs)
            if n == 0:
                continue
            flip = sum(rr['boxed_insufficient'] for rr in recs) / n
            coh = sum(rr['coherent'] for rr in recs) / n
            useful = sum(rr['boxed_insufficient'] and rr['coherent'] for rr in recs) / n
            summary_rows.append({
                'model': model_slug, 'dataset': dataset,
                'condition_base': base, 'alpha': alpha, 'quadrant': quad,
                'n': n, 'flip_rate': flip, 'coherence_rate': coh,
                'useful_flip_rate': useful,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    # 9) Plot
    plot_curves(summary_df, out_dir / f"{dataset}_curves.png", model_slug, dataset)

    # 10) Print headline
    print(f"\n  --- HEADLINE: {model_slug}/{dataset} useful_flip_rate ---")
    q1_summary = summary_df[summary_df['quadrant'] == 'Q1']
    pivot = q1_summary.pivot(index='condition_base', columns='alpha', values='useful_flip_rate')
    print(pivot.round(3).to_string())
    print(f"\n  --- Q3 false-abstention (should stay near 0 unless model is breaking) ---")
    q3_summary = summary_df[summary_df['quadrant'] == 'Q3']
    pivot_q3 = q3_summary.pivot(index='condition_base', columns='alpha', values='flip_rate')
    print(pivot_q3.round(3).to_string())

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def plot_curves(summary_df, save_path, model_slug, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    q1_df = summary_df[summary_df['quadrant'] == 'Q1']
    q3_df = summary_df[summary_df['quadrant'] == 'Q3']

    # Panel 1: Q1 useful flip
    ax = axes[0]
    for cond in q1_df['condition_base'].unique():
        sub = q1_df[q1_df['condition_base'] == cond].sort_values('alpha')
        if cond == 'baseline':
            ax.axhline(sub['useful_flip_rate'].iloc[0], color='black', linestyle='--',
                       label=f"baseline ({sub['useful_flip_rate'].iloc[0]:.2f})")
        else:
            ax.plot(sub['alpha'], sub['useful_flip_rate'], marker='o', label=cond)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('alpha')
    ax.set_ylabel('Q1 useful flip rate\n(strict boxed-Insufficient AND coherent)')
    ax.set_title(f'{model_slug} / {dataset}: Q1 flip rate')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 2: Q3 false abstention (lower = better for direction-specificity)
    ax = axes[1]
    for cond in q3_df['condition_base'].unique():
        sub = q3_df[q3_df['condition_base'] == cond].sort_values('alpha')
        if cond == 'baseline':
            ax.axhline(sub['flip_rate'].iloc[0], color='black', linestyle='--',
                       label=f"baseline ({sub['flip_rate'].iloc[0]:.2f})")
        else:
            ax.plot(sub['alpha'], sub['flip_rate'], marker='o', label=cond)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('alpha')
    ax.set_ylabel('Q3 false abstention rate\n(solvable problems wrongly marked Insufficient)')
    ax.set_title('Q3 sanity: did alpha break the model?')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        help='Single HF model id, or use --all_models')
    parser.add_argument('--all_models', action='store_true',
                        help=f'Run on default model list: {DEFAULT_MODELS}')
    parser.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    parser.add_argument('--all_datasets', action='store_true')
    parser.add_argument('--n_q1', type=int, default=50)
    parser.add_argument('--n_q3', type=int, default=20)
    parser.add_argument('--n_dim', type=int, default=200,
                        help='Number of train prompts per class for DIM')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Generation batch size. Reduce if OOM.')
    parser.add_argument('--max_new_tokens', type=int, default=400)
    parser.add_argument('--force', action='store_true', help='Recompute all')
    args = parser.parse_args()

    if args.all_models:
        models = DEFAULT_MODELS
    elif args.model:
        models = [args.model]
    else:
        raise ValueError("Use --model or --all_models")

    if args.all_datasets:
        datasets = ['umwp', 'treecut']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise ValueError("Use --dataset or --all_datasets")

    for model_name in models:
        for dataset in datasets:
            try:
                run_model_dataset(model_name, dataset, args)
            except Exception as e:
                print(f"\n[ERROR] {model_name}/{dataset}: {e}")
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()