"""
================================================================================
E3 (FULL): Comprehensive Layer Steering Experiment
================================================================================

GOAL OF THIS EXPERIMENT:
  Rigorously determine whether the probe direction is a CAUSAL mediator of
  insufficiency-related output behavior, or merely a reportable (non-causal)
  representation. This is the "X != Y" test.

WHY THIS REPLACES THE PREVIOUS exp13_steering.py:
  The previous code had three issues that make its negative results unreliable:

  (1) STEERING WAS APPLIED ONLY TO THE LAST TOKEN AT THE LAYER OUTPUT.
      After generation begins, new tokens are not perturbed, so the
      injected signal washes out within a few generation steps.
      Following Arditi et al. (2024), we must perturb at every newly
      generated token across all post-instruction positions.

  (2) ONLY ADDITION WAS TESTED. ABLATION WAS NOT.
      Addition can push activations off-manifold (alpha=150 -> gibberish).
      Ablation (removing the direction's component) is a much cleaner causal
      test: if the direction is causal, ablating it should change behavior
      while keeping the rest of the representation on-distribution.

  (3) ONLY THE PROBE NORMAL DIRECTION WAS TRIED.
      Probe normal != difference-in-means (DIM). DIM is the actual geometric
      direction between class centroids; probe normal is the optimal linear
      classifier boundary normal. In curved high-dim data they differ.
      Arditi et al. show DIM is the right direction for causal steering.
      If we want to claim "probes don't read the causal direction", we have
      to also try the direction that IS most likely causal (DIM).

WHAT WE TEST:
  Five steering conditions on Q1 (hallucination) samples:

    1. baseline             - No intervention. Sanity baseline.
    2. probe_normal_add     - Add alpha * v_probe to all post-instruction tokens
                              and to every new generated token. Probe direction
                              from exp10 unified probe.
    3. diff_in_means_add    - Same as above but v = DIM(insuff_acts - suff_acts).
                              This is the Arditi-style steering vector.
    4. diff_in_means_ablate - Project out v_DIM from every residual stream
                              activation (Arditi-style directional ablation).
                              Tests if removing the direction changes anything.
    5. random_add           - Add a random unit vector. Control for "alpha
                              too high breaks everything regardless".

  Alpha sweep: [0.5, 1, 2, 5, 10, 20, 50]
    Smaller alphas covered to catch the case where the original code missed
    the sweet spot.

WHY ONLY Q1:
  Q1 (insufficient + model hallucinated) is where steering MATTERS for our
  story. We want to know: can steering convert Q1 -> Q2 (model abstains)?
  Q3 (sufficient + correct) is only useful as a *coherence sanity check*:
  at the same alpha, can the model still produce numeric answers on Q3?
  If random_add at alpha=20 destroys Q3 coherence the same way it destroys
  probe_normal at alpha=20, then alpha=20 is "too high for any direction"
  and we know our probe-normal steering result is not a special failure.

  So we run a small Q3 batch (20 samples) per condition just to compute a
  coherence_rate, but the primary metric is on Q1.

METRICS:
  Computed per (condition, alpha):
  - flip_rate (Q1): fraction of Q1 samples whose generation now contains
                    insufficient-indicator phrases. Higher = steering worked.
  - coherence_rate (Q1+Q3): fraction of samples whose generation is
                    readable English of length >= 20 tokens without
                    degenerate repetition.
  - useful_flip_rate: flip_rate measured ONLY on samples whose generation
                      is also coherent. This is the headline metric.

INTERPRETATION TABLE:
  Imagine these as four caricature outcomes after running the experiment:

    A) All conditions: useful_flip_rate near 0.
       -> Even the right direction (DIM) cannot flip the model. Strong
          evidence that the probe reads a non-causal latent (Y).
          This is the paper's strongest two-track support.

    B) diff_in_means works, probe_normal does not.
       -> Causal direction exists in latent space, but the probe found a
          correlated-but-different direction. Probes read Y; DIM finds X.
          This is an EVEN STRONGER finding -- it gives us the X direction.

    C) Both work.
       -> No two-track. Probe normal IS the causal direction. We need to
          revisit our "steering failed" claim (it didn't, the old code did).

    D) Random_add also flips some.
       -> Our flip-rate metric has noise. Need to use random as floor.

OUTPUTS:
  experiments/steering_full/{model_slug}/{dataset}_results.json
      Per (condition, alpha): full generations and per-sample flip/coherence flags.
  experiments/steering_full/{model_slug}/{dataset}_summary.csv
      Aggregated metrics per (condition, alpha).
  experiments/steering_full/{model_slug}/{dataset}_curves.png
      Visualization of flip_rate vs alpha for each condition.

USAGE:
  python E3_layer_steering_full.py \\
      --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
      --dataset umwp \\
      --n_q1 50 --n_q3 20
================================================================================
"""

import argparse
import json
import os
import re
import gc
from pathlib import Path
from collections import Counter

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
OUT_DIR = os.path.join(EXPORT_BASE, 'experiments/steering_full')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# AUTOMATED EVALUATION
# ============================================================================

INSUFF_PATTERNS = [
    r'\\boxed\{insufficient',
    r'\\boxed\{not enough',
    r'\\boxed\{cannot',
    r'\\boxed\{undetermined',
    r'\\boxed\{missing',
    r'\\boxed\{unknown',
    r'insufficient information',
    r'cannot be determined',
    r'not enough information',
    r'missing information',
    r'unable to determine',
    r'no unique answer',
    r'not uniquely determined',
]
INSUFF_REGEX = re.compile('|'.join(INSUFF_PATTERNS), re.IGNORECASE)


def detected_insufficient(text):
    """Heuristic for 'model abstained / claimed insufficiency'."""
    return bool(INSUFF_REGEX.search(text))


def is_coherent(text, min_tokens=20, max_repeat_ratio=0.4):
    """
    Heuristic for 'generation is readable English of nontrivial length without
    degenerate looping'. We use this to detect when alpha is so high that the
    model collapses regardless of direction.
    """
    text = text.strip()
    if len(text) < min_tokens:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    counts = Counter(words)
    max_ratio = max(counts.values()) / len(words)
    return max_ratio < max_repeat_ratio


# ============================================================================
# STEERING HOOKS
# ============================================================================

class AdditionHook:
    """
    Add alpha * v to ALL token positions in the residual stream at this layer.
    This runs on every forward pass, including during generation, so newly
    produced tokens also get the perturbation.
    """
    def __init__(self, v, alpha):
        # v is a unit-norm tensor on CPU; we move to target device on first call
        self.v = v
        self.alpha = alpha
        self._v_on_device = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._v_on_device is None or self._v_on_device.device != hs.device:
            self._v_on_device = self.v.to(hs.device, dtype=hs.dtype)
        # Perturb every token position (post-instruction is everywhere here;
        # the instruction-only positions don't matter for generation)
        modified = hs + self.alpha * self._v_on_device
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class AblationHook:
    """
    Project out the component of the residual stream along v at every position.
    For each x: x' = x - (x . v_hat) * v_hat
    Following Arditi et al., this should be applied at EVERY layer for full
    directional ablation. We register it on every transformer block.
    """
    def __init__(self, v):
        self.v = v  # expected unit norm
        self._v_on_device = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._v_on_device is None or self._v_on_device.device != hs.device:
            self._v_on_device = self.v.to(hs.device, dtype=hs.dtype)
        # x - (x . v) * v, where v is unit norm
        # hs shape: (batch, seq, hidden); v shape: (hidden,)
        proj = (hs * self._v_on_device).sum(dim=-1, keepdim=True)
        modified = hs - proj * self._v_on_device
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


# ============================================================================
# DIRECTION COMPUTATION
# ============================================================================

def probe_normal_direction(probe):
    """
    Effective probe weight direction.
    For a Pipeline(StandardScaler, LogisticRegression):
        score(x) = (clf.coef . ((x - scaler.mean) / scaler.scale)) + clf.intercept
        => effective weight on raw x is clf.coef / scaler.scale
    Returns a unit-norm vector.
    """
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def diff_in_means_direction(X_insuff, X_suff):
    """
    Difference-in-means direction (Arditi et al. style).
    Points FROM sufficient centroid TO insufficient centroid, so adding this
    should make activations 'more insufficient'.
    Returns a unit-norm vector.
    """
    mu_insuff = X_insuff.mean(axis=0)
    mu_suff = X_suff.mean(axis=0)
    v = mu_insuff - mu_suff
    return (v / np.linalg.norm(v)).astype(np.float32)


def random_direction(hidden_dim, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(hidden_dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ============================================================================
# HIDDEN STATE EXTRACTION FOR DIM
# ============================================================================

def extract_t0_for_dim(model, tokenizer, prompts, layer):
    """
    Extract hidden state at the final prompt token, at the given layer.
    This matches what the exp10 unified probe was trained on, so the DIM
    direction lives in the same space as the probe normal.
    """
    states = []
    for prompt in tqdm(prompts, desc=f"  extracting t=0 acts (layer {layer})", leave=False):
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        idx = inp['input_ids'].shape[1] - 1
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[layer][0, idx, :].to(torch.float32).cpu().numpy()
            states.append(h)
            del out
            torch.cuda.empty_cache()
    return np.array(states)


# ============================================================================
# UTILITIES
# ============================================================================

def get_layer_modules(model):
    """Locate the transformer block module list across common architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Cannot locate transformer layer modules on this model")


def load_q_samples(model_slug, dataset, quadrant, max_n):
    """Load prompts for a given epistemic quadrant from exp2 eval data."""
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

    prompts = [g['prompt'] for g, e in zip(gen_data, eval_data)
               if e.get('epistemic_quadrant') == quadrant]
    return prompts[:max_n]


def generate_one(model, tokenizer, prompt, max_new_tokens=400):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def run_condition(model, tokenizer, prompts, hooks_setup, label, max_new_tokens=400):
    """
    hooks_setup is a callable that takes the model, registers hooks, and returns
    a list of handles. We call it before each prompt and remove handles after.
    Set hooks_setup=None for baseline (no intervention).
    """
    records = []
    for prompt in tqdm(prompts, desc=f"  {label}", leave=False):
        handles = hooks_setup(model) if hooks_setup is not None else []
        try:
            text = generate_one(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        finally:
            for h in handles:
                h.remove()

        records.append({
            'prompt': prompt,
            'generation': text,
            'detected_insufficient': detected_insufficient(text),
            'coherent': is_coherent(text),
            'tail_200': text[-200:]
        })
    return records


# ============================================================================
# HOOK SETUP HELPERS
# ============================================================================

def make_addition_setup(target_module, v_tensor, alpha):
    """Returns a function that, when called, registers an AdditionHook and
    returns the handle list."""
    def setup(_model):
        hook = AdditionHook(v_tensor, alpha)
        handle = target_module.register_forward_hook(hook)
        return [handle]
    return setup


def make_ablation_setup(layer_modules, v_tensor):
    """Register ablation hooks on every layer for full directional ablation."""
    def setup(_model):
        handles = []
        for lm in layer_modules:
            hook = AblationHook(v_tensor)
            handles.append(lm.register_forward_hook(hook))
        return handles
    return setup


# ============================================================================
# METRICS AGGREGATION
# ============================================================================

def aggregate_records(records):
    n = len(records)
    if n == 0:
        return dict(n=0, flip_rate=0.0, coherence_rate=0.0, useful_flip_rate=0.0)
    n_flip = sum(r['detected_insufficient'] for r in records)
    n_coh = sum(r['coherent'] for r in records)
    n_useful = sum(r['detected_insufficient'] and r['coherent'] for r in records)
    return dict(
        n=n,
        flip_rate=n_flip / n,
        coherence_rate=n_coh / n,
        useful_flip_rate=n_useful / n,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(summary_df, save_path, model_slug, dataset):
    """Plot useful_flip_rate vs alpha for each condition (Q1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    q1_df = summary_df[summary_df['quadrant'] == 'Q1']

    # Panel 1: useful flip rate on Q1 (the main story metric)
    ax = axes[0]
    for cond in q1_df['condition'].unique():
        sub = q1_df[q1_df['condition'] == cond].sort_values('alpha')
        if cond == 'baseline':
            ax.axhline(sub['useful_flip_rate'].iloc[0], color='black',
                       linestyle='--', label=f'baseline ({sub["useful_flip_rate"].iloc[0]:.2f})')
        else:
            ax.plot(sub['alpha'], sub['useful_flip_rate'], marker='o', label=cond)
    ax.set_xlabel('alpha')
    ax.set_ylabel('useful flip rate on Q1\n(insufficient AND coherent)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_title(f'{model_slug} / {dataset} - Did steering flip Q1?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: coherence rate (mixed over Q1 and Q3) -- sanity check
    ax = axes[1]
    for cond in summary_df['condition'].unique():
        sub = summary_df[summary_df['condition'] == cond].sort_values('alpha')
        if cond == 'baseline':
            ax.axhline(sub['coherence_rate'].iloc[0], color='black',
                       linestyle='--', label=f'baseline ({sub["coherence_rate"].iloc[0]:.2f})')
        else:
            ax.plot(sub['alpha'], sub['coherence_rate'], marker='o', label=cond)
    ax.set_xlabel('alpha')
    ax.set_ylabel('coherence rate (all samples)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_title('Coherence: when did alpha break the model?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def run(args):
    model_name = args.model
    model_slug = model_name.split('/')[-1]
    dataset = args.dataset

    out_dir = Path(OUT_DIR) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate the unified probe and its layer
    csv_path = os.path.join(EXP_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    df = pd.read_csv(csv_path)
    row = df[df['Model'] == model_slug]
    if row.empty:
        print(f"ERROR: no exp10 entry for {model_slug}/{dataset}")
        return
    best_layer = int(row['Optimal_Layer'].iloc[0])

    probe_path = os.path.join(
        EXP_DIR, 'probes_proportional', dataset, model_slug,
        f"unified_probe_layer{best_layer}.joblib"
    )
    probe = joblib.load(probe_path)
    v_probe = probe_normal_direction(probe)
    hidden_dim = v_probe.shape[0]

    print(f"Model: {model_slug}")
    print(f"  Best layer: {best_layer}")
    print(f"  Hidden dim: {hidden_dim}")

    # 2) Load Q1 and Q3 prompts
    q1_prompts = load_q_samples(model_slug, dataset, 'Q1_Hallucination', args.n_q1)
    q3_prompts = load_q_samples(model_slug, dataset, 'Q3_Solved_Correctly', args.n_q3)
    print(f"  Q1 samples: {len(q1_prompts)}, Q3 samples: {len(q3_prompts)}")

    # 3) Load model
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    num_gpus = torch.cuda.device_count()
    memory_map = {0: "65GB"}
    for i in range(1, num_gpus):
        memory_map[i] = "78GB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", max_memory=memory_map,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    layer_modules = get_layer_modules(model)
    target_module = layer_modules[best_layer]

    # 4) Compute difference-in-means direction
    #    Need balanced sufficient + insufficient prompts to compute centroids.
    #    Use prompts from exp10 train split which the probe was trained on,
    #    so v_DIM lives in the same activation space as v_probe.
    print(f"  Computing difference-in-means direction at layer {best_layer}...")
    train_gen_path = os.path.join(
        EXPORT_BASE,
        f"experiments/dynamic_tracking_train/math/{model_slug}/{dataset}_cot_generations.json"
    )
    with open(train_gen_path) as f:
        train_gen = json.load(f)
    suff_prompts_dim = [g['prompt'] for g in train_gen if g.get('is_sufficient', True)][:args.n_dim]
    insuff_prompts_dim = [g['prompt'] for g in train_gen if not g.get('is_sufficient', True)][:args.n_dim]
    X_suff = extract_t0_for_dim(model, tokenizer, suff_prompts_dim, best_layer)
    X_insuff = extract_t0_for_dim(model, tokenizer, insuff_prompts_dim, best_layer)
    v_dim = diff_in_means_direction(X_insuff, X_suff)

    # Report the angle between the two directions -- this is informative on its own
    cos_sim = float(np.dot(v_probe, v_dim))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
    print(f"  cos(v_probe, v_DIM) = {cos_sim:.4f}  (angle = {angle_deg:.1f} deg)")

    v_rand = random_direction(hidden_dim, seed=42)
    cos_pr = float(np.dot(v_probe, v_rand))
    cos_dr = float(np.dot(v_dim, v_rand))
    print(f"  cos(v_probe, v_rand) = {cos_pr:.4f}")
    print(f"  cos(v_DIM,   v_rand) = {cos_dr:.4f}")

    # Convert to torch tensors (CPU; hooks move to device on first call)
    v_probe_t = torch.tensor(v_probe, dtype=torch.float32)
    v_dim_t = torch.tensor(v_dim, dtype=torch.float32)
    v_rand_t = torch.tensor(v_rand, dtype=torch.float32)

    # 5) Define all conditions to run
    #    Each entry: (condition_name, alphas_to_run, hook_setup_factory_fn)
    #    where hook_setup_factory_fn(alpha) returns the setup callable.
    conditions = []

    # Baseline -- no intervention, alpha=0 placeholder
    conditions.append(('baseline', [0], lambda a: None))

    # Probe-normal addition
    conditions.append(('probe_normal_add', ALPHAS,
                       lambda a: make_addition_setup(target_module, v_probe_t, a)))

    # DIM addition (Arditi-style)
    conditions.append(('diff_in_means_add', ALPHAS,
                       lambda a: make_addition_setup(target_module, v_dim_t, a)))

    # DIM ablation -- alpha is ignored (set to a sentinel value 1)
    conditions.append(('diff_in_means_ablate', [1],
                       lambda a: make_ablation_setup(layer_modules, v_dim_t)))

    # Random direction addition
    conditions.append(('random_add', ALPHAS,
                       lambda a: make_addition_setup(target_module, v_rand_t, a)))

    # 6) Run everything, with resume
    results_path = out_dir / f"{dataset}_results.json"
    summary_path = out_dir / f"{dataset}_summary.csv"

    if results_path.exists() and not args.force:
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  [RESUME] loaded {len(all_results)} existing records")
    else:
        all_results = []

    done_keys = {(r['condition'], r['alpha'], r['quadrant']) for r in all_results}

    for cond_name, alphas, setup_factory in conditions:
        for alpha in alphas:
            for quadrant_label, prompts in [('Q1', q1_prompts), ('Q3', q3_prompts)]:
                key = (cond_name, alpha, quadrant_label)
                if key in done_keys:
                    continue

                print(f"\n  Condition: {cond_name}, alpha={alpha}, quadrant={quadrant_label}")
                setup_fn = setup_factory(alpha)
                records = run_condition(
                    model, tokenizer, prompts, setup_fn,
                    label=f"{cond_name}_a{alpha}_{quadrant_label}"
                )

                metrics = aggregate_records(records)
                entry = {
                    'condition': cond_name,
                    'alpha': alpha,
                    'quadrant': quadrant_label,
                    'metrics': metrics,
                    'records': records,
                }
                all_results.append(entry)

                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

                print(f"    -> flip={metrics['flip_rate']:.3f}, "
                      f"coh={metrics['coherence_rate']:.3f}, "
                      f"useful_flip={metrics['useful_flip_rate']:.3f}")

    # 7) Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 8) Build summary CSV
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'model': model_slug,
            'dataset': dataset,
            'condition': r['condition'],
            'alpha': r['alpha'],
            'quadrant': r['quadrant'],
            **r['metrics'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    # 9) Plot
    plot_results(summary_df, out_dir / f"{dataset}_curves.png", model_slug, dataset)

    # 10) Print headline numbers
    print("\n" + "=" * 80)
    print("HEADLINE RESULTS (Q1 useful_flip_rate by condition):")
    print("=" * 80)
    q1 = summary_df[summary_df['quadrant'] == 'Q1']
    pivot = q1.pivot(index='condition', columns='alpha', values='useful_flip_rate')
    print(pivot.round(3).to_string())
    print(f"\nAll results -> {results_path}")
    print(f"Summary CSV  -> {summary_path}")
    print(f"Plot         -> {out_dir / f'{dataset}_curves.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='HF model id, e.g. Qwen/Qwen2.5-Math-1.5B-Instruct')
    parser.add_argument('--dataset', default='umwp', choices=['umwp', 'treecut'])
    parser.add_argument('--n_q1', type=int, default=50,
                        help='Number of Q1 (hallucination) samples to test')
    parser.add_argument('--n_q3', type=int, default=20,
                        help='Number of Q3 (correctly solved) samples for coherence check')
    parser.add_argument('--n_dim', type=int, default=200,
                        help='Number of train prompts per class used to compute DIM direction')
    parser.add_argument('--force', action='store_true',
                        help='Ignore existing results.json and rerun')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()