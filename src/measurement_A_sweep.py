"""
================================================================================
Measurement A — Geometric Alignment Sweep (Full 21-Model Coverage)
================================================================================

Strips probe_unembed_alignment.py down to just the static geometric measurement:

    z = W_U @ v_probe                   (no forward passes through samples)
    margin = mean(z over T_abs) - mean(z over T_num)

Plus matched v_random control. v_DIM optional (off by default to keep the sweep
fast on big models — v_DIM needs forward passes on train prompts).

Designed for the model-coverage gap: F7's current 5-model result becomes a
21-model result, which is the strongest single robustness boost the paper can
get with a few hours of compute.

Per-model timing rough estimate (on 2x A100 80GB):
    1.5B-8B:   ~1 min  (model load + W_U projection + token classification)
    12B-27B:  ~3 min
    32B-70B:  ~8 min   (sharded across both GPUs)
Total for 21 models, both datasets: ~1-2 hours.

OUTPUT:
    experiment_result/causal_results/_measurement_A_sweep/
      summary.csv             one row per (model, dataset)
      per_model_token_tables/{slug}_{dataset}_top50.csv
      fig_margin_sweep.png    bar chart across all 21 models
      meta.json               run config

RUN:
    # smoke test (one model, both datasets, ~1 min):
    python src/measurement_A_sweep.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --all_datasets

    # full sweep (no v_DIM, just v_probe + v_random, ~1.5 hours):
    nohup python src/measurement_A_sweep.py --all_models --all_datasets \\
        > measurement_A_sweep.log 2>&1 &

    # full sweep WITH v_DIM (slower, ~3-4 hours):
    nohup python src/measurement_A_sweep.py --all_models --all_datasets \\
        --with_dim > measurement_A_sweep.log 2>&1 &
================================================================================
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SOURCE_BASE = '/export/fs06/hwang302/CARDS'
OUTPUT_BASE = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results/_measurement_A_sweep'
EXP10_DIR = os.path.join(SOURCE_BASE, 'exp_temporal_new')

# Full 21-model list (matches exp10 / causal_probe_test FULL_MODELS).
FULL_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B',                'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B',                       'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B',                  'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B',          'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think',               'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B',                      'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it',                 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct',
]

ABSTENTION_PHRASES = [
    "Insufficient", "Cannot", "Unable", "Unknown", "Indeterminate",
    "Impossible", "Undefined", "Undetermined", "Missing", "Not",
    "Ins", "Insuf",
]


# =============================================================================
# Token sets
# =============================================================================
def build_abstention_token_ids(tokenizer):
    ids = set()
    for phrase in ABSTENTION_PHRASES:
        for variant in [phrase, " " + phrase, phrase.lower(), " " + phrase.lower()]:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            if tids:
                ids.add(tids[0])
    return sorted(ids)


def build_numeric_token_ids(tokenizer):
    ids = set()
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    digit_starters = re.compile(r"^\s?[-+]?\.?\d")
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s or len(s) > 5:
            continue
        if digit_starters.match(s):
            ids.add(tid)
    return sorted(ids)


def get_unembed_weight(model):
    """Return the unembedding weight matrix W_U with shape (V, d)."""
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head.weight
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head.weight
    if (hasattr(model, 'model') and hasattr(model.model, 'language_model')
            and hasattr(model.model.language_model, 'lm_head')):
        return model.model.language_model.lm_head.weight
    raise RuntimeError("Could not locate lm_head")


# =============================================================================
# Direction computation
# =============================================================================
def probe_normal_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)


@torch.no_grad()
def extract_t0_states(model, tokenizer, prompts, target_layer, batch_size=4):
    states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        last_idx = (enc['attention_mask'].sum(dim=1) - 1).tolist()
        h = out.hidden_states[target_layer]
        for b, idx in enumerate(last_idx):
            states.append(h[b, idx, :].detach().to(torch.float32).cpu().numpy())
        del out
    return np.array(states)


def compute_dim_direction(model, tokenizer, dataset, slug, best_layer, n_per_class=200):
    """Build v_DIM from train-set t=0 hidden states at the probe's best layer."""
    path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_train/math/{slug}/{dataset}_cot_generations.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        train = json.load(f)
    suff = [g['prompt'] for g in train if g.get('is_sufficient', True)][:n_per_class]
    insuff = [g['prompt'] for g in train if not g.get('is_sufficient', True)][:n_per_class]
    if len(suff) < 50 or len(insuff) < 50:
        return None, None
    X_suff = extract_t0_states(model, tokenizer, suff, best_layer)
    X_insuff = extract_t0_states(model, tokenizer, insuff, best_layer)
    v = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32), norm


# =============================================================================
# Measurement A on a single (model, dataset) pair
# =============================================================================
def summarize_z(z, abs_ids, num_ids):
    z_arr = z if isinstance(z, np.ndarray) else z.cpu().numpy()
    z_abs = float(z_arr[abs_ids].mean())
    z_num = float(z_arr[num_ids].mean())
    ranks = (-z_arr).argsort().argsort()
    rank_abs_best = int(ranks[abs_ids].min())
    rank_num_best = int(ranks[num_ids].min())
    median_rank_abs = float(np.median(ranks[abs_ids]))
    median_rank_num = float(np.median(ranks[num_ids]))
    return {
        'z_mean_abs': z_abs,
        'z_mean_num': z_num,
        'margin_abs_num': z_abs - z_num,
        'best_rank_abs': rank_abs_best,
        'best_rank_num': rank_num_best,
        'median_rank_abs_pct': median_rank_abs / z_arr.shape[0],
        'median_rank_num_pct': median_rank_num / z_arr.shape[0],
        'vocab_size': int(z_arr.shape[0]),
    }


def token_table(z, tokenizer, name, abs_ids, num_ids, top_k=50):
    z_arr = z if isinstance(z, np.ndarray) else z.cpu().numpy()
    top = (-z_arr).argsort()[:top_k]
    abs_set = set(abs_ids); num_set = set(num_ids)
    rows = []
    for rank, tid in enumerate(top):
        try:
            tok_str = tokenizer.decode([int(tid)])
        except Exception:
            tok_str = '<?>'
        rows.append({
            'direction': name, 'rank': rank, 'token_id': int(tid),
            'token': repr(tok_str), 'z_value': float(z_arr[int(tid)]),
            'is_abs': int(int(tid) in abs_set), 'is_num': int(int(tid) in num_set),
        })
    return rows


# =============================================================================
# Per-model run (BOTH datasets in one load)
# =============================================================================
def run_one_model(model_name, datasets, args):
    """Load model once, compute Measurement A for each dataset, save."""
    slug = model_name.split('/')[-1]

    # Check what's needed: any of the requested (model, dataset) pairs already done?
    needed = []
    for ds in datasets:
        out_pair_dir = Path(OUTPUT_BASE) / 'per_pair' / f'{slug}__{ds}'
        if out_pair_dir.exists() and (out_pair_dir / 'DONE').exists() and not args.force:
            continue
        # also need exp10 probe + best layer
        exp10_csv = os.path.join(EXP10_DIR, 'results', f'exp10_ultimate_proportional_{ds}.csv')
        if not os.path.exists(exp10_csv):
            print(f"  [skip] {slug}/{ds}: no exp10 csv")
            continue
        df10 = pd.read_csv(exp10_csv)
        row = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
        if row.empty:
            print(f"  [skip] {slug}/{ds}: no exp10 entry")
            continue
        best_layer = int(row['Optimal_Layer'].iloc[0])
        probe_path = os.path.join(EXP10_DIR, 'probes_proportional', ds, slug,
                                  f'unified_probe_layer{best_layer}.joblib')
        if not os.path.exists(probe_path):
            print(f"  [skip] {slug}/{ds}: no probe at {probe_path}")
            continue
        needed.append({'dataset': ds, 'best_layer': best_layer, 'probe_path': probe_path,
                       'base_f1': float(row['Unified_Test_F1'].iloc[0]),
                       'out_dir': out_pair_dir})
    if not needed:
        print(f"[{slug}] nothing to do, skipping")
        return []

    # Load model
    print(f"[{slug}] loading model ({len(needed)} dataset(s) to process)...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    n_gpus = torch.cuda.device_count()
    max_mem = {i: '78GiB' for i in range(n_gpus)} if n_gpus else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', max_memory=max_mem,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation='sdpa',
        )
    except ValueError as e:
        if 'scaled_dot_product_attention' not in str(e) and 'sdpa' not in str(e).lower():
            raise
        print(f"  sdpa unsupported for {slug}, retrying with eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', max_memory=max_mem,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation='eager',
        )
    model.eval()
    W_U = get_unembed_weight(model).detach().to(torch.float32)         # (V, d)
    abs_ids = build_abstention_token_ids(tok)
    num_ids = build_numeric_token_ids(tok)

    summary_rows = []
    for spec in needed:
        ds = spec['dataset']; best_layer = spec['best_layer']
        print(f"  -> {slug}/{ds}  (best_layer={best_layer}, base_F1={spec['base_f1']:.3f})")
        probe = joblib.load(spec['probe_path'])
        v_probe = probe_normal_direction(probe)
        d = v_probe.shape[0]
        v_rand = random_direction(d, seed=42)

        # Project onto W_U
        v_probe_t = torch.tensor(v_probe, dtype=torch.float32, device=W_U.device)
        v_rand_t  = torch.tensor(v_rand,  dtype=torch.float32, device=W_U.device)
        z_probe = (W_U @ v_probe_t).cpu().numpy()
        z_rand  = (W_U @ v_rand_t).cpu().numpy()

        z_dim = None; v_dim = None; dim_mag = None
        if args.with_dim:
            v_dim, dim_mag = compute_dim_direction(model, tok, ds, slug, best_layer,
                                                    n_per_class=args.n_dim)
            if v_dim is not None:
                v_dim_t = torch.tensor(v_dim, dtype=torch.float32, device=W_U.device)
                z_dim = (W_U @ v_dim_t).cpu().numpy()

        # Summarize
        s_probe = summarize_z(z_probe, abs_ids, num_ids)
        s_rand  = summarize_z(z_rand,  abs_ids, num_ids)
        s_dim   = summarize_z(z_dim,   abs_ids, num_ids) if z_dim is not None else None

        # Save per-pair output
        out_dir = spec['out_dir']
        out_dir.mkdir(parents=True, exist_ok=True)
        # Token table
        rows = (token_table(z_probe, tok, 'v_probe', abs_ids, num_ids) +
                token_table(z_rand,  tok, 'v_rand',  abs_ids, num_ids))
        if z_dim is not None:
            rows += token_table(z_dim, tok, 'v_dim', abs_ids, num_ids)
        pd.DataFrame(rows).to_csv(out_dir / 'top50_tokens.csv', index=False)
        # Meta
        with open(out_dir / 'measurement_A.json', 'w') as f:
            json.dump({
                'model': slug, 'dataset': ds,
                'best_layer': best_layer, 'base_F1_at_0pct': spec['base_f1'],
                'hidden_dim': int(d), 'vocab_size': s_probe['vocab_size'],
                'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
                'v_probe': s_probe, 'v_rand': s_rand, 'v_dim': s_dim,
                'dim_magnitude': dim_mag,
            }, f, indent=2)
        (out_dir / 'DONE').touch()

        summary_rows.append({
            'model': slug, 'dataset': ds,
            'best_layer': best_layer, 'base_F1': spec['base_f1'],
            'hidden_dim': int(d), 'vocab_size': s_probe['vocab_size'],
            'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
            'v_probe_margin': s_probe['margin_abs_num'],
            'v_rand_margin':  s_rand['margin_abs_num'],
            'v_dim_margin':   s_dim['margin_abs_num'] if s_dim else float('nan'),
            'v_probe_z_abs': s_probe['z_mean_abs'],
            'v_probe_z_num': s_probe['z_mean_num'],
            'v_rand_z_abs':  s_rand['z_mean_abs'],
            'v_rand_z_num':  s_rand['z_mean_num'],
            'v_dim_z_abs':   s_dim['z_mean_abs'] if s_dim else float('nan'),
            'v_dim_z_num':   s_dim['z_mean_num'] if s_dim else float('nan'),
            'v_probe_abs_med_pct': s_probe['median_rank_abs_pct'],
            'v_probe_num_med_pct': s_probe['median_rank_num_pct'],
            'v_probe_n_abs_top50': sum(1 for r in token_table(z_probe, tok, 'v_probe', abs_ids, num_ids) if r['is_abs']),
            'v_probe_n_num_top50': sum(1 for r in token_table(z_probe, tok, 'v_probe', abs_ids, num_ids) if r['is_num']),
        })

    # Release model
    del model, tok, W_U
    gc.collect()
    torch.cuda.empty_cache()
    return summary_rows


# =============================================================================
# Aggregation + plot
# =============================================================================
def aggregate_and_plot(out_root):
    """Read all per-pair measurement_A.json files, build summary.csv and plot."""
    per_pair_dir = Path(out_root) / 'per_pair'
    if not per_pair_dir.exists():
        return
    rows = []
    for pair_dir in sorted(per_pair_dir.iterdir()):
        fp = pair_dir / 'measurement_A.json'
        if not fp.exists(): continue
        d = json.load(open(fp))
        rows.append({
            'model': d['model'], 'dataset': d['dataset'],
            'best_layer': d['best_layer'], 'base_F1': d['base_F1_at_0pct'],
            'hidden_dim': d['hidden_dim'], 'vocab_size': d['vocab_size'],
            'n_T_abs': d['n_T_abs'], 'n_T_num': d['n_T_num'],
            'v_probe_margin': d['v_probe']['margin_abs_num'],
            'v_rand_margin':  d['v_rand']['margin_abs_num'],
            'v_dim_margin':   d['v_dim']['margin_abs_num'] if d.get('v_dim') else float('nan'),
            'v_probe_z_abs': d['v_probe']['z_mean_abs'],
            'v_probe_z_num': d['v_probe']['z_mean_num'],
            'v_rand_z_abs':  d['v_rand']['z_mean_abs'],
            'v_rand_z_num':  d['v_rand']['z_mean_num'],
            'v_dim_z_abs':   d['v_dim']['z_mean_abs'] if d.get('v_dim') else float('nan'),
            'v_dim_z_num':   d['v_dim']['z_mean_num'] if d.get('v_dim') else float('nan'),
            'v_probe_abs_med_pct': d['v_probe']['median_rank_abs_pct'],
            'v_probe_num_med_pct': d['v_probe']['median_rank_num_pct'],
        })
    df = pd.DataFrame(rows).sort_values(['model', 'dataset'])
    df.to_csv(Path(out_root) / 'summary.csv', index=False)
    print(f"\nWrote summary across {len(df)} (model, dataset) pairs to {out_root}/summary.csv")

    # Print headline
    if len(df) == 0:
        return
    print(f"\nHEADLINE  v_probe margin range across {len(df)} pairs: "
          f"[{df['v_probe_margin'].min():+.4f}, {df['v_probe_margin'].max():+.4f}]")
    print(f"          v_rand  margin range across {len(df)} pairs: "
          f"[{df['v_rand_margin'].min():+.4f}, {df['v_rand_margin'].max():+.4f}]")
    n_both_zero = ((df['v_probe_abs_med_pct'] > 0.0) & (df['v_probe_num_med_pct'] > 0.0)).sum()
    print(f"          n pairs with no abs/num tokens in v_probe top-50: see per_pair tables")

    # Plot: dumbbell-style: v_probe vs v_rand margin per (model, dataset)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.3 * len(df))))
    df_p = df.copy().reset_index(drop=True)
    df_p['label'] = df_p['model'] + ' / ' + df_p['dataset']
    y = np.arange(len(df_p))
    ax.scatter(df_p['v_rand_margin'],  y, color='#7f7f7f', s=40, label='v_rand')
    ax.scatter(df_p['v_probe_margin'], y, color='#1f77b4', s=60, label='v_probe', zorder=3)
    if df_p['v_dim_margin'].notna().any():
        sub = df_p[df_p['v_dim_margin'].notna()]
        ax.scatter(sub['v_dim_margin'], sub.index, color='#2ca02c', s=40, label='v_dim', zorder=2)
    ax.axvline(0, color='black', lw=0.5)
    ax.axvspan(-0.02, 0.02, alpha=0.1, color='red', label=r'$|\mathrm{margin}|<0.02$')
    ax.set_yticks(y); ax.set_yticklabels(df_p['label'], fontsize=8)
    ax.set_xlabel(r'margin = $\overline{z}$(abs tokens) − $\overline{z}$(num tokens)')
    ax.set_title(f'Measurement A across {len(df_p)} (model, dataset) pairs')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(Path(out_root) / 'fig_margin_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {out_root}/fig_margin_sweep.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--with_dim', action='store_true',
                   help='Also compute v_DIM (requires forward pass on train prompts; slower).')
    p.add_argument('--n_dim', type=int, default=200)
    p.add_argument('--force', action='store_true')
    p.add_argument('--aggregate_only', action='store_true',
                   help='Skip model runs; just re-aggregate existing per-pair outputs.')
    args = p.parse_args()

    if args.aggregate_only:
        aggregate_and_plot(OUTPUT_BASE)
        return

    if args.all_models:    models = FULL_MODELS
    elif args.model:       models = [args.model]
    else:                  raise SystemExit("--model or --all_models required")
    if args.all_datasets:  datasets = ['umwp', 'treecut']
    elif args.dataset:     datasets = [args.dataset]
    else:                  raise SystemExit("--dataset or --all_datasets required")

    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    all_rows = []
    for m in models:
        try:
            rows = run_one_model(m, datasets, args)
            all_rows.extend(rows)
        except Exception as e:
            print(f"\n[ERROR] {m}: {e}")
            import traceback; traceback.print_exc()

    aggregate_and_plot(OUTPUT_BASE)


if __name__ == '__main__':
    main()
