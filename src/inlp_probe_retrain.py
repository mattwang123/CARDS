"""
================================================================================
Track 1 — INLP probe-retrain curve across 21 models × 2 datasets (FIXED)
================================================================================
Purpose: reframe "rank-K ablation didn't change behavior at K=16" as a finding
about signal high-dimensionality, not a weakness.

For each (model, dataset) at the probe's best layer:
  1. Train baseline logistic probe → F1_0
  2. Iteratively project out the probe direction (Gram-Schmidt orthogonalized
     against prior basis), retrain a new probe on the projected stream, record F1
  3. Repeat for K = 1, 2, ..., K_MAX

Output:
  experiment_result/causal_results/_inlp/
    per_pair/{slug}__{dataset}_curve.csv     # K, f1_test, f1_train
    per_pair/{slug}__{dataset}_basis.npy
    summary_curves.csv                        # long-form: model, dataset, K, f1
    fig_inlp_curves.png                       # all 42 curves + mean

Input data is cached from exp10:
  /export/fs06/hwang302/CARDS/exp_temporal_new/embeddings_proportional/{ds}/{slug}/
    X_train_layer{best_layer}.npy   shape [N, 6_pct, D]
    X_test_layer{best_layer}.npy    shape [N, 6_pct, D]
    y_train.npy, y_test.npy

The best_layer is read from exp10's CSV per (model, dataset). This matches
the layer at which causal_probe_test_v2.py's manipulation-check operates,
so INLP curve and v2 manipulation-check share the same anchor.

FIXES vs initial:
  - find_emb_files now uses exp10's best_layer instead of arbitrary first
    match. INLP curve is at the same layer as v_probe in F6/F7. If the
    expected file is missing, falls back to the first available with a
    warning (was a silent miss before).
  - K=0 sanity check: if retrained F1 deviates from exp10's reported F1 by
    more than 0.01, print a warning. Catches sklearn hyperparam drift between
    exp10's saved probe and our retrain.
  - K_MAX_DEFAULT lowered to 64. At K_MAX=128, runtime is ~14h on 42 pairs;
    K_MAX=64 is ~7h and signal saturates well before 64 in practice.
    Override with --k_max if needed.

CPU-only, no GPU needed. Runtime ~7h on a single workstation.

USAGE:
  python src/inlp_probe_retrain.py                  # all 42 pairs, K_MAX=64
  python src/inlp_probe_retrain.py --smoke          # 2 pairs, K_MAX=16 (sanity)
  python src/inlp_probe_retrain.py --k_max 128      # custom K (slow!)
  python src/inlp_probe_retrain.py --aggregate_only # rebuild summary + plot
================================================================================
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

EMB_BASE  = '/export/fs06/hwang302/CARDS/exp_temporal_new/embeddings_proportional'
EXP10_CSV = '/export/fs06/hwang302/CARDS/exp_temporal_new/results/exp10_ultimate_proportional_{ds}.csv'
OUT_BASE  = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results/_inlp'

K_MAX_DEFAULT = 64
DATASETS = ['umwp', 'treecut']
SMOKE_MODELS = ['Qwen2.5-Math-1.5B-Instruct', 'gemma-3-12b-it']
K0_F1_TOLERANCE = 0.01


def make_probe(seed):
    """Hyperparams match exp10's unified probe + causal_probe_test_v2.py's
    manipulation_check_curve. Any change here must be mirrored in the other
    locations to preserve the K=0 == exp10 invariant."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           C=1.0, solver='lbfgs', n_jobs=-1, random_state=seed)
    )


def probe_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    w = clf.coef_[0] / scaler.scale_
    n = np.linalg.norm(w)
    return (w / n).astype(np.float32) if n > 1e-12 else np.zeros_like(w, dtype=np.float32)


def inlp_curve(X_train, y_train, X_test, y_test, k_max, seed=42):
    """Iteratively project out probe directions and retrain.
    Returns rows: [{'K': k, 'f1_train': ..., 'f1_test': ...}]."""
    rows = []
    X_tr = X_train.astype(np.float32, copy=True)
    X_te = X_test.astype(np.float32,  copy=True)
    basis = []

    for k in range(0, k_max + 1):
        probe = make_probe(seed + k)
        probe.fit(X_tr, y_train)
        f1_te = f1_score(y_test,  probe.predict(X_te))
        f1_tr = f1_score(y_train, probe.predict(X_tr))
        rows.append({'K': k, 'f1_train': round(float(f1_tr), 4),
                              'f1_test':  round(float(f1_te), 4)})

        if k == k_max:
            break

        v = probe_direction(probe)
        # Gram-Schmidt against existing basis (numerical safety against drift)
        for u in basis:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n < 1e-10:
            # Exhausted the readable subspace; replicate last F1 and stop early.
            for kk in range(k + 1, k_max + 1):
                rows.append({'K': kk, 'f1_train': rows[-1]['f1_train'],
                                     'f1_test':  rows[-1]['f1_test']})
            break
        v = (v / n).astype(np.float32)
        basis.append(v)

        # Project out from both train and test
        X_tr = X_tr - np.outer(X_tr @ v, v)
        X_te = X_te - np.outer(X_te @ v, v)

    return rows, basis


def get_best_layer(slug, dataset):
    """Read exp10's CSV to find the probe's best_layer for this (model, dataset)."""
    csv_path = EXP10_CSV.format(ds=dataset)
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    row = df[(df['Model'] == slug) & (df['Percentage'] == '0%')]
    if row.empty:
        return None, None
    best_layer = int(row['Optimal_Layer'].iloc[0])
    exp10_f1   = float(row['Unified_Test_F1'].iloc[0])
    return best_layer, exp10_f1


def find_emb_files(slug, dataset, best_layer):
    """Locate cached embeddings for one (model, dataset) at the exp10
    best_layer. Returns (X_tr, y_tr, X_te, y_te, layer_actually_loaded)
    or None if missing.

    Strict mode: requires X_train_layer{best_layer}.npy. If absent, falls
    back to the first available layer file and warns — this should be rare.
    """
    d = Path(EMB_BASE) / dataset / slug
    if not d.exists():
        return None

    xtr_target = d / f'X_train_layer{best_layer}.npy'
    xte_target = d / f'X_test_layer{best_layer}.npy'
    y_tr_path  = d / 'y_train.npy'
    y_te_path  = d / 'y_test.npy'
    if not y_tr_path.exists() or not y_te_path.exists():
        return None

    if xtr_target.exists() and xte_target.exists():
        xtr_path = xtr_target
        xte_path = xte_target
        loaded_layer = best_layer
    else:
        xtr_files = sorted(d.glob('X_train_layer*.npy'))
        xte_files = sorted(d.glob('X_test_layer*.npy'))
        if not xtr_files or not xte_files:
            return None
        xtr_path = xtr_files[0]
        xte_path = xte_files[0]
        try:
            loaded_layer = int(xtr_path.stem.replace('X_train_layer', ''))
        except ValueError:
            loaded_layer = -1
        print(f"  [WARN] {slug}/{dataset}: X_train_layer{best_layer}.npy not "
              f"found; falling back to {xtr_path.name} (layer={loaded_layer}). "
              f"INLP curve will be at a DIFFERENT layer than exp10's probe.")

    X_tr = np.load(xtr_path)
    X_te = np.load(xte_path)
    y_tr = np.load(y_tr_path)
    y_te = np.load(y_te_path)
    return X_tr, y_tr, X_te, y_te, loaded_layer


def flatten_concat(X, y):
    """Flatten the 6-cutoff dimension into the sample dimension. exp10 convention.

    Expected: X shape [N, n_pct, D] → [N*n_pct, D]; y [N] → [N*n_pct] via repeat.
    If X is already 2D, returns as-is.
    """
    if X.ndim == 2:
        return X, y
    if X.ndim != 3:
        raise ValueError(f"Unexpected X.ndim={X.ndim}; expected 2 or 3")
    N, n_pct, D = X.shape
    return X.reshape(N * n_pct, D), np.repeat(y, n_pct)


def process_pair(slug, dataset, k_max, force=False):
    out_dir = Path(OUT_BASE) / 'per_pair'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'{slug}__{dataset}_curve.csv'
    if out_csv.exists() and not force:
        return None  # cached

    # 1. Look up exp10's best_layer + reported F1
    best_layer, exp10_f1 = get_best_layer(slug, dataset)
    if best_layer is None:
        return f'no exp10 entry for {slug}/{dataset}'

    # 2. Load embeddings at exp10's best_layer
    res = find_emb_files(slug, dataset, best_layer)
    if res is None:
        return f'no embeddings for {slug}/{dataset}'
    X_tr, y_tr, X_te, y_te, loaded_layer = res
    X_tr_f, y_tr_f = flatten_concat(X_tr, y_tr)
    X_te_f, y_te_f = flatten_concat(X_te, y_te)

    # 3. Run INLP
    rows, basis = inlp_curve(X_tr_f, y_tr_f, X_te_f, y_te_f, k_max=k_max)

    # 4. K=0 sanity: K=0 retrained F1 should match exp10's reported F1 if
    # sklearn hyperparams are aligned.
    f1_at_k0 = rows[0]['f1_test']
    delta = abs(f1_at_k0 - exp10_f1)
    sanity_warning = None
    if delta > K0_F1_TOLERANCE:
        sanity_warning = (f'K=0 retrained F1 ({f1_at_k0:.4f}) differs from exp10 '
                          f'F1 ({exp10_f1:.4f}) by {delta:.4f} > {K0_F1_TOLERANCE}. '
                          f'Possible sklearn hyperparam drift.')
        print(f"  [WARN] {slug}/{dataset}: {sanity_warning}")

    # 5. Write curve CSV
    df = pd.DataFrame([{'model': slug, 'dataset': dataset,
                         'best_layer': best_layer,
                         'loaded_layer': loaded_layer,
                         'exp10_anchor_f1': exp10_f1,
                         **r}
                       for r in rows])
    df.to_csv(out_csv, index=False)

    # 6. Save basis + sanity metadata
    np.save(out_dir / f'{slug}__{dataset}_basis.npy',
            np.stack(basis) if basis else np.zeros((0, X_tr_f.shape[1]), dtype=np.float32))
    meta = {
        'model': slug, 'dataset': dataset,
        'best_layer': best_layer, 'loaded_layer': loaded_layer,
        'exp10_anchor_f1': exp10_f1,
        'k0_retrained_f1': float(f1_at_k0),
        'k0_anchor_delta': float(delta),
        'k0_warning': sanity_warning,
        'k_max': k_max,
        'n_train_samples': int(X_tr_f.shape[0]),
        'n_test_samples':  int(X_te_f.shape[0]),
        'hidden_dim':      int(X_tr_f.shape[1]),
    }
    with open(out_dir / f'{slug}__{dataset}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return None


def aggregate_and_plot(out_base):
    out_base = Path(out_base)
    per_pair = out_base / 'per_pair'
    if not per_pair.exists():
        print(f"No per_pair dir at {per_pair}")
        return
    dfs = []
    for f in sorted(per_pair.glob('*_curve.csv')):
        dfs.append(pd.read_csv(f))
    if not dfs:
        print("No per-pair curves to aggregate")
        return
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(out_base / 'summary_curves.csv', index=False)
    print(f"Wrote summary_curves.csv: {len(big)} rows, "
          f"{big[['model','dataset']].drop_duplicates().shape[0]} pairs")

    # Sanity summary: which pairs have K=0 anchor mismatch?
    if 'exp10_anchor_f1' in big.columns:
        k0 = big[big['K'] == 0].copy()
        if not k0.empty:
            k0['anchor_delta'] = (k0['f1_test'] - k0['exp10_anchor_f1']).abs()
            mismatched = k0[k0['anchor_delta'] > K0_F1_TOLERANCE]
            if not mismatched.empty:
                print(f"\n[ANCHOR MISMATCH] {len(mismatched)} pairs deviate from "
                      f"exp10 at K=0 by > {K0_F1_TOLERANCE}:")
                for _, r in mismatched.iterrows():
                    print(f"  {r['model']}/{r['dataset']}: "
                          f"exp10={r['exp10_anchor_f1']:.4f} vs "
                          f"retrained={r['f1_test']:.4f} (Δ={r['anchor_delta']:.4f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = big[big['dataset'] == ds]
        for slug, g in sub.groupby('model'):
            g = g.sort_values('K')
            ax.plot(g['K'], g['f1_test'], color='gray', alpha=0.35, lw=0.8)
        mean = sub.groupby('K')['f1_test'].mean()
        ax.plot(mean.index, mean.values, color='black', lw=2.0, label='mean across models')
        ax.set_xlabel('K (directions projected out)')
        ax.set_xscale('symlog')
        ax.set_ylabel('retrained probe test F1')
        ax.set_title(f'INLP curve / {ds}  (n = {sub["model"].nunique()} models)')
        ax.axhline(0.5, color='red', ls='--', lw=0.5, label='chance')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_base / 'fig_inlp_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {out_base/'fig_inlp_curves.png'}")

    # Headline numbers
    headline_rows = []
    for ds in DATASETS:
        s = big[big['dataset'] == ds]
        for K in [1, 4, 16, 64, 128]:
            if (s['K'] == K).any():
                mean_f1 = s[s['K'] == K]['f1_test'].mean()
                std_f1  = s[s['K'] == K]['f1_test'].std()
                headline_rows.append({'dataset': ds, 'K': K,
                                       'mean_f1': round(float(mean_f1), 4),
                                       'std_f1':  round(float(std_f1),  4)})
    print('\nHeadline F1 at selected K:')
    print(pd.DataFrame(headline_rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k_max', type=int, default=K_MAX_DEFAULT)
    p.add_argument('--smoke', action='store_true',
                   help='Run only on Qwen2.5-Math-1.5B-Instruct + gemma-3-12b-it, K_MAX=16')
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--dataset', type=str, default=None, choices=DATASETS)
    p.add_argument('--force', action='store_true')
    p.add_argument('--aggregate_only', action='store_true')
    args = p.parse_args()

    if args.aggregate_only:
        aggregate_and_plot(OUT_BASE)
        return

    k_max = 16 if args.smoke else args.k_max

    # Discover pairs from filesystem
    pairs = []
    for ds in DATASETS:
        if args.dataset and ds != args.dataset:
            continue
        d = Path(EMB_BASE) / ds
        if not d.exists():
            continue
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            slug = sub.name
            if args.model and slug != args.model:
                continue
            if args.smoke and slug not in SMOKE_MODELS:
                continue
            pairs.append((slug, ds))
    pairs = sorted(set(pairs))
    print(f"Pairs to process: {len(pairs)} (K_MAX={k_max})")

    errors = []
    for slug, ds in tqdm(pairs, desc='INLP pairs'):
        try:
            err = process_pair(slug, ds, k_max=k_max, force=args.force)
            if err:
                errors.append((slug, ds, err))
        except Exception as e:
            errors.append((slug, ds, str(e)))
            print(f'  [ERROR] {slug}/{ds}: {e}')

    if errors:
        print(f'\n{len(errors)} pair(s) errored:')
        for slug, ds, err in errors:
            print(f'  {slug}/{ds}: {err}')

    aggregate_and_plot(OUT_BASE)


if __name__ == '__main__':
    main()