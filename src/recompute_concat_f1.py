"""
Backfill concat_f1_curve.csv for pairs that completed before the sanity-gate fix.
Uses saved probes + saved hidden states; no model load needed.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

OUTPUT_BASE = Path('/export/fs06/hwang302/CARDS/exp_temporal_new/multi_layer')


def _load_X(pair_dir, name):
    """Load hidden states. Prefer .pt (bf16); fall back to .npy (fp16, may be corrupt)."""
    pt_path = pair_dir / 'hidden_states' / f'{name}.pt'
    npy_path = pair_dir / 'hidden_states' / f'{name}.npy'
    if pt_path.exists():
        return torch.load(pt_path).to(torch.float32).numpy()
    arr = np.load(npy_path).astype(np.float32)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"{npy_path} has {n_bad} non-finite values (fp16 overflow). "
            f"Re-extract this pair with --force to regenerate as bf16 .pt.")
    return arr


def backfill(pair_dir):
    meta = json.load(open(pair_dir / 'meta.json'))
    saved_layers = meta['saved_layers']
    X_test = _load_X(pair_dir, 'X_test')
    y_test = np.load(pair_dir / 'hidden_states' / 'y_test.npy')
    X_train = _load_X(pair_dir, 'X_train')
    y_train = np.load(pair_dir / 'hidden_states' / 'y_train.npy')
    n_pct = X_test.shape[1]
    n_saved = X_test.shape[2]
    D = X_test.shape[3]
    X_test_flat  = X_test.reshape(-1, n_saved, D)
    X_train_flat = X_train.reshape(-1, n_saved, D)
    y_test_flat  = np.repeat(y_test,  n_pct)
    y_train_flat = np.repeat(y_train, n_pct)

    rows = []
    for li, layer in enumerate(saved_layers):
        probe = joblib.load(pair_dir / 'probes' / f'unified_probe_L{layer}.joblib')
        f1_tr = f1_score(y_train_flat, probe.predict(X_train_flat[:, li, :]))
        f1_te = f1_score(y_test_flat,  probe.predict(X_test_flat[:, li, :]))
        rows.append({
            'model': meta['model'], 'dataset': meta['dataset'],
            'layer': int(layer),
            'concat_train_f1': round(float(f1_tr), 4),
            'concat_test_f1':  round(float(f1_te), 4),
            'is_best_layer':  int(layer == meta['best_layer_from_exp10']),
            'is_final_layer': int(layer == meta['final_layer']),
        })
    df = pd.DataFrame(rows)
    df.to_csv(pair_dir / 'concat_f1_curve.csv', index=False)

    best_row = df.iloc[df['concat_test_f1'].idxmax()]
    print(f"{meta['model']}/{meta['dataset']}: "
          f"reproduced best_layer = L{int(best_row['layer'])} "
          f"(concat F1 = {best_row['concat_test_f1']:.4f})  "
          f"exp10 best_layer = L{meta['best_layer_from_exp10']}")


if __name__ == '__main__':
    for marker in OUTPUT_BASE.rglob('_COMPLETE'):
        pair_dir = marker.parent
        if not (pair_dir / 'concat_f1_curve.csv').exists():
            backfill(pair_dir)
