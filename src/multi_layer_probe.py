"""
================================================================================
multi_layer_probe.py (FIXED)
================================================================================
Extends exp10 by saving probe artifacts at the layers needed for paper claims:

  F2 robustness   : does the layer-best-F1 trajectory hold across layers,
                    not just at L*?
  F6 strong form  : is v_probe orthogonal to Δu at L*, AND at L (final layer
                    — the layer closest to unembedding)?

Reuses exp10's extraction protocol (full forward pass on prompt+CoT, pull
hidden states at six normalized cutoffs); the only structural change is that
we keep multiple saved layers instead of slicing to argmax.

LAYER SELECTION (three modes):
  --minimal_layers (RECOMMENDED for the claim X + L-anchor design):
      Save {L*-1, L*, L*+1, L_final}. ~4 layers. Storage-light, fast.
      Sufficient for: F6 at L* + F6 at L, plus immediate neighbors of L*
      as a local robustness check.

  default (dense, except large models):
      Save every layer (except sparse models). Useful for the bonus
      "decoupling holds across all layers" claim Y.

  --sparse_only:
      Save sparse layers for ALL models, not just SPARSE_MODELS.
      {0, L/8, L/4, L/2, 3L/4, L_final, L*-1, L*, L*+1}.

Outputs per (model, dataset):
  /export/fs06/hwang302/CARDS/exp_temporal_new/multi_layer/{dataset}/{slug}/
    meta.json
    hidden_states/
      X_train.pt        # [N_train, 6_pct, n_saved, D]  bfloat16 (torch.save)
      X_test.pt         # [N_test,  6_pct, n_saved, D]  bfloat16
      y_train.npy
      y_test.npy
    probes/
      unified_probe_L{l}.joblib       # one per saved layer
    v_probe_per_layer.npy             # [n_saved, D]  unit-normed
    v_dim_per_layer.npy               # [n_saved, D]  unit-normed
    f1_curve.csv                      # (layer, pct) → unified test F1
    concat_f1_curve.csv               # (layer)      → concat test F1 (exp10 metric)
    alignment.csv                     # (layer)      → margin_A for v_probe, v_dim, v_rand
    _COMPLETE                         # resume marker (written last)

Aggregate master CSVs:
  experiment_result/causal_results/_multi_layer/
    f1_curve_by_layer.csv
    alignment_by_layer.csv
    run_meta.csv

FIXES vs initial:
  - Added --minimal_layers flag. Default dense extraction is 8-16 GB per
    pair on large models; minimal mode is ~5%. Run minimal first; switch
    to dense only if the data argues for it.
  - Added --sparse_only flag to force sparse mode for all models.
  - get_saved_layers signature extended with `minimal` arg.
  - Aligned exp10 anchor loading with inlp_probe_retrain.py and
    causal_probe_test_v2.py (same csv path, same column names).

USAGE:
  # RECOMMENDED: minimal (L*-1, L*, L*+1, final), ~2 days on 21 models
  nohup python src/multi_layer_probe.py --all_models --all_datasets --minimal_layers \
      > multi_layer.log 2>&1 &

  # smoke test (Qwen-1.5B-Inst + Gemma-12B on both datasets)
  python src/multi_layer_probe.py --smoke --minimal_layers

  # full dense sweep (slow, large storage; only if you want claim Y)
  nohup python src/multi_layer_probe.py --all_models --all_datasets \
      > multi_layer_dense.log 2>&1 &

  # rebuild master CSVs from existing per-pair outputs
  python src/multi_layer_probe.py --aggregate_only
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
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# CONFIG
# ============================================================================

PERCENTAGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

SOURCE_BASE = '/export/fs06/hwang302/CARDS'
TRAIN_DIR = os.path.join(SOURCE_BASE, 'experiments/dynamic_tracking_train')
TEST_DIR = os.path.join(SOURCE_BASE, 'experiments/dynamic_tracking_test')
EXP10_DIR = os.path.join(SOURCE_BASE, 'exp_temporal_new')
OUTPUT_BASE = os.path.join(SOURCE_BASE, 'exp_temporal_new', 'multi_layer')
MASTER_CSV_DIR = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results/_multi_layer'

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

SMOKE_MODELS = ['Qwen/Qwen2.5-Math-1.5B-Instruct', 'google/gemma-3-12b-it']

# Models that get sparse-layer subsampling even in dense mode.
SPARSE_MODELS = {
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct',
}

ABSTENTION_PHRASES = [
    "Insufficient", "Cannot", "Unable", "Unknown", "Indeterminate",
    "Impossible", "Undefined", "Undetermined", "Missing", "Not",
    "Ins", "Insuf",
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_exp2_data(model_name, dataset, split_dir):
    model_slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{model_slug}/{dataset}_cot_generations.json"
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        data = json.load(f)
    labels = [1 if not d.get('is_sufficient', True) else 0 for d in data]
    return data, np.array(labels)


def load_balanced_train_prompts(slug, dataset, n_per_class=200):
    """Return (suff_prompts, insuff_prompts) for v_DIM computation."""
    path = os.path.join(TRAIN_DIR, 'math', slug, f'{dataset}_cot_generations.json')
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        train = json.load(f)
    suff = [g['prompt'] for g in train if g.get('is_sufficient', True)][:n_per_class]
    insuff = [g['prompt'] for g in train if not g.get('is_sufficient', True)][:n_per_class]
    return suff, insuff


# ============================================================================
# LAYER SET
# ============================================================================

def get_saved_layers(n_hidden_outputs, best_layer, sparse=False, minimal=False):
    """Decide which layers to extract.

    Args:
      n_hidden_outputs: len(outputs.hidden_states) = num_layers + 1
      best_layer: L* from exp10
      sparse: True for SPARSE_MODELS or --sparse_only
      minimal: True for --minimal_layers (overrides sparse/dense)

    Layer sets:
      minimal: {L*-1, L*, L*+1, L_final}
      sparse : {0, L/8, L/4, L/2, 3L/4, L_final, L*-1, L*, L*+1}
      dense  : every layer in [0, n_hidden_outputs)
    """
    L = n_hidden_outputs - 1
    if minimal:
        cand = {best_layer, max(0, best_layer - 1), min(L, best_layer + 1), L}
        return sorted(cand)
    if not sparse:
        return list(range(n_hidden_outputs))
    cand = {0, L // 8, L // 4, L // 2, (3 * L) // 4, L,
            best_layer, max(0, best_layer - 1), min(L, best_layer + 1)}
    return sorted(cand)


# ============================================================================
# HIDDEN STATE EXTRACTION
# ============================================================================

@torch.no_grad()
def detect_num_hidden_outputs(model, tokenizer):
    """Run one tiny forward pass to discover len(hidden_states)."""
    enc = tokenizer("Hello.", return_tensors='pt').to(model.device)
    out = model(**enc, output_hidden_states=True)
    n = len(out.hidden_states)
    del out
    torch.cuda.empty_cache()
    return n


@torch.no_grad()
def extract_all_layers(model, tokenizer, data, labels, saved_layers, desc=""):
    """Returns (X, y) where X is [N_valid, n_pct, n_saved, D], fp32 numpy."""
    extracted = []
    valid_labels = []
    saved_set = list(saved_layers)
    for item, label in tqdm(zip(data, labels), total=len(data), desc=desc):
        prompt_text = item['prompt']
        gen_text = item.get('generated_response', '')
        prompt_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'][0]
        full_ids = tokenizer(prompt_text + gen_text, return_tensors='pt')['input_ids'][0]
        p_len = prompt_ids.shape[0]
        total_len = full_ids.shape[0]
        cot_len = total_len - p_len
        if cot_len < 10:
            continue
        target_indices = []
        for pct in PERCENTAGES:
            idx = p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0)
            target_indices.append(min(idx, total_len - 1))

        inputs = tokenizer(prompt_text + gen_text, return_tensors='pt').to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        # Slice on GPU, then move to CPU. saved x n_pct x D.
        per_layer = []
        for l in saved_set:
            h = outputs.hidden_states[l][0, target_indices, :].to(torch.float32).cpu()
            per_layer.append(h)
        stacked = torch.stack(per_layer, dim=0)            # [n_saved, n_pct, D]
        stacked = stacked.permute(1, 0, 2).numpy()         # [n_pct, n_saved, D]
        extracted.append(stacked)
        valid_labels.append(label)
        del outputs, inputs
        torch.cuda.empty_cache()
    X = np.array(extracted, dtype=np.float32) if extracted else np.zeros((0, 0, 0, 0), dtype=np.float32)
    y = np.array(valid_labels)
    return X, y


@torch.no_grad()
def extract_t0_all_layers(model, tokenizer, prompts, saved_layers, batch_size=4):
    """Last-prompt-token hidden states at each saved layer.
    Returns [N, n_saved, D] fp32 numpy.

    Tokenizer is configured with padding_side='left', so the last real token is
    always at position seq_len-1 for every item in the batch. Using
    (attention_mask.sum-1) would index a PAD position under left padding.
    """
    assert tokenizer.padding_side == 'left', "extract_t0_all_layers assumes left padding"
    states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        bsz, seq_len = enc['input_ids'].shape
        last_idx = seq_len - 1
        for b in range(bsz):
            per_layer = []
            for l in saved_layers:
                per_layer.append(out.hidden_states[l][b, last_idx, :].to(torch.float32).cpu())
            states.append(torch.stack(per_layer, dim=0).numpy())   # [n_saved, D]
        del out
        torch.cuda.empty_cache()
    return np.array(states, dtype=np.float32)


# ============================================================================
# PROBING
# ============================================================================

def make_probe():
    """Hyperparams match exp10, inlp_probe_retrain, causal_probe_test_v2.
    Changing any param here breaks the cross-script comparability."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0,
                           solver='lbfgs', n_jobs=-1)
    )


def probe_normal_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    n = np.linalg.norm(W)
    if n < 1e-12:
        return np.zeros_like(W, dtype=np.float32)
    return (W / n).astype(np.float32)


def train_probes_per_layer(X_train, y_train, X_test, y_test, saved_layers):
    """Train one unified probe per saved layer (concat across cutoffs).
    Returns: probes (list), f1_rows (per (layer,pct)), concat_rows (per layer),
             v_probes [n_saved, D].

    concat_F1 is the metric exp10 uses to pick best_layer: F1 on the
    flattened test set (all cutoffs pooled).
    """
    N_tr, n_pct, n_saved, D = X_train.shape
    X_train_flat = X_train.reshape(-1, n_saved, D)
    y_train_flat = np.repeat(y_train, n_pct)
    X_test_flat = X_test.reshape(-1, n_saved, D)
    y_test_flat = np.repeat(y_test, n_pct)

    probes = []
    f1_rows = []
    concat_rows = []
    v_probes = np.zeros((n_saved, D), dtype=np.float32)

    for li, layer in enumerate(tqdm(saved_layers, desc='  fitting probes')):
        Xtr = X_train_flat[:, li, :]
        probe = make_probe()
        probe.fit(Xtr, y_train_flat)
        probes.append(probe)
        v_probes[li] = probe_normal_direction(probe)

        # Per-(layer, pct) F1 (matches exp10 CSV cells)
        for pi, pct in enumerate(PERCENTAGES):
            Xte_pct = X_test[:, pi, li, :]
            Xtr_pct = X_train[:, pi, li, :]
            f1_train = f1_score(y_train, probe.predict(Xtr_pct))
            f1_test = f1_score(y_test, probe.predict(Xte_pct))
            f1_rows.append({
                'layer': int(layer),
                'pct': f'{int(pct * 100)}%',
                'unified_train_f1': round(float(f1_train), 4),
                'unified_test_f1': round(float(f1_test), 4),
            })

        # Concat F1 (matches exp10's best_layer selection metric)
        f1_concat_train = f1_score(y_train_flat, probe.predict(X_train_flat[:, li, :]))
        f1_concat_test = f1_score(y_test_flat,  probe.predict(X_test_flat[:, li, :]))
        concat_rows.append({
            'layer': int(layer),
            'concat_train_f1': round(float(f1_concat_train), 4),
            'concat_test_f1': round(float(f1_concat_test), 4),
        })
    return probes, f1_rows, concat_rows, v_probes


# ============================================================================
# Δu / MEASUREMENT A
# ============================================================================

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
    # Try the common paths, including Gemma3 multimodal.
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head.weight
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head.weight
    raise RuntimeError("Could not locate lm_head")


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)


def summarize_z(z, abs_ids, num_ids):
    z_abs = float(z[abs_ids].mean())
    z_num = float(z[num_ids].mean())
    ranks = (-z).argsort().argsort()
    return {
        'z_mean_abs': z_abs,
        'z_mean_num': z_num,
        'margin_abs_num': z_abs - z_num,
        'median_rank_abs_pct': float(np.median(ranks[abs_ids])) / z.shape[0],
        'median_rank_num_pct': float(np.median(ranks[num_ids])) / z.shape[0],
    }


def measurement_A_per_layer(W_U_np, v_probes, v_dims, v_rand, saved_layers,
                             abs_ids, num_ids):
    rows = []
    for li, layer in enumerate(saved_layers):
        z_probe = W_U_np @ v_probes[li]
        z_rand = W_U_np @ v_rand
        s_p = summarize_z(z_probe, abs_ids, num_ids)
        s_r = summarize_z(z_rand, abs_ids, num_ids)
        row = {
            'layer': int(layer),
            'v_probe_margin': s_p['margin_abs_num'],
            'v_rand_margin':  s_r['margin_abs_num'],
            'v_probe_abs_med_pct': s_p['median_rank_abs_pct'],
            'v_probe_num_med_pct': s_p['median_rank_num_pct'],
        }
        if v_dims is not None:
            z_dim = W_U_np @ v_dims[li]
            s_d = summarize_z(z_dim, abs_ids, num_ids)
            row.update({
                'v_dim_margin': s_d['margin_abs_num'],
                'v_dim_abs_med_pct': s_d['median_rank_abs_pct'],
                'v_dim_num_med_pct': s_d['median_rank_num_pct'],
            })
        else:
            row.update({
                'v_dim_margin': float('nan'),
                'v_dim_abs_med_pct': float('nan'),
                'v_dim_num_med_pct': float('nan'),
            })
        rows.append(row)
    return rows


# ============================================================================
# PER-(MODEL, DATASET) PIPELINE
# ============================================================================

def process_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    pair_dir = Path(OUTPUT_BASE) / dataset / slug
    complete_marker = pair_dir / '_COMPLETE'
    if complete_marker.exists() and not args.force:
        print(f"[skip] {slug}/{dataset}: _COMPLETE marker present")
        return

    # 1) exp10 anchor
    exp10_csv = os.path.join(EXP10_DIR, 'results',
                              f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[skip] {slug}/{dataset}: no exp10 csv at {exp10_csv}")
        return
    df10 = pd.read_csv(exp10_csv)
    row10 = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row10.empty:
        print(f"[skip] {slug}/{dataset}: no exp10 entry for {slug}")
        return
    best_layer = int(row10['Optimal_Layer'].iloc[0])
    exp10_best_f1 = float(row10['Unified_Test_F1'].iloc[0])

    # 2) data
    train_data, train_labels = load_exp2_data(model_name, dataset, TRAIN_DIR)
    test_data, test_labels = load_exp2_data(model_name, dataset, TEST_DIR)
    if train_data is None or test_data is None:
        print(f"[skip] {slug}/{dataset}: missing exp2 data")
        return

    # 3) load model
    print(f"\n[{slug}/{dataset}] loading model...")
    model = None
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        n_gpus = torch.cuda.device_count()
        max_mem = None
        if n_gpus > 0:
            max_mem = {0: '65GiB'}
            for i in range(1, n_gpus):
                max_mem[i] = '78GiB'
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', max_memory=max_mem,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model.eval()

        # 4) layer set
        n_hidden_outputs = detect_num_hidden_outputs(model, tok)
        # Sparse mode is enabled either by model membership in SPARSE_MODELS
        # or globally by --sparse_only. --minimal_layers overrides both.
        sparse = (model_name in SPARSE_MODELS) or args.sparse_only
        saved_layers = get_saved_layers(
            n_hidden_outputs, best_layer,
            sparse=sparse, minimal=args.minimal_layers,
        )
        mode_tag = ('minimal' if args.minimal_layers
                    else ('sparse' if sparse else 'dense'))
        print(f"[{slug}/{dataset}] best_layer(exp10)={best_layer}, "
              f"n_hidden_outputs={n_hidden_outputs}, "
              f"saved_layers={len(saved_layers)} "
              f"({mode_tag}): {saved_layers}")

        # 5) extract train + test
        print(f"[{slug}/{dataset}] extracting train hidden states (N={len(train_data)})...")
        X_train, y_train = extract_all_layers(model, tok, train_data, train_labels,
                                              saved_layers, desc='train')
        print(f"[{slug}/{dataset}] extracting test hidden states (N={len(test_data)})...")
        X_test, y_test = extract_all_layers(model, tok, test_data, test_labels,
                                            saved_layers, desc='test')
        if X_train.size == 0 or X_test.size == 0:
            print(f"[skip] {slug}/{dataset}: no valid samples after extraction")
            return

        D = X_train.shape[-1]

        # 6) save hidden states (bf16 via torch.save to preserve range)
        hs_dir = pair_dir / 'hidden_states'
        hs_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(X_train).to(torch.bfloat16), hs_dir / 'X_train.pt')
        torch.save(torch.from_numpy(X_test).to(torch.bfloat16),  hs_dir / 'X_test.pt')
        np.save(hs_dir / 'y_train.npy', y_train)
        np.save(hs_dir / 'y_test.npy',  y_test)

        # 7) probes per layer
        print(f"[{slug}/{dataset}] fitting {len(saved_layers)} unified probes...")
        probes, f1_rows, concat_rows, v_probes = train_probes_per_layer(
            X_train, y_train, X_test, y_test, saved_layers)
        probes_dir = pair_dir / 'probes'
        probes_dir.mkdir(parents=True, exist_ok=True)
        for layer, probe in zip(saved_layers, probes):
            joblib.dump(probe, probes_dir / f'unified_probe_L{layer}.joblib')
        np.save(pair_dir / 'v_probe_per_layer.npy', v_probes)

        # sanity gates — use exp10's exact metrics for apples-to-apples comparison
        best_concat = max(concat_rows, key=lambda r: r['concat_test_f1'])
        reproduced_best_layer = best_concat['layer']
        reproduced_concat_f1 = best_concat['concat_test_f1']
        # exp10_best_f1 is F1 at (best_layer, pct=0%). Find the same cell in our run.
        anchor_match = [r for r in f1_rows
                        if r['layer'] == best_layer and r['pct'] == '0%']
        if anchor_match:
            new_f1_at_anchor = anchor_match[0]['unified_test_f1']
        else:
            # best_layer not in saved_layers (shouldn't happen in minimal/dense
            # mode, can happen in sparse if best_layer not in cand set after
            # dedup); fall back to closest saved layer.
            new_f1_at_anchor = float('nan')
        layer_best_f1 = max(r['unified_test_f1'] for r in f1_rows)

        warnings = []
        if layer_best_f1 < 0.5:
            warnings.append(f"max test_f1 {layer_best_f1:.3f} < 0.5")
        if abs(reproduced_best_layer - best_layer) > 2:
            warnings.append(f"best_layer drift (concat-F1 argmax): exp10={best_layer} vs new={reproduced_best_layer}")
        if not np.isnan(new_f1_at_anchor) and abs(new_f1_at_anchor - exp10_best_f1) > 0.005:
            warnings.append(f"F1 mismatch at exp10 anchor (L={best_layer}, pct=0%): "
                            f"exp10={exp10_best_f1:.4f} vs new={new_f1_at_anchor:.4f}")
        for w in warnings:
            print(f"  [WARN] {slug}/{dataset}: {w}")

        # 8) v_DIM per layer (t=0 protocol, matches exp13_steering_new)
        suff_prompts, insuff_prompts = load_balanced_train_prompts(slug, dataset, args.n_dim)
        v_dims = None
        dim_norms = None
        if len(suff_prompts) >= 50 and len(insuff_prompts) >= 50:
            print(f"[{slug}/{dataset}] extracting t=0 states for v_DIM "
                  f"(suff={len(suff_prompts)}, insuff={len(insuff_prompts)})...")
            X_suff_t0 = extract_t0_all_layers(model, tok, suff_prompts, saved_layers)
            X_insuff_t0 = extract_t0_all_layers(model, tok, insuff_prompts, saved_layers)
            diff = X_insuff_t0.mean(axis=0) - X_suff_t0.mean(axis=0)     # [n_saved, D]
            dim_norms = np.linalg.norm(diff, axis=1)                      # [n_saved]
            safe = np.where(dim_norms > 1e-12, dim_norms, 1.0)
            v_dims = (diff / safe[:, None]).astype(np.float32)
            np.save(pair_dir / 'v_dim_per_layer.npy', v_dims)
        else:
            print(f"[{slug}/{dataset}] skipping v_DIM (not enough balanced prompts)")

        # 9) measurement A per layer
        print(f"[{slug}/{dataset}] computing Measurement A per layer...")
        W_U = get_unembed_weight(model).detach().to(torch.float32).cpu().numpy()
        abs_ids = build_abstention_token_ids(tok)
        num_ids = build_numeric_token_ids(tok)
        v_rand = random_direction(D, seed=42)
        align_rows = measurement_A_per_layer(W_U, v_probes, v_dims, v_rand,
                                              saved_layers, abs_ids, num_ids)

        # 10) per-pair csvs + meta
        final_layer = n_hidden_outputs - 1
        f1_df = pd.DataFrame([{
            'model': slug, 'dataset': dataset, **r,
            'is_best_layer': int(r['layer'] == best_layer),
            'is_final_layer': int(r['layer'] == final_layer),
        } for r in f1_rows])
        f1_df.to_csv(pair_dir / 'f1_curve.csv', index=False)

        concat_df = pd.DataFrame([{
            'model': slug, 'dataset': dataset, **r,
            'is_best_layer': int(r['layer'] == best_layer),
            'is_final_layer': int(r['layer'] == final_layer),
        } for r in concat_rows])
        concat_df.to_csv(pair_dir / 'concat_f1_curve.csv', index=False)

        align_df = pd.DataFrame([{
            'model': slug, 'dataset': dataset, **r,
            'is_best_layer': int(r['layer'] == best_layer),
            'is_final_layer': int(r['layer'] == final_layer),
        } for r in align_rows])
        align_df.to_csv(pair_dir / 'alignment.csv', index=False)

        meta = {
            'model': slug, 'dataset': dataset,
            'best_layer_from_exp10': best_layer,
            'exp10_best_f1': exp10_best_f1,
            'reproduced_best_layer': int(reproduced_best_layer),
            'reproduced_concat_f1': float(reproduced_concat_f1),
            'new_f1_at_anchor': (float(new_f1_at_anchor)
                                 if not np.isnan(new_f1_at_anchor) else None),
            'reproduced_max_cell_f1': float(layer_best_f1),
            'final_layer': int(final_layer),
            'hidden_dim': int(D),
            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
            'percentages': PERCENTAGES,
            'saved_layers': [int(x) for x in saved_layers],
            'layer_mode': mode_tag,
            'sparse': bool(sparse),
            'minimal': bool(args.minimal_layers),
            'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
            'vocab_size': int(W_U.shape[0]),
            'has_v_dim': v_dims is not None,
            'v_dim_norms': dim_norms.tolist() if dim_norms is not None else None,
            'warnings': warnings,
        }
        with open(pair_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        complete_marker.touch()
        print(f"[{slug}/{dataset}] DONE  (max test_f1={layer_best_f1:.3f} at L{reproduced_best_layer})")
    finally:
        # Always release GPU memory, even if the body raised. This prevents
        # a dead model from squatting on VRAM across the next iteration.
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            # If CUDA is already in a bad state (e.g. cudaErrorLaunchFailure),
            # empty_cache itself will raise. Swallow so we don't mask the
            # original exception from the try block.
            pass


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate(out_root):
    out_root = Path(out_root)
    f1_dfs, align_dfs, meta_rows = [], [], []
    for meta_path in out_root.rglob('meta.json'):
        pair_dir = meta_path.parent
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta_rows.append({
                'model': meta['model'], 'dataset': meta['dataset'],
                'best_layer_from_exp10': meta['best_layer_from_exp10'],
                'exp10_best_f1': meta['exp10_best_f1'],
                'reproduced_best_layer': meta['reproduced_best_layer'],
                'reproduced_concat_f1': meta.get('reproduced_concat_f1'),
                'new_f1_at_anchor': meta.get('new_f1_at_anchor'),
                'reproduced_max_cell_f1': meta.get('reproduced_max_cell_f1',
                                                    meta.get('reproduced_max_f1')),
                'final_layer': meta['final_layer'],
                'hidden_dim': meta['hidden_dim'],
                'n_saved_layers': len(meta['saved_layers']),
                'layer_mode': meta.get('layer_mode', 'unknown'),
                'sparse': meta.get('sparse', False),
                'minimal': meta.get('minimal', False),
                'has_v_dim': meta.get('has_v_dim', False),
                'warnings': '; '.join(meta.get('warnings', [])),
            })
            if (pair_dir / 'f1_curve.csv').exists():
                f1_dfs.append(pd.read_csv(pair_dir / 'f1_curve.csv'))
            if (pair_dir / 'alignment.csv').exists():
                align_dfs.append(pd.read_csv(pair_dir / 'alignment.csv'))
        except Exception as e:
            print(f"  [aggregate] skipping {pair_dir}: {e}")
    Path(MASTER_CSV_DIR).mkdir(parents=True, exist_ok=True)
    if f1_dfs:
        f1_master = pd.concat(f1_dfs, ignore_index=True)
        f1_master.to_csv(Path(MASTER_CSV_DIR) / 'f1_curve_by_layer.csv', index=False)
        print(f"Wrote f1_curve_by_layer.csv: {len(f1_master)} rows, "
              f"{f1_master[['model','dataset']].drop_duplicates().shape[0]} pairs")
    if align_dfs:
        a_master = pd.concat(align_dfs, ignore_index=True)
        a_master.to_csv(Path(MASTER_CSV_DIR) / 'alignment_by_layer.csv', index=False)
        print(f"Wrote alignment_by_layer.csv: {len(a_master)} rows, "
              f"{a_master[['model','dataset']].drop_duplicates().shape[0]} pairs")
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(Path(MASTER_CSV_DIR) / 'run_meta.csv', index=False)
        print(f"Wrote run_meta.csv: {len(meta_rows)} pairs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='Run Qwen2.5-Math-1.5B-Instruct + gemma-3-12b-it on both datasets')
    p.add_argument('--n_dim', type=int, default=200,
                   help='Prompts per class for v_DIM (default 200)')
    p.add_argument('--minimal_layers', action='store_true',
                   help='RECOMMENDED. Save only {L*-1, L*, L*+1, L_final}. '
                        'Sufficient for claim X + L-anchor design.')
    p.add_argument('--sparse_only', action='store_true',
                   help='Use sparse layer subsampling for ALL models, not just '
                        '70B+ models. Ignored if --minimal_layers is set.')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing _COMPLETE marker')
    p.add_argument('--aggregate_only', action='store_true',
                   help='Skip extraction; just rebuild master CSVs')
    args = p.parse_args()

    if args.aggregate_only:
        aggregate(OUTPUT_BASE)
        return

    if args.smoke:
        models = SMOKE_MODELS
        datasets = ['umwp', 'treecut']
    else:
        if args.all_models:
            models = FULL_MODELS
        elif args.model:
            models = [args.model]
        else:
            raise SystemExit("Use --model, --all_models, or --smoke")
        if args.all_datasets:
            datasets = ['umwp', 'treecut']
        elif args.dataset:
            datasets = [args.dataset]
        else:
            raise SystemExit("Use --dataset or --all_datasets")

    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    for m in models:
        for d in datasets:
            try:
                process_one(m, d, args)
            except Exception as e:
                print(f"\n[ERROR] {m}/{d}: {e}")
                import traceback; traceback.print_exc()

    aggregate(OUTPUT_BASE)


if __name__ == '__main__':
    main()