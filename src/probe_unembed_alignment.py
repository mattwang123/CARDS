"""
================================================================================
[DEPRECATED, 2026-05-14] Superseded by src/measurement_A_sweep.py.
Measurements B and C are no longer used in the paper; their outputs remain
on disk as audit trail only. Do not run this script for new analyses.
================================================================================
Probe-Unembedding Alignment Test
================================================================================

CORE QUESTION:
  The probe finds a direction v_Y in residual stream at layer L* that
  classifies underspecified inputs with high F1. The model behaviorally
  hallucinates on those same inputs. Why doesn't the unembedding (W_U)
  read v_Y to push the next-token decision toward abstention tokens?

  Two possibilities:
    H_geom: v_Y is geometrically orthogonal (in W_U-space) to the
            abstention-vs-numeric distinction. The probe reads a feature
            that the unembedding does not consult for that decision.
            "Generative override" at the geometric level.
    H_mag:  v_Y *does* push for abstention through W_U, but the magnitude
            of v_Y-content in real residual streams is too small relative
            to other forces (numeric content built up during CoT).

  These have different interventions:
    Under H_geom: amplifying v_Y in residual stream cannot help; the
                  unembedding never uses it. Intervention must be at the
                  unembedding side or via a different direction.
    Under H_mag:  amplifying v_Y *should* help; the recognition direction
                  is the right direction, just too quiet in natural
                  generation.

THREE MEASUREMENTS PER (model, dataset). All sharp, no layer-wise busywork.

  (A) Static alignment of v_Y with W_U's abstention preference.
        Compute:   z = W_U @ v_Y    in R^V.
        Summaries: mean(z over T_abs), mean(z over T_num), and the gap.
        Compare against v_DIM and v_random (same metric).
        Distinguishes H_geom from H_mag at the population level.

  (B) Per-sample correlation between probe activation and final logit
      margin, on the same inputs.
        For each sample (Q1, Q2, Q3) at the force-decode position:
            r = h_{L*}[p] . v_Y                          (recognition score)
            m = LN_final(h_L[p]) . Delta_u               (logit margin abs vs num)
        Pearson correlation of (r, m) within Q1 union Q2 (insufficient inputs).
        Tests whether higher recognition predicts more abstention at output.

  (C) Cross-cutoff trajectories of r and m for Q1.
        How do recognition and decision margin evolve as CoT lengthens?
        If r stays flat and m drifts negative -> commitment grows at the
        unembedding side while recognition is preserved. This is what
        "generative override" predicts directly.

WHAT THIS SCRIPT DOES NOT DO:
  - Does not study why the probe trajectory has the cliff-and-recover
    shape (separate experiment for that).
  - Does not localize to specific blocks beyond L* and L. By design.

OUTPUTS  per (model, dataset)  under
  experiment_result/causal_results/{slug}/{dataset}/alignment/:
    alignment_summary.json   all measurements in one place
    A_token_table.csv        top-50 tokens by z = W_U @ v_Y (and same for DIM, rand)
    B_per_sample.csv         per-sample (r, m) and quadrant
    C_trajectory.csv         per-(quadrant, cutoff) mean r, mean m
    fig_A_token_ranks.png    where do abstention vs numeric tokens land?
    fig_B_scatter.png        r vs m, colored by quadrant
    fig_C_trajectory.png     r and m vs CoT cutoff for Q1

RUN:
  Smoke test (1 model, 30 samples per quadrant, ~3 min):
    python src/probe_unembed_alignment.py \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset umwp --n_per_q 30
  Full:
    nohup python src/probe_unembed_alignment.py --all_models --all_datasets \
        > alignment_run.log 2>&1 &
================================================================================
"""

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SOURCE_BASE  = '/export/fs06/hwang302/CARDS'
OUTPUT_BASE  = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results'
EXP10_DIR    = os.path.join(SOURCE_BASE, 'exp_temporal_new')

CUTOFFS = [0.0, 0.20, 0.40, 0.60, 1.00]
FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"

DEFAULT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
]

ABSTENTION_PHRASES = [
    "Insufficient", "Cannot", "Unable", "Unknown", "Indeterminate",
    "Impossible", "Undefined", "Undetermined", "Missing", "Not",
    "Ins", "Insuf",
]


# =============================================================================
# token sets
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


def get_final_norm_and_unembed(model):
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        norm = model.transformer.ln_f
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        norm = model.model.language_model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'norm'):
        norm = model.language_model.norm
    else:
        raise RuntimeError("Could not locate final norm")
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight
    else:
        raise RuntimeError("Could not locate lm_head")
    return norm, W_U


# =============================================================================
# cutoff utilities (verbatim from causal_probe_test.py)
# =============================================================================
@dataclass
class CutoffResult:
    text: str
    token_count: int
    total_tokens: int


def _token_ids(tok, text):
    return tok(text, add_special_tokens=False)['input_ids']


def _sentence_boundary_candidates(text):
    candidates = []
    for m in re.finditer(r'\n{2,}', text):            candidates.append((m.end(), 0))
    for m in re.finditer(r'[.!?](?:\s+)(?=[A-Z]|$)', text):
        candidates.append((m.end(), 1))
    for m in re.finditer(r'[.!?]\n', text):           candidates.append((m.end(), 2))
    for m in re.finditer(r'[.!?]', text):              candidates.append((m.end(), 3))
    best = {}
    for pos, prio in candidates:
        if pos not in best or prio < best[pos]:
            best[pos] = prio
    return [(pos, prio) for pos, prio in best.items()]


def find_cutoff(text, target_pct, tok):
    ids = _token_ids(tok, text)
    total = len(ids)
    if total == 0:                           return CutoffResult('', 0, 0)
    if target_pct <= 0.0:                     return CutoffResult('', 0, total)
    target = max(1, min(total, int(target_pct * total)))
    if target_pct >= 1.0:                     return CutoffResult(text, total, total)
    cands = _sentence_boundary_candidates(text)
    if not cands:
        return CutoffResult(tok.decode(ids[:target], skip_special_tokens=True),
                            target, total)
    scored = []
    for pos, prio in cands:
        ttext = text[:pos].rstrip()
        ttok = len(_token_ids(tok, ttext))
        if ttok == 0:
            continue
        scored.append((abs(ttok - target), prio, ttok, pos))
    if not scored:
        return CutoffResult(tok.decode(ids[:target], skip_special_tokens=True),
                            target, total)
    _, _, actual, char_pos = min(scored, key=lambda x: (x[0], x[1]))
    return CutoffResult(text[:char_pos].rstrip(), actual, total)


# =============================================================================
# data
# =============================================================================
def load_quadrant_samples(model_slug, dataset, max_per_q=100):
    eval_path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json')
    gen_path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json')
    if not os.path.exists(eval_path) or not os.path.exists(gen_path):
        return None
    with open(eval_path) as f:
        eval_data = json.load(f).get('data', [])
    with open(gen_path) as f:
        gen_data = json.load(f)
    out = {'Q1': [], 'Q2': [], 'Q3': []}
    quad_key = {
        'Q1_Hallucination': 'Q1',
        'Q2_Correct_Rejection': 'Q2',
        'Q3_Solved_Correctly': 'Q3',
    }
    for idx, (e, g) in enumerate(zip(eval_data, gen_data)):
        q = quad_key.get(e.get('epistemic_quadrant', ''))
        if q is None or len(out[q]) >= max_per_q:
            continue
        out[q].append({
            'sample_id': str(e.get('question_idx', e.get('sample_id', idx))),
            'prompt': g.get('prompt'),
            'cot': g.get('generated_response', g.get('model_output', '')),
            'is_sufficient': g.get('is_sufficient', True),
        })
    return out


# =============================================================================
# Probe direction (StandardScaler-corrected, unit norm)
# =============================================================================
def probe_normal_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf    = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    v = W / np.linalg.norm(W)
    return v.astype(np.float32)


# =============================================================================
# Forward to collect r (probe activation at L*) and m (final logit margin)
# =============================================================================
@torch.no_grad()
def forward_collect(model, tokenizer, forced_inputs, target_layer, v_probe, delta_u,
                    batch_size=4):
    """
    For each forced input:
      r = h_{L*}[p] . v_probe       (scalar, recognition score)
      m = LN_final(h_L[p]) . delta_u   (scalar, logit margin abs vs num)
    Returns: np.array(r), np.array(m) both shape (N,)
    """
    final_norm, _ = get_final_norm_and_unembed(model)
    v_probe_d = torch.tensor(v_probe, dtype=torch.float32, device=model.device)
    delta_u_d = delta_u.to(model.device).to(torch.float32)

    r_all, m_all = [], []
    for i in range(0, len(forced_inputs), batch_size):
        batch = forced_inputs[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        last_idx = (enc['attention_mask'].sum(dim=1) - 1).tolist()
        B = enc['input_ids'].shape[0]
        # h at target_layer  (L*)
        h_Lstar = hs[target_layer]                  # (B, T, d)
        # final residual (last block output)
        h_final = hs[-1]                            # (B, T, d)
        for b in range(B):
            r = (h_Lstar[b, last_idx[b], :].to(torch.float32) * v_probe_d).sum().item()
            h_normed = final_norm(h_final[b:b+1, last_idx[b], :])      # (1, d)
            m = (h_normed.to(torch.float32) * delta_u_d).sum().item()
            r_all.append(r); m_all.append(m)
        del out, hs
        torch.cuda.empty_cache()
    return np.array(r_all), np.array(m_all)


# =============================================================================
# Per-(model, dataset) run
# =============================================================================
def run_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    out_dir = Path(OUTPUT_BASE) / slug / dataset / 'alignment'
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / 'DONE'
    if done.exists() and not args.force:
        print(f"[DONE] {slug}/{dataset}/alignment — skipping")
        return

    # ---- exp10 probe + best layer ----
    exp10_csv = os.path.join(EXP10_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[skip] missing exp10 csv {exp10_csv}"); return
    df10 = pd.read_csv(exp10_csv)
    row = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row.empty:
        print(f"[skip] no exp10 entry for {slug}/{dataset}"); return
    best_layer = int(row['Optimal_Layer'].iloc[0])
    probe_path = os.path.join(EXP10_DIR, 'probes_proportional', dataset, slug,
                              f'unified_probe_layer{best_layer}.joblib')
    if not os.path.exists(probe_path):
        print(f"[skip] missing probe at {probe_path}"); return
    probe = joblib.load(probe_path)
    v_probe = probe_normal_direction(probe)        # unit norm, shape (d,)

    samples = load_quadrant_samples(slug, dataset, max_per_q=args.n_per_q)
    if samples is None or len(samples['Q1']) == 0:
        print(f"[skip] missing samples"); return
    n_per_q = {q: len(samples[q]) for q in samples}
    print(f"[RUN]  {slug}/{dataset}/alignment   L*={best_layer}   {n_per_q}")

    # ---- load model ----
    print("  loading model...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    n_gpus = torch.cuda.device_count()
    max_mem = {i: '78GiB' for i in range(n_gpus)} if n_gpus else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', max_memory=max_mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()

    # ---- token sets + W_U + delta_u ----
    abs_ids = build_abstention_token_ids(tok)
    num_ids = build_numeric_token_ids(tok)
    if not abs_ids or not num_ids:
        print(f"  ! degenerate token sets: |abs|={len(abs_ids)}, |num|={len(num_ids)}"); return
    _, W_U = get_final_norm_and_unembed(model)
    W_U_f32 = W_U.to(torch.float32).detach()                         # (V, d)
    delta_u = W_U_f32[abs_ids].mean(dim=0) - W_U_f32[num_ids].mean(dim=0)   # (d,)
    print(f"  |T_abs|={len(abs_ids)}, |T_num|={len(num_ids)}, ||delta_u||={delta_u.norm().item():.3f}")

    # ----------------------------------------------------------------------
    # Measurement A: static alignment of v_Y with W_U's abstention preference
    # ----------------------------------------------------------------------
    print("  [A] static alignment  z = W_U @ v_probe ...")
    v_probe_t = torch.tensor(v_probe, dtype=torch.float32, device=W_U_f32.device)
    z_probe = (W_U_f32 @ v_probe_t).cpu().numpy()                    # (V,)

    # Compute DIM and random for comparison. Use Q1+Q2 t=0 residuals to build DIM.
    print("  [A] building v_DIM and v_random at L*...")
    v_random = np.random.RandomState(42).randn(W_U_f32.shape[1]).astype(np.float32)
    v_random /= np.linalg.norm(v_random)
    # DIM from train data t=0:
    train_path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_train/math/{slug}/{dataset}_cot_generations.json')
    v_dim = None
    if os.path.exists(train_path):
        with open(train_path) as f: train = json.load(f)
        suff_p = [g['prompt'] for g in train if g.get('is_sufficient', True)][:200]
        insuff_p = [g['prompt'] for g in train if not g.get('is_sufficient', True)][:200]
        if len(suff_p) >= 50 and len(insuff_p) >= 50:
            # extract t=0 hidden states at L*
            @torch.no_grad()
            def extract(prompts):
                states = []
                for k in range(0, len(prompts), 4):
                    enc = tok(prompts[k:k+4], return_tensors='pt', padding=True).to(model.device)
                    out = model(**enc, output_hidden_states=True)
                    last_idx = (enc['attention_mask'].sum(dim=1) - 1).tolist()
                    h = out.hidden_states[best_layer]
                    for b, idx in enumerate(last_idx):
                        states.append(h[b, idx, :].detach().to(torch.float32).cpu().numpy())
                    del out
                return np.array(states)
            X_suff = extract(suff_p)
            X_insuff = extract(insuff_p)
            v_dim_raw = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
            v_dim = v_dim_raw / np.linalg.norm(v_dim_raw)

    v_dim_t = torch.tensor(v_dim, dtype=torch.float32, device=W_U_f32.device) if v_dim is not None else None
    v_rand_t = torch.tensor(v_random, dtype=torch.float32, device=W_U_f32.device)
    z_dim = (W_U_f32 @ v_dim_t).cpu().numpy() if v_dim_t is not None else None
    z_rand = (W_U_f32 @ v_rand_t).cpu().numpy()

    def summarize_z(z):
        if z is None: return None
        z_abs = float(z[abs_ids].mean())
        z_num = float(z[num_ids].mean())
        # rank-based summaries
        ranks = (-z).argsort().argsort()       # rank 0 = highest z
        rank_abs_best = int(ranks[abs_ids].min())
        rank_num_best = int(ranks[num_ids].min())
        median_rank_abs = float(np.median(ranks[abs_ids]))
        median_rank_num = float(np.median(ranks[num_ids]))
        # top-10 token ids by z
        top_k = (-z).argsort()[:50]
        return {
            'z_mean_abs': z_abs, 'z_mean_num': z_num, 'margin_abs_num': z_abs - z_num,
            'best_rank_abs': rank_abs_best, 'best_rank_num': rank_num_best,
            'median_rank_abs': median_rank_abs, 'median_rank_num': median_rank_num,
            'vocab_size': int(z.shape[0]),
            'top_50_token_ids': top_k.tolist(),
        }

    A_results = {
        'v_probe': summarize_z(z_probe),
        'v_dim':   summarize_z(z_dim),
        'v_rand':  summarize_z(z_rand),
    }

    # Save the top-50 token table for inspection
    def token_table(z, name):
        if z is None: return pd.DataFrame()
        top_k = (-z).argsort()[:50]
        rows = []
        for rank, tid in enumerate(top_k):
            try:
                tok_str = tok.decode([int(tid)])
            except Exception:
                tok_str = '<?>'
            rows.append({
                'direction': name, 'rank': rank, 'token_id': int(tid),
                'token': repr(tok_str), 'z_value': float(z[int(tid)]),
                'is_abs': int(int(tid) in set(abs_ids)),
                'is_num': int(int(tid) in set(num_ids)),
            })
        return pd.DataFrame(rows)
    A_table = pd.concat([
        token_table(z_probe, 'v_probe'),
        token_table(z_dim,   'v_dim'),
        token_table(z_rand,  'v_rand'),
    ], ignore_index=True)
    A_table.to_csv(out_dir / 'A_token_table.csv', index=False)

    print(f"  [A] v_probe : margin (z_abs - z_num) = {A_results['v_probe']['margin_abs_num']:+.4f}")
    if A_results['v_dim'] is not None:
        print(f"  [A] v_dim   : margin (z_abs - z_num) = {A_results['v_dim']['margin_abs_num']:+.4f}")
    print(f"  [A] v_rand  : margin (z_abs - z_num) = {A_results['v_rand']['margin_abs_num']:+.4f}")

    # ----------------------------------------------------------------------
    # Measurement B: per-sample (r, m) at cutoff 100% (where decisions are
    # finalized in baseline behavior)
    # ----------------------------------------------------------------------
    print("  [B] per-sample (r, m) at cutoff 100% ...")
    B_rows = []
    for q in ['Q1', 'Q2', 'Q3']:
        forced = []
        for s in samples[q]:
            c = find_cutoff(s['cot'], 1.0, tok)
            forced.append(s['prompt'] + c.text + FORCE_DECODE_SUFFIX)
        if not forced: continue
        r_arr, m_arr = forward_collect(model, tok, forced, best_layer, v_probe, delta_u,
                                       batch_size=args.batch_size)
        for sid, r, m in zip([s['sample_id'] for s in samples[q]], r_arr, m_arr):
            B_rows.append({'quadrant': q, 'sample_id': sid, 'r': float(r), 'm': float(m), 'cutoff': 100})

    B_df = pd.DataFrame(B_rows)
    B_df.to_csv(out_dir / 'B_per_sample.csv', index=False)

    insuff_df = B_df[B_df['quadrant'].isin(['Q1', 'Q2'])]
    if len(insuff_df) >= 5:
        rho, pval = pearsonr(insuff_df['r'], insuff_df['m'])
    else:
        rho, pval = float('nan'), float('nan')
    # Quadrant means
    quad_summary = B_df.groupby('quadrant').agg(
        mean_r=('r', 'mean'), std_r=('r', 'std'),
        mean_m=('m', 'mean'), std_m=('m', 'std'),
        n=('r', 'count'),
    ).reset_index()
    print("  [B] per-quadrant means at cutoff 100%:")
    print("  " + quad_summary.to_string(index=False).replace('\n', '\n  '))
    print(f"  [B] Pearson(r, m) within Q1 union Q2: rho={rho:+.4f}  p={pval:.3e}  n={len(insuff_df)}")

    # ----------------------------------------------------------------------
    # Measurement C: trajectory of r and m for Q1 across cutoffs
    # ----------------------------------------------------------------------
    print("  [C] trajectories of (r, m) across cutoffs for Q1, Q2, Q3 ...")
    C_rows = []
    for q in ['Q1', 'Q2', 'Q3']:
        for cutoff in CUTOFFS:
            ck = int(round(cutoff * 100))
            forced = []
            for s in samples[q]:
                if cutoff <= 0.0: truncated = ''
                else:
                    c = find_cutoff(s['cot'], cutoff, tok); truncated = c.text
                forced.append(s['prompt'] + truncated + FORCE_DECODE_SUFFIX)
            if not forced: continue
            r_arr, m_arr = forward_collect(model, tok, forced, best_layer, v_probe, delta_u,
                                           batch_size=args.batch_size)
            C_rows.append({
                'quadrant': q, 'cutoff_pct': ck, 'n': len(forced),
                'mean_r': float(r_arr.mean()), 'std_r': float(r_arr.std()),
                'mean_m': float(m_arr.mean()), 'std_m': float(m_arr.std()),
            })
    C_df = pd.DataFrame(C_rows)
    C_df.to_csv(out_dir / 'C_trajectory.csv', index=False)

    # ----------------------------------------------------------------------
    # Save everything
    # ----------------------------------------------------------------------
    meta = {
        'model': slug, 'dataset': dataset,
        'probe_best_layer': best_layer,
        'probe_base_f1_at_0pct': float(row['Unified_Test_F1'].iloc[0]),
        'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
        'delta_u_norm': float(delta_u.norm()),
        'n_per_q': n_per_q,
        'measurement_A': A_results,
        'measurement_B_correlation_r_m_within_insuff': {
            'pearson_rho': float(rho), 'p_value': float(pval),
            'n_samples_insuff': int(len(insuff_df)),
        },
        'measurement_B_per_quadrant_means_at_100': quad_summary.to_dict('records'),
    }
    with open(out_dir / 'alignment_summary.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # plots
    make_plots(A_results, B_df, C_df, out_dir, slug, dataset, best_layer)

    done.touch()
    del model, tok, W_U, W_U_f32, delta_u
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  -> done. {out_dir}")


def make_plots(A, B_df, C_df, out_dir, slug, dataset, best_layer):
    # ----- A plot: margin per direction, plus rank of best abstention vs numeric token
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    dirs = []; margins = []
    for d_name in ['v_probe', 'v_dim', 'v_rand']:
        r = A.get(d_name)
        if r is None: continue
        dirs.append(d_name); margins.append(r['margin_abs_num'])
    ax.bar(dirs, margins, color=['#1f77b4', '#2ca02c', '#7f7f7f'])
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel(r'mean $z$ on abstention − mean $z$ on numeric')
    ax.set_title(f'{slug}/{dataset}: $z = W_U \\cdot v$ pushes for abstention?')
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    width = 0.35
    xs = np.arange(len(dirs))
    abs_ranks = [A[d]['median_rank_abs'] / A[d]['vocab_size'] for d in dirs]
    num_ranks = [A[d]['median_rank_num'] / A[d]['vocab_size'] for d in dirs]
    ax.bar(xs - width/2, abs_ranks, width, label='abstention tokens (median rank percentile)', color='#1f77b4')
    ax.bar(xs + width/2, num_ranks, width, label='numeric tokens (median rank percentile)', color='#d62728')
    ax.set_xticks(xs); ax.set_xticklabels(dirs)
    ax.set_ylabel('median rank / vocab size  (lower = more activated)')
    ax.set_title('Where do abstention and numeric tokens rank under each direction?')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(out_dir / 'fig_A_token_ranks.png', dpi=150, bbox_inches='tight'); plt.close()

    # ----- B plot: scatter of (r, m), colored by quadrant
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {'Q1': '#d62728', 'Q2': '#2ca02c', 'Q3': '#7f7f7f'}
    for q in ['Q3', 'Q1', 'Q2']:
        sub = B_df[B_df['quadrant'] == q]
        if sub.empty: continue
        ax.scatter(sub['r'], sub['m'], color=colors[q], alpha=0.55,
                   label=f'{q} (n={len(sub)})', s=24)
    ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel(r'recognition score:  $r = h_{L^{*}}^{(p)} \cdot v_Y$' + f'    (L*={best_layer})')
    ax.set_ylabel(r'final logit margin:  $m = \mathrm{LN}_\mathrm{final}(h_L^{(p)}) \cdot \Delta u$')
    ax.set_title(f'{slug}/{dataset}: probe activation vs unembedding decision  (cutoff = 100%)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / 'fig_B_scatter.png', dpi=150, bbox_inches='tight'); plt.close()

    # ----- C plot: trajectory of mean_r and mean_m per quadrant across cutoffs
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, col, ylab in [(axes[0], 'mean_r', r'mean recognition score $\bar r$'),
                          (axes[1], 'mean_m', r'mean final logit margin $\bar m$')]:
        for q in ['Q1', 'Q2', 'Q3']:
            sub = C_df[C_df['quadrant'] == q].sort_values('cutoff_pct')
            if sub.empty: continue
            ax.plot(sub['cutoff_pct'], sub[col], color={'Q1': '#d62728', 'Q2': '#2ca02c', 'Q3': '#7f7f7f'}[q],
                    lw=2, marker='o', label=f'{q} (n={int(sub.iloc[0]["n"])})')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('CoT cutoff %'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend()
    plt.suptitle(f'{slug}/{dataset}: recognition vs decision across CoT length', y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / 'fig_C_trajectory.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved 3 figures to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--n_per_q', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    if args.all_models:    models = DEFAULT_MODELS
    elif args.model:       models = [args.model]
    else:                  raise SystemExit("Specify --model or --all_models")

    if args.all_datasets:  datasets = ['umwp', 'treecut']
    elif args.dataset:     datasets = [args.dataset]
    else:                  raise SystemExit("Specify --dataset or --all_datasets")

    for m in models:
        for ds in datasets:
            try: run_one(m, ds, args)
            except Exception as e:
                print(f"\n[ERROR] {m}/{ds}: {e}")
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
