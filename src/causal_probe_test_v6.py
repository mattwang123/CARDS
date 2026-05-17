"""
================================================================================
causal_probe_test_v6.py
================================================================================
Unified F7 causal intervention. Two intervention modes share ~85% scaffolding
(exp10 anchor, v_DIM, INLP bases, alpha scale, manipulation check, conditions)
and only differ in the decode step.

  decision    (Option A, "mechanism evidence", FAST)
    Input  : prompt + exp2_cot truncated at first \\boxed{, keeping the
             opening brace. The \\boxed{ in the input is exp2's OWN brace
             (NOT a second appended one — V3's bug).
    Decode : greedy, ≤30 new tokens, stop at first '}'.
    Tests  : "at the commit step, does intervention change the next token?"
             Maps to paper claim's literal wording (abstain-vs-numeric token).
    Speed  : per pair ~30-90 min.

  natural     (Option B, "deployment evidence", SLOW)
    Input  : prompt only.
    Decode : greedy, ≤4096 new tokens, hook active throughout, stop when all
             sequences emit a closed \\boxed{...}.
    Tests  : "does intervention change deployment-level outcome end-to-end?"
             Stronger claim, mixes reasoning + decision effects, ~3-4 hrs/pair.

Baseline (both modes): exp2 cached classification (deterministic by quadrant
selection — Q1/Q3 insuff_rate=0.0, Q2/Q4 insuff_rate=1.0). Injected into
summary.csv as 'baseline_exp2' rows with zero CI width. No baseline forward
pass needed. With --include_baseline_check, an additional no-hook condition
'baseline_check' is run as a protocol-drift sanity check.

Sample shortage: math models rarely over-abstain so Q4 is sparse (N=0..87
across pairs). Cells with N < N_MIN_PER_QUADRANT=30 are flagged in
summary.csv (included=False) and dropped from aggregate plots.

Output structure (per pair):
  causal_results_v6/{slug}/{dataset}/
    meta.json                        # shared config + per-mode N
    manipulation_check.csv           # K vs retrained F1, shared
    basis_probe.npy
    basis_dim.npy
    decision/
      summary.csv                    # mode-specific
      sample_major.json
      DONE
    natural/
      summary.csv
      sample_major.json
      DONE

Resume:
  - Per-pair: both DONE → skip.
  - Per-mode: that mode's DONE → skip mode, keep going with the other.
  - Within mode: sample_major.json tracks per-(sample, condition) state.

Usage:
  python causal_probe_test_v6.py --test                       # smoke, both modes
  python causal_probe_test_v6.py --mode decision --all_models --all_datasets
  python causal_probe_test_v6.py --mode natural  --all_models --all_datasets
  python causal_probe_test_v6.py --mode both     --all_models --all_datasets
  python causal_probe_test_v6.py --aggregate_only            # plots from existing
================================================================================
"""

import argparse
import gc
import json
import os
import re
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse low-level mechanics from v1
from causal_probe_test import (
    SubspaceAblationHook,
    EXP10_DIR,
    SOURCE_BASE,
    classify_answer,
    diff_in_means_direction,
    extract_t0_states,
    extract_boxed,
    get_layer_modules,
    get_q1_q3_samples,
    load_existing_records,
    load_train_balanced_prompts,
    register_addition_hook,
    remove_handles,
)

# Reuse from v3
from causal_probe_test_v3 import (
    make_probe,
    direction_from_probe,
    build_inlp_basis,
    random_orthonormal_basis,
    compute_basis_means,
    load_exp10_embeddings,
    get_q2_q4_samples,
    MeanReplacementAblationHook,
    register_ablation,
    bootstrap_ci_rate,
    bootstrap_ci_f1,
    manipulation_check_curve,
    compute_layer_residual_norm,
    _alpha_to_str,
)


# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_BASE = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results_v6'

# Mode-specific decode parameters
BOXED_OPEN = "\\boxed{"
BOXED_PATTERN = re.compile(r'\\boxed\{[^{}]*\}')

MAX_NEW_TOKENS_DECISION = 30        # decision mode: short, just boxed content + }
MAX_NEW_TOKENS_NATURAL = 4096       # natural mode: long CoT, stop_at_boxed terminates earlier

BATCH_DECISION = 16                  # short gen, can batch wider
BATCH_NATURAL = 4                    # long gen, whole-batch wait is costly

# Mode registry
MODE_DECISION = 'decision'
MODE_NATURAL = 'natural'
MODES_ALL = [MODE_DECISION, MODE_NATURAL]

# Condition matrix (shared)
ALPHAS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
RANK_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
RANK_K_MAX = 128
ABLATION_LAYERS = ['all', 'lstar_only']
ABLATION_METHODS = ['zero', 'mean']
DIRECTIONS = ['probe', 'dim', 'rand']
N_BOOTSTRAP = 1000

N_Q1 = 200
N_Q2 = 200
N_Q3 = 100
N_Q4 = 100
N_DIM_PER_CLASS = 300

N_MIN_PER_QUADRANT = 30              # bootstrap CI useless below this
SAMPLE_WARN_FRAC = 0.5

K0_F1_TOLERANCE = 0.005

PAPER_MODELS_V6 = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
]


# ============================================================================
# STOP CRITERIA (mode-specific)
# ============================================================================

class StopAtClosingBrace(StoppingCriteria):
    """For DECISION mode: stop when every sequence emits '}' (closes the
    \\boxed{ that's at the end of the input)."""
    def __init__(self, tokenizer, prompt_len, batch_size, check_every=2):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.done = [False] * batch_size
        self._step = 0
        self.check_every = check_every

    def __call__(self, input_ids, scores, **kwargs):
        self._step += 1
        if self._step % self.check_every != 0:
            return all(self.done)
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            new_tokens = input_ids[i, self.prompt_len:]
            if new_tokens.numel() < 1:
                continue
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if '}' in text:
                self.done[i] = True
        return all(self.done)


class StopWhenAllBoxed(StoppingCriteria):
    """For NATURAL mode: stop when every sequence emits a closed \\boxed{...}."""
    def __init__(self, tokenizer, prompt_len, batch_size, check_every=2):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.done = [False] * batch_size
        self._step = 0
        self.check_every = check_every

    def __call__(self, input_ids, scores, **kwargs):
        self._step += 1
        if self._step % self.check_every != 0:
            return all(self.done)
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            new_tokens = input_ids[i, self.prompt_len:]
            if new_tokens.numel() < 5:
                continue
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if BOXED_PATTERN.search(text):
                self.done[i] = True
        return all(self.done)


# ============================================================================
# DECODE HELPERS (mode-specific)
# ============================================================================

def make_decision_input(prompt, exp2_cot):
    """Construct the input that places model at the exact state right
    before committing the boxed answer. The \\boxed{ in the returned
    string is exp2's OWN opening brace (or a fallback-appended one if
    exp2 didn't use \\boxed{}). NEVER double-appended."""
    idx = exp2_cot.find(BOXED_OPEN)
    if idx == -1:
        return prompt + exp2_cot + BOXED_OPEN
    return prompt + exp2_cot[:idx + len(BOXED_OPEN)]


@torch.no_grad()
def batched_decode_decision(model, tokenizer, inputs, batch_size,
                             max_new_tokens=MAX_NEW_TOKENS_DECISION):
    """DECISION mode: short generation, stop at first '}'. Hook may be
    attached. Returns list of {'raw_output', 'boxed_content'}."""
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=False).to(model.device)
        prompt_len = enc['input_ids'].shape[1]
        stopping = StoppingCriteriaList([
            StopAtClosingBrace(tokenizer, prompt_len, len(batch),
                                check_every=2),
        ])
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            stopping_criteria=stopping, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        for j in range(len(batch)):
            new_tokens = out.sequences[j, prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            close_idx = text.find('}')
            boxed_content = text[:close_idx] if close_idx >= 0 else text
            results.append({
                'raw_output': text,
                'boxed_content': boxed_content,
            })
        del enc, out
        torch.cuda.empty_cache()
    return results


@torch.no_grad()
def batched_decode_natural(model, tokenizer, inputs, batch_size,
                            max_new_tokens=MAX_NEW_TOKENS_NATURAL):
    """NATURAL mode: long autoregressive generation, stop when all
    sequences emit closed \\boxed{...}. Hook is active throughout. Returns
    list of {'raw_output', 'boxed_content'}."""
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=False).to(model.device)
        prompt_len = enc['input_ids'].shape[1]
        stopping = StoppingCriteriaList([
            StopWhenAllBoxed(tokenizer, prompt_len, len(batch),
                              check_every=2),
        ])
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            stopping_criteria=stopping, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        for j in range(len(batch)):
            new_tokens = out.sequences[j, prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            m = BOXED_PATTERN.search(text)
            if m:
                text = text[:m.end()]
            boxed = extract_boxed(text)
            results.append({
                'raw_output': text,
                'boxed_content': boxed if boxed else '',
            })
        del enc, out
        torch.cuda.empty_cache()
    return results


# ============================================================================
# MODE DISPATCH
# ============================================================================

def make_inputs(samples, mode):
    if mode == MODE_DECISION:
        return [make_decision_input(s['prompt'], s.get('cot', ''))
                for s in samples]
    elif mode == MODE_NATURAL:
        return [s['prompt'] for s in samples]
    raise ValueError(f"unknown mode: {mode}")


def decode_for_mode(model, tokenizer, inputs, mode, batch_size=None):
    if mode == MODE_DECISION:
        bs = batch_size or BATCH_DECISION
        return batched_decode_decision(model, tokenizer, inputs, bs)
    elif mode == MODE_NATURAL:
        bs = batch_size or BATCH_NATURAL
        return batched_decode_natural(model, tokenizer, inputs, bs)
    raise ValueError(f"unknown mode: {mode}")


def is_coherent(out_dict, mode):
    raw = out_dict.get('raw_output', '')
    boxed = out_dict.get('boxed_content', '')
    if mode == MODE_DECISION:
        if not raw or boxed is None:
            return False
        s = boxed.strip()
        if len(s) == 0 or len(s) > 50:
            return False
        return True
    else:  # natural
        if boxed is None or boxed == '' and not raw:
            return False
        if not raw or len(raw.strip()) < 5:
            return False
        # 5-gram degeneracy heuristic
        words = raw.split()
        if len(words) >= 25:
            from collections import Counter
            five_grams = [' '.join(words[k:k+5])
                          for k in range(len(words) - 4)]
            mc = Counter(five_grams).most_common(1)
            if mc and mc[0][1] > 4:
                return False
        return True


def is_verbalize_correct(boxed_content, is_sufficient):
    kind = classify_answer(boxed_content) if boxed_content else 'other'
    if is_sufficient and kind == 'numeric':
        return True
    if (not is_sufficient) and kind == 'insufficient':
        return True
    return False


def parse_output(out_dict, sample, mode):
    boxed = out_dict.get('boxed_content', '')
    kind = classify_answer(boxed) if boxed else 'other'
    coh = is_coherent(out_dict, mode)
    correct = is_verbalize_correct(boxed, sample.get('is_sufficient', True))
    rec = {
        'boxed_content': boxed,
        'answer_kind': kind,
        'coherent': coh,
        'verbalize_correct': correct,
    }
    if mode == MODE_NATURAL:
        # store truncated raw for inspection
        rec['raw_output'] = out_dict.get('raw_output', '')[:1000]
    return rec


# ============================================================================
# CONDITION BUILDER (shared between modes)
# ============================================================================

def _register_steering(target_module, v_t, a_raw, model=None):
    return register_addition_hook(target_module, v_t, a_raw)


def _register_baseline(model=None):
    return []


def build_conditions(target_module, layer_modules, lstar,
                     directions_with_bases, alpha_scale, args):
    """Default: no baseline force-decode (baseline = exp2 deployment,
    injected into summary post-hoc with definitional values per quadrant).
    With --include_baseline_check, prepend one no-hook condition for
    protocol sanity."""
    conditions = []
    if getattr(args, 'include_baseline_check', False):
        conditions.append(('baseline_check', _register_baseline))

    if not args.skip_steering:
        for name, v_unit, _basis, _means in directions_with_bases:
            for a in args.alphas:
                a_raw = a * alpha_scale
                v_t = torch.tensor(v_unit, dtype=torch.float32)
                conditions.append((
                    f'steer_{name}_{_alpha_to_str(a)}',
                    partial(_register_steering, target_module, v_t, a_raw),
                ))

    if not args.skip_ablation:
        for K in args.rank_K_values:
            for name, _v, basis_full, mean_full in directions_with_bases:
                K_eff = min(K, basis_full.shape[0])
                B = basis_full[:K_eff].copy()
                M = mean_full[:K_eff].copy()
                for layer_mode in args.ablation_layers:
                    lm_short = 'all' if layer_mode == 'all' else 'lstar'
                    for method in args.ablation_methods:
                        mv = None if method == 'zero' else M
                        conditions.append((
                            f'ablate_{name}_K{K}_{lm_short}_{method}',
                            partial(register_ablation, layer_modules, B, mv,
                                    layer_mode, lstar),
                        ))
    return conditions


# ============================================================================
# AGGREGATION (shared between modes)
# ============================================================================

QUADRANT_BASELINE = {
    'Q1_Hallucination':    {'insuff_rate': 0.0, 'numeric_rate': 1.0,
                            'verbalize_acc': 0.0},
    'Q2_Correct_Abstain':  {'insuff_rate': 1.0, 'numeric_rate': 0.0,
                            'verbalize_acc': 1.0},
    'Q3_Correct_Solve':    {'insuff_rate': 0.0, 'numeric_rate': 1.0,
                            'verbalize_acc': 1.0},
    'Q4_Over_Abstain':     {'insuff_rate': 1.0, 'numeric_rate': 0.0,
                            'verbalize_acc': 0.0},
}


def aggregate_summary(sample_records, slug, dataset, out_path,
                     n_boot=N_BOOTSTRAP, n_min=N_MIN_PER_QUADRANT):
    """Schema: model, dataset, condition, quadrant, n,
       insuff_rate (+ CI), numeric_rate (+ CI), coherence_rate (+ CI),
       verbalize_acc (+ CI), included (bool).

    Injects synthetic 'baseline_exp2' rows per quadrant with deterministic
    rates and zero CI width (Q1/Q3 = numeric, Q2/Q4 = insufficient, by
    quadrant definition)."""
    groups = {}
    quadrant_counts = {}
    for rec in sample_records:
        quad = rec.get('quadrant', '')
        quadrant_counts[quad] = quadrant_counts.get(quad, 0) + 1
        for cond_name, v in rec.get('conditions', {}).items():
            key = (cond_name, quad)
            groups.setdefault(key, []).append(v)

    rows = []

    # exp2-deployment baseline rows (deterministic)
    for quad, n in quadrant_counts.items():
        if quad not in QUADRANT_BASELINE:
            continue
        base = QUADRANT_BASELINE[quad]
        rows.append({
            'model': slug, 'dataset': dataset,
            'condition': 'baseline_exp2', 'quadrant': quad, 'n': n,
            'insuff_rate':       base['insuff_rate'],
            'insuff_ci_low':     base['insuff_rate'],
            'insuff_ci_high':    base['insuff_rate'],
            'numeric_rate':      base['numeric_rate'],
            'numeric_ci_low':    base['numeric_rate'],
            'numeric_ci_high':   base['numeric_rate'],
            'coherence_rate':    1.0,
            'coherence_ci_low':  1.0,
            'coherence_ci_high': 1.0,
            'verbalize_acc':     base['verbalize_acc'],
            'verbalize_ci_low':  base['verbalize_acc'],
            'verbalize_ci_high': base['verbalize_acc'],
            'included':          (n >= n_min),
        })

    # Intervention rows (bootstrap)
    for (cond, quad), vs in sorted(groups.items()):
        n = len(vs)
        if n == 0:
            continue
        ins = np.array([1.0 if v['answer_kind'] == 'insufficient' else 0.0 for v in vs])
        num = np.array([1.0 if v['answer_kind'] == 'numeric'      else 0.0 for v in vs])
        coh = np.array([1.0 if v.get('coherent', False)            else 0.0 for v in vs])
        vc  = np.array([1.0 if v.get('verbalize_correct', False)   else 0.0 for v in vs])
        ins_m, ins_lo, ins_hi = bootstrap_ci_rate(ins, n_boot=n_boot)
        num_m, num_lo, num_hi = bootstrap_ci_rate(num, n_boot=n_boot)
        coh_m, coh_lo, coh_hi = bootstrap_ci_rate(coh, n_boot=n_boot)
        vc_m,  vc_lo,  vc_hi  = bootstrap_ci_rate(vc,  n_boot=n_boot)
        rows.append({
            'model': slug, 'dataset': dataset, 'condition': cond,
            'quadrant': quad, 'n': n,
            'insuff_rate':    round(ins_m, 4),
            'insuff_ci_low':  round(ins_lo, 4),
            'insuff_ci_high': round(ins_hi, 4),
            'numeric_rate':    round(num_m, 4),
            'numeric_ci_low':  round(num_lo, 4),
            'numeric_ci_high': round(num_hi, 4),
            'coherence_rate':    round(coh_m, 4),
            'coherence_ci_low':  round(coh_lo, 4),
            'coherence_ci_high': round(coh_hi, 4),
            'verbalize_acc':    round(vc_m, 4),
            'verbalize_ci_low': round(vc_lo, 4),
            'verbalize_ci_high':round(vc_hi, 4),
            'included':        (n >= n_min),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def record_has_condition(rec, cond_name):
    if rec is None:
        return False
    return rec.get('conditions', {}).get(cond_name) is not None


# ============================================================================
# SAMPLE SHORTAGE WARN
# ============================================================================

def warn_sample_shortage(slug, dataset, q_counts, targets,
                          n_min=N_MIN_PER_QUADRANT,
                          warn_frac=SAMPLE_WARN_FRAC):
    issues = []
    critical = []
    for q, target in zip(['Q1', 'Q2', 'Q3', 'Q4'], targets):
        n = q_counts.get(q, 0)
        if n < n_min:
            critical.append((q, n))
        if target > 0 and n < target * warn_frac:
            issues.append(f'{q}={n}/{target} ({n/target*100:.0f}%)')
        elif target > 0 and n < target:
            issues.append(f'{q}={n}/{target}')
    if issues:
        print(f"  [WARN] {slug}/{dataset}: sample shortage: {', '.join(issues)}")
    if critical:
        crit_str = ', '.join(f'{q}={n}' for q, n in critical)
        print(f"  [WARN] {slug}/{dataset}: {crit_str} below n_min={n_min} — "
              f"excluded from aggregate (included=False)")
    return {'critical': critical, 'shortages': issues}


# ============================================================================
# PATHS
# ============================================================================

def out_paths(output_base, slug, dataset, mode=None):
    pair_dir = Path(output_base) / slug / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    p = {
        'pair_dir':           pair_dir,
        'meta':               pair_dir / 'meta.json',
        'manipulation_check': pair_dir / 'manipulation_check.csv',
        'basis_probe':        pair_dir / 'basis_probe.npy',
        'basis_dim':          pair_dir / 'basis_dim.npy',
    }
    if mode is not None:
        mode_dir = pair_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        p['mode_dir']     = mode_dir
        p['summary']      = mode_dir / 'summary.csv'
        p['sample_major'] = mode_dir / 'sample_major.json'
        p['done']         = mode_dir / 'DONE'
    return p


# ============================================================================
# PER-MODE INTERVENTION LOOP
# ============================================================================

def run_one_mode(model, tok, all_samples, mode, slug, dataset,
                  conditions, paths_mode, args):
    """Run all intervention conditions for one mode. Resumes from
    sample_major.json if present."""
    sample_records = load_existing_records(paths_mode['sample_major'])
    for r in sample_records:
        if 'conditions' not in r:
            r['conditions'] = {}
    by_idx = {r['sample_id']: r for r in sample_records}

    inputs_by_id = {}
    inputs_list = make_inputs(all_samples, mode)
    for s, inp in zip(all_samples, inputs_list):
        inputs_by_id[s['sample_id']] = inp

    batch_size = (args.batch_decision if mode == MODE_DECISION
                   else args.batch_natural)
    max_new = (MAX_NEW_TOKENS_DECISION if mode == MODE_DECISION
                else MAX_NEW_TOKENS_NATURAL)
    decode_fn = (batched_decode_decision if mode == MODE_DECISION
                  else batched_decode_natural)

    desc = f'  {mode}'
    for cond_name, register_fn in tqdm(conditions, desc=desc):
        need = [s for s in all_samples
                if not record_has_condition(by_idx.get(s['sample_id']),
                                             cond_name)]
        if not need:
            continue

        handles = register_fn(model)
        try:
            batch_inputs = [inputs_by_id[s['sample_id']] for s in need]
            gen_results = decode_fn(model, tok, batch_inputs, batch_size,
                                     max_new_tokens=max_new)
            for s, gr in zip(need, gen_results):
                parsed = parse_output(gr, s, mode)
                rec = by_idx.get(s['sample_id'])
                if rec is None:
                    rec = {
                        'sample_id': s['sample_id'],
                        'quadrant': s.get('quadrant', ''),
                        'is_sufficient': s.get('is_sufficient'),
                        'exp2_answer_kind': classify_answer(
                            extract_boxed(s.get('cot', ''))) or 'other',
                        'conditions': {},
                    }
                    sample_records.append(rec)
                    by_idx[s['sample_id']] = rec
                rec['conditions'][cond_name] = parsed
        finally:
            remove_handles(handles)

        with open(paths_mode['sample_major'], 'w') as f:
            json.dump(sample_records, f, indent=2)
        aggregate_summary(sample_records, slug, dataset, paths_mode['summary'],
                          n_boot=args.n_bootstrap)

    aggregate_summary(sample_records, slug, dataset, paths_mode['summary'],
                      n_boot=args.n_bootstrap)
    paths_mode['done'].touch()


# ============================================================================
# PER-(MODEL, DATASET) RUN
# ============================================================================

def run_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    paths = out_paths(args.output_dir, slug, dataset)

    # Per-mode DONE markers
    modes_to_run = []
    for m in args.modes:
        pm = out_paths(args.output_dir, slug, dataset, mode=m)
        if pm['done'].exists() and not args.force:
            print(f"[DONE] {slug}/{dataset}/{m} — skipping")
            continue
        modes_to_run.append(m)
    if not modes_to_run:
        print(f"[DONE] {slug}/{dataset} — all modes complete")
        return

    # ─── exp10 anchor ───────────────────────────────────────────────────
    exp10_csv = os.path.join(EXP10_DIR, 'results',
                              f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[SKIP] {slug}/{dataset}: no exp10 csv"); return
    df10 = pd.read_csv(exp10_csv)
    row10 = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row10.empty:
        print(f"[SKIP] {slug}/{dataset}: no exp10 entry"); return
    best_layer = int(row10['Optimal_Layer'].iloc[0])
    exp10_anchor_f1 = float(row10['Unified_Test_F1'].iloc[0])

    probe_path = os.path.join(EXP10_DIR, 'probes_proportional', dataset, slug,
                               f'unified_probe_layer{best_layer}.joblib')
    if not os.path.exists(probe_path):
        print(f"[SKIP] {slug}/{dataset}: no exp10 probe"); return
    exp10_probe = joblib.load(probe_path)
    v_probe = direction_from_probe(exp10_probe)
    hidden_dim = v_probe.shape[0]

    emb = load_exp10_embeddings(slug, dataset, best_layer)
    if emb is None:
        print(f"[SKIP] {slug}/{dataset}: no exp10 embeddings"); return
    X_tr_emb, y_tr_emb, X_te_emb, y_te_emb = emb

    print(f"[RUN] {slug}/{dataset}: modes={modes_to_run}, "
          f"best_layer={best_layer}, exp10_F1={exp10_anchor_f1:.4f}")

    # ─── Q1-Q4 sampling (shared) ────────────────────────────────────────
    q1_samples, q3_samples = get_q1_q3_samples(slug, dataset, args.n_q1, args.n_q3)
    q2_samples, q4_samples = get_q2_q4_samples(slug, dataset, args.n_q2, args.n_q4)
    all_samples = q1_samples + q2_samples + q3_samples + q4_samples
    q_counts = {'Q1': len(q1_samples), 'Q2': len(q2_samples),
                'Q3': len(q3_samples), 'Q4': len(q4_samples)}
    print(f"  samples: Q1={q_counts['Q1']} Q2={q_counts['Q2']} "
          f"Q3={q_counts['Q3']} Q4={q_counts['Q4']}")
    shortage_info = warn_sample_shortage(
        slug, dataset, q_counts,
        [args.n_q1, args.n_q2, args.n_q3, args.n_q4],
        n_min=N_MIN_PER_QUADRANT,
    )
    if len(all_samples) == 0:
        print(f"  ! no samples — skipping"); return

    # ─── Load model ONCE ────────────────────────────────────────────────
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    num_gpus = torch.cuda.device_count()
    max_mem = {i: '78GiB' for i in range(num_gpus)} if num_gpus else None
    # gemma-3 hybrid attention (sliding window + global) on multi-GPU sdpa
    # causes sticky CUDA context failures. Use eager for gemma family.
    attn_impl = 'eager' if 'gemma' in model_name.lower() else 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', max_memory=max_mem,
        dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    layer_modules = get_layer_modules(model)
    target_module = layer_modules[best_layer]
    n_layers = len(layer_modules)

    # ─── Shared scaffolding: v_DIM ──────────────────────────────────────
    suff_pmpts, insuff_pmpts = load_train_balanced_prompts(slug, dataset, args.n_dim)
    if (not suff_pmpts or not insuff_pmpts
            or len(suff_pmpts) < 50 or len(insuff_pmpts) < 50):
        print(f"  ! insufficient train prompts for v_DIM; falling back to test")
        suff_pmpts = [s['prompt'] for s in q3_samples][:200]
        insuff_pmpts = [s['prompt'] for s in q1_samples][:200]

    print(f"  Computing v_DIM (t=0 diff-of-means)...")
    X_suff_t0 = extract_t0_states(model, tok, suff_pmpts, best_layer)
    X_insuff_t0 = extract_t0_states(model, tok, insuff_pmpts, best_layer)
    v_dim_raw, dim_mag = diff_in_means_direction(X_insuff_t0, X_suff_t0)
    nrm = max(float(np.linalg.norm(v_dim_raw)), 1e-12)
    v_dim = (v_dim_raw / nrm).astype(np.float32)
    cos_probe_dim = float(np.dot(v_probe, v_dim))
    print(f"  cos(v_probe, v_DIM) = {cos_probe_dim:.4f}")

    # ─── Shared: INLP bases ─────────────────────────────────────────────
    K_max_used = min(args.rank_K_max, hidden_dim - 1)
    if paths['basis_probe'].exists() and paths['basis_dim'].exists() and not args.force:
        probe_basis = np.load(paths['basis_probe'])
        dim_basis = np.load(paths['basis_dim'])
        print(f"  Loaded existing INLP bases")
    else:
        print(f"  Building INLP bases (K_max={K_max_used})...")
        probe_basis = build_inlp_basis(v_probe, X_tr_emb, y_tr_emb,
                                        K_max=K_max_used, seed=42)
        dim_basis   = build_inlp_basis(v_dim,   X_tr_emb, y_tr_emb,
                                        K_max=K_max_used, seed=43)
        np.save(paths['basis_probe'], probe_basis)
        np.save(paths['basis_dim'],   dim_basis)
    rand_basis = random_orthonormal_basis(hidden_dim, K_max_used, seed=123)

    probe_means = compute_basis_means(X_tr_emb, probe_basis)
    dim_means   = compute_basis_means(X_tr_emb, dim_basis)
    rand_means  = compute_basis_means(X_tr_emb, rand_basis)

    # ─── Shared: alpha scale ─────────────────────────────────────────────
    norm_pool = (suff_pmpts + insuff_pmpts)[:200]
    res_mean_norm, res_std_norm = compute_layer_residual_norm(
        model, tok, norm_pool, best_layer, batch_size=4)
    alpha_scale = res_mean_norm / np.sqrt(hidden_dim)
    v_rand = rand_basis[0]

    # ─── Shared: manipulation check ─────────────────────────────────────
    if paths['manipulation_check'].exists() and not args.force:
        print(f"  Loaded existing manipulation_check.csv")
    else:
        print("  Manipulation check (probe retrain at each K)...")
        K_list_for_mc = [k for k in args.rank_K_values if k <= K_max_used]
        mc_rows = []
        for basis_full, name, anchor_check in [
            (probe_basis, 'probe', exp10_anchor_f1),
            (dim_basis,   'dim',   None),
            (rand_basis,  'rand',  None),
        ]:
            rows = manipulation_check_curve(
                X_tr_emb, y_tr_emb, X_te_emb, y_te_emb, basis_full,
                K_list_for_mc, exp10_f1=anchor_check,
                label=f'{slug}/{dataset}/{name}')
            for r in rows:
                mc_rows.append({'model': slug, 'dataset': dataset,
                                'direction': name, **r})
        pd.DataFrame(mc_rows).to_csv(paths['manipulation_check'], index=False)

    K_list_for_mc = [k for k in args.rank_K_values if k <= K_max_used]

    # ─── Shared: meta ────────────────────────────────────────────────────
    cos_probe_rand = float(np.dot(v_probe, v_rand))
    cos_dim_rand   = float(np.dot(v_dim,   v_rand))
    meta = {
        'model': slug, 'dataset': dataset, 'version': 'v6',
        'modes_configured': args.modes,
        'modes_actually_run_this_invocation': modes_to_run,
        'designs': {
            MODE_DECISION: 'decision_step_force_decode',
            MODE_NATURAL:  'full_natural_generation_with_hook',
        },
        'best_layer': best_layer, 'final_layer': n_layers,
        'hidden_dim': hidden_dim,
        'exp10_anchor_f1': exp10_anchor_f1,
        'cos_probe_dim': cos_probe_dim,
        'cos_probe_rand': cos_probe_rand,
        'cos_dim_rand': cos_dim_rand,
        'v_dim_raw_magnitude': float(dim_mag),
        'residual_norm_mean_at_best_layer': res_mean_norm,
        'residual_norm_std_at_best_layer': res_std_norm,
        'alpha_scale_raw': float(alpha_scale),
        'alphas_in_residual_units': args.alphas,
        'rank_K_values': K_list_for_mc,
        'K_max_used': K_max_used,
        'ablation_layers': args.ablation_layers,
        'ablation_methods': args.ablation_methods,
        'n_q1': q_counts['Q1'], 'n_q2': q_counts['Q2'],
        'n_q3': q_counts['Q3'], 'n_q4': q_counts['Q4'],
        'n_q1_target': args.n_q1, 'n_q2_target': args.n_q2,
        'n_q3_target': args.n_q3, 'n_q4_target': args.n_q4,
        'n_min_per_quadrant': N_MIN_PER_QUADRANT,
        'sample_shortage_critical': shortage_info['critical'],
        'n_bootstrap': args.n_bootstrap,
        'max_new_tokens': {
            MODE_DECISION: MAX_NEW_TOKENS_DECISION,
            MODE_NATURAL:  MAX_NEW_TOKENS_NATURAL,
        },
        'batch_size': {
            MODE_DECISION: args.batch_decision,
            MODE_NATURAL:  args.batch_natural,
        },
        'include_baseline_check': args.include_baseline_check,
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    # ─── Build conditions (shared) ──────────────────────────────────────
    args.rank_K_values = K_list_for_mc
    args.rank_K_max = K_max_used
    directions_with_bases = [
        ('probe', v_probe, probe_basis, probe_means),
        ('dim',   v_dim,   dim_basis,   dim_means),
        ('rand',  v_rand,  rand_basis,  rand_means),
    ]
    conditions = build_conditions(
        target_module, layer_modules, best_layer,
        directions_with_bases, alpha_scale, args,
    )
    print(f"  Built {len(conditions)} conditions")

    # ─── Run each mode ───────────────────────────────────────────────────
    for mode in modes_to_run:
        print(f"\n  ── Mode: {mode} ──")
        paths_mode = out_paths(args.output_dir, slug, dataset, mode=mode)
        t_mode = time.time()
        run_one_mode(model, tok, all_samples, mode, slug, dataset,
                      conditions, paths_mode, args)
        print(f"  {mode} done in {(time.time()-t_mode)/60:.1f} min "
              f"→ {paths_mode['summary']}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# PLOTS
# ============================================================================

DIR_COLORS = {'probe': 'tab:blue', 'dim': 'tab:orange', 'rand': 'tab:green'}


def aggregate_per_mode(output_base, mode):
    """Concatenate all summary.csv for one mode across (slug, dataset)."""
    base = Path(output_base)
    dfs = []
    for p in base.glob(f'*/*/{mode}/summary.csv'):
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    big = pd.concat(dfs, ignore_index=True)
    return big


def plot_per_mode(output_base, mode, plot_dir):
    plot_dir = Path(plot_dir) / mode
    plot_dir.mkdir(parents=True, exist_ok=True)
    base = Path(output_base)

    big = aggregate_per_mode(output_base, mode)
    if big is None or big.empty:
        print(f'No summary data for mode {mode}'); return

    if 'included' in big.columns:
        big_f = big[big['included'] == True].copy()
    else:
        big_f = big[big['n'] >= N_MIN_PER_QUADRANT].copy()
    dropped = len(big) - len(big_f)
    if dropped:
        print(f'[{mode}] Dropped {dropped} cells with included=False or n<{N_MIN_PER_QUADRANT}')

    big_f.to_csv(base / f'all_pairs_summary_{mode}.csv', index=False)
    n_pairs = big_f[['model', 'dataset']].drop_duplicates().shape[0]
    print(f'[{mode}] Wrote all_pairs_summary_{mode}.csv: {len(big_f)} rows, '
          f'{n_pairs} pairs')

    big_int = big_f[~big_f['condition'].isin(['baseline_exp2', 'baseline_check'])]

    # Ablation box plots
    for quad_prefix, label in [('Q1', 'q1'), ('Q2', 'q2'),
                                ('Q3', 'q3'), ('Q4', 'q4')]:
        sub = big_int[(big_int['quadrant'].str.startswith(quad_prefix))
                      & big_int['condition'].str.startswith('ablate_')].copy()
        if sub.empty:
            continue
        sub['K'] = sub['condition'].str.extract(r'_K(\d+)_').astype(float).iloc[:, 0]
        sub['direction'] = sub['condition'].str.extract(r'ablate_(probe|dim|rand)_')[0]
        sub['layer_mode'] = sub['condition'].str.extract(r'_(all|lstar)_')[0]
        sub['method'] = sub['condition'].str.extract(r'_(zero|mean)$')[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        for ax, (lm, mth) in zip(axes.flat,
                                  [(lm_, mth_) for lm_ in ['all', 'lstar']
                                                for mth_ in ['zero', 'mean']]):
            cell = sub[(sub['layer_mode'] == lm) & (sub['method'] == mth)]
            if cell.empty:
                ax.set_visible(False); continue
            for dir_ in DIRECTIONS:
                d = cell[cell['direction'] == dir_]
                if d.empty:
                    continue
                Ks = sorted(d['K'].unique())
                data = [d[d['K'] == K]['insuff_rate'].values for K in Ks]
                positions = np.arange(len(Ks)) + (DIRECTIONS.index(dir_) - 1) * 0.25
                bp = ax.boxplot(data, positions=positions, widths=0.2,
                                patch_artist=True, labels=[''] * len(Ks))
                for patch in bp['boxes']:
                    patch.set_facecolor(DIR_COLORS[dir_]); patch.set_alpha(0.5)
            if 'Ks' in dir():
                ax.set_xticks(np.arange(len(Ks)))
                ax.set_xticklabels([int(k) for k in Ks])
            ax.set_xlabel('K')
            ax.set_ylabel(f'{label.upper()} insuff_rate')
            ax.set_title(f'[{mode}] Ablate {quad_prefix}: layer={lm}, method={mth}')
            # baseline reference
            base_val = QUADRANT_BASELINE.get(
                {'Q1': 'Q1_Hallucination', 'Q2': 'Q2_Correct_Abstain',
                 'Q3': 'Q3_Correct_Solve', 'Q4': 'Q4_Over_Abstain'}[quad_prefix],
                {}).get('insuff_rate')
            if base_val is not None:
                ax.axhline(base_val, color='black', ls='--', lw=0.8,
                           label=f'baseline_exp2 ({base_val})')
            ax.grid(alpha=0.3)
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=DIR_COLORS[d], alpha=0.5, label=d)
                              for d in DIRECTIONS]
            ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_dir / f'box_ablate_K_{label}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    # Steering box plots
    for quad_prefix, label in [('Q1', 'q1'), ('Q2', 'q2'),
                                ('Q3', 'q3'), ('Q4', 'q4')]:
        sub = big_int[(big_int['quadrant'].str.startswith(quad_prefix))
                      & big_int['condition'].str.startswith('steer_')].copy()
        if sub.empty:
            continue
        sub['direction'] = sub['condition'].str.extract(r'steer_(probe|dim|rand)_')[0]
        sub['alpha_str'] = sub['condition'].str.extract(
            r'steer_(?:probe|dim|rand)_(.+)$')[0]

        def decode_alpha(a):
            s = a[1:]
            sign = -1 if s.startswith('n') else 1
            return sign * float(s[1:].replace('_', '.'))
        sub['alpha'] = sub['alpha_str'].apply(decode_alpha)

        fig, ax = plt.subplots(figsize=(12, 6))
        for dir_ in DIRECTIONS:
            d = sub[sub['direction'] == dir_]
            if d.empty:
                continue
            alphas = sorted(d['alpha'].unique())
            data = [d[d['alpha'] == a]['insuff_rate'].values for a in alphas]
            positions = np.array(alphas) + (DIRECTIONS.index(dir_) - 1) * 0.15
            bp = ax.boxplot(data, positions=positions, widths=0.12,
                            patch_artist=True, labels=[''] * len(alphas))
            for patch in bp['boxes']:
                patch.set_facecolor(DIR_COLORS[dir_]); patch.set_alpha(0.5)
        ax.set_xlabel('alpha (residual-units)')
        ax.set_ylabel(f'{label.upper()} insuff_rate')
        ax.set_title(f'[{mode}] Steering, {quad_prefix}')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        base_val = QUADRANT_BASELINE.get(
            {'Q1': 'Q1_Hallucination', 'Q2': 'Q2_Correct_Abstain',
             'Q3': 'Q3_Correct_Solve', 'Q4': 'Q4_Over_Abstain'}[quad_prefix],
            {}).get('insuff_rate')
        if base_val is not None:
            ax.axhline(base_val, color='black', ls='--', lw=0.8,
                       label=f'baseline_exp2 ({base_val})')
        ax.grid(alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=DIR_COLORS[d], alpha=0.5, label=d)
                            for d in DIRECTIONS],
                  loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_dir / f'box_steer_alpha_{label}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()


def plot_manipulation_check(output_base, plot_dir):
    """manipulation_check is at pair level (shared between modes), so plot once."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    base = Path(output_base)

    dfs = []
    for p in base.glob('*/*/manipulation_check.csv'):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    if not dfs:
        print(f'No manipulation_check.csv under {base}'); return
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(base / 'all_pairs_manipulation.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dir_ in zip(axes, DIRECTIONS):
        sub = big[big['direction'] == dir_]
        if sub.empty:
            ax.set_visible(False); continue
        for (m, d), g in sub.groupby(['model', 'dataset']):
            g = g.sort_values('K')
            ax.plot(g['K'], g['f1'], color='gray', alpha=0.4, lw=0.8)
        mean = sub.groupby('K')['f1'].mean()
        ax.plot(mean.index, mean.values, color='black', lw=2.0,
                label='mean across pairs')
        ax.set_xlabel('K (rank projected out)')
        ax.set_ylabel('retrained probe F1')
        ax.set_title(f'Manipulation check — {dir_}')
        ax.set_xscale('symlog')
        ax.axhline(0.5, color='red', ls='--', lw=0.5, label='chance')
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig(plot_dir / 'manipulation_check_K_curve.png', dpi=150,
                bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--output_dir', default=OUTPUT_BASE)
    p.add_argument('--mode', default='both',
                   choices=['decision', 'natural', 'both'],
                   help='Which intervention mode(s) to run')
    p.add_argument('--n_q1', type=int, default=N_Q1)
    p.add_argument('--n_q2', type=int, default=N_Q2)
    p.add_argument('--n_q3', type=int, default=N_Q3)
    p.add_argument('--n_q4', type=int, default=N_Q4)
    p.add_argument('--n_dim', type=int, default=N_DIM_PER_CLASS)
    p.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    p.add_argument('--rank_K_values', type=int, nargs='+', default=RANK_K_VALUES)
    p.add_argument('--rank_K_max', type=int, default=RANK_K_MAX)
    p.add_argument('--ablation_layers', nargs='+', default=ABLATION_LAYERS,
                   choices=['all', 'lstar_only'])
    p.add_argument('--ablation_methods', nargs='+', default=ABLATION_METHODS,
                   choices=['zero', 'mean'])
    p.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP)
    p.add_argument('--batch_decision', type=int, default=BATCH_DECISION)
    p.add_argument('--batch_natural',  type=int, default=BATCH_NATURAL)
    p.add_argument('--skip_steering', action='store_true')
    p.add_argument('--skip_ablation', action='store_true')
    p.add_argument('--include_baseline_check', action='store_true',
                   help='Adds 1 no-hook condition per mode as protocol sanity')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing DONE markers and rebuild bases')
    p.add_argument('--aggregate_only', action='store_true')
    p.add_argument('--test', action='store_true')
    args = p.parse_args()

    args.modes = ([MODE_DECISION] if args.mode == 'decision'
                   else [MODE_NATURAL] if args.mode == 'natural'
                   else MODES_ALL)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        plot_manipulation_check(args.output_dir,
                                 Path(args.output_dir) / '_plots')
        for m in MODES_ALL:
            plot_per_mode(args.output_dir, m,
                           Path(args.output_dir) / '_plots')
        return

    if args.test:
        args.n_q1 = 5; args.n_q2 = 5; args.n_q3 = 5; args.n_q4 = 5
        args.n_dim = 20
        args.alphas = [1.0, -1.0]
        args.rank_K_values = [1, 2]
        args.rank_K_max = 2
        args.ablation_layers = ['all']
        args.ablation_methods = ['zero']
        args.n_bootstrap = 100

    if args.all_models:
        models = PAPER_MODELS_V6
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("Specify --model or --all_models")

    if args.all_datasets:
        datasets = ['umwp', 'treecut']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Specify --dataset or --all_datasets")

    print(f"V6 unified — modes={args.modes}")
    print(f"Models ({len(models)}): {[m.split('/')[-1] for m in models]}")
    print(f"Datasets: {datasets}")
    print(f"N_min_per_quadrant: {N_MIN_PER_QUADRANT}")
    print(f"Baseline: exp2-deployment definitional (injected into summary.csv)")
    if args.include_baseline_check:
        print(f"  + no-hook 'baseline_check' condition for protocol sanity")

    for m in models:
        for ds in datasets:
            t0 = time.time()
            try:
                run_one(m, ds, args)
                print(f"  [{m}/{ds}] total {(time.time()-t0)/60:.1f} min")
            except Exception as e:
                print(f"\n[ERROR] {m}/{ds}: {e}")
                import traceback; traceback.print_exc()

    plot_manipulation_check(args.output_dir,
                             Path(args.output_dir) / '_plots')
    for m in MODES_ALL:
        plot_per_mode(args.output_dir, m,
                       Path(args.output_dir) / '_plots')


if __name__ == '__main__':
    main()