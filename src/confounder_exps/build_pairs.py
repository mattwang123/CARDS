"""
================================================================================
EXP A: Build Minimal Contrastive Pairs
================================================================================
GOAL: Produce pairs (q+, q-) where:
  q+ = a KNOWN-SOLVABLE question from our test set (is_sufficient=True)
  q- = a MINIMALLY-EDITED version that is unsolvable (1-3 words changed)

WHY SAMPLE FROM OUR DATASET (not generate new questions)?
  - Solvability of q+ is GUARANTEED by the dataset label. No LLM needs to
    judge the original, removing the hardest verification task entirely.
  - Pairs stay in the exact domain/distribution the probe was trained on.
    A passing result is therefore MORE convincing, not less.
  - The probe was trained on the TRAIN split; we sample from TEST only.
    No data leakage.

WHY TWO DIFFERENT MODELS (gen vs verifier)?
  - gen_model (gpt-4.5-preview): powerful enough for clean surgical edits
  - ver_model (gpt-4.1-mini): cheaper, independent; only judges the EDIT
    (not the original -- the dataset label handles that)
  - Different models = genuine independence of judgment

VERIFICATION (simplified vs prior approach):
  Since solvable_original is guaranteed by the label, the verifier checks ONLY:
    (a) Is the edited version genuinely unsolvable?
        (no single reasonable assumption should yield a unique answer)
    (b) Is the edit truly minimal?
        (<=5 words changed, names/scenario/structure preserved)

OUTPUT:
  src/data/processed/minimal_pairs/verified_pairs_{dataset}.json
  Each record: {q_plus, q_minus, removed_value, edit_distance,
                lex_overlap, verifier_judgment, verified}

USAGE:
  python exp_confounder_A_build_pairs.py --dataset both
  python exp_confounder_A_build_pairs.py --dataset umwp \
      --gen_model gpt-4.5-preview --ver_model gpt-4.1-mini --n 250
================================================================================
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ---- CONFIG ----
DATASETS = {
    'umwp':    'src/data/processed/insufficient_dataset_umwp/umwp_test.json',
    'treecut': 'src/data/processed/treecut/treecut_test.json'
}
OUT_DIR = 'src/data/processed/minimal_pairs'
N_CANDIDATES = 250   # ~150 expected to pass both stages
SEED = 42

# Generator: powerful model makes a surgical edit
GENERATION_PROMPT = """\
You are editing a math word problem. The original IS SOLVABLE.
Make a MINIMAL edit so it becomes UNSOLVABLE due to missing information.

STRICT RULES:
1. Change 1-3 words MAXIMUM. Smaller is always better.
2. Replace exactly ONE numerical/quantitative value with a vague placeholder:
   "some", "several", "a number of", "an unknown amount", or "a few".
3. Do NOT change: names, scenario, sentence structure, or the question asked.
4. The result must be GENUINELY unsolvable — no single reasonable assumption
   should yield a unique numerical answer.

Return ONLY this JSON (no markdown):
{{"edited": "<full modified problem>", "removed": "<exact value removed>", "n_words_changed": <int>}}

ORIGINAL:
{question}"""

# Verifier: cheaper model, only judges the EDIT (original solvability is guaranteed)
VERIFICATION_PROMPT = """\
A math problem was minimally edited to make it unsolvable. Judge the edit strictly.
The ORIGINAL is confirmed solvable (do not question this).

ORIGINAL (confirmed solvable):
{original}

EDITED (claimed unsolvable):
{edited}

Check ONLY these two things:
1. unsolvable_edited: Is the EDITED version GENUINELY unsolvable?
   Ask: could a student make ONE reasonable assumption and still arrive at a
   unique numerical answer? If yes -> false (still effectively solvable).
2. minimal_edit: Does the edit change AT MOST 5 words AND preserve all names,
   scenario, and structure? Count changed words carefully.

Return ONLY this JSON (no markdown):
{{"unsolvable_edited": <bool>, "minimal_edit": <bool>, "reason": "<one sentence>"}}"""


# ---- UTILITIES ----

def parse_json(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else None


def call_llm(client, model, prompt, temperature=None):
    """Call LLM with JSON output. temperature=None uses model default (required for GPT-5)."""
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,   # GPT-5 needs ample headroom for reasoning + JSON output
            response_format={"type": "json_object"},
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
        parsed = parse_json(raw)
        if parsed is None:
            return {"_error": f"parse_failed_raw={(raw or '')[:200]}"}
        return parsed
    except Exception as e:
        return {"_error": str(e)}


def lex_overlap(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def word_edit_distance(a, b):
    A, B = a.split(), b.split()
    n, m = len(A), len(B)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] if A[i-1] == B[j-1] \
                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def _save(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


# ---- PIPELINE ----

def build_pairs(client, gen_model, ver_model, dataset, target_verified):
    """
    Goal: end up with `target_verified` verified pairs, each from a DISTINCT
    original question.

    Pipeline:
      1. Load cached records. Count verified.
      2. If we already have >= target_verified, done. (no API calls)
      3. Otherwise, sample fresh originals (never seen before) AND optionally
         retry past API/parse failures (those still have an unused original).
      4. For each: generate minimal edit -> auto-filter -> verify.
      5. Stop as soon as we hit target_verified.

    "Never seen before" means: we do NOT re-attempt originals that are already
    cached as either verified or as a non-API rejection (auto_reject /
    ver_rejected). These are settled and shouldn't be re-rolled.

    Past API/parse failures (gen_failed / ver_failed) ARE re-attempted because
    they were errors, not real rejections of that original.
    """
    with open(DATASETS[dataset]) as f:
        data = json.load(f)

    all_sufficient = [d for d in data if d.get('is_sufficient', True)]

    out_path = Path(OUT_DIR) / f"verified_pairs_{dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.load(open(out_path)) if out_path.exists() else []

    n_verified = sum(1 for r in existing if r.get('verified'))
    if n_verified >= target_verified:
        print(f"[{dataset}] Already have {n_verified} verified pairs (>= target {target_verified}). Done.")
        return existing

    # Settled: keep these, never reprocess
    def is_settled(r):
        if r.get('verified'): return True
        fr = r.get('fail_reason') or ''
        return 'auto_reject' in fr or 'ver_rejected' in fr

    settled = [r for r in existing if is_settled(r)]
    api_failures = [r for r in existing if not is_settled(r)]
    settled_questions = {r['q_plus'] for r in settled}

    # Build the to-process queue:
    #   (a) Past API failures we want to retry — already in dataset, already in cache
    #   (b) Fresh originals — never seen, sampled deterministically with our seed
    retry_queue = [r['q_plus'] for r in api_failures]

    # Stable random sample of fresh originals (deterministic across runs)
    random.seed(SEED)
    shuffled = all_sufficient.copy()
    random.shuffle(shuffled)
    fresh_pool = [d['question'] for d in shuffled if d['question'] not in settled_questions
                                                    and d['question'] not in set(retry_queue)]

    # Reset cache to just settled records; we'll rebuild failures + new attempts
    existing = list(settled)

    # Process retries first (cheap to fix), then fresh originals as needed
    process_queue = retry_queue + fresh_pool
    print(f"[{dataset}] Settled: {len(settled)} ({n_verified} verified). "
          f"Need: {target_verified - n_verified} more. "
          f"Queue: {len(retry_queue)} retries + up to {len(fresh_pool)} fresh.")

    pbar = tqdm(total=target_verified - n_verified, desc=f"{dataset}")
    for q_plus in process_queue:
        if n_verified >= target_verified:
            break

        record = {'q_plus': q_plus, 'q_minus': None, 'verified': False}

        # Stage 1: generate minimal edit
        gen = call_llm(client, gen_model,
                       GENERATION_PROMPT.format(question=q_plus))
        if not gen or '_error' in gen or not gen.get('edited'):
            record['fail_reason'] = f"gen_failed: {(gen or {}).get('_error', 'no_edit')}"
            existing.append(record); _save(out_path, existing); continue

        q_minus = gen['edited']
        ed = word_edit_distance(q_plus, q_minus)
        lo = round(lex_overlap(q_plus, q_minus), 3)
        record.update({'q_minus': q_minus, 'removed_value': gen.get('removed', ''),
                       'edit_distance': ed, 'lex_overlap': lo})

        # Stage 2: auto-filter
        if ed > 6 or lo < 0.80:
            record['fail_reason'] = f"auto_reject: ed={ed}, lex={lo:.2f}"
            existing.append(record); _save(out_path, existing); continue

        # Stage 3: verify
        ver = call_llm(client, ver_model,
                       VERIFICATION_PROMPT.format(original=q_plus, edited=q_minus))
        if not ver or '_error' in ver:
            record['fail_reason'] = f"ver_failed: {(ver or {}).get('_error', 'parse')}"
            existing.append(record); _save(out_path, existing); continue

        record['verifier_judgment'] = ver
        record['verified'] = bool(ver.get('unsolvable_edited') and ver.get('minimal_edit'))
        if not record['verified']:
            record['fail_reason'] = f"ver_rejected: {ver.get('reason', '')}"
        else:
            n_verified += 1
            pbar.update(1)

        existing.append(record)
        _save(out_path, existing)

    pbar.close()
    final_verified = sum(1 for r in existing if r.get('verified'))
    print(f"[{dataset}] Done. Verified: {final_verified}/{target_verified} target "
          f"(total records in file: {len(existing)})")
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['umwp', 'treecut', 'both'])
    parser.add_argument('--gen_model', default='gpt-5-2025-08-07',
                        help="Powerful model for generating minimal edits")
    parser.add_argument('--ver_model', default='gpt-4.1-2025-04-14',
                        help="Cheaper model for verifying the edit")
    parser.add_argument('--n', type=int, default=N_CANDIDATES,
                        help="Target number of VERIFIED pairs (each from a distinct original)")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    targets = ['umwp', 'treecut'] if args.dataset == 'both' else [args.dataset]
    for ds in targets:
        build_pairs(client, args.gen_model, args.ver_model, ds, args.n)


if __name__ == '__main__':
    main()