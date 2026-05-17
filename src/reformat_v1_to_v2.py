"""
================================================================================
Post-hoc Reformatter: v1 -> v2 sample-major format
================================================================================
Converts v1 results (condition-major) to v2 sample-major format and re-applies
strict \\boxed{Insufficient} flip detection. Useful for inspecting the existing
Qwen2.5-Math-1.5B-Instruct/umwp results without rerunning the experiment.

USAGE:
  python reformat_v1_to_v2.py \\
      --v1_results experiments/steering_full/Qwen2.5-Math-1.5B-Instruct/umwp_results.json \\
      --out_dir   experiments/steering_full_v2/Qwen2.5-Math-1.5B-Instruct \\
      --dataset   umwp
================================================================================
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd


BOXED_INSUFF_REGEX = re.compile(
    r'\\boxed\s*\{\s*('
    r'insufficient|not enough(?: information)?|cannot be determined|'
    r'undetermined|missing(?: information)?|unknown|unsolvable|'
    r'no unique answer|cannot determine|unable to determine'
    r')\s*\}',
    re.IGNORECASE
)


def strict_boxed_insufficient(text):
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


def reformat(v1_path, out_dir, dataset):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(v1_path) as f:
        v1 = json.load(f)

    # Build sample-major dict: prompt -> {quadrant, variants}
    by_prompt = {}
    for entry in v1:
        cond = entry['condition']
        alpha = entry['alpha']
        quadrant = entry['quadrant']
        if cond == 'baseline':
            key = 'baseline'
        elif cond == 'diff_in_means_ablate':
            key = 'diff_in_means_ablate'
        else:
            key = f"{cond}_a{alpha}"

        for record in entry['records']:
            p = record['prompt']
            if p not in by_prompt:
                by_prompt[p] = {'prompt': p, 'quadrant': quadrant, 'variants': {}}
            text = record['generation']
            by_prompt[p]['variants'][key] = {
                'text': text,
                'boxed_insufficient': strict_boxed_insufficient(text),
                'coherent': is_coherent(text),
                'tail_200': text[-200:],
            }

    sample_records = list(by_prompt.values())
    results_path = out / f"{dataset}_results_sample_major.json"
    with open(results_path, 'w') as f:
        json.dump(sample_records, f, indent=2)
    print(f"Wrote {len(sample_records)} sample records to {results_path}")

    # Build summary with STRICT detection
    summary_rows = []
    all_cond_keys = set()
    for r in sample_records:
        all_cond_keys.update(r['variants'].keys())

    for cond_key in sorted(all_cond_keys):
        if cond_key == 'baseline':
            alpha, base = 0.0, 'baseline'
        elif cond_key == 'diff_in_means_ablate':
            alpha, base = float('nan'), 'diff_in_means_ablate'
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
                'condition_base': base, 'alpha': alpha, 'quadrant': quad,
                'n': n, 'flip_rate': flip, 'coherence_rate': coh,
                'useful_flip_rate': useful,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out / f"{dataset}_summary_strict.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote strict summary to {summary_path}")

    print("\n=== Q1 useful_flip_rate (strict) ===")
    q1 = summary_df[summary_df['quadrant'] == 'Q1']
    print(q1.pivot(index='condition_base', columns='alpha', values='useful_flip_rate').round(3).to_string())

    print("\n=== Q3 false-abstention rate ===")
    q3 = summary_df[summary_df['quadrant'] == 'Q3']
    print(q3.pivot(index='condition_base', columns='alpha', values='flip_rate').round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1_results', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    reformat(args.v1_results, args.out_dir, args.dataset)


if __name__ == '__main__':
    main()