"""
================================================================================
EXPERIMENT 15C: Wait Extension Analysis
================================================================================

Aggregates evaluated wait-extension outputs into:
  A. Behavioral shift tables  (Q rates vs n_waits per model)
  B. Q1→Q2 flip rates         (headline: can wait tokens fix hallucinations?)
  C. Model-family breakdown   (thinking models vs standard)
  D. Probe-EOS vs flip rate   (cross-reference with Exp 11 P(insufficient) at EOS)
================================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp15_wait_extension_generate import (  # noqa: E402
    DATASETS,
    EXP15_OUTPUT_BASE,
    SOURCE_EXPORT_BASE,
    WAIT_COUNTS,
    get_model_family,
    model_slug,
    select_models,
)

VALID_QUADRANTS = [
    'Q1_Hallucination', 'Q2_Correct_Rejection',
    'Q3_Solved_Correctly', 'Q4_Competence_Failure',
]
QUADRANT_COLORS = {
    'Q1_Hallucination':      '#d62728',
    'Q2_Correct_Rejection':  '#1f77b4',
    'Q3_Solved_Correctly':   '#2ca02c',
    'Q4_Competence_Failure': '#ff7f0e',
}
FAMILY_COLORS = {
    'think_olmo':     '#9467bd',
    'think_deepseek': '#8c564b',
    'standard':       '#7f7f7f',
}

DEFAULT_EVAL_DIR     = os.path.join(EXP15_OUTPUT_BASE, 'experiments/wait_extension_evaluation')
DEFAULT_EXP2_EVAL_DIR = os.path.join(SOURCE_EXPORT_BASE, 'experiments/dynamic_tracking_test_evaluation')
DEFAULT_RESULTS_DIR  = os.path.join(EXP15_OUTPUT_BASE, 'results')
DEFAULT_EXP11_DIR    = os.path.join(SOURCE_EXPORT_BASE, 'exp_temporal_new/results')
DEFAULT_PLOT_DIR     = os.path.join(EXP15_OUTPUT_BASE, 'paper_plots/exp15_wait')

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 300,
})


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def safe_rate(n: int, d: int) -> float:
    return float(n / d) if d else np.nan


def eval_path(eval_dir: Path, slug: str, dataset: str, n_waits: int) -> Path:
    return eval_dir / slug / f'{dataset}_wait{n_waits}_evaluated.json'


def exp2_eval_path(exp2_dir: Path, slug: str, dataset: str) -> Path:
    return exp2_dir / 'math' / slug / f'{dataset}_evaluated_traces.json'


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_exp2_quadrants(exp2_dir: Path, slug: str, dataset: str) -> dict[str, str]:
    """Return {question_idx -> original_epistemic_quadrant} from Exp 2 evaluation."""
    path = exp2_eval_path(exp2_dir, slug, dataset)
    if not path.exists():
        return {}
    payload = load_json(path)
    data = payload if isinstance(payload, list) else payload.get('data', [])
    return {str(item.get('question_idx', i)): item.get('epistemic_quadrant', '')
            for i, item in enumerate(data)}


def load_exp11_eos_probs(exp11_dir: Path, dataset: str, slug: str) -> dict[str, float]:
    """Return {sample_id -> Prob_100%} from Exp 11 sample-wise CSVs."""
    path = Path(exp11_dir) / 'exp11_sample_wise' / f'traj_{dataset}_{slug}.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    col = 'Prob_100%'
    if col not in df.columns or 'Sample_ID' not in df.columns:
        return {}
    return dict(zip(df['Sample_ID'].astype(str), df[col].astype(float)))


# ---------------------------------------------------------------------------
# Analysis A — per-(model, dataset, n_waits) aggregation
# ---------------------------------------------------------------------------
def summarize_one(payload: dict, slug: str, dataset: str, n_waits: int,
                  exp2_quads: dict[str, str]) -> dict:
    data = payload.get('data', [])
    valid = [r for r in data if r.get('epistemic_quadrant') in VALID_QUADRANTS]
    counts = {q: 0 for q in VALID_QUADRANTS}
    for r in valid:
        counts[r['epistemic_quadrant']] += 1

    total  = len(valid)
    insuff = counts['Q1_Hallucination'] + counts['Q2_Correct_Rejection']
    suff   = counts['Q3_Solved_Correctly'] + counts['Q4_Competence_Failure']

    # Q1→Q2 flip: among samples originally Q1 in Exp 2, how many are now Q2?
    orig_q1_now_q2 = orig_q1_total = 0
    for r in valid:
        q_idx = str(r.get('question_idx', ''))
        orig_q = exp2_quads.get(q_idx, '')
        if orig_q == 'Q1_Hallucination':
            orig_q1_total += 1
            if r.get('epistemic_quadrant') == 'Q2_Correct_Rejection':
                orig_q1_now_q2 += 1

    # Token stats
    strip_pcts = [r['strip_pct'] for r in valid if r.get('strip_pct') is not None]
    natural_count = sum(1 for r in valid if r.get('answer_status') == 'natural')

    return {
        'Dataset':          dataset,
        'Model':            slug,
        'Model_Family':     get_model_family(slug),  # slug may not contain org prefix
        'N_Waits':          n_waits,
        'N_Valid':          total,
        'N_Skipped':        int(payload.get('quadrant_counts', {}).get('skipped', 0)),
        'Q1_Count':         counts['Q1_Hallucination'],
        'Q2_Count':         counts['Q2_Correct_Rejection'],
        'Q3_Count':         counts['Q3_Solved_Correctly'],
        'Q4_Count':         counts['Q4_Competence_Failure'],
        'Q1_Rate_All':      safe_rate(counts['Q1_Hallucination'],     total),
        'Q2_Rate_All':      safe_rate(counts['Q2_Correct_Rejection'], total),
        'Q3_Rate_All':      safe_rate(counts['Q3_Solved_Correctly'],  total),
        'Q4_Rate_All':      safe_rate(counts['Q4_Competence_Failure'],total),
        'Q1_Rate_Insuff':   safe_rate(counts['Q1_Hallucination'],     insuff),
        'Q2_Rate_Insuff':   safe_rate(counts['Q2_Correct_Rejection'], insuff),
        'Q3_Rate_Suff':     safe_rate(counts['Q3_Solved_Correctly'],  suff),
        'Q4_Rate_Suff':     safe_rate(counts['Q4_Competence_Failure'],suff),
        'Q1_to_Q2_Flip_Rate': safe_rate(orig_q1_now_q2, orig_q1_total),
        'Q1_to_Q2_Flipped':   orig_q1_now_q2,
        'Q1_Original_Count':  orig_q1_total,
        'Natural_Answer_Rate': safe_rate(natural_count, total),
        'Avg_Strip_Pct':   float(np.mean(strip_pcts)) if strip_pcts else np.nan,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_behavioral_shift(df: pd.DataFrame, plot_dir: Path,
                           dataset: str, slug: str) -> None:
    sub = df[(df['Dataset'] == dataset) & (df['Model'] == slug)].sort_values('N_Waits')
    if sub.empty:
        return
    x = np.arange(len(sub))
    bottoms = np.zeros(len(sub))
    fig, ax = plt.subplots(figsize=(7, 4))
    for quad in VALID_QUADRANTS:
        short = quad.split('_', 1)[0]
        vals = sub[f'{short}_Rate_All'].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottoms, label=short, color=QUADRANT_COLORS[quad])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"wait={int(n)}" for n in sub['N_Waits']])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Wait Count')
    ax.set_ylabel('Quadrant Proportion')
    ax.set_title(f'{dataset.upper()} Wait Extension: {slug}')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / f'behavioral_shift_{dataset}_{slug}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_flip_rates_by_family(df: pd.DataFrame, plot_dir: Path, dataset: str) -> None:
    sub = df[df['Dataset'] == dataset].dropna(subset=['Q1_to_Q2_Flip_Rate'])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for family, fdf in sub.groupby('Model_Family'):
        for model, mdf in fdf.groupby('Model'):
            mdf_sorted = mdf.sort_values('N_Waits')
            color = FAMILY_COLORS.get(family, '#333333')
            ax.plot(mdf_sorted['N_Waits'], mdf_sorted['Q1_to_Q2_Flip_Rate'],
                    marker='o', color=color, alpha=0.6, linewidth=1.2)
    # Legend proxy
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, label=f, linewidth=2)
               for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, frameon=False, title='Family')
    ax.set_xlabel('N Wait Tokens')
    ax.set_ylabel('Q1→Q2 Flip Rate')
    ax.set_title(f'{dataset.upper()} Q1→Q2 Flip Rate vs Wait Count')
    fig.tight_layout()
    fig.savefig(plot_dir / f'flip_rate_by_family_{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_probe_eos_vs_flip(df: pd.DataFrame, eval_dir: Path, exp11_dir: Path,
                            exp2_eval_dir: Path, plot_dir: Path, dataset: str) -> None:
    """Analysis D: per-sample Prob_100% (Exp 11) vs whether the sample flipped Q1→Q2."""
    rows = []
    for slug in df[(df['Dataset'] == dataset)]['Model'].unique():
        eos_probs = load_exp11_eos_probs(exp11_dir, dataset, slug)
        exp2_quads = load_exp2_quadrants(exp2_eval_dir, slug, dataset)
        if not eos_probs or not exp2_quads:
            continue
        # Use wait=1 as the base condition for this scatter.
        wait1_path = eval_dir / slug / f'{dataset}_wait1_evaluated.json'
        if not wait1_path.exists():
            continue
        payload = load_json(wait1_path)
        for item in payload.get('data', []):
            q_idx = str(item.get('question_idx', ''))
            if exp2_quads.get(q_idx) != 'Q1_Hallucination':
                continue
            sid = str(item.get('sample_id', q_idx))
            p = eos_probs.get(sid, eos_probs.get(q_idx))
            if p is None:
                continue
            flipped = 1 if item.get('epistemic_quadrant') == 'Q2_Correct_Rejection' else 0
            rows.append({'slug': slug, 'p_insuff_eos': p, 'flipped': flipped,
                         'family': get_model_family(slug)})

    if not rows:
        return
    sdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for family, fdf in sdf.groupby('family'):
        # Bin P(insufficient) into deciles and show flip rate per bin.
        fdf = fdf.copy()
        fdf['bin'] = pd.cut(fdf['p_insuff_eos'], bins=10, labels=False)
        binned = fdf.groupby('bin').agg(flip_rate=('flipped', 'mean'),
                                         p_mid=('p_insuff_eos', 'mean')).dropna()
        ax.plot(binned['p_mid'], binned['flip_rate'], marker='o',
                color=FAMILY_COLORS.get(family, '#333333'), label=family, linewidth=1.5)
    ax.set_xlabel('Exp 11 P(insufficient) at EOS (original trace)')
    ax.set_ylabel('Q1→Q2 Flip Rate (wait=1)')
    ax.set_title(f'{dataset.upper()} Probe-EOS vs Wait Flip Rate')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / f'probe_eos_vs_flip_{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_analysis(args) -> None:
    eval_dir      = Path(args.eval_dir)
    exp2_eval_dir = Path(args.exp2_eval_dir)
    exp11_dir     = Path(args.exp11_results_dir)
    results_dir   = Path(args.results_dir)
    plot_dir      = Path(args.plot_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    datasets    = args.datasets or DATASETS
    wait_counts = sorted(set(int(n) for n in args.wait_counts))

    rows = []
    for model_name in select_models(args):
        slug = model_slug(model_name)
        for dataset in datasets:
            exp2_quads = load_exp2_quadrants(exp2_eval_dir, slug, dataset)
            for n_waits in wait_counts:
                path = eval_dir / slug / f'{dataset}_wait{n_waits}_evaluated.json'
                if not path.exists():
                    print(f"  ! Missing: {path}")
                    continue
                rows.append(summarize_one(load_json(path), slug, dataset, n_waits, exp2_quads))

    if not rows:
        print("No evaluated files found; nothing to analyse.")
        return

    df = pd.DataFrame(rows).sort_values(['Dataset', 'Model', 'N_Waits'])

    for dataset in datasets:
        out_csv = results_dir / f'exp15_wait_analysis_{dataset}.csv'
        df[df['Dataset'] == dataset].to_csv(out_csv, index=False)
        print(f"  -> {out_csv}")

        for slug in df[df['Dataset'] == dataset]['Model'].unique():
            plot_behavioral_shift(df, plot_dir, dataset, slug)

        plot_flip_rates_by_family(df, plot_dir, dataset)
        plot_probe_eos_vs_flip(df, eval_dir, exp11_dir, exp2_eval_dir, plot_dir, dataset)

    # Global flip-rate table (headline numbers).
    flip_df = df[['Dataset', 'Model', 'Model_Family', 'N_Waits',
                  'Q1_to_Q2_Flip_Rate', 'Q1_to_Q2_Flipped', 'Q1_Original_Count']].copy()
    flip_csv = results_dir / 'exp15_q1_flip_rates.csv'
    flip_df.to_csv(flip_csv, index=False)
    print(f"  -> {flip_csv}")

    combined = results_dir / 'exp15_wait_analysis_all.csv'
    df.to_csv(combined, index=False)
    print(f"  -> {combined}")
    print(f"  -> Plots saved under {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir',          type=str, default=DEFAULT_EVAL_DIR)
    parser.add_argument('--exp2_eval_dir',     type=str, default=DEFAULT_EXP2_EVAL_DIR)
    parser.add_argument('--results_dir',       type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--exp11_results_dir', type=str, default=DEFAULT_EXP11_DIR)
    parser.add_argument('--plot_dir',          type=str, default=DEFAULT_PLOT_DIR)
    parser.add_argument('--datasets',    nargs='+', choices=DATASETS)
    parser.add_argument('--wait_counts', nargs='+', type=int, default=WAIT_COUNTS)
    parser.add_argument('--model',       type=str)
    parser.add_argument('--all_models',  action='store_true')
    args = parser.parse_args()
    build_analysis(args)


if __name__ == '__main__':
    main()
