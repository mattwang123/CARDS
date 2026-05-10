"""
================================================================================
EXPERIMENT 14C: Early Cutoff Analysis
================================================================================

Aggregates early-cutoff force-decode evaluations into behavioral shift tables,
compares them against Exp 11 probe trajectories, and estimates generated-token
savings from early intervention.
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
from exp14_early_cutoff_generate import (  # noqa: E402
    CUTOFF_PERCENTAGES,
    DATASETS,
    EXP14_OUTPUT_BASE,
    EXPORT_BASE,
    FULL_MODELS,
    REPRESENTATIVE_MODELS,
    cutoff_label,
    model_slug,
)


VALID_QUADRANTS = ['Q1_Hallucination', 'Q2_Correct_Rejection', 'Q3_Solved_Correctly', 'Q4_Competence_Failure']
QUADRANT_COLORS = {
    'Q1_Hallucination': '#d62728',
    'Q2_Correct_Rejection': '#1f77b4',
    'Q3_Solved_Correctly': '#2ca02c',
    'Q4_Competence_Failure': '#ff7f0e',
}

BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
DEFAULT_EVAL_DIR = os.path.join(EXP14_OUTPUT_BASE, 'experiments/early_cutoff_evaluation')
DEFAULT_RESULTS_DIR = os.path.join(EXP14_OUTPUT_BASE, 'results')
DEFAULT_EXP11_RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DEFAULT_PLOT_DIR = os.path.join(EXP14_OUTPUT_BASE, 'paper_plots/exp14_cutoff')


plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})


def select_models(args) -> list[str]:
    if args.model:
        return [args.model]
    return FULL_MODELS if args.all_models else REPRESENTATIVE_MODELS


def eval_path(eval_dir: Path, slug: str, dataset: str, cutoff_pct: float) -> Path:
    return eval_dir / slug / f'{dataset}_cutoff{cutoff_label(cutoff_pct)}_evaluated.json'


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def load_exp11_averages(results_dir: Path) -> pd.DataFrame:
    path = results_dir / 'exp11_average_trajectories.csv'
    if not path.exists():
        print(f"  ! Missing Exp 11 averages at {path}; probe columns will be NaN.")
        return pd.DataFrame()
    return pd.read_csv(path)


def probe_average(exp11_df: pd.DataFrame, dataset: str, slug: str, quadrant: str, pct: int) -> float:
    if exp11_df.empty:
        return np.nan
    col = f'Prob_{pct}%'
    if col not in exp11_df.columns:
        return np.nan
    rows = exp11_df[
        (exp11_df['Dataset'] == dataset)
        & (exp11_df['Model'] == slug)
        & (exp11_df['Quadrant'] == quadrant)
    ]
    if rows.empty:
        return np.nan
    return float(rows[col].iloc[0])


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def summarize_evaluation(payload: dict, dataset: str, slug: str, cutoff_pct: float, exp11_df: pd.DataFrame) -> dict:
    data = payload.get("data", [])
    valid = [row for row in data if row.get("epistemic_quadrant") in VALID_QUADRANTS]
    counts = {quad: 0 for quad in VALID_QUADRANTS}
    for row in valid:
        counts[row["epistemic_quadrant"]] += 1

    total_valid = len(valid)
    insuff_total = counts['Q1_Hallucination'] + counts['Q2_Correct_Rejection']
    suff_total = counts['Q3_Solved_Correctly'] + counts['Q4_Competence_Failure']

    token_rows = [
        row for row in valid
        if row.get("cutoff_token_count") is not None and row.get("total_cot_tokens")
    ]
    avg_cutoff_tokens = np.mean([row["cutoff_token_count"] for row in token_rows]) if token_rows else np.nan
    avg_total_tokens = np.mean([row["total_cot_tokens"] for row in token_rows]) if token_rows else np.nan
    avg_actual_boundary_pct = np.mean([row.get("actual_boundary_pct", np.nan) for row in token_rows]) if token_rows else np.nan
    token_savings_rate = 1.0 - (avg_cutoff_tokens / avg_total_tokens) if avg_total_tokens and not np.isnan(avg_total_tokens) else np.nan

    # Boundary drift: how far the sentence-boundary snap moved us from the requested %.
    # Computed over ALL rows (including skipped) to characterise the algorithm, not
    # the evaluation outcome. Positive = overshot, negative = undershot.
    drift_values = [row["boundary_drift_pct"] for row in data if "boundary_drift_pct" in row]
    avg_boundary_drift = float(np.mean(drift_values)) if drift_values else np.nan
    avg_abs_boundary_drift = float(np.mean([abs(v) for v in drift_values])) if drift_values else np.nan

    pct_int = cutoff_label(cutoff_pct)
    return {
        "Dataset": dataset,
        "Model": slug,
        "Cutoff_Pct": pct_int,
        "N_Valid": total_valid,
        "N_Skipped": int(payload.get("quadrant_counts", {}).get("skipped", 0)),
        "Q1_Count": counts['Q1_Hallucination'],
        "Q2_Count": counts['Q2_Correct_Rejection'],
        "Q3_Count": counts['Q3_Solved_Correctly'],
        "Q4_Count": counts['Q4_Competence_Failure'],
        "Q1_Rate_All": safe_rate(counts['Q1_Hallucination'], total_valid),
        "Q2_Rate_All": safe_rate(counts['Q2_Correct_Rejection'], total_valid),
        "Q3_Rate_All": safe_rate(counts['Q3_Solved_Correctly'], total_valid),
        "Q4_Rate_All": safe_rate(counts['Q4_Competence_Failure'], total_valid),
        "Q1_Rate_Insufficient": safe_rate(counts['Q1_Hallucination'], insuff_total),
        "Q2_Rate_Insufficient": safe_rate(counts['Q2_Correct_Rejection'], insuff_total),
        "Q3_Rate_Sufficient": safe_rate(counts['Q3_Solved_Correctly'], suff_total),
        "Q4_Rate_Sufficient": safe_rate(counts['Q4_Competence_Failure'], suff_total),
        "Avg_Cutoff_Tokens": float(avg_cutoff_tokens) if not np.isnan(avg_cutoff_tokens) else np.nan,
        "Avg_Full_CoT_Tokens": float(avg_total_tokens) if not np.isnan(avg_total_tokens) else np.nan,
        "Avg_Actual_Boundary_Pct": float(avg_actual_boundary_pct) if not np.isnan(avg_actual_boundary_pct) else np.nan,
        "Token_Savings_Rate": float(token_savings_rate) if not np.isnan(token_savings_rate) else np.nan,
        # Boundary drift: signed mean and absolute mean of (actual_pct - requested_pct).
        # Computed over all samples (including skipped) to measure algorithm bias/spread.
        # Positive = boundary snapped later than requested; negative = earlier.
        "Avg_Boundary_Drift_Pct": avg_boundary_drift,
        "Avg_Abs_Boundary_Drift_Pct": avg_abs_boundary_drift,
        "Probe_PInsuff_Q1": probe_average(exp11_df, dataset, slug, 'Q1_Hallucination', pct_int),
        "Probe_PInsuff_Q2": probe_average(exp11_df, dataset, slug, 'Q2_Correct_Rejection', pct_int),
    }


def plot_behavioral_shift(df: pd.DataFrame, plot_dir: Path, dataset: str, slug: str) -> None:
    model_df = df[(df['Dataset'] == dataset) & (df['Model'] == slug)].sort_values('Cutoff_Pct')
    if model_df.empty:
        return

    x = np.arange(len(model_df))
    bottoms = np.zeros(len(model_df))
    fig, ax = plt.subplots(figsize=(7, 4))
    for quad in VALID_QUADRANTS:
        short = quad.split('_', 1)[0]
        values = model_df[f'{short}_Rate_All'].fillna(0.0).to_numpy()
        ax.bar(x, values, bottom=bottoms, label=short, color=QUADRANT_COLORS[quad])
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}%" for p in model_df['Cutoff_Pct']])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forced Decode Cutoff")
    ax.set_ylabel("Quadrant Proportion")
    ax.set_title(f"{dataset.upper()} Early Cutoff Behavior: {slug}")
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / f'behavioral_shift_{dataset}_{slug}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_probe_behavior_scatter(df: pd.DataFrame, plot_dir: Path, dataset: str) -> None:
    ds_df = df[df['Dataset'] == dataset].copy()
    ds_df = ds_df.dropna(subset=['Probe_PInsuff_Q2', 'Q2_Rate_Insufficient'])
    if ds_df.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for cutoff_pct, cutoff_df in ds_df.groupby('Cutoff_Pct'):
        ax.scatter(
            cutoff_df['Probe_PInsuff_Q2'],
            cutoff_df['Q2_Rate_Insufficient'],
            label=f"{int(cutoff_pct)}%",
            s=45,
            alpha=0.85,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Exp 11 Mean P(insufficient), Q2 Trajectory")
    ax.set_ylabel("Forced Decode Q2 Rate")
    ax.set_title(f"{dataset.upper()} Probe-Behavior Alignment")
    ax.legend(title="Cutoff", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / f'probe_behavior_scatter_{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)


def build_analysis(args) -> None:
    eval_dir = Path(args.eval_dir)
    results_dir = Path(args.results_dir)
    exp11_results_dir = Path(args.exp11_results_dir)
    plot_dir = Path(args.plot_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp11_df = load_exp11_averages(exp11_results_dir)
    datasets = args.datasets or DATASETS
    cutoffs = sorted({round(float(value), 2) for value in args.cutoffs})

    rows = []
    for model_name in select_models(args):
        slug = model_slug(model_name)
        for dataset in datasets:
            for cutoff_pct in cutoffs:
                path = eval_path(eval_dir, slug, dataset, cutoff_pct)
                if not path.exists():
                    print(f"  ! Missing evaluated cutoff file: {path}")
                    continue
                rows.append(summarize_evaluation(load_json(path), dataset, slug, cutoff_pct, exp11_df))

    if not rows:
        print("No evaluated cutoff files found; nothing to analyze.")
        return

    df = pd.DataFrame(rows).sort_values(["Dataset", "Model", "Cutoff_Pct"])
    for dataset in datasets:
        out_csv = results_dir / f'exp14_cutoff_analysis_{dataset}.csv'
        df[df['Dataset'] == dataset].to_csv(out_csv, index=False)
        print(f"  -> Wrote {out_csv}")

        for slug in df[df['Dataset'] == dataset]['Model'].unique():
            plot_behavioral_shift(df, plot_dir, dataset, slug)
        plot_probe_behavior_scatter(df, plot_dir, dataset)

    combined_csv = results_dir / 'exp14_cutoff_analysis_all.csv'
    df.to_csv(combined_csv, index=False)
    print(f"  -> Wrote {combined_csv}")
    print(f"  -> Plots saved under {plot_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default=DEFAULT_EVAL_DIR)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--exp11_results_dir', type=str, default=DEFAULT_EXP11_RESULTS_DIR)
    parser.add_argument('--plot_dir', type=str, default=DEFAULT_PLOT_DIR)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--cutoffs', nargs='+', type=float, default=CUTOFF_PERCENTAGES)
    parser.add_argument('--model', type=str, help='Analyze a single HuggingFace model name.')
    parser.add_argument('--all_models', action='store_true', help='Analyze all 21 Exp 10/11 models.')
    args = parser.parse_args()
    build_analysis(args)


if __name__ == '__main__':
    main()
