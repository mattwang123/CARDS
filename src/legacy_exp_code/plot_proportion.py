"""
================================================================================
EXPERIMENT 7: Relative Epistemic Awakening (Scatter Plot)
================================================================================
Maps fixed-timestep embeddings (t=0, 2, 4...) to relative generation proportions 
(t / total_length) and plots the Unified Probe's predicted probability over time.
================================================================================
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import joblib
from transformers import AutoTokenizer
from tqdm import tqdm

# ANSI colors for terminal debug readability
COLOR_RESET = "\033[0m"
COLOR_HEADER = "\033[95m"
COLOR_META = "\033[93m"
COLOR_COT = "\033[96m"

# =============================================================================
# NATURE-STYLE PLOT CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,   
    'axes.spines.right': False, 
    'figure.dpi': 300
})

# --- CONFIGURATION ---
MODELS = [
    # --- SMALL/MEDIUM SCALE (~1.5B - 4B) ---
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',

    # --- MEDIUM/LARGE SCALE (~7B - 9B) ---
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think',
    'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',

    # --- LARGE SCALE (14B - 32B) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it',
    'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',

    # --- MASSIVE SCALE (70B+) ---
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]
DATASET = 'umwp'
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
TIMESTEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256]
# Trailing rolling mean over scatter points sorted by x (most recent `window` dots).
ROLLING_WINDOW = 100
# Common x-axis for cross-model aggregation of normalized (proportion) binned curves.
X_AGG_GRID = np.linspace(0.0, 1.0, 129)
AGGREGATE_QUADRANTS = ('Q1_Hallucination', 'Q2_Correct_Rejection')
CACHE_FORMAT_VERSION = 2
DEFAULT_CACHE_DIR = os.path.join('paper_plots', 'exp7_awakening', '.plot_proportion_cache')
# Grand aggregate: drop this many lowest / highest model values per x before averaging.
GRAND_AGG_TRIM_LOW = 2
GRAND_AGG_TRIM_HIGH = 2


def trimmed_mean_across_models(Y, trim_low=GRAND_AGG_TRIM_LOW, trim_high=GRAND_AGG_TRIM_HIGH):
    """
    Y: (n_models, n_x). For each x index, sort values across models, drop the `trim_low`
    smallest and `trim_high` largest, then mean the rest. If too few non-NaN values,
    falls back to the mean of all available values in that column.
    """
    n_models, n_x = Y.shape
    out = np.empty(n_x, dtype=float)
    for j in range(n_x):
        col = Y[:, j]
        col = col[~np.isnan(col)]
        m = len(col)
        if m == 0:
            out[j] = np.nan
            continue
        if m <= trim_low + trim_high:
            out[j] = float(np.mean(col))
            continue
        s = np.sort(col)
        out[j] = float(np.mean(s[trim_low : m - trim_high]))
    return out


def save_cot_length_histogram(trace_json_path, model_name, out_path, title):
    """Save a CoT-length histogram with mean line and +/-1 std bar."""
    with open(trace_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Trace JSON must be a non-empty list: {trace_json_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lengths = np.array(
        [
            len(tokenizer.encode(item.get("generated_response", ""), add_special_tokens=False))
            for item in data
        ],
        dtype=np.int64,
    )

    mean = float(np.mean(lengths))
    std = float(np.std(lengths))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts, _, _ = ax.hist(
        lengths,
        bins=40,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvline(mean, color="#D62728", linestyle="--", linewidth=2.0, label=f"Mean = {mean:.1f}")
    y_bar = (counts.max() * 0.92) if len(counts) > 0 else 1.0
    x_left = max(0.0, mean - std)
    x_right = mean + std
    ax.hlines(y=y_bar, xmin=x_left, xmax=x_right, color="#2CA02C", linewidth=4, label=f"±1 SD ({std:.1f})")
    ax.plot([x_left, x_right], [y_bar, y_bar], "|", color="#2CA02C", markersize=14, markeredgewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Generated CoT Length (tokens)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def rolling_trailing_mean_sorted(df, x_col, y_col='Probability', window=ROLLING_WINDOW):
    """Sort by x_col, return (x, y_roll) with trailing rolling mean of y (min_periods ~ window/5)."""
    d = df.sort_values(x_col, kind='mergesort').copy()
    min_periods = max(1, min(window // 5, window))
    d['_roll'] = d[y_col].rolling(window=window, min_periods=min_periods).mean()
    return d[x_col].to_numpy(), d['_roll'].to_numpy()


QUADRANT_STYLES = {
    'Q1_Hallucination': {
        'color': '#d62728',
        'title': 'Epistemic Wandering and Awakening in Hallucinations (Q1)'
    },
    'Q2_Correct_Rejection': {
        'color': '#2ca02c',
        'title': 'Epistemic Dynamics in Correct Rejections (Q2)'
    },
    'Q3_True_Positive': {
        'color': '#1f77b4',
        'title': 'Epistemic Dynamics in True Positives (Q3)'
    },
    'Q4_True_Negative': {
        'color': '#9467bd',
        'title': 'Epistemic Dynamics in True Negatives (Q4)'
    },
}

def get_available_model_slugs(dataset):
    """
    Discover available model slugs from embedding folder names.
    Prefer EXPORT_BASE, but also support local in-repo experiment_result path.
    """
    candidates = [
        os.path.join(BASE_DIR, 'embeddings', dataset),
        os.path.join('experiment_result', 'exp_temporal_new', 'embeddings', dataset),
    ]
    for emb_dir in candidates:
        if os.path.isdir(emb_dir):
            return sorted(
                d for d in os.listdir(emb_dir)
                if os.path.isdir(os.path.join(emb_dir, d))
            )
    return []


def _model_cache_path(cache_dir, dataset, model_slug):
    safe = model_slug.replace('/', '_')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{dataset}__{safe}.pkl")


def _cache_payload_valid(payload, dataset, model_name, unified_layer):
    if not isinstance(payload, dict) or payload.get('version') != CACHE_FORMAT_VERSION:
        return False
    if payload.get('dataset') != dataset:
        return False
    if payload.get('model_name') != model_name:
        return False
    if int(payload.get('unified_layer', -1)) != int(unified_layer):
        return False
    if tuple(payload.get('timesteps', ())) != tuple(TIMESTEPS):
        return False
    return 'df' in payload and isinstance(payload['df'], pd.DataFrame)


def extract_probe_dataframe(
    model_name,
    model_slug,
    unified_layer,
    unified_probe,
    eval_data,
    gen_data,
    tokenizer,
    print_cot_debug=True,
):
    """
    Build the long-form DataFrame (proportion / timestep / probe prob / quadrant / EOS flag).
    Heavy path: loads embeddings and runs predict_proba.
    """
    lengths = []
    quadrants = []

    if len(gen_data) != len(eval_data):
        raise ValueError(f"Length mismatch: gen={len(gen_data)} vs eval={len(eval_data)}")

    for i in tqdm(range(len(gen_data)), desc=f"Lengths ({model_slug})", leave=False):
        gen_item = gen_data[i]
        eval_item = eval_data[i]
        text = gen_item.get('generated_response', '')
        lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        quadrants.append(eval_item['epistemic_quadrant'])

    lengths = np.array(lengths)
    quadrants = np.array(quadrants)

    if print_cot_debug:
        print(f"{COLOR_HEADER}Sample CoT examples used for length normalization:{COLOR_RESET}")
        sample_indices = [0, len(gen_data) // 2, len(gen_data) - 1]
        for idx in sample_indices:
            text = gen_data[idx].get('generated_response', '')
            print(
                f"{COLOR_META}  idx={idx:<4} len={lengths[idx]:<4} "
                f"quadrant={quadrants[idx]}{COLOR_RESET}"
            )
            print(f"{COLOR_COT}  full_cot={text!r}{COLOR_RESET}")

    records = []
    print("Mapping temporal embeddings to relative proportions...")
    for t in TIMESTEPS:
        emb_path = os.path.join(BASE_DIR, 'embeddings', DATASET, model_slug, f"t_{t}_test.npy")
        if not os.path.exists(emb_path):
            continue

        X_t = np.load(emb_path)[:, unified_layer, :]
        probs_t = unified_probe.predict_proba(X_t)[:, 1]

        for i in range(len(lengths)):
            if lengths[i] > 0 and t <= lengths[i]:
                proportion = t / lengths[i]
                records.append({
                    'Proportion': proportion,
                    'Timestep': t,
                    'Probability': probs_t[i],
                    'Quadrant': quadrants[i],
                    'IsEOS': False
                })

    eos_path = os.path.join(BASE_DIR, 'embeddings', DATASET, model_slug, "t_eos_test.npy")
    if os.path.exists(eos_path):
        X_eos = np.load(eos_path)[:, unified_layer, :]
        probs_eos = unified_probe.predict_proba(X_eos)[:, 1]
        for i in range(len(lengths)):
            records.append({
                'Proportion': 1.0,
                'Timestep': np.nan,
                'Probability': probs_eos[i],
                'Quadrant': quadrants[i],
                'IsEOS': True
            })

    return pd.DataFrame(records)


def parse_plot_args():
    p = argparse.ArgumentParser(description='Exp7 relative / absolute awakening plots (+ cache)')
    p.add_argument(
        '--cache-dir',
        default=DEFAULT_CACHE_DIR,
        help='Directory for per-model extraction caches (pickle).',
    )
    p.add_argument(
        '--recompute',
        action='store_true',
        help='Ignore cache, rerun embedding extraction, overwrite cache files.',
    )
    p.add_argument(
        '--viz-only',
        action='store_true',
        help='Only load caches and plot; skip models with no valid cache (no extraction).',
    )
    p.add_argument(
        '--no-cache-write',
        action='store_true',
        help='After extraction, do not write cache files.',
    )
    p.add_argument(
        '--save-histograms',
        action='store_true',
        help='Save CoT-length histograms for all models (uses *_cot_generations.json).',
    )
    p.add_argument(
        '--hist-only',
        action='store_true',
        help='Only write histograms (skip probe/embedding plots entirely).',
    )
    return p.parse_args()


def run_relative_scatter(args=None):
    if args is None:
        args = parse_plot_args()

    output_dir = os.path.join('paper_plots', 'exp7_awakening', DATASET)
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Cache: {os.path.abspath(args.cache_dir)} "
        f"(default read/write; --recompute to rebuild; --viz-only plot-only; --no-cache-write skip save)"
    )

    # Load Exp 5 data for Unified Layers
    exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{DATASET}.json')
    if not os.path.exists(exp5_path):
        print("Missing Exp 5 json.")
        return
    with open(exp5_path, 'r') as f:
        exp5_data = json.load(f)

    available_slugs = set(get_available_model_slugs(DATASET))
    if available_slugs:
        models_to_run = [m for m in MODELS if m.split('/')[-1] in available_slugs]
        skipped = [m for m in MODELS if m.split('/')[-1] not in available_slugs]
        print(f"Found {len(available_slugs)} embedding model folders for dataset '{DATASET}'.")
        if skipped:
            print(f"Skipping {len(skipped)} models not present as embedding folders.")
    else:
        print(f"Could not find embedding folders for dataset '{DATASET}'. Using MODELS as-is.")
        models_to_run = MODELS

    # Per-quadrant: list of (model_slug, y_on_X_AGG_GRID) for grand aggregate plots.
    aggregate_normalized = {q: [] for q in AGGREGATE_QUADRANTS}

    # Optional: write histograms for all models (standalone; no probes/embeddings needed).
    if args.hist_only or args.save_histograms:
        hist_dir = os.path.join(output_dir, 'histograms')
        os.makedirs(hist_dir, exist_ok=True)
        print(f"Writing CoT length histograms to: {hist_dir}")
        for model_name in models_to_run:
            model_slug = model_name.split('/')[-1]
            trace_path = os.path.join(
                EXPORT_BASE,
                f"experiments/dynamic_tracking_test/math/{model_slug}/{DATASET}_cot_generations.json"
            )
            if not os.path.exists(trace_path):
                print(f"  [skip] missing trace: {trace_path}")
                continue
            out_path = os.path.join(hist_dir, f"cot_len_hist_{model_slug}_{DATASET}.pdf")
            title = f"CoT Length Distribution: {model_slug} ({DATASET})"
            try:
                save_cot_length_histogram(trace_path, model_name, out_path, title)
                print(f"  [ok] {out_path}")
            except Exception as e:
                print(f"  [fail] {model_slug}: {e}")

        if args.hist_only:
            return

    for model_name in models_to_run:
        model_slug = model_name.split('/')[-1]
        print(f"\nProcessing {model_slug}...")

        if model_name not in exp5_data or "unified_layer" not in exp5_data[model_name]:
            continue
            
        unified_layer = exp5_data[model_name]["unified_layer"]
        probe_path = os.path.join(BASE_DIR, 'probes', DATASET, model_slug, f"unified_probe_layer{unified_layer}.joblib")
        
        if not os.path.exists(probe_path):
            continue
            
        unified_probe = joblib.load(probe_path)

        cache_path = _model_cache_path(args.cache_dir, DATASET, model_slug)
        df = None
        if not args.recompute and os.path.isfile(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    pl = pickle.load(f)
                if _cache_payload_valid(pl, DATASET, model_name, unified_layer):
                    df = pl['df']
                    print(f"[cache hit] {cache_path}")
            except Exception as e:
                print(f"[cache corrupt?] {cache_path}: {e}")

        if df is None:
            if args.viz_only:
                print(f"[viz-only] missing or invalid cache, skipping model: {model_slug}")
                continue

            eval_path = os.path.join(
                EXPORT_BASE,
                f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{DATASET}_evaluated_traces.json"
            )
            with open(eval_path, 'r') as f:
                eval_data = json.load(f).get("data", [])

            gen_path = os.path.join(
                EXPORT_BASE,
                f"experiments/dynamic_tracking_test/math/{model_slug}/{DATASET}_cot_generations.json"
            )
            if not os.path.exists(gen_path):
                print(f"Missing generation trace file: {gen_path}")
                continue
            with open(gen_path, 'r') as f:
                gen_data = json.load(f)

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                continue

            try:
                df = extract_probe_dataframe(
                    model_name,
                    model_slug,
                    unified_layer,
                    unified_probe,
                    eval_data,
                    gen_data,
                    tokenizer,
                    print_cot_debug=True,
                )
            except ValueError as e:
                print(f"Skipping model: {e}")
                continue

            if not args.no_cache_write:
                payload = {
                    'version': CACHE_FORMAT_VERSION,
                    'dataset': DATASET,
                    'model_slug': model_slug,
                    'model_name': model_name,
                    'unified_layer': int(unified_layer),
                    'timesteps': tuple(TIMESTEPS),
                    'df': df,
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(payload, f, protocol=4)
                print(f"[cache write] {cache_path}")
        
        # 4. PLOTTING: Generate one plot per quadrant (Q1-Q4)
        for quadrant, style in QUADRANT_STYLES.items():
            df_q = df[df['Quadrant'] == quadrant].copy()
            if df_q.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 6))

            # A. Scatter points
            sns.scatterplot(
                data=df_q,
                x='Proportion',
                y='Probability',
                alpha=0.15,
                color=style['color'],
                edgecolor=None,
                s=20,
                ax=ax
            )

            # B. Binned trend line
            df_q['Bin'] = pd.cut(df_q['Proportion'], bins=np.linspace(0, 1.01, 15), right=False)
            bin_means = df_q.groupby('Bin').mean(numeric_only=True).dropna()

            ax.plot(
                bin_means['Proportion'],
                bin_means['Probability'],
                color='#000000',
                linewidth=3.5,
                marker='D',
                markersize=8,
                zorder=10,
                label='Average Trend (Awakening Curve)'
            )

            x_roll, y_roll = rolling_trailing_mean_sorted(df_q, 'Proportion')
            ax.plot(
                x_roll,
                y_roll,
                color='#ff7f0e',
                linewidth=2.2,
                linestyle='-',
                zorder=11,
                alpha=0.95,
                label=f'Rolling mean (last {ROLLING_WINDOW} pts, sorted by proportion)',
            )

            # Grand aggregate uses rolling-mean trajectories (not the binned black curve).
            if quadrant in aggregate_normalized:
                roll_df = pd.DataFrame({'x': x_roll, 'y': y_roll}).dropna()
                if not roll_df.empty:
                    roll_df = (
                        roll_df.groupby('x', as_index=False)['y']
                        .mean()
                        .sort_values('x')
                    )
                    xp = roll_df['x'].to_numpy(dtype=float)
                    yp = roll_df['y'].to_numpy(dtype=float)
                    if len(xp) >= 2:
                        y_grid = np.interp(X_AGG_GRID, xp, yp, left=float(yp[0]), right=float(yp[-1]))
                    else:
                        y_grid = np.full_like(X_AGG_GRID, float(yp[0]), dtype=float)
                    aggregate_normalized[quadrant].append((model_slug, y_grid))

            # C. Faint cumulative count curve on right-side y-axis
            cdf_x = np.sort(df_q['Proportion'].to_numpy())
            cdf_y = np.arange(1, len(cdf_x) + 1)
            ax2 = ax.twinx()
            ax2.plot(
                cdf_x,
                cdf_y,
                color='#1f77b4',
                linewidth=2.0,
                alpha=0.35,
                linestyle='-',
                zorder=5,
                label='Cumulative Count (points left of x)'
            )
            ax2.set_ylabel('Cumulative Number of Points', color='#1f77b4')
            ax2.tick_params(axis='y', colors='#1f77b4')
            ax2.set_ylim(0, len(cdf_x) * 1.02)

            ax.set_xlim(-0.02, 1.05)
            ax.set_ylim(-0.05, 1.05)

            # Formatting X-axis to percentages
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0% (Start)', '25%', '50%', '75%', '100% (EOS)'])

            ax.set_xlabel('Reasoning Progression (Relative to Total CoT Length)')
            ax.set_ylabel('Latent Probability of "Insufficient"')
            ax.set_title(
                f"{style['title']}\nModel: {model_slug} | Dataset: {DATASET}",
                pad=15
            )

            # Add a decision boundary line
            ax.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)
            ax.text(0.02, 0.52, 'Decision Boundary', color='gray', fontsize=10)

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc='lower center')

            plt.tight_layout()
            quadrant_short = quadrant.split('_')[0]
            out_path = os.path.join(
                output_dir,
                f"Fig_Scatter_{quadrant_short}_{model_slug.replace('/', '_')}_{DATASET}.pdf",
            )
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {out_path}")

            # 5. Absolute-timestep view (exclude EOS), with mean evaluated at each timestep.
            df_q_abs = df_q[df_q['IsEOS'] == False].copy()
            if df_q_abs.empty:
                continue

            fig_abs, ax_abs = plt.subplots(figsize=(9, 6))

            sns.scatterplot(
                data=df_q_abs,
                x='Timestep',
                y='Probability',
                alpha=0.15,
                color=style['color'],
                edgecolor=None,
                s=20,
                ax=ax_abs
            )

            timestep_means = (
                df_q_abs.groupby('Timestep', as_index=False)['Probability']
                .mean()
                .sort_values('Timestep')
            )
            ax_abs.plot(
                timestep_means['Timestep'],
                timestep_means['Probability'],
                color='#000000',
                linewidth=3.5,
                marker='D',
                markersize=8,
                zorder=10,
                label='Average at each timestep'
            )

            ax_abs.set_xlim(min(TIMESTEPS) - 2, max(TIMESTEPS) + 8)
            ax_abs.set_ylim(-0.05, 1.05)
            ax_abs.set_xticks(TIMESTEPS)
            ax_abs.set_xlabel('Reasoning Progression (Absolute Timestep)')
            ax_abs.set_ylabel('Latent Probability of "Insufficient"')
            ax_abs.set_title(
                f"{style['title']} (Absolute Timesteps, EOS Removed)\n"
                f"Model: {model_slug} | Dataset: {DATASET}",
                pad=15
            )
            ax_abs.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)
            ax_abs.text(min(TIMESTEPS) + 2, 0.52, 'Decision Boundary', color='gray', fontsize=10)
            ax_abs.legend(frameon=False, loc='lower center')

            plt.tight_layout()
            out_path_abs = os.path.join(
                output_dir,
                f"Fig_Scatter_{quadrant_short}_{model_slug.replace('/', '_')}_{DATASET}_absolute_t.pdf",
            )
            plt.savefig(out_path_abs, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {out_path_abs}")

    # --- Grand aggregate: normalized proportion curves across models (Q1, Q2 only) ---
    for quadrant in AGGREGATE_QUADRANTS:
        rows = aggregate_normalized[quadrant]
        if not rows:
            print(f"No curves collected for grand aggregate ({quadrant}).")
            continue
        style = QUADRANT_STYLES[quadrant]
        n_models = len(rows)
        model_colors = plt.cm.turbo(np.linspace(0.08, 0.92, n_models))

        # Normal aspect ratio for data axes; legend in its own row (no overlap with x-axis).
        fig_g = plt.figure(figsize=(6.8, 5.4))
        gs = GridSpec(2, 1, figure=fig_g, height_ratios=[3.25, 1.05], hspace=0.55)
        ax_g = fig_g.add_subplot(gs[0])
        leg_ax = fig_g.add_subplot(gs[1])

        mat = []
        for i, (slug, y_grid) in enumerate(rows):
            c = model_colors[i]
            ax_g.plot(
                X_AGG_GRID,
                y_grid,
                color=c,
                alpha=0.55,
                linewidth=1.35,
                zorder=2,
                label=slug,
            )
            mat.append(y_grid)
        Y = np.vstack(mat)
        grand_mean = np.nanmean(Y, axis=0)
        ax_g.plot(
            X_AGG_GRID,
            grand_mean,
            color='#000000',
            linewidth=3.5,
            zorder=10,
            label=f'Mean of rolling curves across {n_models} models',
        )
        trim_mean = trimmed_mean_across_models(Y)
        ax_g.plot(
            X_AGG_GRID,
            trim_mean,
            color='#0055aa',
            linewidth=3.2,
            linestyle='--',
            zorder=12,
            solid_capstyle='round',
            label=(
                f'Mean excluding {GRAND_AGG_TRIM_LOW} lowest & {GRAND_AGG_TRIM_HIGH} '
                f'highest model(s) per x'
            ),
        )
        ax_g.set_xlim(-0.02, 1.05)
        # Dynamic y-rescale for expressiveness: zoom to aggregate curve range with padding.
        y_pool = np.concatenate([grand_mean, trim_mean])
        y_pool = y_pool[~np.isnan(y_pool)]
        if len(y_pool) > 0:
            y_min = float(np.min(y_pool))
            y_max = float(np.max(y_pool))
            span = max(1e-4, y_max - y_min)
            pad = max(0.03, 0.12 * span)
            lo = max(0.0, y_min - pad)
            hi = min(1.0, y_max + pad)
            if hi - lo < 0.08:
                mid = 0.5 * (hi + lo)
                lo = max(0.0, mid - 0.04)
                hi = min(1.0, mid + 0.04)
            ax_g.set_ylim(lo, hi)
        else:
            ax_g.set_ylim(-0.05, 1.05)
        ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_g.set_xticklabels(['0% (Start)', '25%', '50%', '75%', '100% (EOS)'])
        ax_g.set_xlabel('Reasoning Progression (Relative to Total CoT Length)')
        ax_g.set_ylabel('Latent Probability of "Insufficient"')
        ax_g.set_title(
            f"{style['title']} — aggregate across models\nDataset: {DATASET} ({n_models} models)",
            pad=12,
        )
        ax_g.axhline(0.5, color='gray', linestyle='dashed', alpha=0.5, zorder=1)

        n_entries = n_models + 2  # per-model + plain mean + trimmed mean
        ncol = min(6, max(4, int(np.ceil(n_entries / 5))))
        handles, labels = ax_g.get_legend_handles_labels()
        leg_ax.set_axis_off()
        leg_ax.legend(
            handles,
            labels,
            loc='center',
            ncol=ncol,
            frameon=False,
            fontsize=6,
            columnspacing=0.75,
            handletextpad=0.45,
            handlelength=1.15,
        )

        fig_g.tight_layout()
        q_short = quadrant.split('_')[0]
        out_g = os.path.join(
            output_dir,
            f"Fig_GrandAggregate_{q_short}_normalized_{DATASET}.pdf",
        )
        plt.savefig(out_g, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved grand aggregate plot to {out_g}")

if __name__ == '__main__':
    run_relative_scatter(parse_plot_args())