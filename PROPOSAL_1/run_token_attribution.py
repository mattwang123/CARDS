"""
Run token-level attribution analysis to see which words trigger sufficiency detectors

Usage:
    python run_token_attribution.py \
        --data_path data/insufficient_dataset_umwp/umwp_test.json \
        --probe_dir experiments/probe_results/results \
        --model_name qwen2.5-math-1.5b \
        --layer 18 \
        --output_dir experiments/probe_results/mech_interp \
        --num_examples 6 \
        --device cpu
"""
import argparse
import os
import glob

from viz.advanced_mech_interp import token_attribution_analysis


def find_best_layer_probe(probe_dir, metrics_path):
    """Find the best performing layer from metrics"""
    import json

    # Load metrics
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # Find layer with best F1
    best_layer = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_f1'])
    best_f1 = all_metrics[best_layer]['test_f1']

    # Find corresponding probe file (.pkl for linear probes)
    probe_path = os.path.join(probe_dir, f'layer_{best_layer}_probe.pkl')

    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Linear probe file not found: {probe_path}")

    return probe_path, best_layer, best_f1


def main():
    parser = argparse.ArgumentParser(description='Token-level attribution analysis')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to JSON data file (test set recommended)')
    parser.add_argument('--probe_dir', type=str, default='experiments/probe_results_linear/results',
                        help='Directory containing trained LINEAR probes')
    parser.add_argument('--metrics_path', type=str, default='experiments/probe_results_linear/all_metrics.json',
                        help='Path to metrics JSON (to find best layer)')
    parser.add_argument('--model_name', type=str, default='qwen2.5-math-1.5b',
                        help='Model name from config')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to analyze (default: auto-select best layer)')
    parser.add_argument('--output_dir', type=str, default='experiments/probe_results_linear/mech_interp',
                        help='Output directory')
    parser.add_argument('--num_examples', type=int, default=6,
                        help='Number of examples to visualize (half sufficient, half insufficient)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')

    args = parser.parse_args()

    print("="*80)
    print("TOKEN ATTRIBUTION ANALYSIS")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which layer to use
    if args.layer is None:
        print("\nAuto-selecting best layer based on metrics...")
        probe_path, layer_idx, f1_score = find_best_layer_probe(args.probe_dir, args.metrics_path)
        print(f"  Selected layer {layer_idx} (F1: {f1_score:.3f})")
    else:
        layer_idx = args.layer
        probe_path = os.path.join(args.probe_dir, f'layer_{layer_idx}_probe.pkl')
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"Linear probe file not found: {probe_path}")
        print(f"\nUsing specified layer {layer_idx}")

    # Output path
    output_path = os.path.join(args.output_dir, f'token_attribution_layer{layer_idx}.png')

    # Run attribution
    token_attribution_analysis(
        data_path=args.data_path,
        probe_path=probe_path,
        model_name=args.model_name,
        layer_idx=layer_idx,
        output_path=output_path,
        num_examples=args.num_examples,
        device=args.device
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_path}")
    print(f"\nThis visualization shows which tokens (words/subwords) in each question")
    print(f"contribute most to the probe's classification decision:")
    print(f"  - Blue bars → tokens that signal 'Sufficient'")
    print(f"  - Red bars → tokens that signal 'Insufficient'")
    print(f"  - Longer bars → stronger contribution")


if __name__ == '__main__':
    main()
