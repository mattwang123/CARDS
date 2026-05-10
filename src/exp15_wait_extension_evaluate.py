"""
================================================================================
EXPERIMENT 15B: Wait Extension Evaluation
================================================================================

Evaluates wait-extended outputs using the same GPT judge used in Exp 2 / Exp 14,
assigning each output to Q1/Q2/Q3/Q4.
================================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_evaluate import extract_regex_naive, get_llm_judgment  # noqa: E402
from exp15_wait_extension_generate import (  # noqa: E402
    DATASETS,
    EXP15_OUTPUT_BASE,
    FULL_MODELS,
    REPRESENTATIVE_MODELS,
    WAIT_COUNTS,
    model_slug,
    select_models,
)

DATASET_PATHS = {
    'umwp':    'src/data/processed/insufficient_dataset_umwp/umwp_test.json',
    'treecut': 'src/data/processed/treecut/treecut_test.json',
}

DEFAULT_INPUT_DIR  = os.path.join(EXP15_OUTPUT_BASE, 'experiments/wait_extension')
DEFAULT_OUTPUT_DIR = os.path.join(EXP15_OUTPUT_BASE, 'experiments/wait_extension_evaluation')


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)


def input_path(input_dir: Path, slug: str, dataset: str, n_waits: int) -> Path:
    return input_dir / slug / f'{dataset}_wait{n_waits}.json'


def output_path(output_dir: Path, slug: str, dataset: str, n_waits: int) -> Path:
    return output_dir / slug / f'{dataset}_wait{n_waits}_evaluated.json'


def checkpoint_path(output_dir: Path, slug: str, dataset: str, n_waits: int) -> Path:
    return output_dir / slug / f'{dataset}_wait{n_waits}_eval_checkpoint.json'


def load_ground_truth(dataset: str) -> dict[str, dict]:
    with open(DATASET_PATHS[dataset], 'r') as f:
        raw_gt = json.load(f)
    return {str(item.get('idx', item.get('id', i))): item for i, item in enumerate(raw_gt)}


# ---------------------------------------------------------------------------
# Quadrant logic (identical to Exp 14)
# ---------------------------------------------------------------------------
def quadrant_from_judgment(is_sufficient: bool, judge_correct: bool) -> str:
    if not is_sufficient and not judge_correct:
        return 'Q1_Hallucination'
    if not is_sufficient and judge_correct:
        return 'Q2_Correct_Rejection'
    if is_sufficient and judge_correct:
        return 'Q3_Solved_Correctly'
    return 'Q4_Competence_Failure'


def initial_stats() -> dict[str, int]:
    return {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 'skipped': 0, 'errors': 0}


def update_stats(stats: dict[str, int], quadrant: str, status: str) -> None:
    if status != 'success':
        stats['skipped'] += 1
        return
    key = {'Q1_Hallucination': 'q1', 'Q2_Correct_Rejection': 'q2',
           'Q3_Solved_Correctly': 'q3', 'Q4_Competence_Failure': 'q4'}.get(quadrant)
    if key:
        stats[key] += 1


def summarize(stats: dict[str, int], results: list[dict]) -> dict:
    suff_total   = stats['q3'] + stats['q4']
    insuff_total = stats['q1'] + stats['q2']
    return {
        'accuracy_sufficient':   float(stats['q3'] / suff_total)   if suff_total   else 0.0,
        'accuracy_insufficient': float(stats['q2'] / insuff_total) if insuff_total else 0.0,
        'quadrant_counts': stats,
        'data': results,
    }


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------
def evaluate_file(args, client, slug: str, dataset: str, n_waits: int) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    in_path    = input_path(input_dir,  slug, dataset, n_waits)
    final_path = output_path(output_dir, slug, dataset, n_waits)
    ckpt_path  = checkpoint_path(output_dir, slug, dataset, n_waits)

    if final_path.exists() and not args.test and not args.overwrite:
        print(f"  [RESUME HIT] {slug}/{dataset}/wait{n_waits} already evaluated.")
        return
    if not in_path.exists():
        print(f"  ! Missing wait-extension generations: {in_path}")
        return

    gt_data = load_ground_truth(dataset)
    gen_data = load_json(in_path)
    if args.test:
        gen_data = gen_data[:args.test_samples]

    results, stats, start_idx = [], initial_stats(), 0
    if ckpt_path.exists() and not args.test and not args.overwrite:
        try:
            ckpt = load_json(ckpt_path)
            results    = ckpt.get('data', [])
            stats      = ckpt.get('stats', initial_stats())
            start_idx  = len(results)
            print(f"  Resuming eval {slug}/{dataset}/wait{n_waits} at {start_idx}/{len(gen_data)}")
        except json.JSONDecodeError:
            pass

    for idx in tqdm(range(start_idx, len(gen_data)),
                    desc=f"Eval {slug}/{dataset}/wait{n_waits}"):
        item = gen_data[idx]
        q_idx = str(item.get('question_idx', item.get('sample_id', idx)))
        gt_item = gt_data.get(q_idx, {})

        generation_status = item.get('status', 'success')
        is_sufficient = gt_item.get('is_sufficient', item.get('is_sufficient'))
        if is_sufficient is None:
            is_sufficient = True
        ground_truth = gt_item.get('answer', gt_item.get('ground_truth', ''))

        raw_output = item.get('raw_output', '')
        extracted  = extract_regex_naive(raw_output)

        base_row = {
            'sample_id':        item.get('sample_id'),
            'question_idx':     q_idx,
            'is_sufficient':    bool(is_sufficient),
            'ground_truth':     ground_truth,
            'n_waits':          item.get('n_waits', n_waits),
            'model_family':     item.get('model_family'),
            'strip_status':     item.get('strip_status'),
            'answer_status':    item.get('answer_status'),
            'strip_pct':        item.get('strip_pct'),
            'extracted_text':   extracted,
            'generated_answer': item.get('generated_answer'),
        }

        if generation_status != 'success':
            results.append({**base_row, 'judge_correct': None,
                            'epistemic_quadrant': 'Skipped', 'judge_status': generation_status})
            update_stats(stats, 'Skipped', generation_status)
        else:
            judge_correct, judge_status = get_llm_judgment(
                client=client, model_name=args.judge_model,
                extracted_text=extracted, is_sufficient=bool(is_sufficient),
                ground_truth=ground_truth,
            )
            if judge_status != 'success':
                stats['errors'] += 1
            quadrant = quadrant_from_judgment(bool(is_sufficient), judge_correct)
            row = {**base_row, 'judge_correct': judge_correct,
                   'epistemic_quadrant': quadrant, 'judge_status': judge_status}
            if args.keep_raw_output:
                row['raw_output'] = raw_output
            results.append(row)
            update_stats(stats, quadrant, 'success')

        if (idx + 1) % args.save_interval == 0 or (idx + 1) == len(gen_data):
            save_json(ckpt_path, {'stats': stats, 'data': results})

    save_json(final_path, summarize(stats, results))
    if ckpt_path.exists() and not args.test:
        ckpt_path.unlink()
    print(f"  -> {slug}/{dataset}/wait{n_waits} | "
          f"Q1:{stats['q1']} Q2:{stats['q2']} Q3:{stats['q3']} Q4:{stats['q4']} "
          f"Skipped:{stats['skipped']} Errors:{stats['errors']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_evaluation(args) -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
        base_url=args.judge_base_url,
    )
    datasets    = args.datasets or DATASETS
    wait_counts = sorted(set(int(n) for n in args.wait_counts))
    for model_name in select_models(args):
        slug = model_slug(model_name)
        print(f"\n{'=' * 80}\nMODEL: {slug}\n{'=' * 80}")
        for dataset in datasets:
            for n_waits in wait_counts:
                evaluate_file(args, client, slug, dataset, n_waits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--datasets',    nargs='+', choices=DATASETS)
    parser.add_argument('--wait_counts', nargs='+', type=int, default=WAIT_COUNTS)
    parser.add_argument('--model',       type=str)
    parser.add_argument('--all_models',  action='store_true')
    parser.add_argument('--judge_model',    type=str, default='gpt-4o-mini')
    parser.add_argument('--judge_base_url', type=str, default=None)
    parser.add_argument('--save_interval',  type=int, default=20)
    parser.add_argument('--keep_raw_output', action='store_true')
    parser.add_argument('--test',         action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--overwrite',    action='store_true')
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
