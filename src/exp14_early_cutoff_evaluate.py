"""
================================================================================
EXPERIMENT 14B: Early Cutoff Evaluation
================================================================================

Evaluates force-decoded early-cutoff answers with the same GPT judge conventions
used in Exp 2, assigning each cutoff output to Q1/Q2/Q3/Q4.
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


DATASET_PATHS = {
    'umwp': 'src/data/processed/insufficient_dataset_umwp/umwp_test.json',
    'treecut': 'src/data/processed/treecut/treecut_test.json',
}

DEFAULT_INPUT_DIR = os.path.join(EXP14_OUTPUT_BASE, 'experiments/early_cutoff')
DEFAULT_OUTPUT_DIR = os.path.join(EXP14_OUTPUT_BASE, 'experiments/early_cutoff_evaluation')


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)


def select_models(args) -> list[str]:
    if args.model:
        return [args.model]
    return FULL_MODELS if args.all_models else REPRESENTATIVE_MODELS


def input_path(input_dir: Path, slug: str, dataset: str, cutoff_pct: float) -> Path:
    return input_dir / slug / f'{dataset}_cutoff{cutoff_label(cutoff_pct)}.json'


def output_path(output_dir: Path, slug: str, dataset: str, cutoff_pct: float) -> Path:
    return output_dir / slug / f'{dataset}_cutoff{cutoff_label(cutoff_pct)}_evaluated.json'


def checkpoint_path(output_dir: Path, slug: str, dataset: str, cutoff_pct: float) -> Path:
    return output_dir / slug / f'{dataset}_cutoff{cutoff_label(cutoff_pct)}_eval_checkpoint.json'


def load_ground_truth(dataset: str) -> dict[str, dict]:
    with open(DATASET_PATHS[dataset], 'r') as f:
        raw_gt = json.load(f)
    return {str(item.get('idx', item.get('id', i))): item for i, item in enumerate(raw_gt)}


def quadrant_from_judgment(is_sufficient: bool, judge_correct: bool) -> str:
    if not is_sufficient and not judge_correct:
        return "Q1_Hallucination"
    if not is_sufficient and judge_correct:
        return "Q2_Correct_Rejection"
    if is_sufficient and judge_correct:
        return "Q3_Solved_Correctly"
    return "Q4_Competence_Failure"


def initial_stats() -> dict[str, int]:
    return {"q1": 0, "q2": 0, "q3": 0, "q4": 0, "skipped": 0, "errors": 0}


def update_stats(stats: dict[str, int], quadrant: str, status: str) -> None:
    if status != "success":
        stats["skipped"] += 1
        return
    if quadrant == "Q1_Hallucination":
        stats["q1"] += 1
    elif quadrant == "Q2_Correct_Rejection":
        stats["q2"] += 1
    elif quadrant == "Q3_Solved_Correctly":
        stats["q3"] += 1
    elif quadrant == "Q4_Competence_Failure":
        stats["q4"] += 1


def summarize(stats: dict[str, int], results: list[dict]) -> dict:
    suff_total = stats["q3"] + stats["q4"]
    insuff_total = stats["q1"] + stats["q2"]
    return {
        "accuracy_sufficient": float(stats["q3"] / suff_total) if suff_total else 0.0,
        "accuracy_insufficient": float(stats["q2"] / insuff_total) if insuff_total else 0.0,
        "quadrant_counts": stats,
        "data": results,
    }


def evaluate_file(args, client, slug: str, dataset: str, cutoff_pct: float) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    in_path = input_path(input_dir, slug, dataset, cutoff_pct)
    final_path = output_path(output_dir, slug, dataset, cutoff_pct)
    ckpt_path = checkpoint_path(output_dir, slug, dataset, cutoff_pct)

    if final_path.exists() and not args.test and not args.overwrite:
        print(f"  [RESUME HIT] {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)} already evaluated.")
        return
    if not in_path.exists():
        print(f"  ! Missing cutoff generations: {in_path}")
        return

    gt_data = load_ground_truth(dataset)
    cutoff_data = load_json(in_path)
    if args.test:
        cutoff_data = cutoff_data[:args.test_samples]

    results = []
    stats = initial_stats()
    start_idx = 0
    if ckpt_path.exists() and not args.test and not args.overwrite:
        try:
            ckpt = load_json(ckpt_path)
            results = ckpt.get("data", [])
            stats = ckpt.get("stats", initial_stats())
            start_idx = len(results)
            print(f"  Resuming eval {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)} at {start_idx}/{len(cutoff_data)}")
        except json.JSONDecodeError:
            print(f"  ! Corrupt checkpoint at {ckpt_path}; restarting.")

    for idx in tqdm(range(start_idx, len(cutoff_data)), desc=f"Eval {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)}"):
        item = cutoff_data[idx]
        q_idx = str(item.get('question_idx', item.get('sample_id', idx)))
        gt_item = gt_data.get(q_idx, {})

        generation_status = item.get("status", "success")
        is_sufficient = gt_item.get('is_sufficient', item.get('is_sufficient'))
        if is_sufficient is None:
            is_sufficient = True
        ground_truth = gt_item.get('answer', gt_item.get('ground_truth', ''))

        raw_output = item.get('raw_output', '')
        extracted_raw_text = extract_regex_naive(raw_output)

        if generation_status != "success":
            result_row = {
                "sample_id": item.get("sample_id"),
                "question_idx": q_idx,
                "is_sufficient": is_sufficient,
                "ground_truth": ground_truth,
                "cutoff_pct": item.get("cutoff_pct", cutoff_pct),
                "actual_boundary_pct": item.get("actual_boundary_pct"),
                "cutoff_token_count": item.get("cutoff_token_count"),
                "total_cot_tokens": item.get("total_cot_tokens"),
                "extracted_raw_text": extracted_raw_text,
                "generated_answer": item.get("generated_answer"),
                "judge_correct": None,
                "epistemic_quadrant": "Skipped",
                "judge_status": generation_status,
            }
            results.append(result_row)
            update_stats(stats, "Skipped", generation_status)
        else:
            judge_correct, judge_status = get_llm_judgment(
                client=client,
                model_name=args.judge_model,
                extracted_text=extracted_raw_text,
                is_sufficient=bool(is_sufficient),
                ground_truth=ground_truth,
            )
            if judge_status != "success":
                stats["errors"] += 1

            quadrant = quadrant_from_judgment(bool(is_sufficient), judge_correct)
            result_row = {
                "sample_id": item.get("sample_id"),
                "question_idx": q_idx,
                "is_sufficient": bool(is_sufficient),
                "ground_truth": ground_truth,
                "cutoff_pct": item.get("cutoff_pct", cutoff_pct),
                "actual_boundary_pct": item.get("actual_boundary_pct"),
                "cutoff_token_count": item.get("cutoff_token_count"),
                "total_cot_tokens": item.get("total_cot_tokens"),
                "boundary_kind": item.get("boundary_kind"),
                "extracted_raw_text": extracted_raw_text,
                "generated_answer": item.get("generated_answer"),
                "judge_correct": judge_correct,
                "epistemic_quadrant": quadrant,
                "judge_status": judge_status,
            }
            if args.keep_raw_output:
                result_row["raw_output"] = raw_output
            results.append(result_row)
            update_stats(stats, quadrant, "success")

        if (idx + 1) % args.save_interval == 0 or (idx + 1) == len(cutoff_data):
            save_json(ckpt_path, {"stats": stats, "data": results})

    save_json(final_path, summarize(stats, results))
    if ckpt_path.exists() and not args.test:
        ckpt_path.unlink()

    print(
        f"  -> {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)} | "
        f"Q1:{stats['q1']} Q2:{stats['q2']} Q3:{stats['q3']} Q4:{stats['q4']} "
        f"Skipped:{stats['skipped']} Errors:{stats['errors']}"
    )


def run_evaluation(args) -> None:
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=args.judge_base_url,
    )

    datasets = args.datasets or DATASETS
    cutoffs = sorted({round(float(value), 2) for value in args.cutoffs})
    for model_name in select_models(args):
        slug = model_slug(model_name)
        print(f"\n{'=' * 80}\nMODEL: {slug}\n{'=' * 80}")
        for dataset in datasets:
            for cutoff_pct in cutoffs:
                evaluate_file(args, client, slug, dataset, cutoff_pct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--cutoffs', nargs='+', type=float, default=CUTOFF_PERCENTAGES)
    parser.add_argument('--model', type=str, help='Run a single HuggingFace model name.')
    parser.add_argument('--all_models', action='store_true', help='Evaluate all 21 Exp 10/11 models.')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--judge_base_url', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--keep_raw_output', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
