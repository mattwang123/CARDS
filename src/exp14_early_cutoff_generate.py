"""
================================================================================
EXPERIMENT 14A: Early Cutoff + Force Decode Generation
================================================================================

Takes full CoT traces from Exp 2. For nonzero cutoffs, truncates the generated
reasoning at sentence boundaries near that fraction of the generated-token length.
Cutoff 0 uses no CoT (prompt-only), then appends a forced final-answer prefix
and asks the same model to commit immediately.
================================================================================
"""

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.answer_parser import extract_answer_from_boxed  # noqa: E402


FULL_MODELS = [
    # --- SMALL/MEDIUM SCALE (~1.5B - 4B) ---
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',

    # --- MEDIUM/LARGE SCALE (~7B - 9B) ---
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',

    # --- LARGE SCALE (14B - 32B) ---
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',

    # --- MASSIVE SCALE (70B+) ---
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]

REPRESENTATIVE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'allenai/Olmo-3-7B-Think',
    'google/gemma-3-27b-it',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
]

DATASETS = ['umwp', 'treecut']
# 0.0 = no CoT appended before the forced \\boxed continuation (prompt-only).
CUTOFF_PERCENTAGES = [0.0, 0.20, 0.40, 0.60, 1.00]
FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"

SOURCE_EXPORT_BASE = '/export/fs06/hwang302/CARDS'
EXP14_OUTPUT_BASE = '/export/fs06/xwang397/CARDS/results_new'
EXPORT_BASE = SOURCE_EXPORT_BASE
DEFAULT_GENERATION_DIR = os.path.join(SOURCE_EXPORT_BASE, 'experiments/dynamic_tracking_test')
DEFAULT_OUTPUT_DIR = os.path.join(EXP14_OUTPUT_BASE, 'experiments/early_cutoff')


@dataclass
class CutoffResult:
    text: str
    actual_pct: float
    token_count: int
    target_token_idx: int
    total_tokens: int
    boundary_kind: str


class StopOnCloseBrace:
    """Stop once the forced ``\\boxed{`` continuation closes the brace."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
        generated_ids = input_ids[0, self.prompt_len:]
        if generated_ids.numel() == 0:
            return False
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return "}" in generated_text


def model_slug(model_name: str) -> str:
    return model_name.split('/')[-1]


def cutoff_label(cutoff_pct: float) -> int:
    return int(round(cutoff_pct * 100))


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)['input_ids']


def _sentence_boundary_candidates(text: str) -> list[tuple[int, int, str]]:
    """
    Return (char_pos, priority, kind) candidates.

    Lower priority is better. The final choice still primarily minimizes token
    distance from the requested cutoff.
    """
    candidates: list[tuple[int, int, str]] = []

    for match in re.finditer(r'\n{2,}', text):
        candidates.append((match.end(), 0, 'paragraph_break'))

    for match in re.finditer(r'[.!?](?:\s+)(?=[A-Z]|$)', text):
        candidates.append((match.end(), 1, 'sentence_punct_whitespace'))

    for match in re.finditer(r'[.!?]\n', text):
        candidates.append((match.end(), 2, 'sentence_punct_newline'))

    for match in re.finditer(r'[.!?]', text):
        candidates.append((match.end(), 3, 'bare_sentence_punct'))

    # Remove exact duplicate positions, keeping the best priority/kind.
    best_by_pos: dict[int, tuple[int, str]] = {}
    for pos, priority, kind in candidates:
        if pos not in best_by_pos or priority < best_by_pos[pos][0]:
            best_by_pos[pos] = (priority, kind)

    return [(pos, priority, kind) for pos, (priority, kind) in best_by_pos.items()]


def zero_cutoff_result(cot_text: str, tokenizer) -> CutoffResult:
    """No reasoning text before force decode; still record full-CoT length for metrics."""
    ids = token_ids(tokenizer, cot_text)
    total_tokens = len(ids)
    return CutoffResult("", 0.0, 0, 0, total_tokens, "no_cot")


def find_sentence_boundary_cutoff(text: str, target_pct: float, tokenizer) -> CutoffResult:
    """
    Truncate ``text`` near ``target_pct`` of its token length at a sentence boundary.

    If no boundary exists, falls back to the raw token cutoff. Very short
    generated texts are not skipped here; callers can enforce a minimum.
    """
    ids = token_ids(tokenizer, text)
    total_tokens = len(ids)
    if total_tokens == 0:
        return CutoffResult("", 0.0, 0, 0, 0, "empty")

    target_token_idx = max(1, min(total_tokens, int(target_pct * total_tokens)))
    if target_pct >= 1.0:
        return CutoffResult(text, 1.0, total_tokens, target_token_idx, total_tokens, "full_trace")

    candidates = _sentence_boundary_candidates(text)
    if not candidates:
        raw_text = tokenizer.decode(ids[:target_token_idx], skip_special_tokens=True)
        return CutoffResult(
            raw_text,
            target_token_idx / total_tokens,
            target_token_idx,
            target_token_idx,
            total_tokens,
            "raw_token_cutoff",
        )

    scored = []
    for pos, priority, kind in candidates:
        candidate_text = text[:pos].rstrip()
        candidate_tokens = len(token_ids(tokenizer, candidate_text))
        if candidate_tokens == 0:
            continue
        distance = abs(candidate_tokens - target_token_idx)
        scored.append((distance, priority, candidate_tokens, pos, kind))

    if not scored:
        raw_text = tokenizer.decode(ids[:target_token_idx], skip_special_tokens=True)
        return CutoffResult(
            raw_text,
            target_token_idx / total_tokens,
            target_token_idx,
            target_token_idx,
            total_tokens,
            "raw_token_cutoff",
        )

    _, _, actual_tokens, char_pos, kind = min(scored, key=lambda x: (x[0], x[1]))
    truncated = text[:char_pos].rstrip()
    return CutoffResult(
        truncated,
        actual_tokens / total_tokens,
        actual_tokens,
        target_token_idx,
        total_tokens,
        kind,
    )


def truncate_at_first_close_brace(text: str) -> str:
    close_idx = text.find("}")
    if close_idx == -1:
        return text
    return text[:close_idx + 1]


def select_models(args) -> list[str]:
    if args.model:
        return [args.model]
    return FULL_MODELS if args.all_models else REPRESENTATIVE_MODELS


def load_model_and_tokenizer(model_name: str, dtype: str):
    import importlib.util

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = getattr(torch, dtype)
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_accelerate and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        memory_map = {0: "65GB"} if num_gpus > 0 else None
        if num_gpus > 1:
            for gpu_idx in range(1, num_gpus):
                memory_map[gpu_idx] = "78GB"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=memory_map,
            dtype=model_dtype,
            trust_remote_code=True,
        )
    else:
        print("  ~ accelerate is unavailable; loading model on a single device.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")
    model.eval()
    return model, tokenizer


def expected_generation_path(generation_dir: Path, slug: str, dataset: str) -> Path:
    return generation_dir / 'math' / slug / f'{dataset}_cot_generations.json'


def output_path(output_dir: Path, slug: str, dataset: str, cutoff_pct: float) -> Path:
    return output_dir / slug / f'{dataset}_cutoff{cutoff_label(cutoff_pct)}.json'


def existing_complete(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and len(payload) >= expected_count


def generate_force_decode(model, tokenizer, forced_input: str, max_new_tokens: int) -> tuple[str, str]:
    import torch
    from transformers import StoppingCriteriaList

    inputs = tokenizer(forced_input, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    stopping = StoppingCriteriaList([StopOnCloseBrace(tokenizer, prompt_len)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )

    continuation = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    continuation = truncate_at_first_close_brace(continuation).strip()
    raw_output = FORCE_DECODE_SUFFIX + continuation
    return continuation, raw_output


def iter_cutoffs(cutoff_values: Iterable[float]) -> list[float]:
    return sorted({round(float(value), 2) for value in cutoff_values})


def run_generation(args) -> None:
    generation_dir = Path(args.generation_dir)
    output_dir = Path(args.output_dir)
    datasets = args.datasets or DATASETS
    cutoffs = iter_cutoffs(args.cutoffs)
    models = select_models(args)
    failures = []

    for model_name in models:
        slug = model_slug(model_name)
        print(f"\n{'=' * 80}\nMODEL: {model_name}\n{'=' * 80}")

        needed_work = []
        missing_inputs = []
        for dataset in datasets:
            gen_path = expected_generation_path(generation_dir, slug, dataset)
            if not gen_path.exists():
                print(f"  ! Missing Exp 2 generations for {slug}/{dataset}: {gen_path}")
                missing_inputs.append(str(gen_path))
                continue

            gen_data = load_json(gen_path)
            if args.test:
                gen_data = gen_data[:args.test_samples]
            for cutoff_pct in cutoffs:
                out_path = output_path(output_dir, slug, dataset, cutoff_pct)
                if existing_complete(out_path, len(gen_data)) and not args.overwrite:
                    print(f"  [RESUME HIT] {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)} complete.")
                    continue
                needed_work.append((dataset, cutoff_pct, gen_data, out_path))

        if not needed_work:
            print(f"  - Nothing to generate for {slug}.")
            if args.model and missing_inputs:
                failures.append(f"{slug}: missing Exp 2 generations")
            continue

        try:
            model, tokenizer = load_model_and_tokenizer(model_name, args.dtype)
        except Exception as exc:
            print(f"  ! Failed to load {model_name}: {exc}")
            failures.append(f"{slug}: model load failed: {exc}")
            continue

        try:
            for dataset, cutoff_pct, gen_data, out_path in needed_work:
                results = []
                start_idx = 0
                if out_path.exists() and not args.overwrite:
                    try:
                        results = load_json(out_path)
                        start_idx = len(results)
                        print(f"  Resuming {slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)} at {start_idx}/{len(gen_data)}")
                    except json.JSONDecodeError:
                        print(f"  ! Corrupt output at {out_path}; regenerating.")
                        results = []

                for idx in tqdm(range(start_idx, len(gen_data)), desc=f"{slug}/{dataset}/cutoff{cutoff_label(cutoff_pct)}"):
                    item = gen_data[idx]
                    prompt = item.get('prompt', '')
                    cot_text = item.get('generated_response', item.get('model_output', ''))
                    if cutoff_pct <= 0.0:
                        cutoff = zero_cutoff_result(cot_text, tokenizer)
                    else:
                        cutoff = find_sentence_boundary_cutoff(cot_text, cutoff_pct, tokenizer)

                    sample_id = item.get('sample_id', item.get('question_idx', item.get('id', idx)))
                    if cutoff_pct > 0.0 and cutoff.token_count < args.min_cutoff_tokens:
                        results.append({
                            "sample_id": sample_id,
                            "question_idx": item.get('question_idx', item.get('id', idx)),
                            "question": item.get('question'),
                            "is_sufficient": item.get('is_sufficient'),
                            "cutoff_pct": cutoff_pct,
                            "actual_boundary_pct": cutoff.actual_pct,
                            "boundary_drift_pct": round(cutoff.actual_pct - cutoff_pct, 4),
                            "cutoff_token_count": cutoff.token_count,
                            "total_cot_tokens": cutoff.total_tokens,
                            "target_token_idx": cutoff.target_token_idx,
                            "boundary_kind": cutoff.boundary_kind,
                            "status": "skipped_short_cutoff",
                            "truncated_cot": cutoff.text,
                            "generated_answer": None,
                            "raw_output": "",
                        })
                        continue

                    forced_input = prompt + cutoff.text + FORCE_DECODE_SUFFIX
                    continuation, raw_output = generate_force_decode(
                        model,
                        tokenizer,
                        forced_input,
                        max_new_tokens=args.max_new_tokens,
                    )
                    parsed_answer = extract_answer_from_boxed(raw_output)

                    results.append({
                        "sample_id": sample_id,
                        "question_idx": item.get('question_idx', item.get('id', idx)),
                        "question": item.get('question'),
                        "is_sufficient": item.get('is_sufficient'),
                        "cutoff_pct": cutoff_pct,
                        "actual_boundary_pct": cutoff.actual_pct,
                        "boundary_drift_pct": round(cutoff.actual_pct - cutoff_pct, 4),
                        "cutoff_token_count": cutoff.token_count,
                        "total_cot_tokens": cutoff.total_tokens,
                        "target_token_idx": cutoff.target_token_idx,
                        "boundary_kind": cutoff.boundary_kind,
                        "status": "success",
                        "truncated_cot": cutoff.text,
                        "generated_answer": parsed_answer,
                        "generated_continuation": continuation,
                        "raw_output": raw_output,
                    })

                    if (idx + 1) % args.save_interval == 0 or (idx + 1) == len(gen_data):
                        save_json(out_path, results)

                save_json(out_path, results)
                print(f"  -> Saved {len(results)} rows to {out_path}")
        finally:
            import torch

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if args.model and failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_dir', type=str, default=DEFAULT_GENERATION_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--cutoffs', nargs='+', type=float, default=CUTOFF_PERCENTAGES)
    parser.add_argument('--model', type=str, help='Run a single HuggingFace model name.')
    parser.add_argument('--all_models', action='store_true', help='Run all 21 Exp 10/11 models instead of the representative subset.')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--min_cutoff_tokens', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run_generation(args)


if __name__ == '__main__':
    main()
