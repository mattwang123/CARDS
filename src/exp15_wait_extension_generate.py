"""
================================================================================
EXPERIMENT 15: Wait-Token Reasoning Extension
================================================================================

Takes full CoT traces from Exp 2, strips the final conclusion/answer, injects
N repetitions of a model-family-appropriate "Wait" signal, and lets the model
extend its reasoning before committing to a new answer.

Two answer modes:
  natural      - model naturally produces a new \\boxed{} in the continuation
  force_decoded - no \\boxed{} appeared; FORCE_DECODE_SUFFIX appended as fallback
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
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.answer_parser import extract_answer_from_boxed  # noqa: E402


# ---------------------------------------------------------------------------
# Model list (identical to Exp 10/11/14)
# ---------------------------------------------------------------------------
FULL_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct',
]

REPRESENTATIVE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'allenai/Olmo-3-7B-Think',
    'google/gemma-3-27b-it',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
]

DATASETS = ['umwp', 'treecut']
WAIT_COUNTS = [1, 3, 5]

SOURCE_EXPORT_BASE = '/export/fs06/hwang302/CARDS'
EXP15_OUTPUT_BASE = '/export/fs06/xwang397/CARDS/results_new'
DEFAULT_GENERATION_DIR = os.path.join(SOURCE_EXPORT_BASE, 'experiments/dynamic_tracking_test')
DEFAULT_OUTPUT_DIR = os.path.join(EXP15_OUTPUT_BASE, 'experiments/wait_extension')

FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"

# Think-block close tags by model family; open tags used to preserve context.
THINK_CLOSE_TAGS = {
    'think_olmo':     '<|/thought|>',
    'think_deepseek': '</think>',
}

# Wait string injected after stripping; in-distribution for thinking models.
WAIT_STRINGS = {
    'think_olmo':     'Wait,',
    'think_deepseek': 'Wait,',
    'standard':       '\nWait, let me re-read the problem.',
}

MIN_STRIPPED_TOKENS = 20


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StripResult:
    stripped_cot: str
    conclusion_fragment: str
    strip_status: str        # 'stripped' | 'think_tag_stripped' | 'no_conclusion' | 'no_boundary'
    stripped_token_count: int
    original_token_count: int

    @property
    def strip_pct(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return round(1.0 - self.stripped_token_count / self.original_token_count, 4)


# ---------------------------------------------------------------------------
# Helpers shared with exp14
# ---------------------------------------------------------------------------
def model_slug(model_name: str) -> str:
    return model_name.split('/')[-1]


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)['input_ids']


def select_models(args) -> list[str]:
    if args.model:
        return [args.model]
    return FULL_MODELS if args.all_models else REPRESENTATIVE_MODELS


def output_path(output_dir: Path, slug: str, dataset: str, n_waits: int) -> Path:
    return output_dir / slug / f'{dataset}_wait{n_waits}.json'


def existing_complete(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and len(payload) >= expected_count


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------
def get_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if 'think' in lower:
        return 'think_olmo'
    if 'deepseek-r1' in lower or 'r1-distill' in lower:
        return 'think_deepseek'
    return 'standard'


# ---------------------------------------------------------------------------
# Conclusion stripping
# ---------------------------------------------------------------------------
def _strip_last_boxed_sentence(text: str) -> tuple[str, str]:
    """
    Scan backward from the last \\boxed{ to the nearest preceding sentence or
    paragraph boundary.  Returns (text_before_conclusion, removed_fragment).
    """
    boxed_idx = text.rfind(r'\boxed{')
    if boxed_idx == -1:
        return text, ''

    prefix = text[:boxed_idx]

    # Priority order: paragraph break first, then sentence-ending punctuation.
    for pattern in [r'\n{2,}', r'[.!?][ \t]+', r'[.!?]\n', r'\n']:
        matches = list(re.finditer(pattern, prefix))
        if matches:
            cut_pos = matches[-1].end()
            return text[:cut_pos].rstrip(), text[cut_pos:]

    # Last resort: strip the entire last line containing \boxed{.
    last_newline = prefix.rfind('\n')
    if last_newline != -1:
        return text[:last_newline].rstrip(), text[last_newline:]

    return '', text   # nothing usable left before the conclusion


def strip_conclusion(text: str, model_family: str, tokenizer) -> StripResult:
    """
    Remove the final-answer sentence(s) from a CoT trace.

    For thinking models the close-think tag and the answer block outside it are
    removed first; then the last conclusion sentence inside the think block is
    stripped, leaving an open think context for Wait injection.
    """
    original_tokens = len(token_ids(tokenizer, text))

    if model_family in THINK_CLOSE_TAGS:
        close_tag = THINK_CLOSE_TAGS[model_family]
        tag_idx = text.rfind(close_tag)
        if tag_idx != -1:
            inside_think = text[:tag_idx]
            stripped, frag = _strip_last_boxed_sentence(inside_think)
            if not stripped:
                # No \\boxed{ inside the think block — keep all inside-think content.
                stripped, frag = inside_think, ''
            conclusion_fragment = frag + text[tag_idx:]
            stripped_tokens = len(token_ids(tokenizer, stripped))
            return StripResult(
                stripped_cot=stripped,
                conclusion_fragment=conclusion_fragment,
                strip_status='think_tag_stripped',
                stripped_token_count=stripped_tokens,
                original_token_count=original_tokens,
            )
        # Think model but no close tag in output (partial generation) — fall through.

    stripped, frag = _strip_last_boxed_sentence(text)
    if not frag:
        return StripResult(
            stripped_cot=text,
            conclusion_fragment='',
            strip_status='no_conclusion',
            stripped_token_count=original_tokens,
            original_token_count=original_tokens,
        )

    stripped_tokens = len(token_ids(tokenizer, stripped))
    return StripResult(
        stripped_cot=stripped,
        conclusion_fragment=frag,
        strip_status='stripped',
        stripped_token_count=stripped_tokens,
        original_token_count=original_tokens,
    )


# ---------------------------------------------------------------------------
# Wait injection
# ---------------------------------------------------------------------------
def build_wait_injection(model_family: str, n_waits: int) -> str:
    """Repeat the family-specific wait string n_waits times with space separators."""
    wait_str = WAIT_STRINGS[model_family].strip()
    return (' ' + wait_str) * n_waits


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str, dtype: str):
    import importlib.util

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, dtype)
    has_accelerate = importlib.util.find_spec('accelerate') is not None
    if has_accelerate and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        memory_map = {0: '65GB'} if num_gpus > 0 else None
        if num_gpus > 1:
            for i in range(1, num_gpus):
                memory_map[i] = '78GB'
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', max_memory=memory_map,
            torch_dtype=torch_dtype, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        import torch as _torch
        if _torch.cuda.is_available():
            model.to('cuda')
    model.eval()
    return model, tokenizer


def extract_last_boxed_raw(text: str) -> Optional[str]:
    """Return the full \\boxed{...} token of the LAST match, or None."""
    matches = list(re.finditer(r'\\boxed\{[^{}]*\}', text))
    return matches[-1].group(0) if matches else None


def generate_free(model, tokenizer, full_input: str, max_new_tokens: int) -> str:
    """Generate freely; return only the newly generated text."""
    import torch
    inputs = tokenizer(full_input, return_tensors='pt').to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def force_decode(model, tokenizer, prefix: str) -> str:
    """Append FORCE_DECODE_SUFFIX and generate ≤50 tokens; return full raw_output."""
    import torch
    from transformers import StoppingCriteriaList

    class _StopOnBrace:
        def __init__(self, plen):
            self.plen = plen
        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[0, self.plen:]
            if gen.numel() == 0:
                return False
            return '}' in tokenizer.decode(gen, skip_special_tokens=True)

    full_input = prefix + FORCE_DECODE_SUFFIX
    inputs = tokenizer(full_input, return_tensors='pt').to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopOnBrace(prompt_len)]),
        )
    cont = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    close = cont.find('}')
    if close != -1:
        cont = cont[:close + 1]
    return FORCE_DECODE_SUFFIX + cont


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def run_generation(args) -> None:
    generation_dir = Path(args.generation_dir)
    output_dir = Path(args.output_dir)
    datasets = args.datasets or DATASETS
    wait_counts = sorted(set(int(n) for n in args.wait_counts))
    models = select_models(args)

    for model_name in models:
        slug = model_slug(model_name)
        model_family = get_model_family(model_name)
        print(f"\n{'=' * 80}\nMODEL: {model_name}  (family: {model_family})\n{'=' * 80}")

        # Check which (dataset, n_waits) combos still need work.
        needed_work = []
        for dataset in datasets:
            gen_path = generation_dir / 'math' / slug / f'{dataset}_cot_generations.json'
            if not gen_path.exists():
                print(f"  ! Missing Exp 2 generations: {gen_path}")
                continue
            gen_data = load_json(gen_path)
            if args.test:
                gen_data = gen_data[:args.test_samples]
            for n_waits in wait_counts:
                out_path = output_path(output_dir, slug, dataset, n_waits)
                if existing_complete(out_path, len(gen_data)) and not args.overwrite:
                    print(f"  [RESUME HIT] {slug}/{dataset}/wait{n_waits} complete.")
                    continue
                needed_work.append((dataset, n_waits, gen_data, out_path))

        if not needed_work:
            print(f"  - Nothing to generate for {slug}.")
            continue

        try:
            model, tokenizer = load_model_and_tokenizer(model_name, args.dtype)
        except Exception as exc:
            print(f"  ! Failed to load {model_name}: {exc}")
            continue

        try:
            for dataset, n_waits, gen_data, out_path in needed_work:
                wait_injection = build_wait_injection(model_family, n_waits)
                results = []
                start_idx = 0
                if out_path.exists() and not args.overwrite:
                    try:
                        results = load_json(out_path)
                        start_idx = len(results)
                        print(f"  Resuming {slug}/{dataset}/wait{n_waits} at {start_idx}/{len(gen_data)}")
                    except json.JSONDecodeError:
                        results = []

                desc = f"{slug}/{dataset}/wait{n_waits}"
                for idx in tqdm(range(start_idx, len(gen_data)), desc=desc):
                    item = gen_data[idx]
                    prompt = item.get('prompt', '')
                    cot_text = item.get('generated_response', item.get('model_output', ''))
                    sample_id = item.get('sample_id', item.get('question_idx', item.get('id', idx)))
                    q_idx = item.get('question_idx', item.get('id', idx))

                    strip = strip_conclusion(cot_text, model_family, tokenizer)

                    base_row = {
                        'sample_id': sample_id,
                        'question_idx': q_idx,
                        'question': item.get('question'),
                        'is_sufficient': item.get('is_sufficient'),
                        'model_family': model_family,
                        'n_waits': n_waits,
                        'wait_string_used': WAIT_STRINGS[model_family],
                        'strip_status': strip.strip_status,
                        'stripped_cot_token_count': strip.stripped_token_count,
                        'original_cot_token_count': strip.original_token_count,
                        'strip_pct': strip.strip_pct,
                        'conclusion_fragment': strip.conclusion_fragment,
                    }

                    if strip.stripped_token_count < MIN_STRIPPED_TOKENS:
                        results.append({
                            **base_row,
                            'status': 'skipped_too_short',
                            'answer_status': None,
                            'generated_new_cot': None,
                            'generated_answer': None,
                            'raw_output': '',
                        })
                    else:
                        full_input = prompt + strip.stripped_cot + wait_injection
                        new_cot = generate_free(model, tokenizer, full_input, args.max_new_tokens)

                        if extract_last_boxed_raw(new_cot):
                            answer_status = 'natural'
                            raw_output = new_cot
                        else:
                            # Fallback: force-decode from the intervention point.
                            raw_output = force_decode(model, tokenizer, full_input)
                            answer_status = 'force_decoded'

                        results.append({
                            **base_row,
                            'status': 'success',
                            'answer_status': answer_status,
                            'generated_new_cot': new_cot,
                            'generated_answer': extract_answer_from_boxed(raw_output),
                            'raw_output': raw_output,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_dir', type=str, default=DEFAULT_GENERATION_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--wait_counts', nargs='+', type=int, default=WAIT_COUNTS)
    parser.add_argument('--model', type=str, help='Run a single HuggingFace model name.')
    parser.add_argument('--all_models', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_samples', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    run_generation(args)


if __name__ == '__main__':
    main()
