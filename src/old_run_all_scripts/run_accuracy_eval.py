"""
Accuracy Evaluation: Test models on SUFFICIENT questions

This script:
1. Loads sufficient questions from test sets
2. Generates model answers using existing MathSolver
3. Judges answers using GPT-4o-mini
4. Computes accuracy for each model-dataset pair
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.inference import MathSolver

# Dataset configurations
DATASETS = {
    'umwp': {
        'test': 'data/processed/insufficient_dataset_umwp/umwp_test.json'
    },
    'gsm8k': {
        'test': 'data/processed/gsm8k/gsm8k_test.json'
    },
    'treecut': {
        'test': 'data/processed/treecut/treecut_test.json'
    }
}

# Model names from config
MODELS = ['qwen2.5-math-1.5b', 'qwen2.5-1.5b', 'llama-3.2-3b-instruct', 'qwen2.5-math-7b']


def load_judge_prompt():
    """Load GPT-4o-mini judge prompt"""
    prompt_path = Path(__file__).parent / "judge_prompts" / "gpt_answer_judge.md"

    with open(prompt_path, 'r') as f:
        return f.read()


def load_sufficient_questions(dataset_name):
    """Load only sufficient questions from test set"""
    data_path = DATASETS[dataset_name]['test']

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Filter for sufficient questions only
    sufficient = [item for item in data if item.get('is_sufficient', False)]

    return sufficient


def judge_answer(question, model_output, ground_truth, judge_prompt, openai_client):
    """
    Use GPT-4o-mini to judge if model answer is correct

    Returns:
        dict: {verdict, model_answer, ground_truth, explanation}
    """
    # Create the evaluation prompt
    evaluation_prompt = f"""{judge_prompt}

## YOUR TASK

**Question**: {question}

**Ground Truth Answer**: {ground_truth}

**Model Output**:
{model_output}

Please evaluate whether the model's answer is correct. Respond with JSON only:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a mathematics answer evaluator. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Deterministic
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"  ⚠ Error in GPT-4o-mini judgment: {e}")
        return {
            "verdict": "error",
            "model_answer": None,
            "ground_truth": ground_truth,
            "explanation": f"API error: {str(e)}"
        }


def evaluate_model_on_dataset(model_name, dataset_name, device, openai_client, judge_prompt,
                               output_dir, test_mode=False, test_samples=3):
    """
    Evaluate one model on one dataset

    Returns:
        dict: Full results for this model-dataset pair
    """
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} | DATASET: {dataset_name}")
    print(f"{'='*80}")

    # Load sufficient questions
    questions = load_sufficient_questions(dataset_name)
    print(f"Loaded {len(questions)} sufficient questions from test set")

    if test_mode:
        questions = questions[:test_samples]
        print(f"TEST MODE: Using only {len(questions)} samples")

    # Check for checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoints',
                                   f"{model_name}_{dataset_name}_checkpoint.json")

    if os.path.exists(checkpoint_path) and not test_mode:
        print(f"  ✓ Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        results = checkpoint['results']
        start_idx = len(results)
        print(f"  → Resuming from question {start_idx + 1}")
    else:
        results = []
        start_idx = 0

    # Skip if already complete
    if start_idx >= len(questions):
        print(f"  ✓ Already completed!")
        return load_results(model_name, dataset_name, output_dir)

    # Load model (only if we need to generate answers)
    print(f"\n[1/2] Loading model: {model_name}")
    solver = MathSolver(model_name, device=device, max_new_tokens=512, temperature=0.7)

    # Process each question
    print(f"\n[2/2] Generating answers and judging...")

    for idx in tqdm(range(start_idx, len(questions)), desc="Evaluating"):
        item = questions[idx]
        question = item['question']
        ground_truth = item['answer']

        # Generate model answer
        try:
            model_output = solver.solve(question)
        except Exception as e:
            print(f"\n  ⚠ Error generating answer for Q{idx + 1}: {e}")
            model_output = f"[ERROR: {str(e)}]"

        # Judge with GPT-4o-mini
        judgment = judge_answer(question, model_output, ground_truth, judge_prompt, openai_client)

        # Store result
        result = {
            'question_idx': idx,
            'question': question,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'judgment': judgment
        }
        results.append(result)

        # Save checkpoint (every 10 questions or in test mode after each)
        if not test_mode and (len(results) % 10 == 0 or idx == len(questions) - 1):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'model': model_name,
                    'dataset': dataset_name,
                    'total_questions': len(questions),
                    'completed': len(results),
                    'results': results
                }, f, indent=2)

    # Compute metrics
    correct = sum(1 for r in results if r['judgment']['verdict'] == 'correct')
    incorrect = sum(1 for r in results if r['judgment']['verdict'] == 'incorrect')
    errors = sum(1 for r in results if r['judgment']['verdict'] == 'error')

    accuracy = correct / len(results) if len(results) > 0 else 0.0

    # Create final results
    final_results = {
        'model': model_name,
        'dataset': dataset_name,
        'split': 'test',
        'filter': 'sufficient_only',
        'total_questions': len(results),
        'correct': correct,
        'incorrect': incorrect,
        'errors': errors,
        'accuracy': accuracy,
        'detailed_results': results
    }

    # Save final results
    if not test_mode:
        results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✓ Saved results: {results_path}")

        # Remove checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"Total questions: {len(results)}")
    print(f"Correct: {correct} ({accuracy*100:.2f}%)")
    print(f"Incorrect: {incorrect}")
    if errors > 0:
        print(f"Errors: {errors}")
    print(f"{'='*80}")

    return final_results


def load_results(model_name, dataset_name, output_dir):
    """Load existing results if available"""
    results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def run_test_mode(device, openai_client, judge_prompt, test_samples=3):
    """Quick test mode: 1 model, 1 dataset, 3 samples with detailed output"""
    model_name = 'qwen2.5-math-1.5b'
    dataset_name = 'umwp'

    print("="*80)
    print("TEST MODE - Accuracy Evaluation")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} (test split, sufficient questions only)")
    print(f"Samples: {test_samples}")
    print("="*80)

    # Load questions
    questions = load_sufficient_questions(dataset_name)[:test_samples]

    # Load model
    print(f"\nLoading model...")
    solver = MathSolver(model_name, device=device, max_new_tokens=512, temperature=0.7)

    results = []

    for idx, item in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{test_samples}]")
        print(f"{'='*80}")

        question = item['question']
        ground_truth = item['answer']

        print(f"\nQuestion:")
        print(f"  {question}")
        print(f"\nGround Truth:")
        print(f"  {ground_truth}")

        # Generate
        print(f"\nGenerating model answer...")
        model_output = solver.solve(question)
        print(f"\nModel Output:")
        print(f"  {model_output}")

        # Judge
        print(f"\nCalling GPT-4o-mini judge...")
        judgment = judge_answer(question, model_output, ground_truth, judge_prompt, openai_client)
        print(f"\nJudgment:")
        print(f"  {json.dumps(judgment, indent=2)}")

        verdict_symbol = "✓" if judgment['verdict'] == 'correct' else "✗"
        print(f"\n{verdict_symbol} {judgment['verdict'].upper()}")

        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'judgment': judgment
        })

    # Summary
    correct = sum(1 for r in results if r['judgment']['verdict'] == 'correct')
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Correct: {correct}/{len(results)}")
    print(f"Accuracy: {correct/len(results)*100:.1f}%")
    print(f"{'='*80}")


def estimate_cost(all_models, all_datasets):
    """Estimate API cost for GPT-4o-mini"""
    total_questions = 0

    print("\n" + "="*80)
    print("COST ESTIMATION (GPT-4o-mini)")
    print("="*80)

    for dataset_name in all_datasets:
        questions = load_sufficient_questions(dataset_name)
        print(f"{dataset_name}: {len(questions)} sufficient questions")
        total_questions += len(questions)

    total_calls = total_questions * len(all_models)

    # GPT-4o-mini pricing (as of Nov 2024)
    input_cost_per_1m = 0.150   # $0.15 per 1M input tokens
    output_cost_per_1m = 0.600  # $0.60 per 1M output tokens

    # Estimate tokens
    avg_input_tokens = 1000   # Prompt + question + model output
    avg_output_tokens = 150   # JSON response

    total_input_tokens = total_calls * avg_input_tokens
    total_output_tokens = total_calls * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    print(f"\nTotal sufficient questions: {total_questions}")
    print(f"Models: {len(all_models)}")
    print(f"Total API calls: {total_calls}")
    print(f"Estimated input tokens: {total_input_tokens:,}")
    print(f"Estimated output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print("="*80 + "\n")

    return total_cost


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model accuracy on sufficient questions using GPT-4o-mini judge'
    )
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for model inference')
    parser.add_argument('--output_dir', type=str, default='experiments/accuracy_eval',
                        help='Directory to save results')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: 1 model, 1 dataset, 3 samples with detailed output')
    parser.add_argument('--test_samples', type=int, default=3,
                        help='Number of samples in test mode')
    parser.add_argument('--skip_confirmation', action='store_true',
                        help='Skip cost confirmation prompt')

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("\n✗ Error: OPENAI_API_KEY not found")
        print("Set it in .env file or environment:")
        print("  export OPENAI_API_KEY='your_key_here'")
        return

    openai_client = OpenAI(api_key=api_key)

    # Load judge prompt
    judge_prompt = load_judge_prompt()
    print("✓ Loaded GPT-4o-mini judge prompt")

    # Test mode
    if args.test:
        run_test_mode(args.device, openai_client, judge_prompt, args.test_samples)
        return

    # Full evaluation mode
    print("="*80)
    print("ACCURACY EVALUATION - ALL MODELS ON ALL DATASETS")
    print("="*80)
    print(f"Models: {MODELS}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Device: {args.device}")
    print("="*80)

    # Estimate cost
    if not args.skip_confirmation:
        estimate_cost(MODELS, list(DATASETS.keys()))
        response = input("Continue with evaluation? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all evaluations
    all_results = {}

    for model_name in MODELS:
        model_results = {}

        for dataset_name in DATASETS.keys():
            result = evaluate_model_on_dataset(
                model_name, dataset_name, args.device,
                openai_client, judge_prompt, args.output_dir
            )
            model_results[dataset_name] = {
                'accuracy': result['accuracy'],
                'correct': result['correct'],
                'total': result['total_questions']
            }

        all_results[model_name] = model_results

    # Save summary
    summary_path = os.path.join(args.output_dir, 'accuracy_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            acc = all_results[model_name][dataset_name]['accuracy']
            row.append(f"{acc*100:>5.1f}%")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ Individual results saved in: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
