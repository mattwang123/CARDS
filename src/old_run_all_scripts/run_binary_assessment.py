"""
Direct Binary Assessment

This script:
1. Loads ALL questions (sufficient + insufficient) from test sets
2. Prompts models: "Can this math problem be solved with the given information?"
3. Extracts Yes/No responses and evaluates against ground truth sufficiency
4. Computes assessment accuracy and output token efficiency for each model-dataset pair
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
import re

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
MODELS = ['qwen2.5-math-1.5b', 'qwen2.5-1.5b', 'qwen2.5-math-7b']


def load_judge_prompt():
    """Load GPT-4o-mini judge prompt for binary assessment"""
    prompt_path = Path(__file__).parent / "judge_prompts" / "gpt_binary_judge.md"
    
    if not prompt_path.exists():
        # Create default binary judge prompt if it doesn't exist
        default_prompt = """You are evaluating whether a model correctly assessed if a math problem can be solved.

The model was asked: "Can this math problem be solved with the given information?"
The model should answer "Yes" if sufficient information is provided, "No" if insufficient.

Evaluate if the model's assessment matches the ground truth.

Return JSON with:
- "verdict": "correct" or "incorrect" 
- "model_answer": extracted Yes/No from model output
- "ground_truth": the correct Yes/No answer
- "explanation": brief reasoning"""
        
        os.makedirs(prompt_path.parent, exist_ok=True)
        with open(prompt_path, 'w') as f:
            f.write(default_prompt)
        print(f"✓ Created default binary judge prompt: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read()


def load_all_questions(dataset_name):
    """Load ALL questions from test set (both sufficient and insufficient)"""
    data_path = DATASETS[dataset_name]['test']

    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def extract_yes_no_answer(model_output):
    """Extract Yes/No from model output"""
    # Look for boxed answer first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_output, re.IGNORECASE)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if 'yes' in answer.lower():
            return 'Yes'
        elif 'no' in answer.lower():
            return 'No'
    
    # Look for explicit Yes/No patterns
    output_lower = model_output.lower()
    
    # Count occurrences
    yes_count = len(re.findall(r'\byes\b', output_lower))
    no_count = len(re.findall(r'\bno\b', output_lower))
    
    # Look for final answer patterns
    final_patterns = [
        r'final answer[:\s]*([yn]o?)',
        r'answer[:\s]*([yn]o?)',
        r'therefore[,:\s]*([yn]o?)',
        r'so[,:\s]*([yn]o?)',
        r'conclusion[:\s]*([yn]o?)'
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, output_lower)
        if match:
            answer = match.group(1)
            if answer.startswith('y'):
                return 'Yes'
            elif answer.startswith('n'):
                return 'No'
    
    # If no clear pattern, use counts
    if yes_count > no_count:
        return 'Yes'
    elif no_count > yes_count:
        return 'No'
    
    return None  # Unclear


def judge_binary_assessment(question, model_output, ground_truth, judge_prompt, openai_client):
    """Use GPT-4o-mini to judge if binary assessment is correct"""
    # First try to extract answer directly
    extracted_answer = extract_yes_no_answer(model_output)
    
    if extracted_answer is not None:
        # Direct comparison
        is_correct = (extracted_answer == ground_truth)
        return {
            "verdict": "correct" if is_correct else "incorrect",
            "model_answer": extracted_answer,
            "ground_truth": ground_truth,
            "explanation": f"Model predicted '{extracted_answer}', ground truth is '{ground_truth}'"
        }
    
    # Fall back to GPT-4o-mini if extraction fails
    evaluation_prompt = f"""{judge_prompt}

## YOUR TASK

**Question**: {question}

**Ground Truth Answer**: {ground_truth}

**Model Output**:
{model_output}

Please evaluate whether the model's assessment is correct. Respond with JSON only:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a binary assessment evaluator. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
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
    """Evaluate one model on one dataset for binary assessment"""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} | DATASET: {dataset_name} | BINARY ASSESSMENT")
    print(f"{'='*80}")

    # Load ALL questions
    questions = load_all_questions(dataset_name)
    print(f"Loaded {len(questions)} total questions from test set")
    
    # Print distribution
    sufficient_count = sum(1 for q in questions if q.get('is_sufficient', False))
    insufficient_count = len(questions) - sufficient_count
    print(f"  Sufficient: {sufficient_count}")
    print(f"  Insufficient: {insufficient_count}")

    if test_mode:
        questions = questions[:test_samples]
        print(f"TEST MODE: Using only {len(questions)} samples")

    # Check for checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoints',
                                   f"{model_name}_{dataset_name}_binary_checkpoint.json")

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

    # Load model
    print(f"\n[1/2] Loading model: {model_name}")
    solver = MathSolver(model_name, device=device, max_new_tokens=512, temperature=0.7)

    # Process each question
    print(f"\n[2/2] Generating assessments and judging...")

    for idx in tqdm(range(start_idx, len(questions)), desc="Evaluating"):
        item = questions[idx]
        question = item['question']
        is_sufficient = item.get('is_sufficient', False)
        ground_truth = 'Yes' if is_sufficient else 'No'

        # Generate binary assessment
        try:
            model_output, output_tokens = solver.assess(question)
        except Exception as e:
            print(f"\n  ⚠ Error generating assessment for Q{idx + 1}: {e}")
            model_output = f"[ERROR: {str(e)}]"
            output_tokens = 0

        # Judge assessment
        judgment = judge_binary_assessment(question, model_output, ground_truth, judge_prompt, openai_client)

        # Store result
        result = {
            'question_idx': idx,
            'question': question,
            'is_sufficient': is_sufficient,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'output_tokens': output_tokens,
            'judgment': judgment
        }
        results.append(result)

        # Save checkpoint (every 10 questions)
        if not test_mode and (len(results) % 10 == 0 or idx == len(questions) - 1):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'model': model_name,
                    'dataset': dataset_name,
                    'mode': 'binary_assessment',
                    'total_questions': len(questions),
                    'completed': len(results),
                    'results': results
                }, f, indent=2)

    # Compute accuracy metrics
    correct = sum(1 for r in results if r['judgment']['verdict'] == 'correct')
    incorrect = sum(1 for r in results if r['judgment']['verdict'] == 'incorrect')
    errors = sum(1 for r in results if r['judgment']['verdict'] == 'error')

    accuracy = correct / len(results) if len(results) > 0 else 0.0

    # Compute per-class metrics
    sufficient_results = [r for r in results if r['is_sufficient']]
    insufficient_results = [r for r in results if not r['is_sufficient']]
    
    sufficient_correct = sum(1 for r in sufficient_results if r['judgment']['verdict'] == 'correct')
    insufficient_correct = sum(1 for r in insufficient_results if r['judgment']['verdict'] == 'correct')
    
    sufficient_accuracy = sufficient_correct / len(sufficient_results) if sufficient_results else 0.0
    insufficient_accuracy = insufficient_correct / len(insufficient_results) if insufficient_results else 0.0

    # Compute simple efficiency metrics
    total_output_tokens = sum(r['output_tokens'] for r in results)
    avg_output_tokens = total_output_tokens / len(results) if results else 0
    tokens_per_correct = total_output_tokens / correct if correct > 0 else 0

    # Create final results
    final_results = {
        'model': model_name,
        'dataset': dataset_name,
        'mode': 'binary_assessment',
        'total_questions': len(results),
        'correct': correct,
        'incorrect': incorrect,
        'errors': errors,
        'accuracy': accuracy,
        'sufficient_accuracy': sufficient_accuracy,
        'insufficient_accuracy': insufficient_accuracy,
        'sufficient_count': len(sufficient_results),
        'insufficient_count': len(insufficient_results),
        # Simple efficiency metrics
        'total_output_tokens': total_output_tokens,
        'avg_output_tokens': avg_output_tokens,
        'tokens_per_correct': tokens_per_correct,
        'detailed_results': results
    }

    # Save final results
    if not test_mode:
        results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_binary_assessment.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✓ Saved results: {results_path}")

        # Remove checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} on {dataset_name} (Binary Assessment)")
    print(f"{'='*80}")
    print(f"Total questions: {len(results)}")
    print(f"Overall accuracy: {correct}/{len(results)} ({accuracy*100:.2f}%)")
    print(f"Sufficient accuracy: {sufficient_correct}/{len(sufficient_results)} ({sufficient_accuracy*100:.2f}%)")
    print(f"Insufficient accuracy: {insufficient_correct}/{len(insufficient_results)} ({insufficient_accuracy*100:.2f}%)")
    if errors > 0:
        print(f"Errors: {errors}")
    
    print(f"Efficiency: {total_output_tokens:,} tokens, {avg_output_tokens:.1f} avg, {tokens_per_correct:.1f} per correct")
    print(f"{'='*80}")

    return final_results


def load_results(model_name, dataset_name, output_dir):
    """Load existing results if available"""
    results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_binary_assessment.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def run_test_mode(device, openai_client, judge_prompt, test_samples=3):
    """Quick test mode: 1 model, 1 dataset, 3 samples with detailed output"""
    model_name = 'qwen2.5-math-1.5b'
    dataset_name = 'umwp'

    print("="*80)
    print("TEST MODE - Binary Assessment Evaluation")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} (all questions)")
    print(f"Samples: {test_samples}")
    print("="*80)

    # Load questions
    questions = load_all_questions(dataset_name)[:test_samples]

    # Load model
    print(f"\nLoading model...")
    solver = MathSolver(model_name, device=device, max_new_tokens=512, temperature=0.7)

    results = []

    for idx, item in enumerate(questions):
        print(f"\n{'='*80}")
        print(f"[{idx + 1}/{test_samples}]")
        print(f"{'='*80}")

        question = item['question']
        is_sufficient = item.get('is_sufficient', False)
        ground_truth = 'Yes' if is_sufficient else 'No'

        print(f"\nQuestion:")
        print(f"  {question}")
        print(f"\nIs Sufficient: {is_sufficient}")
        print(f"Ground Truth: {ground_truth}")

        # Generate assessment
        print(f"\nGenerating binary assessment...")
        model_output, output_tokens = solver.assess(question)
        print(f"\nModel Output:")
        print(f"  {model_output}")
        print(f"\nOutput tokens: {output_tokens}")

        # Judge
        print(f"\nJudging assessment...")
        judgment = judge_binary_assessment(question, model_output, ground_truth, judge_prompt, openai_client)
        print(f"\nJudgment:")
        print(f"  {json.dumps(judgment, indent=2)}")

        verdict_symbol = "✓" if judgment['verdict'] == 'correct' else "✗"
        print(f"\n{verdict_symbol} {judgment['verdict'].upper()}")

        results.append({
            'question': question,
            'is_sufficient': is_sufficient,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'output_tokens': output_tokens,
            'judgment': judgment
        })

    # Summary
    correct = sum(1 for r in results if r['judgment']['verdict'] == 'correct')
    total_tokens = sum(r['output_tokens'] for r in results)
    avg_tokens = total_tokens / len(results)
    
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Correct assessments: {correct}/{len(results)}")
    print(f"Assessment accuracy: {correct/len(results)*100:.1f}%")
    print(f"Total output tokens: {total_tokens}")
    print(f"Avg output tokens: {avg_tokens:.1f}")
    print(f"Tokens per correct: {total_tokens/correct if correct > 0 else 0:.1f}")
    print(f"{'='*80}")


def estimate_cost(all_models, all_datasets):
    """Estimate API cost for GPT-4o-mini"""
    total_questions = 0

    print("\n" + "="*80)
    print("COST ESTIMATION (GPT-4o-mini for Binary Assessment)")
    print("="*80)

    for dataset_name in all_datasets:
        questions = load_all_questions(dataset_name)
        print(f"{dataset_name}: {len(questions)} total questions")
        total_questions += len(questions)

    total_calls = total_questions * len(all_models)

    # GPT-4o-mini pricing
    input_cost_per_1m = 0.150
    output_cost_per_1m = 0.600

    # Estimate tokens (smaller for binary assessment)
    avg_input_tokens = 800   # Prompt + question + model output
    avg_output_tokens = 100   # JSON response

    total_input_tokens = total_calls * avg_input_tokens
    total_output_tokens = total_calls * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    print(f"\nTotal questions: {total_questions}")
    print(f"Models: {len(all_models)}")
    print(f"Total API calls: {total_calls}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print("="*80 + "\n")

    return total_cost


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1B: Direct Binary Assessment - Can this be solved?'
    )
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for model inference')
    parser.add_argument('--output_dir', type=str, default='experiments/binary_assessment',
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
    print("✓ Loaded binary assessment judge prompt")

    # Test mode
    if args.test:
        run_test_mode(args.device, openai_client, judge_prompt, args.test_samples)
        return

    # Full evaluation mode
    print("="*80)
    print("BINARY ASSESSMENT EVALUATION - ALL MODELS ON ALL DATASETS")
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
                'sufficient_accuracy': result['sufficient_accuracy'],
                'insufficient_accuracy': result['insufficient_accuracy'],
                'correct': result['correct'],
                'total': result['total_questions'],
                'avg_output_tokens': result['avg_output_tokens'],
                'tokens_per_correct': result['tokens_per_correct']
            }

        all_results[model_name] = model_results

    # Save summary
    summary_path = os.path.join(args.output_dir, 'binary_assessment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("BINARY ASSESSMENT SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            acc = all_results[model_name][dataset_name]['accuracy']
            row.append(f"{acc*100:>5.1f}%")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n{'='*80}")
    print("EFFICIENCY SUMMARY (Avg Output Tokens)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            tokens = all_results[model_name][dataset_name]['avg_output_tokens']
            row.append(f"{tokens:>8.1f}")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ Individual results saved in: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()