"""
Experiment 1C: Behavioral Evaluation - Three-valued outcomes

This script:
1. Loads ALL questions (sufficient + insufficient) from test sets
2. Prompts models with standard solving prompts (with explicit refusal permission)
3. Classifies responses as Correct/Incorrect/Refused
4. Saves per-sample results for correlation analysis
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
    """Load GPT-4o-mini judge prompt for behavioral evaluation"""
    prompt_path = Path(__file__).parent / "judge_prompts" / "gpt_behavioral_judge.md"
    
    if not prompt_path.exists():
        # Create default behavioral judge prompt if it doesn't exist
        default_prompt = """You are evaluating math problem solving behavior and correctness.

BEHAVIOR CLASSIFICATION:
- "attempted": Model provides mathematical work, calculations, or numerical answer (even if wrong)
- "refused": Model explicitly states it cannot solve, needs more information, or refuses to answer

CORRECTNESS EVALUATION (for attempts only):
- "correct": Final numerical answer matches ground truth
- "incorrect": Final numerical answer is wrong or unclear
- "n/a": For refusals (no answer to evaluate)

Return JSON with:
- "behavior": "attempted" or "refused"
- "correctness": "correct", "incorrect", or "n/a"
- "extracted_answer": numerical answer found or "refused"
- "explanation": brief reasoning"""
        
        os.makedirs(prompt_path.parent, exist_ok=True)
        with open(prompt_path, 'w') as f:
            f.write(default_prompt)
        print(f"✓ Created default behavioral judge prompt: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read()


def load_all_questions(dataset_name):
    """Load ALL questions from test set (both sufficient and insufficient)"""
    data_path = DATASETS[dataset_name]['test']

    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def classify_behavioral_outcome(model_output, ground_truth, is_sufficient, judge_prompt, openai_client):
    """
    Classify behavioral outcome into three categories: correct/incorrect/refused
    
    Returns:
        str: 'correct', 'incorrect', or 'refused'
    """
    output_lower = model_output.lower()
    
    # Strong refusal indicators
    refusal_patterns = [
        r'cannot.*be.*solved', r'cannot.*solve', r'insufficient.*information',
        r'not.*enough.*information', r'missing.*information', r'unable.*to.*solve',
        r'not.*solvable', r'impossible.*to.*solve', r'lack.*information',
        r'need.*more.*information', r'cannot.*determine'
    ]
    
    # Check for explicit refusal first
    for pattern in refusal_patterns:
        if re.search(pattern, output_lower):
            return 'refused'
    
    # Check for boxed answer (clear attempt)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_output)
    if boxed_match:
        if is_sufficient:
            # Use GPT-4o-mini to judge correctness for sufficient problems
            return judge_correctness_with_gpt(model_output, ground_truth, judge_prompt, openai_client)
        else:
            # Any attempt on insufficient problem is incorrect
            return 'incorrect'
    
    # Check for mathematical work without boxed answer
    has_calculations = bool(re.search(r'[\+\-\*/=]', model_output))
    has_numbers = bool(re.search(r'\d+', model_output))
    
    if has_calculations and has_numbers:
        # Model attempted but didn't follow format properly
        if is_sufficient:
            return judge_correctness_with_gpt(model_output, ground_truth, judge_prompt, openai_client)
        else:
            return 'incorrect'
    
    # No clear mathematical work or refusal - treat as refused
    return 'refused'


def judge_correctness_with_gpt(model_output, ground_truth, judge_prompt, openai_client):
    """Use GPT-4o-mini to judge correctness of attempted solutions"""
    evaluation_prompt = f"""{judge_prompt}

## YOUR TASK

**Ground Truth Answer**: {ground_truth}

**Model Output**:
{model_output}

The model attempted to solve. Please evaluate correctness. Respond with JSON only:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating math problem solving behavior and correctness. Respond only with valid JSON."
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
        return result.get('correctness', 'incorrect')

    except Exception as e:
        print(f"  ⚠ Error in GPT-4o-mini judgment: {e}")
        return 'incorrect'  # Default to incorrect on error


def evaluate_model_on_dataset(model_name, dataset_name, device, openai_client, judge_prompt,
                               output_dir, test_mode=False, test_samples=3):
    """Evaluate behavioral outcomes on ALL questions"""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} | DATASET: {dataset_name} | BEHAVIORAL EVALUATION")
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
                                   f"{model_name}_{dataset_name}_behavioral_checkpoint.json")

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
    print(f"\n[2/2] Generating solutions and classifying behavior...")

    for idx in tqdm(range(start_idx, len(questions)), desc="Evaluating"):
        item = questions[idx]
        question = item['question']
        is_sufficient = item.get('is_sufficient', False)
        ground_truth = item.get('answer', 'N/A')

        # Generate solution attempt
        try:
            model_output, output_tokens = solver.solve(question)
        except Exception as e:
            print(f"\n  ⚠ Error generating solution for Q{idx + 1}: {e}")
            model_output = f"[ERROR: {str(e)}]"
            output_tokens = 0

        # Classify behavioral outcome
        behavioral_outcome = classify_behavioral_outcome(
            model_output, ground_truth, is_sufficient, judge_prompt, openai_client
        )

        # Store result
        result = {
            'question_idx': idx,
            'question': question,
            'is_sufficient': is_sufficient,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'output_tokens': output_tokens,
            'behavioral_outcome': behavioral_outcome  # 'correct'/'incorrect'/'refused'
        }
        results.append(result)

        # Save checkpoint (every 10 questions)
        if not test_mode and (len(results) % 10 == 0 or idx == len(questions) - 1):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'model': model_name,
                    'dataset': dataset_name,
                    'mode': 'behavioral_evaluation',
                    'total_questions': len(questions),
                    'completed': len(results),
                    'results': results
                }, f, indent=2)

    # Compute behavioral metrics
    correct_count = sum(1 for r in results if r['behavioral_outcome'] == 'correct')
    incorrect_count = sum(1 for r in results if r['behavioral_outcome'] == 'incorrect')
    refused_count = sum(1 for r in results if r['behavioral_outcome'] == 'refused')
    
    total = len(results)
    correct_rate = correct_count / total if total > 0 else 0.0
    incorrect_rate = incorrect_count / total if total > 0 else 0.0
    refusal_rate = refused_count / total if total > 0 else 0.0

    # Compute by sufficiency type
    sufficient_results = [r for r in results if r['is_sufficient']]
    insufficient_results = [r for r in results if not r['is_sufficient']]
    
    # Sufficient questions
    sufficient_correct = sum(1 for r in sufficient_results if r['behavioral_outcome'] == 'correct')
    sufficient_incorrect = sum(1 for r in sufficient_results if r['behavioral_outcome'] == 'incorrect')
    sufficient_refused = sum(1 for r in sufficient_results if r['behavioral_outcome'] == 'refused')
    
    # Insufficient questions  
    insufficient_correct = sum(1 for r in insufficient_results if r['behavioral_outcome'] == 'correct')
    insufficient_incorrect = sum(1 for r in insufficient_results if r['behavioral_outcome'] == 'incorrect')
    insufficient_refused = sum(1 for r in insufficient_results if r['behavioral_outcome'] == 'refused')

    # Compute efficiency metrics
    total_output_tokens = sum(r['output_tokens'] for r in results)
    avg_output_tokens = total_output_tokens / len(results) if results else 0

    # Create final results
    final_results = {
        'model': model_name,
        'dataset': dataset_name,
        'mode': 'behavioral_evaluation',
        'total_questions': len(results),
        
        # Overall behavioral outcomes
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'refused_count': refused_count,
        'correct_rate': correct_rate,
        'incorrect_rate': incorrect_rate,
        'refusal_rate': refusal_rate,
        
        # By sufficiency type
        'sufficient_count': len(sufficient_results),
        'insufficient_count': len(insufficient_results),
        'sufficient_correct': sufficient_correct,
        'sufficient_incorrect': sufficient_incorrect,
        'sufficient_refused': sufficient_refused,
        'insufficient_correct': insufficient_correct,
        'insufficient_incorrect': insufficient_incorrect,
        'insufficient_refused': insufficient_refused,
        
        # Efficiency
        'avg_output_tokens': avg_output_tokens,
        
        # Detailed results for correlation analysis
        'detailed_results': results
    }

    # Save final results
    if not test_mode:
        results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_behavioral_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✓ Saved results: {results_path}")

        # Remove checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} on {dataset_name} (Behavioral Evaluation)")
    print(f"{'='*80}")
    print(f"Total questions: {len(results)}")
    print(f"\nOverall Behavioral Outcomes:")
    print(f"  Correct: {correct_count} ({correct_rate*100:.1f}%)")
    print(f"  Incorrect: {incorrect_count} ({incorrect_rate*100:.1f}%)")
    print(f"  Refused: {refused_count} ({refusal_rate*100:.1f}%)")
    print(f"\nBy Question Type:")
    print(f"  Sufficient Questions ({len(sufficient_results)}):")
    print(f"    Correct: {sufficient_correct}, Incorrect: {sufficient_incorrect}, Refused: {sufficient_refused}")
    print(f"  Insufficient Questions ({len(insufficient_results)}):")
    print(f"    Correct: {insufficient_correct}, Incorrect: {insufficient_incorrect}, Refused: {insufficient_refused}")
    print(f"\nEfficiency:")
    print(f"  Avg output tokens: {avg_output_tokens:.1f}")
    print(f"{'='*80}")

    return final_results


def load_results(model_name, dataset_name, output_dir):
    """Load existing results if available""" results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_behavioral_evaluation.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def run_test_mode(device, openai_client, judge_prompt, test_samples=3):
    """Quick test mode: 1 model, 1 dataset, 3 samples with detailed output"""
    model_name = 'qwen2.5-math-1.5b'
    dataset_name = 'umwp'

    print("="*80)
    print("TEST MODE - Behavioral Evaluation")
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
        ground_truth = item.get('answer', 'N/A')

        print(f"\nQuestion:")
        print(f"  {question}")
        print(f"\nIs Sufficient: {is_sufficient}")
        print(f"Ground Truth: {ground_truth}")

        # Generate solution
        print(f"\nGenerating solution...")
        model_output, output_tokens = solver.solve(question)
        print(f"\nModel Output:")
        print(f"  {model_output}")
        print(f"\nOutput tokens: {output_tokens}")

        # Classify behavior
        print(f"\nClassifying behavioral outcome...")
        behavioral_outcome = classify_behavioral_outcome(
            model_output, ground_truth, is_sufficient, judge_prompt, openai_client
        )
        
        print(f"\nBehavioral Outcome: {behavioral_outcome}")

        # Symbols for display
        outcome_symbols = {'correct': '✓', 'incorrect': '✗', 'refused': '🚫'}
        symbol = outcome_symbols.get(behavioral_outcome, '?')
        
        print(f"\n{symbol} {behavioral_outcome.upper()}")

        results.append({
            'question': question,
            'is_sufficient': is_sufficient,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'output_tokens': output_tokens,
            'behavioral_outcome': behavioral_outcome
        })

    # Summary
    correct = sum(1 for r in results if r['behavioral_outcome'] == 'correct')
    incorrect = sum(1 for r in results if r['behavioral_outcome'] == 'incorrect')
    refused = sum(1 for r in results if r['behavioral_outcome'] == 'refused')
    
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"Incorrect: {incorrect}/{len(results)} ({incorrect/len(results)*100:.1f}%)")
    print(f"Refused: {refused}/{len(results)} ({refused/len(results)*100:.1f}%)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 1C: Behavioral Evaluation - Three-valued outcomes'
    )
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for model inference')
    parser.add_argument('--output_dir', type=str, default='experiments/behavioral_evaluation',
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
    print("✓ Loaded behavioral evaluation judge prompt")

    # Test mode
    if args.test:
        run_test_mode(args.device, openai_client, judge_prompt, args.test_samples)
        return

    # Full evaluation mode
    print("="*80)
    print("BEHAVIORAL EVALUATION - ALL MODELS ON ALL DATASETS")
    print("="*80)
    print(f"Models: {MODELS}")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Device: {args.device}")
    print("="*80)

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
                'correct_rate': result['correct_rate'],
                'incorrect_rate': result['incorrect_rate'],
                'refusal_rate': result['refusal_rate'],
                'sufficient_correct': result['sufficient_correct'],
                'insufficient_refused': result['insufficient_refused'],
                'avg_output_tokens': result['avg_output_tokens']
            }

        all_results[model_name] = model_results

    # Save summary
    summary_path = os.path.join(args.output_dir, 'behavioral_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("BEHAVIORAL EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            rate = all_results[model_name][dataset_name]['correct_rate']
            row.append(f"{rate*100:>5.1f}%")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n{'='*80}")
    print("REFUSAL ANALYSIS (Refusal Rate)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'UMWP':<12} {'GSM8K':<12} {'TreeCut':<12}")
    print("-" * 65)

    for model_name in MODELS:
        row = [model_name[:24]]
        for dataset_name in ['umwp', 'gsm8k', 'treecut']:
            rate = all_results[model_name][dataset_name]['refusal_rate']
            row.append(f"{rate*100:>5.1f}%")
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12}")

    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ Individual results saved in: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()