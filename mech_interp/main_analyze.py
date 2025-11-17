"""
Main script to analyze model responses on insufficient dataset
"""
import argparse
import json
import os
import sys
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.inference import MathSolver
from models.answer_parser import extract_numerical_answer, extract_binary_answer
from viz.analyze_responses import analyze_responses, print_analysis_report, save_analysis_report


def load_dataset(data_path, max_examples=None):
    """Load insufficient dataset"""
    print(f"Loading dataset from {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)

    original_size = len(data)

    if max_examples is not None and max_examples < len(data):
        print(f"Loaded {original_size} examples, limiting to {max_examples}")
        data = data[:max_examples]
    else:
        print(f"Loaded {len(data)} examples")

    return data


def run_model_inference(solver, dataset, mode='solve'):
    """
    Run model on all examples and collect responses
    
    Args:
        solver: MathSolver instance
        dataset: List of examples
        mode: 'solve' or 'assess'
    """
    print(f"\nRunning model inference in {mode} mode...")
    
    responses_data = []
    
    for item in tqdm(dataset):
        print(f"\nProcessing: {item['question']}...")

        # Choose method based on mode
        if mode == 'solve':
            response = solver.solve(item['question'])
            model_answer = extract_numerical_answer(response)
        elif mode == 'assess':
            response = solver.assess(item['question'])
            model_answer = extract_binary_answer(response)
        
        # Rest stays exactly the same...
        result = {
            'question': item['question'],
            'answer': item['answer'],
            'is_sufficient': item['is_sufficient'],
            'response': response
        }
        
        if not item['is_sufficient']:
            result['original_question'] = item.get('original_question', '')
            result['removed_value'] = item.get('removed_value', '')
            result['removed_description'] = item.get('removed_description', '')

        responses_data.append(result)
        
        print(f"  Model answer: {model_answer}")
        print(f"  Response preview: {response}")

    return responses_data


def save_responses(responses_data, output_path):
    """Save all responses to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(responses_data, f, indent=2)
    print(f"\nResponses saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze model responses on insufficient dataset'
    )
    parser.add_argument('--model_name', type=str, default='llama-3.2-3b-instruct',
                        help='Model name from config (default: llama-3.2-3b-instruct)')
    parser.add_argument('--data_path', type=str,
                        default='data/insufficient_dataset/gsm8k_train_insufficient.json',
                        help='Path to insufficient dataset')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum tokens to generate (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples to analyze (default: None = all)')
    parser.add_argument('--mode', type=str, default='solve', 
                    choices=['solve', 'assess'],
                    help='Analysis mode: solve (default) or assess (binary assessment)')

    args = parser.parse_args()

    print("="*80)
    print("MODEL RESPONSE ANALYSIS PIPELINE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.data_path, args.max_examples)

    # Initialize model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    solver = MathSolver(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Run inference
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    responses_data = run_model_inference(solver, dataset, args.mode)

    # Save responses
    output_basename = Path(args.data_path).stem
    responses_path = os.path.join(args.output_dir, f'{output_basename}_responses.json')
    save_responses(responses_data, responses_path)

    # Analyze responses
    print("\n" + "="*80)
    print("ANALYZING RESPONSES")
    print("="*80)
    analysis = analyze_responses(responses_data)

    # Print report
    print_analysis_report(analysis)

    # Save report
    report_path = os.path.join(args.output_dir, f'{output_basename}_analysis.txt')
    save_analysis_report(analysis, report_path)

    # Save detailed analysis as JSON
    analysis_json_path = os.path.join(args.output_dir, f'{output_basename}_analysis.json')
    with open(analysis_json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Detailed analysis saved to: {analysis_json_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {responses_path}")
    print(f"  - {report_path}")
    print(f"  - {analysis_json_path}")
    print()


if __name__ == '__main__':
    main()
