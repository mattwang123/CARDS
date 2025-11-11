"""
Create insufficient dataset by using GPT-4o to remove critical numerical values
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


def get_gpt_prompt(question):
    """
    Create detailed prompt for GPT-4o to generate insufficient version

    Args:
        question: Original math question

    Returns:
        str: Prompt for GPT-4o
    """
    prompt = f"""You are a research assistant helping to create a dataset for studying logical insufficiency in mathematical reasoning problems.

DEFINITION OF LOGICAL INSUFFICIENCY:
A math problem is "logically insufficient" when it lacks a critical piece of information needed to arrive at a unique, definitive answer. This is NOT about ambiguous language or unclear questions - it's about missing concrete numerical values or constraints that are mathematically necessary to solve the problem.

ORIGINAL QUESTION:
{question}

YOUR TASK:
Transform this question into a logically insufficient version by removing EXACTLY ONE critical numerical value.

DETAILED REQUIREMENTS:

1. WHAT TO REMOVE:
   - Remove ONE number that is essential for solving the problem
   - The removed number should be one that, without it, multiple valid answers become possible
   - Choose a number whose absence makes the problem genuinely unsolvable, not just harder

2. HOW TO REPLACE IT:
   - Replace the removed number with natural language like:
     * "some" (e.g., "12 tickets" → "some tickets")
     * "several" (e.g., "15 students" → "several students")
     * "a certain amount" (e.g., "$30" → "a certain amount")
     * "a certain number of" (e.g., "8 hours" → "a certain number of hours")
   - The replacement should sound natural in context
   - Do NOT use phrases like "X" or "an unknown value" - be conversational

3. PRESERVE EVERYTHING ELSE:
   - Keep ALL other numbers unchanged
   - Keep ALL other text EXACTLY as written
   - Maintain perfect grammatical correctness
   - The question should read naturally, as if it were always written this way

4. QUALITY CHECKS:
   - The insufficient question should look professionally written
   - Someone reading it should not immediately notice it's "broken"
   - But mathematically, it should be impossible to solve uniquely
   - All other mathematical relationships and operations should remain intact

EXAMPLE TRANSFORMATION:
Original: "A concert ticket costs $40. Mr. Benson bought 12 tickets and received a 5% discount for every ticket bought that exceeds 10. How much did Mr. Benson pay in all?"

Insufficient: "A concert ticket costs $40. Mr. Benson bought some tickets and received a 5% discount for every ticket bought that exceeds 10. How much did Mr. Benson pay in all?"

Removed value: "12"
Description: "number of tickets bought"

Why this works: Without knowing the exact number of tickets, we cannot calculate the final cost, even though we know the price per ticket and discount structure.

OUTPUT FORMAT:
Return ONLY a valid JSON object (no markdown, no code blocks, no extra text) with these exact fields:
{{
  "insufficient_question": "the modified question text",
  "removed_value": "the exact number or value you removed as a string",
  "removed_description": "a brief description of what this value represents (e.g., 'number of tickets', 'hours worked', 'price per item')"
}}

Now generate the insufficient version:"""

    return prompt


def create_insufficient_version(question, client, max_retries=3):
    """
    Call GPT-4o to create insufficient version of the question

    Args:
        question: Original question text
        client: OpenAI client instance
        max_retries: Number of retries on failure

    Returns:
        dict: Contains insufficient_question, removed_value, removed_description
        None: If all retries fail
    """
    prompt = get_gpt_prompt(question)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise research assistant that creates logically insufficient math problems. You always return valid JSON without markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)

            # Validate required fields
            required_fields = ['insufficient_question', 'removed_value', 'removed_description']
            if all(field in result for field in required_fields):
                return result
            else:
                print(f"Warning: Missing fields in response. Attempt {attempt + 1}/{max_retries}")
                continue

        except json.JSONDecodeError as e:
            print(f"JSON decode error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    print(f"Failed to process question after {max_retries} attempts: {question[:100]}...")
    return None


def process_dataset(input_file, output_dir, client, insufficient_ratio=0.5, test_mode=False, random_seed=42):
    """
    Process dataset to create sufficient/insufficient split

    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output
        client: OpenAI client
        insufficient_ratio: Ratio of insufficient examples (default 0.5)
        test_mode: If True, only process 10 examples
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)

    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    original_count = len(data)
    print(f"Loaded {original_count} examples")

    # Test mode: only take 10 examples
    if test_mode:
        print("\n*** TEST MODE: Processing only 10 examples ***")
        data = data[:10]

    # Shuffle data
    random.shuffle(data)

    # Split into sufficient and insufficient
    num_insufficient = int(len(data) * insufficient_ratio)
    num_sufficient = len(data) - num_insufficient

    print(f"\nDataset split:")
    print(f"  Sufficient: {num_sufficient} ({(1-insufficient_ratio)*100:.0f}%)")
    print(f"  Insufficient: {num_insufficient} ({insufficient_ratio*100:.0f}%)")

    # Process sufficient examples (no modification needed)
    print("\nProcessing sufficient examples...")
    sufficient_data = []
    for item in tqdm(data[:num_sufficient], desc="Sufficient"):
        sufficient_data.append({
            'question': item['question'],
            'answer': item['answer'],
            'is_sufficient': True
        })

    # Process insufficient examples (use GPT-4o)
    print("\nProcessing insufficient examples with GPT-4o...")
    insufficient_data = []
    successful = 0
    failed = 0

    for item in tqdm(data[num_sufficient:], desc="Insufficient"):
        result = create_insufficient_version(item['question'], client)

        if result:
            insufficient_data.append({
                'question': result['insufficient_question'],
                'answer': item['answer'],  # Keep original answer for reference
                'is_sufficient': False,
                'original_question': item['question'],
                'removed_value': result['removed_value'],
                'removed_description': result['removed_description']
            })
            successful += 1
        else:
            # If GPT fails, keep original as sufficient
            sufficient_data.append({
                'question': item['question'],
                'answer': item['answer'],
                'is_sufficient': True
            })
            failed += 1

        # Rate limiting: small delay between requests
        time.sleep(0.5)

    # Combine and shuffle
    combined_data = sufficient_data + insufficient_data
    random.shuffle(combined_data)

    print(f"\nProcessing complete:")
    print(f"  Successfully created insufficient: {successful}")
    print(f"  Failed (kept as sufficient): {failed}")
    print(f"  Total examples: {len(combined_data)}")

    # Save output
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename based on input
    input_basename = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{input_basename}_insufficient.json")

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Print example insufficient case
    insufficient_examples = [ex for ex in combined_data if not ex['is_sufficient']]
    if insufficient_examples:
        print("\n" + "="*80)
        print("EXAMPLE INSUFFICIENT TRANSFORMATION:")
        print("="*80)
        example = insufficient_examples[0]
        print(f"\nOriginal: {example['original_question']}")
        print(f"\nInsufficient: {example['question']}")
        print(f"\nRemoved value: {example['removed_value']}")
        print(f"Description: {example['removed_description']}")
        print("="*80)

    return output_file


def estimate_cost(num_examples, insufficient_ratio=0.5):
    """
    Estimate API cost for processing

    Args:
        num_examples: Number of examples to process
        insufficient_ratio: Ratio that will be insufficient
    """
    num_insufficient = int(num_examples * insufficient_ratio)

    # GPT-4o pricing (as of 2024)
    input_cost_per_1m = 2.50  # $2.50 per 1M input tokens
    output_cost_per_1m = 10.00  # $10.00 per 1M output tokens

    # Average tokens per request
    avg_input_tokens = 500  # Prompt + question
    avg_output_tokens = 150  # Response

    # Calculate cost
    total_input_tokens = num_insufficient * avg_input_tokens
    total_output_tokens = num_insufficient * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    print("\n" + "="*60)
    print("ESTIMATED API COST (GPT-4o)")
    print("="*60)
    print(f"Total examples: {num_examples}")
    print(f"Insufficient examples (requiring API calls): {num_insufficient}")
    print(f"Estimated input tokens: {total_input_tokens:,}")
    print(f"Estimated output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create insufficient dataset using GPT-4o'
    )
    parser.add_argument('--train_file', type=str, default='data/raw/gsm8k_train.json',
                        help='Path to training data JSON file')
    parser.add_argument('--test_file', type=str, default='data/raw/gsm8k_test.json',
                        help='Path to test data JSON file')
    parser.add_argument('--output_dir', type=str, default='data/insufficient_dataset',
                        help='Directory to save output')
    parser.add_argument('--insufficient_ratio', type=float, default=0.5,
                        help='Ratio of insufficient examples (default: 0.5)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode: only process 10 examples')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip processing training data')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip processing test data')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    print("="*80)
    print("INSUFFICIENT DATASET CREATION")
    print("="*80)

    # Estimate costs
    if not args.test_mode:
        # Load files to count
        if not args.skip_train and os.path.exists(args.train_file):
            with open(args.train_file, 'r') as f:
                train_count = len(json.load(f))
            estimate_cost(train_count, args.insufficient_ratio)

        if not args.skip_test and os.path.exists(args.test_file):
            with open(args.test_file, 'r') as f:
                test_count = len(json.load(f))
            estimate_cost(test_count, args.insufficient_ratio)

        response = input("Continue with processing? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Process training data
    if not args.skip_train:
        if os.path.exists(args.train_file):
            print(f"\n{'='*80}")
            print("PROCESSING TRAINING DATA")
            print('='*80)
            process_dataset(
                args.train_file,
                args.output_dir,
                client,
                args.insufficient_ratio,
                args.test_mode,
                args.random_seed
            )
        else:
            print(f"Warning: Training file not found: {args.train_file}")

    # Process test data
    if not args.skip_test:
        if os.path.exists(args.test_file):
            print(f"\n{'='*80}")
            print("PROCESSING TEST DATA")
            print('='*80)
            process_dataset(
                args.test_file,
                args.output_dir,
                client,
                args.insufficient_ratio,
                args.test_mode,
                args.random_seed
            )
        else:
            print(f"Warning: Test file not found: {args.test_file}")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
