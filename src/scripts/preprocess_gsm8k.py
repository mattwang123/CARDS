"""
Download and preprocess GSM8K dataset using GPT-4o for sophisticated insufficient variants

GSM8K: Grade School Math 8K dataset
Dataset: https://huggingface.co/datasets/openai/gsm8k

This script uses GPT-4o with a sophisticated prompt to create natural-sounding
insufficient variants that are NOT obviously artificial (no crude "some apples" substitutions).

See INSUFFICIENT_GENERATION_PROMPT.md for the full methodology.
"""
import argparse
import json
import os
import random
import time
from datasets import load_dataset
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


def load_prompt_guide():
    """Load the sophisticated prompt guide from markdown file"""
    guide_path = Path(__file__).parent / "INSUFFICIENT_GENERATION_PROMPT.md"

    if guide_path.exists():
        with open(guide_path, 'r') as f:
            return f.read()
    else:
        # Fallback to embedded version if file not found
        return """
HIGH-QUALITY INSUFFICIENT PROBLEM CREATION GUIDE

CRITICAL: Do NOT create obviously broken questions with crude substitutions like "12 apples" → "some apples".

Instead, use these sophisticated strategies:

1. AMBIGUOUS REFERENCE: Make references unclear without crude substitution
   - "A rectangle has one side of 10cm and another side of 5cm" (which are adjacent?)
   - "Sarah spent money on notebooks and pens" (not "some money")

2. UNDERSPECIFIED SYSTEM: Create mathematically valid but unsolvable scenarios
   - "Tom has twice as many apples as Jerry. Tom has apples." (ratio but no anchor)

3. REMOVED RELATIONSHIP: Keep numbers but remove how they relate
   - "Class A has 25 students. Class B has a different number." (not "some students")

4. CONTEXT-DEPENDENT AMBIGUITY: Missing context makes it unsolvable
   - "A store marks up items from wholesale" (not "by some percent")

QUALITY CHECKLIST:
- Does it sound natural? Could this be in a real textbook?
- Is grammar perfect?
- Is it genuinely unsolvable (not just harder)?
- Are there multiple valid answers?
- Is the insufficiency SUBTLE?
- Would a student try solving before realizing info is missing?
"""


def get_gpt_prompt(question, guide_text):
    """
    Create sophisticated prompt for GPT-4o using the detailed guide

    Args:
        question: Original GSM8K math question
        guide_text: Full text from INSUFFICIENT_GENERATION_PROMPT.md

    Returns:
        str: Comprehensive prompt for GPT-4o
    """
    prompt = f"""You are an expert at creating high-quality logically insufficient math problems for academic research.

IMPORTANT: Read the following guide carefully. Your task is to create a sophisticated insufficient variant that follows these principles:

{guide_text}

---

NOW YOUR TASK:

Transform the following GSM8K question into a logically insufficient version using the strategies above.

ORIGINAL QUESTION:
{question}

REQUIREMENTS:
1. Use ONE of the sophisticated strategies (Ambiguous Reference, Underspecified System, Removed Relationship, Context-Dependent Ambiguity)
2. Make it sound NATURAL - like a real textbook question with incomplete information
3. NO crude substitutions like "some", "a certain number", "X" - be creative!
4. The question should sound professional and grammatically perfect
5. A student should try solving it before realizing information is missing
6. Preserve all context and relationships that remain

Think step-by-step:
1. What is the critical missing information that makes this unsolvable?
2. How can I remove it WITHOUT making it obvious?
3. What natural language makes this sound like incomplete information gathering?
4. Does this read like a real question someone might ask?

OUTPUT FORMAT (JSON only, no markdown):
{{
  "insufficient_question": "The transformed question - must sound natural and professional",
  "removed_value": "What was removed - be specific",
  "removed_description": "Brief explanation: why this makes the problem unsolvable"
}}"""

    return prompt


def create_insufficient_version(question, client, guide_text, max_retries=3):
    """
    Call GPT-4o to create sophisticated insufficient version

    Args:
        question: Original GSM8K question text
        client: OpenAI client instance
        guide_text: Full prompt guide text
        max_retries: Number of retries on failure

    Returns:
        dict: Contains insufficient_question, removed_value, removed_description
        None: If all retries fail
    """
    prompt = get_gpt_prompt(question, guide_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert research assistant specializing in creating natural-sounding logically insufficient math problems.

Your transformations must be SUBTLE and SOPHISTICATED - never use crude substitutions like "some apples" or "a certain number".

Think like a textbook author who accidentally omitted critical information, not like someone doing find-replace on numbers.

Always return valid JSON without markdown formatting."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.8,  # Slightly higher for creativity
                max_tokens=700
            )

            result = json.loads(response.choices[0].message.content)

            # Validate required fields
            required_fields = ['insufficient_question', 'removed_value', 'removed_description']
            if all(field in result for field in required_fields):
                insuff_q = result['insufficient_question'].lower()
                return result
            else:
                print(f"  ⚠ Missing fields in response. Attempt {attempt + 1}/{max_retries}")
                continue

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON decode error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            print(f"  ⚠ Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    print(f"  ✗ Failed to process question after {max_retries} attempts")
    return None


def download_gsm8k(cache_dir=None):
    """Download GSM8K dataset from HuggingFace"""
    print("Downloading GSM8K from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Test examples: {len(dataset['test'])}")
    return dataset


def parse_gsm8k_example(example: Dict) -> Dict:
    """
    Parse GSM8K example to extract question and answer

    GSM8K format:
    - question: the math word problem
    - answer: step-by-step solution ending with "#### {answer}"
    """
    question = example['question']
    answer_text = example['answer']

    # Extract final answer (format: "#### 123")
    if '####' in answer_text:
        final_answer = answer_text.split('####')[-1].strip()
    else:
        final_answer = answer_text.strip()

    return {
        'question': question,
        'answer': f"#### {final_answer}"
    }


def create_balanced_split(
    dataset_split,
    client,
    guide_text,
    num_samples=None,
    insufficient_ratio=0.5,
    random_seed=42,
    use_api=True
):
    """
    Create a balanced dataset from GSM8K split with sophisticated insufficient variants

    Args:
        dataset_split: HuggingFace dataset split
        client: OpenAI client (required if use_api=True)
        guide_text: Full prompt guide text
        num_samples: Total number of samples (None = use all)
        insufficient_ratio: Ratio of insufficient examples
        random_seed: Random seed
        use_api: If False, skip API calls (for testing)
    """
    random.seed(random_seed)

    # Convert to list and shuffle
    data = list(dataset_split)
    random.shuffle(data)

    # Limit samples if specified
    if num_samples is not None:
        data = data[:num_samples]

    num_insufficient = int(len(data) * insufficient_ratio)
    num_sufficient = len(data) - num_insufficient

    print(f"\nCreating balanced split:")
    print(f"  Total samples: {len(data)}")
    print(f"  Sufficient: {num_sufficient}")
    print(f"  Insufficient: {num_insufficient}")

    # Process sufficient examples
    print("\nProcessing sufficient examples...")
    sufficient_examples = []
    for example in tqdm(data[:num_sufficient], desc="Sufficient"):
        parsed = parse_gsm8k_example(example)
        sufficient_examples.append({
            'question': parsed['question'],
            'answer': parsed['answer'],
            'is_sufficient': True
        })

    # Process insufficient examples with GPT-4o
    print(f"\n{'Processing insufficient examples with GPT-4o (sophisticated transformations)...' if use_api else 'Creating test insufficient examples...'}")
    insufficient_examples = []
    successful = 0
    failed = 0

    for example in tqdm(data[num_sufficient:], desc="Insufficient"):
        parsed = parse_gsm8k_example(example)

        if use_api:
            result = create_insufficient_version(parsed['question'], client, guide_text)

            if result:
                insufficient_examples.append({
                    'question': result['insufficient_question'],
                    'answer': 'N/A',
                    'is_sufficient': False,
                    'original_question': parsed['question'],
                    'removed_value': result['removed_value'],
                    'removed_description': result['removed_description']
                })
                successful += 1
            else:
                # If GPT fails after retries, keep as sufficient
                sufficient_examples.append({
                    'question': parsed['question'],
                    'answer': parsed['answer'],
                    'is_sufficient': True
                })
                failed += 1

            # Rate limiting
            time.sleep(0.6)
        else:
            # Test mode: placeholder
            insufficient_examples.append({
                'question': parsed['question'],
                'answer': 'N/A',
                'is_sufficient': False,
                'original_question': parsed['question'],
                'removed_value': 'TEST_MODE',
                'removed_description': 'Test mode - no API transformation'
            })

    if use_api:
        print(f"\nAPI Processing Results:")
        print(f"  ✓ Successfully created: {successful}")
        print(f"  ✗ Failed (kept as sufficient): {failed}")

    print(f"\nFinal counts:")
    print(f"  Sufficient: {len(sufficient_examples)}")
    print(f"  Insufficient: {len(insufficient_examples)}")

    # Combine and shuffle
    combined = sufficient_examples + insufficient_examples
    random.shuffle(combined)

    return combined


def save_dataset(data: List[Dict], output_path: str):
    """Save dataset to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"  Total examples: {len(data)}")
    print(f"  Sufficient: {sum(1 for x in data if x['is_sufficient'])}")
    print(f"  Insufficient: {sum(1 for x in data if not x['is_sufficient'])}")


def estimate_api_cost(num_insufficient_examples):
    """Estimate GPT-4o API cost"""
    # GPT-4o pricing (Jan 2025)
    input_cost_per_1m = 2.50
    output_cost_per_1m = 10.00

    # With full guide text, prompts are longer
    avg_input_tokens = 1200   # Guide + question + instructions
    avg_output_tokens = 200   # Response

    total_input_tokens = num_insufficient_examples * avg_input_tokens
    total_output_tokens = num_insufficient_examples * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost

    print("\n" + "="*80)
    print("ESTIMATED GPT-4o API COST")
    print("="*80)
    print(f"Insufficient examples requiring API calls: {num_insufficient_examples}")
    print(f"Estimated input tokens:  {total_input_tokens:,}")
    print(f"Estimated output tokens: {total_output_tokens:,}")
    print(f"Estimated total cost: ${total_cost:.2f}")
    print("="*80 + "\n")

    return total_cost


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess GSM8K with sophisticated insufficient variants via GPT-4o'
    )
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/gsm8k',
                        help='Directory to save processed data')
    parser.add_argument('--train_samples', type=int, default=4160,
                        help='Number of training samples (default: 4160)')
    parser.add_argument('--test_samples', type=int, default=1040,
                        help='Number of test samples (default: 1040)')
    parser.add_argument('--insufficient_ratio', type=float, default=0.5,
                        help='Ratio of insufficient examples (default: 0.5)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for HuggingFace datasets')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode: 10 examples, no API calls')
    parser.add_argument('--skip_confirmation', action='store_true',
                        help='Skip cost confirmation prompt')

    args = parser.parse_args()

    print("="*80)
    print("GSM8K PREPROCESSING WITH SOPHISTICATED INSUFFICIENT VARIANTS")
    print("="*80)
    print("\nUsing advanced GPT-4o prompting for natural-sounding transformations")
    print("(No crude 'some apples' substitutions!)")
    print("="*80)

    # Load the sophisticated prompt guide
    guide_text = load_prompt_guide()
    print(f"\n✓ Loaded prompt guide ({len(guide_text)} chars)")

    # Test mode setup
    if args.test_mode:
        print("\n*** TEST MODE: 10 examples, no API calls ***\n")
        args.train_samples = 10
        args.test_samples = 10
        use_api = False
        client = None
    else:
        use_api = True

        # Load OpenAI API key
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            print("\n✗ Error: OPENAI_API_KEY not found")
            print("Set it in .env file or environment:")
            print("  export OPENAI_API_KEY='your_key_here'")
            return

        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client initialized")

        # Estimate and confirm costs
        num_insufficient_train = int(args.train_samples * args.insufficient_ratio)
        num_insufficient_test = int(args.test_samples * args.insufficient_ratio)
        total_insufficient = num_insufficient_train + num_insufficient_test

        estimated_cost = estimate_api_cost(total_insufficient)

        if not args.skip_confirmation:
            response = input("Continue with processing? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return

    # Download dataset
    dataset = download_gsm8k(cache_dir=args.cache_dir)

    # Process train split
    print("\n" + "="*80)
    print("PROCESSING TRAIN SPLIT")
    print("="*80)
    train_data = create_balanced_split(
        dataset['train'],
        client,
        guide_text,
        num_samples=args.train_samples,
        insufficient_ratio=args.insufficient_ratio,
        random_seed=args.random_seed,
        use_api=use_api
    )

    # Process test split
    print("\n" + "="*80)
    print("PROCESSING TEST SPLIT")
    print("="*80)
    test_data = create_balanced_split(
        dataset['test'],
        client,
        guide_text,
        num_samples=args.test_samples,
        insufficient_ratio=args.insufficient_ratio,
        random_seed=args.random_seed + 1,
        use_api=use_api
    )

    # Save datasets
    print("\n" + "="*80)
    print("SAVING DATASETS")
    print("="*80)

    train_path = os.path.join(args.output_dir, 'gsm8k_train.json')
    test_path = os.path.join(args.output_dir, 'gsm8k_test.json')

    save_dataset(train_data, train_path)
    save_dataset(test_data, test_path)

    # Print examples
    if train_data:
        print("\n" + "="*80)
        print("EXAMPLE SUFFICIENT:")
        print("="*80)
        sufficient_examples = [x for x in train_data if x['is_sufficient']]
        if sufficient_examples:
            ex = sufficient_examples[0]
            print(json.dumps(ex, indent=2))

        print("\n" + "="*80)
        print("EXAMPLE INSUFFICIENT TRANSFORMATION:")
        print("="*80)
        insufficient_examples = [x for x in train_data if not x['is_sufficient']]
        if insufficient_examples:
            ex = insufficient_examples[0]

            if 'original_question' in ex and ex['original_question'] != ex['question']:
                print("ORIGINAL:")
                print(f"  {ex['original_question']}\n")
                print("TRANSFORMED (INSUFFICIENT):")
                print(f"  {ex['question']}\n")
                print("WHAT WAS REMOVED:")
                print(f"  Value: {ex['removed_value']}")
                print(f"  Why: {ex['removed_description']}")
            else:
                print(json.dumps(ex, indent=2))
        print("="*80)

    print("\n✓ GSM8K preprocessing complete!")
    print("\nRemember: These are SOPHISTICATED transformations, not crude substitutions!")


if __name__ == '__main__':
    main()
