"""
Download and preprocess TreeCut dataset

TreeCut: A Synthetic Unanswerable Math Word Problem Dataset
Dataset: https://huggingface.co/datasets/jouyang/treecut-math

This script:
1. Downloads TreeCut from HuggingFace
2. Reformats to match our standard JSON structure
3. Creates train/test splits with balanced sufficient/insufficient examples
"""
import argparse
import json
import os
import random
from datasets import load_dataset
from typing import List, Dict


def download_treecut(cache_dir=None):
    """Download TreeCut dataset from HuggingFace"""
    print("Downloading TreeCut from HuggingFace...")
    dataset = load_dataset("jouyang/treecut-math", cache_dir=cache_dir)
    print(f"  Available splits: {list(dataset.keys())}")
    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])} examples")
    return dataset


def is_answerable(example: Dict) -> bool:
    """
    Determine if a TreeCut example is answerable

    TreeCut uses:
    - answer: "unknown" for unanswerable
    - answer: numerical value for answerable
    """
    answer = example.get('answer', '')

    # Check if answer is "unknown" or empty
    if isinstance(answer, str) and (answer.lower() == 'unknown' or answer == ''):
        return False

    # Check proof text for unanswerability indicators
    proof = example.get('proof', '')
    if isinstance(proof, str):
        unanswerable_indicators = [
            'cannot calculate',
            'cannot determine',
            'not enough information',
            'insufficient information',
            'variables but only',
            'underdetermined'
        ]
        if any(indicator in proof.lower() for indicator in unanswerable_indicators):
            return False

    return True


def reformat_treecut_example(example: Dict, is_answerable_flag: bool) -> Dict:
    """
    Reformat a TreeCut example to our standard format

    TreeCut format:
    - problem: the math word problem
    - answer: numerical answer (for answerable) or "unknown" (for unanswerable)
    - proof: explanation/proof text

    Args:
        example: TreeCut example dict
        is_answerable_flag: Whether this is an answerable example

    Returns:
        Reformatted dict matching our standard format
    """
    problem = example.get('problem', '')
    answer = example.get('answer', 'unknown')
    proof = example.get('proof', '')

    if is_answerable_flag:
        # Sufficient example
        answer_value = answer

        # Convert to int if whole number
        try:
            if isinstance(answer_value, str):
                answer_float = float(answer_value)
                if answer_float.is_integer():
                    answer_value = int(answer_float)
                else:
                    answer_value = answer_float
        except (ValueError, TypeError):
            pass

        return {
            'question': problem,
            'answer': f"#### {answer_value}",
            'is_sufficient': True
        }
    else:
        # Insufficient example
        # Try to extract what makes it unanswerable from proof
        removed_description = "TreeCut synthetic: "
        if "variables but only" in proof.lower():
            # Extract the key information about missing constraints
            import re
            match = re.search(r'(\d+)\s+variables?\s+but\s+only\s+(\d+)', proof, re.IGNORECASE)
            if match:
                num_vars = match.group(1)
                num_eqs = match.group(2)
                removed_description += f"Underdetermined system: {num_vars} variables, {num_eqs} equations"
            else:
                removed_description += "Underdetermined system of equations"
        else:
            removed_description += "Insufficient constraints to solve uniquely"

        return {
            'question': problem,
            'answer': 'N/A',
            'is_sufficient': False,
            'original_question': 'N/A',  # TreeCut doesn't provide original
            'removed_value': 'N/A',
            'removed_description': removed_description
        }


def create_balanced_split(dataset_split, num_samples=None, insufficient_ratio=0.5, random_seed=42):
    """
    Create a balanced dataset from TreeCut split

    Args:
        dataset_split: HuggingFace dataset split or list of examples
        num_samples: Total number of samples (None = use all)
        insufficient_ratio: Ratio of insufficient examples
        random_seed: Random seed
    """
    random.seed(random_seed)

    if num_samples is None:
        num_samples = len(dataset_split)

    num_insufficient = int(num_samples * insufficient_ratio)
    num_sufficient = num_samples - num_insufficient

    print(f"\nCreating balanced split:")
    print(f"  Total samples: {num_samples}")
    print(f"  Sufficient: {num_sufficient}")
    print(f"  Insufficient: {num_insufficient}")

    # Separate answerable and unanswerable examples
    answerable = []
    unanswerable = []

    for idx, example in enumerate(dataset_split):
        if is_answerable(example):
            answerable.append((idx, example))
        else:
            unanswerable.append((idx, example))

    print(f"\nAvailable in source:")
    print(f"  Answerable: {len(answerable)}")
    print(f"  Unanswerable: {len(unanswerable)}")

    # Sample sufficient examples
    random.shuffle(answerable)
    sufficient_examples = []
    for i in range(min(num_sufficient, len(answerable))):
        idx, example = answerable[i]
        reformatted = reformat_treecut_example(example, is_answerable_flag=True)
        if reformatted['question']:  # Only add if question is not empty
            sufficient_examples.append(reformatted)

    # Sample insufficient examples
    random.shuffle(unanswerable)
    insufficient_examples = []
    for i in range(min(num_insufficient, len(unanswerable))):
        idx, example = unanswerable[i]
        reformatted = reformat_treecut_example(example, is_answerable_flag=False)
        if reformatted['question']:  # Only add if question is not empty
            insufficient_examples.append(reformatted)

    print(f"\nActual created:")
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


def main():
    parser = argparse.ArgumentParser(description='Preprocess TreeCut dataset')
    parser.add_argument('--output_dir', type=str,
                        default='src/data/processed/treecut',
                        help='Directory to save processed data')
    parser.add_argument('--train_samples', type=int, default=4160,
                        help='Number of training samples (default: 4160, matching UMWP)')
    parser.add_argument('--test_samples', type=int, default=1040,
                        help='Number of test samples (default: 1040, matching UMWP)')
    parser.add_argument('--insufficient_ratio', type=float, default=0.5,
                        help='Ratio of insufficient examples (default: 0.5)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training if no separate test split (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for HuggingFace datasets')

    args = parser.parse_args()

    print("="*80)
    print("TREECUT PREPROCESSING PIPELINE")
    print("="*80)

    # Download dataset
    dataset = download_treecut(cache_dir=args.cache_dir)

    # TreeCut might have 'train' and 'test' or just 'train'
    # Check what splits are available
    if 'train' in dataset and 'test' in dataset:
        # Use existing splits
        print("\nUsing existing train/test splits from TreeCut")
        train_data = create_balanced_split(
            dataset['train'],
            num_samples=args.train_samples,
            insufficient_ratio=args.insufficient_ratio,
            random_seed=args.random_seed
        )
        test_data = create_balanced_split(
            dataset['test'],
            num_samples=args.test_samples,
            insufficient_ratio=args.insufficient_ratio,
            random_seed=args.random_seed + 1
        )
    elif 'train' in dataset:
        # Create our own split
        print("\nCreating train/test split from available data")
        all_data = list(dataset['train'])
        random.seed(args.random_seed)
        random.shuffle(all_data)

        split_point = int(len(all_data) * args.train_ratio)
        train_split = all_data[:split_point]
        test_split = all_data[split_point:]

        print(f"Split into {len(train_split)} train, {len(test_split)} test")

        train_data = create_balanced_split(
            train_split,
            num_samples=args.train_samples,
            insufficient_ratio=args.insufficient_ratio,
            random_seed=args.random_seed
        )
        test_data = create_balanced_split(
            test_split,
            num_samples=args.test_samples,
            insufficient_ratio=args.insufficient_ratio,
            random_seed=args.random_seed + 1
        )
    else:
        raise ValueError(f"Unexpected dataset splits: {list(dataset.keys())}")

    # Save datasets
    print("\n" + "="*80)
    print("SAVING DATASETS")
    print("="*80)

    train_path = os.path.join(args.output_dir, 'treecut_train.json')
    test_path = os.path.join(args.output_dir, 'treecut_test.json')

    save_dataset(train_data, train_path)
    save_dataset(test_data, test_path)

    # Print examples
    if train_data:
        print("\n" + "="*80)
        print("EXAMPLE SUFFICIENT:")
        print("="*80)
        sufficient_examples = [x for x in train_data if x['is_sufficient']]
        if sufficient_examples:
            print(json.dumps(sufficient_examples[0], indent=2))

        print("\n" + "="*80)
        print("EXAMPLE INSUFFICIENT:")
        print("="*80)
        insufficient_examples = [x for x in train_data if not x['is_sufficient']]
        if insufficient_examples:
            print(json.dumps(insufficient_examples[0], indent=2))
        print("="*80)

    print("\n✓ TreeCut preprocessing complete!")


if __name__ == '__main__':
    main()
