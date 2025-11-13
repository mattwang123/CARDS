"""
Download GSM8K dataset from HuggingFace
"""
import argparse
import json
import os
import re
from datasets import load_dataset


def count_numbers_in_text(text):
    """
    Count how many numbers appear in the text

    Args:
        text: Input string

    Returns:
        int: Number of numeric values found
    """
    # Find all numbers (integers and decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    return len(numbers)


def filter_by_number_count(data, min_numbers=4):
    """
    Filter dataset to only include questions with at least min_numbers numbers

    Args:
        data: List of dataset items
        min_numbers: Minimum number of numbers required in question

    Returns:
        Filtered list of items
    """
    filtered = []
    for item in data:
        num_count = count_numbers_in_text(item['question'])
        if num_count >= min_numbers:
            filtered.append(item)

    return filtered


def download_gsm8k(output_dir, num_samples=None):
    """
    Download GSM8K dataset and save to output_dir

    Args:
        output_dir: Directory to save the data
        num_samples: Number of samples to download (None = all)
    """
    print(f"Downloading GSM8K dataset...")

    # Load dataset from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main")

    # Get train split
    train_data = list(dataset['train'])
    test_data = list(dataset['test'])

    print(f"Original train samples: {len(train_data)}")
    print(f"Original test samples: {len(test_data)}")

    # Filter to only keep questions with at least 4 numbers
    print("\nFiltering for questions with at least 4 numbers...")
    train_data = filter_by_number_count(train_data, min_numbers=4)
    test_data = filter_by_number_count(test_data, min_numbers=4)

    print(f"After filtering - Train samples: {len(train_data)}")
    print(f"After filtering - Test samples: {len(test_data)}")

    # Limit samples if specified
    if num_samples is not None:
        train_data = train_data[:min(num_samples, len(train_data))]
        test_samples = min(num_samples // 5, len(test_data))  # 20% for test
        test_data = test_data[:test_samples]

    print(f"\nFinal train samples: {len(train_data)}")
    print(f"Final test samples: {len(test_data)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    train_path = os.path.join(output_dir, 'gsm8k_train.json')
    test_path = os.path.join(output_dir, 'gsm8k_test.json')

    # Convert to list of dicts
    train_list = [{'question': item['question'], 'answer': item['answer']}
                  for item in train_data]
    test_list = [{'question': item['question'], 'answer': item['answer']}
                 for item in test_data]

    with open(train_path, 'w') as f:
        json.dump(train_list, f, indent=2)

    with open(test_path, 'w') as f:
        json.dump(test_list, f, indent=2)

    print(f"\nSaved to:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    # Print example
    print(f"\nExample problem:")
    example_q = train_list[0]['question']
    example_num_count = count_numbers_in_text(example_q)
    print(f"Q: {example_q}")
    print(f"Numbers in question: {example_num_count}")
    print(f"A: {train_list[0]['answer'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description='Download GSM8K dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Directory to save the data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to download (default: 100)')

    args = parser.parse_args()

    download_gsm8k(args.output_dir, args.num_samples)


if __name__ == '__main__':
    main()
