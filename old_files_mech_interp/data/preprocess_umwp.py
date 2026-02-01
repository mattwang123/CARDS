"""
Preprocess UMWP dataset to match our insufficient dataset format
"""
import argparse
import json
import os


# Category descriptions from UMWP paper
CATEGORY_DESCRIPTIONS = {
    1: "Missing critical information",
    2: "Incomplete constraint information",
    3: "Contradictory information",
    4: "Irrelevant question",
    5: "Other type of unanswerability"
}


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def reformat_umwp(input_file, output_dir, train_ratio=0.8, random_seed=42):
    """
    Reformat UMWP dataset to match our insufficient dataset format

    Args:
        input_file: Path to StandardDataset.jsonl
        output_dir: Directory to save reformatted data
        train_ratio: Ratio of data for training (default: 0.8)
        random_seed: Random seed for reproducibility
    """
    import random
    random.seed(random_seed)

    print(f"Loading UMWP dataset from {input_file}...")
    data = load_jsonl(input_file)

    print(f"Total examples: {len(data)}")

    # Split into answerable and unanswerable
    answerable = [x for x in data if x['answerable']]
    unanswerable = [x for x in data if not x['answerable']]

    print(f"  Answerable: {len(answerable)}")
    print(f"  Unanswerable: {len(unanswerable)}")

    # Shuffle each group independently
    random.shuffle(answerable)
    random.shuffle(unanswerable)

    # Split answerable into train/test
    train_size_ans = int(len(answerable) * train_ratio)
    answerable_train = answerable[:train_size_ans]
    answerable_test = answerable[train_size_ans:]

    # Split unanswerable into train/test
    train_size_unans = int(len(unanswerable) * train_ratio)
    unanswerable_train = unanswerable[:train_size_unans]
    unanswerable_test = unanswerable[train_size_unans:]

    print(f"\nTrain split:")
    print(f"  Answerable: {len(answerable_train)}")
    print(f"  Unanswerable: {len(unanswerable_train)}")
    print(f"\nTest split:")
    print(f"  Answerable: {len(answerable_test)}")
    print(f"  Unanswerable: {len(unanswerable_test)}")

    # Create a mapping from ID to original question for unanswerable questions
    id_to_question = {x['id']: x['question'] for x in data}

    def process_items(answerable_items, unanswerable_items):
        """Process and reformat items"""
        reformatted = []

        # Process answerable questions (is_sufficient = True)
        for item in answerable_items:
            # Format answer like GSM8K (with ####)
            answer_value = item['answer'][0] if item['answer'] else 0
            # Convert to int if whole number
            if isinstance(answer_value, float) and answer_value.is_integer():
                answer_value = int(answer_value)

            reformatted_item = {
                'question': item['question'],
                'answer': f"#### {answer_value}",  # GSM8K format for compatibility
                'is_sufficient': True
            }
            reformatted.append(reformatted_item)

        # Process unanswerable questions (is_sufficient = False)
        for item in unanswerable_items:
            # Get original question if available
            original_question = "N/A"
            if item['relevant_ids'] and len(item['relevant_ids']) > 0:
                original_id = item['relevant_ids'][0]
                if original_id in id_to_question:
                    original_question = id_to_question[original_id]

            category_desc = CATEGORY_DESCRIPTIONS.get(item['category'], "Unknown")

            reformatted_item = {
                'question': item['question'],
                'answer': "N/A",  # No answer for unanswerable
                'is_sufficient': False,
                'original_question': original_question,
                'removed_value': "N/A",  # UMWP doesn't specify exact removed value
                'removed_description': f"Category {item['category']}: {category_desc}"
            }
            reformatted.append(reformatted_item)

        # Shuffle to mix answerable and unanswerable
        random.shuffle(reformatted)
        return reformatted

    # Process train and test separately
    train_data = process_items(answerable_train, unanswerable_train)
    test_data = process_items(answerable_test, unanswerable_test)

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, 'umwp_train.json')
    test_file = os.path.join(output_dir, 'umwp_test.json')

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n" + "="*80)
    print("SAVED FILES:")
    print("="*80)
    print(f"Train: {train_file}")
    print(f"  Total: {len(train_data)}")
    print(f"  Sufficient: {sum(1 for x in train_data if x['is_sufficient'])}")
    print(f"  Insufficient: {sum(1 for x in train_data if not x['is_sufficient'])}")

    print(f"\nTest: {test_file}")
    print(f"  Total: {len(test_data)}")
    print(f"  Sufficient: {sum(1 for x in test_data if x['is_sufficient'])}")
    print(f"  Insufficient: {sum(1 for x in test_data if not x['is_sufficient'])}")

    # Print examples
    print("\n" + "="*80)
    print("EXAMPLE SUFFICIENT (TRAIN):")
    print("="*80)
    sufficient_ex = [x for x in train_data if x['is_sufficient']][0]
    print(json.dumps(sufficient_ex, indent=2))

    print("\n" + "="*80)
    print("EXAMPLE INSUFFICIENT (TRAIN):")
    print("="*80)
    insufficient_ex = [x for x in train_data if not x['is_sufficient']][0]
    print(json.dumps(insufficient_ex, indent=2))
    print("="*80)

    return train_file, test_file


def main():
    parser = argparse.ArgumentParser(description='Preprocess UMWP dataset')
    parser.add_argument('--input_file', type=str,
                        default='data/raw_umwp/StandardDataset.jsonl',
                        help='Path to UMWP JSONL file')
    parser.add_argument('--output_dir', type=str,
                        default='data/insufficient_dataset_umwp',
                        help='Directory to save reformatted data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    reformat_umwp(args.input_file, args.output_dir, args.train_ratio, args.random_seed)


if __name__ == '__main__':
    main()
