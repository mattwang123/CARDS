"""
Training-Free Intervention Pipeline

This script implements probe-guided intervention for logical insufficiency:
1. Train linear probes on best layers (using cached embeddings)
2. Detect insufficiency in test questions
3. If detected, reprompt model with insufficiency warning
4. Judge model's acknowledgment and identification of missing information
5. Compute metrics and save results
"""

import argparse
import json
import os
import sys
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inference import MathSolver

# Server paths
SERVER_EMBEDDING_BASE = "/export/fs06/psingh54/CARDS/src/data/embeddings"
SERVER_PROBE_RESULTS = "/export/fs06/psingh54/CARDS/src/experiments/all_probes_linear/best_layers_linear.json"
SERVER_DATA_BASE = "/export/fs06/psingh54/CARDS/src/data/processed"

# Dataset configurations
DATASETS = {
    'umwp': {
        'train': 'insufficient_dataset_umwp/umwp_train.json',
        'test': 'insufficient_dataset_umwp/umwp_test.json'
    },
    'gsm8k': {
        'train': 'gsm8k/gsm8k_train.json',
        'test': 'gsm8k/gsm8k_test.json'
    }
}

# Model configurations (matching embedding filenames)
MODELS = [
    'qwen2.5-math-1.5b',
    'qwen2.5-1.5b',
    'llama-3.2-3b-instruct',
    'qwen2.5-math-7b'
]


def load_best_layers(results_path):
    """Load best layer configuration for each model-dataset pair"""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_dataset(dataset_name, split='train'):
    """Load dataset from server processed files"""
    dataset_path = os.path.join(SERVER_DATA_BASE, DATASETS[dataset_name][split])
    with open(dataset_path, 'r') as f:
        return json.load(f)


def load_embeddings(dataset_name, model_name, split='train', pooling='last_token'):
    """Load embeddings from server"""
    # Construct filename following the pattern
    base_name = f"{dataset_name}_{split}_{model_name}_{pooling}"

    embeddings_file = os.path.join(SERVER_EMBEDDING_BASE, f"{base_name}.npy")
    metadata_file = os.path.join(SERVER_EMBEDDING_BASE, f"{base_name}_metadata.json")

    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return embeddings, metadata


def train_probe(X_train, y_train):
    """Train linear probe (logistic regression) with balanced class weights"""
    probe = LogisticRegression(
        max_iter=7000,
        solver='lbfgs',
        C=1.0,  # L2 regularization (lambda = 1.0)
        class_weight='balanced',
        random_state=42
    )

    probe.fit(X_train, y_train)
    return probe


def predict_with_probe(probe, embedding):
    """Predict insufficiency using trained probe"""
    # Reshape for single prediction
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    prediction = probe.predict(embedding)[0]
    probability = probe.predict_proba(embedding)[0]

    return {
        'prediction': int(prediction),  # 0 = sufficient, 1 = insufficient
        'probability_insufficient': float(probability[1]),
        'probability_sufficient': float(probability[0])
    }


def create_reprompt(question):
    """Create reprompt with insufficiency warning"""
    prompt = f"""Question: {question}

Note: This question is believed to be logically insufficient - it may be missing critical information or constraints needed for a unique answer.

Please identify what information is missing and what you would need to solve this problem."""

    return prompt


def load_judge_prompt():
    """Load or create GPT-4o-mini judge prompt for intervention evaluation"""
    prompt_path = Path(__file__).parent / "judge_prompts" / "intervention_judge.md"

    if not prompt_path.exists():
        # Create default judge prompt
        default_prompt = """You are evaluating whether a model correctly acknowledged and identified missing information in a logically insufficient math problem.

The model was informed that the question might be logically insufficient and was asked to identify what information is missing.

You will be given:
1. The ORIGINAL sufficient question (with all information)
2. The INSUFFICIENT question (what the model saw, with something removed)
3. What was removed (ground truth)
4. The model's response

Your task is to evaluate TWO things:

## 1. ACKNOWLEDGMENT
Does the model acknowledge that information is missing?
- Answer "YES" if the model recognizes insufficiency and attempts to identify missing info
- Answer "NO" if the model ignores the warning and just attempts to solve the problem with assumptions

## 2. CORRECT IDENTIFICATION (only if acknowledged)
If the model acknowledged insufficiency, does it correctly identify what's missing?
- Answer "YES" if the model identifies the specific missing value/constraint
- Answer "NO" if the model identifies wrong information or is too vague
- Answer "N/A" if the model didn't acknowledge insufficiency

## OUTPUT FORMAT
Respond with valid JSON only:
{
  "acknowledged": "YES" | "NO",
  "correctly_identified": "YES" | "NO" | "N/A",
  "explanation": "<brief reasoning for your judgment>"
}

## EXAMPLES

### Example 1: Acknowledged + Correctly Identified
Original: "John has 5 apples and buys 3 more. How many does he have?"
Insufficient: "John has some apples and buys 3 more. How many does he have?"
Removed: "5 apples" - initial number of apples
Model Response: "I cannot determine the exact answer because the initial number of apples John has is not specified. I would need to know how many apples John started with."

Your Response:
{
  "acknowledged": "YES",
  "correctly_identified": "YES",
  "explanation": "Model correctly identified that the initial number of apples is missing"
}

### Example 2: Acknowledged + Incorrectly Identified
Original: "Mary reads 20 pages per day for 5 days. How many pages total?"
Insufficient: "Mary reads some pages per day for 5 days. How many pages total?"
Removed: "20 pages" - pages per day
Model Response: "This problem is missing information. I need to know the total number of pages in the book to answer this question."

Your Response:
{
  "acknowledged": "YES",
  "correctly_identified": "NO",
  "explanation": "Model acknowledged insufficiency but identified wrong missing info (total pages instead of pages per day)"
}

### Example 3: Not Acknowledged (Makes Assumptions)
Original: "A triangle has sides 10cm and 15cm with a 90-degree angle between them. What is the area?"
Insufficient: "A triangle has sides 10cm and 15cm. What is the area?"
Removed: "90-degree angle between them" - the included angle
Model Response: "Let me assume this is a right triangle. Using the formula Area = 0.5 × base × height = 0.5 × 10 × 15 = 75 square cm. The answer is 75."

Your Response:
{
  "acknowledged": "NO",
  "correctly_identified": "N/A",
  "explanation": "Model made assumptions and solved the problem instead of acknowledging missing information"
}

## IMPORTANT
- Be strict: model must explicitly state what's missing, not just say "something is missing"
- Focus on whether model identifies the SAME information that was removed
- Generic statements like "more context needed" without specifics → correctly_identified = "NO"
"""

        os.makedirs(prompt_path.parent, exist_ok=True)
        with open(prompt_path, 'w') as f:
            f.write(default_prompt)
        print(f"✓ Created intervention judge prompt: {prompt_path}")

    with open(prompt_path, 'r') as f:
        return f.read()


def judge_intervention(original_question, insufficient_question, removed_info, model_response,
                       judge_prompt, openai_client):
    """Use GPT-4o-mini to judge model's acknowledgment and identification"""

    evaluation_prompt = f"""{judge_prompt}

## YOUR TASK

**Original Question (with all information)**:
{original_question}

**Insufficient Question (what model saw)**:
{insufficient_question}

**What Was Removed (ground truth)**:
{removed_info}

**Model's Response**:
{model_response}

Please evaluate the model's response. Respond with JSON only:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intervention evaluator. Respond only with valid JSON."
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
            "acknowledged": "ERROR",
            "correctly_identified": "ERROR",
            "explanation": f"API error: {str(e)}"
        }


def run_intervention_pipeline(model_name, dataset_name, device, openai_client, judge_prompt,
                              best_layers, output_dir):
    """Run full intervention pipeline for one model-dataset pair"""

    print(f"\n{'='*80}")
    print(f"INTERVENTION PIPELINE: {model_name} on {dataset_name}")
    print(f"{'='*80}")

    # Get best layer for this configuration
    train_key = f"train_on_{dataset_name}"
    best_layer = best_layers[model_name][train_key]['best_layer']
    print(f"Using best layer: {best_layer} (from {train_key})")

    # Load datasets
    print(f"\n[1/7] Loading datasets...")
    train_data = load_dataset(dataset_name, 'train')
    test_data = load_dataset(dataset_name, 'test')

    # Filter to only insufficient questions for test
    test_insufficient = [item for item in test_data if not item.get('is_sufficient', True)]
    print(f"  Train: {len(train_data)} total")
    print(f"  Test: {len(test_insufficient)} insufficient questions")

    # Load embeddings
    print(f"\n[2/7] Loading embeddings from server...")
    train_embeddings, train_metadata = load_embeddings(dataset_name, model_name, 'train')
    test_embeddings, test_metadata = load_embeddings(dataset_name, model_name, 'test')

    # Extract best layer embeddings
    train_X = train_embeddings[:, best_layer, :]
    test_X = test_embeddings[:, best_layer, :]

    # Get labels
    train_y = np.array([0 if item.get('is_sufficient', True) else 1 for item in train_data])

    print(f"  Train embeddings: {train_X.shape}")
    print(f"  Test embeddings: {test_X.shape}")

    # Train probe
    print(f"\n[3/7] Training linear probe on layer {best_layer}...")
    start_train = time.time()
    probe = train_probe(train_X, train_y)
    train_time = time.time() - start_train
    print(f"  ✓ Probe trained in {train_time:.2f}s")

    # Save probe
    probe_dir = os.path.join(output_dir, 'probes')
    os.makedirs(probe_dir, exist_ok=True)
    probe_path = os.path.join(probe_dir, f"{model_name}_{dataset_name}_layer{best_layer}.pkl")
    with open(probe_path, 'wb') as f:
        pickle.dump(probe, f)
    print(f"  ✓ Saved probe: {probe_path}")

    # Evaluate probe on train set for sanity check
    train_pred = probe.predict(train_X)
    train_acc = accuracy_score(train_y, train_pred)
    train_f1 = f1_score(train_y, train_pred, average='macro')
    print(f"  Train accuracy: {train_acc:.3f}, F1: {train_f1:.3f}")

    # Load model for generation
    print(f"\n[4/7] Loading model for generation...")
    solver = MathSolver(model_name, device=device, max_new_tokens=512, temperature=0.7)

    # Run intervention on test insufficient questions
    print(f"\n[5/7] Running intervention on {len(test_insufficient)} insufficient questions...")

    results = []
    probe_times = []
    model_times = []
    judge_times = []

    probe_detected = 0
    probe_missed = 0
    acknowledged_count = 0
    correctly_identified_count = 0

    for idx, item in enumerate(tqdm(test_insufficient, desc="Processing")):
        insufficient_question = item['question']
        original_question = item.get('original_question', insufficient_question)
        removed_value = item.get('removed_value', 'N/A')
        removed_description = item.get('removed_description', 'N/A')

        # Find corresponding test embedding
        # Match by question text
        test_idx = None
        for i, test_item in enumerate(test_data):
            if test_item['question'] == insufficient_question:
                test_idx = i
                break

        if test_idx is None:
            print(f"\n  ⚠ Warning: Could not find embedding for question {idx}")
            continue

        question_embedding = test_X[test_idx]

        # 1. Use probe to detect insufficiency
        start_probe = time.time()
        probe_result = predict_with_probe(probe, question_embedding)
        probe_time = time.time() - start_probe
        probe_times.append(probe_time)

        result = {
            'question_idx': idx,
            'insufficient_question': insufficient_question,
            'original_question': original_question,
            'removed_value': removed_value,
            'removed_description': removed_description,
            'probe_prediction': probe_result['prediction'],
            'probe_probability': probe_result['probability_insufficient'],
            'probe_time': probe_time
        }

        # Check if probe detected insufficiency
        if probe_result['prediction'] == 0:  # Predicted sufficient (false negative)
            probe_missed += 1
            result['probe_detected'] = False
            result['model_response'] = None
            result['judgment'] = None
            result['model_time'] = 0
            result['judge_time'] = 0
        else:  # Predicted insufficient (true positive)
            probe_detected += 1
            result['probe_detected'] = True

            # 2. Reprompt model with insufficiency warning
            reprompt = create_reprompt(insufficient_question)

            start_model = time.time()
            try:
                model_response, _ = solver.generate(reprompt)
            except Exception as e:
                print(f"\n  ⚠ Error generating response for Q{idx}: {e}")
                model_response = f"[ERROR: {str(e)}]"
            model_time = time.time() - start_model
            model_times.append(model_time)

            result['model_response'] = model_response
            result['model_time'] = model_time

            # 3. Judge with GPT-4o-mini
            removed_info = f"Removed: {removed_value}\nDescription: {removed_description}"

            start_judge = time.time()
            judgment = judge_intervention(
                original_question,
                insufficient_question,
                removed_info,
                model_response,
                judge_prompt,
                openai_client
            )
            judge_time = time.time() - start_judge
            judge_times.append(judge_time)

            result['judgment'] = judgment
            result['judge_time'] = judge_time

            # Update counts
            if judgment['acknowledged'] == 'YES':
                acknowledged_count += 1
                if judgment['correctly_identified'] == 'YES':
                    correctly_identified_count += 1

        results.append(result)

    # Compute metrics
    print(f"\n[6/7] Computing metrics...")

    total_questions = len(test_insufficient)
    probe_detection_rate = probe_detected / total_questions if total_questions > 0 else 0
    probe_miss_rate = probe_missed / total_questions if total_questions > 0 else 0

    acknowledgment_rate = acknowledged_count / probe_detected if probe_detected > 0 else 0
    identification_rate = correctly_identified_count / acknowledged_count if acknowledged_count > 0 else 0
    identification_rate_overall = correctly_identified_count / probe_detected if probe_detected > 0 else 0

    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'best_layer': best_layer,
        'total_insufficient_questions': total_questions,
        'probe_metrics': {
            'detected': probe_detected,
            'missed': probe_missed,
            'detection_rate': probe_detection_rate,
            'miss_rate': probe_miss_rate
        },
        'intervention_metrics': {
            'acknowledged': acknowledged_count,
            'correctly_identified': correctly_identified_count,
            'acknowledgment_rate': acknowledgment_rate,  # Of detected
            'identification_rate_given_acknowledged': identification_rate,  # Of acknowledged
            'identification_rate_overall': identification_rate_overall  # Of detected
        },
        'timing': {
            'probe_avg': np.mean(probe_times) if probe_times else 0,
            'probe_total': np.sum(probe_times) if probe_times else 0,
            'model_avg': np.mean(model_times) if model_times else 0,
            'model_total': np.sum(model_times) if model_times else 0,
            'judge_avg': np.mean(judge_times) if judge_times else 0,
            'judge_total': np.sum(judge_times) if judge_times else 0,
            'total': np.sum(probe_times) + np.sum(model_times) + np.sum(judge_times)
        },
        'probe_training': {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'train_time': train_time
        }
    }

    # Save results
    print(f"\n[7/7] Saving results...")
    results_data = {
        'metrics': metrics,
        'detailed_results': results
    }

    results_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_intervention.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  ✓ Saved: {results_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"Total insufficient questions: {total_questions}")
    print(f"Probe detected: {probe_detected} ({probe_detection_rate*100:.1f}%)")
    print(f"Probe missed: {probe_missed} ({probe_miss_rate*100:.1f}%)")
    print(f"\nOf detected by probe:")
    print(f"  Acknowledged insufficiency: {acknowledged_count} ({acknowledgment_rate*100:.1f}%)")
    print(f"  Correctly identified: {correctly_identified_count} ({identification_rate_overall*100:.1f}%)")
    print(f"\nTiming (avg per question):")
    print(f"  Probe: {metrics['timing']['probe_avg']*1000:.1f}ms")
    print(f"  Model: {metrics['timing']['model_avg']:.2f}s")
    print(f"  Judge: {metrics['timing']['judge_avg']:.2f}s")
    print(f"{'='*80}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Training-Free Intervention Pipeline with Probe-Guided Reprompting'
    )
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for model inference')
    parser.add_argument('--output_dir', type=str, default='experiments/intervention',
                        help='Directory to save results and probes')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to run (otherwise all models)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to run (otherwise all datasets)')

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("\n✗ Error: OPENAI_API_KEY not found")
        print("Set it in .env file or environment")
        return

    openai_client = OpenAI(api_key=api_key)

    # Load best layers configuration
    print("="*80)
    print("TRAINING-FREE INTERVENTION PIPELINE")
    print("="*80)
    print(f"Loading best layer configuration from: {SERVER_PROBE_RESULTS}")

    if not os.path.exists(SERVER_PROBE_RESULTS):
        print(f"\n✗ Error: Best layers file not found: {SERVER_PROBE_RESULTS}")
        print("Make sure you're running on the server with access to probe results")
        return

    best_layers = load_best_layers(SERVER_PROBE_RESULTS)

    # Load judge prompt
    judge_prompt = load_judge_prompt()
    print("✓ Loaded intervention judge prompt")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which models and datasets to run
    models_to_run = [args.model] if args.model else MODELS
    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    print(f"\nModels: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print("="*80)

    # Run pipeline for each model-dataset pair
    all_metrics = {}

    for model_name in models_to_run:
        model_metrics = {}

        for dataset_name in datasets_to_run:
            metrics = run_intervention_pipeline(
                model_name,
                dataset_name,
                args.device,
                openai_client,
                judge_prompt,
                best_layers,
                args.output_dir
            )
            model_metrics[dataset_name] = metrics

        all_metrics[model_name] = model_metrics

    # Save summary
    summary_path = os.path.join(args.output_dir, 'intervention_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("INTERVENTION PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {args.output_dir}/")
    print(f"Summary: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
