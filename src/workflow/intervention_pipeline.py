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
import torch
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
    """Create reprompt with insufficiency warning as system + user"""
    # System message with warning
    system_msg = """You are a helpful math assistant. The following question might be missing some information needed to solve it completely. If information is missing, clearly identify what specific information you would need to provide a definitive answer. Do not make assumptions - if you cannot solve it with the given information, explain what is missing."""

    # User message with question
    user_msg = f"""Question: {question}

Please analyze this question and:
1. Determine if you have all the information needed to solve it
2. If something is missing, identify EXACTLY what information you need
3. Explain why you cannot provide a definitive numerical answer without this information"""

    # Format as conversation (most models expect this format)
    prompt = f"""System: {system_msg}

User: {user_msg}"""

    return prompt


def load_judge_prompt():
    """Load intervention judge prompt from file"""
    prompt_path = Path(__file__).parent / "judge_prompts" / "intervention_judge.md"

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Judge prompt not found: {prompt_path}")

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

    # Evaluate probe on test set first
    print(f"\n[4/7] Evaluating probe on test set...")
    test_y = np.array([0 if item.get('is_sufficient', True) else 1 for item in test_data])
    test_pred = probe.predict(test_X)
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average='macro')
    test_precision = precision_score(test_y, test_pred, average='macro')
    test_recall = recall_score(test_y, test_pred, average='macro')
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Test F1: {test_f1:.3f}")
    print(f"  Test precision: {test_precision:.3f}")
    print(f"  Test recall: {test_recall:.3f}")

    # Load model for generation
    print(f"\n[5/7] Loading model for generation...")
    solver = MathSolver(model_name, device=device, max_new_tokens=2048, temperature=0.7)

    # Helper function to generate from custom prompt
    def generate_from_prompt(prompt_text):
        """Generate response from custom prompt (bypass create_prompt)"""
        inputs = solver.tokenizer(prompt_text, return_tensors='pt')
        inputs = {k: v.to(solver.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = solver.model.generate(
                **inputs,
                max_new_tokens=solver.max_new_tokens,
                temperature=solver.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=solver.tokenizer.eos_token_id
            )

        response = solver.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if prompt_text in response:
            response = response.replace(prompt_text, '').strip()

        # Count output tokens
        output_tokens = len(outputs[0]) - len(inputs['input_ids'][0])

        return response, output_tokens

    # Run intervention on test insufficient questions
    print(f"\n[6/7] Running intervention on {len(test_insufficient)} insufficient questions...")
    print(f"{'='*80}")

    results = []
    probe_times = []
    model_times = []
    judge_times = []

    probe_detected = 0
    probe_missed = 0
    acknowledged_count = 0
    correctly_identified_count = 0

    for idx, item in enumerate(test_insufficient):
        print(f"\n{'─'*80}")
        print(f"Question {idx+1}/{len(test_insufficient)}")
        print(f"{'─'*80}")

        insufficient_question = item['question']
        original_question = item.get('original_question', insufficient_question)
        removed_value = item.get('removed_value', 'N/A')
        removed_description = item.get('removed_description', 'N/A')

        print(f"\n📝 INSUFFICIENT QUESTION:")
        print(f"   {insufficient_question}")
        print(f"\n🔍 GROUND TRUTH (what was removed):")
        print(f"   Value: {removed_value}")
        print(f"   Description: {removed_description}")

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
        print(f"\n🤖 PROBE DETECTION:")
        start_probe = time.time()
        probe_result = predict_with_probe(probe, question_embedding)
        probe_time = time.time() - start_probe
        probe_times.append(probe_time)

        print(f"   Prediction: {'INSUFFICIENT' if probe_result['prediction'] == 1 else 'SUFFICIENT'}")
        print(f"   Probability: {probe_result['probability_insufficient']:.3f}")
        print(f"   Time: {probe_time*1000:.1f}ms")

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
            print(f"\n❌ PROBE MISSED - Skipping intervention")
        else:  # Predicted insufficient (true positive)
            probe_detected += 1
            result['probe_detected'] = True
            print(f"\n✓ PROBE DETECTED - Running intervention...")

            # 2. Reprompt model with insufficiency warning
            reprompt = create_reprompt(insufficient_question)

            print(f"\n💬 REPROMPT TO MODEL:")
            print(f"   {reprompt}")

            start_model = time.time()
            try:
                model_response, output_tokens = generate_from_prompt(reprompt)
                print(f"\n📤 MODEL RESPONSE:")
                print(f"   {model_response}")
            except Exception as e:
                print(f"\n  ⚠ Error generating response for Q{idx}: {e}")
                model_response = f"[ERROR: {str(e)}]"
                output_tokens = 0
            model_time = time.time() - start_model
            model_times.append(model_time)
            print(f"   Generation time: {model_time:.2f}s ({output_tokens} tokens)")

            result['model_response'] = model_response
            result['model_time'] = model_time

            # 3. Judge with GPT-4o-mini
            removed_info = f"Removed: {removed_value}\nDescription: {removed_description}"

            print(f"\n⚖️  GPT-4o-mini JUDGMENT:")
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

            print(f"   Acknowledged: {judgment['acknowledged']}")
            print(f"   Correctly Identified: {judgment['correctly_identified']}")
            print(f"   Explanation: {judgment['explanation']}")
            print(f"   Judge time: {judge_time:.2f}s")

            result['judgment'] = judgment
            result['judge_time'] = judge_time

            # Update counts
            if judgment['acknowledged'] == 'YES':
                acknowledged_count += 1
                if judgment['correctly_identified'] == 'YES':
                    correctly_identified_count += 1

        results.append(result)

        # Print running stats every 10 questions
        if (idx + 1) % 10 == 0:
            print(f"\n{'='*80}")
            print(f"RUNNING STATS (after {idx+1} questions):")
            print(f"  Probe detected: {probe_detected}/{idx+1} ({probe_detected/(idx+1)*100:.1f}%)")
            print(f"  Acknowledged: {acknowledged_count}/{probe_detected} ({acknowledged_count/probe_detected*100:.1f}% of detected)" if probe_detected > 0 else "  Acknowledged: 0/0")
            print(f"  Correctly identified: {correctly_identified_count}/{probe_detected} ({correctly_identified_count/probe_detected*100:.1f}% of detected)" if probe_detected > 0 else "  Correctly identified: 0/0")
            print(f"{'='*80}")

    # Compute metrics
    print(f"\n[7/7] Computing metrics...")

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
        },
        'probe_test_performance': {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
    }

    # Save results
    print(f"\nSaving results...")
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
    print(f"\nProbe Test Performance:")
    print(f"  Accuracy: {test_acc:.3f}")
    print(f"  F1: {test_f1:.3f}")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall: {test_recall:.3f}")
    print(f"\nIntervention on {total_questions} insufficient questions:")
    print(f"  Probe detected: {probe_detected} ({probe_detection_rate*100:.1f}%)")
    print(f"  Probe missed: {probe_missed} ({probe_miss_rate*100:.1f}%)")
    print(f"\nOf {probe_detected} detected by probe:")
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
