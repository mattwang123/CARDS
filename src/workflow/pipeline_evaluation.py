"""
Stage 2: Compute metrics from saved GPTEvaluator results.

This module reads the JSON output from Stage 1 (run_experiment.py) and computes
aggregate metrics without making any API calls.
"""
import json
from typing import Dict, List, Any


def compute_metrics_from_json(results_path: str) -> Dict[str, Any]:
    """
    Compute aggregate metrics from Stage 1 results JSON.
    
    Args:
        results_path: Path to JSON file saved by Stage 1
    
    Returns:
        dict with summary metrics for workflow and baselines
    """
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    examples = data['examples']
    
    # Workflow metrics
    workflow_metrics = _compute_agent_metrics(examples, 'workflow', include_question_match=True)
    
    # Baseline metrics
    baseline_metrics = {}
    baseline_types = [k.replace('baseline_', '') for k in examples[0].keys() if k.startswith('baseline_')]
    
    for baseline_type in baseline_types:
        baseline_key = f'baseline_{baseline_type}'
        baseline_metrics[baseline_type] = _compute_agent_metrics(examples, baseline_key, include_question_match=False)
    
    return {
        'workflow': workflow_metrics,
        **{f'baseline_{bt}': baseline_metrics[bt] for bt in baseline_types}
    }


def _compute_agent_metrics(examples: List[Dict], agent_key: str, include_question_match: bool = False) -> Dict[str, Any]:
    """
    Compute metrics for any agent (workflow or baseline).
    
    Args:
        examples: List of example dicts from Stage 1 results
        agent_key: Key for the agent in each example ('workflow' or 'baseline_*')
        include_question_match: Whether to compute question_match_rate (only for workflow)
    
    Returns:
        dict with metrics: behavior_correctness, answer_accuracy, avg_tokens, total_tokens, num_examples,
        and optionally question_match_rate
    """
    total_behavior_correctness = 0.0
    total_answer_correct = 0
    num_answer_cases = 0
    total_tokens = 0
    question_match_count = 0
    question_match_total = 0
    
    for ex in examples:
        agent = ex[agent_key]
        eval_initial = agent['evaluation_initial']
        eval_final = agent.get('evaluation_final_answer')
        is_sufficient = ex['ground_truth_sufficient']
        
        # Behavior correctness (from initial evaluation)
        # Support both old "redundancy" and new "behavior_correctness" for backward compatibility
        behavior_correctness = eval_initial.get('behavior_correctness', eval_initial.get('redundancy', 0.0))
        total_behavior_correctness += behavior_correctness
        total_tokens += agent['tokens_used']
        
        # Answer correctness
        # Priority: final_response > initial_response
        # For sufficient examples: if agent asked question (incorrect behavior), we have final_response
        # For insufficient examples: if user disclosed info, we have final_response
        if eval_final and eval_final.get('answer_correct') is not None:
            # Final answer available (agent asked question and got user response)
            num_answer_cases += 1
            if eval_final['answer_correct']:
                total_answer_correct += 1
        elif eval_initial.get('answer_correct') is not None:
            # No final answer, check initial response
            num_answer_cases += 1
            if eval_initial['answer_correct']:
                total_answer_correct += 1
        
        # Question match rate (only for workflow, insufficient questions)
        if include_question_match and not is_sufficient:
            q_match = eval_initial.get('question_matches_missing_info')
            if q_match is not None:
                question_match_total += 1
                if q_match:
                    question_match_count += 1
    
    metrics = {
        'behavior_correctness': total_behavior_correctness / len(examples) if examples else 0.0,
        'answer_accuracy': (total_answer_correct / num_answer_cases) if num_answer_cases > 0 else 0.0,
        'avg_tokens': total_tokens / len(examples) if examples else 0.0,
        'total_tokens': total_tokens,
        'num_examples': len(examples)
    }
    
    if include_question_match:
        metrics['question_match_rate'] = (question_match_count / question_match_total) if question_match_total > 0 else 0.0
    
    return metrics


def print_summary(metrics: Dict[str, Any]):
    """Print a formatted summary of metrics."""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    print(f"\nWorkflow:")
    wf = metrics['workflow']
    print(f"  Behavior correctness: {wf['behavior_correctness']:.4f}")
    print(f"  Answer accuracy: {wf.get('answer_accuracy', 0.0):.4f}")
    if 'question_match_rate' in wf:
        print(f"  Question match rate: {wf['question_match_rate']:.4f}")
    print(f"  Avg tokens: {wf['avg_tokens']:.2f}")
    
    for key in metrics:
        if key.startswith('baseline_'):
            baseline_type = key.replace('baseline_', '')
            bl = metrics[key]
            print(f"\nBaseline ({baseline_type}):")
            print(f"  Behavior correctness: {bl['behavior_correctness']:.4f}")
            print(f"  Answer accuracy: {bl.get('answer_accuracy', 0.0):.4f}")
            print(f"  Avg tokens: {bl['avg_tokens']:.2f}")

