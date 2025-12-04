"""
Main experiment script for probe-based workflow

Stage 1: Generate responses and run GPTEvaluator, save results to JSON.
Stage 2: Compute metrics from saved results (via pipeline_evaluation.py).
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.probe_inference import ProbeInference, load_best_probe
from workflow.agent import WorkflowAgent
from workflow.baselines import BaselineAgent
from workflow.user_simulators import GPT5Simulator, RAGSimulator, GPT5RecallSimulator
from workflow.gpt_evaluator import GPTEvaluator
from workflow.pipeline_evaluation import compute_metrics_from_json, print_summary
from models.inference import MathSolver
from tqdm import tqdm


@dataclass
class ExampleState:
    """
    Data class to track the state of an example through the workflow pipeline.
    
    This class maintains all intermediate results and decisions for a single example,
    making it easy to track what happened at each stage and why.
    """
    # Original dataset fields
    question: str
    original_question: Optional[str] = None
    missing_info: Optional[str] = None
    ground_truth_sufficient: bool = True
    ground_truth_answer: Optional[str] = None
    
    # Probe results
    probe_prediction: Optional[bool] = None  # True = sufficient, False = insufficient
    probe_confidence: Optional[float] = None
    prompt_appended: bool = False
    
    # Initial agent response
    first_response: Optional[str] = None
    first_response_tokens: int = 0
    
    # Initial evaluation (after first response)
    evaluation_initial: Optional[Dict[str, Any]] = None
    agent_asked_question: Optional[bool] = None  # From evaluation_initial['asked_question']
    agent_attempted_answer: Optional[bool] = None  # From evaluation_initial['answer_attempted']
    
    # User simulator interaction
    user_response: Optional[str] = None
    user_disclosed_info: bool = False  # Whether user decided to disclose
    
    # Follow-up agent response (only if user disclosed)
    final_response: Optional[str] = None
    final_response_tokens: int = 0
    
    # Final answer evaluation (only if final_response exists)
    evaluation_final_answer: Optional[Dict[str, Any]] = None
    
    # Total tokens (first + final)
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'question': self.question,
            'original_question': self.original_question,
            'missing_info': self.missing_info,
            'ground_truth_sufficient': self.ground_truth_sufficient,
            'ground_truth_answer': self.ground_truth_answer,
            'workflow': {
                'first_response': self.first_response,
                'final_response': self.final_response,
                'tokens_used': self.total_tokens,
                'probe_prediction': self.probe_prediction,
                'probe_confidence': self.probe_confidence,
                'prompt_appended': self.prompt_appended,
                'user_provided_info': self.user_disclosed_info,
                'evaluation_initial': self.evaluation_initial,
                'evaluation_final_answer': self.evaluation_final_answer,
            }
        }


def _convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_python_types(item) for item in obj]
    else:
        return obj


def _get_missing_info(example):
    """Extract missing information from an example."""
    removed_value = example.get('removed_value', '')
    removed_description = example.get('removed_description', '')
    
    if removed_value and removed_value != 'N/A':
        return removed_value
    if removed_description and removed_description != 'N/A':
        return removed_description
    return 'Unknown'


def _get_ground_truth_answer(example):
    """Extract ground-truth answer from example."""
    if not example.get('is_sufficient', True) and 'original_answer' in example:
        return example['original_answer']
    return example.get('answer', None)


def _process_sufficient_examples(sufficient_examples, workflow_agent, solver,
                                  baseline_agents, baseline_types, evaluator, batch_size=8):
    """
    Process sufficient examples: probe → solver → evaluate → if agent asked question,
    simulate user response "Please answer my original question" → follow-up solver → evaluate final answer.
    
    This ensures we can always calculate accuracy for all methods, even when agents
    incorrectly ask questions for sufficient problems.
    
    Returns:
        List of result dicts with first_response, final_response (if agent asked), evaluations
    """
    results = []
    
    # Initialize ExampleState objects for workflow
    workflow_states = []
    for example in sufficient_examples:
        state = ExampleState(
            question=example['question'],
            ground_truth_sufficient=True,
            ground_truth_answer=example.get('answer')
        )
        workflow_states.append(state)
    
    # Batch probe predictions
    all_questions = [ex['question'] for ex in sufficient_examples]
    if hasattr(workflow_agent.probe_inference, 'predict_batch'):
        probe_results = workflow_agent.probe_inference.predict_batch(all_questions)
    else:
        probe_results = [workflow_agent.probe_inference.predict(q) for q in all_questions]
    
    # Store probe results in states
    for state, (is_sufficient, confidence) in zip(workflow_states, probe_results):
        state.probe_prediction = bool(is_sufficient)
        state.probe_confidence = float(confidence)
    
    # Prepare questions (with prompts if probe incorrectly detected insufficiency)
    initial_questions = []
    for state, (is_sufficient, _) in zip(workflow_states, probe_results):
        question = state.question
        if not is_sufficient:
            # Probe incorrectly detected insufficiency - append prompt anyway
            question = f"{question}\n\n{workflow_agent.system_prompt}"
            state.prompt_appended = True
        else:
            state.prompt_appended = False
        initial_questions.append(question)
    
    # Batch solve
    initial_responses = solver.batch_solve(initial_questions, batch_size=batch_size)
    
    # Store initial responses and count tokens
    for state, response in zip(workflow_states, initial_responses):
        state.first_response = response
        state.first_response_tokens = len(solver.tokenizer.encode(response))
    
    # Evaluate initial responses FIRST
    print("Evaluating initial workflow responses...")
    for state in tqdm(workflow_states, desc="Evaluating workflow", unit="example"):
        state.evaluation_initial = evaluator.evaluate(
            state.question,
            state.first_response,
            ground_truth_sufficient=True,
            missing_info=None,
            ground_truth_answer=state.ground_truth_answer
        )
        state.agent_asked_question = state.evaluation_initial.get('asked_question', False)
        state.agent_attempted_answer = state.evaluation_initial.get('answer_attempted', False)
    
    # If agent asked a question (incorrect behavior), simulate user response
    # User says: "Please answer my original question."
    follow_up_questions = []
    follow_up_indices = []
    for i, state in enumerate(workflow_states):
        if state.agent_asked_question:
            # Simulate user response asking agent to answer the original question
            user_response = "Please answer my original question."
            state.user_response = user_response
            state.user_disclosed_info = True  # User provided instruction to answer
            
            # Use original question for follow-up (not the question with appended prompt)
            follow_up_question = state.question
            follow_up_questions.append(follow_up_question)
            follow_up_indices.append(i)
        else:
            state.user_response = None
            state.user_disclosed_info = False
    
    # Follow-up solver calls (if agent asked question)
    if follow_up_questions:
        print(f"Calling follow-up solver for {len(follow_up_questions)} examples (agent asked question)...")
        follow_up_responses = solver.batch_solve(follow_up_questions, batch_size=batch_size)
        for idx, response in zip(follow_up_indices, follow_up_responses):
            state = workflow_states[idx]
            state.final_response = response
            state.final_response_tokens = len(solver.tokenizer.encode(response))
    
    # Calculate total tokens
    for state in workflow_states:
        state.total_tokens = state.first_response_tokens + state.final_response_tokens
    
    # Process baselines (similar conditional flow)
    baseline_states = {bt: [] for bt in baseline_types}
    
    # Initialize baseline states
    for baseline_type in baseline_types:
        for example in sufficient_examples:
            state = ExampleState(
                question=example['question'],
                ground_truth_sufficient=True,
                ground_truth_answer=example.get('answer')
            )
            baseline_states[baseline_type].append(state)
    
    # Baseline questions and initial responses
    baseline_questions = {bt: [] for bt in baseline_types}
    for example in sufficient_examples:
        question = example['question']
        for baseline_type in baseline_types:
            if baseline_type == 'always_prompt':
                baseline_questions[baseline_type].append(
                    f"{question}\n\n{baseline_agents[baseline_type].system_prompt}"
                )
                baseline_states[baseline_type][len(baseline_questions[baseline_type])-1].prompt_appended = True
            elif baseline_type == 'just_answer':
                baseline_questions[baseline_type].append(question)
                baseline_states[baseline_type][len(baseline_questions[baseline_type])-1].prompt_appended = False
            else:
                raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    baseline_responses = {}
    for baseline_type in baseline_types:
        baseline_responses[baseline_type] = solver.batch_solve(
            baseline_questions[baseline_type], batch_size=batch_size
        )
        # Store responses
        for state, response in zip(baseline_states[baseline_type], baseline_responses[baseline_type]):
            state.first_response = response
            state.first_response_tokens = len(solver.tokenizer.encode(response))
    
    # Evaluate baseline initial responses
    print("Evaluating initial baseline responses...")
    for baseline_type in baseline_types:
        for state in tqdm(baseline_states[baseline_type], desc=f"Evaluating {baseline_type}", unit="example"):
            state.evaluation_initial = evaluator.evaluate(
                state.question,
                state.first_response,
                ground_truth_sufficient=True,
                missing_info=None,
                ground_truth_answer=state.ground_truth_answer
            )
            state.agent_asked_question = state.evaluation_initial.get('asked_question', False)
            state.agent_attempted_answer = state.evaluation_initial.get('answer_attempted', False)
    
    # If baseline asked a question, simulate user response
    for baseline_type in baseline_types:
        follow_up_questions = []
        follow_up_indices = []
        for i, state in enumerate(baseline_states[baseline_type]):
            if state.agent_asked_question:
                user_response = "Please answer my original question."
                state.user_response = user_response
                state.user_disclosed_info = True
                follow_up_question = state.question
                follow_up_questions.append(follow_up_question)
                follow_up_indices.append(i)
            else:
                state.user_response = None
                state.user_disclosed_info = False
        
        if follow_up_questions:
            follow_up_responses = solver.batch_solve(follow_up_questions, batch_size=batch_size)
            for idx, response in zip(follow_up_indices, follow_up_responses):
                state = baseline_states[baseline_type][idx]
                state.final_response = response
                state.final_response_tokens = len(solver.tokenizer.encode(response))
        
        # Calculate total tokens for baselines
        for state in baseline_states[baseline_type]:
            state.total_tokens = state.first_response_tokens + state.final_response_tokens
    
    # Compile results from states
    for i, (example, workflow_state) in enumerate(zip(sufficient_examples, workflow_states)):
        result = workflow_state.to_dict()
        
        # Add baseline results
        for baseline_type in baseline_types:
            baseline_state = baseline_states[baseline_type][i]
            result[f'baseline_{baseline_type}'] = {
                'first_response': baseline_state.first_response,
                'final_response': baseline_state.final_response,
                'tokens_used': baseline_state.total_tokens,
                'prompt_appended': baseline_state.prompt_appended,
                'user_provided_info': baseline_state.user_disclosed_info,
                'evaluation_initial': baseline_state.evaluation_initial,
                'evaluation_final_answer': baseline_state.evaluation_final_answer,
            }
        
        results.append(result)
    
    return results


def _process_insufficient_examples(insufficient_examples, workflow_agent, user_simulator, solver,
                                    baseline_agents, baseline_types, evaluator, batch_size=8):
    """
    Process insufficient examples with conditional user interaction.
    
    Flow:
    1. Probe → initial solver response
    2. Evaluate initial response FIRST to determine if agent asked a question
    3. Only call user simulator if agent asked a question
    4. Only call follow-up solver if user disclosed information
    5. Compile results with ExampleState objects
    
    Returns:
        List of result dicts with first_response, final_response (or None), evaluations
    """
    results = []
    
    # Initialize ExampleState objects for workflow
    workflow_states = []
    for example in insufficient_examples:
        state = ExampleState(
            question=example['question'],
            original_question=example.get('original_question'),
            missing_info=_get_missing_info(example),
            ground_truth_sufficient=False,
            ground_truth_answer=example.get('original_answer')
        )
        workflow_states.append(state)
    
    # Batch probe predictions
    all_questions = [ex['question'] for ex in insufficient_examples]
    if hasattr(workflow_agent.probe_inference, 'predict_batch'):
        probe_results = workflow_agent.probe_inference.predict_batch(all_questions)
    else:
        probe_results = [workflow_agent.probe_inference.predict(q) for q in all_questions]
    
    # Store probe results in states
    for state, (is_sufficient, confidence) in zip(workflow_states, probe_results):
        state.probe_prediction = bool(is_sufficient)
        state.probe_confidence = float(confidence)
    
    # Prepare initial questions (with prompts if probe detected insufficiency)
    initial_questions = []
    for state, (is_sufficient, _) in zip(workflow_states, probe_results):
        question = state.question
        if not is_sufficient:
            question = f"{question}\n\n{workflow_agent.system_prompt}"
            state.prompt_appended = True
        else:
            state.prompt_appended = False
        initial_questions.append(question)
    
    # Batch solve initial questions
    initial_responses = solver.batch_solve(initial_questions, batch_size=batch_size)
    
    # Store initial responses and count tokens
    for state, response in zip(workflow_states, initial_responses):
        state.first_response = response
        state.first_response_tokens = len(solver.tokenizer.encode(response))
    
    # Evaluate initial responses FIRST (before user simulator)
    print("Evaluating initial workflow responses...")
    for state in tqdm(workflow_states, desc="Evaluating workflow", unit="example"):
        state.evaluation_initial = evaluator.evaluate(
            state.question,
            state.first_response,
            ground_truth_sufficient=False,
            missing_info=state.missing_info,
            ground_truth_answer=None  # Not needed for insufficient initial evaluation
        )
        state.agent_asked_question = state.evaluation_initial.get('asked_question', False)
        state.agent_attempted_answer = state.evaluation_initial.get('answer_attempted', False)
    
    # User simulator: only call if agent asked a question
    print("Interacting with user simulator (only if agent asked)...")
    for state in tqdm(workflow_states, desc="User simulator", unit="example"):
        if state.agent_asked_question:
            user_response, disclosed = user_simulator.respond(
                state.first_response,
                state.original_question or state.question,
                state.missing_info
            )
            state.user_response = user_response
            state.user_disclosed_info = disclosed
        else:
            # Agent didn't ask, so user doesn't respond
            state.user_response = None
            state.user_disclosed_info = False
    
    # Follow-up solver: only call if user disclosed info
    follow_up_questions = []
    follow_up_indices = []
    for i, state in enumerate(workflow_states):
        if state.user_disclosed_info and state.user_response:
            complete_question = f"{state.question}\n\nAdditional information: {state.user_response}"
            follow_up_questions.append(complete_question)
            follow_up_indices.append(i)
    
    if follow_up_questions:
        print(f"Calling follow-up solver for {len(follow_up_questions)} examples...")
        follow_up_responses = solver.batch_solve(follow_up_questions, batch_size=batch_size)
        for idx, response in zip(follow_up_indices, follow_up_responses):
            state = workflow_states[idx]
            state.final_response = response
            state.final_response_tokens = len(solver.tokenizer.encode(response))
    
    # Calculate total tokens
    for state in workflow_states:
        state.total_tokens = state.first_response_tokens + state.final_response_tokens
    
    # Process baselines (similar conditional flow)
    baseline_states = {bt: [] for bt in baseline_types}
    
    # Initialize baseline states
    for baseline_type in baseline_types:
        for example in insufficient_examples:
            state = ExampleState(
                question=example['question'],
                original_question=example.get('original_question'),
                missing_info=_get_missing_info(example),
                ground_truth_sufficient=False,
                ground_truth_answer=example.get('original_answer')
            )
            baseline_states[baseline_type].append(state)
    
    # Baseline questions and initial responses
    baseline_questions = {bt: [] for bt in baseline_types}
    for example in insufficient_examples:
        question = example['question']
        for baseline_type in baseline_types:
            if baseline_type == 'always_prompt':
                baseline_questions[baseline_type].append(
                    f"{question}\n\n{baseline_agents[baseline_type].system_prompt}"
                )
                baseline_states[baseline_type][len(baseline_questions[baseline_type])-1].prompt_appended = True
            elif baseline_type == 'just_answer':
                baseline_questions[baseline_type].append(question)
                baseline_states[baseline_type][len(baseline_questions[baseline_type])-1].prompt_appended = False
            else:
                raise ValueError(f"Unrecognized baseline type: {baseline_type}")
    
    baseline_responses = {}
    for baseline_type in baseline_types:
        baseline_responses[baseline_type] = solver.batch_solve(
            baseline_questions[baseline_type], batch_size=batch_size
        )
        # Store responses
        for state, response in zip(baseline_states[baseline_type], baseline_responses[baseline_type]):
            state.first_response = response
            state.first_response_tokens = len(solver.tokenizer.encode(response))
    
    # Evaluate baseline initial responses
    print("Evaluating initial baseline responses...")
    for baseline_type in baseline_types:
        for state in tqdm(baseline_states[baseline_type], desc=f"Evaluating {baseline_type}", unit="example"):
            state.evaluation_initial = evaluator.evaluate(
                state.question,
                state.first_response,
                ground_truth_sufficient=False,
                missing_info=state.missing_info,
                ground_truth_answer=None
            )
            state.agent_asked_question = state.evaluation_initial.get('asked_question', False)
            state.agent_attempted_answer = state.evaluation_initial.get('answer_attempted', False)
    
    # User simulator for baselines (only if agent asked)
    print("Interacting with user simulator for baselines...")
    for baseline_type in baseline_types:
        for state in tqdm(baseline_states[baseline_type], desc=f"User sim {baseline_type}", unit="example"):
            if state.agent_asked_question:
                user_response, disclosed = user_simulator.respond(
                    state.first_response,
                    state.original_question or state.question,
                    state.missing_info
                )
                state.user_response = user_response
                state.user_disclosed_info = disclosed
            else:
                state.user_response = None
                state.user_disclosed_info = False
    
    # Follow-up solver for baselines (only if user disclosed)
    baseline_follow_up_dicts = {}
    for baseline_type in baseline_types:
        follow_up_questions = []
        follow_up_indices = []
        for i, state in enumerate(baseline_states[baseline_type]):
            if state.user_disclosed_info and state.user_response:
                complete_question = f"{state.question}\n\nAdditional information: {state.user_response}"
                follow_up_questions.append(complete_question)
                follow_up_indices.append(i)
        
        if follow_up_questions:
            follow_up_responses = solver.batch_solve(follow_up_questions, batch_size=batch_size)
            for idx, response in zip(follow_up_indices, follow_up_responses):
                state = baseline_states[baseline_type][idx]
                state.final_response = response
                state.final_response_tokens = len(solver.tokenizer.encode(response))
            baseline_follow_up_dicts[baseline_type] = {idx: baseline_states[baseline_type][idx].final_response 
                                                      for idx in follow_up_indices}
        else:
            baseline_follow_up_dicts[baseline_type] = {}
        
        # Calculate total tokens for baselines
        for state in baseline_states[baseline_type]:
            state.total_tokens = state.first_response_tokens + state.final_response_tokens
    
    # Compile results from states
    for i, (example, workflow_state) in enumerate(zip(insufficient_examples, workflow_states)):
        result = workflow_state.to_dict()
        
        # Add baseline results
        for baseline_type in baseline_types:
            baseline_state = baseline_states[baseline_type][i]
            result[f'baseline_{baseline_type}'] = {
                'first_response': baseline_state.first_response,
                'final_response': baseline_state.final_response,
                'tokens_used': baseline_state.total_tokens,
                'prompt_appended': baseline_state.prompt_appended,
                'user_provided_info': baseline_state.user_disclosed_info,
                'evaluation_initial': baseline_state.evaluation_initial,
                'evaluation_final_answer': baseline_state.evaluation_final_answer,
            }
        
        results.append(result)
    
    return results


def _evaluate_and_save_stage1(examples, evaluator, output_path):
    """
    Stage 1: Run GPTEvaluator on all examples and save results.
    
    For sufficient examples:
    - evaluation_initial: Already done in _process_sufficient_examples (skip if exists)
    - evaluation_final_answer: GPTEvaluator.evaluate_answer_only() on final_response (if exists)
      (final_response exists if agent incorrectly asked a question)
    
    For insufficient examples:
    - evaluation_initial: Already done in _process_insufficient_examples (skip if exists)
    - evaluation_final_answer: GPTEvaluator.evaluate_answer_only() on final_response (if exists)
      (final_response exists if user disclosed missing information)
    
    For baselines:
    - evaluation_initial: Already done in processing functions (skip if exists)
    - evaluation_final_answer: GPTEvaluator.evaluate_answer_only() on final_response (if exists)
      (final_response exists if agent asked question and got user response)
    """
    print("\n" + "="*80)
    print("STAGE 1: GPT EVALUATION")
    print("="*80)
    
    evaluated_examples = []
    
    for ex in tqdm(examples, desc="Evaluating examples", unit="example"):
        is_sufficient = ex['ground_truth_sufficient']
        missing_info = None if is_sufficient else ex.get('missing_info')
        gt_answer = ex.get('ground_truth_answer')  # Preserved from original dataset
        
        # Workflow evaluation
        workflow = ex['workflow']
        
        # Initial evaluation: only if not already done (for insufficient examples, it's done in processing)
        if workflow.get('evaluation_initial') is None:
            eval_initial = evaluator.evaluate(
                ex['question'],
                workflow['first_response'],
                is_sufficient,
                missing_info=missing_info,
                ground_truth_answer=gt_answer if is_sufficient else None,
            )
            workflow['evaluation_initial'] = eval_initial
        
        # Final answer evaluation: for any example with final_response
        # (sufficient examples can have final_response if agent incorrectly asked a question)
        if workflow.get('final_response'):
            eval_final_answer = evaluator.evaluate_answer_only(
                ex.get('original_question', ex['question']),
                workflow['final_response'],
                ground_truth_answer=ex.get('ground_truth_answer'),
            )
            workflow['evaluation_final_answer'] = eval_final_answer
        
        # Baseline evaluations
        for key in ex.keys():
            if key.startswith('baseline_'):
                baseline = ex[key]
                
                # Initial evaluation: only if not already done
                if baseline.get('evaluation_initial') is None:
                    baseline_eval_initial = evaluator.evaluate(
                        ex['question'],
                        baseline['first_response'],
                        is_sufficient,
                        missing_info=missing_info,
                        ground_truth_answer=gt_answer if is_sufficient else None,
                    )
                    baseline['evaluation_initial'] = baseline_eval_initial
                
                # Final answer evaluation: for any baseline with final_response
                # (sufficient examples can have final_response if agent incorrectly asked a question)
                if baseline.get('final_response'):
                    if ex.get('ground_truth_answer'):
                        baseline_eval_final = evaluator.evaluate_answer_only(
                            ex.get('original_question', ex['question']),
                            baseline['final_response'],
                            ground_truth_answer=ex.get('ground_truth_answer'),
                        )
                        baseline['evaluation_final_answer'] = baseline_eval_final
        
        evaluated_examples.append(ex)
    
    # Save Stage 1 results
    # Get experiment_config from first example if it exists
    experiment_config = {}
    if evaluated_examples and 'experiment_config' in evaluated_examples[0]:
        experiment_config = evaluated_examples[0]['experiment_config']
        # Remove it from the example (it's at the top level)
        del evaluated_examples[0]['experiment_config']
    
    stage1_output = {
        'experiment_config': experiment_config,
        'examples': evaluated_examples
    }
    
    converted_output = _convert_to_python_types(stage1_output)
    with open(output_path, 'w') as f:
        json.dump(converted_output, f, indent=2)
    
    print(f"\nStage 1 results saved to: {output_path}")
    return evaluated_examples


def run_experiment(
    probe_path,
    probe_type,
    layer_idx,
    model_name,
    dataset_path,
    user_simulator_type,
    baseline_types,
    device='cpu',
    max_examples=None,
    train_config=None,
    openai_client=None,
        batch_size=8,
    output_path=None,
):
    """
    Run workflow experiment (Stage 1 + Stage 2).
    
    Stage 1: Generate responses and run GPTEvaluator, save to JSON.
    Stage 2: Compute metrics from saved results.

    Args:
        probe_path: Path to probe file or experiment directory
        probe_type: 'linear' or 'mlp'
        layer_idx: Layer index (None to use best from metrics)
        model_name: Model name for workflow agent
        dataset_path: Path to test dataset JSON
        user_simulator_type: 'gpt5', 'rag', or 'gpt5_recall'
        baseline_types: List of baseline types ['just_answer', 'always_prompt']
        device: 'cpu' or 'cuda'
        max_examples: Maximum number of examples (None for all)
        train_config: Training configuration (e.g., 'train_on_ALL')
        openai_client: OpenAI client instance
        batch_size: Batch size for parallel processing
        output_path: Path to save results JSON

    Returns:
        dict: Results with metrics for each method
    """
    print("="*80)
    print("PROBE-BASED WORKFLOW EXPERIMENT")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Split into sufficient and insufficient
    sufficient_examples = [ex for ex in dataset if ex.get('is_sufficient', True)]
    insufficient_examples = [ex for ex in dataset if not ex.get('is_sufficient', True)]
    
    if max_examples:
        # Limit both proportionally
        total = len(sufficient_examples) + len(insufficient_examples)
        if total > max_examples:
            sufficient_ratio = len(sufficient_examples) / total
            sufficient_examples = sufficient_examples[:int(max_examples * sufficient_ratio)]
            insufficient_examples = insufficient_examples[:max_examples - len(sufficient_examples)]
    
    print(f"Processing {len(sufficient_examples)} sufficient examples")
    print(f"Processing {len(insufficient_examples)} insufficient examples")

    # Load probe
    probe_device = 'cpu'
    print(f"\nLoading probe...")
    if layer_idx is None:
        if os.path.isdir(probe_path):
            experiment_dir = probe_path
        else:
            experiment_dir = os.path.dirname(os.path.dirname(probe_path))
        probe_inference = load_best_probe(
            experiment_dir, probe_type, model_name, train_config, device=probe_device
        )
    else:
        probe_inference = ProbeInference(
            probe_path=probe_path,
            probe_type=probe_type,
            layer_idx=layer_idx,
            model_name=model_name,
            device=probe_device
        )

    # Create solver
    print(f"Loading solver: {model_name} (device: {device})")
    solver = MathSolver(model_name, device=device)

    # Create agents
    workflow_agent = WorkflowAgent(probe_inference, solver)
    baseline_agents = {bt: BaselineAgent(solver) for bt in baseline_types}
    
    # Create user simulator (only for insufficient examples)
    user_simulator = None
    if insufficient_examples:
        print(f"Creating user simulator: {user_simulator_type}")
    if user_simulator_type == 'gpt5':
        user_simulator = GPT5Simulator()
    elif user_simulator_type == 'rag':
        user_simulator = RAGSimulator()
    elif user_simulator_type == 'gpt5_recall':
        user_simulator = GPT5RecallSimulator()
    else:
        raise ValueError(f"Unknown user simulator type: {user_simulator_type}")

    # Create evaluator
    evaluator = GPTEvaluator(openai_client=openai_client)

    # Stage 1: Process examples and generate responses
    print("\n" + "="*80)
    print("STAGE 1: GENERATING RESPONSES")
    print("="*80)

    all_results = []
    
    # Process sufficient examples
    if sufficient_examples:
        print(f"\nProcessing {len(sufficient_examples)} sufficient examples...")
        sufficient_results = _process_sufficient_examples(
            sufficient_examples,
            workflow_agent,
            solver,
            baseline_agents,
            baseline_types,
            evaluator,
            batch_size=batch_size
        )
        all_results.extend(sufficient_results)
    
    # Process insufficient examples
    if insufficient_examples:
        print(f"\nProcessing {len(insufficient_examples)} insufficient examples...")
        insufficient_results = _process_insufficient_examples(
            insufficient_examples,
            workflow_agent,
            user_simulator,
            solver,
            baseline_agents,
            baseline_types,
            evaluator,
            batch_size=batch_size
        )
        all_results.extend(insufficient_results)
    
    # Add experiment config to first example
    if all_results:
        all_results[0]['experiment_config'] = {
            'probe_path': probe_path,
            'probe_type': probe_type,
            'model_name': model_name,
            'dataset_path': dataset_path,
            'user_simulator': user_simulator_type,
            'baselines': baseline_types,
            'device': device,
            'num_examples': len(all_results),
            'evaluator': 'gpt-4o-mini'
        }
    
    # Stage 1: Evaluate with GPTEvaluator
    stage1_path = output_path or 'workflow_stage1_results.json'
    evaluated_examples = _evaluate_and_save_stage1(
        all_results,
        evaluator,
        stage1_path
    )
    
    # Stage 2: Compute metrics
    print("\n" + "="*80)
    print("STAGE 2: COMPUTING METRICS")
    print("="*80)
    
    metrics = compute_metrics_from_json(stage1_path)
    print_summary(metrics)
    
    # Compile final results
    results = {
        'experiment_config': all_results[0].get('experiment_config', {}) if all_results else {},
        'summary_metrics': metrics,
        'examples': evaluated_examples
    }
    
    # Save final results (with metrics)
    if output_path:
        converted_results = _convert_to_python_types(results)
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        print(f"\nFinal results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run probe-based workflow experiment')
    parser.add_argument('--probe_path', type=str, required=True,
                        help='Path to probe file or experiment directory')
    parser.add_argument('--probe_type', type=str, default='linear',
                        choices=['linear', 'mlp'], help='Probe type')
    parser.add_argument('--layer_idx', type=int, default=None,
                        help='Layer index (None to use best from metrics)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for workflow agent')
    parser.add_argument('--train_config', type=str, default=None,
                        help='Training configuration (e.g., train_on_ALL)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to test dataset JSON')
    parser.add_argument('--user_simulator', type=str, default='gpt5',
                        choices=['gpt5', 'rag', 'gpt5_recall'],
                        help='User simulator type')
    parser.add_argument('--baselines', type=str, nargs='+',
                        default=['just_answer', 'always_prompt'],
                        choices=['just_answer', 'always_prompt'],
                        help='Baseline types to run')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Maximum number of examples')
    parser.add_argument('--output_path', type=str, default='workflow_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (or use OPENAI_API_KEY env var)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')

    args = parser.parse_args()

    # Setup OpenAI client
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n✗ Error: OPENAI_API_KEY not found")
        print("Set it via --openai_api_key or environment variable")
        return
    openai_client = OpenAI(api_key=api_key)

    results = run_experiment(
        probe_path=args.probe_path,
        probe_type=args.probe_type,
        layer_idx=args.layer_idx,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        user_simulator_type=args.user_simulator,
        baseline_types=args.baselines,
        device=args.device,
        max_examples=args.max_examples,
        train_config=args.train_config,
        openai_client=openai_client,
        batch_size=args.batch_size,
        output_path=args.output_path
    )
    
    print(f"\nExperiment complete!")


if __name__ == '__main__':
    main()
