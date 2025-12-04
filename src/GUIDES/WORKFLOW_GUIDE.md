# Probe-Based Workflow System Guide

## Dataset Structure

The workflow system expects datasets in a unified JSON format. Each example must follow the structure below.

### For Sufficient Examples

**Required fields:**
- `question` (str): The math problem text
- `answer` (str): The ground truth answer (format: `"#### <number>"` for GSM8K)
- `is_sufficient` (bool): Must be `true`

**Example:**
```json
{
  "question": "At its current growth rate, a certain plant will be 80 feet tall after a year. If the plant grows at the same rate every month and is currently 20 feet tall, what's its monthly growth rate in feet?",
  "answer": "#### 5",
  "is_sufficient": true
}
```

### For Insufficient Examples

**Required fields:**
- `question` (str): The insufficient/incomplete problem text
- `answer` (str): Should be `"N/A"` (indicates unsolvable)
- `is_sufficient` (bool): Must be `false`
- `original_question` (str): The complete, solvable version
- `original_answer` (str): The correct answer to the original question (format: `"#### <number>"` for GSM8K) — **required for final answer evaluation**
- `removed_value` (str): The specific missing information (e.g., `"$200 flat fee"`)
- `removed_description` (str, optional): Description of why this info is needed

**Example:**
```json
{
  "question": "Mark is trying to choose between two venues for a surprise party for his wife. The first venue charges a flat fee, regardless of how many guests attend. The second charges a per-person rate, which includes food for each guest. However, the first venue does not include food, which Mark estimates will cost $5 for each person who attends. How many guests are necessary for the two venues to be equal in cost?",
  "answer": "N/A",
  "is_sufficient": false,
  "original_question": "Mark is trying to choose between two venues for a surprise party for his wife. The first venue charges a flat fee of $200, regardless of how many guests attend. While the second charges, $25 per person who attends. However, the first venue does not include food, which Mark estimates will cost $5 for each person who attends. At the second venue, food for each guest is already included in the price. How many guests are necessary for the two venues to be equal in cost?",
  "original_answer": "#### 10",
  "removed_value": "$200 flat fee for the first venue",
  "removed_description": "The flat fee is crucial as it provides the base cost for the first venue, necessary for comparison with the per-person cost of the second venue."
}
```

## Overview

The probe-based workflow system detects incomplete math word problems **before** expensive model inference, reducing computation cost while improving accuracy by encouraging users to disclose missing information.

### Key Goals

1. **Reduce computation cost**: Skip full inference when probe detects insufficient questions
2. **Improve accuracy**: Encourage users to disclose missing information through targeted questions
3. **Evaluate behavior**: Measure how well agents distinguish between sufficient and insufficient questions

### Two-Stage Architecture

The system uses a **two-stage design** for clean separation of concerns:

- **Stage 1** (`run_experiment.py`): Generate responses and run GPTEvaluator, save results to JSON
- **Stage 2** (`pipeline_evaluation.py`): Compute metrics from saved results (no API calls)

This allows:
- Re-running metrics with different aggregations without regenerating responses
- Easy extension of evaluation logic
- Clear separation between generation and analysis

## Core Workflow Process

### Main Workflow (Probe-Based Agent)

1. **User provides a puzzle/question** (may be incomplete)
2. **Probe classification**: Extract embeddings → Run probe prediction (sufficient/insufficient)
3. **Conditional system prompt**: 
   - If **insufficient** → Append: *"This query might be incomplete. Could you follow up with a concise question to the user about what might be missing?"*
   - If **sufficient** → No prompt appended
4. **Model rollout**: Run inference on question (with or without prompt) → `first_response`
5. **User simulator interaction**:
   - User simulator receives agent's `first_response`
   - Decides whether to disclose missing information (based on simulator type)
   - If information disclosed → Run follow-up solver call → `final_response`
6. **Final response**: Either `first_response` (if user didn't provide info) or `final_response` (if user provided info)

### Baselines

Baselines follow the **same workflow** but with different prompt strategies:

- **`just_answer`**: Never appends prompt, always solves directly
- **`always_prompt`**: Always appends prompt regardless of probe output

Both baselines also interact with user simulators (same as workflow).

## Components

### 1. Workflow Agent (`src/workflow/agent.py`)

Main agent that integrates probe + model inference:

```python
WorkflowAgent.process(question):
  1. Run probe prediction → (is_sufficient, confidence)
  2. If not is_sufficient:
     - Append system prompt to question
  3. Run solver on (modified) question → response
  4. Return: {
      'response': response,
      'tokens_used': total_tokens,
      'probe_prediction': is_sufficient,
      'probe_confidence': confidence,
      'prompt_appended': bool
    }
```

### 2. Baseline Agents (`src/workflow/baselines.py`)

- **`BaselineAgent`**: Base class for baseline methods
- **`just_answer`**: Always solves without checking sufficiency
- **`always_prompt`**: Always appends system prompt

### 3. User Simulators (`src/workflow/user_simulators.py`)

Three types of user simulators to test different interaction patterns:

#### GPT-5 Simulator (`gpt5`)
- **Behavior**: Sophisticated user simulated by GPT-4o (proxy for GPT-5)
- **Decision logic**: Decides whether to disclose missing information based on agent's response
- **Key characteristic**: Only discloses information if workflow agent **explicitly asks** about what's missing
- **Requires**: OpenAI API key (`OPENAI_API_KEY`)
- **Use case**: Most realistic user behavior

#### RAG Simulator (`rag`)
- **Behavior**: Simple cosine similarity matching
- **Decision logic**: 
  - Embeds agent's query and missing_info
  - Computes cosine similarity
  - Returns missing_info probabilistically based on similarity
- **Key characteristic**: Success rate proportional to similarity
- **Requires**: OpenAI API key (for embeddings only)
- **Use case**: Testing with constrained user behavior

#### GPT-5 Recall Simulator (`gpt5_recall`)
- **Behavior**: GPT-5 with access to a recall tool (RAG-like database)
- **Decision logic**: 
  - GPT-5 generates a query based on agent's response
  - RAG retrieval uses cosine similarity between query embedding and missing sentence embedding
  - Probability of successful recall is proportional to cosine similarity
- **Key characteristic**: Middle ground between RAG and GPT-5
- **Requires**: OpenAI API key
- **Use case**: Testing with realistic user that has imperfect memory/recall

### 4. GPT Evaluator (`src/workflow/gpt_evaluator.py`)

**The only evaluator** used by the workflow. Uses GPT-4o-mini to judge:

- **Redundancy**: Did the agent ask vs answer appropriately?
  - For **insufficient** questions: redundancy = 1 if agent asks a question, 0 if attempts to answer
  - For **sufficient** questions: redundancy = 1 if agent answers, 0 if asks unnecessary question
- **Answer correctness**: When ground-truth answer available, checks if numeric answer matches
- **Question match**: For insufficient questions, checks if clarification targets true missing info

**Two evaluation modes:**

1. **`evaluate()`**: Full behavioral evaluation on `first_response`
   - Returns: `redundancy`, `answer_attempted`, `answer_correct`, `asked_question`, `question_matches_missing_info`
2. **`evaluate_answer_only()`**: Numeric answer correctness only
   - Used for `final_response` after user provides missing info
   - Returns: `answer_attempted`, `answer_correct`

### 5. Probe Inference (`src/workflow/probe_inference.py`)

Loads and uses trained probes for sufficiency prediction:

- Supports both linear (pickle) and MLP (PyTorch) probes
- Auto-detects best layer from metrics
- Handles both old and new probe directory structures
- Batch prediction support for efficiency

### 6. Pipeline Evaluation (`src/workflow/pipeline_evaluation.py`)

Stage 2: Computes aggregate metrics from Stage 1 JSON:

- **Unified metric computation**: Same function for workflow and baselines
- **Metrics computed**:
  - `redundancy`: Mean redundancy score
  - `answer_accuracy`: Fraction with correct answers
  - `question_match_rate`: Fraction where clarification matches missing info (workflow only)
  - `avg_tokens`, `total_tokens`: Token cost metrics

## Evaluation Metrics

### 1. Redundancy (Behavioral Correctness)

Binary metric (0/1) indicating appropriate ask-vs-answer behavior:

- **For insufficient questions**:
  - ✅ **Correct (1)**: Agent asks a question
  - ❌ **Incorrect (0)**: Agent attempts to answer the problem directly
- **For sufficient questions**:
  - ✅ **Correct (1)**: Agent provides an answer
  - ❌ **Incorrect (0)**: Agent asks an unnecessary question

### 2. Answer Correctness/Accuracy

Numeric answer correctness:

- **For sufficient questions**: Check `evaluation_initial['answer_correct']` on `first_response`
- **For insufficient questions**: Check `evaluation_final_answer['answer_correct']` on `final_response` (if user provided info)
- Extracts numerical answer from response (e.g., from `#### 42` format)
- Compares to ground truth answer

### 3. Question Match Rate

For insufficient questions only (workflow):

- Checks if agent's clarification question targets the true missing information
- From `evaluation_initial['question_matches_missing_info']`
- Only computed for workflow (not baselines)

### 4. Total Inference Cost

Measured by **output tokens only** (agent's response):

- **Components**:
  - Initial solver call output tokens
  - Follow-up solver call output tokens (if user provides info)
- **Note**: Does NOT count user tokens, system messages, or manually prepended prompts
- **Goal**: Lower cost than baselines (especially `just_answer` which always does full inference)

## Data Format

### Input Dataset Format

All datasets use unified JSON format:

```json
{
  "question": "Jake has 9 fewer peaches than Steven...",
  "answer": "#### 27",
  "is_sufficient": true
}
```

For insufficient examples:
```json
{
  "question": "Ali is collecting bottle caps. He has red ones and green ones...",
  "answer": "N/A",
  "is_sufficient": false,
  "original_question": "Ali is collecting bottle caps. He has 125 bottle caps...",
  "original_answer": "#### 125",
  "removed_value": "125",
  "removed_description": "Missing critical information: total count"
}
```

**Required fields:**
- `question`: The math word problem text
- `is_sufficient`: Boolean flag (True = sufficient, False = insufficient)
- `answer`: Answer string (numeric for sufficient, "N/A" for insufficient)

**For insufficient examples:**
- `original_question`: Original complete question before modification
- `original_answer`: Original correct answer (for evaluation)
- `removed_value`: What was removed (for analysis)
- `removed_description`: Description of missing information

### Stage 1 Output Format

Each example in the results JSON contains:

```json
{
  "question": "...",
  "ground_truth_sufficient": true/false,
  "ground_truth_answer": "#### 27",  // For sufficient or insufficient with original_answer
  "missing_info": "125",  // Only for insufficient
  "workflow": {
    "first_response": "...",  // Agent's initial response
    "final_response": "...",  // Final answer after user interaction (or None)
    "tokens_used": 123,
    "probe_prediction": true/false,
    "probe_confidence": 0.87,
    "prompt_appended": true/false,
    "user_provided_info": true/false,
    "evaluation_initial": {  // GPTEvaluator.evaluate() on first_response
      "redundancy": 1.0,
      "answer_attempted": true,
      "answer_correct": true,
      "asked_question": false,
      "question_matches_missing_info": null
    },
    "evaluation_final_answer": {  // GPTEvaluator.evaluate_answer_only() on final_response
      "answer_attempted": true,
      "answer_correct": true
    }  // Or null if not applicable
  },
  "baseline_just_answer": {
    "first_response": "...",
    "final_response": "...",  // Can be None if user didn't respond
    "tokens_used": 45,
    "prompt_appended": false,
    "user_provided_info": true/false,
    "evaluation_initial": { ... },
    "evaluation_final_answer": { ... }  // Or null
  },
  "baseline_always_prompt": { ... }
}
```

**Key points:**
- **Sufficient questions**: `evaluation_final_answer` is `None` (not applicable)
- **Insufficient questions**: `evaluation_final_answer` exists if `final_response` exists (user provided info)
- **Baselines**: Same structure as workflow (they also interact with user simulators)

## Running Experiments

### Prerequisites

1. **Trained Probes**: Ensure probes are trained and saved
   ```bash
   cd src
   python run_all_probes.py --probe_type linear --pooling mean --device cuda
   ```

2. **Dataset**: Ensure test dataset exists (e.g., GSM8K)
   ```bash
   # Preprocess GSM8K (includes original_answer for insufficient examples)
   python scripts/preprocess_gsm8k.py
   ```

3. **OpenAI API Key** (for GPT simulators):
   ```bash
   export OPENAI_API_KEY=your_key_here
   # Or add to .env file
   ```

### Basic Usage

```bash
cd src
python workflow/run_experiment.py \
    --probe_path experiments/all_probes_linear_max \
    --probe_type linear \
    --model_name qwen2.5-math-1.5b \
    --dataset_path data/processed/gsm8k/gsm8k_test.json \
    --user_simulator gpt5 \
    --baselines just_answer always_prompt \
    --device cuda \
    --max_examples 100 \
    --output_path workflow_results.json
```

### Command-Line Arguments

**Required:**
- `--probe_path`: Path to probe experiment directory or specific probe file
- `--model_name`: Model name for workflow agent (e.g., `qwen2.5-math-1.5b`)
- `--dataset_path`: Path to test dataset JSON

**Optional:**
- `--probe_type`: `linear` or `mlp` (default: `linear`)
- `--layer_idx`: Specific layer index (default: `None` to use best from metrics)
- `--train_config`: Training configuration (e.g., `train_on_ALL`, default: auto-detect)
- `--user_simulator`: `gpt5`, `rag`, or `gpt5_recall` (default: `gpt5`)
- `--baselines`: List of baselines to run (default: `just_answer always_prompt`)
- `--device`: `cpu` or `cuda` (default: `cpu`)
- `--max_examples`: Maximum number of examples (default: `None` for all)
- `--batch_size`: Batch size for parallel processing (default: `8`)
- `--output_path`: Path to save results JSON (default: `workflow_results.json`)
- `--openai_api_key`: OpenAI API key (or use `OPENAI_API_KEY` env var)

### Probe Path Options

**Option 1: Experiment directory (recommended)**
```bash
--probe_path experiments/all_probes_linear_max
--train_config train_on_ALL
```
Auto-detects best layer from metrics.

**Option 2: Specific probe file**
```bash
--probe_path experiments/all_probes_linear_max/qwen2.5-math-1.5b/train_on_ALL/probes_linear/layer_18_probe
--layer_idx 18
```

### Processing Both Sufficient and Insufficient Examples

The system automatically:
- Splits dataset into sufficient and insufficient examples
- Processes sufficient examples (no user simulator needed)
- Processes insufficient examples (with user simulator)
- Combines results for unified evaluation

## Results and Metrics

### Stage 1 Output

Saves JSON with:
- `experiment_config`: Configuration used
- `examples`: List of example results with evaluations

### Stage 2 Output

Computes and prints:
- **Redundancy**: Mean redundancy score (0-1)
- **Answer accuracy**: Fraction with correct answers
- **Question match rate**: Fraction where clarification matches missing info (workflow only)
- **Avg tokens**: Average output tokens per example
- **Total tokens**: Total output tokens

### Example Output

```
================================================================================
METRICS SUMMARY
================================================================================

Workflow:
  Redundancy: 0.8500
  Answer accuracy: 0.7200
  Question match rate: 0.6800
  Avg tokens: 245.30

Baseline (just_answer):
  Redundancy: 0.3000
  Answer accuracy: 0.4500
  Avg tokens: 580.20

Baseline (always_prompt):
  Redundancy: 0.7000
  Answer accuracy: 0.6500
  Avg tokens: 620.10
```

## Key Design Decisions

### Why Two Stages?

1. **Separation of concerns**: Generation vs. evaluation
2. **Flexibility**: Re-run metrics without regenerating responses
3. **Extensibility**: Easy to add new metrics or aggregations
4. **Debugging**: Can inspect Stage 1 results before computing metrics

### Why GPT Evaluator?

- **Flexible**: Can judge nuanced behavior (asking vs. answering)
- **Accurate**: Better than regex patterns for detecting questions/answers
- **Consistent**: Single evaluator for all metrics (no regex fallback)

### Why User Simulators?

- **Controlled testing**: Reproducible user behavior
- **Cost-effective**: No need for human annotators
- **Variety**: Test different interaction patterns (RAG, GPT-5, GPT-5 recall)

### Why Baselines Also Interact?

- **Fair comparison**: Baselines get same opportunity to ask questions
- **Realistic**: Tests if baselines can also benefit from user interaction
- **Comprehensive**: Evaluates both prompt strategies (always vs. conditional)

## Troubleshooting

### Probes Not Found

**Error**: `FileNotFoundError: Could not find all_metrics.json`

**Solution**: 
- Verify probes were saved: Check `experiments/all_probes_linear_max/{model}/train_on_ALL/probes_linear/`
- Ensure probe training completed successfully
- Check probe path matches experiment directory structure

### API Key Errors

**Error**: `ValueError: OPENAI_API_KEY not found`

**Solution**:
- Add to `~/.bashrc`: `export OPENAI_API_KEY=your_key`
- Source it: `source ~/.bashrc`
- Or use RAG simulator (requires API key for embeddings, but cheaper)

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'workflow'`

**Solution**:
- Ensure you're in `src/` directory
- Check `PYTHONPATH=.` is set
- Verify `src/workflow/__init__.py` exists

### Out of Memory

**Error**: CUDA out of memory

**Solution**:
- Reduce `--batch_size` (default: 8)
- Use `--device cpu` for probe inference (probe uses CPU by default)
- Reduce `--max_examples` for testing

### Missing Ground Truth Answers

**Error**: `ValueError: GPTEvaluator.evaluate called with ground_truth_sufficient=True but ground_truth_answer=None`

**Solution**:
- Ensure dataset has `answer` field for sufficient examples
- For insufficient examples, ensure `original_answer` field exists
- Check dataset preprocessing saved original answers correctly

## File Structure

```
src/workflow/
├── agent.py              # WorkflowAgent: probe + solver integration
├── baselines.py          # BaselineAgent: just_answer, always_prompt
├── user_simulators.py    # GPT5Simulator, RAGSimulator, GPT5RecallSimulator
├── gpt_evaluator.py      # GPTEvaluator: only evaluator used
├── pipeline_evaluation.py # Stage 2: metric computation
├── probe_inference.py    # Probe loading and inference
├── run_experiment.py     # Stage 1: main experiment script
└── tests/                # Test suite
```

## Testing

Run automated tests:
```bash
cd src
pytest workflow/tests/ -v
```

Run specific test:
```bash
pytest workflow/tests/test_integration.py -v
```

## Next Steps

1. **Run experiments**: Test workflow vs baselines on GSM8K
2. **Analyze results**: Compare accuracy and cost metrics
3. **Extend**: Add new user simulators or evaluation metrics
4. **Optimize**: Tune probe thresholds or system prompts

## References

- **Probe Training**: See `src/run_all_probes.py`
- **Dataset Preprocessing**: See `src/scripts/preprocess_*.py`
- **Judge Prompts**: See `src/judge_prompts/gpt_workflow_judge.md`
- **Cursor Rules**: See `.cursor/rules/workflow-evaluator.mdc`


