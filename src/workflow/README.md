# Training-Free Intervention Pipeline

This directory contains the implementation of probe-guided intervention for detecting and handling logically insufficient questions.

## Overview

The pipeline tests whether linear probes trained to detect insufficiency can be used to guide models toward acknowledging missing information instead of hallucinating answers.

## Pipeline Steps

1. **Load Best Layers**: Read optimal probe layers from previous probing experiments
2. **Load Data**: Load UMWP and GSM8K datasets (train + test splits)
3. **Load Embeddings**: Load cached embeddings from server (`/export/fs06/psingh54/CARDS/src/data/embeddings/`)
4. **Train Probes**: Train linear probes on train data at best layer per model-dataset pair
5. **Detect Insufficiency**: For each test insufficient question, use probe to predict
6. **Intervene**: If probe detects insufficiency, reprompt model with warning
7. **Judge Response**: Use GPT-4o-mini to evaluate if model acknowledged and correctly identified missing info
8. **Save Results**: Save probes (.pkl), detailed results, and metrics

## Usage

### Run All Models and Datasets (8 experiments)

```bash
cd /export/fs06/psingh54/CARDS/src/workflow
python intervention_pipeline.py --device cuda --output_dir ../../experiments/intervention
```

### Run Specific Model or Dataset

```bash
# Single model
python intervention_pipeline.py --model qwen2.5-math-1.5b --device cuda

# Single dataset
python intervention_pipeline.py --dataset umwp --device cuda

# Both
python intervention_pipeline.py --model llama-3.2-3b-instruct --dataset gsm8k --device cuda
```

## Requirements

- Must run on server with access to:
  - `/export/fs06/psingh54/CARDS/src/data/embeddings/` (cached embeddings)
  - `/export/fs06/psingh54/CARDS/src/experiments/all_probes_linear/best_layers_linear.json` (best layer config)
- OpenAI API key in `.env` for GPT-4o-mini judging
- GPU recommended for model inference

## Outputs

### Per Model-Dataset Pair

**`{model}_{dataset}_intervention.json`**:
```json
{
  "metrics": {
    "total_insufficient_questions": 520,
    "probe_metrics": {
      "detected": 468,
      "missed": 52,
      "detection_rate": 0.90,
      "miss_rate": 0.10
    },
    "intervention_metrics": {
      "acknowledged": 345,
      "correctly_identified": 280,
      "acknowledgment_rate": 0.737,  // Of detected
      "identification_rate_given_acknowledged": 0.812,  // Of acknowledged
      "identification_rate_overall": 0.598  // Of detected
    },
    "timing": {
      "probe_avg": 0.015,  // seconds
      "model_avg": 1.2,
      "judge_avg": 0.8,
      "total": 1040.5
    }
  },
  "detailed_results": [...]
}
```

**`probes/{model}_{dataset}_layer{N}.pkl`**: Trained sklearn probe (can be loaded with pickle)

### Summary

**`intervention_summary.json`**: Aggregated metrics across all model-dataset pairs

## Evaluation Metrics

### Probe Performance
- **Detection Rate**: % of insufficient questions correctly identified by probe
- **Miss Rate**: % of insufficient questions probe failed to detect (false negatives)

### Intervention Performance (on probe-detected questions)
- **Acknowledgment Rate**: % of questions where model acknowledged insufficiency
- **Identification Rate (given acknowledged)**: % where model correctly identified missing info (of those that acknowledged)
- **Identification Rate (overall)**: % where model correctly identified missing info (of all detected by probe)

### Timing
- **Probe Time**: Time for probe inference per question
- **Model Time**: Time for model generation per question
- **Judge Time**: Time for GPT-4o-mini judgment per question

## Judge Prompt

The GPT-4o-mini judge evaluates two things:

1. **Acknowledgment**: Did the model recognize the question is insufficient?
   - YES: Model states something is missing and attempts to identify it
   - NO: Model ignores warning and tries to solve with assumptions

2. **Correct Identification**: Did the model identify the right missing information?
   - YES: Model identifies the specific value/constraint that was removed
   - NO: Model identifies wrong information or is too vague
   - N/A: Model didn't acknowledge insufficiency

The judge has access to:
- Original sufficient question
- Insufficient question (what model saw)
- Ground truth of what was removed
- Model's response

## Example Flow

**Input (Insufficient Question)**:
> "John has some apples and buys 3 more. How many does he have?"

**Probe Detection**: Predicts insufficient (probability: 0.92)

**Reprompt to Model**:
> Question: John has some apples and buys 3 more. How many does he have?
>
> Note: This question is believed to be logically insufficient - it may be missing critical information or constraints needed for a unique answer.
>
> Please identify what information is missing and what you would need to solve this problem.

**Model Response**:
> "I cannot determine the exact answer because the initial number of apples John has is not specified. I would need to know how many apples John started with to calculate the total."

**GPT-4o-mini Judgment**:
```json
{
  "acknowledged": "YES",
  "correctly_identified": "YES",
  "explanation": "Model correctly identified that initial number of apples is missing"
}
```

## Notes

- Only tests on **insufficient** questions (not sufficient ones)
- Probe must detect insufficiency for intervention to occur
- If probe misses (predicts sufficient), that question is skipped for intervention
- All timings are measured separately for analysis
