# CARDS - Usage Guide

## Setup

```bash
conda env create -f environment.yml
conda activate cards
```

---

## Data Preparation

### Option 1: UMWP Dataset (Recommended)

The UMWP (Unanswerable Math Word Problems) dataset is already included in this repository at `data/raw_umwp/StandardDataset.jsonl` (5,200 examples).

**Process the UMWP dataset:**

```bash
python data/preprocess_umwp.py
```

This creates:
- `data/insufficient_dataset_umwp/umwp_train.json` (4,160 examples: 2,080 sufficient, 2,080 insufficient)
- `data/insufficient_dataset_umwp/umwp_test.json` (1,040 examples: 520 sufficient, 520 insufficient)

**Args:**
- `--input_file`: Path to UMWP JSONL (default: data/raw_umwp/StandardDataset.jsonl)
- `--output_dir`: Where to save processed data (default: data/insufficient_dataset_umwp)
- `--train_ratio`: Train/test split ratio (default: 0.8)
- `--random_seed`: Random seed (default: 42)

---

### Option 2: GSM8K Dataset (GPT-4o Generated Insufficient Examples)

**Step 1: Download GSM8K Data**

```bash
python data/download.py --output_dir data/raw --num_samples 100
```

**Args:**
- `--output_dir`: Where to save data (default: data/raw)
- `--num_samples`: Number of samples (default: 100)

**Output:**
- `data/raw/gsm8k_train.json`
- `data/raw/gsm8k_test.json`

**Step 2: Create Insufficient Dataset (Requires OpenAI API)**

First, create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here
```

**Test mode (10 examples only):**
```bash
python data/create_insufficient.py --test_mode
```

**Full processing:**
```bash
python data/create_insufficient.py \
  --train_file data/raw/gsm8k_train.json \
  --test_file data/raw/gsm8k_test.json \
  --output_dir data/insufficient_dataset \
  --insufficient_ratio 0.5
```

**Output:**
- `data/insufficient_dataset/gsm8k_train_insufficient.json`
- `data/insufficient_dataset/gsm8k_test_insufficient.json`

---

## Model Analysis

Analyze model responses on insufficient dataset to detect overconfidence:

```bash
# Using UMWP dataset (recommended - larger dataset)
python main_analyze.py \
  --model_name qwen1.5-chat \
  --data_path data/insufficient_dataset_umwp/umwp_train.json \
  --max_examples 50 \
  --device cpu

# Using GSM8K dataset
python main_analyze.py \
  --model_name qwen1.5-chat \
  --data_path data/insufficient_dataset/gsm8k_train_insufficient.json \
  --device cpu
```

**Args:**
- `--model_name`: Model from config (default: llama-3.2-3b-instruct)
- `--data_path`: Path to insufficient dataset
- `--output_dir`: Where to save results (default: results/)
- `--device`: cpu or cuda (default: cpu)
- `--max_new_tokens`: Max tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_examples`: Limit number of examples to analyze (default: None = all)

**Output:**
- `results/[dataset_name]_responses.json` - All model responses
- `results/[dataset_name]_analysis.txt` - Summary report
- `results/[dataset_name]_analysis.json` - Detailed metrics

**What it analyzes:**
- **Sufficient questions**: Does model get correct answer?
- **Insufficient questions**: Does model answer anyway (overconfidence)?
- **Overconfidence rate**: % of insufficient questions answered

---

## Extract Hidden States (For Future Probe Training)

Extract embeddings from frozen LLM for all layers:

```bash
python models/extractor.py \
  --model_name gpt2 \
  --data_path data/insufficient_dataset_umwp/umwp_train.json \
  --output_dir data/embeddings \
  --pooling last_token \
  --device cpu
```

**Args:**
- `--model_name`: Model from config (gpt2, llama-3.2-1b, etc.)
- `--data_path`: Path to JSON data
- `--output_dir`: Where to save embeddings
- `--pooling`: last_token / mean / max
- `--layers`: all or specific layers
- `--device`: cpu or cuda

**Output:**
- `data/embeddings/[filename]_[model]_[pooling].npy` (shape: num_samples × num_layers × hidden_dim)
- `data/embeddings/[filename]_[model]_[pooling]_metadata.json`

---

## Available Models

See `models/config.py`:
- `gpt2`: 12 layers, 768 hidden dim (no auth required)
- `gpt2-medium`: 24 layers, 1024 hidden dim (no auth required)
- `qwen1.5-chat`: 28 layers, 1536 hidden dim (no auth required) ⭐
- `qwen2.5-math-1.5b`: 28 layers, 1536 hidden dim (no auth required, specialized for math)
- `phi-3-mini`: 32 layers, 3072 hidden dim (no auth required)
- `gemma-2-2b`: 26 layers, 2304 hidden dim (no auth required)
- `llama-3.2-1b-instruct`: 16 layers, 2048 hidden dim (requires HF token, gated)
- `llama-3.2-3b-instruct`: 28 layers, 3072 hidden dim (requires HF token, gated)

---

## Project Status

✅ **Completed:**
- Data download pipeline (GSM8K with ≥4 numbers filter)
- UMWP dataset preprocessing (train/test split)
- Insufficient dataset creation (GPT-4o for GSM8K)
- Model configuration (8 models available)
- Model inference pipeline
- Response analysis (overconfidence detection)
- Hidden state extraction (all layers, 3 pooling methods)
- 2-layer MLP probe architecture

🔜 **Next Steps:**
- Training pipeline for probes
- Layer-wise probe evaluation
- Visualizations (overconfidence plots, layer analysis)
- Activation steering experiments

---

## Directory Structure

```
CARDS/
├── data/
│   ├── raw_umwp/                    # UMWP raw data (committed to git)
│   │   └── StandardDataset.jsonl
│   ├── insufficient_dataset_umwp/   # Processed UMWP (generated, not committed)
│   │   ├── umwp_train.json
│   │   └── umwp_test.json
│   ├── raw/                         # GSM8K data (generated, not committed)
│   ├── insufficient_dataset/        # GPT-4o generated (generated, not committed)
│   └── embeddings/                  # Hidden states (generated, not committed)
│
├── models/
│   ├── config.py                    # Model registry
│   ├── inference.py                 # Model loading + generation
│   ├── answer_parser.py             # Extract answers from \boxed{}
│   ├── extractor.py                 # Hidden state extraction
│   └── probe.py                     # 2-layer MLP probe
│
├── viz/
│   └── analyze_responses.py         # Response analysis logic
│
├── results/                         # Model outputs (generated, not committed)
│
├── main_analyze.py                  # Main analysis script
└── environment.yml                  # Conda environment
```

---

## Notes

- **Results are not committed to git** (added to .gitignore)
- **UMWP raw data is committed** for easy setup
- **Generated datasets are not committed** (run preprocessing scripts to create)
- Use `--max_examples` for faster testing on large datasets
- Model responses are saved for later analysis
