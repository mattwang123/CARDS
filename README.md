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

## Train Linear Probes

Train linear or MLP probes on extracted embeddings:

```bash
python main.py \
  --train_embeddings data/embeddings/umwp_train_qwen2.5-math-1.5b_last_token.npy \
  --train_labels data/insufficient_dataset_umwp/umwp_train.json \
  --test_embeddings data/embeddings/umwp_test_qwen2.5-math-1.5b_last_token.npy \
  --test_labels data/insufficient_dataset_umwp/umwp_test.json \
  --probe_type linear \
  --output_dir experiments/probe_results
```

**Args:**
- `--train_embeddings`: Path to train embeddings .npy file
- `--train_labels`: Path to train labels JSON file
- `--test_embeddings`: Path to test embeddings .npy file
- `--test_labels`: Path to test labels JSON file
- `--output_dir`: Directory to save results (default: experiments/probe_results)
- `--probe_type`: 'linear' (logistic regression) or 'mlp' (2-layer neural network) (default: linear)
- `--hidden_dim`: Hidden dimension for MLP probe (default: 128)
- `--num_epochs`: Number of training epochs for MLP (default: 50)
- `--lr`: Learning rate for MLP (default: 0.001)
- `--device`: cpu or cuda (default: cpu)
- `--visualize_all`: Generate 3D PCA for ALL layers (can be slow)
- `--skip_pca`: Skip PCA visualization for faster execution

**Output:**
```
experiments/probe_results/
├── pca_3d/                           # 3D PCA plots for all layers (if --visualize_all)
├── mech_interp/                      # Mechanistic interpretability analysis
│   ├── activation_statistics.png
│   ├── cosine_similarity.png
│   └── weight_importance.png         # Linear probes only
├── results/
│   ├── layer_XX_probe.pkl            # Trained probe models
│   ├── layer_performance.png         # Performance across layers
│   └── confusion_matrices_top3.png
└── all_metrics.json                  # Complete metrics for all layers
```

**What it does:**
1. Trains one probe per layer (28 probes for 28-layer model)
2. Evaluates which layers encode sufficiency information best
3. Generates activation statistics and cosine similarity analysis
4. Visualizes weight importance for linear probes
5. Identifies best-performing layers

---

## Advanced Mechanistic Interpretability

Run comprehensive mech interp analyses on embeddings:

```bash
python run_advanced_interp.py \
  --train_embeddings data/embeddings/umwp_train_qwen2.5-math-1.5b_last_token.npy \
  --train_labels data/insufficient_dataset_umwp/umwp_train.json \
  --test_embeddings data/embeddings/umwp_test_qwen2.5-math-1.5b_last_token.npy \
  --test_labels data/insufficient_dataset_umwp/umwp_test.json \
  --output_dir experiments/probe_results
```

**Args:**
- `--train_embeddings`: Path to train embeddings .npy file
- `--train_labels`: Path to train labels JSON file
- `--test_embeddings`: Path to test embeddings .npy file
- `--test_labels`: Path to test labels JSON file
- `--output_dir`: Base output directory (default: experiments/probe_results)
- `--skip_analyses`: List of analyses to skip (optional)
  - Choices: selectivity, control, subspace, knn, cka, lda, umap, intrinsic

**Generated Analyses:**

1. **Selectivity Analysis**: Which dimensions are most correlated with sufficiency?
2. **Control Task Probing**: Validates results aren't spurious (random/shuffled labels)
3. **Subspace Dimensionality**: Minimum dimensions needed for good performance
4. **K-Nearest Neighbors**: Local structure and clustering analysis
5. **CKA (Centered Kernel Alignment)**: Layer similarity matrix
6. **LDA Visualization**: Supervised dimensionality reduction (better than PCA)
7. **UMAP Visualization**: Non-linear embedding visualization
8. **Intrinsic Dimensionality**: Estimates true dimensionality of embedding manifold

**Output:**
```
experiments/probe_results/mech_interp/
├── selectivity_analysis.png          # Most selective dimensions per layer
├── control_task_probing.png          # Validation against random labels
├── subspace_dimensionality.png       # Performance vs # dimensions
├── knn_analysis.png                  # k-NN accuracy across layers
├── cka_similarity.png                # Layer-to-layer similarity heatmap
├── lda_visualization.png             # LDA projections for key layers
├── umap_visualization.png            # UMAP 2D embeddings
└── intrinsic_dimensionality.png      # Estimated ID per layer
```

**Note:** UMAP requires `pip install umap-learn` (optional)

---

## Token-Level Attribution

Analyze which specific words/tokens in questions trigger the sufficiency detectors:

```bash
python run_token_attribution.py \
  --data_path data/insufficient_dataset_umwp/umwp_test.json \
  --model_name qwen2.5-math-1.5b \
  --num_examples 6 \
  --device cpu
```

**Args:**
- `--data_path`: Path to JSON data file (test set recommended)
- `--probe_dir`: Directory with trained probes (default: experiments/probe_results/results)
- `--metrics_path`: Path to metrics JSON (default: experiments/probe_results/all_metrics.json)
- `--model_name`: Model name from config (default: qwen2.5-math-1.5b)
- `--layer`: Layer to analyze (default: auto-selects best F1 layer)
- `--output_dir`: Output directory (default: experiments/probe_results/mech_interp)
- `--num_examples`: Number of examples to visualize (default: 6)
- `--device`: cpu or cuda (default: cpu)

**What it does:**
1. Auto-selects the best-performing probe layer (or uses specified layer)
2. Loads the trained linear probe weights
3. Projects each token's embedding onto the probe direction
4. Visualizes which tokens push toward "Sufficient" (blue) vs "Insufficient" (red)

**Output:**
```
experiments/probe_results/mech_interp/
└── token_attribution_layer18.png    # Token-level heatmap for 6 examples
```

**Example insights:**
- Numbers and quantities → Blue (push toward sufficient)
- Vague references ("it", "that value") → Red (push toward insufficient)
- Question words ("how many", "what is") → Variable based on context

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
- Linear and MLP probe architectures
- **Training pipeline for probes** (main.py)
- **Layer-wise probe evaluation**
- **Basic mechanistic interpretability** (activation stats, cosine similarity, weight importance)
- **Advanced mechanistic interpretability** (selectivity, control tasks, LDA, UMAP, CKA, k-NN, intrinsic dim)
- **Token-level attribution** (which words trigger sufficiency detectors)

🔜 **Next Steps:**
- Activation steering experiments
- Intervention analysis (causal probing)
- Generate progress report for milestone

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
│   ├── analyze_responses.py         # Response analysis logic
│   ├── plot_embeddings.py           # 3D PCA visualization
│   ├── plot_probe_results.py        # Layer performance plots
│   ├── mech_interp.py               # Basic mech interp (activation stats, cosine sim)
│   └── advanced_mech_interp.py      # Advanced analyses (selectivity, LDA, UMAP, etc.)
│
├── results/                         # Model outputs (generated, not committed)
├── experiments/                     # Probe results (generated, not committed)
│
├── main_analyze.py                  # Model analysis script
├── main.py                          # Probe training script
├── run_advanced_interp.py           # Advanced mech interp runner
├── run_token_attribution.py         # Token-level attribution analysis
└── environment.yml                  # Conda environment
```

---

## Notes

- **Results are not committed to git** (added to .gitignore)
- **UMWP raw data is committed** for easy setup
- **Generated datasets are not committed** (run preprocessing scripts to create)
- Use `--max_examples` for faster testing on large datasets
- Model responses are saved for later analysis
