"""
Extract hidden states from frozen LLM for all layers
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # CHANGED: Needed for Loss/Perplexity
from tqdm import tqdm

# Handle both direct execution and module import
try:
    from .config import get_model_config
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.config import get_model_config


class HiddenStateExtractor:
    """Extract hidden states from frozen LLM"""

    def __init__(self, model_name, device='cpu'):
        """
        Args:
            model_name: Name of model from config
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device

        # Get model config
        self.config = get_model_config(model_name)

        print(f"Loading model: {self.config['name']}")
        print(f"Device: {device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'])

        # Add padding token if missing (for some models like GPT-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CHANGED: Use CausalLM to enable loss calculation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            output_hidden_states=True,
            device_map='auto' if device == 'cuda' else None,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        self.model.eval()

        print(f"Model loaded with {self.config['num_layers']} layers")

    def extract_hidden_states(self, text, layers='all', pooling='last_token'):
        """
        Extract hidden states AND metrics for a single text
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            # CHANGED: Pass labels to calculate loss/perplexity
            outputs = self.model(**inputs, labels=inputs['input_ids'])

        # CHANGED: Calculate metrics
        loss = outputs.loss.item()
        perplexity = torch.exp(outputs.loss).item()
        seq_len = inputs['attention_mask'].sum(dim=1).item()
        metrics = {'loss': loss, 'perplexity': perplexity, 'length': seq_len}

        # hidden_states is a tuple: (embedding_layer, layer_0, layer_1, ..., layer_N)
        hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        # Determine which layers to extract
        if layers == 'all':
            layer_indices = list(range(len(hidden_states)))
        else:
            layer_indices = layers

        # Extract and pool for each layer
        embeddings = {}
        for idx in layer_indices:
            layer_hidden = hidden_states[idx]  # Shape: (batch_size, seq_len, hidden_dim)

            if pooling == 'last_token':
                embedding = layer_hidden[0, seq_len - 1, :].cpu().numpy()
            elif pooling == 'mean':
                mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = layer_hidden * mask
                sum_hidden = masked_hidden.sum(dim=1)
                mean_hidden = sum_hidden / mask.sum(dim=1)
                embedding = mean_hidden[0].cpu().numpy()
            elif pooling == 'max':
                mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = layer_hidden.clone()
                masked_hidden[mask == 0] = -float('inf')
                max_hidden = masked_hidden.max(dim=1)[0]
                embedding = max_hidden[0].cpu().numpy()
            else:
                raise ValueError(f"Invalid pooling: {pooling}")

            embeddings[idx] = embedding

        # CHANGED: Return both embeddings and metrics
        return embeddings, metrics

    def extract_dataset(self, data, layers='all', pooling='last_token'):
        """
        Extract hidden states and metrics for entire dataset
        """
        all_embeddings = []
        all_metrics = []  # CHANGED: Store metrics

        print(f"Extracting embeddings with pooling={pooling}")
        print(f"Processing {len(data)} samples...")

        for item in tqdm(data):
            question = item['question']
            # CHANGED: Unpack metrics
            embeddings, metrics = self.extract_hidden_states(question, layers, pooling)

            # Stack embeddings from all layers
            layer_embeddings = [embeddings[i] for i in sorted(embeddings.keys())]
            all_embeddings.append(layer_embeddings)
            all_metrics.append(metrics)  # CHANGED: Save metrics

        embeddings_array = np.array(all_embeddings)

        print(f"Extracted embeddings shape: {embeddings_array.shape}")
        # CHANGED: Return tuple
        return embeddings_array, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Extract hidden states from frozen LLM')
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='Model name from config')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to GSM8K JSON file')
    parser.add_argument('--output_dir', type=str, default='data/embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--pooling', type=str, default='last_token',
                        choices=['last_token', 'mean', 'max'],
                        help='Pooling method for embeddings')
    parser.add_argument('--layers', type=str, default='all',
                        help='Which layers to extract (default: all)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Create extractor
    extractor = HiddenStateExtractor(args.model_name, args.device)

    # Extract embeddings
    # CHANGED: Unpack result
    embeddings, metrics = extractor.extract_dataset(
        data,
        layers=args.layers,
        pooling=args.pooling
    )

    # Save embeddings
    os.makedirs(args.output_dir, exist_ok=True)

    # Create filename
    split_name = os.path.basename(args.data_path).replace('.json', '')
    filename = f"{split_name}_{args.model_name}_{args.pooling}.npy"
    output_path = os.path.join(args.output_dir, filename)

    np.save(output_path, embeddings)
    print(f"\nSaved embeddings to: {output_path}")

    # CHANGED: Save metrics to separate JSON
    metrics_path = output_path.replace('.npy', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'model_hf_name': extractor.config['name'],
        'num_samples': embeddings.shape[0],
        'num_layers': embeddings.shape[1],
        'hidden_size': embeddings.shape[2],
        'pooling': args.pooling,
        'data_path': args.data_path,
        'metrics_path': metrics_path  # CHANGED: Reference metrics file
    }

    metadata_path = output_path.replace('.npy', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")


if __name__ == '__main__':
    main()