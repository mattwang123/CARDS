"""
Probe inference module for detecting insufficient questions
"""
import pickle
import os
import json
import numpy as np
import torch

from models.extractor import HiddenStateExtractor
from models.probe import MLPProbe
from models.config import get_model_config


class ProbeInference:
    """
    Load and use trained probes for sufficiency prediction
    """

    def __init__(self, probe_path, probe_type, layer_idx, model_name, device='cpu'):
        """
        Args:
            probe_path: Path to probe file (without .pkl or .pt extension)
            probe_type: 'linear' or 'mlp'
            layer_idx: Layer index the probe was trained on
            model_name: Model name from config (for embedding extraction)
            device: 'cpu' or 'cuda'
        """
        self.probe_type = probe_type
        self.layer_idx = layer_idx
        self.device = device

        # Load probe
        if probe_type == 'linear':
            with open(probe_path + '.pkl', 'rb') as f:
                self.probe = pickle.load(f)
        else:  # mlp
            # Load model config to get hidden size
            config = get_model_config(model_name)
            hidden_size = config['hidden_size']
            
            # Create probe architecture
            self.probe = MLPProbe(input_dim=hidden_size, hidden_dim=128, num_classes=2)
            self.probe.load_state_dict(torch.load(probe_path + '.pt', map_location=device))
            self.probe.to(device)
            self.probe.eval()

        # Load extractor for embedding extraction
        self.extractor = HiddenStateExtractor(model_name, device=device)

    def predict(self, question):
        """
        Predict if question is sufficient

        Args:
            question: Question text string

        Returns:
            tuple: (is_sufficient: bool, confidence: float)
                is_sufficient: True if sufficient, False if insufficient
                confidence: Probability of sufficiency (0-1)
        """
        import time
        
        # Extract embeddings for all layers
        extract_start = time.time()
        embeddings = self.extractor.extract_hidden_states(
            question, layers='all', pooling='last_token'
        )
        extract_time = time.time() - extract_start

        # Get embedding for the specific layer
        layer_embedding = embeddings[self.layer_idx]  # numpy array

        # Run probe prediction
        probe_start = time.time()
        if self.probe_type == 'linear':
            # LogisticRegression: predict_proba returns (n_samples, n_classes)
            proba = self.probe.predict_proba(layer_embedding.reshape(1, -1))[0]
            is_sufficient = self.probe.predict(layer_embedding.reshape(1, -1))[0] == 1
            confidence = proba[1] if is_sufficient else proba[0]
        else:  # mlp
            # Convert to tensor
            x = torch.FloatTensor(layer_embedding).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.probe(x)
                probs = torch.softmax(logits, dim=1)[0]
                is_sufficient = torch.argmax(logits, dim=1)[0].item() == 1
                confidence = probs[1].item() if is_sufficient else probs[0].item()
        probe_time = time.time() - probe_start
        
        total_time = time.time() - extract_start
        print(f"      [TIMING] Probe breakdown - extract: {extract_time:.3f}s, probe: {probe_time:.3f}s, total: {total_time:.3f}s")

        return is_sufficient, confidence

    def predict_batch(self, questions):
        """
        Predict sufficiency for a batch of questions

        Args:
            questions: List of question text strings

        Returns:
            list: List of (is_sufficient, confidence) tuples
        """
        import time
        
        batch_start = time.time()
        
        # Extract embeddings for all questions in batch
        extract_start = time.time()
        all_embeddings = []
        for question in questions:
            embeddings = self.extractor.extract_hidden_states(
                question, layers='all', pooling='last_token'
            )
            all_embeddings.append(embeddings[self.layer_idx])
        extract_time = time.time() - extract_start
        
        # Run probe predictions in batch
        probe_start = time.time()
        results = []
        for layer_embedding in all_embeddings:
            if self.probe_type == 'linear':
                proba = self.probe.predict_proba(layer_embedding.reshape(1, -1))[0]
                is_sufficient = self.probe.predict(layer_embedding.reshape(1, -1))[0] == 1
                confidence = proba[1] if is_sufficient else proba[0]
            else:  # mlp
                x = torch.FloatTensor(layer_embedding).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.probe(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    is_sufficient = torch.argmax(logits, dim=1)[0].item() == 1
                    confidence = probs[1].item() if is_sufficient else probs[0].item()
            results.append((is_sufficient, confidence))
        probe_time = time.time() - probe_start
        
        total_time = time.time() - batch_start
        print(f"    [TIMING] Batch probe ({len(questions)} questions) - extract: {extract_time:.3f}s, probe: {probe_time:.3f}s, total: {total_time:.3f}s ({total_time/len(questions):.3f}s per question)")
        
        return results


def load_best_probe(experiment_dir, probe_type='linear', model_name=None, train_config=None, device='cpu'):
    """
    Load the best probe from an experiment directory
    
    Supports two directory structures:
    1. Old structure (from train_probes.py): {experiment_dir}/probes_{type}/
    2. New structure (from run_all_probes.py): {experiment_dir}/{model_name}/train_on_{dataset}/probes_{type}/

    Args:
        experiment_dir: Path to experiment directory
        probe_type: 'linear' or 'mlp'
        model_name: Model name for embedding extraction (required for new structure)
        train_config: Training configuration (e.g., 'train_on_umwp', 'train_on_ALL')
                     If None, tries old structure first, then looks for any train_on_* directory
        device: 'cpu' or 'cuda' (default: 'cpu')

    Returns:
        ProbeInference instance
    """
    if model_name is None:
        raise ValueError("model_name must be provided")
    
    # Try new structure first (from run_all_probes.py)
    if train_config:
        probes_dir = os.path.join(experiment_dir, model_name, train_config, f'probes_{probe_type}')
    else:
        # Try to find any train_on_* directory
        model_dir = os.path.join(experiment_dir, model_name)
        if os.path.exists(model_dir):
            # Look for train_on_* directories
            train_configs = [d for d in os.listdir(model_dir) if d.startswith('train_on_')]
            if train_configs:
                # Prefer 'train_on_ALL' if available, otherwise use first one
                if 'train_on_ALL' in train_configs:
                    train_config = 'train_on_ALL'
                else:
                    train_config = train_configs[0]
                probes_dir = os.path.join(model_dir, train_config, f'probes_{probe_type}')
            else:
                probes_dir = None
        else:
            probes_dir = None
    
    # Fall back to old structure (from train_probes.py)
    if probes_dir is None or not os.path.exists(probes_dir):
        probes_dir = os.path.join(experiment_dir, f'probes_{probe_type}')
    
    metrics_path = os.path.join(probes_dir, 'all_metrics.json')
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Could not find all_metrics.json in {probes_dir}. "
            f"Expected structure: {experiment_dir}/{{model_name}}/train_on_{{dataset}}/probes_{{type}}/ "
            f"or {experiment_dir}/probes_{{type}}/"
        )

    # Load metrics to find best layer
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Find layer with highest F1 score
    best_metric = max(metrics, key=lambda m: m['test_f1'])
    best_layer = best_metric['layer']

    # Get probe path
    probe_path = os.path.join(probes_dir, f'layer_{best_layer}_probe')

    # Create ProbeInference instance
    return ProbeInference(
        probe_path=probe_path,
        probe_type=probe_type,
        layer_idx=best_layer,
        model_name=model_name,
        device=device
    )


def load_probe_from_path(probe_path, probe_type, layer_idx, model_name, device='cpu'):
    """
    Load a probe directly from a file path
    
    Args:
        probe_path: Path to probe file (without .pkl/.pt extension)
        probe_type: 'linear' or 'mlp'
        layer_idx: Layer index
        model_name: Model name for embedding extraction
        device: 'cpu' or 'cuda'
    
    Returns:
        ProbeInference instance
    """
    return ProbeInference(
        probe_path=probe_path,
        probe_type=probe_type,
        layer_idx=layer_idx,
        model_name=model_name,
        device=device
    )

