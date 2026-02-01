"""
Model configuration registry
"""

MODELS = {
    'llama-3.2-1b': {
        'name': 'meta-llama/Llama-3.2-1B',
        'num_layers': 16,
        'hidden_size': 2048,
        'requires_auth': True  # May need HuggingFace token
    },
    'llama-3.2-1b-instruct': {
        'name': 'meta-llama/Llama-3.2-1B-Instruct',
        'num_layers': 16,
        'hidden_size': 2048,
        'requires_auth': True
    },
    'llama-3.2-3b': {
        'name': 'meta-llama/Llama-3.2-3B',
        'num_layers': 28,
        'hidden_size': 3072,
        'requires_auth': True
    },
    'llama-3.2-3b-instruct': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'num_layers': 28,
        'hidden_size': 3072,
        'requires_auth': True
    },
    'gpt2': {
        'name': 'gpt2',
        'num_layers': 12,
        'hidden_size': 768,
        'requires_auth': False
    },
    'gpt2-medium': {
        'name': 'gpt2-medium',
        'num_layers': 24,
        'hidden_size': 1024,
        'requires_auth': False
    },
    'qwen2.5-math-1.5b': {
        'name': 'Qwen/Qwen2.5-Math-1.5B-Instruct',
        'num_layers': 28,
        'hidden_size': 1536,
        'requires_auth': False
    },
    'phi-3-mini': {
        'name': 'microsoft/Phi-3-mini-4k-instruct',
        'num_layers': 32,
        'hidden_size': 3072,
        'requires_auth': False
    },
    'gemma-2-2b': {
        'name': 'google/gemma-2-2b-it',
        'num_layers': 26,
        'hidden_size': 2304,
        'requires_auth': False
    },
    'qwen1.5-chat': {
        'name': 'Qwen/Qwen1.5-0.5B-Chat',
        'num_layers': 28,
        'hidden_size': 1536,
        'requires_auth': False
    }
}


def get_model_config(model_name):
    """
    Get configuration for a model

    Args:
        model_name: Name of the model (key in MODELS dict)

    Returns:
        dict: Model configuration

    Raises:
        ValueError: If model_name not found
    """
    if model_name not in MODELS:
        available = ', '.join(MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    return MODELS[model_name]


def list_models():
    """Print all available models"""
    print("Available models:")
    for name, config in MODELS.items():
        print(f"  {name}:")
        print(f"    HF name: {config['name']}")
        print(f"    Layers: {config['num_layers']}")
        print(f"    Hidden size: {config['hidden_size']}")
        print(f"    Auth required: {config['requires_auth']}")
        print()
