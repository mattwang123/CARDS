"""
Model configuration registry

This file defines the 4 core models for our experiments:
1. qwen2.5-math-1.5b: Math-specialized small model (1.5B)
2. qwen2.5-1.5b: General-purpose small model (1.5B, non-math)
3. llama-3.2-3b-instruct: General instruction-tuned model (3B)
4. qwen2.5-math-7b: Math-specialized larger model (7B)
"""

MODELS = {
    'qwen2.5-math-1.5b': {
        'name': 'Qwen/Qwen2.5-Math-1.5B-Instruct',
        'num_layers': 28,
        'hidden_size': 1536,
        'requires_auth': False,
        'description': 'Math-specialized 1.5B model from Qwen team'
    },
    'qwen2.5-1.5b': {
        'name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'num_layers': 28,
        'hidden_size': 1536,
        'requires_auth': False,
        'description': 'General-purpose 1.5B instruction-tuned model'
    },
    'llama-3.2-3b-instruct': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'num_layers': 28,
        'hidden_size': 3072,
        'requires_auth': True,
        'description': 'Meta Llama 3.2 3B instruction-tuned model'
    },
    'qwen2.5-math-7b': {
        'name': 'Qwen/Qwen2.5-Math-7B-Instruct',
        'num_layers': 28,
        'hidden_size': 3584,
        'requires_auth': False,
        'description': 'Math-specialized 7B model from Qwen team'
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
    """Print all available models with details"""
    print("="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    for name, config in MODELS.items():
        print(f"\n{name}:")
        print(f"  HF name:     {config['name']}")
        print(f"  Layers:      {config['num_layers']}")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Auth req:    {config['requires_auth']}")
        print(f"  Description: {config['description']}")
    print("="*80)


if __name__ == '__main__':
    list_models()
