"""
2-layer MLP probe for binary classification
"""
import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    """
    2-layer MLP for binary classification on LLM embeddings

    Architecture:
        input_dim -> hidden_dim (Linear + ReLU)
        hidden_dim -> 2 (Linear) for binary classification
    """

    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        """
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension (default: 128)
            num_classes: Number of output classes (default: 2 for binary)
        """
        super(MLPProbe, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Layer 1: input -> hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Layer 2: hidden -> output
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            logits of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        """
        Predict class labels

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x):
        """
        Predict class probabilities

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


def create_probe(input_dim, hidden_dim=128, num_classes=2):
    """
    Factory function to create a probe

    Args:
        input_dim: Dimension of input embeddings
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes

    Returns:
        MLPProbe instance
    """
    return MLPProbe(input_dim, hidden_dim, num_classes)


if __name__ == '__main__':
    # Test the probe
    print("Testing MLPProbe...")

    # Create a probe for embeddings of dimension 768 (like GPT-2)
    probe = MLPProbe(input_dim=768, hidden_dim=128, num_classes=2)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 768)

    # Forward pass
    logits = probe(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")

    # Predictions
    predictions = probe.predict(dummy_input)
    print(f"Predictions: {predictions}")

    # Probabilities
    probs = probe.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities:\n{probs}")

    print("\nProbe test passed!")
