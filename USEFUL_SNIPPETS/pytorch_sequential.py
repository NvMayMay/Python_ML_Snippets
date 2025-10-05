'''
Simple PyTorch Sequential Model Example with Multiple Layers
'''


import torch
import torch.nn as nn
# Define a simple sequential model using nn.Sequential
# This creates a feed-forward neural network with linear layers and activations
model = nn.Sequential(
    nn.Linear(784, 128),  # First layer: 784 input features (e.g., flattened 28x28 image) to 128 hidden units
    nn.ReLU(),            # Activation function: ReLU for non-linearity
    nn.Linear(128, 64),   # Second layer: 128 hidden units to 64 hidden units
    nn.ReLU(),            # Another ReLU activation
    nn.Linear(64, 10),    # Output layer: 64 hidden units to 10 output classes (e.g., for MNIST)
    nn.Softmax(dim=1)     # Softmax activation for multi-class classification (probabilities)
)

# Example usage: Create a random input tensor (batch of 1, 784 features)
input_tensor = torch.randn(1, 784)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape (should be [1, 10] for 10 classes)
print("Output shape:", output.shape)
print("Output:", output)