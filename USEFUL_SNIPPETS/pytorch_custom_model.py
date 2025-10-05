'''
Custom PyTorch model definition using nn.Module subclassing.
'''

import torch  
import torch.nn as nn  
# Define a custom neural network model by subclassing nn.Module
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()  # Call the parent class constructor
        # Define layers in the __init__ method
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.softmax = nn.Softmax(dim=1)  # Softmax for output probabilities

    def forward(self, x):
        # Define the forward pass in the forward method
        x = self.fc1(x)  # Pass input through first layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through second layer
        x = self.softmax(x)  # Apply softmax for classification
        return x  # Return the output

# Example usage
# Create an instance of the model
model = CustomModel(input_size=784, hidden_size=128, output_size=10)

# Create a random input tensor (batch of 1, 784 features)
input_tensor = torch.randn(1, 784)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape (should be [1, 10] for 10 classes)
print("Output shape:", output.shape)
print("Output:", output)

'''
other layers include:
Linear, Conv2d, MaxPool2d, BatchNorm2d, Dropout, ReLU, 
Sigmoid, Tanh, LSTM, RNN, Embedding, Flatten, ConvTranspose2d, 
AdaptiveAvgPool2d, LayerNorm, GRU, Softmax, CrossEntropyLoss, MSELoss, 
BCEWithLogitsLoss, Sequential, ModuleList, ModuleDict, Identity, ZeroPad2d, 
AvgPool2d, Upsample, Bilinear, PixelShuffle, GroupNorm, InstanceNorm2d, 
SyncBatchNorm, TransformerEncoderLayer, TransformerDecoderLayer, 
MultiheadAttention, PositionalEncoding, NLLLoss, L1Loss, HuberLoss, 
KLDivLoss, CosineSimilarity, PairwiseDistance, TripletMarginLoss, CTCLoss, 
PoissonNLLLoss, GaussianNLLLoss, NegativeLogLikelihood, MarginRankingLoss, 
HingeEmbeddingLoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss, 
MultiMarginLoss, SmoothL1Loss, SoftMarginLoss, CosineEmbeddingLoss, 
TripletMarginWithDistanceLoss.


other activation functions include:
ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU, SELU, PReLU, RReLU, 
Hardtanh, Hardshrink, Softplus, Softshrink, Softsign, Tanhshrink, 
Threshold, LogSigmoid, Hardsigmoid, Hardswish, SiLU, Mish, Swish, GLU, 
Bilinear, Identity.
'''