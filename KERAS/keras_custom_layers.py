# This builds a Keras Functional API model
# Install required libraries
# python3 -m pip install tensorflow pydot graphviz
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

'''
This example demonstrates how to create a custom layer in Keras by subclassing the Layer class.
The custom layer will take an input tensor and apply a simple transformation to it.
It will then be integrated into a Sequential model.
'''
class CustomDenseLayer(Layer):  # Define a custom Keras layer by subclassing the base Layer class.
    def __init__(self, units=32): # number of neurons
        super(CustomDenseLayer, self).__init__()  # Initialize the parent Layer class.
        self.units = units  # Store the number of output units (neurons) for this layer.

    def build(self, input_shape):  # Called once to create the layer's weights based on input shape.
        self.w = self.add_weight(shape=(input_shape[-1], self.units),  # Create a trainable weight matrix for inputs to outputs.
                                 initializer='random_normal',  # Initialize weights with a random normal distribution.
                                 trainable=True)  # Allow weights to be updated during training.
        self.b = self.add_weight(shape=(self.units,),  # Create a trainable bias vector for each output unit.
                                 initializer='zeros',  # Initialize biases to zero.
                                 trainable=True)  # Allow biases to be updated during training.
    def call(self, inputs):  # Defines the computation performed at every call (forward pass).
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)  # Apply a linear transformation followed by ReLU activation.
    
# Create a Sequential model and add the custom layer
from tensorflow.keras.layers import Softmax
# Define the model with Softmax in the output layer
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),  # Hidden layer with ReLU activation
    Softmax()              # Output layer with Softmax activation for multi-class classification
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building:")
model.summary()

# Build the model to show parameters
model.build((1000, 20))
# Uncomment the line below to visualize the model architecture
# print("\nModel summary after building:")
# model.summary()

# Train the model on random data for demonstration purposes
import numpy as np 
# Generate random data 
x_train = np.random.random((1000, 20)) 
y_train = np.random.randint(10, size=(1000, 1)) 
# Convert labels to categorical one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) 
model.fit(x_train, y_train, epochs=10, batch_size=32) 

# Evaluate the model on random test data
# Generate random test data 
x_test = np.random.random((200, 20)) 
y_test = np.random.randint(10, size=(200, 1)) 
# Convert labels to categorical one-hot encoding 
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) 
# Evaluate the model 
loss = model.evaluate(x_test, y_test) 
# uncomment the line below to print the test loss
# print(f'Test loss: {loss}') 

# Can visualize the model architecture using plot_model
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

