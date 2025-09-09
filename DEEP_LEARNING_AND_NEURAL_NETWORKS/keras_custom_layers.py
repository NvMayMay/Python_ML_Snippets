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
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)  # Apply a linear transformation followed by