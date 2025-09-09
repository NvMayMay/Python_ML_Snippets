# This builds a Keras Functional API model
# Install required libraries
# python3 -m pip install tensorflow
import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

'''
The Functional API is a way to create models that are more flexible than the Sequential API.
The user must define the input, hidden, and output layers, and then create the model by specifying the inputs and outputs.
The hidden layers can defined in a number of ways, i.e. dense, ReLU activation etc.
'''

# Define the input layer with 20 features
input_layer = Input(shape=(20,))
# Define hidden laters with 64 neurons and ReLU activation
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
# Define the output layer with 1 neuron and sigmoid activation for binary classification
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
# Instantiate the model by specifying the inputs and outputs, uncomment the line below to print the model summary
model = Model(inputs=input_layer, outputs=output_layer)
#model.summary()

# Compile the model with binary crossentropy loss and Adam optimizer with accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on training data, with random data for demonstration purposes
import numpy as np
X_train = np.random.rand(1000, 20) # 1000 samples, 20 features
y_train = np.random.randint(2, size=(1000, 1)) # 1000 binary labels
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on test data, with random data for demonstration purposes
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
# uncomment the lines below to print the test loss and accuracy
# print(f'Test loss: {loss}') 
# print(f'Test accuracy: {accuracy}') 

'''
Can add dropout layers, batch normalization, and other layers as needed.
Below are examples of adding dropout and batch normalization layers.
'''
### Dropout example
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
# Define the input layer
input_layer = Input(shape=(20,))
# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)
# Add a Dropout layer
dropout_layer = Dropout(rate=0.5)(hidden_layer)
# Add another hidden layer after Dropout
hidden_layer2 = Dense(64, activation='relu')(dropout_layer)
# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
# Summary of the model, uncomment the line below to print
# model.summary()

### Batch Normalization example
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
# Define the input layer
input_layer = Input(shape=(20,))
# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)
# Add a BatchNormalization layer
batch_norm_layer = BatchNormalization()(hidden_layer)
# Add another hidden layer after BatchNormalization
hidden_layer2 = Dense(64, activation='relu')(batch_norm_layer)
# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
# Summary of the model
model.summary()