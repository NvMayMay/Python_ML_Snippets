# This builds an autoencoder using Keras to compress and decompress images from the MNIST dataset.
# python3 -m pip install tensorflow keras numpy matplotlib scikit-learn

import numpy as np 
from tensorflow.keras.datasets import mnist 
# Load the dataset 
(x_train, _), (x_test, _) = mnist.load_data() 
# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255. 
# Flatten the images from 28x28 to 784-dimensional vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) 
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) 

# Build the autoencoder model, takes 784 dimensional input and compresses it to 32 dimensions
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
# Encoder 
input_layer = Input(shape=(784,)) 
encoded = Dense(64, activation='relu')(input_layer) 
# Bottleneck 
bottleneck = Dense(32, activation='relu')(encoded) 
# Decoder 
decoded = Dense(64, activation='relu')(bottleneck) 
output_layer = Dense(784, activation='sigmoid')(decoded) 
# Autoencoder model 
autoencoder = Model(input_layer, output_layer) 
# Compile the model 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 
# Summary of the model 
autoencoder.summary()

# 1. Define the Encoder:
# Create an input layer with 784 neurons.
# Add a Dense layer with 64 neurons and ReLU activation.
# 2. Define the Bottleneck:
# Add a Dense layer with 32 neurons and ReLU activation.
# 3. Define the Decoder:
# Add a Dense layer with 64 neurons and ReLU activation.
# Add an output layer with 784 neurons and sigmoid activation.
# 4. Compile the Model:
# Use the Adam optimizer and binary crossentropy loss.

#Train model
autoencoder.fit(
    x_train, x_train,  
    epochs=25,  
    batch_size=256,  
    shuffle=True,  
    validation_data=(x_test, x_test)
)

# Evaluate the model on test data
import matplotlib.pyplot as plt 
# Predict the test data 
reconstructed = autoencoder.predict(x_test) 
# Visualize the results 
n = 10  # Number of digits to display 
plt.figure(figsize=(20, 4)) 
for i in range(n): 
    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(x_test[i].reshape(28, 28)) 
    plt.gray() 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 

    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(reconstructed[i].reshape(28, 28)) 
    plt.gray() 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
plt.show()

# Freeze all layers of the autoencoder
for layer in autoencoder.layers:
    layer.trainable = False

# Check trainable status of each layer, uncomment to see the status
for i, layer in enumerate(autoencoder.layers):
    # print(f"Layer {i}: {layer.name}, Trainable = {layer.trainable}")
    pass

# Unfreeze the top layers of the encoder
for layer in autoencoder.layers[-4:]:
    layer.trainable = True 
# Compile the model again
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 
# Train the model again
autoencoder.fit(x_train, x_train,  
                epochs=10,  
                batch_size=256,  
                shuffle=True,  
                validation_data=(x_test, x_test))

import numpy as np
import matplotlib.pyplot as plt
# Add noise to the data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# Train the autoencoder with noisy data
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=20,
    batch_size=512,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Denoise the test images
reconstructed_noisy = autoencoder.predict(x_test_noisy)
# Visualize the results
n = 10  # Number of digits to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Display noisy images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display original images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()