# Install required libraries
# python3 -m pip install tensorflow matplotlib scipy
# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset for training images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Normalize the pixel values for augmentation
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Display a sample of the training images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.show()

# Create sample.png image for augmentation demonstration
from PIL import Image, ImageDraw
# Create a blank white image
image = Image.new('RGB', (224, 224), color = (255, 255, 255))
# Draw a red square
draw = ImageDraw.Draw(image)
draw.rectangle([(50, 50), (174, 174)], fill=(255, 0, 0))
# Save the image
image.save('sample.jpg')
import numpy as np 
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
# Load a sample image 
img_path = 'sample.jpg' 
img = load_img(img_path) 
x = img_to_array(img) 
x = np.expand_dims(x, axis=0) 

# These augmentations can be applied to the image of a red square and produced the following images:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# Load the sample image
img_path = 'sample.jpg'
img = load_img(img_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
# Create an instance of ImageDataGenerator with basic augmentations
datagen = ImageDataGenerator(
rotation_range=40,           # Randomly rotate images in the range (degrees, 0 to 40)
width_shift_range=0.2,       # Randomly shift images horizontally by up to 20% of the width
height_shift_range=0.2,      # Randomly shift images vertically by up to 20% of the height
shear_range=0.2,             # Apply random shearing transformations up to 0.2 radians
zoom_range=0.2,              # Randomly zoom in or out on images by up to 20%
horizontal_flip=True,        # Randomly flip images horizontally
fill_mode='nearest'          # Fill in new pixels after transformations using the nearest pixel values
)
# Generate batches of augmented images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# Feature-wise and sample-wise normalization
# Create an instance of ImageDataGenerator with normalization options
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    samplewise_std_normalization=True
)
# Load the sample image again and fit the generator (normally done on the training set)
datagen.fit(x)
# Generate batches of normalized images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# Custom Data Augmentation Function, generates random noise
# Define a custom data augmentation function
def add_random_noise(image):
    noise = np.random.normal(0, 0.1, image.shape)
    return image + noise
# Create an instance of ImageDataGenerator with the custom augmentation
datagen = ImageDataGenerator(preprocessing_function=add_random_noise)
# Generate batches of augmented images with noise
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break

plt.show()