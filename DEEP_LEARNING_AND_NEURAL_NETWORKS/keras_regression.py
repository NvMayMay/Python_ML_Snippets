# This builds a basic neural network to predict the strength of concrete
# Install required libraries
# python3 -m pip install numpy pandas tensorflow_cpu==2.18.0
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

import warnings
warnings.simplefilter('ignore', FutureWarning)

filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

#Check how many data points, uncomment the line below to print
# print(concrete_data.shape)
# Check the dataset for any null values, uncomment the line below to print
# print(concrete_data.isnull().sum())

# split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()

# save the number of predictors to n_cols
n_cols = predictors_norm.shape[1] # number of predictors

# Build a neural network model
# define regression model, 2 hidden layers with 50 nodes each, this can be edited.
# More nodes and more layers generally improve the accuracy of the model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)