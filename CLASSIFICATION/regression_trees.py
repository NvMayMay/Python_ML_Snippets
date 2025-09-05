#install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#This dataset is for New York City taxi trip records, the goal is to predict the tip amount based on other features

# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
ssl._create_default_https_context = ssl._create_unverified_context
raw_data = pd.read_csv(url)
raw_data

# use correlation of target variable against input features, delete hash in below section to print
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))
#plt.show()

# normalize data to standardize feature scales
# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')
# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)
# get the feature matrix used for training
X = proc_data.values
# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor
# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)

# fit the model to the training data
dt_reg.fit(X_train, y_train)

#   evaluate models using test data, delete hash in below section to print
#   run inference using the sklearn model
y_pred = dt_reg.predict(X_test)
#   evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
#print('MSE score : {0:.3f}'.format(mse_score))
r2_score = dt_reg.score(X_test,y_test)
#print('R^2 score : {0:.3f}'.format(r2_score))