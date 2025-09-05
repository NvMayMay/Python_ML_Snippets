#install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#import dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
ssl._create_default_https_context = ssl._create_unverified_context
data = pd.read_csv(file_path)

# Checking for null values to clean data
# delete hash from below to print
# print(data.isnull().sum())

# Dataset summary
# delete hash from below to print
# print(data.info())
# print(data.describe())

# Standardizing continuous numerical features to improve model performance
# finds all columns with float64 data type
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
# scales features to have mean=0 and standard deviation=1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
# Converting to a DataFrame with same column names
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
# Combining with the original dataset after dropping original continuous columns to keep scaled versions
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column
# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
# Predictions
y_pred_ova = model_ova.predict(X_test)
# Evaluation metrics for OvA, remove hash to print
#print("One-vs-All (OvA) Strategy")
#print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
# Predictions
y_pred_ovo = model_ovo.predict(X_test)
# Evaluation metrics for OvO, remove hash to print
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

#   use for loop to evaluate the difference in accuracy when using different test sizes
#test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
#for size in test_sizes:
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42, stratify=y)
#    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
#    model_ovo.fit(X_train, y_train)
#    y_pred_ovo = model_ovo.predict(X_test)
#    print(f"Test Size: {size}, Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")  

#   Feature importance for both models, OVA is immediately accessible
#   For OVO model, we need to aggregate coefficients from each binary classifier
#feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
#plt.barh(X.columns, feature_importance)
#plt.title("Feature Importance")
#plt.xlabel("Importance")
#plt.show()
#   For One vs One model
#   Collect all coefficients from each underlying binary classifier
#coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])
#   Now take the mean across all those classifiers
#feature_importance = np.mean(np.abs(coefs), axis=0)
# Plot feature importance
#plt.barh(X.columns, feature_importance)
#plt.title("Feature Importance (One-vs-One)")
#plt.xlabel("Importance")
#plt.show()
