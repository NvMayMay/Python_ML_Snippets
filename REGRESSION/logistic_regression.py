# must install numpy, pandas, matplotlib, scikit-learn if not already installed
# python3 -m pip install numpy pandas matplotlib scikit-learn

import pandas as pd
import numpy as np
import ssl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# churn_df = pd.read_csv("ChurnData.csv") model data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
ssl._create_default_https_context = ssl._create_unverified_context
churn_df = pd.read_csv(url)

# preprocess data by selecting relevant features and converting boolean to integer
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
# convert to arrays; one for input features and one for target variable
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])
# standardize the data
X_norm = StandardScaler().fit(X).transform(X)
#create training and testing sets, 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)
# build logistic regression model
LR = LogisticRegression().fit(X_train,y_train)
# predict using the testing set
yhat = LR.predict(X_test)
# get probability estimates for each class
yhat_prob = LR.predict_proba(X_test)

# output can be evaluated by examining the role of each feature in the model
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
# positive LR coefficient indicates that as the value of the feature increases, the odds of the target being 1 (churn) increase
# negative LR coefficient indicates that as the value of the feature increases, the odds of the target being 1 (churn) decrease

# find the log loss value for the model and dataset
log_loss(y_test, yhat_prob)