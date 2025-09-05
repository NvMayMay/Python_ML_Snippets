#install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn
# Import the libraries we need 
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')
# this dataset is for credit card transactions, the goal is to identify fraudulent transactions based on other features
# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
ssl._create_default_https_context = ssl._create_unverified_context
raw_data=pd.read_csv(url)

# visualise the distribution of the target variable, delete hash in below section to print
# get the set of distinct classes
labels = raw_data.Class.unique()
# get the count of each class
sizes = raw_data.Class.value_counts().values
# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
#plt.show()

# show correlation of target variable against input features, delete hash in below section to print
#correlation_values = raw_data.corr()['Class'].drop('Class')
#correlation_values.plot(kind='barh', figsize=(10, 6))
#plt.show()

# preprocess data by normalizing input features and deleting 'Time' column
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values
# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]
# y: labels vector
y = data_matrix[:, 30]
# data normalization
X = normalize(X, norm="l1")

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# due to class imbalance, we compute sample weights to be used in training the model
w_train = compute_sample_weight('balanced', y_train)
# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)

# unlike decision trees, SVMs do not have a native way to handle unbalanced classes
# so we use the class_weight='balanced' parameter to adjust weights inversely proportional to class
# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)

#   calculate probabilities of test samples evaluate accuracy using ROC-AUC score
#   ROC-AUC score is a preferred metric for unbalanced datasets, stands for Receiver Operating Characteristic - Area Under Curve
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
#   delete hash in below line to print
#print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

#   compute probability of test sample belonging to class of fraudulent transactions
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
#   delete hash in below line to print
#print('SVM ROC-AUC score : {0:.3f}'.format(roc_auc_svm))

#   Model's ROC-AUC score can be improved by removing features that are not correlated with the target variable
#   This code finds the 6 features most correlated with the target variable
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
#   Reassign X to a matrix with only those 6 features
X = raw_data[correlation_values.index].values