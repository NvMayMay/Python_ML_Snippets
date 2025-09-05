#install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn
import numpy as np 
import pandas as pd
import ssl
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# This dataset is for a set of patients with same illness who took one of five drugs
# This exercise will build a model to predict which drug would be most appropriate for a patient based on their features

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
ssl._create_default_https_context = ssl._create_unverified_context
my_data = pd.read_csv(path)

# Checking for null values and to get data types run code without hash
#print(my_data.isnull().sum())
#print(my_data.info())

# Preprocessing: convert categorical variables to numerical using LabelEncoder
# print data afterwards (delete hash from below to print) to see what parameter has mapped on to what value
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
#print(my_data)
# map target variable (drug) to numerical values
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
#print(my_data)

#   find correlation of input features to target variable
my_data.drop('Drug',axis=1).corr()['Drug_num']

#   Create frequency table for target variable
#   delete hash from below to print
#category_counts = my_data['Drug'].value_counts()
#   Plot the count plot
#plt.bar(category_counts.index, category_counts.values, color='blue')
#plt.xlabel('Drug')
#plt.ylabel('Count')
#plt.title('Category Distribution')
#plt.xticks(rotation=45)  # Rotate labels for better readability if needed
#plt.show()

# seperate target variable from input features
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)
# split data into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
# build decision tree model, defining entropy as the information gain criteria and a max depth of 4
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

#   predict on testing set and check accuracy (print code without hash to see accuracy)
tree_predictions = drugTree.predict(X_testset)
#print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

#   visualize the decision tree (print code without hash to see tree)
#plot_tree(drugTree)
#plt.show()