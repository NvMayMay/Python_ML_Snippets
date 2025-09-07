# install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn seaborn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import ssl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# This dataset is a customer base segmented by service usage patterns, we are to customize offers for each prospective customer
# Target field is custcat, corresponding to 4 customer categories

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
ssl._create_default_https_context = ssl._create_unverified_context

# Visualise the distribution of the target variable, delete hash in below section to print
#print(df['custcat'].value_counts())

# Visualise how each feature correlates to the target variable, delete hash in below section to print
#correlation_matrix = df.corr()
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Sort features by correlation value with target variable, delete hash in below section to print
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
#print(correlation_values)

# Separate data into features and target label
X = df.drop('custcat', axis=1)
y = df['custcat']

# Normalize data to standardize feature scales
X_norm = StandardScaler().fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Establish KNN model with k=3
k = 3
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# Use trained model to make predictions
yhat = knn_model.predict(X_test)

# Evaluate model accuracy, delete hash in below section to print
#print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# Use a for loop to determine the best value of k
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    # Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

#   Plot model accuracy for different values of k, delete hash in below section to print
#plt.plot(range(1,Ks+1),acc,'g')
#plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
#plt.legend(('Accuracy value', 'Standard Deviation'))
#plt.ylabel('Model Accuracy')
#plt.xlabel('Number of Neighbors (K)')
#plt.tight_layout()
#plt.show()
#print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 
