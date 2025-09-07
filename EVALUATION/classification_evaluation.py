# must install numpy, pandas, matplotlib, scikit-learn, seaborn if not already installed
# python3 -m pip install numpy pandas matplotlib scikit-learn seaborn==0.13.2
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names
# Can describe the dataset by uncommenting the line below
# Dataset comprised of 569 samples with 30 features each, binary target (malignant=0, benign=1)
#print(data.DESCR)
# Can print the target names by uncommenting the line below
#print(data.target_names)

# Standardise data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Add some noise to simulate random measurement error
# Add Gaussian noise to the data set
np.random.seed(42)  # For reproducibility
noise_factor = 0.5 # Adjust this to control the amount of noise
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)
# Compare original and noisy data by uncommenting the block below
#plt.figure(figsize=(12, 6))
# # Original Feature Distribution (Noise-Free)
#plt.subplot(1, 2, 1)
#plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original')
#plt.title('Original Feature Distribution')
#plt.xlabel(feature_names[5])
#plt.ylabel('Frequency')
## Noisy Feature Distribution
#plt.subplot(1, 2, 2)
#plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy') 
#plt.title('Noisy Feature Distribution')
#plt.xlabel(feature_names[5])  
#plt.ylabel('Frequency')
#plt.tight_layout()  # Ensures proper spacing between subplots
#plt.show()

# Split data into training and test sets and initialise KNN and SVM models
# Split the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
# Initialize the models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1, random_state=42)
# Fit the models to the training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
# Make predictions on the test data
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)
# # Print accuracy scores and classification reports for both models
# # Uncomment the lines below to print, prints precision, recall, f1-score, support for each class
# print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
# print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")
# print("\nKNN Testing Data Classification Report:")
# print(classification_report(y_test, y_pred_knn))
# print("\nSVM Testing Data Classification Report:")
# print(classification_report(y_test, y_pred_svm))

# Plot confusion matrices for both models, uncomment to print
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
#plt.show()