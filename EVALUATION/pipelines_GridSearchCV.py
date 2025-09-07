# must install numpy, pandas, matplotlib, scikit-learn if not already installed
# python3 -m pip install matplotlib scikit-learn seaborn

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Model will be built on iris dataset 
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

#Instantiate a pipeline consisting of standard scaler, PCA, and KNN classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('pca', PCA(n_components=2),),       # Step 2: Reduce dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step 3: K-Nearest Neighbors classifier
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Fit the pipeline on training data
pipeline.fit(X_train, y_train)
# Measure the pipeline accuracy on the test data
test_score = pipeline.score(X_test, y_test)
# Uncomment the line below to print the test accuracy
#print(f"{test_score:.3f}")
y_pred = pipeline.predict(X_test)
# Enter your code here
conf_matrix = confusion_matrix(y_test, ...)
# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(..., annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Show the plot
plt.tight_layout()
# uncomment the line below to display the plot
#plt.show()

# generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Create a plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Show the plot
plt.tight_layout()
# uncomment the line below to display the plot
#plt.show()

# Model create using pipeline above can be improved with different hyperparameters
# One way to do this is with cross-validation
# make a pipeline without specifying any parameters yet
pipeline = Pipeline(
                    [('scaler', StandardScaler()),
                     ('pca', PCA()),
                     ('knn', KNeighborsClassifier()) 
                    ]
                   )
# Hyperparameter search grid for numbers of PCA components and KNN neighbors
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
             }
# Choose cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Determine best parameters and save best model to variable
best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                         )
# Fit the best GridSearch model on the training data and print the best parameters
# uncomment the line below to print the best parameters
# print(best_model.fit(X_train, y_train))
# The output of this will be something like:
# Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=3)),
#                 ('knn', KNeighborsClassifier(n_neighbors=3))])
# StandardScaler
# ?
# StandardScaler()
# PCA
# ?
# PCA(n_components=3)
# KNeighborsClassifier
# ?
# KNeighborsClassifier(n_neighbors=3)

# Evaluate the best model on the test data
test_score = best_model.score(X_test, y_test)
# Uncomment the lines below to print the test accuracy of the best model and best parameters
#print(f"{test_score:.3f}")
#print(best_model.best_params_)

# Plot confusion matric for prediction on test set
y_pred = best_model.predict(X_test)
# Generate the confusion matrix for KNN
conf_matrix = confusion_matrix(y_test, y_pred)
# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
# Set the title and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
# Show the plot
plt.tight_layout()
plt.show()
