# must install numpy, pandas, matplotlib, scikit-learn, seaborn if not already installed
# python3 -m pip install numpy pandas matplotlib scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

# Definte function to display evaluation metrics
def regression_results(y_true, y_pred, regr_type):
    # Regression metrics
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred) 
    mse = mean_squared_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred)
    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ',  round(ev,4)) 
    print('r2: ', round(r2,4))
    print('MAE: ', round(mae,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print()

# Generate synthetic data
noise=1
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + noise*np.random.randn(1000, 1)  # Linear relationship with some noise
y_ideal =  4 + 3 * X
# Specify the portion of the dataset to add outliers (e.g., the last 20%)
y_outlier = pd.Series(y.reshape(-1).copy())
# Identify indices where the feature variable X is greater than a certain threshold
threshold = 1.5  # Example threshold to add outliers for larger feature values
outlier_indices = np.where(X.flatten() > threshold)[0]
# Add outliers at random locations within the specified portion
num_outliers = 5  # Number of outliers to add
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)
# Modify the target values at these indices to create outliers (add significant noise)
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)
plt.figure(figsize=(12, 6))
# Scatter plot of the original data with outliers
plt.scatter(X, y_outlier, alpha=0.4,ec='k', label='Original Data with Outliers')
plt.plot(X, y_ideal,  linewidth=3, color='g',label='Ideal, noise free data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
# Delete hash in below line for plots
#plt.show()

# Fit a simple linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

# Fit a ridge regression model (regularization to control large coefficients)
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

# Fit a lasso regression model (regularization to control large coefficients)
lasso_reg = Lasso(alpha=.2)
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

# Print regression results, uncomment the lines below to print
# regression_results(y, y_outlier_pred_lin, 'Ordinary')
# regression_results(y, y_outlier_pred_ridge, 'Ridge')
# regression_results(y, y_outlier_pred_lasso, 'Lasso')

# Create high dimensional synthetic data with small number of informative features
from sklearn.datasets import make_regression
X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)
# Get the ideal predictions based on the informative coefficients used in the regression model
ideal_predictions = X @ ideal_coef
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()
# Fit the models
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)
# Predict on the test set
y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
# Print regression results, uncomment the lines below to print
# regression_results(y_test, y_pred_linear, 'Ordinary')
# regression_results(y_test, y_pred_ridge, 'Ridge')
# regression_results(y_test, y_pred_lasso, 'Lasso')

# Compaure performance of models, uncomment the block below to print
# fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
# axes[0,0].scatter(y_test, y_pred_linear, color="red", label="Linear")
# axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,0].set_title("Linear Regression")
# axes[0,0].set_xlabel("Actual",)
# axes[0,0].set_ylabel("Predicted",)
# axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
# axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,2].set_title("Lasso Regression",)
# axes[0,2].set_xlabel("Actual",)
# axes[0,1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
# axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,1].set_title("Ridge Regression",)
# axes[0,1].set_xlabel("Actual",)
# axes[0,2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
# axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
# axes[0,2].set_title("Lasso Regression",)
# axes[0,2].set_xlabel("Actual",)
# # Line plots for predictions compared to actual and ideal predictions
# axes[1,0].plot(y_test, label="Actual", lw=2)
# axes[1,0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
# axes[1,0].set_title("Linear vs Ideal",)
# axes[1,0].legend()
 # axes[1,1].plot(y_test, label="Actual", lw=2)
# # axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
# axes[1,1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
# axes[1,1].set_title("Ridge vs Ideal",)
# axes[1,1].legend()
 # axes[1,2].plot(y_test, label="Actual", lw=2)
# axes[1,2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
# axes[1,2].set_title("Lasso vs Ideal",)
# axes[1,2].legend()
 # plt.tight_layout()
# plt.show()