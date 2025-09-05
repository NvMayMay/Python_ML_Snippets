# must install numpy, pandas, matplotlib, scikit-learn if not already installed
# python3 -m pip install numpy pandas matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ssl

# creates a secure SSL context and takes 5 random samples from the dataset at the url
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv(url)
df.sample(5)
#delete hash in below line for table
#print(df.sample(5))
# Drop categoricals and any useless columns
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)

# shows correlation matrix, with 1 being perfect correlation, -1 being perfect negative correlation, and 0 being no correlation
# can used .drop method to drop any with low correlation to the target variable
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
# delete hash in below line for table
#print(df.corr())

# plot scatter matrix, showing scatter plots for each pair of input features.
# diagonal plots show the distribution of each feature
axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
#delete hash in below line for plots
#plt.show()

# Extract the required columns (ENGINESIZE and FUELCONSUMPTION_COMB_MPG) and convert the resulting dataframes to NumPy arrays.
X = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()

#perform preprocessing (Standardization) to ensure that each feature contributes equally to the distance computations
# should be done after splitting into training and testing sets to avoid data leakage, but done here for simplicity
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# create training and testing sets, 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# build multiple linear regression model
from sklearn import linear_model
# create a model object
regressor = linear_model.LinearRegression()
# train the model in the training data
regressor.fit(X_train, y_train)
# Print the coefficients
coef_ =  regressor.coef_
intercept_ = regressor.intercept_
# delete hash in below lines for output
#print ('Coefficients: ',coef_)
#print ('Intercept: ',intercept_)
# coefficients and intercept describe the best fit hyperplane

# transform parameters back to original feature space prior to standardization
# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)
# delete hash in below lines for output
#print ('Coefficients: ', coef_original)
#print ('Intercept: ', intercept_original)

# visualize model outputs in 3D as plane over data points
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Ensure X1, X2, and y_test have compatible shapes for 3D plotting
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)
# Create a mesh grid for plotting the regression plane
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                               np.linspace(X2.min(), X2.max(), 100))
y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf
# Predict y values using trained regression model to compare with actual y_test for above/below plane colors
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]
# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
# Plot the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')
# Plot the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')
# Set view and labels
ax.view_init(elev=10)
ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
#delete hash in below line for plot
#plt.show()
