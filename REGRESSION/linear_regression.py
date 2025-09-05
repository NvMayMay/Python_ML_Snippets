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

# selects features for regression model and takes 9 random samples from the dataframe
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)
#delete hash in below line for table
#print(cdf.sample(9))

# plot histogram for specific features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
#delete hash in below line for plots
#plt.show()

# scatter plot to show relationship between FUELCONSUMPTION_COMB and CO2EMISSIONS
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
#delete has in below line for plots
#plt.show()

# split dataset into X and Y variable sets
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
#split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# build simple linear regression model
from sklearn import linear_model
regressor = linear_model.LinearRegression()
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)
# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)
# coefficients are line of best fit (y=mx+b)

# visualize model outputs
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
# plots regression model as red line by using y=mx+b formula, can be evaluated by eye

# evaluate model performance using mean absolute error, mean squared error, root mean square error, and R2 score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))
# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))