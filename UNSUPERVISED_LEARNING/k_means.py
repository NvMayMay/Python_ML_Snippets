# install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn plotly

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# Generate synthetic dataset with 4 centers using make_blobs
# n_samples is the number of data points divided equally among clusters
# centers is the number of centers to generate, or the fixed center locations
# cluster_std is the standard deviation of the clusters
# X is an array of shape [n_samples, n_features] with the generated samples
# y is an array of shape [n_samples] with the integer labels for cluster membership of each sample
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
# remove hash in below to print
#plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)

# Set up KMeans clustering
# init is the method for initialization, k-means++ is preferred
# n_clusters is the number of clusters to form as well as the number of centroids to generate
# n_init is the number of time the k-means algorithm will be run with different centroid seeds
kmeans = KMeans(init="k-means++", n_clusters=4, n_init=12)
# Fit the model to the data 
kmeans.fit(X)
# get label for each point in the model and save to variable
k_means_labels = kmeans.labels_
# get the coordinates of the cluster centers and save to variable
k_means_cluster_centers = kmeans.cluster_centers_

# Create visual plot of k-means clustered data sets.
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot, delete hash in below line to print
#plt.show()

# Running with k=3
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
# Delete hash in below line to print
#plt.show()

# Using csv data file of customer segmentation
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

# Data preprocessing, dropping categorial columns
cust_df = cust_df.drop('Address', axis=1)
cust_df = cust_df.dropna()
# Normalize data
X = cust_df.values[:,1:] # leaves out `Customer ID`
Clus_dataSet = StandardScaler().fit_transform(X)

# Run KMeans on customer dataset using k=3
clusterNum = 3
k_mean = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_mean.fit(X)
labels = k_mean.labels_
# Apply k-means cluster lables to each row in dataframe
cust_df["Clus_km"] = labels
# Check centroids by averaging the features in each cluster
cust_df.groupby('Clus_km').mean()
#Examine distribution of customers based on their education, income and age
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
# Detele hash in below line to print
#plt.show()

# Create interactive 3D scatter plot
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))
fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot
# Delete hash in below line to print
#fig.show()

# From observations, 3 clusters are:
# Cluster 0: Higher income, older age, moderate education
# Cluster 1: Lower income, younger age
# Cluster 2: Moderate income, moderate age