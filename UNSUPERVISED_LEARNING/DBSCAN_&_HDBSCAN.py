# install relevant libraries
# python3 -m pip install numpy pandas matplotlib scikit-learn
# python3 -m pip install hdbscan geopandas contextily shapely

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
import ssl
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

# Download the canada map
import requests
import zipfile
import io
import os

# URL of the ZIP file on the cloud server
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'
# Directory to save the extracted TIFF file
output_dir = './'
os.makedirs(output_dir, exist_ok=True)
# Step 1: Download the ZIP file
response = requests.get(zip_file_url)
response.raise_for_status()  # Ensure the request was successful
# Step 2: Open the ZIP file in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Step 3: Iterate over the files in the ZIP
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):  # Check if it's a TIFF file
            # Step 4: Extract the TIFF file
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")

#   Write a function that plots clustered locations and overlays them on a basemap.
def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
        Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    # Show the plot
    plt.show()

# Download the data into a pandas dataframe
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv(url, encoding = "ISO-8859-1")

#Preprocess the data, keeping only museum entries and only the latitude and longitude columns.
df = df[df.ODCAF_Facility_Type == 'museum']
df = df[['Latitude', 'Longitude']]
# remove those with null values and convert to floats not objects
df = df[df.Latitude != '..']
df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float')

# In this case we know how to scale the coordinates. Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]

# Apply DBSCAN with euclidian distance metric to scaled coordinates
min_samples=3 # minimum number of samples needed to form a neighbourhood
eps=1.0 # neighbourhood search radius
metric='euclidean' # distance measure 
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)

# Add the cluster labels to the original dataframe
df['Cluster'] = dbscan.fit_predict(coords_scaled)  # Assign the cluster labels
# Display the size of each cluster, with -1 being the noise cluster
# Delete hash in below line to print
#print(df['Cluster'].value_counts())
#plot_clustered_locations(df, title='Museums Clustered by Proximity')

'''
Build a HDBSCAN model and compare results to DBSCAN
'''

# Initialize HDBSCAN model
min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')
# Assign the cluster labels
df['Cluster'] = hdb.fit_predict(coords_scaled)