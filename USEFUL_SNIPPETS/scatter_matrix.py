# Requires inputs: df
# Produces outputs: scatter matrix plot
# df is a pandas dataframe containing the data to be visualized in a scatter matrix 
# scatter matrix helps visualize relationships between pairs of features in the dataset
# and the distribution of individual features along the diagonal

import pandas as pd
import matplotlib.pyplot as plt
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