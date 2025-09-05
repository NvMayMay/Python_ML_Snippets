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