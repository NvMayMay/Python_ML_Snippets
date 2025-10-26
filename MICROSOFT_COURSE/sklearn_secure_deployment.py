'''
unsafe deployment
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset (Flaw: No data validation or sanitization)
data = pd.read_csv('user_data.csv')

# Split the dataset into features and target (Flaw: No input validation)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets (Flaw: Fixed random state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model (Flaw: No model security checks)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to disk (Flaw: Unencrypted model saving)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk for later use (Flaw: No integrity checks on the loaded model)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')

'''

secure deployment

'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import hashlib
import cryptography.fernet

# Validate and sanitize input data
def validate_data(df):
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values. Please clean the data before processing.")
    return df

# Load and validate dataset
data = validate_data(pd.read_csv('user_data.csv'))

# Split the dataset into features and target with validation
X = validate_data(data.iloc[:, :-1])
y = validate_data(data.iloc[:, -1])

# Split the data into training and testing sets with a securely managed random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=os.urandom(16))

# Train a logistic regression model with added security considerations
model = LogisticRegression()
model.fit(X_train, y_train)

# Encrypt model before saving
key = cryptography.fernet.Fernet.generate_key()
cipher = cryptography.fernet.Fernet(key)

# Save the encrypted model to disk
filename = 'finalized_model.sav'
encrypted_model = cipher.encrypt(pickle.dumps(model))
with open(filename, 'wb') as f:
    f.write(encrypted_model)

# Load the encrypted model from disk and verify its integrity
with open(filename, 'rb') as f:
    encrypted_model = f.read()
    decrypted_model = cipher.decrypt(encrypted_model)

loaded_model = pickle.loads(decrypted_model)

# Compute hash of the loaded model
loaded_model_hash = hashlib.sha256(decrypted_model).hexdigest()

# Verify that the loaded model's hash matches the original
original_model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
if loaded_model_hash != original_model_hash:
    raise ValueError("Model integrity check failed. The model may have been tampered with.")

result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')
