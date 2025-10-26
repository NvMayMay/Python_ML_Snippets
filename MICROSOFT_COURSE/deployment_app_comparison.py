'''
 You have been hired by a telecommunications company to develop a machine-learning model that predicts customer churn. 
 The company wants to identify customers who are likely to cancel their service so they can take proactive steps to retain them. 
 The model you develop will be integrated into the company’s customer relationship management (CRM) system 
 and used by the marketing team to target at-risk customers with retention offers.
 
 The same task is performed on TensorFlow, PyTorch, and Scikit-learn frameworks for comparison.
'''
#TensorFlow imports
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' These steps are common for all three frameworks: data loading, preprocessing, and splitting. '''
# Load dataset
data = pd.read_csv('customer_churn.csv')
# preprocess the data
data = data.drop(columns=['CustomerID']) #Simplify the dataset
data = data.dropna()  # Simple example of dropping missing values
# Handle missing values and simplify the dataset
data = pd.get_dummies(data, drop_first=True)
# Split the dataset into features and target
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Development and training'''
#TensorFlow Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#PyTorch Model
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ChurnModel()

#Scikit-learn Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

'''Compile and train the models'''
#TensorFlow
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#PyTorch
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified example)
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train.values).float())
    loss = criterion(outputs.squeeze(), torch.tensor(y_train.values).float())
    loss.backward()
    optimizer.step()

#Scikit-learn
model.fit(X_train, y_train)

'''Evaluate the models'''
#TensorFlow
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

#PyTorch
model.eval()
outputs = model(torch.tensor(X_test.values).float())
predictions = (outputs.squeeze().detach().numpy() > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.values)
print(f'Test accuracy: {accuracy}')

#Scikit-learn
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')

'''Optimise for deployment'''
#TensorFlow: Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

#PyTorch: apply dynamic quantization
# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

#Scikit-learn: simplify model by limiting depth
# Simplify model by limiting its maximum depth
pruned_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt') 
pruned_model.fit(X_train, y_train) 
pruned_predictions = pruned_model.predict(X_test) 
pruned_accuracy = accuracy_score(y_test, pruned_predictions) 
print(f'Pruned Test accuracy: {pruned_accuracy}')

'''Save models for deployment'''
#TensorFlow
model.save('churn_model.h5')

#PyTorch
torch.save(model.state_dict(), 'churn_model.pth')

#Scikit-learn
import joblib
joblib.dump(pruned_model, 'churn_model.pkl')