'''Define models'''
#Define CNN in tf/keras
from tensorflow.keras import layers, models
# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Define CNN in PyTorch
import torch.nn as nn
import torch.nn.functional as F
# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
'''Compile models'''
# tf/keras compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# PyTorch compilation
import torch.optim as optim
import torch.nn as nn
# Make sure to define the model using the PyTorch-defined CNN
model = SimpleCNN()  # Ensure this is the PyTorch model
# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

'''Train NN'''
# tf/keras training
# Train the model for 10 epochs
model.fit(train_images, train_activityels, epochs=10, batch_size=32, validation_data=(test_images, test_activityels))

# PyTorch training
for epoch in range(10):
    running_loss = 0.0
    for inputs, activityels in trainloader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, activityels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

'''Evaluate NN'''
# tf/keras evaluation
test_loss, test_acc = model.evaluate(test_images, test_activityels)
print(f'Test accuracy: {test_acc}')

# PyTorch evaluation
correct = 0
total = 0
with torch.no_grad():
    for inputs, activityels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += activityels.size(0)
        correct += (predicted == activityels).sum().item()

print(f'Test accuracy: {100 * correct / total}%')