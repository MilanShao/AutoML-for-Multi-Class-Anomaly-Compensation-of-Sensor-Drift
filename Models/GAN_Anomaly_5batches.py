import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data up to Batch 5 for training
list_features_train = []
list_targets_train = []

for i in range(1, 6):  # Load batches 1 to 5 for training
    X_train, y_train = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{i}.dat', dtype=np.float64)
    X_train = pd.DataFrame(X_train.toarray())
    y_train = pd.Series(y_train)
    
    list_features_train.append(X_train)
    list_targets_train.append(y_train)

X_train = pd.concat(list_features_train, ignore_index=True)
y_train = pd.concat(list_targets_train, ignore_index=True)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # Adjust output units for multiclass classification
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]

discriminator = Discriminator(input_dim)
criterion = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss for multiclass classification
optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Use torch.long for class labels

# Train the Discriminator model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = discriminator(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Load the rest of the data for testing (Batches 6 to 10)
list_features_test = []
list_targets_test = []

for i in range(6, 11):  # Load batches 6 to 10 for testing
    X_test, y_test = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{i}.dat', dtype=np.float64)
    X_test = pd.DataFrame(X_test.toarray())
    y_test = pd.Series(y_test)
    
    list_features_test.append(X_test)
    list_targets_test.append(y_test)

X_test = pd.concat(list_features_test, ignore_index=True)
y_test = pd.concat(list_targets_test)

# Convert test data to PyTorch tensor
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)  # Use torch.long for class labels

# Evaluate the Discriminator model
with torch.no_grad():
    test_outputs = discriminator(X_test_tensor)
    _, test_predictions = torch.max(test_outputs, 1)

# Display results
accuracy = accuracy_score(y_test, test_predictions)
conf_matrix = confusion_matrix(y_test, test_predictions)
class_report = classification_report(y_test, test_predictions)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Visualize the confusion matrix
sns.set(font_scale=1)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
