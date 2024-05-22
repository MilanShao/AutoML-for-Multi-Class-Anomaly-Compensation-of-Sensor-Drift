import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.utils import shuffle

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

# Define the Graph Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
encoding_dim = 32

autoencoder = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

# Train the Autoencoder model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Calculate reconstruction errors for training set
reconstructions = autoencoder(X_train_tensor)
train_errors = torch.mean(torch.abs(reconstructions - X_train_tensor), axis=1)

# Compute 95th percentile reconstruction error per class
thresholds = []
for label in np.unique(y_train):
    class_errors = train_errors[y_train == label]
    threshold = torch.sort(class_errors)[0][int(len(class_errors) * 0.75)]
    thresholds.append(threshold)

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

# Calculate reconstruction errors for test set
test_reconstructions = autoencoder(X_test_tensor)
test_errors = torch.mean(torch.abs(test_reconstructions - X_test_tensor), axis=1)

# Classify instances as anomalies based on thresholds
anomaly_predictions = []
for i in range(len(X_test)):
    label = int(y_test.iloc[i])  # Convert label to integer
    if label < len(thresholds):  # Check if threshold is defined for the label
        error = test_errors[i]
        threshold = thresholds[label]
        if error > threshold:
            anomaly_predictions.append(1)  # Anomaly
        else:
            anomaly_predictions.append(0)  # Normal
    else:
        # If threshold is not defined, classify as anomaly
        anomaly_predictions.append(1)

# Display results
accuracy = accuracy_score(y_test, anomaly_predictions)
conf_matrix = confusion_matrix(y_test, anomaly_predictions)
class_report = classification_report(y_test, anomaly_predictions)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
