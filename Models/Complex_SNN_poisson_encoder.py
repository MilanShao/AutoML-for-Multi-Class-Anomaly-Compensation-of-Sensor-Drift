import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from batches
def load_data(batch_numbers):
    features, targets = [], []

    for batch_number in batch_numbers:
        data_path = f'/Dataset/batch{batch_number}.dat'
        X_batch, y_batch = load_svmlight_file(data_path, dtype=np.float64)
        features.append(pd.DataFrame(X_batch.toarray()))
        targets.append(pd.Series(y_batch))

    X = pd.concat(features, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)

    return X, y

# Custom Poisson Encoder
def poisson_encoder(input_data, time=300, dt=1.0, encoding_rate=20.0):
    spikes = torch.rand_like(input_data) < encoding_rate * dt / 1000.0
    spike_train = spikes.repeat(1, time)
    return spike_train.float()

# Define a more complex SNN model
class ComplexSNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexSNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size1)
        self.hidden_layer1 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden_layer2 = nn.Linear(hidden_size2, output_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)  # Batch normalization
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)  # Batch normalization
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = torch.relu(self.hidden_layer1(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        return self.hidden_layer2(x)

# Load and preprocess training data
X_train, y_train = load_data(range(1, 6))
X_train_tensor = torch.from_numpy(X_train.values).float()
y_train_tensor = torch.tensor(y_train.values - 1).long()

# Encode input features using Poisson Encoder
X_train_encoded = poisson_encoder(X_train_tensor)

# Initialize SNN model, criterion, and optimizer
input_size = X_train_encoded.shape[1]
hidden_size1 = 256
hidden_size2 = 128
output_size = 6
snn = ComplexSNN(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(snn.parameters(), lr=0.001)

# Training loop
for epoch in range(300):
    output = snn(X_train_encoded)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Load and preprocess test data
X_test, y_test = load_data(range(6, 11))
X_test_tensor = torch.from_numpy(X_test.values).float()
X_test_encoded = poisson_encoder(X_test_tensor)

# Testing loop
with torch.no_grad():
    output_test = snn(X_test_encoded)
    predictions_test = torch.argmax(output_test, dim=1).numpy()

# Calculate accuracy and display confusion matrix and classification report
accuracy = accuracy_score(y_test, predictions_test)
conf_matrix = confusion_matrix(y_test, predictions_test)
class_report = classification_report(y_test, predictions_test)

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
