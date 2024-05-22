import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(x)
        x = self.decoder(x)
        x = torch.relu(x)
        return x

# Define the training function
def train_autoencoder(model, dataloader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch[0]  # Extract the data part of the batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Compare reconstruction with original input
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print("Epoch {}: Loss = {:.4f}".format(epoch, total_loss / len(dataloader)))


# Prepare data for training Autoencoder
autoencoder_models = []

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


# Train Autoencoder models for the first three classes
train_autoencoder_models = []
test_features = []
test_targets = []

for class_idx in range(1, 4):  # Use batches 1 to 3 for training
    # Load data for the current class
    X_train, y_train = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{class_idx}.dat', dtype=np.float64)
    X_train = pd.DataFrame(X_train.toarray())
    y_train = pd.Series(y_train)
    
    # Convert data to PyTorch tensors and create DataLoader
    dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define and initialize the Autoencoder model
    autoencoder = Autoencoder(input_size=X_train.shape[1], hidden_size=64)
    
    # Print the model architecture
    print("Autoencoder architecture for class", class_idx)
    print(autoencoder)
    
    # Train the Autoencoder model
    train_autoencoder(autoencoder, dataloader)
    
    # Add trained Autoencoder model to the list
    train_autoencoder_models.append(autoencoder)

# Load data for testing from batches 4 and 5
for i in range(4, 6):  # Load batches 4 and 5 for testing
    X_test_batch, y_test_batch = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{i}.dat', dtype=np.float64)
    X_test_batch = pd.DataFrame(X_test_batch.toarray())
    y_test_batch = pd.Series(y_test_batch)
    
    test_features.append(X_test_batch)
    test_targets.append(y_test_batch)

# Concatenate test data
X_test = pd.concat(test_features, ignore_index=True)
y_test = pd.concat(test_targets, ignore_index=True)

# Convert test data to PyTorch tensors and create DataLoader
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32))
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Perform predictions on the test set
all_predictions = []

for autoencoder in train_autoencoder_models:
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch[0]
            outputs = autoencoder(inputs)
            mse = ((inputs - outputs) ** 2).mean(dim=1)  # Calculate Mean Squared Error
            predictions.extend(mse.numpy())
    all_predictions.append(predictions)

# Combine predictions from all autoencoders and make a majority vote
final_predictions = np.mean(np.array(all_predictions), axis=0)  # Using mean as a voting mechanism

# Convert predictions to binary (anomaly or not)
threshold = 0.9  # Adjust threshold as needed
final_predictions_binary = [1 if pred > threshold else 0 for pred in final_predictions]

# Calculate accuracy and display confusion matrix and classification report
accuracy = accuracy_score(y_test, final_predictions_binary)
conf_matrix = confusion_matrix(y_test, final_predictions_binary)
class_report = classification_report(y_test, final_predictions_binary)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
