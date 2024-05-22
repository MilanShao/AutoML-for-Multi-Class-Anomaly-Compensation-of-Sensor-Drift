import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Define Normalizing Flow architecture
class NormalizingFlowModel(torch.nn.Module):
    def __init__(self, num_input_features, num_flows, num_hidden_units):
        super(NormalizingFlowModel, self).__init__()
        self.flows = torch.nn.ModuleList([dist.transforms.Planar(num_hidden_units) for _ in range(num_flows)])
        self.fc = torch.nn.Linear(num_input_features, num_hidden_units)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        for flow in self.flows:
            x = flow(x)
        return x

def train_normalizing_flow_model(model, dataloader, num_epochs=100, learning_rate=0.001):
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(model, model, optimizer, loss=Trace_ELBO())

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            total_loss += svi.step(batch[0])
        if epoch % 10 == 0:
            print("Epoch {}: Loss = {:.4f}".format(epoch, total_loss / len(dataloader)))

# Prepare data for training Normalizing Flow models
nf_models = []

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

# Train Normalizing Flow models for each class
for class_idx in range(1, 7):  # 6 classes
    
    # Filter data for the current class
    X_class = X_train[y_train == class_idx]
    
    # Convert data to PyTorch tensors and create DataLoader
    dataset = TensorDataset(torch.tensor(X_class.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define and initialize the Normalizing Flow model
    nf_model = NormalizingFlowModel(num_input_features=X_class.shape[1], num_flows=5, num_hidden_units=64)
    
    # Train the Normalizing Flow model
    train_normalizing_flow_model(nf_model, dataloader)
    
    # Add trained Normalizing Flow model to the list
    nf_models.append(nf_model)
