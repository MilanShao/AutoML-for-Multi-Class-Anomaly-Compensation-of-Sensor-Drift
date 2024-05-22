import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_inputs = tokenizer(list(X_train[0].astype(str)), return_tensors='pt', padding=True, truncation=True)

# Create PyTorch tensors
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
labels = torch.tensor(y_train.values)

# Create dataset and dataloader
dataset = TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pre-trained DistilBERT model for sequence classification
num_classes = 6
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Check unique class labels in training set
#print("Unique labels in y_train:", np.unique(y_train))

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        labels = (labels.to(device) - 1).long()  # Convert labels to Long type with 0-based indexing

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

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

# Tokenization for test data
tokenized_inputs_test = tokenizer(list(X_test[0].astype(str)), return_tensors='pt', padding=True, truncation=True)
input_ids_test = tokenized_inputs_test['input_ids']
attention_mask_test = tokenized_inputs_test['attention_mask']
labels_test = torch.tensor(y_test.values)

# Create dataset and dataloader for test data
dataset_test = TensorDataset(input_ids_test, attention_mask_test, labels_test)
dataloader_test = DataLoader(dataset_test, batch_size=32)

# Evaluation
model.eval()
all_preds = []
all_labels = []
for batch in dataloader_test:
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and display confusion matrix and classification report
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
