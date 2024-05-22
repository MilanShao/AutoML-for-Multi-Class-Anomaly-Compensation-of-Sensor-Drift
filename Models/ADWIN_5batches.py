
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

adwin = ADWIN()

# Load data from the first two batches
list_features = []
list_targets = []

for i in range(1, 6):  # Load batches 1 to 5 for training
    X, y = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=float)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)
    
    list_features.append(X)
    list_targets.append(y)
    
X_train = pd.concat(list_features, ignore_index=True)
y_train = pd.concat(list_targets, ignore_index=True)

# Create and train a Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Initialize ADWIN for each feature
adwin_detectors = [ADWIN() for _ in range(X_train.shape[1])]

# Load the rest of the data for testing
list_features_test = []
list_targets_test = []

for i in range(6, 11):  # Load batches 6 to 10 for testing
    X_test, y_test = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=np.float64)
    X_test = pd.DataFrame(X_test.toarray())
    y_test = pd.Series(y_test)
    
    list_features_test.append(X_test)
    list_targets_test.append(y_test)

X_test = pd.concat(list_features_test, ignore_index=True)
y_test = pd.concat(list_targets_test, ignore_index=True)

# Perform predictions on the test set with drift detection
predictions = []
for i in range(X_test.shape[0]):
    instance = X_test.iloc[[i]]
    
    # Check for concept drift in each feature
    drift_detected = any(adwin_detector.add_element(instance.iloc[0, j]) for j, adwin_detector in enumerate(adwin_detectors))
    
    if drift_detected:
        # Handle drift (e.g., update model, retrain, etc.)
        # For simplicity, retrain the entire model in this example
        random_forest.fit(X_train, y_train)
        
        # Reset ADWIN detectors
        adwin_detectors = [ADWIN() for _ in range(X_train.shape[1])]
    
    # Make predictions with the model
    prediction = random_forest.predict(instance)
    predictions.append(prediction[0])

# Calculate accuracy and display confusion matrix and classification report
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)