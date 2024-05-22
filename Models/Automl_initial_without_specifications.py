import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from autosklearn.classification import AutoSklearnClassifier

# Load data for training
list_features_train = []
list_targets_train = []

for i in range(1, 6):  
    X_train, y_train = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=np.float64)
    X_train = pd.DataFrame(X_train.toarray())
    y_train = pd.Series(y_train)
    
    list_features_train.append(X_train)
    list_targets_train.append(y_train)

X_train = pd.concat(list_features_train, ignore_index=True)
y_train = pd.concat(list_targets_train, ignore_index=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train AutoML for hyperparameter tuning
automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, ensemble_size=1, initial_configurations_via_metalearning=0)
automl.fit(X_train, y_train)

# Perform predictions on the test set
automl_predictions = automl.predict(X_test)

# Calculate accuracy and display confusion matrix and classification report
automl_accuracy = accuracy_score(y_test, automl_predictions)
automl_conf_matrix = confusion_matrix(y_test, automl_predictions)
automl_class_report = classification_report(y_test, automl_predictions)

print("AutoML Accuracy:", automl_accuracy)
print("\nAutoML Confusion Matrix:\n", automl_conf_matrix)
print("\nAutoML Classification Report:\n", automl_class_report)
