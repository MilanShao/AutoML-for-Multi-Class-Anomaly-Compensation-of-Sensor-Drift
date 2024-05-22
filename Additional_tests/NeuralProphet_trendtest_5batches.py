from neuralprophet import NeuralProphet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming 'ds' is your datetime column and 'y' is your target column
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

# Convert target labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Prepare the DataFrame with 'ds' column for dates
df_train = pd.DataFrame({
    'ds': pd.date_range(start="2022-01-01", periods=len(X_train), freq='D'),  # Adjust the start date as needed
    'y': y_train_encoded
})

# Initialize NeuralProphet
model = NeuralProphet()

# Fit the model
model.fit(df_train)

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

# Convert target labels to integers
y_test_encoded = label_encoder.transform(y_test)

# Prepare the DataFrame with 'ds' column for dates
df_test = pd.DataFrame({
    'ds': pd.date_range(start="2022-01-01", periods=len(X_test), freq='D'),  # Adjust the start date as needed
    'y': y_test_encoded
})

# Create future DataFrame for prediction
future = model.make_future_dataframe(df_test, periods=len(X_test))

# Perform classification with thresholding
classification_forecast = model.predict(future)
your_classification_threshold = 0.5  # Adjust the threshold as needed
predicted_classes = (classification_forecast['yhat1'] > your_classification_threshold).astype(int)

# Print Classification Report
print('\n\n--- NeuralProphet Classifier ---')
print('Accuracy:', accuracy_score(y_test_encoded, predicted_classes))
print('\nClassification Report:\n', classification_report(y_test_encoded, predicted_classes))
print('\nConfusion Matrix:\n', confusion_matrix(y_test_encoded, predicted_classes))
