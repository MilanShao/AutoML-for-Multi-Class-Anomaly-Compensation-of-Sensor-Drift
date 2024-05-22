import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MeanShift
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Mean-Shift-Algorithmus sucht nach den lokalen Maxima in der Dichtefunktion der Daten

# Load data from the first two batches
list_features = []
list_targets = []

for i in range(1, 6):  # Load only the first five batches
    X, y = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=np.float64)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)
    
    list_features.append(X)
    list_targets.append(y)
    
X_train = pd.concat(list_features, ignore_index=True)
y_train = pd.concat(list_targets, ignore_index=True)

# Create and train a Mean Shift model for drift detection
mean_shift = MeanShift()
mean_shift.fit(X_train)

# Identify clusters and their centers
clusters = mean_shift.predict(X_train)
cluster_centers = mean_shift.cluster_centers_

# Visualize clusters
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 8], c=clusters, cmap='viridis', marker='.')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 8], marker='x', s=150, linewidths=5, color='r')
plt.title('Mean Shift Clustering')
plt.xlabel('Feature 0')
plt.ylabel('Feature 8')
plt.show()

# Identify points that are far from cluster centers (potential drift points)
distances = np.linalg.norm(X_train - cluster_centers[clusters], axis=1)
# Set threshold based on a quantile of distances
quantile_threshold = 0.75
threshold = np.quantile(distances, quantile_threshold)
potential_drift_points = X_train[distances > threshold]

# Remove irrelevant plots (for visualization)
relevant_features = [0, 8, 72, 64, 112]
for feature_idx in relevant_features:
    plt.figure(figsize=(10, 6))
    
    # Histogram for the training set
    plt.hist(X_train.iloc[:, feature_idx], bins=50, alpha=0.5, label='Train (Batches 1-5)', color='blue')
    
    # Scatter plot for potential drift points
    plt.scatter(potential_drift_points.iloc[:, feature_idx], np.zeros_like(potential_drift_points.iloc[:, feature_idx]), 
                color='red', marker='x', label='Potential Drift Points')
    
    plt.title(f'Histogram of Feature {feature_idx} with Drift Points')
    plt.xlabel(f'Feature {feature_idx} Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.show()

# Create and train a Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

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

# Perform predictions on the test set
predictions = random_forest.predict(X_test)

# Calculate accuracy and display confusion matrix and classification report
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

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
