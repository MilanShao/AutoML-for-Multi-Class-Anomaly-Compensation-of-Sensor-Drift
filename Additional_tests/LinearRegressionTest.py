from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd

# Load data up to Batch 10 for training and testing
list_features = []
list_targets = []

for i in range(1, 11):  # Load all batches for training and testing
    X, y = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=np.float64)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)
    
    list_features.append(X)
    list_targets.append(y)

X_all = pd.concat(list_features, ignore_index=True)
y_all = pd.concat(list_targets)

# Create a Linear Regression model
linear_model = LinearRegression()

# Perform k-fold cross-validation
k = 10  # Number of folds
mse_scores = cross_val_score(linear_model, X_all, y_all, scoring='neg_mean_squared_error', cv=k)

# Convert negative MSE scores to positive
mse_scores = -mse_scores

# Print the MSE scores for each fold
for fold, mse in enumerate(mse_scores, start=1):
    print(f"Fold {fold}: Mean Squared Error = {mse}")

# Print the average MSE across all folds
average_mse = np.mean(mse_scores)
print(f"\nAverage Mean Squared Error across {k}-fold cross-validation: {average_mse}")
