import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import kruskal, shapiro, mannwhitneyu
from warnings import filterwarnings
import itertools

filterwarnings('ignore')

# Load the first two batches for training
list_features_train = []
list_targets_train = []

for i in range(1, 3):
    X_train, y_train = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{i}.dat', dtype=np.float64)
    X_train = pd.DataFrame(X_train.toarray())
    y_train = pd.Series(y_train)

    list_features_train.append(X_train)
    list_targets_train.append(y_train)

X_train = pd.concat(list_features_train, ignore_index=True)
y_train = pd.concat(list_targets_train, ignore_index=True)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.995)  # Adjust the explained variance ratio as needed
X_train_pca = pca.fit_transform(X_train)

# Load the rest of the batches for testing
list_features_test = []
list_targets_test = []

for i in range(3, 11):
    X_test, y_test = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch{i}.dat', dtype=np.float64)
    X_test = pd.DataFrame(X_test.toarray())
    y_test = pd.Series(y_test)

    list_features_test.append(X_test)
    list_targets_test.append(y_test)

X_test = pd.concat(list_features_test, ignore_index=True)
y_test = pd.concat(list_targets_test, ignore_index=True)

# Apply PCA transformation on the test data
X_test_pca = pca.transform(X_test)

# Create a Voting Classifier with base classifiers
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
tree = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=10)
forest = RandomForestClassifier(n_estimators=110, random_state=10)
mlp = MLPClassifier(hidden_layer_sizes=(55, 55, 50), activation='tanh', solver='lbfgs', max_iter=2000, random_state=10)
bagging = BaggingClassifier(mlp, n_estimators=10)

voting = VotingClassifier(estimators=[('knn', knn), ('tree', tree), ('forest', forest), ('mlp', mlp), ('bagging', bagging)], voting='hard')

# Fit the Voting Classifier on the training data with PCA
voting.fit(X_train_pca, y_train)

# Make predictions on the test data with PCA
predictions = voting.predict(X_test_pca)

# Evaluate the performance
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
