import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.stats import kruskal, shapiro, mannwhitneyu
from warnings import filterwarnings

# Suppressing warnings for cleaner output
filterwarnings('ignore')

# Load and preprocess the data
list_features = []
list_targets = []
for i in range(1, 11):
    X, y = load_svmlight_file(f'/home/adduser/Projekte/Sensordrift/Dataset/batch' + str(i) + '.dat', dtype=np.float64)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)
    
    list_features.append(X)
    list_targets.append(y)
    
X = pd.concat(list_features, ignore_index=True)
y = pd.concat(list_targets, ignore_index=True)

import matplotlib.pyplot as plt

# Plot des Features X
plt.figure(figsize=(15, 8))
plt.plot(X)
plt.title('Sensor Measurements')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Plot des Targets y
plt.figure(figsize=(15, 8))
plt.plot(y)
plt.title('Targets')
plt.xlabel('Index')
plt.ylabel('Classes')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardisiere die Daten
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA durchführen
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotte die ersten beiden Hauptkomponenten
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('PCA - Erste zwei Hauptkomponenten')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Klasse')
plt.show()


# Load the dataset for PCA (if needed)
# dataset = pd.read_csv('/home/adduser/Projekte/Sensordrift/Dataset/dataset_pca.csv')
# X = dataset.drop('target', axis=1)
# y = dataset.loc[:, 'target']

# Split the dataset into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Create Dummy Classifier as the baseline model
dummy_classifier = DummyClassifier(strategy='most_frequent')

# Fit the Dummy Classifier on the training set
dummy_classifier.fit(X_train, y_train)

# Make predictions on the test set
dummy_pred = dummy_classifier.predict(X_test)

# Evaluate the baseline model
baseline_accuracy = accuracy_score(y_test, dummy_pred)
print(f'Baseline Accuracy: {baseline_accuracy:.5f}')

# Print classification report for detailed metrics
print('Classification Report for Dummy Classifier:')
print(classification_report(y_test, dummy_pred))

# Visualize the confusion matrix
sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, dummy_pred), annot=True, cmap='Blues', fmt='d', ax=ax)
plt.title('Confusion Matrix for Dummy Classifier')
plt.show()

# PCA auf den Trainingsdaten durchführen
X_train_scaled = scaler.transform(X_train)
X_train_pca = pca.transform(X_train_scaled)

# PCA auf den Testdaten durchführen
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Plotte die Verteilung der Klassen in der ersten und zweiten Hauptkomponente für Trainingsdaten
plt.figure(figsize=(12, 8))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('PCA - Erste zwei Hauptkomponenten - Trainingsdaten')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Klasse')
plt.show()

# Plotte die Verteilung der Klassen in der ersten und zweiten Hauptkomponente für Testdaten
plt.figure(figsize=(12, 8))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('PCA - Erste zwei Hauptkomponenten - Testdaten')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Klasse')
plt.show()

# Plotte die Klassengrenzen auf den Trainingsdaten
h = .02  # Schrittgröße im Gitter
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z_train = dummy_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z_train = Z_train.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z_train, cmap='viridis', alpha=0.3)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('Klassengrenzen - Trainingsdaten')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Klasse')
plt.show()

# Plotte die Klassengrenzen auf den Testdaten
Z_test = dummy_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z_test = Z_test.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z_test, cmap='viridis', alpha=0.3)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('Klassengrenzen - Testdaten')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Klasse')
plt.show()
