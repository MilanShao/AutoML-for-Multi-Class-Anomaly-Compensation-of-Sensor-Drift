import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools

from scipy.stats import kruskal
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

from warnings import filterwarnings
filterwarnings('ignore')

list_features = []
list_targets = []
for i in range(1, 11):
    X, y = load_svmlight_file(f='/Dataset/batch' + str(i) + '.dat', dtype=np.float64)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)
    
    list_features.append(X)
    list_targets.append(y)
    
X = pd.concat(list_features, ignore_index=True)
y = pd.concat(list_targets, ignore_index=True)

print(X.shape)
X.head()

# write csv
# y.name = 'target'
# dataset = pd.concat([X, y], axis=1)
# dataset.to_csv('/home/adduser/Projekte/Sensordrift/Dataset/dataset.csv', index=False)

# write PCA csv
# scaler = MinMaxScaler(feature_range=(-1, 1))
# cols = X.columns
# X = scaler.fit_transform(X)
# X = pd.DataFrame(X, columns=cols)
# pca = PCA().fit(X)
# X_pca = PCA(0.995).fit_transform(X) # PCA(30) com 1/4 das features
# X_pca = pd.DataFrame(X_pca)
# y.name = 'target'
# dataset_pca = pd.concat([X_pca, y], axis=1)
# dataset_pca.to_csv('/home/adduser/Projekte/Sensordrift/Dataset/dataset_pca.csv', index=False)

correlation = X.corr()
correlation.head()

fig, ax = plt.subplots(figsize=(18,18))
sns.heatmap(correlation, cmap='seismic')
plt.savefig('correlation_seismic.png')

high_correlation = correlation.applymap(lambda x: x if x > 0.85 or x < -0.85 else 0)

fig, ax = plt.subplots(figsize=(18,18))
sns.heatmap(high_correlation, cmap='seismic') # 'Blues'
plt.savefig('correlation_85.png')

# y.name = 'target'
# data = pd.concat([X, y], axis=1)
# data.head()

# cols_to_plot = [i for i in range(0, 128, 8)]
# cols_to_plot.append('target')
# cols_to_plot

# sns.pairplot(data[cols_to_plot], hue='target')
# plt.savefig('pairplot.png')

# Save the boxplot for each column
# count = 0
# for i in range(2):
#     for j in range(4):
#         plt.figure(figsize=(8, 6))  # Create a new figure for each boxplot
#         plt.boxplot(X.iloc[:, cols_to_plot[count]])
#         plt.xticks([])
#         plt.xlabel(str(cols_to_plot[count]))
#         plt.savefig(f'boxplot_col_{cols_to_plot[count]}.png')
#         count += 1

# # Show the histograms
# plt.show()

# # Save the histogram
# def count_outliers(column):
#     q1 = column.quantile(0.25)
#     q3 = column.quantile(0.75)
#     iqr = q3 - q1
#     return column[(column < (q1 - 1.5 * iqr)) | (column > (q3 + 1.5 * iqr))].count()

# outliers_columns= X.apply(count_outliers)
# outliers_columns[outliers_columns == 0]
# plt.figure(figsize=(8, 6))
# outliers_columns.hist(bins=30)
# plt.savefig('outliers_histogram.png')

# Set the random seed
seed = 10

# Load the dataset
dataset = pd.read_csv('/Dataset/dataset_pca.csv')

# Split the dataset into features (X) and target variable (y)
X = dataset.drop('target', axis=1)
y = dataset.loc[:, 'target']

# Define test size and create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

# Set scoring metric, number of cross-validation folds, and create StratifiedKFold object
scoring = 'accuracy'
cv = 10
folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

# Create a KNeighborsClassifier with desired parameters
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')

# Perform cross-validation for kNN and print results
knn_scores = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(knn_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(knn_scores), np.std(knn_scores)))

# Perform cross-validation for Decisiontree and print results
tree = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=seed)
tree_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(tree_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(tree_scores), np.std(tree_scores)))

# Perform cross-validation for Randomforrest and print results
forest = RandomForestClassifier(n_estimators=110, random_state=seed)
forest_scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(forest_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(forest_scores), np.std(forest_scores)))

# Perform cross-validation for MLP and print results
mlp = MLPClassifier(hidden_layer_sizes=(55, 55, 50), activation='tanh', solver='lbfgs', max_iter=2000, random_state=seed)
mlp_scores = cross_val_score(estimator=mlp, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(mlp_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(mlp_scores), np.std(mlp_scores)))

#BaggingClassifier
bagging = BaggingClassifier(mlp, n_estimators=10)
bagging_scores = cross_val_score(estimator=bagging, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(bagging_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(bagging_scores), np.std(bagging_scores)))

#VotingClassifier
voting = VotingClassifier(estimators=[('knn', knn), ('tree', tree), ('forest', forest), ('mlp', mlp)])
voting_scores = cross_val_score(estimator=voting, X=X_train, y=y_train, cv=folds, scoring=scoring)
print(voting_scores)
print('Mean: %.5f, std: %.5f' % (np.mean(voting_scores), np.std(voting_scores)))

# Perform Normality test
def normality_test(sample, sample_name):
    stat, p = shapiro(sample)
    alpha = 0.05
    if p > alpha:
        print(sample_name + ': Normal distribution (fail to reject H0)')
    else:
        print(sample_name + ' No normal distribution (reject H0)')
        
clf_results = [(knn_scores, 'KNN'), (tree_scores, 'DF'), (forest_scores, 'RF'), (mlp_scores, 'MLP'), (bagging_scores, 'BG'), (voting_scores, 'VT')]

for clf_result in clf_results:
    normality_test(clf_result[0], clf_result[1])

#Kruskal-Willis   
stat, p = kruskal(knn_scores, tree_scores, forest_scores, mlp_scores, bagging_scores, voting_scores)
print(stat, p)
alpha = 0.05

if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
    
def compare_samples(pair):
    sample_names = pair[0][1] + ' x ' + pair[1][1]
    sample_1 = pair[0][0]
    sample_2 = pair[1][0]
    stat, p = mannwhitneyu(sample_1, sample_2)
    alpha = 0.05
    if p > alpha:
        print(sample_names + ': Same distributions (fail to reject H0). p-value: ' + str(p))
    else:
        print(sample_names + ': Different distributions (reject H0). p-value: '+ str(p))
        
for pair in itertools.combinations((clf_results), 2):
    compare_samples(pair)

# Compare the results
results = pd.DataFrame([knn_scores, tree_scores, forest_scores, mlp_scores, bagging_scores, voting_scores])
results.index = ['KNN', 'DT', 'RF', 'MLP', 'BG', 'VT']

# Save the results to a CSV file
results.to_csv('classification_results.csv')

fig, ax = plt.subplots(figsize=(12,7))

ax.boxplot(results)
plt.xticks([1, 2, 3, 4, 5, 6], ['KNN', 'DT', 'RF', 'MLP', 'BG', 'VT'])
plt.savefig('boxplot_results.png')
#plt.show()

knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Accuracy: %.5f' % accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(12,7))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues', ax=ax, fmt='d')
plt.savefig('confusion_matrix_voting.png')



