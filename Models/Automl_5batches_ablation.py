import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from autosklearn.pipeline.components.classification import random_forest

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

# Load data for testing
list_features_test = []
list_targets_test = []

for i in range(6, 11):  
    X_test, y_test = load_svmlight_file(f'/Dataset/batch{i}.dat', dtype=np.float64)
    X_test = pd.DataFrame(X_test.toarray())
    y_test = pd.Series(y_test)
    
    list_features_test.append(X_test)
    list_targets_test.append(y_test)

X_test = pd.concat(list_features_test, ignore_index=True)
y_test = pd.concat(list_targets_test, ignore_index=True)

# automl = AutoSklearnClassifier(
#     time_left_for_this_task=3600,  
#     per_run_time_limit=800,         
#     initial_configurations_via_metalearning=25,  # Metalearning auf 25 setzen
#     ensemble_size=20,   
#     ensemble_nbest=50,  
#     max_models_on_disc=50, 
#     seed=1, 
#     memory_limit=3072, 
#     include={"data_preprocessor": ["feature_type"]},
#     exclude=None,  
#     resampling_strategy='cv-iterative-fit',      
#     resampling_strategy_arguments={'folds': 5, 'stratified': True},  
#     tmp_folder=None, 
#     delete_tmp_folder_after_terminate=True, 
#     n_jobs=None, 
#     dask_client=None, 
#     disable_evaluator_output=False, 
#     get_smac_object_callback=None, 
#     smac_scenario_args=None,
#     logging_config=None, 
#     metadata_directory=None, 
#     metric=None, 
#     scoring_functions=None, 
#     load_models=True, 
#     get_trials_callback=None, 
#     dataset_compression=True, 
#     allow_string_features=True
# )



# automl = AutoSklearnClassifier(
#     time_left_for_this_task=120,  
#     per_run_time_limit=30,         
#     initial_configurations_via_metalearning=25,  # Metalearning auf 25 setzen
#     ensemble_size=20,   
#     ensemble_nbest=50,  
#     max_models_on_disc=50, 
#     seed=1, 
#     memory_limit=3072, 
#     include={"data_preprocessor": ["feature_type"], "feature_preprocessor": ["no_preprocessing"]},  # Alle Datenvorverarbeitungskomponenten ausschließen
#     resampling_strategy='cv-iterative-fit',      
#     resampling_strategy_arguments={'folds': 5, 'stratified': True},  
#     tmp_folder=None, 
#     delete_tmp_folder_after_terminate=True, 
#     n_jobs=None, 
#     dask_client=None, 
#     disable_evaluator_output=False, 
#     get_smac_object_callback=None, 
#     smac_scenario_args=None,
#     logging_config=None, 
#     metadata_directory=None, 
#     metric=None, 
#     scoring_functions=None, 
#     load_models=True, 
#     get_trials_callback=None, 
#     dataset_compression=True, 
#     allow_string_features=True
# )


automl = AutoSklearnClassifier(
    time_left_for_this_task=120,  
    per_run_time_limit=30,         
    initial_configurations_via_metalearning=0,  # no Metalearning
    ensemble_size=20,   
    ensemble_nbest=50,  
    max_models_on_disc=50, 
    seed=1, 
    memory_limit=3072, 
    include={"classifier": ["random_forest"]},  
    resampling_strategy='cv-iterative-fit',      
    resampling_strategy_arguments={'folds': 5, 'stratified': True},  
    tmp_folder=None, 
    delete_tmp_folder_after_terminate=True, 
    n_jobs=None, 
    dask_client=None, 
    disable_evaluator_output=False, 
    get_smac_object_callback=None, 
    smac_scenario_args=None,
    logging_config=None, 
    metadata_directory=None, 
    metric=None, 
    scoring_functions=None, 
    load_models=True, 
    get_trials_callback=None, 
    dataset_compression=True, 
    allow_string_features=True
)



# include_classifiers = ["random_forest"]

# automl = AutoSklearnClassifier(
#     time_left_for_this_task=120,  
#     per_run_time_limit=30,         
#     initial_configurations_via_metalearning=25,  
#     ensemble_size=1,   # Ensemble deaktivieren, indem die Größe auf 1 gesetzt wird
#     ensemble_nbest=1,  # Beste Modelle auf 1 beschränken
#     max_models_on_disc=1,  # Modelle auf 1 beschränken
#     seed=1, 
#     memory_limit=3072, 
#     include={"classifier": include_classifiers, "data_preprocessor": ["feature_type"]},
#     exclude=None,  
#     resampling_strategy='cv-iterative-fit',      
#     resampling_strategy_arguments={'folds': 5, 'stratified': True},  
#     tmp_folder=None, 
#     delete_tmp_folder_after_terminate=True, 
#     n_jobs=None, 
#     dask_client=None, 
#     disable_evaluator_output=False, 
#     get_smac_object_callback=None, 
#     smac_scenario_args=None,
#     logging_config=None, 
#     metadata_directory=None, 
#     metric=None, 
#     scoring_functions=None, 
#     load_models=True, 
#     get_trials_callback=None, 
#     dataset_compression=True, 
#     allow_string_features=True
# )



# # Train AutoML with Preprocessing, Balancing, and Feature Preprocessing
# automl = AutoSklearnClassifier(
#     time_left_for_this_task=120,  
#     per_run_time_limit=30,         
#     ensemble_size=20,   
#     initial_configurations_via_metalearning=4,
#     ensemble_class="default",  
#     ensemble_nbest=20,          
#     include={"data_preprocessor": ["feature_type"]},
#     smac_scenario_args={"runcount_limit": 5},
#     #include_preprocessors=["no_preprocessing", "random_trees_embedding", "pca"],  
#     resampling_strategy='cv-iterative-fit',      # Resampling Balancing
#     resampling_strategy_arguments={'folds': 5, 'stratified': True},  
#     #include_estimators=["random_forest", "extra_trees"],  
#     #include_preprocessors=["no_preprocessing", "polynomial"]  
# )
# automl.fit(X_train, y_train)

# # Perform predictions on the test set
# automl_predictions = automl.predict(X_test)

# # Calculate accuracy and display confusion matrix and classification report
# automl_accuracy = accuracy_score(y_test, automl_predictions)
# automl_conf_matrix = confusion_matrix(y_test, automl_predictions)
# automl_class_report = classification_report(y_test, automl_predictions)

# print("AutoML Accuracy:", automl_accuracy)
# print("\nAutoML Confusion Matrix:\n", automl_conf_matrix)
# print("\nAutoML Classification Report:\n", automl_class_report)

# # best models and their weights
# models_with_weights = automl.get_models_with_weights()

# for model, weight in models_with_weights:
#     print("Model:", model)
#     print("Weight:", weight)
#     print()

# Find the indices of rows corresponding to class 6
class_6_indices_train = y_train[y_train == 6].index
class_6_indices_test = y_test[y_test == 6].index

# Remove rows corresponding to class 6 from training and test data
X_train_filtered = X_train.drop(class_6_indices_train)
y_train_filtered = y_train.drop(class_6_indices_train)

X_test_filtered = X_test.drop(class_6_indices_test)
y_test_filtered = y_test.drop(class_6_indices_test)

# Train AutoML with the filtered data
automl.fit(X_train_filtered, y_train_filtered)

# Perform predictions on the filtered test set
automl_predictions_filtered = automl.predict(X_test_filtered)

# Calculate accuracy and display confusion matrix and classification report for filtered data
automl_accuracy_filtered = accuracy_score(y_test_filtered, automl_predictions_filtered)
automl_conf_matrix_filtered = confusion_matrix(y_test_filtered, automl_predictions_filtered)
automl_class_report_filtered = classification_report(y_test_filtered, automl_predictions_filtered)

print("AutoML Accuracy (Filtered):", automl_accuracy_filtered)
print("\nAutoML Confusion Matrix (Filtered):\n", automl_conf_matrix_filtered)
print("\nAutoML Classification Report (Filtered):\n", automl_class_report_filtered)

# best models and their weights
models_with_weights_filtered = automl.get_models_with_weights()

for model, weight in models_with_weights_filtered:
    print("Model:", model)
    print("Weight:", weight)
    print()
