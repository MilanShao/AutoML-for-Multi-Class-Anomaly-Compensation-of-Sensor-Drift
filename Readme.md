AutoML for Multi-Class Anomaly Compensation of Sensor Drift - The official repository to the NeurIPS submission

## Benchmarking Results on the Proposed Sensor Drift Compensation Setting

The table below presents the performance of various models on our proposed sensor drift compensation setting. The metrics used for evaluation are Precision, Recall, and F1 score. Our model, AutoML-DC, achieves the highest scores across all metrics.

| **Model**               | **Precision** | **Recall** | **F1**   |
|-------------------------|---------------|------------|----------|
| Random Forest           | 0.68          | 0.57       | 0.56     |
| SVM (RBF Kernel)        | 0.52          | 0.43       | 0.43     |
| Logistic Regression     | 0.57          | 0.53       | 0.50     |
| XG Boost                | 0.66          | 0.53       | 0.51     |
| CatBoost                | 0.49          | 0.54       | 0.50     |
| KNN                     | 0.68          | 0.57       | 0.56     |
| SNN                     | 0.16          | 0.13       | 0.11     |
| LSTM                    | 0.58          | 0.61       | 0.57     |
| CNN                     | 0.65          | 0.62       | 0.60     |
| Decision Tree           | 0.50          | 0.38       | 0.40     |
| Gradient Boosting       | 0.49          | 0.51       | 0.49     |
| Gaussian Naive Bayes    | 0.50          | 0.32       | 0.32     |
| AdaBoost                | 0.42          | 0.42       | 0.40     |
| Bagging                 | 0.48          | 0.40       | 0.39     |
| Ensemble Model          | 0.29          | 0.32       | 0.29     |
| GRU                     | 0.40          | 0.32       | 0.28     |
| BERT-TS                 | 0.05          | 0.06       | 0.04     |
| Anomaly-GAN             | 0.21          | 0.35       | 0.26     |
| **AutoML-DC (ours)**    | **0.77**      | **0.76**   | **0.76** |