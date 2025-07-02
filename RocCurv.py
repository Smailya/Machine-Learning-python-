
"""
This code demonstrates evaluation of classification models on an imbalanced dataset, where one class (class 1) dominates 95% of the data.

Problem with Accuracy:
Accuracy can be misleading on imbalanced data. A model that always predicts the majority class achieves high accuracy but is useless for detecting the minority class.

Alternative Metric - AUC-ROC:
The ROC curve plots the True Positive Rate (sensitivity) vs False Positive Rate (1 - specificity) across thresholds. The AUC (Area Under Curve) quantifies the model’s ability to discriminate between classes regardless of threshold.

Two Hypothetical Models:

Model 1 predicts all samples as majority class → high accuracy but poor class 0 detection.

Model 2 predicts probabilities with some overlap → lower accuracy but better balanced performance.

Plotting ROC Curve:
Visualizes model performance at different thresholds, showing the trade-off between sensitivity and specificity.

AUC Score:
Provides a threshold-independent measure of model quality, especially useful for imbalanced datasets.
"""

# Import necessary libraries  pip install numpy scikit-learn matplotlib

import numpy as np  # For numerical operations and array handling
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve  # Evaluation metrics
import matplotlib.pyplot as plt  # For plotting ROC curve

# Step 1: Create an imbalanced dataset
n = 10000          # Total number of samples
ratio = 0.95       # Proportion of class 1 (majority class)
n_0 = int((1 - ratio) * n)  # Number of samples in class 0 (minority)
n_1 = int(ratio * n)        # Number of samples in class 1 (majority)

# True labels: 0 for minority class, 1 for majority class
y = np.array([0] * n_0 + [1] * n_1)

# Step 2: Hypothetical model 1 - always predicts majority class (class 1)
y_proba = np.array([1] * n)  # Predicted probabilities: always 100% for class 1
y_pred = y_proba > 0.5       # Predicted classes based on threshold 0.5

# Evaluate model 1 using accuracy and confusion matrix
print(f'Accuracy score (model 1): {accuracy_score(y, y_pred)}')
cf_mat = confusion_matrix(y, y_pred)
print('Confusion matrix (model 1):')
print(cf_mat)
print(f'Class 0 accuracy (model 1): {cf_mat[0][0] / n_0}')  # Accuracy on minority class
print(f'Class 1 accuracy (model 1): {cf_mat[1][1] / n_1}')  # Accuracy on majority class

# Explanation:
# Model 1 achieves very high overall accuracy (~95%) by always predicting the majority class (1).
# However, it completely fails to predict the minority class (0), so it is not useful.

# Step 3: Hypothetical model 2 - predicts probabilities with some uncertainty
# For class 0 samples, probabilities are uniformly distributed between 0 and 0.7
# For class 1 samples, probabilities are uniformly distributed between 0.3 and 1
y_proba_2 = np.array(
    np.random.uniform(0, 0.7, n_0).tolist() +
    np.random.uniform(0.3, 1, n_1).tolist()
)
y_pred_2 = y_proba_2 > 0.5  # Predicted classes based on threshold 0.5

# Evaluate model 2 using accuracy and confusion matrix
print(f'\nAccuracy score (model 2): {accuracy_score(y, y_pred_2)}')
cf_mat_2 = confusion_matrix(y, y_pred_2)
print('Confusion matrix (model 2):')
print(cf_mat_2)
print(f'Class 0 accuracy (model 2): {cf_mat_2[0][0] / n_0}')  # Accuracy on minority class
print(f'Class 1 accuracy (model 2): {cf_mat_2[1][1] / n_1}')  # Accuracy on majority class

# Explanation:
# Model 2 has lower overall accuracy than model 1 but performs better on both classes.
# It is more balanced and informative, despite lower accuracy.

# Step 4: Define a function to plot the ROC curve
def plot_roc_curve(true_y, y_prob):
    """
    Plots the ROC curve given true labels and predicted probabilities.

    Parameters:
    - true_y: true binary labels (0 or 1)
    - y_prob: predicted probabilities for the positive class (class 1)
    """
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)  # Calculate false positive rate and true positive rate
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')  # Diagonal line for random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Step 5: Plot ROC curve and calculate AUC for model 1
plot_roc_curve(y, y_proba)
auc_score_1 = roc_auc_score(y, y_proba)
print(f'Model 1 AUC score: {auc_score_1}')

# The ROC curve plots the trade-off between true positive rate and false positive rate at various thresholds.
# The AUC (Area Under the Curve) summarizes the ROC curve into a single value between 0 and 1.
# AUC close to 1 means excellent model, 0.5 means random guessing.
# Model 1, despite high accuracy, will have low AUC because it cannot distinguish between classes properly.
