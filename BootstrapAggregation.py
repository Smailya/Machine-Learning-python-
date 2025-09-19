
"""
What is Bagging?
Bagging, short for Bootstrap Aggregating, is an ensemble learning technique in machine learning designed to improve the accuracy and stability of predictive models. 
The main idea is to train multiple models (often called "base learners" or "weak learners") independently on different random subsets of the training data, 
and then combine their predictions to produce a final result. This process helps to reduce model variance and mitigate overfitting, especially for models that are sensitive to noise, such as decision trees
How Bagging Works:

Bootstrap Sampling: Random subsets of the original dataset are created by sampling with replacement (so some samples may appear multiple times in a subset, and some not at all).

Model Training: A separate model is trained on each bootstrap sample.

Prediction Aggregation: For classification tasks, the final prediction is made by majority voting among all models; for regression, predictions are averaged.

Result: The ensemble prediction is typically more accurate and robust than any single model
"""
# Import necessary libraries for data handling, modeling, and evaluation
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt

# Load a sample dataset (Wine dataset) for demonstration
data = datasets.load_wine(as_frame=True)
X = data.data        # Features
y = data.target      # Labels

# Split the dataset into training and test sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22
)

# --- Base Model: Single Decision Tree ---

# Initialize a Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=22)

# Train the Decision Tree on the training data
dtree.fit(X_train, y_train)

# Predict labels for the test set
y_pred = dtree.predict(X_test)

# Evaluate and print accuracy on both train and test sets
print('Train data accuracy:', accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)))
print('Test data accuracy:', accuracy_score(y_true=y_test, y_pred=y_pred))
# Single trees often overfit: high train accuracy, lower test accuracy

# --- Bagging Ensemble ---

# Try different numbers of estimators (trees) to see effect on accuracy
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
models = []
scores = []

for n_estimators in estimator_range:
    # Create a BaggingClassifier with n_estimators Decision Trees
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
    clf.fit(X_train, y_train)  # Train ensemble on training data
    models.append(clf)
    # Evaluate accuracy on test set
    scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))

# Plot the number of estimators vs. test accuracy
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)
plt.xlabel('n_estimators', fontsize=18)
plt.ylabel('score', fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
# This plot shows how increasing the number of trees affects performance

# --- Out-of-Bag (OOB) Evaluation ---

# Initialize a BaggingClassifier with OOB score enabled
oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
oob_model.fit(X_train, y_train)
print("OOB Score:", oob_model.oob_score_)
# OOB score estimates accuracy using only the training data, leveraging samples not used in each bootstrap

# --- Visualizing an Individual Tree from the Ensemble ---

from sklearn.tree import plot_tree

# Visualize the first tree in the ensemble
plt.figure(figsize=(30, 20))
plot_tree(models[2].estimators_[0], feature_names=X.columns)
plt.show()
# This helps interpret how a single tree in the bagging ensemble makes decisions
