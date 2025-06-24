"""
What is Logistic Regression?
Logistic Regression is a classification algorithm used to predict categorical outcomes (e.g., yes/no, 0/1).

Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class.

For binary classification (two possible outcomes), logistic regression models the log-odds of the outcome as a linear function of the input variables.

The output probability is between 0 and 1, indicating the likelihood of the positive class (e.g., tumor is cancerous).

How the Code Works:
Data Preparation:
Tumor sizes (X) and their corresponding cancer status (y) are prepared. X is reshaped to a 2D array as required by scikit-learn.

Model Training:
A logistic regression model is created and trained (fit) on the data to learn the relationship between tumor size and cancer status.

Prediction:
The model predicts whether a tumor of size 3.46 cm is cancerous or not.

Interpreting Coefficients:
The model coefficient represents the change in log-odds per unit increase in tumor size. Exponentiating it gives the odds ratio, which is easier to interpret (e.g., odds increase 4 times per cm increase).

Probability Calculation:
The logit2prob function converts the model’s log-odds output into probabilities between 0 and 1, showing the likelihood of cancer for each tumor size.

 Results: A tumor size of 3.46 cm is predicted not to be cancerous (``).

The odds ratio (~4) means each 1 cm increase in tumor size multiplies the odds of cancer by about 4.

Probabilities for each tumor size give a nuanced understanding of cancer risk rather than a simple yes/no prediction.

This example demonstrates how logistic regression can be used to classify binary outcomes and interpret the results probabilistically.
"""

# Import dependencies pip install numpy scikit-learn

import numpy as np  # For numerical operations and array handling
from sklearn import linear_model  # For logistic regression model

# Step 1: Prepare the data
# X: Tumor sizes in centimeters (independent variable)
# Reshape to 2D array as required by sklearn (n_samples, n_features)
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)

# y: Tumor status (dependent variable)
# 0 means tumor is benign (not cancerous), 1 means malignant (cancerous)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Step 2: Create logistic regression model and fit it to the data
logr = linear_model.LogisticRegression()
logr.fit(X, y)

# Step 3: Predict if a tumor of size 3.46 cm is cancerous (1) or not (0)
predicted = logr.predict(np.array([3.46]).reshape(-1, 1))
print("Prediction for tumor size 3.46 cm:", predicted)  # Output: [0] means benign

# Step 4: Understand the model coefficient (log-odds)
# Coefficient indicates how the log-odds of cancer changes with tumor size
log_odds_coef = logr.coef_
odds = np.exp(log_odds_coef)  # Convert log-odds to odds ratio
print("Odds ratio for 1 cm increase in tumor size:", odds)  # ~4 means odds increase 4x per cm

# Step 5: Define a function to convert log-odds to probability
def logit2prob(logr, x):
    """
    Convert logistic regression output (log-odds) to probability.
    Parameters:
      logr: trained logistic regression model
      x: input tumor size(s), can be scalar or array-like
    Returns:
      Probability of tumor being cancerous.
    """
    log_odds = logr.coef_ * x + logr.intercept_  # Calculate log-odds
    odds = np.exp(log_odds)                       # Convert log-odds to odds
    probability = odds / (1 + odds)                # Convert odds to probability
    return probability

# Step 6: Use the function to compute probabilities for all tumor sizes in X
probabilities = logit2prob(logr, X)
print("Probabilities of tumor being cancerous for given sizes:\n", probabilities)

# Expected output example:
# [[0.60749955]
#  [0.19268876]
#  [0.12775886]
#  [0.00955221]
#  [0.08038616]
#  [0.07345637]
#  [0.88362743]
#  [0.77901378]
#  [0.88924409]
#  [0.81293497]
#  [0.57719129]
#  [0.96664243]]
