# Import dependencies
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
