# Cross-Validation (CV) is a technique to evaluate model performance on unseen data by splitting the dataset multiple times.
# It helps prevent overfitting and information leakage during hyperparameter tuning.

# Import necessary libraries
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, cross_val_score

# Load the Iris dataset (features X and labels y)
X, y = datasets.load_iris(return_X_y=True)

# Initialize the Decision Tree classifier with a fixed random state for reproducibility
clf = DecisionTreeClassifier(random_state=42)

# -------------------------
# 1. K-Fold Cross Validation
# -------------------------
# K-Fold splits the data into k equal parts (folds). The model trains on k-1 folds and validates on the remaining fold.
# This process repeats k times, each time with a different fold as validation.
k_folds = KFold(n_splits=5)

# Perform cross-validation and get accuracy scores for each fold
scores_kfold = cross_val_score(clf, X, y, cv=k_folds)

print("K-Fold Cross Validation Scores: ", scores_kfold)
print("Average K-Fold CV Score: ", scores_kfold.mean())
print("Number of K-Fold CV Scores used in Average: ", len(scores_kfold))


# -----------------------------
# 2. Stratified K-Fold Cross Validation
# -----------------------------
# Stratified K-Fold maintains the class distribution in each fold, useful for imbalanced datasets.
sk_folds = StratifiedKFold(n_splits=5)

# Perform stratified cross-validation
scores_stratified = cross_val_score(clf, X, y, cv=sk_folds)

print("\nStratified K-Fold Cross Validation Scores: ", scores_stratified)
print("Average Stratified K-Fold CV Score: ", scores_stratified.mean())
print("Number of Stratified K-Fold CV Scores used in Average: ", len(scores_stratified))


# -------------------------
# 3. Leave-One-Out (LOO) Cross Validation
# -------------------------
# LOO uses one sample as validation and the rest as training, repeated for every sample.
# This is exhaustive and computationally expensive but gives an almost unbiased estimate.
loo = LeaveOneOut()

# Perform LOO cross-validation
scores_loo = cross_val_score(clf, X, y, cv=loo)

print("\nLeave-One-Out Cross Validation Scores: ", scores_loo)
print("Average LOO CV Score: ", scores_loo.mean())
print("Number of LOO CV Scores used in Average: ", len(scores_loo))


# -------------------------
# 4. Leave-P-Out (LPO) Cross Validation
# -------------------------
# LPO leaves out p samples for validation and trains on the rest, repeated for all combinations.
# More exhaustive than LOO when p > 1.
lpo = LeavePOut(p=2)

# Perform LPO cross-validation
scores_lpo = cross_val_score(clf, X, y, cv=lpo)

print("\nLeave-P-Out Cross Validation Scores: ", scores_lpo)
print("Average LPO CV Score: ", scores_lpo.mean())
print("Number of LPO CV Scores used in Average: ", len(scores_lpo))


# -------------------------
# 5. Shuffle Split Cross Validation
# -------------------------
# ShuffleSplit randomly splits the data multiple times into training and validation sets.
# Allows flexible control over train/test sizes.
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)

# Perform ShuffleSplit cross-validation
scores_ss = cross_val_score(clf, X, y, cv=ss)

print("\nShuffle Split Cross Validation Scores: ", scores_ss)
print("Average Shuffle Split CV Score: ", scores_ss.mean())
print("Number of Shuffle Split CV Scores used in Average: ", len(scores_ss))


# Summary:
# - Cross-validation provides a better estimate of model performance on unseen data.
# - K-Fold is a standard method; Stratified K-Fold is preferred for imbalanced classes.
# - LOO and LPO are exhaustive but computationally expensive.
# - ShuffleSplit offers flexible random splits.
# - Always perform data preprocessing and tuning within CV folds to avoid leakage.
