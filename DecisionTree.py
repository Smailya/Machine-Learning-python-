# Import necessary libraries
import pandas as pd  # For data handling and manipulation
from sklearn.tree import DecisionTreeClassifier  # To create the decision tree model
import matplotlib.pyplot as plt  # For plotting the decision tree
from sklearn import tree  # For plotting utilities

# Step 1: Load the dataset from a CSV file
df = pd.read_csv("data.csv")  # Dataset contains info about comedians and if the person went to their show

# Step 2: Convert categorical (non-numeric) columns to numeric values
# 'Nationality' column: map 'UK'->0, 'USA'->1, 'N'->2
df['Nationality'] = df['Nationality'].map({'UK': 0, 'USA': 1, 'N': 2})

# 'Go' column (target): map 'YES'->1, 'NO'->0
df['Go'] = df['Go'].map({'YES': 1, 'NO': 0})

# Step 3: Separate feature columns (inputs) and target column (output)
features = ['Age', 'Experience', 'Rank', 'Nationality']  # Features used to predict
X = df[features]  # Input features
y = df['Go']      # Target variable (whether the person went or not)

# Step 4: Create and train the Decision Tree classifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)  # Fit model to the data

# Step 5: Visualize the trained decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(dtree, feature_names=features, filled=True, rounded=True)
plt.show()

# Step 6: Use the trained model to predict new data
# Example: Predict if person would go to a show with a 40-year-old American comedian,
# 10 years experience, rank 7, nationality USA (mapped to 1)
print(dtree.predict([[40, 10, 7, 1]]))  # Output: [1] means "YES"

# Another example with rank 6 instead of 7
print(dtree.predict([[40, 10, 6, 1]]))  # Output might be [0] meaning "NO"
