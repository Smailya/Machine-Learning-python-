

# Required dependencies:
# - scikit-learn: for machine learning tools and datasets
# - To install, run: pip install scikit-learn
# - then write the code

from sklearn import datasets  # Import datasets module to load sample datasets
from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier model
from sklearn.metrics import accuracy_score  # Import function to calculate accuracy of the model

# Load the wine dataset as a pandas DataFrame (as_frame=True preserves feature names)
data = datasets.load_wine(as_frame=True)

# Separate the dataset into input features (X) and target labels (y)
X = data.data  # Features such as chemical properties of wine
y = data.target  # Target variable representing wine classes

# Split the data into training and testing sets
# 25% of data is reserved for testing, random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Initialize the Decision Tree Classifier with a fixed random state for reproducibility
dtree = DecisionTreeClassifier(random_state=22)

# Train (fit) the decision tree model on the training data
dtree.fit(X_train, y_train)

# Use the trained model to predict the classes of the test data
y_pred = dtree.predict(X_test)

# Calculate and print the accuracy on the training data (how well the model fits training data)
print("Train accuracy:", accuracy_score(y_train, dtree.predict(X_train)))

# Calculate and print the accuracy on the test data (how well the model generalizes to unseen data)
print("Test accuracy:", accuracy_score(y_test, y_pred))

