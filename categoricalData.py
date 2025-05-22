# Required dependencies:
# - pandas: for data manipulation and handling CSV files
# - scikit-learn: for linear regression model
# Install them using:
# pip install pandas scikit-learn

import pandas  # Import pandas library for data handling
from sklearn import linear_model  # Import linear regression model from scikit-learn

# Load the dataset from a CSV file named 'data.csv'
cars = pandas.read_csv("data.csv")

# Convert the categorical 'Car' column into dummy/indicator variables (one-hot encoding)
# This creates new columns for each unique car brand with 0/1 values
ohe_cars = pandas.get_dummies(cars[['Car']])

# Combine the numeric features 'Volume' and 'Weight' with the one-hot encoded car brand columns
X = pandas.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)

# Set the target variable as the 'CO2' column (emission values)
y = cars['CO2']

# Create a linear regression model object
regr = linear_model.LinearRegression()

# Train the model on the features (X) and target (y)
regr.fit(X, y)

# Predict CO2 emission for a specific car with:
# Weight = 2300 kg, Volume = 1300 cm3, and car brand encoded as one-hot vector
# The one-hot vector here has 18 zeros and a 1 at the 19th position corresponding to 'VW'
predictedCO2 = regr.predict([[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

# Print the predicted CO2 emission value
print(predictedCO2)
