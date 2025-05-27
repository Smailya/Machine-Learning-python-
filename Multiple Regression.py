# Import pandas for data handling and reading CSV files
import pandas as pd

# Import LinearRegression from sklearn for building the multiple regression model
from sklearn.linear_model import LinearRegression

# Load the dataset from a CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Define independent variables (features) as a DataFrame with columns 'Weight' and 'Volume'
X = df[['Weight', 'Volume']]

# Define dependent variable (target) as the 'CO2' column
y = df['CO2']

# Create a LinearRegression object
regr = LinearRegression()

# Fit the model to the data (train the regression model)
regr.fit(X, y)

# Predict CO2 emission for a car with weight=2300kg and volume=1300cm3
predictedCO2 = regr.predict([[2300, 1300]])

print("Predicted CO2 emission:", predictedCO2)

# Print the coefficients of the regression model
# These represent the effect of each independent variable on CO2 emission
print("Coefficients:", regr.coef_)

# Example: Predict CO2 emission for a car with increased weight (3300kg) but same volume (1300cm3)
predictedCO2_increased_weight = regr.predict([[3300, 1300]])
print("Predicted CO2 emission with increased weight:", predictedCO2_increased_weight)
