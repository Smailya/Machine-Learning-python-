# Import necessary libraries
import numpy  # NumPy is used for numerical operations and generating random data
import matplotlib.pyplot as plt  # Matplotlib is used for plotting graphs
from sklearn.metrics import r2_score  # scikit-learn's r2_score measures model fit quality

# Set random seed for reproducibility of results
numpy.random.seed(2)

# Generate synthetic data:
# x: Normally distributed around 3 (mean) with std dev 1, 100 data points
# y: Normally distributed around 150 with std dev 40, divided by x to simulate dependent variable
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

# Split data into training and testing sets
# First 80 points for training (80%), last 20 for testing (20%)
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

# Visualize the training data with a scatter plot
plt.scatter(train_x, train_y)
plt.title("Training Set Scatter Plot")
plt.xlabel("Minutes before purchase")
plt.ylabel("Money spent")
plt.show()

# Visualize the testing data similarly
plt.scatter(test_x, test_y)
plt.title("Testing Set Scatter Plot")
plt.xlabel("Minutes before purchase")
plt.ylabel("Money spent")
plt.show()

# Fit a polynomial regression model of degree 4 on the training data
# numpy.polyfit fits a polynomial, numpy.poly1d creates a polynomial function
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

# Create a range of x values to plot the polynomial regression line smoothly
myline = numpy.linspace(0, 6, 100)

# Plot training data and the polynomial regression line
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline), color='red')
plt.title("Polynomial Regression Fit on Training Data")
plt.xlabel("Minutes before purchase")
plt.ylabel("Money spent")
plt.show()

# Evaluate model fit on training data using R-squared score
r2_train = r2_score(train_y, mymodel(train_x))
print(f"R-squared on training data: {r2_train:.3f}")  # ~0.799 indicates decent fit

# Evaluate model fit on testing data using R-squared score
r2_test = r2_score(test_y, mymodel(test_x))
print(f"R-squared on testing data: {r2_test:.3f}")  # ~0.809 indicates model generalizes well

# Predict how much money a customer will spend if they stay 5 minutes in the shop
prediction = mymodel(5)
print(f"Predicted spending for 5 minutes: ${prediction:.2f}")
s
