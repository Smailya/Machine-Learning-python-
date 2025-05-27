# Import matplotlib.pyplot for plotting graphs and visualizations
import matplotlib.pyplot as plt

# Import stats module from scipy for statistical functions including linear regression
from scipy import stats

# Define the data arrays:
# x represents the age of each car (independent variable)
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

# y represents the speed of each car (dependent variable)
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Perform linear regression using scipy.stats.linregress
# This function returns:
# slope: slope of the regression line
# intercept: y-intercept of the regression line
# r: correlation coefficient (strength and direction of linear relationship)
# p: p-value for hypothesis test (significance of the relationship)
# std_err: standard error of the estimated slope
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Define a function that calculates predicted y values from x using the regression line equation
def myfunc(x):
    return slope * x + intercept

# Apply the regression function to each x value to get predicted y values (the regression line)
mymodel = list(map(myfunc, x))

# Plot the original data points as a scatter plot
plt.scatter(x, y)

# Plot the linear regression line
plt.plot(x, mymodel)

# Show the plot with data points and regression line
plt.show()

# Print the correlation coefficient to understand how well the data fits a linear model
print("Correlation coefficient (r):", r)

# Predict the speed of a car that is 10 years old using the regression model
predicted_speed = myfunc(10)
print("Predicted speed of a 10-year-old car:", predicted_speed)
