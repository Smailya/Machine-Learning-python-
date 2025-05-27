# Import the matplotlib.pyplot module for plotting graphs and visualizations
import matplotlib.pyplot as plt

# Import numpy for numerical operations, especially for generating random data
import numpy

# Example 1: Simple scatter plot with predefined data points

# x array represents the age of each car
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]

# y array represents the speed of each car
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Use plt.scatter() to create a scatter plot
# Each point is plotted with x[i] on the horizontal axis and y[i] on the vertical axis
plt.scatter(x, y)

# Display the scatter plot
plt.show()


# Example 2: Scatter plot with randomly generated data from normal distributions

# Generate 1000 random values for x from a normal distribution with mean=5.0 and std dev=1.0
x = numpy.random.normal(5.0, 1.0, 1000)

# Generate 1000 random values for y from a normal distribution with mean=10.0 and std dev=2.0
y = numpy.random.normal(10.0, 2.0, 1000)

# Plot the scatter plot of these randomly generated points
plt.scatter(x, y)

# Show the plot
plt.show()
