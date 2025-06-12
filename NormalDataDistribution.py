# Import necessary libraries pip install numpy matplotlib

import numpy  # NumPy is required for generating random numbers and handling arrays
import matplotlib.pyplot as plt  # Matplotlib is needed for plotting the histogram

# Generate 100,000 random numbers following a normal (Gaussian) distribution
# Arguments to numpy.random.normal():
#   5.0: mean (center) of the distribution
#   1.0: standard deviation (spread/width of the bell curve)
#   100000: number of random values to generate
x = numpy.random.normal(5.0, 1.0, 100000)

# Plot a histogram of the generated data
# Arguments to plt.hist():
#   x: the data array
#   100: number of bins (bars) in the histogram
plt.hist(x, 100)

# Display the histogram plot
plt.show()
