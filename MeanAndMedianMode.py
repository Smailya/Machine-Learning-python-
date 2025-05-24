# Required dependencies:
# You need to have NumPy and SciPy installed.
# You can install them using pip if you don't have them already:
# pip install numpy scipy

import numpy as np
from scipy import stats

# List of recorded speeds of 13 cars
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Calculate the mean (average) speed
# Mean is calculated by summing all values and dividing by the number of values
mean_speed = np.mean(speed)

# Calculate the median (middle) speed
# Median is the middle value when the data is sorted
median_speed = np.median(speed)

# Calculate the mode (most common) speed
# Mode is the value that appears most frequently in the data
mode_speed = stats.mode(speed)

# Print the results
print(f"Mean speed: {mean_speed:.2f}")  # Rounded to 2 decimal places
print(f"Median speed: {median_speed}")
print(f"Mode speed: {mode_speed.mode[0]} (appears {mode_speed.count[0]} times)")

"""
Explanation:
- We use NumPy's mean() and median() functions to compute the average and middle values respectively.
- We use SciPy's stats.mode() function to find the most frequently occurring value.
- The output gives a quick statistical summary of the speeds recorded.
"""
