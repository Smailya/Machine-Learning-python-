# Dependencies:
# This code requires the NumPy library, a powerful numerical computing package in Python.
# If you don't have it installed, you can install it using:
# pip install numpy

import numpy as np

# Example 1: Calculate Standard Deviation of car speeds
# Standard deviation measures how spread out the numbers are from the mean (average).
# A low standard deviation means values are close to the mean.
# A high standard deviation means values are more spread out.

speed1 = [86, 87, 88, 86, 87, 85, 86]

# Calculate standard deviation using NumPy's std() method
std_dev1 = np.std(speed1)

print(f"Standard Deviation of speed1: {std_dev1:.2f}")

# Example 2: Calculate Variance and Standard Deviation of another set of car speeds
# Variance is the average of the squared differences from the mean.
# Standard deviation is the square root of the variance.

speed2 = [32, 111, 138, 28, 59, 77, 97]

# Calculate variance using NumPy's var() method
variance = np.var(speed2)

# Calculate standard deviation using NumPy's std() method
std_dev2 = np.std(speed2)

print(f"Variance of speed2: {variance:.2f}")
print(f"Standard Deviation of speed2: {std_dev2:.2f}")

"""
Explanation:
- np.var() computes the variance, which quantifies the spread of the data by averaging squared deviations from the mean.
- np.std() computes the standard deviation, which is the square root of the variance, giving a measure of spread in the original units.
- These statistics help understand how consistent or variable the data values are.
"""
