# Import the NumPy library for numerical operations
import numpy

# List of ages of people living on a street
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2, 8, 6, 25, 36, 27, 61, 31]

# Calculate the 75th percentile (value below which 75% of ages fall)
percentile_75 = numpy.percentile(ages, 75)

# Print the 75th percentile value
print(f"The 75th percentile is: {percentile_75}")

# Calculate the 90th percentile (value below which 90% of ages fall)
percentile_90 = numpy.percentile(ages, 90)

# Print the 90th percentile value
print(f"The 90th percentile is: {percentile_90}")
