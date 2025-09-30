# Required dependencies:
# - matplotlib: for plotting graphs
# - scikit-learn: for KMeans clustering algorithm
# Install them using:
# pip install matplotlib scikit-learn

import matplotlib.pyplot as plt  # Import plotting library for visualization
from sklearn.cluster import KMeans  # Import KMeans clustering algorithm from scikit-learn

# Sample data points (x and y coordinates)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combine x and y into a list of coordinate pairs (tuples)
data = list(zip(x, y))

# Use the elbow method to find the best number of clusters (K)
inertias = []  # List to store inertia values for each K

# Test K values from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)  # Initialize KMeans with k clusters
    kmeans.fit(data)  # Fit the model to the data
    inertias.append(kmeans.inertia_)  # Store the inertia (sum of squared distances)

# Plot the inertia values to visualize the "elbow"
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')  # Title of the plot
plt.xlabel('Number of clusters (K)')  # X-axis label
plt.ylabel('Inertia')  # Y-axis label
plt.show()  # Display the plot

# From the elbow plot, choose K=2 (where the elbow appears)
kmeans = KMeans(n_clusters=2)  # Initialize KMeans with 2 clusters
kmeans.fit(data)  # Fit the model to the data

# Visualize the clustered data points with colors showing cluster assignments
plt.scatter(x, y, c=kmeans.labels_)  # Scatter plot colored by cluster labels
plt.title('K-means Clustering with K=2')  # Title of the plot
plt.show()  # Display the plot


