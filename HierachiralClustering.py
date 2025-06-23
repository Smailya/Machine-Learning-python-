# Import necessary libraries
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting data and dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage  # For hierarchical clustering and dendrogram plotting
from sklearn.cluster import AgglomerativeClustering  # For agglomerative hierarchical clustering implementation

# Step 1: Create sample data points (two variables x and y)
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combine x and y into a list of coordinate pairs (2D points)
data = list(zip(x, y))

# Step 2: Visualize the data points with a scatter plot
plt.scatter(x, y)
plt.title("Scatter plot of data points")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

# Step 3: Perform hierarchical/agglomerative clustering using SciPy's linkage function
# 'ward' linkage minimizes variance within clusters
# 'euclidean' distance measures straight-line distance between points
linkage_data = linkage(data, method='ward', metric='euclidean')

# Step 4: Plot the dendrogram to visualize the hierarchical clustering
dendrogram(linkage_data)
plt.title("Dendrogram of hierarchical clustering")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

# Step 5: Use scikit-learn's AgglomerativeClustering to cluster the data into 2 clusters
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

# Fit the model and predict cluster labels for each data point
labels = hierarchical_cluster.fit_predict(data)

# Step 6: Visualize the clustered data points with colors indicating cluster membership
plt.scatter(x, y, c=labels)
plt.title("Agglomerative Clustering Result")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()
