import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Load the Data
data = pd.read_csv("Iris.csv")

# Step 2: Data Preparation
X = data.iloc[:, 1:5].values  # Considering all numeric attributes for clustering

# Print data before clustering
print("Data Before Clustering:")
print(data.head())


# Custom K-Means Implementation
def kmeans(X, n_clusters, max_iter=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    # Precompute pairwise distances
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    start_time = time.time()
    for step in range(max_iter):
        # Assign points to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    # Calculate SSE
    sse = calculate_sse(X, labels, centroids)

    total_time = round((time.time() - start_time) * 1000, 2)  # Round to 2 decimal places (milliseconds)
    return labels, centroids, sse, step + 1, total_time


# Custom K-Medoids Implementation
def kmedoids(X, n_clusters, max_iter=100):
    # Randomly initialize medoids
    medoid_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    medoids = X[medoid_indices]

    # Precompute pairwise distances
    distances = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))

    start_time = time.time()
    for step in range(max_iter):
        # Assign points to the nearest medoid
        labels = np.argmin(distances[medoid_indices, :], axis=0)

        # Update medoids
        new_medoids = np.copy(medoids)
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]  # Indices of points in cluster i
            if len(cluster_indices) > 0:
                cluster_distances = np.sum(distances[cluster_indices[:, np.newaxis], cluster_indices], axis=0)
                min_index = cluster_indices[np.argmin(cluster_distances)]
                new_medoids[i] = X[min_index]

        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids
        medoid_indices = np.where(np.all(X == medoids[:, np.newaxis], axis=-1))[1]

    # Calculate SSE
    sse = calculate_sse(X, labels, medoids)

    total_time = round((time.time() - start_time) * 1000, 2)  # Round to 2 decimal places (milliseconds)
    return labels, medoids, sse, step + 1, total_time


# Calculate SSE
def calculate_sse(X, labels, centroids):
    sse = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroid) ** 2)
    return sse


# Perform Clustering
n_clusters = 3
kmeans_labels, kmeans_centroids, kmeans_sse, kmeans_steps, kmeans_time = kmeans(X, n_clusters)
kmedoids_labels, kmedoids_medoids, kmedoids_sse, kmedoids_steps, kmedoids_time = kmedoids(X, n_clusters)

# Print Data After Clustering
data['K-Means Cluster'] = kmeans_labels
data['K-Medoids Cluster'] = kmedoids_labels
print("Data After Clustering:")
print(data.head())

# Plotting
plt.figure(figsize=(12, 10))

# Plot for K-Means
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.tight_layout()

# Plot for K-Medoids
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmedoids_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmedoids_medoids[:, 0], kmedoids_medoids[:, 1], c='red', s=200, marker='o', label='Medoids')
plt.title('K-Medoids Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.tight_layout()

# Plot for K-Means (Other Features)
plt.subplot(2, 2, 3)
plt.scatter(X[:, 2], X[:, 3], c=kmeans_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans_centroids[:, 2], kmeans_centroids[:, 3], c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.tight_layout()

# Plot for K-Medoids (Other Features)
plt.subplot(2, 2, 4)
plt.scatter(X[:, 2], X[:, 3], c=kmedoids_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmedoids_medoids[:, 2], kmedoids_medoids[:, 3], c='red', s=200, marker='o', label='Medoids')
plt.title('K-Medoids Clustering')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.tight_layout()

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Add space between subplots
plt.show()

# Print SSE, Cluster Counts, Steps, and Time Taken
print("K-Means SSE:", kmeans_sse)
print("K-Medoids SSE:", kmedoids_sse)

kmeans_cluster_counts = np.bincount(kmeans_labels)
kmedoids_cluster_counts = np.bincount(kmedoids_labels)

print("K-Means Cluster Counts:", kmeans_cluster_counts)
print("K-Medoids Cluster Counts:", kmedoids_cluster_counts)

print("K-Means Total Steps:", kmeans_steps)
print("K-Medoids Total Steps:", kmedoids_steps)

print("K-Means Time Taken (milliseconds):", kmeans_time)
print("K-Medoids Time Taken (milliseconds):", kmedoids_time)
