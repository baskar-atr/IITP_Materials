import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

def dbscan_clustering(X, eps, min_samples):
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_id = 0

    for i in range(X.shape[0]):
        if labels[i] != 0:
            continue

        neighbors = find_neighbors(X, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id
        expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels

def find_neighbors(X, index, eps):
    neighbors = []
    for i in range(X.shape[0]):
        if np.linalg.norm(X[index] - X[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(X, labels, index, neighbors, cluster_id, eps, min_samples):
    for neighbor in neighbors:
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_id
        elif labels[neighbor] == 0:
            labels[neighbor] = cluster_id
            new_neighbors = find_neighbors(X, neighbor, eps)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)

def hierarchical_clustering(X, n_clusters):
    linkage_matrix = linkage(X, method='ward')
    clusters = dendrogram(linkage_matrix, p=n_clusters, no_plot=True)['ivl']
    cluster_labels = np.zeros(X.shape[0])
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[int(idx)] = i + 1
    return cluster_labels, linkage_matrix

# Read the dataset
df = pd.read_csv('students.csv')

# Select the required columns
selected_columns = ['age', 'sports', 'music', 'shopping', 'NumberOffriends']
df = df[selected_columns]

# Drop rows with missing values
df = df.dropna()

# Convert 'age' column to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Drop rows with non-numeric values in 'age' column
df = df.dropna(subset=['age'])

# Convert 'age' column to integer
df['age'] = df['age'].astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# DBSCAN Clustering
eps = 0.5
min_samples = 5
dbscan_clusters = dbscan_clustering(X_scaled, eps, min_samples)

# Print the total number of data points for each cluster in DBSCAN
cluster_counts_dbscan = {}
for label in set(dbscan_clusters):
    if label == -1:
        continue
    cluster_counts_dbscan[label] = np.sum(dbscan_clusters == label)

print("Total Data Points in Each Cluster (DBSCAN):")
for cluster_id, count in cluster_counts_dbscan.items():
    print(f"Cluster {cluster_id}: {count} data points")

# Calculate silhouette score for DBSCAN Clustering
silhouette_score_dbscan = silhouette_score(X_scaled, dbscan_clusters)
print("Silhouette Score for DBSCAN Clustering:", silhouette_score_dbscan)

# Hierarchical Clustering
n_clusters_hierarchical = 15
hierarchical_clusters, linkage_matrix = hierarchical_clustering(X_scaled, n_clusters_hierarchical)

# Calculate silhouette score for Hierarchical Clustering
silhouette_score_hierarchical = silhouette_score(X_scaled, hierarchical_clusters)
print("Silhouette Score for Hierarchical Clustering:", silhouette_score_hierarchical)
truncate_threshold = 100  # Set a threshold for the number of clusters to display in the dendrogram

# Plot DBSCAN Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.colorbar(label='Cluster')
plt.show()

# Plot DBSCAN Clustering for age vs music
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=dbscan_clusters, cmap='viridis')
plt.title('DBSCAN Clustering: Age vs Music')
plt.xlabel('Age')
plt.ylabel('Music')
plt.colorbar(label='Cluster')
plt.show()

# Plot DBSCAN Clustering for age vs shopping
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 3], c=dbscan_clusters, cmap='viridis')
plt.title('DBSCAN Clustering: Age vs Shopping')
plt.xlabel('Age')
plt.ylabel('Shopping')
plt.colorbar(label='Cluster')
plt.show()

# Plot DBSCAN Clustering for age vs NumberOffriends
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 4], c=dbscan_clusters, cmap='viridis')
plt.title('DBSCAN Clustering: Age vs Number of Friends')
plt.xlabel('Age')
plt.ylabel('Number of Friends')
plt.colorbar(label='Cluster')
plt.show()


# Plot Hierarchical Clustering Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_clusters, cmap='viridis')
plt.title('Hierarchical Clustering Scatter Plot')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.colorbar(label='Cluster')
plt.show()

# Plotting dendrogram for Hierarchical Clustering
# Plotting dendrogram for Hierarchical Clustering
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10, truncate_mode='lastp')
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
