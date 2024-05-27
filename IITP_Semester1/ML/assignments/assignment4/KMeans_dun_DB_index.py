import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# Load the dataset
iris_data = pd.read_csv("Iris.csv")

# Drop any missing values if present
iris_data.dropna(inplace=True)

# Separate features and target variable
X = iris_data.iloc[:, :-1]  # Features

# Apply the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply cluster validity Dunn index to find the better cluster
def dunn_index(X, labels):
    clusters = []
    for cluster in np.unique(labels):
        clusters.append(X[labels == cluster])

    min_inter_cluster_distances = []
    max_intra_cluster_distances = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            inter_cluster_distance = np.linalg.norm(clusters[i].mean(axis=0) - clusters[j].mean(axis=0))
            min_inter_cluster_distances.append(inter_cluster_distance)

    for cluster in clusters:
        intra_cluster_distance = pdist(cluster).max()
        max_intra_cluster_distances.append(intra_cluster_distance)

    dunn_index = np.min(min_inter_cluster_distances) / np.max(max_intra_cluster_distances)
    return dunn_index

dunn_scores = []
pairwise_dists = squareform(pdist(X))
for num_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(X)
    dunn_scores.append(dunn_index(X.values, cluster_labels))

optimal_clusters_dunn = dunn_scores.index(max(dunn_scores)) + 2
print("Optimal clusters using Dunn Index:", optimal_clusters_dunn)

# Apply cluster validity Davies Bouldin index to find the better cluster
db_scores = []
for num_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(X)
    db_scores.append(davies_bouldin_score(X, cluster_labels))

optimal_clusters_db = db_scores.index(min(db_scores)) + 2
print("Optimal clusters using Davies Bouldin Index:", optimal_clusters_db)

# Applying the Silhouette index to find the similarity between the clusters
silhouette_scores = []
for num_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, cluster_labels))

optimal_clusters_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
print("Optimal clusters using Silhouette Score:", optimal_clusters_silhouette)

# Plot pairplot for Elbow Method clusters
kmeans_elbow = KMeans(n_clusters=optimal_clusters_silhouette)
cluster_labels_elbow = kmeans_elbow.fit_predict(X)
iris_data['Cluster_Labels_Elbow'] = cluster_labels_elbow
sns.pairplot(iris_data, hue='Cluster_Labels_Elbow')
plt.suptitle(f'Pairplot for Elbow Method Clusters (Clusters: {optimal_clusters_silhouette})', fontsize=16, fontweight='bold')
plt.show()

# Plot pairplot for Dunn Index clusters
kmeans_dunn = KMeans(n_clusters=optimal_clusters_dunn)
cluster_labels_dunn = kmeans_dunn.fit_predict(X)
iris_data['Cluster_Labels_Dunn'] = cluster_labels_dunn
sns.pairplot(iris_data, hue='Cluster_Labels_Dunn')
plt.suptitle(f'Pairplot for Dunn Index Clusters (Clusters: {optimal_clusters_dunn})', fontsize=16, fontweight='bold')
plt.show()

# Plot pairplot for Davies Bouldin Index clusters
kmeans_db = KMeans(n_clusters=optimal_clusters_db)
cluster_labels_db = kmeans_db.fit_predict(X)
iris_data['Cluster_Labels_DB'] = cluster_labels_db
sns.pairplot(iris_data, hue='Cluster_Labels_DB')
plt.suptitle(f'Pairplot for Davies Bouldin Index Clusters (Clusters: {optimal_clusters_db})', fontsize=16, fontweight='bold')
plt.show()

# Plot pairplot for Silhouette Index clusters
kmeans_silhouette = KMeans(n_clusters=optimal_clusters_silhouette)
cluster_labels_silhouette = kmeans_silhouette.fit_predict(X)
iris_data['Cluster_Labels_Silhouette'] = cluster_labels_silhouette
sns.pairplot(iris_data, hue='Cluster_Labels_Silhouette')
plt.suptitle(f'Pairplot for Silhouette Index Clusters (Clusters: {optimal_clusters_silhouette})', fontsize=16, fontweight='bold')
plt.show()

# Compare silhouette scores to determine the best clustering method
silhouette_scores = {
    "Elbow Method": silhouette_score(X, cluster_labels_elbow),
    "Dunn Index": silhouette_score(X, cluster_labels_dunn),
    "Davies Bouldin Index": silhouette_score(X, cluster_labels_db),
    "Silhouette Index": silhouette_score(X, cluster_labels_silhouette)
}

best_method = max(silhouette_scores, key=silhouette_scores.get)
optimal_clusters = {
    "Elbow Method": optimal_clusters_silhouette,
    "Dunn Index": optimal_clusters_dunn,
    "Davies Bouldin Index": optimal_clusters_db,
    "Silhouette Index": optimal_clusters_silhouette
}

print("Silhouette Scores:")
for method, score in silhouette_scores.items():
    print(f"{method}: {score}")

print("Best Clustering Method:", best_method)
print("Optimal Clusters for Best Method:", optimal_clusters[best_method])
