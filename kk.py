# Required Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

# Simulated Datasets (Replace with actual datasets for real analysis)
np.random.seed(42)
adult_sample = pd.DataFrame(np.random.rand(500, 10), columns=[f"feature{i}" for i in range(10)])
bank_sample = pd.DataFrame(np.random.rand(500, 10), columns=[f"feature{i}" for i in range(10)])

# Dimensionality Reduction
# PCA
pca_adult_reduced = PCA(n_components=2).fit_transform(adult_sample)
pca_bank_reduced = PCA(n_components=2).fit_transform(bank_sample)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_adult_reduced = umap_reducer.fit_transform(adult_sample)
umap_bank_reduced = umap_reducer.fit_transform(bank_sample)

# Function to calculate clustering metrics
def evaluate_clustering(data, labels, dataset_name, method="Original"):
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    print(f"\nClustering Metrics for {dataset_name} ({method} Data):")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")

# Function to determine the optimal number of clusters using the Elbow Method
def elbow_method(data, dataset_name, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title(f"Elbow Method for Optimal Clusters ({dataset_name})")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.grid()
    plt.show()

# Evaluate clustering for Original, PCA, and UMAP data
def analyze_clustering(data, dataset_name, method="Original"):
    print(f"\nAnalyzing Clustering for {dataset_name} ({method} Data)...")
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    evaluate_clustering(data, kmeans_labels, dataset_name, method=method)
    elbow_method(data, dataset_name)

# Clustering Analysis for Adult Dataset
analyze_clustering(adult_sample, "Adult Dataset", method="Original")
analyze_clustering(pca_adult_reduced, "Adult Dataset", method="PCA")
analyze_clustering(umap_adult_reduced, "Adult Dataset", method="UMAP")

# Clustering Analysis for Bank Dataset
analyze_clustering(bank_sample, "Bank Dataset", method="Original")
analyze_clustering(pca_bank_reduced, "Bank Dataset", method="PCA")
analyze_clustering(umap_bank_reduced, "Bank Dataset", method="UMAP")
