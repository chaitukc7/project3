import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# File paths
adult_path = 'C:/Users/chaitu/OneDrive/Desktop/adult.csv'
bank_full_path = 'C:/Users/chaitu/OneDrive/Desktop/bank-full.csv'

# Load the datasets
adult_df = pd.read_csv(adult_path)
bank_full_df = pd.read_csv(bank_full_path, sep=';', encoding='utf-8')

# Preprocessing
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Encode categorical variables for Adult Dataset
categorical_columns_adult = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "gender", "native-country", "income"
]
for col in categorical_columns_adult:
    if col in adult_df.columns:
        adult_df[col] = label_encoder.fit_transform(adult_df[col])

# Normalize numerical features for Adult Dataset
numeric_columns_adult = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
adult_df[numeric_columns_adult] = scaler.fit_transform(adult_df[numeric_columns_adult])

# Encode categorical variables for Bank Dataset
categorical_columns_bank = [
    "job", "marital", "education", "default", "housing", "loan", 
    "contact", "month", "poutcome", "y"
]
for col in categorical_columns_bank:
    if col in bank_full_df.columns:
        bank_full_df[col] = label_encoder.fit_transform(bank_full_df[col])

# Normalize numerical features for Bank Dataset
numeric_columns_bank = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
bank_full_df[numeric_columns_bank] = scaler.fit_transform(bank_full_df[numeric_columns_bank])

# Sampling for analysis
adult_sample = adult_df.sample(n=500, random_state=42).drop(columns=["income"], errors='ignore')
bank_sample = bank_full_df.sample(n=500, random_state=42).drop(columns=["y"], errors='ignore')

# K-Means Clustering Visualization
def visualize_kmeans_all_features(data, dataset_name, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title(f"K-Means Clustering ({dataset_name}) with k={k}")
    plt.xlabel("Dimension 1 (from data)")
    plt.ylabel("Dimension 2 (from data)")
    plt.grid()
    plt.show()

# Hierarchical Clustering Visualization
def visualize_hierarchical_all_features(data, dataset_name):
    linkage_matrix = linkage(data, method="ward")
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode="level", p=5)
    plt.title(f"Hierarchical Clustering Dendrogram ({dataset_name})")
    plt.xlabel("Data Points (truncated)")
    plt.ylabel("Distance")
    plt.show()

# PCA Visualization
def visualize_pca_all_features(data, dataset_name, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_.sum()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, s=50, c='blue')
    plt.title(f"PCA - {dataset_name} (Explained Variance: {explained_variance:.2f})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()

# UMAP Visualization
def visualize_umap(data, dataset_name, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_data = reducer.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6, s=50, c='green')
    plt.title(f"UMAP - {dataset_name}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.grid()
    plt.show()

# Perform and visualize analysis for Adult Dataset
print("Visualizing Analysis for Adult Dataset...")
visualize_kmeans_all_features(adult_sample, "Adult Dataset")
visualize_hierarchical_all_features(adult_sample, "Adult Dataset")
visualize_pca_all_features(adult_sample, "Adult Dataset")
visualize_umap(adult_sample, "Adult Dataset")

# Perform and visualize analysis for Bank Dataset
print("Visualizing Analysis for Bank Dataset...")
visualize_kmeans_all_features(bank_sample, "Bank Dataset")
visualize_hierarchical_all_features(bank_sample, "Bank Dataset")
visualize_pca_all_features(bank_sample, "Bank Dataset")
visualize_umap(bank_sample, "Bank Dataset")


# Step 1: Dimensionality Reduction Results
# PCA-Reduced Data
pca_adult_reduced = PCA(n_components=2).fit_transform(adult_sample)
pca_bank_reduced = PCA(n_components=2).fit_transform(bank_sample)

# UMAP-Reduced Data
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_adult_reduced = umap_reducer.fit_transform(adult_sample)
umap_bank_reduced = umap_reducer.fit_transform(bank_sample)

# Step 2: Re-run K-means and Hierarchical Clustering on Dimensionality-Reduced Data
def re_run_clustering_on_reduced_data(reduced_data, dataset_name, method="PCA"):
    print(f"Re-running K-Means on {dataset_name} ({method}-Reduced)...")
    visualize_kmeans_all_features(pd.DataFrame(reduced_data), f"{dataset_name} ({method}-Reduced)")

    print(f"Re-running Hierarchical Clustering on {dataset_name} ({method}-Reduced)...")
    visualize_hierarchical_all_features(pd.DataFrame(reduced_data), f"{dataset_name} ({method}-Reduced)")

# Adult Dataset
re_run_clustering_on_reduced_data(pca_adult_reduced, "Adult Dataset", method="PCA")
re_run_clustering_on_reduced_data(umap_adult_reduced, "Adult Dataset", method="UMAP")

# Bank Dataset
re_run_clustering_on_reduced_data(pca_bank_reduced, "Bank Dataset", method="PCA")
re_run_clustering_on_reduced_data(umap_bank_reduced, "Bank Dataset", method="UMAP")

# Step 3: Train AdaBoost and Random Forest on Original and Reduced Data
def train_ensemble_models(data, target, dataset_name, method="Original"):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest ({dataset_name} - {method}): Accuracy = {rf_acc:.4f}")

    # AdaBoost
    ab_model = AdaBoostClassifier(random_state=42)
    ab_model.fit(X_train, y_train)
    ab_preds = ab_model.predict(X_test)
    ab_acc = accuracy_score(y_test, ab_preds)
    print(f"AdaBoost ({dataset_name} - {method}): Accuracy = {ab_acc:.4f}")

# Prepare Target Variables for Original Data
adult_target = adult_df["income"] if "income" in adult_df else [0] * len(adult_df)
bank_target = bank_full_df["y"] if "y" in bank_full_df else [0] * len(bank_full_df)

# Train on Original Data
print("\nTraining on Original Data...")
train_ensemble_models(adult_sample, adult_target[:500], "Adult Dataset", method="Original")
train_ensemble_models(bank_sample, bank_target[:500], "Bank Dataset", method="Original")

# Train on PCA-Reduced Data
print("\nTraining on PCA-Reduced Data...")
train_ensemble_models(pca_adult_reduced, adult_target[:500], "Adult Dataset", method="PCA")
train_ensemble_models(pca_bank_reduced, bank_target[:500], "Bank Dataset", method="PCA")

# Train on UMAP-Reduced Data
print("\nTraining on UMAP-Reduced Data...")
train_ensemble_models(umap_adult_reduced, adult_target[:500], "Adult Dataset", method="UMAP")
train_ensemble_models(umap_bank_reduced, bank_target[:500], "Bank Dataset", method="UMAP")

