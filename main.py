# Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import umap.umap_ as umap
import matplotlib.pyplot as plt

# File Paths
adult_path = 'C:/Users/chaitu/OneDrive/Desktop/adult.csv'
bank_full_path = 'C:/Users/chaitu/OneDrive/Desktop/bank-full.csv'

# Load Datasets
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

adult_target = adult_df["income"].sample(n=500, random_state=42).values
bank_target = bank_full_df["y"].sample(n=500, random_state=42).values

# Dimensionality Reduction
# PCA
pca_adult_reduced = PCA(n_components=2).fit_transform(adult_sample)
pca_bank_reduced = PCA(n_components=2).fit_transform(bank_sample)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_adult_reduced = umap_reducer.fit_transform(adult_sample)
umap_bank_reduced = umap_reducer.fit_transform(bank_sample)

# K-Means and Hierarchical Clustering Visualization Functions
def visualize_kmeans(data, dataset_name):
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title(f"K-Means Clustering ({dataset_name})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid()
    plt.show()

def visualize_hierarchical(data, dataset_name):
    linkage_matrix = linkage(data, method="ward")
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode="level", p=5)
    plt.title(f"Hierarchical Clustering Dendrogram ({dataset_name})")
    plt.xlabel("Data Points (truncated)")
    plt.ylabel("Distance")
    plt.show()

# Re-run Clustering on Reduced Data
print("\nRe-running Clustering on Reduced Data (PCA & UMAP)...")
visualize_kmeans(pca_adult_reduced, "Adult Dataset (PCA)")
visualize_hierarchical(pca_adult_reduced, "Adult Dataset (PCA)")
visualize_kmeans(umap_adult_reduced, "Adult Dataset (UMAP)")
visualize_hierarchical(umap_adult_reduced, "Adult Dataset (UMAP)")

visualize_kmeans(pca_bank_reduced, "Bank Dataset (PCA)")
visualize_hierarchical(pca_bank_reduced, "Bank Dataset (PCA)")
visualize_kmeans(umap_bank_reduced, "Bank Dataset (UMAP)")
visualize_hierarchical(umap_bank_reduced, "Bank Dataset (UMAP)")

# Ensemble Models (AdaBoost and Random Forest) with Hyperparameter Tuning
def tune_and_evaluate(data, target, dataset_name, method="Original"):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    
    # Hyperparameter grids
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    ab_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='accuracy', verbose=0)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_train_score = rf_best.score(X_train, y_train)
    rf_test_score = rf_best.score(X_test, y_test)
    
    # AdaBoost
    ab = AdaBoostClassifier(random_state=42)
    ab_grid = GridSearchCV(ab, ab_param_grid, cv=3, scoring='accuracy', verbose=0)
    ab_grid.fit(X_train, y_train)
    ab_best = ab_grid.best_estimator_
    ab_train_score = ab_best.score(X_train, y_train)
    ab_test_score = ab_best.score(X_test, y_test)
    
    # Results
    results = pd.DataFrame({
        "Model": ["Random Forest", "AdaBoost"],
        "Train Score": [rf_train_score, ab_train_score],
        "Test Score": [rf_test_score, ab_test_score],
        "Best Params": [rf_grid.best_params_, ab_grid.best_params_]
    })
    print(f"\nResults for {dataset_name} ({method} Data):")
    print(results)
    return results

# Evaluate on Original and Reduced Data
print("\nTuning and Evaluating Models...")
adult_original_results = tune_and_evaluate(adult_sample, adult_target, "Adult Dataset", method="Original")
adult_pca_results = tune_and_evaluate(pca_adult_reduced, adult_target, "Adult Dataset", method="PCA")
adult_umap_results = tune_and_evaluate(umap_adult_reduced, adult_target, "Adult Dataset", method="UMAP")

bank_original_results = tune_and_evaluate(bank_sample, bank_target, "Bank Dataset", method="Original")
bank_pca_results = tune_and_evaluate(pca_bank_reduced, bank_target, "Bank Dataset", method="PCA")
bank_umap_results = tune_and_evaluate(umap_bank_reduced, bank_target, "Bank Dataset", method="UMAP")
