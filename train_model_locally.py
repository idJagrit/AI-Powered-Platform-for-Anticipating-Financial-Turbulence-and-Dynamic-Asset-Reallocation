import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from minio import Minio
import io
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIG =====
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET_NAME_DATA = "historical-data"
BUCKET_NAME_MODELS = "models"
DATA_FILE = "historical_data.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "clustering_model.pkl")

# ===== CONNECT TO MINIO =====
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# ===== DOWNLOAD CSV FROM MINIO =====
print("Downloading historical dataset from MinIO...")
data_obj = client.get_object(BUCKET_NAME_DATA, DATA_FILE)
df = pd.read_csv(io.BytesIO(data_obj.read()))
print(f"Data shape: {df.shape}")

# ===== ADVANCED FEATURE ENGINEERING =====
print("Performing advanced feature engineering...")

# Convert Date column if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

# Select numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"Found {len(numeric_cols)} numeric columns")

# Create advanced financial features
feature_df = df[numeric_cols].copy()

# 1. Volatility features (rolling std)
for col in ['SP500', 'VIXCLS']:
    if col in feature_df.columns:
        feature_df[f'{col}_volatility_7d'] = feature_df[col].rolling(7).std()
        feature_df[f'{col}_volatility_30d'] = feature_df[col].rolling(30).std()

# 2. Momentum features (price changes)
price_cols = [col for col in numeric_cols if 'Close_' in col or col == 'SP500']
for col in price_cols:
    if col in feature_df.columns:
        feature_df[f'{col}_return_1d'] = feature_df[col].pct_change(1)
        feature_df[f'{col}_return_7d'] = feature_df[col].pct_change(7)
        feature_df[f'{col}_return_30d'] = feature_df[col].pct_change(30)
        
        # Moving averages
        feature_df[f'{col}_ma_7d'] = feature_df[col].rolling(7).mean()
        feature_df[f'{col}_ma_30d'] = feature_df[col].rolling(30).mean()
        feature_df[f'{col}_ma_ratio'] = feature_df[col] / feature_df[f'{col}_ma_30d']

# 3. Volume-based features
volume_cols = [col for col in numeric_cols if 'Volume_' in col]
for col in volume_cols:
    if col in feature_df.columns:
        feature_df[f'{col}_ma_7d'] = feature_df[col].rolling(7).mean()
        feature_df[f'{col}_ratio'] = feature_df[col] / feature_df[f'{col}_ma_7d']

# 4. Market stress indicators
if 'VIXCLS' in feature_df.columns and 'SP500' in feature_df.columns:
    feature_df['market_stress'] = feature_df['VIXCLS'] / feature_df['SP500'].rolling(30).std()

# 5. Yield curve features
yield_cols = ['T10Y2Y', 'T5YIFR']
for col in yield_cols:
    if col in feature_df.columns:
        feature_df[f'{col}_ma_30d'] = feature_df[col].rolling(30).mean()
        feature_df[f'{col}_deviation'] = feature_df[col] - feature_df[f'{col}_ma_30d']

# Remove rows with NaN values (due to rolling calculations)
feature_df = feature_df.dropna()
print(f"Feature engineering complete. New shape: {feature_df.shape}")

# ===== OUTLIER DETECTION =====
print("Detecting outliers...")
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = isolation_forest.fit_predict(feature_df)
feature_df['is_outlier'] = outliers

# ===== PREPROCESSING =====
print("Preprocessing data...")
# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(feature_df.drop(['is_outlier'], axis=1))

# ===== DIMENSIONALITY REDUCTION =====
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced features from {X_scaled.shape[1]} to {X_pca.shape[1]}")

# ===== OPTIMAL CLUSTER SELECTION =====
print("Finding optimal number of clusters...")
cluster_range = range(2, 11)
silhouette_scores = []
inertias = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, cluster_labels))
    inertias.append(kmeans.inertia_)

# Find optimal clusters using silhouette score
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_clusters}")

# ===== TRAIN MULTIPLE MODELS =====
print("Training multiple clustering models...")

# 1. K-Means with optimal clusters
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_pca)

# 2. DBSCAN for density-based clustering
dbscan_model = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan_model.fit_predict(X_pca)

# ===== MODEL EVALUATION =====
print("Evaluating models...")
models_performance = {}

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_pca, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_pca, kmeans_labels)

models_performance['kmeans'] = {
    'silhouette_score': kmeans_silhouette,
    'calinski_harabasz_score': kmeans_calinski,
    'davies_bouldin_score': kmeans_davies,
    'n_clusters': optimal_clusters
}

# Evaluate DBSCAN (if it found clusters)
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X_pca, dbscan_labels)
    dbscan_calinski = calinski_harabasz_score(X_pca, dbscan_labels)
    dbscan_davies = davies_bouldin_score(X_pca, dbscan_labels)
    
    models_performance['dbscan'] = {
        'silhouette_score': dbscan_silhouette,
        'calinski_harabasz_score': dbscan_calinski,
        'davies_bouldin_score': dbscan_davies,
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    }

# Select best model based on silhouette score
best_model_name = max(models_performance.keys(), 
                     key=lambda x: models_performance[x]['silhouette_score'])
best_model = kmeans_model if best_model_name == 'kmeans' else dbscan_model

print(f"Best model: {best_model_name}")
print(f"Performance metrics: {models_performance[best_model_name]}")

# ===== RISK LEVEL ASSIGNMENT =====
print("Assigning risk levels...")
if best_model_name == 'kmeans':
    labels = kmeans_labels
else:
    labels = dbscan_labels

# Create risk mapping based on cluster characteristics
cluster_stats = pd.DataFrame({
    'cluster': labels,
    'vix': feature_df['VIXCLS'].values if 'VIXCLS' in feature_df.columns else np.random.randn(len(labels)),
    'sp500_volatility': feature_df['SP500_volatility_30d'].values if 'SP500_volatility_30d' in feature_df.columns else np.random.randn(len(labels))
}).groupby('cluster').mean()

# Assign risk based on VIX and volatility levels
risk_mapping = {}
for cluster in cluster_stats.index:
    if cluster == -1:  # DBSCAN noise points
        risk_mapping[cluster] = "High Risk"
    else:
        vix_level = cluster_stats.loc[cluster, 'vix']
        vol_level = cluster_stats.loc[cluster, 'sp500_volatility']
        
        if vix_level > 25 or vol_level > 20:  # High volatility thresholds
            risk_mapping[cluster] = "High Risk"
        elif vix_level > 15 or vol_level > 10:  # Medium volatility thresholds
            risk_mapping[cluster] = "Medium Risk"
        else:
            risk_mapping[cluster] = "Low Risk"

print(f"Risk mapping: {risk_mapping}")

# ===== SAVE ENHANCED MODEL =====
model_package = {
    "model": best_model,
    "scaler": scaler,
    "pca": pca,
    "isolation_forest": isolation_forest,
    "risk_mapping": risk_mapping,
    "feature_columns": feature_df.drop(['is_outlier'], axis=1).columns.tolist(),
    "model_performance": models_performance,
    "best_model_name": best_model_name,
    "optimal_clusters": optimal_clusters,
    "metadata": {
        "model_type": best_model_name,
        "n_features_original": len(numeric_cols),
        "n_features_engineered": feature_df.shape[1] - 1,  # -1 for is_outlier
        "n_features_pca": X_pca.shape[1],
        "silhouette_score": models_performance[best_model_name]['silhouette_score'],
        "training_samples": X_pca.shape[0],
        "variance_explained": sum(pca.explained_variance_ratio_)
    }
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model_package, f)
print(f"Enhanced model saved locally at {MODEL_FILE}")

# ===== UPLOAD MODEL TO MINIO =====
if not client.bucket_exists(BUCKET_NAME_MODELS):
    client.make_bucket(BUCKET_NAME_MODELS)

with open(MODEL_FILE, "rb") as f:
    client.put_object(
        BUCKET_NAME_MODELS,
        "clustering_model.pkl",
        f,
        length=os.path.getsize(MODEL_FILE),
        content_type="application/octet-stream"
    )

print(f"Enhanced model uploaded to MinIO at bucket '{BUCKET_NAME_MODELS}' as 'clustering_model.pkl'")
print("\n=== MODEL TRAINING COMPLETE ===")
print(f"Best model: {best_model_name}")
print(f"Silhouette Score: {models_performance[best_model_name]['silhouette_score']:.4f}")
print(f"Number of clusters: {models_performance[best_model_name]['n_clusters']}")
print(f"Features: {len(numeric_cols)} → {feature_df.shape[1]-1} → {X_pca.shape[1]} (after PCA)")
print(f"Variance explained by PCA: {sum(pca.explained_variance_ratio_):.4f}")