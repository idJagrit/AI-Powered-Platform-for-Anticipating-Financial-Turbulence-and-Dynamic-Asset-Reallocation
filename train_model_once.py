import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from minio import Minio
import io

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

# ===== PREPROCESS & TRAIN MODEL =====
print("Preprocessing and training model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

model = KMeans(n_clusters=3, random_state=42)
model.fit(X_scaled)

# ===== SAVE MODEL LOCALLY =====
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"scaler": scaler, "model": model}, f)
print(f"Model saved locally at {MODEL_FILE}")

# ===== UPLOAD MODEL TO MINIO =====
if not client.bucket_exists(BUCKET_NAME_MODELS):
    client.make_bucket(BUCKET_NAME_MODELS)

with open(MODEL_FILE, "rb") as f:
    client.put_object(
        BUCKET_NAME_MODELS,
        "clustering_model.pkl",  # file name in bucket
        f,
        length=os.path.getsize(MODEL_FILE),
        content_type="application/octet-stream"
    )

print(f"Model uploaded to MinIO at bucket '{BUCKET_NAME_MODELS}' as 'clustering_model.pkl'")
