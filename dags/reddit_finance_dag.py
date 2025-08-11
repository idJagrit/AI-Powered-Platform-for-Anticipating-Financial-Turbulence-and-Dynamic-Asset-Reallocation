from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import praw
import json
from minio import Minio
from io import BytesIO
import os

# Finance-related keywords
FINANCE_KEYWORDS = [
    "stock market", "investing", "finance", "cryptocurrency", "bitcoin", "forex",
    "trading", "NASDAQ", "S&P 500", "economy", "inflation", "interest rates"
]

# MinIO configuration
MINIO_CLIENT = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
BUCKET_NAME = "reddit-finance"

def fetch_reddit_posts():
    """Fetch Reddit posts for given finance keywords and upload to MinIO."""
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

    all_posts = []
    for keyword in FINANCE_KEYWORDS:
        for submission in reddit.subreddit("all").search(keyword, limit=20):
            all_posts.append({
                "title": submission.title,
                "score": submission.score,
                "url": submission.url,
                "created_utc": submission.created_utc,
                "keyword": keyword
            })

    # Convert to JSON
    data_bytes = BytesIO(json.dumps(all_posts, indent=2).encode("utf-8"))

    # Ensure bucket exists
    if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
        MINIO_CLIENT.make_bucket(BUCKET_NAME)

    # Save with timestamp
    file_name = f"reddit_posts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    MINIO_CLIENT.put_object(
        BUCKET_NAME,
        file_name,
        data_bytes,
        length=len(data_bytes.getvalue()),
        content_type="application/json"
    )

    print(f"✅ Uploaded {file_name} to MinIO bucket '{BUCKET_NAME}'.")

# Airflow DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reddit_finance_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    fetch_posts_task = PythonOperator(
        task_id="fetch_reddit_posts",
        python_callable=fetch_reddit_posts
    )

    fetch_posts_task
