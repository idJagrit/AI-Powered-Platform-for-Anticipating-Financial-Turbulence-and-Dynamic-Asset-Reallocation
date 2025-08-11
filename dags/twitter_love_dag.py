from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import tweepy
import json
import os
import pandas as pd
from minio import Minio
from io import BytesIO

# Paths
CREDENTIALS_PATH = "/opt/airflow/data/twitter_api.json"

# Read credentials
with open(CREDENTIALS_PATH) as f:
    creds = json.load(f)

BEARER_TOKEN = creds["bearer_token"]

# MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)

def fetch_tweets(**kwargs):
    """Fetch tweets containing 'love' in English"""
    client = tweepy.Client(bearer_token=BEARER_TOKEN)

    query = "love lang:en -is:retweet"
    tweets = client.search_recent_tweets(
        query=query,
        max_results=50,  # Max 100
        tweet_fields=["created_at", "author_id", "text"]
    )

    # Convert to DataFrame
    tweet_data = []
    for tweet in tweets.data or []:
        tweet_data.append({
            "created_at": tweet.created_at,
            "author_id": tweet.author_id,
            "text": tweet.text
        })

    df = pd.DataFrame(tweet_data)
    print(f"Fetched {len(df)} tweets")
    kwargs['ti'].xcom_push(key="tweet_df", value=df.to_json())

def save_to_minio(**kwargs):
    """Save tweets DataFrame to MinIO"""
    ti = kwargs['ti']
    df_json = ti.xcom_pull(key="tweet_df", task_ids="fetch_tweets")
    df = pd.read_json(df_json)

    if not minio_client.bucket_exists("twitter-data"):
        minio_client.make_bucket("twitter-data")

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    file_name = f"tweets_love_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    minio_client.put_object(
        "twitter-data",
        file_name,
        buffer,
        length=buffer.getbuffer().nbytes,
        content_type="text/csv"
    )
    print(f"Saved {file_name} to MinIO bucket 'twitter-data'")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "twitter_love_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="fetch_tweets",
        python_callable=fetch_tweets,
        provide_context=True
    )

    t2 = PythonOperator(
        task_id="save_to_minio",
        python_callable=save_to_minio,
        provide_context=True
    )

    t1 >> t2
