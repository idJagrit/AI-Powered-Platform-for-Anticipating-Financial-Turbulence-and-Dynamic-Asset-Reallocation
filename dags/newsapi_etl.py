from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from newsapi import NewsApiClient
import boto3
import json
import psycopg2
from io import BytesIO

MINIO_ENDPOINT = "http://minio:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
BUCKET_NAME = "financial-news"

PG_CONN = {
    "dbname": "airflow",
    "user": "airflow",
    "password": "airflow",
    "host": "postgres",
    "port": 5432
}

def extract_newsapi(**context):
    newsapi = NewsApiClient(api_key="91629e011d1e4b0d87ac3b4e0f0e196b")
    data = newsapi.get_everything(
        q="finance OR stock OR economy OR business",
        domains="bloomberg.com,reuters.com,cnbc.com,ft.com",  # Financial sources
        language="en",
        page_size=20,
        sort_by="publishedAt")
    context["ti"].xcom_push(key="raw_data", value=data)

def save_to_minio(**context):
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    raw_data = context["ti"].xcom_pull(key="raw_data", task_ids="extract_newsapi")
    key = f"raw/newsapi/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps(raw_data))

def save_metadata_postgres(**context):
    raw_data = context["ti"].xcom_pull(key="raw_data", task_ids="extract_newsapi")
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_metadata (
            source TEXT,
            title TEXT,
            published_at TIMESTAMP
        )
    """)
    for article in raw_data.get("articles", []):
        cur.execute(
            "INSERT INTO news_metadata (source, title, published_at) VALUES (%s, %s, %s)",
            (article["source"]["name"], article["title"], article["publishedAt"])
        )
    conn.commit()
    cur.close()
    conn.close()

with DAG(
    "newsapi_etl",
    start_date=datetime(2025, 8, 8),
    schedule_interval="@daily",
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id="extract_newsapi",
        python_callable=extract_newsapi
    )

    t2 = PythonOperator(
        task_id="save_to_minio",
        python_callable=save_to_minio
    )

    t3 = PythonOperator(
        task_id="save_metadata_postgres",
        python_callable=save_metadata_postgres
    )

    t1 >> t2 >> t3
