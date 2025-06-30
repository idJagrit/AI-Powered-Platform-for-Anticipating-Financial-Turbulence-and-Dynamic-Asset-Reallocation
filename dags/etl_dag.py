from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from etl_pipeline import run_etl

with DAG(
    dag_id="financial_etl",
    start_date=datetime(2025, 6, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:
    etl_task = PythonOperator(
        task_id="run_etl_pipeline",
        python_callable=run_etl
    )