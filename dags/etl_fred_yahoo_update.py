# dags/etl_fred_yahoo_update.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
import io
import logging
import pandas as pd
import numpy as np
import boto3
from botocore.client import Config
from fredapi import Fred
import yfinance as yf

# ----------------- CONFIG -----------------
BUCKET = "historical-data"
FILE_KEY = "historical_data.csv"  # your existing file in MinIO
FRED_SERIES = ['SP500', 'VIXCLS', 'T10Y2Y', 'T5YIFR', 'WLEMUINDXD']
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'WMT']

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MINIO_CONN_ID = "minio_default"  # optional Airflow connection name

# ----------------- HELPERS -----------------
def get_s3_client():
    """Return a boto3 S3 client. Prefer Airflow connection if available."""
    try:
        conn = BaseHook.get_connection(MINIO_CONN_ID)
        extra = conn.extra_dejson or {}
        endpoint = extra.get("endpoint_url", f"http://{conn.host}:9000") if conn.host else extra.get("endpoint_url", "http://minio:9000")
        aws_access = conn.login
        aws_secret = conn.password
    except Exception:
        endpoint = "http://minio:9000"
        aws_access = "minio"
        aws_secret = "minio123"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=aws_access,
        aws_secret_access_key=aws_secret,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )
    return s3

def read_historical(s3):
    """Read existing CSV from MinIO and return DataFrame indexed by Date."""
    obj = s3.get_object(Bucket=BUCKET, Key=FILE_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=[0], index_col=0)
    df.index.name = "Date"
    df = df.sort_index()
    return df

def build_column_order():
    """Return exact column order: FRED series then Close/Volume/HighLow_Spread for each ticker."""
    cols = []
    cols.extend(FRED_SERIES)
    for t in TICKERS:
        cols.append(f"Close_{t}")
        cols.append(f"Volume_{t}")
        cols.append(f"HighLow_Spread_{t}")
    return cols

def fetch_fred_yahoo(start_date, end_date, fred_key):
    """
    Fetch FRED and Yahoo data for the inclusive date range start_date..end_date.
    Returns DataFrame reindexed on daily calendar.
    """
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    fred_df = pd.DataFrame(index=idx)
    fred = Fred(api_key=fred_key)

    # FRED: series may be lower frequency — reindex to daily then fill later
    for s in FRED_SERIES:
        try:
            ser = fred.get_series(s, observation_start=start_date, observation_end=end_date)
            ser.index = pd.to_datetime(ser.index)
            fred_df[s] = ser.reindex(idx)
        except Exception as e:
            logging.warning(f"FRED fetch failed for {s}: {e}")
            fred_df[s] = np.nan

    yahoo_df = pd.DataFrame(index=idx)
    for t in TICKERS:
        try:
            # yfinance 'end' is exclusive; use end_date + 1 day to include end_date
            yf_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            stock = yf.download(t, start=start_date, end=yf_end, progress=False, auto_adjust=False)
            if stock is None or stock.empty:
                logging.warning(f"No yahoo data for {t} in range {start_date}..{end_date}")
                yahoo_df[f"Close_{t}"] = np.nan
                yahoo_df[f"Volume_{t}"] = np.nan
                yahoo_df[f"HighLow_Spread_{t}"] = np.nan
                continue
            stock.index = pd.to_datetime(stock.index)
            yahoo_df[f"Close_{t}"] = stock["Close"].reindex(idx)
            yahoo_df[f"Volume_{t}"] = stock["Volume"].reindex(idx)
            yahoo_df[f"HighLow_Spread_{t}"] = (stock["High"] - stock["Low"]).reindex(idx)
        except Exception as e:
            logging.warning(f"Yahoo fetch failed for {t}: {e}")
            yahoo_df[f"Close_{t}"] = np.nan
            yahoo_df[f"Volume_{t}"] = np.nan
            yahoo_df[f"HighLow_Spread_{t}"] = np.nan

    combined = pd.concat([fred_df, yahoo_df], axis=1)
    return combined

def fill_no_missing(combined, original_historical):
    """
    Fill missing values with the strategy:
      1) For price-like & fred columns: forward-fill then back-fill (last known value carried forward).
      2) For volume columns: fill NaN with 0 for non-trading days.
      3) If an entire column is still NaN (extremely rare), fill with 0.
    This ensures no NaNs remain.
    """
    df = combined.copy()
    price_cols = [c for c in df.columns if c.startswith("Close_") or c.startswith("HighLow_Spread_") or c in FRED_SERIES]
    vol_cols = [c for c in df.columns if c.startswith("Volume_")]

    # Forward/backward fill price and FRED series so we carry last known value
    if price_cols:
        df.loc[:, price_cols] = df.loc[:, price_cols].ffill().bfill()

    # For volumes: for newly introduced rows that are NaN because a market was closed, set 0
    if vol_cols:
        # set remaining NaNs to 0
        df.loc[:, vol_cols] = df.loc[:, vol_cols].fillna(0)

    # As additional safety: any column still all-NaN -> fill with 0
    for c in df.columns:
        if df[c].isna().all():
            df[c] = 0.0

    return df

# ----------------- DAG TASK -----------------
def etl_task(**context):
    logging.info("Starting ETL task for FRED + Yahoo update.")
    s3 = get_s3_client()

    # FRED API key must be set in Airflow Variables
    fred_key = Variable.get("FRED_API_KEY", default_var=None)
    if not fred_key:
        raise ValueError("Airflow Variable FRED_API_KEY is not set. Set it in Airflow UI -> Admin -> Variables.")

    # Read historical CSV
    try:
        hist = read_historical(s3)
        logging.info(f"Loaded historical CSV from s3://{BUCKET}/{FILE_KEY} shape={hist.shape}")
    except Exception as e:
        raise RuntimeError(f"Could not load historical CSV from s3://{BUCKET}/{FILE_KEY}: {e}")

    last_date = hist.index.max()
    start_fetch = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_fetch = datetime.utcnow().strftime("%Y-%m-%d")

    if pd.to_datetime(start_fetch) <= pd.to_datetime(end_fetch):
        logging.info(f"Fetching new data: {start_fetch} -> {end_fetch}")
        new_df = fetch_fred_yahoo(start_fetch, end_fetch, fred_key)

        # Append (prefer new rows where overlapping)
        combined = pd.concat([hist, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        logging.info("No new dates to fetch. Validating and re-writing historical CSV with fill rules.")
        combined = hist.copy()

    # Ensure continuous daily index from earliest to latest
    idx_full = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
    combined = combined.reindex(idx_full)

    # Fill missing values using the chosen strategy
    combined_filled = fill_no_missing(combined, hist)

    # Ensure exact column order and presence
    ordered_cols = build_column_order()
    for c in ordered_cols:
        if c not in combined_filled.columns:
            combined_filled[c] = 0.0
    combined_final = combined_filled[ordered_cols]

    # Write Date as first column and upload to MinIO
    out_df = combined_final.copy()
    out_df.index.name = "Date"
    csv_bytes = out_df.to_csv().encode("utf-8")

    s3.put_object(Bucket=BUCKET, Key=FILE_KEY, Body=csv_bytes, ContentType="text/csv")
    logging.info(f"Wrote updated dataset to s3://{BUCKET}/{FILE_KEY} shape={out_df.shape}")

# ----------------- DAG DEFINITION -----------------
with DAG(
    dag_id="etl_fred_yahoo_update",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:
    run_etl = PythonOperator(
        task_id="fetch_and_update_historical",
        python_callable=etl_task,
    )

    run_etl
