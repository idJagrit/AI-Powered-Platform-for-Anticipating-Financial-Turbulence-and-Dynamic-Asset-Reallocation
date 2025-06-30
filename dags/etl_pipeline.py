import pandas as pd
import yfinance as yf
import requests
import tweepy
import praw
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lower, regexp_replace
from datetime import datetime
import os
from dotenv import load_dotenv

# Initialize Spark in local mode
spark = SparkSession.builder.appName("FinancialETL").master("local[*]").getOrCreate()

# Constants (replace with your API keys)
load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Extract: Market Data (yfinance)
def extract_market_data():
    ticker = "SPY"  # Example: S&P 500 ETF
    data = yf.download(ticker, start="2025-06-01", end="2025-06-30")
    data['date'] = data.index
    return spark.createDataFrame(data.reset_index(drop=True))

# Extract: Macro Data (FRED API)
def extract_macro_data():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GDP",
        "api_key": os.getenv("FRED_API_KEY"),
        "file_type": "json",
        "observation_start": "2025-01-01"
    }
    response = requests.get(url, params=params)
    data = response.json()['observations']
    df = pd.DataFrame(data)
    return spark.createDataFrame(df)

# Extract: News Data (NewsAPI)
def extract_news_data():
    url = f"https://newsapi.org/v2/everything?q=finance&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json()['articles']
    df = pd.DataFrame(articles)
    return spark.createDataFrame(df)

# Extract: Twitter Data (tweepy)
def extract_twitter_data():
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    query = "#finance OR #stocks -is:retweet"
    tweets = client.search_recent_tweets(query=query, max_results=100)
    data = [{"text": tweet.text, "created_at": tweet.created_at} for tweet in tweets.data]
    return spark.createDataFrame(data)

# Extract: Reddit Data (praw)
def extract_reddit_data():
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
    subreddit = reddit.subreddit("wallstreetbets")
    posts = [{"title": post.title, "selftext": post.selftext, "created": post.created} for post in subreddit.hot(limit=100)]
    return spark.createDataFrame(posts)

# Transform: Clean and preprocess data
def transform_data(df, source):
    if source in ["news", "twitter", "reddit"]:
        # Clean text: remove URLs, emojis, lowercase
        df = df.withColumn("text_clean", lower(regexp_replace(col("text"), r"http\S+|www\S+", "")))
        df = df.withColumn("text_clean", regexp_replace(col("text_clean"), r"[^\w\s]", ""))
    elif source in ["market", "macro"]:
        # Handle missing values and normalize dates
        df = df.na.fill(0).withColumn("date", to_date(col("date")))
    return df

# Load: Save to Parquet
def load_data(df, source):
    output_path = f"D:/finance/data/{source}/{datetime.now().strftime('%Y%m%d')}"
    df.write.mode("overwrite").parquet(output_path)

# Main ETL function
def run_etl():
    # Extract
    market_df = extract_market_data()
    macro_df = extract_macro_data()
    news_df = extract_news_data()
    twitter_df = extract_twitter_data()
    reddit_df = extract_reddit_data()

    # Transform
    market_df = transform_data(market_df, "market")
    macro_df = transform_data(macro_df, "macro")
    news_df = transform_data(news_df, "news")
    twitter_df = transform_data(twitter_df, "twitter")
    reddit_df = transform_data(reddit_df, "reddit")

    # Load
    load_data(market_df, "market")
    load_data(macro_df, "macro")
    load_data(news_df, "news")
    load_data(twitter_df, "twitter")
    load_data(reddit_df, "reddit")

if __name__ == "__main__":
    run_etl()