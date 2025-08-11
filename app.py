import streamlit as st
import boto3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from botocore.client import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Page Configuration
# -------------------
st.set_page_config(
    page_title="Financial News Sentiment Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# -------------------
# Custom CSS for Professional Look
# -------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e8f4fd;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .news-card {
        border: 1px solid #e0e6ed;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .source-badge {
        background: #f8f9fa;
        color: #495057;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# -------------------
# S3 / MinIO Settings (Unchanged)
# -------------------
S3_ENDPOINT = "http://localhost:9000"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"
BUCKET_NAME = "financial-news"
PREFIX = "raw/newsapi/"

# -------------------
# Load FinBERT (Unchanged)
# -------------------
@st.cache_resource
def load_finbert():
    with st.spinner("Loading FinBERT model..."):
        model_name = "yiyanghkust/finbert-tone"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# -------------------
# S3 Client (Unchanged)
# -------------------
@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

# -------------------
# Fetch Latest Articles (Enhanced)
# -------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_articles():
    s3 = get_s3_client()

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    if "Contents" not in response:
        st.error("No files found in the bucket path.")
        return [], None

    sorted_files = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
    latest_file_key = sorted_files[0]["Key"]
    last_modified = sorted_files[0]["LastModified"]

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=latest_file_key)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    return data.get("articles", []), last_modified

# -------------------
# Enhanced Sentiment Analysis
# -------------------
def analyze_sentiment_batch(texts, tokenizer, model):
    """Analyze sentiment for multiple texts efficiently"""
    results = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            label_id = torch.argmax(probs).item()
        
        labels = ["negative", "neutral", "positive"]
        results.append({
            'sentiment': labels[label_id],
            'confidence': probs[label_id].item(),
            'scores': {
                'negative': probs[0].item(),
                'neutral': probs[1].item(),
                'positive': probs[2].item()
            }
        })
    
    return results

def extract_keywords(text, top_n=10):
    """Extract keywords from text"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter out common words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    words = [word for word in words if word not in stop_words and len(word) > 3]
    return Counter(words).most_common(top_n)

# -------------------
# Main App
# -------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Financial News Sentiment Intelligence</h1>
    <p>Real-time sentiment analysis of financial news using advanced NLP and FinBERT</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🎛️ Dashboard Controls")

# Load model
with st.spinner("Initializing FinBERT model..."):
    tokenizer, model = load_finbert()

# Fetch articles
with st.spinner("Fetching latest financial news..."):
    articles, last_modified = fetch_latest_articles()

if not articles:
    st.error("❌ No articles found. Please check your MinIO setup.")
    st.stop()

# Sidebar info
st.sidebar.success(f"✅ Loaded {len(articles)} articles")
if last_modified:
    st.sidebar.info(f"📅 Last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar filters
st.sidebar.subheader("🔍 Filters")
num_articles = st.sidebar.slider("Number of articles to analyze", 5, min(len(articles), 100), 20)
sentiment_filter = st.sidebar.multiselect(
    "Filter by sentiment", 
    ["positive", "negative", "neutral"], 
    default=["positive", "negative", "neutral"]
)

show_images = st.sidebar.checkbox("Show article images", True)
show_detailed_analysis = st.sidebar.checkbox("Show detailed analysis", False)

# -------------------
# Analyze Sentiment
# -------------------
st.header("🤖 AI-Powered Sentiment Analysis")

# Prepare data for analysis
articles_subset = articles[:num_articles]
texts_to_analyze = [f"{article.get('title', '')} {article.get('description', '')}" for article in articles_subset]

# Perform batch sentiment analysis
with st.spinner(f"Analyzing sentiment for {len(articles_subset)} articles..."):
    sentiment_results = analyze_sentiment_batch(texts_to_analyze, tokenizer, model)

# Add sentiment results to articles
for i, article in enumerate(articles_subset):
    article['sentiment_analysis'] = sentiment_results[i]

# Filter by sentiment
filtered_articles = [
    article for article in articles_subset 
    if article['sentiment_analysis']['sentiment'] in sentiment_filter
]

# -------------------
# Dashboard Metrics
# -------------------
st.subheader("📈 Real-time Sentiment Metrics")

sentiment_counts = Counter([article['sentiment_analysis']['sentiment'] for article in articles_subset])
total_articles = len(articles_subset)

col1, col2, col3, col4 = st.columns(4)

with col1:
    positive_pct = (sentiment_counts['positive'] / total_articles) * 100 if total_articles > 0 else 0
    st.metric(
        "🟢 Positive Sentiment", 
        f"{sentiment_counts['positive']}", 
        delta=f"{positive_pct:.1f}%"
    )

with col2:
    negative_pct = (sentiment_counts['negative'] / total_articles) * 100 if total_articles > 0 else 0
    st.metric(
        "🔴 Negative Sentiment", 
        f"{sentiment_counts['negative']}", 
        delta=f"{negative_pct:.1f}%"
    )

with col3:
    neutral_pct = (sentiment_counts['neutral'] / total_articles) * 100 if total_articles > 0 else 0
    st.metric(
        "⚪ Neutral Sentiment", 
        f"{sentiment_counts['neutral']}", 
        delta=f"{neutral_pct:.1f}%"
    )

with col4:
    avg_confidence = np.mean([article['sentiment_analysis']['confidence'] for article in articles_subset])
    st.metric(
        "🎯 Avg Confidence", 
        f"{avg_confidence:.3f}",
        delta="Model confidence"
    )

# -------------------
# Visualizations
# -------------------
st.subheader("📊 Sentiment Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    # Sentiment distribution pie chart
    fig_pie = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Overall Sentiment Distribution",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#4682B4'
        },
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Confidence distribution
    confidences = [article['sentiment_analysis']['confidence'] for article in articles_subset]
    sentiments = [article['sentiment_analysis']['sentiment'] for article in articles_subset]
    
    fig_conf = px.box(
        x=sentiments,
        y=confidences,
        title="Sentiment Confidence Distribution",
        color=sentiments,
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#4682B4'
        }
    )
    fig_conf.update_layout(xaxis_title="Sentiment", yaxis_title="Confidence Score")
    st.plotly_chart(fig_conf, use_container_width=True)

# -------------------
# Advanced Analytics
# -------------------
if show_detailed_analysis:
    st.subheader("🔬 Advanced Sentiment Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "☁️ Word Cloud", "📊 Source Analysis"])
    
    with tab1:
        # Sentiment timeline (simulated based on article order)
        timeline_data = []
        for i, article in enumerate(articles_subset):
            timeline_data.append({
                'index': i,
                'sentiment': article['sentiment_analysis']['sentiment'],
                'confidence': article['sentiment_analysis']['confidence'],
                'title': article.get('title', '')[:50] + '...'
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig_timeline = px.scatter(
            df_timeline,
            x='index',
            y='confidence',
            color='sentiment',
            title="Sentiment Confidence Over Articles",
            hover_data=['title'],
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#4682B4'
            }
        )
        fig_timeline.update_layout(xaxis_title="Article Index", yaxis_title="Confidence Score")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        # Word cloud
        all_text = ' '.join([f"{article.get('title', '')} {article.get('description', '')}" for article in filtered_articles])
        
        if all_text.strip():
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis'
                ).generate(all_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.info("Word cloud generation requires additional setup. Showing keyword analysis instead.")
                
                # Extract keywords
                keywords = extract_keywords(all_text, 20)
                if keywords:
                    df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                    fig_keywords = px.bar(
                        df_keywords.head(10),
                        x='Frequency',
                        y='Keyword',
                        title="Top Keywords in Financial News",
                        orientation='h'
                    )
                    fig_keywords.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_keywords, use_container_width=True)
    
    with tab3:
        # Source analysis
        source_sentiment = {}
        for article in articles_subset:
            source = article['source'].get('name', 'Unknown')
            sentiment = article['sentiment_analysis']['sentiment']
            
            if source not in source_sentiment:
                source_sentiment[source] = {'positive': 0, 'negative': 0, 'neutral': 0}
            source_sentiment[source][sentiment] += 1
        
        # Convert to DataFrame for visualization
        source_data = []
        for source, sentiments in source_sentiment.items():
            for sentiment, count in sentiments.items():
                source_data.append({'Source': source, 'Sentiment': sentiment, 'Count': count})
        
        if source_data:
            df_sources = pd.DataFrame(source_data)
            fig_sources = px.bar(
                df_sources,
                x='Source',
                y='Count',
                color='Sentiment',
                title="Sentiment Distribution by News Source",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'negative': '#DC143C',
                    'neutral': '#4682B4'
                }
            )
            fig_sources.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_sources, use_container_width=True)

# -------------------
# News Articles Display
# -------------------
st.header("📰 Latest Financial News with Sentiment Analysis")

if filtered_articles:
    st.info(f"Showing {len(filtered_articles)} articles matching your filters")
    
    for i, article in enumerate(filtered_articles, 1):
        sentiment_data = article['sentiment_analysis']
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        
        # Determine sentiment styling
        if sentiment == 'positive':
            sentiment_class = "sentiment-positive"
            sentiment_emoji = "🟢"
        elif sentiment == 'negative':
            sentiment_class = "sentiment-negative"
            sentiment_emoji = "🔴"
        else:
            sentiment_class = "sentiment-neutral"
            sentiment_emoji = "⚪"
        
        # Article container
        with st.container():
            st.markdown('<div class="news-card">', unsafe_allow_html=True)
            
            # Article header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {i}. {article.get('title', 'No Title')}")
                
                # Source and timestamp
                source_name = article['source'].get('name', 'Unknown Source')
                published_at = article.get('publishedAt', '')
                if published_at:
                    try:
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_date = published_at
                else:
                    formatted_date = 'Unknown Date'
                
                st.markdown(f'<span class="source-badge">{source_name}</span> | 📅 {formatted_date}', 
                          unsafe_allow_html=True)
            
            with col2:
                # Sentiment badge
                st.markdown(f"""
                <div style="text-align: right;">
                    <span class="{sentiment_class}">{sentiment_emoji} {sentiment.title()}</span><br>
                    <small>Confidence: {confidence:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Article description
            description = article.get('description', 'No description available.')
            st.write(description)
            
            # Detailed sentiment scores
            if show_detailed_analysis:
                scores = sentiment_data['scores']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Positive", f"{scores['positive']:.3f}")
                with col2:
                    st.metric("Neutral", f"{scores['neutral']:.3f}")
                with col3:
                    st.metric("Negative", f"{scores['negative']:.3f}")
            
            # Article image
            if show_images and article.get("urlToImage"):
                try:
                    st.image(article["urlToImage"], use_container_width=True, caption="Article Image")
                except:
                    st.info("Image could not be loaded")
            
            # Read more link
            article_url = article.get('url', '#')
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <a href="{article_url}" target="_blank" 
                   style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 0.5rem 1rem; border-radius: 5px; 
                          text-decoration: none; font-weight: bold;">
                    📖 Read Full Article
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

else:
    st.warning("No articles match your current filters. Please adjust your selection.")

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px; margin-top: 2rem;'>
    🚀 Financial News Sentiment Intelligence Dashboard | 
    Powered by FinBERT & MinIO | 
    Last Analysis: {timestamp}
</div>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)