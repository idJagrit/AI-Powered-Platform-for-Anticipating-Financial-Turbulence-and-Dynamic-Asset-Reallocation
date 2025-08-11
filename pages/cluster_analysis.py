import streamlit as st
import pandas as pd
import pickle
import boto3
from botocore.client import Config
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# -------------------
# Page Configuration
# -------------------
st.set_page_config(
    page_title="Advanced Risk Behavior Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------
# MinIO / S3 Settings (Unchanged)
# -------------------
S3_ENDPOINT = "http://localhost:9000"
AWS_ACCESS_KEY_ID = "minio"
AWS_SECRET_ACCESS_KEY = "minio123"

DATA_BUCKET = "historical-data"
DATA_FILE = "historical_data.csv"

MODEL_BUCKET = "models"
MODEL_FILE = "clustering_model.pkl"

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
# Load Data Functions (Enhanced)
# -------------------
@st.cache_data
def load_historical_data():
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=DATA_FILE)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    
    # Convert Date column if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    return df

@st.cache_resource
def load_enhanced_model():
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_FILE)
    loaded_obj = pickle.loads(obj["Body"].read())
    
    return loaded_obj

# -------------------
# Feature Engineering Function
# -------------------
@st.cache_data
def engineer_features(df):
    """Apply the same feature engineering as in training"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_df = df[numeric_cols].copy()
    
    # 1. Volatility features
    for col in ['SP500', 'VIXCLS']:
        if col in feature_df.columns:
            feature_df[f'{col}_volatility_7d'] = feature_df[col].rolling(7).std()
            feature_df[f'{col}_volatility_30d'] = feature_df[col].rolling(30).std()
    
    # 2. Momentum features
    price_cols = [col for col in numeric_cols if 'Close_' in col or col == 'SP500']
    for col in price_cols:
        if col in feature_df.columns:
            feature_df[f'{col}_return_1d'] = feature_df[col].pct_change(1)
            feature_df[f'{col}_return_7d'] = feature_df[col].pct_change(7)
            feature_df[f'{col}_return_30d'] = feature_df[col].pct_change(30)
            feature_df[f'{col}_ma_7d'] = feature_df[col].rolling(7).mean()
            feature_df[f'{col}_ma_30d'] = feature_df[col].rolling(30).mean()
            feature_df[f'{col}_ma_ratio'] = feature_df[col] / feature_df[f'{col}_ma_30d']
    
    # 3. Volume features
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
    
    return feature_df.dropna()

# -------------------
# Main App
# -------------------
st.title("🚀 Advanced Investor Risk Behavior Analysis")
st.markdown("### Advanced clustering analysis with machine learning insights")

# Sidebar for controls
st.sidebar.header("📊 Analysis Controls")

# Load data and model
try:
    with st.spinner("Loading historical data..."):
        df_raw = load_historical_data()
        st.sidebar.success(f"✅ Data loaded: {df_raw.shape[0]:,} records")
        
    with st.spinner("Loading enhanced model..."):
        model_package = load_enhanced_model()
        st.sidebar.success(f"✅ Model loaded: {model_package['best_model_name'].upper()}")
        
except Exception as e:
    st.error(f"Error loading data/model: {e}")
    st.stop()

# -------------------
# Model Information Panel
# -------------------
st.header("🤖 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

metadata = model_package.get('metadata', {})
with col1:
    st.metric(
        "Model Type", 
        model_package['best_model_name'].upper(),
        delta=f"Silhouette: {metadata.get('silhouette_score', 0):.3f}"
    )
with col2:
    st.metric(
        "Clusters Found", 
        model_package.get('optimal_clusters', 'N/A'),
        delta=f"Samples: {metadata.get('training_samples', 0):,}"
    )
with col3:
    st.metric(
        "Feature Engineering", 
        f"{metadata.get('n_features_original', 0)} → {metadata.get('n_features_engineered', 0)}",
        delta=f"PCA: {metadata.get('n_features_pca', 0)}"
    )
with col4:
    st.metric(
        "Variance Explained", 
        f"{metadata.get('variance_explained', 0):.1%}",
        delta="by PCA"
    )

# -------------------
# Data Processing and Prediction
# -------------------
with st.spinner("Processing features and generating predictions..."):
    # Engineer features
    df_features = engineer_features(df_raw)
    
    # Get recent data for display (last 100 records)
    recent_data = df_features.tail(100).copy()
    
    # Make predictions using the loaded model components
    model = model_package['model']
    scaler = model_package['scaler']
    pca = model_package['pca']
    risk_mapping = model_package['risk_mapping']
    feature_columns = model_package['feature_columns']
    
    # Ensure we have the right columns
    prediction_data = recent_data[feature_columns]
    
    # Apply same preprocessing pipeline
    X_scaled = scaler.transform(prediction_data)
    X_pca = pca.transform(X_scaled)
    
    # Predict clusters
    clusters = model.predict(X_pca)
    
    # Map to risk levels
    recent_data['Cluster'] = clusters
    recent_data['Risk_Level'] = recent_data['Cluster'].map(risk_mapping).fillna("Unknown")
    
    # Add outlier detection
    isolation_forest = model_package['isolation_forest']
    outliers = isolation_forest.predict(X_scaled)
    recent_data['Is_Outlier'] = outliers == -1

# -------------------
# Time Period Selector
# -------------------
if 'Date' in df_raw.columns:
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df_raw['Date'].max() - timedelta(days=90), df_raw['Date'].max()),
        min_value=df_raw['Date'].min(),
        max_value=df_raw['Date'].max()
    )
    
    if len(date_range) == 2:
        mask = (df_raw['Date'] >= pd.to_datetime(date_range[0])) & (df_raw['Date'] <= pd.to_datetime(date_range[1]))
        display_data = recent_data[recent_data.index.isin(df_raw[mask].index)]
    else:
        display_data = recent_data
else:
    display_data = recent_data

# -------------------
# Risk Distribution Dashboard
# -------------------
st.header("📈 Risk Distribution Analysis")

# Calculate risk distribution
risk_counts = display_data['Risk_Level'].value_counts().reset_index()
risk_counts.columns = ['Risk_Level', 'Count']
risk_counts['Percentage'] = (risk_counts['Count'] / len(display_data) * 100).round(2)

col1, col2 = st.columns(2)

with col1:
    # Enhanced pie chart
    fig_pie = px.pie(
        risk_counts, 
        names="Risk_Level", 
        values="Count",
        title="Current Risk Level Distribution",
        color_discrete_map={
            'Low Risk': '#2E8B57',
            'Medium Risk': '#FFD700', 
            'High Risk': '#DC143C'
        },
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Enhanced bar chart
    fig_bar = px.bar(
        risk_counts, 
        x="Risk_Level", 
        y="Count",
        text="Count",
        title="Risk Level Count Distribution",
        color="Risk_Level",
        color_discrete_map={
            'Low Risk': '#2E8B57',
            'Medium Risk': '#FFD700', 
            'High Risk': '#DC143C'
        }
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------
# Advanced Analytics
# -------------------
st.header("🔬 Advanced Risk Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "🎯 Cluster Analysis", "⚠️ Outlier Detection", "📋 Data Table"])

with tab1:
    if 'Date' in df_raw.columns:
        # Time series of risk levels
        time_series_data = df_raw.tail(200).copy()
        time_features = engineer_features(time_series_data)
        
        if len(time_features) > 0:
            # Predict for time series
            X_ts_scaled = scaler.transform(time_features[feature_columns])
            X_ts_pca = pca.transform(X_ts_scaled)
            ts_clusters = model.predict(X_ts_pca)
            
            time_features['Cluster'] = ts_clusters
            time_features['Risk_Level'] = time_features['Cluster'].map(risk_mapping).fillna("Unknown")
            time_features['Date'] = time_series_data['Date'].iloc[-len(time_features):].values
            
            # Risk level over time
            fig_timeline = px.scatter(
                time_features.reset_index(), 
                x='Date', 
                y='Risk_Level',
                color='Risk_Level',
                title="Risk Level Evolution Over Time",
                color_discrete_map={
                    'Low Risk': '#2E8B57',
                    'Medium Risk': '#FFD700', 
                    'High Risk': '#DC143C'
                }
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Market indicators vs Risk
            if 'VIXCLS' in time_features.columns and 'SP500' in time_features.columns:
                fig_indicators = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('VIX vs Risk Level', 'S&P 500 vs Risk Level', 
                                   'VIX Over Time', 'S&P 500 Over Time'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # VIX vs Risk scatter
                for risk_level in time_features['Risk_Level'].unique():
                    mask = time_features['Risk_Level'] == risk_level
                    fig_indicators.add_trace(
                        go.Scatter(
                            x=time_features[mask]['VIXCLS'],
                            y=time_features[mask]['Risk_Level'],
                            mode='markers',
                            name=f'{risk_level} (VIX)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # SP500 vs Risk scatter
                for risk_level in time_features['Risk_Level'].unique():
                    mask = time_features['Risk_Level'] == risk_level
                    fig_indicators.add_trace(
                        go.Scatter(
                            x=time_features[mask]['SP500'],
                            y=time_features[mask]['Risk_Level'],
                            mode='markers',
                            name=f'{risk_level} (SP500)',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                # VIX timeline
                fig_indicators.add_trace(
                    go.Scatter(
                        x=time_features['Date'],
                        y=time_features['VIXCLS'],
                        mode='lines',
                        name='VIX',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                # SP500 timeline  
                fig_indicators.add_trace(
                    go.Scatter(
                        x=time_features['Date'],
                        y=time_features['SP500'],
                        mode='lines',
                        name='S&P 500',
                        line=dict(color='blue')
                    ),
                    row=2, col=2
                )
                
                fig_indicators.update_layout(height=600, title_text="Market Indicators Analysis")
                st.plotly_chart(fig_indicators, use_container_width=True)

with tab2:
    st.subheader("🎯 Cluster Characteristics")
    
    # Cluster statistics
    cluster_stats = display_data.groupby(['Cluster', 'Risk_Level']).agg({
        'VIXCLS': ['mean', 'std'] if 'VIXCLS' in display_data.columns else 'count',
        'SP500': ['mean', 'std'] if 'SP500' in display_data.columns else 'count',
    }).round(3)
    
    st.write("**Cluster Statistics:**")
    st.dataframe(cluster_stats)
    
    # Feature importance for clustering (using PCA components)
    if hasattr(pca, 'components_'):
        feature_importance = pd.DataFrame(
            pca.components_[:3].T,  # First 3 components
            columns=[f'PC{i+1}' for i in range(3)],
            index=feature_columns
        )
        
        fig_importance = px.bar(
            feature_importance.abs().sum(axis=1).sort_values(ascending=True).tail(10).reset_index(),
            x='index',
            y=0,
            orientation='h',
            title="Top 10 Most Important Features for Clustering"
        )
        fig_importance.update_layout(xaxis_title="Feature Importance", yaxis_title="Features")
        st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.subheader("⚠️ Outlier Detection Analysis")
    
    outlier_summary = display_data['Is_Outlier'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Points", outlier_summary.get(False, 0))
        st.metric("Outliers Detected", outlier_summary.get(True, 0))
    
    with col2:
        if len(display_data) > 0:
            outlier_pct = (outlier_summary.get(True, 0) / len(display_data)) * 100
            st.metric("Outlier Percentage", f"{outlier_pct:.2f}%")
        
        # Show outliers by risk level
        outlier_risk = display_data[display_data['Is_Outlier'] == True]['Risk_Level'].value_counts()
        if not outlier_risk.empty:
            fig_outlier_risk = px.bar(
                x=outlier_risk.index,
                y=outlier_risk.values,
                title="Outliers by Risk Level",
                labels={'x': 'Risk Level', 'y': 'Count'},
                color=outlier_risk.index,
                color_discrete_map={
                    'Low Risk': '#2E8B57',
                    'Medium Risk': '#FFD700', 
                    'High Risk': '#DC143C'
                }
            )
            st.plotly_chart(fig_outlier_risk, use_container_width=True)
    
    # Outlier details table
    if display_data['Is_Outlier'].sum() > 0:
        st.write("**Recent Outlier Records:**")
        outlier_data = display_data[display_data['Is_Outlier'] == True][
            ['Risk_Level', 'Cluster'] + 
            ([col for col in ['Date', 'SP500', 'VIXCLS'] if col in display_data.columns])
        ].tail(10)
        st.dataframe(outlier_data, use_container_width=True)

with tab4:
    st.subheader("📋 Recent Data Analysis")
    
    # Data filtering options
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=display_data['Risk_Level'].unique(),
            default=display_data['Risk_Level'].unique()
        )
    
    with col2:
        show_outliers = st.checkbox("Include Outliers Only", False)
    
    # Apply filters
    filtered_data = display_data[display_data['Risk_Level'].isin(risk_filter)]
    if show_outliers:
        filtered_data = filtered_data[filtered_data['Is_Outlier'] == True]
    
    # Display summary statistics
    st.write(f"**Showing {len(filtered_data)} records**")
    
    # Key metrics table
    summary_cols = []
    if 'SP500' in filtered_data.columns:
        summary_cols.append('SP500')
    if 'VIXCLS' in filtered_data.columns:
        summary_cols.append('VIXCLS')
    if 'T10Y2Y' in filtered_data.columns:
        summary_cols.append('T10Y2Y')
    
    # Add some stock prices
    stock_cols = [col for col in filtered_data.columns if 'Close_' in col][:5]
    summary_cols.extend(stock_cols)
    
    if summary_cols:
        summary_stats = filtered_data[summary_cols + ['Risk_Level']].groupby('Risk_Level').agg(['mean', 'std', 'min', 'max']).round(3)
        st.write("**Summary Statistics by Risk Level:**")
        st.dataframe(summary_stats, use_container_width=True)
    
    # Recent data table with key columns
    display_cols = ['Risk_Level', 'Cluster', 'Is_Outlier']
    if 'Date' in df_raw.columns:
        # Add date from original data
        date_mapping = dict(zip(df_raw.index, df_raw['Date']))
        filtered_data_with_date = filtered_data.copy()
        filtered_data_with_date['Date'] = filtered_data_with_date.index.map(date_mapping)
        display_cols = ['Date'] + display_cols
        display_data_final = filtered_data_with_date
    else:
        display_data_final = filtered_data
    
    # Add key financial indicators
    key_indicators = ['SP500', 'VIXCLS', 'T10Y2Y']
    for indicator in key_indicators:
        if indicator in display_data_final.columns:
            display_cols.append(indicator)
    
    # Add top stock prices
    stock_prices = [col for col in display_data_final.columns if 'Close_' in col][:3]
    display_cols.extend(stock_prices)
    
    st.write("**Recent Data Records:**")
    display_table = display_data_final[display_cols].tail(50)
    st.dataframe(display_table, use_container_width=True)

# -------------------
# Advanced Insights Section
# -------------------
st.header("🧠 AI-Powered Insights")

# Generate insights based on recent data
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Current Market Sentiment")
    
    # Risk level insights
    current_risk_dist = display_data['Risk_Level'].value_counts(normalize=True)
    dominant_risk = current_risk_dist.idxmax()
    dominant_pct = current_risk_dist.max() * 100
    
    if dominant_risk == "High Risk":
        sentiment_color = "🔴"
        sentiment_desc = "CAUTIOUS - High risk conditions detected"
    elif dominant_risk == "Medium Risk":
        sentiment_color = "🟡"
        sentiment_desc = "MODERATE - Mixed risk signals"
    else:
        sentiment_color = "🟢"
        sentiment_desc = "OPTIMISTIC - Low risk environment"
    
    st.markdown(f"""
    **{sentiment_color} Market Sentiment: {sentiment_desc}**
    
    - **Dominant Risk Level**: {dominant_risk} ({dominant_pct:.1f}% of recent data)
    - **Model Confidence**: {metadata.get('silhouette_score', 0):.3f} (Silhouette Score)
    - **Outliers Detected**: {display_data['Is_Outlier'].sum()} out of {len(display_data)} recent records
    """)

with col2:
    st.subheader("🎯 Key Risk Drivers")
    
    # Show key metrics that might be driving risk
    if 'VIXCLS' in display_data.columns:
        current_vix = display_data['VIXCLS'].iloc[-1]
        vix_trend = "↗️" if display_data['VIXCLS'].tail(5).diff().mean() > 0 else "↘️"
        
        vix_level = "High" if current_vix > 25 else "Moderate" if current_vix > 15 else "Low"
        st.write(f"**VIX Level**: {current_vix:.2f} ({vix_level}) {vix_trend}")
    
    if 'SP500' in display_data.columns:
        sp500_return = display_data['SP500'].pct_change().tail(5).mean() * 100
        sp500_trend = "↗️" if sp500_return > 0 else "↘️"
        st.write(f"**S&P 500 Trend**: {sp500_return:+.2f}% (5-day avg) {sp500_trend}")
    
    # Show cluster distribution
    cluster_info = display_data.groupby(['Cluster', 'Risk_Level']).size().reset_index(name='count')
    st.write("**Active Clusters**:")
    for _, row in cluster_info.iterrows():
        st.write(f"- Cluster {row['Cluster']}: {row['Risk_Level']} ({row['count']} records)")

# -------------------
# Model Performance Details
# -------------------
with st.expander("🔍 Advanced Model Details"):
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Clustering Performance:**")
        performance = model_package.get('model_performance', {})
        best_model = model_package.get('best_model_name', 'unknown')
        
        if best_model in performance:
            metrics = performance[best_model]
            st.write(f"- Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
            st.write(f"- Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}")
            st.write(f"- Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
            st.write(f"- Number of Clusters: {metrics.get('n_clusters', 'N/A')}")
    
    with col2:
        st.write("**Feature Engineering Summary:**")
        st.write(f"- Original Features: {metadata.get('n_features_original', 'N/A')}")
        st.write(f"- Engineered Features: {metadata.get('n_features_engineered', 'N/A')}")
        st.write(f"- PCA Components: {metadata.get('n_features_pca', 'N/A')}")
        st.write(f"- Variance Explained: {metadata.get('variance_explained', 0):.1%}")
        st.write(f"- Training Samples: {metadata.get('training_samples', 'N/A'):,}")
    
    st.write("**Risk Level Mapping:**")
    risk_mapping_df = pd.DataFrame(list(risk_mapping.items()), columns=['Cluster', 'Risk_Level'])
    st.dataframe(risk_mapping_df, use_container_width=True)

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    🚀 Advanced Investor Risk Behavior Analysis | 
    Powered by Machine Learning & MinIO | 
    Model: {model_type} | 
    Last Updated: {timestamp}
</div>
""".format(
    model_type=model_package.get('best_model_name', 'Unknown').upper(),
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
), unsafe_allow_html=True)