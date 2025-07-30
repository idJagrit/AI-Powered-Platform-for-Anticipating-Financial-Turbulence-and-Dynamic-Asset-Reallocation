# AI-Powered-Platform-for-Anticipating-Financial-Turbulence-and-Dynamic-Asset-Reallocation
## Project Objective: 
To build a scalable, AI-driven platform that anticipates financial turbulence using multi-source data and provides real-time, personalized asset reallocation strategies across various asset classes. This enables proactive risk management tailored to market conditions.


Features
- **ETL Pipeline**: Aggregates structured (market, macroeconomic, CDS) and unstructured (news, social media) financial data.
- **Sentiment Analysis**: NLP models (BERT/FinBERT/GPT) analyze sentiment from Twitter, Reddit, and financial news sources.
- **Investor Behavior Clustering**: Unsupervised learning to identify and group investor risk profiles.
- **Reinforcement Learning Engine**: Bayesian optimization + Black-Litterman models for dynamic portfolio reallocation.
- **Market Forecasting**: Predicts 7/30/90-day trends using hybrid LSTM-CNN models with attention mechanisms.
- **Explainability**: SHAP values and counterfactual analysis for transparency in model decisions.
- **Interactive Dashboard**: Mobile-first UI showing risk scores, asset heatmaps, and scenario-based simulations (e.g., oil shocks, stagflation).
 
## Team Members:
- [Jagrit](https://github.com/idJagrit)
- [Saumya Vaidya](https://github.com/samthedoctor)
- [Vansh Raj Singh](https://github.com/vanshraj07)
- [Rashi Singh](https://github.com/RashiS26)

---
##  Key Technologies

- **Frontend**: HTML, CSS, JS, Node.js, Tailwind (optionally)
- **Backend**: Python (Flask/FastAPI), Express.js
- **Data & ML**: PyTorch, TensorFlow, Scikit-learn, HuggingFace, Pandas, NumPy
- **NLP Models**: FinBERT, GPT, BERT
- **Forecasting**: LSTM, CNN, Attention Mechanisms
- **ETL**: Apache Airflow DAGs
- **Orchestration**: Docker, Docker Compose

---
## Prerequisites

| Tool         | Version         | Notes                             |
|--------------|------------------|------------------------------------|
| Python       | 3.8+             | For backend and ML models          |
| Node.js      | 16+              | For frontend and dashboard         |
| Docker       | Latest           | For containerized orchestration    |
| Git          | Any              | To clone the repo                  |
| Airflow      | 2.x (optional)   | For DAG-based ETL pipelines        |
| AlphaVantage | API Key          | Or yfinance for stock data         |

---
## Frontend Pages
/login → Login screen with login.jpg/login.webp and signin.css

/signup → User registration UI styled with signup.css

/dashboard → Portfolio analysis page using dashboard.css

/analysis → Forecasting and risk overview from analysis.css

---


##  Key Variables & Configurations

| File                | Variable                 | Description                                                   |
|---------------------|--------------------------|---------------------------------------------------------------|
| `etl_pipeline.py`   | `api_key`                | Your API key for financial data (e.g., Alpha Vantage, yfinance) |
| `etl_pipeline.py`   | `data_sources`           | URLs or endpoints to fetch raw structured/unstructured data   |
| `etl_dag.py`        | `schedule_interval`      | Time interval for DAG execution (e.g., `@daily`, `@hourly`)   |
| `index.js`          | `PORT`                   | Port for running the web server (default: 3000 or 5000)       |
| `index.js`          | `APP_SECRET`             | JWT secret or session key for auth                            |
| `.env`              | `DB_URI`, `API_KEY`, etc | Secure environment config variables                           |
| `docker-compose.yml`| `services`               | Specifies backend/frontend service containers                 |

---

### Core Functions
<table>
  <tr>
    <td style ="width: 50%; vertical-align: top; padding-right: 20px;">
      <h4>1. <code>`fetch_market_data(symbol, start, end)`</code></h4>
      <ul>
        <li><strong>Purpose</strong>:<br>Fetch historical market data for a specific asset (stock, ETF, index).</li>
        <li><strong>Parameters</strong>:
          <ul>
            <li><code>title </code>: `symbol` (str): Stock ticker symbol (e.g., 'AAPL')</li>
            <li><code>dailyPrice</code>:`start` (str): Start date (format: 'YYYY-MM-DD')</li>
            <li><code>image </code>: `end` (str): End date (format: 'YYYY-MM-DD')</li>
          </ul>
        </li>
        <li><strong>Logic</strong>:
          <ul>
            <li>  Uses `yfinance` to download historical data (Open, High, Low, Close, Volume) between the specified dates and returns a DataFrame.</li>
            
 </ul>
        </li>
      </ul>
    </td>
</table>
