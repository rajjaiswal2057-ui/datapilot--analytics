# 📊 DataPilot Analytics — Intelligent Data Analytics & AI Insights Platform

**Upload data. Discover insights. Predict the future.**

An AI-powered analytics platform that takes a raw CSV/Excel file and automatically performs data cleaning, exploratory analysis, visualization, machine learning, and business insight generation — with a natural-language chat interface layered on top.

## Why this project

Built to demonstrate the complete data workflow used by Data Analysts, Data Scientists, and AI/ML engineers:

```
Upload → Clean → Analyze → Visualize → Predict → Chat → Report
```

## Features (Version 1)

- ✅ CSV / Excel upload
- ✅ One-click automatic data cleaning (missing values, duplicates, outliers)
- ✅ Before/after cleaning comparison
- ✅ Exploratory dashboard: KPIs, distribution charts, category breakdowns, correlation heatmap
- ✅ Rule-based AI business insights (no API cost — pure logic on your data)
- ✅ Machine learning: regression or classification, with model comparison and feature importance
- 🔜 Natural-language "Chat with data" (requires an LLM API key — see below)

## Quick start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd datapilot-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload `sample_data/retail_sales_sample.csv` to try it immediately.

## Enabling "Chat with data"

The chat tab expects an LLM API key. Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Then wire up a call in `app.py`'s chat tab: pass the dataset's summary statistics + the user's question to the API, and display the response. Keep the prompt grounded strictly in your dataframe's actual numbers to avoid hallucinated answers.

## Project structure

```
datapilot-ai/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── sample_data/
│   └── retail_sales_sample.csv
├── utils/
│   ├── data_cleaner.py     # Cleaning + profiling
│   ├── eda.py               # Summary stats, correlation, column detection
│   ├── ml_models.py         # Regression & classification training
│   └── insights.py          # Rule-based business insight generation
└── .streamlit/
    └── config.toml          # Theme (premium blue palette)
```

## Tech stack

Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly

## Deployment

Push to GitHub, then deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud) — point it at `app.py`.

## Roadmap (Version 2)

- SQL query workspace on uploaded data
- PDF/Excel report export
- Time-series forecasting
- Anomaly detection
- Light/dark mode toggle
- Analysis history (SQLite persistence)

## Interview talking points

- **Cleaning:** median imputation for numeric, mode for categorical, IQR method for outlier detection
- **ML:** compares Linear Regression vs Random Forest (or Logistic Regression vs Random Forest for classification), reports R²/MAE or Accuracy/Precision/Recall/F1, shows feature importance
- **Insights:** rule-based logic keeps the dashboard fast and free — LLM is reserved only for the conversational layer
