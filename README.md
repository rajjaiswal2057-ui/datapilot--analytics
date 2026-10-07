# ðŸ“Š DataPilot Analytics

**Transform raw data into actionable business intelligence.**

DataPilot Analytics is an AI-powered web app that takes any CSV or Excel file and automatically cleans it, explores it, visualizes it, trains machine learning models on it, and lets you ask questions about it in plain English, with no coding required.

ðŸ”— **Live app:** [Open DataPilot Analytics](https://datapilot-analyticsai.streamlit.app)

<!-- Add a screenshot: put an image in a "screenshots" folder and uncomment the line below -->
<!-- ![DataPilot Analytics](screenshots/app.png) -->

---

## What it does

```
Upload â†’ Clean â†’ Analyze â†’ Visualize â†’ Predict â†’ Chat
```

| Tab | What you get |
|---|---|
| ðŸ§¹ **Clean data** | One-click cleaning of missing values, duplicates and outliers, with a before/after comparison |
| ðŸ“ˆ **Dashboard** | KPIs, distribution charts, category breakdowns, correlation heatmap and automatic business insights |
| ðŸ”® **Predict** | Train Regression or Classification models, compare them, and see which features matter most |
| ðŸ’¬ **Chat with data** | Ask questions about your dataset in natural language, powered by Google Gemini |

## Features

- Upload CSV or Excel files (`.csv`, `.xlsx`)
- Automatic data profiling and cleaning
  - Median imputation for numeric columns, mode for categorical columns
  - Duplicate removal
  - Outlier detection using the IQR method
- Interactive Plotly charts
- Rule-based business insights (fast, free, no API needed)
- Machine learning with scikit-learn
  - Regression: Linear Regression vs Random Forest (RÂ², MAE)
  - Classification: Logistic Regression vs Random Forest (Accuracy, Precision, Recall, F1)
  - Feature importance chart and best-model selection
- Natural-language chat grounded in your dataset's actual statistics

## How to use it

1. Open the app and upload a CSV or Excel file from the sidebar
   (or try the included `sample_data/retail_sales_sample.csv`).
2. Go to **Clean data** and click the cleaning button.
3. Explore the **Dashboard** for charts and insights.
4. In **Predict**, pick a task (regression or classification), choose a target column and click **Train models**.
5. Use **Chat with data** to ask questions like *"Which category has the highest profit?"*

## Tech stack

Python Â· Streamlit Â· Pandas Â· NumPy Â· Scikit-learn Â· Plotly Â· Google Gemini API

## Run it locally

```bash
# 1. Clone the repository
git clone https://github.com/rajjaiswal2057-ui/datapilot--analytics.git
cd datapilot--analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Enable "Chat with data"
#    Create .streamlit/secrets.toml and add your free Gemini API key:
#    GEMINI_API_KEY = "your-key-here"

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey).
> Never commit your key. `.streamlit/secrets.toml` is already in `.gitignore`.
> All other tabs work without any API key.

## Project structure

```
datapilot--analytics/
â”œâ”€â”€ app.py                    # Main Streamlit app
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ sample_data/
â”‚   â””â”€â”€ retail_sales_sample.csv
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ data_cleaner.py       # Profiling and cleaning
â”‚   â”œâ”€â”€ eda.py                # Summary stats, correlations, column detection
â”‚   â”œâ”€â”€ ml_models.py          # Regression and classification training
â”‚   â”œâ”€â”€ insights.py           # Rule-based business insights
â”‚   â”œâ”€â”€ chat_engine.py        # Gemini-powered chat over the dataset
â”‚   â””â”€â”€ dashboard_helpers.py  # Chart and KPI helpers
â””â”€â”€ .streamlit/
    â””â”€â”€ config.toml           # Theme
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own copy, fork this repo, point Streamlit Cloud at `app.py`, and add `GEMINI_API_KEY` under **Settings â†’ Secrets**.

## Roadmap

- SQL query workspace on uploaded data
- PDF / Excel report export
- Time-series forecasting
- Anomaly detection
- Light / dark mode toggle
- Analysis history

## Author

**Raj Jaiswal**
GitHub: [@rajjaiswal2057-ui](https://github.com/rajjaiswal2057-ui)

If you find this project useful, please give it a â­
