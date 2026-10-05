"""Helper functions for the premium Business Intelligence dashboard."""
import pandas as pd
import numpy as np


def detect_profit_column(df: pd.DataFrame):
    for col in df.columns:
        if "profit" in col.lower() or "earning" in col.lower():
            return col
    return None


def detect_order_column(df: pd.DataFrame):
    for col in df.columns:
        if "order" in col.lower() and "id" in col.lower():
            return col
    return None


def calculate_profit_margin(df: pd.DataFrame, revenue_col: str, profit_col: str) -> float:
    total_revenue = df[revenue_col].sum()
    if not total_revenue:
        return 0.0
    return round(df[profit_col].sum() / total_revenue * 100, 1)


def pareto_data(df: pd.DataFrame, cat_col: str, value_col: str) -> pd.DataFrame:
    """Returns per-category totals with cumulative % of the value column, sorted descending."""
    grouped = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False).reset_index()
    total = grouped[value_col].sum()
    grouped["cumulative_pct"] = grouped[value_col].cumsum() / total * 100 if total else 0
    return grouped


def pareto_summary(pareto_df: pd.DataFrame, cat_col: str) -> str:
    """'Top X% of categories generate Y% of total revenue.'"""
    n_total = len(pareto_df)
    n_80 = (pareto_df["cumulative_pct"] <= 80).sum() + 1
    n_80 = min(n_80, n_total)
    pct_of_items = round(n_80 / n_total * 100)
    return f"Top {pct_of_items}% of {cat_col} values ({n_80} of {n_total}) generate ~80% of total."


def strongest_drivers(corr_df: pd.DataFrame, target: str, top_n: int = 3):
    """Strongest positive/negative correlations with the target column, excluding itself."""
    if target not in corr_df.columns:
        return [], []
    series = corr_df[target].drop(index=target, errors="ignore").sort_values(ascending=False)
    positive = series[series > 0].head(top_n)
    negative = series[series < 0].sort_values().head(top_n)
    return list(positive.items()), list(negative.items())


def data_quality_score(profile: dict) -> int:
    """0-100 score from the cleaning profile dict (rows, missing_pct, duplicates, outliers)."""
    score = 100
    score -= min(profile.get("missing_pct", 0) * 2, 30)
    dup_pct = (profile.get("duplicates", 0) / profile["rows"] * 100) if profile.get("rows") else 0
    score -= min(dup_pct * 2, 20)
    outlier_pct = (profile.get("outliers", 0) / profile["rows"] * 100) if profile.get("rows") else 0
    score -= min(outlier_pct * 0.3, 20)
    return max(round(score), 0)


def profitability_quadrant(df: pd.DataFrame, cat_col: str, revenue_col: str, profit_col: str) -> pd.DataFrame:
    """Classify each category into Star / Revenue Risk / Growth Opportunity / Low Priority
    based on revenue and profit margin relative to the dataset's own medians."""
    grouped = df.groupby(cat_col).agg(
        revenue=(revenue_col, "sum"), profit=(profit_col, "sum")
    )
    grouped["margin"] = (grouped["profit"] / grouped["revenue"] * 100).round(1)
    rev_median = grouped["revenue"].median()
    margin_median = grouped["margin"].median()

    def classify(row):
        high_rev = row["revenue"] >= rev_median
        high_margin = row["margin"] >= margin_median
        if high_rev and high_margin:
            return "Star"
        if high_rev and not high_margin:
            return "Revenue Risk"
        if not high_rev and high_margin:
            return "Growth Opportunity"
        return "Low Priority"

    grouped["quadrant"] = grouped.apply(classify, axis=1)
    return grouped.reset_index()


def detect_anomalies(df: pd.DataFrame, value_col: str, top_n: int = 5):
    """IQR-based anomaly detection on a numeric column. Returns (count, pct, top_rows)."""
    q1, q3 = df[value_col].quantile(0.25), df[value_col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, 0.0, df.head(0)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    anomalies = df[(df[value_col] < lower) | (df[value_col] > upper)]
    pct = round(len(anomalies) / len(df) * 100, 1) if len(df) else 0
    top_rows = anomalies.reindex(anomalies[value_col].abs().sort_values(ascending=False).index).head(top_n)
    return len(anomalies), pct, top_rows


def categorized_insights(df: pd.DataFrame, revenue_col: str, profit_col: str, cat_col: str) -> dict:
    """Rule-based Risks / Opportunities / Recommendations, grounded in the actual dataframe."""
    risks, opportunities, recommendations = [], [], []

    if revenue_col and profit_col and cat_col:
        grouped = df.groupby(cat_col).agg(
            revenue=(revenue_col, "sum"), profit=(profit_col, "sum")
        )
        grouped["margin"] = (grouped["profit"] / grouped["revenue"] * 100).round(1)
        grouped = grouped.sort_values("revenue", ascending=False)

        avg_margin = grouped["margin"].mean()
        for name, row in grouped.iterrows():
            if row["revenue"] > grouped["revenue"].median() and row["margin"] < avg_margin - 5:
                risks.append(
                    f"**{name}** generates high revenue ({row['revenue']:,.0f}) but below-average "
                    f"margin ({row['margin']}% vs {avg_margin:.1f}% average) — review pricing or costs."
                )
            if row["revenue"] < grouped["revenue"].median() and row["margin"] > avg_margin + 5:
                opportunities.append(
                    f"**{name}** has strong margin ({row['margin']}%) but lower revenue — "
                    "potential to grow with more marketing focus."
                )

        best = grouped["margin"].idxmax()
        recommendations.append(
            f"**Finding:** {best} has the highest profit margin ({grouped.loc[best, 'margin']}%). "
            f"**Recommendation:** Prioritize inventory and marketing spend toward {best}."
        )

    return {
        "risks": risks or ["No major risks detected from available data."],
        "opportunities": opportunities or ["No standout opportunities detected from available data."],
        "recommendations": recommendations or ["Not enough data to generate specific recommendations."],
    }