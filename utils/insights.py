"""Rule-based business insight generation for DataPilot Analytics.

No LLM call required for this module — insights are derived directly
from the dataframe. This keeps the core dashboard fast and free to run;
the LLM is reserved for the natural-language chat feature only.
"""
import pandas as pd
from utils.eda import top_category_column, numeric_target_column


def generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    target_col = numeric_target_column(df)
    cat_col = top_category_column(df)

    if target_col and cat_col:
        grouped = df.groupby(cat_col)[target_col].sum().sort_values(ascending=False)
        if len(grouped) > 0:
            top_segment = grouped.index[0]
            top_share = round(grouped.iloc[0] / grouped.sum() * 100, 1)
            insights.append(
                f"{top_segment} contributes the most to {target_col}, at {top_share}% of the total."
            )
        if len(grouped) > 1:
            bottom_segment = grouped.index[-1]
            insights.append(
                f"{bottom_segment} has the lowest {target_col} — consider investigating why."
            )

    if target_col:
        missing_pct = round(df[target_col].isna().mean() * 100, 1)
        if missing_pct > 5:
            insights.append(
                f"{target_col} has {missing_pct}% missing values, which may affect prediction reliability."
            )

    # Simple trend detection if a date-like column exists
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols and target_col:
        try:
            date_col = date_cols[0]
            temp = df[[date_col, target_col]].dropna()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna().sort_values(date_col)
            if len(temp) > 4:
                first_half = temp[target_col].iloc[: len(temp) // 2].mean()
                second_half = temp[target_col].iloc[len(temp) // 2:].mean()
                if first_half > 0:
                    change = round((second_half - first_half) / first_half * 100, 1)
                    direction = "increased" if change > 0 else "decreased"
                    insights.append(f"{target_col} has {direction} by {abs(change)}% over the observed period.")
        except Exception:
            pass

    if not insights:
        insights.append("Upload a dataset with a numeric and categorical column to generate insights.")

    return insights
