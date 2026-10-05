"""Exploratory data analysis utilities for DataPilot Analytics."""
import pandas as pd
import numpy as np


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def top_category_column(df: pd.DataFrame):
    """Best-guess categorical column for region/category-style breakdowns."""
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) == 0:
        return None
    # Prefer columns with low-moderate cardinality (looks like a category, not an ID)
    candidates = [c for c in cat_cols if 2 <= df[c].nunique() <= 20]
    return candidates[0] if candidates else cat_cols[0]


def numeric_target_column(df: pd.DataFrame):
    """Best-guess numeric column for revenue/sales-style KPIs."""
    num_cols = df.select_dtypes(include=np.number).columns
    priority_keywords = ["sales", "revenue", "profit", "amount", "price", "total"]
    for keyword in priority_keywords:
        for col in num_cols:
            if keyword in col.lower():
                return col
    return num_cols[0] if len(num_cols) else None
