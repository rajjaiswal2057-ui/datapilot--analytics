"""Data cleaning utilities for DataPilot Analytics."""
import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a summary profile used for the KPI cards and before/after view."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_count = 0
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count += int(((df[col] < lower) | (df[col] > upper)).sum())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) else 0,
        "duplicates": int(df.duplicated().sum()),
        "outliers": outlier_count,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One-click cleaning: drop duplicates, fix dtypes, impute missing values."""
    cleaned = df.copy()

    # Drop exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Impute missing values: median for numeric, mode for categorical
    for col in cleaned.columns:
        if cleaned[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        else:
            mode = cleaned[col].mode()
            fill_value = mode[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    # Attempt to parse obvious date columns
    for col in cleaned.columns:
        if cleaned[col].dtype == object and "date" in col.lower():
            try:
                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
            except Exception:
                pass

    return cleaned
