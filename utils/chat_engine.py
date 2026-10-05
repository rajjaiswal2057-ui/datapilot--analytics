"""Natural-language chat over the uploaded dataset, using the Anthropic API.

The LLM is never shown the raw dataframe — only a compact statistical summary
built from it. This keeps the answer grounded in real numbers and avoids
sending large amounts of data to the API.
"""
import pandas as pd


def build_data_context(df: pd.DataFrame) -> str:
    """Compact text summary of the dataframe for the LLM prompt."""
    lines = [f"Dataset has {len(df)} rows and {len(df.columns)} columns.", ""]
    lines.append("Columns: " + ", ".join(df.columns))
    lines.append("")

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols[:8]:
        lines.append(
            f"{col}: sum={df[col].sum():.2f}, mean={df[col].mean():.2f}, "
            f"min={df[col].min():.2f}, max={df[col].max():.2f}"
        )


    cat_cols = df.select_dtypes(include="object").columns
    key_numeric = [c for c in numeric_cols if c.lower() in ("sales", "profit", "quantity", "discount")]

    for col in cat_cols:
        if df[col].nunique() <= 20:
            top = df[col].value_counts().head(5)
            lines.append(f"\n{col} breakdown (row counts):")
            for val, count in top.items():
                lines.append(f"  {val}: {count} rows")

            for num_col in key_numeric[:2]:
                grouped = df.groupby(col)[num_col].sum().sort_values(ascending=False)
                lines.append(f"\n{num_col} by {col}:")
                for val, total in grouped.items():
                    lines.append(f"  {val}: {total:.2f}")

    return "\n".join(lines)


def ask_gemini(model, question: str) -> str:
    """model is a configured google.generativeai.GenerativeModel instance already
    primed with the system instruction (see app.py)."""
    response = model.generate_content(question)
    return response.text
