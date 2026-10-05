"""DataPilot Analytics — Intelligent Data Analytics Platform.

Run locally with:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils.data_cleaner import profile_dataset, clean_dataset
from utils.eda import summary_statistics, correlation_matrix, top_category_column, numeric_target_column
from utils.ml_models import train_regression, train_classification
from utils.insights import generate_insights
from utils.chat_engine import build_data_context, ask_gemini
from utils.dashboard_helpers import (
    detect_profit_column, detect_order_column, calculate_profit_margin,
    pareto_data, pareto_summary, strongest_drivers, data_quality_score,
    categorized_insights, profitability_quadrant, detect_anomalies,
)

st.set_page_config(page_title="DataPilot Analytics", page_icon="📊", layout="wide")

# ---------- Premium UI styling ----------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #64748B; font-size: 13px; }
[data-testid="stMetricValue"] { font-size: 22px; font-weight: 600; }
.dp-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    background: #DCFCE7;
    color: #166534;
    font-size: 12px;
    font-weight: 500;
}
.dp-insight {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #2563EB;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
}
.dp-risk { border-left-color: #DC2626 !important; }
.dp-opportunity { border-left-color: #16A34A !important; }
.dp-recommendation { border-left-color: #D97706 !important; }
h1, h2, h3 { font-weight: 600 !important; }

/* Landing page */
.hero-container {
    text-align:center;
    padding: 52px 24px 28px;
    background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 60%, #FFFFFF 100%);
    border-radius: 18px;
    margin-bottom: 28px;
    border: 1px solid #E2E8F0;
}
.hero-eyebrow {
    display:inline-block;
    font-size:11px;
    font-weight:700;
    letter-spacing:0.1em;
    color:#2563EB;
    background:#DBEAFE;
    padding:5px 16px;
    border-radius:20px;
    margin-bottom:18px;
    text-transform:uppercase;
}
.hero-title {
    font-size:34px;
    font-weight:800;
    color:#0F172A;
    margin-bottom:14px;
    line-height:1.22;
    letter-spacing:-0.02em;
}
.hero-subtitle {
    font-size:15.5px;
    color:#64748B;
    max-width:620px;
    margin:0 auto;
    line-height:1.65;
}
.stats-strip {
    display:flex;
    justify-content:center;
    gap:56px;
    margin-top:32px;
    padding-top:24px;
    border-top:1px solid #E2E8F0;
}
.stat-item { text-align:center; }
.stat-number { font-size:24px; font-weight:800; color:#2563EB; }
.stat-label { font-size:11.5px; color:#94A3B8; margin-top:3px; text-transform:uppercase; letter-spacing:0.03em; }
.section-label {
    text-align:center;
    font-size:11px;
    font-weight:700;
    letter-spacing:0.1em;
    color:#94A3B8;
    text-transform:uppercase;
    margin: 36px 0 4px;
}
.section-title {
    text-align:center;
    font-size:22px;
    font-weight:700;
    color:#0F172A;
    margin-bottom:28px;
}
.step-card { text-align:center; padding: 8px 12px; }
.step-number {
    width:36px; height:36px;
    border-radius:50%;
    background:#0F172A;
    color:#FFFFFF;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 14px;
    font-weight:700;
    font-size:14px;
}
.step-title { font-weight:600; font-size:14px; margin-bottom:6px; color:#0F172A; }
.step-desc { font-size:12.5px; color:#64748B; line-height:1.5; }
.feature-card {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:24px 18px;
    text-align:center;
    height:172px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.feature-icon-badge {
    width:50px; height:50px;
    border-radius:13px;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 14px;
    font-size:23px;
}
.trust-strip {
    text-align:center;
    margin-top:40px;
    padding-top:20px;
    border-top:1px solid #E2E8F0;
    font-size:12px;
    color:#94A3B8;
}
.trust-strip b { color:#64748B; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("📊 DataPilot Analytics")
    st.caption("Transform Raw Data into Actionable Business Intelligence")

# ---------- Sidebar: upload ----------
st.sidebar.header("Upload dataset")
uploaded_file = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx"])

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None


def read_uploaded_file(file):
    """Read CSV or Excel, falling back to common alternate encodings for CSV."""
    if not file.name.endswith(".csv"):
        return pd.read_excel(file)
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", encoding_errors="replace")


if uploaded_file is not None:
    raw_df = read_uploaded_file(uploaded_file)

    tab_clean, tab_dashboard, tab_ml, tab_chat = st.tabs(
        ["🧹 Clean data", "📈 Dashboard", "🤖 Predict", "💬 Chat with data"]
    )

    # ---------- Tab 1: Cleaning ----------
    with tab_clean:
        before = profile_dataset(raw_df)
        cleaned_df = clean_dataset(raw_df)
        after = profile_dataset(cleaned_df)
        st.session_state.cleaned_df = cleaned_df

        st.subheader("Before vs after cleaning")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", after["rows"])
        c2.metric("Missing values", after["missing_values"], delta=int(after["missing_values"] - before["missing_values"]))
        c3.metric("Duplicates removed", before["duplicates"] - after["duplicates"])
        c4.metric("Outliers flagged", after["outliers"])

        st.markdown('<span class="dp-badge">Data cleaned</span>', unsafe_allow_html=True)
        st.dataframe(cleaned_df.head(20), width='stretch')

    # ---------- Tab 2: Dashboard (upgraded BI style) ----------
    with tab_dashboard:
        base_df = st.session_state.cleaned_df
        target_col = numeric_target_column(base_df)
        cat_col = top_category_column(base_df)
        profit_col = detect_profit_column(base_df)
        order_col = detect_order_column(base_df)

        st.markdown("### Business Performance Intelligence")
        st.caption("Interactive analytics, performance monitoring & AI-powered business insights")
        h1, h2, h3 = st.columns(3)
        h1.caption(f"📄 Dataset: **{uploaded_file.name}**")
        h2.caption(f"📊 Records: **{len(base_df):,}**")
        h3.caption(f"🕒 Refreshed: **{datetime.now().strftime('%H:%M:%S')}**")

        st.markdown("##### Filters")
        filter_cols = [c for c in [cat_col] if c]
        second_cat = next(
            (c for c in base_df.select_dtypes(include="object").columns
             if c != cat_col and 2 <= base_df[c].nunique() <= 15),
            None,
        )
        if second_cat:
            filter_cols.append(second_cat)

        df = base_df.copy()
        filter_col_widgets = st.columns(len(filter_cols) + 1) if filter_cols else None
        active_filters = 0
        if filter_cols:
            for i, fcol in enumerate(filter_cols):
                options = sorted(base_df[fcol].dropna().unique().tolist())
                selected = filter_col_widgets[i].multiselect(fcol, options, default=[])
                if selected:
                    df = df[df[fcol].isin(selected)]
                    active_filters += 1
            with filter_col_widgets[-1]:
                st.caption(f"Active filters: {active_filters}")
                if st.button("Reset filters"):
                    st.rerun()

        if df.empty:
            st.warning("No rows match the current filters.")
            df = base_df

        st.markdown("##### Executive KPIs")
        k1, k2, k3, k4 = st.columns(4)
        if target_col:
            k1.metric(f"💰 {target_col}", f"{df[target_col].sum():,.0f}")
        if profit_col:
            k2.metric(f"💵 {profit_col}", f"{df[profit_col].sum():,.0f}")
        if profit_col and target_col:
            k3.metric("📈 Profit margin", f"{calculate_profit_margin(df, target_col, profit_col)}%")
        if order_col:
            n_orders = df[order_col].nunique()
            k4.metric("🧾 Orders", f"{n_orders:,}")
        elif target_col:
            k4.metric(f"📐 Avg {target_col}", f"{df[target_col].mean():,.2f}")

        col_chart1, col_chart2 = st.columns([1.4, 1])
        with col_chart1:
            if target_col:
                fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
                fig.update_traces(marker_color="#2563EB")
                st.plotly_chart(fig, width='stretch')
        with col_chart2:
            if cat_col and target_col:
                grouped = df.groupby(cat_col)[target_col].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(grouped, x=cat_col, y=target_col, title=f"{target_col} by {cat_col}")
                fig.update_traces(marker_color="#2563EB")
                st.plotly_chart(fig, width='stretch')

        sub_col = next(
            (c for c in df.select_dtypes(include="object").columns
             if c not in (cat_col,) and 2 <= df[c].nunique() <= 25),
            None,
        )
        if sub_col and target_col:
            st.markdown("##### Top & bottom performers")
            perf_tab1, perf_tab2 = st.tabs(["Top performers", "Bottom performers"])
            ranked = df.groupby(sub_col)[target_col].sum().sort_values(ascending=False)
            with perf_tab1:
                top10 = ranked.head(10).reset_index()
                fig = px.bar(top10, x=target_col, y=sub_col, orientation="h", title=f"Top 10 {sub_col} by {target_col}")
                fig.update_traces(marker_color="#16A34A")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, width='stretch')
            with perf_tab2:
                bottom10 = ranked.tail(10).reset_index()
                fig = px.bar(bottom10, x=target_col, y=sub_col, orientation="h", title=f"Bottom 10 {sub_col} by {target_col}")
                fig.update_traces(marker_color="#DC2626")
                fig.update_layout(yaxis={"categoryorder": "total descending"})
                st.plotly_chart(fig, width='stretch')

        if cat_col and target_col:
            st.markdown("##### Revenue concentration — Pareto analysis")
            pdata = pareto_data(df, cat_col, target_col)
            fig = px.bar(pdata, x=cat_col, y=target_col, title=f"{target_col} by {cat_col} with cumulative %")
            fig.update_traces(marker_color="#2563EB")
            fig.add_scatter(x=pdata[cat_col], y=pdata["cumulative_pct"], yaxis="y2", mode="lines+markers", name="Cumulative %", line=dict(color="#DC2626"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="Cumulative %"))
            st.plotly_chart(fig, width='stretch')
            st.caption(pareto_summary(pdata, cat_col))

        if cat_col and target_col and profit_col:
            st.markdown("##### Profitability quadrant")
            st.caption("Revenue vs. profit margin, benchmarked against this dataset's own medians.")
            quad_df = profitability_quadrant(df, cat_col, target_col, profit_col)
            color_map = {
                "Star": "#16A34A", "Revenue Risk": "#DC2626",
                "Growth Opportunity": "#2563EB", "Low Priority": "#94A3B8",
            }
            fig = px.scatter(
                quad_df, x="revenue", y="margin", size="revenue", color="quadrant",
                hover_name=cat_col, color_discrete_map=color_map,
                title=f"{cat_col}: Revenue vs Profit Margin",
                labels={"revenue": target_col, "margin": "Profit margin (%)"},
            )
            fig.add_hline(y=quad_df["margin"].median(), line_dash="dot", line_color="#94A3B8")
            fig.add_vline(x=quad_df["revenue"].median(), line_dash="dot", line_color="#94A3B8")
            st.plotly_chart(fig, width='stretch')
            for quadrant_name in ["Star", "Revenue Risk", "Growth Opportunity", "Low Priority"]:
                members = quad_df[quad_df["quadrant"] == quadrant_name][cat_col].tolist()
                if members:
                    st.caption(f"**{quadrant_name}:** {', '.join(str(m) for m in members)}")

        if target_col:
            st.markdown("##### Anomaly & outlier detection")
            n_anom, pct_anom, top_anom = detect_anomalies(df, target_col)
            a1, a2 = st.columns(2)
            a1.metric("Anomalies detected", n_anom)
            a2.metric("Anomaly rate", f"{pct_anom}%")
            if n_anom > 0:
                st.caption(f"Most significant outliers in {target_col} (flagged using the IQR method — not removed from the data):")
                display_cols = [c for c in [cat_col, target_col, profit_col] if c]
                st.dataframe(top_anom[display_cols] if display_cols else top_anom, width='stretch')

        corr = correlation_matrix(df)
        if not corr.empty:
            st.markdown("##### What drives performance?")
            st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Correlation heatmap", color_continuous_scale="Blues"), width='stretch')
            if target_col:
                pos, neg = strongest_drivers(corr, target_col)
                for name, val in pos:
                    st.caption(f"🟢 {name} is associated with higher {target_col} (correlation {val:.2f})")
                for name, val in neg:
                    st.caption(f"🔴 {name} is associated with lower {target_col} (correlation {val:.2f})")

        st.markdown("##### AI Business Analyst")
        cat_insights = categorized_insights(df, target_col, profit_col, cat_col)
        st.markdown("**🔴 Risks**")
        for r in cat_insights["risks"]:
            st.markdown(f'<div class="dp-insight dp-risk">{r}</div>', unsafe_allow_html=True)
        st.markdown("**🟢 Opportunities**")
        for o in cat_insights["opportunities"]:
            st.markdown(f'<div class="dp-insight dp-opportunity">{o}</div>', unsafe_allow_html=True)
        st.markdown("**💡 Recommendations**")
        for rec in cat_insights["recommendations"]:
            st.markdown(f'<div class="dp-insight dp-recommendation">{rec}</div>', unsafe_allow_html=True)

        st.markdown("##### Quick insights")
        for insight in generate_insights(df):
            st.markdown(f'<div class="dp-insight">💡 {insight}</div>', unsafe_allow_html=True)

        st.markdown("##### Data quality")
        profile = profile_dataset(df)
        score = data_quality_score(profile)
        q1, q2, q3 = st.columns(3)
        q1.metric("Data quality score", f"{score}/100")
        q2.metric("Missing %", f"{profile['missing_pct']}%")
        q3.metric("Duplicates", profile["duplicates"])

        st.download_button(
            "⬇ Download filtered data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
        )

    # ---------- Tab 3: ML ----------
    with tab_ml:
        df = st.session_state.cleaned_df
        task = st.radio("Prediction task", ["Regression (predict a number)", "Classification (predict a category)"], horizontal=True)
        target = st.selectbox("Target column to predict", df.columns)

        if st.button("Train models"):
            unique_ratio = df[target].nunique() / len(df)
            if task.startswith("Classification") and unique_ratio > 0.5:
                st.warning(
                    f"'{target}' has {df[target].nunique()} unique values across {len(df)} rows — "
                    "this looks like an ID column, not a category. Pick a column with repeated "
                    "values like Region, Category, or Ship Mode instead."
                )
            elif task.startswith("Regression") and not pd.api.types.is_numeric_dtype(df[target]):
                st.warning(
                    f"'{target}' is a text column, not a number — regression can't predict it. "
                    "Switch to 'Classification' above, or pick a numeric column like Sales or Profit."
                )
            else:
                with st.spinner("Training models..."):
                    if task.startswith("Regression"):
                        results, best_name, importance = train_regression(df, target)
                        cols = st.columns(len(results))
                        for col, (name, r) in zip(cols, results.items()):
                            with col:
                                st.metric(name, f"R² {r['r2']}", f"MAE {r['mae']}")
                        st.success(f"Best model: {best_name}")
                    else:
                        results, best_name, importance = train_classification(df, target)
                        cols = st.columns(len(results))
                        for col, (name, r) in zip(cols, results.items()):
                            with col:
                                st.metric(name, f"Acc {r['accuracy']}", f"F1 {r['f1']}")
                        st.success(f"Best model: {best_name}")

                    if importance is not None:
                        st.subheader("Feature importance")
                        st.plotly_chart(px.bar(importance.head(10), title="Top features"), width='stretch')

    # ---------- Tab 4: Chat ----------
    with tab_chat:
        df = st.session_state.cleaned_df
        api_key = st.secrets.get("GEMINI_API_KEY", "")

        if not api_key:
            st.info(
                "Connect a free Gemini API key in `.streamlit/secrets.toml` to enable "
                "natural-language Q&A over this dataset. See README for setup."
            )
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.write(msg)

            question = st.chat_input("Ask a question about your data")
            if question:
                st.session_state.chat_history.append(("user", question))
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            context = build_data_context(df)
                            system_prompt = (
                                "You are a data analyst assistant. Answer the user's question "
                                "using ONLY the dataset summary provided below. If the summary "
                                "doesn't contain enough information, say so instead of guessing.\n\n"
                                f"DATASET SUMMARY:\n{context}"
                            )
                            model = genai.GenerativeModel(
                                "gemini-flash-latest", system_instruction=system_prompt
                            )
                            answer = ask_gemini(model, question)
                        except Exception as e:
                            answer = f"Error calling the API: {e}"
                        st.write(answer)
                        st.session_state.chat_history.append(("assistant", answer))

else:
    st.markdown("""
    <div class="hero-container">
        <span class="hero-eyebrow">AI-Powered Business Intelligence</span>
        <div class="hero-title">Turn raw data into decisions —<br>automatically</div>
        <div class="hero-subtitle">
            Upload any CSV or Excel file and get instant cleaning, executive dashboards,
            ML predictions, and an AI analyst you can chat with in plain English.
        </div>
        <div class="stats-strip">
            <div class="stat-item"><div class="stat-number">4</div><div class="stat-label">Analysis Modules</div></div>
            <div class="stat-item"><div class="stat-number">2</div><div class="stat-label">ML Algorithms</div></div>
            <div class="stat-item"><div class="stat-number">&lt;30s</div><div class="stat-label">To First Insight</div></div>
            <div class="stat-item"><div class="stat-number">100%</div><div class="stat-label">Free To Run</div></div>
        </div>
    </div>

    <div class="section-label">Simple workflow</div>
    <div class="section-title">From raw file to business insight, in 3 steps</div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    steps = [
        ("1", "Upload your data", "Drop in any CSV or Excel file — no formatting or setup required"),
        ("2", "Auto-analysis runs", "Cleaning, KPIs, Pareto analysis, and ML models generate instantly"),
        ("3", "Ask & act", "Chat with your data in plain English and export what you need"),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">What you get</div><div class="section-title">Everything a data analyst needs, built in</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🧹", "#DBEAFE", "Auto-clean", "Missing values, duplicates, and outliers handled automatically"),
        ("📊", "#DCFCE7", "Executive dashboard", "KPIs, Pareto analysis, top performers, profitability quadrant"),
        ("🤖", "#FEF3C7", "ML predictions", "Regression & classification with model comparison"),
        ("💬", "#FCE7F3", "AI chat", "Ask questions about your data in plain English"),
    ]
    for col, (icon, bg, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon-badge" style="background:{bg};">{icon}</div>
                <div style="font-weight:600; font-size:14px; margin-bottom:6px;">{title}</div>
                <div style="font-size:12px; color:#64748B; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(" Upload a CSV or Excel file from the sidebar to get started.")
    st.caption("Try any sales, HR, or retail dataset from Kaggle to see the dashboard in action.")

    st.markdown("""
    <div class="trust-strip">
        Built with <b>Python</b> · <b>Streamlit</b> · <b>Scikit-learn</b> · <b>Plotly</b> · <b>Gemini AI</b>
    </div>
    """, unsafe_allow_html=True)