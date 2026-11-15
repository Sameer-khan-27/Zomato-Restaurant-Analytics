"""
app.py  –  Zomato Restaurant Analytics Dashboard
=================================================
Streamlit app reading ONLY from data/processed/.

Tabs
----
  Tab 1 – SQL & Market Analysis
  Tab 2 – Hypothesis Testing
  Tab 3 – Power BI Star Schema
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from src/ and data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARQUET_PATH = os.path.join(PROCESSED_DIR, "zomato_clean.parquet")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zomato Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom minimal CSS
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    h1 {font-size: 1.8rem !important;}
    .stMetric label {font-size: 0.8rem !important;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data bootstrap – run data_loader if processed files are missing
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing data…")
def bootstrap_data() -> None:
    if not os.path.exists(PARQUET_PATH):
        from data.data_loader import export_all  # noqa: PLC0415
        export_all()


bootstrap_data()


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


@st.cache_data(show_spinner=False)
def load_dim_location() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, "dim_location.csv"))


@st.cache_data(show_spinner=False)
def load_dim_cuisine() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROCESSED_DIR, "dim_cuisine.csv"))


@st.cache_data(show_spinner=False)
def load_fact() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PROCESSED_DIR, "fact_restaurants.csv.gz"),
        compression="gzip",
    )


@st.cache_data(show_spinner=False)
def run_sql_queries() -> dict[str, pd.DataFrame]:
    from src.sql_runner import run_all_queries  # noqa: PLC0415
    return run_all_queries(PARQUET_PATH)


@st.cache_data(show_spinner=False)
def run_stats() -> pd.DataFrame:
    from src.statistical_analysis import run_all_tests, results_to_dataframe  # noqa: PLC0415
    return results_to_dataframe(run_all_tests(PARQUET_PATH))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_clean()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🍽️ Zomato Bangalore Restaurant Analytics")
st.caption("Source: Kaggle Zomato Bangalore Dataset  |  Data processed from `data/processed/`")

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
total = len(df)
avg_rating = df["rate"].mean()
median_cost = df["approx_cost(for two)"].median()
pct_online = (df["online_order"].str.capitalize() == "Yes").mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏪 Total Restaurants", f"{total:,}")
k2.metric("⭐ Avg Rating", f"{avg_rating:.2f} / 5")
k3.metric("💰 Median Cost (₹)", f"{median_cost:,.0f}")
k4.metric("📱 Online Order", f"{pct_online:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 SQL & Market Analysis",
    "🧪 Hypothesis Testing",
    "🗂️ Power BI Star Schema",
])


# ============================================================
# TAB 1 – SQL & Market Analysis
# ============================================================
with tab1:
    st.subheader("SQL Window Functions – Top Restaurant per Location")
    st.caption("DENSE_RANK() OVER (PARTITION BY location ORDER BY votes DESC) · NTILE(4) · AVG() OVER()")

    with st.spinner("Running SQL queries…"):
        sql_results = run_sql_queries()

    top_df = sql_results.get("top_restaurants_per_location", pd.DataFrame())
    if not top_df.empty:
        st.dataframe(
            top_df.rename(columns={
                "avg_rating_in_location": "Avg Rating (Location)",
                "avg_cost_in_location": "Avg Cost (Location)",
                "top_restaurant": "Restaurant",
            }),
            use_container_width=True,
            height=350,
        )

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💲 Cost vs. Rating")
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="approx_cost(for two)",
            y="rate",
            color="online_order",
            hover_name="name",
            hover_data=["location", "votes"],
            labels={
                "approx_cost(for two)": "Approx Cost (₹ for 2)",
                "rate": "Rating",
                "online_order": "Online Order",
            },
            opacity=0.6,
            color_discrete_map={"Yes": "#3b82d4", "No": "#e57373"},
            title="Cost vs. Rating (sample n=1,500)",
        )
        fig_scatter.update_layout(
            height=400, margin=dict(t=40, b=20), legend_title_text="Online Order"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("🍜 Top 15 Cuisines by Restaurant Count")
        cuisine_counts = (
            df["cuisines"]
            .str.split(r",\s*")
            .explode()
            .str.strip()
            .value_counts()
            .head(15)
            .reset_index()
        )
        cuisine_counts.columns = ["Cuisine", "Count"]
        fig_bar = px.bar(
            cuisine_counts,
            x="Count",
            y="Cuisine",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            title="Top 15 Cuisines",
        )
        fig_bar.update_layout(
            height=400,
            margin=dict(t=40, b=20),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Online adoption heatmap
    st.markdown("---")
    st.subheader("📍 Online Order Adoption by Location (Top 20)")
    online_df = sql_results.get("online_adoption_by_location", pd.DataFrame())
    if not online_df.empty:
        top20_online = online_df.head(20)
        fig_online = px.bar(
            top20_online,
            x="online_pct",
            y="location",
            orientation="h",
            color="online_pct",
            color_continuous_scale="Teal",
            labels={"online_pct": "Online Order %", "location": "Location"},
            title="Online Order Adoption Rate by Location (Top 20)",
            text="online_pct",
        )
        fig_online.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_online.update_layout(
            height=460,
            margin=dict(t=40, b=20),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_online, use_container_width=True)


# ============================================================
# TAB 2 – Hypothesis Testing
# ============================================================
with tab2:
    st.subheader("Statistical Hypothesis Tests (α = 0.05)")
    st.caption(
        "Welch's Two-Sample T-Test and One-Way ANOVA comparing ratings and votes "
        "across Online Order and Book Table groups."
    )

    with st.spinner("Running statistical tests…"):
        stats_df = run_stats()

    # Colour-code significant rows
    def _highlight(row: pd.Series) -> list[str]:
        color = "background-color: #d4edda" if row["significant"] else "background-color: #f8d7da"
        return [color] * len(row)

    display_df = stats_df[["test", "statistic", "p_value", "significant", "interpretation"]].copy()
    display_df.columns = ["Test", "Statistic", "p-value", "Significant", "Interpretation"]

    st.dataframe(
        display_df.style.apply(_highlight, axis=1),
        use_container_width=True,
        height=420,
    )

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("⭐ Rating by Online Order")
        fig_box1 = px.box(
            df,
            x="online_order",
            y="rate",
            color="online_order",
            points="outliers",
            color_discrete_map={"Yes": "#3b82d4", "No": "#e57373"},
            labels={"online_order": "Online Order", "rate": "Rating"},
            title="Rating Distribution: Online Order Yes vs No",
        )
        fig_box1.update_layout(height=380, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_box1, use_container_width=True)

    with col_b2:
        st.subheader("⭐ Rating by Book Table")
        fig_box2 = px.box(
            df,
            x="book_table",
            y="rate",
            color="book_table",
            points="outliers",
            color_discrete_map={"Yes": "#7c5cd8", "No": "#ffb74d"},
            labels={"book_table": "Book Table", "rate": "Rating"},
            title="Rating Distribution: Book Table Yes vs No",
        )
        fig_box2.update_layout(height=380, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_box2, use_container_width=True)

    col_b3, col_b4 = st.columns(2)
    with col_b3:
        st.subheader("🗳️ Votes by Online Order")
        fig_box3 = px.box(
            df,
            x="online_order",
            y="votes",
            color="online_order",
            points="outliers",
            color_discrete_map={"Yes": "#3b82d4", "No": "#e57373"},
            labels={"online_order": "Online Order", "votes": "Votes"},
            title="Vote Distribution: Online Order Yes vs No",
        )
        fig_box3.update_layout(height=380, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_box3, use_container_width=True)

    with col_b4:
        st.subheader("🗳️ Votes by Book Table")
        fig_box4 = px.box(
            df,
            x="book_table",
            y="votes",
            color="book_table",
            points="outliers",
            color_discrete_map={"Yes": "#7c5cd8", "No": "#ffb74d"},
            labels={"book_table": "Book Table", "votes": "Votes"},
            title="Vote Distribution: Book Table Yes vs No",
        )
        fig_box4.update_layout(height=380, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_box4, use_container_width=True)


# ============================================================
# TAB 3 – Power BI Star Schema
# ============================================================
with tab3:
    st.subheader("🗂️ Power BI Star Schema – Table Previews")
    st.caption(
        "Processed files in `data/processed/`. "
        "Connect Power BI Desktop to these files via **Get Data → Text/CSV / Parquet**."
    )

    dim_location = load_dim_location()
    dim_cuisine = load_dim_cuisine()
    fact = load_fact()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**dim_location.csv**")
        st.dataframe(dim_location.head(10), use_container_width=True, height=260)
        st.caption(f"{len(dim_location)} locations")

    with c2:
        st.markdown("**dim_cuisine.csv**")
        st.dataframe(dim_cuisine.head(10), use_container_width=True, height=260)
        st.caption(f"{len(dim_cuisine)} unique cuisines")

    with c3:
        st.markdown("**fact_restaurants.csv.gz** (preview)")
        st.dataframe(fact.head(10), use_container_width=True, height=260)
        st.caption(f"{len(fact):,} restaurants")

    st.markdown("---")
    st.subheader("📐 Key DAX Formulas")

    dax_formulas = [
        (
            "Avg Rating",
            "Avg Rating = AVERAGE(fact_restaurants[rate])",
            "Calculates the mean rating across all restaurants in the current filter context.",
        ),
        (
            "Online Order % ",
            (
                "Online Order % = \n"
                "DIVIDE(\n"
                "    COUNTROWS(FILTER(fact_restaurants, fact_restaurants[online_order] = \"Yes\")),\n"
                "    COUNTROWS(fact_restaurants)\n"
                ") * 100"
            ),
            "Percentage of restaurants offering online ordering.",
        ),
        (
            "Avg Cost by Location",
            (
                "Avg Cost by Location = \n"
                "CALCULATE(\n"
                "    AVERAGE(fact_restaurants[approx_cost(for two)]),\n"
                "    ALLEXCEPT(fact_restaurants, dim_location[location])\n"
                ")"
            ),
            "Average cost per two persons, isolated to the selected location dimension.",
        ),
    ]

    for title_dax, formula, desc in dax_formulas:
        with st.expander(f"📌 {title_dax}", expanded=True):
            st.code(formula, language="dax")
            st.caption(desc)

    st.markdown("---")
    st.subheader("🔗 Schema Diagram")
    st.markdown("""
    ```
    ┌──────────────────────┐         ┌───────────────────────────────────────────────┐
    │   dim_location       │         │              fact_restaurants                  │
    │──────────────────────│         │───────────────────────────────────────────────│
    │ PK  location_id INT  │◄────────│ FK  location_id  INT                           │
    │     location   TEXT  │         │ PK  restaurant_id INT                          │
    └──────────────────────┘         │     name          TEXT                         │
                                     │     online_order  TEXT (Yes/No)                │
    ┌──────────────────────┐         │     book_table    TEXT (Yes/No)                │
    │   dim_cuisine        │         │     rate          REAL                         │
    │──────────────────────│         │     votes         INT                          │
    │ PK  cuisine_id  INT  │         │     rest_type     TEXT                         │
    │     cuisine_name TEXT│         │     cuisines      TEXT (FK → dim_cuisine)      │
    └──────────────────────┘         │     approx_cost   REAL                         │
                                     │     listed_in     TEXT                         │
                                     └───────────────────────────────────────────────┘
    ```
    """)
