# 🍽️ Zomato Restaurant Analytics — Bangalore

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQL-Window_Functions-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Hypothesis_Testing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Star_Schema-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)

> **End-to-end restaurant intelligence platform** — transforming raw Kaggle data into an optimized Parquet warehouse, SQL window-function KPIs, SciPy statistical hypothesis tests, a Power BI star schema, and an interactive Streamlit analytics dashboard.

## 📋 Table of Contents

1. [Executive Summary & Business Problem](#executive-summary--business-problem)
2. [GitHub 100 MB Limit — Data Optimization Solution](#github-100-mb-limit--data-optimization-solution)
3. [System Architecture & Tech Stack](#system-architecture--tech-stack)
4. [SQL Window Functions & Statistical Testing](#sql-window-functions--statistical-testing)
5. [Project Directory Structure](#project-directory-structure)
6. [Local Setup & Run Instructions](#local-setup--run-instructions)

## Executive Summary & Business Problem

### Business Problem

Bangalore's food-service market is hyper-competitive, with **31,936+ active restaurant listings** across 90+ neighborhoods. Operators frequently make pricing, cuisine, and delivery decisions on intuition rather than empirical evidence. This platform answers four core business questions:

| # | Business Question | Analytical Method |
|---|---|---|
| **1** | Which locations and outlets drive the highest customer engagement (votes)? | SQL `DENSE_RANK() OVER (PARTITION BY location)` |
| **2** | Do online-ordering restaurants earn significantly higher ratings than dine-in only? | Welch's Two-Sample T-Test (`scipy.stats.ttest_ind`) |
| **3** | How does meal cost segment into pricing tiers, and how do ratings vary across tiers? | `CASE` segmentation + `NTILE(4)` + One-Way ANOVA |
| **4** | How do individual outlets perform relative to their neighborhood's baseline? | Windowed `AVG() OVER (PARTITION BY location)` benchmarks |

### Key Platform Metrics & Findings

- **Dataset Scale:** **31,936** deduplicated, cleaned restaurant records across Bangalore.
- **Market Baselines:** **3.75 / 5.0** average rating | **₹450** median cost for two | **60.1%** online-order adoption.
- **Neighborhood Dominance:** **BTM, Koramangala, and Indiranagar** lead total vote volume across major cuisine categories.
- **Digital Delivery Lift:** Online-order restaurants achieve statistically higher ratings than offline-only outlets ($p < 0.001$).

## GitHub 100 MB Limit — Data Optimization Solution

### The Bottleneck

The raw Kaggle dataset (`data/raw/zomato.csv`) exceeds **550 MB** due to unindexed, free-text review arrays (`reviews_list`, `menu_item`) that exceed GitHub's 100 MB single-file limit and bloat repository clone times.

### The ETL Compression Pipeline

`data/data_loader.py` cleans, deduplicates, downcasts data types, and exports compressed columnar storage alongside dimension and fact tables for BI reporting:

```python
# 1. Strip redundant free-text blobs and standardize rating/cost fields
df.drop(columns=["reviews_list", "menu_item", "dish_liked"], errors="ignore", inplace=True)

# 2. Downcast numeric and categorical types for memory efficiency
df["rate"] = pd.to_numeric(df["rate"].astype(str).str.replace("/5", ""), errors="coerce").astype("float32")
df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int32")

# 3. Export compressed Parquet + Star Schema CSV dimensions
df.to_parquet("data/processed/zomato_clean.parquet", compression="snappy", index=False)
```

### Data Pipeline Artefacts

| Artefact | Location | Format | Purpose |
|---|---|---|---|
| **Cleaned Analytical Dataset** | `data/processed/zomato_clean.parquet` | Apache Parquet (Snappy) | Ultra-fast columnar storage for SQLite & Streamlit |
| **Dimension Tables** | `data/processed/dim_location.csv`, `dim_cuisine.csv` | Normalized CSV | Lookup dimensions for Power BI star schema |
| **Fact Table** | `data/processed/fact_restaurants.csv` | Transactional CSV | Core metrics fact table for BI modeling |
| **Raw Kaggle Dataset** | `data/raw/zomato.csv` | Raw CSV (550 MB) | Excluded from Git via `.gitignore` |

## System Architecture & Tech Stack

### End-to-End Pipeline Architecture

| Pipeline Stage | Directory / Module | Key Components | Output |
|---|---|---|---|
| **1. Data Ingestion & ETL** | `data/data_loader.py` | Deduplication, type downcasting, Snappy compression | `zomato_clean.parquet` & star-schema CSVs |
| **2. Relational & SQL Layer** | `src/sql_runner.py`, `sql/zomato_analytics.sql` | In-memory SQLite engine, `DENSE_RANK`, `NTILE(4)`, `SUM() OVER` | Location leaderboards & rating distributions |
| **3. Statistical Engine** | `src/statistical_analysis.py` | Welch's Two-Sample T-Test, One-Way ANOVA (`scipy.stats`) | Hypothesis validation ($p$-values & interpretations) |
| **4. Presentation & BI** | `app.py`, Power BI Desktop | Multi-tab Streamlit web app, Plotly visuals, DAX star schema | Interactive executive & operational dashboards |

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Engineering** | Python, Pandas, PyArrow | ETL pipeline, deduplication, Snappy Parquet serialization |
| **In-Memory SQL** | SQLite (`sqlite3`) | Zero-setup relational engine executing analytical views |
| **SQL Analytics** | ANSI SQL Window Functions | `DENSE_RANK`, `NTILE(4)`, `SUM() OVER`, `AVG() OVER` |
| **Statistical Testing** | SciPy (`scipy.stats`) | Welch's Two-Sample T-Test & One-Way ANOVA |
| **BI Modeling** | Power BI Star Schema | Dimensional modeling (`fact_restaurants` + `dim_*` tables) |
| **Web Application** | Streamlit + Plotly | Interactive multi-tab analytics dashboard |

### Power BI Star Schema Model

| Table Name | Role | Key Fields & Relationships |
|---|---|---|
| **`fact_restaurants`** | Fact Table (1 row per restaurant) | `restaurant_id` (PK), `location_id` (FK), `cuisine_id` (FK), `rate`, `votes`, `approx_cost`, `online_order`, `book_table` |
| **`dim_location`** | Location Dimension | `location_id` (PK), `location`, neighborhood benchmark metrics |
| **`dim_cuisine`** | Cuisine Dimension | `cuisine_id` (PK), `cuisine` categories |

## SQL Window Functions & Statistical Testing

### Analytical SQL Queries (`src/sql_runner.py` & `sql/zomato_analytics.sql`)

**1. Location Leaderboard & Cost-Tier Benchmarking**

```sql
SELECT
    restaurant_id,
    name,
    location,
    rate,
    votes,
    "approx_cost(for two)" AS cost,
    CASE
        WHEN "approx_cost(for two)" < 300  THEN 'Budget'
        WHEN "approx_cost(for two)" < 700  THEN 'Mid-Range'
        WHEN "approx_cost(for two)" < 1200 THEN 'Premium'
        ELSE                                    'Luxury'
    END AS cost_tier,
    DENSE_RANK() OVER (PARTITION BY location ORDER BY votes DESC)      AS rank_in_location,
    NTILE(4)     OVER (ORDER BY votes DESC)                            AS votes_quartile,
    ROUND(AVG(rate) OVER (PARTITION BY location), 2)                   AS avg_rating_in_location,
    ROUND(AVG("approx_cost(for two)") OVER (PARTITION BY location), 0) AS avg_cost_in_location
FROM restaurants
ORDER BY location, rank_in_location
LIMIT 500;
```

**2. Cumulative Rating Band Distribution**

```sql
WITH bands AS (
    SELECT
        CASE
            WHEN rate < 3.0 THEN '1-Poor (<3)'
            WHEN rate < 3.5 THEN '2-Fair (3-3.5)'
            WHEN rate < 4.0 THEN '3-Good (3.5-4)'
            WHEN rate < 4.5 THEN '4-Very Good (4-4.5)'
            ELSE                 '5-Excellent (>=4.5)'
        END AS rating_band,
        COUNT(*) AS restaurant_count
    FROM restaurants
    GROUP BY rating_band
)
SELECT
    rating_band,
    restaurant_count,
    SUM(restaurant_count) OVER (ORDER BY rating_band) AS running_total
FROM bands
ORDER BY rating_band;
```

### Statistical Hypothesis Testing (`src/statistical_analysis.py`)

| Statistical Test | Hypothesis Evaluated | Method | Decision Rule |
|---|---|---|---|
| **Welch's T-Test** | Online-order vs. offline-only restaurant ratings & votes | `scipy.stats.ttest_ind(..., equal_var=False)` | Reject $H_0$ if $p < 0.05$ |
| **Welch's T-Test** | Table-booking vs. walk-in restaurant ratings & votes | `scipy.stats.ttest_ind(..., equal_var=False)` | Reject $H_0$ if $p < 0.05$ |
| **One-Way ANOVA** | Rating variance across pricing tiers (Budget to Luxury) | `scipy.stats.f_oneway` | Reject $H_0$ if $p < 0.05$ |

## Project Directory Structure

| Path | Type | Description |
|---|---|---|
| `app.py` | Application | Main Streamlit dashboard entry point |
| `README.md` | Documentation | Project overview, architecture, and setup guide |
| `.gitignore` | Configuration | Excludes `data/raw/`, `.venv/`, and `__pycache__/` |
| `data/data_loader.py` | ETL Script | Cleans raw CSV and exports Parquet + Star Schema CSVs |
| `data/raw/` | Directory | Git-ignored folder for raw Kaggle `zomato.csv` |
| `data/processed/` | Directory | Stores `zomato_clean.parquet`, `dim_*.csv`, and `fact_restaurants.csv` |
| `sql/zomato_analytics.sql` | SQL Script | Analytical SQL views and window-function queries |
| `src/sql_runner.py` | Python Module | In-memory SQLite engine and KPI query runner |
| `src/statistical_analysis.py` | Python Module | SciPy Welch's T-Test and One-Way ANOVA module |

## Local Setup & Run Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/Sameer-khan-27/Zomato-Restaurant-Analytics.git](https://github.com/Sameer-khan-27/Zomato-Restaurant-Analytics.git)
cd Zomato-Restaurant-Analytics
```

### 2. Create & Activate Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install streamlit pandas pyarrow plotly scipy
```

### 4. *(Optional)* Rebuild Processed Data from Raw CSV

If `data/processed/zomato_clean.parquet` is not already present, download `zomato.csv` from [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), place it at `data/raw/zomato.csv`, and run:

```powershell
python data/data_loader.py
```

### 5. Launch the Streamlit Dashboard

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser to explore the interactive tabs:
- **SQL & Market Analysis** — Top restaurants per location, online adoption rates, and rating band distributions.
- **Hypothesis Testing** — Color-coded statistical significance table (`p-value`, test statistics, and business interpretations).
- **Power BI Star Schema** — Dimensional model preview and export-ready tables.

## Author

**MD Sameer Khan**  
GitHub: [@Sameer-khan-27](https://github.com/Sameer-khan-27)