# 🍽️ Zomato Restaurant Analytics — Bangalore

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQL-Window_Functions-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Hypothesis_Testing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Star_Schema-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)

> **End-to-end restaurant intelligence platform** — transforming raw Kaggle data into an optimized Parquet warehouse, SQL window-function KPIs, SciPy statistical hypothesis tests, a Power BI star schema, and an interactive Streamlit analytics dashboard.

---

## 📋 Table of Contents

1. [Executive Summary & Business Problem](#executive-summary--business-problem)
2. [GitHub 100 MB Limit — Data Optimization Solution](#github-100-mb-limit--data-optimization-solution)
3. [Technical Architecture & Tech Stack](#technical-architecture--tech-stack)
4. [SQL Window Functions & Statistical Testing](#sql-window-functions--statistical-testing)
5. [Project Directory Structure](#project-directory-structure)
6. [Local Setup & Run Instructions](#local-setup--run-instructions)

---

## Executive Summary & Business Problem

### Business Problem

Bangalore's food-service market is hyper-competitive, with **31,936+ active restaurant listings** across 90+ neighborhoods. Operators frequently make pricing, cuisine, and delivery decisions on intuition rather than empirical evidence. This platform answers four core business questions:

| # | Business Question | Analytical Method |
|---|---|---|
| 1 | Which locations and outlets drive the highest customer engagement (votes)? | SQL `DENSE_RANK() OVER (PARTITION BY location)` |
| 2 | Do online-ordering restaurants earn significantly higher ratings than dine-in only? | Welch's Two-Sample T-Test (`scipy.stats.ttest_ind`) |
| 3 | How does meal cost segment into pricing tiers, and how do ratings vary across tiers? | `CASE` segmentation + `NTILE(4)` + One-Way ANOVA |
| 4 | How do individual outlets perform relative to their neighborhood's baseline? | Windowed `AVG() OVER (PARTITION BY location)` benchmarks |

### Key Platform Metrics & Findings

- **Dataset Scale:** **31,936** deduplicated, cleaned restaurant records across Bangalore.
- **Market Baselines:** **3.75 / 5.0** average rating | **₹450** median cost for two | **60.1%** online-order adoption.
- **Neighborhood Dominance:** **BTM, Koramangala, and Indiranagar** lead total vote volume across major cuisine categories.
- **Digital Delivery Lift:** Online-order restaurants achieve statistically higher ratings than offline-only outlets ($p < 0.001$).

---

## GitHub 100 MB Limit — Data Optimization Solution

### The Bottleneck

The raw Kaggle dataset (`data/raw/zomato.csv`) exceeds **550 MB** largely due to unindexed, free-text review arrays (`reviews_list`, `menu_item`) that exceed GitHub's 100 MB single-file limit and bloat repository clone times.

### The ETL Compression Pipeline

`data/data_loader.py` cleans, deduplicates, downcasts data types, and exports compressed columnar storage alongside dimension/fact tables for BI reporting:

```python
# 1. Strip redundant free-text blobs and standardize rating/cost fields
df.drop(columns=["reviews_list", "menu_item", "dish_liked"], errors="ignore", inplace=True)

# 2. Downcast numeric and categorical types for memory efficiency
df["rate"] = pd.to_numeric(df["rate"].astype(str).str.replace("/5", ""), errors="coerce").astype("float32")
df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype("int32")

# 3. Export compressed Parquet + Star Schema CSV dimensions
df.to_parquet("data/processed/zomato_clean.parquet", compression="snappy", index=False)

| Artefact | Location | Purpose |
|---|---|---|
| `zomato_clean.parquet` | `data/processed/` | Ultra-fast columnar storage for SQLite & Streamlit |
| `dim_location.csv` / `dim_cuisine.csv` | `data/processed/` | Normalized lookup dimensions for Power BI |
| `fact_restaurants.csv` | `data/processed/` | Core transactional fact table |
| ~~`zomato.csv` (raw)~~ | `data/raw/` | Excluded from version control via `.gitignore` |

---

## Technical Architecture & Tech Stack

```text
┌──────────────────────────────────────────────────────────────────────┐
│                   Zomato Restaurant Analytics                        │
├─────────────────┬────────────────────┬───────────────────────────────┤
│   Data Layer    │  Analytics Layer   │      Presentation Layer       │
│                 │                    │                               │
│ data/raw/       │ src/               │ app.py (Streamlit)            │
│ └ zomato.csv    │ ├ sql_runner.py    │ ├ Executive KPI Header        │
│   (git-ignored) │ └ statistical_     │ ├ SQL & Market Analysis       │
│                 │   analysis.py      │ ├ Hypothesis Testing          │
│ data/processed/ │                    │ └ Power BI Star Schema View   │
│ ├ zomato_clean  │ sql/               │                               │
│ │ .parquet      │ └ zomato_          │ Power BI Desktop              │
│ └ dim/fact CSVs │   analytics.sql    │ └ Star Schema & DAX Measures  │
└─────────────────┴────────────────────┴───────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Engineering** | Python, Pandas, PyArrow | ETL pipeline, deduplication, Snappy Parquet serialization |
| **In-Memory SQL** | SQLite (`sqlite3`) | Zero-setup relational engine executing analytical views |
| **SQL Analytics** | ANSI SQL Window Functions | `DENSE_RANK`, `NTILE(4)`, `SUM() OVER`, `AVG() OVER` |
| **Statistical Testing** | SciPy (`scipy.stats`) | Welch's Two-Sample T-Test & One-Way ANOVA |
| **BI Modeling** | Power BI Star Schema | Dimensional modeling (`fact_restaurants` + `dim_*` tables) |
| **Web Application** | Streamlit + Plotly | Interactive multi-tab analytics dashboard |

### Power BI Star Schema Design

```text
              ┌────────────────────┐
              │  fact_restaurants  │  ← Grain: 1 row per restaurant
              │  ────────────────  │
              │  restaurant_id PK  │
              │  location_id   FK  │───► dim_location (location, neighborhood metrics)
              │  cuisine_id    FK  │───► dim_cuisine  (cuisine categories)
              │  rate              │
              │  votes             │
              │  approx_cost       │
              │  online_order      │
              │  book_table        │
              └────────────────────┘
```
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

---

## Project Directory Structure

```text
Zomato-Restaurant-Analytics/
│
├── app.py                          # Main Streamlit dashboard application
├── README.md                       # Project documentation
├── .gitignore                      # Excludes data/raw/, .venv/, and __pycache__/
│
├── data/
│   ├── data_loader.py              # ETL script: cleans raw CSV → Parquet + Star Schema CSVs
│   ├── raw/                        # Git-ignored — place raw Kaggle zomato.csv here
│   └── processed/
│       ├── zomato_clean.parquet    # Primary compressed analytical dataset
│       ├── dim_cuisine.csv         # Cuisine dimension table
│       ├── dim_location.csv        # Location dimension table
│       └── fact_restaurants.csv    # Fact table for Power BI
│
├── sql/
│   └── zomato_analytics.sql        # DDL views and analytical SQL queries
│
├── src/
│   ├── sql_runner.py               # In-memory SQLite engine & window-function runner
│   └── statistical_analysis.py     # SciPy T-Test & ANOVA module
│
└── .venv/                          # Local Python virtual environment (git-ignored)
```

---

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

If `data/processed/zomato_clean.parquet` is not already present, download `zomato.csv` from [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), place it in `data/raw/zomato.csv`, and run:

```powershell
python data/data_loader.py
```

### 5. Launch the Streamlit Dashboard

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** (or **http://localhost:8502**) in your browser to explore the interactive tabs:
- **SQL & Market Analysis** — Top restaurants per location, online adoption rates, and rating band distributions.
- **Hypothesis Testing** — Color-coded statistical significance table (`p-value`, test statistics, and business interpretations).
- **Power BI Star Schema** — Dimensional model preview and export-ready tables.

---

## Author

**MD Sameer Khan**  
GitHub: [@Sameer-khan-27](https://github.com/Sameer-khan-27)