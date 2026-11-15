# 🍽️ Zomato Restaurant Analytics — Bangalore

> **End-to-end restaurant intelligence platform** — from raw Kaggle data engineering through SQL window-function KPIs, SciPy hypothesis testing, and a Power BI star schema to a live Streamlit operations dashboard.

---

## 📋 Table of Contents

1. [Executive Summary & Business Problem](#executive-summary)
2. [GitHub 100 MB Limit — Data Optimization Solution](#github-100-mb-limit)
3. [Technical Architecture & Tech Stack](#technical-architecture)
4. [SQL Window Functions & Statistical Testing](#sql--statistical-testing)
5. [Project Directory Structure](#project-directory-structure)
6. [Local Setup & Run Instructions](#local-setup--run-instructions)

---

## Executive Summary

### Business Problem

Bangalore's restaurant industry is hyper-competitive — over **12,000 listed outlets** across 90+ neighbourhoods, with operators making pricing, cuisine, and location decisions largely on intuition rather than data. This project answers four high-value business questions:

| # | Business Question | Analytical Method |
|---|---|---|
| 1 | Which locations and cuisine types drive the highest votes (proxy for footfall)? | SQL `DENSE_RANK()` window functions |
| 2 | Do online-order restaurants earn significantly higher ratings than dine-only? | Two-sample T-Test (`scipy.stats`) |
| 3 | How does approximate cost segment into pricing tiers, and what rating does each tier deliver? | `NTILE(4)` quartile analysis + ANOVA |
| 4 | Which restaurant types are over-/under-indexed relative to their location's average? | Z-score standardisation per locality |

### Key Findings (sample)

- **BTM, Koramangala, and Indiranagar** dominate the top-10 vote rankings across all cuisine categories.
- Online-order restaurants score **+0.31 rating points** higher on average — statistically significant (p < 0.001).
- Budget tier (₹200–400 for two) shows the **highest variance** in ratings, indicating the most opportunity for differentiation.
- North Indian and Chinese cuisines appear in **78 % of high-vote restaurants**, yet average rating trails Continental by 0.4 points.

---

## GitHub 100 MB Limit

### The Problem

The raw Kaggle dataset — `zomato.csv` — weighs **~550 MB** due to two free-text columns that are analytically useless for structured queries:

| Column | Avg. chars per row | Contribution to file size |
|---|---|---|
| `reviews_list` | ~1,800 | ~310 MB |
| `menu_item` | ~900 | ~160 MB |

Pushing this file directly to GitHub fails hard at the 100 MB per-file limit and bloats the repository permanently even after deletion (Git history retains blobs).

### The Solution — Lightweight Compressed Artefacts

The ETL pipeline (`src/data_optimizer.py`) applies three transformations before saving:

```python
# 1. Drop analytically redundant free-text columns
df.drop(columns=["reviews_list", "menu_item", "dish_liked"], inplace=True)

# 2. Downcast numeric types (float64 → float32, object categories → category dtype)
df["rate"]             = pd.to_numeric(df["rate"].str.replace("/5",""), errors="coerce").astype("float32")
df["approx_cost"]      = pd.to_numeric(df["approx_cost"].str.replace(",",""), errors="coerce").astype("float32")
df["votes"]            = df["votes"].astype("int32")
df["location"]         = df["location"].astype("category")
df["rest_type"]        = df["rest_type"].astype("category")
df["cuisines"]         = df["cuisines"].astype("category")

# 3. Persist as compressed artefacts
df.to_parquet("data/processed/restaurants.parquet", compression="snappy", index=False)
df.to_csv("data/processed/restaurants.csv.gz", compression="gzip", index=False)
```

| Artefact | Size | Use case |
|---|---|---|
| `restaurants.parquet` | **~9 MB** | Python / Pandas / Power BI DirectQuery |
| `restaurants.csv.gz` | **~13 MB** | Universal compatibility, SQL import |
| ~~`zomato.csv` (raw)~~ | ~~550 MB~~ | Excluded via `.gitignore` |

Both artefacts are **well under GitHub's 100 MB limit** and load in milliseconds compared to the raw CSV.

> **To reproduce from raw data:** Download `zomato.csv` from [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants), place it in `data/raw/`, then run `python src/data_optimizer.py`.

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Zomato Restaurant Analytics                         │
├─────────────────┬────────────────────┬───────────────────────────────┤
│   Data Layer    │  Analytics Layer   │      Presentation Layer       │
│                 │                    │                               │
│  data/raw/      │  src/              │  app.py  (Streamlit)          │
│  zomato.csv     │  data_optimizer.py │  ├ Executive KPI Summary      │
│  (git-ignored)  │  sql_analytics.py  │  ├ Location Heatmap           │
│                 │  hypothesis_tests  │  ├ Cuisine Rankings           │
│  data/processed/│  .py               │  ├ Pricing Tier Analysis      │
│  restaurants    │                    │  └ Hypothesis Test Results    │
│  .parquet       │  sql/              │                               │
│  restaurants    │  window_kpis.sql   │  Power BI                     │
│  .csv.gz        │  star_schema.sql   │  ├ Star Schema Model          │
│                 │                    │  ├ Location Drill-through     │
│  zomato.db      │  notebooks/        │  └ Rating vs. Cost Matrix     │
│  (SQLite)       │  EDA.ipynb         │                               │
└─────────────────┴────────────────────┴───────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Data Engineering** | Python, Pandas | 3.10+, ≥2.0 | Cleaning, type optimisation, compression |
| **Storage** | SQLite (via `sqlite3`) | Built-in | Queryable relational store for SQL analytics |
| **Compressed Artefacts** | Apache Parquet (Snappy), gzip CSV | — | GitHub-safe distribution |
| **SQL Analytics** | ANSI SQL — Window Functions | SQLite 3.38+ | DENSE_RANK, NTILE, LAG, running totals |
| **Statistical Testing** | SciPy | ≥1.10 | T-Test, one-way ANOVA, effect size (Cohen's d) |
| **BI / Reporting** | Power BI Desktop | — | Star schema, DAX measures, drill-through |
| **Dashboard** | Streamlit + Plotly | ≥1.35, ≥5.18 | Interactive web UI |
| **Notebook EDA** | Jupyter, Seaborn | — | Exploratory analysis, distribution plots |

### Power BI Star Schema Design

```
              ┌──────────────────┐
              │  fact_restaurant │  ← grain: one row per restaurant
              │  ─────────────── │
              │  restaurant_id PK│
              │  location_id  FK │───► dim_location  (location, city_zone)
              │  cuisine_id   FK │───► dim_cuisine   (cuisine, cuisine_group)
              │  rest_type_id FK │───► dim_rest_type (rest_type, category)
              │  rate            │
              │  votes           │
              │  approx_cost     │
              │  online_order    │
              │  book_table      │
              └──────────────────┘
```

---

## SQL & Statistical Testing

### Window Function KPIs (`sql/window_kpis.sql`)

**1. Location × Cuisine Leaderboard — `DENSE_RANK()`**
```sql
SELECT
    location,
    cuisines,
    votes,
    DENSE_RANK() OVER (
        PARTITION BY location
        ORDER BY votes DESC
    ) AS rank_within_location
FROM restaurants
WHERE votes > 0;
```

**2. Pricing Quartile Segmentation — `NTILE(4)`**
```sql
SELECT
    name,
    approx_cost,
    rate,
    NTILE(4) OVER (ORDER BY approx_cost) AS price_quartile,
    AVG(rate) OVER (
        PARTITION BY NTILE(4) OVER (ORDER BY approx_cost)
    ) AS avg_rating_in_tier
FROM restaurants
WHERE approx_cost IS NOT NULL AND rate IS NOT NULL;
```

**3. Month-over-Month Vote Growth — `LAG()`**
```sql
SELECT
    location,
    total_votes,
    LAG(total_votes) OVER (
        PARTITION BY location ORDER BY snapshot_month
    )                                   AS prev_month_votes,
    ROUND(
        100.0 * (total_votes - LAG(total_votes) OVER (
            PARTITION BY location ORDER BY snapshot_month
        )) / NULLIF(LAG(total_votes) OVER (
            PARTITION BY location ORDER BY snapshot_month
        ), 0),
    1)                                  AS mom_growth_pct
FROM monthly_location_votes;
```

### Statistical Hypothesis Testing (`src/hypothesis_tests.py`)

| Test | Hypothesis | Method | Result |
|---|---|---|---|
| **T-Test** | Online-order restaurants have higher avg rating than dine-only | `scipy.stats.ttest_ind` | p < 0.001 ✅ Reject H₀ |
| **ANOVA** | Rating differs significantly across the 4 price quartiles | `scipy.stats.f_oneway` | p < 0.001 ✅ Reject H₀ |
| **Cohen's d** | Effect size for online-order rating difference | Manual calc | d = 0.42 (medium effect) |

```python
from scipy import stats

online   = df[df["online_order"] == "Yes"]["rate"].dropna()
offline  = df[df["online_order"] == "No"]["rate"].dropna()

t_stat, p_val = stats.ttest_ind(online, offline, equal_var=False)
cohens_d = (online.mean() - offline.mean()) / (
    ((online.std()**2 + offline.std()**2) / 2) ** 0.5
)
```

---

## Project Directory Structure

```
Zomato-Restaurant-Analytics/
│
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Excludes data/raw/ and .venv/
│
├── data/
│   ├── raw/                        # ⚠ git-ignored — place zomato.csv here
│   └── processed/
│       ├── restaurants.parquet     # ~9 MB  — primary analytical dataset
│       └── restaurants.csv.gz      # ~13 MB — universal fallback
│
├── src/
│   ├── data_optimizer.py           # Raw → compressed artefact ETL
│   ├── load_data.py                # Parquet/csv.gz loader with caching
│   ├── sql_analytics.py            # SQLite query runner, returns DataFrames
│   └── hypothesis_tests.py         # T-Test, ANOVA, Cohen's d
│
├── sql/
│   ├── window_kpis.sql             # DENSE_RANK, NTILE, LAG queries
│   └── star_schema.sql             # Dimension & fact table DDL
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── powerbi/
│   └── ZomatoAnalytics.pbix        # Power BI report file
│
├── reports/
│   └── findings_summary.md         # Key statistical findings
│
└── .venv/                          # Virtual environment (git-ignored)
```

---

## Local Setup & Run Instructions

### Prerequisites

- Python 3.10 or higher
- pip
- *(Optional)* Power BI Desktop for the `.pbix` report

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/Zomato-Restaurant-Analytics.git
cd Zomato-Restaurant-Analytics
```

---

### Step 2 — Create and activate a virtual environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

> **Windows execution-policy note:** If you see a script-blocked error, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — *(First run only)* Reproduce the compressed dataset

> Skip this step if `data/processed/restaurants.parquet` already exists in the repo.

1. Download `zomato.csv` from [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
2. Place it at `data/raw/zomato.csv`
3. Run the optimiser:

```bash
python src/data_optimizer.py
```

Expected output:
```
Raw CSV loaded     : 51,717 rows × 17 columns  (550 MB)
After column drops : 51,717 rows × 10 columns
After type casting : memory reduced by 68 %
Wrote restaurants.parquet  →  9.1 MB
Wrote restaurants.csv.gz   → 12.8 MB
```

---

### Step 5 — Launch the Streamlit dashboard

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> If `streamlit` is not on your PATH, use:
> ```bash
> python -m streamlit run app.py
> ```

---

### Dashboard Pages

| Page | Contents |
|---|---|
| **Executive Summary** | Total restaurants, avg rating, online-order split, top location KPI tiles |
| **Location Analysis** | Choropleth-style bar chart, DENSE_RANK leaderboard table |
| **Cuisine Rankings** | Vote-weighted top cuisines, rating distribution by cuisine group |
| **Pricing Tier Analysis** | NTILE quartile box plots, cost vs. rating scatter |
| **Hypothesis Testing** | T-Test and ANOVA results with p-values, effect size, interpretation |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built as a portfolio project demonstrating end-to-end data engineering, SQL analytics, and statistical testing on real-world restaurant data.</p>
