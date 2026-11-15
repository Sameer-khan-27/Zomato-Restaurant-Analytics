"""
sql_runner.py
=============
Loads zomato_clean.parquet into an in-memory SQLite database, creates the
views defined in sql/zomato_analytics.sql, then runs each named query and
returns results as a dict of DataFrames.

Usage
-----
    from src.sql_runner import run_all_queries
    results = run_all_queries()   # dict keyed by query name
"""

from __future__ import annotations

import os
import sqlite3
import textwrap

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "processed", "zomato_clean.parquet")
SQL_PATH = os.path.join(BASE_DIR, "sql", "zomato_analytics.sql")


# ---------------------------------------------------------------------------
# Inline SQL for each named query (avoids parsing the .sql file at runtime)
# ---------------------------------------------------------------------------

_SQL_TOP_RESTAURANTS = textwrap.dedent("""
    WITH top_per_location AS (
        SELECT *
        FROM   v_ranked_by_location
        WHERE  rank_in_location = 1
    )
    SELECT
        t.location,
        t.name                      AS top_restaurant,
        t.votes,
        t.rate,
        t.cost_tier,
        t.avg_rating_in_location,
        t.avg_cost_in_location
    FROM   top_per_location t
    ORDER  BY t.votes DESC
    LIMIT  30;
""")

_SQL_ONLINE_ADOPTION = textwrap.dedent("""
    WITH online_stats AS (
        SELECT
            location,
            COUNT(*)                                                AS total,
            SUM(CASE WHEN online_order = 'Yes' THEN 1 ELSE 0 END)  AS online_count
        FROM   restaurants
        GROUP  BY location
    )
    SELECT
        location,
        total,
        online_count,
        ROUND(100.0 * online_count / total, 1)  AS online_pct
    FROM   online_stats
    ORDER  BY online_pct DESC;
""")

_SQL_RATING_BANDS = textwrap.dedent("""
    WITH bands AS (
        SELECT
            CASE
                WHEN rate < 3.0 THEN '1-Poor (<3)'
                WHEN rate < 3.5 THEN '2-Fair (3-3.5)'
                WHEN rate < 4.0 THEN '3-Good (3.5-4)'
                WHEN rate < 4.5 THEN '4-Very Good (4-4.5)'
                ELSE                 '5-Excellent (>=4.5)'
            END                 AS rating_band,
            COUNT(*)            AS restaurant_count
        FROM   restaurants
        GROUP  BY rating_band
    ),
    running AS (
        SELECT
            rating_band,
            restaurant_count,
            SUM(restaurant_count) OVER (ORDER BY rating_band)  AS running_total
        FROM   bands
    )
    SELECT * FROM running ORDER BY rating_band;
""")

_SQL_FULL_WINDOW = textwrap.dedent("""
    SELECT
        restaurant_id,
        name,
        location,
        rate,
        votes,
        "approx_cost(for two)"                                                  AS cost,
        CASE
            WHEN "approx_cost(for two)" < 300  THEN 'Budget'
            WHEN "approx_cost(for two)" < 700  THEN 'Mid-Range'
            WHEN "approx_cost(for two)" < 1200 THEN 'Premium'
            ELSE                                    'Luxury'
        END                                                                      AS cost_tier,
        DENSE_RANK() OVER (PARTITION BY location ORDER BY votes DESC)            AS rank_in_location,
        NTILE(4)     OVER (ORDER BY votes DESC)                                  AS votes_quartile,
        ROUND(AVG(rate) OVER (PARTITION BY location), 2)                         AS avg_rating_in_location,
        ROUND(AVG("approx_cost(for two)") OVER (PARTITION BY location), 0)       AS avg_cost_in_location
    FROM restaurants
    ORDER BY location, rank_in_location
    LIMIT 500;
""")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _get_connection(parquet_path: str = PARQUET_PATH) -> sqlite3.Connection:
    """Return an in-memory SQLite connection pre-loaded with the restaurants table."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Processed parquet not found at {parquet_path}.\n"
            "Run:  python data/data_loader.py"
        )
    df = pd.read_parquet(parquet_path)
    con = sqlite3.connect(":memory:")
    df.to_sql("restaurants", con, if_exists="replace", index=False)

    # Create views from SQL file
    with open(SQL_PATH, "r", encoding="utf-8") as fh:
        raw_sql = fh.read()

    # Execute only CREATE VIEW statements (skip bare SELECTs / CTEs)
    for statement in raw_sql.split(";"):
        stmt = statement.strip()
        if stmt.upper().startswith("CREATE VIEW"):
            try:
                con.execute(stmt)
            except sqlite3.OperationalError:
                pass  # view may already exist
    con.commit()
    return con


def run_all_queries(parquet_path: str = PARQUET_PATH) -> dict[str, pd.DataFrame]:
    """Execute all analytics queries and return results as named DataFrames."""
    con = _get_connection(parquet_path)
    results: dict[str, pd.DataFrame] = {}
    queries = {
        "top_restaurants_per_location": _SQL_TOP_RESTAURANTS,
        "online_adoption_by_location": _SQL_ONLINE_ADOPTION,
        "rating_band_distribution": _SQL_RATING_BANDS,
        "full_window_functions": _SQL_FULL_WINDOW,
    }
    for name, sql in queries.items():
        results[name] = pd.read_sql_query(sql, con)
    con.close()
    return results


if __name__ == "__main__":
    res = run_all_queries()
    for key, df in res.items():
        print(f"\n=== {key} ===")
        print(df.head(5).to_string(index=False))
