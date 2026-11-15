"""
data_loader.py
==============
Loads the Kaggle Zomato Bangalore CSV (or generates a 5 000-row fallback),
drops bulky text columns, cleans the numeric fields, deduplicates, imputes
nulls, and exports a Star-Schema to data/processed/.

Outputs
-------
data/processed/zomato_clean.parquet   – cleaned fact+dim joined
data/processed/dim_location.csv       – dimension: location
data/processed/dim_cuisine.csv        – dimension: cuisine (exploded)
data/processed/fact_restaurants.csv.gz – compressed fact table
"""

from __future__ import annotations

import os
import random
import re
import string

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "zomato.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns to keep (Kaggle Zomato Bangalore schema subset)
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "name",
    "online_order",
    "book_table",
    "rate",
    "votes",
    "location",
    "rest_type",
    "cuisines",
    "approx_cost(for two)",
    "listed_in(type)",
]

BULKY_COLS = ["reviews_list", "menu_item", "dish_liked", "url", "address", "phone"]

# ---------------------------------------------------------------------------
# Fallback data generation
# ---------------------------------------------------------------------------
LOCATIONS = [
    "Banashankari", "Basavanagudi", "Mysore Road", "Jayanagar", "Kumaraswamy Layout",
    "Rajarajeshwari Nagar", "Bellandur", "Sarjapur Road", "Whitefield", "Old Airport Road",
    "Indiranagar", "Koramangala 5th Block", "Koramangala 6th Block", "HSR", "BTM",
    "Electronic City", "Marathahalli", "Brookefield", "Frazer Town", "MG Road",
    "Brigade Road", "Lavelle Road", "Church Street", "Residency Road", "Shivajinagar",
    "Rajajinagar", "Malleshwaram", "Hebbal", "Yelahanka", "Kalyan Nagar",
]
REST_TYPES = [
    "Quick Bites", "Casual Dining", "Cafe", "Delivery", "Dessert Parlor",
    "Fine Dining", "Bar", "Pub", "Food Court", "Bakery",
]
CUISINE_LIST = [
    "North Indian", "South Indian", "Chinese", "Fast Food", "Biryani",
    "Pizza", "Italian", "Continental", "Cafe", "Desserts",
    "Mughlai", "Beverages", "Seafood", "Street Food", "Thai",
    "Mexican", "American", "Rolls", "Sandwich", "Burger",
]
LISTED_TYPES = ["Buffet", "Cafes", "Delivery", "Desserts", "Dine-out", "Drinks & Nightlife", "Pubs and Bars"]


def _rand_name() -> str:
    prefixes = ["The", "Spice", "Royal", "Urban", "Desi", "Golden", "Olive", "Pepper", "Curry", "Star"]
    suffixes = ["Kitchen", "Grill", "Bites", "House", "Cafe", "Corner", "Garden", "Palace", "Junction", "Hub"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)} {''.join(random.choices(string.ascii_uppercase, k=2))}"


def _generate_fallback(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    names = [_rand_name() for _ in range(n)]
    online = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    book = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    rates = rng.uniform(2.0, 5.0, n).round(1)
    # ~8 % NaN in rate to simulate real data
    nan_mask = rng.random(n) < 0.08
    rates_str = [f"{r}/5" if not nan_mask[i] else "NEW" for i, r in enumerate(rates)]
    votes = rng.integers(0, 5000, n).tolist()
    locations = rng.choice(LOCATIONS, n).tolist()
    rest_types = rng.choice(REST_TYPES, n).tolist()
    cuisines = [
        ", ".join(rng.choice(CUISINE_LIST, rng.integers(1, 4), replace=False).tolist())
        for _ in range(n)
    ]
    costs = rng.integers(100, 3000, n)
    # add comma formatting for ~30 % to mimic raw data
    costs_str = [f"{c:,}" if rng.random() < 0.3 else str(c) for c in costs]
    listed = rng.choice(LISTED_TYPES, n).tolist()

    return pd.DataFrame({
        "name": names,
        "online_order": online,
        "book_table": book,
        "rate": rates_str,
        "votes": votes,
        "location": locations,
        "rest_type": rest_types,
        "cuisines": cuisines,
        "approx_cost(for two)": costs_str,
        "listed_in(type)": listed,
    })


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_rate(series: pd.Series) -> pd.Series:
    """Strip '/5', coerce non-numeric tokens (NEW, -) to NaN, cast to float."""
    cleaned = series.astype(str).str.replace(r"/5\s*$", "", regex=True).str.strip()
    cleaned = cleaned.replace({"NEW": np.nan, "-": np.nan, "nan": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _clean_cost(series: pd.Series) -> pd.Series:
    """Remove commas, cast to float."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_clean() -> pd.DataFrame:
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV, low_memory=False)
        
        # Clean column names by stripping trailing/leading whitespaces
        df.columns = df.columns.str.strip()
        
        # Find cost column dynamically
        cost_col = None
        for col in df.columns:
            if "cost" in col.lower() and ("two" in col.lower() or "approx" in col.lower()):
                cost_col = col
                break
        
        if cost_col and cost_col != "approx_cost(for two)":
            df.rename(columns={cost_col: "approx_cost(for two)"}, inplace=True)
            
        # Keep only schema columns that exist
        present = [c for c in KEEP_COLS if c in df.columns]
        df = df[present].copy()
    else:
        print("[data_loader] Raw CSV not found — generating 5,000-row fallback dataset.")
        df = _generate_fallback(5000)

    # --- Clean numeric fields ---
    if "rate" in df.columns:
        df["rate"] = _clean_rate(df["rate"])
    if "approx_cost(for two)" in df.columns:
        df["approx_cost(for two)"] = _clean_cost(df["approx_cost(for two)"])

    # --- Normalise binary columns ---
    for col in ["online_order", "book_table"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()

    # --- Deduplicate ---
    df.drop_duplicates(inplace=True)

    # --- Impute nulls ---
    df["rate"].fillna(df["rate"].median(), inplace=True)
    df["approx_cost(for two)"].fillna(df["approx_cost(for two)"].median(), inplace=True)
    df["votes"].fillna(0, inplace=True)
    for col in ["location", "rest_type", "cuisines", "listed_in(type)"]:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def build_star_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (dim_location, dim_cuisine, fact_restaurants)."""
    # Dim Location
    dim_location = (
        df[["location"]].drop_duplicates().reset_index(drop=True)
        .rename_axis("location_id")
        .reset_index()
    )
    dim_location["location_id"] += 1

    # Dim Cuisine (explode multi-value)
    cuisine_exploded = (
        df["cuisines"].str.split(r",\s*").explode().str.strip()
        .dropna().drop_duplicates().reset_index(drop=True)
        .rename("cuisine_name")
        .to_frame()
        .rename_axis("cuisine_id")
        .reset_index()
    )
    cuisine_exploded["cuisine_id"] += 1
    dim_cuisine = cuisine_exploded

    # Fact table
    loc_map = dict(zip(dim_location["location"], dim_location["location_id"]))
    fact = df.copy()
    fact.insert(0, "restaurant_id", range(1, len(fact) + 1))
    fact["location_id"] = fact["location"].map(loc_map)
    fact_cols = [
        "restaurant_id", "name", "location_id", "online_order", "book_table",
        "rate", "votes", "rest_type", "cuisines", "approx_cost(for two)", "listed_in(type)",
    ]
    fact_cols = [c for c in fact_cols if c in fact.columns]
    fact = fact[fact_cols]

    return dim_location, dim_cuisine, fact


def export_all() -> pd.DataFrame:
    df = load_and_clean()
    dim_location, dim_cuisine, fact = build_star_schema(df)

    # 1. Full clean parquet (compressed by default via snappy)
    df.to_parquet(os.path.join(PROCESSED_DIR, "zomato_clean.parquet"), index=False)

    # 2. Dimension tables as CSV
    dim_location.to_csv(os.path.join(PROCESSED_DIR, "dim_location.csv"), index=False)
    dim_cuisine.to_csv(os.path.join(PROCESSED_DIR, "dim_cuisine.csv"), index=False)

    # 3. Fact table as compressed gzip CSV
    fact.to_csv(os.path.join(PROCESSED_DIR, "fact_restaurants.csv.gz"), index=False, compression="gzip")

    print(f"[data_loader] Exported {len(df):,} rows.")
    print(f"  → {PROCESSED_DIR}/zomato_clean.parquet")
    print(f"  → {PROCESSED_DIR}/dim_location.csv  ({len(dim_location)} rows)")
    print(f"  → {PROCESSED_DIR}/dim_cuisine.csv  ({len(dim_cuisine)} rows)")
    print(f"  → {PROCESSED_DIR}/fact_restaurants.csv.gz  ({len(fact):,} rows)")
    return df


if __name__ == "__main__":
    export_all()
