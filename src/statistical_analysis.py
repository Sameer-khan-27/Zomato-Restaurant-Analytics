"""
statistical_analysis.py
========================
Runs hypothesis tests comparing restaurant ratings and vote counts
across online_order (Yes/No) and book_table (Yes/No) groups.

Tests performed
---------------
1. Two-sample independent T-Test  (scipy.stats.ttest_ind)
   – rate:  online_order Yes vs No
   – votes: online_order Yes vs No
   – rate:  book_table   Yes vs No
   – votes: book_table   Yes vs No

2. One-way ANOVA  (scipy.stats.f_oneway)
   – rate  across all 4 (online_order × book_table) groups
   – votes across all 4 groups

Returns
-------
A dict mapping descriptive test names → TestResult(statistic, pvalue, significant).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "processed", "zomato_clean.parquet")

ALPHA = 0.05


@dataclass
class TestResult:
    test_name: str
    statistic: float
    pvalue: float
    significant: bool
    interpretation: str

    def __repr__(self) -> str:
        sig = "✓ Significant" if self.significant else "✗ Not significant"
        return (
            f"{self.test_name}\n"
            f"  stat={self.statistic:.4f}  p={self.pvalue:.4f}  [{sig}]\n"
            f"  {self.interpretation}"
        )


def _load(parquet_path: str = PARQUET_PATH) -> pd.DataFrame:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Processed parquet not found at {parquet_path}.\n"
            "Run:  python data/data_loader.py"
        )
    return pd.read_parquet(parquet_path)


def _ttest(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    label_a: str = "Yes",
    label_b: str = "No",
) -> TestResult:
    a = df.loc[df[group_col] == label_a, metric].dropna()
    b = df.loc[df[group_col] == label_b, metric].dropna()
    stat, pval = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
    sig = pval < ALPHA
    test_name = f"T-Test | {metric} | {group_col} ({label_a} vs {label_b})"
    interp = (
        f"Mean {metric}: {label_a}={a.mean():.3f}, {label_b}={b.mean():.3f}. "
        + ("Groups differ significantly." if sig else "No significant difference.")
    )
    return TestResult(test_name, float(stat), float(pval), sig, interp)


def _anova(df: pd.DataFrame, metric: str, group_col: str) -> TestResult:
    groups = [grp[metric].dropna().values for _, grp in df.groupby(group_col)]
    stat, pval = stats.f_oneway(*groups)
    sig = pval < ALPHA
    test_name = f"ANOVA  | {metric} | by {group_col}"
    interp = (
        f"{'At least one group mean differs significantly.' if sig else 'No significant difference between groups.'}"
    )
    return TestResult(test_name, float(stat), float(pval), sig, interp)


def _interaction_anova(df: pd.DataFrame, metric: str) -> TestResult:
    """Four-group ANOVA across the 2×2 interaction of online_order × book_table."""
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for ol in ["Yes", "No"]:
        for bt in ["Yes", "No"]:
            mask = (df["online_order"] == ol) & (df["book_table"] == bt)
            g = df.loc[mask, metric].dropna().values
            if len(g) > 1:
                groups.append(g)
                labels.append(f"OL={ol}/BT={bt}")
    stat, pval = stats.f_oneway(*groups)
    sig = pval < ALPHA
    test_name = f"ANOVA  | {metric} | online_order × book_table (4 groups)"
    n_desc = ", ".join(f"{l}:n={len(g)}" for l, g in zip(labels, groups))
    interp = (
        f"Groups: {n_desc}. "
        + ("Significant interaction effect." if sig else "No significant interaction.")
    )
    return TestResult(test_name, float(stat), float(pval), sig, interp)


def run_all_tests(parquet_path: str = PARQUET_PATH) -> dict[str, TestResult]:
    df = _load(parquet_path)

    results: dict[str, TestResult] = {}

    # --- T-Tests ---
    for metric in ["rate", "votes"]:
        for group_col in ["online_order", "book_table"]:
            r = _ttest(df, metric, group_col)
            results[r.test_name] = r

    # --- One-way ANOVA by online_order and book_table ---
    for metric in ["rate", "votes"]:
        r = _anova(df, metric, "online_order")
        results[r.test_name] = r
        r = _anova(df, metric, "book_table")
        results[r.test_name] = r

    # --- 2×2 Interaction ANOVA ---
    for metric in ["rate", "votes"]:
        r = _interaction_anova(df, metric)
        results[r.test_name] = r

    return results


def results_to_dataframe(results: dict[str, TestResult]) -> pd.DataFrame:
    rows = []
    for r in results.values():
        rows.append({
            "test": r.test_name,
            "statistic": round(r.statistic, 4),
            "p_value": round(r.pvalue, 4),
            "significant": r.significant,
            "interpretation": r.interpretation,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run_all_tests()
    for r in res.values():
        print(r)
        print()
