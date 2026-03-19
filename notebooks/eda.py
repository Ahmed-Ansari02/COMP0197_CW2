"""
Member B — Exploratory Data Analysis & Quality Assurance
=========================================================

Run as a script or import into a Jupyter notebook.
Produces diagnostic plots and summary statistics for all structural features.

Usage:
    python notebooks/member_b_eda.py --parquet data/intermediate/structural_features.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map missingness over time AND across countries.

    Shared exploration checklist item 1: conflict data is often
    missing-not-at-random (MNAR). Data disappearance itself may
    signal state collapse, media blackout, or institutional failure.

    Returns a summary DataFrame.
    """
    # Overall missingness by feature
    overall = df.drop(columns=["gwcode", "year_month"]).isnull().mean().sort_values(ascending=False)
    print("\n=== OVERALL MISSINGNESS ===")
    for col, pct in overall.items():
        if pct > 0:
            print(f"  {col}: {pct:.1%}")

    # Missingness by year (temporal pattern)
    df_temp = df.copy()
    df_temp["year"] = df_temp["year_month"].str[:4].astype(int)
    yearly_missing = df_temp.groupby("year").apply(
        lambda g: g.drop(columns=["gwcode", "year_month", "year"]).isnull().mean()
    )
    print("\n=== MISSINGNESS BY YEAR (selected features) ===")
    check_cols = [c for c in ["v2x_libdem", "tenure_months", "fx_volatility",
                               "food_price_anomaly", "gdp_growth"] if c in yearly_missing.columns]
    if check_cols:
        print(yearly_missing[check_cols].round(3).to_string())

    # Countries with highest missingness
    country_missing = df.groupby("gwcode").apply(
        lambda g: g.drop(columns=["gwcode", "year_month"]).isnull().mean().mean()
    ).sort_values(ascending=False)
    print("\n=== TOP 20 COUNTRIES BY MISSINGNESS ===")
    print(country_missing.head(20).round(3).to_string())

    return overall


def distribution_profiling(df: pd.DataFrame) -> None:
    """
    Shared exploration checklist item 3: profile feature distributions.

    Checks for:
    - Zero-inflation (economic features in stable countries)
    - Heavy tails (conflict-adjacent features)
    - Candidates for log-transformation
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {"gwcode"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    print("\n=== DISTRIBUTION PROFILES ===")
    print(f"{'Feature':<30} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} {'%Zero':>8} {'%NaN':>8}")
    print("-" * 90)

    candidates_for_log = []

    for col in sorted(numeric_cols):
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        mean = vals.mean()
        std = vals.std()
        skew = vals.skew()
        kurt = vals.kurtosis()
        pct_zero = (vals == 0).mean() * 100
        pct_nan = df[col].isna().mean() * 100

        print(f"  {col:<28} {mean:>8.3f} {std:>8.3f} {skew:>8.2f} {kurt:>8.1f} {pct_zero:>7.1f}% {pct_nan:>7.1f}%")

        # Flag log-transform candidates: positive, skewness > 2
        if skew > 2 and vals.min() >= 0:
            candidates_for_log.append(col)

    if candidates_for_log:
        print(f"\n  Log-transform candidates (skew > 2, non-negative): {candidates_for_log}")


def temporal_alignment_check(df: pd.DataFrame) -> None:
    """
    Shared exploration checklist item 2: verify temporal alignment.

    Checks that:
    - V-Dem features are constant within each year (as expected for annual data)
    - REIGN features can change monthly
    - Economic features show expected monthly variation
    """
    print("\n=== TEMPORAL ALIGNMENT CHECK ===")

    df_temp = df.copy()
    df_temp["year"] = df_temp["year_month"].str[:4].astype(int)

    # V-Dem: should be constant within each (country, year) after expansion
    vdem_cols = [c for c in df.columns if c.startswith("v2x_")]
    if vdem_cols:
        # After lag, values within a year should be nearly constant
        # (except at year boundaries where the lag shifts)
        vdem_std = df_temp.groupby(["gwcode", "year"])[vdem_cols[:3]].std().mean()
        print(f"  V-Dem within-year std (should be ~0 except at boundaries):")
        for col in vdem_cols[:3]:
            if col in vdem_std.index:
                print(f"    {col}: {vdem_std[col]:.4f}")

    # Economic: should show monthly variation
    econ_cols = [c for c in ["fx_volatility", "fx_pct_change"] if c in df.columns]
    if econ_cols:
        monthly_var = df.groupby("gwcode")[econ_cols].std().mean()
        print(f"  Economic monthly variation (should be > 0):")
        for col in econ_cols:
            if col in monthly_var.index:
                print(f"    {col}: {monthly_var[col]:.4f}")


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature selection Stage 1: identify highly correlated feature pairs
    (Spearman ρ > 0.9) for potential removal.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = {"gwcode"}
    numeric_cols = [c for c in numeric_cols if c not in exclude and not c.endswith("_available")]

    if len(numeric_cols) < 2:
        print("\n  Not enough numeric features for correlation analysis")
        return pd.DataFrame()

    corr = df[numeric_cols].corr(method="spearman")

    # Find highly correlated pairs
    print("\n=== HIGHLY CORRELATED PAIRS (|ρ| > 0.9) ===")
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            rho = corr.iloc[i, j]
            if abs(rho) > 0.9:
                pairs.append((numeric_cols[i], numeric_cols[j], rho))
                print(f"  {numeric_cols[i]} ↔ {numeric_cols[j]}: ρ = {rho:.3f}")

    if not pairs:
        print("  None found (good)")

    return corr


def reign_gap_analysis(df: pd.DataFrame) -> None:
    """Analyse the impact of the REIGN Aug 2021 cutoff."""
    print("\n=== REIGN COVERAGE GAP ANALYSIS ===")

    reign_cols = ["tenure_months", "age", "coup_event", "regime_change"]
    available = [c for c in reign_cols if c in df.columns]

    if not available:
        print("  No REIGN features found")
        return

    pre = df[df["year_month"] <= "2021-08"]
    post = df[df["year_month"] > "2021-08"]

    for col in available:
        pre_cov = pre[col].notna().mean() * 100
        post_cov = post[col].notna().mean() * 100
        print(f"  {col}: pre-cutoff={pre_cov:.1f}%, post-cutoff={post_cov:.1f}%")


def run_full_eda(parquet_path: str | Path) -> None:
    """Run complete EDA suite."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        print(f"Parquet file not found: {parquet_path}")
        print("Run the pipeline first: python run_member_b.py")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Countries: {df['gwcode'].nunique()}, Months: {df['year_month'].nunique()}")
    print(f"Period: {df['year_month'].min()} to {df['year_month'].max()}")

    missingness_audit(df)
    distribution_profiling(df)
    temporal_alignment_check(df)
    correlation_analysis(df)
    reign_gap_analysis(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", type=str,
        default="data/intermediate/structural_features.parquet",
    )
    args = parser.parse_args()
    run_full_eda(args.parquet)
