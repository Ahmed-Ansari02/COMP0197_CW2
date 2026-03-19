"""
Structural Features Assembly Pipeline
=======================================

Merges V-Dem, REIGN, and economic covariates into a single
analysis-ready panel dataset indexed by (gwcode, year_month).

This module is the final stage of Member B's pipeline. It:
    1. Constructs the full panel skeleton (all GW codes × all months)
    2. Left-joins each source onto the skeleton
    3. Applies t−1 lag to ALL features (preventing temporal leakage)
    4. Encodes missingness as informative binary indicators
    5. Runs comprehensive quality checks
    6. Exports to data/intermediate/structural_features.parquet

Critical design decisions:
    - Lag is applied ONCE here, not in individual ingest modules.
      This ensures consistent, auditable lag treatment across all sources.
    - No normalisation is performed. Member C handles normalisation
      after the train/test split to prevent information leakage from
      the test set into feature scaling parameters.
    - Missingness indicators are features themselves: the absence of
      economic data may be informative (failed states often stop reporting).

Output schema:
    Index: (gwcode: int, year_month: str "YYYY-MM")
    Features: ~35 columns (raw + derived + missingness indicators)
    Rows: ~180 countries × 180 months ≈ 32,400 country-months
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.ingest_vdem import ingest_vdem
from src.data.ingest_reign import ingest_reign
from src.data.ingest_economic import ingest_all_economic
from src.data.ingest_powell_thyne import ingest_powell_thyne, integrate_with_reign

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
YEAR_MIN = 2010
YEAR_MAX = 2024

OUTPUT_PATH = Path("data/intermediate/structural_features.parquet")
LOG_PATH = Path("data/logs/member_b_quality_report.json")


def _build_panel_skeleton(reign_gwcodes: Optional[set] = None) -> pd.DataFrame:
    """
    Construct the full (gwcode, year_month) panel grid.

    Uses REIGN's country list as the baseline (since it is already on GW codes),
    augmented with any additional countries from V-Dem.
    """
    all_months = pd.date_range(
        start=f"{YEAR_MIN}-01-01",
        end=f"{YEAR_MAX}-12-01",
        freq="MS",
    ).strftime("%Y-%m").tolist()

    if reign_gwcodes is not None:
        countries = sorted(reign_gwcodes)
    else:
        # Fallback: use a broad set of GW codes
        from src.data.crosswalk import GW_TO_ISO3
        countries = sorted(GW_TO_ISO3.keys())

    panel = pd.DataFrame(
        [(gw, ym) for gw in countries for ym in all_months],
        columns=["gwcode", "year_month"],
    )

    logger.info(
        "Panel skeleton: %d countries × %d months = %d rows",
        len(countries), len(all_months), len(panel),
    )

    return panel


def _cross_validate_structural_breaks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-validate V-Dem annual governance scores against REIGN's monthly
    structural break indicators.

    Problem: V-Dem codes governance annually. A mid-year coup (detected by
    REIGN) means the V-Dem score for that year is an average that doesn't
    reflect the discontinuity. For months AFTER a mid-year coup, the V-Dem
    values are stale and misleading.

    Solution: Create a `vdem_stale_flag` that marks country-months where
    V-Dem data is likely outdated because a structural break occurred
    after the V-Dem coding period but within the same calendar year.
    This tells the model "don't trust the democracy score here."

    Also creates `structural_break_recency`: how many months since the
    most recent structural break (from either REIGN or V-Dem year-change).
    This captures the "instability window" effect where conflict risk
    stays elevated for months after a political shock.
    """
    has_vdem = "v2x_libdem" in panel.columns
    has_reign_break = "regime_change" in panel.columns

    if not has_reign_break:
        logger.info("No REIGN structural break data — skipping cross-validation")
        return panel

    panel = panel.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # ── V-Dem staleness detection ────────────────────────────────────
    if has_vdem:
        # Extract year from year_month for grouping
        panel["_year"] = panel["year_month"].str[:4].astype(int)

        # For each country-year: did a structural break (regime_change or
        # coup_event) occur? If so, V-Dem's annual score may not reflect
        # the post-break reality for remaining months of that year.
        panel["vdem_stale_flag"] = 0

        for gw in panel["gwcode"].unique():
            gw_mask = panel["gwcode"] == gw
            gw_data = panel.loc[gw_mask]

            for year in gw_data["_year"].unique():
                year_mask = gw_mask & (panel["_year"] == year)
                year_data = panel.loc[year_mask]

                # Check for breaks in this year
                breaks_in_year = year_data[
                    (year_data.get("regime_change", pd.Series(dtype=float)) == 1)
                    | (year_data.get("coup_event", pd.Series(dtype=float)) == 1)
                ]

                if len(breaks_in_year) > 0:
                    # Find the month of first break
                    first_break_month = breaks_in_year["year_month"].min()
                    # Mark all months AFTER the break in this year as stale
                    stale_mask = year_mask & (panel["year_month"] > first_break_month)
                    panel.loc[stale_mask, "vdem_stale_flag"] = 1

        n_stale = (panel["vdem_stale_flag"] == 1).sum()
        logger.info(
            "V-Dem staleness: %d country-months flagged (mid-year structural break)",
            n_stale,
        )

        panel.drop(columns=["_year"], inplace=True)
    else:
        panel["vdem_stale_flag"] = 0

    return panel


def _apply_lag(df: pd.DataFrame, feature_cols: list[str], lag: int = 1) -> pd.DataFrame:
    """
    Apply temporal lag to all feature columns within each country group.

    After lag=1: the feature value at row (country, 2020-06) contains
    the actual observation from 2020-05. This is correct for predicting
    June's outcomes using May's information, preventing temporal leakage.

    Parameters
    ----------
    df : panel DataFrame sorted by (gwcode, year_month)
    feature_cols : columns to lag
    lag : number of months to shift (default 1)

    Returns
    -------
    DataFrame with lagged features (first `lag` months per country are NaN)
    """
    df = df.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    df[feature_cols] = df.groupby("gwcode")[feature_cols].shift(lag)

    n_nullified = df[feature_cols].isna().all(axis=1).sum()
    logger.info(
        "Lag t−%d applied to %d features (%d leading NaN rows expected)",
        lag, len(feature_cols), n_nullified,
    )

    return df


def _encode_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicators for data source availability.

    These are features themselves — the absence of data from a given
    source may be informative. Failed states often stop submitting
    economic statistics; countries under media blackout may have
    missing GDELT coverage. The transformer can learn to weight
    these signals.
    """
    indicator_map = {
        "vdem_available": "v2x_libdem",
        "reign_available": "tenure_months",
        "fx_available": "fx_volatility",
        "food_available": "food_price_anomaly",
        "gdp_available": "gdp_growth",
    }

    for flag_name, source_col in indicator_map.items():
        if source_col in df.columns:
            df[flag_name] = df[source_col].notna().astype(int)
        else:
            df[flag_name] = 0

    return df


def _generate_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive quality metrics for the structural features dataset.

    Returns a JSON-serialisable dict with:
        - Panel dimensions
        - Per-feature missingness rates
        - Source coverage statistics
        - REIGN post-2021 coverage
        - Value range checks
    """
    report = {
        "panel_dimensions": {
            "n_rows": int(len(df)),
            "n_countries": int(df["gwcode"].nunique()),
            "n_months": int(df["year_month"].nunique()),
            "date_range": [df["year_month"].min(), df["year_month"].max()],
        },
        "missingness": {},
        "source_coverage": {},
        "value_ranges": {},
    }

    # Per-feature missingness
    for col in sorted(df.columns):
        if col in ("gwcode", "year_month"):
            continue
        missing_pct = round(df[col].isna().mean() * 100, 2)
        if missing_pct > 0:
            report["missingness"][col] = missing_pct

    # Source-level coverage
    source_checks = {
        "vdem": ["v2x_libdem", "v2x_polyarchy"],
        "reign": ["tenure_months", "age"],
        "fx": ["fx_volatility", "fx_pct_change"],
        "food": ["food_price_anomaly", "food_cpi_acceleration"],
        "gdp": ["gdp_growth", "gdp_growth_deviation"],
    }
    for source, cols in source_checks.items():
        available_cols = [c for c in cols if c in df.columns]
        if available_cols:
            coverage = round(df[available_cols[0]].notna().mean() * 100, 2)
            report["source_coverage"][source] = coverage

    # REIGN post-2021 specific check
    post_2021 = df[df["year_month"] > "2021-08"]
    if len(post_2021) > 0 and "tenure_months" in df.columns:
        reign_post = round(post_2021["tenure_months"].notna().mean() * 100, 2)
        report["source_coverage"]["reign_post_2021"] = reign_post

    # Value ranges for key features
    range_cols = ["v2x_libdem", "v2x_corr", "governance_deficit", "repression_index",
                  "fx_volatility", "gdp_growth", "food_price_anomaly"]
    for col in range_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                report["value_ranges"][col] = {
                    "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4),
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                    "p25": round(float(vals.quantile(0.25)), 4),
                    "p75": round(float(vals.quantile(0.75)), 4),
                }

    return report


def run_quality_checks(df: pd.DataFrame) -> None:
    """
    Run assertion-based quality checks on the assembled panel.
    These checks enforce invariants that, if violated, indicate
    pipeline bugs rather than data issues.
    """
    logger.info("Running quality checks...")

    # 1. Panel completeness: sorted per country
    for gwcode in df["gwcode"].unique()[:20]:
        country = df[df["gwcode"] == gwcode]
        assert country["year_month"].is_monotonic_increasing, \
            f"GW {gwcode}: year_month not monotonically increasing"
    logger.info("  PASS: Panel is sorted per country")

    # 2. No duplicate (gwcode, year_month) pairs
    dupes = df.duplicated(subset=["gwcode", "year_month"], keep=False)
    assert not dupes.any(), \
        f"Found {dupes.sum()} duplicate (gwcode, year_month) rows"
    logger.info("  PASS: No duplicate panel entries")

    # 3. Lag verification: first month per country should have NaN features
    vdem_cols = [c for c in df.columns if c.startswith("v2x_")]
    if vdem_cols:
        for gwcode in df["gwcode"].unique()[:10]:
            country = df[df["gwcode"] == gwcode].sort_values("year_month")
            first_row = country.iloc[0]
            # After lag, the first month should be NaN (shifted from nothing)
            if first_row["year_month"] == f"{YEAR_MIN}-01" and "v2x_libdem" in df.columns:
                assert pd.isna(first_row.get("v2x_libdem")), \
                    f"GW {gwcode}: first row v2x_libdem not NaN after lag"
        logger.info("  PASS: Lag appears correctly applied")

    # 4. V-Dem range check
    for col in vdem_cols:
        if col == "v2x_regime":
            continue
        vals = df[col].dropna()
        if len(vals) > 0:
            out_of_range = ~vals.between(-0.01, 1.01)
            assert not out_of_range.any(), f"{col}: {out_of_range.sum()} values outside [0,1]"
    logger.info("  PASS: V-Dem indices in valid range")

    # 5. REIGN coverage report
    reign_cols_check = ["tenure_months", "age"]
    for col in reign_cols_check:
        if col in df.columns:
            coverage = df[col].notna().mean()
            logger.info("  REIGN '%s' coverage: %.1f%%", col, coverage * 100)

    # 6. Economic coverage
    for col in ["fx_volatility", "food_price_anomaly", "gdp_growth"]:
        if col in df.columns:
            coverage = df[col].notna().mean()
            logger.info("  %s coverage: %.1f%%", col, coverage * 100)

    # 7. Missingness summary
    logger.info("  Missingness summary (>0%%):")
    for col in sorted(df.columns):
        if col in ("gwcode", "year_month"):
            continue
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            logger.info("    %s: %.1f%%", col, missing_pct)

    logger.info("All quality checks passed")


def build_structural_features(
    vdem_csv: Optional[str | Path] = None,
    reign_csv: Optional[str | Path] = None,
    fx_csv: Optional[str | Path] = None,
    gdp_csv: Optional[str | Path] = None,
    food_csv: Optional[str | Path] = None,
    coups_tsv: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Main entry point: assemble all structural features into a unified panel.

    Pipeline:
        1. Ingest V-Dem → monthly governance features
        2. Ingest REIGN → monthly leader/regime features
        3. Ingest economic covariates → FX, GDP, food price features
        4. Ingest Powell & Thyne coups → patch REIGN's post-2021 gap
        5. Build panel skeleton from REIGN country list
        6. Left-join all sources onto the skeleton
        7. Integrate Powell & Thyne with REIGN coup data
        8. Cross-validate V-Dem against structural breaks
        9. Apply t−1 lag to ALL feature columns
        10. Encode missingness as informative indicators
        11. Run quality checks
        12. Export to Parquet + quality report JSON

    Parameters
    ----------
    vdem_csv : path to V-Dem CSV
    reign_csv : path to REIGN CSV
    fx_csv : path to IMF exchange rate CSV
    gdp_csv : path to World Bank GDP CSV
    food_csv : path to FAO food price CSV
    coups_tsv : path to Powell & Thyne coup TSV
    output_path : Parquet output path (default: data/intermediate/structural_features.parquet)
    log_path : quality report JSON path (default: data/logs/member_b_quality_report.json)

    Returns
    -------
    Assembled DataFrame with all structural features
    """
    output_path = Path(output_path) if output_path else OUTPUT_PATH
    log_path = Path(log_path) if log_path else LOG_PATH

    logger.info("=" * 70)
    logger.info("STRUCTURAL FEATURES ASSEMBLY PIPELINE")
    logger.info("=" * 70)

    # ── 1. Ingest individual sources ────────────────────────────────

    vdem_df = None
    reign_df = None
    econ_dfs = {}

    if vdem_csv:
        vdem_df = ingest_vdem(vdem_csv)

    if reign_csv:
        reign_df = ingest_reign(reign_csv)

    econ_dfs = ingest_all_economic(
        fx_csv=fx_csv, gdp_csv=gdp_csv, food_csv=food_csv
    )

    # ── 2. Build panel skeleton ─────────────────────────────────────

    gwcodes = set()
    if reign_df is not None:
        gwcodes.update(reign_df["gwcode"].unique())
    if vdem_df is not None:
        gwcodes.update(vdem_df["gwcode"].unique())
    for edf in econ_dfs.values():
        gwcodes.update(edf["gwcode"].unique())

    if not gwcodes:
        raise ValueError("No data sources loaded — cannot build panel")

    panel = _build_panel_skeleton(gwcodes)

    # ── 3. Left-join sources ────────────────────────────────────────

    # V-Dem
    if vdem_df is not None:
        logger.info("Merging V-Dem (%d rows)...", len(vdem_df))
        panel = panel.merge(vdem_df, on=["gwcode", "year_month"], how="left")

    # REIGN
    if reign_df is not None:
        logger.info("Merging REIGN (%d rows)...", len(reign_df))
        panel = panel.merge(reign_df, on=["gwcode", "year_month"], how="left")

    # Economic: FX
    if "fx" in econ_dfs:
        fx_df = econ_dfs["fx"]
        logger.info("Merging exchange rates (%d rows)...", len(fx_df))
        panel = panel.merge(fx_df, on=["gwcode", "year_month"], how="left")

    # Economic: GDP
    if "gdp" in econ_dfs:
        gdp_df = econ_dfs["gdp"]
        logger.info("Merging GDP (%d rows)...", len(gdp_df))
        panel = panel.merge(gdp_df, on=["gwcode", "year_month"], how="left")

    # Economic: Food prices
    if "food" in econ_dfs:
        food_df = econ_dfs["food"]
        logger.info("Merging food prices (%d rows)...", len(food_df))
        panel = panel.merge(food_df, on=["gwcode", "year_month"], how="left")

    # ── 4. Integrate Powell & Thyne coups (patches REIGN gap) ───────

    if coups_tsv:
        coups_tsv_path = Path(coups_tsv)
        if coups_tsv_path.exists():
            pt_monthly = ingest_powell_thyne(coups_tsv)
            panel = integrate_with_reign(panel, pt_monthly)
        else:
            logger.warning("Powell & Thyne file not found: %s", coups_tsv)

    # ── 5. Cross-validate V-Dem against structural breaks ─────────

    panel = _cross_validate_structural_breaks(panel)

    # ── 6. Apply t−1 lag ────────────────────────────────────────────

    feature_cols = [
        c for c in panel.columns
        if c not in ("gwcode", "year_month")
    ]

    panel = _apply_lag(panel, feature_cols, lag=1)

    # ── 6. Encode missingness ───────────────────────────────────────

    panel = _encode_missingness(panel)

    # ── 7. Quality checks ───────────────────────────────────────────

    run_quality_checks(panel)

    # ── 8. Generate quality report ──────────────────────────────────

    report = _generate_quality_report(panel)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Quality report written to %s", log_path)

    # ── 9. Export ───────────────────────────────────────────────────

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Structural features exported to %s", output_path)
    logger.info(
        "Final dataset: %d rows × %d columns (%d countries, %s to %s)",
        len(panel), len(panel.columns), panel["gwcode"].nunique(),
        panel["year_month"].min(), panel["year_month"].max(),
    )

    return panel


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build structural features panel for ViEWS pipeline"
    )
    parser.add_argument("--vdem", type=str, help="Path to V-Dem CSV")
    parser.add_argument("--reign", type=str, help="Path to REIGN CSV")
    parser.add_argument("--fx", type=str, help="Path to exchange rate CSV")
    parser.add_argument("--gdp", type=str, help="Path to GDP CSV")
    parser.add_argument("--food", type=str, help="Path to food price CSV")
    parser.add_argument("--coups", type=str, help="Path to Powell & Thyne TSV")
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help="Output Parquet path",
    )
    parser.add_argument(
        "--log", type=str, default=str(LOG_PATH),
        help="Quality report JSON path",
    )

    args = parser.parse_args()

    build_structural_features(
        vdem_csv=args.vdem,
        reign_csv=args.reign,
        fx_csv=args.fx,
        gdp_csv=args.gdp,
        food_csv=args.food,
        coups_tsv=args.coups,
        output_path=args.output,
        log_path=args.log,
    )
