"""
V-Dem Ingestion & Feature Engineering
======================================

Ingests the Varieties of Democracy (V-Dem) Country-Year dataset,
selects conflict-relevant governance indicators, expands annual
observations to monthly resolution (repeat, NOT interpolate),
maps to Gleditsch-Ward codes, and derives composite indices for
the ViEWS fatality prediction pipeline.

Design decisions:
    1. Annual → monthly via repetition, not interpolation. V-Dem is
       expert-coded once per year; interpolation would fabricate
       sub-annual variation that does not exist in the underlying
       measurement process.
    2. Regime type (v2x_regime ∈ {0,1,2,3}) is one-hot encoded to
       avoid imposing ordinal assumptions on the transformer encoder.
    3. Year-on-year democracy change (Δ liberal democracy index) is
       computed to capture democratic backsliding / transition dynamics.
    4. No normalisation — indices are already on [0, 1] by construction.
       Member C is informed via the feature registry.

Reference:
    Coppedge, M. et al. (2024). V-Dem Dataset v15.
    Varieties of Democracy Institute, University of Gothenburg.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.crosswalk import ISO3_TO_GW

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Column specifications
# ──────────────────────────────────────────────────────────────────────
VDEM_ID_COLS = ["country_text_id", "year"]

VDEM_FEATURES = [
    "v2x_libdem",          # Liberal democracy index [0, 1]
    "v2x_polyarchy",       # Electoral democracy index (Polity analog) [0, 1]
    "v2x_civlib",          # Civil liberties index [0, 1]
    "v2x_rule",            # Rule of law index [0, 1]
    "v2x_corr",            # Political corruption index [0, 1] (higher = more corrupt)
    "v2x_clphy",           # Physical integrity / freedom from state violence [0, 1]
    "v2x_clpol",           # Political civil liberties [0, 1]
    "v2x_freexp_altinf",   # Freedom of expression & alternative info sources [0, 1]
    "v2xcs_ccsi",          # Core civil society index [0, 1]
    "v2x_regime",          # Regime classification: 0=closed autoc, 1=electoral autoc,
                           #   2=electoral dem, 3=liberal dem
    "v2x_partipdem",       # Participatory democracy index [0, 1]
    "v2xnp_regcorr",       # Regime corruption [0, 1]
    "v2x_execorr",         # Executive corruption [0, 1]
    "v2x_frassoc_thick",   # Freedom of association (thick) [0, 1]
]

# Study period: 2009 included for forward-fill buffer into Jan 2010
YEAR_MIN = 2009
YEAR_MAX = 2024


def load_vdem(
    csv_path: str | Path,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> pd.DataFrame:
    """
    Load V-Dem CSV with minimal column selection and temporal filtering.

    Parameters
    ----------
    csv_path : path to V-Dem-CY-Full+Others-v15.csv (or equivalent)
    year_min : earliest year to retain (inclusive)
    year_max : latest year to retain (inclusive)

    Returns
    -------
    DataFrame with columns: country_text_id, year, + VDEM_FEATURES
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"V-Dem CSV not found: {csv_path}")

    logger.info("Loading V-Dem from %s (years %d–%d)", csv_path, year_min, year_max)

    # Only read the columns we need — V-Dem full file has 4000+ columns
    vdem = pd.read_csv(
        csv_path,
        usecols=VDEM_ID_COLS + VDEM_FEATURES,
        low_memory=False,
    )

    # Temporal filter
    vdem = vdem[(vdem["year"] >= year_min) & (vdem["year"] <= year_max)].copy()
    logger.info("V-Dem loaded: %d country-years", len(vdem))

    return vdem


def _expand_annual_to_monthly(vdem: pd.DataFrame) -> pd.DataFrame:
    """
    Expand annual V-Dem observations to monthly by repeating each year's
    values across all 12 months. This is NOT interpolation — it preserves
    the actual measurement cadence of the expert coding process.

    The resulting year_month column uses YYYY-MM string format for
    consistent panel indexing.
    """
    # Vectorised expansion: cross-join each row with months 1–12
    months = pd.DataFrame({"month": range(1, 13)})
    vdem = vdem.assign(_key=1)
    months = months.assign(_key=1)
    expanded = vdem.merge(months, on="_key").drop(columns="_key")

    # Create canonical year_month index
    expanded["year_month"] = (
        expanded["year"].astype(str) + "-" + expanded["month"].astype(str).str.zfill(2)
    )
    expanded.drop(columns=["month"], inplace=True)

    logger.info("Expanded to %d country-months", len(expanded))
    return expanded


def _map_to_gwcodes(vdem_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Map V-Dem's country_text_id (ISO3) to Gleditsch-Ward numeric codes.
    Logs unmapped countries for manual resolution.
    """
    vdem_monthly["gwcode"] = vdem_monthly["country_text_id"].map(ISO3_TO_GW)

    unmapped = vdem_monthly[vdem_monthly["gwcode"].isna()]["country_text_id"].unique()
    if len(unmapped) > 0:
        logger.warning(
            "V-Dem countries with no GW mapping (%d): %s",
            len(unmapped),
            ", ".join(sorted(unmapped)),
        )
        # Drop unmapped — these are typically historical/dissolved entities
        vdem_monthly = vdem_monthly.dropna(subset=["gwcode"]).copy()

    vdem_monthly["gwcode"] = vdem_monthly["gwcode"].astype(int)
    return vdem_monthly


def _encode_regime_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode v2x_regime (0–3) into four binary columns.
    This avoids imposing ordinal assumptions on the regime
    classification when fed to the transformer encoder.
    """
    regime_dummies = pd.get_dummies(
        df["v2x_regime"].astype("Int64"),  # nullable int handles NaN
        prefix="regime_type",
    ).astype(int)

    # Ensure all four categories exist even if some are absent in the data
    for i in range(4):
        col = f"regime_type_{i}"
        if col not in regime_dummies.columns:
            regime_dummies[col] = 0

    df = pd.concat([df, regime_dummies], axis=1)
    return df


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive composite governance indices from raw V-Dem scores.

    Features:
        governance_deficit: 1 − mean(libdem, polyarchy, rule_of_law)
            Higher values indicate weaker institutional quality.
            Theoretically grounded in Hegre et al. (2001): weak institutions
            create permissive conditions for organised violence.

        repression_index: mean(corruption, exec_corruption) + (1 − physical_integrity) / 2
            Captures the coercive capacity of the state and incentive
            structures for grievance-based mobilisation (Gurr, 1970).

        libdem_yoy_change: Δ in liberal democracy index (12-month diff)
            Detects democratic backsliding or rapid democratisation,
            both of which are associated with elevated conflict risk
            (Mansfield & Snyder, 2005).
    """
    # Governance deficit: inverse of institutional quality
    df["governance_deficit"] = 1.0 - df[["v2x_libdem", "v2x_polyarchy", "v2x_rule"]].mean(axis=1)

    # Repression index: corruption + physical integrity violations
    df["repression_index"] = (
        df[["v2x_corr", "v2x_execorr"]].mean(axis=1) + (1.0 - df["v2x_clphy"])
    ) / 2.0

    # Year-on-year change in liberal democracy (12-month diff since monthly)
    df = df.sort_values(["gwcode", "year_month"])
    df["libdem_yoy_change"] = df.groupby("gwcode")["v2x_libdem"].diff(12)

    return df


def _validate_ranges(df: pd.DataFrame) -> None:
    """Assert that all V-Dem indices remain in their valid [0, 1] range."""
    bounded_cols = [c for c in VDEM_FEATURES if c != "v2x_regime"]
    for col in bounded_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        out_of_range = ~vals.between(-0.001, 1.001)
        if out_of_range.any():
            n_bad = out_of_range.sum()
            logger.error("%s: %d values outside [0, 1]", col, n_bad)
            raise ValueError(f"{col} has {n_bad} values outside [0, 1]")

    # Derived features approximate range check
    for col in ["governance_deficit", "repression_index"]:
        vals = df[col].dropna()
        if len(vals) > 0 and not vals.between(-0.1, 1.5).all():
            logger.warning("%s has values outside expected range", col)

    logger.info("V-Dem range validation passed")


def _flag_large_democracy_shifts(df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
    """
    Flag country-months where the year-on-year change in liberal democracy
    exceeds a threshold. These may indicate coups, revolutions, or data
    artefacts and warrant manual review.
    """
    large = df[df["libdem_yoy_change"].abs() > threshold].copy()
    if len(large) > 0:
        logger.warning(
            "%d country-months with |Δlibdem| > %.2f:",
            len(large),
            threshold,
        )
        for _, row in large.head(10).iterrows():
            logger.warning(
                "  GW %d, %s: libdem=%.3f, Δ=%.3f",
                row["gwcode"], row["year_month"],
                row["v2x_libdem"], row["libdem_yoy_change"],
            )
    return df


def ingest_vdem(csv_path: str | Path) -> pd.DataFrame:
    """
    Full V-Dem ingestion pipeline: load → expand → map → encode → derive → validate.

    Parameters
    ----------
    csv_path : path to V-Dem CSV

    Returns
    -------
    DataFrame indexed by (gwcode, year_month) with all raw + derived V-Dem features.
    Lag is NOT applied here — that is done in structural_features.py.
    """
    logger.info("=" * 60)
    logger.info("V-Dem ingestion pipeline starting")
    logger.info("=" * 60)

    # 1. Load
    vdem = load_vdem(csv_path)

    # 2. Expand annual → monthly (repeat, not interpolate)
    vdem_monthly = _expand_annual_to_monthly(vdem)

    # 3. Map to GW codes
    vdem_monthly = _map_to_gwcodes(vdem_monthly)

    # 4. Forward-fill for years where V-Dem data is not yet available
    #    (e.g., 2024 if V-Dem v15 covers only through 2023)
    max_year_with_data = vdem_monthly.dropna(subset=["v2x_libdem"])["year"].max()
    if max_year_with_data < YEAR_MAX:
        logger.info(
            "V-Dem data ends at %d; forward-filling to %d",
            max_year_with_data, YEAR_MAX,
        )
        vdem_monthly = vdem_monthly.sort_values(["gwcode", "year_month"])
        feature_cols = [c for c in VDEM_FEATURES if c in vdem_monthly.columns]
        vdem_monthly[feature_cols] = vdem_monthly.groupby("gwcode")[feature_cols].ffill()

    # 5. One-hot encode regime type
    vdem_monthly = _encode_regime_type(vdem_monthly)

    # 6. Derive composite features
    vdem_monthly = _derive_features(vdem_monthly)

    # 7. Validate ranges
    _validate_ranges(vdem_monthly)
    vdem_monthly = _flag_large_democracy_shifts(vdem_monthly)

    # 8. Filter to study period (drop 2009 buffer year)
    vdem_monthly = vdem_monthly[vdem_monthly["year_month"] >= "2010-01"].copy()

    # 9. Clean up columns
    drop_cols = ["country_text_id", "year"]
    vdem_monthly.drop(columns=[c for c in drop_cols if c in vdem_monthly.columns], inplace=True)

    logger.info(
        "V-Dem ingestion complete: %d rows, %d columns, %d countries",
        len(vdem_monthly),
        len(vdem_monthly.columns),
        vdem_monthly["gwcode"].nunique(),
    )

    return vdem_monthly
