"""
REIGN Ingestion & Feature Engineering
======================================

Ingests the Rulers, Elections, and Irregular Governance (REIGN) dataset,
which provides monthly leader-level characteristics, regime classifications,
election timing, and irregular power transitions for all sovereign states.

Critical constraint:
    REIGN data collection ceased in August 2021. The pipeline handles this
    via Option A (recommended in the project spec): forward-fill slow-moving
    features (regime type, leader demographics) and mark fast-changing event
    features (coups, irregular transitions) as NaN post-cutoff.

Design decisions:
    1. REIGN uses native GW codes (ccode) — no crosswalk mapping required.
    2. Regime type strings are one-hot encoded for the transformer.
    3. Structural break detection: regime changes and coups are flagged as
       binary indicators, as these invalidate forward-fill assumptions.
    4. Leader vulnerability is encoded as a binary risk flag based on
       empirical age distributions of leaders during instability events.

Reference:
    Bell, C. (2016). The Rulers, Elections, and Irregular Governance Dataset.
    One Earth Future Foundation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Column specifications
# ──────────────────────────────────────────────────────────────────────
REIGN_RAW_COLS = [
    "ccode",              # GW country code (native)
    "year", "month",
    "government",         # Regime type string (e.g., "Presidential Democracy")
    "leader",             # Leader name
    "elected",            # Was the leader elected? (binary)
    "tenure_months",      # Months in power
    "age",                # Leader age
    "male",               # Leader gender (binary)
    "militarycareer",     # Military background (binary)
    "lastelection",       # Date of last election
    "loss",               # Did the leader lose power this month? (binary)
    "irregular",          # Irregular leadership change (coup, revolution)
    "prev_conflict",      # Previous conflict indicator
    "precip",             # Precipitation anomaly (SPI)
]

# Slow-moving features: safe to forward-fill past Aug 2021
SLOW_FEATURES = [
    "government", "elected", "male", "militarycareer",
    "leader", "prev_conflict",
]

# Fast-changing/event features: must be NaN after cutoff
FAST_FEATURES = ["loss", "irregular"]

REIGN_CUTOFF = "2021-08"

# Study period
YEAR_MIN = 2010
YEAR_MAX = 2024


def load_reign(csv_path: str | Path) -> pd.DataFrame:
    """
    Load REIGN CSV with basic validation.

    Parameters
    ----------
    csv_path : path to REIGN CSV (e.g., REIGN_2021_8.csv)

    Returns
    -------
    DataFrame with selected REIGN columns
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"REIGN CSV not found: {csv_path}")

    logger.info("Loading REIGN from %s", csv_path)

    reign = pd.read_csv(csv_path, low_memory=False)

    # Select available columns (REIGN schema may vary across versions)
    available = [c for c in REIGN_RAW_COLS if c in reign.columns]
    missing = set(REIGN_RAW_COLS) - set(available)
    if missing:
        logger.warning("REIGN columns not found: %s", missing)

    reign = reign[available].copy()

    # Temporal filter
    reign = reign[(reign["year"] >= YEAR_MIN) & (reign["year"] <= YEAR_MAX)].copy()

    logger.info("REIGN loaded: %d country-months", len(reign))
    return reign


def _create_temporal_index(reign: pd.DataFrame) -> pd.DataFrame:
    """Create canonical (gwcode, year_month) index from REIGN's native fields."""
    reign["year_month"] = (
        reign["year"].astype(int).astype(str)
        + "-"
        + reign["month"].astype(int).astype(str).str.zfill(2)
    )
    reign["gwcode"] = reign["ccode"].astype(int)
    return reign


def _handle_coverage_gap(reign: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the REIGN post-August 2021 coverage gap.

    Strategy (Option A from project spec):
        - Slow-moving features: forward-fill from last known observation
        - Fast-changing event features: set to NaN (cannot be inferred)
        - Tenure months: extrapolate linearly under same-leader assumption
          (with caveat flag)

    This approach is conservative: it preserves regime context while
    avoiding fabrication of discrete events (coups, elections).
    """
    # Identify the last observation per country
    last_obs = reign.groupby("gwcode")["year_month"].max().reset_index()
    last_obs.columns = ["gwcode", "last_reign_month"]

    # Generate missing months for each country up to YEAR_MAX
    all_months = pd.date_range(
        start=f"{YEAR_MIN}-01-01",
        end=f"{YEAR_MAX}-12-01",
        freq="MS",
    ).strftime("%Y-%m").tolist()

    # Build a complete panel skeleton
    countries = reign["gwcode"].unique()
    skeleton = pd.DataFrame(
        [(gw, ym) for gw in countries for ym in all_months],
        columns=["gwcode", "year_month"],
    )

    # Merge existing data onto skeleton
    reign_extended = skeleton.merge(reign, on=["gwcode", "year_month"], how="left")

    # Sort for forward-fill
    reign_extended = reign_extended.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # Forward-fill slow features within each country
    slow_available = [c for c in SLOW_FEATURES if c in reign_extended.columns]
    reign_extended[slow_available] = reign_extended.groupby("gwcode")[slow_available].ffill()

    # NaN out fast features after the REIGN cutoff
    fast_available = [c for c in FAST_FEATURES if c in reign_extended.columns]
    post_cutoff = reign_extended["year_month"] > REIGN_CUTOFF
    reign_extended.loc[post_cutoff, fast_available] = np.nan

    # Forward-fill numeric slow features (age, tenure) with caution
    if "age" in reign_extended.columns:
        # Age can be extrapolated: increment by 1/12 per month from last known
        reign_extended["age"] = reign_extended.groupby("gwcode")["age"].ffill()
        # Approximate age increment for forward-filled months
        for gw in countries:
            mask = (reign_extended["gwcode"] == gw) & post_cutoff
            if mask.any():
                last_age_idx = reign_extended.loc[
                    (reign_extended["gwcode"] == gw)
                    & (reign_extended["year_month"] <= REIGN_CUTOFF)
                ].index
                if len(last_age_idx) > 0:
                    last_idx = last_age_idx[-1]
                    n_months_after = mask.sum()
                    increments = np.arange(1, n_months_after + 1) / 12.0
                    base_age = reign_extended.loc[last_idx, "age"]
                    if pd.notna(base_age):
                        reign_extended.loc[mask, "age"] = base_age + increments

    if "tenure_months" in reign_extended.columns:
        # Tenure: extrapolate linearly (same-leader assumption)
        reign_extended["tenure_months"] = reign_extended.groupby("gwcode")["tenure_months"].ffill()
        for gw in countries:
            mask = (reign_extended["gwcode"] == gw) & post_cutoff
            if mask.any():
                last_tenure_idx = reign_extended.loc[
                    (reign_extended["gwcode"] == gw)
                    & (reign_extended["year_month"] <= REIGN_CUTOFF)
                ].index
                if len(last_tenure_idx) > 0:
                    last_idx = last_tenure_idx[-1]
                    n_months_after = mask.sum()
                    base_tenure = reign_extended.loc[last_idx, "tenure_months"]
                    if pd.notna(base_tenure):
                        reign_extended.loc[mask, "tenure_months"] = (
                            base_tenure + np.arange(1, n_months_after + 1)
                        )

    n_filled = post_cutoff.sum()
    logger.info(
        "REIGN coverage gap: %d country-months forward-filled after %s",
        n_filled, REIGN_CUTOFF,
    )

    return reign_extended


def _derive_features(reign: pd.DataFrame) -> pd.DataFrame:
    """
    Derive conflict-relevant features from REIGN raw fields.

    Features:
        leader_age_risk: Binary flag for age-based vulnerability.
            Leaders under 40 or over 75 are empirically associated with
            higher regime instability (Bueno de Mesquita et al., 2003).

        months_since_election: Proxy for democratic legitimacy decay.
            Longer gaps between elections correlate with weakening
            accountability mechanisms (Lindberg, 2006).

        regime_change: Binary flag for month-on-month regime type transition.
            Structural breaks that invalidate steady-state assumptions.

        coup_event: Binary indicator derived from the 'irregular' field.
            Direct conflict escalation risk factor.
    """
    # Leader age risk
    if "age" in reign.columns:
        reign["leader_age_risk"] = np.where(
            (reign["age"] < 40) | (reign["age"] > 75), 1, 0
        )
        # Preserve NaN where age is unknown
        reign.loc[reign["age"].isna(), "leader_age_risk"] = np.nan
    else:
        reign["leader_age_risk"] = np.nan

    # Months since last election
    if "lastelection" in reign.columns:
        # Parse election dates robustly
        reign["lastelection_parsed"] = pd.to_datetime(
            reign["lastelection"], errors="coerce"
        )
        reign["year_month_dt"] = pd.to_datetime(reign["year_month"] + "-01")
        reign["months_since_election"] = (
            (reign["year_month_dt"] - reign["lastelection_parsed"]).dt.days / 30.44
        )
        # Clamp negative values (election in the future relative to REIGN coding)
        reign.loc[reign["months_since_election"] < 0, "months_since_election"] = np.nan
        reign.drop(columns=["lastelection_parsed", "year_month_dt"], inplace=True)
    else:
        reign["months_since_election"] = np.nan

    # Regime change detection (structural break)
    if "government" in reign.columns:
        reign = reign.sort_values(["gwcode", "year_month"])
        reign["regime_change"] = (
            reign.groupby("gwcode")["government"].shift(1) != reign["government"]
        ).astype(float)
        # First observation per country is not a "change"
        first_mask = ~reign.duplicated(subset=["gwcode"], keep="first")
        reign.loc[first_mask, "regime_change"] = 0.0
        # NaN where government is unknown
        reign.loc[reign["government"].isna(), "regime_change"] = np.nan
    else:
        reign["regime_change"] = np.nan

    # Coup event (from irregular transitions)
    if "irregular" in reign.columns:
        reign["coup_event"] = reign["irregular"].fillna(0).astype(float)
        # But keep NaN for post-cutoff period where irregular is NaN
        reign.loc[reign["irregular"].isna(), "coup_event"] = np.nan
    else:
        reign["coup_event"] = np.nan

    return reign


def _encode_regime_type(reign: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode REIGN's string-valued government/regime type."""
    if "government" not in reign.columns:
        return reign

    regime_dummies = pd.get_dummies(
        reign["government"],
        prefix="reign_regime",
    ).astype(int)

    reign = pd.concat([reign, regime_dummies], axis=1)
    logger.info(
        "REIGN regime types encoded: %d categories",
        len(regime_dummies.columns),
    )
    return reign


def _detect_structural_breaks(reign: pd.DataFrame) -> pd.DataFrame:
    """
    Detect structural breaks and compute proximity / reliability features.

    A structural break (coup, regime transition) has two consequences:
        1. Forward-filled values from BEFORE the break are invalid for
           months AFTER the break. We flag this with `reign_ffill_reliable`.
        2. Proximity to a structural break is itself predictive of conflict
           escalation — instability cascades in the months following a
           regime change (Cederman et al., 2010). We encode this with
           `months_since_structural_break`.

    These features go beyond detection-and-logging: they give the model
    explicit signals about when background assumptions are violated.
    """
    reign = reign.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # ── Detect and log breaks ────────────────────────────────────────
    if "regime_change" in reign.columns:
        changes = reign[reign["regime_change"] == 1]
        if len(changes) > 0:
            logger.info("Detected %d regime transitions:", len(changes))
            for _, row in changes.head(20).iterrows():
                logger.info(
                    "  GW %d, %s: %s",
                    row.get("gwcode", "?"),
                    row.get("year_month", "?"),
                    row.get("government", "?"),
                )

    if "coup_event" in reign.columns:
        coups = reign[reign["coup_event"] == 1]
        if len(coups) > 0:
            logger.info("Detected %d coup/irregular events:", len(coups))
            for _, row in coups.head(20).iterrows():
                logger.info(
                    "  GW %d, %s",
                    row.get("gwcode", "?"),
                    row.get("year_month", "?"),
                )

    # ── Compute months_since_structural_break ────────────────────────
    # Combines regime changes and coups into a single "structural break"
    # indicator, then computes forward-looking distance from last break.
    reign["structural_break"] = 0.0
    if "regime_change" in reign.columns:
        reign["structural_break"] = reign["structural_break"].where(
            reign["regime_change"] != 1, 1.0
        )
    if "coup_event" in reign.columns:
        reign["structural_break"] = reign["structural_break"].where(
            reign["coup_event"] != 1, 1.0
        )

    # For each country, compute months since the last structural break.
    # This captures the "instability window" — the 6-12 months after a
    # regime change when conflict risk is empirically elevated.
    def _months_since_break(group):
        result = pd.Series(np.nan, index=group.index)
        last_break_idx = None
        for i, (idx, val) in enumerate(group.items()):
            if val == 1.0:
                last_break_idx = i
                result.iloc[i] = 0
            elif last_break_idx is not None:
                result.iloc[i] = i - last_break_idx
        return result

    reign["months_since_structural_break"] = reign.groupby("gwcode")[
        "structural_break"
    ].transform(_months_since_break)

    # ── Forward-fill reliability flag ────────────────────────────────
    # After the REIGN cutoff (2021-08), forward-filled values are only
    # reliable if NO structural break occurred in the last 6 months
    # before the cutoff. Countries that had recent instability before
    # the cutoff are more likely to have changed since — flag this.
    reign["reign_ffill_reliable"] = 1.0

    post_cutoff = reign["year_month"] > REIGN_CUTOFF
    reign.loc[post_cutoff, "reign_ffill_reliable"] = 1.0  # Default: trust ffill

    # Check each country: if a structural break happened within 6 months
    # of the cutoff, mark post-cutoff forward-fill as unreliable
    for gw in reign["gwcode"].unique():
        gw_mask = reign["gwcode"] == gw
        pre_cutoff_breaks = reign.loc[
            gw_mask
            & (reign["year_month"] > "2021-02")  # within 6 months of cutoff
            & (reign["year_month"] <= REIGN_CUTOFF)
            & (reign["structural_break"] == 1.0)
        ]
        if len(pre_cutoff_breaks) > 0:
            reign.loc[gw_mask & post_cutoff, "reign_ffill_reliable"] = 0.0
            logger.warning(
                "  GW %d: structural break near REIGN cutoff — "
                "post-2021 forward-fill flagged unreliable",
                gw,
            )

    # Clean up intermediate column
    reign.drop(columns=["structural_break"], inplace=True)

    n_unreliable = (reign.loc[post_cutoff, "reign_ffill_reliable"] == 0).sum()
    logger.info(
        "Forward-fill reliability: %d post-cutoff country-months flagged unreliable",
        n_unreliable,
    )

    return reign


def ingest_reign(csv_path: str | Path) -> pd.DataFrame:
    """
    Full REIGN ingestion pipeline: load → index → gap-fill → derive → encode → validate.

    Parameters
    ----------
    csv_path : path to REIGN CSV

    Returns
    -------
    DataFrame indexed by (gwcode, year_month) with all raw + derived REIGN features.
    Lag is NOT applied here — that is done in structural_features.py.
    """
    logger.info("=" * 60)
    logger.info("REIGN ingestion pipeline starting")
    logger.info("=" * 60)

    # 1. Load
    reign = load_reign(csv_path)

    # 2. Create temporal index
    reign = _create_temporal_index(reign)

    # 3. Handle coverage gap (post-Aug 2021)
    reign = _handle_coverage_gap(reign)

    # 4. Derive features
    reign = _derive_features(reign)

    # 5. Encode regime type
    reign = _encode_regime_type(reign)

    # 6. Detect and log structural breaks
    reign = _detect_structural_breaks(reign)

    # 7. Select output columns
    #    Keep gwcode, year_month, and all engineered features
    meta_cols = ["gwcode", "year_month"]
    feature_cols = [
        "tenure_months", "age", "male", "militarycareer", "elected",
        "leader_age_risk", "months_since_election",
        "regime_change", "coup_event", "prev_conflict", "precip",
        "months_since_structural_break", "reign_ffill_reliable",
    ]
    # Add one-hot regime columns
    regime_cols = [c for c in reign.columns if c.startswith("reign_regime_")]
    keep_cols = meta_cols + [c for c in feature_cols if c in reign.columns] + regime_cols

    reign = reign[keep_cols].copy()

    # 8. Deduplicate (some REIGN editions have duplicate country-months)
    reign = reign.drop_duplicates(subset=["gwcode", "year_month"], keep="last")

    logger.info(
        "REIGN ingestion complete: %d rows, %d columns, %d countries",
        len(reign),
        len(reign.columns),
        reign["gwcode"].nunique(),
    )

    return reign
