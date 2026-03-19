"""
Powell & Thyne Coup d'État Dataset — Ingestion & Integration
==============================================================

Ingests the Powell & Thyne (2011) global coup dataset, which provides
event-level records of every successful and failed coup attempt from
1950 to present (updated within weeks of new events).

This module serves as a **patch** for REIGN's post-August 2021 gap:
REIGN's `coup_event` and `irregular` fields are NaN after its cutoff,
but coups continued to happen (Myanmar 2021, Guinea 2021, Sudan 2021,
Burkina Faso 2022, Niger 2023, Gabon 2023, etc.). Powell & Thyne
provides ground-truth coup data through the present day.

Integration strategy:
    - For 2010-01 to 2021-08: REIGN's coup coding is authoritative
      (finer-grained, includes non-coup irregular transitions).
    - For 2021-09 onwards: Powell & Thyne replaces REIGN's NaN values.
    - For the full period: additional derived features (cumulative coup
      history, failed coup attempts) supplement both sources.

Data format:
    TSV (tab-separated), one row per coup attempt.
    Columns: country, ccode (GW), year, month, day, coup (1=failed, 2=success)
    Native Gleditsch-Ward codes — no crosswalk needed.

Reference:
    Powell, J.M. & Thyne, C.L. (2011). Global Instances of Coups from
    1950 to 2010: A New Dataset. Journal of Peace Research, 48(2), 249-259.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POWELL_THYNE_URL = (
    "http://www.uky.edu/~clthyn2/coup_data/powell_thyne_coups_final.txt"
)

REIGN_CUTOFF = "2021-08"
YEAR_MIN = 2010
YEAR_MAX = 2024


def download_powell_thyne(output_path: Optional[str | Path] = None) -> Path:
    """
    Download the Powell & Thyne coup dataset from UKY.

    Returns path to downloaded TSV.
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else raw_dir / "powell_thyne_coups.tsv"

    if output_path.exists() and output_path.stat().st_size > 100:
        logger.info("Powell & Thyne already downloaded: %s", output_path)
        return output_path

    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    logger.info("Downloading Powell & Thyne coup data from %s", POWELL_THYNE_URL)
    response = requests.get(POWELL_THYNE_URL, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.info(
        "Powell & Thyne saved to %s (%d bytes)",
        output_path, len(response.content),
    )
    return output_path


def load_powell_thyne(tsv_path: str | Path) -> pd.DataFrame:
    """
    Load the Powell & Thyne event-level coup dataset.

    Parameters
    ----------
    tsv_path : path to powell_thyne_coups_final.txt (TSV)

    Returns
    -------
    DataFrame with columns: gwcode, year, month, day, coup_successful, coup_failed
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Powell & Thyne file not found: {tsv_path}")

    logger.info("Loading Powell & Thyne from %s", tsv_path)

    # Tab-separated, with quoted strings
    pt = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Normalise column names (handle minor schema variations across versions)
    col_map = {}
    for c in pt.columns:
        cl = c.strip().lower()
        if cl == "ccode":
            col_map[c] = "gwcode"
        elif cl == "country":
            col_map[c] = "country_name"
        elif cl in ("coup",):
            col_map[c] = "coup_code"
    pt = pt.rename(columns=col_map)

    # Use ccode as GW code (Powell & Thyne uses GW natively)
    if "gwcode" not in pt.columns:
        raise ValueError("Expected 'ccode' column in Powell & Thyne data")

    pt["gwcode"] = pd.to_numeric(pt["gwcode"], errors="coerce")
    pt = pt.dropna(subset=["gwcode"]).copy()
    pt["gwcode"] = pt["gwcode"].astype(int)

    # Parse coup outcome: 1 = failed, 2 = successful
    pt["coup_code"] = pd.to_numeric(pt["coup_code"], errors="coerce")
    pt["pt_coup_successful"] = (pt["coup_code"] == 2).astype(int)
    pt["pt_coup_failed"] = (pt["coup_code"] == 1).astype(int)

    # Temporal filter
    pt["year"] = pd.to_numeric(pt["year"], errors="coerce").astype("Int64")
    pt["month"] = pd.to_numeric(pt["month"], errors="coerce").astype("Int64")
    pt = pt.dropna(subset=["year", "month"]).copy()
    pt = pt[(pt["year"] >= YEAR_MIN) & (pt["year"] <= YEAR_MAX)].copy()

    logger.info(
        "Powell & Thyne loaded: %d coup events (%d-%d), %d successful, %d failed",
        len(pt), pt["year"].min(), pt["year"].max(),
        pt["pt_coup_successful"].sum(), pt["pt_coup_failed"].sum(),
    )

    return pt


def aggregate_to_country_month(pt: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-level coup data to country-month panel format.

    For each (gwcode, year_month), produces:
        pt_coup_event: 1 if any coup attempt (successful or failed) this month
        pt_coup_successful: 1 if a successful coup this month
        pt_coup_failed: 1 if a failed coup attempt this month
        pt_coup_count: total number of coup attempts this month (usually 0 or 1)

    Returns DataFrame indexed by (gwcode, year_month).
    """
    pt = pt.copy()
    pt["year_month"] = (
        pt["year"].astype(int).astype(str)
        + "-"
        + pt["month"].astype(int).astype(str).str.zfill(2)
    )

    # Aggregate: max (binary) and sum (count) per country-month
    monthly = pt.groupby(["gwcode", "year_month"]).agg(
        pt_coup_successful=("pt_coup_successful", "max"),
        pt_coup_failed=("pt_coup_failed", "max"),
        pt_coup_count=("coup_code", "count"),
    ).reset_index()

    # Any coup event (successful or failed)
    monthly["pt_coup_event"] = (
        (monthly["pt_coup_successful"] == 1) | (monthly["pt_coup_failed"] == 1)
    ).astype(int)

    logger.info(
        "Aggregated to %d country-months with coup activity",
        len(monthly),
    )

    return monthly


def derive_coup_history(
    panel: pd.DataFrame,
    monthly_coups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive historical coup features for the full panel.

    Features:
        pt_coup_event: binary, any coup attempt this month
        pt_coup_successful: binary, successful coup this month
        pt_coup_failed: binary, failed coup this month
        pt_cumulative_coups: running count of all coup attempts for this
            country since 2010. Captures "coup-prone" states.
        pt_months_since_coup: months since last coup attempt (any type).
            Captures the "coup trap" — countries that had a recent coup
            are empirically more likely to have another (Powell, 2012).

    Parameters
    ----------
    panel : full (gwcode, year_month) panel skeleton
    monthly_coups : output of aggregate_to_country_month()

    Returns
    -------
    Panel with coup features added (0 for months with no coup activity)
    """
    # Merge coup events onto panel (left join: no-coup months get NaN)
    panel = panel.merge(monthly_coups, on=["gwcode", "year_month"], how="left")

    # Fill NaN with 0 (no coup activity)
    coup_cols = ["pt_coup_event", "pt_coup_successful", "pt_coup_failed", "pt_coup_count"]
    for col in coup_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)

    # Sort for cumulative computations
    panel = panel.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # Cumulative coup count per country (running total)
    panel["pt_cumulative_coups"] = panel.groupby("gwcode")["pt_coup_event"].cumsum()

    # Months since last coup attempt
    def _months_since_event(group):
        result = pd.Series(np.nan, index=group.index)
        last_event_pos = None
        for i, (idx, val) in enumerate(group.items()):
            if val == 1:
                last_event_pos = i
                result.iloc[i] = 0
            elif last_event_pos is not None:
                result.iloc[i] = i - last_event_pos
        return result

    panel["pt_months_since_coup"] = panel.groupby("gwcode")[
        "pt_coup_event"
    ].transform(_months_since_event)

    return panel


def integrate_with_reign(
    panel: pd.DataFrame,
    monthly_coups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate Powell & Thyne coup data with existing REIGN coup fields.

    Strategy:
        - 2010-01 to 2021-08: REIGN `coup_event` is authoritative.
          Powell & Thyne is used for supplementary features only.
        - 2021-09 onwards: REIGN `coup_event` is NaN.
          Powell & Thyne `pt_coup_event` replaces it.

    Also patches `reign_ffill_reliable`: if Powell & Thyne detects a
    post-2021 coup for a country, the REIGN forward-fill for that
    country is definitively unreliable (the leader/regime changed).

    Parameters
    ----------
    panel : assembled panel with REIGN features (coup_event may be NaN post-2021)
    monthly_coups : output of aggregate_to_country_month()

    Returns
    -------
    Panel with integrated coup features
    """
    # First, add all Powell & Thyne features to the panel
    panel = derive_coup_history(panel, monthly_coups)

    # Patch REIGN's coup_event NaN values with Powell & Thyne data
    if "coup_event" in panel.columns:
        post_cutoff = panel["year_month"] > REIGN_CUTOFF
        reign_nan = panel["coup_event"].isna()
        patch_mask = post_cutoff & reign_nan

        n_patched = patch_mask.sum()
        panel.loc[patch_mask, "coup_event"] = panel.loc[patch_mask, "pt_coup_event"].astype(float)

        # Count how many actual coups were filled in
        coups_filled = panel.loc[patch_mask & (panel["pt_coup_event"] == 1)]
        logger.info(
            "Patched %d post-cutoff REIGN coup_event NaN values with Powell & Thyne "
            "(%d actual coup events discovered)",
            n_patched, len(coups_filled),
        )

        # Log the specific coups discovered post-REIGN
        if len(coups_filled) > 0:
            from src.data.crosswalk import GW_TO_ISO3
            for _, row in coups_filled.iterrows():
                iso3 = GW_TO_ISO3.get(row["gwcode"], "???")
                success = "SUCCESSFUL" if row.get("pt_coup_successful", 0) == 1 else "FAILED"
                logger.info(
                    "  Post-REIGN coup: GW %d (%s), %s — %s",
                    row["gwcode"], iso3, row["year_month"], success,
                )

        # Patch reign_ffill_reliable: if a post-2021 coup happened,
        # the REIGN forward-fill for that country is definitively wrong
        if "reign_ffill_reliable" in panel.columns:
            post_coup_countries = panel.loc[
                post_cutoff & (panel["pt_coup_successful"] == 1),
                "gwcode"
            ].unique()

            for gw in post_coup_countries:
                # Find the month of the coup
                coup_month = panel.loc[
                    (panel["gwcode"] == gw)
                    & post_cutoff
                    & (panel["pt_coup_successful"] == 1),
                    "year_month"
                ].min()

                # Mark all months from the coup onwards as unreliable
                invalidate_mask = (
                    (panel["gwcode"] == gw)
                    & (panel["year_month"] >= coup_month)
                )
                panel.loc[invalidate_mask, "reign_ffill_reliable"] = 0.0

                from src.data.crosswalk import GW_TO_ISO3
                iso3 = GW_TO_ISO3.get(gw, "???")
                logger.info(
                    "  GW %d (%s): REIGN forward-fill invalidated from %s "
                    "(successful coup detected by Powell & Thyne)",
                    gw, iso3, coup_month,
                )

    return panel


def ingest_powell_thyne(tsv_path: str | Path) -> pd.DataFrame:
    """
    Full Powell & Thyne ingestion pipeline.

    Parameters
    ----------
    tsv_path : path to powell_thyne_coups_final.txt

    Returns
    -------
    Aggregated country-month DataFrame ready for panel merge.
    """
    logger.info("=" * 60)
    logger.info("Powell & Thyne coup data ingestion")
    logger.info("=" * 60)

    pt = load_powell_thyne(tsv_path)
    monthly = aggregate_to_country_month(pt)

    # Log summary of post-REIGN coups
    post_reign = monthly[monthly["year_month"] > REIGN_CUTOFF]
    if len(post_reign) > 0:
        logger.info(
            "Post-REIGN coup events (after %s): %d country-months",
            REIGN_CUTOFF, len(post_reign),
        )
        from src.data.crosswalk import GW_TO_ISO3
        for _, row in post_reign.iterrows():
            iso3 = GW_TO_ISO3.get(row["gwcode"], "???")
            ctype = []
            if row["pt_coup_successful"]:
                ctype.append("successful")
            if row["pt_coup_failed"]:
                ctype.append("failed")
            logger.info(
                "  GW %d (%s), %s: %s",
                row["gwcode"], iso3, row["year_month"], "/".join(ctype),
            )

    return monthly
