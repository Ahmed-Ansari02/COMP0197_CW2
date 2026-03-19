"""
Economic Covariates Ingestion & Feature Engineering
=====================================================

Ingests three complementary economic data streams that capture the material
conditions theorised to mediate conflict risk:

    1. Exchange rates (IMF IFS): currency instability as a proxy for
       macroeconomic fragility and capital flight.
    2. GDP growth (World Bank WDI): annual economic performance deviation
       from country-specific trends.
    3. Food prices (FAO FPMA / FAOSTAT): food security pressure, which has
       direct theoretical linkage to protest mobilisation and civil unrest
       (Bellemare, 2015; Hendrix & Haggard, 2015).

Design decisions:
    - Exchange rate volatility is computed as the 3-month rolling standard
      deviation of month-on-month percentage changes, following the
      realised volatility approach (Andersen et al., 2003).
    - Sudden depreciation is flagged using a country-specific 2σ threshold,
      which is robust to the heteroskedasticity across currency regimes.
    - GDP deviation is measured relative to a 5-year rolling trend to
      capture country-specific recession signals rather than level effects.
    - Food price anomaly is the ratio of current CPI to its 12-month
      rolling mean, capturing acute price shocks irrespective of base levels.
    - All features are delivered in raw (unnormalised) form; Member C
      handles normalisation on the training split.

References:
    Bellemare, M.F. (2015). Rising food prices, food price volatility,
        and social unrest. Am. J. Agr. Econ., 97(1), 1-21.
    Hendrix, C.S. & Haggard, S. (2015). Global food prices, regime type,
        and urban unrest in the developing world. J. Peace Res., 52(2), 143-157.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.crosswalk import ISO3_TO_GW, IMF_TO_ISO3, gw_from_iso3

logger = logging.getLogger(__name__)

YEAR_MIN = 2010
YEAR_MAX = 2024


# ══════════════════════════════════════════════════════════════════════
# 1. Exchange Rates (IMF International Financial Statistics)
# ══════════════════════════════════════════════════════════════════════

def ingest_exchange_rates(csv_path: str | Path) -> pd.DataFrame:
    """
    Ingest IMF IFS exchange rate data (domestic currency per USD, period average).

    Expected CSV columns: country_code (or iso3), year, month, exchange_rate
    The exact schema depends on the IMF bulk download format; we handle
    common variants.

    Returns DataFrame with columns: gwcode, year_month, fx_pct_change,
    fx_volatility, fx_depreciation_flag.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Exchange rate CSV not found: {csv_path}")

    logger.info("Loading exchange rates from %s", csv_path)
    econ = pd.read_csv(csv_path, low_memory=False)

    logger.info("FX CSV columns (first 10): %s", list(econ.columns[:10]))
    import re

    # ── Detect format ─────────────────────────────────────────────────
    has_series_code = "SERIES_CODE" in econ.columns
    has_indicator = "INDICATOR" in econ.columns
    imf_sdmx = any(c.strip().upper() in ("REF_AREA", "TIME_PERIOD", "OBS_VALUE") for c in econ.columns)

    # IMF Data Explorer wide format: metadata columns + time period columns
    # Period columns are "YYYY", "YYYY-QN", or "YYYY-MDD"
    monthly_period_cols = [c for c in econ.columns if re.match(r"^\d{4}-M\d{2}$", c.strip())]

    if has_series_code and monthly_period_cols:
        # ── IMF Data Explorer wide/pivot format ──────────────────────
        # Rows are series (one per country × indicator × frequency × transformation)
        # Columns include metadata + "2009-M01", "2009-M02", ..., "2024-M12"
        logger.info(
            "Detected IMF Data Explorer wide format: %d series, %d monthly period columns",
            len(econ), len(monthly_period_cols),
        )

        # Filter to target indicator and frequency
        if has_indicator:
            # Keep only "Domestic currency per US Dollar" (or similar)
            usd_mask = econ["INDICATOR"].str.contains("Domestic currency per US Dollar", na=False)
            econ = econ[usd_mask].copy()
            logger.info("Filtered to 'Domestic currency per US Dollar': %d series", len(econ))

        if "FREQUENCY" in econ.columns:
            econ = econ[econ["FREQUENCY"] == "Monthly"].copy()
            logger.info("Filtered to Monthly: %d series", len(econ))

        # Filter to period average (not end-of-period) if TYPE_OF_TRANSFORMATION exists
        if "TYPE_OF_TRANSFORMATION" in econ.columns:
            pa_mask = econ["TYPE_OF_TRANSFORMATION"].str.contains("Period average", na=False)
            if pa_mask.any():
                econ = econ[pa_mask].copy()
                logger.info("Filtered to Period average: %d series", len(econ))

        if len(econ) == 0:
            raise ValueError(
                "No 'Domestic currency per US Dollar' monthly period-average series found. "
                "Check the IMF download filters."
            )

        # Extract ISO3 from SERIES_CODE (format: "LKA.XDC_USD.PA_RT.M")
        econ["iso3"] = econ["SERIES_CODE"].str.split(".").str[0].str.strip().str.upper()

        # Deduplicate: if multiple series per country (e.g., PA_RT and EOP_RT),
        # keep the first one per country
        econ = econ.drop_duplicates(subset=["iso3"], keep="first")
        logger.info("Unique countries: %d", econ["iso3"].nunique())

        # Melt: wide → long (only monthly columns)
        id_cols = [c for c in econ.columns if c not in monthly_period_cols]
        econ = econ.melt(
            id_vars=["iso3"],
            value_vars=monthly_period_cols,
            var_name="period",
            value_name="exchange_rate",
        )

        # Parse "2009-M01" → year=2009, month=1
        econ["year"] = econ["period"].str[:4].astype(int)
        econ["month"] = econ["period"].str[6:].astype(int)
        econ["exchange_rate"] = pd.to_numeric(econ["exchange_rate"], errors="coerce")

    elif imf_sdmx:
        # ── IMF SDMX-CSV long format ─────────────────────────────────
        logger.info("Detected IMF SDMX format")
        sdmx_map = {}
        for c in econ.columns:
            cu = c.strip().upper()
            if cu == "REF_AREA":
                sdmx_map[c] = "iso3"
            elif cu == "OBS_VALUE":
                sdmx_map[c] = "exchange_rate"
            elif cu == "TIME_PERIOD":
                sdmx_map[c] = "time_period"
        econ = econ.rename(columns=sdmx_map)

        econ["exchange_rate"] = pd.to_numeric(econ["exchange_rate"], errors="coerce")
        econ["time_period"] = econ["time_period"].astype(str).str.strip()
        econ["year"] = econ["time_period"].str[:4].astype(int)
        econ["month"] = econ["time_period"].str[5:7].astype(int)
        econ["iso3"] = econ["iso3"].str.strip().str.upper()

        # Filter to monthly if FREQ column exists
        freq_col = [c for c in econ.columns if c.strip().upper() == "FREQ"]
        if freq_col:
            econ = econ[econ[freq_col[0]].astype(str).str.strip().str.upper() == "M"].copy()

    else:
        # ── Generic format ────────────────────────────────────────────
        col_map = {}
        for c in econ.columns:
            cl = c.lower().strip()
            if cl in ("country_code", "iso3", "iso", "country_iso3", "ref_area"):
                col_map[c] = "iso3"
            elif cl in ("imf_code", "imf_country_code", "country_code_imf"):
                col_map[c] = "imf_code"
            elif cl in ("year",):
                col_map[c] = "year"
            elif cl in ("month",):
                col_map[c] = "month"
            elif cl in ("exchange_rate", "value", "rate", "fx_rate", "obs_value"):
                col_map[c] = "exchange_rate"
        econ = econ.rename(columns=col_map)

    # Map to GW codes
    if "iso3" in econ.columns:
        econ["gwcode"] = econ["iso3"].map(ISO3_TO_GW)
    elif "imf_code" in econ.columns:
        imf_to_gw = {k: ISO3_TO_GW.get(v) for k, v in IMF_TO_ISO3.items()}
        econ["gwcode"] = econ["imf_code"].map(imf_to_gw)
    else:
        raise ValueError(
            "Exchange rate CSV must contain a country code column. "
            f"Found columns: {list(econ.columns)}"
        )

    econ["exchange_rate"] = pd.to_numeric(econ.get("exchange_rate", pd.Series(dtype=float)), errors="coerce")
    econ = econ.dropna(subset=["gwcode", "exchange_rate"]).copy()
    econ["gwcode"] = econ["gwcode"].astype(int)

    # Create temporal index
    econ["year_month"] = (
        econ["year"].astype(int).astype(str)
        + "-"
        + econ["month"].astype(int).astype(str).str.zfill(2)
    )

    # Filter to study period
    econ = econ[(econ["year"] >= YEAR_MIN) & (econ["year"] <= YEAR_MAX)].copy()

    # Sort for rolling computations
    econ = econ.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # ── Feature engineering ──────────────────────────────────────────

    # Month-on-month percentage change
    econ["fx_pct_change"] = econ.groupby("gwcode")["exchange_rate"].pct_change()

    # 3-month rolling volatility (realised volatility estimator)
    econ["fx_volatility"] = econ.groupby("gwcode")["fx_pct_change"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )

    # Log-transform volatility to handle heavy tails.
    # FX volatility is right-skewed: many stable months, few crisis spikes.
    # log(1 + x) preserves zeros while compressing the tail, making the
    # distribution closer to normal for the transformer's attention mechanism.
    econ["fx_volatility_log"] = np.log1p(econ["fx_volatility"])

    # Sudden depreciation flag: country-specific 2σ threshold
    # This is robust to the heteroskedasticity across different currency regimes
    country_stats = econ.groupby("gwcode")["fx_pct_change"].agg(["mean", "std"]).reset_index()
    country_stats.columns = ["gwcode", "fx_mean_hist", "fx_std_hist"]
    econ = econ.merge(country_stats, on="gwcode", how="left")
    econ["fx_depreciation_flag"] = (
        econ["fx_pct_change"] > (econ["fx_mean_hist"] + 2 * econ["fx_std_hist"])
    ).astype(int)
    # Where std is NaN or zero, flag is 0
    econ.loc[econ["fx_std_hist"].isna() | (econ["fx_std_hist"] == 0), "fx_depreciation_flag"] = 0

    # Country-normalised FX change: z-score each country's pct_change
    # against its own history. This makes a 5% depreciation in Turkey
    # (routine) distinguishable from 5% in Switzerland (crisis).
    # NOTE: this is within-country normalisation (not cross-sectional),
    # so it does not leak test-set information.
    econ["fx_pct_change_zscore"] = econ.groupby("gwcode")["fx_pct_change"].transform(
        lambda x: (x - x.expanding(min_periods=6).mean())
        / x.expanding(min_periods=6).std().clip(lower=1e-8)
    )

    # Clean up
    output_cols = [
        "gwcode", "year_month",
        "fx_pct_change", "fx_volatility", "fx_volatility_log",
        "fx_depreciation_flag", "fx_pct_change_zscore",
    ]
    econ = econ[output_cols].copy()

    # Forward-fill gaps up to 3 months (reporting delays)
    econ = econ.sort_values(["gwcode", "year_month"])
    fill_cols = ["fx_pct_change", "fx_volatility", "fx_volatility_log", "fx_pct_change_zscore"]
    for col in fill_cols:
        econ[col] = econ.groupby("gwcode")[col].ffill(limit=3)

    logger.info(
        "Exchange rates processed: %d rows, %d countries",
        len(econ), econ["gwcode"].nunique(),
    )

    return econ


# ══════════════════════════════════════════════════════════════════════
# 2. GDP Growth (World Bank WDI)
# ══════════════════════════════════════════════════════════════════════

def ingest_gdp(csv_path: str | Path) -> pd.DataFrame:
    """
    Ingest World Bank GDP growth data (NY.GDP.MKTP.KD.ZG).

    Annual data is expanded to monthly (repeat, not interpolate) following
    the same logic as V-Dem: GDP is measured annually and sub-annual
    interpolation would fabricate variation.

    Derived feature:
        gdp_growth_deviation: z-scored deviation from country's own
        5-year rolling mean. Captures whether the economy is performing
        abnormally relative to its own trend — a stronger conflict
        predictor than absolute growth levels (Miguel et al., 2004).

    Expected CSV: iso3 (or country_code), year, gdp_growth
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GDP CSV not found: {csv_path}")

    # Guard against empty files (e.g., failed API download)
    if csv_path.stat().st_size < 10:
        raise ValueError(f"GDP CSV is empty or corrupted: {csv_path}")

    logger.info("Loading GDP data from %s", csv_path)
    gdp = pd.read_csv(csv_path, low_memory=False)

    if len(gdp) == 0:
        raise ValueError(f"GDP CSV has no data rows: {csv_path}")

    # Normalise columns
    col_map = {}
    for c in gdp.columns:
        cl = c.lower().strip()
        if cl in ("country_code", "iso3", "iso", "country_iso3", "countrycode"):
            col_map[c] = "iso3"
        elif cl in ("year", "date"):
            col_map[c] = "year"
        elif cl in ("gdp_growth", "value", "ny.gdp.mktp.kd.zg", "gdp_growth_annual"):
            col_map[c] = "gdp_growth"
    gdp = gdp.rename(columns=col_map)

    if "iso3" not in gdp.columns or "gdp_growth" not in gdp.columns:
        raise ValueError("GDP CSV must contain 'iso3' and 'gdp_growth' columns")

    # Map to GW codes
    gdp["gwcode"] = gdp["iso3"].map(ISO3_TO_GW)
    gdp = gdp.dropna(subset=["gwcode"]).copy()
    gdp["gwcode"] = gdp["gwcode"].astype(int)
    gdp["year"] = gdp["year"].astype(int)

    # Filter: keep 2005+ for 5-year rolling window buffer
    gdp = gdp[gdp["year"] >= 2005].copy()
    gdp = gdp.sort_values(["gwcode", "year"]).reset_index(drop=True)

    # GDP growth deviation from 5-year rolling trend (z-scored).
    # This is the key normalisation for GDP: raw growth percentages are
    # not comparable across countries (5% growth is normal for Ethiopia,
    # exceptional for Japan). The deviation captures whether the economy
    # is performing abnormally relative to ITS OWN trend — a stronger
    # predictor of conflict than absolute levels (Miguel et al., 2004).
    gdp["gdp_growth_deviation"] = gdp.groupby("gwcode")["gdp_growth"].transform(
        lambda x: (
            (x - x.rolling(5, min_periods=3).mean())
            / x.rolling(5, min_periods=3).std().clip(lower=0.01)
        )
    )

    # Negative growth shock indicator: binary flag for country-years
    # where GDP growth is below the country's own 5-year mean by > 1 std.
    # Negative income shocks are empirically the strongest economic
    # predictor of civil conflict onset (Miguel et al., 2004).
    gdp["gdp_negative_shock"] = (gdp["gdp_growth_deviation"] < -1.0).astype(int)

    # Filter to study period after computing rolling features
    gdp = gdp[gdp["year"] >= YEAR_MIN].copy()

    # Expand annual → monthly (repeat, not interpolate)
    months = pd.DataFrame({"month": range(1, 13)})
    gdp = gdp.assign(_key=1)
    months = months.assign(_key=1)
    gdp_monthly = gdp.merge(months, on="_key").drop(columns="_key")
    gdp_monthly["year_month"] = (
        gdp_monthly["year"].astype(str)
        + "-"
        + gdp_monthly["month"].astype(str).str.zfill(2)
    )

    # Forward-fill for years where World Bank data is delayed
    gdp_monthly = gdp_monthly.sort_values(["gwcode", "year_month"])
    max_year = gdp_monthly.dropna(subset=["gdp_growth"])["year"].max()
    if max_year < YEAR_MAX:
        logger.info("GDP data ends at %d; forward-filling to %d", max_year, YEAR_MAX)
        fill_gdp = ["gdp_growth", "gdp_growth_deviation", "gdp_negative_shock"]
        gdp_monthly[fill_gdp] = (
            gdp_monthly.groupby("gwcode")[fill_gdp].ffill()
        )

    output_cols = ["gwcode", "year_month", "gdp_growth", "gdp_growth_deviation", "gdp_negative_shock"]
    gdp_monthly = gdp_monthly[output_cols].copy()

    logger.info(
        "GDP processed: %d rows, %d countries",
        len(gdp_monthly), gdp_monthly["gwcode"].nunique(),
    )

    return gdp_monthly


# ══════════════════════════════════════════════════════════════════════
# 3. Food Prices (FAO FPMA / FAOSTAT CPI Food)
# ══════════════════════════════════════════════════════════════════════

def ingest_food_prices(csv_path: str | Path) -> pd.DataFrame:
    """
    Ingest FAO food price CPI data.

    Features:
        food_price_anomaly: current CPI / 12-month rolling mean.
            Values > 1 indicate above-trend food prices.
            Empirically linked to protest onset (Bellemare, 2015).

        food_cpi_acceleration: month-on-month change in year-on-year rate.
            Captures whether food inflation is accelerating — a stronger
            predictor of unrest than the level itself.

    Expected CSV: iso3 (or country_code), year, month, food_cpi
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Food price CSV not found: {csv_path}")

    logger.info("Loading food prices from %s", csv_path)

    # FAOSTAT downloads are ZIPs containing a CSV — handle both
    if csv_path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(csv_path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {csv_path}")
            with zf.open(csv_names[0]) as f:
                food = pd.read_csv(f, low_memory=False, encoding="utf-8")
    else:
        food = pd.read_csv(csv_path, low_memory=False)

    logger.info("Food CSV columns: %s", list(food.columns))

    # ── Handle FAOSTAT native format ─────────────────────────────────
    # FAOSTAT CSVs have columns like:
    #   "Area Code (M49)", "Area Code", "Area", "Item Code", "Item",
    #   "Year Code", "Year", "Months Code", "Months", "Value", "Flag"
    # Months are text: "January", "February", etc.

    MONTH_NAME_TO_NUM = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    # Detect FAOSTAT format by checking for "Months" column
    faostat_format = any(
        c.strip().lower() in ("months", "months code") for c in food.columns
    )

    if faostat_format:
        logger.info("Detected FAOSTAT native format")

        # ── Map area codes to ISO3 ───────────────────────────────────
        # FAOSTAT uses M49 numeric codes in "Area Code (M49)" column.
        # We need to convert M49 → ISO3 for our GW crosswalk.

        m49_col = None
        iso3_col = None
        for c in food.columns:
            cl = c.strip().lower()
            if "area code" in cl and "iso3" in cl:
                iso3_col = c
                break
            if "area code" in cl and "m49" in cl:
                m49_col = c
            if "area code" in cl and "m49" not in cl and "iso3" not in cl:
                # Generic "Area Code" — check if values are numeric or alpha
                sample = food[c].dropna().astype(str).iloc[:5].tolist()
                if sample and all(s.isdigit() or s.startswith("'") for s in sample):
                    m49_col = m49_col or c  # numeric = M49
                elif sample and all(len(s) <= 3 and s.isalpha() for s in sample):
                    iso3_col = iso3_col or c  # alpha = ISO3

        if iso3_col:
            food = food.rename(columns={iso3_col: "iso3"})
            food["iso3"] = food["iso3"].astype(str).str.strip().str.upper()
        elif m49_col:
            logger.info("FAOSTAT uses M49 numeric codes; converting to ISO3")
            # Build M49 → ISO3 mapping
            try:
                import pycountry
                m49_to_iso3 = {}
                for c_obj in pycountry.countries:
                    m49_to_iso3[int(c_obj.numeric)] = c_obj.alpha_3
                food["iso3"] = (
                    pd.to_numeric(food[m49_col], errors="coerce")
                    .astype("Int64")
                    .map(m49_to_iso3)
                )
            except ImportError:
                # Fallback: build from our GW↔ISO3 crosswalk + hardcoded M49 map
                logger.info("pycountry not available; using built-in M49→ISO3 map")
                from src.data._m49_map import M49_TO_ISO3
                food["iso3"] = (
                    pd.to_numeric(food[m49_col], errors="coerce")
                    .astype("Int64")
                    .map(M49_TO_ISO3)
                )
            mapped = food["iso3"].notna().sum()
            total = len(food)
            logger.info("M49→ISO3 mapping: %d/%d rows mapped (%.1f%%)", mapped, total, mapped/total*100)
        else:
            raise ValueError(
                "FAOSTAT CSV must contain an area code column (M49 or ISO3). "
                f"Found columns: {list(food.columns)}"
            )

        # Parse month names to numbers
        # Prefer text "Months" column over numeric "Months Code"
        months_col = None
        for c in food.columns:
            if c.strip().lower() == "months":
                months_col = c
                break
        if months_col is None:
            for c in food.columns:
                if c.strip().lower() == "months code":
                    months_col = c
                    break

        if months_col:
            # Check if it's text ("January") or numeric
            sample_val = str(food[months_col].dropna().iloc[0]).strip().lower()
            if sample_val in MONTH_NAME_TO_NUM:
                food["month"] = food[months_col].str.strip().str.lower().map(MONTH_NAME_TO_NUM)
            else:
                food["month"] = pd.to_numeric(food[months_col], errors="coerce")
                # FAOSTAT "Months Code" uses 7001-7012 for Jan-Dec
                if food["month"].dropna().min() > 7000:
                    food["month"] = food["month"] - 7000
            food = food.dropna(subset=["month"]).copy()

        # Year column
        year_col = None
        for c in food.columns:
            if c.strip().lower() in ("year", "year code"):
                year_col = c
                break
        if year_col:
            food = food.rename(columns={year_col: "year"})

        # Value column
        value_col = None
        for c in food.columns:
            if c.strip().lower() == "value":
                value_col = c
                break
        if value_col:
            food = food.rename(columns={value_col: "food_cpi"})

        food["food_cpi"] = pd.to_numeric(food["food_cpi"], errors="coerce")

    else:
        # Generic format: normalise column names
        col_map = {}
        for c in food.columns:
            cl = c.lower().strip()
            if cl in ("country_code", "iso3", "iso", "area_code"):
                col_map[c] = "iso3"
            elif cl in ("year",):
                col_map[c] = "year"
            elif cl in ("month",):
                col_map[c] = "month"
            elif cl in ("food_cpi", "value", "cpi_food", "food_price_index"):
                col_map[c] = "food_cpi"
        food = food.rename(columns=col_map)

    # Map to GW codes
    food["gwcode"] = food["iso3"].map(ISO3_TO_GW)
    food = food.dropna(subset=["gwcode", "food_cpi"]).copy()
    food["gwcode"] = food["gwcode"].astype(int)

    # Create temporal index
    food["year_month"] = (
        food["year"].astype(int).astype(str)
        + "-"
        + food["month"].astype(int).astype(str).str.zfill(2)
    )

    # Filter to study period (with buffer for rolling windows)
    food = food[food["year"] >= YEAR_MIN - 1].copy()
    food = food.sort_values(["gwcode", "year_month"]).reset_index(drop=True)

    # ── Feature engineering ──────────────────────────────────────────

    # Apply t-1 lag to raw CPI before computing features
    # (prevents look-ahead in the rolling windows)
    food["food_cpi_lagged"] = food.groupby("gwcode")["food_cpi"].shift(1)

    # Food price anomaly: ratio to 12-month rolling mean
    food["food_price_anomaly"] = (
        food["food_cpi_lagged"]
        / food.groupby("gwcode")["food_cpi_lagged"].transform(
            lambda x: x.rolling(12, min_periods=6).mean()
        )
    )

    # Year-on-year CPI change
    food["food_cpi_yoy"] = food.groupby("gwcode")["food_cpi"].pct_change(12)

    # CPI acceleration: month-on-month change in YoY rate
    food["food_cpi_acceleration"] = food.groupby("gwcode")["food_cpi_yoy"].diff(1)

    # Log-transform the anomaly ratio to handle right-skewed spikes.
    # Food crises produce extreme anomaly values (e.g., 2.0+ during
    # hyperinflation). log(anomaly) compresses these while preserving
    # the sign: log(1.3) > 0 (above trend), log(0.9) < 0 (below trend).
    food["food_price_anomaly_log"] = np.log(food["food_price_anomaly"].clip(lower=1e-6))

    # Food price spike flag: anomaly > 1.15 (15% above 12-month trend).
    # Threshold chosen based on Bellemare (2015): protest onset probability
    # increases significantly when food prices exceed ~15% above trend.
    food["food_price_spike"] = (food["food_price_anomaly"] > 1.15).astype(int)

    # Filter to study period
    food = food[food["year_month"] >= f"{YEAR_MIN}-01"].copy()

    output_cols = [
        "gwcode", "year_month",
        "food_price_anomaly", "food_price_anomaly_log",
        "food_cpi_acceleration", "food_price_spike",
    ]
    food = food[output_cols].copy()

    logger.info(
        "Food prices processed: %d rows, %d countries",
        len(food), food["gwcode"].nunique(),
    )

    return food


# ══════════════════════════════════════════════════════════════════════
# Unified economic ingestion
# ══════════════════════════════════════════════════════════════════════

def ingest_all_economic(
    fx_csv: Optional[str | Path] = None,
    gdp_csv: Optional[str | Path] = None,
    food_csv: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Ingest all available economic data sources. Gracefully skips
    sources whose CSVs are not provided or not found.

    Returns
    -------
    Dict mapping source name to processed DataFrame
    """
    results = {}

    if fx_csv is not None:
        try:
            results["fx"] = ingest_exchange_rates(fx_csv)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Exchange rate ingestion skipped: %s", e)

    if gdp_csv is not None:
        try:
            results["gdp"] = ingest_gdp(gdp_csv)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("GDP ingestion skipped: %s", e)

    if food_csv is not None:
        try:
            results["food"] = ingest_food_prices(food_csv)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Food price ingestion skipped: %s", e)

    logger.info("Economic ingestion complete: %d sources loaded", len(results))
    return results
