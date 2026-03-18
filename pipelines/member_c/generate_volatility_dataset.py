"""
Member C Data Acquisition & Exploration Script
================================================
Downloads, explores, and profiles:
  1. GPR (Geopolitical Risk Index) — global + country-level monthly data
  2. GDELT tone/sentiment — via Google BigQuery (all countries in one query)

Then runs:
  Task 1: Missingness Audit
  Task 2: Temporal Alignment (lag application, panel reshaping)
  Task 3: Distribution Profiling (stats, transforms, correlations, autocorrelation)
  + Feature Registry output

Requirements:
  pip install pandas openpyxl xlrd requests google-cloud-bigquery db-dtypes \
              matplotlib seaborn scipy

Setup for BigQuery:
  1. Create a Google Cloud project (free tier gives 1TB/month of queries)
  2. Enable the BigQuery API
  3. Authenticate via one of:
     a) Run: gcloud auth application-default login
     b) Set env var: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  4. Set your project ID in the GCP_PROJECT variable below

Output:
  - data/gpr_global.csv         → Monthly global GPR, GPRT (threats), GPRA (acts)
  - data/gpr_country.csv        → Monthly country-level GPR for ~44 countries
  - data/gdelt_tone.csv         → Monthly tone per country (from BigQuery)
  - data/fips_to_iso_mapping.csv→ Mapping file for GDELT country codes
  - output/missingness/         → Heatmaps and reports per source
  - output/profiles/            → Distribution profiles per feature
  - output/feature_registry.csv → Completed feature registry
  - output/member_c_final.csv → Final (country, month) panel with t-1 lag

Notes:
  - GPR data is freely available from Caldara & Iacoviello (Fed Board of Governors)
  - GDELT is free on BigQuery; Google free tier = 1TB queries/month (more than enough)
  - BigQuery pulls ALL countries at once — no looping, no rate limits
  - Set USE_DOC_API_FALLBACK = True if you can't set up GCP
"""

import os
import warnings
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from scipy import stats as scipystats
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — adjust these to match your panel
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_c")
REPORT_DIR = os.path.join(BASE_DIR, "analysis", "member_c")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "missingness"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "profiles"), exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Date range for the analysis panel (filter GPR to relevant period)
PANEL_START = "1985-01"
PANEL_END = "2025-12"

# Redundant features to drop (identified by Spearman |r| > 0.9)
# event_count ↔ total_articles (0.997), hostile_event_count (0.978), cooperative_event_count (0.994)
# gpr_global ↔ gpr_threats (0.912)
# Keep: event_count (most interpretable volume proxy), gpr_global (benchmark index)
REDUNDANT_FEATURES = [
    "total_articles",           # r=0.997 with event_count
    "hostile_event_count",      # r=0.978 with event_count
    "cooperative_event_count",  # r=0.994 with event_count
    "gpr_threats",              # r=0.912 with gpr_global
    "goldstein_min",            # near-constant (-10 for 99%+ of rows)
    # Brent oil — r=0.982 with WTI, and WTI has better coverage (2000 vs 2007)
    "brent_oil_mean", "brent_oil_vol", "brent_oil_close", "brent_oil_pct_chg",
    # _close columns — r>0.99 with _mean for every macro indicator
    "vix_close", "wti_oil_close", "gold_close", "dxy_close",
    "us_10y_yield_close", "wheat_close", "copper_close", "us_13w_tbill_close",
]

# Features to log1p-transform (heavy-tailed, non-negative)
LOG1P_FEATURES = [
    "gpr_global", "gpr_acts", "gpr_country",
    "event_count", "tone_std",
]

# ISO3-to-FIPS mapping for merging GDELT (FIPS codes) with GPR (ISO3 codes)
# Inverted from the FIPS-to-ISO mapping in save_fips_iso_mapping()
ISO3_TO_FIPS = {
    "ARG": "AR", "AUS": "AS", "BEL": "BE", "BRA": "BR", "CAN": "CA",
    "CHE": "SZ", "CHL": "CI", "CHN": "CH", "COL": "CO", "DEU": "GM",
    "DNK": "DA", "EGY": "EG", "ESP": "SP", "FIN": "FI", "FRA": "FR",
    "GBR": "UK", "HKG": "HK", "HUN": "HU", "IDN": "ID", "IND": "IN",
    "ISR": "IS", "ITA": "IT", "JPN": "JA", "KOR": "KS", "MEX": "MX",
    "MYS": "MY", "NLD": "NL", "NOR": "NO", "PER": "PE", "PHL": "RP",
    "POL": "PL", "PRT": "PO", "RUS": "RS", "SAU": "SA", "SWE": "SW",
    "THA": "TH", "TUN": "TS", "TUR": "TU", "TWN": "TW", "UKR": "UP",
    "USA": "US", "VEN": "VE", "VNM": "VM", "ZAF": "SF",
}

# Macro/volatility indicators from Yahoo Finance (global, monthly)
# Each entry: (ticker, feature_name, description)
MACRO_TICKERS = [
    ("^VIX",      "vix",            "CBOE Volatility Index (fear gauge)"),
    ("BZ=F",      "brent_oil",      "Brent crude oil price (USD/barrel)"),
    ("CL=F",      "wti_oil",        "WTI crude oil price (USD/barrel)"),
    ("GC=F",      "gold",           "Gold futures price (USD/oz)"),
    ("DX-Y.NYB",  "dxy",            "US Dollar Index"),
    ("^TNX",      "us_10y_yield",   "US 10-Year Treasury yield"),
    ("ZW=F",      "wheat",          "Wheat futures price (USD/bushel)"),
    ("HG=F",      "copper",         "Copper futures price (USD/lb)"),
    ("^IRX",      "us_13w_tbill",   "US 13-week T-bill yield (Fed Funds proxy)"),
]

# Your Google Cloud project ID (needed for BigQuery billing)
# Free tier: 1TB of query processing per month
GCP_PROJECT = "deeplearninggdelt"

# Date range for GDELT queries
GDELT_START_YEAR = 1985
GDELT_END_YEAR = 2025

# Set to True to use the GDELT Doc API fallback instead of BigQuery
# (slower, rate-limited, but no GCP setup needed)
USE_DOC_API_FALLBACK = False


# ============================================================
# 1. GPR DATA — download from Caldara & Iacoviello
# ============================================================

def download_gpr_global():
    """
    Downloads the global monthly GPR index (benchmark, 1985–present).
    Source: https://www.matteoiacoviello.com/gpr.htm

    The Excel file contains:
      - GPR: benchmark geopolitical risk index
      - GPRT: geopolitical threats sub-index
      - GPRA: geopolitical acts sub-index
    """
    print("=" * 60)
    print("Downloading Global GPR Index...")
    print("=" * 60)

    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_excel(BytesIO(response.content))
        print(f"  Downloaded: {len(df)} rows, columns: {list(df.columns)}")

        out_path = os.path.join(OUTPUT_DIR, "gpr_global.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")
        return df

    except Exception as e:
        print(f"  ERROR downloading global GPR: {e}")
        print("  Try manually downloading from: https://www.matteoiacoviello.com/gpr.htm")
        return None


def download_gpr_country():
    """
    Downloads country-specific monthly GPR indices (~44 countries).
    Source: https://www.matteoiacoviello.com/gpr_country.htm
    """
    print("\n" + "=" * 60)
    print("Downloading Country-Level GPR Index...")
    print("=" * 60)

    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_country_export.xls"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_excel(BytesIO(response.content))
        print(f"  Downloaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Countries available: {[c for c in df.columns if c.lower() not in ['month', 'date']]}")

        out_path = os.path.join(OUTPUT_DIR, "gpr_country.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")
        return df

    except Exception as e:
        print(f"  ERROR downloading country GPR: {e}")
        print("  Try manually: https://www.matteoiacoviello.com/gpr_country.htm")
        return None


# ============================================================
# 2. GDELT TONE DATA — via Google BigQuery (primary method)
# ============================================================

def download_gdelt_tone_bigquery(project_id, start_year, end_year):
    """
    Downloads monthly tone statistics for ALL countries in a single BigQuery query.

    Uses the public GDELT v2 events table on BigQuery:
      `gdelt-bq.gdeltv2.events`

    Key columns used:
      - ActionGeo_CountryCode: FIPS country code where the event took place
      - AvgTone: average tone of all source articles for this event (-100 to +100)
      - SQLDATE: date of the event (YYYYMMDD integer)
      - Year: year of the event
      - NumArticles: number of source articles (used as weight)
      - GoldsteinScale: cooperative (+10) to hostile (-10) scale

    Returns a DataFrame with one row per (country_code, year_month) containing:
      - tone_mean: article-count-weighted average tone
      - tone_min: most negative single-event tone (captures worst days)
      - tone_max: most positive single-event tone
      - tone_std: standard deviation of tone (sentiment volatility)
      - event_count: number of events recorded
      - total_articles: total source articles (proxy for media coverage volume)
      - goldstein_mean/min: cooperative-hostile scale averages
      - hostile/cooperative event counts

    Cost estimate: ~2-5 GB per query (well within the 1TB/month free tier).
    """
    print("\n" + "=" * 60)
    print("Downloading GDELT Tone via BigQuery...")
    print("=" * 60)

    try:
        from google.cloud import bigquery
    except ImportError:
        print("  ERROR: google-cloud-bigquery not installed.")
        print("  Run: pip install google-cloud-bigquery db-dtypes")
        return None

    date_fmt = "%Y%m%d"
    month_fmt = "%Y-%m"
    query = f"""
    SELECT
        ActionGeo_CountryCode AS country_code,
        FORMAT_DATE('{month_fmt}', SAFE.PARSE_DATE('{date_fmt}', CAST(SQLDATE AS STRING))) AS year_month,

        -- Weighted average tone (weighted by number of source articles)
        SAFE_DIVIDE(
            SUM(AvgTone * NumArticles),
            SUM(NumArticles)
        ) AS tone_mean,

        -- Extremes
        MIN(AvgTone) AS tone_min,
        MAX(AvgTone) AS tone_max,

        -- Volatility of tone within the month
        STDDEV(AvgTone) AS tone_std,

        -- Volume metrics
        COUNT(*) AS event_count,
        SUM(NumArticles) AS total_articles,

        -- Cooperative vs hostile breakdown (Goldstein scale)
        COUNTIF(GoldsteinScale < -5) AS hostile_event_count,
        COUNTIF(GoldsteinScale > 5) AS cooperative_event_count,
        AVG(GoldsteinScale) AS goldstein_mean,
        MIN(GoldsteinScale) AS goldstein_min

    FROM (
        -- GDELT v1 (1979–2015) and v2 (2015–present) combined
        SELECT SQLDATE, ActionGeo_CountryCode, AvgTone, NumArticles, GoldsteinScale, Year
        FROM `gdelt-bq.full.events`
        WHERE Year BETWEEN {start_year} AND 2014

        UNION ALL

        SELECT SQLDATE, ActionGeo_CountryCode, AvgTone, NumArticles, GoldsteinScale, Year
        FROM `gdelt-bq.gdeltv2.events`
        WHERE Year BETWEEN 2015 AND {end_year}
    )

    WHERE
        ActionGeo_CountryCode IS NOT NULL
        AND ActionGeo_CountryCode != ''
        AND SAFE.PARSE_DATE('{date_fmt}', CAST(SQLDATE AS STRING)) IS NOT NULL

    GROUP BY
        country_code, year_month

    HAVING
        year_month IS NOT NULL

    ORDER BY
        country_code, year_month
    """

    print(f"  Querying GDELT events table ({start_year}-{end_year})...")
    print(f"  This pulls ALL countries in one query.")

    try:
        client = bigquery.Client(project=project_id)
        df = client.query(query).to_dataframe()

        print(f"  Retrieved: {len(df)} country-month rows")
        print(f"  Countries: {df['country_code'].nunique()}")
        print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")

        print("\n  NOTE: GDELT uses FIPS country codes, not ISO.")
        print("  e.g. 'UK' not 'GB', 'AS' not 'AU', 'BM' not 'MM'.")
        print("  Use data/fips_to_iso_mapping.csv to convert.")

        out_path = os.path.join(OUTPUT_DIR, "gdelt_tone.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")

        save_fips_iso_mapping()
        return df

    except Exception as e:
        print(f"  ERROR running BigQuery: {e}")
        print("\n  Troubleshooting:")
        print("  1. pip install google-cloud-bigquery db-dtypes")
        print("  2. Run: gcloud auth application-default login")
        print(f"  3. Check project ID: '{project_id}'")
        print("  4. Enable BigQuery API in GCP console")
        return None


def save_fips_iso_mapping():
    """
    Saves a FIPS-to-ISO country code mapping file.

    GDELT uses FIPS 10-4 country codes, which differ from ISO 3166-1 alpha-2.
    This mapping is essential when joining GDELT data with ViEWS (which uses GW codes).

    Your integration pipeline should: GDELT FIPS -> ISO alpha-2 -> GW code
    """
    fips_to_iso = {
        "AF": "AF", "AL": "AL", "AG": "DZ", "AO": "AO", "AC": "AG",
        "AR": "AR", "AM": "AM", "AS": "AU", "AU": "AT", "AJ": "AZ",
        "BA": "BH", "BG": "BD", "BB": "BB", "BO": "BY", "BE": "BE",
        "BH": "BZ", "BN": "BJ", "BT": "BT", "BL": "BO", "BK": "BA",
        "BC": "BW", "BR": "BR", "BX": "BN", "BU": "BG", "UV": "BF",
        "BY": "BI", "CB": "KH", "CM": "CM", "CA": "CA", "CV": "CV",
        "CT": "CF", "CD": "TD", "CI": "CL", "CH": "CN", "CO": "CO",
        "CN": "KM", "CG": "CD", "CF": "CG", "CS": "CR", "IV": "CI",
        "HR": "HR", "CU": "CU", "CY": "CY", "EZ": "CZ", "DA": "DK",
        "DJ": "DJ", "DR": "DO", "EC": "EC", "EG": "EG", "ES": "SV",
        "EK": "GQ", "ER": "ER", "EN": "EE", "ET": "ET", "FJ": "FJ",
        "FI": "FI", "FR": "FR", "GB": "GA", "GA": "GM", "GG": "GE",
        "GM": "DE", "GH": "GH", "GR": "GR", "GT": "GT", "GV": "GN",
        "PU": "GW", "GY": "GY", "HA": "HT", "HO": "HN", "HU": "HU",
        "IC": "IS", "IN": "IN", "ID": "ID", "IR": "IR", "IZ": "IQ",
        "EI": "IE", "IS": "IL", "IT": "IT", "JM": "JM", "JA": "JP",
        "JO": "JO", "KZ": "KZ", "KE": "KE", "KN": "KP", "KS": "KR",
        "KU": "KW", "KG": "KG", "LA": "LA", "LG": "LV", "LE": "LB",
        "LT": "LS", "LI": "LR", "LY": "LY", "LH": "LT", "LU": "LU",
        "MK": "MK", "MA": "MG", "MI": "MW", "MY": "MY", "MV": "MV",
        "ML": "ML", "MT": "MT", "MR": "MR", "MP": "MU", "MX": "MX",
        "MD": "MD", "MG": "MN", "MJ": "ME", "MO": "MA", "MZ": "MZ",
        "BM": "MM", "WA": "NA", "NP": "NP", "NL": "NL", "NZ": "NZ",
        "NU": "NI", "NG": "NE", "NI": "NG", "NO": "NO", "MU": "OM",
        "PK": "PK", "PM": "PA", "PP": "PG", "PA": "PY", "PE": "PE",
        "RP": "PH", "PL": "PL", "PO": "PT", "QA": "QA", "RO": "RO",
        "RS": "RU", "RW": "RW", "SA": "SA", "SG": "SN", "RI": "RS",
        "SE": "SC", "SL": "SL", "SN": "SG", "LO": "SK", "SI": "SI",
        "BP": "SB", "SO": "SO", "SF": "ZA", "OD": "SS", "SP": "ES",
        "CE": "LK", "SU": "SD", "NS": "SR", "WZ": "SZ", "SW": "SE",
        "SZ": "CH", "SY": "SY", "TW": "TW", "TI": "TJ", "TZ": "TZ",
        "TH": "TH", "TT": "TL", "TO": "TG", "TD": "TT", "TS": "TN",
        "TU": "TR", "TX": "TM", "UG": "UG", "UP": "UA", "AE": "AE",
        "UK": "GB", "US": "US", "UY": "UY", "UZ": "UZ", "VE": "VE",
        "VM": "VN", "YM": "YE", "ZA": "ZM", "ZI": "ZW",
    }

    mapping_df = pd.DataFrame(
        list(fips_to_iso.items()),
        columns=["fips_code", "iso_code"]
    )

    out_path = os.path.join(OUTPUT_DIR, "fips_to_iso_mapping.csv")
    mapping_df.to_csv(out_path, index=False)
    print(f"  FIPS-to-ISO mapping saved to: {out_path}")


# ============================================================
# 2b. FALLBACK — GDELT Doc API (if BigQuery not available)
# ============================================================

# Countries to query if using the Doc API fallback (ISO-2 codes)
# Expand to match your ViEWS panel
DOC_API_COUNTRIES = [
    "AF", "IQ", "SY", "YE", "SO", "SS", "CD", "NG", "ML", "UA",
    "MM", "ET", "LY", "SD", "CF", "MZ", "PK", "CM", "BF", "TD",
]


def download_gdelt_tone_doc_api(start_date, end_date, countries):
    """
    Fallback method using the GDELT Doc 2.0 API.
    Much slower than BigQuery (one HTTP request per country per year chunk),
    but requires no GCP setup.

    Only use this if BigQuery is not an option.
    """
    import time
    from datetime import datetime, timedelta

    print("\n" + "=" * 60)
    print("Downloading GDELT Tone via Doc API (fallback)...")
    print("=" * 60)
    print("  WARNING: This is much slower than BigQuery.")
    print(f"  Querying {len(countries)} countries — expect ~{len(countries) * 2} minutes.\n")

    all_tone = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    for i, country_code in enumerate(countries):
        print(f"  [{i+1}/{len(countries)}] {country_code}", end="")

        current_start = start
        while current_start < end:
            current_end = min(current_start + timedelta(days=365), end)
            sd = current_start.strftime("%Y%m%d") + "000000"
            ed = current_end.strftime("%Y%m%d") + "235959"

            params = {
                "query": f"sourcecountry:{country_code}",
                "mode": "timelinetone",
                "startdatetime": sd,
                "enddatetime": ed,
                "format": "csv",
            }

            try:
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()

                if response.text.strip():
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    df["country_code"] = country_code
                    all_tone.append(df)
                    print(".", end="", flush=True)

            except Exception as e:
                print(f"x", end="", flush=True)

            current_start = current_end + timedelta(days=1)
            time.sleep(2)

        print()  # Newline after each country

    if not all_tone:
        print("  No data retrieved.")
        return None

    raw_df = pd.concat(all_tone, ignore_index=True)

    # Aggregate to monthly
    date_col = raw_df.columns[0]
    raw_df[date_col] = pd.to_datetime(raw_df[date_col])
    raw_df["year_month"] = raw_df[date_col].dt.to_period("M").astype(str)

    numeric_cols = raw_df.select_dtypes(include="number").columns
    tone_col = numeric_cols[0] if len(numeric_cols) > 0 else raw_df.columns[1]

    monthly = raw_df.groupby(["country_code", "year_month"]).agg(
        tone_mean=(tone_col, "mean"),
        tone_min=(tone_col, "min"),
        tone_max=(tone_col, "max"),
        tone_std=(tone_col, "std"),
        days_with_data=(tone_col, "count"),
    ).reset_index()

    out_path = os.path.join(OUTPUT_DIR, "gdelt_tone.csv")
    monthly.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    return monthly


# ============================================================
# 3. MACRO / VOLATILITY INDICATORS — via Yahoo Finance
# ============================================================

def download_macro_indicators():
    """
    Downloads monthly macro/volatility indicators from Yahoo Finance.
    Aggregates daily data to monthly: close → mean and volatility (std).
    Also computes month-over-month % change for price features.
    """
    print("\n" + "=" * 60)
    print("Downloading Macro/Volatility Indicators (Yahoo Finance)...")
    print("=" * 60)

    all_monthly = []

    for ticker, feat_name, description in MACRO_TICKERS:
        print(f"  {feat_name:20s} ({ticker:10s}): ", end="", flush=True)
        try:
            data = yf.download(ticker, start="1984-01-01", end="2026-01-01", progress=False)
            if len(data) == 0:
                print("NO DATA")
                continue

            # Flatten multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Resample daily → monthly
            monthly = data["Close"].resample("MS").agg(["mean", "std", "last"]).rename(
                columns={"mean": f"{feat_name}_mean", "std": f"{feat_name}_vol", "last": f"{feat_name}_close"}
            )
            # Month-over-month % change on closing price
            monthly[f"{feat_name}_pct_chg"] = monthly[f"{feat_name}_close"].pct_change() * 100

            print(f"{monthly.index.min().date()} to {monthly.index.max().date()} ({len(monthly)} months)")
            all_monthly.append(monthly)

        except Exception as e:
            print(f"ERROR: {e}")

    if not all_monthly:
        print("  No macro data retrieved.")
        return None

    # Merge all indicators on date index
    merged = pd.concat(all_monthly, axis=1)
    merged.index.name = "date"
    merged["year_month"] = merged.index.to_period("M").astype(str)

    out_path = os.path.join(OUTPUT_DIR, "macro_indicators.csv")
    merged.to_csv(out_path)
    print(f"\n  Saved to: {out_path}")
    print(f"  Shape: {merged.shape}")
    print(f"  Features: {[c for c in merged.columns if c != 'year_month']}")

    return merged


def load_macro_indicators(path=None):
    """Load macro indicators CSV if it exists."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "macro_indicators.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "year_month" not in df.columns:
        df["year_month"] = pd.to_datetime(df.index).to_period("M").astype(str)
    return df


# ============================================================
# 4. LOAD LOCAL DATA (if already downloaded / from .xls)
# ============================================================

def load_gpr_from_xls(path=None):
    """Load GPR data from local .xls or .csv file and reshape to long format."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "gpr_data.xls")

    # Try .xls first, then fall back to gpr_global.csv (which has the same columns)
    if os.path.exists(path):
        df = pd.read_excel(path)
    else:
        csv_path = os.path.join(OUTPUT_DIR, "gpr_global.csv")
        if os.path.exists(csv_path):
            print(f"  GPR .xls not found, loading from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print(f"  GPR file not found at {path}")
            return None, None
    df["month"] = pd.to_datetime(df["month"])
    # Drop metadata rows
    df = df.drop(columns=["var_name", "var_label"], errors="ignore")

    # --- Global GPR features ---
    global_cols = ["month", "GPR", "GPRT", "GPRA"]
    gpr_global = df[global_cols].dropna(subset=["GPR"]).copy()
    gpr_global = gpr_global.rename(columns={
        "GPR": "gpr_global", "GPRT": "gpr_threats", "GPRA": "gpr_acts"
    })
    gpr_global["year_month"] = gpr_global["month"].dt.to_period("M").astype(str)

    # --- Country-level GPR: pivot wide -> long ---
    country_cols = [c for c in df.columns if c.startswith("GPRC_")]
    if not country_cols:
        return gpr_global, None

    country_df = df[["month"] + country_cols].dropna(subset=[country_cols[0]]).copy()
    country_long = country_df.melt(
        id_vars=["month"], value_vars=country_cols,
        var_name="country_iso3", value_name="gpr_country"
    )
    country_long["country_iso3"] = country_long["country_iso3"].str.replace("GPRC_", "")
    country_long["year_month"] = country_long["month"].dt.to_period("M").astype(str)

    return gpr_global, country_long


def load_gdelt_tone(path=None):
    """Load GDELT tone CSV if it exists."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "gdelt_tone.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


# ============================================================
# TASK 1: MISSINGNESS AUDIT
# ============================================================

def missingness_audit(gpr_global, gpr_country, gdelt_tone, macro=None):
    """
    For each source, compute missingness matrix, generate stats,
    produce heatmap, and flag informative missingness.
    """
    print("\n" + "=" * 60)
    print("  TASK 1: MISSINGNESS AUDIT")
    print("=" * 60)

    sources = {}

    # --- GPR Global ---
    if gpr_global is not None:
        mask = gpr_global[["year_month", "gpr_global", "gpr_threats", "gpr_acts"]].copy()
        mask = mask[(mask["year_month"] >= PANEL_START) & (mask["year_month"] <= PANEL_END)]
        feat_cols = ["gpr_global", "gpr_threats", "gpr_acts"]
        sources["gpr_global"] = (mask, feat_cols, "year_month", None)

    # --- GPR Country ---
    if gpr_country is not None:
        mask = gpr_country[(gpr_country["year_month"] >= PANEL_START) &
                           (gpr_country["year_month"] <= PANEL_END)].copy()
        sources["gpr_country"] = (mask, ["gpr_country"], "year_month", "country_iso3")

    # --- GDELT Tone ---
    if gdelt_tone is not None and len(gdelt_tone) > 0:
        mask = gdelt_tone.copy()
        if "year_month" in mask.columns:
            mask = mask[(mask["year_month"] >= PANEL_START) & (mask["year_month"] <= PANEL_END)]
        if len(mask) > 0:
            tone_feats = [c for c in mask.columns if c not in ["country_code", "year_month"]]
            sources["gdelt_tone"] = (mask, tone_feats, "year_month", "country_code")

    # --- Macro Indicators ---
    if macro is not None:
        mask = macro.copy()
        mask = mask[(mask["year_month"] >= PANEL_START) & (mask["year_month"] <= PANEL_END)]
        macro_feats = [c for c in mask.columns if c != "year_month"]
        if len(macro_feats) > 0:
            sources["macro"] = (mask, macro_feats, "year_month", None)

    for source_name, (df, feat_cols, time_col, country_col) in sources.items():
        print(f"\n  --- {source_name} ---")

        # Overall % missing per feature
        missing_pct = df[feat_cols].isnull().mean() * 100
        print(f"  Overall missingness:")
        for feat, pct in missing_pct.items():
            flag = " *** >40% — candidate for dropping" if pct > 40 else ""
            print(f"    {feat}: {pct:.1f}%{flag}")

        # Per-year missingness
        df["_year"] = df[time_col].str[:4]
        year_miss = df.groupby("_year")[feat_cols].apply(lambda x: x.isnull().mean() * 100)
        print(f"  Per-year missingness (first feature):")
        for yr, row in year_miss.iterrows():
            val = row.iloc[0] if hasattr(row, "iloc") else row
            if val > 0:
                print(f"    {yr}: {val:.1f}%")
        df.drop(columns=["_year"], inplace=True)

        # Missingness heatmap (country x time)
        if country_col is not None:
            pivot = df.pivot_table(
                index=country_col, columns=time_col,
                values=feat_cols[0], aggfunc="count"
            )
            # Convert to boolean: 0 = missing, 1 = present
            all_months = pd.period_range(PANEL_START, PANEL_END, freq="M").astype(str)
            pivot = pivot.reindex(columns=all_months, fill_value=0)
            is_present = (pivot > 0).astype(int)

            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(["#d32f2f", "#4caf50"])
            fig, ax = plt.subplots(figsize=(max(16, len(all_months) * 0.15), max(8, len(is_present) * 0.2)))
            sns.heatmap(is_present, cmap=cmap, vmin=0, vmax=1,
                        cbar_kws={"label": "0=Missing  1=Present", "ticks": [0, 1]},
                        ax=ax, xticklabels=12)
            ax.set_title(f"Missingness Heatmap: {source_name}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Country")
            plt.tight_layout()
            fig.savefig(os.path.join(REPORT_DIR, "missingness", f"{source_name}_heatmap.png"), dpi=150)
            plt.close(fig)
            print(f"  Heatmap saved: output/missingness/{source_name}_heatmap.png")

            # Per-country missingness summary
            country_miss_pct = (1 - is_present.mean(axis=1)) * 100
            countries_with_gaps = country_miss_pct[country_miss_pct > 0]
            if len(countries_with_gaps) > 0:
                print(f"  Countries with gaps:")
                for c, pct in countries_with_gaps.sort_values(ascending=False).head(10).items():
                    print(f"    {c}: {pct:.1f}% missing")
            else:
                print(f"  All {len(is_present)} countries have complete coverage in {PANEL_START}–{PANEL_END}")
        else:
            # Time-series only (global GPR) — bar chart of missingness over time
            miss_by_month = df.set_index(time_col)[feat_cols].isnull().mean(axis=1)
            if miss_by_month.sum() > 0:
                fig, ax = plt.subplots(figsize=(14, 4))
                miss_by_month.plot(kind="bar", ax=ax)
                ax.set_title(f"Missingness Over Time: {source_name}")
                plt.tight_layout()
                fig.savefig(os.path.join(REPORT_DIR, "missingness", f"{source_name}_heatmap.png"), dpi=150)
                plt.close(fig)
                print(f"  Heatmap saved: output/missingness/{source_name}_heatmap.png")
            else:
                print(f"  No missingness in panel period — no heatmap needed.")

        # Missingness report summary
        report = missing_pct.to_frame("pct_missing").reset_index()
        report.columns = ["feature", "pct_missing"]
        report["informative_missingness"] = False  # Will be updated when target data is available
        report.to_csv(os.path.join(REPORT_DIR, "missingness", f"{source_name}_report.csv"), index=False)
        print(f"  Report saved: output/missingness/{source_name}_report.csv")

    print("\n  NOTE: Cross-referencing missingness with conflict intensity requires")
    print("  the ViEWS target variable (UCDP fatalities). is_missing_* flags will be")
    print("  created during the final integration merge by Member C.")


# ============================================================
# TASK 2: TEMPORAL ALIGNMENT
# ============================================================

def temporal_alignment(gpr_global, gpr_country, gdelt_tone, macro=None):
    """
    All sources are already monthly.
    Steps: filter to panel period, apply t-1 lag, merge into panel, validate.
    """
    print("\n" + "=" * 60)
    print("  TASK 2: TEMPORAL ALIGNMENT")
    print("=" * 60)

    all_months = pd.period_range(PANEL_START, PANEL_END, freq="M").astype(str)
    panels = []

    # --- GPR Global (monthly, no country dimension — broadcast to all countries later) ---
    if gpr_global is not None:
        gpr_g = gpr_global[["year_month", "gpr_global", "gpr_threats", "gpr_acts"]].copy()
        gpr_g = gpr_g[(gpr_g["year_month"] >= PANEL_START) & (gpr_g["year_month"] <= PANEL_END)]
        gpr_g = gpr_g.sort_values("year_month").reset_index(drop=True)

        # Apply t-1 lag: shift features down by 1 row (row for month M gets features from M-1)
        for col in ["gpr_global", "gpr_threats", "gpr_acts"]:
            gpr_g[col] = gpr_g[col].shift(1)
        gpr_g = gpr_g.dropna(subset=["gpr_global"])

        print(f"  GPR Global: {len(gpr_g)} months after t-1 lag")
        print(f"    Range: {gpr_g['year_month'].min()} to {gpr_g['year_month'].max()}")
        panels.append(("gpr_global", gpr_g))

    # --- GPR Country (monthly, long format) ---
    if gpr_country is not None:
        gpr_c = gpr_country[["year_month", "country_iso3", "gpr_country"]].copy()
        gpr_c = gpr_c[(gpr_c["year_month"] >= PANEL_START) & (gpr_c["year_month"] <= PANEL_END)]

        # Apply t-1 lag per country
        gpr_c = gpr_c.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)
        gpr_c["gpr_country"] = gpr_c.groupby("country_iso3")["gpr_country"].shift(1)
        gpr_c = gpr_c.dropna(subset=["gpr_country"])

        print(f"  GPR Country: {len(gpr_c)} country-months after t-1 lag")
        print(f"    Countries: {gpr_c['country_iso3'].nunique()}")
        panels.append(("gpr_country", gpr_c))

    # --- GDELT Tone (monthly, already per country) ---
    if gdelt_tone is not None:
        gt = gdelt_tone.copy()
        country_col = "country_code"
        if "year_month" in gt.columns:
            gt = gt[(gt["year_month"] >= PANEL_START) & (gt["year_month"] <= PANEL_END)]
        tone_feats = [c for c in gt.columns if c not in [country_col, "year_month"]]

        # Apply t-1 lag per country
        gt = gt.sort_values([country_col, "year_month"]).reset_index(drop=True)
        for col in tone_feats:
            gt[col] = gt.groupby(country_col)[col].shift(1)
        gt = gt.dropna(subset=[tone_feats[0]])

        print(f"  GDELT Tone: {len(gt)} country-months after t-1 lag")
        print(f"    Countries: {gt[country_col].nunique()}")
        panels.append(("gdelt_tone", gt))

    # --- Merge into a single panel ---
    if not panels:
        print("  No data to align.")
        return None

    # Start with GPR country as base (it has the country dimension)
    if gpr_country is not None:
        merged = gpr_country[["year_month", "country_iso3", "gpr_country"]].copy()
        merged = merged[(merged["year_month"] >= PANEL_START) & (merged["year_month"] <= PANEL_END)]
        merged = merged.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)
        merged["gpr_country"] = merged.groupby("country_iso3")["gpr_country"].shift(1)

        # Merge global GPR (broadcast to all countries)
        if gpr_global is not None:
            global_cols = ["gpr_global", "gpr_threats", "gpr_acts"]
            gpr_g_lagged = gpr_global[["year_month"] + global_cols].copy()
            gpr_g_lagged = gpr_g_lagged[(gpr_g_lagged["year_month"] >= PANEL_START) &
                                         (gpr_g_lagged["year_month"] <= PANEL_END)]
            gpr_g_lagged = gpr_g_lagged.sort_values("year_month").reset_index(drop=True)
            for col in global_cols:
                gpr_g_lagged[col] = gpr_g_lagged[col].shift(1)
            merged = merged.merge(gpr_g_lagged, on="year_month", how="left")

        # Merge GDELT tone via ISO3→FIPS mapping
        if gdelt_tone is not None and len(gdelt_tone) > 0:
            gt_lagged = gdelt_tone.copy()
            gt_lagged = gt_lagged[(gt_lagged["year_month"] >= PANEL_START) &
                                   (gt_lagged["year_month"] <= PANEL_END)]
            tone_feats = [c for c in gt_lagged.columns if c not in ["country_code", "year_month"]]
            gt_lagged = gt_lagged.sort_values(["country_code", "year_month"]).reset_index(drop=True)
            for col in tone_feats:
                gt_lagged[col] = gt_lagged.groupby("country_code")[col].shift(1)

            # Map ISO3 → FIPS so we can join
            merged["_fips"] = merged["country_iso3"].map(ISO3_TO_FIPS)
            merged = merged.merge(
                gt_lagged, left_on=["_fips", "year_month"],
                right_on=["country_code", "year_month"], how="left"
            )
            merged.drop(columns=["_fips", "country_code"], inplace=True)
            gdelt_matched = merged[tone_feats[0]].notna().sum()
            print(f"  GDELT tone merged: {gdelt_matched}/{len(merged)} rows matched via FIPS mapping")

        # Merge macro indicators (global — broadcast to all countries)
        if macro is not None:
            macro_lagged = macro.copy()
            macro_lagged = macro_lagged[(macro_lagged["year_month"] >= PANEL_START) &
                                         (macro_lagged["year_month"] <= PANEL_END)]
            macro_feats = [c for c in macro_lagged.columns if c != "year_month"]
            macro_lagged = macro_lagged.sort_values("year_month").reset_index(drop=True)
            # Apply t-1 lag
            for col in macro_feats:
                macro_lagged[col] = macro_lagged[col].shift(1)
            merged = merged.merge(macro_lagged[["year_month"] + macro_feats],
                                  on="year_month", how="left")
            macro_matched = merged[macro_feats[0]].notna().sum()
            print(f"  Macro indicators merged: {macro_matched}/{len(merged)} rows matched")
            print(f"    Features added: {len(macro_feats)}")

        merged = merged.dropna(subset=["gpr_country"])
    else:
        merged = panels[0][1]

    # --- Drop redundant features ---
    to_drop = [f for f in REDUNDANT_FEATURES if f in merged.columns]
    if to_drop:
        merged.drop(columns=to_drop, inplace=True)
        print(f"  Dropped redundant: {to_drop}")

    # --- Apply log1p transforms ---
    log_applied = []
    for feat in LOG1P_FEATURES:
        if feat in merged.columns:
            merged[f"{feat}_log1p"] = np.log1p(merged[feat])
            log_applied.append(feat)
    if log_applied:
        print(f"  Applied log1p to: {log_applied}")

    # Save aligned panel
    out_path = os.path.join(OUTPUT_DIR, "member_c_final.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n  Volatility dataset saved: {out_path}")
    print(f"  Shape: {merged.shape}")

    # --- Spot-check validation ---
    print("\n  SPOT-CHECK VALIDATION (5 random country-months):")
    if len(merged) >= 5:
        sample = merged.sample(5, random_state=42)
        feat_cols = [c for c in merged.columns if c not in ["year_month", "country_iso3"]]
        for _, row in sample.iterrows():
            country = row.get("country_iso3", "GLOBAL")
            month = row.get("year_month", "N/A")
            gpr_val = row.get("gpr_country", "N/A")
            tone_val = row.get("tone_mean", "N/A")
            gpr_str = f"gpr={gpr_val:.3f}" if isinstance(gpr_val, float) else f"gpr={gpr_val}"
            tone_str = f"tone={tone_val:.3f}" if isinstance(tone_val, float) else f"tone={tone_val}"
            print(f"    {country} / {month}: {gpr_str}, {tone_str}")
        print("  --> Verify: value for month M should match raw value from month M-1")
        print(f"  Total features in panel: {len(feat_cols)}")

    return merged


# ============================================================
# TASK 3: DISTRIBUTION PROFILING
# ============================================================

def distribution_profiling(gpr_global, gpr_country, gdelt_tone, macro=None):
    """
    Compute summary stats, classify distributions, recommend transforms,
    run correlation and autocorrelation analysis.
    """
    print("\n" + "=" * 60)
    print("  TASK 3: DISTRIBUTION PROFILING")
    print("=" * 60)

    # Collect all features into one DataFrame for profiling
    all_features = {}

    if gpr_global is not None:
        gpr_g = gpr_global[(gpr_global["year_month"] >= PANEL_START) &
                            (gpr_global["year_month"] <= PANEL_END)]
        for col in ["gpr_global", "gpr_threats", "gpr_acts"]:
            if col in gpr_g.columns:
                all_features[col] = gpr_g[col].dropna().values

    if gpr_country is not None:
        gpr_c = gpr_country[(gpr_country["year_month"] >= PANEL_START) &
                             (gpr_country["year_month"] <= PANEL_END)]
        all_features["gpr_country"] = gpr_c["gpr_country"].dropna().values

    if gdelt_tone is not None and len(gdelt_tone) > 0:
        gt = gdelt_tone.copy()
        if "year_month" in gt.columns:
            gt = gt[(gt["year_month"] >= PANEL_START) & (gt["year_month"] <= PANEL_END)]
        if len(gt) > 0:
            tone_feats = [c for c in gt.columns if c not in ["country_code", "year_month"]]
            for col in tone_feats:
                vals = gt[col].dropna().values
                if len(vals) > 0:
                    all_features[col] = vals

    if macro is not None:
        m = macro.copy()
        m = m[(m["year_month"] >= PANEL_START) & (m["year_month"] <= PANEL_END)]
        macro_feats = [c for c in m.columns if c != "year_month"]
        for col in macro_feats:
            vals = m[col].dropna().values
            if len(vals) > 0:
                all_features[col] = vals

    if not all_features:
        print("  No features to profile.")
        return None

    # --- 1. Summary statistics ---
    profiles = []
    for feat_name, values in all_features.items():
        vals = pd.Series(values).astype(float)
        n = len(vals)
        profile = {
            "feature": feat_name,
            "n": n,
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
            "skewness": vals.skew(),
            "kurtosis": vals.kurtosis(),
            "pct_zero": (vals == 0).mean() * 100,
            "p01": vals.quantile(0.01),
            "p05": vals.quantile(0.05),
            "p25": vals.quantile(0.25),
            "p75": vals.quantile(0.75),
            "p95": vals.quantile(0.95),
            "p99": vals.quantile(0.99),
        }

        # --- 2. Classify distribution ---
        dist_type = "normal-ish"
        if profile["pct_zero"] > 50:
            dist_type = "zero-inflated"
        elif profile["kurtosis"] > 5 or (profile["median"] > 0 and profile["p99"] > 10 * profile["median"]):
            dist_type = "heavy-tailed"
        elif profile["std"] < 0.01 * abs(profile["mean"]) if profile["mean"] != 0 else False:
            dist_type = "near-zero-variance"

        # Check for bounded features (indices on 0-1 or similar)
        if profile["min"] >= 0 and profile["max"] <= 1.01:
            dist_type = "bounded"

        profile["distribution_type"] = dist_type

        # --- 3. Transform recommendation ---
        if dist_type == "heavy-tailed" and profile["min"] >= 0:
            profile["transform_rec"] = "log1p"
        elif dist_type == "zero-inflated" and profile["min"] >= 0:
            profile["transform_rec"] = "log1p"
        elif dist_type == "bounded":
            profile["transform_rec"] = "raw (bounded index)"
        elif dist_type == "near-zero-variance":
            profile["transform_rec"] = "drop candidate"
        else:
            profile["transform_rec"] = "raw"

        profiles.append(profile)

        # Print summary
        print(f"\n  --- {feat_name} ---")
        print(f"    n={n}, mean={profile['mean']:.3f}, median={profile['median']:.3f}, "
              f"std={profile['std']:.3f}")
        print(f"    skew={profile['skewness']:.2f}, kurtosis={profile['kurtosis']:.2f}, "
              f"zeros={profile['pct_zero']:.1f}%")
        print(f"    range=[{profile['min']:.3f}, {profile['max']:.3f}], "
              f"p99={profile['p99']:.3f}")
        print(f"    Type: {dist_type} | Transform: {profile['transform_rec']}")

    profiles_df = pd.DataFrame(profiles)

    # --- 4. Outlier analysis ---
    print("\n  OUTLIER ANALYSIS (values > 99.5th percentile):")
    for feat_name, values in all_features.items():
        vals = pd.Series(values).astype(float)
        threshold = vals.quantile(0.995)
        outliers = vals[vals > threshold]
        if len(outliers) > 0:
            print(f"    {feat_name}: {len(outliers)} outliers above {threshold:.2f}, "
                  f"max={outliers.max():.2f}")

    # --- 5. Correlation analysis ---
    if len(all_features) > 1:
        print("\n  CORRELATION ANALYSIS (Spearman, |r| > 0.7):")
        # Build a combined DataFrame — align by index for features with same length
        # For global vs country features, use global features (shorter)
        # We'll compute pairwise on available data
        feat_names = list(all_features.keys())
        corr_pairs = []
        for i in range(len(feat_names)):
            for j in range(i + 1, len(feat_names)):
                a = pd.Series(all_features[feat_names[i]])
                b = pd.Series(all_features[feat_names[j]])
                min_len = min(len(a), len(b))
                if min_len > 30:
                    r, p = scipystats.spearmanr(a.iloc[:min_len], b.iloc[:min_len])
                    if abs(r) > 0.7:
                        flag = " *** REDUNDANCY CANDIDATE" if abs(r) > 0.9 else ""
                        print(f"    {feat_names[i]} <-> {feat_names[j]}: r={r:.3f}{flag}")
                        corr_pairs.append((feat_names[i], feat_names[j], r))

        # Save correlation matrix heatmap
        if len(feat_names) >= 2:
            # Use the minimum length across all features
            min_n = min(len(v) for v in all_features.values())
            corr_df = pd.DataFrame({k: v[:min_n] for k, v in all_features.items()})
            corr_matrix = corr_df.corr(method="spearman")
            fig, ax = plt.subplots(figsize=(max(8, len(feat_names) * 0.8),
                                             max(6, len(feat_names) * 0.6)))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_title("Spearman Correlation Matrix")
            plt.tight_layout()
            fig.savefig(os.path.join(REPORT_DIR, "profiles", "correlation_matrix.png"), dpi=150)
            plt.close(fig)
            print(f"    Correlation heatmap saved: output/profiles/correlation_matrix.png")
    else:
        corr_pairs = []

    # --- 6. Temporal autocorrelation ---
    print("\n  TEMPORAL AUTOCORRELATION (lag-1, per country where applicable):")
    if gpr_country is not None:
        gpr_c = gpr_country[(gpr_country["year_month"] >= PANEL_START) &
                             (gpr_country["year_month"] <= PANEL_END)].copy()
        gpr_c = gpr_c.sort_values(["country_iso3", "year_month"])
        autocorrs = []
        for country, grp in gpr_c.groupby("country_iso3"):
            series = grp["gpr_country"].values
            if len(series) > 12:
                ac1 = pd.Series(series).autocorr(lag=1)
                autocorrs.append((country, ac1))
        if autocorrs:
            ac_df = pd.DataFrame(autocorrs, columns=["country", "autocorr_lag1"])
            mean_ac = ac_df["autocorr_lag1"].mean()
            high_ac = ac_df[ac_df["autocorr_lag1"] > 0.8]
            print(f"    gpr_country: mean lag-1 autocorr = {mean_ac:.3f}")
            print(f"    Countries with autocorr > 0.8: {len(high_ac)}/{len(ac_df)}")
            if mean_ac > 0.8:
                print(f"    --> Consider adding month-over-month DIFFERENCE as additional feature")

    if gpr_global is not None:
        gpr_g = gpr_global[(gpr_global["year_month"] >= PANEL_START) &
                            (gpr_global["year_month"] <= PANEL_END)]
        for col in ["gpr_global", "gpr_threats", "gpr_acts"]:
            if col in gpr_g.columns:
                series = gpr_g[col].dropna()
                if len(series) > 12:
                    ac1 = series.autocorr(lag=1)
                    ac3 = series.autocorr(lag=3)
                    ac12 = series.autocorr(lag=12)
                    print(f"    {col}: lag1={ac1:.3f}, lag3={ac3:.3f}, lag12={ac12:.3f}")

    # --- Distribution plots (histogram + KDE + log1p + boxplot) ---
    for feat_name, values in all_features.items():
        vals = pd.Series(values).astype(float)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: histogram
        axes[0, 0].hist(vals, bins=50, edgecolor="black", alpha=0.7, color="#2196F3")
        axes[0, 0].axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.2f}")
        axes[0, 0].axvline(vals.median(), color="orange", linestyle="--", label=f"median={vals.median():.2f}")
        axes[0, 0].legend()
        axes[0, 0].set_title(f"{feat_name} — Histogram")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")

        # Top-right: KDE
        try:
            vals.plot.kde(ax=axes[0, 1], color="#2196F3")
            axes[0, 1].set_title(f"{feat_name} — KDE")
        except Exception:
            axes[0, 1].text(0.5, 0.5, "KDE failed (too few unique values)", ha="center", va="center")

        # Bottom-left: log1p histogram (if applicable)
        if vals.min() >= 0:
            log_vals = np.log1p(vals)
            axes[1, 0].hist(log_vals, bins=50, edgecolor="black", alpha=0.7, color="#4CAF50")
            axes[1, 0].set_title(f"log(1 + {feat_name})")
            axes[1, 0].set_xlabel("log1p(Value)")
            axes[1, 0].set_ylabel("Frequency")
        else:
            axes[1, 0].hist(vals, bins=50, edgecolor="black", alpha=0.7, color="#FF9800")
            axes[1, 0].set_title(f"{feat_name} (contains negatives — no log1p)")

        # Bottom-right: boxplot
        axes[1, 1].boxplot(vals.dropna(), vert=True)
        axes[1, 1].set_title(f"{feat_name} — Boxplot")
        axes[1, 1].set_ylabel("Value")

        fig.suptitle(f"Distribution Profile: {feat_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "profiles", f"{feat_name}_dist.png"), dpi=150)
        plt.close(fig)

    # --- Time series plot for global features ---
    if gpr_global is not None:
        gpr_g = gpr_global[(gpr_global["year_month"] >= PANEL_START) &
                            (gpr_global["year_month"] <= PANEL_END)].copy()
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        for i, col in enumerate(["gpr_global", "gpr_threats", "gpr_acts"]):
            if col in gpr_g.columns:
                axes[i].plot(gpr_g["year_month"], gpr_g[col], linewidth=1.5)
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        axes[0].set_title("GPR Global Indices Over Time")
        axes[-1].set_xlabel("Month")
        # Show every 6th label to avoid clutter
        for ax in axes:
            ticks = ax.get_xticks()
            ax.set_xticks(ticks[::6] if len(ticks) > 12 else ticks)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "profiles", "gpr_global_timeseries.png"), dpi=150)
        plt.close(fig)

    # --- Country GPR heatmap (top 20 highest-risk countries) ---
    if gpr_country is not None:
        gpr_c = gpr_country[(gpr_country["year_month"] >= PANEL_START) &
                             (gpr_country["year_month"] <= PANEL_END)]
        country_means = gpr_c.groupby("country_iso3")["gpr_country"].mean().sort_values(ascending=False)
        top20 = country_means.head(20).index.tolist()
        top_data = gpr_c[gpr_c["country_iso3"].isin(top20)]
        pivot = top_data.pivot_table(index="country_iso3", columns="year_month",
                                      values="gpr_country", aggfunc="mean")
        pivot = pivot.loc[top20]  # preserve sort order
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(pivot, cmap="YlOrRd", ax=ax, xticklabels=6)
        ax.set_title("GPR Country Index — Top 20 Highest-Risk Countries")
        ax.set_xlabel("Month")
        ax.set_ylabel("Country (ISO3)")
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "profiles", "gpr_country_risk_heatmap.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Distribution plots saved to: output/profiles/")

    # Save profiles
    profiles_df.to_csv(os.path.join(REPORT_DIR, "profiles", "distribution_profiles.csv"), index=False)
    print(f"  Profile table saved: output/profiles/distribution_profiles.csv")

    return profiles_df, corr_pairs


# ============================================================
# FEATURE REGISTRY
# ============================================================

def build_feature_registry(profiles_df, corr_pairs):
    """Build the shared feature registry CSV."""
    print("\n" + "=" * 60)
    print("  FEATURE REGISTRY")
    print("=" * 60)

    if profiles_df is None:
        print("  No profiles available — skipping registry.")
        return

    # Map feature names to metadata
    source_map = {
        "gpr_global": "GPR", "gpr_threats": "GPR", "gpr_acts": "GPR",
        "gpr_country": "GPR",
        "tone_mean": "GDELT", "tone_min": "GDELT", "tone_max": "GDELT",
        "tone_std": "GDELT", "event_count": "GDELT", "total_articles": "GDELT",
        "hostile_event_count": "GDELT", "cooperative_event_count": "GDELT",
        "goldstein_mean": "GDELT", "goldstein_min": "GDELT",
    }
    # Auto-populate macro features
    for _, feat_name, _ in MACRO_TICKERS:
        for suffix in ["_mean", "_vol", "_close", "_pct_chg"]:
            source_map[f"{feat_name}{suffix}"] = "Yahoo Finance"

    resolution_map = {
        "gpr_global": "monthly", "gpr_threats": "monthly", "gpr_acts": "monthly",
        "gpr_country": "monthly",
    }
    # Macro indicators are aggregated from daily → monthly
    for _, feat_name, _ in MACRO_TICKERS:
        for suffix in ["_mean", "_vol", "_close", "_pct_chg"]:
            resolution_map[f"{feat_name}{suffix}"] = "daily (agg to monthly)"

    # Build correlation lookup
    corr_lookup = {}
    if corr_pairs:
        for a, b, r in corr_pairs:
            if abs(r) > 0.9:
                corr_lookup.setdefault(a, []).append(f"{b}({r:.2f})")
                corr_lookup.setdefault(b, []).append(f"{a}({r:.2f})")

    registry = []
    for _, row in profiles_df.iterrows():
        feat = row["feature"]

        # Determine status
        if feat in REDUNDANT_FEATURES:
            status = "drop"
            known_issues = f"redundant (r>0.9 with {corr_lookup.get(feat, ['?'])[0].split('(')[0]})"
        elif row["distribution_type"] == "near-zero-variance":
            status = "review"
            known_issues = "near-zero variance"
        else:
            status = "keep"
            known_issues = ""

        # Determine actual transform applied
        if feat in LOG1P_FEATURES:
            transform = "log1p"
        elif row["transform_rec"] == "raw (bounded index)":
            transform = "raw (bounded)"
        else:
            transform = "raw"

        entry = {
            "feature_name": feat,
            "source": source_map.get(feat, "GDELT"),
            "original_resolution": resolution_map.get(feat, "monthly"),
            "aggregation_method": "none (already monthly)",
            "lag_applied": "t-1",
            "transform": transform,
            "pct_missing": 0.0,
            "missingness_informative": False,
            "distribution_type": row["distribution_type"],
            "correlated_with": "; ".join(corr_lookup.get(feat, [])),
            "known_issues": known_issues,
            "status": status,
        }

        registry.append(entry)

    registry_df = pd.DataFrame(registry)
    out_path = os.path.join(OUTPUT_DIR, "feature_registry.csv")
    registry_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  {len(registry_df)} features registered")
    print()
    print(registry_df.to_string(index=False))

    return registry_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  MEMBER C — DATA PIPELINE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Report directory: {REPORT_DIR}/")
    print(f"  Panel period: {PANEL_START} to {PANEL_END}")
    print(f"  GDELT date range: {GDELT_START_YEAR}-{GDELT_END_YEAR}")
    print(f"  GDELT method: {'Doc API (fallback)' if USE_DOC_API_FALLBACK else 'BigQuery'}")
    print()

    # ==========================
    # STEP 0: Load local data
    # ==========================
    print("=" * 60)
    print("  LOADING LOCAL DATA")
    print("=" * 60)
    gpr_global, gpr_country = load_gpr_from_xls()
    gdelt_tone = load_gdelt_tone()
    macro = load_macro_indicators()

    if gpr_global is not None:
        print(f"  GPR Global: {len(gpr_global)} months loaded from .xls")
    if gpr_country is not None:
        print(f"  GPR Country: {gpr_country['country_iso3'].nunique()} countries, "
              f"{len(gpr_country)} rows")
    if gdelt_tone is not None:
        print(f"  GDELT Tone: {len(gdelt_tone)} rows loaded")
    else:
        print("  GDELT Tone: not found — run download or set USE_DOC_API_FALLBACK=True")
    if macro is not None:
        macro_feats = [c for c in macro.columns if c != "year_month"]
        print(f"  Macro Indicators: {len(macro)} months, {len(macro_feats)} features loaded")
    else:
        print("  Macro Indicators: not found — will download from Yahoo Finance")

    # ==========================
    # STEP 1: Download (optional — skip if data already local)
    # ==========================
    if gpr_global is None:
        print("\n  No local GPR data found — attempting download...")
        gpr_dl = download_gpr_global()
        download_gpr_country()
        gpr_global, gpr_country = load_gpr_from_xls()

    if gdelt_tone is None:
        print("\n  No GDELT tone data — attempting download...")
        if USE_DOC_API_FALLBACK:
            gdelt_tone = download_gdelt_tone_doc_api(
                f"{GDELT_START_YEAR}-01-01",
                f"{GDELT_END_YEAR}-12-31",
                DOC_API_COUNTRIES,
            )
        else:
            gdelt_tone = download_gdelt_tone_bigquery(
                GCP_PROJECT, GDELT_START_YEAR, GDELT_END_YEAR
            )

    if macro is None:
        macro = download_macro_indicators()
        if macro is not None:
            macro = load_macro_indicators()  # reload from CSV for consistent format

    # ==========================
    # TASK 1: Missingness Audit
    # ==========================
    missingness_audit(gpr_global, gpr_country, gdelt_tone, macro)

    # ==========================
    # TASK 2: Temporal Alignment
    # ==========================
    aligned = temporal_alignment(gpr_global, gpr_country, gdelt_tone, macro)

    # ==========================
    # TASK 3: Distribution Profiling
    # ==========================
    result = distribution_profiling(gpr_global, gpr_country, gdelt_tone, macro)

    # ==========================
    # Feature Registry
    # ==========================
    if result is not None:
        profiles_df, corr_pairs = result
        build_feature_registry(profiles_df, corr_pairs)

    # ==========================
    # Summary
    # ==========================
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print("  Outputs:")
    print(f"    Missingness reports: {REPORT_DIR}/missingness/")
    print(f"    Distribution profiles: {REPORT_DIR}/profiles/")
    print(f"    Volatility dataset: {OUTPUT_DIR}/member_c_final.csv")
    print(f"    Feature registry: {OUTPUT_DIR}/feature_registry.csv")
    print(f"    Macro raw data: {OUTPUT_DIR}/macro_indicators.csv")
    print()
    print("  NEXT STEPS:")
    print("  1. Review heatmaps and distribution plots")
    print("  2. Verify 5 spot-checks against raw data")
    print("  3. Share feature_registry.csv with team for integration")


if __name__ == "__main__":
    main()