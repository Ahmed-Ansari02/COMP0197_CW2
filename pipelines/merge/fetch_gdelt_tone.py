"""
Fetch GDELT tone data for ALL countries via BigQuery.

Member C's pipeline only stored tone for 44 GPR countries because it used
gpr_country as the merge scaffold. This script fetches the same GDELT tone
features for all ~200 countries directly.

Requires:
  pip install google-cloud-bigquery db-dtypes
  gcloud auth application-default login

Output: data/processed/merge/gdelt_tone_all.csv
  Columns: country_iso3, year_month, tone_mean, tone_min, tone_max,
           tone_std, event_count, goldstein_mean

These are the same features member C produced, just for all countries.
The merge pipeline reads this file and uses it to fill in the gaps.
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "merge"
FIPS_MAPPING_PATH = BASE_DIR / "data" / "processed" / "member_c" / "fips_to_iso_mapping.csv"

GCP_PROJECT = None  # loaded from config/config.yaml
START_YEAR = 1985
END_YEAR = 2025

# Columns to keep (same as member C's output, minus redundant ones)
KEEP_COLUMNS = [
    "tone_mean", "tone_min", "tone_max", "tone_std",
    "event_count", "goldstein_mean",
]


def build_fips_to_iso3() -> dict[str, str]:
    """Load FIPS-to-ISO3 mapping from member C's mapping file."""
    mapping = pd.read_csv(FIPS_MAPPING_PATH)
    # The file has fips_code -> iso_code (ISO2)
    # We need FIPS -> ISO3, so we'll use pycountry or a manual approach

    # Build FIPS -> ISO2 from the file
    fips_to_iso2 = dict(zip(mapping["fips_code"], mapping["iso_code"]))

    # ISO2 -> ISO3 mapping (standard)
    try:
        import pycountry
        iso2_to_iso3 = {}
        for c in pycountry.countries:
            iso2_to_iso3[c.alpha_2] = c.alpha_3
        # Some GDELT-specific overrides
        iso2_to_iso3["XK"] = "XKX"  # Kosovo
    except ImportError:
        # Fallback: build from member A/B data
        print("  pycountry not installed, building ISO2→ISO3 from panel data")
        iso2_to_iso3 = _build_iso2_to_iso3_fallback()

    # Chain: FIPS -> ISO2 -> ISO3
    fips_to_iso3 = {}
    for fips, iso2 in fips_to_iso2.items():
        if iso2 in iso2_to_iso3:
            fips_to_iso3[fips] = iso2_to_iso3[iso2]

    return fips_to_iso3


def _build_iso2_to_iso3_fallback() -> dict[str, str]:
    """Build ISO2→ISO3 from a hardcoded list of common countries."""
    # Standard ISO 3166 mapping for countries in our panel
    return {
        "AF": "AFG", "AL": "ALB", "DZ": "DZA", "AO": "AGO", "AR": "ARG",
        "AM": "ARM", "AU": "AUS", "AT": "AUT", "AZ": "AZE", "BH": "BHR",
        "BD": "BGD", "BY": "BLR", "BE": "BEL", "BJ": "BEN", "BT": "BTN",
        "BO": "BOL", "BA": "BIH", "BW": "BWA", "BR": "BRA", "BN": "BRN",
        "BG": "BGR", "BF": "BFA", "BI": "BDI", "KH": "KHM", "CM": "CMR",
        "CA": "CAN", "CV": "CPV", "CF": "CAF", "TD": "TCD", "CL": "CHL",
        "CN": "CHN", "CO": "COL", "KM": "COM", "CG": "COG", "CD": "COD",
        "CR": "CRI", "CI": "CIV", "HR": "HRV", "CU": "CUB", "CY": "CYP",
        "CZ": "CZE", "DK": "DNK", "DJ": "DJI", "DO": "DOM", "EC": "ECU",
        "EG": "EGY", "SV": "SLV", "GQ": "GNQ", "ER": "ERI", "EE": "EST",
        "ET": "ETH", "FI": "FIN", "FR": "FRA", "GA": "GAB", "GM": "GMB",
        "GE": "GEO", "DE": "DEU", "GH": "GHA", "GR": "GRC", "GT": "GTM",
        "GN": "GIN", "GW": "GNB", "GY": "GUY", "HT": "HTI", "HN": "HND",
        "HK": "HKG", "HU": "HUN", "IS": "ISL", "IN": "IND", "ID": "IDN",
        "IR": "IRN", "IQ": "IRQ", "IE": "IRL", "IL": "ISR", "IT": "ITA",
        "JM": "JAM", "JP": "JPN", "JO": "JOR", "KZ": "KAZ", "KE": "KEN",
        "KW": "KWT", "KG": "KGZ", "LA": "LAO", "LV": "LVA", "LB": "LBN",
        "LS": "LSO", "LR": "LBR", "LY": "LBY", "LT": "LTU", "MG": "MDG",
        "MW": "MWI", "MY": "MYS", "MV": "MDV", "ML": "MLI", "MR": "MRT",
        "MX": "MEX", "MD": "MDA", "MN": "MNG", "ME": "MNE", "MA": "MAR",
        "MZ": "MOZ", "MM": "MMR", "NA": "NAM", "NP": "NPL", "NL": "NLD",
        "NZ": "NZL", "NI": "NIC", "NE": "NER", "NG": "NGA", "KP": "PRK",
        "NO": "NOR", "OM": "OMN", "PK": "PAK", "PS": "PSE", "PA": "PAN",
        "PG": "PNG", "PY": "PRY", "PE": "PER", "PH": "PHL", "PL": "POL",
        "PT": "PRT", "QA": "QAT", "RO": "ROU", "RU": "RUS", "RW": "RWA",
        "SA": "SAU", "SN": "SEN", "RS": "SRB", "SL": "SLE", "SG": "SGP",
        "SK": "SVK", "SI": "SVN", "SO": "SOM", "ZA": "ZAF", "KR": "KOR",
        "SS": "SSD", "ES": "ESP", "LK": "LKA", "SD": "SDN", "SR": "SUR",
        "SZ": "SWZ", "SE": "SWE", "CH": "CHE", "SY": "SYR", "TW": "TWN",
        "TJ": "TJK", "TZ": "TZA", "TH": "THA", "TL": "TLS", "TG": "TGO",
        "TT": "TTO", "TN": "TUN", "TR": "TUR", "TM": "TKM", "UG": "UGA",
        "UA": "UKR", "AE": "ARE", "GB": "GBR", "US": "USA", "UY": "URY",
        "UZ": "UZB", "VE": "VEN", "VN": "VNM", "YE": "YEM", "ZM": "ZMB",
        "ZW": "ZWE", "XK": "XKX", "BZ": "BLZ",
    }


def fetch_gdelt_tone(project_id: str, start_year: int, end_year: int) -> pd.DataFrame | None:
    """Run BigQuery to get monthly tone stats for all countries."""
    try:
        from google.cloud import bigquery
    except ImportError:
        print("ERROR: google-cloud-bigquery not installed.")
        print("Run: pip install google-cloud-bigquery db-dtypes")
        return None

    date_fmt = "%Y%m%d"
    month_fmt = "%Y-%m"

    query = f"""
    SELECT
        ActionGeo_CountryCode AS country_code,
        FORMAT_DATE('{month_fmt}', SAFE.PARSE_DATE('{date_fmt}', CAST(SQLDATE AS STRING))) AS year_month,
        SAFE_DIVIDE(
            SUM(AvgTone * NumArticles),
            SUM(NumArticles)
        ) AS tone_mean,
        MIN(AvgTone) AS tone_min,
        MAX(AvgTone) AS tone_max,
        STDDEV(AvgTone) AS tone_std,
        COUNT(*) AS event_count,
        AVG(GoldsteinScale) AS goldstein_mean
    FROM (
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

    print(f"Querying GDELT BigQuery ({start_year}-{end_year})...")
    try:
        client = bigquery.Client(project=project_id)
        df = client.query(query).to_dataframe()
        print(f"  Retrieved: {len(df):,} rows, {df['country_code'].nunique()} FIPS countries")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        print("Troubleshooting:")
        print("  1. pip install google-cloud-bigquery db-dtypes")
        print("  2. gcloud auth application-default login")
        print(f"  3. Check project ID: '{project_id}'")
        return None


def convert_fips_to_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FIPS country codes to ISO3 and apply t-1 lag."""
    fips_to_iso3 = build_fips_to_iso3()

    df["country_iso3"] = df["country_code"].map(fips_to_iso3)

    n_before = len(df)
    unmapped = df[df["country_iso3"].isna()]["country_code"].unique()
    df = df.dropna(subset=["country_iso3"])
    print(f"  FIPS→ISO3: {len(df):,}/{n_before:,} rows mapped, {len(unmapped)} unmapped codes")

    if len(unmapped) > 0 and len(unmapped) <= 20:
        print(f"  Unmapped: {sorted(unmapped)}")

    # Apply t-1 lag per country (same convention as all other pipelines)
    df = df.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)
    tone_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    for col in tone_cols:
        df[col] = df.groupby("country_iso3")[col].shift(1)

    # Drop the FIPS code column
    df = df[["country_iso3", "year_month"] + tone_cols]

    print(f"  Final: {len(df):,} rows, {df['country_iso3'].nunique()} ISO3 countries")
    print(f"  Date range: {df['year_month'].min()} to {df['year_month'].max()}")

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "gdelt_tone_all.csv"

    # Load project ID from config
    global GCP_PROJECT
    import yaml
    config_path = BASE_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    GCP_PROJECT = config.get("gcp_project")
    if not GCP_PROJECT:
        print("ERROR: Set gcp_project in config/config.yaml")
        return None

    # Check if already fetched
    if output_path.exists():
        existing = pd.read_csv(output_path)
        print(f"Already fetched: {output_path} ({len(existing):,} rows)")
        print("Delete the file to re-fetch.")
        return existing

    # Fetch from BigQuery
    raw = fetch_gdelt_tone(GCP_PROJECT, START_YEAR, END_YEAR)
    if raw is None:
        return None

    # Convert FIPS → ISO3 + lag
    result = convert_fips_to_iso3(raw)

    # Filter to panel date range
    result = result[
        (result["year_month"] >= "1985-01") & (result["year_month"] <= "2025-12")
    ]

    # Save
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    main()
