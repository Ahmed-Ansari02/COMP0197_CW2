"""
Data Download Utilities
========================

Programmatic fetching of raw data from public APIs and repositories.
Each function downloads to data/raw/ and returns the local path.

Sources:
    - V-Dem: v-dem.net (manual download required — file is ~300MB)
    - REIGN: GitHub archive (OEF Data Science)
    - GDP: World Bank API (indicator NY.GDP.MKTP.KD.ZG)
    - Exchange rates: IMF IFS API or bulk download
    - Food prices: FAO FAOSTAT API

Note: V-Dem requires manual download due to file size and terms of use.
      All other sources can be fetched programmatically.
"""

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def ensure_raw_dir() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def download_reign(output_path: Optional[str | Path] = None) -> Path:
    """
    Download REIGN dataset from the archived GitHub repository.

    REIGN (Rulers, Elections, and Irregular Governance) ceased data
    collection in August 2021. The dataset is now hosted as a static
    archive.

    Returns path to downloaded CSV.
    """
    ensure_raw_dir()
    output_path = Path(output_path) if output_path else RAW_DIR / "REIGN_2021_8.csv"

    if output_path.exists() and output_path.stat().st_size > 100:
        logger.info("REIGN already downloaded: %s", output_path)
        return output_path

    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloads: pip install requests")

    url = "https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/data_sets/REIGN_2021_8.csv"

    logger.info("Downloading REIGN from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.info("REIGN saved to %s (%d bytes)", output_path, len(response.content))

    return output_path


def download_gdp_worldbank(
    output_path: Optional[str | Path] = None,
    start_year: int = 2005,
    end_year: int = 2024,
) -> Path:
    """
    Download GDP growth data from World Bank API.

    Uses indicator NY.GDP.MKTP.KD.ZG (GDP growth, annual %).
    Downloads for all countries via the CSV bulk-download endpoint
    (more reliable than the JSON API which can timeout or paginate).

    Falls back to JSON API with pagination if CSV endpoint fails.

    Returns path to downloaded CSV.
    """
    ensure_raw_dir()
    output_path = Path(output_path) if output_path else RAW_DIR / "gdp_growth_worldbank.csv"

    # Skip re-download only if file exists AND is non-empty
    if output_path.exists() and output_path.stat().st_size > 100:
        logger.info("GDP data already downloaded: %s", output_path)
        return output_path

    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloads: pip install requests")

    indicator = "NY.GDP.MKTP.KD.ZG"

    # Strategy 1: CSV bulk download (more reliable, no pagination needed)
    csv_url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?date={start_year}:{end_year}&format=csv&per_page=20000"
    )

    records = []

    try:
        logger.info("Downloading GDP from World Bank (CSV endpoint)...")
        response = requests.get(csv_url, timeout=180)
        response.raise_for_status()

        # CSV endpoint returns a ZIP file containing the data CSV
        import zipfile
        zip_path = RAW_DIR / "_gdp_temp.zip"
        zip_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the data CSV (not metadata)
            csv_names = [n for n in zf.namelist()
                         if n.endswith(".csv") and "Metadata" not in n]
            if csv_names:
                with zf.open(csv_names[0]) as f:
                    raw = pd.read_csv(f)

                # World Bank CSV format: Country Code (ISO3), year columns
                id_col = [c for c in raw.columns if "Country Code" in c]
                if id_col:
                    iso3_col = id_col[0]
                    year_cols = [c for c in raw.columns if c.strip().isdigit()]
                    for _, row in raw.iterrows():
                        iso3 = str(row[iso3_col]).strip()
                        if len(iso3) != 3:
                            continue
                        for yc in year_cols:
                            val = row[yc]
                            if pd.notna(val):
                                try:
                                    records.append({
                                        "iso3": iso3,
                                        "year": int(yc.strip()),
                                        "gdp_growth": float(val),
                                    })
                                except (ValueError, TypeError):
                                    continue

        zip_path.unlink(missing_ok=True)

    except Exception as e:
        logger.warning("CSV endpoint failed (%s), trying JSON API...", e)

        # Strategy 2: JSON API with pagination
        base_url = "https://api.worldbank.org/v2"
        page = 1
        total_pages = 1

        while page <= total_pages:
            url = (
                f"{base_url}/country/all/indicator/{indicator}"
                f"?date={start_year}:{end_year}&format=json"
                f"&per_page=5000&page={page}"
            )
            try:
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
                data = resp.json()
            except Exception as api_err:
                logger.warning("JSON API page %d failed: %s", page, api_err)
                break

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                break

            # Update total pages from pagination header
            total_pages = data[0].get("pages", 1)

            for entry in data[1]:
                # Use countryiso3code (reliable ISO3), NOT country.id
                iso3 = entry.get("countryiso3code", "")
                if not iso3:
                    iso3 = entry.get("country", {}).get("id", "")
                year = entry.get("date", "")
                value = entry.get("value")
                if value is not None and len(iso3) == 3:
                    records.append({
                        "iso3": iso3,
                        "year": int(year),
                        "gdp_growth": float(value),
                    })

            page += 1

    if not records:
        logger.error(
            "World Bank API returned 0 records. The API may be down or rate-limiting.\n"
            "Manual download: https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG\n"
            "Save as CSV with columns: iso3, year, gdp_growth"
        )
        raise ValueError("World Bank API returned no GDP data")

    df = pd.DataFrame(records)
    # Filter out aggregate regions (ISO3 codes that aren't real countries)
    # Real country ISO3 codes are mapped in our crosswalk
    df.to_csv(output_path, index=False)
    logger.info("GDP saved to %s (%d records, %d countries)",
                output_path, len(df), df["iso3"].nunique())

    return output_path


def download_fao_food_cpi(output_path: Optional[str | Path] = None) -> Path:
    """
    Download FAO food CPI data from FAOSTAT.

    Downloads the Consumer Price Index - Food series for all available
    countries. Monthly resolution for ~126 countries.

    Note: FAO API can be slow. Falls back to FAOSTAT bulk download
    if the API is unavailable.
    """
    ensure_raw_dir()
    output_path = Path(output_path) if output_path else RAW_DIR / "fao_food_cpi.csv"

    if output_path.exists():
        logger.info("FAO food CPI already downloaded: %s", output_path)
        return output_path

    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloads: pip install requests")

    # FAOSTAT bulk download URL for Consumer Prices domain
    url = "https://fenixservices.fao.org/faostat/static/bulkdownloads/ConsumerPriceIndices_E_All_Data.zip"

    logger.info("Downloading FAO food CPI from FAOSTAT...")
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()

    import zipfile
    zip_path = RAW_DIR / "fao_cpi_raw.zip"
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in FAO ZIP archive")
        zf.extract(csv_names[0], RAW_DIR)
        extracted = RAW_DIR / csv_names[0]

    # Parse and normalise
    fao = pd.read_csv(extracted, encoding="latin-1", low_memory=False)

    # Filter to food CPI items and monthly frequency
    food_items = fao[
        fao["Item"].str.contains("Food", case=False, na=False)
        & (fao["Months"].notna())
    ].copy()

    if len(food_items) == 0:
        logger.warning("No food CPI items found; saving raw file for manual inspection")
        fao.to_csv(output_path, index=False)
    else:
        food_items.to_csv(output_path, index=False)

    # Clean up
    zip_path.unlink(missing_ok=True)
    extracted.unlink(missing_ok=True)

    logger.info("FAO food CPI saved to %s", output_path)
    return output_path


def check_vdem_exists(path: str | Path = None) -> bool:
    """
    Check if V-Dem CSV exists. V-Dem must be downloaded manually from
    https://v-dem.net due to file size (~300MB) and terms of use.
    """
    path = Path(path) if path else RAW_DIR / "V-Dem-CY-Full+Others-v15.csv"
    if path.exists():
        logger.info("V-Dem found at %s", path)
        return True
    logger.warning(
        "V-Dem CSV not found at %s. Please download manually from "
        "https://v-dem.net/data/the-v-dem-dataset/",
        path,
    )
    return False
