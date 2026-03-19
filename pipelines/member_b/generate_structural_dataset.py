"""
Member B — Structural/Contextual Features Pipeline
=====================================================

Loads, cleans, and assembles governance, leader, economic, and coup data
into a single (country, month) panel covering 1985-01 to 2025-12.

Sources:
  1. V-Dem v15/v16 (annual expert-coded governance indices, ~180 countries)
  2. REIGN (monthly leader/regime data, stops Aug 2021)
  3. IMF IFS exchange rates (monthly currency data)
  4. World Bank WDI GDP growth (annual, expanded to monthly)
  5. FAO FAOSTAT food CPI (monthly, ~126 countries)
  6. Powell & Thyne coup d'etat dataset (event-level, 1950–present)

Requirements:
  pip install pandas numpy pyarrow requests pyyaml matplotlib seaborn

Setup:
  - V-Dem CSV must be downloaded manually from https://v-dem.net (~300MB)
  - REIGN, GDP, and Powell & Thyne can be auto-downloaded with --download
  - IMF and FAO data require manual download (see notes in config section)

Output:
  - data/processed/member_b/vdem_governance.csv      -> V-Dem governance indices
  - data/processed/member_b/reign_leader.csv         -> REIGN leader/regime data
  - data/processed/member_b/fx_exchange_rates.csv    -> IMF exchange rate features
  - data/processed/member_b/gdp_growth.csv           -> World Bank GDP growth
  - data/processed/member_b/food_prices.csv          -> FAO food CPI features
  - data/processed/member_b/powell_thyne_coups.csv   -> Powell & Thyne coup events
  - data/processed/member_b/member_b_final.csv       -> assembled panel (merged)
  - data/processed/member_b/feature_registry.csv     -> feature metadata
  - data/processed/member_b/quality_report.json      -> coverage & range stats
  - analysis/member_b/missingness/                    -> missingness heatmaps
  - analysis/member_b/profiles/                       -> distribution plots

Notes:
  - Panel index: (country_iso3, year_month) — consistent with members A/C
  - Annual data expanded to monthly via repetition (NOT interpolation)
  - All features lagged by t-1 to prevent temporal leakage
  - Missingness encoded as informative binary indicators
  - No normalisation — Member C handles this after train/test split
  - REIGN ends Aug 2021: no forward-fill, features are NaN post-cutoff
  - reign_available missingness flag lets the model learn from the gap
  - Powell & Thyne is an independent source covering 1985–present (not a REIGN patch)

"""

import io
import json
import logging
import os
import re
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_b")
REPORT_DIR = os.path.join(BASE_DIR, "analysis", "member_b")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "missingness"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "profiles"), exist_ok=True)

PANEL_START = "1985-01"
PANEL_END = "2025-12"
YEAR_MIN = 1985
YEAR_MAX = 2025
LAG_MONTHS = 1

# Default raw file paths
VDEM_PATH = os.path.join(RAW_DIR, "V-Dem-CY-Full+Others-v16.csv")
REIGN_PATH = os.path.join(RAW_DIR, "REIGN_2021_8.csv")
FX_PATH = os.path.join(RAW_DIR, "imf_exchange_rates.csv")
GDP_PATH = os.path.join(RAW_DIR, "gdp_growth_worldbank.csv")
FOOD_PATH = os.path.join(RAW_DIR, "ConsumerPriceIndices_E_All_Data", "ConsumerPriceIndices_E_All_Data_NOFLAG.csv")
COUPS_PATH = os.path.join(RAW_DIR, "powell_thyne_coups.tsv")

REIGN_CUTOFF = "2021-08"

POWELL_THYNE_URL = (
    "http://www.uky.edu/~clthyn2/coup_data/powell_thyne_coups_final.txt"
)

logger = logging.getLogger(__name__)


# ============================================================
# COUNTRY CODE CROSSWALK (GW <-> ISO3)
# ============================================================

GW_TO_ISO3 = {
    # Americas
    2: "USA", 20: "CAN", 31: "BHS", 40: "CUB", 41: "HTI", 42: "DOM",
    51: "JAM", 52: "TTO", 53: "BRB", 54: "DMA", 55: "GRD", 56: "SLC",
    57: "SVG", 58: "ATG", 60: "KNA", 70: "MEX", 80: "BLZ", 90: "GTM",
    91: "HND", 92: "SLV", 93: "NIC", 94: "CRI", 95: "PAN", 100: "COL",
    101: "VEN", 110: "GUY", 115: "SUR", 130: "ECU", 135: "PER", 140: "BRA",
    145: "BOL", 150: "PRY", 155: "CHL", 160: "ARG", 165: "URY",
    # Europe
    200: "GBR", 205: "IRL", 210: "NLD", 211: "BEL", 212: "LUX",
    220: "FRA", 225: "CHE", 230: "ESP", 235: "PRT", 255: "DEU",
    260: "DEU", 265: "DEU",
    290: "POL", 305: "AUT", 310: "HUN", 316: "CZE", 317: "SVK",
    325: "ITA", 331: "SMR", 338: "MLT", 339: "ALB", 341: "MNE",
    343: "MKD", 344: "HRV", 345: "SRB", 346: "BIH",
    347: "XKX", 349: "SVN",
    350: "GRC", 352: "CYP", 355: "BGR", 359: "MDA", 360: "ROU",
    365: "RUS", 366: "EST", 367: "LVA", 368: "LTU", 369: "UKR",
    370: "BLR", 371: "ARM", 372: "GEO", 373: "AZE",
    375: "FIN", 380: "SWE", 385: "NOR", 390: "DNK", 395: "ISL",
    # Africa
    402: "CPV", 404: "GNB", 411: "GIN", 420: "GMB", 432: "MLI",
    433: "SEN", 434: "BEN", 435: "MRT", 436: "NER", 437: "CIV",
    438: "GIN", 439: "BFA", 450: "LBR", 451: "SLE", 452: "GHA",
    461: "TGO", 471: "CMR", 475: "NGA", 481: "GAB", 482: "CAF",
    483: "TCD", 484: "COG", 490: "COD", 500: "UGA", 501: "KEN",
    510: "TZA", 516: "BDI", 517: "RWA", 520: "SOM", 522: "DJI",
    530: "ETH", 531: "ERI", 540: "AGO", 541: "MOZ",
    551: "ZMB", 552: "ZWE", 553: "MWI", 560: "ZAF",
    565: "NAM", 570: "LSO", 571: "BWA", 572: "SWZ",
    580: "MDG", 581: "COM", 590: "MUS", 591: "SYC",
    600: "MAR", 615: "DZA", 616: "TUN", 620: "LBY",
    625: "SDN", 626: "SSD",
    630: "IRN", 640: "TUR",
    645: "IRQ", 651: "EGY", 652: "SYR", 660: "LBN",
    663: "JOR", 666: "ISR", 670: "SAU", 678: "YEM",
    679: "YEM", 680: "YEM",
    690: "KWT", 692: "BHR", 694: "QAT", 696: "ARE", 698: "OMN",
    # Asia
    700: "AFG", 701: "TKM", 702: "TJK", 703: "KGZ", 704: "UZB",
    705: "KAZ", 710: "CHN", 711: "MNG", 712: "TWN", 713: "TWN",
    730: "KOR", 731: "PRK", 740: "JPN",
    750: "IND", 760: "PAK", 770: "BGD", 771: "BGD",
    775: "MMR", 780: "LKA", 781: "MDV",
    790: "NPL", 800: "THA", 811: "KHM", 812: "LAO",
    816: "VNM", 820: "MYS", 830: "SGP",
    835: "BRN", 840: "PHL", 850: "IDN", 860: "TLS",
    # Oceania
    900: "AUS", 910: "PNG", 920: "NZL",
    935: "VUT", 940: "SLB", 946: "KIR",
    947: "TUV", 950: "FJI", 955: "TON", 970: "NRU",
    983: "MHL", 986: "PLW", 987: "FSM", 990: "WSM",
}

ISO3_TO_GW = {}
for _gw, _iso in GW_TO_ISO3.items():
    if _iso not in ISO3_TO_GW:
        ISO3_TO_GW[_iso] = _gw

IMF_TO_ISO3 = {
    111: "USA", 156: "CAN", 213: "FRA", 218: "DEU", 223: "ITA",
    112: "GBR", 138: "NLD", 124: "BEL", 137: "LUX", 128: "DNK",
    144: "SWE", 142: "NOR", 172: "FIN", 176: "ISL", 122: "AUT",
    146: "CHE", 182: "PRT", 184: "ESP", 174: "GRC", 178: "IRL",
    936: "BGD", 534: "IND", 564: "PAK", 524: "LKA", 518: "MMR",
    566: "NPL", 548: "MYS", 576: "SGP", 578: "THA", 536: "IDN",
    582: "VNM", 556: "PHL", 516: "KHM",
    612: "DZA", 744: "QAT", 456: "SAU", 466: "KWT", 429: "IRN",
    436: "ISR", 439: "IRQ", 446: "LBN", 474: "ARE", 449: "JOR",
    443: "EGY", 672: "SYR", 678: "TUN", 686: "MAR",
    694: "NGA", 646: "ZAF", 648: "ZWE", 636: "KEN", 738: "TZA",
    734: "UGA", 618: "AGO", 622: "BEN", 748: "SEN", 684: "MOZ",
    644: "ETH", 632: "GHA", 662: "CIV",
    298: "ARG", 248: "MEX", 253: "PER", 238: "CHL",
    964: "CHN", 542: "KOR", 158: "JPN",
    926: "RUS", 960: "MNG",
    944: "AUS", 196: "NZL",
}

M49_TO_ISO3 = {
    4: "AFG", 8: "ALB", 12: "DZA", 20: "AND", 24: "AGO",
    28: "ATG", 32: "ARG", 51: "ARM", 36: "AUS", 40: "AUT",
    31: "AZE", 44: "BHS", 48: "BHR", 50: "BGD", 52: "BRB",
    112: "BLR", 56: "BEL", 84: "BLZ", 204: "BEN", 64: "BTN",
    68: "BOL", 70: "BIH", 72: "BWA", 76: "BRA", 96: "BRN",
    100: "BGR", 854: "BFA", 108: "BDI", 132: "CPV", 116: "KHM",
    120: "CMR", 124: "CAN", 140: "CAF", 148: "TCD", 152: "CHL",
    156: "CHN", 170: "COL", 174: "COM", 178: "COG", 180: "COD",
    188: "CRI", 384: "CIV", 191: "HRV", 192: "CUB", 196: "CYP",
    203: "CZE", 208: "DNK", 262: "DJI", 212: "DMA", 214: "DOM",
    218: "ECU", 818: "EGY", 222: "SLV", 226: "GNQ", 232: "ERI",
    233: "EST", 748: "SWZ", 231: "ETH", 242: "FJI", 246: "FIN",
    250: "FRA", 266: "GAB", 270: "GMB", 268: "GEO", 276: "DEU",
    288: "GHA", 300: "GRC", 308: "GRD", 320: "GTM", 324: "GIN",
    624: "GNB", 328: "GUY", 332: "HTI", 340: "HND", 348: "HUN",
    352: "ISL", 356: "IND", 360: "IDN", 364: "IRN", 368: "IRQ",
    372: "IRL", 376: "ISR", 380: "ITA", 388: "JAM", 392: "JPN",
    400: "JOR", 398: "KAZ", 404: "KEN", 296: "KIR", 408: "PRK",
    410: "KOR", 414: "KWT", 417: "KGZ", 418: "LAO", 428: "LVA",
    422: "LBN", 426: "LSO", 430: "LBR", 434: "LBY", 438: "LIE",
    440: "LTU", 442: "LUX", 450: "MDG", 454: "MWI", 458: "MYS",
    462: "MDV", 466: "MLI", 470: "MLT", 584: "MHL", 478: "MRT",
    480: "MUS", 484: "MEX", 583: "FSM", 498: "MDA", 492: "MCO",
    496: "MNG", 499: "MNE", 504: "MAR", 508: "MOZ", 104: "MMR",
    516: "NAM", 520: "NRU", 524: "NPL", 528: "NLD", 554: "NZL",
    558: "NIC", 562: "NER", 566: "NGA", 807: "MKD", 578: "NOR",
    512: "OMN", 586: "PAK", 585: "PLW", 591: "PAN", 598: "PNG",
    600: "PRY", 604: "PER", 608: "PHL", 616: "POL", 620: "PRT",
    634: "QAT", 642: "ROU", 643: "RUS", 646: "RWA", 659: "KNA",
    662: "LCA", 670: "VCT", 882: "WSM", 674: "SMR", 678: "STP",
    682: "SAU", 686: "SEN", 688: "SRB", 690: "SYC", 694: "SLE",
    702: "SGP", 703: "SVK", 705: "SVN", 90: "SLB", 706: "SOM",
    710: "ZAF", 728: "SSD", 724: "ESP", 144: "LKA", 729: "SDN",
    740: "SUR", 752: "SWE", 756: "CHE", 760: "SYR", 762: "TJK",
    834: "TZA", 764: "THA", 626: "TLS", 768: "TGO", 776: "TON",
    780: "TTO", 788: "TUN", 792: "TUR", 795: "TKM", 798: "TUV",
    800: "UGA", 804: "UKR", 784: "ARE", 826: "GBR", 840: "USA",
    858: "URY", 860: "UZB", 548: "VUT", 862: "VEN", 704: "VNM",
    887: "YEM", 894: "ZMB", 716: "ZWE",
    275: "PSE", 158: "TWN", 344: "HKG", 446: "MAC",
    630: "PRI", 660: "AIA", 533: "ABW", 535: "BES",
    136: "CYM", 531: "CUW", 254: "GUF", 312: "GLP",
    474: "MTQ", 540: "NCL", 258: "PYF", 638: "REU", 534: "SXM",
}


def gw_from_iso3(iso3):
    return ISO3_TO_GW.get(iso3)


def iso3_from_gw(gwcode):
    return GW_TO_ISO3.get(gwcode)


# ============================================================
# DOWNLOAD UTILITIES
# ============================================================

def download_reign(output_path=None):
    output_path = output_path or os.path.join(RAW_DIR, "REIGN_2021_8.csv")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"  REIGN already downloaded: {output_path}")
        return output_path
    import requests
    url = "https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/data_sets/REIGN_2021_8.csv"
    print(f"  Downloading REIGN from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"  REIGN saved ({len(response.content)} bytes)")
    return output_path


def download_gdp_worldbank(output_path=None, start_year=1980, end_year=2025):
    output_path = output_path or os.path.join(RAW_DIR, "gdp_growth_worldbank.csv")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"  GDP already downloaded: {output_path}")
        return output_path
    import requests
    indicator = "NY.GDP.MKTP.KD.ZG"
    csv_url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?date={start_year}:{end_year}&format=csv&per_page=20000"
    )
    records = []
    try:
        print("  Downloading GDP from World Bank (CSV endpoint)...")
        response = requests.get(csv_url, timeout=180)
        response.raise_for_status()
        zip_path = os.path.join(RAW_DIR, "_gdp_temp.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist()
                         if n.endswith(".csv") and "Metadata" not in n]
            if csv_names:
                with zf.open(csv_names[0]) as fh:
                    raw = pd.read_csv(fh)
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
                                        "iso3": iso3, "year": int(yc.strip()),
                                        "gdp_growth": float(val),
                                    })
                                except (ValueError, TypeError):
                                    continue
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"  CSV endpoint failed ({e}), trying JSON API...")
        base_url = "https://api.worldbank.org/v2"
        page, total_pages = 1, 1
        while page <= total_pages:
            url = (
                f"{base_url}/country/all/indicator/{indicator}"
                f"?date={start_year}:{end_year}&format=json&per_page=5000&page={page}"
            )
            try:
                import requests as req
                resp = req.get(url, timeout=180)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break
            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                break
            total_pages = data[0].get("pages", 1)
            for entry in data[1]:
                iso3 = entry.get("countryiso3code", "") or entry.get("country", {}).get("id", "")
                year = entry.get("date", "")
                value = entry.get("value")
                if value is not None and len(iso3) == 3:
                    records.append({"iso3": iso3, "year": int(year), "gdp_growth": float(value)})
            page += 1
    if not records:
        raise ValueError("World Bank API returned no GDP data")
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"  GDP saved ({len(df)} records, {df['iso3'].nunique()} countries)")
    return output_path


def download_powell_thyne(output_path=None):
    output_path = output_path or os.path.join(RAW_DIR, "powell_thyne_coups.tsv")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"  Powell & Thyne already downloaded: {output_path}")
        return output_path
    import requests
    print(f"  Downloading Powell & Thyne from {POWELL_THYNE_URL}")
    response = requests.get(POWELL_THYNE_URL, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"  Powell & Thyne saved ({len(response.content)} bytes)")
    return output_path


# ============================================================
# PANEL UTILITIES
# ============================================================

def build_full_panel(countries, start=PANEL_START, end=PANEL_END):
    months = pd.period_range(start=start, end=end, freq="M").astype(str)
    idx = pd.MultiIndex.from_product([countries, months], names=["country_iso3", "year_month"])
    return pd.DataFrame(index=idx).reset_index()


def apply_lag(df, feature_cols, lag=1):
    df = df.sort_values(["country_iso3", "year_month"]).copy()
    df[feature_cols] = df.groupby("country_iso3")[feature_cols].shift(lag)
    return df


# ============================================================
# V-DEM INGESTION
# ============================================================

VDEM_ID_COLS = ["country_text_id", "year"]
VDEM_FEATURES = [
    "v2x_libdem", "v2x_polyarchy", "v2x_civlib", "v2x_rule", "v2x_corr",
    "v2x_clphy", "v2x_clpol", "v2x_freexp_altinf", "v2xcs_ccsi",
    "v2x_regime", "v2x_partipdem", "v2xnp_regcorr", "v2x_execorr",
    "v2x_frassoc_thick",
]


def ingest_vdem(csv_path):
    print("\n[V-Dem] loading...")
    if not os.path.exists(csv_path):
        print(f"  V-Dem CSV not found: {csv_path} — skipping")
        return None

    vdem = pd.read_csv(csv_path, usecols=VDEM_ID_COLS + VDEM_FEATURES, low_memory=False)
    # Include YEAR_MIN-1 as buffer for year-on-year change computation
    vdem = vdem[(vdem["year"] >= YEAR_MIN - 1) & (vdem["year"] <= YEAR_MAX)].copy()
    print(f"  Loaded {len(vdem)} country-years")

    # Expand annual -> monthly (repeat, NOT interpolate)
    months_df = pd.DataFrame({"month": range(1, 13)})
    vdem = vdem.assign(_key=1)
    months_df = months_df.assign(_key=1)
    expanded = vdem.merge(months_df, on="_key").drop(columns="_key")
    expanded["year_month"] = (
        expanded["year"].astype(str) + "-" + expanded["month"].astype(str).str.zfill(2)
    )
    expanded.drop(columns=["month"], inplace=True)

    # Map ISO3 -> country_iso3 (V-Dem uses country_text_id = ISO3)
    expanded["country_iso3"] = expanded["country_text_id"].str.strip().str.upper()
    # Drop countries with no ISO3 in our crosswalk
    valid_iso3 = set(ISO3_TO_GW.keys())
    unmapped = expanded[~expanded["country_iso3"].isin(valid_iso3)]["country_iso3"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unmapped V-Dem countries ({len(unmapped)}): {', '.join(sorted(unmapped)[:10])}")
    expanded = expanded[expanded["country_iso3"].isin(valid_iso3)].copy()

    # Forward-fill if V-Dem data ends before YEAR_MAX
    max_year_with_data = expanded.dropna(subset=["v2x_libdem"])["year"].max()
    if max_year_with_data < YEAR_MAX:
        print(f"  V-Dem data ends at {max_year_with_data}; forward-filling to {YEAR_MAX}")
        expanded = expanded.sort_values(["country_iso3", "year_month"])
        feature_cols = [c for c in VDEM_FEATURES if c in expanded.columns]
        expanded[feature_cols] = expanded.groupby("country_iso3")[feature_cols].ffill()

    # One-hot encode regime type (0=closed autoc, 1=electoral autoc, 2=electoral dem, 3=liberal dem)
    regime_dummies = pd.get_dummies(
        expanded["v2x_regime"].astype("Int64"), prefix="regime_type"
    ).astype(int)
    for i in range(4):
        col = f"regime_type_{i}"
        if col not in regime_dummies.columns:
            regime_dummies[col] = 0
    expanded = pd.concat([expanded, regime_dummies], axis=1)

    # Derive composite features
    expanded["governance_deficit"] = 1.0 - expanded[["v2x_libdem", "v2x_polyarchy", "v2x_rule"]].mean(axis=1)
    expanded["repression_index"] = (
        expanded[["v2x_corr", "v2x_execorr"]].mean(axis=1) + (1.0 - expanded["v2x_clphy"])
    ) / 2.0
    expanded = expanded.sort_values(["country_iso3", "year_month"])
    expanded["libdem_yoy_change"] = expanded.groupby("country_iso3")["v2x_libdem"].diff(12)

    # Validate ranges
    bounded_cols = [c for c in VDEM_FEATURES if c != "v2x_regime"]
    for col in bounded_cols:
        vals = expanded[col].dropna()
        if len(vals) > 0:
            out = ~vals.between(-0.001, 1.001)
            if out.any():
                print(f"  WARNING: {col} has {out.sum()} values outside [0, 1]")

    # Flag large democracy shifts
    large = expanded[expanded["libdem_yoy_change"].abs() > 0.15]
    if len(large) > 0:
        print(f"  {len(large)} country-months with |delta_libdem| > 0.15")

    # Filter to study period (drop 2009 buffer)
    expanded = expanded[expanded["year_month"] >= PANEL_START].copy()

    # Clean up
    drop_cols = ["country_text_id", "year"]
    expanded.drop(columns=[c for c in drop_cols if c in expanded.columns], inplace=True)

    print(f"  V-Dem complete: {len(expanded)} rows, {expanded['country_iso3'].nunique()} countries")
    return expanded


# ============================================================
# REIGN INGESTION
# ============================================================

REIGN_RAW_COLS = [
    "ccode", "year", "month", "government", "leader", "elected",
    "tenure_months", "age", "male", "militarycareer",
    "lastelection", "loss", "irregular", "prev_conflict", "precip",
]


def ingest_reign(csv_path):
    print("\n[REIGN] loading...")
    if not os.path.exists(csv_path):
        print(f"  REIGN CSV not found: {csv_path} — skipping")
        return None

    reign = pd.read_csv(csv_path, low_memory=False)
    available = [c for c in REIGN_RAW_COLS if c in reign.columns]
    reign = reign[available].copy()
    reign = reign[(reign["year"] >= YEAR_MIN) & (reign["year"] <= YEAR_MAX)].copy()
    print(f"  Loaded {len(reign)} country-months")

    # Create temporal index
    reign["year_month"] = (
        reign["year"].astype(int).astype(str) + "-"
        + reign["month"].astype(int).astype(str).str.zfill(2)
    )
    reign["gwcode"] = reign["ccode"].astype(int)

    # Map GW -> ISO3 for panel consistency
    reign["country_iso3"] = reign["gwcode"].map(GW_TO_ISO3)
    unmapped = reign[reign["country_iso3"].isna()]["gwcode"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unmapped REIGN GW codes: {unmapped[:10]}")
    reign = reign.dropna(subset=["country_iso3"]).copy()

    # No forward-fill: only use actual observed data (up to Aug 2021)
    reign = reign[reign["year_month"] <= REIGN_CUTOFF].copy()
    reign = reign.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)
    print(f"  Using observed data only (up to {REIGN_CUTOFF}): {len(reign)} rows")

    # Derive features
    if "age" in reign.columns:
        reign["leader_age_risk"] = np.where(
            (reign["age"] < 40) | (reign["age"] > 75), 1, 0
        )
        reign.loc[reign["age"].isna(), "leader_age_risk"] = np.nan

    if "lastelection" in reign.columns:
        reign["lastelection_parsed"] = pd.to_datetime(reign["lastelection"], errors="coerce")
        reign["year_month_dt"] = pd.to_datetime(reign["year_month"] + "-01")
        reign["months_since_election"] = (
            (reign["year_month_dt"] - reign["lastelection_parsed"]).dt.days / 30.44
        )
        reign.loc[reign["months_since_election"] < 0, "months_since_election"] = np.nan
        reign.drop(columns=["lastelection_parsed", "year_month_dt"], inplace=True)

    if "government" in reign.columns:
        reign["regime_change"] = (
            reign.groupby("country_iso3")["government"].shift(1) != reign["government"]
        ).astype(float)
        first_mask = ~reign.duplicated(subset=["country_iso3"], keep="first")
        reign.loc[first_mask, "regime_change"] = 0.0
        reign.loc[reign["government"].isna(), "regime_change"] = np.nan

    if "irregular" in reign.columns:
        reign["coup_event"] = reign["irregular"].fillna(0).astype(float)
        reign.loc[reign["irregular"].isna(), "coup_event"] = np.nan

    # One-hot encode regime type
    if "government" in reign.columns:
        regime_dummies = pd.get_dummies(reign["government"], prefix="reign_regime").astype(int)
        reign = pd.concat([reign, regime_dummies], axis=1)

    # Structural break detection
    reign["structural_break"] = 0.0
    if "regime_change" in reign.columns:
        reign["structural_break"] = reign["structural_break"].where(
            reign["regime_change"] != 1, 1.0
        )
    if "coup_event" in reign.columns:
        reign["structural_break"] = reign["structural_break"].where(
            reign["coup_event"] != 1, 1.0
        )

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

    reign["months_since_structural_break"] = reign.groupby("country_iso3")[
        "structural_break"
    ].transform(_months_since_break)

    reign.drop(columns=["structural_break"], inplace=True)

    # Select output columns
    meta_cols = ["country_iso3", "year_month"]
    feature_cols = [
        "tenure_months", "age", "male", "militarycareer", "elected",
        "leader_age_risk", "months_since_election",
        "regime_change", "coup_event", "prev_conflict", "precip",
        "months_since_structural_break",
    ]
    regime_cols = [c for c in reign.columns if c.startswith("reign_regime_")]
    keep_cols = meta_cols + [c for c in feature_cols if c in reign.columns] + regime_cols
    reign = reign[keep_cols].copy()

    # Deduplicate
    reign = reign.drop_duplicates(subset=["country_iso3", "year_month"], keep="last")

    print(f"  REIGN complete: {len(reign)} rows, {reign['country_iso3'].nunique()} countries")
    return reign


# ============================================================
# ECONOMIC INGESTION
# ============================================================

def ingest_exchange_rates(csv_path):
    print("\n[FX] loading exchange rates...")
    if not os.path.exists(csv_path):
        print(f"  FX CSV not found: {csv_path} — skipping")
        return None

    econ = pd.read_csv(csv_path, low_memory=False)

    # Detect format
    has_series_code = "SERIES_CODE" in econ.columns
    has_indicator = "INDICATOR" in econ.columns
    imf_sdmx = any(c.strip().upper() in ("REF_AREA", "TIME_PERIOD", "OBS_VALUE") for c in econ.columns)
    monthly_period_cols = [c for c in econ.columns if re.match(r"^\d{4}-M\d{2}$", c.strip())]

    if has_series_code and monthly_period_cols:
        # IMF Data Explorer wide format
        if has_indicator:
            usd_mask = econ["INDICATOR"].str.contains("Domestic currency per US Dollar", na=False)
            econ = econ[usd_mask].copy()
        if "FREQUENCY" in econ.columns:
            econ = econ[econ["FREQUENCY"] == "Monthly"].copy()
        if "TYPE_OF_TRANSFORMATION" in econ.columns:
            pa_mask = econ["TYPE_OF_TRANSFORMATION"].str.contains("Period average", na=False)
            if pa_mask.any():
                econ = econ[pa_mask].copy()
        if len(econ) == 0:
            raise ValueError("No matching FX series found in IMF data")
        econ["iso3"] = econ["SERIES_CODE"].str.split(".").str[0].str.strip().str.upper()
        econ = econ.drop_duplicates(subset=["iso3"], keep="first")
        econ = econ.melt(id_vars=["iso3"], value_vars=monthly_period_cols,
                         var_name="period", value_name="exchange_rate")
        econ["year"] = econ["period"].str[:4].astype(int)
        econ["month"] = econ["period"].str[6:].astype(int)
        econ["exchange_rate"] = pd.to_numeric(econ["exchange_rate"], errors="coerce")
    elif imf_sdmx:
        sdmx_map = {}
        for c in econ.columns:
            cu = c.strip().upper()
            if cu == "REF_AREA": sdmx_map[c] = "iso3"
            elif cu == "OBS_VALUE": sdmx_map[c] = "exchange_rate"
            elif cu == "TIME_PERIOD": sdmx_map[c] = "time_period"
        econ = econ.rename(columns=sdmx_map)
        econ["exchange_rate"] = pd.to_numeric(econ["exchange_rate"], errors="coerce")
        econ["time_period"] = econ["time_period"].astype(str).str.strip()
        econ["year"] = econ["time_period"].str[:4].astype(int)
        econ["month"] = econ["time_period"].str[5:7].astype(int)
        econ["iso3"] = econ["iso3"].str.strip().str.upper()
    else:
        col_map = {}
        for c in econ.columns:
            cl = c.lower().strip()
            if cl in ("country_code", "iso3", "iso", "country_iso3", "ref_area"):
                col_map[c] = "iso3"
            elif cl in ("imf_code", "imf_country_code"):
                col_map[c] = "imf_code"
            elif cl == "year": col_map[c] = "year"
            elif cl == "month": col_map[c] = "month"
            elif cl in ("exchange_rate", "value", "rate", "fx_rate", "obs_value"):
                col_map[c] = "exchange_rate"
        econ = econ.rename(columns=col_map)

    # Map to ISO3 (via GW crosswalk)
    if "iso3" in econ.columns:
        valid_iso3 = set(ISO3_TO_GW.keys())
        econ["country_iso3"] = econ["iso3"].where(econ["iso3"].isin(valid_iso3))
    elif "imf_code" in econ.columns:
        econ["country_iso3"] = econ["imf_code"].map(IMF_TO_ISO3)
    else:
        raise ValueError(f"FX CSV needs a country code column. Found: {list(econ.columns)}")

    econ["exchange_rate"] = pd.to_numeric(econ.get("exchange_rate", pd.Series(dtype=float)), errors="coerce")
    econ = econ.dropna(subset=["country_iso3", "exchange_rate"]).copy()

    econ["year_month"] = (
        econ["year"].astype(int).astype(str) + "-" + econ["month"].astype(int).astype(str).str.zfill(2)
    )
    econ = econ[(econ["year"] >= YEAR_MIN) & (econ["year"] <= YEAR_MAX)].copy()
    econ = econ.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    # Feature engineering
    econ["fx_pct_change"] = econ.groupby("country_iso3")["exchange_rate"].pct_change()
    econ["fx_volatility"] = econ.groupby("country_iso3")["fx_pct_change"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )
    econ["fx_volatility_log"] = np.log1p(econ["fx_volatility"])

    country_stats = econ.groupby("country_iso3")["fx_pct_change"].agg(["mean", "std"]).reset_index()
    country_stats.columns = ["country_iso3", "fx_mean_hist", "fx_std_hist"]
    econ = econ.merge(country_stats, on="country_iso3", how="left")
    econ["fx_depreciation_flag"] = (
        econ["fx_pct_change"] > (econ["fx_mean_hist"] + 2 * econ["fx_std_hist"])
    ).astype(int)
    econ.loc[econ["fx_std_hist"].isna() | (econ["fx_std_hist"] == 0), "fx_depreciation_flag"] = 0

    econ["fx_pct_change_zscore"] = econ.groupby("country_iso3")["fx_pct_change"].transform(
        lambda x: (x - x.expanding(min_periods=6).mean()) / x.expanding(min_periods=6).std().clip(lower=1e-8)
    )

    output_cols = [
        "country_iso3", "year_month",
        "fx_pct_change", "fx_volatility", "fx_volatility_log",
        "fx_depreciation_flag", "fx_pct_change_zscore",
    ]
    econ = econ[output_cols].copy()

    # Forward-fill gaps up to 3 months
    econ = econ.sort_values(["country_iso3", "year_month"])
    fill_cols = ["fx_pct_change", "fx_volatility", "fx_volatility_log", "fx_pct_change_zscore"]
    for col in fill_cols:
        econ[col] = econ.groupby("country_iso3")[col].ffill(limit=3)

    print(f"  FX complete: {len(econ)} rows, {econ['country_iso3'].nunique()} countries")
    return econ


def ingest_gdp(csv_path):
    print("\n[GDP] loading...")
    if not os.path.exists(csv_path):
        print(f"  GDP CSV not found: {csv_path} — skipping")
        return None
    if os.path.getsize(csv_path) < 10:
        print(f"  GDP CSV is empty: {csv_path} — skipping")
        return None

    gdp = pd.read_csv(csv_path, low_memory=False)
    if len(gdp) == 0:
        return None

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

    valid_iso3 = set(ISO3_TO_GW.keys())
    gdp["country_iso3"] = gdp["iso3"].where(gdp["iso3"].isin(valid_iso3))
    gdp = gdp.dropna(subset=["country_iso3"]).copy()
    gdp["year"] = gdp["year"].astype(int)

    gdp = gdp[gdp["year"] >= YEAR_MIN - 5].copy()
    gdp = gdp.sort_values(["country_iso3", "year"]).reset_index(drop=True)

    gdp["gdp_growth_deviation"] = gdp.groupby("country_iso3")["gdp_growth"].transform(
        lambda x: (x - x.rolling(5, min_periods=3).mean()) / x.rolling(5, min_periods=3).std().clip(lower=0.01)
    )
    gdp["gdp_negative_shock"] = (gdp["gdp_growth_deviation"] < -1.0).astype(int)

    gdp = gdp[gdp["year"] >= YEAR_MIN].copy()

    # Expand annual -> monthly
    months_df = pd.DataFrame({"month": range(1, 13)})
    gdp = gdp.assign(_key=1)
    months_df = months_df.assign(_key=1)
    gdp_monthly = gdp.merge(months_df, on="_key").drop(columns="_key")
    gdp_monthly["year_month"] = (
        gdp_monthly["year"].astype(str) + "-" + gdp_monthly["month"].astype(str).str.zfill(2)
    )

    # Forward-fill delayed data
    gdp_monthly = gdp_monthly.sort_values(["country_iso3", "year_month"])
    max_year = gdp_monthly.dropna(subset=["gdp_growth"])["year"].max()
    if max_year < YEAR_MAX:
        print(f"  GDP data ends at {max_year}; forward-filling to {YEAR_MAX}")
        fill_gdp = ["gdp_growth", "gdp_growth_deviation", "gdp_negative_shock"]
        gdp_monthly[fill_gdp] = gdp_monthly.groupby("country_iso3")[fill_gdp].ffill()

    output_cols = ["country_iso3", "year_month", "gdp_growth", "gdp_growth_deviation", "gdp_negative_shock"]
    gdp_monthly = gdp_monthly[output_cols].copy()

    print(f"  GDP complete: {len(gdp_monthly)} rows, {gdp_monthly['country_iso3'].nunique()} countries")
    return gdp_monthly


def ingest_food_prices(csv_path):
    print("\n[Food] loading food prices...")
    if not os.path.exists(csv_path):
        print(f"  Food CPI CSV not found: {csv_path} — skipping")
        return None

    food = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")

    # Detect FAOSTAT bulk wide format (year columns like Y2000, Y2001, ...)
    year_cols = [c for c in food.columns if re.match(r"^Y\d{4}$", c)]
    is_bulk_wide = len(year_cols) > 0

    if is_bulk_wide:
        # FAOSTAT bulk download: wide format with Y2000..Y2025 columns
        # Filter to Food CPI item only (Item Code 23013)
        if "Item Code" in food.columns:
            food = food[food["Item Code"] == 23013].copy()
        print(f"  Bulk format detected: {len(food)} rows, {len(year_cols)} year columns")

        # Map M49 -> ISO3
        m49_col = None
        for c in food.columns:
            if "area code" in c.lower() and "m49" in c.lower():
                m49_col = c
                break
        if m49_col is None:
            m49_col = "Area Code (M49)"

        food["m49_clean"] = food[m49_col].astype(str).str.replace("'", "").str.strip()
        food["iso3"] = pd.to_numeric(food["m49_clean"], errors="coerce").astype("Int64").map(M49_TO_ISO3)

        # Parse month from Months Code (7001=Jan, 7002=Feb, ..., 7012=Dec)
        food["month"] = pd.to_numeric(food["Months Code"], errors="coerce") - 7000
        food = food.dropna(subset=["month", "iso3"]).copy()

        # Melt wide year columns to long format
        id_cols = ["iso3", "month"]
        food_long = food.melt(id_vars=id_cols, value_vars=year_cols,
                              var_name="year_str", value_name="food_cpi")
        food_long["year"] = food_long["year_str"].str[1:].astype(int)
        food_long["food_cpi"] = pd.to_numeric(food_long["food_cpi"], errors="coerce")
        food_long.drop(columns=["year_str"], inplace=True)
        food = food_long
    else:
        # Long format (original fao_food_cpi.csv or similar)
        MONTH_NAME_TO_NUM = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        # Detect area code column
        m49_col, iso3_col = None, None
        for c in food.columns:
            cl = c.strip().lower()
            if "area code" in cl and "iso3" in cl:
                iso3_col = c
                break
            if "area code" in cl and "m49" in cl:
                m49_col = c
        if iso3_col:
            food = food.rename(columns={iso3_col: "iso3"})
            food["iso3"] = food["iso3"].astype(str).str.strip().str.upper()
        elif m49_col:
            food["iso3"] = (
                pd.to_numeric(food[m49_col], errors="coerce").astype("Int64").map(M49_TO_ISO3)
            )
        else:
            col_map = {}
            for c in food.columns:
                cl = c.lower().strip()
                if cl in ("country_code", "iso3", "iso", "area_code"):
                    col_map[c] = "iso3"
                elif cl == "year": col_map[c] = "year"
                elif cl == "month": col_map[c] = "month"
                elif cl in ("food_cpi", "value", "cpi_food", "food_price_index"):
                    col_map[c] = "food_cpi"
            food = food.rename(columns=col_map)

        # Parse months
        months_col = None
        for c in food.columns:
            if c.strip().lower() in ("months", "months code"):
                months_col = c
                break
        if months_col:
            sample_val = str(food[months_col].dropna().iloc[0]).strip().lower()
            if sample_val in MONTH_NAME_TO_NUM:
                food["month"] = food[months_col].str.strip().str.lower().map(MONTH_NAME_TO_NUM)
            else:
                food["month"] = pd.to_numeric(food[months_col], errors="coerce")
                if food["month"].dropna().min() > 7000:
                    food["month"] = food["month"] - 7000
            food = food.dropna(subset=["month"]).copy()

        if "Value" in food.columns:
            food = food.rename(columns={"Value": "food_cpi"})
        food["food_cpi"] = pd.to_numeric(food.get("food_cpi", pd.Series(dtype=float)), errors="coerce")

    valid_iso3 = set(ISO3_TO_GW.keys())
    food["country_iso3"] = food["iso3"].where(food["iso3"].isin(valid_iso3))
    food = food.dropna(subset=["country_iso3", "food_cpi"]).copy()

    food["year_month"] = (
        food["year"].astype(int).astype(str) + "-" + food["month"].astype(int).astype(str).str.zfill(2)
    )
    food = food[food["year"] >= YEAR_MIN - 1].copy()
    food = food.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    # Feature engineering
    food["food_cpi_lagged"] = food.groupby("country_iso3")["food_cpi"].shift(1)
    food["food_price_anomaly"] = (
        food["food_cpi_lagged"]
        / food.groupby("country_iso3")["food_cpi_lagged"].transform(
            lambda x: x.rolling(12, min_periods=6).mean()
        )
    )
    food["food_cpi_yoy"] = food.groupby("country_iso3")["food_cpi"].pct_change(12)
    food["food_cpi_acceleration"] = food.groupby("country_iso3")["food_cpi_yoy"].diff(1)
    food["food_price_anomaly_log"] = np.log(food["food_price_anomaly"].clip(lower=1e-6))
    food["food_price_spike"] = (food["food_price_anomaly"] > 1.15).astype(int)

    food = food[food["year_month"] >= PANEL_START].copy()

    output_cols = [
        "country_iso3", "year_month",
        "food_price_anomaly", "food_price_anomaly_log",
        "food_cpi_acceleration", "food_price_spike",
    ]
    food = food[output_cols].copy()

    print(f"  Food complete: {len(food)} rows, {food['country_iso3'].nunique()} countries")
    return food


# ============================================================
# POWELL & THYNE COUP INGESTION
# ============================================================

def ingest_powell_thyne(tsv_path):
    print("\n[Powell & Thyne] loading coup data...")
    if not os.path.exists(tsv_path):
        print(f"  Powell & Thyne file not found: {tsv_path} — skipping")
        return None

    pt = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    col_map = {}
    for c in pt.columns:
        cl = c.strip().lower()
        if cl == "ccode": col_map[c] = "gwcode"
        elif cl == "country": col_map[c] = "country_name"
        elif cl == "coup": col_map[c] = "coup_code"
    pt = pt.rename(columns=col_map)

    pt["gwcode"] = pd.to_numeric(pt["gwcode"], errors="coerce")
    pt = pt.dropna(subset=["gwcode"]).copy()
    pt["gwcode"] = pt["gwcode"].astype(int)
    pt["country_iso3"] = pt["gwcode"].map(GW_TO_ISO3)
    pt = pt.dropna(subset=["country_iso3"]).copy()

    pt["coup_code"] = pd.to_numeric(pt["coup_code"], errors="coerce")
    pt["pt_coup_successful"] = (pt["coup_code"] == 2).astype(int)
    pt["pt_coup_failed"] = (pt["coup_code"] == 1).astype(int)

    pt["year"] = pd.to_numeric(pt["year"], errors="coerce").astype("Int64")
    pt["month"] = pd.to_numeric(pt["month"], errors="coerce").astype("Int64")
    pt = pt.dropna(subset=["year", "month"]).copy()
    pt = pt[(pt["year"] >= YEAR_MIN) & (pt["year"] <= YEAR_MAX)].copy()

    print(f"  {len(pt)} coup events, {pt['pt_coup_successful'].sum()} successful, {pt['pt_coup_failed'].sum()} failed")

    # Aggregate to country-month
    pt["year_month"] = (
        pt["year"].astype(int).astype(str) + "-" + pt["month"].astype(int).astype(str).str.zfill(2)
    )
    monthly = pt.groupby(["country_iso3", "year_month"]).agg(
        pt_coup_successful=("pt_coup_successful", "max"),
        pt_coup_failed=("pt_coup_failed", "max"),
        pt_coup_count=("coup_code", "count"),
    ).reset_index()
    monthly["pt_coup_event"] = (
        (monthly["pt_coup_successful"] == 1) | (monthly["pt_coup_failed"] == 1)
    ).astype(int)

    return monthly


def integrate_coups_with_panel(panel, monthly_coups):
    """Merge Powell & Thyne coup data as an independent source across the full panel."""
    if monthly_coups is None:
        return panel

    # Merge coup events onto panel
    panel = panel.merge(monthly_coups, on=["country_iso3", "year_month"], how="left")
    coup_cols = ["pt_coup_event", "pt_coup_successful", "pt_coup_failed", "pt_coup_count"]
    for col in coup_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)

    panel = panel.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    # Cumulative coups and months-since
    panel["pt_cumulative_coups"] = panel.groupby("country_iso3")["pt_coup_event"].cumsum()

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

    panel["pt_months_since_coup"] = panel.groupby("country_iso3")[
        "pt_coup_event"
    ].transform(_months_since_event)

    return panel


# ============================================================
# CROSS-VALIDATION & ASSEMBLY
# ============================================================

def cross_validate_structural_breaks(panel):
    has_vdem = "v2x_libdem" in panel.columns
    has_reign_break = "regime_change" in panel.columns
    has_pt_coup = "pt_coup_event" in panel.columns

    if not has_reign_break and not has_pt_coup:
        panel["vdem_stale_flag"] = 0
        return panel

    panel = panel.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    if has_vdem:
        panel["_year"] = panel["year_month"].str[:4].astype(int)
        panel["vdem_stale_flag"] = 0

        for c in panel["country_iso3"].unique():
            c_mask = panel["country_iso3"] == c
            c_data = panel.loc[c_mask]
            for year in c_data["_year"].unique():
                year_mask = c_mask & (panel["_year"] == year)
                year_data = panel.loc[year_mask]
                # Check for breaks from both REIGN and Powell & Thyne
                break_cond = pd.Series(False, index=year_data.index)
                if has_reign_break:
                    break_cond = break_cond | (year_data["regime_change"] == 1)
                if has_pt_coup:
                    break_cond = break_cond | (year_data["pt_coup_event"] == 1)
                breaks_in_year = year_data[break_cond]
                if len(breaks_in_year) > 0:
                    first_break_month = breaks_in_year["year_month"].min()
                    stale_mask = year_mask & (panel["year_month"] > first_break_month)
                    panel.loc[stale_mask, "vdem_stale_flag"] = 1

        panel.drop(columns=["_year"], inplace=True)
    else:
        panel["vdem_stale_flag"] = 0

    return panel


# ============================================================
# MISSINGNESS ENCODING
# ============================================================

def encode_missingness(df):
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


# ============================================================
# QUALITY CHECKS & REPORTING
# ============================================================

def run_quality_checks(df):
    print("\n[Quality] running checks...")

    # Panel sorted per country
    for c in list(df["country_iso3"].unique())[:20]:
        country = df[df["country_iso3"] == c]
        assert country["year_month"].is_monotonic_increasing, f"{c}: year_month not sorted"
    print("  PASS: Panel sorted per country")

    # No duplicates
    dupes = df.duplicated(subset=["country_iso3", "year_month"], keep=False)
    assert not dupes.any(), f"Found {dupes.sum()} duplicate rows"
    print("  PASS: No duplicate panel entries")

    # V-Dem range check
    vdem_cols = [c for c in df.columns if c.startswith("v2x_")]
    for col in vdem_cols:
        if col == "v2x_regime":
            continue
        vals = df[col].dropna()
        if len(vals) > 0:
            out = ~vals.between(-0.01, 1.01)
            assert not out.any(), f"{col}: {out.sum()} values outside [0,1]"
    print("  PASS: V-Dem indices in valid range")

    # Coverage report
    for col in ["v2x_libdem", "tenure_months", "fx_volatility", "food_price_anomaly", "gdp_growth"]:
        if col in df.columns:
            coverage = df[col].notna().mean()
            print(f"  {col} coverage: {coverage:.1%}")

    print("  All quality checks passed")


def generate_quality_report(df):
    report = {
        "panel_dimensions": {
            "n_rows": int(len(df)),
            "n_countries": int(df["country_iso3"].nunique()),
            "n_months": int(df["year_month"].nunique()),
            "date_range": [df["year_month"].min(), df["year_month"].max()],
        },
        "missingness": {},
        "source_coverage": {},
        "value_ranges": {},
    }

    for col in sorted(df.columns):
        if col in ("country_iso3", "year_month"):
            continue
        missing_pct = round(df[col].isna().mean() * 100, 2)
        if missing_pct > 0:
            report["missingness"][col] = missing_pct

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


# ============================================================
# ANALYSIS & PROFILING (matching member_a/c format)
# ============================================================

def missingness_report(df, name):
    miss = df.drop(columns=["country_iso3", "year_month"], errors="ignore").isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n  [{name}] missingness ({len(miss)} features with nulls):")
    if miss.empty:
        print("    None")
        return

    for feat, pct in miss.head(20).items():
        print(f"    {feat}: {pct:.1%}")

    if HAS_PLOTTING and len(miss) > 0:
        sample = df[miss.index].sample(min(500, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(max(8, len(miss) * 0.4), 6))
        sns.heatmap(sample.isnull(), cbar=False, yticklabels=False, ax=ax)
        ax.set_title(f"Missingness — {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "missingness", f"{name}_missingness.png"), dpi=100)
        plt.close(fig)


def distribution_profile(df, feature_cols, name):
    if not HAS_PLOTTING:
        return
    for col in feature_cols:
        series = df[col].dropna()
        if series.empty or series.nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].hist(series, bins=50, edgecolor="none", color="steelblue", alpha=0.8)
        axes[0].set_title(f"{col} — raw")
        axes[1].hist(np.log1p(series.clip(lower=0)), bins=50, edgecolor="none", color="darkorange", alpha=0.8)
        axes[1].set_title(f"{col} — log1p")
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "profiles", f"{name}_{col.replace('/', '_')}.png"), dpi=80)
        plt.close(fig)


# ============================================================
# FEATURE REGISTRY
# ============================================================

def save_feature_registry(df):
    feature_cols = [c for c in df.columns if c not in ["country_iso3", "year_month"]]
    records = []
    for col in feature_cols:
        series = df[col].dropna()
        source = "V-Dem"
        if col.startswith("v2x_") or col.startswith("regime_type_") or col in ("governance_deficit", "repression_index", "libdem_yoy_change"):
            source = "V-Dem"
        elif col.startswith("reign_") or col in ("tenure_months", "age", "male", "militarycareer", "elected",
                                                   "leader_age_risk", "months_since_election", "regime_change",
                                                   "coup_event", "prev_conflict", "precip",
                                                   "months_since_structural_break", "vdem_stale_flag"):
            source = "REIGN"
        elif col.startswith("fx_"):
            source = "IMF IFS"
        elif col.startswith("gdp_"):
            source = "World Bank"
        elif col.startswith("food_"):
            source = "FAO"
        elif col.startswith("pt_"):
            source = "Powell & Thyne"
        elif col.endswith("_available"):
            source = "Derived (missingness)"

        records.append({
            "feature_name": col,
            "source": source,
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isna().mean() * 100, 2),
            "mean": round(series.mean(), 4) if len(series) > 0 else None,
            "std": round(series.std(), 4) if len(series) > 0 else None,
            "min": round(series.min(), 4) if len(series) > 0 else None,
            "max": round(series.max(), 4) if len(series) > 0 else None,
            "lag_months": 1,
        })
    path = os.path.join(OUTPUT_DIR, "feature_registry.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\n[REGISTRY] {len(records)} features -> {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Member B — structural/contextual features pipeline")
    parser.add_argument("--vdem", type=str, default=VDEM_PATH)
    parser.add_argument("--reign", type=str, default=REIGN_PATH)
    parser.add_argument("--fx", type=str, default=FX_PATH)
    parser.add_argument("--gdp", type=str, default=GDP_PATH)
    parser.add_argument("--food", type=str, default=FOOD_PATH)
    parser.add_argument("--coups", type=str, default=COUPS_PATH)
    parser.add_argument("--download", action="store_true", help="Download available sources")
    args = parser.parse_args()

    print("Member B — structural/contextual features pipeline")
    print("=" * 60)

    # Optionally download
    if args.download:
        print("\n[Download] fetching available data sources...")
        try:
            download_reign()
        except Exception as e:
            print(f"  REIGN download failed: {e}")
        try:
            download_gdp_worldbank()
        except Exception as e:
            print(f"  GDP download failed: {e}")
        try:
            download_powell_thyne()
        except Exception as e:
            print(f"  Powell & Thyne download failed: {e}")

    # Ingest each source
    vdem_df = ingest_vdem(args.vdem) if os.path.exists(args.vdem) else None
    reign_df = ingest_reign(args.reign) if os.path.exists(args.reign) else None
    fx_df = ingest_exchange_rates(args.fx) if os.path.exists(args.fx) else None
    gdp_df = ingest_gdp(args.gdp) if os.path.exists(args.gdp) else None
    food_df = ingest_food_prices(args.food) if os.path.exists(args.food) else None
    coups_df = ingest_powell_thyne(args.coups) if os.path.exists(args.coups) else None

    sources = {k: v for k, v in {
        "V-Dem": vdem_df, "REIGN": reign_df, "FX": fx_df,
        "GDP": gdp_df, "Food": food_df, "Coups": coups_df,
    }.items() if v is not None}

    if not sources:
        print("\n  ERROR: No data sources found. Place raw data in data/raw/ or use --download.")
        return

    print(f"\n[Assembly] sources available: {', '.join(sources.keys())}")

    # Save each source as a separate CSV (matching member_a/c output pattern)
    source_files = {
        "vdem_governance.csv": vdem_df,
        "reign_leader.csv": reign_df,
        "fx_exchange_rates.csv": fx_df,
        "gdp_growth.csv": gdp_df,
        "food_prices.csv": food_df,
        "powell_thyne_coups.csv": coups_df,
    }
    for filename, df in source_files.items():
        if df is not None:
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  [OUTPUT] {df.shape} -> {path}")

    # Collect all country ISO3 codes
    all_countries = set()
    for df in sources.values():
        if "country_iso3" in df.columns:
            all_countries.update(df["country_iso3"].unique())

    # Build panel skeleton
    panel = build_full_panel(sorted(all_countries))
    print(f"  Panel skeleton: {panel['country_iso3'].nunique()} countries x {panel['year_month'].nunique()} months = {len(panel)} rows")

    # Left-join all sources
    for name, df in [("V-Dem", vdem_df), ("REIGN", reign_df), ("FX", fx_df), ("GDP", gdp_df), ("Food", food_df)]:
        if df is not None:
            print(f"  Merging {name} ({len(df)} rows)...")
            panel = panel.merge(df, on=["country_iso3", "year_month"], how="left")

    # Integrate Powell & Thyne as independent source
    panel = integrate_coups_with_panel(panel, coups_df)

    # Cross-validate V-Dem against structural breaks
    panel = cross_validate_structural_breaks(panel)

    # Apply t-1 lag
    feature_cols = [c for c in panel.columns if c not in ("country_iso3", "year_month")]
    panel = apply_lag(panel, feature_cols, lag=LAG_MONTHS)
    print(f"  Lag t-{LAG_MONTHS} applied to {len(feature_cols)} features")

    # Encode missingness
    panel = encode_missingness(panel)

    # Quality checks
    run_quality_checks(panel)

    # Generate quality report
    report = generate_quality_report(panel)
    report_path = os.path.join(OUTPUT_DIR, "quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] quality report -> {report_path}")

    # Analysis outputs (matching member_a/c format)
    missingness_report(panel, "structural")
    vdem_cols = [c for c in panel.columns if c.startswith("v2x_") or c in ("governance_deficit", "repression_index")]
    econ_cols = [c for c in panel.columns if c.startswith("fx_") or c.startswith("gdp_") or c.startswith("food_")]
    distribution_profile(panel, vdem_cols, "vdem")
    distribution_profile(panel, econ_cols, "economic")

    # Save final dataset
    csv_path = os.path.join(OUTPUT_DIR, "member_b_final.csv")
    panel.to_csv(csv_path, index=False)
    print(f"\n[OUTPUT] {panel.shape} -> {csv_path}")

    # Feature registry
    save_feature_registry(panel)

    print(f"\nPipeline complete.")
    print(f"  Countries: {panel['country_iso3'].nunique()}")
    print(f"  Months: {panel['year_month'].nunique()}")
    print(f"  Features: {len([c for c in panel.columns if c not in ('country_iso3', 'year_month')])}")


if __name__ == "__main__":
    main()
