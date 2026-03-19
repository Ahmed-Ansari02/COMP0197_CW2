"""
Loads, cleans, and aggregates conflict event data from three sources into a
(country, month) panel covering 1985-01 to 2025-12.

Sources:
  1. UCDP GED v25.1 (1989-2024) + Candidate v26.0.1 (2025+)
  2. ACLED export CSVs (1997-present)
  3. GDELT 1.0 via BigQuery — conflict and protest events (1985-present)

Requirements:
  pip install pandas pycountry matplotlib seaborn google-cloud-bigquery db-dtypes

Setup for BigQuery:
  1. Create a Google Cloud project (free tier gives 1TB/month of queries)
  2. Enable the BigQuery API
  3. Authenticate: gcloud auth application-default login
  4. Set your project ID in GCP_PROJECT below

Setup for ACLED:
  - Download a global CSV export from https://acleddata.com/data-export-tool/
  - Place the file(s) in data/raw/member_a/acled/

Output:
  - data/processed/member_a/ucdp_panel.csv      → monthly UCDP features per country
  - data/processed/member_a/acled_panel.csv      → monthly ACLED features per country
  - data/processed/member_a/gdelt_events.csv     → monthly GDELT conflict features per country
  - data/processed/member_a/member_a_final.csv   → merged panel, log1p-transformed, t-1 lagged
  - data/processed/member_a/feature_registry.csv → feature metadata
  - analysis/member_a/                           → missingness heatmaps, distribution plots

Notes:
  - UCDP type_of_violence: 1=state-based, 2=non-state, 3=one-sided violence
  - UCDP candidate events (2025+) contain unverified records — included for coverage
  - GDELT CAMEO codes: 14=protest, 18=assault, 19=fight, 20=mass violence
  - Country-months with no events are zero-filled (not dropped)
  - All features lagged by t-1 relative to the target window

"""

import os
import warnings
import pandas as pd
import numpy as np
import pycountry
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_a")
REPORT_DIR = os.path.join(BASE_DIR, "analysis", "member_a")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "missingness"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "profiles"), exist_ok=True)

PANEL_START = "1985-01"
PANEL_END   = "2025-12"
GCP_PROJECT = "gdelt-490620"

UCDP_GED_PATH       = os.path.join(RAW_DIR, "GEDEvent_v25_1.csv")
UCDP_CANDIDATE_PATH = os.path.join(RAW_DIR, "GEDEvent_v26_0_1.csv")
ACLED_DIR           = os.path.join(RAW_DIR, "member_a", "acled")

# UCDP type_of_violence: 1=state-based, 2=non-state, 3=one-sided
# GDELT CAMEO: 14=protest, 18=assault, 19=fight, 20=mass violence

LOG1P_FEATURES = [
    "ucdp_fatalities_best", "ucdp_fatalities_high", "ucdp_event_count",
    "ucdp_peak_event_fatalities", "ucdp_civilian_deaths",
    "acled_fatalities", "acled_event_count", "acled_battle_count",
    "gdelt_conflict_event_count",
]

ACLED_CACHE = os.path.join(RAW_DIR, "member_a", "acled_cache.csv")

# UCDP uses GW country names — many don't match ISO standards directly
GW_NAME_TO_ISO3 = {
    "Cambodia (Kampuchea)":            "KHM",
    "DR Congo (Zaire)":                "COD",
    "Myanmar (Burma)":                 "MMR",
    "Russia (Soviet Union)":           "RUS",
    "Serbia (Yugoslavia)":             "SRB",
    "Yemen (North Yemen)":             "YEM",
    "Zimbabwe (Rhodesia)":             "ZWE",
    "Madagascar (Malagasy)":           "MDG",
    "Kingdom of eSwatini (Swaziland)": "SWZ",
    "Ivory Coast":                     "CIV",
    "Bosnia-Herzegovina":              "BIH",
    "North Macedonia":                 "MKD",
    "South Sudan":                     "SSD",
    "Trinidad and Tobago":             "TTO",
    "United States of America":        "USA",
    "United Kingdom":                  "GBR",
    "Iran":                            "IRN",
    "Syria":                           "SYR",
    "Bolivia":                         "BOL",
    "Venezuela":                       "VEN",
    "Tanzania":                        "TZA",
    "Congo":                           "COG",
    "Laos":                            "LAO",
    "South Korea":                     "KOR",
    "North Korea":                     "PRK",
    "Taiwan":                          "TWN",
    "Moldova":                         "MDA",
    "Solomon Islands":                 "SLB",
    "Papua New Guinea":                "PNG",
    "Central African Republic":        "CAF",
    "Gambia":                          "GMB",
    "Guinea-Bissau":                   "GNB",
    "Kyrgyzstan":                      "KGZ",
    "Tajikistan":                      "TJK",
    "Uzbekistan":                      "UZB",
    "Turkmenistan":                    "TKM",
    "Azerbaijan":                      "AZE",
    "Armenia":                         "ARM",
    "Georgia":                         "GEO",
    "Turkey":                          "TUR",
}


def country_name_to_iso3(name):
    if name in GW_NAME_TO_ISO3:
        return GW_NAME_TO_ISO3[name]
    try:
        result = pycountry.countries.search_fuzzy(name)
        if result:
            return result[0].alpha_3
    except LookupError:
        pass
    return None


def build_full_panel(countries, start=PANEL_START, end=PANEL_END):
    months = pd.period_range(start=start, end=end, freq="M").astype(str)
    idx = pd.MultiIndex.from_product([countries, months], names=["country_iso3", "year_month"])
    return pd.DataFrame(index=idx).reset_index()


def apply_lag(df, feature_cols, lag=1):
    df = df.sort_values(["country_iso3", "year_month"]).copy()
    df[feature_cols] = df.groupby("country_iso3")[feature_cols].shift(lag)
    return df


def missingness_report(df, name):
    miss = df.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n  [{name}] missingness ({len(miss)} features with nulls):")
    print(miss.to_string())
    if miss.empty:
        return
    sample = df[miss.index].sample(min(500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(max(8, len(miss) * 0.5), 6))
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title(f"Missingness — {name}")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "missingness", f"{name}_missingness.png"), dpi=100)
    plt.close(fig)


def distribution_profile(df, feature_cols, name):
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


def load_ucdp():
    print("\n[UCDP] loading...")
    ged  = pd.read_csv(UCDP_GED_PATH, low_memory=False)
    cand = pd.read_csv(UCDP_CANDIDATE_PATH, low_memory=False)
    print(f"  GED: {len(ged):,} events ({ged['year'].min()}–{ged['year'].max()})")
    print(f"  Candidate: {len(cand):,} events ({cand['year'].min()}–{cand['year'].max()})")
    print(f"  Candidate code_status:\n{cand['code_status'].value_counts().to_string()}")

    df = pd.concat([ged, cand], ignore_index=True)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    df["year_month"] = df["date_start"].dt.to_period("M").astype(str)
    df = df[(df["year_month"] >= PANEL_START) & (df["year_month"] <= PANEL_END)]

    name_map = {n: country_name_to_iso3(n) for n in df["country"].unique()}
    unresolved = [k for k, v in name_map.items() if v is None]
    if unresolved:
        print(f"  WARNING: unresolved countries: {unresolved}")

    df["country_iso3"] = df["country"].map(name_map)
    df = df.dropna(subset=["country_iso3", "year_month"])
    print(f"  {len(df):,} events across {df['country_iso3'].nunique()} countries")
    return df


def aggregate_ucdp(df):
    print("\n[UCDP] aggregating to country-month...")
    df["fatality_range"] = df["high"] - df["low"]

    agg = df.groupby(["country_iso3", "year_month"]).agg(
        ucdp_event_count          =("id",               "count"),
        ucdp_fatalities_best      =("best",             "sum"),
        ucdp_fatalities_high      =("high",             "sum"),
        ucdp_civilian_deaths      =("deaths_civilians", "sum"),
        ucdp_peak_event_fatalities=("best",             "max"),
        ucdp_fatality_uncertainty =("fatality_range",   "mean"),
        ucdp_state_based_events   =("type_of_violence", lambda x: (x == 1).sum()),
        ucdp_non_state_events     =("type_of_violence", lambda x: (x == 2).sum()),
        ucdp_one_sided_events     =("type_of_violence", lambda x: (x == 3).sum()),
    ).reset_index()
    agg["ucdp_has_conflict"] = 1

    panel = build_full_panel(sorted(agg["country_iso3"].unique()))
    panel = panel.merge(agg, on=["country_iso3", "year_month"], how="left")

    fill_cols = [c for c in panel.columns if c.startswith("ucdp_") and c != "ucdp_has_conflict"]
    panel[fill_cols] = panel[fill_cols].fillna(0)
    panel["ucdp_has_conflict"] = panel["ucdp_has_conflict"].fillna(0).astype(int)

    print(f"  {panel.shape[0]:,} rows | {(panel['ucdp_has_conflict']==1).sum():,} active, {(panel['ucdp_has_conflict']==0).sum():,} zero-filled")
    return panel


def download_acled_api():
    """
    Download ACLED events via cookie-based auth. Reads credentials from
    ACLED_EMAIL and ACLED_PASSWORD environment variables.
    Results are cached to data/raw/member_a/acled_cache.csv to avoid
    re-downloading on every run.
    """
    import requests

    if os.path.exists(ACLED_CACHE):
        print(f"\n[ACLED] loading from cache: {ACLED_CACHE}")
        df = pd.read_csv(ACLED_CACHE, low_memory=False)
        print(f"  {len(df):,} events")
        return df

    email    = os.environ.get("ACLED_EMAIL")
    password = os.environ.get("ACLED_PASSWORD")
    if not email or not password:
        print("\n[ACLED] ACLED_EMAIL / ACLED_PASSWORD not set — skipping")
        return None

    print("\n[ACLED] authenticating...")
    session = requests.Session()
    resp = session.post(
        "https://acleddata.com/user/login?_format=json",
        json={"name": email, "pass": password},
        headers={"Content-Type": "application/json"},
    )
    if not resp.ok:
        print(f"  ERROR: login failed ({resp.status_code})")
        return None
    print("  logged in")

    os.makedirs(os.path.dirname(ACLED_CACHE), exist_ok=True)
    fields  = "event_date|country|event_type|fatalities"
    limit   = 5000
    offset  = 0
    records = []

    print("  downloading (this takes ~10 minutes for the full dataset)...")
    while True:
        r = session.get(
            "https://acleddata.com/api/acled/read",
            params={
                "fields":  fields,
                "limit":   limit,
                "offset":  offset,
                "event_date":       f"{PANEL_START[:4]}-01-01",
                "event_date_where": ">=",
            },
        )
        if not r.ok:
            print(f"  ERROR: request failed at offset {offset} ({r.status_code})")
            break
        batch = r.json().get("data", [])
        if not batch:
            break
        records.extend(batch)
        offset += limit
        if offset % 50000 == 0:
            print(f"  {offset:,} events downloaded...")

    df = pd.DataFrame(records)
    df.to_csv(ACLED_CACHE, index=False)
    print(f"  done — {len(df):,} events cached to {ACLED_CACHE}")
    return df


def aggregate_acled(df):
    print("\n[ACLED] aggregating to country-month...")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["year_month"] = df["event_date"].dt.to_period("M").astype(str)

    name_map = {n: country_name_to_iso3(n) for n in df["country"].unique()}
    unresolved = [k for k, v in name_map.items() if v is None]
    if unresolved:
        print(f"  WARNING: unresolved countries: {unresolved}")

    df["country_iso3"] = df["country"].map(name_map)
    df = df.dropna(subset=["country_iso3", "year_month"])
    df = df[(df["year_month"] >= PANEL_START) & (df["year_month"] <= PANEL_END)]
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)

    agg = df.groupby(["country_iso3", "year_month"]).agg(
        acled_event_count    =("event_type",  "count"),
        acled_fatalities     =("fatalities",  "sum"),
        acled_peak_fatalities=("fatalities",  "max"),
        acled_battle_count   =("event_type",  lambda x: (x == "Battles").sum()),
        acled_explosion_count=("event_type",  lambda x: (x == "Explosions/Remote violence").sum()),
        acled_violence_count =("event_type",  lambda x: (x == "Violence against civilians").sum()),
        acled_protest_count  =("event_type",  lambda x: (x == "Protests").sum()),
        acled_riot_count     =("event_type",  lambda x: (x == "Riots").sum()),
    ).reset_index()

    panel = build_full_panel(sorted(agg["country_iso3"].unique()))
    panel = panel.merge(agg, on=["country_iso3", "year_month"], how="left")
    acled_cols = [c for c in panel.columns if c.startswith("acled_")]
    panel[acled_cols] = panel[acled_cols].fillna(0)
    print(f"  {panel.shape[0]:,} rows")
    return panel


def download_gdelt_conflict_bigquery(project_id):
    try:
        from google.cloud import bigquery
    except ImportError:
        print("\n[GDELT] google-cloud-bigquery not installed — skipping")
        return pd.DataFrame()

    print(f"\n[GDELT] querying BigQuery ({project_id})...")
    client = bigquery.Client(project=project_id)

    query = """
    SELECT
        ActionGeo_CountryCode AS fips_code,
        FORMAT_DATE('%Y-%m', PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS year_month,
        COUNTIF(EventRootCode IN ('18','19','20')) AS gdelt_conflict_event_count,
        COUNTIF(EventRootCode = '14')              AS gdelt_protest_event_count,
        AVG(GoldsteinScale)                        AS gdelt_goldstein_mean
    FROM `gdelt-bq.full.events`
    WHERE
        EventRootCode IN ('14','18','19','20')
        AND SQLDATE IS NOT NULL
        AND ActionGeo_CountryCode IS NOT NULL
        AND SQLDATE BETWEEN 19850101 AND 20251231
    GROUP BY fips_code, year_month
    HAVING year_month IS NOT NULL
    ORDER BY fips_code, year_month
    """

    print("  running query...")
    df = client.query(query).to_dataframe()
    print(f"  {len(df):,} rows | {df['year_month'].min()} to {df['year_month'].max()}")

    fips_map_path = os.path.join(BASE_DIR, "data", "processed", "member_c", "fips_to_iso_mapping.csv")
    if os.path.exists(fips_map_path):
        fips_map = pd.read_csv(fips_map_path)
        def iso2_to_iso3(code):
            try:
                result = pycountry.countries.get(alpha_2=code)
                return result.alpha_3 if result else None
            except (AttributeError, LookupError):
                return None
        fips_map["iso3"] = fips_map["iso_code"].map(iso2_to_iso3)
        fips_dict = dict(zip(fips_map["fips_code"], fips_map["iso3"]))
        df["country_iso3"] = df["fips_code"].map(fips_dict)
        n_unmapped = df["country_iso3"].isna().sum()
        if n_unmapped > 0:
            unmapped = df[df["country_iso3"].isna()]["fips_code"].unique()
            print(f"  WARNING: {len(unmapped)} FIPS codes unmapped (territories/zones, dropping)")
        df = df.dropna(subset=["country_iso3"])
    else:
        print("  WARNING: fips_to_iso_mapping.csv not found")
        df["country_iso3"] = df["fips_code"]

    df = df.drop(columns=["fips_code"])
    df = df.groupby(["country_iso3", "year_month"]).agg(
        gdelt_conflict_event_count=("gdelt_conflict_event_count", "sum"),
        gdelt_protest_event_count =("gdelt_protest_event_count",  "sum"),
        gdelt_goldstein_mean      =("gdelt_goldstein_mean",       "mean"),
    ).reset_index()

    panel = build_full_panel(sorted(df["country_iso3"].unique()))
    panel = panel.merge(df, on=["country_iso3", "year_month"], how="left")
    panel[["gdelt_conflict_event_count", "gdelt_protest_event_count"]] = (
        panel[["gdelt_conflict_event_count", "gdelt_protest_event_count"]].fillna(0)
    )
    print(f"  panel: {panel.shape[0]:,} rows")
    return panel


def merge_panels(ucdp, acled, gdelt):
    print("\n[MERGE] combining panels...")
    panel = ucdp.copy()

    if acled is not None and not acled.empty:
        panel = panel.merge(acled, on=["country_iso3", "year_month"], how="outer")
        for col in [c for c in panel.columns if c.startswith("acled_")]:
            panel[col] = panel[col].fillna(0)

    if gdelt is not None and not gdelt.empty:
        panel = panel.merge(gdelt, on=["country_iso3", "year_month"], how="outer")
        for col in [c for c in panel.columns if c.startswith("gdelt_") and c != "gdelt_goldstein_mean"]:
            panel[col] = panel[col].fillna(0)

    panel = panel[
        (panel["year_month"] >= PANEL_START) & (panel["year_month"] <= PANEL_END)
    ].sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    print(f"  {panel.shape[0]:,} rows | {panel['country_iso3'].nunique()} countries | {panel['year_month'].nunique()} months")
    return panel


def save_feature_registry(df):
    feature_cols = [c for c in df.columns if c not in ["country_iso3", "year_month"]]
    records = []
    for col in feature_cols:
        series = df[col].dropna()
        source = "UCDP" if col.startswith("ucdp_") else "ACLED" if col.startswith("acled_") else "GDELT"
        records.append({
            "feature_name":  col,
            "source":        source,
            "dtype":         str(df[col].dtype),
            "missing_pct":   round(df[col].isna().mean() * 100, 2),
            "mean":          round(series.mean(), 4) if len(series) > 0 else None,
            "std":           round(series.std(),  4) if len(series) > 0 else None,
            "min":           round(series.min(),  4) if len(series) > 0 else None,
            "max":           round(series.max(),  4) if len(series) > 0 else None,
            "log1p_applied": col in LOG1P_FEATURES,
            "lag_months":    1,
        })
    path = os.path.join(OUTPUT_DIR, "feature_registry.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\n[REGISTRY] {len(records)} features → {path}")


def main():
    print("Member A — conflict data pipeline")

    ucdp_raw   = load_ucdp()
    ucdp_panel = aggregate_ucdp(ucdp_raw)
    ucdp_panel.to_csv(os.path.join(OUTPUT_DIR, "ucdp_panel.csv"), index=False)
    missingness_report(ucdp_panel, "ucdp")
    distribution_profile(ucdp_panel, [c for c in ucdp_panel.columns if c.startswith("ucdp_")], "ucdp")

    acled_raw   = download_acled_api()
    acled_panel = None
    if acled_raw is not None:
        acled_panel = aggregate_acled(acled_raw)
        acled_panel.to_csv(os.path.join(OUTPUT_DIR, "acled_panel.csv"), index=False)
        missingness_report(acled_panel, "acled")
        distribution_profile(acled_panel, [c for c in acled_panel.columns if c.startswith("acled_")], "acled")

    gdelt_panel = download_gdelt_conflict_bigquery(GCP_PROJECT)
    if not gdelt_panel.empty:
        gdelt_panel.to_csv(os.path.join(OUTPUT_DIR, "gdelt_events.csv"), index=False)
        missingness_report(gdelt_panel, "gdelt")
        distribution_profile(gdelt_panel, [c for c in gdelt_panel.columns if c.startswith("gdelt_")], "gdelt")

    final = merge_panels(ucdp_panel, acled_panel, gdelt_panel)

    for col in LOG1P_FEATURES:
        if col in final.columns:
            final[col] = np.log1p(final[col].clip(lower=0))

    feature_cols = [c for c in final.columns if c not in ["country_iso3", "year_month"]]
    final = apply_lag(final, feature_cols, lag=1)

    final.to_csv(os.path.join(OUTPUT_DIR, "member_a_final.csv"), index=False)
    print(f"\n[OUTPUT] {final.shape} → member_a_final.csv")

    save_feature_registry(final)
    print("\ndone.")


if __name__ == "__main__":
    main()
