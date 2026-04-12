"""
Merge member_a, member_b, member_c datasets into a unified panel.

Join key: (country_iso3, year_month)
Strategy: outer join to preserve all country-month observations.
Output: data/processed/merge/merged_panel.csv

We decide to drop these columns:
- gpr_country: GPR country-level index only exists for 44 countries
  (Caldara & Iacoviello only publish it for major economies). Including it
  would mean 78% NaN, and a missingness indicator would just proxy for
  "is this a major economy" rather than actual geopolitical risk.
  We keep gpr_global and gpr_acts instead.
  These features acc have much better coverage.

Broadcast columns:
- pipeline c macro indicators (VIX, oil, gold, DXY, yields, wheat, copper,
  T-bills) and GPR global/acts are country-invariant (one value per month).
  prev pipeline had only 44 GPR countries, but they apply
  globally. We broadcast them to all countries after merging.
- refill the GDELT tone columns (tone_mean, tone_min, tone_max, tone_std, event_count,
  goldstein_mean) to the full datatset.

Includes merge diagnostics to verify join quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "merge"

# ISO-3 region mapping for imputation fallback
REGION_MAP = {
    "Eastern Africa": ["BDI", "COM", "DJI", "ERI", "ETH", "KEN", "MDG", "MWI",
                        "MOZ", "RWA", "SOM", "SSD", "TZA", "UGA", "ZMB", "ZWE"],
    "Western Africa": ["BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB",
                        "LBR", "MLI", "MRT", "NER", "NGA", "SEN", "SLE", "TGO"],
    "Central Africa": ["AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "STP"],
    "Northern Africa": ["DZA", "EGY", "LBY", "MAR", "SDN", "TUN"],
    "Southern Africa": ["BWA", "SWZ", "LSO", "NAM", "ZAF"],
    "Central Asia": ["KAZ", "KGZ", "TJK", "TKM", "UZB"],
    "Eastern Asia": ["CHN", "JPN", "MNG", "PRK", "KOR", "TWN"],
    "Southern Asia": ["AFG", "BGD", "BTN", "IND", "IRN", "MDV", "NPL", "PAK", "LKA"],
    "South-Eastern Asia": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP",
                            "THA", "TLS", "VNM"],
    "Western Asia": ["ARM", "AZE", "BHR", "CYP", "GEO", "IRQ", "ISR", "JOR",
                      "KWT", "LBN", "OMN", "PSE", "QAT", "SAU", "SYR", "TUR",
                      "ARE", "YEM"],
    "Eastern Europe": ["BLR", "BGR", "CZE", "HUN", "MDA", "POL", "ROU", "RUS",
                        "SVK", "UKR"],
    "South America": ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY",
                       "PER", "SUR", "URY", "VEN"],
    "Central America": ["BLZ", "CRI", "SLV", "GTM", "HND", "MEX", "NIC", "PAN"],
    "Caribbean": ["CUB", "DOM", "HTI", "JAM", "TTO"],
}

COUNTRY_TO_REGION = {}
for region, countries in REGION_MAP.items():
    for iso3 in countries:
        COUNTRY_TO_REGION[iso3] = region


# Columns from member C that are the same value for every country in a given month
# (global macro indicators — no country dimension)
GLOBAL_COLUMNS = [
    "gpr_global", "gpr_acts",
    "vix_mean", "vix_vol", "vix_pct_chg",
    "wti_oil_mean", "wti_oil_vol", "wti_oil_pct_chg",
    "gold_mean", "gold_vol", "gold_pct_chg",
    "dxy_mean", "dxy_vol", "dxy_pct_chg",
    "us_10y_yield_mean", "us_10y_yield_vol", "us_10y_yield_pct_chg",
    "wheat_mean", "wheat_vol", "wheat_pct_chg",
    "copper_mean", "copper_vol", "copper_pct_chg",
    "us_13w_tbill_mean", "us_13w_tbill_vol", "us_13w_tbill_pct_chg",
]

# Drop from member C: only exists for 44 countries, would be a confounded proxy
DROP_COLUMNS = ["gpr_country"]


def load_member_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean the three member datasets."""
    a = pd.read_csv(PROCESSED_DIR / "member_a" / "member_a_final.csv")
    b = pd.read_csv(PROCESSED_DIR / "member_b" / "member_b_final.csv")
    c = pd.read_csv(PROCESSED_DIR / "member_c" / "member_c_final.csv")

    # Strip whitespace from ALL string columns and column names
    # (member C's CSV has trailing spaces in country codes and column names)
    for df in [a, b, c]:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

    # Drop gpr_country — only 44 countries, confounded with "major economy" status
    c = c.drop(columns=[col for col in DROP_COLUMNS if col in c.columns])
    print(f"Dropped {DROP_COLUMNS} from member C (only 44 countries, confounded)")

    print(f"Member A: {a.shape} | {a.country_iso3.nunique()} countries | {a.year_month.min()} to {a.year_month.max()}")
    print(f"Member B: {b.shape} | {b.country_iso3.nunique()} countries | {b.year_month.min()} to {b.year_month.max()}")
    print(f"Member C: {c.shape} | {c.country_iso3.nunique()} countries | {c.year_month.min()} to {c.year_month.max()}")

    return a, b, c


def diagnose_join_keys(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> None:
    """Print detailed diagnostics about join key compatibility."""
    print(f"\n{'='*60}")
    print("JOIN KEY DIAGNOSTICS")
    print(f"{'='*60}")

    a_iso = set(a.country_iso3.unique())
    b_iso = set(b.country_iso3.unique())
    c_iso = set(c.country_iso3.unique())

    print(f"\nCountry overlap:")
    print(f"  A ∩ B: {len(a_iso & b_iso)} countries")
    print(f"  A ∩ C: {len(a_iso & c_iso)} countries")
    print(f"  B ∩ C: {len(b_iso & c_iso)} countries")
    print(f"  A ∩ B ∩ C: {len(a_iso & b_iso & c_iso)} countries")

    only_a = a_iso - b_iso - c_iso
    only_b = b_iso - a_iso - c_iso
    only_c = c_iso - a_iso - b_iso
    if only_a:
        print(f"  Only in A ({len(only_a)}): {sorted(only_a)}")
    if only_b:
        print(f"  Only in B ({len(only_b)}): {sorted(only_b)}")
    if only_c:
        print(f"  Only in C ({len(only_c)}): {sorted(only_c)}")

    # Date format check
    print(f"\nDate format samples:")
    print(f"  A: {list(a.year_month.head(3))}")
    print(f"  B: {list(b.year_month.head(3))}")
    print(f"  C: {list(c.year_month.head(3))}")

    # Check for duplicates on join key
    for name, df in [("A", a), ("B", b), ("C", c)]:
        n_dups = df.duplicated(subset=["country_iso3", "year_month"]).sum()
        if n_dups > 0:
            print(f"  WARNING: {name} has {n_dups} duplicate (iso3, year_month) rows!")
        else:
            print(f"  {name}: no duplicate keys ✓")

    # Column overlap
    a_cols = set(a.columns) - {"country_iso3", "year_month"}
    b_cols = set(b.columns) - {"country_iso3", "year_month"}
    c_cols = set(c.columns) - {"country_iso3", "year_month"}

    overlap_ab = a_cols & b_cols
    overlap_ac = a_cols & c_cols
    overlap_bc = b_cols & c_cols

    if overlap_ab or overlap_ac or overlap_bc:
        print(f"\nWARNING — Overlapping feature columns:")
        if overlap_ab:
            print(f"  A ∩ B: {sorted(overlap_ab)}")
        if overlap_ac:
            print(f"  A ∩ C: {sorted(overlap_ac)}")
        if overlap_bc:
            print(f"  B ∩ C: {sorted(overlap_bc)}")
    else:
        print(f"\nNo overlapping feature columns ✓")

    print(f"{'='*60}")


def merge_panels(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Outer join all three member datasets on (country_iso3, year_month)."""
    join_keys = ["country_iso3", "year_month"]

    # First merge: A + B
    merged = pd.merge(a, b, on=join_keys, how="outer", suffixes=("", "_dup_b"))
    print(f"After A+B merge: {merged.shape}")

    # Second merge: (A+B) + C
    merged = pd.merge(merged, c, on=join_keys, how="outer", suffixes=("", "_dup_c"))
    print(f"After A+B+C merge: {merged.shape}")

    # Handle any duplicate columns
    dup_cols = [c for c in merged.columns if "_dup_b" in c or "_dup_c" in c]
    if dup_cols:
        print(f"Dropping {len(dup_cols)} duplicate columns: {dup_cols}")
        merged = merged.drop(columns=dup_cols)

    # Add region column
    merged["region"] = merged["country_iso3"].map(COUNTRY_TO_REGION)

    # Sort
    merged = merged.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    return merged


def broadcast_global_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill global (country-invariant) columns across all countries.

    Member C's pipeline only stored these for 44 GPR countries, but macro
    indicators like VIX, oil prices, etc. are the same for every country
    in a given month. Extract one value per month and broadcast.
    """
    globals_present = [col for col in GLOBAL_COLUMNS if col in df.columns]
    if not globals_present:
        return df

    # Extract one value per month (take first non-null)
    monthly_globals = df.groupby("year_month")[globals_present].first().reset_index()

    # Drop the sparse versions, merge the full ones back
    df = df.drop(columns=globals_present)
    df = df.merge(monthly_globals, on="year_month", how="left")

    filled = df[globals_present].notna().mean().mean() * 100
    print(f"Broadcast {len(globals_present)} global columns to all countries ({filled:.0f}% coverage)")

    return df


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year and month from year_month."""
    df["year"] = df["year_month"].str[:4].astype(int)
    df["month"] = df["year_month"].str[5:7].astype(int)
    return df


def validate_merge(merged: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> None:
    """Post-merge validation checks."""
    print(f"\n{'='*60}")
    print("POST-MERGE VALIDATION")
    print(f"{'='*60}")

    n_rows = len(merged)
    n_countries = merged.country_iso3.nunique()
    n_months = merged.year_month.nunique()

    print(f"Merged: {n_rows:,} rows × {merged.shape[1]} cols")
    print(f"Countries: {n_countries}, Months: {n_months}")
    print(f"Expected max rows (countries × months): {n_countries * n_months:,}")

    # Check: did C's data actually join?
    c_countries = set(c.country_iso3.unique())
    c_feature = [col for col in c.columns if col not in ["country_iso3", "year_month"]][0]
    c_data_in_merged = merged[merged.country_iso3.isin(c_countries)][c_feature].notna().sum()
    print(f"\nMember C join check ('{c_feature}' non-null for C countries): {c_data_in_merged:,}")
    if c_data_in_merged == 0:
        print("  ✗ FAILED — Member C data did not join! Check country codes.")
    else:
        print(f"  ✓ OK — {c_data_in_merged:,} values joined from C")

    # Check: did A's data survive?
    a_feature = [col for col in a.columns if col not in ["country_iso3", "year_month"]][0]
    a_notnull = merged[a_feature].notna().sum()
    print(f"Member A data present ('{a_feature}'): {a_notnull:,} / {n_rows:,}")

    # Check: did B's data survive?
    b_feature = [col for col in b.columns if col not in ["country_iso3", "year_month"]][0]
    b_notnull = merged[b_feature].notna().sum()
    print(f"Member B data present ('{b_feature}'): {b_notnull:,} / {n_rows:,}")

    # Missingness overview by source
    a_cols = [c for c in a.columns if c not in ["country_iso3", "year_month"]]
    b_cols = [c for c in b.columns if c not in ["country_iso3", "year_month"]]
    c_cols = [c for c in c.columns if c not in ["country_iso3", "year_month"]]

    a_cols_present = [c for c in a_cols if c in merged.columns]
    b_cols_present = [c for c in b_cols if c in merged.columns]
    c_cols_present = [c for c in c_cols if c in merged.columns]

    a_missing_pct = merged[a_cols_present].isnull().mean().mean() * 100
    b_missing_pct = merged[b_cols_present].isnull().mean().mean() * 100
    c_missing_pct = merged[c_cols_present].isnull().mean().mean() * 100

    print(f"\nAvg missingness by source:")
    print(f"  Member A columns: {a_missing_pct:.1f}%")
    print(f"  Member B columns: {b_missing_pct:.1f}%")
    print(f"  Member C columns: {c_missing_pct:.1f}%")

    # Row count sanity
    print(f"\nRow count check:")
    print(f"  A rows: {len(a):,}")
    print(f"  B rows: {len(b):,}")
    print(f"  C rows: {len(c):,}")
    print(f"  Merged rows: {n_rows:,}")

    # Expectation: outer join should be <= sum of unique keys
    a_keys = set(zip(a.country_iso3, a.year_month))
    b_keys = set(zip(b.country_iso3, b.year_month))
    c_keys = set(zip(c.country_iso3, c.year_month))
    expected = len(a_keys | b_keys | c_keys)
    print(f"  Union of all keys: {expected:,}")
    if n_rows == expected:
        print(f"  ✓ Row count matches expected union")
    else:
        print(f"  ✗ Row count mismatch! Merged={n_rows:,}, Expected={expected:,}")

    print(f"{'='*60}")


def report_missingness(df: pd.DataFrame) -> None:
    """Print missingness summary."""
    n_rows = len(df)
    print(f"\nMissingness by column (top 20):")

    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(1)
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False).head(20)
    for col, pct in missing_report.items():
        print(f"  {col}: {pct}%")

    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\nTotal: {total_missing:,}/{total_cells:,} cells missing ({total_missing/total_cells*100:.1f}%)")


def backfill_gdelt_tone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill GDELT tone columns for countries missing from member C's output.

    Member C only had tone for 44 GPR countries. If fetch_gdelt_tone.py has
    been run, we have tone for all ~200 countries and can fill the gaps.
    """
    gdelt_path = OUTPUT_DIR / "gdelt_tone_all.csv"
    if not gdelt_path.exists():
        print(f"No full GDELT tone data at {gdelt_path} — run fetch_gdelt_tone.py first")
        return df

    tone_full = pd.read_csv(gdelt_path)
    print(f"Loading full GDELT tone: {tone_full.shape} ({tone_full.country_iso3.nunique()} countries)")

    tone_cols = ["tone_mean", "tone_min", "tone_max", "tone_std", "event_count", "goldstein_mean"]
    tone_cols_present = [c for c in tone_cols if c in df.columns and c in tone_full.columns]

    if not tone_cols_present:
        print("  No matching tone columns to backfill")
        return df

    # For each tone column: where the merged panel has NaN, fill from full GDELT
    before_nan = df[tone_cols_present].isnull().sum().sum()

    tone_lookup = tone_full.set_index(["country_iso3", "year_month"])[tone_cols_present]
    merged_idx = df.set_index(["country_iso3", "year_month"])

    for col in tone_cols_present:
        mask = merged_idx[col].isna()
        fill_values = tone_lookup[col].reindex(merged_idx.index)
        merged_idx.loc[mask, col] = fill_values[mask]

    df[tone_cols_present] = merged_idx[tone_cols_present].values

    after_nan = df[tone_cols_present].isnull().sum().sum()
    filled = before_nan - after_nan
    print(f"  Backfilled {filled:,} tone values ({before_nan:,} → {after_nan:,} NaN)")

    return df


def main():
    # Load
    a, b, c = load_member_data()

    # Pre-merge diagnostics
    diagnose_join_keys(a, b, c)

    # Merge
    merged = merge_panels(a, b, c)
    merged = broadcast_global_columns(merged)
    merged = backfill_gdelt_tone(merged)
    merged = add_temporal_columns(merged)

    # Post-merge validation
    validate_merge(merged, a, b, c)

    # Missingness report
    report_missingness(merged)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "merged_panel.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return merged


if __name__ == "__main__":
    main()
