"""
Filters member_a_final.csv down to the subset of features used for modelling,
and adds log-difference change features to capture month-on-month escalation.

Dropped features (redundant or overlap with member C):
  - gdelt_protest_event_count: overlap with member C's GDELT tone features

Added change features (log-difference handles zero-heavy data gracefully):
  - ucdp_fatalities_best_ld, ucdp_event_count_ld, gdelt_conflict_event_count_ld

Run after generate_conflict_dataset.py.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_a")

DROP_COLS = [
    "gdelt_protest_event_count",
]

# Features to compute log-difference on (already log1p-transformed in the main pipeline)
# log-diff = log1p(t-1) - log1p(t-2), captures escalation without division-by-zero
CHANGE_COLS = [
    "ucdp_fatalities_best",
    "ucdp_event_count",
    "gdelt_conflict_event_count",
    "acled_fatalities",
    "acled_event_count",
    "acled_political_violence_count",
]

df = pd.read_csv(os.path.join(OUTPUT_DIR, "member_a_final.csv"))
filtered = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Zero-fill count/fatality features for countries present in the panel but absent
# from a given source (e.g. GDELT-only countries have no UCDP record = 0 events,
# not missing). Goldstein left as NaN — no events means no meaningful score.
zero_fill_cols = [
    c for c in filtered.columns
    if c not in ["country_iso3", "year_month", "gdelt_goldstein_mean"]
    and "ld" not in c
]
filtered[zero_fill_cols] = filtered[zero_fill_cols].fillna(0)

# Compute log-differences within each country
# Since features are already log1p-transformed and lagged, we take first difference
filtered = filtered.sort_values(["country_iso3", "year_month"])
for col in CHANGE_COLS:
    if col in filtered.columns:
        filtered[f"{col}_ld"] = (
            filtered.groupby("country_iso3")[col].diff()
        )

filtered.to_csv(os.path.join(OUTPUT_DIR, "member_a_model_features.csv"), index=False)

print(f"Full:     {df.shape}")
print(f"Filtered: {filtered.shape}")
print(f"Features: {filtered.columns.tolist()}")
