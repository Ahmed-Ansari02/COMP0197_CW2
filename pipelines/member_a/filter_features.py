"""
Filters member_a_final.csv down to the subset of features used for modelling,
and adds log-difference change features to capture month-on-month escalation.

Dropped features (redundant or overlap with member C):
  - acled_protest_count / acled_riot_count: overlap with member C's GDELT protest signal
  - gdelt_protest_event_count: overlap with member C's GDELT tone features
  - acled_peak_fatalities: redundant with ucdp_peak_event_fatalities and acled_fatalities
  - acled_event_count: redundant once event type breakdown is available

Added change features (log-difference handles zero-heavy data gracefully):
  - ucdp_fatalities_best_ld, ucdp_event_count_ld
  - acled_fatalities_ld, gdelt_conflict_event_count_ld

Run after generate_conflict_dataset.py.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_a")

DROP_COLS = [
    "acled_protest_count",
    "acled_riot_count",
    "gdelt_protest_event_count",
    "acled_peak_fatalities",
    "acled_event_count",
]

# Features to compute log-difference on (already log1p-transformed in the main pipeline)
# log-diff = log1p(t-1) - log1p(t-2), captures escalation without division-by-zero
CHANGE_COLS = [
    "ucdp_fatalities_best",
    "ucdp_event_count",
    "acled_fatalities",
    "gdelt_conflict_event_count",
]

df = pd.read_csv(os.path.join(OUTPUT_DIR, "member_a_final.csv"))
filtered = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

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
