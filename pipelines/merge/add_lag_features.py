"""
Add autoregressive features to the merged panel.

Motivation: the ViEWS competition rank 1 model is literally just the last
12 months of actuals. CCEW_tft (rank 5, best deep learning entry) added
3 rolling statistics of lagged fatalities as custom features. Autoregressive
signal dominates all other features for conflict forecasting.

Reads:  data/processed/merge/merged_panel.csv
Writes: data/processed/merge/merged_panel_ar.csv

Run this BEFORE preprocess.py. The preprocess pipeline will auto-detect
these as numeric features and standardise them like everything else.

If you skip this script, preprocess.py still works on merged_panel.csv —
you just won't have the autoregressive features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MERGE_DIR = BASE_DIR / "data" / "processed" / "merge"

TARGET = "ucdp_fatalities_best"

# Rolling windows to compute statistics over (in months)
WINDOWS = [3, 6, 12]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-country rolling statistics of the target variable.

    All features are lagged by 1 month (use only past data, no leakage).
    The target is already log1p-transformed by pipeline A.

    Features added per window W:
    - target_roll{W}_mean: rolling mean of past W months
    - target_roll{W}_median: rolling median of past W months
    - target_roll{W}_std: rolling std of past W months
    - target_roll{W}_max: rolling max of past W months
    - target_roll{W}_zscore: (t-1 value - rolling mean) / rolling std

    Also adds:
    - target_lag1: t-1 value (most recent month)
    - target_lag2: t-2 value
    - target_lag3: t-3 value
    - target_diff1: change from t-2 to t-1
    - target_months_since_spike: months since last y > log1p(500) ≈ 6.2
    """
    df = df.sort_values(["country_iso3", "year_month"]).copy()

    print(f"Adding autoregressive features from '{TARGET}'")
    print(f"  Rolling windows: {WINDOWS}")

    new_cols = []

    for iso3, group in df.groupby("country_iso3"):
        idx = group.index
        y = group[TARGET].values  # already log1p-transformed

        # Simple lags (shifted by 1 = no leakage)
        df.loc[idx, "target_lag1"] = pd.Series(y, index=idx).shift(1).values
        df.loc[idx, "target_lag2"] = pd.Series(y, index=idx).shift(2).values
        df.loc[idx, "target_lag3"] = pd.Series(y, index=idx).shift(3).values

        # First difference
        df.loc[idx, "target_diff1"] = (
            pd.Series(y, index=idx).shift(1) - pd.Series(y, index=idx).shift(2)
        ).values

        # Rolling statistics (shift(1) ensures we only use past data)
        s = pd.Series(y, index=idx).shift(1)
        for w in WINDOWS:
            roll = s.rolling(window=w, min_periods=1)
            df.loc[idx, f"target_roll{w}_mean"] = roll.mean().values
            df.loc[idx, f"target_roll{w}_median"] = roll.median().values
            df.loc[idx, f"target_roll{w}_std"] = roll.std().values
            df.loc[idx, f"target_roll{w}_max"] = roll.max().values

            # Z-score: how unusual is the most recent value vs rolling history
            roll_mean = roll.mean()
            roll_std = roll.std()
            df.loc[idx, f"target_roll{w}_zscore"] = (
                (s - roll_mean) / roll_std.replace(0, np.nan)
            ).values

        # Months since last spike (log1p(500) ≈ 6.21)
        spike_threshold = np.log1p(500)
        lagged = pd.Series(y, index=idx).shift(1)
        is_spike = lagged > spike_threshold
        months_since = pd.Series(np.nan, index=idx)
        counter = np.nan
        for i, (ix, spike) in enumerate(is_spike.items()):
            if spike:
                counter = 0
            elif not np.isnan(counter):
                counter += 1
            months_since.iloc[i] = counter
        df.loc[idx, "target_months_since_spike"] = months_since.values

    # Report what was added
    ar_cols = [c for c in df.columns if c.startswith("target_")]
    print(f"  Added {len(ar_cols)} autoregressive features:")
    for col in sorted(ar_cols):
        n_valid = df[col].notna().sum()
        print(f"    {col}: {n_valid:,}/{len(df):,} non-null")

    return df


def main():
    input_path = MERGE_DIR / "merged_panel.csv"
    output_path = MERGE_DIR / "merged_panel_ar.csv"

    df = pd.read_csv(input_path)
    print(f"Loaded {input_path}: {df.shape}")

    df = add_lag_features(df)

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}: {df.shape}")
    print(f"Run preprocess.py next (it will auto-detect the new features)")


if __name__ == "__main__":
    main()
