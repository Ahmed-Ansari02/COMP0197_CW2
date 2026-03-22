"""
Preprocess the merged panel into model-ready features.

Reads:  data/processed/merge/merged_panel.csv
Writes: data/processed/merge/model_ready.csv

Steps:
1. Drop identifiers, availability flags, and known duplicate columns
2. Log1p-transform heavy-tailed features (skew > 3, non-negative, max > 100)
   — some pipeline a and c features are already log1p'd by their pipeline, skip those
3. Drop redundant features (|Pearson r| > 0.9 on training data)
   — greedy: for each correlated pair, drop the one with less correlation to target
4. Z-score standardise using training period stats only
5. Clip extreme outliers beyond ±10 std (dont want to blow up LSTM grads)
6. Fill remaining NaN with 0 (post-standardisation, 0 = population mean), so this is like 'avg' value
7. Save with a feature registry

Does NOT touch: country_iso3, year_month, target column (kept as-is for splits)
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MERGE_DIR = BASE_DIR / "data" / "processed" / "merge"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

# Features already log1p-transformed by member A's pipeline
ALREADY_LOG1P = {
    "ucdp_event_count", "ucdp_fatalities_best", "ucdp_fatalities_high",
    "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
    "ucdp_fatality_uncertainty", "gdelt_conflict_event_count",
    "gdelt_goldstein_mean",
}

# Features already log1p-transformed by member C's pipeline
ALREADY_LOG1P |= {
    "gpr_global", "gpr_acts", "event_count", "tone_std",
}

# Columns to exclude from features entirely
DROP_COLUMNS = {
    # Identifiers
    "country_iso3", "year_month", "year", "month", "region",
    # Availability flags (not real features, just metadata)
    "vdem_stale_flag", "vdem_available", "reign_available",
    "fx_available", "food_available", "gdp_available",
    # Coup event is a near-duplicate of pt_coup_event
    "coup_event",
}

# Genuinely redundant features to drop before correlation filter.
# These are known duplicates/derivations from inspecting the pipelines:
#   - fx_volatility_log: log of fx_volatility (r=1.0 after our log1p)
#   - food_price_anomaly: raw version of food_price_anomaly_log (r=0.997)
#   - pt_coup_event: identical to pt_coup_count (r=1.0)
#   - governance_deficit: derived from v2x_libdem, v2x_polyarchy, v2x_rule
#   - repression_index: derived from v2x_corr, v2x_execorr, v2x_clphy
#   - ucdp_peak_event_fatalities: near-identical to ucdp_fatalities_high (r=0.96)
#   - v2x_civlib: composite of v2x_clphy + v2x_clpol (r>0.95 with both)
#   - v2x_clpol: composite of v2x_freexp_altinf + v2x_frassoc_thick (r>0.97)
#   - v2xnp_regcorr: near-identical to v2x_execorr (r=0.988)
#   - v2x_corr: near-identical to v2x_execorr (r=0.947)
KNOWN_REDUNDANT = {
    "fx_volatility_log",
    "food_price_anomaly",
    "pt_coup_event",
    "governance_deficit",
    "repression_index",
    "ucdp_peak_event_fatalities",
    "v2x_civlib",
    "v2x_clpol",
    "v2xnp_regcorr",
    "v2x_corr",
}

# Target — keep in output but don't standardise
TARGET = "ucdp_fatalities_best"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def identify_log1p_candidates(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Find non-negative, heavy-tailed features that need log1p."""
    candidates = []
    for col in features:
        if col in ALREADY_LOG1P:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if vals.min() >= 0 and vals.max() > 100 and vals.skew() > 3:
            candidates.append(col)
    return candidates


def drop_redundant_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = 0.9,
) -> tuple[list[str], list[str]]:
    """
    Drop one feature from each pair with |Pearson r| > threshold.

    Greedy approach: for each correlated pair, keep the one that has
    higher absolute correlation with the target variable (more useful
    for prediction). This way we keep the most informative version.

    Returns:
        (kept_features, dropped_features)
    """
    corr = df[feature_cols].corr()

    # Correlation of each feature with target
    target_corr = {}
    for col in feature_cols:
        both = df[[col, target_col]].dropna()
        if len(both) > 10:
            target_corr[col] = abs(both[col].corr(both[target_col]))
        else:
            target_corr[col] = 0.0

    to_drop = set()
    drop_reasons = []

    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in to_drop:
                continue
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                fi, fj = feature_cols[i], feature_cols[j]
                # Drop the one less correlated with target
                if target_corr[fi] >= target_corr[fj]:
                    to_drop.add(fj)
                    drop_reasons.append((fj, fi, round(r, 3)))
                else:
                    to_drop.add(fi)
                    drop_reasons.append((fi, fj, round(r, 3)))

    kept = [c for c in feature_cols if c not in to_drop]
    dropped = sorted(to_drop)

    if drop_reasons:
        print(f"\nDropping {len(dropped)} redundant features (|r| > {threshold}):")
        for dropped_feat, kept_feat, r in sorted(drop_reasons):
            print(f"  {dropped_feat:40s} (r={r} with {kept_feat})")

    return kept, dropped


def preprocess(df: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline.

    Returns:
        (processed_df, info_dict with feature lists and stats)
    """
    # Separate identifiers and target
    id_cols = ["country_iso3", "year_month"]
    ids = df[id_cols].copy()

    # Determine feature columns
    all_cols = set(df.columns)
    feature_cols = sorted([
        c for c in all_cols
        if c not in DROP_COLUMNS
        and c != TARGET
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ])

    print(f"Features: {len(feature_cols)}")
    print(f"Target: {TARGET}")

    features = df[feature_cols].copy()
    target = df[TARGET].copy()

    # Step 1: log1p transform heavy-tailed features
    log1p_cols = identify_log1p_candidates(features, feature_cols)
    if log1p_cols:
        print(f"\nLog1p transforming {len(log1p_cols)} heavy-tailed features:")
        for col in log1p_cols:
            old_max = features[col].max()
            features[col] = np.log1p(features[col])
            print(f"  {col}: max {old_max:.0f} → {features[col].max():.2f}")

    # Step 2: drop known redundant features
    redundant_present = [c for c in feature_cols if c in KNOWN_REDUNDANT]
    if redundant_present:
        print(f"\nDropping {len(redundant_present)} known redundant features:")
        for col in sorted(redundant_present):
            print(f"  {col}")
        feature_cols = [c for c in feature_cols if c not in KNOWN_REDUNDANT]
        features = features[feature_cols]
    redundant_dropped = redundant_present

    # Step 3: z-score standardise using training period only
    train_mask = df["year_month"] <= train_end
    n_train = train_mask.sum()
    print(f"\nComputing z-score stats from training period (≤{train_end}): {n_train:,} rows")

    train_means = features[train_mask].mean()
    train_stds = features[train_mask].std()

    # Avoid division by zero for constant features
    constant_features = train_stds[train_stds < 1e-10].index.tolist()
    if constant_features:
        print(f"  Dropping {len(constant_features)} constant features: {constant_features}")
        features = features.drop(columns=constant_features)
        feature_cols = [c for c in feature_cols if c not in constant_features]
        train_means = train_means.drop(constant_features)
        train_stds = train_stds.drop(constant_features)

    features = (features - train_means) / train_stds

    # Step 3: clip extreme outliers (> 10 std from mean, post-standardisation)
    n_clipped = (features.abs() > 10).sum().sum()
    if n_clipped > 0:
        features = features.clip(-10, 10)
        print(f"Clipped {n_clipped:,} values beyond ±10 std")

    # Step 4: fill remaining NaN with 0 (= population mean post-standardisation)
    n_nan = features.isnull().sum().sum()
    total = features.shape[0] * features.shape[1]
    print(f"\nFilling {n_nan:,}/{total:,} NaN with 0 ({n_nan/total*100:.1f}%)")
    features = features.fillna(0)

    # Reassemble
    result = pd.concat([ids, features, target.rename(TARGET)], axis=1)

    info = {
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "log1p_transformed": log1p_cols,
        "redundant_dropped": redundant_dropped,
        "constant_dropped": constant_features,
        "train_end": train_end,
        "train_rows": int(n_train),
        "total_rows": len(df),
    }

    return result, info


def main():
    config = load_config()
    train_end = config["splits"]["train_end"]

    # Load merged panel (prefer AR version if available)
    ar_path = MERGE_DIR / "merged_panel_ar.csv"
    plain_path = MERGE_DIR / "merged_panel.csv"
    if ar_path.exists():
        input_path = ar_path
        print(f"Using autoregressive version: {ar_path.name}")
    else:
        input_path = plain_path
        print(f"No AR features found — using {plain_path.name}")
    df = pd.read_csv(input_path)
    print(f"Loaded {input_path}: {df.shape}")

    # Preprocess
    result, info = preprocess(df, train_end)

    # Save
    output_path = MERGE_DIR / "model_ready.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}: {result.shape}")

    # Save feature registry (kept features + dropped features)
    kept = pd.DataFrame({
        "feature": info["feature_cols"],
        "status": "kept",
        "log1p_applied": [c in info["log1p_transformed"] for c in info["feature_cols"]],
        "was_already_log1p": [c in ALREADY_LOG1P for c in info["feature_cols"]],
    })
    dropped = pd.DataFrame({
        "feature": info["redundant_dropped"],
        "status": "dropped_redundant",
        "log1p_applied": False,
        "was_already_log1p": False,
    })
    registry = pd.concat([kept, dropped], ignore_index=True)
    registry_path = MERGE_DIR / "feature_registry.csv"
    registry.to_csv(registry_path, index=False)
    print(f"Feature registry: {registry_path}")

    print(f"\nSummary:")
    print(f"  {info['n_features']} features (from {info['n_features'] + len(info['redundant_dropped'])})")
    print(f"  {len(info['log1p_transformed'])} log1p-transformed")
    print(f"  {len(info['redundant_dropped'])} redundant features dropped (|r| > 0.9)")
    print(f"  {len(info['constant_dropped'])} constant features dropped")
    print(f"  z-score stats from ≤{train_end} ({info['train_rows']:,} rows)")


if __name__ == "__main__":
    main()
