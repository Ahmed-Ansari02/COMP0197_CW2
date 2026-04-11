"""
Evaluate the trained TFT model against ViEWS benchmarks.

ViEWS competition window: July 2024 - June 2025.
Only evaluates on months with available UCDP ground truth (7 of 12).

Multi-horizon evaluation: 1, 3, and 6 months ahead.
  For horizon h, the model sees data up to t-h and predicts t.
  This simulates real forecasting with a known lead time.

Baselines (point-estimate only, MAE/RMSE):
  - Persistence: predict y[t-h] (last observed value)
  - Moving average (6-month): predict mean(y[t-h-5 : t-h+1])
  - XGBoost: trained on features[t-h] -> y[t], same train split as TFT
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

from src.models.tft.model import ConflictForecaster
from src.models.tft.train import get_feature_list
from src.evaluation.baselines import VIEWS_WINDOW_START, VIEWS_WINDOW_END
from src.evaluation.metrics import mae_point, rmse_point
from src.evaluation.eval_runner import (
    find_usable_months,
    compare_models,
    per_month_comparison,
    print_comparison,
    print_diagnostics,
)


def build_test_samples(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "ucdp_fatalities_best",
    window_size: int = 24,
    test_start: str = VIEWS_WINDOW_START,
    test_end: str | None = VIEWS_WINDOW_END,
    horizon: int = 1,
):
    """
    Build test windows where the TARGET month is in [test_start, test_end].

    For horizon h, the window covers [i - window_size - h + 1 : i - h + 1],
    so the model only sees data up to h months before the target.
    """
    x_all, y_all, dates, countries = [], [], [], []

    for iso3 in df["country_iso3"].unique():
        country_df = df[df["country_iso3"] == iso3].sort_values("year_month")
        n = len(country_df)
        if n <= window_size + horizon - 1:
            continue

        feature_data = country_df[features].values.astype(np.float32)
        targets = np.expm1(country_df[target_col].values.astype(np.float32))
        year_months = country_df["year_month"].values

        for i in range(window_size + horizon - 1, n):
            ym = year_months[i]
            y = targets[i]
            if ym < test_start:
                continue
            if test_end and ym > test_end:
                continue
            if np.isnan(y):
                continue
            x_start = i - window_size - horizon + 1
            x_end = i - horizon + 1
            x_all.append(feature_data[x_start:x_end])
            y_all.append(y)
            dates.append(ym)
            countries.append(iso3)

    return x_all, np.array(y_all), np.array(dates), np.array(countries)


def build_baselines(
    df: pd.DataFrame,
    target_col: str,
    countries: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    ma_window: int = 6,
) -> dict[str, np.ndarray]:
    """
    Compute persistence and moving-average baseline predictions for each
    test observation. Returns raw fatality predictions (not log-space).
    """
    # Build lookup: (iso3, year_month) -> raw fatalities
    lookup = {}
    for iso3 in df["country_iso3"].unique():
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month")
        raw = np.expm1(cdf[target_col].values.astype(np.float32))
        for ym, val in zip(cdf["year_month"].values, raw):
            lookup[(iso3, ym)] = val

    # Build a sorted month list for offset arithmetic
    all_months = sorted(df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months)}

    persistence, moving_avg = [], []
    for iso3, ym in zip(countries, dates):
        t_idx = month_to_idx.get(ym)
        if t_idx is None or t_idx - horizon < 0:
            persistence.append(np.nan)
            moving_avg.append(np.nan)
            continue

        # Persistence: y[t - horizon]
        ref_month = all_months[t_idx - horizon]
        persistence.append(lookup.get((iso3, ref_month), np.nan))

        # Moving average: mean of y[t-horizon-ma_window+1 : t-horizon+1]
        ma_vals = []
        for k in range(ma_window):
            idx_k = t_idx - horizon - k
            if idx_k < 0:
                break
            m_k = all_months[idx_k]
            v = lookup.get((iso3, m_k))
            if v is not None and not np.isnan(v):
                ma_vals.append(v)
        moving_avg.append(np.mean(ma_vals) if ma_vals else np.nan)

    return {
        "Persistence": np.array(persistence),
        "MovingAvg-6m": np.array(moving_avg),
    }


def build_xgboost_baseline(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    countries_test: np.ndarray,
    dates_test: np.ndarray,
    horizon: int,
    train_end: str = "2024-03",
) -> np.ndarray:
    """
    Train an XGBoost model on features[t-horizon] -> y[t] using the training
    split, then predict on the test observations.

    Uses a single feature vector per observation (no temporal window),
    making it a fair point-in-time ML baseline vs the TFT's 24-month window.
    Target is raw fatalities (expm1 of stored log1p values).
    """
    all_months = sorted(df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months)}

    # Build training set: for each (iso3, t) where t <= train_end, use features[t-horizon]
    X_train, y_train = [], []
    for iso3 in df["country_iso3"].unique():
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month").reset_index(drop=True)
        feat = cdf[features].values.astype(np.float32)
        tgts = np.expm1(cdf[target_col].values.astype(np.float32))
        yms  = cdf["year_month"].values
        for i, ym in enumerate(yms):
            if ym > train_end:
                continue
            t_idx = month_to_idx.get(ym)
            if t_idx is None or t_idx - horizon < 0:
                continue
            src_ym = all_months[t_idx - horizon]
            src_rows = cdf[cdf["year_month"] == src_ym]
            if len(src_rows) == 0:
                continue
            src_idx = src_rows.index[0] - cdf.index[0]
            if src_idx < 0 or src_idx >= len(feat):
                continue
            if np.isnan(tgts[i]):
                continue
            X_train.append(feat[src_idx])
            y_train.append(tgts[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train XGBoost in log1p space for stability
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, np.log1p(y_train))

    # Build test feature matrix: for each test obs, look up features[t-horizon]
    X_test = []
    for iso3, ym in zip(countries_test, dates_test):
        t_idx = month_to_idx.get(ym)
        if t_idx is None or t_idx - horizon < 0:
            X_test.append(np.full(len(features), np.nan))
            continue
        src_ym = all_months[t_idx - horizon]
        src_rows = df[(df["country_iso3"] == iso3) & (df["year_month"] == src_ym)]
        if len(src_rows) == 0:
            X_test.append(np.full(len(features), np.nan))
        else:
            X_test.append(src_rows[features].values[0].astype(np.float32))

    X_test = np.array(X_test)
    valid = ~np.isnan(X_test).any(axis=1)
    preds = np.full(len(X_test), np.nan)
    if valid.sum() > 0:
        preds[valid] = np.expm1(model.predict(X_test[valid]))

    return preds, model


def generate_predictions(model, x_list, device, n_samples=1000, batch_size=128):
    """Generate probabilistic samples from model in batches."""
    all_samples = []
    for start in tqdm(range(0, len(x_list), batch_size), desc="Predicting"):
        end = min(start + batch_size, len(x_list))
        batch_x = torch.tensor(np.stack(x_list[start:end])).to(device)
        with torch.no_grad():
            pred = model.predict(batch_x, n_samples=n_samples)
            all_samples.append(pred["samples"].cpu().numpy())
    return np.concatenate(all_samples, axis=0)


def print_baseline_table(
    y_true: np.ndarray,
    baselines: dict[str, np.ndarray],
    tft_samples: np.ndarray,
    horizon: int,
):
    """Print MAE/RMSE comparison: TFT vs baselines."""
    from src.evaluation.metrics import mae, rmse
    print(f"\n  Baseline comparison — horizon {horizon}m")
    print(f"  {'Model':<20s} {'MAE':>8s} {'RMSE':>8s}")
    print("  " + "-" * 38)
    print(f"  {'TFT (ours)':<20s} {mae(y_true, tft_samples):8.1f} {rmse(y_true, tft_samples):8.1f}")
    for name, y_pred in baselines.items():
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            continue
        m = mae_point(y_true[mask], y_pred[mask])
        r = rmse_point(y_true[mask], y_pred[mask])
        print(f"  {name:<20s} {m:8.1f} {r:8.1f}")
    print()


def evaluate_horizon(
    model,
    df: pd.DataFrame,
    features: list[str],
    device,
    horizon: int,
    n_samples: int,
    batch_size: int,
    window_size: int,
    views_scores_path: str,
):
    """Run full evaluation for a single forecast horizon."""
    print(f"\n{'='*72}")
    print(f"  HORIZON: {horizon}-MONTH AHEAD")
    print(f"{'='*72}")

    x_all, y_all, dates_all, countries_all = build_test_samples(
        df, features, window_size=window_size, horizon=horizon,
    )

    usable, empty = find_usable_months(y_all, dates_all)
    usable_mask = np.isin(dates_all, usable)
    x_usable = [x_all[i] for i in range(len(x_all)) if usable_mask[i]]
    y_usable = y_all[usable_mask]
    dates_usable = dates_all[usable_mask]
    countries_usable = countries_all[usable_mask]

    if len(x_usable) == 0:
        print("  No usable observations — skipping.")
        return None

    print(f"  Evaluated on: {', '.join(usable)}  ({len(usable)}/{len(usable)+len(empty)} months)")
    if empty:
        print(f"  Excluded (no UCDP data): {', '.join(empty)}")

    # Model predictions
    samples = generate_predictions(model, x_usable, device, n_samples, batch_size)

    # Baselines
    baselines = build_baselines(df, "ucdp_fatalities_best", countries_usable, dates_usable, horizon)
    print(f"  Training XGBoost baseline (h={horizon})…")
    xgb_preds, _ = build_xgboost_baseline(
        df, features, "ucdp_fatalities_best", countries_usable, dates_usable, horizon,
    )
    baselines["XGBoost"] = xgb_preds

    # Per-month CRPS vs ViEWS (only for horizon=1 where ViEWS is directly comparable)
    if horizon == 1:
        print(f"\n  Per-month CRPS (ours vs top ViEWS entries):")
        per_month_comparison(
            y_usable, samples, dates_usable,
            model_name="Ours",
            views_csv_path=views_scores_path,
            usable_months=usable,
        )

    # Full comparison table
    comparison = compare_models(
        y_usable,
        {f"TFT_simplified (h={horizon})": samples},
        dates=dates_usable,
        views_csv_path=views_scores_path if horizon == 1 else "SKIP",
        usable_months=usable,
    )
    print_comparison(comparison)

    # Baselines
    print_baseline_table(y_usable, baselines, samples, horizon)

    # Diagnostics
    print(f"\n  DIAGNOSTIC (horizon={horizon}m)")
    print_diagnostics(y_usable, samples)

    return comparison


def evaluate(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_path: str = "data/processed/merge/model_ready.csv",
    views_scores_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    n_samples: int = 1000,
    batch_size: int = 128,
    hidden_dim: int = 64,
    n_lstm_layers: int = 1,
    n_attention_heads: int = 2,
    n_mixture_components: int = 3,
    window_size: int = 24,
    horizons: list[int] = [1, 3, 6],
):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    df = pd.read_csv(data_path)
    features = get_feature_list(df)
    print(f"Loaded {df.shape}, {len(features)} features")

    model = ConflictForecaster(
        n_features=len(features),
        hidden_dim=hidden_dim,
        n_lstm_layers=n_lstm_layers,
        n_attention_heads=n_attention_heads,
        n_mixture_components=n_mixture_components,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.eval()

    print(f"\nViEWS window: {VIEWS_WINDOW_START} to {VIEWS_WINDOW_END}")
    print(f"Horizons: {horizons}-month ahead")

    results = {}
    for h in horizons:
        results[h] = evaluate_horizon(
            model, df, features, device,
            horizon=h,
            n_samples=n_samples,
            batch_size=batch_size,
            window_size=window_size,
            views_scores_path=views_scores_path,
        )

    return results


if __name__ == "__main__":
    evaluate()
