"""
Evaluate the trained TFT model against ViEWS benchmarks.

ViEWS competition window: July 2024 - June 2025.
Only evaluates on months with available UCDP ground truth (7 of 12).
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.tft.model import ConflictForecaster
from src.models.tft.train import get_feature_list
from src.evaluation.baselines import VIEWS_WINDOW_START, VIEWS_WINDOW_END
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
):
    """Build test windows where the TARGET month is in [test_start, test_end]."""
    x_all, y_all, dates, countries = [], [], [], []

    for iso3 in df["country_iso3"].unique():
        country_df = df[df["country_iso3"] == iso3].sort_values("year_month")
        if len(country_df) <= window_size:
            continue

        feature_data = country_df[features].values.astype(np.float32)
        targets = np.expm1(country_df[target_col].values.astype(np.float32))
        year_months = country_df["year_month"].values

        for i in range(window_size, len(country_df)):
            ym = year_months[i]
            y = targets[i]
            if ym < test_start:
                continue
            if test_end and ym > test_end:
                continue
            if np.isnan(y):
                continue
            x_all.append(feature_data[i - window_size:i])
            y_all.append(y)
            dates.append(ym)
            countries.append(iso3)

    return x_all, np.array(y_all), np.array(dates), np.array(countries)


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
):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Load data + model
    df = pd.read_csv(data_path)
    features = get_feature_list(df)

    model = ConflictForecaster(
        n_features=len(features),
        hidden_dim=hidden_dim,
        n_lstm_layers=n_lstm_layers,
        n_attention_heads=n_attention_heads,
        n_mixture_components=n_mixture_components,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.eval()

    # Build full ViEWS window, then filter to usable months
    x_all, y_all, dates_all, countries_all = build_test_samples(
        df, features, window_size=window_size,
    )

    usable, empty = find_usable_months(y_all, dates_all)
    usable_mask = np.isin(dates_all, usable)
    x_usable = [x_all[i] for i in range(len(x_all)) if usable_mask[i]]
    y_usable = y_all[usable_mask]
    dates_usable = dates_all[usable_mask]

    # Generate predictions
    samples = generate_predictions(model, x_usable, device, n_samples, batch_size)

    # Print header
    print()
    print("=" * 72)
    print(f"  ViEWS BENCHMARK COMPARISON")
    print(f"  Window: {VIEWS_WINDOW_START} to {VIEWS_WINDOW_END}")
    print(f"  Evaluated on: {', '.join(usable)}  ({len(usable)}/{len(usable)+len(empty)} months)")
    if empty:
        print(f"  Excluded (no UCDP data): {', '.join(empty)}")
    print("=" * 72)

    # Per-month breakdown
    print(f"\n  Per-month CRPS (ours vs top ViEWS entries):")
    per_month_comparison(
        y_usable, samples, dates_usable,
        model_name="Ours",
        views_csv_path=views_scores_path,
        usable_months=usable,
    )

    # Full comparison table (our metrics + ViEWS benchmarks)
    comparison = compare_models(
        y_usable,
        {">> TFT_simplified (ours)": samples},
        dates=dates_usable,
        views_csv_path=views_scores_path,
        usable_months=usable,
    )
    print_comparison(comparison)

    # Diagnostics
    print(f"\n{'='*72}")
    print(f"  DIAGNOSTIC")
    print(f"{'='*72}")
    print_diagnostics(y_usable, samples)

    return comparison


if __name__ == "__main__":
    evaluate()