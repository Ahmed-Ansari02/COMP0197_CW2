"""
ViEWS Prediction Competition benchmark scores.

Loads per-month scores from cm_monthly_scores_full_Jul-Jun.csv and filters
to the months where we have UCDP ground truth (currently Jul 2024 – Jan 2025).

Source: https://viewsforecasting.org/research/prediction-challenge-2023/
Paper: https://arxiv.org/abs/2407.11045
"""

import pandas as pd
import numpy as np

VIEWS_WINDOW_START = "2024-07"
VIEWS_WINDOW_END = "2025-06"

# Month ID → year_month mapping for the competition window
MONTH_ID_MAP = {
    535: "2024-07", 536: "2024-08", 537: "2024-09", 538: "2024-10",
    539: "2024-11", 540: "2024-12", 541: "2025-01", 542: "2025-02",
    543: "2025-03", 544: "2025-04", 545: "2025-05", 546: "2025-06",
}

# Reference models to highlight in comparisons
TOP_REFS = [
    "VIEWS_bm_ph_conflictology_country12",
    "ConflictForecast",
    "Randahl_Vegelius_markov_omm",
    "CCEW_tft",
]


def load_views_monthly_scores(csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv") -> pd.DataFrame:
    """Load raw per-month ViEWS competition scores."""
    df = pd.read_csv(csv_path)
    df["year_month"] = df["Month ID"].map(MONTH_ID_MAP)
    return df


def get_views_benchmarks(
    csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    usable_months: list[str] | None = None,
) -> dict[str, dict]:
    """
    Load ViEWS scores and average over usable months only.

    Returns:
        {model_name: {"crps": float, "ign": float, "mis": float}, ...}
    """
    df = load_views_monthly_scores(csv_path)

    if usable_months is not None:
        df = df[df["year_month"].isin(usable_months)]

    benchmarks = {}
    for model_name in df["Model"].unique():
        model_df = df[df["Model"] == model_name]
        if len(model_df) > 0:
            benchmarks[model_name] = {
                "crps": float(model_df["CRPS"].mean()),
                "ign": float(model_df["ab-Log Score"].mean()),
                "mis": float(model_df["MIS"].mean()),
            }

    return benchmarks


def get_views_monthly(
    csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    usable_months: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-month scores filtered to usable months."""
    df = load_views_monthly_scores(csv_path)
    if usable_months is not None:
        df = df[df["year_month"].isin(usable_months)]
    return df


def print_views_benchmarks(
    csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    usable_months: list[str] | None = None,
):
    """Print ViEWS leaderboard filtered to usable months."""
    benchmarks = get_views_benchmarks(csv_path, usable_months)
    n_months = len(usable_months) if usable_months else 12

    print(f"\nViEWS Leaderboard — Country-level, {n_months} usable months")
    print(f"{'Rank':>4s}  {'Model':42s} {'CRPS':>7s} {'ab-Log':>7s} {'MIS':>9s}")
    print("-" * 73)

    sorted_models = sorted(benchmarks.items(), key=lambda x: x[1]["crps"])
    for rank, (name, scores) in enumerate(sorted_models, 1):
        marker = " *" if name.startswith("VIEWS_bm") else ""
        print(f"  {rank:2d}  {name:42s} {scores['crps']:7.2f} {scores['ign']:7.2f} {scores['mis']:9.1f}{marker}")

    print("-" * 73)
    print("* = benchmark model")
