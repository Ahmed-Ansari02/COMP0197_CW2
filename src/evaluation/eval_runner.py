"""
Unified evaluation runner.

Runs all models through the same evaluation pipeline and
produces comparison tables + ViEWS benchmark context.
"""

import numpy as np
import pandas as pd
from typing import Callable

from src.evaluation.metrics import full_evaluation, crps_sample
from src.evaluation.baselines import VIEWS_BENCHMARKS, VIEWS_COMPETITION_ENTRIES, print_views_benchmarks


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    samples: np.ndarray,
    spike_threshold: float = 500,
) -> dict:
    """Evaluate a single model and return results dict."""
    results = full_evaluation(y_true, samples, spike_threshold)
    results["model"] = name
    return results


def compare_models(
    y_true: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    spike_threshold: float = 500,
) -> pd.DataFrame:
    """
    Compare multiple models on all metrics.

    Args:
        y_true: [N] observed fatalities
        model_predictions: {model_name: [N, S] samples}

    Returns:
        DataFrame with one row per model, columns = metrics
    """
    rows = []

    for name, samples in model_predictions.items():
        result = evaluate_model(name, y_true, samples, spike_threshold)
        rows.append({
            "Model": name,
            "CRPS": result["crps"],
            "IGN": result["ign"],
            "MIS_90": result["mis_90"],
            "MIS_50": result["mis_50"],
            "Spike_Recall": result["spikes"]["recall"],
            "Spike_Precision": result["spikes"]["precision"],
            "Spike_F1": result["spikes"]["f1"],
            "Calib_chi2_p": result["calibration"]["chi2_pval"],
            "N": result["n_observations"],
        })

    # Add ViEWS reference rows (benchmarks + competition entries)
    for source, label in [(VIEWS_BENCHMARKS, "bm"), (VIEWS_COMPETITION_ENTRIES, "entry")]:
        for bm_name, bm_info in source.items():
            rows.append({
                "Model": f"[{label}] {bm_name}",
                "CRPS": bm_info["crps"],
                "IGN": bm_info.get("ab_log", np.nan),
                "MIS_90": bm_info.get("mis", np.nan),
                "MIS_50": np.nan,
                "Spike_Recall": np.nan,
                "Spike_Precision": np.nan,
                "Spike_F1": np.nan,
                "Calib_chi2_p": np.nan,
                "N": np.nan,
            })

    df = pd.DataFrame(rows)
    return df.sort_values("CRPS").reset_index(drop=True)


def evaluate_by_period(
    y_true: np.ndarray,
    samples: np.ndarray,
    dates: np.ndarray,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Breakdown evaluation by year.

    Args:
        y_true: [N] observed fatalities
        samples: [N, S] samples
        dates: [N] year-month strings (e.g., "2020-01")

    Returns:
        DataFrame with CRPS per year
    """
    years = np.array([d[:4] for d in dates])
    unique_years = sorted(set(years))

    rows = []
    for year in unique_years:
        mask = years == year
        if mask.sum() == 0:
            continue
        crps_vals = crps_sample(y_true[mask], samples[mask])
        rows.append({
            "Year": year,
            "Model": model_name,
            "CRPS": float(crps_vals.mean()),
            "N": int(mask.sum()),
            "N_nonzero": int((y_true[mask] > 0).sum()),
        })

    return pd.DataFrame(rows)


def print_comparison(comparison_df: pd.DataFrame) -> None:
    """Pretty-print model comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON (sorted by CRPS, lower = better)")
    print("=" * 90)

    # Format the table
    fmt_df = comparison_df.copy()
    for col in ["CRPS", "IGN", "MIS_90", "MIS_50"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "—")
    for col in ["Spike_Recall", "Spike_Precision", "Spike_F1", "Calib_chi2_p"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "—")

    print(fmt_df.to_string(index=False))
    print("=" * 90)
    print("Ref rows are ViEWS competition benchmarks (CRPS only — other metrics not published)")
    print()
