"""
Unified evaluation runner.

Runs all models through the same evaluation pipeline and
produces comparison tables against ViEWS benchmarks.

Benchmarks are loaded from cm_monthly_scores_full_Jul-Jun.csv
and filtered to the months where we have UCDP ground truth.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import full_evaluation, crps_sample
from src.evaluation.baselines import (
    get_views_benchmarks,
    get_views_monthly,
    TOP_REFS,
    VIEWS_WINDOW_START,
    VIEWS_WINDOW_END,
)


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


def find_usable_months(y_true: np.ndarray, dates: np.ndarray):
    """Months where at least some countries have non-zero fatalities."""
    usable, empty = [], []
    for m in sorted(set(dates)):
        mask = dates == m
        if (y_true[mask] > 0).sum() > 0:
            usable.append(m)
        else:
            empty.append(m)
    return usable, empty


def compare_models(
    y_true: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    dates: np.ndarray | None = None,
    views_csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    usable_months: list[str] | None = None,
    spike_threshold: float = 500,
) -> pd.DataFrame:
    """
    Compare multiple models on all metrics against ViEWS benchmarks.

    Args:
        y_true: [N] observed fatalities
        model_predictions: {model_name: [N, S] samples}
        dates: [N] year-month strings (for per-month breakdown)
        views_csv_path: path to ViEWS competition scores CSV
        usable_months: months to filter benchmarks to
        spike_threshold: fatality threshold for spike detection

    Returns:
        DataFrame with one row per model, columns = metrics, sorted by CRPS
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

    # Add ViEWS benchmark rows from CSV
    benchmarks = get_views_benchmarks(views_csv_path, usable_months)
    for bm_name, bm_scores in benchmarks.items():
        rows.append({
            "Model": f"[views] {bm_name}",
            "CRPS": bm_scores["crps"],
            "IGN": bm_scores["ign"],
            "MIS_90": bm_scores["mis"],
            "MIS_50": np.nan,
            "Spike_Recall": np.nan,
            "Spike_Precision": np.nan,
            "Spike_F1": np.nan,
            "Calib_chi2_p": np.nan,
            "N": np.nan,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("CRPS").reset_index(drop=True)


def per_month_comparison(
    y_true: np.ndarray,
    samples: np.ndarray,
    dates: np.ndarray,
    model_name: str,
    views_csv_path: str = "cm_monthly_scores_full_Jul-Jun.csv",
    usable_months: list[str] | None = None,
) -> None:
    """Print per-month CRPS: our model vs top ViEWS entries."""
    views_df = get_views_monthly(views_csv_path, usable_months)
    benchmarks = get_views_benchmarks(views_csv_path, usable_months)
    months = sorted(set(dates)) if usable_months is None else usable_months

    # Filter refs to those present in data
    refs = [r for r in TOP_REFS if r in benchmarks]

    header = f"  {'Month':<10s} {model_name:>8s}"
    for ref in refs:
        header += f" {ref[:12]:>12s}"
    print(header)
    print("  " + "-" * (10 + 8 + 12 * len(refs) + len(refs)))

    our_monthly = {}
    for m in months:
        mask = dates == m
        if mask.sum() == 0:
            continue
        our_monthly[m] = float(crps_sample(y_true[mask], samples[mask]).mean())
        line = f"  {m:<10s} {our_monthly[m]:8.2f}"
        for ref in refs:
            ref_df = views_df[(views_df["Model"] == ref) & (views_df["year_month"] == m)]
            if len(ref_df) > 0:
                line += f" {ref_df['CRPS'].values[0]:12.2f}"
            else:
                line += f" {'--':>12s}"
        print(line)

    print("  " + "-" * (10 + 8 + 12 * len(refs) + len(refs)))
    our_avg = np.mean(list(our_monthly.values()))
    line = f"  {'AVG':<10s} {our_avg:8.2f}"
    for ref in refs:
        line += f" {benchmarks[ref]['crps']:12.2f}"
    print(line)

    return our_monthly


def print_comparison(comparison_df: pd.DataFrame) -> None:
    """Pretty-print model comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON (sorted by CRPS, lower = better)")
    print("=" * 90)

    fmt_df = comparison_df.copy()
    for col in ["CRPS", "IGN", "MIS_90", "MIS_50"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "--")
    for col in ["Spike_Recall", "Spike_Precision", "Spike_F1", "Calib_chi2_p"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "--")

    print(fmt_df.to_string(index=False))
    print("=" * 90)


def print_diagnostics(y_true: np.ndarray, samples: np.ndarray) -> None:
    """Print model diagnostic breakdown."""
    crps_all = crps_sample(y_true, samples)
    zero_mask = y_true == 0
    pos_mask = y_true > 0
    spike_mask = y_true > 500

    pred_means = samples.mean(axis=1)
    pred_pzero = (samples == 0).mean(axis=1)

    print(f"""
  Observations:     {len(y_true)} ({(~zero_mask).sum()} non-zero, {spike_mask.sum()} spikes)

  CRPS by segment:
    y = 0:          {crps_all[zero_mask].mean():8.2f}  (n={zero_mask.sum()})
    0 < y <= 500:   {crps_all[pos_mask & ~spike_mask].mean():8.2f}  (n={(pos_mask & ~spike_mask).sum()})
    y > 500:        {crps_all[spike_mask].mean():8.2f}  (n={spike_mask.sum()})

  Model calibration:
    Avg P(zero):    {pred_pzero.mean():.3f}   (actual zero rate: {zero_mask.mean():.3f})
    Mean pred:      {pred_means.mean():.1f}   (actual mean: {y_true.mean():.1f})
    Median pred:    {np.median(pred_means):.1f}   (actual median: {np.median(y_true):.1f})
""")
