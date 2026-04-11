"""
Evaluation metrics for probabilistic conflict forecasting.

Implements the ViEWS competition evaluation protocol:
- CRPS (primary metric): rewards calibration + sharpness
- IGN (Ignorance/Log Score): probability assigned to observed values
- MIS (Mean Interval Score): prediction interval width + coverage
- Calibration: PIT histogram uniformity
- Spike metrics: recall/precision for high-fatality events
"""

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────
# CRPS — Continuous Ranked Probability Score
# ──────────────────────────────────────────────────────────────

def crps_sample(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    CRPS from empirical samples (the ViEWS standard approach).

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        y_true: [N] observed values
        samples: [N, S] S samples per observation

    Returns:
        [N] CRPS values (lower is better)
    """
    N, S = samples.shape

    # E|X - y|: mean absolute error of samples vs observation
    abs_diff = np.abs(samples - y_true[:, None])
    term1 = abs_diff.mean(axis=1)

    # E|X - X'|: mean pairwise absolute difference between samples
    # Efficient computation using sorted samples
    sorted_samples = np.sort(samples, axis=1)
    # For sorted values, E|X-X'| = 2 * sum_i (2*i - S) * x_i / S^2
    weights = 2 * np.arange(S) - S
    term2 = (sorted_samples * weights[None, :]).sum(axis=1) / (S * S)

    return term1 - term2


def crps_mean(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Average CRPS across all observations."""
    return float(crps_sample(y_true, samples).mean())


# ──────────────────────────────────────────────────────────────
# IGN — Ignorance Score (Log Score)
# ──────────────────────────────────────────────────────────────

def ign_score(y_true: np.ndarray, samples: np.ndarray, bandwidth: str = "silverman") -> np.ndarray:
    """
    Ignorance score: -log(f(y)) where f is KDE of predictive distribution.

    Args:
        y_true: [N] observed values
        samples: [N, S] samples per observation
        bandwidth: KDE bandwidth method

    Returns:
        [N] IGN values (lower is better)
    """
    N = len(y_true)
    ign = np.zeros(N)

    for i in range(N):
        s = samples[i]
        if s.std() < 1e-10:
            # Degenerate distribution
            ign[i] = 0.0 if np.abs(y_true[i] - s.mean()) < 1e-6 else 50.0
            continue

        try:
            kde = stats.gaussian_kde(s, bw_method=bandwidth)
            pdf_val = kde(y_true[i])[0]
            ign[i] = -np.log(max(pdf_val, 1e-10))
        except Exception:
            ign[i] = 50.0  # penalty for failed KDE

    return ign


def ign_mean(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Average IGN across all observations."""
    return float(ign_score(y_true, samples).mean())


# ──────────────────────────────────────────────────────────────
# MIS — Mean Interval Score
# ──────────────────────────────────────────────────────────────

def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Interval score for a (1-alpha) prediction interval.

    IS = (upper - lower) + (2/alpha) * (lower - y) * I(y < lower)
                         + (2/alpha) * (y - upper) * I(y > upper)

    Args:
        y_true: [N] observed values
        lower: [N] lower bound of prediction interval
        upper: [N] upper bound of prediction interval
        alpha: significance level (default 0.1 for 90% interval)

    Returns:
        [N] interval scores (lower is better)
    """
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)

    return width + (2 / alpha) * below + (2 / alpha) * above


def mis_from_samples(
    y_true: np.ndarray,
    samples: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """MIS computed from samples using quantile-derived intervals."""
    lower = np.quantile(samples, alpha / 2, axis=1)
    upper = np.quantile(samples, 1 - alpha / 2, axis=1)
    return float(interval_score(y_true, lower, upper, alpha).mean())


# ──────────────────────────────────────────────────────────────
# Calibration — PIT Histogram
# ──────────────────────────────────────────────────────────────

def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform values.

    PIT(y) = proportion of samples <= y.
    For a calibrated model, PIT values should be Uniform(0,1).

    Returns:
        [N] PIT values in [0, 1]
    """
    return (samples <= y_true[:, None]).mean(axis=1)


def pit_reliability(y_true: np.ndarray, samples: np.ndarray, n_bins: int = 10) -> dict:
    """
    PIT histogram for calibration assessment.

    Returns:
        dict with bin_edges, observed_freq, expected_freq, chi2_stat, chi2_pval
    """
    pits = pit_values(y_true, samples)
    observed, bin_edges = np.histogram(pits, bins=n_bins, range=(0, 1))
    observed_freq = observed / len(pits)
    expected_freq = 1.0 / n_bins

    # Chi-squared test for uniformity
    chi2_stat, chi2_pval = stats.chisquare(observed, f_exp=np.full(n_bins, len(pits) / n_bins))

    return {
        "bin_edges": bin_edges,
        "observed_freq": observed_freq,
        "expected_freq": expected_freq,
        "chi2_stat": float(chi2_stat),
        "chi2_pval": float(chi2_pval),
    }


# ──────────────────────────────────────────────────────────────
# Point Prediction Accuracy
# ──────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, samples: np.ndarray) -> float:
    """MAE using sample mean as point estimate."""
    y_pred = samples.mean(axis=1)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, samples: np.ndarray) -> float:
    """RMSE using sample mean as point estimate."""
    y_pred = samples.mean(axis=1)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae_point(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE for point-estimate baselines."""
    return float(np.abs(y_true - y_pred).mean())


def rmse_point(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE for point-estimate baselines."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


# ──────────────────────────────────────────────────────────────
# Spike Metrics — High-fatality event detection
# ──────────────────────────────────────────────────────────────

def _auroc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC computed via trapezoidal integration (no sklearn dependency)."""
    n_pos = y_true_binary.sum()
    n_neg = len(y_true_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    desc = np.argsort(y_score)[::-1]
    y_sorted = y_true_binary[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return float(np.trapezoid(tpr, fpr))


def spike_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
    threshold: float = 500,
    alert_quantile: float = 0.9,
) -> dict:
    """
    Evaluate how well the model detects high-fatality spikes.

    A spike is y_true > threshold.
    An alert is raised when the model's alert_quantile exceeds threshold.
    AUROC uses P(sample > threshold) as the continuous score.

    Returns:
        dict with recall, precision, f1, auroc, n_spikes, n_alerts
    """
    is_spike = y_true > threshold
    predicted_upper = np.quantile(samples, alert_quantile, axis=1)
    is_alert = predicted_upper > threshold

    n_spikes = int(is_spike.sum())
    n_alerts = int(is_alert.sum())

    if n_spikes == 0:
        recall = float("nan")
    else:
        recall = float((is_spike & is_alert).sum() / n_spikes)

    if n_alerts == 0:
        precision = float("nan")
    else:
        precision = float((is_spike & is_alert).sum() / n_alerts)

    if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * recall * precision / (recall + precision)

    # AUROC: use fraction of samples exceeding threshold as continuous score
    y_score = (samples > threshold).mean(axis=1)
    auroc = _auroc(is_spike.astype(float), y_score)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "n_spikes": n_spikes,
        "n_alerts": n_alerts,
    }


# ──────────────────────────────────────────────────────────────
# Combined Report
# ──────────────────────────────────────────────────────────────

def full_evaluation(
    y_true: np.ndarray,
    samples: np.ndarray,
    spike_threshold: float = 500,
) -> dict:
    """
    Run all metrics and return a combined report.

    Args:
        y_true: [N] observed fatalities
        samples: [N, S] sampled predictions

    Returns:
        dict with all metrics
    """
    return {
        "mae": mae(y_true, samples),
        "rmse": rmse(y_true, samples),
        "crps": crps_mean(y_true, samples),
        "ign": ign_mean(y_true, samples),
        "mis_90": mis_from_samples(y_true, samples, alpha=0.1),
        "mis_50": mis_from_samples(y_true, samples, alpha=0.5),
        "calibration": pit_reliability(y_true, samples),
        "spikes": spike_metrics(y_true, samples, threshold=spike_threshold),
        "n_observations": len(y_true),
        "mean_predicted": float(samples.mean()),
        "mean_observed": float(y_true.mean()),
    }
