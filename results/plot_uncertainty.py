"""Visualise uncertainty calibration and epistemic vs aleatoric uncertainty.

Produces:
  - Calibration plot: does higher uncertainty correlate with larger errors?
  - Epistemic uncertainty (MC Dropout std) distribution
  - Uncertainty vs prediction error scatter
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent

df = pd.read_csv(RESULTS / "a_hurdle_test_predictions.csv")

# Focus on non-zero samples where we have severity predictions
nz = df[df.y_true > 0].copy()
nz["y_log"] = np.log1p(nz.y_true)
nz["abs_error"] = (nz.y_log - nz.mu_mean).abs()

# ── 1. Uncertainty calibration: bin by sigma, check error ────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel A: Aleatoric uncertainty (sigma) vs error
nz["sigma_bin"] = pd.qcut(nz.sigma_mean, 5, duplicates="drop")
cal_sigma = nz.groupby("sigma_bin", observed=True).agg(
    avg_sigma=("sigma_mean", "mean"),
    avg_error=("abs_error", "mean"),
    count=("abs_error", "count"),
).reset_index()

axes[0].bar(range(len(cal_sigma)), cal_sigma.avg_error, color="#4C72B0", alpha=0.7,
            label="Avg |error|")
axes[0].plot(range(len(cal_sigma)), cal_sigma.avg_sigma, "ro-", label="Avg σ")
axes[0].set_xticks(range(len(cal_sigma)))
axes[0].set_xticklabels([f"{r.avg_sigma:.2f}" for _, r in cal_sigma.iterrows()],
                        fontsize=8)
axes[0].set_xlabel("Aleatoric uncertainty (σ) quintile")
axes[0].set_ylabel("Value")
axes[0].set_title("Aleatoric Calibration:\nhigher σ → larger error?")
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)

corr_sigma = nz.sigma_mean.corr(nz.abs_error)
axes[0].text(0.95, 0.95, f"ρ = {corr_sigma:.3f}", transform=axes[0].transAxes,
             fontsize=10, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

# Panel B: Epistemic uncertainty (mu_std from MC Dropout) vs error
nz["epistemic_bin"] = pd.qcut(nz.mu_std, 5, duplicates="drop")
cal_epi = nz.groupby("epistemic_bin", observed=True).agg(
    avg_mu_std=("mu_std", "mean"),
    avg_error=("abs_error", "mean"),
    count=("abs_error", "count"),
).reset_index()

axes[1].bar(range(len(cal_epi)), cal_epi.avg_error, color="#55A868", alpha=0.7,
            label="Avg |error|")
axes[1].plot(range(len(cal_epi)), cal_epi.avg_mu_std, "ro-", label="Avg μ_std")
axes[1].set_xticks(range(len(cal_epi)))
axes[1].set_xticklabels([f"{r.avg_mu_std:.3f}" for _, r in cal_epi.iterrows()],
                        fontsize=8)
axes[1].set_xlabel("Epistemic uncertainty (μ_std) quintile")
axes[1].set_ylabel("Value")
axes[1].set_title("Epistemic Calibration:\nhigher μ_std → larger error?")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.3)

corr_epi = nz.mu_std.corr(nz.abs_error)
axes[1].text(0.95, 0.95, f"ρ = {corr_epi:.3f}", transform=axes[1].transAxes,
             fontsize=10, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

# Panel C: Scatter of total uncertainty vs error
nz["total_unc"] = nz.sigma_mean + nz.mu_std
axes[2].scatter(nz.total_unc, nz.abs_error, alpha=0.2, s=8, c="steelblue")
axes[2].set_xlabel("Total uncertainty (σ + μ_std)")
axes[2].set_ylabel("|Prediction error| (log1p space)")
axes[2].set_title("Uncertainty vs Error")
axes[2].grid(alpha=0.3)

corr_total = nz.total_unc.corr(nz.abs_error)
axes[2].text(0.95, 0.95, f"ρ = {corr_total:.3f}", transform=axes[2].transAxes,
             fontsize=10, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

fig.suptitle("Uncertainty Calibration — A Hurdle (test set, non-zero months)",
             fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "plot_uncertainty_calibration.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_uncertainty_calibration.png'}")
plt.close()

# ── 2. Distribution of epistemic uncertainty across all samples ──────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].hist(df.p_conflict_std, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
axes[0].set_xlabel("P(conflict) epistemic std")
axes[0].set_ylabel("Count")
axes[0].set_title("Epistemic Uncertainty on Conflict Probability")
axes[0].axvline(df.p_conflict_std.median(), color="red", linestyle="--",
                label=f"Median = {df.p_conflict_std.median():.4f}")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

axes[1].hist(nz.mu_std, bins=50, color="#55A868", alpha=0.7, edgecolor="white")
axes[1].set_xlabel("μ epistemic std (from MC Dropout)")
axes[1].set_ylabel("Count")
axes[1].set_title("Epistemic Uncertainty on Severity (non-zero months)")
axes[1].axvline(nz.mu_std.median(), color="red", linestyle="--",
                label=f"Median = {nz.mu_std.median():.4f}")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Distribution of Epistemic Uncertainty (MC Dropout)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "plot_uncertainty_distribution.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_uncertainty_distribution.png'}")
plt.close()
