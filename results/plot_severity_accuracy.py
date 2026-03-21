"""Evaluate severity predictions: predicted vs actual fatalities.

Produces:
  - Scatter plot of predicted vs true log1p(fatalities) for non-zero months
  - Per-country time series for top conflict countries
  - Feature group comparison bar chart
  - Spike detection analysis
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

RUNS = {
    "A — Conflict":       "a_hurdle_test_predictions.csv",
    "A+B — +Structural":  "ab_hurdle_test_predictions.csv",
    "A+B+C — +Volatility": "abc_hurdle_test_predictions.csv",
}

# ── 1. Scatter: predicted vs true (log1p space, non-zero only) ──────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (label, fname) in zip(axes, RUNS.items()):
    df = pd.read_csv(RESULTS / fname)
    nz = df[df.y_true > 0].copy()
    nz["y_log"] = np.log1p(nz.y_true)

    ax.scatter(nz.y_log, nz.mu_mean, alpha=0.25, s=10, c="steelblue")
    lims = [0, max(nz.y_log.max(), nz.mu_mean.max()) + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.6, label="Perfect prediction")
    ax.set_xlabel("True log1p(fatalities)")
    ax.set_ylabel("Predicted μ (log1p space)")
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    mae = (nz.y_log - nz.mu_mean).abs().mean()
    corr = nz.y_log.corr(nz.mu_mean)
    ax.text(0.05, 0.92, f"MAE={mae:.2f}\nCorr={corr:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

fig.suptitle("Severity Prediction — Predicted vs True (non-zero months)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "plot_severity_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_severity_scatter.png'}")
plt.close()

# ── 2. Feature group comparison bar chart ───────────────────────────────────

metrics = []
for label, fname in RUNS.items():
    df = pd.read_csv(RESULTS / fname)
    nz = df[df.y_true > 0].copy()
    nz["y_log"] = np.log1p(nz.y_true)
    mae_log = (nz.y_log - nz.mu_mean).abs().mean()
    rmse_log = np.sqrt(((nz.y_log - nz.mu_mean) ** 2).mean())
    corr = nz.y_log.corr(nz.mu_mean)
    metrics.append({"Run": label, "MAE (log1p)": mae_log,
                    "RMSE (log1p)": rmse_log, "Correlation": corr})

# Add baseline
df_b = pd.read_csv(RESULTS / "a_baseline_test_predictions.csv")
nz_b = df_b[df_b.y_true > 0].copy()
nz_b["y_log"] = np.log1p(nz_b.y_true)
mae_b = (nz_b.y_log - nz_b.y_pred_mean).abs().mean()
rmse_b = np.sqrt(((nz_b.y_log - nz_b.y_pred_mean) ** 2).mean())
corr_b = nz_b.y_log.corr(nz_b.y_pred_mean)
metrics.append({"Run": "A — Baseline (MSE)", "MAE (log1p)": mae_b,
                "RMSE (log1p)": rmse_b, "Correlation": corr_b})

mdf = pd.DataFrame(metrics)
print("\n" + mdf.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for ax, col in zip(axes, ["MAE (log1p)", "RMSE (log1p)", "Correlation"]):
    bars = ax.bar(range(len(mdf)), mdf[col], color=colors)
    ax.set_xticks(range(len(mdf)))
    ax.set_xticklabels(mdf.Run, rotation=25, ha="right", fontsize=8)
    ax.set_title(col, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mdf[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Feature Group Comparison — Severity Metrics (non-zero months)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "plot_feature_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {RESULTS / 'plot_feature_comparison.png'}")
plt.close()

# ── 3. Spike detection analysis ─────────────────────────────────────────────

df = pd.read_csv(RESULTS / "a_hurdle_test_predictions.csv")
thresholds = [0, 10, 50, 100, 500, 1000, 5000]
rows = []
for i in range(len(thresholds) - 1):
    lo, hi = thresholds[i], thresholds[i + 1]
    mask = (df.y_true > lo) & (df.y_true <= hi)
    if mask.sum() == 0:
        continue
    sub = df[mask]
    detected = (sub.p_conflict_mean > 0.5).mean()
    avg_mu = sub.mu_mean.mean()
    avg_true_log = np.log1p(sub.y_true).mean()
    rows.append({"Range": f"{lo+1}–{hi}", "Count": mask.sum(),
                 "Detected %": detected * 100,
                 "Avg μ": avg_mu, "Avg log1p(true)": avg_true_log})

# Add >5000
mask_big = df.y_true > 5000
if mask_big.sum() > 0:
    sub = df[mask_big]
    rows.append({"Range": ">5000", "Count": mask_big.sum(),
                 "Detected %": (sub.p_conflict_mean > 0.5).mean() * 100,
                 "Avg μ": sub.mu_mean.mean(),
                 "Avg log1p(true)": np.log1p(sub.y_true).mean()})

spike_df = pd.DataFrame(rows)
print("\nSpike Detection Analysis (A Hurdle, test set):")
print(spike_df.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 5))
x = range(len(spike_df))
w = 0.35

bars1 = ax1.bar([i - w / 2 for i in x], spike_df["Detected %"],
                width=w, color="#4C72B0", label="Detection Rate (%)")
ax1.set_ylabel("Detection Rate (%)", color="#4C72B0")
ax1.set_ylim(0, 110)

ax2 = ax1.twinx()
bars2 = ax2.bar([i + w / 2 for i in x], spike_df["Avg μ"],
                width=w, color="#C44E52", alpha=0.7, label="Avg predicted μ")
ax2.plot(x, spike_df["Avg log1p(true)"], "ko-", markersize=6,
         label="Avg true log1p(y)")
ax2.set_ylabel("log1p(fatalities)")

ax1.set_xticks(x)
ax1.set_xticklabels(spike_df.Range, fontsize=9)
ax1.set_xlabel("Fatality Range")
ax1.set_title("Spike Detection: Detection Rate & Severity Accuracy by Fatality Range")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "plot_spike_detection.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {RESULTS / 'plot_spike_detection.png'}")
plt.close()
