"""Plot per-country time series: true vs predicted fatalities for top conflict countries.

Shows the model's month-by-month predictions alongside actual fatality counts,
with uncertainty bands from the Student-t distribution and MC Dropout.
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

# Select top 8 countries by total fatalities in the test set
top_countries = (
    df.groupby("country_iso3")["y_true"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for ax, country in zip(axes, top_countries):
    sub = df[df.country_iso3 == country].sort_values("year_month").copy()
    sub["y_log"] = np.log1p(sub.y_true)

    # Predicted expected fatality (hurdle * severity)
    sub["pred_log"] = sub.p_conflict_mean * sub.mu_mean

    # Uncertainty band: mu +/- 1.5 * sigma (in log space), scaled by P(conflict)
    sub["upper"] = sub.p_conflict_mean * (sub.mu_mean + 1.5 * sub.sigma_mean)
    sub["lower"] = sub.p_conflict_mean * np.maximum(sub.mu_mean - 1.5 * sub.sigma_mean, 0)

    x = range(len(sub))
    ax.fill_between(x, sub.lower, sub.upper, alpha=0.2, color="steelblue",
                    label="±1.5σ band")
    ax.plot(x, sub.y_log, "k-", linewidth=1.2, alpha=0.8, label="True log1p(y)")
    ax.plot(x, sub.pred_log, "b-", linewidth=1.0, alpha=0.7, label="Predicted")

    # Mark spikes
    spikes = sub[sub.y_true > sub.y_true.quantile(0.95)]
    if len(spikes) > 0:
        spike_x = [list(sub.index).index(i) for i in spikes.index]
        ax.scatter(spike_x, np.log1p(spikes.y_true), c="red", s=30, zorder=5,
                   label="Spike (>95th pctl)")

    tick_step = max(1, len(sub) // 6)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(sub.year_month.values[::tick_step], rotation=30, fontsize=7)
    ax.set_title(f"{country}  (total fatalities: {sub.y_true.sum():,.0f})", fontsize=11)
    ax.set_ylabel("log1p(fatalities)")
    ax.grid(alpha=0.3)

    if ax == axes[0]:
        ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Country Time Series — True vs Predicted (A Hurdle, test set 2022–2025)",
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(RESULTS / "plot_country_timeseries.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_country_timeseries.png'}")
plt.close()
