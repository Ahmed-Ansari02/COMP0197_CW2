"""
Member A — Exploratory Data Analysis
Run after generate_conflict_dataset.py and filter_features.py.
Produces:
  - analysis/member_a/correlation_matrix.png
  - analysis/member_a/temporal/   — global time series plots per feature group
  - analysis/member_a/profiles/distribution_profiles.csv — summary stats

GenAI usage: Claude assisted with plot structure and layout.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "member_a")
REPORT_DIR = os.path.join(BASE_DIR, "analysis", "member_a")

os.makedirs(os.path.join(REPORT_DIR, "temporal"), exist_ok=True)
os.makedirs(os.path.join(REPORT_DIR, "cross_source"), exist_ok=True)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "member_a_model_features.csv"))
feature_cols = [c for c in df.columns if c not in ["country_iso3", "year_month"]]


# ── 1. Correlation matrix ──────────────────────────────────────────────────

print("Plotting correlation matrix...")
corr = df[feature_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.4, ax=ax, annot_kws={"size": 6}
)
ax.set_title("Spearman correlation — member A model features", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "correlation_matrix.png"), dpi=120)
plt.close(fig)
print("  saved correlation_matrix.png")


# ── 2. Distribution profiles summary CSV ──────────────────────────────────

print("Building distribution profiles summary...")
records = []
for col in feature_cols:
    s = df[col].dropna()
    records.append({
        "feature":      col,
        "n_nonmissing": len(s),
        "missing_pct":  round(df[col].isna().mean() * 100, 2),
        "pct_zero":     round((s == 0).mean() * 100, 2),
        "mean":         round(s.mean(), 4),
        "median":       round(s.median(), 4),
        "std":          round(s.std(), 4),
        "skewness":     round(s.skew(), 4),
        "p5":           round(s.quantile(0.05), 4),
        "p95":          round(s.quantile(0.95), 4),
        "max":          round(s.max(), 4),
    })
pd.DataFrame(records).to_csv(
    os.path.join(REPORT_DIR, "profiles", "distribution_profiles.csv"), index=False
)
print("  saved distribution_profiles.csv")


# ── 3. Temporal trends — global monthly aggregates ─────────────────────────

print("Plotting temporal trends...")

# Group features for cleaner plots
groups = {
    "fatalities": ["ucdp_fatalities_best", "ucdp_fatalities_high", "ucdp_civilian_deaths"],
    "event_counts": ["ucdp_event_count", "ucdp_state_based_events", "ucdp_non_state_events", "ucdp_one_sided_events"],
    "conflict_signal": ["gdelt_conflict_event_count", "gdelt_goldstein_mean"],
    "escalation": ["ucdp_fatalities_best_ld", "ucdp_event_count_ld", "gdelt_conflict_event_count_ld"],
}

for group_name, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    if not cols:
        continue

    global_ts = df.groupby("year_month")[cols].sum().reset_index()
    global_ts["year_month"] = pd.to_datetime(global_ts["year_month"])

    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(global_ts["year_month"], global_ts[col], linewidth=0.8, color="steelblue")
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Global monthly totals — {group_name}", fontsize=11)
    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "temporal", f"{group_name}_trend.png"), dpi=100)
    plt.close(fig)
    print(f"  saved temporal/{group_name}_trend.png")


# ── 4. Conflict has_conflict rate over time ────────────────────────────────

print("Plotting conflict prevalence over time...")
prev = df.groupby("year_month")["ucdp_has_conflict"].mean().reset_index()
prev["year_month"] = pd.to_datetime(prev["year_month"])

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(prev["year_month"], prev["ucdp_has_conflict"] * 100, linewidth=0.9, color="crimson")
ax.set_title("% of countries with active conflict per month (UCDP)")
ax.set_ylabel("% countries")
ax.set_xlabel("Year")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "temporal", "conflict_prevalence.png"), dpi=100)
plt.close(fig)
print("  saved temporal/conflict_prevalence.png")

print("\nDone.")
