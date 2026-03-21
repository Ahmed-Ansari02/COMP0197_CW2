"""Evaluate and plot the conflict classification head (P(fatalities > 0)).

Produces:
  - ROC curves for all hurdle runs
  - Precision-Recall curves for all hurdle runs
  - Confusion matrix for the best run
  - Summary metrics table printed to stdout
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

RESULTS = Path(__file__).parent

RUNS = {
    "A — Conflict":     "a_hurdle_test_predictions.csv",
    "A+B — +Structural": "ab_hurdle_test_predictions.csv",
    "A+B+C — +Volatility": "abc_hurdle_test_predictions.csv",
}

# ── ROC and PR curves ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

print("=" * 65)
print(f"{'Run':30s}  {'ROC-AUC':>8s}  {'PR-AUC':>8s}  {'N':>6s}")
print("-" * 65)

for label, fname in RUNS.items():
    df = pd.read_csv(RESULTS / fname)
    y_bin = (df.y_true > 0).astype(int).values
    p = df.p_conflict_mean.values

    roc_auc = roc_auc_score(y_bin, p)
    pr_auc = average_precision_score(y_bin, p)
    print(f"{label:30s}  {roc_auc:8.4f}  {pr_auc:8.4f}  {len(df):6d}")

    fpr, tpr, _ = roc_curve(y_bin, p)
    axes[0].plot(fpr, tpr, label=f"{label}  (AUC={roc_auc:.3f})")

    prec, rec, _ = precision_recall_curve(y_bin, p)
    axes[1].plot(rec, prec, label=f"{label}  (AP={pr_auc:.3f})")

print("=" * 65)

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Conflict Detection")
axes[0].legend(fontsize=8, loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Conflict Detection")
axes[1].legend(fontsize=8, loc="lower left")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "plot_conflict_roc_pr.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {RESULTS / 'plot_conflict_roc_pr.png'}")
plt.close()

# ── Confusion matrix for best run (A) ───────────────────────────────────────

df = pd.read_csv(RESULTS / "a_hurdle_test_predictions.csv")
y_bin = (df.y_true > 0).astype(int)
y_pred_bin = (df.p_conflict_mean > 0.5).astype(int)

fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_bin, y_pred_bin)
disp = ConfusionMatrixDisplay(cm, display_labels=["Peace", "Conflict"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix — A Hurdle (threshold=0.5)")
fig.tight_layout()
fig.savefig(RESULTS / "plot_confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_confusion_matrix.png'}")

print("\nClassification Report (A Hurdle, threshold=0.5):")
print(classification_report(y_bin, y_pred_bin, target_names=["Peace", "Conflict"]))
plt.close()
