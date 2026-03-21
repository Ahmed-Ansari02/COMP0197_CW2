"""Plot training loss curves for all runs side by side."""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent

RUNS = {
    "A — Hurdle":    "a_hurdle_training_log.csv",
    "A+B — Hurdle":  "ab_hurdle_training_log.csv",
    "A+B+C — Hurdle": "abc_hurdle_training_log.csv",
    "A — Baseline (MSE)": "a_baseline_training_log.csv",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, fname in RUNS.items():
    df = pd.read_csv(RESULTS / fname)
    axes[0].plot(df.epoch, df.train_loss, label=label)
    axes[1].plot(df.epoch, df.val_loss, label=label, marker="o", markersize=3)

axes[0].set_title("Training Loss", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation Loss", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle("Training Curves — Feature Ablation", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "plot_training_curves.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'plot_training_curves.png'}")
plt.close()
