import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from submission.train import ConvTransformer, get_feature_list
from src.evaluation.evaluate_tft import generate_predictions
from src.evaluation.metrics import (
    pit_values, crps_sample, spike_metrics, _auroc, mae, rmse,
    mae_point, rmse_point,
)
from src.evaluation.baselines import (
    load_views_monthly_scores, VIEWS_WINDOW_START, VIEWS_WINDOW_END, TOP_REFS,
)

CHECKPOINT   = "submission/best_model.pt"
DATA_PATH    = "submission/data/merge/model_ready.csv"
VIEWS_CSV    = "submission/cm_monthly_scores_full_Jul-Jun.csv"
OUT_DIR      = "analysis/evaluation"
N_SAMPLES    = 1000
BATCH_SIZE   = 128
WINDOW_SIZE  = 24
HORIZONS     = [1, 3, 6]
SPIKE_THRESH = 500

def build_test_samples(
    df: pd.DataFrame,
    features: list,
    target_col: str = "ucdp_fatalities_best",
    window_size: int = 24,
    horizon: int = 1,
    test_start: str = VIEWS_WINDOW_START,
    test_end: str = VIEWS_WINDOW_END,
):
    """
    Build sliding-window test samples matching submission/test.py exactly:
    - NaN targets treated as 0 (UCDP absence = no conflict)
    - Short sequences zero-padded so all countries are included
    - horizon=h: features from [i-window_size-(h-1) : i-(h-1)], target at i
    """
    x_all, y_all, dates, countries = [], [], [], []
    shift = horizon - 1

    for iso3 in df["country_iso3"].unique():
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month")

        feat_data = cdf[features].values.astype(np.float32)
        raw_targets = cdf[target_col].values.astype(np.float32)
        raw_targets = np.nan_to_num(raw_targets, nan=0.0)  # NaN → 0
        targets = np.expm1(raw_targets)
        yms = cdf["year_month"].values

        # Zero-pad short sequences so every country is included
        needed = window_size + shift
        if len(feat_data) < needed:
            pad = needed - len(feat_data)
            feat_data = np.concatenate([np.zeros((pad, feat_data.shape[1]), dtype=np.float32), feat_data])
            targets = np.concatenate([np.zeros(pad, dtype=np.float32), targets])
            yms = np.concatenate([np.full(pad, ""), yms])

        for i in range(needed, len(feat_data)):
            ym = yms[i]
            if ym < test_start:
                continue
            if test_end and ym > test_end:
                continue
            x_all.append(feat_data[i - window_size - shift: i - shift])
            y_all.append(targets[i])
            dates.append(ym)
            countries.append(iso3)

    return x_all, np.array(y_all), np.array(dates), np.array(countries)


def build_baselines(
    df: pd.DataFrame,
    target_col: str,
    countries: np.ndarray,
    dates: np.ndarray,
    horizon: int,
) -> dict:
    """Persistence and 6-month moving-average baselines."""
    persistence = np.full(len(countries), np.nan)
    movavg6m    = np.full(len(countries), np.nan)

    for idx, (iso3, ym) in enumerate(zip(countries, dates)):
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month")
        all_ym  = cdf["year_month"].values
        targets = np.expm1(cdf[target_col].values.astype(np.float32))
        pos = np.searchsorted(all_ym, ym)
        lag_pos = pos - horizon
        if lag_pos >= 0:
            persistence[idx] = targets[lag_pos]
        if lag_pos >= 6:
            movavg6m[idx] = targets[lag_pos - 5: lag_pos + 1].mean()
        elif lag_pos >= 0:
            movavg6m[idx] = targets[: lag_pos + 1].mean()

    return {"Persistence": persistence, "MovingAvg-6m": movavg6m}


def build_xgboost_baseline(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    countries: np.ndarray,
    dates: np.ndarray,
    horizon: int,
) -> tuple:
    """
    Gradient-boosted tree baseline (sklearn HistGradientBoostingRegressor).
    Uses single-timestep features — no temporal window — trained on pre-test data.
    sklearn is used instead of XGBoost to avoid macOS OpenMP segfaults.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    test_key_to_idx = {(c, d): j for j, (c, d) in enumerate(zip(countries, dates))}

    train_rows, train_targets = [], []
    test_rows  = [None] * len(countries)
    test_found = [False] * len(countries)

    for iso3 in df["country_iso3"].unique():
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month")
        all_ym = cdf["year_month"].values
        feats  = cdf[features].values.astype(np.float32)
        raw_t  = cdf[target_col].values.astype(np.float32)

        for i in range(horizon, len(cdf)):
            ym    = all_ym[i]
            y_val = raw_t[i]
            if np.isnan(y_val):
                continue
            x_row = feats[i - horizon]
            key = (iso3, ym)
            if key in test_key_to_idx:
                j = test_key_to_idx[key]
                test_rows[j]  = x_row
                test_found[j] = True
            elif ym < VIEWS_WINDOW_START:
                train_rows.append(x_row)
                train_targets.append(y_val)

    found_idx = [j for j, ok in enumerate(test_found) if ok]
    if len(train_rows) == 0 or len(found_idx) == 0:
        return np.full(len(countries), np.nan), None

    X_train = np.stack(train_rows).astype(np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    X_test  = np.stack([test_rows[j] for j in found_idx]).astype(np.float32)

    model = HistGradientBoostingRegressor(
        max_iter=400, max_depth=5, learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds_raw = model.predict(X_test)
    preds_out = np.full(len(countries), np.nan)
    for j, pred in zip(found_idx, preds_raw):
        preds_out[j] = float(max(pred, 0))

    return preds_out, model


colour_palette = {
    "ours":        "#2563EB",
    "persistence": "#DC2626",
    "movavg":      "#D97706",
    "views_best":  "#059669",
    "neutral":     "#6B7280",
}


def load_model_and_data():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    df = pd.read_csv(DATA_PATH)
    features = get_feature_list(df)
    model = ConvTransformer(n_features=len(features)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=device))
    model.eval()
    return model, df, features, device


def collect_horizon_data(model, df, features, device, horizons):
    """For each horizon: (x_list, y, dates, countries, samples)."""
    data = {}
    for h in horizons:
        print(f"  Generating predictions for h={h}...")
        x_all, y_all, dates_all, countries_all = build_test_samples(
            df, features, window_size=WINDOW_SIZE, horizon=h,
        )
        from src.evaluation.eval_runner import find_usable_months
        usable, _ = find_usable_months(y_all, dates_all)
        mask = np.isin(dates_all, usable)
        x_u = [x_all[i] for i in range(len(x_all)) if mask[i]]
        y_u = y_all[mask]
        d_u = dates_all[mask]
        c_u = countries_all[mask]
        samples = generate_predictions(model, x_u, device, N_SAMPLES, BATCH_SIZE)
        baselines = build_baselines(df, "ucdp_fatalities_best", c_u, d_u, h)
        print(f"  Training XGBoost baseline (h={h})...")
        xgb_preds, _ = build_xgboost_baseline(df, features, "ucdp_fatalities_best", c_u, d_u, h)
        baselines["XGBoost"] = xgb_preds
        data[h] = dict(y=y_u, dates=d_u, countries=c_u, samples=samples, baselines=baselines)
    return data


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_pit(data):
    print("Plot 1: PIT histogram...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle("Calibration -- PIT Histogram (uniform = perfectly calibrated)", fontsize=13)

    for ax, h in zip(axes, HORIZONS):
        pits = pit_values(data[h]["y"], data[h]["samples"])
        n_bins = 10
        counts, edges = np.histogram(pits, bins=n_bins, range=(0, 1))
        freq = counts / len(pits)
        expected = 1 / n_bins
        centers = (edges[:-1] + edges[1:]) / 2
        colours = [colour_palette["ours"] if abs(f - expected) > 0.02 else colour_palette["neutral"]
                   for f in freq]
        ax.bar(centers, freq, width=0.09, color=colours, edgecolor="white", linewidth=0.5)
        ax.axhline(expected, color="black", linestyle="--", linewidth=1, label="Ideal")
        ax.set_title(f"Horizon {h}m  (n={len(pits)})")
        ax.set_xlabel("PIT value")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(freq) * 1.4)

    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    savefig("pit_histogram.png")


def plot_crps_per_month(data):
    print("Plot 2: Per-month CRPS...")
    if not os.path.exists(VIEWS_CSV):
        print("  (skipped -- ViEWS CSV not found)")
        return

    views_df = load_views_monthly_scores(VIEWS_CSV)
    h1 = data[1]
    months = sorted(set(h1["dates"]))

    our_crps = {}
    for m in months:
        mask = h1["dates"] == m
        our_crps[m] = float(crps_sample(h1["y"][mask], h1["samples"][mask]).mean())

    refs = [r for r in TOP_REFS if r in views_df["Model"].values][:3]
    ref_crps = {r: {} for r in refs}
    for r in refs:
        for m in months:
            row = views_df[(views_df["Model"] == r) & (views_df["year_month"] == m)]
            if len(row):
                ref_crps[r][m] = float(row["CRPS"].values[0])

    x = np.arange(len(months))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x, [our_crps[m] for m in months], width, label="TFT (ours)",
           color=colour_palette["ours"], zorder=3)
    ref_colours = ["#059669", "#7C3AED", "#B45309"]
    for i, (r, col) in enumerate(zip(refs, ref_colours)):
        vals = [ref_crps[r].get(m, np.nan) for m in months]
        ax.bar(x + (i + 1) * width, vals, width, label=r[:28], color=col, zorder=3, alpha=0.85)

    ax.set_xticks(x + width * (len(refs) / 2))
    ax.set_xticklabels(months, rotation=25, ha="right")
    ax.set_ylabel("CRPS (lower = better)")
    ax.set_title("Per-month CRPS -- TFT vs top ViEWS entries (h=1)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    savefig("crps_per_month.png")


def plot_horizon_comparison(data):
    print("Plot 3: Horizon comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Performance across forecast horizons", fontsize=13)

    metrics = {
        "CRPS": lambda d: float(crps_sample(d["y"], d["samples"]).mean()),
        "MAE":  lambda d: mae(d["y"], d["samples"]),
        "RMSE": lambda d: rmse(d["y"], d["samples"]),
    }

    for ax, (metric_name, metric_fn) in zip(axes, metrics.items()):
        ours_vals, pers_vals, mova_vals, xgb_vals = [], [], [], []
        for h in HORIZONS:
            d = data[h]
            ours_vals.append(metric_fn(d))
            bl = d["baselines"]
            mask_p = ~np.isnan(bl["Persistence"])
            mask_m = ~np.isnan(bl["MovingAvg-6m"])
            mask_x = ~np.isnan(bl["XGBoost"])
            pers_vals.append(mae_point(d["y"][mask_p], bl["Persistence"][mask_p])
                             if metric_name == "MAE"
                             else rmse_point(d["y"][mask_p], bl["Persistence"][mask_p])
                             if metric_name == "RMSE"
                             else float("nan"))
            mova_vals.append(mae_point(d["y"][mask_m], bl["MovingAvg-6m"][mask_m])
                             if metric_name == "MAE"
                             else rmse_point(d["y"][mask_m], bl["MovingAvg-6m"][mask_m])
                             if metric_name == "RMSE"
                             else float("nan"))
            xgb_vals.append(mae_point(d["y"][mask_x], bl["XGBoost"][mask_x])
                            if metric_name == "MAE"
                            else rmse_point(d["y"][mask_x], bl["XGBoost"][mask_x])
                            if metric_name == "RMSE"
                            else float("nan"))

        ax.plot(HORIZONS, ours_vals, "o-", color=colour_palette["ours"],
                linewidth=2, markersize=7, label="TFT (ours)", zorder=3)

        if metric_name in ("MAE", "RMSE"):
            ax.plot(HORIZONS, pers_vals, "s--", color=colour_palette["persistence"],
                    linewidth=1.5, markersize=6, label="Persistence", zorder=2)
            ax.plot(HORIZONS, mova_vals, "^--", color=colour_palette["movavg"],
                    linewidth=1.5, markersize=6, label="MovAvg-6m", zorder=2)
            ax.plot(HORIZONS, xgb_vals, "D--", color="#7C3AED",
                    linewidth=1.5, markersize=6, label="XGBoost", zorder=2)

        if metric_name == "CRPS" and os.path.exists(VIEWS_CSV):
            views_df = load_views_monthly_scores(VIEWS_CSV)
            usable = sorted(set(data[1]["dates"]))
            filt = views_df[views_df["year_month"].isin(usable)]
            best_views = filt.groupby("Model")["CRPS"].mean().min()
            ax.axhline(best_views, color=colour_palette["views_best"], linestyle=":",
                       linewidth=1.5, label=f"ViEWS best ({best_views:.1f})", zorder=1)

        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"{h}m" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig("horizon_comparison.png")


def plot_roc(data):
    print("Plot 4: ROC curves...")
    fig, ax = plt.subplots(figsize=(6, 6))
    colours = [colour_palette["ours"], "#7C3AED", "#D97706"]

    for h, col in zip(HORIZONS, colours):
        d = data[h]
        is_spike = (d["y"] > SPIKE_THRESH).astype(float)
        y_score = (d["samples"] > SPIKE_THRESH).mean(axis=1)

        n_pos = is_spike.sum()
        n_neg = len(is_spike) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        desc = np.argsort(y_score)[::-1]
        y_sorted = is_spike[desc]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = np.concatenate([[0], tp / n_pos, [1]])
        fpr = np.concatenate([[0], fp / n_neg, [1]])
        auc = float(_trapz(tpr, fpr))
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"h={h}m  (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Spike Detection ROC  (threshold >{SPIKE_THRESH} fatalities)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("roc_curve.png")


def plot_pred_vs_actual(data):
    print("Plot 5: Predicted vs actual...")
    d = data[1]
    nonzero = d["y"] > 0
    y_true = d["y"][nonzero]
    y_pred = d["samples"][nonzero].mean(axis=1)
    y_lo   = np.quantile(d["samples"][nonzero], 0.05, axis=1)
    y_hi   = np.quantile(d["samples"][nonzero], 0.95, axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.errorbar(y_true, y_pred,
                yerr=[np.maximum(y_pred - y_lo, 0), np.maximum(y_hi - y_pred, 0)],
                fmt="none", ecolor=colour_palette["ours"], alpha=0.15, linewidth=0.5, zorder=1)
    sc = ax.scatter(y_true, y_pred, c=np.log1p(y_true), cmap="viridis",
                    s=18, alpha=0.7, zorder=2)
    plt.colorbar(sc, ax=ax, label="log1p(actual fatalities)")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.6, label="Perfect")
    ax.set_xlabel("Actual fatalities")
    ax.set_ylabel("Predicted mean fatalities")
    ax.set_title("Predicted vs Actual -- non-zero months (h=1)\nerror bars = 90% prediction interval")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    savefig("pred_vs_actual.png")


def plot_country_timeseries(data, df):
    print("Plot 6: Country time series...")
    d = data[1]

    test_totals = (
        pd.DataFrame({"iso3": d["countries"], "month": d["dates"], "y": d["y"]})
        .groupby("iso3")["y"].sum()
        .nlargest(5)
    )
    top_countries = test_totals.index.tolist()

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=False)
    fig.suptitle("Top-5 conflict countries -- predicted distribution vs actual (h=1)", fontsize=12)

    for ax, iso3 in zip(axes, top_countries):
        mask = d["countries"] == iso3
        months = d["dates"][mask]
        y_true  = d["y"][mask]
        samp    = d["samples"][mask]

        median = np.median(samp, axis=1)
        lo90   = np.quantile(samp, 0.05, axis=1)
        hi90   = np.quantile(samp, 0.95, axis=1)
        lo50   = np.quantile(samp, 0.25, axis=1)
        hi50   = np.quantile(samp, 0.75, axis=1)

        sort_idx = np.argsort(months)
        months_s = months[sort_idx]
        y_true_s = y_true[sort_idx]
        median_s = median[sort_idx]
        lo90_s, hi90_s = lo90[sort_idx], hi90[sort_idx]
        lo50_s, hi50_s = lo50[sort_idx], hi50[sort_idx]

        x = np.arange(len(months_s))
        ax.fill_between(x, lo90_s, hi90_s, alpha=0.2, color=colour_palette["ours"], label="90% PI")
        ax.fill_between(x, lo50_s, hi50_s, alpha=0.35, color=colour_palette["ours"], label="50% PI")
        ax.plot(x, median_s, color=colour_palette["ours"], linewidth=1.5, label="Median pred")
        ax.plot(x, y_true_s, "o-", color="black", markersize=5, linewidth=1.5, label="Actual")
        ax.set_title(f"{iso3}  (total actual: {int(y_true_s.sum()):,})")
        ax.set_xticks(x)
        ax.set_xticklabels(months_s, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Fatalities")
        ax.grid(alpha=0.2)

    handles = [
        mpatches.Patch(color=colour_palette["ours"], alpha=0.2, label="90% PI"),
        mpatches.Patch(color=colour_palette["ours"], alpha=0.5, label="50% PI"),
        plt.Line2D([0], [0], color=colour_palette["ours"], linewidth=1.5, label="Median pred"),
        plt.Line2D([0], [0], color="black", marker="o", markersize=4, linewidth=1.5, label="Actual"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    savefig("country_timeseries.png")


def plot_crps_by_bucket(data):
    print("Plot 7: CRPS by fatality bucket...")
    buckets = [
        ("y = 0",          lambda y: y == 0),
        ("0 < y <= 25",    lambda y: (y > 0) & (y <= 25)),
        ("25 < y <= 100",  lambda y: (y > 25) & (y <= 100)),
        ("100 < y <= 500", lambda y: (y > 100) & (y <= 500)),
        ("y > 500",        lambda y: y > 500),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(buckets))
    width = 0.25
    colours = [colour_palette["ours"], "#7C3AED", "#D97706"]

    for i, h in enumerate(HORIZONS):
        d = data[h]
        crps_all = crps_sample(d["y"], d["samples"])
        vals = []
        for _, mask_fn in buckets:
            mask = mask_fn(d["y"])
            vals.append(float(crps_all[mask].mean()) if mask.sum() > 0 else 0)
        ax.bar(x + i * width, vals, width, label=f"h={h}m",
               color=colours[i], zorder=3, alpha=0.85)

    counts = [mask_fn(data[1]["y"]).sum() for _, mask_fn in buckets]
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{label}\n(n={c})" for (label, _), c in zip(buckets, counts)])
    ax.set_ylabel("Mean CRPS (lower = better)")
    ax.set_title("CRPS by fatality magnitude -- where does the model struggle?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    savefig("crps_by_bucket.png")


def plot_failure_analysis(data):
    print("Plot 8: Failure analysis...")
    d = data[1]
    crps_all = crps_sample(d["y"], d["samples"])
    top_k = 15

    idx = np.argsort(crps_all)[-top_k:][::-1]
    labels = [f"{d['countries'][i]}  {d['dates'][i]}" for i in idx]
    crps_vals = crps_all[idx]
    y_true_vals = d["y"][idx]
    y_pred_vals = d["samples"][idx].mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, crps_vals, color=colour_palette["ours"], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CRPS (lower = better)")
    ax.set_title(f"Top-{top_k} worst-predicted country-months (h=1)\nbar labels: actual | predicted mean")
    for bar, actual, pred in zip(bars, y_true_vals, y_pred_vals):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"actual={int(actual):,}  pred={int(pred):,}",
                va="center", fontsize=7.5, color="#374151")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("failure_analysis.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model and data...")
    model, df, features, device = load_model_and_data()

    print("Collecting predictions across horizons...")
    data = collect_horizon_data(model, df, features, device, HORIZONS)

    plot_pit(data)
    plot_crps_per_month(data)
    plot_horizon_comparison(data)
    plot_roc(data)
    plot_pred_vs_actual(data)
    plot_country_timeseries(data, df)
    plot_crps_by_bucket(data)
    plot_failure_analysis(data)

    print(f"\nAll plots saved to {OUT_DIR}/")
