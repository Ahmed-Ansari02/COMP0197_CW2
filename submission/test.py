"""
test.py — Load saved Conv-Transformer, evaluate against ViEWS benchmarks,
           produce metrics and visual results.

Imports model architecture from train.py.

Additional packages (beyond base env): pandas, matplotlib, scipy
GenAI disclosure: LLMs for brainstorming.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from train import (
    ConvTransformer, HurdleStudentT, get_feature_list,
    DATA_DIR, TARGET, EXCLUDE_COLS,
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def crps_sample(y_true, samples):
    """CRPS from empirical samples. Lower is better."""
    N, S = samples.shape
    term1 = np.abs(samples - y_true[:, None]).mean(axis=1)
    sorted_s = np.sort(samples, axis=1)
    weights = 2 * np.arange(S) - S
    term2 = (sorted_s * weights[None, :]).sum(axis=1) / (S * S)
    return term1 - term2


def ign_score(y_true, samples, bandwidth="silverman"):
    """Ignorance score: -log(f(y)) via KDE. Lower is better."""
    N = len(y_true)
    ign = np.zeros(N)
    for i in range(N):
        s = samples[i]
        if s.std() < 1e-10:
            ign[i] = 0.0 if np.abs(y_true[i] - s.mean()) < 1e-6 else 50.0
            continue
        try:
            kde = stats.gaussian_kde(s, bw_method=bandwidth)
            ign[i] = -np.log(max(kde(y_true[i])[0], 1e-10))
        except Exception:
            ign[i] = 50.0
    return ign


def interval_score(y_true, lower, upper, alpha=0.1):
    """Interval score for a (1-alpha) prediction interval."""
    width = upper - lower
    below = np.maximum(lower - y_true, 0)
    above = np.maximum(y_true - upper, 0)
    return width + (2 / alpha) * below + (2 / alpha) * above


def mis_from_samples(y_true, samples, alpha=0.1):
    lower = np.quantile(samples, alpha / 2, axis=1)
    upper = np.quantile(samples, 1 - alpha / 2, axis=1)
    return float(interval_score(y_true, lower, upper, alpha).mean())


def pit_values(y_true, samples):
    """PIT: proportion of samples <= y. Should be Uniform(0,1) if calibrated."""
    return (samples <= y_true[:, None]).mean(axis=1)


def pit_reliability(y_true, samples, n_bins=10):
    pits = pit_values(y_true, samples)
    observed, bin_edges = np.histogram(pits, bins=n_bins, range=(0, 1))
    observed_freq = observed / len(pits)
    chi2_stat, chi2_pval = stats.chisquare(
        observed, f_exp=np.full(n_bins, len(pits) / n_bins))
    return {
        "bin_edges": bin_edges, "observed_freq": observed_freq,
        "expected_freq": 1.0 / n_bins,
        "chi2_stat": float(chi2_stat), "chi2_pval": float(chi2_pval),
    }


def spike_metrics(y_true, samples, threshold=500, alert_quantile=0.9):
    is_spike = y_true > threshold
    is_alert = np.quantile(samples, alert_quantile, axis=1) > threshold
    n_spikes, n_alerts = int(is_spike.sum()), int(is_alert.sum())
    recall = float((is_spike & is_alert).sum() / n_spikes) if n_spikes > 0 else float("nan")
    precision = float((is_spike & is_alert).sum() / n_alerts) if n_alerts > 0 else float("nan")
    if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {"recall": recall, "precision": precision, "f1": f1,
            "n_spikes": n_spikes, "n_alerts": n_alerts}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ViEWS BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

VIEWS_WINDOW_START = "2024-07"
VIEWS_WINDOW_END = "2025-06"

MONTH_ID_MAP = {
    535: "2024-07", 536: "2024-08", 537: "2024-09", 538: "2024-10",
    539: "2024-11", 540: "2024-12", 541: "2025-01", 542: "2025-02",
    543: "2025-03", 544: "2025-04", 545: "2025-05", 546: "2025-06",
}


def load_views_scores(csv_path="cm_monthly_scores_full_Jul-Jun.csv"):
    df = pd.read_csv(csv_path)
    df["year_month"] = df["Month ID"].map(MONTH_ID_MAP)
    return df


def get_views_benchmarks(csv_path, usable_months=None):
    df = load_views_scores(csv_path)
    if usable_months:
        df = df[df["year_month"].isin(usable_months)]
    benchmarks = {}
    for model_name in df["Model"].unique():
        mdf = df[df["Model"] == model_name]
        if len(mdf) > 0:
            benchmarks[model_name] = {
                "crps": float(mdf["CRPS"].mean()),
                "ign": float(mdf["ab-Log Score"].mean()),
                "mis": float(mdf["MIS"].mean()),
            }
    return benchmarks


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def build_test_samples(df, features, window_size=24,
                       test_start=VIEWS_WINDOW_START, test_end=VIEWS_WINDOW_END):
    """Build test windows where the TARGET month is in the ViEWS window.

    NaN targets treated as zero (UCDP absence = no conflict = 0 fatalities).
    Short sequences zero-padded so all countries are included.
    """
    x_all, y_all, dates, countries = [], [], [], []
    for iso3 in df["country_iso3"].unique():
        cdf = df[df["country_iso3"] == iso3].sort_values("year_month")

        feat = cdf[features].values.astype(np.float32)
        raw_targets = cdf[TARGET].values.astype(np.float32)
        raw_targets = np.nan_to_num(raw_targets, nan=0.0)
        targets = np.expm1(raw_targets)
        yms = cdf["year_month"].values

        # Zero-pad short sequences
        if len(feat) < window_size:
            pad_len = window_size - len(feat)
            feat = np.concatenate([
                np.zeros((pad_len, feat.shape[1]), dtype=np.float32), feat])
            targets = np.concatenate([np.zeros(pad_len, dtype=np.float32), targets])
            yms = np.concatenate([np.full(pad_len, ""), yms])

        for i in range(window_size, len(feat)):
            ym = yms[i]
            if ym < test_start or (test_end and ym > test_end):
                continue
            x_all.append(feat[i - window_size:i])
            y_all.append(targets[i])
            dates.append(ym)
            countries.append(iso3)
    return x_all, np.array(y_all), np.array(dates), np.array(countries)


def find_usable_months(y_true, dates):
    usable, empty = [], []
    for m in sorted(set(dates)):
        mask = dates == m
        if (y_true[mask] > 0).sum() > 0:
            usable.append(m)
        else:
            empty.append(m)
    return usable, empty


def generate_predictions(model, x_list, device, n_samples=1000, batch_size=128):
    model.eval()
    all_samples = []
    for start in range(0, len(x_list), batch_size):
        end = min(start + batch_size, len(x_list))
        bx = torch.tensor(np.stack(x_list[start:end])).to(device)
        with torch.no_grad():
            pred = model.predict(bx, n_samples=n_samples)
            all_samples.append(pred["samples"].cpu().numpy())
        print(f"  Predicted {end}/{len(x_list)}", end="\r")
    print()
    return np.concatenate(all_samples, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_pit_histogram(y_true, samples, save_path="pit_histogram.png"):
    """PIT calibration histogram."""
    rel = pit_reliability(y_true, samples)
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = rel["bin_edges"]
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    ax.bar(centers, rel["observed_freq"], width=widths * 0.9,
           color="steelblue", edgecolor="white", alpha=0.8)
    ax.axhline(rel["expected_freq"], color="red", linestyle="--",
               label=f"Uniform (p={rel['chi2_pval']:.3f})")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Frequency")
    ax.set_title("PIT Calibration Histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_crps_comparison(our_crps, benchmarks, save_path="crps_comparison.png"):
    """Horizontal bar chart: our model vs ViEWS benchmarks."""
    models = {f"[ViEWS] {k}": v["crps"] for k, v in benchmarks.items()}
    models[">> Ours (Conv-Transformer)"] = our_crps

    # Sort by CRPS and take top 15
    sorted_models = sorted(models.items(), key=lambda x: x[1])[:15]
    names = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]
    colors = ["#e74c3c" if "Ours" in n else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("CRPS (lower is better)")
    ax.set_title("Model Comparison — Country-level CRPS")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_monthly_crps(y_true, samples, dates, benchmarks, views_csv,
                      usable_months, save_path="monthly_crps.png"):
    """Per-month CRPS line plot."""
    views_df = load_views_scores(views_csv)
    if usable_months:
        views_df = views_df[views_df["year_month"].isin(usable_months)]

    months = sorted(usable_months)
    our_monthly = []
    for m in months:
        mask = dates == m
        if mask.sum() > 0:
            our_monthly.append(float(crps_sample(y_true[mask], samples[mask]).mean()))
        else:
            our_monthly.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(months, our_monthly, "o-", color="#e74c3c", linewidth=2,
            markersize=6, label="Ours", zorder=5)

    top_refs = ["VIEWS_bm_ph_conflictology_country12", "ConflictForecast", "CCEW_tft"]
    colors_ref = ["#3498db", "#2ecc71", "#9b59b6"]
    for ref, color in zip(top_refs, colors_ref):
        ref_monthly = []
        for m in months:
            rdf = views_df[(views_df["Model"] == ref) & (views_df["year_month"] == m)]
            ref_monthly.append(rdf["CRPS"].values[0] if len(rdf) > 0 else np.nan)
        ax.plot(months, ref_monthly, "s--", color=color, markersize=4,
                alpha=0.7, label=ref[:25])

    ax.set_xlabel("Month")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Per-month CRPS Comparison")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(our_metrics, benchmarks):
    """Print sorted comparison table."""
    rows = [{"Model": ">> Conv-Transformer (ours)", **our_metrics}]
    for name, scores in benchmarks.items():
        rows.append({"Model": f"[ViEWS] {name}", "crps": scores["crps"],
                      "ign": scores["ign"], "mis_90": scores["mis"]})

    df = pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)

    print(f"\n{'='*80}")
    print("MODEL COMPARISON (sorted by CRPS, lower = better)")
    print(f"{'='*80}")
    print(f"{'Model':50s} {'CRPS':>7s} {'IGN':>7s} {'MIS_90':>9s}")
    print("-" * 80)
    for _, row in df.iterrows():
        crps = f"{row['crps']:.2f}" if pd.notna(row.get('crps')) else "--"
        ign = f"{row.get('ign', np.nan):.2f}" if pd.notna(row.get('ign')) else "--"
        mis = f"{row.get('mis_90', np.nan):.1f}" if pd.notna(row.get('mis_90')) else "--"
        print(f"{row['Model']:50s} {crps:>7s} {ign:>7s} {mis:>9s}")
    print(f"{'='*80}")


def print_diagnostics(y_true, samples):
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

  Calibration:
    Avg P(zero):    {pred_pzero.mean():.3f}   (actual: {zero_mask.mean():.3f})
    Mean pred:      {pred_means.mean():.1f}   (actual: {y_true.mean():.1f})
    Median pred:    {np.median(pred_means):.1f}   (actual: {np.median(y_true):.1f})
""")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate Conv-Transformer")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--data", default="data/merge/model_ready.csv")
    parser.add_argument("--views-csv", default="cm_monthly_scores_full_Jul-Jun.csv")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.data)
    features = get_feature_list(df)
    print(f"Loaded {df.shape}, {len(features)} features")

    # Load model
    model = ConvTransformer(
        n_features=len(features), window_size=args.window_size,
        d_model=args.hidden_dim, n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        dim_feedforward=args.hidden_dim * 2,
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build test samples
    x_all, y_all, dates_all, countries_all = build_test_samples(
        df, features, window_size=args.window_size)
    usable, empty = find_usable_months(y_all, dates_all)
    usable_mask = np.isin(dates_all, usable)
    x_usable = [x_all[i] for i in range(len(x_all)) if usable_mask[i]]
    y_usable = y_all[usable_mask]
    dates_usable = dates_all[usable_mask]

    print(f"\n{'='*72}")
    print(f"  ViEWS BENCHMARK COMPARISON")
    print(f"  Window: {VIEWS_WINDOW_START} to {VIEWS_WINDOW_END}")
    print(f"  Evaluated on: {', '.join(usable)}  ({len(usable)}/{len(usable)+len(empty)} months)")
    if empty:
        print(f"  Excluded (no UCDP data): {', '.join(empty)}")
    print(f"{'='*72}")

    # Generate predictions
    samples = generate_predictions(model, x_usable, device, args.n_samples)

    # Compute metrics
    our_crps = float(crps_sample(y_usable, samples).mean())
    our_ign = float(ign_score(y_usable, samples).mean())
    our_mis_90 = mis_from_samples(y_usable, samples, alpha=0.1)
    our_mis_50 = mis_from_samples(y_usable, samples, alpha=0.5)
    calib = pit_reliability(y_usable, samples)
    spikes = spike_metrics(y_usable, samples)

    our_metrics = {
        "crps": our_crps, "ign": our_ign, "mis_90": our_mis_90,
        "mis_50": our_mis_50, "calib_p": calib["chi2_pval"],
        "spike_recall": spikes["recall"], "spike_f1": spikes["f1"],
    }

    # Load ViEWS benchmarks
    benchmarks = get_views_benchmarks(args.views_csv, usable)

    # Print results
    print_comparison_table(our_metrics, benchmarks)

    print(f"\n{'='*72}")
    print(f"  DIAGNOSTICS")
    print(f"{'='*72}")
    print_diagnostics(y_usable, samples)

    # Visualizations
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Generating visualizations...")
    plot_pit_histogram(y_usable, samples, results_dir / "pit_histogram.png")
    plot_crps_comparison(our_crps, benchmarks, results_dir / "crps_comparison.png")
    plot_monthly_crps(y_usable, samples, dates_usable, benchmarks,
                      args.views_csv, usable, results_dir / "monthly_crps.png")

    print(f"\nDone. Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
