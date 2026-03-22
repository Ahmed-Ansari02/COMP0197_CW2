"""
evaluate.py — Evaluation for Bayesian LSTM conflict fatality model

Metrics:
    - CRPS  (Continuous Ranked Probability Score) — primary metric
    - IGN   (Ignorance score = mean NLL)
    - MAE   on point prediction (median of predictive distribution)
    - Brier score for P(fatalities > threshold)
    - Calibration: empirical coverage of predictive intervals

Outputs:
    - results/bayesian_lstm/metrics.json       — scalar metrics
    - results/bayesian_lstm/predictions.csv    — per-sample predictions
    - results/bayesian_lstm/calibration.png    — reliability diagram
    - results/bayesian_lstm/uncertainty.png    — epistemic vs aleatoric
"""

import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from dataset import get_dataloaders, ConflictDataset
from model import BayesianLSTM, zinb_nll

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_PATH    = BASE_DIR / "data" / "processed" / "merge" / "model_ready.csv"
CKPT_PATH    = BASE_DIR / "checkpoints" / "bayesian_lstm" / "best_model.pt"
RESULTS_DIR  = BASE_DIR / "results" / "bayesian_lstm"
CONFIG_PATH  = BASE_DIR / "config" / "config.yaml"

MC_SAMPLES   = 50
BATCH_SIZE   = 256
THRESHOLDS   = [1.0, 2.3, 3.9]   # log1p scale: ~1, ~9, ~48 fatalities


# ── CRPS ─────────────────────────────────────────────────────────────────────

def crps_sample_based(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    CRPS via sample-based estimator.
    samples : [n_samples, N]
    y_true  : [N]
    Returns : [N] per-sample CRPS (lower is better)
    """
    S = samples.shape[0]
    # E[|X - y|]
    term1 = np.abs(samples - y_true[None, :]).mean(axis=0)
    # 0.5 * E[|X - X'|] — estimate with pairs
    idx = np.random.choice(S, size=(S, 2), replace=True)
    term2 = 0.5 * np.abs(samples[idx[:, 0]] - samples[idx[:, 1]]).mean(axis=0)
    return term1 - term2


# ── Brier score ───────────────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, prob_pred: np.ndarray) -> float:
    """Binary Brier score for exceedance probability."""
    return float(np.mean((prob_pred - y_true) ** 2))


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> BayesianLSTM:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["train_cfg"]

    model = BayesianLSTM(
        n_dynamic  = ckpt["n_dynamic"],
        n_static   = ckpt["n_static"],
        hidden_dim = cfg["hidden_dim"],
        n_layers   = cfg["n_layers"],
        static_dim = cfg["static_dim"],
        fusion_dim = cfg["fusion_dim"],
        dropout    = cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val NLL={ckpt['val_loss']:.4f})")
    return model


# ── Inference over a dataloader ───────────────────────────────────────────────

def run_inference(model, loader, device, n_samples=MC_SAMPLES):
    """
    Run MC dropout inference over full dataloader.

    Returns dict of numpy arrays, each length N (observed samples only):
        mu_mean, epistemic_var, aleatoric_var,
        mu_mc_samples [n_samples, N],
        y_true, nll
    """
    all_mu_samples   = []   # [S, N]
    all_alpha_samples = []
    all_pi_samples   = []
    all_epi          = []
    all_ale          = []
    all_mu_mean      = []
    all_target       = []
    all_nll          = []
    all_mask         = []

    model.train()  # MC dropout on

    with torch.no_grad():
        for dyn, sta, tgt, mask in loader:
            dyn  = dyn.to(device)
            sta  = sta.to(device)
            tgt  = tgt.to(device)
            mask = mask.to(device)

            (mu_samples, alpha_samples, pi_samples,
             epi_var, ale_var, mu_mean) = model.predict_with_uncertainty(
                dyn, sta, n_samples=n_samples
            )

            # NLL on mean prediction
            nll = zinb_nll(tgt, mu_mean, alpha_samples.mean(0), pi_samples.mean(0))

            all_mu_samples.append(mu_samples.cpu().numpy())     # [S, B]
            all_alpha_samples.append(alpha_samples.cpu().numpy())
            all_pi_samples.append(pi_samples.cpu().numpy())
            all_epi.append(epi_var.cpu().numpy())
            all_ale.append(ale_var.cpu().numpy())
            all_mu_mean.append(mu_mean.cpu().numpy())
            all_target.append(tgt.cpu().numpy())
            all_nll.append(nll.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    # Concatenate across batches
    mu_samples_all   = np.concatenate(all_mu_samples,   axis=1)  # [S, N_total]
    alpha_samples_all = np.concatenate(all_alpha_samples, axis=1)
    pi_samples_all   = np.concatenate(all_pi_samples,   axis=1)
    epi_all          = np.concatenate(all_epi)
    ale_all          = np.concatenate(all_ale)
    mu_mean_all      = np.concatenate(all_mu_mean)
    target_all       = np.concatenate(all_target)
    nll_all          = np.concatenate(all_nll)
    mask_all         = np.concatenate(all_mask).astype(bool)

    # Filter to observed only
    return {
        "mu_samples"   : mu_samples_all[:, mask_all],    # [S, N_obs]
        "alpha_samples": alpha_samples_all[:, mask_all],
        "pi_samples"   : pi_samples_all[:, mask_all],
        "epistemic"    : epi_all[mask_all],
        "aleatoric"    : ale_all[mask_all],
        "mu_mean"      : mu_mean_all[mask_all],
        "y_true"       : target_all[mask_all],
        "nll"          : nll_all[mask_all],
    }


# ── Calibration plot ──────────────────────────────────────────────────────────

def plot_calibration(results: dict, out_path: Path):
    """
    Reliability diagram — empirical coverage vs nominal coverage.
    For each confidence level α, check if y_true falls in the α% interval.
    """
    mu_samples = results["mu_samples"]   # [S, N]
    y_true     = results["y_true"]       # [N]

    levels     = np.arange(0.05, 1.0, 0.05)
    coverages  = []

    for level in levels:
        lo = np.percentile(mu_samples, (1 - level) / 2 * 100, axis=0)
        hi = np.percentile(mu_samples, (1 + level) / 2 * 100, axis=0)
        coverage = np.mean((y_true >= lo) & (y_true <= hi))
        coverages.append(coverage)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(levels, coverages, "o-", color="#1D9E75", lw=2, label="Bayesian LSTM")
    ax.fill_between(levels, levels, coverages,
                    alpha=0.15, color="#1D9E75")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Calibration plot: {out_path}")


# ── Uncertainty decomposition plot ────────────────────────────────────────────

def plot_uncertainty(results: dict, out_path: Path):
    """
    Scatter: epistemic vs aleatoric uncertainty, coloured by y_true.
    """
    epi = results["epistemic"]
    ale = results["aleatoric"]
    y   = results["y_true"]

    # Cap for visibility
    epi_plot = np.clip(epi, 0, np.percentile(epi, 99))
    ale_plot = np.clip(ale, 0, np.percentile(ale, 99))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(ale_plot, epi_plot, c=y, cmap="YlOrRd",
                    alpha=0.4, s=8, rasterized=True)
    plt.colorbar(sc, ax=ax, label="y_true (log1p fatalities)")
    ax.set_xlabel("Aleatoric uncertainty (ZINB variance)")
    ax.set_ylabel("Epistemic uncertainty (MC variance)")
    ax.set_title("Uncertainty decomposition")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Uncertainty plot: {out_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    import yaml

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Splits
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    splits = cfg["splits"]

    # Data — test split only
    _, _, test_loader = get_dataloaders(
        data_path  = DATA_PATH,
        seq_len    = 24,
        batch_size = BATCH_SIZE,
        train_end  = splits["train_end"],
        val_start  = splits["val_start"],
        val_end    = splits["val_end"],
        test_start = splits["test_start"],
    )

    # Model
    model = load_model(CKPT_PATH, device)

    # Inference
    print(f"\nRunning MC inference ({MC_SAMPLES} samples)...")
    results = run_inference(model, test_loader, device, n_samples=MC_SAMPLES)

    N = len(results["y_true"])
    print(f"Observed test samples: {N:,}")

    # ── Metrics ───────────────────────────────────────────────────────────────

    # 1. IGN (mean NLL)
    ign = float(results["nll"].mean())

    # 2. CRPS
    crps_vals = crps_sample_based(results["y_true"], results["mu_samples"])
    crps_mean = float(crps_vals.mean())

    # 3. MAE on median prediction
    median_pred = np.median(results["mu_samples"], axis=0)
    mae = float(np.abs(median_pred - results["y_true"]).mean())

    # 4. Brier scores at thresholds
    brier_scores = {}
    for thresh in THRESHOLDS:
        y_exc   = (results["y_true"] > thresh).astype(float)
        p_exc   = (results["mu_samples"] > thresh).mean(axis=0)
        brier_scores[f"brier_gt_{thresh:.1f}"] = brier_score(y_exc, p_exc)

    # 5. Uncertainty decomposition summary
    epi_mean = float(results["epistemic"].mean())
    ale_mean = float(results["aleatoric"].mean())
    epi_frac = epi_mean / (epi_mean + ale_mean + 1e-8)

    # 6. 80% interval coverage
    # 6. 80% interval coverage (includes structural zero prediction)
    lo_80 = np.percentile(results["mu_samples"], 10, axis=0)
    hi_80 = np.percentile(results["mu_samples"], 90, axis=0)
    pi_mean = results["pi_samples"].mean(axis=0)

    in_interval    = (results["y_true"] >= lo_80) & (results["y_true"] <= hi_80)
    predicted_zero = pi_mean > 0.5
    true_zero      = results["y_true"] == 0

    covered        = in_interval | (true_zero & predicted_zero)
    coverage_80 = float(covered.mean())

    metrics = {
        "n_test_obs"       : N,
        "ign"              : round(ign, 4),
        "crps"             : round(crps_mean, 4),
        "mae"              : round(mae, 4),
        "coverage_80pct"   : round(coverage_80, 4),
        "epistemic_mean"   : round(epi_mean, 4),
        "aleatoric_mean"   : round(ale_mean, 4),
        "epistemic_fraction": round(epi_frac, 4),
        **{k: round(v, 4) for k, v in brier_scores.items()},
    }

    print(f"\n{'='*40}")
    print("TEST METRICS")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")
    print(f"{'='*40}")

    # ── Save results ──────────────────────────────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: {metrics_path}")

    # Predictions CSV
    preds_df = pd.DataFrame({
        "y_true"        : results["y_true"],
        "mu_mean"       : results["mu_mean"],
        "mu_median"     : median_pred,
        "lo_80"         : lo_80,
        "hi_80"         : hi_80,
        "pi_mean"       : pi_mean, 
        "epistemic"     : results["epistemic"],
        "aleatoric"     : results["aleatoric"],
        "crps"          : crps_vals,
        "nll"           : results["nll"],
    })
    preds_path = RESULTS_DIR / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"Predictions: {preds_path}")

    # Plots
    plot_calibration(results, RESULTS_DIR / "calibration.png")
    plot_uncertainty(results, RESULTS_DIR / "uncertainty.png")

    return metrics


if __name__ == "__main__":
    evaluate()