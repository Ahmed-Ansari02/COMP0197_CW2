"""Train the Conv-Transformer fatality prediction model.

Usage examples:
    # Hurdle-Student-t model with conflict features only
    python train.py --features a

    # Hurdle-Student-t model with all features
    python train.py --features abc

    # Deterministic MSE baseline
    python train.py --features ab --baseline

    # Custom hyperparameters
    python train.py --features ab --lr 1e-4 --epochs 50 --batch-size 128
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from configs.default import Config
from model.conv_transformer import ConvTransformer
from model.dataset import build_dataloaders
from model.losses import hurdle_loss, mse_baseline_loss
from model.utils import (
    EarlyStopping,
    TrainingLogger,
    save_checkpoint,
    load_checkpoint,
    mc_dropout_inference,
)


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, baseline):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y_count, y_binary in loader:
        x = x.to(device)
        y_count = y_count.to(device)
        y_binary = y_binary.to(device)

        optimizer.zero_grad()

        if baseline:
            y_pred = model(x)
            loss = mse_baseline_loss(y_pred, y_count)
        else:
            conflict_logit, mu, sigma, nu = model(x)
            loss, _, _ = hurdle_loss(
                conflict_logit, mu, sigma, nu,
                y_count, y_binary,
                pos_weight=cfg.train.pos_weight,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, cfg, device, baseline):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y_count, y_binary in loader:
        x = x.to(device)
        y_count = y_count.to(device)
        y_binary = y_binary.to(device)

        if baseline:
            y_pred = model(x)
            loss = mse_baseline_loss(y_pred, y_count)
        else:
            conflict_logit, mu, sigma, nu = model(x)
            loss, _, _ = hurdle_loss(
                conflict_logit, mu, sigma, nu,
                y_count, y_binary,
                pos_weight=cfg.train.pos_weight,
            )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Save predictions CSV
# ---------------------------------------------------------------------------

def save_predictions(
    result: dict,
    meta: dict,
    indices: list,
    path: str,
    baseline: bool,
):
    """Write per-sample prediction CSV for the evaluation team."""
    samples = meta["samples"]
    ym = meta["year_month"]
    cids = meta["country_iso3"]

    rows = []
    for i, global_idx in enumerate(indices):
        _, target_idx = samples[global_idx]
        row = {
            "country_iso3": cids[target_idx],
            "year_month": ym[target_idx],
            "y_true": result["y_true"][i],
        }
        if baseline:
            row["y_pred_mean"] = result["y_pred_mean"][i]
            row["y_pred_std"] = result["y_pred_std"][i]
        else:
            row["p_conflict_mean"] = result["p_conflict_mean"][i]
            row["p_conflict_std"] = result["p_conflict_std"][i]
            row["mu_mean"] = result["mu_mean"][i]
            row["mu_std"] = result["mu_std"][i]
            row["sigma_mean"] = result["sigma_mean"][i]
            row["sigma_std"] = result["sigma_std"][i]
            row["nu_mean"] = result["nu_mean"][i]
            row["nu_std"] = result["nu_std"][i]
        rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[SAVE] predictions -> {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Conv-Transformer fatality model")
    parser.add_argument("--features", choices=["a", "ab", "abc"], default="a",
                        help="Feature group: a (conflict), ab (+structural), abc (+volatility)")
    parser.add_argument("--baseline", action="store_true",
                        help="Train deterministic MSE baseline instead of hurdle model")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()

    # Override defaults from CLI
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.window_size is not None:
        cfg.model.window_size = args.window_size
    if args.patience is not None:
        cfg.train.patience = args.patience
    if args.mc_samples is not None:
        cfg.train.mc_samples = args.mc_samples
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model_tag = "baseline" if args.baseline else "hurdle"
    run_name = f"{args.features}_{model_tag}"
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}  |  device: {device}")
    print(f"{'='*60}\n")

    # ---- Data -----------------------------------------------------------
    train_loader, val_loader, test_loader, n_features, meta = build_dataloaders(
        cfg, args.features,
    )

    if len(train_loader) == 0:
        print("[ERROR] No training batches. Check data paths and window size.")
        sys.exit(1)

    # ---- Model ----------------------------------------------------------
    model = ConvTransformer(
        n_features=n_features,
        window_size=cfg.model.window_size,
        patch_size=cfg.model.patch_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_transformer_layers=cfg.model.n_transformer_layers,
        n_conv_layers=cfg.model.n_conv_layers,
        conv_kernel_size=cfg.model.conv_kernel_size,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        baseline=args.baseline,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] parameters: {n_params:,}")

    # ---- Optimiser & scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, cfg.train.warmup_steps, total_steps)

    # ---- Training -------------------------------------------------------
    early_stop = EarlyStopping(patience=cfg.train.patience)
    logger = TrainingLogger()
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{run_name}.pt")
    best_val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, args.baseline,
        )
        val_loss = validate(model, val_loader, cfg, device, args.baseline)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(epoch, train_loss, val_loss, current_lr)

        improved = early_stop.step(val_loss)
        marker = " *" if improved else ""
        print(f"  epoch {epoch:3d}/{cfg.train.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={current_lr:.2e}  ({elapsed:.1f}s){marker}")

        if improved:
            best_val_loss = val_loss
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss)

        if early_stop.should_stop:
            print(f"\n  Early stopping at epoch {epoch} (patience={cfg.train.patience})")
            break

    # ---- Restore best model ---------------------------------------------
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
        print(f"\n[LOAD] Best model from {ckpt_path}  (val_loss={best_val_loss:.4f})")

    # ---- MC Dropout inference on val & test -----------------------------
    print(f"\n[MC] Running {cfg.train.mc_samples} stochastic forward passes ...")

    val_result = mc_dropout_inference(
        model, val_loader, cfg.train.mc_samples, device, args.baseline,
    )
    test_result = mc_dropout_inference(
        model, test_loader, cfg.train.mc_samples, device, args.baseline,
    )

    # ---- Save outputs ---------------------------------------------------
    save_predictions(
        val_result, meta, meta["val_indices"],
        os.path.join(cfg.output_dir, f"{run_name}_val_predictions.csv"),
        args.baseline,
    )
    save_predictions(
        test_result, meta, meta["test_indices"],
        os.path.join(cfg.output_dir, f"{run_name}_test_predictions.csv"),
        args.baseline,
    )
    logger.save(os.path.join(cfg.output_dir, f"{run_name}_training_log.csv"))

    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint:    {ckpt_path}")
    print(f"  Results:       {cfg.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
