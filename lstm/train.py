"""
train.py — Training loop for Bayesian LSTM conflict fatality model

Features:
    - ZINB negative log-likelihood loss (masked on unobserved targets)
    - Gradient clipping (max norm 1.0)
    - Early stopping on validation loss
    - Saves best checkpoint by val loss
    - GPU/CPU auto-detection
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import time
from pathlib import Path

from dataset import get_dataloaders
from model import BayesianLSTM, zinb_nll

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
DATA_PATH   = BASE_DIR / "data" / "processed" / "merge" / "model_ready.csv"
CKPT_DIR    = BASE_DIR / "checkpoints" / "bayesian_lstm"

TRAIN_CFG = {
    # Data
    "seq_len"    : 24,
    "batch_size" : 256,
    # Model
    "hidden_dim" : 128,
    "n_layers"   : 2,
    "static_dim" : 32,
    "fusion_dim" : 128,
    "dropout"    : 0.3,
    # Training
    "lr"         : 1e-3,
    "weight_decay": 1e-4,
    "max_epochs" : 100,
    "patience"   : 10,       # early stopping patience
    "clip_norm"  : 1.0,      # gradient clipping
    "mc_samples" : 50,       # MC passes at inference
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_splits(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["splits"]


def masked_zinb_loss(mu, alpha, pi, target, mask):
    """
    ZINB NLL averaged over observed samples only.
    Unobserved targets (mask=False) are excluded from loss.
    """
    nll = zinb_nll(target, mu, alpha, pi)   # [B]
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nll[mask].mean()


def run_epoch(model, loader, optimizer, device, clip_norm, is_train):
    """Single epoch — train or val."""
    model.train() if is_train else model.eval()

    # For val we still want MC dropout on — handled by MCDropout class
    # but we disable grad for val
    total_loss  = 0.0
    total_obs   = 0
    n_batches   = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for dyn, sta, tgt, mask in loader:
            dyn  = dyn.to(device)
            sta  = sta.to(device)
            tgt  = tgt.to(device)
            mask = mask.to(device)

            mu, alpha, pi = model(dyn, sta)
            loss = masked_zinb_loss(mu, alpha, pi, tgt, mask)

            if is_train and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            n_obs = mask.sum().item()
            total_loss += loss.item() * n_obs
            total_obs  += n_obs
            n_batches  += 1

    avg_loss = total_loss / max(total_obs, 1)
    return avg_loss


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Splits from config
    splits = load_splits(CONFIG_PATH)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path  = DATA_PATH,
        seq_len    = TRAIN_CFG["seq_len"],
        batch_size = TRAIN_CFG["batch_size"],
        train_end  = splits["train_end"],
        val_start  = splits["val_start"],
        val_end    = splits["val_end"],
        test_start = splits["test_start"],
    )

    # Infer feature dims from first batch
    dyn_sample, sta_sample, _, _ = next(iter(train_loader))
    n_dynamic = dyn_sample.shape[2]
    n_static  = sta_sample.shape[1]
    print(f"n_dynamic={n_dynamic}, n_static={n_static}")

    # Model
    model = BayesianLSTM(
        n_dynamic  = n_dynamic,
        n_static   = n_static,
        hidden_dim = TRAIN_CFG["hidden_dim"],
        n_layers   = TRAIN_CFG["n_layers"],
        static_dim = TRAIN_CFG["static_dim"],
        fusion_dim = TRAIN_CFG["fusion_dim"],
        dropout    = TRAIN_CFG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Checkpoint dir
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CKPT_DIR / "best_model.pt"

    # Training loop
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print(f"\n{'Epoch':>6} {'Train NLL':>10} {'Val NLL':>10} {'Time':>8} {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, TRAIN_CFG["max_epochs"] + 1):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            TRAIN_CFG["clip_norm"], is_train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, device,
            TRAIN_CFG["clip_norm"], is_train=False
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} "
            f"{elapsed:>7.1f}s {lr_now:>10.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
        })

        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : val_loss,
                "train_cfg"  : TRAIN_CFG,
                "n_dynamic"  : n_dynamic,
                "n_static"   : n_static,
            }, best_ckpt)
            print(f"         ✓ saved best checkpoint (val={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= TRAIN_CFG["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {TRAIN_CFG['patience']} epochs)")
                break

    print(f"\nBest val NLL: {best_val_loss:.4f}")
    print(f"Checkpoint:   {best_ckpt}")

    # Save training history
    import json
    history_path = CKPT_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History:      {history_path}")

    return model, best_ckpt


if __name__ == "__main__":
    train()