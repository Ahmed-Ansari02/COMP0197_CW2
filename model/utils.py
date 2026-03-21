"""Training utilities: early stopping, checkpointing, logging, MC inference."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False  # not improved


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Training logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Accumulates per-epoch metrics and writes to CSV."""

    def __init__(self):
        self.rows: List[Dict] = []

    def log(self, epoch: int, train_loss: float, val_loss: float, lr: float,
            **extra):
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": lr}
        row.update(extra)
        self.rows.append(row)

    def save(self, path: str):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = self.rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def _enable_mc_dropout(model: nn.Module):
    """Turn on dropout layers while keeping everything else in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    loader: DataLoader,
    n_samples: int = 50,
    device: str = "cpu",
    baseline: bool = False,
) -> Dict[str, np.ndarray]:
    """Run T stochastic forward passes and aggregate distribution params.

    Returns dict with keys:
      hurdle mode  -> y_true, p_conflict_mean/std, mu_mean/std, sigma_mean/std, nu_mean/std
      baseline mode -> y_true, y_pred_mean/std
    """
    _enable_mc_dropout(model)

    all_y = []
    if baseline:
        all_preds = []
    else:
        all_conflict, all_mu, all_sigma, all_nu = [], [], [], []

    for x, y_count, y_binary in loader:
        x = x.to(device)
        B = x.size(0)

        if baseline:
            preds = torch.zeros(n_samples, B, device=device)
            for t in range(n_samples):
                preds[t] = model(x)
            all_preds.append(preds.cpu().numpy())
        else:
            conflict_samples = torch.zeros(n_samples, B, device=device)
            mu_samples = torch.zeros(n_samples, B, device=device)
            sigma_samples = torch.zeros(n_samples, B, device=device)
            nu_samples = torch.zeros(n_samples, B, device=device)
            for t in range(n_samples):
                cl, m, s, n = model(x)
                conflict_samples[t] = torch.sigmoid(cl)
                mu_samples[t] = m
                sigma_samples[t] = s
                nu_samples[t] = n
            all_conflict.append(conflict_samples.cpu().numpy())
            all_mu.append(mu_samples.cpu().numpy())
            all_sigma.append(sigma_samples.cpu().numpy())
            all_nu.append(nu_samples.cpu().numpy())

        all_y.append(y_count.numpy())

    y_true = np.concatenate(all_y)
    result = {"y_true": y_true}

    if baseline:
        preds = np.concatenate(all_preds, axis=1)  # (T, N)
        result["y_pred_mean"] = preds.mean(axis=0)
        result["y_pred_std"] = preds.std(axis=0)
    else:
        conflict = np.concatenate(all_conflict, axis=1)
        mu = np.concatenate(all_mu, axis=1)
        sigma = np.concatenate(all_sigma, axis=1)
        nu = np.concatenate(all_nu, axis=1)
        result["p_conflict_mean"] = conflict.mean(axis=0)
        result["p_conflict_std"] = conflict.std(axis=0)
        result["mu_mean"] = mu.mean(axis=0)
        result["mu_std"] = mu.std(axis=0)
        result["sigma_mean"] = sigma.mean(axis=0)
        result["sigma_std"] = sigma.std(axis=0)
        result["nu_mean"] = nu.mean(axis=0)
        result["nu_std"] = nu.std(axis=0)

    model.train()
    return result
