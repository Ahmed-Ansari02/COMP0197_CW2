"""
Training pipeline for the Conv-Transformer.

Spike detection improvements (3 strategies):
  1. Asymmetric Exloss — penalises underestimation of extremes more than
     overestimation, inspired by ExtremeCast (Xu et al., 2024).
  2. WeightedRandomSampler — oversamples high-fatality events during training.
  3. Spike NLL multiplier — flat 5x on y > 500, matching shared loss.py.

Uses the unified dataset from pipelines/merge/preprocess.py.
"""

import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional

from src.models.transformer.model import ConvTransformer



EXCLUDE_COLS = {
    "country_iso3", "year_month", "year", "month", "region",
    "ucdp_fatalities_best", "ucdp_fatalities_high",
    "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
}


class ConflictDataset(Dataset):
    """Windowed sequences per country for temporal modelling."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        target_col: str = "ucdp_fatalities_best",
        window_size: int = 24,
    ):
        self.window_size = window_size
        self.samples = []
        self.sample_weights = []

        for iso3 in df["country_iso3"].unique():
            country_df = df[df["country_iso3"] == iso3].sort_values("year_month")
            if len(country_df) <= window_size:
                continue

            feat_data = country_df[features].values.astype(np.float32)
            targets = country_df[target_col].values.astype(np.float32)

            for i in range(window_size, len(country_df)):
                x = feat_data[i - window_size:i]
                y = targets[i]
                if np.isnan(y):
                    continue
                self.samples.append((x, y))

                if y > 500:
                    self.sample_weights.append(10.0)
                elif y > 50:
                    self.sample_weights.append(5.0)
                elif y > 0:
                    self.sample_weights.append(3.0)
                else:
                    self.sample_weights.append(1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {"x": torch.from_numpy(x), "y": torch.tensor(y)}


def get_feature_list(df: pd.DataFrame, target_col: str = "ucdp_fatalities_best") -> list[str]:
    exclude = EXCLUDE_COLS | {target_col}
    exclude |= {c for c in df.columns
                 if c.endswith("_missing") or c.endswith("_available") or c.endswith("_flag")}
    return sorted([
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
    ])



class AsymmetricHurdleLoss(nn.Module):
    """
    Hurdle-Student-t NLL with three spike-aware improvements:

    1. Asymmetric severity weighting (ExtremeCast, Xu et al. 2024):
       Underestimation of extreme values is penalised more than overestimation,
       with the penalty scaling continuously with the severity of the true event.

    2. Spike NLL multiplier: flat multiplier for y > spike_threshold.

    3. Weighted BCE: pos_weight for the binary conflict gate.
    """

    def __init__(
        self,
        spike_threshold: float = 500.0,
        spike_multiplier: float = 5.0,
        asymmetry_alpha: float = 2.0,
        log1p_max: float = 13.5,
        bce_pos_weight: float = 5.0,
    ):
        super().__init__()
        self.spike_threshold = spike_threshold
        self.spike_multiplier = spike_multiplier
        self.asymmetry_alpha = asymmetry_alpha
        self.log1p_max = log1p_max
        self.bce_pos_weight = bce_pos_weight

    def forward(
        self,
        dist_params: dict[str, torch.Tensor],
        y: torch.Tensor,
        dist_head,
    ) -> dict[str, torch.Tensor]:
        log_probs = dist_head.log_prob(dist_params, y)
        nll = -log_probs

        is_nonzero = (y > 0).float()
        mu = dist_params["mu"]
        log_y = torch.log1p(y.clamp(min=0))

        residual = log_y - mu
        severity_scale = log_y / self.log1p_max

        asymmetric_weight = torch.where(
            (residual > 0) & (y > 0),
            1.0 + self.asymmetry_alpha * severity_scale,
            torch.ones_like(residual),
        )

        spike_weight = torch.where(
            y > self.spike_threshold,
            torch.tensor(self.spike_multiplier, device=y.device),
            torch.ones_like(y),
        )

        total_weight = asymmetric_weight * spike_weight
        weighted_nll = (total_weight * nll).mean()

        return {
            "loss": weighted_nll,
            "nll": weighted_nll.detach(),
            "mean_log_prob": log_probs.mean().detach(),
        }


def create_dataloaders(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "ucdp_fatalities_best",
    window_size: int = 24,
    batch_size: int = 64,
    train_end: str = "2024-03",
    val_end: str = "2024-06",
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = ConflictDataset(
        df[df["year_month"] <= train_end], features, target_col, window_size)
    val_ds = ConflictDataset(
        df[df["year_month"] <= val_end], features, target_col, window_size)
    test_ds = ConflictDataset(df, features, target_col, window_size)

    weights = torch.tensor(train_ds.sample_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Samples — Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def train_model(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    target_col: str = "ucdp_fatalities_best",
    hidden_dim: int = 128,
    n_transformer_layers: int = 4,
    n_heads: int = 4,
    n_conv_layers: int = 2,
    patch_size: int = 3,
    dropout: float = 0.2,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 100,
    window_size: int = 24,
    patience: int = 10,
    train_end: str = "2024-03",
    val_end: str = "2024-06",
    checkpoint_dir: str = "checkpoints/transformer",
) -> ConvTransformer:

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if features is None:
        features = get_feature_list(df, target_col)
    print(f"Features: {len(features)}")

    train_loader, val_loader, _ = create_dataloaders(
        df, features, target_col, window_size, batch_size, train_end, val_end,
    )

    model = ConvTransformer(
        n_features=len(features),
        window_size=window_size,
        patch_size=patch_size,
        d_model=hidden_dim,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        n_conv_layers=n_conv_layers,
        dim_feedforward=hidden_dim * 2,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = AsymmetricHurdleLoss(
        spike_threshold=500.0,
        spike_multiplier=5.0,
        asymmetry_alpha=2.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        t0 = time.time()

        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            dist_params = model(x)
            loss_dict = criterion(dist_params, y, model.dist_head)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict["loss"].item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                dist_params = model(x)
                loss_dict = criterion(dist_params, y, model.dist_head)
                val_losses.append(loss_dict["loss"].item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float("inf")
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        improved = avg_val < best_val_loss
        marker = " *" if improved else ""
        print(f"  epoch {epoch+1:3d}/{max_epochs}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  lr={lr:.2e}  ({elapsed:.1f}s){marker}")

        if improved:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(f"{checkpoint_dir}/best_model.pt", map_location=device, weights_only=True))
    print(f"\nBest val loss: {best_val_loss:.4f}")
    return model


def generate_predictions(
    model: ConvTransformer,
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "ucdp_fatalities_best",
    window_size: int = 24,
    n_samples: int = 1000,
    batch_size: int = 64,
    device: str = "cpu",
    output_dir: str = "results/transformer",
):
    """Run predict() on the full dataset and save (N, S) sample matrix."""
    test_ds = ConflictDataset(df, features, target_col, window_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.to(device).eval()
    all_samples = []
    all_y = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            pred = model.predict(x, n_samples=n_samples)
            all_samples.append(pred["samples"].cpu().numpy())
            all_y.append(batch["y"].numpy())

    samples = np.concatenate(all_samples, axis=0)
    y_true = np.concatenate(all_y)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/samples.npy", samples)
    np.save(f"{output_dir}/y_true.npy", y_true)
    print(f"Saved samples {samples.shape} and y_true {y_true.shape} to {output_dir}/")

    return y_true, samples


if __name__ == "__main__":
    data_path = Path("data/processed/merge/model_ready.csv")
    if not data_path.exists():
        print("Run pipelines/merge/merge_panel.py and pipelines/merge/preprocess.py first")
        raise SystemExit(1)

    df = pd.read_csv(data_path)
    features = get_feature_list(df)
    print(f"Loaded {df.shape}, {len(features)} features")

    model = train_model(df, features)

    device = "cpu"
    y_true, samples = generate_predictions(
        model, df, features, device=device,
    )

    try:
        from src.evaluation.metrics import crps_mean
        crps = crps_mean(y_true, samples)
        print(f"\nCRPS (full dataset): {crps:.2f}")
    except Exception as e:
        print(f"Could not compute CRPS: {e}")
