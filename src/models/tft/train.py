"""
Training pipeline for the simplified TFT.

Adapted for the merged panel data (country_iso3 + year_month format).
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm

from src.models.tft.model import ConflictForecaster
from src.models.common.loss import ConflictForecastLoss


# Columns that are identifiers/targets, not features
EXCLUDE_COLS = {
    "country_iso3", "year_month", "year", "month", "region",
    # UCDP target candidates
    "ucdp_fatalities_best", "ucdp_fatalities_high",
    "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
}


class ConflictDataset(Dataset):
    """
    Produces windowed sequences per country for temporal modelling.
    Each sample: (x_temporal[window_size, n_features], y).
    """

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

            feature_data = country_df[features].values.astype(np.float32)
            # Target is log1p'd from member A's pipeline — undo for ZILNM
            # (ZILNM log_prob expects raw counts: does its own log(y) internally)
            targets = np.expm1(country_df[target_col].values.astype(np.float32))

            for i in range(window_size, len(country_df)):
                x = feature_data[i - window_size:i]
                y = targets[i]

                # Skip if target is NaN
                if np.isnan(y):
                    continue

                self.samples.append((x, y))

                # Stratified sampling weights
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
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y),
        }


def get_feature_list(df: pd.DataFrame, target_col: str = "ucdp_fatalities_best") -> list[str]:
    """Auto-detect numeric feature columns, excluding identifiers and target."""
    exclude = EXCLUDE_COLS | {target_col}
    # Also exclude missingness indicators and availability flags
    exclude |= {c for c in df.columns if c.endswith("_missing") or c.endswith("_available") or c.endswith("_flag")}

    features = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]
    return sorted(features)


def create_dataloaders(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "ucdp_fatalities_best",
    window_size: int = 24,
    batch_size: int = 64,
    train_end: str = "2024-03",
    val_end: str = "2024-06",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with stratified sampling."""
    train_df = df[df["year_month"] <= train_end]
    val_df = df[(df["year_month"] > train_end) & (df["year_month"] <= val_end)]
    test_df = df[df["year_month"] > val_end]

    print(f"Train: {len(train_df):,} rows ({train_df['year_month'].min()} to {train_df['year_month'].max()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['year_month'].min()} to {val_df['year_month'].max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['year_month'].min()} to {test_df['year_month'].max()})")

    # Build datasets (use full df up to split point for windowing)
    train_ds = ConflictDataset(df[df["year_month"] <= train_end], features, target_col, window_size)
    val_ds = ConflictDataset(df[df["year_month"] <= val_end], features, target_col, window_size)
    test_ds = ConflictDataset(df, features, target_col, window_size)

    # Stratified sampler for training
    weights = torch.tensor(train_ds.sample_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Samples — Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    return train_loader, val_loader, test_loader


def train_model(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "ucdp_fatalities_best",
    hidden_dim: int = 64,
    n_lstm_layers: int = 1,
    n_attention_heads: int = 2,
    n_mixture_components: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 100,
    window_size: int = 24,
    patience: int = 10,
    train_end: str = "2024-03",
    val_end: str = "2024-06",
    checkpoint_dir: str = "checkpoints",
) -> ConflictForecaster:
    """Full training pipeline."""
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if features is None:
        features = get_feature_list(df, target_col)
    print(f"Features: {len(features)}")

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        df, features, target_col, window_size, batch_size, train_end, val_end,
    )

    # Build model
    model = ConflictForecaster(
        n_features=len(features),
        hidden_dim=hidden_dim,
        n_lstm_layers=n_lstm_layers,
        n_attention_heads=n_attention_heads,
        n_mixture_components=n_mixture_components,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss + optimizer
    criterion = ConflictForecastLoss(spike_threshold=500, spike_loss_multiplier=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False):
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

        print(f"Epoch {epoch+1}: train={avg_train:.4f}, val={avg_val:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping + checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
            print(f"  Saved best model (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pt", weights_only=True))
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    data_path = Path("data/processed/merge/model_ready.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        features = get_feature_list(df)
        print(f"Loaded {df.shape}, {len(features)} features")
        model = train_model(df, features)
    else:
        print("Run src/data/merge_panel.py first")
