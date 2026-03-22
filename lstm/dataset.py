"""
dataset.py — Conflict fatality prediction: Bayesian LSTM
Loads model_ready.csv, builds per-country sliding windows.
Separates static (slow-moving) and dynamic (monthly) features.

>python lstm/dataset.py data/processed/merge/model_ready.csv
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Feature groups ────────────────────────────────────────────────────────────

# Static: country-level, slow-moving — concatenated to LSTM hidden state
# NOT fed as sequence steps
STATIC_FEATURES = [
    "v2x_polyarchy", "v2x_libdem", "v2x_freexp_altinf",
    "v2x_frassoc_thick", "v2x_clphy", "v2x_execorr",
    "v2x_rule", "v2x_jucon", "v2x_legcon",
    "reign_democracy", "reign_personal", "reign_military",
    "reign_party", "reign_monarch",
]

# Target
TARGET = "ucdp_fatalities_best"

# Identifiers — never fed to model
ID_COLS = ["country_iso3", "year_month"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ConflictDataset(Dataset):
    """
    Sliding window dataset over country-month panel.

    For each country, slides a window of length `seq_len` months.
    Returns:
        dynamic  : [seq_len, n_dynamic]  float32
        static   : [n_static]            float32  (last timestep values)
        target   : scalar                float32  (fatalities at t+1, log1p scale)
        mask     : scalar                bool     (True if target is observed)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 24,
        split: str = "train",      # "train" | "val" | "test"
        train_end: str = "2024-03",
        val_start: str = "2024-04",
        val_end: str = "2024-06",
        test_start: str = "2024-07",
    ):
        self.seq_len = seq_len

        # ── Split filtering ───────────────────────────────────────────────────────
        if split == "train":
            df = df[df["year_month"] <= train_end].copy()
        elif split == "val":
            # keep seq_len months of context before val_start so windows can form
            all_months = sorted(df["year_month"].unique())
            val_start_idx = all_months.index(val_start)
            context_start = all_months[max(0, val_start_idx - seq_len)]
            df = df[
                (df["year_month"] >= context_start) &
                (df["year_month"] <= val_end)
            ].copy()
        elif split == "test":
            all_months = sorted(df["year_month"].unique())
            test_start_idx = all_months.index(test_start)
            context_start = all_months[max(0, test_start_idx - seq_len)]
            df = df[df["year_month"] >= context_start].copy()
        else:
            raise ValueError(f"split must be train/val/test, got {split}")
        

        # ── Identify dynamic features ─────────────────────────────────────────
        exclude = set(ID_COLS) | {TARGET} | set(STATIC_FEATURES)
        self.dynamic_cols = sorted([
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ])
        self.static_cols = [c for c in STATIC_FEATURES if c in df.columns]

        self.n_dynamic = len(self.dynamic_cols)
        self.n_static = len(self.static_cols)

        # ── Build windows per country ─────────────────────────────────────────
        self.windows = []  # list of (dynamic, static, target, mask)

        for country, grp in df.groupby("country_iso3"):
            grp = grp.sort_values("year_month").reset_index(drop=True)

            dyn = grp[self.dynamic_cols].values.astype(np.float32)   # [T, n_dyn]
            sta = grp[self.static_cols].values.astype(np.float32)    # [T, n_sta]
            tgt = grp[TARGET].values.astype(np.float32)              # [T]

            T = len(grp)
            # Need seq_len steps as input + 1 step as target
            if T < seq_len + 1:
                continue

            for t in range(seq_len, T):
                dyn_window = dyn[t - seq_len : t]        # [seq_len, n_dyn]
                sta_window = sta[t - 1]                  # last timestep static
                target_val = tgt[t]
                observed   = not np.isnan(target_val)

                # Replace NaN target with 0 for loss masking
                if not observed:
                    target_val = 0.0

                self.windows.append((
                    torch.tensor(dyn_window, dtype=torch.float32),
                    torch.tensor(sta_window, dtype=torch.float32),
                    torch.tensor(target_val, dtype=torch.float32),
                    torch.tensor(observed,   dtype=torch.bool),
                ))

        print(
            f"[{split}] {len(self.windows):,} windows | "
            f"{self.n_dynamic} dynamic + {self.n_static} static features"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# ── Convenience loaders ───────────────────────────────────────────────────────

def get_dataloaders(
    data_path: str | Path,
    seq_len: int = 24,
    batch_size: int = 256,
    train_end: str = "2024-03",
    val_start: str = "2024-04",
    val_end: str = "2024-06",
    test_start: str = "2024-07",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    df = pd.read_csv(data_path)
    print(f"Loaded {data_path}: {df.shape}")

    kwargs = dict(
        seq_len=seq_len,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
    )

    train_ds = ConflictDataset(df, split="train", **kwargs)
    val_ds   = ConflictDataset(df, split="val",   **kwargs)
    test_ds  = ConflictDataset(df, split="test",  **kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/merge/model_ready.csv"
    train_loader, val_loader, test_loader = get_dataloaders(path)

    dyn, sta, tgt, mask = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  dynamic : {dyn.shape}")   # [B, seq_len, n_dynamic]
    print(f"  static  : {sta.shape}")   # [B, n_static]
    print(f"  target  : {tgt.shape}")   # [B]
    print(f"  mask    : {mask.shape}")  # [B]
    print(f"  observed in batch: {mask.sum().item()}/{len(mask)}")