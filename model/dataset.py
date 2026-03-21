"""Panel dataset for fatality prediction.

Merges member A/B/C processed CSVs, aligns the raw UCDP target,
builds per-country sliding windows, and provides train/val/test splits.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from configs.default import Config


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_member_a(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df["year_month"] = df["year_month"].astype(str).str.strip()
    return df


def _load_member_b(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df["year_month"] = df["year_month"].astype(str).str.strip()
    return df


def _load_member_c(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df["year_month"] = df["year_month"].astype(str).str.strip()
    # Many columns arrive as strings due to whitespace — coerce to numeric
    for col in df.columns:
        if col not in ("country_iso3", "year_month"):
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df


def _load_target(path: str) -> pd.DataFrame:
    """Load raw (un-lagged, un-transformed) UCDP fatality counts."""
    df = pd.read_csv(path)
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip()
    df["year_month"] = df["year_month"].astype(str).str.strip()
    return df[["country_iso3", "year_month", "ucdp_fatalities_best"]]


# ---------------------------------------------------------------------------
# Merge & build panel
# ---------------------------------------------------------------------------

def build_panel(cfg: Config, feature_group: str) -> pd.DataFrame:
    """Load, merge, and return the full panel with features + raw target.

    Parameters
    ----------
    cfg : Config
    feature_group : one of 'a', 'ab', 'abc'

    Returns
    -------
    pd.DataFrame  with columns: country_iso3, year_month, <features>, target
    """
    feat_cols = cfg.feature_columns(feature_group)

    # --- Load member data ------------------------------------------------
    df_a = _load_member_a(cfg.data.member_a_path)
    panel = df_a[["country_iso3", "year_month"] +
                 [c for c in cfg.data.member_a_features if c in df_a.columns]].copy()

    if feature_group in ("ab", "abc"):
        df_b = _load_member_b(cfg.data.member_b_path)
        b_cols = [c for c in cfg.data.member_b_features if c in df_b.columns]
        panel = panel.merge(
            df_b[["country_iso3", "year_month"] + b_cols],
            on=["country_iso3", "year_month"],
            how="left",
        )

    if feature_group == "abc":
        df_c = _load_member_c(cfg.data.member_c_path)
        c_cols = [c for c in cfg.data.member_c_features if c in df_c.columns]
        panel = panel.merge(
            df_c[["country_iso3", "year_month"] + c_cols],
            on=["country_iso3", "year_month"],
            how="inner",  # restricts to C's 43 countries
        )

    # --- Attach raw target ------------------------------------------------
    target_df = _load_target(cfg.data.target_path)
    target_df = target_df.rename(columns={"ucdp_fatalities_best": "target"})
    panel = panel.merge(target_df, on=["country_iso3", "year_month"], how="inner")

    # --- Keep only columns that exist after merge -------------------------
    present_feat_cols = [c for c in feat_cols if c in panel.columns]
    panel = panel[["country_iso3", "year_month"] + present_feat_cols + ["target"]]

    # --- Sort for windowing -----------------------------------------------
    panel = panel.sort_values(["country_iso3", "year_month"]).reset_index(drop=True)

    return panel, present_feat_cols


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

_BINARY_COLS = {
    "ucdp_has_conflict", "regime_type_0", "regime_type_1", "regime_type_2",
    "regime_type_3", "male", "militarycareer", "elected", "leader_age_risk",
    "regime_change", "coup_event", "prev_conflict", "fx_depreciation_flag",
    "gdp_negative_shock", "food_price_spike", "pt_coup_successful",
    "pt_coup_failed", "pt_coup_event", "vdem_stale_flag", "vdem_available",
    "reign_available", "fx_available", "food_available", "gdp_available",
}

_BINARY_COLS.update({f"reign_regime_{s}" for s in [
    "Dominant Party", "Foreign/Occupied", "Indirect Military", "Military",
    "Military-Personal", "Monarchy", "Oligarchy", "Parliamentary Democracy",
    "Party-Military", "Party-Personal", "Party-Personal-Military Hybrid",
    "Personal Dictatorship", "Presidential Democracy",
    "Provisional - Civilian", "Provisional - Military", "Warlordism",
]})


def impute_and_standardise(
    panel: pd.DataFrame,
    feat_cols: List[str],
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impute NaNs, standardise features using train-set stats.

    Returns feature array (N, F), target array (N,), binary-target array (N,).
    """
    panel = panel.copy()

    # Forward-fill within country, then fill residual
    panel[feat_cols] = panel.groupby("country_iso3")[feat_cols].ffill()
    for col in feat_cols:
        if col in _BINARY_COLS:
            panel[col] = panel[col].fillna(0.0)
        else:
            train_median = panel.loc[train_mask, col].median()
            fill_val = train_median if pd.notna(train_median) else 0.0
            panel[col] = panel[col].fillna(fill_val)

    X = panel[feat_cols].values.astype(np.float32)
    y = panel["target"].values.astype(np.float32)

    # Standardise using train-set statistics
    train_X = X[train_mask]
    mu = train_X.mean(axis=0)
    sigma = train_X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0  # avoid division by zero for constant cols
    X = (X - mu) / sigma

    y_binary = (y > 0).astype(np.float32)
    return X, y, y_binary


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

class PanelDataset(Dataset):
    """PyTorch dataset that yields (window_features, raw_count, binary_label)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_binary: np.ndarray,
        country_ids: np.ndarray,
        window_size: int,
    ):
        self.window_size = window_size
        self.samples: List[Tuple[int, int]] = []  # (start_row, end_row)

        # Build per-country contiguous windows
        unique_countries = np.unique(country_ids)
        for cid in unique_countries:
            idxs = np.where(country_ids == cid)[0]
            if len(idxs) < window_size + 1:
                continue
            for i in range(len(idxs) - window_size):
                window_start = idxs[i]
                window_end = idxs[i + window_size - 1]
                target_idx = idxs[i + window_size]
                self.samples.append((window_start, target_idx))

        self.X = X
        self.y = y
        self.y_binary = y_binary

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window_start, target_idx = self.samples[idx]
        x = self.X[window_start: window_start + self.window_size]
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[target_idx], dtype=torch.float32),
            torch.tensor(self.y_binary[target_idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: Config,
    feature_group: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict]:
    """Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, n_features, meta
        meta contains country_iso3 and year_month arrays for the full panel
        so predictions can be matched back.
    """
    panel, feat_cols = build_panel(cfg, feature_group)
    n_features = len(feat_cols)

    ym = panel["year_month"].values
    train_mask = ym <= cfg.data.train_end
    val_mask = (ym > cfg.data.train_end) & (ym <= cfg.data.val_end)
    test_mask = ym > cfg.data.val_end

    X, y, y_binary = impute_and_standardise(panel, feat_cols, train_mask)

    country_ids = panel["country_iso3"].values

    # Create per-split datasets
    # For windowing we need contiguous country blocks, so we pass the full
    # arrays but restrict which *target* months belong to each split.
    # The window can extend into earlier splits (e.g. val window uses train data).
    all_ds = PanelDataset(X, y, y_binary, country_ids, cfg.model.window_size)

    # Partition samples by which split the *target* month falls into
    train_indices, val_indices, test_indices = [], [], []
    for i, (ws, ti) in enumerate(all_ds.samples):
        t_ym = ym[ti]
        if t_ym <= cfg.data.train_end:
            train_indices.append(i)
        elif t_ym <= cfg.data.val_end:
            val_indices.append(i)
        else:
            test_indices.append(i)

    train_ds = torch.utils.data.Subset(all_ds, train_indices)
    val_ds = torch.utils.data.Subset(all_ds, val_indices)
    test_ds = torch.utils.data.Subset(all_ds, test_indices)

    use_pin = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=cfg.train.batch_size,
        pin_memory=use_pin,
        num_workers=cfg.train.num_workers,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Metadata for mapping predictions back to (country, month)
    meta = {
        "country_iso3": country_ids,
        "year_month": ym,
        "samples": all_ds.samples,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    print(f"[DATA] features={n_features}  group={feature_group}")
    print(f"[DATA] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    return train_loader, val_loader, test_loader, n_features, meta
