"""
train.py — Data pipeline + Conv-Transformer training for conflict fatality forecasting.

Pipeline stages (each skips if output CSV already exists):
  1. Load Member A: UCDP + GDELT conflict events
  2. Load Member B: governance, economic, and regime features
  3. Load Member C: geopolitical risk, GDELT tone, macro indicators
  4. Merge: outer join A+B+C, broadcast globals, backfill GDELT tone
  5. Add autoregressive features (lagged target statistics)
  6. Preprocess: log1p, drop redundant, z-score standardise, clip outliers
  7. Train Conv-Transformer with Hurdle-Student-t distribution head

Additional packages (beyond base env): pandas, matplotlib, scipy
GenAI disclosure: LLMs for brainstorming.
"""

import argparse
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")

# --- Stage 1: Member A (conflict) ---

def load_member_a():
    """Load conflict dataset (UCDP + ACLED + GDELT events)."""
    path = DATA_DIR / "member_a" / "member_a_final.csv"
    if path.exists():
        print(f"[Pipeline] Member A: loading {path}")
        return pd.read_csv(path)
    raise FileNotFoundError(
        f"Member A data not found at {path}. "
        "Regeneration requires ACLED credentials and BigQuery access. "
        "Include the pre-built CSV."
    )


# --- Stage 2: Member B (structural/governance) ---

def load_member_b():
    """Load structural dataset (V-Dem, REIGN, FX, GDP, food CPI, coups)."""
    path = DATA_DIR / "member_b" / "member_b_final.csv"
    if path.exists():
        print(f"[Pipeline] Member B: loading {path}")
        return pd.read_csv(path)
    raise FileNotFoundError(
        f"Member B data not found at {path}. "
        "Regeneration requires V-Dem, IMF, and FAO raw files. "
        "Include the pre-built CSV."
    )


# --- Stage 3: Member C (volatility/tone) ---

def load_member_c():
    """Load volatility dataset (GPR, GDELT tone, macro indicators)."""
    path = DATA_DIR / "member_c" / "member_c_final.csv"
    if path.exists():
        print(f"[Pipeline] Member C: loading {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        return df
    raise FileNotFoundError(
        f"Member C data not found at {path}. "
        "Regeneration requires BigQuery access and GPR Excel files. "
        "Include the pre-built CSV."
    )


# --- Stage 4: Merge ---

# Columns from Member C that are country-invariant (one value per month)
GLOBAL_COLUMNS = [
    "gpr_global", "gpr_acts",
    "vix_mean", "vix_vol", "vix_pct_chg",
    "wti_oil_mean", "wti_oil_vol", "wti_oil_pct_chg",
    "gold_mean", "gold_vol", "gold_pct_chg",
    "dxy_mean", "dxy_vol", "dxy_pct_chg",
    "us_10y_yield_mean", "us_10y_yield_vol", "us_10y_yield_pct_chg",
    "wheat_mean", "wheat_vol", "wheat_pct_chg",
    "copper_mean", "copper_vol", "copper_pct_chg",
    "us_13w_tbill_mean", "us_13w_tbill_vol", "us_13w_tbill_pct_chg",
]

# only 44 countries
DROP_FROM_C = ["gpr_country"]  

def merge_panels(a, b, c):
    """Outer join A+B+C on (country_iso3, year_month)."""
    keys = ["country_iso3", "year_month"]

    for df in [a, b, c]:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

    c = c.drop(columns=[col for col in DROP_FROM_C if col in c.columns], errors="ignore")

    merged = pd.merge(a, b, on=keys, how="outer", suffixes=("", "_dup_b"))
    merged = pd.merge(merged, c, on=keys, how="outer", suffixes=("", "_dup_c"))

    dup_cols = [c for c in merged.columns if "_dup_b" in c or "_dup_c" in c]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    merged = merged.sort_values(keys).reset_index(drop=True)
    print(f"[Merge] Merged panel: {merged.shape}")
    return merged


def broadcast_globals(df):
    """Fill global (country-invariant) columns across all countries."""
    present = [col for col in GLOBAL_COLUMNS if col in df.columns]
    if not present:
        return df
    monthly = df.groupby("year_month")[present].first().reset_index()
    df = df.drop(columns=present)
    df = df.merge(monthly, on="year_month", how="left")
    print(f"[Merge] Broadcast {len(present)} global columns")
    return df


def backfill_gdelt_tone(df):
    """Fill GDELT tone gaps using full-country backfill file."""
    tone_path = DATA_DIR / "merge" / "gdelt_tone_all.csv"
    if not tone_path.exists():
        print("[Merge] No gdelt_tone_all.csv — skipping tone backfill")
        return df

    tone_full = pd.read_csv(tone_path)
    tone_cols = ["tone_mean", "tone_min", "tone_max", "tone_std",
                 "event_count", "goldstein_mean"]
    tone_cols = [c for c in tone_cols if c in df.columns and c in tone_full.columns]
    if not tone_cols:
        return df

    before = df[tone_cols].isnull().sum().sum()
    lookup = tone_full.set_index(["country_iso3", "year_month"])[tone_cols]
    idx = df.set_index(["country_iso3", "year_month"])
    for col in tone_cols:
        mask = idx[col].isna()
        fill = lookup[col].reindex(idx.index)
        idx.loc[mask, col] = fill[mask]
    df[tone_cols] = idx[tone_cols].values
    after = df[tone_cols].isnull().sum().sum()
    print(f"[Merge] Backfilled {before - after:,} GDELT tone values")
    return df


def add_temporal_columns(df):
    df["year"] = df["year_month"].str[:4].astype(int)
    df["month"] = df["year_month"].str[5:7].astype(int)
    return df


# --- Stage 5: Autoregressive features ---

TARGET = "ucdp_fatalities_best"
AR_WINDOWS = [3, 6, 12]


def add_lag_features(df):
    """Add per-country lagged target statistics."""
    df = df.sort_values(["country_iso3", "year_month"]).copy()
    print(f"[AR] Adding autoregressive features")

    for iso3, group in df.groupby("country_iso3"):
        idx = group.index
        y = group[TARGET].values
        s = pd.Series(y, index=idx).shift(1)

        df.loc[idx, "target_lag1"] = s.values
        df.loc[idx, "target_lag2"] = pd.Series(y, index=idx).shift(2).values
        df.loc[idx, "target_lag3"] = pd.Series(y, index=idx).shift(3).values
        df.loc[idx, "target_diff1"] = (s - pd.Series(y, index=idx).shift(2)).values

        for w in AR_WINDOWS:
            roll = s.rolling(window=w, min_periods=1)
            df.loc[idx, f"target_roll{w}_mean"] = roll.mean().values
            df.loc[idx, f"target_roll{w}_median"] = roll.median().values
            df.loc[idx, f"target_roll{w}_std"] = roll.std().values
            df.loc[idx, f"target_roll{w}_max"] = roll.max().values
            rmean, rstd = roll.mean(), roll.std()
            df.loc[idx, f"target_roll{w}_zscore"] = (
                (s - rmean) / rstd.replace(0, np.nan)
            ).values

        spike_thresh = np.log1p(500)
        lagged = pd.Series(y, index=idx).shift(1)
        months_since = pd.Series(np.nan, index=idx)
        counter = np.nan
        for i, (ix, spike) in enumerate((lagged > spike_thresh).items()):
            if spike:
                counter = 0
            elif not np.isnan(counter):
                counter += 1
            months_since.iloc[i] = counter
        df.loc[idx, "target_months_since_spike"] = months_since.values

    ar_cols = [c for c in df.columns if c.startswith("target_")]
    print(f"[AR] Added {len(ar_cols)} features")
    return df


# --- Stage 6: Preprocess ---

ALREADY_LOG1P = {
    "ucdp_event_count", "ucdp_fatalities_best", "ucdp_fatalities_high",
    "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
    "ucdp_fatality_uncertainty", "gdelt_conflict_event_count",
    "gdelt_goldstein_mean",
    "acled_fatalities", "acled_event_count", "acled_peak_fatalities",
    "acled_battle_count", "acled_explosion_count", "acled_violence_count",
    "acled_protest_count", "acled_riot_count", "acled_airstrike_count",
    "acled_armed_clash_count", "acled_political_violence_count",
    "acled_demonstration_count",
    "gpr_global", "gpr_acts", "event_count", "tone_std",
}

PREPROCESS_DROP = {
    "country_iso3", "year_month", "year", "month", "region",
    "vdem_stale_flag", "vdem_available", "reign_available",
    "fx_available", "food_available", "gdp_available",
    "coup_event",
}

KNOWN_REDUNDANT = {
    "fx_volatility_log", "food_price_anomaly", "pt_coup_event",
    "governance_deficit", "repression_index", "ucdp_peak_event_fatalities",
    "v2x_civlib", "v2x_clpol", "v2xnp_regcorr", "v2x_corr",
}


def preprocess(df, train_end="2024-03"):
    """Log1p, drop redundant, z-score standardise, clip, fill NaN."""
    ids = df[["country_iso3", "year_month"]].copy()

    feature_cols = sorted([
        c for c in df.columns
        if c not in PREPROCESS_DROP and c != TARGET
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ])

    features = df[feature_cols].copy()
    target = df[TARGET].copy()

    # Log1p heavy-tailed features
    log1p_cols = []
    for col in feature_cols:
        if col in ALREADY_LOG1P:
            continue
        vals = features[col].dropna()
        if len(vals) > 0 and vals.min() >= 0 and vals.max() > 100 and vals.skew() > 3:
            log1p_cols.append(col)
    for col in log1p_cols:
        features[col] = np.log1p(features[col])
    print(f"[Preprocess] Log1p: {len(log1p_cols)} features")

    # Drop known redundant
    redundant = [c for c in feature_cols if c in KNOWN_REDUNDANT]
    feature_cols = [c for c in feature_cols if c not in KNOWN_REDUNDANT]
    features = features[feature_cols]
    print(f"[Preprocess] Dropped {len(redundant)} redundant features")

    # Z-score using training period only
    train_mask = df["year_month"] <= train_end
    train_means = features[train_mask].mean()
    train_stds = features[train_mask].std()
    constant = train_stds[train_stds < 1e-10].index.tolist()
    if constant:
        features = features.drop(columns=constant)
        feature_cols = [c for c in feature_cols if c not in constant]
        train_means = train_means.drop(constant)
        train_stds = train_stds.drop(constant)
        print(f"[Preprocess] Dropped {len(constant)} constant features")

    features = (features - train_means) / train_stds

    # Clip outliers
    n_clipped = (features.abs() > 10).sum().sum()
    features = features.clip(-10, 10)
    print(f"[Preprocess] Clipped {n_clipped:,} values beyond +/-10 std")

    # Fill NaN with 0 (= population mean post-standardisation)
    n_nan = features.isnull().sum().sum()
    features = features.fillna(0)
    print(f"[Preprocess] Filled {n_nan:,} NaN with 0")

    result = pd.concat([ids, features, target.rename(TARGET)], axis=1)
    print(f"[Preprocess] Final: {result.shape}, {len(feature_cols)} features")
    return result


def run_pipeline():
    """Execute full data pipeline, skipping stages with existing outputs."""
    model_ready_path = DATA_DIR / "merge" / "model_ready.csv"

    if model_ready_path.exists():
        print(f"[Pipeline] model_ready.csv found — skipping all stages")
        return pd.read_csv(model_ready_path)

    # Load member datasets
    a = load_member_a()
    b = load_member_b()
    c = load_member_c()

    # Merge
    merged = merge_panels(a, b, c)
    merged = broadcast_globals(merged)
    merged = backfill_gdelt_tone(merged)
    merged = add_temporal_columns(merged)

    # Add autoregressive features
    # merged = add_lag_features(merged)

    # Preprocess
    result = preprocess(merged)

    # Save
    model_ready_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(model_ready_path, index=False)
    print(f"[Pipeline] Saved {model_ready_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = F.pad(x, (self.pad, 0))
        out = self.drop(F.gelu(self.bn(self.conv(out))))
        return out + res


class CausalConvFrontEnd(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_features if i == 0 else hidden_dim
            layers.append(CausalConvBlock(in_ch, hidden_dim, kernel_size, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_dim, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        n_patches = T // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * D)
        return self.norm(self.proj(x))


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_head, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x):
        T = x.size(2)
        if T > self.cos_cached.size(0):
            self._build_cache(T)
        cos, sin = self.cos_cached[:T], self.sin_cached[:T]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(self.d_head)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HurdleStudentT(nn.Module):
    """Two-part hurdle: P(y>0) gate + Student-t severity in log1p space."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2))
        self.conflict_head = nn.Linear(hidden_dim, 1)
        self.severity_head = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            self.conflict_head.bias.fill_(-1.5)
            self.severity_head.bias.copy_(torch.tensor([3.0, 0.0, 0.0]))

    def forward(self, h):
        h = self.shared(h)
        logit = self.conflict_head(h).squeeze(-1)
        sev = self.severity_head(h)
        mu = sev[:, 0]
        sigma = F.softplus(sev[:, 1]) + 1e-4
        nu = F.softplus(sev[:, 2]) + 2.0
        return {"conflict_logit": logit, "mu": mu, "sigma": sigma, "nu": nu}

    def log_prob(self, params, y):
        logit, mu, sigma, nu = params["conflict_logit"], params["mu"], params["sigma"], params["nu"]
        is_zero = (y == 0).float()
        log_p_zero = F.logsigmoid(-logit)
        log_p_pos_gate = F.logsigmoid(logit)
        log_y = torch.log1p(y.clamp(min=0))
        z = (log_y - mu) / sigma
        log_p_student = (
            torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
            - ((nu + 1) / 2) * torch.log1p(z * z / nu)
        )
        return is_zero * log_p_zero + (1 - is_zero) * (log_p_pos_gate + log_p_student)

    def sample(self, params, n_samples=1000):
        logit, mu, sigma, nu = params["conflict_logit"], params["mu"], params["sigma"], params["nu"]
        B = logit.shape[0]
        dev = mu.device
        is_nonzero = torch.bernoulli(
            torch.sigmoid(logit).unsqueeze(-1).expand(B, n_samples))
        # Chi2 sampling uses _standard_gamma which is not supported on MPS
        nu_exp = nu.unsqueeze(-1).expand(B, n_samples)
        chi2 = torch.distributions.Chi2(nu_exp.cpu())
        chi2_samples = chi2.sample().to(dev)
        z = torch.randn(B, n_samples, device=dev)
        t_samples = z / torch.sqrt(chi2_samples / nu.unsqueeze(-1))
        log_y = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * t_samples
        raw = torch.expm1(log_y.clamp(max=15)).clamp(min=0)
        return torch.round(is_nonzero * raw)


class ConvTransformer(nn.Module):
    """Causal-Conv + Patch-Transformer with Hurdle-Student-t output."""

    def __init__(self, n_features, window_size=24, patch_size=3, d_model=128,
                 n_heads=4, n_transformer_layers=4, n_conv_layers=2,
                 conv_kernel_size=3, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.conv_frontend = CausalConvFrontEnd(
            n_features, d_model, n_conv_layers, conv_kernel_size, dropout)
        self.patch_embed = PatchEmbedding(patch_size, d_model, d_model)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.dist_head = HurdleStudentT(d_model)

    def encode(self, x):
        h = self.conv_frontend(x)
        h = self.patch_embed(h)
        h = self.transformer(h)
        return self.final_norm(h)[:, -1, :]

    def forward(self, x):
        return self.dist_head(self.encode(x))

    def predict(self, x, n_samples=1000):
        self.eval()
        with torch.no_grad():
            params = self.forward(x)
            samples = self.dist_head.sample(params, n_samples)
            return {
                "samples": samples,
                "mean": samples.mean(dim=1),
                "median": samples.median(dim=1).values,
                "p_conflict": torch.sigmoid(params["conflict_logit"]),
            }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAINING
# ═══════════════════════════════════════════════════════════════════════════

EXCLUDE_COLS = {
    "country_iso3", "year_month", "year", "month", "region",
    "ucdp_fatalities_best", "ucdp_fatalities_high",
    "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
}


def get_feature_list(df, target_col="ucdp_fatalities_best"):
    """Auto-detect numeric feature columns."""
    exclude = EXCLUDE_COLS | {target_col}
    exclude |= {c for c in df.columns
                 if c.endswith("_missing") or c.endswith("_available") or c.endswith("_flag")}
    return sorted([
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
    ])


class ConflictDataset(Dataset):
    """Windowed sequences per country. Targets are RAW counts (expm1 applied)."""

    def __init__(self, df, features, target_col="ucdp_fatalities_best", window_size=24):
        self.samples = []
        self.sample_weights = []
        for iso3 in df["country_iso3"].unique():
            cdf = df[df["country_iso3"] == iso3].sort_values("year_month")
            if len(cdf) <= window_size:
                continue
            feat = cdf[features].values.astype(np.float32)
            targets = np.expm1(cdf[target_col].values.astype(np.float32))
            for i in range(window_size, len(cdf)):
                y = targets[i]
                if np.isnan(y):
                    continue
                self.samples.append((feat[i - window_size:i], y))
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
        return {"x": torch.from_numpy(x), "y": torch.tensor(y, dtype=torch.float32)}


class HurdleLoss(nn.Module):
    """BCE (conflict gate) + Student-t NLL (severity)."""

    def __init__(self, pos_weight=5.1):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, dist_params, y, dist_head):
        logit = dist_params["conflict_logit"]
        mu, sigma, nu = dist_params["mu"], dist_params["sigma"], dist_params["nu"]
        y_binary = (y > 0).float()

        pw = torch.tensor([self.pos_weight], device=logit.device, dtype=logit.dtype)
        bce = F.binary_cross_entropy_with_logits(logit, y_binary, pos_weight=pw)

        mask = y > 0.5
        if mask.any():
            log_y = torch.log1p(y[mask])
            z = (log_y - mu[mask]) / sigma[mask]
            nll = -(
                torch.lgamma((nu[mask] + 1) / 2) - torch.lgamma(nu[mask] / 2)
                - 0.5 * torch.log(nu[mask] * math.pi) - torch.log(sigma[mask])
                - ((nu[mask] + 1) / 2) * torch.log1p(z * z / nu[mask])
            ).mean()
        else:
            nll = torch.tensor(0.0, device=logit.device)

        return {"loss": bce + nll, "bce": bce.detach(), "nll": nll.detach()}


def train_model(df, features=None, window_size=24, patch_size=3, hidden_dim=128,
                n_heads=4, n_layers=4, n_conv_layers=2, dropout=0.2,
                lr=5e-4, weight_decay=1e-4, batch_size=64, max_epochs=100,
                patience=15, train_end="2024-03", val_end="2024-06",
                checkpoint_path="best_model.pt"):
    """Train Conv-Transformer and save best checkpoint."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu")
    print(f"Device: {device}")

    if features is None:
        features = get_feature_list(df)
    print(f"Features: {len(features)}")

    # Build datasets
    train_ds = ConflictDataset(
        df[df["year_month"] <= train_end], features, window_size=window_size)
    val_ds = ConflictDataset(
        df[df["year_month"] <= val_end], features, window_size=window_size)

    weights = torch.tensor(train_ds.sample_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Samples — Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # Model
    model = ConvTransformer(
        n_features=len(features), window_size=window_size, patch_size=patch_size,
        d_model=hidden_dim, n_heads=n_heads, n_transformer_layers=n_layers,
        n_conv_layers=n_conv_layers, dim_feedforward=hidden_dim * 2, dropout=dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = HurdleLoss(pos_weight=5.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        t0 = time.time()

        model.train()
        train_losses = []
        for batch in train_loader:
            x, yb = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            loss_dict = criterion(model(x), yb, model.dist_head)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict["loss"].item())

        scheduler.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, yb = batch["x"].to(device), batch["y"].to(device)
                loss_dict = criterion(model(x), yb, model.dist_head)
                val_losses.append(loss_dict["loss"].item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float("inf")
        elapsed = time.time() - t0
        improved = avg_val < best_val_loss
        marker = " *" if improved else ""

        print(f"  epoch {epoch+1:3d}/{max_epochs}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.1f}s){marker}")

        if improved:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"\nBest val loss: {best_val_loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Data pipeline + Conv-Transformer training")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run data pipeline
    df = run_pipeline()

    # Train
    features = get_feature_list(df)
    print(f"\nLoaded {df.shape}, {len(features)} features")

    model = train_model(
        df, features,
        window_size=args.window_size,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
    )

    print(f"\nCheckpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
