from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    window_size: int = 36
    patch_size: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_transformer_layers: int = 4
    n_conv_layers: int = 2
    conv_kernel_size: int = 3
    dropout: float = 0.2
    dim_feedforward: int = 256


@dataclass
class TrainConfig:
    lr: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    grad_clip_norm: float = 1.0
    mc_samples: int = 50
    pos_weight: float = 5.1
    num_workers: int = 0


@dataclass
class DataConfig:
    member_a_path: str = str(_PROJECT_ROOT / "data" / "processed" / "member_a" / "member_a_model_features.csv")
    member_b_path: str = str(_PROJECT_ROOT / "data" / "processed" / "member_b" / "member_b_final.csv")
    member_c_path: str = str(_PROJECT_ROOT / "data" / "processed" / "member_c" / "member_c_final.csv")
    target_path: str = str(_PROJECT_ROOT / "data" / "processed" / "member_a" / "ucdp_panel.csv")

    train_end: str = "2017-12"
    val_end: str = "2021-12"

    target_col: str = "ucdp_fatalities_best"

    member_a_features: List[str] = field(default_factory=lambda: [
        "ucdp_event_count", "ucdp_fatalities_best", "ucdp_fatalities_high",
        "ucdp_civilian_deaths", "ucdp_peak_event_fatalities",
        "ucdp_fatality_uncertainty", "ucdp_state_based_events",
        "ucdp_non_state_events", "ucdp_one_sided_events", "ucdp_has_conflict",
        "gdelt_conflict_event_count", "gdelt_goldstein_mean",
        "ucdp_fatalities_best_ld", "ucdp_event_count_ld",
        "gdelt_conflict_event_count_ld",
    ])

    member_b_features: List[str] = field(default_factory=lambda: [
        "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_freexp_altinf",
        "v2x_frassoc_thick", "v2x_regime", "v2xnp_regcorr", "v2x_civlib",
        "v2x_clpol", "v2x_clphy", "v2x_corr", "v2x_execorr", "v2x_rule",
        "v2xcs_ccsi", "regime_type_0", "regime_type_1", "regime_type_2",
        "regime_type_3", "governance_deficit", "repression_index",
        "libdem_yoy_change", "tenure_months", "age", "male",
        "militarycareer", "elected", "leader_age_risk",
        "months_since_election", "regime_change", "coup_event",
        "prev_conflict", "precip", "months_since_structural_break",
        "reign_regime_Dominant Party", "reign_regime_Foreign/Occupied",
        "reign_regime_Indirect Military", "reign_regime_Military",
        "reign_regime_Military-Personal", "reign_regime_Monarchy",
        "reign_regime_Oligarchy", "reign_regime_Parliamentary Democracy",
        "reign_regime_Party-Military", "reign_regime_Party-Personal",
        "reign_regime_Party-Personal-Military Hybrid",
        "reign_regime_Personal Dictatorship",
        "reign_regime_Presidential Democracy",
        "reign_regime_Provisional - Civilian",
        "reign_regime_Provisional - Military", "reign_regime_Warlordism",
        "fx_pct_change", "fx_volatility", "fx_volatility_log",
        "fx_depreciation_flag", "fx_pct_change_zscore", "gdp_growth",
        "gdp_growth_deviation", "gdp_negative_shock", "food_price_anomaly",
        "food_price_anomaly_log", "food_cpi_acceleration", "food_price_spike",
        "pt_coup_successful", "pt_coup_failed", "pt_coup_count",
        "pt_coup_event", "pt_cumulative_coups", "pt_months_since_coup",
        "vdem_stale_flag", "vdem_available", "reign_available",
        "fx_available", "food_available", "gdp_available",
    ])

    member_c_features: List[str] = field(default_factory=lambda: [
        "gpr_country", "gpr_global", "gpr_acts",
        "tone_mean", "tone_min", "tone_max", "tone_std",
        "event_count", "goldstein_mean",
        "vix_mean", "vix_vol", "vix_pct_chg",
        "wti_oil_mean", "wti_oil_vol", "wti_oil_pct_chg",
        "gold_mean", "gold_vol", "gold_pct_chg",
        "dxy_mean", "dxy_vol", "dxy_pct_chg",
        "us_10y_yield_mean", "us_10y_yield_vol", "us_10y_yield_pct_chg",
        "wheat_mean", "wheat_vol", "wheat_pct_chg",
        "copper_mean", "copper_vol", "copper_pct_chg",
        "us_13w_tbill_mean", "us_13w_tbill_vol", "us_13w_tbill_pct_chg",
    ])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = str(_PROJECT_ROOT / "results")
    checkpoint_dir: str = str(_PROJECT_ROOT / "checkpoints")
    seed: int = 42

    def feature_columns(self, group: str) -> List[str]:
        """Return feature column names for the requested group: 'a', 'ab', or 'abc'."""
        cols = list(self.data.member_a_features)
        if group in ("ab", "abc"):
            cols += self.data.member_b_features
        if group == "abc":
            cols += self.data.member_c_features
        return cols
