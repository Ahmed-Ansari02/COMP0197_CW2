"""
Baseline models and ViEWS benchmark reference scores.

Provides naive baselines for comparison:
1. Exactly-zero: always predicts 0
2. Historical mean: predicts mean of training data
3. Last-value: repeats most recent observation
4. Climatology: uses past 12 months as empirical distribution

Plus ViEWS competition benchmark scores for context.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# ViEWS Prediction Competition (2nd competition, 2023/24)
#
# Source: country_level.html and grid_level.html from
#   https://viewsforecasting.org/research/prediction-challenge-2023/
# Evaluation window: true future, July 2024 – June 2025
# Paper: https://arxiv.org/abs/2407.11045
#
# All scores: lower = better.
# CRPS = Continuous Ranked Probability Score (primary metric)
# ab-Log = ab-Log Score (log-likelihood based)
# MIS = Mean Interval Score
# ──────────────────────────────────────────────────────────────

# Country-level results (cm), July 2024 – June 2025, ranked by CRPS
VIEWS_COUNTRY_LEVEL = {
    # rank: (model, crps, ab_log, mis)
    1:  ("VIEWS_bm_ph_conflictology_country12", 24.92, 0.76, 376.30),
    2:  ("Brandt_NB_GLMM", 25.17, 1.17, 390.80),
    3:  ("Randahl_Vegelius_markov_omm", 25.27, 0.68, 386.80),
    4:  ("ConflictForecast", 25.96, 0.69, 329.70),
    5:  ("CCEW_tft", 26.23, 1.13, 386.60),
    6:  ("VIEWS_bm_last_historical", 26.32, 1.15, 500.80),
    7:  ("Brandt_TW_GLMM", 26.69, 0.93, 466.70),
    8:  ("Bodentien_Rueter_NB", 27.22, 0.73, 468.80),
    9:  ("Drauz_Becker_quantile_forecast", 27.35, 0.78, 469.80),
    10: ("PaCE_ShapeFinder", 32.74, 0.83, 380.20),
    11: ("Brandt_P_GLMM", 33.38, 1.17, 645.10),
    12: ("Randahl_Vegelius_markov_hpmm", 41.40, 0.83, 351.60),
    13: ("Randahl_Vegelius_markov_gpcmm", 47.91, 8.75, 799.00),
    14: ("VIEWS_bm_conflictology_bootstrap240", 54.54, 1.30, 1061.40),
    15: ("VIEWS_bm_exactly_zero", 55.80, 1.83, 1116.00),
    16: ("Muchlinski_Thornhill_zeroinf_GAM", 61.16, 2.43, 1183.50),
    17: ("UNITO_transformer", 64.10, 1.12, 1020.10),
}

# Grid-level results (pgm), July 2024 – June 2025, ranked by CRPS
# just grabbed this but tbh , we only ahve data for countires anyways. 
VIEWS_GRID_LEVEL = {
    1:  ("VIEWS_bm_ph_conflictology_country12", 0.27, 0.11, 4.58),
    2:  ("CCEW_trees_local", 0.29, 0.10, 4.61),
    3:  ("CCEW_trees_global_local", 0.29, 0.09, 4.43),
    4:  ("CCEW_trees_global", 0.29, 0.09, 4.57),
    5:  ("VIEWS_bm_ph_conflictology_neighbors12", 0.30, 0.11, 4.96),
    6:  ("VIEWS_bm_conflictology_bootstrap240", 0.32, 0.12, 6.31),
    7:  ("VIEWS_bm_exactly_zero", 0.32, 0.12, 6.31),
    8:  ("VIEWS_bm_last_historical", 0.32, 0.15, 5.96),
    9:  ("CCEW_tft", 0.33, 0.12, 6.02),
    10: ("ConflictForecast", 0.33, 0.09, 7.91),
}

# Convenience dicts for eval_runner comparison tables
VIEWS_BENCHMARKS = {
    name: {"crps": crps, "ab_log": ab_log, "mis": mis}
    for _, (name, crps, ab_log, mis) in VIEWS_COUNTRY_LEVEL.items()
    if name.startswith("VIEWS_bm")
}

VIEWS_COMPETITION_ENTRIES = {
    name: {"crps": crps, "ab_log": ab_log, "mis": mis}
    for _, (name, crps, ab_log, mis) in VIEWS_COUNTRY_LEVEL.items()
    if not name.startswith("VIEWS_bm")
}


# ──────────────────────────────────────────────────────────────
# Naive Baselines (generate samples for evaluation)
# ──────────────────────────────────────────────────────────────

def exactly_zero_samples(n_observations: int, n_samples: int = 1000) -> np.ndarray:
    """Always predict zero fatalities."""
    return np.zeros((n_observations, n_samples))


def historical_mean_samples(
    y_train: np.ndarray,
    n_observations: int,
    n_samples: int = 1000,
) -> np.ndarray:
    """Sample from a Poisson with rate = training mean."""
    rate = max(y_train.mean(), 0.01)
    return np.random.poisson(rate, size=(n_observations, n_samples)).astype(float)


def last_value_samples(
    y_last: np.ndarray,
    n_samples: int = 1000,
    noise_std: float = 0.1,
) -> np.ndarray:
    """
    Repeat last observed value with small noise.

    Args:
        y_last: [N] most recent observation per country-month
        n_samples: samples per observation
        noise_std: relative noise (fraction of value)
    """
    base = y_last[:, None].repeat(n_samples, axis=1)
    noise = np.random.normal(0, noise_std * np.abs(base) + 0.1, base.shape)
    return np.maximum(base + noise, 0).round()


def climatology_samples(
    y_history: dict[str, np.ndarray],
    n_samples: int = 1000,
    window: int = 12,
) -> np.ndarray:
    """
    Climatology baseline: sample from the empirical distribution
    of the last `window` months for each country.

    This is equivalent to ViEWS conflictology_country12.

    Args:
        y_history: dict mapping country_iso3 -> array of historical fatalities
        n_samples: samples per observation

    Returns:
        [N, n_samples] where N = number of countries
    """
    all_samples = []
    for iso3, history in y_history.items():
        recent = history[-window:] if len(history) >= window else history
        if len(recent) == 0:
            recent = np.array([0.0])
        # Bootstrap from recent history
        samples = np.random.choice(recent, size=n_samples, replace=True)
        all_samples.append(samples)

    return np.array(all_samples)


def print_views_benchmarks():
    """Print ViEWS leaderboard (country-level, July 2024 – June 2025)."""
    print("\nViEWS Live Leaderboard — Country-level, July 2024 – June 2025")
    print("Source: country_level.html from viewsforecasting.org")
    print(f"{'Rank':>4s}  {'Model':42s} {'CRPS':>7s} {'ab-Log':>7s} {'MIS':>9s}")
    print("-" * 73)
    for rank, (name, crps, ab_log, mis) in sorted(VIEWS_COUNTRY_LEVEL.items()):
        marker = " *" if name.startswith("VIEWS_bm") else ""
        print(f"  {rank:2d}  {name:42s} {crps:7.2f} {ab_log:7.2f} {mis:9.1f}{marker}")
    print("-" * 73)
    print("* = benchmark model")
    print("#1 to beat: CRPS < 24.92 (VIEWS_bm_ph_conflictology_country12)")
    print()
