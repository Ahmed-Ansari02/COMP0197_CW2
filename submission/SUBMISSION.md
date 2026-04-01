# Submission Structure

## Files
- `train.py` — Data pipeline + model training (self-contained)
- `test.py` — Evaluation + visualization (imports model from train.py)
- `best_model.pt` — Saved Conv-Transformer weights
- `instruction.pdf` — Reproduction steps + 3 additional packages
- `cm_monthly_scores_full_Jul-Jun.csv` — ViEWS benchmark scores

## Data Directory
```
data/
  member_a/
    ucdp_panel.csv              # UCDP conflict events (raw target source)
    gdelt_events.csv            # GDELT conflict events (from BigQuery)
    member_a_final.csv          # Aggregated conflict features
  member_b/
    vdem_governance.csv         # V-Dem governance indices
    reign_leader.csv            # REIGN leader/regime data
    fx_exchange_rates.csv       # IMF exchange rates
    gdp_growth.csv              # World Bank GDP growth
    food_prices.csv             # FAO food CPI
    powell_thyne_coups.csv      # Coup events
    member_b_final.csv          # Aggregated structural features
  member_c/
    gpr_global.csv              # Geopolitical Risk Index
    gdelt_tone.csv              # GDELT tone/sentiment (from BigQuery)
    macro_indicators.csv        # VIX, oil, gold, etc. (from Yahoo Finance)
    member_c_final.csv          # Aggregated volatility features
  merge/
    gdelt_tone_all.csv          # GDELT tone for all countries
    model_ready.csv             # Final preprocessed training data
```

## Additional Packages (3)
1. `pandas` — data loading and manipulation
2. `matplotlib` — visual results in test.py
3. `scipy` — evaluation metrics (KDE, chi-squared test)

## Pipeline Flow (in train.py)
Each stage checks if its output CSV exists and skips if so.

```
Stage 1: Load Member A — conflict events (UCDP + GDELT)
Stage 2: Load Member B — governance, economic, regime features
Stage 3: Load Member C — geopolitical risk, tone, macro indicators
Stage 4: Merge — outer join A+B+C, broadcast global features, backfill GDELT tone
Stage 5: Add autoregressive features (target lags + rolling statistics)
Stage 6: Preprocess — log1p transforms, drop redundant, z-score, clip, fill NaN
Stage 7: Train Conv-Transformer (Hurdle-Student-t distribution)
```

Since all CSVs are shipped pre-built, stages 1-6 are skipped on first run.
If model_ready.csv is deleted, stages 4-6 rebuild it from member finals.

## Model Architecture
Conv-Transformer with Hurdle-Student-t output distribution:
- Causal Conv front-end (2 layers) → local pattern extraction
- Patch embedding (3-month patches) → quarter-level tokenization
- Rotary-PE Transformer (4 layers, 4 heads) → temporal attention
- Hurdle-Student-t head → P(zero) gate + Student-t severity in log1p space

## Evaluation (in test.py)
- ViEWS competition window: July 2024 – June 2025
- Metrics: CRPS, IGN (log score), MIS (interval score), PIT calibration, spike detection
- Compared against 20 ViEWS competition entries
- Visualizations: PIT histogram, CRPS bar chart, per-month CRPS

## Data Sources
| Source | Coverage | Method |
|--------|----------|--------|
| UCDP GED v25.1 | 1989–2024 | Pre-built CSV |
| GDELT (BigQuery) | 1985–2025 | Pre-built CSV |
| ACLED | 1997–present | Pre-built into member_a_final |
| V-Dem v16 | 1900–2023 | Pre-built CSV |
| REIGN | 1966–2021 | Pre-built CSV |
| IMF IFS (FX) | 1985–2025 | Pre-built CSV |
| World Bank (GDP) | 1980–2025 | Pre-built CSV |
| FAO (food CPI) | 2000–2025 | Pre-built CSV |
| Powell & Thyne (coups) | 1950–present | Pre-built CSV |
| GPR Index | 1985–present | Pre-built CSV |
| Yahoo Finance (macro) | 1984–2026 | Pre-built CSV |
