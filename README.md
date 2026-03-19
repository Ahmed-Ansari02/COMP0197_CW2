# COMP0197 — Coursework 2

Predicting fatality counts as a probability distribution over a monthly rolling window using a transformer model, built on the ViEWS dataset as the panel scaffold.

## Project Structure

```
.
├── data/
│   └── processed/
│       ├── member_a/          # ACLED, UCDP, GDELT event data
│       ├── member_b/          # V-Dem, REIGN, Economic indicators
│       └── member_c/          # GPR, GDELT tone, macro/volatility indicators
│
├── pipelines/
│   ├── member_a/              # Data acquisition & processing scripts
│   ├── member_b/
│   └── member_c/
│
├── analysis/
│   ├── member_a/              # EDA outputs (missingness heatmaps, distribution plots)
│   ├── member_b/
│   └── member_c/
│
├── model/                     # Shared transformer model code
│
├── docs/                      # reports 
│
├── requirements.txt
└── README.md
```

## Data Sources

| Source | Member | Description |
|--------|--------|-------------|
| ACLED | A | Armed conflict events, protests, fatalities |
| UCDP | A | Conflict event and fatality counts (also target variable via ViEWS) |
| GDELT | A, C | Global event database — event counts, tone, Goldstein scores |
| V-Dem | B | Democracy indices, press freedom, corruption |
| REIGN | B | Regime characteristics, leader tenure, elections |
| Economic | B | Exchange rate volatility, GDP growth |
| GPR | C | Geopolitical risk index (global + 44 countries) |
| Macro | C | VIX, WTI oil, gold, DXY, US 10Y yield, wheat, copper, US 13W T-bill |

## Panel Structure

- **Unit of analysis:** (country, month)
- **Temporal range:** 1985-01 to 2025-12
- **Country identifiers:** ISO 3166-1 alpha-3 codes
- **Lag convention:** All features lagged by t-1 (April's row uses March's data)
- **Target variable:** UCDP-based fatality counts from ViEWS

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Pipelines

Each member's pipeline generates their processed dataset and analysis outputs.

```bash
# Member A — UCDP, ACLED, GDELT conflict events
python pipelines/member_a/generate_conflict_dataset.py

# Member C — GPR, GDELT tone, macro/volatility indicators
python pipelines/member_c/generate_volatility_dataset.py
```

### Member A outputs
- `data/processed/member_a/member_a_final.csv` — 86,100 rows x 23 columns (full feature set)
- `data/processed/member_a/member_a_model_features.csv` — 86,100 rows x 18 columns (model-ready, redundant features removed)
- `data/processed/member_a/feature_registry.csv` — metadata for all features
- `analysis/member_a/` — missingness heatmaps, distribution plots

To generate the model-ready file after running the main pipeline:
```bash
python pipelines/member_a/filter_features.py
```

#### Member A features (model-ready set)

All features are lagged by t-1. Features marked log1p are log1p-transformed to handle heavy-tailed distributions.

| Feature | Source | Transform | Description |
|---------|--------|-----------|-------------|
| `ucdp_event_count` | UCDP | log1p | Total conflict events in month |
| `ucdp_fatalities_best` | UCDP | log1p | Best-estimate fatality count |
| `ucdp_fatalities_high` | UCDP | log1p | High-estimate fatality count |
| `ucdp_civilian_deaths` | UCDP | log1p | Civilian fatalities |
| `ucdp_peak_event_fatalities` | UCDP | log1p | Max fatalities in a single event |
| `ucdp_fatality_uncertainty` | UCDP | raw | Mean(high - low) across events |
| `ucdp_state_based_events` | UCDP | raw | Events involving a state actor (type=1) |
| `ucdp_non_state_events` | UCDP | raw | Non-state conflict events (type=2) |
| `ucdp_one_sided_events` | UCDP | raw | One-sided violence events (type=3) |
| `ucdp_has_conflict` | UCDP | binary | Any events recorded this month |
| `acled_fatalities` | ACLED | log1p | Total reported fatalities |
| `acled_battle_count` | ACLED | log1p | Battle events |
| `acled_explosion_count` | ACLED | raw | Explosions/remote violence events |
| `acled_violence_count` | ACLED | raw | Violence against civilians events |
| `gdelt_conflict_event_count` | GDELT | log1p | CAMEO 18/19/20 events — assault, fight, mass violence |
| `gdelt_goldstein_mean` | GDELT | raw | Mean Goldstein hostility score across conflict events |

Added log-difference change features — computed as `log1p(x at t-1) - log1p(x at t-2)`, capturing month-on-month escalation without division-by-zero on sparse conflict data:

| Feature | Description |
|---------|-------------|
| `ucdp_fatalities_best_ld` | Log-difference of verified fatalities (t-1 vs t-2) |
| `ucdp_event_count_ld` | Log-difference of conflict event count (t-1 vs t-2) |
| `acled_fatalities_ld` | Log-difference of ACLED-reported fatalities (t-1 vs t-2) |
| `gdelt_conflict_event_count_ld` | Log-difference of media-reported conflict events (t-1 vs t-2) |

Dropped from full set (`member_a_final.csv`) as redundant or overlapping with member C:
- `acled_event_count` — redundant with event type breakdown
- `acled_peak_fatalities` — redundant with `ucdp_peak_event_fatalities` and `acled_fatalities`
- `acled_protest_count`, `acled_riot_count` — overlap with member C's GDELT protest signal
- `gdelt_protest_event_count` — overlap with member C's GDELT tone features

### Member C outputs
- `data/processed/member_c/member_c_final.csv` — 21,604 rows x 40 columns (combined & cleaned dataset)
- `data/processed/member_c/feature_registry.csv` — metadata for all features
- `analysis/member_c/` — missingness heatmaps, distribution plots, correlation matrix

#### Member C features (kept after redundancy filtering)

| Feature | Source | Description |
|---------|--------|-------------|
| `gpr_global` | GPR | Global geopolitical risk index (log1p) |
| `gpr_acts` | GPR | GPR sub-index for actual conflict acts (log1p) |
| `gpr_country` | GPR | Country-level GPR index (log1p) |
| `tone_mean` | GDELT | Mean news tone for country-month |
| `tone_min` | GDELT | Min news tone (most negative article) |
| `tone_max` | GDELT | Max news tone (most positive article) |
| `tone_std` | GDELT | Tone volatility within month (log1p) |
| `event_count` | GDELT | Total GDELT events (log1p) |
| `goldstein_mean` | GDELT | Mean Goldstein score across all events |
| `vix_mean` | Macro | Monthly mean VIX (equity volatility) |
| `vix_vol` | Macro | Intra-month VIX volatility |
| `vix_pct_chg` | Macro | Month-on-month VIX % change |
| `wti_oil_mean` | Macro | Monthly mean WTI crude oil price |
| `wti_oil_vol` | Macro | Intra-month WTI volatility |
| `wti_oil_pct_chg` | Macro | Month-on-month WTI % change |
| `gold_mean` | Macro | Monthly mean gold price |
| `gold_vol` | Macro | Intra-month gold volatility |
| `gold_pct_chg` | Macro | Month-on-month gold % change |
| `dxy_mean` | Macro | Monthly mean USD index |
| `dxy_vol` | Macro | Intra-month DXY volatility |
| `dxy_pct_chg` | Macro | Month-on-month DXY % change |
| `us_10y_yield_mean` | Macro | Monthly mean US 10-year treasury yield |
| `us_10y_yield_vol` | Macro | Intra-month yield volatility |
| `us_10y_yield_pct_chg` | Macro | Month-on-month yield % change |
| `wheat_mean` | Macro | Monthly mean wheat futures price |
| `wheat_vol` | Macro | Intra-month wheat volatility |
| `wheat_pct_chg` | Macro | Month-on-month wheat % change |
| `copper_mean` | Macro | Monthly mean copper futures price |
| `copper_vol` | Macro | Intra-month copper volatility |
| `copper_pct_chg` | Macro | Month-on-month copper % change |
| `us_13w_tbill_mean` | Macro | Monthly mean US 13-week T-bill rate |
| `us_13w_tbill_vol` | Macro | Intra-month T-bill rate volatility |
| `us_13w_tbill_pct_chg` | Macro | Month-on-month T-bill % change |
