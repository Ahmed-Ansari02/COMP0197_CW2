# COMP0197 — Coursework 2

Predicting fatality counts as a probability distribution over a monthly rolling window using a transformer model, built on the ViEWS dataset as the panel scaffold.

## Project Structure

```
.
├── data/
│   └── processed/
│       ├── member_a/          # ACLED, UCDP, GDELT event data
│       ├── member_b/          # V-Dem, REIGN, Powell & Thyne, Economic indicators
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
| IMF IFS | B | Exchange rate volatility (monthly, 1985–2025) |
| World Bank WDI | B | GDP growth (annual, expanded to monthly) |
| FAO CPI | B | Food consumer price index (monthly, 2000–2025 only) |
| Powell & Thyne | B | Coup d'état events (independent source, 1985–2025) |
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

# Member B — V-Dem, REIGN, economic, coup indicators
python pipelines/member_b/generate_structural_dataset.py

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
| `gdelt_conflict_event_count` | GDELT | log1p | CAMEO 18/19/20 events — assault, fight, mass violence |
| `gdelt_goldstein_mean` | GDELT | raw | Mean Goldstein hostility score across conflict events |

Added log-difference change features — computed as `log1p(x at t-1) - log1p(x at t-2)`, capturing month-on-month escalation without division-by-zero on sparse conflict data:

| Feature | Description |
|---------|-------------|
| `ucdp_fatalities_best_ld` | Log-difference of verified fatalities (t-1 vs t-2) |
| `ucdp_event_count_ld` | Log-difference of conflict event count (t-1 vs t-2) |
| `gdelt_conflict_event_count_ld` | Log-difference of media-reported conflict events (t-1 vs t-2) |

Dropped from full set (`member_a_final.csv`) as overlapping with member C:
- `gdelt_protest_event_count` — overlap with member C's GDELT tone features

### Member B outputs
- `data/processed/member_b/member_b_final.csv` — 92,988 rows x 76 columns (combined & cleaned dataset)
- `data/processed/member_b/vdem_governance.csv` — V-Dem governance indices (171 countries)
- `data/processed/member_b/reign_leader.csv` — REIGN leader/regime data (187 countries)
- `data/processed/member_b/fx_exchange_rates.csv` — IMF exchange rate features (166 countries)
- `data/processed/member_b/gdp_growth.csv` — World Bank GDP growth (185 countries)
- `data/processed/member_b/food_prices.csv` — FAO food CPI features (176 countries)
- `data/processed/member_b/powell_thyne_coups.csv` — Powell & Thyne coup events
- `data/processed/member_b/feature_registry.csv` — metadata for all features
- `data/processed/member_b/quality_report.json` — coverage & range statistics
- `analysis/member_b/` — missingness heatmaps, distribution plots

Manual downloads required (place in `data/raw/`):

| Source | URL | Notes |
|--------|-----|-------|
| V-Dem | https://v-dem.net/data/the-v-dem-dataset/ | Select "Country-Year: V-Dem Full+Others" (~300 MB) |
| IMF IFS | https://data.imf.org/en/Data%20Explorer?datasetUrn=IMF.STA%3AER%284.0.1%29 | Exchange rates, download as CSV |
| FAO CPI | https://www.fao.org/faostat/en/#data/CP | Use "Bulk Downloads → All Data" |

REIGN, GDP, and Powell & Thyne are auto-downloaded with `--download`:
```bash
python pipelines/member_b/generate_structural_dataset.py --download
```

#### Member B features

All features are lagged by t-1. Panel covers 1985-01 to 2025-12.

**V-Dem governance indices (annual, expanded to monthly)**

| Feature | Source | Description |
|---------|--------|-------------|
| `v2x_libdem` | V-Dem | Liberal democracy index |
| `v2x_polyarchy` | V-Dem | Electoral democracy index |
| `v2x_partipdem` | V-Dem | Participatory democracy index |
| `v2x_civlib` | V-Dem | Civil liberties index |
| `v2x_rule` | V-Dem | Rule of law index |
| `v2x_corr` | V-Dem | Political corruption index |
| `v2x_execorr` | V-Dem | Executive corruption index |
| `v2x_clphy` | V-Dem | Physical violence index |
| `v2x_clpol` | V-Dem | Political civil liberties index |
| `v2x_freexp_altinf` | V-Dem | Freedom of expression & alt info index |
| `v2x_frassoc_thick` | V-Dem | Freedom of association (thick) index |
| `v2xcs_ccsi` | V-Dem | Core civil society index |
| `v2xnp_regcorr` | V-Dem | Regime corruption index |
| `v2x_regime` | V-Dem | Regime type (0=closed autoc, 1=electoral autoc, 2=electoral dem, 3=liberal dem) |
| `regime_type_0..3` | V-Dem | One-hot encoded regime type |
| `governance_deficit` | V-Dem | Derived: 1 − mean(libdem, polyarchy, rule) |
| `repression_index` | V-Dem | Derived: mean(corr, execorr) + (1 − clphy) / 2 |
| `libdem_yoy_change` | V-Dem | Year-on-year change in liberal democracy index |

**REIGN leader/regime data (monthly, observed only — ends Aug 2021, NaN after)**

| Feature | Source | Description |
|---------|--------|-------------|
| `tenure_months` | REIGN | Leader's tenure in months |
| `age` | REIGN | Leader's age |
| `male` | REIGN | Leader gender (binary) |
| `militarycareer` | REIGN | Military career background (binary) |
| `elected` | REIGN | Leader was elected (binary) |
| `leader_age_risk` | REIGN | Derived: age < 40 or > 75 (binary) |
| `months_since_election` | REIGN | Months since last election |
| `regime_change` | REIGN | Government type changed this month (binary) |
| `coup_event` | REIGN | Irregular transfer of power (binary) |
| `prev_conflict` | REIGN | Previous conflict indicator |
| `precip` | REIGN | Precipitation anomaly |
| `months_since_structural_break` | REIGN | Months since last regime change or coup |
| `reign_regime_*` | REIGN | One-hot encoded regime type (14 categories) |
| `vdem_stale_flag` | Derived | V-Dem annual data may be stale due to mid-year structural break (uses REIGN + Powell & Thyne) |

**Economic indicators (FAO food CPI available from 2000 only — NaN for 1985–1999)**

| Feature | Source | Description |
|---------|--------|-------------|
| `fx_pct_change` | IMF IFS | Month-on-month exchange rate % change |
| `fx_volatility` | IMF IFS | 3-month rolling std of fx_pct_change |
| `fx_volatility_log` | IMF IFS | log1p of fx_volatility |
| `fx_depreciation_flag` | IMF IFS | Depreciation > 2σ above country mean (binary) |
| `fx_pct_change_zscore` | IMF IFS | Expanding z-score of fx_pct_change |
| `gdp_growth` | World Bank | Annual GDP growth rate (expanded to monthly) |
| `gdp_growth_deviation` | World Bank | Z-score of GDP growth vs 5-year rolling mean |
| `gdp_negative_shock` | World Bank | GDP deviation < −1.0 (binary) |
| `food_price_anomaly` | FAO | Food CPI / 12-month rolling mean |
| `food_price_anomaly_log` | FAO | log of food_price_anomaly |
| `food_cpi_acceleration` | FAO | Month-on-month change in year-on-year food CPI growth |
| `food_price_spike` | FAO | Food price anomaly > 1.15 (binary) |

**Powell & Thyne coup events (independent source, 1985–2025)**

| Feature | Source | Description |
|---------|--------|-------------|
| `pt_coup_event` | Powell & Thyne | Any coup attempt this month (binary) |
| `pt_coup_successful` | Powell & Thyne | Successful coup (binary) |
| `pt_coup_failed` | Powell & Thyne | Failed coup attempt (binary) |
| `pt_coup_count` | Powell & Thyne | Number of coup events in month |
| `pt_cumulative_coups` | Powell & Thyne | Cumulative coup count for country |
| `pt_months_since_coup` | Powell & Thyne | Months since last coup event |

**Missingness indicators**

| Feature | Source | Description |
|---------|--------|-------------|
| `vdem_available` | Derived | V-Dem data available for this country-month (binary) |
| `reign_available` | Derived | REIGN data available (binary, 0 after Aug 2021) |
| `fx_available` | Derived | Exchange rate data available (binary) |
| `gdp_available` | Derived | GDP data available (binary) |
| `food_available` | Derived | Food price data available (binary) |

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
