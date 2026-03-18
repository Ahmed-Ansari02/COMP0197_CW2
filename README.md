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
# Member C — GPR, GDELT tone, macro/volatility indicators
python pipelines/member_c/generate_volatility_dataset.py
```

This produces:
- `data/processed/member_c/member_c_final.csv` — 21,604 rows x 40 columns (combined & cleaned dataset)
- `data/processed/member_c/feature_registry.csv` — metadata for all features
- `analysis/member_c/` — missingness heatmaps, distribution plots, correlation matrix
