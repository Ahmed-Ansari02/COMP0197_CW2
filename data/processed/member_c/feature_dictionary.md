# Feature Dictionary

Panel structure: **(country_iso3, year_month)** — 44 countries, 1985-01 to 2025-12, 21,604 rows, 40 columns.

All features are lagged by **t-1** (April's row contains March's data) to prevent data leakage.

---

## Identifiers

| Column | Description |
|--------|-------------|
| `year_month` | Calendar month (YYYY-MM) |
| `country_iso3` | ISO 3166-1 alpha-3 country code (e.g., USA, GBR, CHN) |

---

## Geopolitical Risk Index (GPR)

Source: [Caldara & Iacoviello](https://www.matteoiacoviello.com/gpr.htm), derived from automated text analysis of 10 major newspapers. Higher values = more geopolitical risk. Monthly, 1985–present.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `gpr_global` | Global geopolitical risk index — aggregate measure across all newspapers | log1p |
| `gpr_acts` | GPR Acts sub-index — counts mentions of actual geopolitical events (wars, terrorist attacks, tensions that materialised) | log1p |
| `gpr_country` | Country-specific GPR index — geopolitical risk score for each individual country | log1p |

**Dropped:** `gpr_threats` (r=0.88 with `gpr_global`, captures threat mentions vs actual events — redundant with global index)

---

## GDELT (Global Database of Events, Language, and Tone)

Source: [GDELT Project](https://www.gdeltproject.org/) via BigQuery. Monitors broadcast, print, and web news worldwide, coding events using the CAMEO taxonomy. Monthly aggregates from daily data, 1985–present (v1: 1985–2014, v2: 2015+).

| Feature | Description | Transform |
|---------|-------------|-----------|
| `tone_mean` | Average article tone across all events for that country-month. Ranges roughly -10 (very negative) to +10 (very positive). | raw |
| `tone_min` | Minimum (most negative) article tone in that country-month. Captures the worst sentiment. | raw |
| `tone_max` | Maximum (most positive) article tone in that country-month. Captures the best sentiment. | raw |
| `tone_std` | Standard deviation of article tone. High values = polarised/mixed coverage. | log1p |
| `event_count` | Total number of coded events for that country-month. Proxy for how much is happening / media attention. | log1p |
| `goldstein_mean` | Mean Goldstein scale score. Ranges -10 (most hostile, e.g., military attack) to +10 (most cooperative, e.g., alliance). Measures the average cooperative–hostile nature of events. | raw |

**Dropped:** `total_articles` (r=0.99 with `event_count`), `hostile_event_count` (r=0.97), `cooperative_event_count` (r=0.99), `goldstein_min` (near-constant at -10, since almost every country-month has at least one maximally hostile event)

---

## VIX (CBOE Volatility Index)

Source: Yahoo Finance (`^VIX`). Measures the market's expectation of 30-day volatility on the S&P 500, often called the "fear gauge". Higher values = more market uncertainty/fear. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `vix_mean` | Monthly average of daily VIX values. Baseline fear/uncertainty level for that month. | raw |
| `vix_vol` | Intra-month standard deviation of daily VIX. High values = volatility of volatility (unstable risk environment). | raw |
| `vix_pct_chg` | Month-over-month percentage change in VIX. Positive = rising fear, negative = calming markets. | raw |

---

## WTI Crude Oil

Source: Yahoo Finance (`CL=F`). West Texas Intermediate crude oil futures — the primary US oil benchmark. Priced in USD per barrel. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `wti_oil_mean` | Monthly average oil price. Captures the general price level. | raw |
| `wti_oil_vol` | Intra-month standard deviation of daily oil prices. High values = supply/demand shocks or geopolitical disruptions. | raw |
| `wti_oil_pct_chg` | Month-over-month percentage change. Captures price momentum / sudden shifts. | raw |

**Dropped:** All Brent oil features (r=0.98 with WTI — nearly identical information)

---

## Gold

Source: Yahoo Finance (`GC=F`). Gold futures — traditional safe-haven asset. Rises during geopolitical uncertainty, inflation fears, and dollar weakness. Priced in USD per troy ounce. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `gold_mean` | Monthly average gold price. | raw |
| `gold_vol` | Intra-month price volatility. Spikes during crises. | raw |
| `gold_pct_chg` | Month-over-month percentage change. | raw |

---

## DXY (US Dollar Index)

Source: Yahoo Finance (`DX-Y.NYB`). Measures the USD against a basket of 6 major currencies (EUR, JPY, GBP, CAD, SEK, CHF). Higher values = stronger dollar. A strong dollar can pressure emerging markets (dollar-denominated debt becomes more expensive). Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `dxy_mean` | Monthly average dollar strength. | raw |
| `dxy_vol` | Intra-month volatility of the dollar. | raw |
| `dxy_pct_chg` | Month-over-month percentage change. | raw |

---

## US 10-Year Treasury Yield

Source: Yahoo Finance (`^TNX`). Yield on 10-year US government bonds — the global benchmark for "risk-free" rate. Rising yields signal tightening financial conditions; falling yields signal flight to safety. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `us_10y_yield_mean` | Monthly average yield (in %). | raw |
| `us_10y_yield_vol` | Intra-month yield volatility. High values = bond market stress. | raw |
| `us_10y_yield_pct_chg` | Month-over-month percentage change in yield. | raw |

---

## Wheat

Source: Yahoo Finance (`ZW=F`). Wheat futures — a key food commodity. Price spikes often coincide with conflict (supply disruptions, export bans) and can themselves trigger instability in import-dependent countries. Priced in USD cents per bushel. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `wheat_mean` | Monthly average wheat price. | raw |
| `wheat_vol` | Intra-month price volatility. | raw |
| `wheat_pct_chg` | Month-over-month percentage change. | raw |

---

## Copper

Source: Yahoo Finance (`HG=F`). Copper futures — often called "Dr. Copper" because its price is considered a barometer of global economic health (used heavily in construction and manufacturing). Priced in USD per pound. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `copper_mean` | Monthly average copper price. | raw |
| `copper_vol` | Intra-month price volatility. | raw |
| `copper_pct_chg` | Month-over-month percentage change. | raw |

---

## US 13-Week Treasury Bill

Source: Yahoo Finance (`^IRX`). Yield on 3-month US T-bills — the shortest-term risk-free rate. Closely tracks Federal Reserve policy. The spread between 13W and 10Y yields (yield curve) is a recession predictor. Daily, aggregated to monthly.

| Feature | Description | Transform |
|---------|-------------|-----------|
| `us_13w_tbill_mean` | Monthly average yield (in %). | raw |
| `us_13w_tbill_vol` | Intra-month yield volatility. | raw |
| `us_13w_tbill_pct_chg` | Month-over-month percentage change. | raw |

---

## Log-Transformed Features

For heavy-tailed features, `log(1 + x)` versions are included to compress extreme values and make distributions more suitable for the transformer. Both raw and transformed versions are in the panel — feature selection will determine which the model prefers.

| Feature | Original |
|---------|----------|
| `gpr_global_log1p` | `gpr_global` |
| `gpr_acts_log1p` | `gpr_acts` |
| `gpr_country_log1p` | `gpr_country` |
| `event_count_log1p` | `event_count` |
| `tone_std_log1p` | `tone_std` |

---

## Summary of Dropped Features (17 total)

| Feature | Reason |
|---------|--------|
| `gpr_threats` | r=0.88 with `gpr_global` |
| `total_articles` | r=0.99 with `event_count` |
| `hostile_event_count` | r=0.97 with `event_count` |
| `cooperative_event_count` | r=0.99 with `event_count` |
| `goldstein_min` | Near-constant at -10 (no variance) |
| `brent_oil_mean` | r=0.98 with `wti_oil_mean` |
| `brent_oil_vol` | Dropped with Brent suite |
| `brent_oil_close` | Dropped with Brent suite |
| `brent_oil_pct_chg` | Dropped with Brent suite |
| `vix_close` | r=0.94 with `vix_mean` |
| `wti_oil_close` | r=0.99 with `wti_oil_mean` |
| `gold_close` | r=1.00 with `gold_mean` |
| `dxy_close` | r=0.99 with `dxy_mean` |
| `us_10y_yield_close` | r=1.00 with `us_10y_yield_mean` |
| `wheat_close` | r=0.99 with `wheat_mean` |
| `copper_close` | r=0.99 with `copper_mean` |
| `us_13w_tbill_close` | r=1.00 with `us_13w_tbill_mean` |
