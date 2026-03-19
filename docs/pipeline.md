
  ---
  The Complete Data Loading Pipeline — Explained Simply

  The Big Picture: What Is This Thing?

  Imagine you're trying to predict where fights might break out in the world next month. To do that, you need to gather clues from many different sources — how democratic a country is, who's leading it, how their money is doing, how expensive food is, and whether there have been recent coups.

  This pipeline is like a factory assembly line that:
  1. Takes in 6 different raw ingredient files (CSV/TSV data)
  2. Cleans, transforms, and combines them
  3. Outputs one single table (a Parquet file) where every row is a (country, month) pair and every column is a "clue" (feature)

  The final table has roughly ~180 countries x 180 months (Jan 2010 – Dec 2024) = ~32,400 rows and ~60 columns.

  ---
  Step 0: How It Starts — run.py

  When you run python run.py, here's what happens:

  1. It sets up logging (both to screen and to data/logs/member_b_pipeline.log)
  2. It checks which raw data files exist in data/raw/
  3. If you passed --download, it auto-downloads REIGN, GDP, and coup data
  4. It skips any source whose file is missing (graceful degradation — doesn't crash)
  5. It calls the main orchestrator: build_structural_features()

  Think of run.py as the "start" button on the factory.

  ---
  Step 1: The Orchestrator — structural_features.py

  build_structural_features() (line 358) is the factory foreman. It runs 10 steps in order:

  Step 1a: Call each ingredient loader

  vdem_df   = ingest_vdem(csv)      → democracy/governance data
  reign_df  = ingest_reign(csv)     → leader/regime data
  econ_dfs  = ingest_all_economic() → FX, GDP, food prices
  pt_df     = ingest_powell_thyne() → coup events

  Each loader returns a clean DataFrame with columns (gwcode, year_month, feature1, feature2, ...).

  Step 1b: Build the "skeleton"

  The function _build_panel_skeleton() (line 56) creates an empty grid of every possible (country, month) combination:

  gwcode  year_month
    2     2010-01
    2     2010-02
    ...
    2     2024-12
    20    2010-01
    ...

  It gets the list of countries from whichever data sources were loaded. This skeleton is the blank table that everything gets attached to.

  Step 1c: Left-join all sources onto the skeleton

  Each cleaned dataset is merged onto the skeleton using (gwcode, year_month) as the key. Left join means: if a source doesn't have data for a particular country-month, that cell becomes NaN (empty).

  This is like having a big attendance sheet and checking off which clues you have for each country-month.

  Step 1d: Integrate Powell & Thyne with REIGN

  This patches the REIGN gap (explained below in the REIGN section). For months after August 2021 where REIGN has no coup data, Powell & Thyne's coup data fills in.

  Step 1e: Cross-validate V-Dem against structural breaks

  _cross_validate_structural_breaks() (line 89) does something clever:

  V-Dem scores are annual (coded once per year). But what if a coup happens in June? The V-Dem score for that year was coded before the coup — it's now stale for the remaining months. This function:

  1. Looks at each country-year
  2. Checks if a coup or regime change happened that year
  3. If yes, marks all months after the coup as vdem_stale_flag = 1

  This tells the model: "Don't fully trust the democracy score here — something big happened that it doesn't reflect."

  Step 1f: Apply t-1 lag

  _apply_lag() (line 162) is the most important safety feature in the whole pipeline.

  What it does: For every feature column, it shifts the value forward by 1 month within each country. So the row for (France, June 2020) now contains the feature values from May 2020.

  Why: If you're predicting whether a fight happens in June, you can only use information that was available before June. Using June's data to predict June would be cheating (temporal leakage). The lag is applied once, here, centrally — not in the individual loaders. This makes it auditable and consistent.

  Consequence: The first month (Jan 2010) for every country becomes NaN — there's nothing before it to shift from.

  Step 1g: Encode missingness

  _encode_missingness() (line 193) creates 5 binary columns:

  ┌─────────────────┬────────────────────────────────┬──────────┐
  │     Column      │            1 if...             │ 0 if...  │
  ├─────────────────┼────────────────────────────────┼──────────┤
  │ vdem_available  │ v2x_libdem has a value         │ it's NaN │
  ├─────────────────┼────────────────────────────────┼──────────┤
  │ reign_available │ tenure_months has a value      │ it's NaN │
  ├─────────────────┼────────────────────────────────┼──────────┤
  │ fx_available    │ fx_volatility has a value      │ it's NaN │
  ├─────────────────┼────────────────────────────────┼──────────┤
  │ food_available  │ food_price_anomaly has a value │ it's NaN │
  ├─────────────────┼────────────────────────────────┼──────────┤
  │ gdp_available   │ gdp_growth has a value         │ it's NaN │
  └─────────────────┴────────────────────────────────┴──────────┘

  Why this matters: A country that stops reporting economic data is itself a signal — failed states and countries in crisis often stop submitting statistics. The absence of data is informative!

  Step 1h: Quality checks

  run_quality_checks() (line 290) runs assertions:
  - Every country's months are in order (monotonically increasing)
  - No duplicate (country, month) rows
  - The lag was actually applied (first month is NaN)
  - V-Dem values are within [0, 1]
  - Reports coverage % for each source

  Step 1i: Quality report + export

  - Writes a JSON report to data/logs/member_b_quality_report.json with missingness %, coverage %, and value ranges
  - Saves the final table to data/intermediate/structural_features.parquet

  ---
  The 6 Data Sources — Each Explained in Detail

  Source 1: V-Dem (Varieties of Democracy) — ingest_vdem.py

  What it is: Expert-coded democracy scores for every country, published once per year by the University of Gothenburg. Think of it as a "report card" for how democratic each country is.

  Raw file: V-Dem-CY-Full+Others-v16.csv (~300MB, 4000+ columns). Must be downloaded manually from v-dem.net.

  The pipeline (6 stages):

  Stage 1 — Load & filter (line 67): Reads the CSV but only picks 14 columns (out of 4000+) plus country_text_id and year. Filters to years 2009–2024 (2009 is a buffer year, explained later).

  The 14 selected features (all scored 0 to 1):

  ┌───────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │      Feature      │                                          What it measures                                           │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_libdem        │ Liberal democracy (the "main" score)                                                                │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_polyarchy     │ Electoral democracy (are elections fair?)                                                           │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_civlib        │ Civil liberties (can people speak freely?)                                                          │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_rule          │ Rule of law (do courts work?)                                                                       │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_corr          │ Political corruption (higher = more corrupt)                                                        │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_clphy         │ Physical integrity (freedom from state violence)                                                    │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_clpol         │ Political civil liberties                                                                           │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_freexp_altinf │ Freedom of expression & media                                                                       │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2xcs_ccsi        │ Civil society strength                                                                              │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_regime        │ Regime type (0=closed autocracy, 1=electoral autocracy, 2=electoral democracy, 3=liberal democracy) │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_partipdem     │ Participatory democracy                                                                             │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2xnp_regcorr     │ Regime corruption                                                                                   │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_execorr       │ Executive corruption                                                                                │
  ├───────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ v2x_frassoc_thick │ Freedom of association                                                                              │
  └───────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Stage 2 — Expand annual to monthly (line 105): V-Dem gives one score per country per year. But our panel is monthly. So each year's score is repeated 12 times (once for each month).

  Crucially, this is NOT interpolation. If a country scored 0.7 in 2019 and 0.6 in 2020, the values are:
  2019-01: 0.7, 2019-02: 0.7, ..., 2019-12: 0.7
  2020-01: 0.6, 2020-02: 0.6, ..., 2020-12: 0.6
  NOT a smooth transition from 0.7 to 0.6. Why? Because the experts coded it once — any in-between values would be fabricated.

  Stage 3 — Map country codes (line 130): V-Dem uses ISO3 codes (like "GBR", "USA"). Our panel uses Gleditsch-Ward numeric codes (like 200, 2). This step converts via the crosswalk dictionary. Countries with no GW mapping (typically dissolved/historical entities) are dropped and logged.

  Stage 4 — Forward-fill (line 276): If V-Dem v15 only covers through 2023, the 2023 values are carried forward into 2024. This is the same "repeat, don't interpolate" logic.

  Stage 5 — One-hot encode regime type (line 151): The v2x_regime column has values {0, 1, 2, 3}. These are categories, not numbers (3 isn't "3 times more" than 1). So they get converted to 4 binary columns:
  regime_type_0 = 1 if closed autocracy, else 0
  regime_type_1 = 1 if electoral autocracy, else 0
  regime_type_2 = 1 if electoral democracy, else 0
  regime_type_3 = 1 if liberal democracy, else 0

  Stage 6 — Derive composite features (line 172): Three new features are calculated:

  1. governance_deficit = 1 − average(libdem, polyarchy, rule_of_law). Higher = weaker institutions. Logic: weak institutions create conditions for violence.
  2. repression_index = average(corruption, exec_corruption) + (1 − physical_integrity) / 2. Captures how much a state relies on coercion. Logic: repressive states generate grievances that fuel conflict.
  3. libdem_yoy_change = this month's libdem − the same month last year's libdem (12-month diff). Detects rapid democratisation OR democratic backsliding — both associated with elevated conflict risk.

  Stage 7 — Validate & flag (lines 206, 228): Asserts all [0,1] indices are actually in [0,1]. Also logs any country-month where |Δlibdem| > 0.15 — these big jumps might be coups, revolutions, or data errors.

  Stage 8 — Drop buffer year: 2009 was loaded only so the year-on-year change could be computed for Jan 2010. Now it's dropped.

  ---
  Source 2: REIGN (Rulers, Elections, and Irregular Governance) — ingest_reign.py

  What it is: Monthly leader-level data for every country. Who's in charge, how they got there, how long they've been there, are they military, when was the last election, did a coup happen.

  Raw file: REIGN_2021_8.csv. Critical problem: REIGN stopped being updated in August 2021. So we have data from 2010 through Aug 2021, but nothing after.

  The pipeline (7 stages):

  Stage 1 — Load (line 73): Reads the CSV, selects 15 columns. REIGN already uses Gleditsch-Ward codes natively (ccode), so no crosswalk needed.

  Stage 2 — Create temporal index (line 108): Combines year and month columns into year_month string format ("2015-03"). Renames ccode to gwcode.

  Stage 3 — Handle the coverage gap (line 119): This is the most complex part. After August 2021, there's NO data. The pipeline uses "Option A" from the project spec:

  For slow-moving features (things that change rarely):
  - government (regime type string), elected, male, militarycareer, leader, prev_conflict
  - These are forward-filled: the last known value (Aug 2021) is carried forward to Dec 2024
  - Logic: If someone was a military dictator in Aug 2021, they're probably still a military dictator in Sep 2021. (Unless a coup happened — handled separately!)

  For fast-changing/event features (things that are discrete events):
  - loss (did leader lose power?), irregular (was there an irregular transition?)
  - These are set to NaN post-cutoff
  - Logic: You can't assume "no coup happened" just because you don't have data. NaN = "we don't know"

  Special handling for age and tenure:
  - age: Forward-filled, then incremented by 1/12 per month (a leader ages ~0.083 years per month)
  - tenure_months: Forward-filled, then incremented by 1 per month (under the assumption the same leader stays)

  The function builds a complete skeleton for all countries × all months, merges the existing data onto it, then applies these fill strategies.

  Stage 4 — Derive features (line 213):

  1. leader_age_risk: Binary flag. = 1 if leader is under 40 or over 75. Based on research showing these age groups correlate with regime instability.
  2. months_since_election: How many months since the last election? Computed from the lastelection date field. Longer gaps → weaker accountability.
  3. regime_change: Binary flag. = 1 if the government string this month differs from last month. Detects when a country's regime type changed (e.g., from "Presidential Democracy" to "Military Junta").
  4. coup_event: Binary flag derived from the irregular field. Directly indicates coups/irregular transitions.

  Stage 5 — One-hot encode regime type (line 283): REIGN's government column has ~14 string categories like "Presidential Democracy", "Military", "Parliamentary Democracy", etc. Each becomes a binary column (reign_regime_Presidential Democracy, reign_regime_Military, etc.).

  Stage 6 — Detect structural breaks (line 301): This is sophisticated:

  - Combines regime_change and coup_event into a unified structural_break indicator
  - Computes months_since_structural_break: a running count of months since the last coup/regime change. This captures the "instability window" — the months after a political shock when conflict risk is highest.
  - Creates reign_ffill_reliable: For post-August 2021 data, this flag = 0 if a structural break happened within 6 months of the cutoff. Logic: if a country was already unstable right before REIGN stopped, the forward-filled values are probably wrong by now.

  Stage 7 — Deduplicate & select (line 449): Some REIGN editions have duplicate country-months. These are removed (keeping the last occurrence). Only the engineered features are kept; raw columns like leader (name string) are dropped.

  ---
  Source 3a: Exchange Rates (IMF IFS) — ingest_exchange_rates()

  What it is: Monthly exchange rate (domestic currency per 1 USD) for each country. A proxy for macroeconomic stability.

  Raw file: imf_exchange_rates.csv. The function handles 3 different IMF download formats automatically:
  1. IMF Data Explorer wide format (periods as columns like "2020-M01")
  2. IMF SDMX-CSV long format (with REF_AREA, TIME_PERIOD, OBS_VALUE)
  3. Generic format (iso3, year, month, exchange_rate)

  Processing:

  1. Format detection & parsing: Inspects column names to determine which format. For the wide format, it melts period columns into rows. Filters to "Domestic currency per USD, monthly, period average".
  2. Map to GW codes: ISO3 or IMF numeric codes → GW via the crosswalk.
  3. Feature engineering (5 derived features):

    - fx_pct_change: Month-on-month % change in exchange rate. A jump means sudden depreciation.
    - fx_volatility: 3-month rolling standard deviation of fx_pct_change (with min_periods=2). This is a "realised volatility" measure — how jittery has the currency been recently?
    - fx_volatility_log: log(1 + volatility). Volatility is extremely right-skewed (lots of calm months, rare crisis spikes). The log transform compresses the tail so the model can handle it better.
    - fx_depreciation_flag: Binary. = 1 if fx_pct_change exceeds the country's own mean + 2 standard deviations. This is country-specific — a 5% depreciation in Turkey (routine) won't trigger it, but 5% in Switzerland (unprecedented) will.
    - fx_pct_change_zscore: An expanding z-score of fx_pct_change within each country. Uses expanding(min_periods=6) so it only considers past data for each point — no information leakage.
  4. Gap-fill: Forward-fills up to 3 months to handle reporting delays.

  ---
  Source 3b: GDP Growth (World Bank WDI) — ingest_gdp()

  What it is: Annual GDP growth rate (%) from the World Bank.

  Processing:

  1. Load & map: Reads CSV, maps ISO3 → GW. Loads from 2005 onward to have a buffer for the rolling window.
  2. Feature engineering:

    - gdp_growth_deviation: This is the key feature. It's a z-score of GDP growth against the country's own 5-year rolling mean. If Ethiopia normally grows at 8% but grew at 3% this year, that's a big negative deviation — even though 3% would be amazing for Japan. Country-specific normalisation.
    - gdp_negative_shock: Binary. = 1 if gdp_growth_deviation < −1.0. Based on research (Miguel et al., 2004) showing that negative income shocks are the single strongest economic predictor of civil conflict.
  3. Expand annual → monthly: Same repeat logic as V-Dem.
  4. Forward-fill: If World Bank hasn't published 2024 data yet, 2023 values carry forward.

  ---
  Source 3c: Food Prices (FAO FAOSTAT) — ingest_food_prices()

  What it is: Monthly Consumer Price Index for food, from the UN Food and Agriculture Organization.

  Processing:

  1. Load: Handles both ZIP files (FAOSTAT bulk downloads) and plain CSVs.
  2. Format detection: FAOSTAT has a unique format with M49 area codes and month names as strings ("January", "February"). The function detects this and converts:
    - M49 codes → ISO3 (via pycountry library or a built-in _m49_map.py fallback)
    - Month names → numbers (1-12)
    - FAOSTAT "Months Code" 7001-7012 → 1-12
  3. Feature engineering (4 derived features):

    - food_price_anomaly: Current CPI / 12-month rolling mean. Values > 1 mean food is more expensive than the recent trend. A value of 1.2 means food is 20% above its recent average.
    - food_price_anomaly_log: log(anomaly). Handles extreme values during hyperinflation. log(1.3) > 0 (above trend), log(0.9) < 0 (below trend).
    - food_cpi_acceleration: Is food inflation speeding up? This is the month-on-month change in the year-on-year rate. A stronger predictor of unrest than the food price level itself.
    - food_price_spike: Binary. = 1 if food_price_anomaly > 1.15 (food is 15%+ above its 12-month trend). The 15% threshold is based on Bellemare (2015) research linking this level to protest onset.

  Note: The raw CPI gets a t-1 lag before computing rolling windows (line 577) to prevent look-ahead in the feature engineering itself.

  ---
  Source 4: Powell & Thyne Coups — ingest_powell_thyne.py

  What it is: An event-level dataset of every coup attempt (successful or failed) globally since 1950. Updated within weeks of new events. Uses GW codes natively.

  Why it's needed: REIGN stopped in Aug 2021, but coups kept happening: Myanmar (Feb 2021), Guinea (Sep 2021), Sudan (Oct 2021), Burkina Faso (Jan 2022), Niger (Jul 2023), Gabon (Aug 2023). Powell & Thyne fills this gap.

  Processing:

  1. Load: Reads TSV. coup column: 1 = failed attempt, 2 = successful coup.
  2. Aggregate to country-month (line 142): Raw data is event-level (one row per coup). This aggregates to country-month with:
    - pt_coup_event: Any coup attempt this month (binary)
    - pt_coup_successful: Successful coup this month (binary)
    - pt_coup_failed: Failed attempt this month (binary)
    - pt_coup_count: Number of attempts this month
  3. Derive coup history (line 181):
    - pt_cumulative_coups: Running total of all coups for this country since 2010. Captures "coup-prone" states.
    - pt_months_since_coup: Running count of months since last coup. Captures the "coup trap" — countries that had a recent coup are more likely to have another.
  4. Integrate with REIGN (line 241):
    - Before Aug 2021: REIGN's coup_event is authoritative (kept as-is). Powell & Thyne provides supplementary history features only.
    - After Aug 2021: REIGN's coup_event is NaN. Powell & Thyne's pt_coup_event replaces it.
    - Critical patch: If Powell & Thyne finds a successful post-2021 coup for a country, reign_ffill_reliable is set to 0 for that country from the coup month onward. Because a successful coup means the leader/regime changed — so all the forward-filled REIGN data (name, age, tenure, regime type) is definitively wrong.

  ---
  The Country Code Crosswalk — crosswalk.py

  This is the Rosetta Stone of the pipeline. Different datasets use different country codes:
  - V-Dem, World Bank, FAO → ISO3 (e.g., "GBR", "USA")
  - REIGN, Powell & Thyne → Gleditsch-Ward numeric (e.g., 200, 2)
  - IMF → IMF numeric codes (e.g., 112, 111)
  - FAO → M49 numeric codes (e.g., 826, 840)

  The crosswalk provides ~100 GW ↔ ISO3 mappings and handles edge cases:
  - South Sudan (GW 626): Didn't exist before July 2011. Pre-2011 data mapped to Sudan (625).
  - Kosovo (GW 347): Not in the official GW system; gets a custom code.
  - Serbia/Montenegro: Split in June 2006. Montenegro (341) only valid after that.

  ---
  The Config File — member_b_config.yaml

  Sets all the parameters in one place:

  panel:
    year_min: 2010
    year_max: 2024
    lag: 1  # months

  feature_selection:
    max_missingness: 0.40   # drop features missing >40%
    min_variance: 0.001     # drop near-zero variance features
    max_correlation: 0.90   # drop one of a highly correlated pair

  ---
  Summary: The Flow in One Diagram

   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  V-Dem CSV   │  │  REIGN CSV   │  │  IMF/WB/FAO  │  │Powell & Thyne│
   │  (annual,    │  │  (monthly,   │  │    CSVs       │  │  (event-     │
   │   ISO3)      │  │   GW codes,  │  │  (mixed freq, │  │   level,     │
   │              │  │   stops 2021)│  │   mixed codes)│  │   GW codes)  │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                  │                  │
     ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
     │ Expand  │      │ Fill gap  │     │ Parse fmt │     │ Aggregate │
     │ annual  │      │ (ffill    │     │ Map codes │     │ to monthly│
     │ →monthly│      │  slow,    │     │ Derive    │     │ Derive    │
     │ Map→GW  │      │  NaN fast)│     │ features  │     │ history   │
     │ Derive  │      │ Derive    │     │           │     │           │
     │ features│      │ features  │     │           │     │           │
     └────┬────┘      └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                  │                  │
          └────────┬────────┴────────┬─────────┘                  │
                   │                 │                             │
            ┌──────▼──────────────────▼──────┐                     │
            │      Panel Skeleton            │                     │
            │  (all countries × all months)  │                     │
            │      LEFT JOIN all sources     │◄────────────────────┘
            └──────────────┬─────────────────┘
                           │
                ┌──────────▼──────────┐
                │ Cross-validate      │
                │ V-Dem vs breaks     │
                │ (stale flag)        │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │ Apply t−1 lag       │
                │ (shift ALL features │
                │  forward 1 month)   │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │ Encode missingness  │
                │ (5 binary flags)    │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │ Quality checks      │
                │ + JSON report       │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │ Export Parquet       │
                │ (~32,400 rows       │
                │  × ~60 columns)     │
                └─────────────────────┘

  ---
  The 3 Most Important Design Decisions to Know for Presenting

  1. Repeat, don't interpolate: Annual data (V-Dem, GDP) is copied 12 times — never smoothed. Because the measurement literally happened once. Inventing in-between values would be lying.
  2. t−1 lag applied centrally: Every feature is shifted by 1 month in ONE place (structural_features.py), not scattered across loaders. This prevents temporal leakage (using future information to predict the present) and makes it easy to audit.
  3. Missingness is a feature, not a problem: Instead of just leaving NaN cells, binary flags explicitly tell the model "this data source had nothing here." A country that stops reporting economic data is itself a warning sign.