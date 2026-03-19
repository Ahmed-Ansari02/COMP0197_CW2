# Structural & Contextual Feature Engineering: Methodology Note

## Data Sources

We assemble structural features from three source families that capture the slow-moving background conditions theorised to mediate conflict risk:

**V-Dem (Varieties of Democracy, v15).** Expert-coded governance indicators at annual resolution covering 179 countries from 1789 to 2023. We select 14 indices spanning liberal democracy, electoral quality, civil liberties, rule of law, corruption, and freedom of expression. Annual values are repeated (not interpolated) to monthly resolution to preserve the actual measurement cadence — interpolation would fabricate sub-annual variation absent from the expert coding process (Coppedge et al., 2024).

**REIGN (Rulers, Elections, and Irregular Governance).** Monthly leader-level data including regime type, leader demographics, election timing, and irregular power transitions (Bell, 2016). REIGN provides the only source of monthly-resolution political dynamics in our feature set. Data collection ceased in August 2021; we handle this via forward-fill of slow-moving features (regime type, leader demographics) and mark event features (coups, irregular transitions) as missing post-cutoff.

**Economic Covariates.** Three complementary streams: (i) exchange rate volatility from IMF International Financial Statistics, computed as the 3-month rolling standard deviation of month-on-month percentage changes; (ii) GDP growth from the World Bank, with deviation from country-specific 5-year trends; (iii) food CPI from FAO FAOSTAT, with anomaly ratios and acceleration metrics linked empirically to protest onset (Bellemare, 2015).

## Panel Structure

All features are mapped to the ViEWS Gleditsch-Ward country code system and indexed by (gwcode, year_month). The crosswalk handles edge cases including the South Sudan independence (July 2011), Kosovo (custom GW code), and the Serbia-Montenegro separation (June 2006).

## Temporal Lag

All features are lagged by t−1 (one calendar month) to prevent temporal leakage. At row (country, June 2020), every feature value corresponds to the May 2020 observation. This ensures the model never accesses contemporaneous information during training or evaluation.

## Derived Features

We compute three composite V-Dem indices: (i) a **governance deficit** score (1 − mean of liberal democracy, electoral democracy, and rule of law indices), capturing institutional weakness as a permissive condition for violence (Hegre et al., 2001); (ii) a **repression index** combining corruption and physical integrity violations, operationalising the coercive state capacity channel (Gurr, 1970); and (iii) **year-on-year democracy change** to detect backsliding or rapid transition, both associated with elevated conflict risk (Mansfield & Snyder, 2005).

From REIGN, we derive **leader age risk** (binary flag for age < 40 or > 75), **months since last election** (legitimacy decay proxy), **regime change** detection, and **coup event** indicators.

Economic derived features include **sudden depreciation flags** (country-specific 2σ threshold), **GDP growth deviation** from the country's own 5-year trend (captures country-specific recessions rather than absolute levels), and **food price acceleration** (month-on-month change in year-on-year inflation rate).

## Missing Data Treatment

Missingness is encoded as informative binary indicators for each source family, as data absence may itself signal institutional failure. V-Dem gaps (~12% of countries) are left as NaN. REIGN post-2021 slow features are forward-filled; event features are NaN. Exchange rate gaps are forward-filled up to 3 months. No imputation is applied to food prices for countries without FAO coverage.

## Normalisation

No normalisation is applied. V-Dem indices are already on [0, 1] by construction. Member C applies z-scoring after the train/test split to prevent information leakage from the test set into scaling parameters.
