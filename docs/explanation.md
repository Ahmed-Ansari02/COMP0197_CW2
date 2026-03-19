# Member B Pipeline — The Full Explanation
## (Written So Anyone Can Understand It)

---

# Part 1: What Is This Project Actually Doing?

## The Big Picture

Imagine you're a weather forecaster, but instead of predicting rain, you're
predicting **political violence** — wars, protests, coups, people getting hurt.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   "Will Country X become dangerous in the next 1, 3, 6 months?" │
  │                                                                 │
  │   We don't give a YES/NO answer.                                │
  │   We give a PROBABILITY — like a weather forecast:              │
  │                                                                 │
  │   "There's a 73% chance of serious instability in Sudan         │
  │    within the next 3 months"                                    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

We do this for **every country, every month**, from 2010 to 2024.

## The Three Team Members

Think of it like building a house. Each person brings different materials:

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        THE TEAM                                      │
  │                                                                      │
  │  MEMBER A               MEMBER B (YOU)         MEMBER C              │
  │  ─────────              ──────────────         ─────────             │
  │  "What's happening      "What kind of          "What are people      │
  │   RIGHT NOW?"            country IS this?"      SAYING about it?"    │
  │                                                                      │
  │  • Battles today        • Is it a democracy?   • News sentiment      │
  │  • Protests today       • Who's the leader?    • Geopolitical risk   │
  │  • Deaths today         • Economy good/bad?    • Media hostility     │
  │  • Armed groups         • Free press?          • Also merges         │
  │                         • Food affordable?       everything together │
  │                                                                      │
  │  Sources:               Sources:                Sources:             │
  │  ACLED, GDELT, UCDP     V-Dem, REIGN, IMF,     GPR, GDELT tone     │
  │                          World Bank, FAO                             │
  │                                                                      │
  │  Like checking           Like checking the      Like checking        │
  │  the SYMPTOMS            PATIENT'S HISTORY      the RUMOURS          │
  └──────────────────────────────────────────────────────────────────────┘
```

**Member A** tracks the fires (active conflict events).
**Member B** (that's us!) tracks the *conditions that make fires possible* — dry wood, no fire department, etc.
**Member C** tracks what people are *saying* about the fire risk, and glues everything together.

---

# Part 2: What Does Member B Actually Do?

## The Analogy

Imagine you're a doctor. Before you can diagnose a patient, you need their
**background health information**:

- Medical history (democracy indices = "how healthy is this government?")
- Who their doctor is and how long they've had them (leader characteristics)
- Can they afford medicine? (economic indicators)
- Are they eating properly? (food prices)

That's what Member B does — we collect the **background conditions** of every
country that might make violence more or less likely.

## Our Three Data Sources

```
  ┌─────────────────────────────────────────────────────────────┐
  │                  MEMBER B's THREE SOURCES                    │
  │                                                              │
  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
  │  │   V-DEM     │  │   REIGN     │  │   ECONOMIC       │    │
  │  │             │  │             │  │                  │    │
  │  │ "Report     │  │ "Report     │  │ "Report card     │    │
  │  │  card on    │  │  card on    │  │  on the          │    │
  │  │  the        │  │  the        │  │  WALLET"         │    │
  │  │  GOVERNMENT"│  │  LEADER"    │  │                  │    │
  │  │             │  │             │  │ • Exchange rate   │    │
  │  │ • Democracy │  │ • Who leads │  │ • GDP growth     │    │
  │  │ • Freedom   │  │ • How long  │  │ • Food prices    │    │
  │  │ • Corruption│  │ • Elected?  │  │                  │    │
  │  │ • Rule of   │  │ • Military? │  │ Monthly/Annual   │    │
  │  │   law       │  │ • Coups?    │  │ IMF, World Bank, │    │
  │  │             │  │             │  │ FAO              │    │
  │  │ Annual      │  │ Monthly     │  │                  │    │
  │  │ (once/year) │  │ (once/month)│  │                  │    │
  │  └─────────────┘  └─────────────┘  └──────────────────┘    │
  │                                                              │
  │  Updated by         Stopped in         Mix of monthly        │
  │  academics          Aug 2021!          and annual            │
  │  (experts vote      (big problem       data                  │
  │   on scores)         we handle)                              │
  └─────────────────────────────────────────────────────────────┘
```

---

# Part 3: The Data Sources — One by One

## 3.1 V-Dem (Varieties of Democracy)

### What it is

Imagine 3,000 political science professors around the world, and every year
you ask them: "On a scale of 0 to 1, how democratic is France? How corrupt
is Nigeria? How free is the press in China?"

They all give scores, and the average becomes the official number.

**That's V-Dem.** Expert-coded scores for ~180 countries, updated once per year.

### The features we extract

```
  ┌────────────────────────────────────────────────────────────────┐
  │                    V-DEM FEATURES                               │
  │                                                                 │
  │  Each is a score from 0.0 to 1.0                               │
  │  (0 = terrible, 1 = excellent)                                 │
  │                                                                 │
  │  v2x_libdem ─────── "How liberal-democratic is this country?"  │
  │  v2x_polyarchy ──── "How well do elections work?"              │
  │  v2x_civlib ─────── "Can citizens live freely?"                │
  │  v2x_rule ───────── "Does rule of law exist?"                  │
  │  v2x_corr ───────── "How corrupt?" (⚠ higher = MORE corrupt)  │
  │  v2x_clphy ──────── "Freedom from state violence?"             │
  │  v2x_clpol ──────── "Political civil liberties?"               │
  │  v2x_freexp_altinf─ "Can people speak freely / get real news?" │
  │  v2xcs_ccsi ─────── "Is there a strong civil society?"         │
  │  v2x_regime ─────── "What type of regime?" (0, 1, 2, or 3)    │
  │  v2x_partipdem ──── "Can people participate in politics?"      │
  │  v2xnp_regcorr ──── "How corrupt is the regime?"               │
  │  v2x_execorr ────── "How corrupt is the president/PM?"         │
  │  v2x_frassoc_thick─ "Can people form organisations?"           │
  │                                                                 │
  │  ALSO: v2x_regime is a CATEGORY, not a score:                  │
  │    0 = Closed autocracy    (e.g., North Korea)                 │
  │    1 = Electoral autocracy (e.g., Russia — has "elections")    │
  │    2 = Electoral democracy (e.g., India)                       │
  │    3 = Liberal democracy   (e.g., Norway)                      │
  └────────────────────────────────────────────────────────────────┘
```

### The big challenge: Annual → Monthly

V-Dem gives us ONE number per country per YEAR. But our prediction model
needs ONE number per country per MONTH.

**Wrong approach (interpolation):**
```
  Year:    2019          2020
  Score:   0.60          0.70

  WRONG — "Interpolate" means we'd guess the months in between:
  Jan=0.61  Feb=0.62  Mar=0.63 ... Nov=0.69  Dec=0.70

  This is WRONG because the experts only measured it ONCE.
  We'd be INVENTING fake data between measurements.

  It's like if your doctor weighed you in January (70kg) and
  December (75kg), and someone wrote in your chart that you
  weighed 72.5kg in June. Nobody measured that! It's made up.
```

**Correct approach (repeat/forward-fill):**
```
  Year:    2019                              2020
  Score:   0.60                              0.70

  RIGHT — Just repeat the same value for all 12 months:
  Jan=0.60  Feb=0.60  Mar=0.60 ... Dec=0.60  Jan=0.70 ...

  We're saying: "The best information we have for any month in 2019
  is the 2019 annual score." Honest about what we know.

  Like saying: "Last time we checked, your weight was 70kg."
  True and honest, even if slightly outdated.
```

Here's the code that does it:

```
  FILE: src/data/ingest_vdem.py — function _expand_annual_to_monthly()

  What it does:

  INPUT (3 rows — one per year):
  ┌──────────┬──────┬────────────┐
  │ country  │ year │ v2x_libdem │
  ├──────────┼──────┼────────────┤
  │ AFG      │ 2019 │ 0.120      │
  │ AFG      │ 2020 │ 0.095      │
  │ AFG      │ 2021 │ 0.040      │  ← Taliban takeover
  └──────────┴──────┴────────────┘

  OUTPUT (36 rows — one per month):
  ┌──────────┬────────────┬────────────┐
  │ country  │ year_month │ v2x_libdem │
  ├──────────┼────────────┼────────────┤
  │ AFG      │ 2019-01    │ 0.120      │
  │ AFG      │ 2019-02    │ 0.120      │  ← Same value
  │ AFG      │ ...        │ 0.120      │     repeated
  │ AFG      │ 2019-12    │ 0.120      │     12 times
  │ AFG      │ 2020-01    │ 0.095      │  ← Now uses 2020
  │ AFG      │ ...        │ 0.095      │     score
  │ AFG      │ 2020-12    │ 0.095      │
  │ AFG      │ 2021-01    │ 0.040      │  ← Dramatic drop
  │ AFG      │ ...        │ ...        │
  └──────────┴────────────┴────────────┘
```

### Derived (computed) features

We don't just use the raw scores — we combine them to create new
features that capture specific theories about why conflict happens:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                  DERIVED V-DEM FEATURES                         │
  │                                                                 │
  │  1. GOVERNANCE DEFICIT                                         │
  │     ─────────────────                                          │
  │     Formula: 1 - average(democracy, elections, rule_of_law)    │
  │                                                                 │
  │     What it means: "How BAD are the institutions?"             │
  │                                                                 │
  │     Democracy=0.8, Elections=0.7, Rule of law=0.9              │
  │     → Average = 0.8 → Deficit = 1 - 0.8 = 0.2 (low deficit)  │
  │                                                                 │
  │     Democracy=0.2, Elections=0.1, Rule of law=0.1              │
  │     → Average = 0.13 → Deficit = 0.87 (high deficit = danger) │
  │                                                                 │
  │     Theory: Weak institutions create space for violence        │
  │             (Hegre et al., 2001)                               │
  │                                                                 │
  │  2. REPRESSION INDEX                                           │
  │     ────────────────                                           │
  │     Formula: (avg(corruption, exec_corruption)                 │
  │              + (1 - physical_safety)) / 2                      │
  │                                                                 │
  │     What it means: "How oppressive is the state?"              │
  │                                                                 │
  │     Corrupt government + state violence against citizens       │
  │     = people have reasons to rebel (grievance theory, Gurr)    │
  │                                                                 │
  │  3. DEMOCRACY CHANGE (Year-on-Year)                            │
  │     ─────────────────────────────────                          │
  │     Formula: this_year's_democracy - last_year's_democracy     │
  │                                                                 │
  │     What it means: "Is democracy getting BETTER or WORSE?"     │
  │                                                                 │
  │     Positive change = democratising (hopeful but unstable)     │
  │     Negative change = BACKSLIDING (very dangerous!)            │
  │                                                                 │
  │     Theory: Both rapid democratisation AND autocratisation     │
  │             increase conflict risk (Mansfield & Snyder, 2005)  │
  └─────────────────────────────────────────────────────────────────┘
```

### One-hot encoding of regime type

The regime type is a category (0, 1, 2, 3), not a number you can average.
"Type 2" is not "twice as much" as "Type 1." So we convert it:

```
  BEFORE (one column, values 0-3):
  ┌─────────┬────────────┐
  │ country │ v2x_regime │
  ├─────────┼────────────┤
  │ PRK     │ 0          │  (North Korea = closed autocracy)
  │ RUS     │ 1          │  (Russia = electoral autocracy)
  │ IND     │ 2          │  (India = electoral democracy)
  │ NOR     │ 3          │  (Norway = liberal democracy)
  └─────────┴────────────┘

  AFTER (four columns, each is 0 or 1):
  ┌─────────┬──────────┬──────────┬──────────┬──────────┐
  │ country │ regime_0 │ regime_1 │ regime_2 │ regime_3 │
  ├─────────┼──────────┼──────────┼──────────┼──────────┤
  │ PRK     │ 1        │ 0        │ 0        │ 0        │
  │ RUS     │ 0        │ 1        │ 0        │ 0        │
  │ IND     │ 0        │ 0        │ 1        │ 0        │
  │ NOR     │ 0        │ 0        │ 0        │ 1        │
  └─────────┴──────────┴──────────┴──────────┴──────────┘

  Why? The AI model treats numbers mathematically.
  If we kept 0,1,2,3 it might think "3 is three times more than 1"
  which makes no sense for regime types.
  One-hot encoding says "you're either THIS type or you're NOT."
```

---

## 3.2 REIGN (Rulers, Elections, and Irregular Governance)

### What it is

A dataset that tracks, month by month:
- **Who** is the leader of each country
- **How** they came to power (elected? coup? inheritance?)
- **How long** they've been in charge
- **What type** of government it is
- Whether there were **coups or irregular power changes**

### The big problem: REIGN stopped in August 2021

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    THE REIGN GAP PROBLEM                        │
  │                                                                 │
  │  ◄──── REIGN has data ────►│◄──── NO DATA ────────────────►   │
  │                             │                                   │
  │  2010 ──────────────── 2021-08 ─────────────────────── 2024   │
  │  ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
  │                             │                                   │
  │                        Data collection                          │
  │                        STOPPED here                             │
  │                                                                 │
  │  But our model needs data through 2024!                        │
  │  What do we do?                                                │
  └─────────────────────────────────────────────────────────────────┘
```

**Our solution (Option A):**

```
  We split REIGN features into two types:

  SLOW-CHANGING features           FAST-CHANGING features
  (things that rarely change)      (things that can happen any month)
  ──────────────────────────       ──────────────────────────────────
  • Government type                • Did a coup happen?
  • Is the leader male?            • Did the leader lose power?
  • Military background?
  • Was the leader elected?

  For SLOW features:               For FAST features:
  We FORWARD-FILL                  We set to NaN (unknown)
  (assume nothing changed)         (we CAN'T guess these)

  Example:                         Example:
  2021-08: Military Junta          2021-08: No coup (0)
  2021-09: Military Junta ← fill   2021-09: ??? (NaN)
  2021-10: Military Junta ← fill   2021-10: ??? (NaN)
  ...                              ...
  2024-12: Military Junta ← fill   2024-12: ??? (NaN)

  It's like if your friend stopped texting you:
  - You can ASSUME they still live at the same address (slow change)
  - You CANNOT assume they didn't get a new job (fast change)
```

### REIGN derived features

```
  ┌─────────────────────────────────────────────────────────────────┐
  │               DERIVED REIGN FEATURES                            │
  │                                                                 │
  │  1. LEADER AGE RISK                                            │
  │     ────────────────                                           │
  │     Is the leader very young (< 40) or very old (> 75)?        │
  │     → Flag as "risky" (binary: 0 or 1)                        │
  │                                                                 │
  │     Why? Very young leaders may lack legitimacy.               │
  │          Very old leaders may face succession crises.           │
  │                                                                 │
  │  2. MONTHS SINCE LAST ELECTION                                 │
  │     ──────────────────────────                                 │
  │     How many months since the country last held elections?      │
  │                                                                 │
  │     Why? The longer since an election, the weaker the          │
  │          democratic legitimacy of the government.               │
  │          Like a "mandate expiration timer."                     │
  │                                                                 │
  │  3. REGIME CHANGE                                              │
  │     ─────────────                                              │
  │     Did the type of government change this month?              │
  │     (e.g., from "democracy" to "military junta")               │
  │                                                                 │
  │     Why? Regime changes are one of the strongest predictors    │
  │          of subsequent violence. Everything is in flux.         │
  │                                                                 │
  │  4. COUP EVENT                                                 │
  │     ──────────                                                 │
  │     Was there an irregular (non-constitutional) power change?  │
  │                                                                 │
  │     Why? Coups often trigger cascading violence.               │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 3.3 Economic Covariates

### Why economics matters for conflict

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  When the economy fails, people get desperate.                 │
  │  When people get desperate, they protest.                      │
  │  When protests are repressed, violence escalates.              │
  │                                                                 │
  │  Economy bad → People angry → Protests → Crackdown → Conflict │
  │                                                                 │
  │  Three economic signals we track:                              │
  │                                                                 │
  │  💱 CURRENCY COLLAPSE     📉 GDP SHRINKING     🍞 FOOD TOO     │
  │                                                 EXPENSIVE      │
  │  "Is the money             "Is the whole        "Can people    │
  │   becoming worthless?"      economy shrinking?"  afford to eat?"│
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### 3.3a Exchange Rates (IMF)

```
  What we get:  "How much local currency buys 1 US dollar" each month

  What we compute:

  1. fx_pct_change — "How much did the currency change this month?"

     Month 1: 100 units per dollar
     Month 2: 105 units per dollar
     → 5% depreciation (currency lost value)

  2. fx_volatility — "How wild are the swings?" (3-month rolling std dev)

     Stable currency:  100, 101, 100, 99, 101  → low volatility
     Unstable currency: 100, 120, 95, 150, 80  → HIGH volatility

     Think of it like a heart rate monitor:
     Normal: ∿∿∿∿∿ (steady)
     Crisis: ∿⌇∿⌇⌇∿⌇ (erratic)

  3. fx_depreciation_flag — "Is this an ABNORMAL crash?"

     We compare this month's change to the country's own history.
     If it's more than 2 standard deviations above normal → FLAG IT.

     Why country-specific? A 5% drop is NORMAL for Argentina.
     A 5% drop is a CRISIS for Switzerland.
```

### 3.3b GDP Growth (World Bank)

```
  What we get:  Annual GDP growth percentage per country

  What we compute:

  1. gdp_growth — The raw number (e.g., "+3.2%" or "-5.1%")
     Expanded to monthly by repetition (same as V-Dem — annual data)

  2. gdp_growth_deviation — "Is this unusual FOR THIS COUNTRY?"

     We compare the current year to the country's own 5-year trend.

     Example:
     Nigeria usually grows at +4% per year.
     This year it grew at +1%.
     → Deviation is NEGATIVE (below its own trend)
     → Even though +1% sounds OK, for Nigeria it's bad.

     VS.

     Japan usually grows at +1% per year.
     This year it grew at +1%.
     → Deviation is ZERO (normal for Japan)

     This is a z-score: (actual - trend_mean) / trend_std_dev

     Why? Absolute GDP numbers are misleading across countries.
     What matters is: "Is this country doing worse than ITS normal?"
```

### 3.3c Food Prices (FAO)

```
  What we get:  Monthly food Consumer Price Index (CPI) per country

  What we compute:

  1. food_price_anomaly — "Are food prices above the recent trend?"

     Formula: current_CPI / average_CPI_over_last_12_months

     Result = 1.0 → exactly at trend (normal)
     Result = 1.3 → 30% ABOVE trend (food crisis!)
     Result = 0.9 → 10% below trend (deflation)

     Why ratio? Because CPI=200 means different things in different
     countries. The RATIO tells us if it's abnormally high for THAT country.

  2. food_cpi_acceleration — "Is food inflation getting FASTER?"

     Not just "are prices high?" but "are they ACCELERATING?"

     Month 1: prices up 5% year-on-year
     Month 2: prices up 8% year-on-year
     → Acceleration = +3 percentage points

     This is the scariest signal: not just expensive food,
     but food getting MORE expensive FASTER.

     Research link: Bellemare (2015) showed food price spikes
     directly predict protest onset in developing countries.
```

---

# Part 4: The Country Code Problem (Crosswalk)

## Why this matters

Every dataset uses different codes for the same countries:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                THE COUNTRY CODE MESS                            │
  │                                                                 │
  │  The same country (Afghanistan) is identified as:              │
  │                                                                 │
  │  V-Dem:       "AFG"          (ISO 3-letter code)               │
  │  REIGN:        700           (Gleditsch-Ward number)           │
  │  World Bank:  "AFG"          (ISO 3-letter code)               │
  │  IMF:          512           (IMF numeric code)                │
  │  ViEWS:        700           (Gleditsch-Ward number)           │
  │                                                                 │
  │  We need EVERYTHING to use the same code system!               │
  │  Our standard: Gleditsch-Ward (GW) numbers ← ViEWS uses this  │
  └─────────────────────────────────────────────────────────────────┘

  FILE: src/data/crosswalk.py

  The "crosswalk" is just a translation dictionary:

  ISO3 → GW                    GW → ISO3
  ────────                     ────────
  "AFG" → 700                  700 → "AFG"
  "USA" → 2                    2   → "USA"
  "GBR" → 200                  200 → "GBR"
  "NGA" → 475                  475 → "NGA"
  ...196 mappings total
```

### Edge cases (the tricky countries)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                  COUNTRY EDGE CASES                             │
  │                                                                 │
  │  1. SOUTH SUDAN (born July 2011)                               │
  │     ─────────────────────────────                              │
  │     Before July 2011: only SUDAN exists (GW 625)              │
  │     After July 2011:  SUDAN (625) + SOUTH SUDAN (626)         │
  │                                                                 │
  │     ┌────────────┬──────────────────┬──────────────────┐       │
  │     │            │  Before 2011-07  │  After 2011-07   │       │
  │     ├────────────┼──────────────────┼──────────────────┤       │
  │     │  SDN data  │  GW 625          │  GW 625          │       │
  │     │  SSD data  │  GW 625 ← MERGE │  GW 626 ← NEW   │       │
  │     └────────────┴──────────────────┴──────────────────┘       │
  │                                                                 │
  │  2. KOSOVO                                                     │
  │     ──────                                                     │
  │     Declared independence 2008, not universally recognised.    │
  │     Not in the official GW system!                             │
  │     → We assign custom code 347                                │
  │                                                                 │
  │  3. SERBIA / MONTENEGRO (split June 2006)                      │
  │     ────────────────────                                       │
  │     Before 2006: Yugoslavia/Serbia-Montenegro (345)            │
  │     After 2006:  Serbia (345) + Montenegro (341)               │
  │                                                                 │
  │  4. IMF codes (two-step mapping)                               │
  │     ─────────                                                  │
  │     IMF 512 → ISO3 "AFG" → GW 700                             │
  │     IMF code → ISO3 code → GW code (two lookups needed)       │
  └─────────────────────────────────────────────────────────────────┘
```

---

# Part 5: The Pipeline — How It All Fits Together

## The Full Data Flow

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │                        THE FULL PIPELINE                                 │
  │                                                                          │
  │   RAW DATA                  INGESTION               ASSEMBLY             │
  │   ────────                  ─────────               ────────             │
  │                                                                          │
  │   V-Dem CSV ──────► ingest_vdem.py ──────┐                              │
  │   (300MB,annual)     • Select 14 cols     │                              │
  │                      • Expand → monthly   │                              │
  │                      • Map ISO3 → GW      │                              │
  │                      • One-hot regime      │       structural            │
  │                      • Derive composites   ├──────► _features.py        │
  │                                            │        • Build panel        │
  │   REIGN CSV ─────► ingest_reign.py ───────┤        • Left-join all      │
  │   (monthly)          • Create index        │        • Apply t-1 LAG     │
  │                      • Fill gap post-2021  │        • Add missingness   │
  │                      • Derive features     │        • Quality checks    │
  │                      • Encode regime type   │        • Export Parquet    │
  │                                            │            │                │
  │   IMF CSV ───┐                             │            ▼                │
  │   GDP CSV ───┼─► ingest_economic.py ──────┘    structural_features     │
  │   FAO CSV ───┘   • FX volatility                    .parquet           │
  │                   • GDP deviation                       │                │
  │                   • Food anomaly                        ▼                │
  │                                                  Member C merges        │
  │                                                  with A's & C's         │
  │                                                  features               │
  └──────────────────────────────────────────────────────────────────────────┘
```

## Step by step

### Step 1: Ingest each source separately

```
  Each source gets its own module (file) that:

  ① Loads the raw CSV
  ② Selects the columns we need
  ③ Converts country codes to GW numbers
  ④ Creates the "year_month" column (e.g., "2020-06")
  ⑤ Engineers derived features
  ⑥ Validates data ranges

  Output: a clean DataFrame with columns [gwcode, year_month, features...]
```

### Step 2: Build the panel skeleton

```
  The "skeleton" is every possible (country, month) combination:

  ┌─────────┬────────────┐
  │ gwcode  │ year_month │    ~180 countries
  ├─────────┼────────────┤    × 180 months (2010-01 to 2024-12)
  │ 2       │ 2010-01    │    = ~32,400 rows
  │ 2       │ 2010-02    │
  │ 2       │ ...        │    Every row is a unique
  │ 2       │ 2024-12    │    (country, month) pair.
  │ 20      │ 2010-01    │
  │ 20      │ 2010-02    │    Even if nothing happened that month,
  │ ...     │ ...        │    the row EXISTS (with zeros/NaN).
  │ 990     │ 2024-12    │
  └─────────┴────────────┘

  Why? Because "nothing happened" IS information.
  A month with zero protests is different from a missing month.
```

### Step 3: Left-join all sources onto the skeleton

```
  "Left join" means: start with the skeleton, and for each row,
  LOOK UP the matching data from each source.

  SKELETON             V-DEM DATA               RESULT
  ┌──────┬───────┐    ┌──────┬───────┬──────┐   ┌──────┬───────┬──────┐
  │ gw   │ month │    │ gw   │ month │ dem  │   │ gw   │ month │ dem  │
  ├──────┼───────┤    ├──────┼───────┼──────┤   ├──────┼───────┼──────┤
  │ 700  │ 2020-01│   │ 700  │2020-01│ 0.12 │   │ 700  │2020-01│ 0.12 │ ← matched!
  │ 700  │ 2020-02│   │ 700  │2020-02│ 0.12 │   │ 700  │2020-02│ 0.12 │ ← matched!
  │ 999  │ 2020-01│   │      │       │      │   │ 999  │2020-01│ NaN  │ ← no match
  └──────┴───────┘    └──────┴───────┴──────┘   └──────┴───────┴──────┘

  If a source doesn't have data for a country → NaN (missing).
  This is FINE — we track missingness as a feature itself.

  We do this for V-Dem, then REIGN, then FX, then GDP, then food.
  Each join adds more columns to the same panel.
```

### Step 4: Apply the LAG (most critical step!)

This is **the single most important step** in the entire pipeline.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │                    WHY WE NEED LAG                              │
  │                                                                 │
  │  Imagine you're betting on a football match.                   │
  │                                                                 │
  │  CHEATING: You bet AFTER seeing the match stats.               │
  │  → You'd always win, but it's useless in real life.            │
  │                                                                 │
  │  FAIR: You bet BEFORE the match, using only PAST information.  │
  │  → This is what lag ensures.                                   │
  │                                                                 │
  │  Without lag, the model would use June's data to predict       │
  │  June's violence. That's cheating! In the real world, when     │
  │  you make a prediction for June, June hasn't happened yet.     │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  HOW LAG WORKS:

  BEFORE lag (raw data):
  ┌──────┬────────────┬──────────┬────────────┐
  │ gw   │ year_month │ democracy│ fx_vol     │
  ├──────┼────────────┼──────────┼────────────┤
  │ 700  │ 2020-01    │ 0.120    │ 0.05       │  ← January's actual values
  │ 700  │ 2020-02    │ 0.120    │ 0.08       │  ← February's actual values
  │ 700  │ 2020-03    │ 0.120    │ 0.12       │  ← March's actual values
  │ 700  │ 2020-04    │ 0.120    │ 0.06       │  ← April's actual values
  └──────┴────────────┴──────────┴────────────┘

  AFTER lag (shifted by 1 month):
  ┌──────┬────────────┬──────────┬────────────┐
  │ gw   │ year_month │ democracy│ fx_vol     │
  ├──────┼────────────┼──────────┼────────────┤
  │ 700  │ 2020-01    │ NaN      │ NaN        │  ← Nothing before Jan!
  │ 700  │ 2020-02    │ 0.120    │ 0.05       │  ← Now has JANUARY's values
  │ 700  │ 2020-03    │ 0.120    │ 0.08       │  ← Now has FEBRUARY's values
  │ 700  │ 2020-04    │ 0.120    │ 0.12       │  ← Now has MARCH's values
  └──────┴────────────┴──────────┴────────────┘

  READ IT AS:
  "When predicting what happens in April 2020,
   the most recent information we're allowed to use
   is from March 2020."

  The first month (2020-01) becomes NaN because there's
  nothing before it — we have no December 2019 data to shift in.
```

**Where in the code:**
```
  FILE: src/data/structural_features.py — function _apply_lag()

  # This one line does all the work:
  df[feature_cols] = df.groupby("gwcode")[feature_cols].shift(1)

  .groupby("gwcode")  → "Do this separately for each country"
  .shift(1)           → "Move everything down by 1 row"

  We do this ONCE, in ONE place, for ALL features.
  This way we can't accidentally forget to lag something.
```

### Step 5: Encode missingness as features

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                 MISSINGNESS AS INFORMATION                      │
  │                                                                 │
  │  Missing data isn't just a nuisance — it's a SIGNAL.          │
  │                                                                 │
  │  Why might a country have no economic data?                    │
  │  → It might be a FAILED STATE that stopped reporting.          │
  │  → That ITSELF predicts conflict!                              │
  │                                                                 │
  │  So we create binary flags:                                    │
  │                                                                 │
  │  ┌──────┬──────────┬─────────┬───────────────┬──────────────┐  │
  │  │ gw   │ fx_vol   │ fx_avail│ food_anomaly  │ food_avail   │  │
  │  ├──────┼──────────┼─────────┼───────────────┼──────────────┤  │
  │  │ 2    │ 0.03     │ 1       │ 1.05          │ 1            │  │
  │  │ 700  │ NaN      │ 0 ← !  │ NaN           │ 0 ← !       │  │
  │  │ 475  │ 0.15     │ 1       │ 1.30          │ 1            │  │
  │  └──────┴──────────┴─────────┴───────────────┴──────────────┘  │
  │                                                                 │
  │  Five flags: vdem_available, reign_available, fx_available,    │
  │              food_available, gdp_available                      │
  └─────────────────────────────────────────────────────────────────┘
```

---

# Part 6: Why We Do NOT Normalise

```
  ┌─────────────────────────────────────────────────────────────────┐
  │               THE NORMALISATION TRAP                            │
  │                                                                 │
  │  "Normalisation" means scaling data to a standard range        │
  │  (e.g., mean=0, std=1).                                       │
  │                                                                 │
  │  WHY it's dangerous if WE do it:                               │
  │                                                                 │
  │  Our data will be split into:                                  │
  │    TRAINING set (2010–2019): model learns from this            │
  │    TEST set (2020–2024): model is evaluated on this            │
  │                                                                 │
  │  If we normalise BEFORE splitting:                             │
  │    The scaling parameters (mean, std) include TEST data!       │
  │    → The model indirectly "knows" about the future.            │
  │    → This is INFORMATION LEAKAGE.                              │
  │                                                                 │
  │  CORRECT approach: Member C normalises AFTER splitting,        │
  │  using ONLY training set statistics.                           │
  │                                                                 │
  │  Exception: V-Dem is already [0, 1] by design. No scaling     │
  │  needed. We document this so Member C skips those columns.     │
  └─────────────────────────────────────────────────────────────────┘
```

---

# Part 7: Powell & Thyne Coup Patch — Filling REIGN's Blind Spot

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                THE PROBLEM: REIGN WENT BLIND                        │
  │                                                                      │
  │  REIGN stopped collecting data in August 2021.                      │
  │  But coups KEPT HAPPENING after that:                               │
  │                                                                      │
  │  Timeline:                                                           │
  │  ──────── REIGN DATA ─────────  ──── BLIND SPOT ────────────────   │
  │  2010 ─────────────── 2021-08 │ 2021-09 ──────────────── 2024-12  │
  │                                │                                    │
  │                                │  Guinea (Sep 2021) — SUCCESSFUL    │
  │                                │  Sudan (Sep 2021)  — SUCCESSFUL    │
  │                                │  Sudan (Oct 2021)  — FAILED        │
  │                                │  Burkina Faso (Jan 2022) — SUCCESS │
  │                                │  Guinea-Bissau (Feb 2022) — FAILED │
  │                                │  Burkina Faso (Sep 2022) — SUCCESS │
  │                                │  São Tomé (Nov 2022) — FAILED      │
  │                                │  Sudan (Apr 2023) — FAILED         │
  │                                │  Niger (Jul 2023) — SUCCESSFUL     │
  │                                │  Gabon (Aug 2023) — SUCCESSFUL     │
  │                                │  Bolivia (Jun 2024) — FAILED       │
  │                                │  Bangladesh (Aug 2024) — SUCCESSFUL│
  │                                                                      │
  │  That's 12 coup events our model would MISS without a patch!        │
  └──────────────────────────────────────────────────────────────────────┘
```

## The Solution: Powell & Thyne (2011) Dataset

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Powell & Thyne Coup d'État Dataset                                 │
  │  ──────────────────────────────────                                 │
  │  • Academic dataset from University of Kentucky                     │
  │  • Records EVERY coup attempt worldwide since 1950                  │
  │  • Updated within weeks of new events (still maintained!)           │
  │  • Uses Gleditsch-Ward codes natively (no crosswalk needed!)       │
  │  • Coding: 1 = failed coup, 2 = successful coup                   │
  │                                                                      │
  │  Reference:                                                          │
  │  Powell, J.M. & Thyne, C.L. (2011). Global Instances of Coups      │
  │  from 1950 to 2010. Journal of Peace Research, 48(2), 249-259.     │
  └──────────────────────────────────────────────────────────────────────┘

  HOW WE INTEGRATE IT:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  2010-01 ──────── 2021-08  │  2021-09 ──────────── 2024-12        │
  │                             │                                       │
  │  REIGN is AUTHORITATIVE    │  Powell & Thyne PATCHES the gap       │
  │  (finer-grained, includes  │  (replaces REIGN's NaN coup_event     │
  │   non-coup irregular       │   with real data from P&T)            │
  │   transitions)             │                                       │
  │                             │  Also invalidates reign_ffill_reliable│
  │  Powell & Thyne provides   │  for countries where a successful     │
  │  SUPPLEMENTARY features    │  coup means the leader changed        │
  │  only (cumulative count,   │  (forward-filled REIGN is now wrong)  │
  │  months since coup)        │                                       │
  └──────────────────────────────────────────────────────────────────────┘
```

**Features from Powell & Thyne:**
```
  FEATURE               WHAT IT MEANS
  ───────               ─────────────
  pt_coup_event         Did any coup attempt happen this month? (0/1)
  pt_coup_successful    Was the coup successful? (0/1)
  pt_coup_failed        Was the coup attempt a failure? (0/1)
  pt_coup_count         How many coup attempts this month? (usually 0 or 1)
  pt_cumulative_coups   Running total of all coups in this country since 2010
                        → captures "coup-prone" states (e.g., Burkina Faso)
  pt_months_since_coup  How many months since the last coup attempt?
                        → captures the "coup trap" (Powell 2012): countries
                          with a recent coup are more likely to have another
```

**Where in the code:**
```
  FILE: src/data/ingest_powell_thyne.py

  download_powell_thyne()     → Downloads TSV from UKY (auto with --download)
  load_powell_thyne()         → Parses event-level data, maps codes
  aggregate_to_country_month()→ Converts events to (gwcode, year_month) panel
  derive_coup_history()       → Computes cumulative coups, months since coup
  integrate_with_reign()      → Patches REIGN's post-2021 NaN values
  ingest_powell_thyne()       → Full pipeline entry point
```

---

# Part 8: Cross-Validation & Structural Break Detection

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │           WHEN DATA SOURCES DISAGREE                                 │
  │                                                                      │
  │  V-Dem says: "Sudan's democracy score for 2021 is 0.12"            │
  │  REIGN says: "Wait, there was a COUP in Sudan in October 2021!"    │
  │                                                                      │
  │  Problem: V-Dem coded the score BEFORE the coup happened.           │
  │  The 0.12 is stale for Nov-Dec 2021 — the real situation           │
  │  is much worse.                                                     │
  │                                                                      │
  │  Solution: We CREATE a flag called vdem_stale_flag:                 │
  │                                                                      │
  │  ┌──────┬────────────┬──────────┬────────────────┐                  │
  │  │ gw   │ year_month │ libdem   │ vdem_stale_flag │                 │
  │  ├──────┼────────────┼──────────┼────────────────┤                  │
  │  │ 625  │ 2021-09    │ 0.12     │ 0              │                  │
  │  │ 625  │ 2021-10    │ 0.12     │ 0 ← coup HERE │                  │
  │  │ 625  │ 2021-11    │ 0.12     │ 1 ← STALE!    │                  │
  │  │ 625  │ 2021-12    │ 0.12     │ 1 ← STALE!    │                  │
  │  └──────┴────────────┴──────────┴────────────────┘                  │
  │                                                                      │
  │  This tells the transformer: "don't trust the democracy score       │
  │  here — something changed that isn't reflected in the number."      │
  └──────────────────────────────────────────────────────────────────────┘
```

## REIGN Forward-Fill Reliability

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │           CAN WE TRUST THE FORWARD-FILL?                            │
  │                                                                      │
  │  After Aug 2021, we forward-fill REIGN's slow features              │
  │  (leader name, regime type, gender, military career).               │
  │                                                                      │
  │  But if a COUP happened (detected by Powell & Thyne), the           │
  │  leader probably CHANGED — so the forward-fill is WRONG.            │
  │                                                                      │
  │  reign_ffill_reliable = 0 means:                                    │
  │    "A structural break happened near or after the REIGN cutoff.     │
  │     The forward-filled values are probably incorrect."              │
  │                                                                      │
  │  Two triggers:                                                       │
  │    1. Structural break within 6 months BEFORE cutoff (2021-02..08)  │
  │    2. Successful coup AFTER cutoff (detected by Powell & Thyne)     │
  │       → 6 countries invalidated: Guinea, Sudan, Burkina Faso,       │
  │         Niger, Gabon, Bangladesh                                    │
  │                                                                      │
  │  months_since_structural_break captures the "instability window":   │
  │    0 = break THIS month; 3 = three months ago; NaN = no break yet   │
  │    Empirically, conflict risk stays elevated for 6-12 months        │
  │    after a regime change (Cederman et al., 2010)                    │
  └──────────────────────────────────────────────────────────────────────┘
```

**Where in the code:**
```
  FILE: src/data/structural_features.py — _cross_validate_structural_breaks()
  FILE: src/data/ingest_reign.py        — _detect_structural_breaks()
  FILE: src/data/ingest_powell_thyne.py  — integrate_with_reign()
```

---

# Part 9: Quality Checks

Before we hand off our data, we run automated checks:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    QUALITY CHECKS                               │
  │                                                                 │
  │  FILE: src/data/structural_features.py — run_quality_checks()  │
  │                                                                 │
  │  ✓ CHECK 1: Panel sorted?                                     │
  │    Every country's months must be in order:                    │
  │    2010-01, 2010-02, ..., 2024-12 (not jumbled)               │
  │                                                                 │
  │  ✓ CHECK 2: No duplicates?                                    │
  │    Each (country, month) pair appears exactly ONCE.            │
  │    Two rows for "Afghanistan, 2020-06" = BUG.                 │
  │                                                                 │
  │  ✓ CHECK 3: Lag correct?                                      │
  │    First month per country must be NaN (nothing to shift from) │
  │                                                                 │
  │  ✓ CHECK 4: V-Dem ranges valid?                               │
  │    All V-Dem indices must be between 0 and 1.                  │
  │    If v2x_libdem = 5.3, something went very wrong.            │
  │                                                                 │
  │  ✓ CHECK 5: REIGN coverage report                             │
  │    How much of REIGN data is present? (esp. post-2021)        │
  │                                                                 │
  │  ✓ CHECK 6: Economic coverage report                          │
  │    How many countries have FX/food/GDP data?                   │
  │                                                                 │
  │  ✓ CHECK 7: Full missingness summary                          │
  │    Every column: what % is NaN?                                │
  └─────────────────────────────────────────────────────────────────┘
```

---

# Part 10: The Output — What Member C Receives

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  FILE: data/intermediate/structural_features.parquet                 │
  │                                                                      │
  │  Format: Apache Parquet (compressed columnar binary)                │
  │  Size:   ~35,460 rows × ~55+ columns (197 countries × 180 months) │
  │  Index:  (gwcode, year_month)                                       │
  │                                                                      │
  │  ┌────────┬────────────┬───────────┬───────┬─────────┬──────────┐   │
  │  │ gwcode │ year_month │ v2x_libdem│ tenure│ fx_vol  │ vdem_    │   │
  │  │        │            │           │_months│         │ available│   │
  │  ├────────┼────────────┼───────────┼───────┼─────────┼──────────┤   │
  │  │ 2      │ 2010-01    │ NaN       │ NaN   │ NaN     │ 0        │   │
  │  │ 2      │ 2010-02    │ 0.891     │ 15    │ 0.012   │ 1        │   │
  │  │ 2      │ 2010-03    │ 0.891     │ 16    │ 0.008   │ 1        │   │
  │  │ ...    │ ...        │ ...       │ ...   │ ...     │ ...      │   │
  │  │ 700    │ 2024-11    │ 0.040     │ 40    │ NaN     │ 1        │   │
  │  │ 700    │ 2024-12    │ 0.040     │ 41    │ NaN     │ 1        │   │
  │  └────────┴────────────┴───────────┴───────┴─────────┴──────────┘   │
  │                                                                      │
  │  EVERY feature has:                                                 │
  │    • t−1 lag already applied ← CRITICAL, Member C does NOT re-lag  │
  │    • NaN for first month (consequence of lag)                      │
  │    • NaN where source data is unavailable                          │
  │    • Raw values (not normalised) ← Member C normalises             │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

# Part 11: File-by-File Guide

```
  PROJECT STRUCTURE:

  COMP0197_CW2/
  │
  ├── src/data/                          ← ALL THE CODE
  │   ├── crosswalk.py                   ← Country code translator (GW ↔ ISO3)
  │   ├── ingest_vdem.py                 ← V-Dem: load, expand, derive
  │   ├── ingest_reign.py                ← REIGN: load, gap-fill, derive
  │   ├── ingest_economic.py             ← FX + GDP + food: load, derive
  │   ├── ingest_powell_thyne.py        ← Powell & Thyne coups: patch REIGN gap
  │   ├── structural_features.py         ← MERGE + LAG + EXPORT (the boss)
  │   └── download.py                    ← Fetch data from APIs
  │
  ├── configs/
  │   ├── member_b_config.yaml           ← All settings in one place
  │   └── feature_registry_member_b.csv  ← Every feature documented
  │
  ├── data/
  │   ├── raw/                           ← Downloaded CSVs go here
  │   ├── intermediate/                  ← Output Parquet goes here
  │   └── logs/                          ← Quality report JSON goes here
  │
  ├── tests/
  │   └── test_pipeline.py              ← 16 automated tests
  │
  ├── notebooks/
  │   └── member_b_eda.py               ← Exploratory analysis script
  │
  ├── docs/
  │   ├── member_b_methodology.md       ← For the paper appendix
  │   └── member_b_explained.md         ← THIS FILE
  │
  ├── run_member_b.py                   ← One command to run everything
  └── requirements.txt                  ← Python packages needed
```

---

# Part 12: How Each File Calls Each Other

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                       CALL GRAPH                                     │
  │                                                                      │
  │  run.py                                                              │
  │       │                                                              │
  │       │ (optionally, with --download flag)                          │
  │       ├──► download.py ──► downloads REIGN, GDP from internet       │
  │       ├──► ingest_powell_thyne.py ──► downloads P&T coup data       │
  │       │                                                              │
  │       │ (main pipeline)                                             │
  │       └──► structural_features.py                                   │
  │                │                                                     │
  │                ├──► ingest_vdem.py                                   │
  │                │        └──► crosswalk.py (ISO3 → GW mapping)       │
  │                │                                                     │
  │                ├──► ingest_reign.py                                  │
  │                │        └──► _detect_structural_breaks()             │
  │                │        (uses native GW codes — no crosswalk)       │
  │                │                                                     │
  │                ├──► ingest_economic.py                               │
  │                │        └──► crosswalk.py (ISO3/IMF → GW mapping)   │
  │                │                                                     │
  │                ├──► ingest_powell_thyne.py                           │
  │                │        └──► integrate_with_reign() (patch coups)   │
  │                │                                                     │
  │                ├──► _build_panel_skeleton()              Build grid  │
  │                ├──► merge (left-join)                    Combine all │
  │                ├──► _cross_validate_structural_breaks()  V-Dem stale │
  │                ├──► _apply_lag()                         Shift t−1   │
  │                ├──► _encode_missingness()                Availability│
  │                ├──► run_quality_checks()                 Verify      │
  │                └──► .to_parquet()                        Export      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

# Part 13: Glossary of Every Term

```
  TERM                    PLAIN ENGLISH
  ────                    ─────────────
  Panel data              A table where each row is a (country, month) pair
  GW code                 A number that identifies a country (like a phone number)
  ISO3                    A 3-letter code for a country (like AFG, USA, GBR)
  Crosswalk               A translation table between two code systems
  Forward-fill            "Copy the last known value into empty cells"
  One-hot encoding        Convert a category into multiple yes/no columns
  Lag (t-1)               "Use LAST month's data, not THIS month's"
  Temporal leakage        Accidentally using future information (cheating)
  NaN                     "Not a Number" = missing/unknown value
  Parquet                 A compressed file format for big tables (like ZIP for data)
  z-score                 "How many standard deviations from the mean?"
  Rolling window          "Average/std over the last N months" (slides forward)
  Missingness indicator   A 0/1 flag that says "was this data available?"
  Derived feature         A new column computed from existing columns
  Panel skeleton          The empty grid of all (country, month) combos
  Left join               "Look up matching data; keep NaN if no match"
  MNAR                    "Missing Not At Random" — the WHY of missingness matters
  Forward-fill            Using the last known value to fill gaps
  Regime type             What kind of government (democracy, autocracy, etc.)
  Structural break        A sudden change that invalidates assumptions (e.g., a coup)
  Volatility              How much something bounces around (stability measure)
  CPI                     Consumer Price Index — measures how expensive things are
  Depreciation            Currency losing value against the dollar
  Powell & Thyne          Academic coup dataset (1950-present, updated in real time)
  Coup trap               Countries with recent coups are more likely to have another
  Stale flag              Marks data that is outdated because something changed
  Forward-fill reliable   0/1 flag: "can we trust the filled-in REIGN values?"
  Cross-validation        Checking one source against another to detect conflicts
  Expanding z-score       z-score using ALL past data, growing over time (no leakage)
  log1p transform         log(1 + x): compresses extreme values while preserving zeros
```

---

# Part 14: The Theory — Why These Features Predict Conflict

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │              WHY THESE FEATURES MATTER                               │
  │              (The Academic Justification)                            │
  │                                                                      │
  │  ┌────────────────────┐                                             │
  │  │  WEAK INSTITUTIONS │  Hegre et al. (2001)                       │
  │  │  (governance_deficit)│  "Countries in the middle — not fully     │
  │  │                    │   democratic, not fully autocratic —        │
  │  │                    │   have the highest conflict risk."         │
  │  └────────┬───────────┘                                             │
  │           │                                                          │
  │           ▼                                                          │
  │  ┌────────────────────┐                                             │
  │  │  STATE REPRESSION  │  Gurr (1970) "Why Men Rebel"               │
  │  │  (repression_index)│  "When governments are corrupt AND         │
  │  │                    │   violent, citizens have both              │
  │  │                    │   grievance and motivation to rebel."      │
  │  └────────┬───────────┘                                             │
  │           │                                                          │
  │           ▼                                                          │
  │  ┌────────────────────┐                                             │
  │  │  DEMOCRATIC        │  Mansfield & Snyder (2005)                 │
  │  │  BACKSLIDING       │  "Transitions — in either direction —      │
  │  │  (libdem_yoy_change)│  destabilise the political order and      │
  │  │                    │   create windows for violence."            │
  │  └────────┬───────────┘                                             │
  │           │                                                          │
  │           ▼                                                          │
  │  ┌────────────────────┐                                             │
  │  │  ECONOMIC SHOCKS   │  Miguel, Satyanath & Sergenti (2004)       │
  │  │  (gdp_deviation,   │  "Negative income shocks increase         │
  │  │   fx_volatility)   │   civil conflict, especially in           │
  │  │                    │   poor countries."                         │
  │  └────────┬───────────┘                                             │
  │           │                                                          │
  │           ▼                                                          │
  │  ┌────────────────────┐                                             │
  │  │  FOOD PRICE SPIKES │  Bellemare (2015)                          │
  │  │  (food_price_anomaly│  "Rising food prices directly predict     │
  │  │   food_cpi_accel)  │   social unrest, particularly protests     │
  │  │                    │   and riots in developing countries."      │
  │  └────────┬───────────┘                                             │
  │           │                                                          │
  │           ▼                                                          │
  │  ┌────────────────────┐                                             │
  │  │  THE COUP TRAP     │  Powell (2012)                              │
  │  │  (pt_cumulative_   │  "Countries that experience a coup are     │
  │  │   coups,           │   significantly more likely to experience  │
  │  │   pt_months_since_ │   another one. The risk decays slowly     │
  │  │   coup)            │   but never fully disappears."            │
  │  └────────────────────┘                                             │
  │                                                                      │
  │  Together, these features capture the STRUCTURAL CONDITIONS         │
  │  that make a country VULNERABLE to conflict — like dry timber       │
  │  waiting for a spark. Member A's features provide the SPARKS.      │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

# Part 15: Common Questions

**Q: Why Parquet and not CSV?**
CSV is text — slow to read, takes lots of space. Parquet is binary — 5-10x
smaller, 10-50x faster to load, and preserves data types (CSV loses the
difference between integers and floats).

**Q: Why 2010–2024?**
ViEWS uses this as the standard evaluation period. Pre-2010 data is spottier,
and we need enough history for the transformer's 12-month rolling window.

**Q: What if a country has NO data from any source?**
It stays in the panel with all NaN values. The missingness flags (all set to 0)
become its only features. The model can still learn from patterns like
"countries with zero data availability tend to be failed states."

**Q: Why not just drop countries with lots of missing data?**
Because those are often the most interesting countries! Somalia, South Sudan,
Yemen — the places with the most conflict often have the worst data. Dropping
them would bias our model toward predicting stable countries (which is easy
and useless).

**Q: What does Member C do with our output?**
Member C takes our Parquet file, Member A's conflict features, and their own
sentiment features, merges them all on (gwcode, year_month), normalises
using training-set-only statistics, and feeds the result into the transformer.
