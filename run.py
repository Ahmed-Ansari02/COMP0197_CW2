#!/usr/bin/env python3
"""
Member B Pipeline Runner
=========================

End-to-end execution of the structural/contextual features pipeline.

Usage:
    # With all data sources available:
    python run_member_b.py

    # With specific paths:
    python run_member_b.py \
        --vdem data/raw/V-Dem-CY-Full+Others-v15.csv \
        --reign data/raw/REIGN_2021_8.csv \
        --fx data/raw/imf_exchange_rates.csv \
        --gdp data/raw/gdp_growth_worldbank.csv \
        --food data/raw/fao_food_cpi.csv

    # Download REIGN and GDP automatically:
    python run_member_b.py --download
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/logs/member_b_pipeline.log", mode="w"),
        ],
    )
    logger = logging.getLogger("run_member_b")

    parser = argparse.ArgumentParser(description="Run Member B structural features pipeline")
    parser.add_argument("--vdem", type=str, default="data/raw/V-Dem-CY-Full+Others-v16.csv")
    parser.add_argument("--reign", type=str, default="data/raw/REIGN_2021_8.csv")
    parser.add_argument("--fx", type=str, default="data/raw/imf_exchange_rates.csv")
    parser.add_argument("--gdp", type=str, default="data/raw/gdp_growth_worldbank.csv")
    parser.add_argument("--food", type=str, default="data/raw/fao_food_cpi.csv")
    parser.add_argument("--coups", type=str, default="data/raw/powell_thyne_coups.tsv")
    parser.add_argument("--download", action="store_true", help="Download available sources")
    parser.add_argument("--output", type=str, default="data/intermediate/structural_features.parquet")
    args = parser.parse_args()

    # Ensure directories exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/intermediate").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # Optionally download data
    if args.download:
        from src.data.download import download_reign, download_gdp_worldbank
        from src.data.ingest_powell_thyne import download_powell_thyne
        logger.info("Downloading available data sources...")
        try:
            download_reign()
        except Exception as e:
            logger.warning("REIGN download failed: %s", e)
        try:
            download_gdp_worldbank()
        except Exception as e:
            logger.warning("GDP download failed: %s", e)
        try:
            download_powell_thyne()
        except Exception as e:
            logger.warning("Powell & Thyne download failed: %s", e)

    # Resolve paths — skip sources that don't exist
    vdem = args.vdem if Path(args.vdem).exists() else None
    reign = args.reign if Path(args.reign).exists() else None
    fx = args.fx if Path(args.fx).exists() else None
    gdp = args.gdp if Path(args.gdp).exists() else None
    food = args.food if Path(args.food).exists() else None
    coups = args.coups if Path(args.coups).exists() else None

    if not any([vdem, reign, fx, gdp, food, coups]):
        logger.error(
            "No data sources found. Place raw data in data/raw/ or use --download.\n"
            "Required files:\n"
            "  V-Dem:    data/raw/V-Dem-CY-Full+Others-v15.csv (manual download from v-dem.net)\n"
            "  REIGN:    data/raw/REIGN_2021_8.csv (--download or manual)\n"
            "  FX:       data/raw/imf_exchange_rates.csv (manual from data.imf.org)\n"
            "  GDP:      data/raw/gdp_growth_worldbank.csv (--download or manual)\n"
            "  Food CPI: data/raw/fao_food_cpi.csv (manual from FAOSTAT)\n"
            "  Coups:    data/raw/powell_thyne_coups.tsv (--download or manual)"
        )
        sys.exit(1)

    sources = {k: v for k, v in
               {"V-Dem": vdem, "REIGN": reign, "FX": fx, "GDP": gdp,
                "Food": food, "Coups": coups}.items()
               if v is not None}
    logger.info("Sources available: %s", ", ".join(sources.keys()))

    # Run pipeline
    from src.data.structural_features import build_structural_features

    df = build_structural_features(
        vdem_csv=vdem,
        reign_csv=reign,
        fx_csv=fx,
        gdp_csv=gdp,
        food_csv=food,
        coups_tsv=coups,
        output_path=args.output,
    )

    logger.info("Pipeline complete. Output: %s", args.output)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))

    return df


if __name__ == "__main__":
    main()
