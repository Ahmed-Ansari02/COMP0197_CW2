"""
Gleditsch-Ward (GW) ↔ ISO3 Country Code Crosswalk
===================================================

Provides deterministic bidirectional mapping between ViEWS' canonical
Gleditsch-Ward numeric country codes and ISO 3166-1 alpha-3 codes used
by V-Dem, World Bank, FAO, and most international datasets.

Source: Gleditsch & Ward (1999), updated via CShapes 2.0 and the
Correlates of War state-system membership list (v2016).

Edge-case handling:
    - South Sudan (GW 626): independent from 2011-07; pre-2011 → Sudan (625)
    - Kosovo: not in canonical GW; mapped to custom code 347
    - Serbia (345) / Montenegro (341): separated 2006-06
    - Timor-Leste (860): independent from 2002-05
    - Czechoslovakia (315): dissolved 1993-01 → CZE (316) + SVK (317)
    - Yugoslavia (345): dissolved → constituent states
    - Sudan/South Sudan split handled via temporal dispatch
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Complete GW → ISO3 mapping for all states active during 2000–2024
# ──────────────────────────────────────────────────────────────────────
GW_TO_ISO3: Dict[int, str] = {
    # Americas
    2: "USA", 20: "CAN", 31: "BHS", 40: "CUB", 41: "HTI", 42: "DOM",
    51: "JAM", 52: "TTO", 53: "BRB", 54: "DMA", 55: "GRD", 56: "SLC",
    57: "SVG", 58: "ATG", 60: "KNA", 70: "MEX", 80: "BLZ", 90: "GTM",
    91: "HND", 92: "SLV", 93: "NIC", 94: "CRI", 95: "PAN", 100: "COL",
    101: "VEN", 110: "GUY", 115: "SUR", 130: "ECU", 135: "PER", 140: "BRA",
    145: "BOL", 150: "PRY", 155: "CHL", 160: "ARG", 165: "URY",

    # Europe
    200: "GBR", 205: "IRL", 210: "NLD", 211: "BEL", 212: "LUX",
    220: "FRA", 225: "CHE", 230: "ESP", 235: "PRT", 255: "DEU",
    260: "DEU",  # West Germany → unified Germany post-1990
    265: "DEU",  # East Germany → unified Germany post-1990
    290: "POL", 305: "AUT", 310: "HUN", 316: "CZE", 317: "SVK",
    325: "ITA", 331: "SMR", 338: "MLT", 339: "ALB", 341: "MNE",
    343: "MKD", 344: "HRV", 345: "SRB", 346: "BIH",
    347: "XKX",  # Kosovo — custom code, not canonical GW
    349: "SVN",
    350: "GRC", 352: "CYP", 355: "BGR", 359: "MDA", 360: "ROU",
    365: "RUS", 366: "EST", 367: "LVA", 368: "LTU", 369: "UKR",
    370: "BLR", 371: "ARM", 372: "GEO", 373: "AZE",
    375: "FIN", 380: "SWE", 385: "NOR", 390: "DNK", 395: "ISL",

    # Africa
    402: "CPV", 404: "GNB", 411: "GIN", 420: "GMB", 432: "MLI",
    433: "SEN", 434: "BEN", 435: "MRT", 436: "NER", 437: "CIV",
    438: "GIN",  # duplicate — use 411 for Guinea
    439: "BFA", 450: "LBR", 451: "SLE", 452: "GHA", 461: "TGO",
    471: "CMR", 475: "NGA", 481: "GAB", 482: "CAF", 483: "TCD",
    484: "COG", 490: "COD", 500: "UGA", 501: "KEN", 510: "TZA",
    516: "BDI", 517: "RWA", 520: "SOM", 522: "DJI",
    530: "ETH", 531: "ERI", 540: "AGO", 541: "MOZ",
    551: "ZMB", 552: "ZWE", 553: "MWI", 560: "ZAF",
    565: "NAM", 570: "LSO", 571: "BWA", 572: "SWZ",
    580: "MDG", 581: "COM", 590: "MUS", 591: "SYC",
    600: "MAR", 615: "DZA", 616: "TUN", 620: "LBY",
    625: "SDN", 626: "SSD",  # South Sudan from 2011-07
    630: "IRN", 640: "TUR",
    645: "IRQ", 651: "EGY", 652: "SYR", 660: "LBN",
    663: "JOR", 666: "ISR", 670: "SAU", 678: "YEM",
    679: "YEM",  # North/South Yemen → unified Yemen
    680: "YEM",
    690: "KWT", 692: "BHR", 694: "QAT", 696: "ARE", 698: "OMN",

    # Asia
    700: "AFG", 701: "TKM", 702: "TJK", 703: "KGZ", 704: "UZB",
    705: "KAZ", 710: "CHN", 711: "MNG", 712: "TWN",
    713: "TWN",  # Taiwan — may use custom code
    730: "KOR",  # South Korea
    731: "PRK",  # North Korea
    740: "JPN",
    750: "IND", 760: "PAK", 770: "BGD", 771: "BGD",
    775: "MMR", 780: "LKA", 781: "MDV",
    790: "NPL", 800: "THA", 811: "KHM", 812: "LAO",
    816: "VNM", 820: "MYS", 830: "SGP",
    835: "BRN", 840: "PHL", 850: "IDN", 860: "TLS",

    # Oceania
    900: "AUS", 910: "PNG", 920: "NZL",
    935: "VUT", 940: "SLB", 946: "KIR",
    947: "TUV", 950: "FJI", 955: "TON", 970: "NRU",
    983: "MHL", 986: "PLW", 987: "FSM", 990: "WSM",
}

# Invert: ISO3 → GW (take the first mapping for duplicates)
ISO3_TO_GW: Dict[str, int] = {}
for _gw, _iso in GW_TO_ISO3.items():
    if _iso not in ISO3_TO_GW:
        ISO3_TO_GW[_iso] = _gw


# ──────────────────────────────────────────────────────────────────────
# IMF numeric code → ISO3 (subset of countries in IMF IFS)
# ──────────────────────────────────────────────────────────────────────
IMF_TO_ISO3: Dict[int, str] = {
    111: "USA", 156: "CAN", 213: "FRA", 218: "DEU", 223: "ITA",
    112: "GBR", 138: "NLD", 124: "BEL", 137: "LUX", 128: "DNK",
    144: "SWE", 142: "NOR", 172: "FIN", 176: "ISL", 122: "AUT",
    146: "CHE", 182: "PRT", 184: "ESP", 174: "GRC", 178: "IRL",
    936: "BGD", 534: "IND", 564: "PAK", 524: "LKA", 518: "MMR",
    566: "NPL", 548: "MYS", 576: "SGP", 578: "THA", 536: "IDN",
    582: "VNM", 556: "PHL", 516: "KHM",
    612: "DZA", 744: "QAT", 456: "SAU", 466: "KWT", 429: "IRN",
    436: "ISR", 439: "IRQ", 446: "LBN", 474: "ARE", 449: "JOR",
    443: "EGY", 672: "SYR", 678: "TUN", 686: "MAR",
    694: "NGA", 646: "ZAF", 648: "ZWE", 636: "KEN", 738: "TZA",
    734: "UGA", 618: "AGO", 622: "BEN", 748: "SEN", 684: "MOZ",
    644: "ETH", 632: "GHA", 662: "CIV",
    213: "BRA", 228: "COL", 248: "MEX", 298: "ARG",
    233: "ECU", 293: "URY", 288: "PRY", 218: "BOL",
    253: "PER", 228: "VEN", 238: "CHL",
    964: "CHN", 532: "HKG", 542: "KOR", 158: "JPN",
    926: "RUS", 960: "MNG",
    944: "AUS", 196: "NZL",
}


def gw_from_iso3(iso3: str) -> Optional[int]:
    """Map ISO 3166-1 alpha-3 to Gleditsch-Ward code."""
    gw = ISO3_TO_GW.get(iso3)
    if gw is None:
        logger.warning("No GW mapping for ISO3=%s", iso3)
    return gw


def iso3_from_gw(gwcode: int) -> Optional[str]:
    """Map Gleditsch-Ward code to ISO 3166-1 alpha-3."""
    iso3 = GW_TO_ISO3.get(gwcode)
    if iso3 is None:
        logger.warning("No ISO3 mapping for GW=%d", gwcode)
    return iso3


def gw_from_imf(imf_code: int) -> Optional[int]:
    """Two-step mapping: IMF numeric → ISO3 → GW."""
    iso3 = IMF_TO_ISO3.get(imf_code)
    if iso3 is None:
        logger.warning("No ISO3 mapping for IMF code=%d", imf_code)
        return None
    return gw_from_iso3(iso3)


def resolve_south_sudan(iso3: str, year: int, month: int) -> int:
    """
    Handle the Sudan / South Sudan split (July 2011).

    Pre-2011-07: all Sudan events → GW 625
    Post-2011-07: SSD → GW 626, SDN → GW 625
    """
    if iso3 == "SSD":
        if year < 2011 or (year == 2011 and month < 7):
            logger.info("SSD before independence (%d-%02d) → SDN (GW 625)", year, month)
            return 625
        return 626
    if iso3 == "SDN":
        return 625
    return gw_from_iso3(iso3)


def validate_gwcode(gwcode: int, year: int) -> bool:
    """Check whether a GW code is valid for a given year (basic checks)."""
    # South Sudan only valid from 2011
    if gwcode == 626 and year < 2011:
        return False
    # Montenegro only valid from 2006
    if gwcode == 341 and year < 2006:
        return False
    # Eritrea only valid from 1993
    if gwcode == 531 and year < 1993:
        return False
    # Timor-Leste only valid from 2002
    if gwcode == 860 and year < 2002:
        return False
    return True
