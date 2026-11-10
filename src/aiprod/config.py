"""Project paths, sample windows, and the industry crosswalk.

Industries are mapped across five sources: BEA GDP-by-industry (output), BLS
CES (employment and hours), Census QWI (employment by age), the
Felten-Raj-Seamans language-modeling AI exposure index (by NAICS), and BEA
detailed fixed assets (IT capital in 1995).
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

# Sample windows (inclusive). COVID is excluded everywhere; growth rates whose
# 4-quarter window touches COVID are excluded too, hence the 2022 end.
COVID = ("2020-01-01", "2022-12-31")
BASELINE = ("2010-01-01", "2019-12-31")
IT_ERA = ("1998-01-01", "2004-12-31")  # BEA annual VA starts 1997, so growth starts 1998
AI_ERA = ("2023-01-01", "2026-12-31")  # ChatGPT released Nov 2022

# QWI age groups: A03 = 22-24 (juniors), A05/A06 = 35-44 / 45-54 (prime age).
QWI_YOUNG = ("A03",)
QWI_PRIME = ("A05", "A06")
# Windows for the junior-employment difference-in-differences (C2).
QWI_IT = {"pre": ("1994-01-01", "1995-12-31"), "post": ("1996-01-01", "2000-12-31")}
QWI_AI = {"pre": ("2015-01-01", "2019-12-31"), "post": ("2022-10-01", "2026-12-31")}

# IT capital in BEA fixed assets: computers, communications, software.
IT_ASSET_CODES = (
    "EP1A", "EP1B", "EP1C", "EP1D", "EP1E", "EP1F", "EP1G", "EP1H",
    "EP20", "ENS1", "ENS2", "ENS3",
)
FA_TOTAL_CODES = ("EQ00", "ST00", "IP00")
IT_BASE_YEAR = 1995


@dataclass(frozen=True)
class Sector:
    """One industry in the output-per-hour panel (claim C1).

    Attributes:
        key (str): Short identifier.
        bea_line (str): Exact industry label in BEA GDP-by-industry tables.
        ces (str): 8-digit CES industry code (supersector + industry).
        ae_id (str): FRED ID for all-employee employment of the CES industry.
        naics2 (tuple[str, ...]): Two-digit NAICS codes (QWI labels) in the sector.
        ae_sub_id (str | None): FRED ID of a sub-industry's employment, used to
            scale CES hours when BEA output covers only part of the CES
            supersector (finance and insurance within financial activities).
    """

    key: str
    bea_line: str
    ces: str
    ae_id: str
    naics2: tuple[str, ...]
    ae_sub_id: str | None = field(default=None)


SECTORS: tuple[Sector, ...] = (
    Sector("mining", "Mining", "10000000", "USMINE", ("21",)),
    Sector("utilities", "Utilities", "44220000", "CES4422000001", ("22",)),
    Sector("construction", "Construction", "20000000", "USCONS", ("23",)),
    Sector("manufacturing", "Manufacturing", "30000000", "MANEMP", ("31-33",)),
    Sector("wholesale", "Wholesale trade", "41420000", "USWTRADE", ("42",)),
    Sector("retail", "Retail trade", "42000000", "USTRADE", ("44-45",)),
    Sector("transport", "Transportation and warehousing", "43000000", "CES4300000001", ("48-49",)),
    Sector("information", "Information", "50000000", "USINFO", ("51",)),
    Sector("finance", "Finance and insurance", "55000000", "USFIRE", ("52",), "CES5552000001"),
    Sector("prof_business", "Professional and business services", "60000000", "USPBS", ("54", "55", "56")),
    Sector(
        "educ_health",
        "Educational services, health care, and social assistance",
        "65000000", "USEHS", ("61", "62"),
    ),
    Sector(
        "leisure",
        "Arts, entertainment, recreation, accommodation, and food services",
        "70000000", "USLAH", ("71", "72"),
    ),
    Sector("other_services", "Other services, except government", "80000000", "USSERV", ("81",)),
)
