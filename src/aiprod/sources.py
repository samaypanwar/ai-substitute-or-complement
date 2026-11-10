"""Download-and-cache readers for every free data source in the project.

Each reader downloads its file into ``data/raw`` once and parses it into a tidy
pandas object. Delete a cached file to force a fresh download.
"""

import gzip
from pathlib import Path

import openpyxl
import pandas as pd
import requests

from aiprod.config import FA_TOTAL_CODES, IT_ASSET_CODES, RAW

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
BEA_VALUE_ADDED = "https://apps.bea.gov/industry/Release/XLS/GDPxInd/ValueAdded.xlsx"
BEA_FIXED_ASSETS = "https://apps.bea.gov/national/FA2004/Details/xls/detailnonres_stk1.xlsx"
LM_AIIE = (
    "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/"
    "Language%20Modeling%20AIOE%20and%20AIIE.xlsx"
)
QWI_NATIONAL = (
    "https://lehd.ces.census.gov/data/qwi/latest_release/us/qwi_us_sa_f_gn_ns_op_u.csv.gz"
)


def download(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless the file is already cached.

    Args:
        url (str): Source URL.
        path (Path): Destination file.

    Returns:
        Path: The cached file.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Default requests User-Agent: FRED stalls requests from unrecognised agents.
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def fred(series_id: str) -> pd.Series:
    """Read one FRED series (latest vintage) as a float Series indexed by date.

    Args:
        series_id (str): FRED series ID, e.g. ``"USINFO"``.

    Returns:
        pd.Series: Observations named ``series_id``; missing values dropped.
    """
    path = download(FRED_CSV.format(series_id), RAW / "fred" / f"{series_id}.csv")
    frame = pd.read_csv(path, parse_dates=["observation_date"], na_values=".")
    series = frame.set_index("observation_date")[series_id].astype(float).dropna()
    return series.rename(series_id)


def bea_value_added(table: str) -> pd.DataFrame:
    """Read one sheet of BEA's GDP-by-industry ``ValueAdded.xlsx`` workbook.

    Args:
        table (str): Sheet name, e.g. ``"TVA103-A"`` (annual chain-type quantity
            index) or ``"TVA103-Q"`` (quarterly).

    Returns:
        pd.DataFrame: Rows are BEA lines (``line``, ``industry`` columns),
        remaining columns are periods as strings ("1997", "2005Q1").
    """
    path = download(BEA_VALUE_ADDED, RAW / "bea_valueadded.xlsx")
    sheet = openpyxl.load_workbook(path, read_only=True)[table]
    rows = list(sheet.iter_rows(values_only=True))
    header_at = next(i for i, row in enumerate(rows) if row[0] == "Line")
    periods = [str(p) for p in rows[header_at][2:] if p is not None]

    # Data rows have an integer line number in the first column; footnotes do not.
    records = []
    for row in rows[header_at + 1:]:
        if row[0] is None or not str(row[0]).isdigit():
            continue
        values = [pd.to_numeric(v, errors="coerce") for v in row[2:2 + len(periods)]]
        records.append([int(row[0]), str(row[1]).strip(), *values])
    return pd.DataFrame(records, columns=["line", "industry", *periods])


def it_capital_stock(year: int) -> pd.DataFrame:
    """IT and total current-cost net capital stock by industry in ``year``.

    IT capital is computers, communications equipment and software (see
    ``IT_ASSET_CODES``); the total is equipment + structures + IPP. Levels are
    returned (millions of dollars) so that industries can be aggregated by
    summation before taking the IT share.

    Args:
        year (int): Calendar year between 1925 and the latest BEA year.

    Returns:
        pd.DataFrame: Columns ``it`` and ``total``, indexed by BEA fixed-asset
        industry code (e.g. ``"5415"``, ``"524A"``).
    """
    path = download(BEA_FIXED_ASSETS, RAW / "bea_fa_detail_stock.xlsx")
    sheet = openpyxl.load_workbook(path, read_only=True)["Datasets"]
    rows = sheet.iter_rows(values_only=True)
    column = [str(c) for c in next(rows)].index(str(year))

    # Row keys look like "K1N5415" + "1" + "EP1B" + ".A": industry, then asset code.
    stocks = pd.DataFrame(
        [(row[0][3:7], row[0][8:12], row[column]) for row in rows if row[0]],
        columns=["industry", "asset", "stock"],
    )
    stocks["stock"] = pd.to_numeric(stocks["stock"], errors="coerce")
    by_type = stocks.pivot_table(index="industry", columns="asset", values="stock", aggfunc="sum")

    return pd.DataFrame({
        "it": by_type[list(IT_ASSET_CODES)].sum(axis=1),
        "total": by_type[list(FA_TOTAL_CODES)].sum(axis=1),
    })


def language_model_exposure() -> pd.Series:
    """Felten-Raj-Seamans language-modeling AI industry exposure (LM AIIE).

    Returns:
        pd.Series: Exposure score indexed by NAICS code string (4-6 digits).
    """
    path = download(LM_AIIE, RAW / "lm_aiie.xlsx")
    frame = pd.read_excel(path, sheet_name="LM AIIE", dtype={"NAICS": str})
    return frame.set_index("NAICS")["Language Modeling AIIE"].rename("lm_aiie")


def qwi_by_age() -> pd.DataFrame:
    """National QWI employment, hires and earnings by NAICS sector and age group.

    Returns:
        pd.DataFrame: Columns ``industry`` (QWI NAICS sector label, e.g. "51",
        "31-33"), ``agegrp`` ("A01".."A08"), ``date`` (quarter start),
        ``emp`` (beginning-of-quarter employment), ``hires`` (all hires),
        ``separations`` and ``earnings`` (average monthly earnings, stable jobs).
        Both sexes combined; sector level only.
    """
    path = download(QWI_NATIONAL, RAW / "qwi_us_sa_f_gn_ns_op_u.csv.gz")
    columns = ["ind_level", "industry", "sex", "agegrp", "year", "quarter", "Emp", "HirA", "Sep", "EarnS"]
    with gzip.open(path, "rt") as handle:
        frame = pd.read_csv(handle, usecols=columns, dtype={"industry": str})

    frame = frame[(frame["ind_level"] == "S") & (frame["sex"] == 0)]
    frame["date"] = pd.PeriodIndex.from_fields(
        year=frame["year"], quarter=frame["quarter"], freq="Q"
    ).to_timestamp()
    renamed = frame.rename(
        columns={"Emp": "emp", "HirA": "hires", "Sep": "separations", "EarnS": "earnings"}
    )
    return renamed[["industry", "agegrp", "date", "emp", "hires", "separations", "earnings"]]
