"""Treat AI (and, symmetrically, IT) services as investment rather than intermediate inputs.

National accounts subtract purchased services from gross output to get value
added. Cloud, software and IT-services purchases are intermediate inputs, so an
industry that buys AI only shows higher value added if its output rises by more
than the bill. Here we re-classify the *unusual* part of those purchases as
investment:

1. AI spending proxy. From BEA's summary input-output use tables, each sector's
   purchases of three commodities: software publishing (511), data processing
   and hosting (514) and computer systems design (5415). Spending is measured as
   a share of the sector's gross output; only the excess over the pre-era path
   counts (AI era: linear 2010-2019 trend; IT era: the 1997 share, because the
   tables start in 1997). Negative excesses are set to zero.
2. Adjusted value added = value added + excess spending.
3. A capital stock built by perpetual inventory, ``K_t = (1 - delta) K_{t-1} + X_t``,
   with depreciation ``delta * K_t``. ``delta = 1`` gives depreciation = spending,
   so the net measures fall back to the current treatment.
4. Net labor share = compensation / (value added - depreciation), with official
   BEA depreciation plus the depreciation of the new capital.

The use tables end in 2023. Purchases for 2024-2025 are extrapolated by growing
each sector's 2023 purchases with the nominal gross output of the three
supplying industries (BEA GDP by industry, which runs to 2025).
"""

import zipfile

import numpy as np
import openpyxl
import pandas as pd

from aiprod import sources
from aiprod.config import FA_TOTAL_CODES, RAW, SECTORS
from aiprod.panels import all_employee_hours, to_naics2

BEA_IO_ZIP = "https://apps.bea.gov/industry/iTables%20Static%20Files/AllTablesIO.zip"
USE_TABLE = "IOUse_Before_Redefinitions_PRO_1997-2023_Summary.xlsx"
BEA_GROSS_OUTPUT = "https://apps.bea.gov/industry/Release/XLS/GDPxInd/GrossOutput.xlsx"
BEA_INTERMEDIATE = "https://apps.bea.gov/industry/Release/XLS/GDPxInd/IntermediateInputs.xlsx"
BEA_DEPRECIATION = "https://apps.bea.gov/national/FA2004/Details/xls/detailnonres_dep1.xlsx"

# Commodities whose purchases proxy AI / IT services, with their GDP-by-industry line labels.
IT_COMMODITIES = {
    "511": "Publishing industries, except internet (includes software)",
    "514": "Data processing, internet publishing, and other information services",
    "5415": "Computer systems design and related services",
}
USE_TOTAL_OUTPUT_ROW = "T018"  # "Total industry output" in the summary use table
AI_TREND = (2010, 2019)
IT_BASE = 1997
DELTAS = (0.33, 0.5, 1.0)
VARIANTS = ("AI", "IT", "AIflat", "ALL")
LAST_USE_YEAR = 2023


def gdp_by_industry(url: str, name: str, table: str) -> pd.DataFrame:
    """Read one sheet of a BEA GDP-by-industry workbook (same layout as ValueAdded.xlsx)."""
    sheet = openpyxl.load_workbook(sources.download(url, RAW / name), read_only=True)[table]
    rows = list(sheet.iter_rows(values_only=True))
    header_at = next(i for i, row in enumerate(rows) if row[0] == "Line")
    header = rows[header_at]
    period_columns = {i: str(label) for i, label in enumerate(header) if i >= 2 and label is not None}
    records = []
    for row in rows[header_at + 1:]:
        if row[0] is None or not str(row[0]).isdigit():
            continue
        values = [pd.to_numeric(row[i], errors="coerce") for i in period_columns]
        records.append([int(row[0]), str(row[1]).strip(), *values])
    return pd.DataFrame(records, columns=["line", "industry", *period_columns.values()])


def bea_line(table: pd.DataFrame, label: str) -> pd.Series:
    """One industry row of a GDP-by-industry table as an annual Series indexed by year (int)."""
    row = table[table["industry"] == label].iloc[0, 2:].astype(float)
    row.index = row.index.astype(int)
    return row


def use_code_sector(code: str) -> str | None:
    """Map a summary use-table industry code (e.g. "5415", "4A0", "3361MV") to a sector key."""
    naics2 = "44-45" if code == "4A0" else to_naics2(code)
    for sector in SECTORS:
        if naics2 in sector.naics2:
            return sector.key
    return None


def use_table_purchases() -> pd.DataFrame:
    """Nominal IT-commodity purchases and total output by sector and year, 1997-2023.

    Returns:
        pd.DataFrame: Columns ``sector``, ``year``, ``it_purchases`` (millions of
        dollars, sum of ``IT_COMMODITIES``), ``use_output`` (total industry output)
        and one column per commodity code.
    """
    path = sources.download(BEA_IO_ZIP, RAW / "bea_io_tables.zip")
    target = RAW / USE_TABLE
    if not target.exists():
        with zipfile.ZipFile(path) as archive:
            target.write_bytes(archive.read(USE_TABLE))

    book = openpyxl.load_workbook(target, read_only=True)
    records = []
    for year in book.sheetnames:
        rows = list(book[year].iter_rows(values_only=True))
        codes = rows[5]
        wanted = {*IT_COMMODITIES, USE_TOTAL_OUTPUT_ROW}
        by_row = {str(r[0]): r for r in rows[7:] if r[0] is not None and str(r[0]) in wanted}
        for column, code in enumerate(codes):
            if column < 2 or code is None:
                continue
            sector = use_code_sector(str(code))
            if sector is None:
                continue
            record = {"sector": sector, "year": int(year)}
            for row_code, row in by_row.items():
                record[row_code] = pd.to_numeric(row[column], errors="coerce")
            records.append(record)

    frame = pd.DataFrame(records).fillna(0.0)
    frame = frame.groupby(["sector", "year"], as_index=False).sum()
    frame["it_purchases"] = frame[list(IT_COMMODITIES)].sum(axis=1)
    return frame.rename(columns={USE_TOTAL_OUTPUT_ROW: "use_output"})


def depreciation_by_sector() -> pd.DataFrame:
    """Official current-cost depreciation (private nonresidential, all assets) by sector and year."""
    path = sources.download(BEA_DEPRECIATION, RAW / "bea_fa_detail_dep.xlsx")
    sheet = openpyxl.load_workbook(path, read_only=True)["Datasets"]
    rows = sheet.iter_rows(values_only=True)
    header = [str(c) for c in next(rows)]
    years = {i: int(h) for i, h in enumerate(header) if h.isdigit() and int(h) >= 1990}

    records = []
    for row in rows:
        if not row[0] or row[0][8:12] not in FA_TOTAL_CODES:
            continue
        industry = row[0][3:7]
        records += [(industry, year, row[i]) for i, year in years.items()]
    dep = pd.DataFrame(records, columns=["industry", "year", "dep"])
    dep["dep"] = pd.to_numeric(dep["dep"], errors="coerce")
    dep["sector"] = dep["industry"].map(use_code_sector)
    return dep.dropna(subset=["sector"]).groupby(["sector", "year"], as_index=False)["dep"].sum()


def excess_share(share: pd.Series, era: str) -> pd.Series:
    """Excess of the IT-purchase share over its pre-era path, floored at zero.

    Args:
        share (pd.Series): IT purchases / gross output for one sector, indexed by year.
        era (str): "AI" (linear trend fitted on ``AI_TREND``), "AIflat" (constant
            2019 share), "IT" (constant ``IT_BASE`` share, 1998-2004 only) or
            "ALL" (no subtraction: all purchases count, every year).

    Returns:
        pd.Series: Excess share by year (zero inside the pre-era window).
    """
    if era == "ALL":
        return share.copy()  # upper bound: every IT-services purchase is investment
    if era == "AIflat":
        excess = share - share.loc[AI_TREND[1]]
        excess[share.index <= AI_TREND[1]] = 0.0
        return excess.clip(lower=0.0)
    if era == "AI":
        fit_years = share.loc[AI_TREND[0]:AI_TREND[1]]
        slope, intercept = np.polyfit(fit_years.index, fit_years.to_numpy(), 1)
        path = intercept + slope * share.index.to_numpy()
        excess = share - path
        excess[share.index <= AI_TREND[1]] = 0.0
    else:
        excess = share - share.loc[IT_BASE]
        excess[(share.index <= IT_BASE) | (share.index > 2004)] = 0.0
    return excess.clip(lower=0.0)


def perpetual_inventory(real_flow: pd.Series, delta: float) -> tuple[pd.Series, pd.Series]:
    """Capital stock and depreciation from a real investment flow.

    ``K_t = (1 - delta) K_{t-1} + X_t`` and depreciation ``D_t = delta * K_t``, so
    with ``delta = 1`` depreciation equals the flow (the expensing benchmark).
    """
    stock, dep, previous = {}, {}, 0.0
    for year, flow in real_flow.items():
        previous = (1 - delta) * previous + flow
        stock[year], dep[year] = previous, delta * previous
    return pd.Series(stock), pd.Series(dep)


def investment_panel() -> pd.DataFrame:
    """Annual sector panel with current, AI-as-investment and gross-output measures.

    Returns:
        pd.DataFrame: One row per sector-year with, among others:
        ``growth`` (current real VA per hour), ``growth_adj_{era}`` (adjusted real
        VA per hour), ``go_growth`` (real gross output per hour), ``input_share``
        (intermediate inputs / gross output, percent), labor shares (gross and
        net, current and adjusted, by delta) and their yearly changes.
    """
    value_added = sources.bea_value_added("TVA105-A")       # nominal VA
    real_va = sources.bea_value_added("TVA106-A")           # chained 2017 dollars
    components = sources.bea_value_added("TVA113-A")
    go_nominal = gdp_by_industry(BEA_GROSS_OUTPUT, "bea_grossoutput.xlsx", "TGO105-A")
    go_quantity = gdp_by_industry(BEA_GROSS_OUTPUT, "bea_grossoutput.xlsx", "TGO103-A")
    go_price = gdp_by_industry(BEA_GROSS_OUTPUT, "bea_grossoutput.xlsx", "TGO104-A")
    ii_nominal = gdp_by_industry(BEA_INTERMEDIATE, "bea_intermediate.xlsx", "TII105-A")

    purchases = use_table_purchases()
    depreciation = depreciation_by_sector()

    # Price of IT services (2017 = 100): output-weighted mean of the suppliers' gross-output prices.
    supplier_go = pd.DataFrame({c: bea_line(go_nominal, label) for c, label in IT_COMMODITIES.items()})
    supplier_price = pd.DataFrame({c: bea_line(go_price, label) for c, label in IT_COMMODITIES.items()})
    weights = supplier_go.div(supplier_go.sum(axis=1), axis=0)
    it_price = (weights * supplier_price).sum(axis=1) / 100
    supplier_growth = supplier_go.sum(axis=1) / supplier_go.sum(axis=1).loc[LAST_USE_YEAR]

    panels = []
    for sector in SECTORS:
        years = pd.Index(range(1997, 2026))
        frame = pd.DataFrame(index=years)
        frame["va"] = bea_line(value_added, sector.bea_line)
        frame["real_va"] = bea_line(real_va, sector.bea_line)
        frame["go"] = bea_line(go_nominal, sector.bea_line)
        frame["go_q"] = bea_line(go_quantity, sector.bea_line)
        frame["ii"] = bea_line(ii_nominal, sector.bea_line)

        row = components.index[components["industry"] == sector.bea_line][0]
        comp = components.iloc[row + 1, 2:].astype(float)
        comp.index = comp.index.astype(int)
        frame["comp"] = comp

        hours = all_employee_hours(sector).resample("YS").mean()
        hours.index = hours.index.year
        frame["hours"] = hours

        sector_dep = depreciation[depreciation["sector"] == sector.key].set_index("year")["dep"]
        frame["dep"] = sector_dep

        own = purchases[purchases["sector"] == sector.key].set_index("year")
        it_spend = own["it_purchases"].reindex(years)
        # Extrapolate 2024-2025 with the suppliers' nominal output.
        late = years > LAST_USE_YEAR
        it_spend[late] = own.loc[LAST_USE_YEAR, "it_purchases"] * supplier_growth.reindex(years[late]).to_numpy()
        frame["it_spend"] = it_spend
        frame["it_share"] = frame["it_spend"] / frame["go"]
        frame["it_extrapolated"] = late.astype(int)

        for era in VARIANTS:
            excess = excess_share(frame["it_share"], era) * frame["go"]
            real_excess = excess / it_price.reindex(years)
            frame[f"x_{era}"] = excess
            frame[f"real_va_adj_{era}"] = frame["real_va"] + real_excess
            frame[f"va_adj_{era}"] = frame["va"] + excess
            frame[f"labor_share_gross_adj_{era}"] = 100 * frame["comp"] / frame[f"va_adj_{era}"]
            for delta in DELTAS:
                _, real_dep = perpetual_inventory(real_excess.fillna(0.0), delta)
                new_dep = real_dep * it_price.reindex(years)  # back to current dollars
                net = frame[f"va_adj_{era}"] - frame["dep"] - new_dep
                frame[f"labor_share_net_adj_{era}_d{int(delta * 100)}"] = 100 * frame["comp"] / net

        frame["labor_share"] = 100 * frame["comp"] / frame["va"]
        frame["labor_share_net"] = 100 * frame["comp"] / (frame["va"] - frame["dep"])
        frame["input_share"] = 100 * frame["ii"] / frame["go"]
        frame["growth"] = 100 * np.log(frame["real_va"] / frame["hours"]).diff()
        frame["go_growth"] = 100 * np.log(frame["go_q"] / frame["hours"]).diff()
        for era in VARIANTS:
            frame[f"growth_adj_{era}"] = 100 * np.log(frame[f"real_va_adj_{era}"] / frame["hours"]).diff()

        share_columns = [c for c in frame.columns if c.startswith("labor_share") or c == "input_share"]
        for column in share_columns:
            frame[f"{column}_change"] = frame[column].diff()

        frame.index = pd.to_datetime(frame.index.astype(str))
        panels.append(frame.assign(sector=sector.key).rename_axis("date").reset_index())
    return pd.concat(panels, ignore_index=True)
