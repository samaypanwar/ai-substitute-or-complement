"""Build the analysis panels: sector productivity, exposures, and QWI age gaps.

Units:
    * ``sector`` panels follow ``config.SECTORS`` (13 private industries) and are
      used for output per hour (claim C1).
    * ``naics2`` panels follow the QWI two-digit NAICS sector labels (19 private
      sectors) and are used for employment by age (claim C2).
"""

import numpy as np
import pandas as pd

from aiprod import sources
from aiprod.config import IT_BASE_YEAR, QWI_PRIME, QWI_YOUNG, SECTORS, Sector

# QWI labels that combine several two-digit NAICS codes.
NAICS2_GROUPS = {"31": "31-33", "32": "31-33", "33": "31-33", "44": "44-45", "45": "44-45",
                 "48": "48-49", "49": "48-49"}


def to_naics2(code: str) -> str:
    """Map a NAICS-like code (e.g. "541500", "524A", "110C") to its QWI sector label.

    Args:
        code (str): Industry code whose first two characters are NAICS digits.

    Returns:
        str: QWI sector label such as "54" or "31-33".
    """
    prefix = code[:2]
    return NAICS2_GROUPS.get(prefix, prefix)


def constructed_hours(sector: Sector) -> pd.Series:
    """All-employee hours built from the long PNE series (pre-2006 proxy).

    Scales the production-and-nonsupervisory (PNE) hours index by the ratio of
    all employees to PNE employees, i.e. assumes supervisors work the same
    average weekly hours as PNE workers.

    Args:
        sector (Sector): Sector definition from ``config.SECTORS``.

    Returns:
        pd.Series: Monthly hours index (arbitrary units) for the whole CES supersector.
    """
    pne_hours = sources.fred(f"CES{sector.ces}34")
    pne_employment = sources.fred(f"CES{sector.ces}06")
    return (pne_hours * sources.fred(sector.ae_id) / pne_employment).dropna()


def all_employee_hours(sector: Sector) -> pd.Series:
    """Monthly all-employee hours: official CES index from 2006, proxy before.

    The official all-employee index (``CES{ces}16``) is used wherever it exists;
    the constructed series fills earlier months, rebased to match at the splice.
    When BEA output covers only part of the CES supersector (finance), hours are
    scaled by that part's share of supersector employment.

    Args:
        sector (Sector): Sector definition from ``config.SECTORS``.

    Returns:
        pd.Series: Monthly hours index (arbitrary units), named ``sector.key``.
    """
    official = sources.fred(f"CES{sector.ces}16")
    proxy = constructed_hours(sector)
    splice = official.index[0]
    early = proxy[proxy.index < splice] * official.iloc[0] / proxy.loc[splice]
    hours = pd.concat([early, official])

    if sector.ae_sub_id:
        hours = hours * sources.fred(sector.ae_sub_id) / sources.fred(sector.ae_id)
    return hours.dropna().rename(sector.key)


def hours_validation(sector: Sector) -> pd.DataFrame:
    """Compare the pre-2006 proxy with the official CES index where both exist.

    Args:
        sector (Sector): Sector definition; the official series is ``CES{ces}16``.

    Returns:
        pd.DataFrame: Columns ``constructed`` and ``official``, both rebased to
        100 in their first common month.
    """
    both = pd.concat(
        {"constructed": constructed_hours(sector), "official": sources.fred(f"CES{sector.ces}16")},
        axis=1,
    ).dropna()
    return 100 * both / both.iloc[0]


def productivity_panel(frequency: str) -> pd.DataFrame:
    """Real value added per hour by sector, with its growth rate.

    Args:
        frequency (str): ``"A"`` for annual (1997+) or ``"Q"`` for quarterly (2005+).

    Returns:
        pd.DataFrame: Columns ``sector``, ``date``, ``output`` (BEA chain-type
        quantity index), ``hours``, ``productivity`` and ``growth`` (100 x log
        change over one year: one period if annual, four if quarterly).
    """
    output_table = sources.bea_value_added(f"TVA103-{frequency}")
    annual = frequency == "A"
    lag = 1 if annual else 4
    resample_rule = "YS" if annual else "QS"
    period_freq = "Y" if annual else "Q"  # pandas 3 alias; BEA sheets still use "-A"

    panels = []
    for sector in SECTORS:
        line = output_table[output_table["industry"] == sector.bea_line]
        output = line.iloc[0, 2:].astype(float)
        output.index = pd.PeriodIndex(output.index, freq=period_freq).to_timestamp()
        hours = all_employee_hours(sector).resample(resample_rule).mean()

        frame = pd.DataFrame({"output": output, "hours": hours}).dropna()
        frame["productivity"] = frame["output"] / frame["hours"]
        frame["growth"] = 100 * np.log(frame["productivity"]).diff(lag)
        panels.append(frame.assign(sector=sector.key).rename_axis("date").reset_index())
    return pd.concat(panels, ignore_index=True)


def naics2_exposures() -> pd.DataFrame:
    """AI and IT exposure for each QWI two-digit NAICS sector.

    AI exposure is the unweighted mean of the language-modeling AIIE across the
    detailed industries in the sector. IT exposure is the IT share of the
    sector's net capital stock in ``IT_BASE_YEAR``, aggregated by summing
    dollars across BEA fixed-asset industries.

    Returns:
        pd.DataFrame: Columns ``ai_exposure`` and ``it_share``, indexed by QWI
        sector label.
    """
    aiie = sources.language_model_exposure()
    ai = aiie.groupby(aiie.index.map(to_naics2)).mean()

    stock = sources.it_capital_stock(IT_BASE_YEAR)
    stock = stock.groupby(stock.index.map(to_naics2)).sum()
    it = stock["it"] / stock["total"]
    return pd.DataFrame({"ai_exposure": ai, "it_share": it}).dropna()


def sector_exposures() -> pd.DataFrame:
    """AI and IT exposure aggregated to the 13 productivity-panel sectors.

    AI exposure averages the member NAICS sectors; IT share sums capital
    stocks across them.

    Returns:
        pd.DataFrame: Columns ``ai_exposure`` and ``it_share``, indexed by sector key.
    """
    aiie = sources.language_model_exposure()
    stock = sources.it_capital_stock(IT_BASE_YEAR)
    aiie_naics2 = aiie.index.map(to_naics2)
    stock_naics2 = stock.index.map(to_naics2)

    rows = {}
    for sector in SECTORS:
        members = stock.loc[stock_naics2.isin(sector.naics2)]
        rows[sector.key] = {
            "ai_exposure": aiie.loc[aiie_naics2.isin(sector.naics2)].mean(),
            "it_share": members["it"].sum() / members["total"].sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def qwi_age_gaps() -> pd.DataFrame:
    """Log junior-minus-prime gaps in employment and hires by sector and quarter.

    Juniors are ``QWI_YOUNG`` (22-24); prime age is ``QWI_PRIME`` (35-54).
    A falling gap means juniors lose ground relative to prime-age workers in
    the same sector and quarter.

    Returns:
        pd.DataFrame: Columns ``industry``, ``date``, ``emp_gap``, ``hire_gap``
        and ``log_emp_total`` (log all-age sector employment, the demand control).
    """
    qwi = sources.qwi_by_age()
    total = qwi[qwi["agegrp"] == "A00"].set_index(["industry", "date"])["emp"]
    qwi["group"] = np.select(
        [qwi["agegrp"].isin(QWI_YOUNG), qwi["agegrp"].isin(QWI_PRIME)], ["young", "prime"], ""
    )
    qwi = qwi[qwi["group"] != ""]

    totals = qwi.groupby(["industry", "date", "group"])[["emp", "hires"]].sum(min_count=1)
    wide = totals.unstack("group")
    gaps = pd.DataFrame({
        "emp_gap": np.log(wide[("emp", "young")]) - np.log(wide[("emp", "prime")]),
        "hire_gap": np.log(wide[("hires", "young")]) - np.log(wide[("hires", "prime")]),
        "log_emp_total": np.log(total.reindex(wide.index)),
    })
    return gaps.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
