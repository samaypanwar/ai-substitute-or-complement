"""Monthly data: a detailed-industry employment panel and a nowcast of the QWI junior gap.

Two problems with the sector panels motivate this module:

* Power. Thirteen to nineteen sectors are few clusters. CES publishes monthly
  employment for well over a hundred detailed industries, which we can match
  directly to the AI exposure index at the NAICS level.
* Timeliness. QWI (employment by age) arrives about four quarters late. The
  monthly employment report arrives within weeks, so we nowcast the junior gap
  from it and check the nowcast out of sample before trusting it.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from aiprod import sources
from aiprod.config import SECTORS
from aiprod.panels import to_naics2

# CES supersector codes by NAICS prefix; manufacturing splits into durable (31) and nondurable (32).
SUPERSECTOR = {"21": "10", "23": "20", "42": "41", "44": "42", "45": "42", "48": "43", "49": "43",
               "22": "44", "51": "50", "52": "55", "53": "55", "54": "60", "55": "60", "56": "60",
               "61": "65", "62": "65", "71": "70", "72": "70", "81": "80"}
DURABLE_PREFIXES = ("321", "327", "331", "332", "333", "334", "335", "336", "337", "339")

# Monthly all-employee employment at the QWI sector level.
QWI_SECTOR_EMPLOYMENT = {
    "21": "USMINE", "22": "CES4422000001", "23": "USCONS", "31-33": "MANEMP", "42": "USWTRADE",
    "44-45": "USTRADE", "48-49": "CES4300000001", "51": "USINFO", "52": "CES5552000001",
    "53": "CES5553000001", "54": "CES6054000001", "55": "CES6055000001", "56": "CES6056000001",
    "61": "CES6561000001", "62": "CES6562000001", "71": "CES7071000001", "72": "CES7072000001",
    "81": "USSERV",
}


def ces_id(naics: str) -> str:
    """CES all-employee series ID for a 6-digit NAICS-style code (e.g. "541500").

    Args:
        naics (str): Six-character industry code, zero-padded.

    Returns:
        str: Candidate FRED ID such as ``"CES6054150001"``.
    """
    if naics[:2] in ("31", "32", "33"):
        supersector = "31" if naics.startswith(DURABLE_PREFIXES) else "32"
    else:
        supersector = SUPERSECTOR[naics[:2]]
    return f"CES{supersector}{naics}01"


def detailed_employment() -> pd.DataFrame:
    """Monthly employment (1990+) for detailed industries matched to AI exposure.

    The exposure index is mostly at 4-digit NAICS; CES publishes some 4-digit
    and some 3-digit industries. Each 4-digit industry is matched at 4 digits
    when CES has it; otherwise 4-digit industries are pooled into their 3-digit
    parent (exposure averaged), but only when no sibling was matched at 4
    digits, so no employment is counted twice.

    Returns:
        pd.DataFrame: Columns ``naics``, ``naics2``, ``date``, ``log_emp`` and
        ``ai_exposure`` (raw LM AIIE).
    """
    exposure = sources.language_model_exposure()
    in_scope = [code for code in exposure.index if code[:2] in SUPERSECTOR or code[:2] in ("31", "32", "33")]
    four_digit = exposure[in_scope].groupby(lambda code: code[:4]).mean()

    matched = {}
    for code4, score in four_digit.items():
        series = sources.fred_if_available(ces_id(f"{code4}00"))
        if series is not None:
            matched[f"{code4}00"] = (series, score)

    # Fall back to 3-digit parents with no 4-digit child already matched.
    covered_parents = {code[:3] for code in matched}
    three_digit = four_digit.groupby(lambda code: code[:3]).mean()
    for code3, score in three_digit.items():
        if code3 in covered_parents:
            continue
        series = sources.fred_if_available(ces_id(f"{code3}000"))
        if series is not None:
            matched[f"{code3}000"] = (series, score)

    frames = []
    for naics, (series, score) in matched.items():
        series = series[series.index >= "1990-01-01"]
        frames.append(pd.DataFrame({
            "naics": naics,
            "naics2": to_naics2(naics),
            "date": series.index,
            "log_emp": np.log(series.to_numpy()),
            "ai_exposure": score,
        }))
    return pd.concat(frames, ignore_index=True)


def supervisory_share() -> pd.DataFrame:
    """Monthly share of supervisory (non-PNE) employees by QWI sector.

    CES splits production/nonsupervisory employees only at the supersector
    level, so every QWI sector inherits its supersector's share.

    Returns:
        pd.DataFrame: Columns ``industry`` (QWI sector), ``date``, ``supervisory_share``.
    """
    frames = []
    for sector in SECTORS:
        share = 1 - sources.fred(f"CES{sector.ces}06") / sources.fred(sector.ae_id)
        for industry in sector.naics2:
            frames.append(pd.DataFrame({"industry": industry, "date": share.index, "supervisory_share": share.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def monthly_predictors() -> pd.DataFrame:
    """Quarterly averages of the monthly predictors, by QWI sector.

    Predictors are 4-quarter changes in: log sector employment, the
    supervisory share, and the young-minus-prime unemployment rate gap
    (CPS ages 20-24 minus 25-54, a national series).

    Returns:
        pd.DataFrame: Columns ``industry``, ``date`` (quarter start),
        ``d_log_emp``, ``d_supervisory`` and ``d_youth_unemployment``.
    """
    employment = pd.concat(
        [pd.DataFrame({"industry": k, "date": s.index, "log_emp": np.log(s.to_numpy())})
         for k, s in ((k, sources.fred(v)) for k, v in QWI_SECTOR_EMPLOYMENT.items())],
        ignore_index=True,
    )
    monthly = employment.merge(supervisory_share(), on=["industry", "date"])

    youth_gap = (sources.fred("LNS14000036") - sources.fred("LNS14000060")).rename("youth_unemployment")
    monthly = monthly.merge(youth_gap, left_on="date", right_index=True)

    monthly["date"] = monthly["date"].dt.to_period("Q").dt.to_timestamp()
    quarterly = monthly.groupby(["industry", "date"]).mean().reset_index()

    by_sector = quarterly.groupby("industry")
    quarterly["d_log_emp"] = by_sector["log_emp"].diff(4)
    quarterly["d_supervisory"] = by_sector["supervisory_share"].diff(4)
    quarterly["d_youth_unemployment"] = by_sector["youth_unemployment"].diff(4)
    return quarterly[["industry", "date", "d_log_emp", "d_supervisory", "d_youth_unemployment"]]


NOWCAST_FORMULA = "d_emp_gap ~ C(industry) + d_log_emp + d_supervisory + d_youth_unemployment"


def junior_gap_nowcast(gaps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Nowcast the 4-quarter change in the QWI junior gap from monthly data.

    The model is fitted on 2001-2019 only, evaluated out of sample on the
    quarters QWI has already published since 2023, and then used to fill the
    quarters QWI has not yet published.

    Args:
        gaps (pd.DataFrame): Output of ``panels.qwi_age_gaps``.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: Panel with ``d_emp_gap``
        (actual, where published) and ``nowcast``; and out-of-sample fit
        statistics (``oos_r2``, ``oos_corr``, ``n_oos``).
    """
    actual = gaps[["industry", "date", "emp_gap"]].copy()
    actual["d_emp_gap"] = actual.groupby("industry")["emp_gap"].diff(4)

    panel = monthly_predictors().merge(actual, on=["industry", "date"], how="left")
    panel = panel.dropna(subset=["d_log_emp", "d_supervisory", "d_youth_unemployment"])

    train = panel[panel["date"].between("2001-01-01", "2019-12-31")].dropna(subset=["d_emp_gap"])
    fit = smf.ols(NOWCAST_FORMULA, data=train).fit()
    panel["nowcast"] = fit.predict(panel)

    test = panel[(panel["date"] >= "2023-01-01")].dropna(subset=["d_emp_gap"])
    errors = test["d_emp_gap"] - test["nowcast"]
    fit_stats = {
        "oos_r2": float(1 - np.sum(errors ** 2) / np.sum((test["d_emp_gap"] - test["d_emp_gap"].mean()) ** 2)),
        "oos_corr": float(np.corrcoef(test["d_emp_gap"], test["nowcast"])[0, 1]),
        "n_oos": len(test),
    }
    return panel, fit_stats
