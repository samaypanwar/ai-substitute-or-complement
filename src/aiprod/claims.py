"""Estimate claims C1 (productivity) and C2 (junior employment) in both eras.

Every specification is estimated twice with the same code: once for the IT
boom (exposure = 1995 IT capital share) and once for the AI era (exposure =
language-modeling AI exposure). Exposures are standardized, so each
coefficient reads "effect of a one-standard-deviation higher exposure".
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from aiprod.config import AI_ERA, BASELINE, COVID, IT_ERA, QWI_AI, QWI_IT


@dataclass(frozen=True)
class Estimate:
    """One difference-in-differences coefficient.

    Attributes:
        claim (str): Claim label, e.g. "C1".
        era (str): "IT" or "AI".
        outcome (str): Dependent variable.
        beta (float): Coefficient on treatment-period x standardized exposure.
        se (float): Standard error clustered by industry.
        n_obs (int): Observations.
        n_clusters (int): Industries.
    """

    claim: str
    era: str
    outcome: str
    beta: float
    se: float
    n_obs: int
    n_clusters: int


def standardize(series: pd.Series) -> pd.Series:
    """Return ``series`` as z-scores (mean 0, standard deviation 1)."""
    return (series - series.mean()) / series.std()


def in_window(dates: pd.Series, window: tuple[str, str]) -> pd.Series:
    """Boolean mask for dates inside an inclusive (start, end) window."""
    return dates.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))


def did(
    frame: pd.DataFrame, outcome: str, unit: str, claim: str, era: str, controls: tuple[str, ...] = ()
) -> Estimate:
    """Two-way fixed-effects difference-in-differences with clustered errors.

    Estimates ``outcome ~ unit FE + date FE + controls + beta * (treated x
    exposure)``, where ``treated`` marks the era window and ``exposure`` is
    standardized. Date fixed effects absorb anything common to all industries
    (the business cycle, monetary policy), so ``beta`` is identified only from
    differences across industries with different exposure.

    Args:
        frame (pd.DataFrame): Columns ``unit``, ``date``, ``outcome``,
            ``treated`` (0/1), ``exposure`` and any ``controls``.
        outcome (str): Dependent-variable column.
        unit (str): Industry identifier column (also the cluster).
        claim (str): Claim label stored in the result.
        era (str): Era label stored in the result.
        controls (tuple[str, ...]): Extra regressor columns.

    Returns:
        Estimate: The coefficient on ``treated:exposure``; ``outcome`` is suffixed
        with ``|controls`` when controls are used.
    """
    data = frame.dropna(subset=[outcome, "exposure", *controls])
    rhs = " + ".join([f"C({unit})", "C(date)", *controls, "treated:exposure"])
    fit = smf.ols(f"{outcome} ~ {rhs}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data[unit])[0]}
    )
    label = f"{outcome}|{','.join(controls)}" if controls else outcome
    return Estimate(
        claim, era, label,
        beta=float(fit.params["treated:exposure"]),
        se=float(fit.bse["treated:exposure"]),
        n_obs=int(fit.nobs),
        n_clusters=int(data[unit].nunique()),
    )


def event_study(
    frame: pd.DataFrame, outcome: str, unit: str, exposure: pd.Series, base_year: int
) -> pd.DataFrame:
    """Year-by-year exposure coefficients, relative to ``base_year``.

    Estimates ``outcome ~ unit FE + date FE + sum_y gamma_y * 1[year = y] x
    exposure`` with ``gamma_{base_year} = 0``. Flat coefficients before the
    treatment date and a break after it support the difference-in-differences
    reading; a pre-existing trend undermines it.

    Args:
        frame (pd.DataFrame): Columns ``unit``, ``date`` and ``outcome``.
        outcome (str): Dependent-variable column.
        unit (str): Industry identifier column (also the cluster).
        exposure (pd.Series): Raw exposure indexed by unit; standardized here.
        base_year (int): Omitted year.

    Returns:
        pd.DataFrame: Columns ``year``, ``gamma``, ``se`` (clustered by unit).
    """
    data = frame.dropna(subset=[outcome]).copy()
    data["exposure"] = data[unit].map(standardize(exposure))
    data = data.dropna(subset=["exposure"])
    years = sorted(y for y in data["date"].dt.year.unique() if y != base_year)
    for year in years:
        data[f"ex_{year}"] = (data["date"].dt.year == year) * data["exposure"]

    terms = " + ".join(f"ex_{year}" for year in years)
    fit = smf.ols(f"{outcome} ~ C({unit}) + C(date) + {terms}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data[unit])[0]}
    )
    rows = [(year, fit.params[f"ex_{year}"], fit.bse[f"ex_{year}"]) for year in years]
    rows.append((base_year, 0.0, 0.0))
    return pd.DataFrame(rows, columns=["year", "gamma", "se"]).sort_values("year", ignore_index=True)


def c1_productivity(panel: pd.DataFrame, exposures: pd.DataFrame) -> list[Estimate]:
    """C1: did output per hour accelerate more in more-exposed sectors?

    Both eras are compared against the same 2010-2019 baseline, using annual
    growth so the two coefficients are on the same footing. COVID years are
    excluded.

    Args:
        panel (pd.DataFrame): Annual output from ``panels.productivity_panel("A")``.
        exposures (pd.DataFrame): Output of ``panels.sector_exposures``.

    Returns:
        list[Estimate]: One estimate per era.
    """
    panel = panel[~in_window(panel["date"], COVID)]
    results = []
    for era, window, exposure in (("IT", IT_ERA, "it_share"), ("AI", AI_ERA, "ai_exposure")):
        treated = in_window(panel["date"], window)
        sample = panel[treated | in_window(panel["date"], BASELINE)].copy()
        sample["treated"] = treated[sample.index].astype(int)
        sample["exposure"] = sample["sector"].map(standardize(exposures[exposure]))
        results.append(did(sample, "growth", "sector", "C1", era))
    return results


def c2_junior_employment(gaps: pd.DataFrame, exposures: pd.DataFrame) -> list[Estimate]:
    """C2: did juniors lose ground relative to prime-age workers in exposed sectors?

    Outcomes are the log junior-minus-prime gaps in employment and in hires.
    A negative coefficient means juniors lost relative ground where exposure is
    higher. Each era uses its own pre and post windows (``QWI_IT``, ``QWI_AI``).

    Args:
        gaps (pd.DataFrame): Output of ``panels.qwi_age_gaps``.
        exposures (pd.DataFrame): Output of ``panels.naics2_exposures``.

    Returns:
        list[Estimate]: Employment and hires estimates for each era.
    """
    results = []
    for era, windows, exposure in (("IT", QWI_IT, "it_share"), ("AI", QWI_AI, "ai_exposure")):
        post = in_window(gaps["date"], windows["post"])
        sample = gaps[post | in_window(gaps["date"], windows["pre"])].copy()
        sample["treated"] = post[sample.index].astype(int)
        sample["exposure"] = sample["industry"].map(standardize(exposures[exposure]))
        # Junior hiring is cyclical: also ask for fewer juniors *given* sector size.
        for outcome in ("emp_gap", "hire_gap"):
            results.append(did(sample, outcome, "industry", "C2", era))
            results.append(did(sample, outcome, "industry", "C2", era, controls=("log_emp_total",)))
    return results
