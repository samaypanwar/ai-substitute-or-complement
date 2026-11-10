"""Estimate claims C1-C3 in both eras with the same difference-in-differences.

Every specification runs twice with the same code: once for the IT boom
(exposure = 1995 IT capital share) and once for the AI era (exposure =
language-modeling AI exposure). Exposures are standardized, so each
coefficient reads "effect of a one-standard-deviation higher exposure".
P-values come from the wild cluster bootstrap because there are only 13-19
industries.
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from aiprod.config import AI_ERA, BASELINE, COVID, IT_ERA, QWI_AI, QWI_IT
from aiprod.inference import standardize, wild_cluster_bootstrap

ERAS = {"IT": (IT_ERA, "it_share"), "AI": (AI_ERA, "ai_exposure")}
QWI_ERAS = {"IT": (QWI_IT, "it_share"), "AI": (QWI_AI, "ai_exposure")}


@dataclass(frozen=True)
class Estimate:
    """One difference-in-differences coefficient.

    Attributes:
        claim (str): Claim label, e.g. "C1".
        era (str): "IT" or "AI".
        outcome (str): Dependent variable, with ``|controls`` appended if any.
        beta (float): Coefficient on treatment-period x standardized exposure.
        se (float): CR1 cluster-robust standard error.
        p_boot (float): Wild cluster bootstrap p-value (two-sided).
        n_obs (int): Observations.
        n_clusters (int): Industries.
    """

    claim: str
    era: str
    outcome: str
    beta: float
    se: float
    p_boot: float
    n_obs: int
    n_clusters: int


def in_window(dates: pd.Series, window: tuple[str, str]) -> pd.Series:
    """Boolean mask for dates inside an inclusive (start, end) window."""
    return dates.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))


def did(
    frame: pd.DataFrame, outcome: str, unit: str, claim: str, era: str, controls: tuple[str, ...] = ()
) -> Estimate:
    """Two-way fixed-effects difference-in-differences with bootstrap inference.

    Estimates ``outcome ~ unit FE + date FE + controls + beta * (treated x
    exposure)``. Date fixed effects absorb everything common to all industries
    (the business cycle, monetary policy), so ``beta`` comes only from
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
        Estimate: The coefficient on ``treated:exposure``.
    """
    data = frame.dropna(subset=[outcome, "exposure", *controls])
    rhs = " + ".join([f"C({unit})", "C(date)", *controls, "treated:exposure"])
    result = wild_cluster_bootstrap(data, f"{outcome} ~ {rhs}", "treated:exposure", unit)

    label = f"{outcome}|{','.join(controls)}" if controls else outcome
    return Estimate(
        claim, era, label,
        beta=result["beta"],
        se=result["beta"] / result["t"],
        p_boot=result["p_boot"],
        n_obs=len(data),
        n_clusters=result["n_clusters"],
    )


def annual_era_did(panel: pd.DataFrame, exposures: pd.DataFrame, outcome: str, claim: str) -> list[Estimate]:
    """Era-versus-baseline DiD on an annual sector panel, for both eras.

    Both eras are compared against the same 2010-2019 baseline so that the IT
    and AI coefficients are on the same footing. COVID years are dropped.

    Args:
        panel (pd.DataFrame): Columns ``sector``, ``date`` and ``outcome``.
        exposures (pd.DataFrame): Output of ``panels.sector_exposures``.
        outcome (str): Dependent-variable column.
        claim (str): Claim label.

    Returns:
        list[Estimate]: One estimate per era.
    """
    panel = panel[~in_window(panel["date"], COVID)]

    results = []
    for era, (window, exposure) in ERAS.items():
        treated = in_window(panel["date"], window)
        sample = panel[treated | in_window(panel["date"], BASELINE)].copy()

        sample["treated"] = treated[sample.index].astype(int)
        sample["exposure"] = sample["sector"].map(standardize(exposures[exposure]))
        results.append(did(sample, outcome, "sector", claim, era))
    return results


def c2_junior_employment(gaps: pd.DataFrame, exposures: pd.DataFrame) -> list[Estimate]:
    """C2: did juniors lose ground relative to prime-age workers in exposed sectors?

    Outcomes are the log junior-minus-prime gaps in employment and hires, with
    and without a control for sector size (junior hiring is cyclical, so the
    substitution test is fewer juniors *given* sector employment).

    Args:
        gaps (pd.DataFrame): Output of ``panels.qwi_age_gaps``.
        exposures (pd.DataFrame): Output of ``panels.naics2_exposures``.

    Returns:
        list[Estimate]: Four estimates per era.
    """
    results = []
    for era, (windows, exposure) in QWI_ERAS.items():
        post = in_window(gaps["date"], windows["post"])
        sample = gaps[post | in_window(gaps["date"], windows["pre"])].copy()

        sample["treated"] = post[sample.index].astype(int)
        sample["exposure"] = sample["industry"].map(standardize(exposures[exposure]))

        for outcome in ("emp_gap", "hire_gap"):
            results.append(did(sample, outcome, "industry", "C2", era))
            results.append(did(sample, outcome, "industry", "C2", era, controls=("log_emp_total",)))
    return results


def event_study(
    frame: pd.DataFrame, outcome: str, unit: str, exposure: pd.Series, base_year: int
) -> pd.DataFrame:
    """Year-by-year exposure coefficients, relative to ``base_year``.

    Estimates ``outcome ~ unit FE + date FE + sum_y gamma_y * 1[year = y] x
    exposure`` with ``gamma_{base_year} = 0``. Flat coefficients before the
    treatment date and a break after it support the DiD reading; a
    pre-existing trend undermines it.

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
