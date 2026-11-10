"""Claim C4: does the market price the substitution signal? A rank-weighted backtest.

Signal. A sector is "substituting" when its juniors lose ground: the 4-quarter
change in the QWI junior-minus-prime employment gap is negative. The score is
minus that change, so a high score means juniors are falling fastest.

Timing. QWI is published about four quarters after the reference quarter, so a
quarter's score becomes usable only 12 months after the quarter ends. Weights
formed at the end of month t earn month t+1 returns.

Portfolio. Rank weights across sectors (long high score, short low score),
$1 long and $1 short. A Bayesian overlay scales the position by the
real-time posterior of the strategy's mean return, starting from a prior
centered at zero, so the strategy learns its own sign and can flip it.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from aiprod import sources

# Ken French 49 industries -> QWI NAICS sectors (sectors without a clean match are left out).
FF49_TO_SECTOR = {
    "Agric": "11", "Gold": "21", "Mines": "21", "Coal": "21", "Oil": "21", "Util": "22", "Cnstr": "23",
    "Food": "31-33", "Soda": "31-33", "Beer": "31-33", "Smoke": "31-33", "Toys": "31-33",
    "Hshld": "31-33", "Clths": "31-33", "MedEq": "31-33", "Drugs": "31-33", "Chems": "31-33",
    "Rubbr": "31-33", "Txtls": "31-33", "BldMt": "31-33", "Steel": "31-33", "FabPr": "31-33",
    "Mach": "31-33", "ElcEq": "31-33", "Autos": "31-33", "Aero": "31-33", "Ships": "31-33",
    "Guns": "31-33", "Hardw": "31-33", "Chips": "31-33", "LabEq": "31-33", "Paper": "31-33",
    "Boxes": "31-33", "Whlsl": "42", "Rtail": "44-45", "Trans": "48-49", "Books": "51",
    "Telcm": "51", "Softw": "51", "Banks": "52", "Insur": "52", "Fin": "52", "RlEst": "53",
    "BusSv": "54", "Hlth": "62", "Fun": "71", "Meals": "72", "PerSv": "81",
}
QWI_LAG_MONTHS = 12
PERIODS = {
    "Full": ("1996-01-01", "2026-12-31"),
    "Pre-2020": ("1996-01-01", "2019-12-31"),
    "2020-22": ("2020-01-01", "2022-12-31"),
    "Post-ChatGPT": ("2023-01-01", "2026-12-31"),
}


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily decimal returns into calendar-month returns (indexed by month start)."""
    monthly = (1 + daily).resample("MS").prod(min_count=1) - 1
    return monthly


def sector_returns() -> pd.DataFrame:
    """Monthly sector returns: equal-weight average of the mapped FF49 industries.

    Returns:
        pd.DataFrame: Month x QWI sector, decimal returns.
    """
    industries = to_monthly(sources.ken_french("49_Industry_Portfolios_daily_CSV"))
    mapped = industries[list(FF49_TO_SECTOR)]
    return mapped.T.groupby(FF49_TO_SECTOR).mean().T


def factors() -> pd.DataFrame:
    """Monthly Fama-French five factors, momentum, risk-free rate, and the 2y yield change.

    Returns:
        pd.DataFrame: Columns ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``,
        ``RF``, ``MOM`` (decimals) and ``d_2y`` (monthly change in the 2-year
        Treasury yield, percentage points).
    """
    five = to_monthly(sources.ken_french("F-F_Research_Data_5_Factors_2x3_daily_CSV"))
    momentum = to_monthly(sources.ken_french("F-F_Momentum_Factor_daily_CSV")).rename(columns=lambda c: "MOM")
    two_year = sources.fred("DGS2").resample("MS").last().diff().rename("d_2y")
    return five.join(momentum).join(two_year)


def substitution_scores(gaps: pd.DataFrame, lag_months: int = QWI_LAG_MONTHS) -> pd.DataFrame:
    """Real-time substitution score by sector, as known at each month end.

    Args:
        gaps (pd.DataFrame): Output of ``panels.qwi_age_gaps``.
        lag_months (int): Months between the end of a QWI quarter and the date
            its data can be traded on.

    Returns:
        pd.DataFrame: Month x sector; score = minus the 4-quarter change in the
        junior employment gap (high = juniors falling fastest = substituting).
    """
    wide = gaps.pivot(index="date", columns="industry", values="emp_gap")
    score = -wide.diff(4)

    # A quarter starting at date d ends 3 months later and is usable lag_months after that.
    score.index = score.index + pd.DateOffset(months=3 + lag_months)
    monthly_index = pd.date_range(score.index.min(), "2026-12-01", freq="MS")
    return score.reindex(monthly_index, method="ffill")


def exposure_scores(exposure: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """A constant score per sector (e.g. AI exposure), repeated for every month.

    Used as the benchmark "AI-exposure factor": long the most exposed sectors,
    short the least, regardless of what the labor data say.

    Args:
        exposure (pd.Series): Exposure by sector.
        index (pd.DatetimeIndex): Months to cover.

    Returns:
        pd.DataFrame: Month x sector, every row equal to ``exposure``.
    """
    return pd.DataFrame([exposure.to_numpy()] * len(index), index=index, columns=exposure.index)


def rank_weights(scores: pd.Series) -> pd.Series:
    """Dollar-neutral rank weights: $1 long the high scores, $1 short the low scores.

    Args:
        scores (pd.Series): Score per sector at one date (NaNs ignored).

    Returns:
        pd.Series: Weights summing to zero, with absolute values summing to 2.
    """
    ranks = scores.dropna().rank()
    centered = ranks - ranks.mean()
    return centered / centered.abs().sum() * 2


def backtest(scores: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Monthly long-short returns from rank weights formed one month earlier.

    Args:
        scores (pd.DataFrame): Month x sector scores (as known at month end).
        returns (pd.DataFrame): Month x sector returns.

    Returns:
        pd.DataFrame: Columns ``ls`` (long-short return) and ``turnover``
        (one-way: half the sum of absolute weight changes).
    """
    sectors = scores.columns.intersection(returns.columns)
    weights = scores[sectors].apply(rank_weights, axis=1).reindex(columns=sectors).fillna(0)

    # Weights set at the end of month t are held over month t+1.
    held = weights.shift(1).reindex(returns.index).dropna(how="all")
    ls = (held * returns[sectors].reindex(held.index)).sum(axis=1, min_count=1)
    turnover = held.diff().abs().sum(axis=1) / 2
    return pd.DataFrame({"ls": ls, "turnover": turnover}).dropna(subset=["ls"])


def bayesian_sizing(ls: pd.Series, prior_sd: float = 0.005, half_life_months: float = 60) -> pd.Series:
    """Real-time position size from the posterior of the strategy's mean return.

    Prior: mu ~ N(0, prior_sd^2), agnostic about the sign. Evidence: past
    monthly returns, exponentially down-weighted with ``half_life_months`` so
    that a regime change (AI is not IT) can overturn old evidence. The size is
    the posterior z-score of mu, clipped to [-1, 1], using only returns
    observed before the month being traded.

    Args:
        ls (pd.Series): Monthly long-short returns.
        prior_sd (float): Prior standard deviation of the monthly mean return.
        half_life_months (float): Half-life of the evidence weights.

    Returns:
        pd.Series: Position multiplier for each month, in [-1, 1].
    """
    decay = 0.5 ** (1 / half_life_months)
    sizes = {}
    weighted_n, weighted_sum, weighted_sq = 0.0, 0.0, 0.0

    for date, value in ls.items():
        variance = max(weighted_sq / weighted_n - (weighted_sum / weighted_n) ** 2, 1e-6) if weighted_n > 12 else ls.var()
        posterior_var = 1 / (1 / prior_sd ** 2 + weighted_n / variance)
        posterior_mean = posterior_var * weighted_sum / variance
        sizes[date] = float(np.clip(posterior_mean / np.sqrt(posterior_var), -1, 1))

        weighted_n = decay * weighted_n + 1
        weighted_sum = decay * weighted_sum + value
        weighted_sq = decay * weighted_sq + value ** 2
    return pd.Series(sizes)


def performance(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    """Annualized mean, volatility, Sharpe ratio, max drawdown, and turnover.

    Args:
        returns (pd.Series): Monthly decimal returns.
        turnover (pd.Series | None): Monthly one-way turnover, if available.

    Returns:
        dict[str, float]: ``mean``, ``vol``, ``sharpe``, ``max_drawdown``,
        ``turnover`` (annual) and ``months``.
    """
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "mean": 12 * returns.mean(),
        "vol": np.sqrt(12) * returns.std(),
        "sharpe": np.sqrt(12) * returns.mean() / returns.std(),
        "max_drawdown": drawdown.min(),
        "turnover": 12 * turnover.mean() if turnover is not None else np.nan,
        "months": len(returns),
    }


def factor_regression(returns: pd.Series, factor_data: pd.DataFrame) -> dict[str, float]:
    """Alpha after FF5 + momentum + 2y-yield changes, with Newey-West errors.

    The 2-year yield term asks whether the strategy is a disguised bet on
    interest rates (the discount-rate channel) rather than on cash flows.

    Args:
        returns (pd.Series): Monthly long-short returns (already excess, dollar-neutral).
        factor_data (pd.DataFrame): Output of ``factors``.

    Returns:
        dict[str, float]: ``alpha`` (annualized), ``alpha_t``, ``beta_mkt``,
        ``beta_2y`` (return per 1 pp rise in the 2y yield) and ``beta_2y_t``.
    """
    data = factor_data.join(returns.rename("strategy"), how="inner").dropna()
    X = sm.add_constant(data[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "d_2y"]])
    fit = sm.OLS(data["strategy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {
        "alpha": 12 * fit.params["const"],
        "alpha_t": fit.tvalues["const"],
        "beta_mkt": fit.params["Mkt-RF"],
        "beta_2y": fit.params["d_2y"],
        "beta_2y_t": fit.tvalues["d_2y"],
    }


def evaluate(
    strategies: dict[str, pd.Series], turnovers: dict[str, pd.Series], factor_data: pd.DataFrame
) -> pd.DataFrame:
    """Performance and factor regressions for each strategy and sub-period.

    Args:
        strategies (dict[str, pd.Series]): Named monthly return series.
        turnovers (dict[str, pd.Series]): Monthly turnover for each named strategy.
        factor_data (pd.DataFrame): Output of ``factors``.

    Returns:
        pd.DataFrame: One row per (strategy, period).
    """
    rows = []
    for name, returns in strategies.items():
        for period, (start, end) in PERIODS.items():
            window = returns[start:end].dropna()
            rows.append({
                "strategy": name,
                "period": period,
                **performance(window, turnovers[name][start:end]),
                **factor_regression(window, factor_data),
            })
    return pd.DataFrame(rows)
