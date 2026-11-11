"""Inference with few industries: wild cluster bootstrap, kink test, Bayesian pooling.

With 13-19 industry clusters, conventional cluster-robust t-statistics
over-reject. The wild cluster bootstrap (restricted, Webb weights) gives
p-values that remain reliable with few clusters (Cameron, Gelbach and Miller
2008; Webb 2023).

The Bayesian combination treats the IT-boom estimate as a prior for the AI-era
coefficient, with ``tau`` measuring how different AI is allowed to be from IT.
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats

# Webb's six-point weights: mean 0, variance 1, better than Rademacher with few clusters.
WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def standardize(series: pd.Series) -> pd.Series:
    """Return ``series`` as z-scores (mean 0, standard deviation 1)."""
    return (series - series.mean()) / series.std()


def cluster_t(
    X: np.ndarray, xtx_inv: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int
) -> tuple[float, float]:
    """OLS coefficient ``k`` and its CR1 cluster-robust t-statistic.

    Args:
        X (np.ndarray): Design matrix, shape (n, p).
        xtx_inv (np.ndarray): (Pseudo-)inverse of X'X, shape (p, p); passed in so
            bootstrap replications reuse it.
        y (np.ndarray): Outcome, shape (n,).
        groups (np.ndarray): Integer cluster codes 0..G-1, shape (n,).
        k (int): Column of the coefficient of interest.

    Returns:
        tuple[float, float]: The coefficient and its t-statistic.
    """
    n, p = X.shape
    n_groups = groups.max() + 1
    beta = xtx_inv @ (X.T @ y)
    residuals = y - X @ beta

    # Variance of beta_k only: sum over clusters of (a' X_g' u_g)^2, a = row k of (X'X)^-1.
    influence = (X @ xtx_inv[k]) * residuals
    cluster_sums = np.bincount(groups, weights=influence, minlength=n_groups)
    correction = n_groups / (n_groups - 1) * (n - 1) / (n - p)
    se = np.sqrt(correction * np.sum(cluster_sums ** 2))
    return beta[k], beta[k] / se


def wild_cluster_bootstrap(
    data: pd.DataFrame, formula: str, term: str, cluster: str, reps: int = 1999, seed: int = 7
) -> dict[str, float]:
    """Restricted wild cluster bootstrap p-value for one regression coefficient.

    The null ``term = 0`` is imposed when generating bootstrap samples (WCR),
    and each sample's cluster-robust t is compared with the observed one.

    Args:
        data (pd.DataFrame): Regression data.
        formula (str): Patsy formula, e.g. ``"y ~ C(unit) + C(date) + treated:exposure"``.
        term (str): Coefficient name as patsy labels it.
        cluster (str): Cluster column in ``data``.
        reps (int): Bootstrap replications.
        seed (int): Random seed.

    Returns:
        dict[str, float]: ``beta``, ``t`` (CR1), ``p_boot`` (two-sided) and ``n_clusters``.
    """
    y_frame, X_frame = dmatrices(formula, data, return_type="dataframe")
    y = y_frame.to_numpy().ravel()
    X = X_frame.to_numpy()
    k = list(X_frame.columns).index(term)
    groups = pd.factorize(data.loc[y_frame.index, cluster])[0]

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta, t_observed = cluster_t(X, xtx_inv, y, groups, k)

    # Fit under the null (drop the tested column) and resample its residuals by cluster.
    X_null = np.delete(X, k, axis=1)
    fitted_null = X_null @ np.linalg.lstsq(X_null, y, rcond=None)[0]
    residuals_null = y - fitted_null

    rng = np.random.default_rng(seed)
    t_boot = np.empty(reps)
    for r in range(reps):
        weights = rng.choice(WEBB_WEIGHTS, size=groups.max() + 1)
        y_star = fitted_null + residuals_null * weights[groups]
        t_boot[r] = cluster_t(X, xtx_inv, y_star, groups, k)[1]

    return {
        "beta": float(beta),
        "t": float(t_observed),
        "p_boot": float(np.mean(np.abs(t_boot) >= abs(t_observed))),
        "n_clusters": int(groups.max() + 1),
    }


def years_since(dates: pd.Series, start: str) -> pd.Series:
    """Years elapsed since ``start``, floored at zero (a linear ramp)."""
    return ((dates - pd.Timestamp(start)).dt.days / 365.25).clip(lower=0)


def kink_test(
    frame: pd.DataFrame, outcome: str, unit: str, exposure: str,
    controls: tuple[str, ...] = (), drop_covid: bool = True,
) -> pd.DataFrame:
    """Did the exposure gradient bend at ChatGPT?

    Main specification (``drop_covid=True``): COVID years (2020-2021) are
    dropped, as everywhere else in the project, because exposed sectors crashed
    and recovered in a V. Three terms enter, each interacted with standardized
    exposure:

    * ``trend`` (years since 2015): the pre-existing slope;
    * ``post_covid`` (1 from 2022): a level shift for lasting COVID scarring;
    * ``ramp_chatgpt`` (years since 2022Q4): the *change* in slope after ChatGPT.

    Alternative (``drop_covid=False``): COVID years kept, with a straight line
    from 2020 (``ramp_2020``) plus ``ramp_chatgpt``. This was our first
    specification; it averages the V into one slope. Both are reported.

    Args:
        frame (pd.DataFrame): Panel with ``unit``, ``date``, ``outcome``,
            the raw ``exposure`` column and any ``controls``.
        outcome (str): Dependent variable.
        unit (str): Industry column (also the cluster).
        exposure (str): Raw exposure column; standardized across units here.
        controls (tuple[str, ...]): Extra regressors.
        drop_covid (bool): Use the main specification (True) or the alternative.

    Returns:
        pd.DataFrame: Rows for each term with ``beta`` (per year, or level for
        ``post_covid``, per SD of exposure), ``t`` and ``p_boot``.
    """
    in_sample = frame["date"] >= "2015-01-01"
    if drop_covid:
        in_sample &= ~frame["date"].dt.year.isin([2020, 2021])
    data = frame[in_sample].copy()

    unit_exposure = data.groupby(unit)[exposure].first()
    data["exposure"] = data[unit].map(standardize(unit_exposure))
    data["trend"] = years_since(data["date"], "2015-01-01")
    data["post_covid"] = (data["date"] >= "2022-01-01").astype(int)
    data["ramp_2020"] = years_since(data["date"], "2020-01-01")
    data["ramp_chatgpt"] = years_since(data["date"], "2022-10-01")
    data = data.dropna(subset=[outcome, "exposure", *controls])

    if drop_covid:
        terms = ("exposure:trend", "exposure:post_covid", "exposure:ramp_chatgpt")
    else:
        terms = ("exposure:ramp_2020", "exposure:ramp_chatgpt")
    rhs = " + ".join([f"C({unit})", "C(date)", *controls, *terms])
    rows = []
    for term in terms:
        result = wild_cluster_bootstrap(data, f"{outcome} ~ {rhs}", term, unit)
        rows.append({"outcome": outcome, "term": term, **result})
    return pd.DataFrame(rows)


def pooled_posteriors(estimates: pd.DataFrame, outcomes: dict[str, str], n_taus: int = 61) -> pd.DataFrame:
    """Posterior for each claim's AI-era coefficient across a grid of prior widths.

    Args:
        estimates (pd.DataFrame): Rows with ``era``, ``outcome``, ``beta``, ``se``
            (as written by the pipeline's DiD table).
        outcomes (dict[str, str]): Claim label -> outcome name to pool,
            e.g. ``{"C2": "emp_gap|log_emp_total"}``.
        n_taus (int): Grid size; tau runs from 0 to 3x the larger of the two
            standard errors.

    Returns:
        pd.DataFrame: ``posterior_given_prior`` output stacked over claims, plus
        ``claim``, ``beta_it`` and ``beta_ai``.
    """
    frames = []
    for claim, outcome in outcomes.items():
        rows = estimates[estimates["outcome"] == outcome].set_index("era")
        it, ai = rows.loc["IT"], rows.loc["AI"]
        taus = np.linspace(0, 3 * max(it["se"], ai["se"]), n_taus)
        posterior = posterior_given_prior(it["beta"], it["se"], ai["beta"], ai["se"], taus)
        frames.append(posterior.assign(claim=claim, beta_it=it["beta"], beta_ai=ai["beta"]))
    return pd.concat(frames, ignore_index=True)


def posterior_given_prior(
    beta_it: float, se_it: float, beta_ai: float, se_ai: float, taus: np.ndarray
) -> pd.DataFrame:
    """Normal-normal posterior for the AI-era coefficient, across prior widths.

    Prior: beta_AI ~ N(beta_IT, se_IT^2 + tau^2), i.e. centered on the IT-boom
    estimate, with ``tau`` the extra uncertainty from "AI is not IT".
    Likelihood: beta_hat_AI ~ N(beta_AI, se_AI^2).

    Args:
        beta_it (float): IT-boom estimate (the prior mean).
        se_it (float): Its standard error.
        beta_ai (float): AI-era estimate.
        se_ai (float): Its standard error.
        taus (np.ndarray): Grid of prior widths; ``tau = 0`` means "AI is exactly IT".

    Returns:
        pd.DataFrame: Columns ``tau``, ``weight_on_ai`` (shrinkage weight on the
        AI-era data), ``mean``, ``sd``, ``low90``, ``high90`` and ``p_negative``.
    """
    prior_var = se_it ** 2 + taus ** 2
    weight = prior_var / (prior_var + se_ai ** 2)
    mean = weight * beta_ai + (1 - weight) * beta_it
    sd = np.sqrt(prior_var * se_ai ** 2 / (prior_var + se_ai ** 2))

    return pd.DataFrame({
        "tau": taus,
        "weight_on_ai": weight,
        "mean": mean,
        "sd": sd,
        "low90": mean - 1.645 * sd,
        "high90": mean + 1.645 * sd,
        "p_negative": stats.norm.cdf(0, loc=mean, scale=sd),
    })
