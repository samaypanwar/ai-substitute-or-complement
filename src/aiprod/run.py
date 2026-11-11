"""Pipeline entry point: build every panel, estimate every claim, write tables and figures.

Run from the project root with ``poetry run python -m aiprod.run``.
Data are downloaded once into ``data/raw``; later runs read the cache.
"""

from dataclasses import asdict

import pandas as pd

from aiprod import claims, inference, nowcast, panels, plots, strategy, tables
from aiprod.config import PROCESSED, SECTORS, TABLES
from aiprod.figures import apply_style

# Outcomes compared across eras and pooled in the Bayesian step.
POOLED = {
    "C1": "growth",
    "C2": "emp_gap|log_emp_total",
    "C3": "labor_share_change",
}
POOLED_LABELS = {
    "C1": "(a) C1 productivity",
    "C2": "(b) C2 junior gap",
    "C3": "(c) C3 labor share",
}


def save_table(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write ``frame`` to ``output/tables/<name>.csv``."""
    frame.to_csv(TABLES / f"{name}.csv", index=index)


def real_side(sector_exp: pd.DataFrame, naics_exp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """C1-C3 in both eras, event studies, kink tests, and the detailed-industry panel."""
    annual = panels.productivity_panel("A")
    incidence = panels.incidence_panel()
    gaps = panels.qwi_age_gaps()
    detailed = nowcast.detailed_employment()

    annual.to_csv(PROCESSED / "productivity_annual.csv", index=False)
    incidence.to_csv(PROCESSED / "incidence_annual.csv", index=False)
    gaps.to_csv(PROCESSED / "qwi_age_gaps.csv", index=False)

    estimates = (
        claims.annual_era_did(annual, sector_exp, "growth", "C1")
        + claims.c2_junior_employment(gaps, naics_exp)
        + claims.annual_era_did(incidence, sector_exp, "labor_share_change", "C3")
        + claims.annual_era_did(incidence, sector_exp, "labor_share", "C3")
        + claims.annual_era_did(incidence, sector_exp, "price_growth", "C3")
    )
    estimates = pd.DataFrame([asdict(e) for e in estimates])
    save_table(estimates, "did_estimates")

    # Pre-trend check for C3: labor share level by year (AI exposure, base 2019).
    c3_study = claims.event_study(
        incidence[incidence["date"].dt.year.between(2010, 2025)], "labor_share", "sector",
        sector_exp["ai_exposure"], 2019,
    )
    save_table(c3_study, "c3_event_study")

    # Pre-trend checks: junior gap by year, both eras.
    eras = {"AI": (2012, 2026, 2019, "ai_exposure"), "IT": (1994, 2004, 1995, "it_share")}
    studies = {}
    for era, (first, last, base, exposure) in eras.items():
        window = gaps[gaps["date"].dt.year.between(first, last)]
        studies[era] = claims.event_study(window, "emp_gap", "industry", naics_exp[exposure], base)
        save_table(studies[era], f"c2_event_study_{era}")

    # Did the exposed-sector trends bend at ChatGPT?
    gaps_with_exposure = gaps.assign(ai_exposure=gaps["industry"].map(naics_exp["ai_exposure"]))
    kinks = pd.concat([
        inference.kink_test(gaps_with_exposure, "emp_gap", "industry", "ai_exposure", ("log_emp_total",))
        .assign(panel="QWI sectors"),
        inference.kink_test(detailed, "log_emp", "naics", "ai_exposure").assign(panel="CES detailed industries"),
        inference.kink_test(gaps_with_exposure, "emp_gap", "industry", "ai_exposure", ("log_emp_total",),
                            drop_covid=False).assign(panel="QWI sectors, linear through COVID"),
    ])
    save_table(kinks, "kink_tests")

    detailed_study = claims.event_study(
        detailed[detailed["date"] >= "2012-01-01"], "log_emp", "naics",
        detailed.groupby("naics")["ai_exposure"].first(), 2019,
    )
    save_table(detailed_study, "detailed_event_study")

    return {"estimates": estimates, "studies": studies, "gaps": gaps, "incidence": incidence,
            "detailed": detailed, "detailed_study": detailed_study, "kinks": kinks, "c3_study": c3_study}


def pricing(gaps: pd.DataFrame, naics_exp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """C4 backtests: substitution signal, Bayesian sizing, AI-exposure benchmark."""
    returns = strategy.sector_returns()
    factor_data = strategy.factors()

    signal = strategy.backtest(strategy.substitution_scores(gaps), returns)
    size = strategy.bayesian_sizing(signal["ls"])

    exposure_months = returns.index[returns.index >= "1996-01-01"]
    exposure = strategy.backtest(strategy.exposure_scores(naics_exp["ai_exposure"], exposure_months), returns)

    strategies = {
        "Long substituting / short complementing": signal["ls"],
        "Same, Bayesian-sized": signal["ls"] * size,
        "Benchmark: long high AI exposure": exposure["ls"],
    }
    turnovers = {
        "Long substituting / short complementing": signal["turnover"],
        "Same, Bayesian-sized": signal["turnover"],
        "Benchmark: long high AI exposure": exposure["turnover"],
    }
    results = strategy.evaluate(strategies, turnovers, factor_data)
    save_table(results, "c4_strategy")
    return {"strategies": strategies, "size": size, "results": results}


def main() -> None:
    """Run every stage and write tables and figures."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    apply_style()

    sector_exp = panels.sector_exposures()
    naics_exp = panels.naics2_exposures()
    save_table(sector_exp, "exposures_sector", index=True)
    save_table(naics_exp, "exposures_naics2", index=True)

    hours_check = {s.key: panels.hours_validation(s).iloc[-1] for s in SECTORS}
    save_table(pd.DataFrame(hours_check).T, "hours_validation", index=True)

    real = real_side(sector_exp, naics_exp)
    nowcast_panel, nowcast_fit = nowcast.junior_gap_nowcast(real["gaps"])
    save_table(nowcast_panel, "nowcast")
    save_table(pd.DataFrame([nowcast_fit]), "nowcast_fit")

    priced = pricing(real["gaps"], naics_exp)
    posteriors = inference.pooled_posteriors(real["estimates"], POOLED)
    save_table(posteriors, "bayesian_pooling")

    plots.junior_gap_by_exposure(real["gaps"], naics_exp["ai_exposure"])
    plots.c2_event_studies(real["studies"])
    plots.coefficient_comparison(real["estimates"], {POOLED_LABELS[c]: o for c, o in POOLED.items()})
    plots.labor_share_by_exposure(real["incidence"], sector_exp["ai_exposure"])
    plots.detailed_event_study(real["detailed_study"])
    plots.nowcast_fit(nowcast_panel)
    plots.strategy_paths(priced["strategies"], priced["size"])
    plots.bayesian_pooling(posteriors, POOLED_LABELS)
    plots.exposure_overlap(sector_exp)
    plots.c3_event_study(real["c3_study"])

    tables.did_table(real["estimates"])
    tables.kink_table(real["kinks"])
    tables.strategy_table(priced["results"])

    print(real["estimates"].round(3).to_string(index=False), "\n")
    print(real["kinks"].round(4).to_string(index=False), "\n")
    print("Nowcast out-of-sample:", {k: round(v, 3) for k, v in nowcast_fit.items()}, "\n")
    print(priced["results"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
