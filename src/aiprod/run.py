"""Pipeline entry point: build panels, estimate C1 and C2, write tables.

Run with ``poetry run python -m aiprod.run`` from the project root.
"""

from dataclasses import asdict

import pandas as pd

from aiprod import claims, panels
from aiprod.config import PROCESSED, SECTORS, TABLES


def main() -> None:
    """Build every panel, save it, estimate both eras, and print a summary."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    annual = panels.productivity_panel("A")
    quarterly = panels.productivity_panel("Q")
    sector_exp = panels.sector_exposures()
    naics_exp = panels.naics2_exposures()
    gaps = panels.qwi_age_gaps()

    annual.to_csv(PROCESSED / "productivity_annual.csv", index=False)
    quarterly.to_csv(PROCESSED / "productivity_quarterly.csv", index=False)
    sector_exp.to_csv(PROCESSED / "exposures_sector.csv")
    naics_exp.to_csv(PROCESSED / "exposures_naics2.csv")
    gaps.to_csv(PROCESSED / "qwi_age_gaps.csv", index=False)

    # Check the all-employee hours construction against the official index (2006+).
    drift = {
        sector.key: panels.hours_validation(sector).iloc[-1].pipe(lambda r: r["constructed"] / r["official"] - 1)
        for sector in SECTORS
    }
    pd.Series(drift, name="end_drift").to_csv(TABLES / "hours_validation.csv")

    estimates = claims.c1_productivity(annual, sector_exp) + claims.c2_junior_employment(gaps, naics_exp)
    results = pd.DataFrame([asdict(e) for e in estimates])
    results["t"] = results["beta"] / results["se"]
    results.to_csv(TABLES / "c1_c2_estimates.csv", index=False)

    # Pre-trend checks for C2: junior employment gap by year, both eras.
    in_era = {"AI": gaps["date"].dt.year.between(2012, 2026), "IT": gaps["date"].dt.year.between(1994, 2004)}
    for era, base, exposure in (("AI", 2019, "ai_exposure"), ("IT", 1995, "it_share")):
        study = claims.event_study(gaps[in_era[era]], "emp_gap", "industry", naics_exp[exposure], base)
        study.to_csv(TABLES / f"c2_event_study_{era}.csv", index=False)
        print(f"C2 event study ({era}, base {base}):\n", study.round(3).to_string(index=False), "\n")

    print("Exposures (sector):\n", sector_exp.round(3), "\n")
    print("Corr(IT share 1995, AI exposure): sector",
          round(sector_exp.corr().iloc[0, 1], 2), "| NAICS2", round(naics_exp.corr().iloc[0, 1], 2), "\n")
    print("Hours construction, constructed/official - 1 at latest month:\n",
          pd.Series(drift).map("{:+.1%}".format), "\n")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
