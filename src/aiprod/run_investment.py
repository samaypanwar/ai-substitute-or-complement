"""C1-C3 with AI (and IT) treated as investment, gross-output measures, and weighted exposure.

Run from the project root with ``python -m aiprod.run_investment``. Writes
``output/tables/investment_*.csv``.
"""

from dataclasses import asdict

import pandas as pd

from aiprod import claims, inference, investment, panels
from aiprod.config import PROCESSED, TABLES

DELTA_TAGS = [f"d{int(d * 100)}" for d in investment.DELTAS]

# Outcome column -> human label, for the summary table.
OUTCOMES = {
    "growth": "C1 | VA per hour, AI as input (current)",
    "growth_inv": "C1 | VA per hour, AI as investment",
    "growth_inv_flat": "C1 | VA per hour, AI as investment (excess over 2019 share)",
    "growth_inv_all": "C1 | VA per hour, ALL IT services as investment (upper bound)",
    "go_growth": "C1 | Gross output per hour",
    "input_share_change": "C1 | Intermediate-input share of gross output, change (pp/yr)",
    "labor_share_change": "C3 | Gross labor share change, current (pp/yr)",
    "labor_share_gross_inv_change": "C3 | Gross labor share change, AI as investment",
    "labor_share_gross_inv_all_change": "C3 | Gross labor share change, ALL IT services as investment",
    "labor_share_net_change": "C3 | Net labor share change, current",
    "labor_share_net_inv_flat_d50_change": "C3 | Net labor share change, AI inv. over 2019 share, delta=50%",
    "labor_share_net_inv_all_d50_change": "C3 | Net labor share change, ALL IT services inv., delta=50%",
    **{f"labor_share_net_inv_{t}_change": f"C3 | Net labor share change, AI as investment, delta={t[1:]}%"
       for t in DELTA_TAGS},
    "labor_share": "C3 | Gross labor share level, current (pp)",
    "labor_share_net": "C3 | Net labor share level, current",
    "labor_share_net_inv_all_d50": "C3 | Net labor share level, ALL IT services inv., delta=50%",
    **{f"labor_share_net_inv_{t}": f"C3 | Net labor share level, AI as investment, delta={t[1:]}%"
       for t in DELTA_TAGS},
}


def combine_eras(panel: pd.DataFrame) -> pd.DataFrame:
    """One column per adjusted measure: IT-era adjustment up to 2004, AI-era adjustment after.

    The two adjustments never overlap (IT excess is zero after 2004, AI excess
    is zero before 2020), so a single series serves both era regressions.
    """
    early = panel["date"].dt.year <= 2004
    pick = lambda stem: panel[f"{stem}_IT"].where(early, panel[f"{stem}_AI"])
    out = panel.copy()
    out["growth_inv"] = pick("growth_adj")
    out["labor_share_gross_inv"] = pick("labor_share_gross_adj")
    for tag in DELTA_TAGS:
        out[f"labor_share_net_inv_{tag}"] = (
            panel[f"labor_share_net_adj_IT_{tag}"].where(early, panel[f"labor_share_net_adj_AI_{tag}"])
        )
    # Bounds: AI excess over the flat 2019 share (IT era as before), and all purchases capitalized.
    out["growth_inv_flat"] = panel["growth_adj_IT"].where(early, panel["growth_adj_AIflat"])
    out["labor_share_net_inv_flat_d50"] = panel["labor_share_net_adj_IT_d50"].where(
        early, panel["labor_share_net_adj_AIflat_d50"])
    out["growth_inv_all"] = panel["growth_adj_ALL"]
    out["labor_share_gross_inv_all"] = panel["labor_share_gross_adj_ALL"]
    out["labor_share_net_inv_all_d50"] = panel["labor_share_net_adj_ALL_d50"]
    for column in ["labor_share_gross_inv", *[f"labor_share_net_inv_{t}" for t in DELTA_TAGS],
                   "labor_share_net_inv_flat_d50", "labor_share_gross_inv_all", "labor_share_net_inv_all_d50"]:
        out[f"{column}_change"] = out.groupby("sector")[column].diff()
    return out


def estimate(panel: pd.DataFrame, gaps: pd.DataFrame, sector_exp, naics_exp, label: str) -> pd.DataFrame:
    """All C1-C3 difference-in-differences for one exposure definition."""
    rows = []
    for outcome in OUTCOMES:
        rows += claims.annual_era_did(panel, sector_exp, outcome, OUTCOMES[outcome][:2])
    rows += claims.c2_junior_employment(gaps, naics_exp)
    frame = pd.DataFrame([asdict(r) for r in rows])
    frame["label"] = frame["outcome"].map(OUTCOMES).fillna("C2 | " + frame["outcome"])
    return frame.assign(exposure=label)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    panel = combine_eras(investment.investment_panel())
    panel.to_csv(PROCESSED / "investment_panel.csv", index=False)
    gaps = panels.qwi_age_gaps()

    exposures = {
        "unweighted LM AIIE (v1)": (panels.sector_exposures(), panels.naics2_exposures()),
        "employment-weighted LM AIIE": (panels.sector_exposures(True), panels.naics2_exposures(True)),
    }
    for label, (sector_exp, naics_exp) in exposures.items():
        tag = "weighted" if "weighted" in label and "un" not in label else "unweighted"
        sector_exp.to_csv(TABLES / f"investment_exposures_sector_{tag}.csv")
        naics_exp.to_csv(TABLES / f"investment_exposures_naics2_{tag}.csv")

    estimates = pd.concat([estimate(panel, gaps, s, n, label) for label, (s, n) in exposures.items()])
    estimates.to_csv(TABLES / "investment_did_estimates.csv", index=False)

    main_est = estimates[estimates["exposure"] == "employment-weighted LM AIIE"]
    pooled = {
        "C1 (VA/hr, AI as investment)": "growth_inv",
        "C1 (gross output/hr)": "go_growth",
        "C2 (junior gap | size)": "emp_gap|log_emp_total",
        "C3 (net labor share change, AI inv., delta=50%)": "labor_share_net_inv_d50_change",
    }
    posteriors = inference.pooled_posteriors(main_est, pooled)
    posteriors.to_csv(TABLES / "investment_bayesian_pooling.csv", index=False)

    # Descriptive: how big is the excess spending? (percent of value added)
    size = panel.assign(year=panel["date"].dt.year)
    size["x_AI_pct_va"] = 100 * size["x_AI"] / size["va"]
    size["x_IT_pct_va"] = 100 * size["x_IT"] / size["va"]
    size["x_AIflat_pct_va"] = 100 * size["x_AIflat"] / size["va"]
    size["it_share_pct"] = 100 * size["it_share"]
    size[size["year"].isin([1997, 2004, 2019, 2023, 2025])].pivot_table(
        index="sector", columns="year", values=["it_share_pct", "x_IT_pct_va", "x_AI_pct_va", "x_AIflat_pct_va"]
    ).round(2).to_csv(TABLES / "investment_spend_size.csv")

    pd.set_option("display.width", 200)
    show = estimates[["exposure", "era", "label", "beta", "se", "p_boot"]].round(3)
    print(show.to_string(index=False))
    summary = posteriors.groupby("claim").apply(
        lambda g: g.iloc[[0, len(g) // 3, -1]][["tau", "weight_on_ai", "mean", "sd", "p_negative"]]
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
