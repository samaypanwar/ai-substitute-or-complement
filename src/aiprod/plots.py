"""One function per paper figure. Each takes data the pipeline already built.

All figures use ``figures.apply_style`` and are saved by ``figures.save`` as
400-dpi PNG and vector PDF in ``output/figures``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiprod.figures import (AI_BLUE, AQUA, COLUMN, ERA_COLORS, FULL, IT_ORANGE, MUTED,
                            event_study_panel, mark_event, save, zero_line)

CHATGPT = pd.Timestamp("2022-11-30")


def junior_gap_by_exposure(gaps: pd.DataFrame, exposure: pd.Series) -> None:
    """Raw data: junior-minus-prime employment gap for high- vs low-AI-exposure sectors.

    Sectors are split into exposure terciles; each line is the tercile average
    of the log gap, re-centered at its 2019 mean so lines start together.
    """
    tercile = pd.qcut(exposure, 3, labels=["Low exposure", "Middle", "High exposure"])
    data = gaps.assign(tercile=gaps["industry"].map(tercile)).dropna(subset=["tercile"])
    data = data[data["date"] >= "2010-01-01"]

    means = data.groupby(["date", "tercile"], observed=True)["emp_gap"].mean().unstack()
    means = means - means.loc["2019"].mean()

    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    ax.plot(means.index, means["High exposure"], color=AI_BLUE, label="High AI exposure")
    ax.plot(means.index, means["Low exposure"], color=MUTED, label="Low AI exposure")
    zero_line(ax)
    mark_event(ax, CHATGPT, "ChatGPT")
    ax.set_ylabel("log(22-24) - log(35-54)\nrelative to 2019")
    ax.legend(loc="lower left")
    save(fig, "fig01_junior_gap_by_exposure")


def c2_event_studies(studies: dict[str, pd.DataFrame]) -> None:
    """Side-by-side event studies of the junior gap: IT boom (left) and AI era (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    titles = {"IT": "IT boom: exposure = 1995 IT capital share", "AI": "AI era: exposure = LM AI exposure"}
    markers = {"IT": (2001, "Dot-com bust"), "AI": (2022.9, "ChatGPT")}

    for ax, era in zip(axes, ("IT", "AI")):
        event_study_panel(ax, studies[era], ERA_COLORS[era])
        mark_event(ax, markers[era][0], markers[era][1])
        ax.set_title(titles[era], loc="left")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Effect of +1 SD exposure on\njunior employment gap (log pts)")
    save(fig, "fig02_c2_event_studies")


def coefficient_comparison(estimates: pd.DataFrame, rows: dict[str, str]) -> None:
    """Dot-and-whisker plot of IT vs AI coefficients for each claim (95% intervals)."""
    fig, axes = plt.subplots(1, len(rows), figsize=(FULL, 1.9))

    for ax, (label, outcome) in zip(axes, rows.items()):
        subset = estimates[estimates["outcome"] == outcome].set_index("era").loc[["IT", "AI"]]
        for position, (era, row) in enumerate(subset.iterrows()):
            ax.errorbar(position, row["beta"], yerr=1.96 * row["se"], fmt="o", color=ERA_COLORS[era],
                        capsize=0, elinewidth=1.4)
            p_label = "p<0.01" if row["p_boot"] < 0.01 else f"p={row['p_boot']:.2f}"
            ax.annotate(p_label, (position, row["beta"]), xytext=(6, 5),
                        textcoords="offset points", fontsize=6.5, color=MUTED, va="bottom")
        zero_line(ax)
        ax.set_xticks([0, 1], ["IT boom", "AI era"])
        ax.set_xlim(-0.5, 1.7)
        ax.set_title(label, loc="left")
    save(fig, "fig03_claims_it_vs_ai")


def labor_share_by_exposure(incidence: pd.DataFrame, exposure: pd.Series) -> None:
    """Raw data: labor share of value added, high vs low AI-exposure sectors (2019 = 0)."""
    split = exposure > exposure.median()
    data = incidence.assign(group=incidence["sector"].map(split.map({True: "High", False: "Low"})))
    means = data.groupby(["date", "group"])["labor_share"].mean().unstack()
    means = means - means.loc["2019"].mean()

    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    ax.plot(means.index, means["High"], color=AI_BLUE, marker="o", label="High AI exposure")
    ax.plot(means.index, means["Low"], color=MUTED, marker="o", label="Low AI exposure")
    zero_line(ax)
    mark_event(ax, CHATGPT, "ChatGPT")
    ax.set_ylabel("Labor share of value added\n(pp, relative to 2019)")
    ax.legend(loc="lower left")
    save(fig, "fig04_labor_share_by_exposure")


def detailed_event_study(study: pd.DataFrame, n_industries: int) -> None:
    """Event study of total employment across detailed CES industries (AI exposure)."""
    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    event_study_panel(ax, study, AQUA)
    mark_event(ax, 2022.9, "ChatGPT")
    ax.set_ylabel("Effect of +1 SD exposure\non log employment")
    ax.set_title(f"{n_industries} detailed industries, monthly CES", loc="left")
    save(fig, "fig05_detailed_employment_event_study")


def nowcast_fit(panel: pd.DataFrame) -> None:
    """Average 4-quarter change in the junior gap: QWI actual vs monthly-data nowcast."""
    averages = panel.groupby("date")[["d_emp_gap", "nowcast"]].mean()
    averages = averages[averages.index >= "2001-01-01"]

    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    ax.axvspan(pd.Timestamp("2001-01-01"), pd.Timestamp("2019-12-31"), color=MUTED, alpha=0.06, linewidth=0)
    ax.plot(averages.index, averages["d_emp_gap"], color=MUTED, label="QWI (published with ~1 year lag)")
    ax.plot(averages.index, averages["nowcast"], color=AI_BLUE, label="Nowcast from monthly CES and CPS")
    zero_line(ax)
    ax.annotate("Training window", (pd.Timestamp("2002-01-01"), 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -8), textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_ylabel("4-quarter change in\njunior gap (log pts)")
    ax.legend(loc="lower left", ncols=2)
    save(fig, "fig06_nowcast")


def strategy_paths(returns: dict[str, pd.Series], size: pd.Series) -> None:
    """Cumulative log returns of the C4 strategies, with the Bayesian position size below."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FULL, 3.4), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    colors = [AI_BLUE, AQUA, MUTED]

    for color, (name, series) in zip(colors, returns.items()):
        top.plot(series.index, np.log1p(series.fillna(0)).cumsum(), color=color, label=name)
    zero_line(top)
    mark_event(top, CHATGPT, "ChatGPT")
    top.set_ylabel("Cumulative log return")
    top.legend(loc="lower left")

    bottom.fill_between(size.index, 0, size, color=AI_BLUE, alpha=0.35, linewidth=0)
    zero_line(bottom)
    bottom.set_ylim(-1.05, 1.05)
    bottom.set_xlim(size.index.min(), size.index.max())
    bottom.set_ylabel("Bayesian size\n(+1 = long S)")
    save(fig, "fig07_strategy")


def bayesian_pooling(posteriors: pd.DataFrame, labels: dict[str, str]) -> None:
    """Posterior of each AI-era coefficient as the prior widens from "AI = IT" to "AI is new"."""
    claims = list(labels)
    fig, axes = plt.subplots(1, len(claims), figsize=(FULL, 2.0))

    for ax, claim in zip(axes, claims):
        data = posteriors[posteriors["claim"] == claim]
        ax.fill_between(data["tau"], data["low90"], data["high90"], color=AI_BLUE, alpha=0.18, linewidth=0)
        ax.plot(data["tau"], data["mean"], color=AI_BLUE)
        ax.axhline(data["beta_it"].iloc[0], color=IT_ORANGE, linestyle=":", linewidth=1)
        ax.axhline(data["beta_ai"].iloc[0], color=AI_BLUE, linestyle=":", linewidth=1)
        zero_line(ax)
        ax.set_title(labels[claim], loc="left")
        ax.set_xlabel(r"Prior width $\tau$ (AI may differ from IT)")
    axes[0].set_ylabel("Posterior mean, 90% band")
    save(fig, "fig08_bayesian_pooling")


def exposure_overlap(exposures: pd.DataFrame) -> None:
    """Scatter of 1995 IT capital share against AI exposure across sectors."""
    # Hand-placed label offsets (points) for sectors that crowd each other near the origin.
    offsets = {"leisure": (-4, -9), "utilities": (4, -7), "other_services": (4, 4),
               "manufacturing": (-48, -9), "transport": (4, 3), "retail": (4, 2)}

    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    ax.scatter(exposures["it_share"], exposures["ai_exposure"], s=18, color=AI_BLUE, zorder=3)
    for sector, row in exposures.iterrows():
        ax.annotate(sector.replace("_", " "), (row["it_share"], row["ai_exposure"]),
                    xytext=offsets.get(sector, (3, 2)), textcoords="offset points", fontsize=6, color=MUTED)
    correlation = exposures.corr().iloc[0, 1]
    ax.set_title(f"Correlation = {correlation:.2f}", loc="left")
    ax.set_xlabel("IT share of capital stock, 1995")
    ax.set_ylabel("Language-model AI exposure")
    ax.grid(axis="x", color="#d9d8d4", linewidth=0.5)
    save(fig, "fig09_exposure_overlap")
