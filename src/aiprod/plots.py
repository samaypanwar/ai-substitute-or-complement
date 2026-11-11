"""One function per paper figure. Each takes data the pipeline already built.

Figures follow ``figures`` conventions (ICML / ICLR style): sized to the paper's
text width, no in-figure titles (the LaTeX caption describes the figure),
"(a)/(b)" labels on multi-panel figures, PNG only.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiprod.figures import (AI_BLUE, AQUA, ERA_COLORS, FULL, HALF, IT_ORANGE, MUTED,
                            event_study_panel, mark_event, panel_label, save, zero_line)

CHATGPT = pd.Timestamp("2022-11-30")
NOTE_SIZE = 7.5


def p_value_label(p: float) -> str:
    """Format a bootstrap p-value for an annotation."""
    return "p < 0.01" if p < 0.01 else f"p = {p:.2f}"


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

    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.plot(means.index, means["High exposure"], color=AI_BLUE, label="High AI exposure")
    ax.plot(means.index, means["Low exposure"], color=MUTED, label="Low AI exposure")
    zero_line(ax)
    mark_event(ax, CHATGPT, "ChatGPT")

    ax.set_ylabel("Junior gap, relative to 2019")
    ax.legend(loc="lower left")
    save(fig, "fig01_junior_gap_by_exposure")


def labor_share_by_exposure(incidence: pd.DataFrame, exposure: pd.Series) -> None:
    """Raw data: labor share of value added, high vs low AI-exposure sectors (2019 = 0)."""
    split = exposure > exposure.median()
    data = incidence.assign(group=incidence["sector"].map(split.map({True: "High", False: "Low"})))
    means = data.groupby(["date", "group"])["labor_share"].mean().unstack()
    means = means - means.loc["2019"].mean()

    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.plot(means.index, means["High"], color=AI_BLUE, marker="o", label="High AI exposure")
    ax.plot(means.index, means["Low"], color=MUTED, marker="o", label="Low AI exposure")
    zero_line(ax)
    mark_event(ax, CHATGPT, "ChatGPT")

    ax.set_ylabel("Labor share (pp, relative to 2019)")
    ax.legend(loc="lower left")
    save(fig, "fig04_labor_share_by_exposure")


def c2_event_studies(studies: dict[str, pd.DataFrame]) -> None:
    """Side-by-side event studies of the junior gap: (a) IT boom and (b) AI era."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    labels = {"IT": "(a) IT boom, exposure = 1995 IT capital share",
              "AI": "(b) AI era, exposure = language-model AI exposure"}
    markers = {"IT": (2001, "Dot-com bust"), "AI": (2022.9, "ChatGPT")}

    for ax, era in zip(axes, ("IT", "AI")):
        event_study_panel(ax, studies[era], ERA_COLORS[era])
        mark_event(ax, *markers[era])
        panel_label(ax, labels[era])
        ax.set_xlabel("Year")

    axes[0].set_ylabel("Effect of +1 SD exposure (log pts)")
    save(fig, "fig02_c2_event_studies")


def coefficient_comparison(estimates: pd.DataFrame, rows: dict[str, str]) -> None:
    """Dot-and-whisker plot of IT vs AI coefficients for each claim (95% intervals)."""
    fig, axes = plt.subplots(1, len(rows), figsize=(FULL, 2.1))

    for ax, (label, outcome) in zip(axes, rows.items()):
        subset = estimates[estimates["outcome"] == outcome].set_index("era").loc[["IT", "AI"]]

        for position, (era, row) in enumerate(subset.iterrows()):
            ax.errorbar(position, row["beta"], yerr=1.96 * row["se"], fmt="o",
                        color=ERA_COLORS[era], capsize=0, elinewidth=1.4)
            ax.annotate(p_value_label(row["p_boot"]), (position, row["beta"]), xytext=(6, 5),
                        textcoords="offset points", fontsize=NOTE_SIZE, color=MUTED, va="bottom")

        zero_line(ax)
        ax.set_xticks([0, 1], ["IT boom", "AI era"])
        ax.set_xlim(-0.5, 1.8)
        ax.grid(axis="x", visible=False)
        panel_label(ax, label)
    save(fig, "fig03_claims_it_vs_ai")


def detailed_event_study(study: pd.DataFrame) -> None:
    """Event study of total employment across detailed CES industries (AI exposure)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    event_study_panel(ax, study, AQUA)
    mark_event(ax, 2022.9, "ChatGPT")

    ax.set_xlabel("Year")
    ax.set_ylabel("Effect of +1 SD exposure (log pts)")
    save(fig, "fig05_detailed_employment_event_study")


def c3_event_study(study: pd.DataFrame) -> None:
    """Event study of the labor share level on AI exposure (base 2019)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    event_study_panel(ax, study, AI_BLUE)
    mark_event(ax, 2022.9, "ChatGPT")

    ax.set_xlabel("Year")
    ax.set_ylabel("Effect of +1 SD exposure (pp)")
    save(fig, "fig10_labor_share_event_study")


def exposure_overlap(exposures: pd.DataFrame) -> None:
    """Scatter of 1995 IT capital share against AI exposure across sectors."""
    # Four low-exposure sectors sit on top of each other near the origin: label them as one group.
    crowded = ["retail", "utilities", "leisure", "other_services"]
    # Labels that would collide are placed at fixed data coordinates (x, y, alignment).
    placed = {"manufacturing": (0.034, -0.74, "right"), "mining": (0.014, -0.92, "left")}

    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.scatter(exposures["it_share"], exposures["ai_exposure"], s=18, color=AI_BLUE, zorder=3)
    for sector, row in exposures.drop(crowded).iterrows():
        point = (row["it_share"], row["ai_exposure"])
        if sector in placed:
            x, y, align = placed[sector]
            ax.annotate(sector, point, xytext=(x, y), textcoords="data", ha=align, fontsize=6.5, color=MUTED)
        else:
            ax.annotate(sector.replace("_", " "), point, xytext=(3, 2), textcoords="offset points",
                        fontsize=6.5, color=MUTED)

    # The four low-exposure sectors overlap near the origin: one label for the group.
    ax.annotate("retail, utilities, leisure,\nother services", (0.046, -0.02), textcoords="data",
                fontsize=6.5, color=MUTED, va="center")

    correlation = exposures.corr().iloc[0, 1]
    ax.text(0.98, 0.04, f"corr = {correlation:.2f}", transform=ax.transAxes, ha="right",
            fontsize=NOTE_SIZE, color=MUTED)
    ax.set_xlabel("IT share of capital stock, 1995")
    ax.set_ylabel("Language-model AI exposure")
    save(fig, "fig09_exposure_overlap")


def nowcast_fit(panel: pd.DataFrame) -> None:
    """Average 4-quarter change in the junior gap: QWI actual vs monthly-data nowcast."""
    averages = panel.groupby("date")[["d_emp_gap", "nowcast"]].mean()
    averages = averages[averages.index >= "2001-01-01"]

    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    ax.axvspan(pd.Timestamp("2001-01-01"), pd.Timestamp("2019-12-31"), color=MUTED, alpha=0.06, linewidth=0)
    ax.plot(averages.index, averages["d_emp_gap"], color=MUTED, label="QWI (published with about a 1-year lag)")
    ax.plot(averages.index, averages["nowcast"], color=AI_BLUE, label="Nowcast from monthly CES and CPS")
    zero_line(ax)

    ax.annotate("Training window", (pd.Timestamp("2002-01-01"), 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -9), textcoords="offset points", fontsize=NOTE_SIZE, color=MUTED)
    ax.set_ylabel("4-quarter change (log pts)")
    ax.legend(loc="lower left", ncols=2)
    save(fig, "fig06_nowcast")


def strategy_paths(returns: dict[str, pd.Series], size: pd.Series) -> None:
    """(a) Cumulative log returns of the C4 strategies; (b) the Bayesian position size."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FULL, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    colors = [AI_BLUE, AQUA, MUTED]

    for color, (name, series) in zip(colors, returns.items()):
        top.plot(series.index, np.log1p(series.fillna(0)).cumsum(), color=color, label=name)
    zero_line(top)
    mark_event(top, CHATGPT, "ChatGPT")
    panel_label(top, "(a) Cumulative log return")
    top.legend(loc="lower left")

    bottom.fill_between(size.index, 0, size, color=AI_BLUE, alpha=0.35, linewidth=0)
    zero_line(bottom)
    bottom.set_ylim(-1.05, 1.05)
    bottom.set_xlim(size.index.min(), size.index.max())
    panel_label(bottom, "(b) Bayesian position size (+1 = fully long substituting sectors)")
    save(fig, "fig07_strategy")


def bayesian_pooling(posteriors: pd.DataFrame, labels: dict[str, str]) -> None:
    """Posterior of each AI-era coefficient as the prior widens from "AI = IT" to "AI is new"."""
    claims = list(labels)
    fig, axes = plt.subplots(1, len(claims), figsize=(FULL, 2.2))

    for ax, claim in zip(axes, claims):
        data = posteriors[posteriors["claim"] == claim]
        ax.fill_between(data["tau"], data["low90"], data["high90"], color=AI_BLUE, alpha=0.18, linewidth=0)
        ax.plot(data["tau"], data["mean"], color=AI_BLUE)
        ax.axhline(data["beta_it"].iloc[0], color=IT_ORANGE, linestyle=":", linewidth=1.2)
        ax.axhline(data["beta_ai"].iloc[0], color=AI_BLUE, linestyle=":", linewidth=1.2)
        zero_line(ax)

        panel_label(ax, labels[claim])
        ax.set_xlabel(r"Prior width $\tau$")
    axes[0].set_ylabel("Posterior mean, 90% band")
    save(fig, "fig08_bayesian_pooling")
