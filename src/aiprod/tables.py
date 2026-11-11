"""Write the paper's tables as LaTeX (booktabs) straight from pipeline results.

The paper ``\\input``s these files, so no number in the text tables is ever
copied by hand. Each function writes ``output/tables/<name>.tex``.
"""

import pandas as pd

from aiprod.config import TABLES

DID_ROWS = [
    ("C1", "growth", "Productivity growth (pp/yr)"),
    ("C2", "emp_gap", "Junior employment gap"),
    ("C2", "emp_gap|log_emp_total", "\\quad given sector size"),
    ("C2", "hire_gap", "Junior hiring gap"),
    ("C2", "hire_gap|log_emp_total", "\\quad given sector size"),
    ("C3", "labor_share_change", "Labor share change (pp/yr)"),
    ("C3", "price_growth", "Value-added price growth (pp/yr)"),
]
KINK_TERMS = {
    "exposure:trend": "Pre-existing trend (per year)",
    "exposure:post_covid": "Post-COVID level shift",
    "exposure:ramp_chatgpt": "Change in slope after ChatGPT (per year)",
}


SHORT_NAMES = {
    "Long substituting / short complementing": "Substitution signal",
    "Same, Bayesian-sized": "Bayesian-sized",
    "Benchmark: long high AI exposure": "AI-exposure benchmark",
}


def num(value: float, spec: str) -> str:
    """Format a number with a typeset minus sign."""
    return f"{value:{spec}}".replace("-", "$-$")


def p_text(p: float) -> str:
    """Format a p-value for a table cell."""
    return "$<$0.01" if p < 0.01 else f"{p:.2f}"


def write(name: str, lines: list[str]) -> None:
    """Write table lines to ``output/tables/<name>.tex``."""
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def did_table(estimates: pd.DataFrame) -> None:
    """Main results: each claim's coefficient in the IT boom and the AI era."""
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{IT boom} & \\multicolumn{2}{c}{AI era} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "Claim & Outcome & $\\hat\\beta$ (s.e.) & $p_{\\text{boot}}$ & $\\hat\\beta$ (s.e.) & $p_{\\text{boot}}$ \\\\",
        "\\midrule",
    ]
    for claim, outcome, label in DID_ROWS:
        cells = []
        for era in ("IT", "AI"):
            row = estimates[(estimates["outcome"] == outcome) & (estimates["era"] == era)].iloc[0]
            cells += [f"{num(row['beta'], '.3f')} ({row['se']:.3f})", p_text(row["p_boot"])]
        lines.append(f"{claim} & {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_did", lines)


def kink_table(kinks: pd.DataFrame) -> None:
    """ChatGPT kink test for the QWI junior gap and for detailed-industry employment."""
    panels = ["QWI sectors", "CES detailed industries"]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Junior gap (19 sectors)} & \\multicolumn{2}{c}{Employment (91 industries)} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Term $\\times$ exposure & $\\hat\\beta$ & $p_{\\text{boot}}$ & $\\hat\\beta$ & $p_{\\text{boot}}$ \\\\",
        "\\midrule",
    ]
    for term, label in KINK_TERMS.items():
        cells = []
        for panel in panels:
            row = kinks[(kinks["panel"] == panel) & (kinks["term"] == term)].iloc[0]
            cells += [num(row["beta"], ".4f"), p_text(row["p_boot"])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_kink", lines)


def strategy_table(results: pd.DataFrame) -> None:
    """C4 backtest: performance and factor-adjusted alpha by strategy and period."""
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Strategy & Period & Mean & Vol. & Sharpe & Max DD & $\\alpha$ ($t$) & $\\beta_{2y}$ ($t$) \\\\",
        "\\midrule",
    ]
    previous = None
    for _, row in results.iterrows():
        name = SHORT_NAMES[row["strategy"]] if row["strategy"] != previous else ""
        if previous is not None and name:
            lines.append("\\addlinespace")
        previous = row["strategy"]
        lines.append(
            f"{name} & {row['period']} & {num(100 * row['mean'], '.1f')}\\% & {100 * row['vol']:.1f}\\% & "
            f"{num(row['sharpe'], '.2f')} & {num(100 * row['max_drawdown'], '.0f')}\\% & "
            f"{num(100 * row['alpha'], '.1f')}\\% ({num(row['alpha_t'], '.1f')}) & "
            f"{num(row['beta_2y'], '.3f')} ({num(row['beta_2y_t'], '.1f')}) \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_strategy", lines)
