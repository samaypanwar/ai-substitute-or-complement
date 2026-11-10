# Who gets paid for AI? Substitution, complementarity, and the macro data

Final project for 46-975 *Macroeconomics for MSCF* (CMU Tepper, Fall 2026).

**Question.** AI is a shock to the production function. Where it *substitutes* for labor, the productivity gain should accrue to capital (margins and cash flows); where it *complements* labor, to wages. We measure which is happening, industry by industry, using scheduled macro releases (BEA GDP by industry, the BLS employment report, JOLTS, and Census QWI), and ask whether equity prices reflect it.

We test everything twice: once on the 1995–2004 IT boom (as a historical prior) and once on the post-ChatGPT era.

## Claims
| | Claim | Data |
|---|---|---|
| C1 | Output per hour accelerated more where exposure is higher | BEA real value added by industry, CES hours |
| C2 | Substitution shows up as missing junior hires | Census QWI by age × industry; CES; JOLTS |
| C3 | Substitution raises margins; complementarity raises wages | BEA value-added components by industry |
| C4 | The market does not separate substituting from complementing industries | Ken French industry returns, FF factors, 2y Treasury |

The full design, including how AI differs from IT and how each difference should show up in the data, is in [`design.md`](design.md).

## Reproduce
```bash
poetry install
poetry run python -m aiprod.run
```
All data are free and downloaded on first run into `data/` (FRED, BEA, Census LEHD, Ken French library, and the Felten–Raj–Seamans AI exposure index). Tables go to `output/tables/`, figures to `output/figures/`.

## Layout
```
src/aiprod/   config.py     sample windows and industry crosswalk
              sources.py    download-and-cache readers for every data source
              panels.py     sector panels: productivity, labor share, exposure, junior gaps
              nowcast.py    detailed CES industries and the monthly nowcast of the QWI junior gap
              claims.py     difference-in-differences and event studies for C1-C3, both eras
              inference.py  wild cluster bootstrap, ChatGPT kink test, Bayesian pooling
              strategy.py   C4 backtest: rank weights, Bayesian sizing, factor regressions
              figures.py    publication figure style (400-dpi PNG + vector PDF)
              plots.py      one function per paper figure
              run.py        pipeline entry point
notebooks/    01_explore.ipynb: interactive companion (kernel "macro-46-975")
output/       tables/ and figures/
```

## Current results
See [`design.md` §6](design.md). In short: AI-exposed sectors show fewer junior workers (with a post-ChatGPT kink) and a falling labor share, but no productivity acceleration and no change in total employment. A long-short strategy on the junior-substitution signal does not earn excess returns.
