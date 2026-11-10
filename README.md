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
src/aiprod/   config.py   sample windows and industry crosswalk
              sources.py  download-and-cache readers for every data source
              panels.py   industry panels (productivity, exposure, junior gaps)
              claims.py   estimators for each claim
              run.py      pipeline entry point
notebooks/    interactive exploration
output/       tables and figures
```
