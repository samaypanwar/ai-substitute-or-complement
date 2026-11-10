# Project design: AI as substitute or complement, and who gets paid

*Living design doc (Samay + Claude). It becomes the Nov 29 one-pager and the skeleton of the paper. Last updated 2026-11-10.*

## 1. Core idea
AI is a shock to the production function. Where it **substitutes** for labor (routine and junior tasks), the productivity gain goes to capital: margins rise, so cash flows $D$ in $P_t=E_t[M_{t,t+j}D_{t+j}]$ rise. Where it **complements** labor (non-routine and senior tasks), the gain goes to wages. The market prices "AI exposure" as one factor. We split it using labor-market flows.

## 2. Claims

| | Claim | Main test | Killed if |
|---|---|---|---|
| C1 | Output per hour accelerated after 2022Q4, more so where AI exposure is high | Panel: productivity growth on Post × Exposure, COVID dummy | No exposure gradient, or Information-sector deflator only |
| C2 | Substitution shows up as missing **junior hires** (flows, not stocks) | Diff-in-diff: young (22–24) vs prime-age (35–54) employment and hires × Exposure | Young decline equal in low-exposure industries, meaning the Fed cycle, not AI |
| C3 | Substitution raises margins; complementarity raises wages | Labor share, unit labor cost, deflator vs productivity | Prices fall one-for-one (gain to consumers) |
| C4 | The market doesn't separate S from C, so a long-short on the S/C score earns alpha | Rank-weighted backtest; alpha vs FF5 + UMD + Δ2y yield | Alpha vanishes under rate beta (discount-rate story) |

COVID (2020Q1–2021Q4) is excluded or dummied throughout.

## 3. The data problem: about 3–4 years of AI
Statistical power comes from three places. We use all three.

**(a) Cross-section, not time.**
- Go below supersectors to detailed knowledge-work industries. Monthly CES employment from 1990 is on FRED for: computer systems design, legal, accounting, consulting, credit intermediation, data processing/hosting, and more.
- About 30–50 industries × about 46 months post-ChatGPT is a real panel.
- Low-exposure industries act as the control group.

**(b) Fast proxies for the slow truth.**
- QWI (age × industry) lags about 4 quarters. It is the ground truth for C2 but too slow to trade.
- The trading signal uses monthly or faster proxies:
  - CES employment and supervisory share;
  - JOLTS hires by industry;
  - possibly Indeed postings (daily).
- We estimate the proxy → QWI mapping on 1993–2019, where the history is long. Then the proxies nowcast QWI in the AI era.

**(c) A prior from history, updated by AI-era data.**
- The *mechanism* is not new. Routine-biased technical change (computers, 1980s–2000s) and the IT-using-industry productivity boom (1995–2004) are earlier substitution and complementarity episodes.
- Estimate the mechanism coefficients on that history. This is the prior.
- Then:
$$\beta_{AI} \sim N(\beta_{hist},\ \tau^2), \qquad \hat\beta_{AI\text{-}era}\mid\beta_{AI} \sim N(\beta_{AI},\ se^2)$$
$$\text{posterior mean} = w\,\hat\beta_{AI\text{-}era} + (1-w)\,\beta_{hist}, \qquad w=\frac{\tau^2}{\tau^2+se^2}$$
- $\tau$ = **"how different is AI from past technology?"**
  - $\tau\to 0$: AI is just more IT; the history decides.
  - $\tau\to\infty$: flat prior; only AI-era data count (plain OLS).
- We report results across a grid of $\tau$. That grid *is* the forward-looking regime discussion the brief asks for.

**(d) The IT-boom prior, concretely (decided 2026-11-10).**
- **Exposure, then:** each industry's IT-capital intensity in 1995, from BEA fixed assets by industry (IT equipment + software share; free, annual). This is the analog of today's AI-exposure index, measured *before* the boom.
- **Estimate on 1995–2004** the same C1–C3 regressions, with IT intensity in place of AI exposure:
  - productivity acceleration (C1);
  - junior vs prime-age employment and hires from QWI (C2, covered from 1994);
  - labor share and margins (C3).
- These coefficients become $\beta_{hist}$ for the AI era.
- **The prior applies only to the real side (C1–C3).** 1995–2004 *returns* are the dot-com bubble and crash. A return-based prior from that window would mostly measure the bubble, so C4's prior stays centered at 0 (§4).
- The bubble becomes a forward-looking point instead: the IT boom delivered real productivity *and* a crash in the producers' stocks. That is the precedent for "AI is real, and the AI trade can still be wrong."

**(e) How AI differs from IT, and where each difference shows up.**
The prior is only as good as the analogy. Each difference below predicts *which* coefficient should move and *in which direction*. Write the AI-era coefficient as $\beta_{AI}=\beta_{IT}+\Delta$; each row is a prediction about $\Delta$. A row we can't sign gets a wide $\tau$.

| # | Difference (IT 1995–2004 vs gen-AI 2022–) | Shows up as | Coefficient | Expected $\Delta$ | How we check |
|---|---|---|---|---|---|
| D1 | **Which tasks.** IT automated routine *middle-skill* tasks (clerks, production) and complemented educated workers. Gen-AI targets non-routine *cognitive* tasks (writing, code, analysis). | Knowledge work flips from complement (IT) to partial substitute (AI). | C2: young × exposure | IT ≈ 0 or positive; AI negative. **Prior likely wrong-signed: wide $\tau$** | Estimate C2 in both eras; the sign flip is itself a finding |
| D2 | **Who is junior.** IT rewarded computer-literate new graduates. AI substitutes for codified entry-level tasks; seniors' tacit knowledge complements it. | Junior/senior gap opens in AI era only. | C2 | negative | QWI 22–24 vs 35–54 by era |
| D3 | **Capex at the user.** IT required heavy user-side capital plus organizational change (productivity "J-curve", long lags). Gen-AI is rented per token; capex sits with producers (hyperscalers). | Shorter adoption→productivity lag; less capital deepening at users, so gains appear as a TFP residual. | C1 lag structure; L3 growth accounting | faster response | Distributed-lag C1 in both eras; compare lag peaks |
| D4 | **Measurement.** Purchased AI services are *intermediate inputs* at the user (Assignment 1 Q1: the LLM license). User value added rises only if output rises by more. IT hardware was capitalized and hedonically deflated. | User-industry real VA may *understate* AI gains; IT-producer VA overstated IT gains. | C1 level | AI understated | Check intermediate-input share by industry (BEA GDP-by-industry gross output vs VA) |
| D5 | **Producer market power.** IT hardware was competitive, with falling prices passing gains to users. AI is concentrated (GPUs, a few model labs) and can extract rents. | Users keep a smaller share of the gain; producer margins absorb it. | C3 incidence | smaller user margin gain | User industries' intermediate costs vs margins |
| D6 | **Policy regime.** 1990s: low inflation; the Fed let productivity growth run (Greenspan). 2022–26: post-inflation hiking, re-hike Sept 2026. | Hiring slowdown confounded with tight policy; discount-rate channel stronger. | C2 confound; C4 rate beta | — | Low-exposure control group; ΔDGS2 control |
| D7 | **Speed of diffusion.** Gen-AI reached mass use in months; IT took a decade. | Effects show up within 3–4 years, if real. | C1, C2 timing | earlier | Event time around Nov 2022 |

**Use in estimation.** Separate $\tau$ per claim: wide for C2 (D1–D2 predict a sign flip), moderate for C1 (D3–D4 shift timing and level), tighter for C3 (incidence economics are similar, but D5 matters). The gap between $\beta_{IT}$ and the AI-era estimate is evidence on "is AI different?", which is the brief's regime question.

## 4. The sign of C4: weak prior, learned in real time
- The prior on the long-S / short-C payoff is centered at 0 and wide. We're agnostic about the sign.
- **No look-ahead.** At each rebalance date $t$:
  - Update the posterior using only data up to $t$.
  - Position = posterior mean scaled by its precision.
  - So early positions are small, and they grow as the data speak.
- The learning rule *is* the strategy. If the sign is wrong, it flips by itself, and the backtest shows that honestly.
- EDA is done on pre-2022 data only, or is explicitly labelled descriptive. We never pick the sign on the full sample and then backtest on the same sample.

## 5. Open items
1. Exposure index (AIOE vs Eloundou et al.) and the occupation→industry mapping (OEWS).
2. Hours and supervisory share for *detailed* CES industries. FRED lacks some, so try BLS directly.
3. Map detailed industries to returns (Ken French 49 industries? ETFs?). This decides the tradable N.
4. ~~Historical prior episode.~~ **Decided 2026-11-10: the 1995–2004 IT boom.** See §3(d).
5. Indeed postings: coverage by occupation, history length.

## 6. Status: first results (pipeline v0, 2026-11-10)
Code: `src/aiprod/`; run `poetry run python -m aiprod.run`.

| | IT boom (exposure = 1995 IT capital share) | AI era (exposure = LM AI exposure) |
|---|---|---|
| | IT boom | AI era |
|---|---|---|
| C1: extra productivity growth per 1 SD exposure (official all-employee hours from 2006) | +0.81 pp/yr (se 0.56) | −0.16 pp/yr (se 0.88) |
| C2: junior-vs-prime employment gap per 1 SD | +0.032 (se 0.016) | −0.049 (se 0.030) |
| C2, controlling for sector employment (juniors *given* sector size) | +0.028 (se 0.017) | −0.071 (se 0.024) |

- **C1: no AI-era productivity gradient.** The earlier "white-collar acceleration" table used nonsupervisory-only hours and no time effects; it does not survive.
- **C2: AI-era juniors lose ground in exposed sectors, and more so given sector size.** The IT era leans the other way. This is a sign flip that is **not yet separable** from the dot-com hiring boom (IT) or from the 2020–22 shocks (AI): the AI-era decline starts in 2020–21, before ChatGPT, with no visible acceleration after Nov 2022.
- 1995 IT share and AI exposure are correlated (0.63 across sectors), so "exposed" means roughly the same white-collar sectors in both eras.
- With 13–19 industries, a wild cluster bootstrap is needed before quoting significance.
- **Not built yet:** C3 (labor share, unit labor cost), C4 (returns, rate beta, backtest), detailed-industry panel, nowcast proxies, Bayesian combination.

## 7. Dates
- Group form: Nov 13.
- One-pager: Nov 29.
- Final: Dec 11.
