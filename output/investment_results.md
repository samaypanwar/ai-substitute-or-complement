# C1–C3 with AI as investment, gross output, and employment-weighted exposure

Run `python -m aiprod.run_investment`. Tables: `output/tables/investment_*.csv`.
Coefficients are per 1 SD of exposure; p-values from the wild cluster bootstrap (as in v1).
The C4 pricing test is not rerun.

## What changed

1. **AI spending proxy.** From BEA's summary use tables (1997–2023), each sector's purchases of
   software publishing (511), data processing and hosting (514) and computer systems design (5415),
   as a share of gross output. 2024–25 are extrapolated by growing 2023 purchases with the suppliers'
   nominal gross output.
   - AI era: only the excess over the sector's linear 2010–19 trend counts (main), or over its flat
     2019 share (bound).
   - IT era: the excess over the 1997 share in 1998–2004 (the tables start in 1997).
   - Upper bound: *all* such purchases capitalized, in every year.
2. **Adjusted value added** = VA + excess spending; real terms use a supplier-weighted gross-output
   price index for those three industries.
3. **Capital stock** by perpetual inventory, K_t = (1−δ)K_{t−1} + X_t, depreciation δ·K_t,
   δ ∈ {33%, 50%, 100%}. At δ = 100% the net measures reduce to the current treatment, which the
   results reproduce exactly (a built-in check).
4. **Net labor share** = compensation ÷ (VA − official BEA depreciation − depreciation of the new capital).
   Official depreciation: BEA detailed fixed assets (private nonresidential, all assets).
5. **Option B:** real gross output per hour (BEA TGO103 ÷ CES hours) and the intermediate-input
   share of gross output (TII105 ÷ TGO105).
6. **Exposure:** LM AIIE averaged within sectors using 2019 County Business Patterns employment
   (4-digit NAICS) as weights, in place of the unweighted mean. The IT score is unchanged.

## How big is the AI spending?

Small, and mostly *on trend*. IT-services purchases are 0.5–5% of gross output. Measured against
each sector's 2010–19 trend, 2023–25 purchases are **at or below trend in 11 of 13 sectors**: the
cloud/software spending boom of the 2010s was already steep. Against the flat 2019 share, the
excess is at most ~1.4% of value added (information, 2025), and below 0.6% in most sectors.
The IT-era excess is similarly small (≤0.34% of VA).

## Results (employment-weighted exposure; v1 unweighted in brackets)

| Measure | IT boom | AI era |
|---|---|---|
| **C1** VA per hour, AI as input (current) | +0.49 (p=0.26) | −0.05 (p=0.98) [−0.26] |
| C1 VA per hour, AI as investment (trend excess) | +0.49 (p=0.27) | −0.02 (p=0.99) |
| C1 VA per hour, AI as investment (over 2019 share) | +0.49 | −0.05 (p=0.98) |
| C1 VA per hour, all IT services capitalized | +0.41 (p=0.39) | −0.05 (p=0.98) |
| **C1** gross output per hour | +0.10 (p=0.78) | +0.19 (p=0.69) [+0.35] |
| C1 intermediate-input share change (pp/yr) | −0.12 (p=0.60) | −0.15 (p=0.62) |
| **C2** junior gap, given sector size | +0.028 (p=0.35) | **−0.064 (p=0.03)** [−0.071, p=0.01] |
| C2 junior gap, no size control | +0.032 (p=0.27) | −0.048 (p=0.15) |
| C2 junior hires gap, given size | +0.016 (p=0.46) | −0.043 (p=0.15) |
| **C3** gross labor share change, current (pp/yr) | +0.01 (p=0.99) | −0.77 (p<0.01) [−0.81] |
| C3 gross labor share change, AI as investment | +0.01 | −0.78 (p<0.01) |
| C3 net labor share change, current | +0.14 (p=0.84) | −1.01 (p<0.01) |
| C3 net labor share change, AI inv., δ=33% / 50% / 100% | +0.14 | −1.02 / −1.01 / −1.01 (all p<0.01) |
| C3 net labor share change, all IT services inv., δ=50% | +0.20 | −1.01 (p<0.01) |
| C3 gross labor share **level** (pp) | +1.73 (p=0.29) | −0.68 (p=0.71) |
| C3 net labor share **level**, AI inv., δ=50% | +3.04 (p=0.08) | +1.84 (p=0.69) |

**Bayesian pooling** (IT prior → AI posterior), P(β<0) at τ = 0 / mid-grid / 3×SE:
C1 (AI as investment) 0.18 / 0.41 / 0.49; C1 (gross output/hr) 0.32 / 0.34 / 0.34;
C2 0.55 / 0.93 / 0.99; C3 (net, δ=50%) 1.00 at every τ.

## Reading

- **Treating AI as investment changes nothing material.** The spending above trend is too small to
  move value added, productivity or labor shares; the choice of δ is irrelevant at this scale. Even
  capitalizing *every* IT-services purchase in both eras leaves C1 and C3 essentially unchanged. So
  the C1 null is not an artifact of expensing AI.
- **Gross output per hour** is positive in the AI era (+0.19) but far from significant, and the
  input share did not rise. There is no sign that exposed sectors swapped workers for purchased
  services at a scale visible in the national accounts.
- **C2 survives** the exposure change: −0.064 (p=0.03), a little smaller than v1.
- **C3:** the net labor share *change* is more negative than the gross one (−1.01 vs −0.77), but the
  *levels* are insignificant under every treatment. As in v1, the change result is mostly the
  2021–22 spike unwinding, so C3 remains inconclusive.

## Caveats

- The use tables end in 2023; 2024–25 purchases are extrapolated (`it_extrapolated` flag in the panel).
- Commodity 511 includes non-software publishing; 514 includes internet publishing. The proxy
  over-counts AI-unrelated spending, which makes the "too small to matter" conclusion conservative.
- Most AI spending may be invisible here: in-house AI development is already capitalized as
  own-account software, and organizational investment (training, workflow redesign) is paid as wages.
- Real adjusted VA adds a deflated flow to chained dollars, which is an approximation.
