# OFI Research Report




## Evidence status — three separate claims

| Claim | Status |
|---|---|
| **Implementation validity** (the code computes what it says) | Tests pass on this run; the one-day raw-to-feature audit (`ofi_research pilot`) must also have been reviewed by hand before these numbers mean anything. |
| **Predictive evidence** (OFI forecasts future mid moves OOS) | Measured on 8 out-of-sample fold(s) across 8 distinct market day(s) — see the day-blocked interval in `13_overlap_aware_uncertainty.csv`, not the point estimate. |
| **Net trading evidence** (that edge survives execution) | Measured on 1 trade(s) over 1 day(s), after full spread crossing, fees, slippage and total latency — see `10_gross_vs_net_performance.csv` and `11_/12_` sensitivity tables. |

Predictive evidence does not imply net trading evidence, and neither follows from implementation validity.

- Label: `INTC_20d_baseline`  |  Primary horizon: `ms1000`  |  Walk-forward: `rolling` (10/2/1 train/val/test days)


## Headline decision

**Verdict:** OFI is predictive and net-positive out of sample, but the decision gate is NOT fully met (failing: not_one_day). Treat as promising-but-unproven; gather more test days / regimes before adding the wave/PDE layer.


**Proceed to impulse-response / Green's-function stage?** NO


### Decision-gate checklist

| Criterion | Status | Basis |
|---|---|---|
| stable_coef_sign | ✅ PASS | beta_1 positive in 100.0% of 8 folds |
| nonzero_oos_corr | ✅ PASS | mean OOS corr 0.02214 in 100.0% of folds |
| sensible_deciles | ✅ PASS | decile Spearman 0.556 |
| net_positive | ✅ PASS | 1 trades executed |
| not_one_day | ❌ FAIL | largest day = 1.000 of total |P&L| over 1 day(s) |
| survives_stress | ✅ PASS | latency/cost sweeps |

## Execution accounting — what the policy actually did

| Quantity | Value |
|---|---|
| Out-of-sample folds | 8 |
| Distinct OOS market dates | 8 |
| OOS dates with eligible predictions | 8 |
| OOS dates with executed trades | 1 |
| Rows with a usable prediction | 857967 |
| … clearing the cost gate |pred| > round-trip cost | 72 |
| … clearing the z gate | 1 |
| Signals (both gates) | 1 |
| Dropped: no executable quote after latency | 0 |
| Dropped: cannot close inside segment | 0 |
| Dropped: insufficient displayed depth | 0 |
| Suppressed by overlap guard | 6 |
| **Executed trades** | **1** |
| Long / short | 1 / 0 |
| Fraction of time holding a position | 0.0000 |
| Turnover (notional) | 99.05 |
| Gross per trade (ticks) | 6.0000 |
| Cost per trade (ticks) | 2.0000 |
| Net per trade (price units) | 0.04000000 |
| Net per trade (ticks) | 4.0000 |
| Net per trade (bps) | 4.1125 |


## Effect size in economic units

A 1-SD increase in `OFI1_ev50` predicts a mid move of:

| Unit | Value |
|---|---|
| price units | 0.00014516 |
| **ticks** | **0.01452** |
| basis points | 0.01492 |
| 95% CI (ticks, across folds) | [0.00902, 0.02001] |
| mean quoted spread (ticks) | 2.530 |
| **predicted move / spread** | **0.00574** |

Read this as: the effect is real and consistently signed, but a 1-SD OFI move buys about 0.0145 of a tick against a 2.53-tick spread. Roughly 174 SD of simultaneous OFI would be needed to cover one round trip. Decile monotonicity of 0.556 is moderate, not strong.


## Phase 0 — front-of-queue passive markout (UPPER BOUND)

Every number here assumes our order is at the FRONT of the queue and fills on every trade at our price. Real queue position is strictly worse, so these are ceilings. There is no queue model, no fill probability and no inventory limit: this is a screen to decide whether market-by-order data is worth buying, not a backtest.

**Maker rebate is 0** — set `costs.maker_rebate_per_unit` from the venue schedule; at a $0.01 tick a typical add rebate is worth 0.2-0.3 ticks per share, which is large relative to the edges below.

### Unconditional (no OFI conditioning)
| Horizon | Markout (ticks) | day-clustered SE | t | fills | days |
|---|---|---|---|---|---|
| ms100 | -0.0900 | 0.0262 | -3.44 | 42301 | 20 |
| ms250 | -0.0834 | 0.0257 | -3.24 | 50937 | 20 |
| ms500 | -0.0670 | 0.0251 | -2.67 | 56782 | 20 |
| ms1000 | -0.0537 | 0.0269 | -1.99 | 59228 | 20 |
| ms2000 | -0.0739 | 0.0283 | -2.61 | 59602 | 20 |
| ms5000 | -0.0386 | 0.0232 | -1.66 | 59593 | 20 |
| ms10000 | 0.0084 | 0.0361 | 0.23 | 59554 | 20 |
| ms30000 | 0.0621 | 0.0596 | 1.04 | 59411 | 20 |

At the primary horizon `ms1000` the unconditional front-of-queue maker earns -0.0537 ticks per fill — NEGATIVE. Adverse selection already exceeds the spread captured at the best possible queue position, so OFI would have to rescue a business that loses money by default, not merely improve a profitable one.

### Conditioned on fill-aligned OFI, known BEFORE the fill

Buckets are cut on OFI signed by the position the fill leaves us in, because adverse selection is symmetric: raw OFI produces a U (both tails toxic) and its monotonicity would read ~0 even if OFI predicted toxicity perfectly. See `21b_` for that raw view and `21c_` for |OFI|.

| Decile (worst->best flow) | markout (ticks) | SE | fills |
|---|---|---|---|
| 0 | -0.0943 | 0.0475 | 5939 |
| 1 | -0.1222 | 0.0450 | 5925 |
| 2 | -0.1169 | 0.0421 | 5939 |
| 3 | -0.1496 | 0.0459 | 5936 |
| 4 | -0.0766 | 0.0532 | 5937 |
| 5 | -0.0496 | 0.0553 | 5919 |
| 6 | -0.0653 | 0.0409 | 5892 |
| 7 | -0.0035 | 0.0332 | 5926 |
| 8 | 0.0529 | 0.0511 | 5901 |
| 9 | 0.0887 | 0.0620 | 5914 |

Decile monotonicity (Spearman) = 0.879; bottom-to-top decile spread = 0.1830 ticks. OFI separates toxic from benign fills, which is the result Phase 1 would be built on.

Decile edges are cut pooled over the whole sample, so they peek at the full period. A positive reading is a reason to run this walk-forward, not a result on its own.


## Answers to the section-29 questions


**1. Does OFI predict future midprice movement OOS?**

Mean OOS pred-corr (M1, ms1000) = 0.02214; positive in 100.0% of folds. Day-blocked bootstrap 95% interval [0.01476, 0.02845] over 8 day(s) — the interval, not the point estimate, is the result.


**2. Strongest horizons?**

See `03_ofi_correlation_by_horizon.csv` (HAC/Newey-West, FDR-adjusted p-values).


**3. Is the OFI-decile relationship monotonic?**

Mean decile monotonicity (Spearman) = 0.556 (1.0 = perfectly monotone). See `04_ofi_decile_response_*`.


**4. Is the OFI coefficient stable across folds?**

beta_1 positive in 100.0% of folds (see `09_coefficient_stability_*`).


**5. Does L2 improve on L1?**

`M2_ofi_spread_depth` vs `M1_ofi`: OOS correlation 0.00728 -> 0.01894 (+0.01165), OOS R^2 -0.000032 -> 0.000296 — adding L2 spread/depth improved the baseline.


**6. Does normalization help?**

`M1N_ofi_normalized` vs `M1_ofi`: OOS correlation 0.00728 -> 0.00468 (-0.00261), OOS R^2 -0.000032 -> -0.000039 — normalizing OFI did not improve the baseline.


**7. Do signed volume / intensities add information?**

`M3_ofi_signedvol` vs `M1_ofi`: OOS correlation 0.00728 -> 0.01114 (+0.00386), OOS R^2 -0.000032 -> 0.000060 — adding signed volume improved the baseline. `M4_ofi_intensity` vs `M1_ofi`: OOS correlation 0.00728 -> 0.00956 (+0.00228), OOS R^2 -0.000032 -> 0.000015 — adding trade intensity improved the baseline. `M5_full` vs `M1_ofi`: OOS correlation 0.00728 -> 0.02214 (+0.01485), OOS R^2 -0.000032 -> 0.000425 — the full feature set improved the baseline.


**8. Profitable after bid/ask execution, fees, slippage?**

Net P&L (M1, pooled test) = 0.04000 over 1 trades (4.0000 ticks/trade, 4.1125 bps/trade) vs gross mid-to-mid 0.06000 (6.0000 ticks/trade); cost 2.0000 ticks/trade.  Day-blocked interval not estimable (fewer than two out-of-sample days WITH TRADES).


**9. Which regimes work/fail?**

regime_spread: only one populated bucket (`narrow`, 0.04000) — no contrast; regime_tod: only one populated bucket (`open`, 0.04000) — no contrast; regime_vol: only one populated bucket (`low_vol`, 0.04000) — no contrast. See `08_regime_results.csv`.


**10. Survives latency & cost stress?**

Net P&L stays positive to 100 ms; all cost scenarios remain non-negative. See `11_`/`12_`.


**11. Concentrated in a few days?**

The single largest day accounts for 1.000 of total absolute P&L across 1 trading day(s); 1.000 of those days positive. (Share is |largest day| / sum of |daily P&L|, so it stays in [0,1] even when the total is negative.)


**12. Proceed to response-kernel stage?**

NO — per the decision gate above.


## Statistical caveats (section 25)

- Targets overlap across rows -> all significance uses Newey-West (HAC) SEs; iid SEs are **not** trusted.
- Many horizon/window combinations are tested -> p-values are FDR-adjusted (Benjamini-Hochberg) and treated as exploratory.
- The conclusion is driven by out-of-sample walk-forward performance, not in-sample p-values.
- Queue position cannot be modeled from top-of-book data; aggressive taker execution is assumed.
- Overlapping targets also invalidate per-row iid intervals, so the headline numbers carry a whole-day block bootstrap and day-clustered SEs (`13_overlap_aware_uncertainty.csv`).
- The contemporaneous CKS regression (`15_contemporaneous_cks_replication.csv`) is a data/formula sanity check; its R^2 is NOT expected out-of-sample predictive performance.
- XNAS.ITCH is a venue-LOCAL book: its mid/spread are not the national NBBO, and cross-venue routing would need consolidated data.
- Statistical significance and economic significance are reported separately: a correlation whose CI excludes zero can still imply a predicted move far below one tick, and only the tick-denominated effect size decides tradability.
- A no-trade run yields NOT_EVALUABLE P&L criteria, never FAIL. Absence of trading evidence is not evidence of unprofitability.
- 8 out-of-sample market days is a small number of independent days; treat all fold-fraction statistics (e.g. '100.0% of folds') as counts out of 8, not as stable probabilities.


## Feature availability notes

- Signed trade volume is the vendor's `signed_trade_volume`, a NET per-event quantity (one canonical event may contain both buy and sell executions, so size x direction would not reproduce it). It is therefore not derived from the direction percentages that follow, which describe the SEPARATE per-trade `trade_direction` used by the intensity features: Aggressor direction came from the vendor side field for 71.34% of trades, from a causal 'lee_ready' classifier for 22.77%, and was UNRESOLVED for 5.88% (direction 0, contributing zero signed volume — unresolved trades are neutral, never dropped and never guessed).
