# OFI Research Report




## Evidence status — three separate claims

| Claim | Status |
|---|---|
| **Implementation validity** (the code computes what it says) | Tests pass on this run; the one-day raw-to-feature audit (`ofi_research pilot`) must also have been reviewed by hand before these numbers mean anything. |
| **Predictive evidence** (OFI forecasts future mid moves OOS) | Measured on 8 out-of-sample fold(s) across 8 distinct market day(s) — see the day-blocked interval in `13_overlap_aware_uncertainty.csv`, not the point estimate. |
| **Net trading evidence** (that edge survives execution) | **NOT EVALUATED** — the policy executed ZERO trades on 8 day(s) of eligible predictions, so no execution assumption was ever exercised. This is an absence of evidence, NOT evidence of unprofitability. |

Predictive evidence does not imply net trading evidence, and neither follows from implementation validity.

- Label: `INTC_20d_passive_walkforward`  |  Primary horizon: `ms1000`  |  Walk-forward: `rolling` (10/2/1 train/val/test days)


## Headline decision

**Verdict:** MAKER PATH SURVIVES ITS FIRST HONEST TEST. Every exercised passive criterion passes, including at the back of the queue. This is a walk-forward result on held-out days with latency charged on both the quote and the cancel — but it is still one symbol over a small number of days, and the queue position remains an assumption no MBP-10 data can pin down.


**Proceed to impulse-response / Green's-function stage?** YES


### Decision-gate checklist

| Criterion | Status | Basis |
|---|---|---|
| stable_coef_sign | ✅ PASS | beta_1 positive in 100.0% of 8 folds |
| nonzero_oos_corr | ✅ PASS | mean OOS corr 0.02214 in 100.0% of folds |
| sensible_deciles | ✅ PASS | decile Spearman 0.556 |
| net_positive | ⬜ NOT_EVALUABLE | taker path OFF by design — settled negative on effect size, not re-derived each run (evaluation.run_taker_backtest) |
| not_one_day | ⬜ NOT_EVALUABLE | taker path OFF by design — settled negative on effect size, not re-derived each run (evaluation.run_taker_backtest) |
| survives_stress | ⬜ NOT_EVALUABLE | taker path OFF by design — settled negative on effect size, not re-derived each run (evaluation.run_taker_backtest) |
| passive_filter_beats_unconditional | ✅ PASS | best gated 0.2716 ticks vs unconditional 0.2551 at the same queue position sweep, zero rebate |
| passive_positive_at_zero_rebate | ✅ PASS | best gated cell 0.2716 ticks/fill before any rebate |
| passive_significant_day_clustered | ✅ PASS | t = 7.81 over 8 days, day-clustered |
| passive_survives_back_of_queue | ✅ PASS | behind all displayed depth: 0.2502 ticks/fill on 1111 fills |
| passive_rebate_not_load_bearing | ✅ PASS | break-even needs -0.272 ticks of rebate (-0.00272 price units per share); a typical US add tier is 0.20-0.30 ticks |

> 3 criteria were NOT EVALUABLE on this run: `net_positive`, `not_one_day`, `survives_stress`. They are not failures — they were never exercised. `proceed` requires PASS on all six, so it remains NO.


## Execution accounting — what the policy actually did

| Quantity | Value |
|---|---|
| Out-of-sample folds | 8 |
| Distinct OOS market dates | 8 |
| OOS dates with eligible predictions | 8 |
| OOS dates with executed trades | 0 |
| Rows with a usable prediction | n/a |
| … clearing the cost gate |pred| > round-trip cost | n/a |
| … clearing the z gate | n/a |
| Signals (both gates) | n/a |
| Dropped: no executable quote after latency | n/a |
| Dropped: cannot close inside segment | n/a |
| Dropped: insufficient displayed depth | n/a |
| Suppressed by overlap guard | n/a |
| **Executed trades** | **0** |
| Long / short | n/a / n/a |
| Fraction of time holding a position | n/a |
| Turnover (notional) | n/a |

> **NO TRADES WERE EXECUTED.** Tradability was NOT evaluated; it was not tested. The mean |prediction| is n/a price units (n/a ticks) against a mean round-trip cost threshold of n/a (n/a ticks) — a ratio of n/a. The largest single prediction in the whole out-of-sample period was n/a ticks. The forecast is smaller than the spread it must cross by orders of magnitude, so no threshold setting makes this policy trade. Read every P&L, cost and latency figure below as UNDEFINED, not as zero.


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


## Passive policy — walk-forward, queued, latency-charged

The gate threshold is fitted on TRAIN days and applied to days the model has not seen. A resting bid is quoted when the model predicts UP and a resting ask when it predicts DOWN, so the side is chosen before the fill rather than read off it — which is the one thing the Phase-0 decile table above cannot claim.

Quotes reach the book 1.30 ms after the decision and join the queue as it stands then; cancels take 1.30 ms to bite and every trade inside that window still fills us. Fills are simulated per event, not on the decision clock.

### How far back in the queue does the edge survive?

Zero rebate, best gate per queue position.

| queue ahead | best gate | markout (ticks) | t | fills/day | days |
|---|---|---|---|---|---|
| 0.00 × depth | q=0.30 | 0.2716 | 7.81 | 9374 | 8 |
| 0.25 × depth | q=0.90 | 0.1447 | 1.26 | 790 | 8 |
| 0.50 × depth | q=0.90 | 0.1767 | 1.01 | 497 | 8 |
| 1.00 × depth | q=0.90 | 0.2502 | 0.75 | 139 | 8 |

At the front of the queue, quoting indiscriminately earns 0.2551 ticks per fill. The best fitted gate (q=0.30) earns 0.2716, a difference of +0.0166 ticks — which is the filter doing its job: the edge is in declining to quote, exactly as Phase 0 suggested.

### How much of this is the rebate?

The best cell before any rebate is 0.2716 ticks per fill (queue 0.00×, gate q=0.30). Break-even needs -0.272 ticks of rebate (-0.00272 per share). The business therefore does not depend on the schedule.

| rebate (ticks) | markout (ticks) | after crossing out |
|---|---|---|
| 0.00 | 0.2716 | -0.9797 |
| 0.10 | 0.3716 | -0.8797 |
| 0.20 | 0.4716 | -0.7797 |
| 0.25 | 0.5216 | -0.7297 |
| 0.30 | 0.5716 | -0.6797 |

These rebate tiers are ILLUSTRATIVE, not a schedule. Replace `passive.rebate_sweep` with the venue's real numbers before any of this is quoted to anyone.


## Answers to the section-29 questions


**1. Does OFI predict future midprice movement OOS?**

Mean OOS pred-corr (M5_full, ms1000) = 0.02214; positive in 100.0% of folds. Day-blocked bootstrap 95% interval [0.01476, 0.02845] over 8 day(s) — the interval, not the point estimate, is the result.


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

**Not evaluated.** The policy executed ZERO trades across 8 out-of-sample day(s), so the reported net P&L of 0.00000 is the sum of an empty ledger, not a measured loss. See the execution-accounting table. No interval: the policy executed ZERO trades, so there is no P&L distribution to resample.


**9. Which regimes work/fail?**

Not evaluable — regime P&L is computed from the trade ledger, and no trades were executed. `08_regime_results.csv` is not written on a no-trade run.


**10. Survives latency & cost stress?**

Not evaluable — both sweeps re-run a policy that never trades, so every row is 0.0 by construction. A flat line of zeros across latencies is NOT evidence of latency-robustness.


**11. Concentrated in a few days?**

Not evaluable — no trades, so there are no trading days to concentrate in (the walk-forward itself covered 8 market days).


**12. Proceed to response-kernel stage?**

YES — per the decision gate above.


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
