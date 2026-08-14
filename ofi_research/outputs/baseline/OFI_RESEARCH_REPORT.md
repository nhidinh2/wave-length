# OFI Research Report

> **⚠️ RECONSTRUCTED REPORT.** The 2026-08-12 run was killed by the OS at
> 06:21 (memory pressure) after completing data prep, the walk-forward,
> deciles, ablation and leave-one-group-out. This report was rebuilt from the
> tables that run left on disk (`01`–`09`, `07b`, `15`), not from a complete
> pipeline pass.
>
> * Recomputed here: `06`, `10`, `13`, `19` and this report.
> * `11`/`12` are derived from a proof, NOT copied from the earlier run:
>   `bt_pass_cost_gate = 0` means no prediction cleared the round-trip cost
>   threshold at ZERO latency, and latency/cost can only make that worse, so
>   every sweep row is exactly zero.
> * NOT regenerated, still from the 2026-08-11 run: `16`, `17`, `18` and all
>   `p_*.png` plots. Treat those as stale.
> * `08_regime_results.csv` and `10b_per_day_net_pnl.csv` do not exist because
>   there were no trades to bucket.





## Evidence status — three separate claims

| Claim | Status |
|---|---|
| **Implementation validity** (the code computes what it says) | Tests pass on this run; the one-day raw-to-feature audit (`ofi_research pilot`) must also have been reviewed by hand before these numbers mean anything. |
| **Predictive evidence** (OFI forecasts future mid moves OOS) | Measured on 8 out-of-sample fold(s) across 8 distinct market day(s) — see the day-blocked interval in `13_overlap_aware_uncertainty.csv`, not the point estimate. |
| **Net trading evidence** (that edge survives execution) | **NOT EVALUATED** — the policy executed ZERO trades on 8 day(s) of eligible predictions, so no execution assumption was ever exercised. This is an absence of evidence, NOT evidence of unprofitability. |

Predictive evidence does not imply net trading evidence, and neither follows from implementation validity.

- Label: `INTC_20d_baseline`  |  Primary horizon: `ms1000`  |  Walk-forward: `rolling` (10/2/1 train/val/test days)


## Headline decision

**Verdict:** PREDICTIVE BUT UNTESTED FOR TRADABILITY. OFI shows a weak, consistently positive out-of-sample relationship with future mid moves, but the trading policy executed ZERO trades, so nothing about profitability was measured — the P&L, cost and latency tables are undefined, not zero. The forecast is roughly 0.0087 of the round-trip cost it must clear, so the gap is one of effect size, not of threshold tuning. Do NOT advance to the wave/PDE layer on this basis, and do not record this run as evidence that the signal is unprofitable.


**Proceed to impulse-response / Green's-function stage?** NO


### Decision-gate checklist

| Criterion | Status | Basis |
|---|---|---|
| stable_coef_sign | ✅ PASS | beta_1 positive in 100.0% of 8 folds |
| nonzero_oos_corr | ✅ PASS | mean OOS corr 0.00728 in 75.0% of folds |
| sensible_deciles | ✅ PASS | decile Spearman 0.556 |
| net_positive | ⬜ NOT_EVALUABLE | no trades executed — never tested |
| not_one_day | ⬜ NOT_EVALUABLE | no trading days to concentrate in |
| survives_stress | ⬜ NOT_EVALUABLE | stress sweeps ran on a policy that never traded |

> 3 criteria were NOT EVALUABLE on this run: `net_positive`, `not_one_day`, `survives_stress`. They are not failures — they were never exercised. `proceed` requires PASS on all six, so it remains NO.


## Execution accounting — what the policy actually did

| Quantity | Value |
|---|---|
| Out-of-sample folds | 8 |
| Distinct OOS market dates | 8 |
| OOS dates with eligible predictions | 8 |
| OOS dates with executed trades | 0 |
| Rows with a usable prediction | 857973 |
| … clearing the cost gate |pred| > round-trip cost | 0 |
| … clearing the z gate | 1 |
| Signals (both gates) | 0 |
| Dropped: no executable quote after latency | 0 |
| Dropped: cannot close inside segment | 0 |
| Dropped: insufficient displayed depth | 0 |
| Suppressed by overlap guard | 0 |
| **Executed trades** | **0** |
| Long / short | 0 / 0 |
| Fraction of time holding a position | 0.0000 |
| Turnover (notional) | 0.00 |

> **NO TRADES WERE EXECUTED.** Tradability was NOT evaluated; it was not tested. The mean |prediction| is 0.00020464 price units (0.0205 ticks) against a mean round-trip cost threshold of 0.023839 (2.384 ticks) — a ratio of 0.00866. The largest single prediction in the whole out-of-sample period was 1.3727 ticks. The forecast is smaller than the spread it must cross by orders of magnitude, so no threshold setting makes this policy trade. Read every P&L, cost and latency figure below as UNDEFINED, not as zero.


## Effect size in economic units

A 1-SD increase in `OFI1_ev50` predicts a mid move of:

| Unit | Value |
|---|---|
| price units | 0.00025110 |
| **ticks** | **0.02511** |
| basis points | 0.02590 |
| 95% CI (ticks, across folds) | [0.02029, 0.02993] |
| mean quoted spread (ticks) | 2.530 |
| **predicted move / spread** | **0.00993** |

Read this as: the effect is real and consistently signed, but a 1-SD OFI move buys about 0.0251 of a tick against a 2.53-tick spread. Roughly 101 SD of simultaneous OFI would be needed to cover one round trip. Decile monotonicity of 0.556 is moderate, not strong.


## Answers to the section-29 questions


**1. Does OFI predict future midprice movement OOS?**

Mean OOS pred-corr (M1, ms1000) = 0.00728; positive in 75.0% of folds. Day-blocked bootstrap 95% interval [0.00249, 0.01188] over 8 day(s) — the interval, not the point estimate, is the result.


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
- 8 out-of-sample market days is a small number of independent days; treat all fold-fraction statistics (e.g. '75.0% of folds') as counts out of 8, not as stable probabilities.


## Feature availability notes

- Signed trade volume is the vendor's `signed_trade_volume`, a NET per-event quantity (one canonical event may contain both buy and sell executions, so size x direction would not reproduce it). It is therefore not derived from the direction percentages that follow, which describe the SEPARATE per-trade `trade_direction` used by the intensity features: Aggressor direction came from the vendor side field for 71.34% of trades, from a causal 'lee_ready' classifier for 22.77%, and was UNRESOLVED for 5.88% (direction 0, contributing zero signed volume — unresolved trades are neutral, never dropped and never guessed).
