# wave-length

A staged research programme building toward a PDE model of price formation from
order-book dynamics.

The eventual model expresses the drift of the midprice as

```
μ_t = θ₁·OFI_t + θ₂·intensity_t + θ₃·Δspread_t + θ₄·u_t
```

where `u_t` is the output of a wave / Duhamel response layer. **`u_t` has not
been built.** Everything to its left has, and this repository currently
contains the work needed to make the comparison `baseline + u_t` vs `baseline`
meaningful — because without a measured baseline, a later improvement cannot be
attributed to the wave layer.

## Roadmap

| stage | status |
|---|---|
| **1. Order-book / OFI baseline** | built and measured — see below |
| 2. Wave / Duhamel response layer (`u_t`) | not started |
| 3. Kinetic / Fokker–Planck layer | not started |

## Layout

```
wave-length/
├── data/pilot/          20 days of INTC mbp-10 (dbn.zst + parquet), 4.7 GB
└── ofi_research/        stage 1 — see ofi_research/README.md for the module map
    ├── configs/         baseline_intc.json
    ├── outputs/
    │   ├── baseline/    tables + OFI_RESEARCH_REPORT.md
    │   └── pilot_audit/ one-day raw-to-feature audit (protocol gate)
    └── tests/           149 tests
```

---

# Stage 1 — findings

**Dataset.** INTC, XNAS.ITCH `mbp-10`, 20 trading days (2026-07-13 → 2026-08-07),
~63M raw records → 2,158,686 decision rows at a 200 ms decision clock.
Walk-forward: rolling 10/2/1 train/validation/test → **8 folds, 8 distinct test
days** (2026-07-29 → 2026-08-07). Primary horizon **1 s**, selected by a fixed
outcome-independent rule.

## 1. OFI predicts — weakly, but real

| | value |
|---|---|
| Mean OOS correlation (M1) | **0.00728** |
| Day-blocked 95% CI | **[0.00249, 0.01188]** |
| Day-clustered t | 2.69 |
| Folds with positive correlation | 6 / 8 |
| Folds with positive β₁ | **8 / 8** |

The interval excludes zero and the coefficient never changes sign. This is a
genuine, consistently-signed relationship.

## 2. It is far too small to trade aggressively

Because models standardize on the training fold, `β₁` is directly "price move
per 1-SD of OFI" and converts straight into ticks:

| | value |
|---|---|
| Predicted move per 1-SD OFI | **0.0251 ticks** |
| Mean quoted spread | **2.53 ticks** |
| Ratio | **0.99%** |
| SD of OFI needed to cover a round trip | **~101** |

INTC in this sample is a **small-tick** name: median spread 3 ticks, only 10.95%
of time at one tick. The round-trip cost gate is therefore the full spread
(~2.4 ticks; fees, slippage and impact are all zero in the baseline config).

## 3. The backtest executed zero trades — and that is a measurement, not a loss

Across **857,973** out-of-sample rows:

| funnel stage | count |
|---|---|
| Rows with a usable prediction | 857,973 |
| … clearing the cost gate | **0** |
| … clearing the z gate | 1 |
| Signals | **0** |
| Executed trades | **0** |
| Largest single prediction, entire period | **1.37 ticks** (vs 2.38 needed) |

Not one prediction in a month cleared the threshold, at *zero* latency. Since
latency only delays execution and higher costs only raise the bar, every
row of the cost and latency sweeps is provably zero.

**Conclusion: OFI alone cannot support an aggressive taker strategy at this
horizon on this symbol.** Tradability was *not evaluated* — it was never
exercised. That is an absence of trading evidence, not evidence of
unprofitability, and the two are now reported differently.

## 4. The full specification is 3× better than OFI alone

| model | OOS corr | OOS R² | dir. acc |
|---|---|---|---|
| M0 constant | 0 | −0.000060 | 49.95% |
| M1 `OFI` | 0.00728 | **−0.000032** | **49.75%** |
| M4 `+intensity` | 0.00956 | +0.000015 | 49.61% |
| M3 `+signed volume` | 0.01114 | +0.000060 | 49.88% |
| M2 `+spread/depth` | 0.01894 | +0.000296 | 51.42% |
| **M5 full** | **0.02214** | **+0.000425** | **51.66%** |

M1 — which every headline number in the original report was based on — has
**negative** out-of-sample R² and directional accuracy **below a coin flip**.
M5 corresponds to `θ₁·OFI + θ₂·intensity + θ₃·Δspread`, i.e. the target
equation minus `u_t`, and most of its gain comes from spread/depth (M2), not
from intensity or signed volume.

**This is the baseline `u_t` must beat.** Benchmarking the wave layer against
M1 would credit it with gains plain spread/depth already delivers.

---

# Work completed (2026-08-12)

## Three reporting bugs, each of which changed a conclusion

1. **Day-count conflation.** One `n_days` field served as both trading-day and
   market-day count, so a report could simultaneously claim 0, 8, and "fewer
   than two" out-of-sample days. Folds, market dates, dates with predictions
   and dates with trades are now four separate printed quantities.

2. **Per-trade P&L averaged over folds, not pooled.** A mean of fold means
   weights a 1-trade fold like a 50-trade fold, and inverted the reported sign:
   **+0.32 ticks/trade against a true −1.36**, sitting beside the pooled total
   in the same table.

3. **Concentration share went negative.** `max_day/total` is undefined when the
   total is negative, so a one-day-dominated **loss** passed the "< 0.6"
   concentration gate. Now `|max day| / Σ|daily|`, bounded in [0, 1]. On
   synthetic data this flipped a spurious PASS to a correct FAIL at 0.849.

## A design error caught by a test

The passive screen originally used the taker study's Spearman monotonicity.
That statistic reads ≈0 on maker data even when OFI predicts toxicity
perfectly, because adverse selection is **symmetric** — negative OFI leaves you
long into falling prices, positive leaves you short into rising ones, so
markout against raw OFI is U-shaped. It would have produced a **false negative**
killing the passive hypothesis. Fixed by conditioning on *fill-aligned* OFI
(OFI signed by the position the fill leaves you in), which folds the U into a
line.

## Reporting made honest

- **Three-valued decision gate**: `PASS` / `FAIL` / `NOT_EVALUABLE`. A criterion
  never exercised is not a failure. With zero trades, `net_positive`,
  `not_one_day` and `survives_stress` are `NOT_EVALUABLE`.
- **Execution accounting** (`19_`): the full signal funnel, so an empty ledger
  can never again masquerade as a loss.
- **Effect size in ticks and bps**, not just correlation.
- **Real ablation deltas** in the answers, instead of "see the CSV".
- Verdict text distinguishes "never traded" from "traded and lost".

## New capability

- `passive.py` — **Phase 0**: a front-of-queue passive-markout screen that
  bounds market-making performance from the MBP-10 data already on disk. Real
  queue position is always worse, so an unprofitable bound is decisive and no
  market-by-order data need be bought. Tables `20`–`22`. **Never yet run on
  real data.**
- `costs.maker_rebate_per_unit` — the cost model previously had no concept of a
  rebate, only a symmetric charge applied twice. Defaults to 0 and says so
  loudly; at a $0.01 tick a typical add rebate is 0.2–0.3 ticks.
- `evaluation.reference_model` (default `M5_full`) — the model every headline
  tracks, replacing `M1_ofi` hardcoded in 13 places.
- `evaluation.run_cks_replication` (default **off**) — CKS is a construction
  audit, not the model.
- `phase0` CLI subcommand — runs the passive screen without the walk-forward.
- 14 new tests (**149 total**).

## CKS bin-count anomaly — diagnosed and fixed

Bin counts should fall monotonically as the window widens. They did not:

| window | expected bins | actual | |
|---|---|---|---|
| 10 s | 46,800 | 46,560 | ✓ |
| 5 s | 93,600 | 93,120 | ✓ |
| 1 s | 468,000 | 463,400 | ✓ |
| 500 ms | 936,000 | 838,046 | ✓ |
| **100 ms** | **4,680,000** | **62,197** | ✗ **1.3%** |

Cause: the decision clock is 200 ms, so a 100 ms bin sits **below the sampling
resolution** — most bins are empty and few have the two observations needed to
form a Δm. Windows narrower than `decision_interval_ms` are now dropped with a
warning. Everything ≥ 500 ms was always sound.

## Housekeeping

228 lines of redundant prose removed (24.4% → 21.7% of non-blank lines), 98
decorative banner lines collapsed, 9 unused imports removed, `ofi_research/`
README rewritten to match the actual repo, stray synthetic outputs deleted, and
stale Aug-11 artifacts quarantined in `outputs/baseline/_stale_20260811/`.

---

# Known issues

- **A full clean run has never completed.** The 2026-08-12 run was killed by the
  OS at 06:21 (17 GB machine, 782k pageouts) after data prep, walk-forward,
  deciles, ablation and leave-one-group-out. The current report is
  **reconstructed** from the tables that run left on disk and is labelled as
  such at the top. Its numbers are sound; `16`, `17`, `18` and all plots are
  stale.
- **`07b_leave_one_group_out` cost 54 minutes** to produce one 7-row table and
  is the peak-memory stage. Disable it or raise `decision_interval_ms` before
  the next full run.
- **The current report is M1-based.** Regenerating against `M5_full` requires a
  fresh run; per-fold M5 predictions were not retained.
- **The execution path has never produced a trade on real data.** It is covered
  by unit tests but has never been exercised end-to-end on INTC. If a stronger
  model does trade, that code runs for the first time — check it deliberately.
- **Phase 0 has never been run on real data.**
- One symbol only. `18_tick_regime` records the preregistration requirement:
  a large-tick name must be a **separate** replication, never pooled.

# Next steps

1. **Re-run against `M5_full`** with `07b` disabled — gives a clean, complete
   baseline at the correct reference model. M5's predictions are ~3× M1's, so
   it may trade, which would exercise the execution path for the first time.
2. **Run Phase 0** to settle whether OFI predicts adverse selection well enough
   for passive quoting. This is the only live *tradability* path; a 100× effect-
   size gap will not be closed by `u_t`.
3. **Then build `u_t`** — see the brief below.

Note that (2) and (3) answer different questions. The wave layer can be
validated as a **predictive improvement** even if the signal is never tradable
through aggressive execution — just don't let a positive Δcorr be read as
"now it trades."

---

# Stage 2 brief — building `u_t`

## The empirical impulse response you are trying to reproduce

Stage 1 already measured how OFI's predictive power decays with horizon. This
curve *is* the response function, measured directly (`OFI1_ref`, HAC t-stats,
FDR-adjusted, n ≈ 1.4–2.2M per row, table `03`):

| horizon | Pearson | HAC t | FDR p | significant? |
|---|---|---|---|---|
| **100 ms** | **0.0308** | 29.9 | ~1e-195 | ✔ strongest |
| 250 ms | 0.0234 | 25.6 | ~1e-143 | ✔ |
| 500 ms | 0.0146 | 16.2 | ~1e-58 | ✔ |
| 1 s | 0.0091 | 9.6 | 2e-21 | ✔ |
| 2 s | 0.0057 | 5.8 | 1.2e-8 | ✔ |
| 5 s | 0.0017 | 1.7 | 0.11 | ✘ |
| 10 s | −0.0004 | −0.4 | 0.72 | ✘ |
| 30 s | −0.0005 | −0.5 | 0.66 | ✘ |

**Read this before choosing anything else.** Three consequences:

1. **The response is monotonically decaying with a half-life of roughly
   400 ms, and is statistically dead beyond ~5 s.** A kernel `G` with a
   characteristic timescale of seconds is fitting noise; the dynamics live
   below one second.
2. **The 1 s primary horizon is not where the signal is.** It was chosen by a
   fixed outcome-independent rule ("smallest clock horizon ≥ 1 s"), and it sits
   3.4× below the 100 ms correlation. Evaluating `u_t` only at 1 s tests the
   tail of the response, not its body. **Test at 100–500 ms as well.**
3. **The 200 ms decision clock is a floor.** At 100 ms only 67% of rows have a
   usable target and ~50% of targets are exactly unchanged (vs 18.6% at 1 s).
   Working below 200 ms means lowering `sampling.decision_interval_ms`, which
   raises memory — already the binding constraint (see Known issues).

Event-clock horizons agree: `ev25`/`ev50` (median realized spans 263 ms /
472 ms) give correlations 0.0127 / 0.0130, matching the 250–500 ms clock rows.
`02b_horizon_and_window_realization.csv` maps every event window to realized
milliseconds.

## Integration contract

`u_t` must reach the pipeline as a **column on the feature frame**, after which
everything downstream works unchanged.

1. **Compute the kernel output** as a new causal feature column, e.g. `wave_u`.
   The available increment series are `OFI_level_1_increment`,
   `OFI_level_2_increment`, `signed_volume_increment`; trailing aggregates
   already exist over event windows `ev10/25/50/100` and time windows
   `ms100/500/1000/5000/10000` (`features.py`, `_rolling_event_sum` /
   `_rolling_time_sum` — reuse these, they handle segment resets).
2. **Add a feature group** in `models.feature_groups`, then a model in
   `models.model_feature_sets`:
   ```python
   "M6_wave": ofi + g["spread"] + g["depth"] + signedvol
              + g["intensity"] + g["wave"],     # == M5_full + u_t
   ```
   and append `"M6_wave"` to `LADDER_ORDER`. The ladder holds raw OFI fixed and
   adds one group at a time precisely so a change is attributable to the group
   added — keep that property.
3. **Retain the column through sampling.** Anything matching the prefixes in
   `sampling.required_columns` survives automatically; a new prefix must be
   added there or it is pruned before the models see it.
4. **Set `evaluation.reference_model`** to compare against. The baseline is
   `M5_full`, not `M1_ofi`.

Nothing in `splits`, `costs`, `backtest`, `diagnostics` or the report needs to
change. The backtest takes a single `predictions` array aligned to rows
(`backtest.run_backtest`), so a kernel model that produces one is a drop-in.

## Non-negotiable constraints

Any `u_t` must satisfy the same contracts as every other feature, or the result
is not comparable to the baseline:

- **Strictly causal.** Value at row `t` uses only rows with timestamp ≤ `t`.
  A Duhamel integral is a convolution over the *past* only.
- **Segment-reset.** Never convolve across a segment boundary (session change,
  book clear, halt, integrity break). Reuse the existing rolling helpers, which
  already reset per `segment_id`.
- **Train-only fitting.** Any kernel parameter estimated from data must be fit
  on the training fold only, like every scaler, winsorization limit and decile
  edge. Fitting `G` on the full sample is leakage.
- **It must pass `tests/test_no_leakage.py`.** The truncation test rebuilds
  features on a strict prefix and requires identical values on retained rows.
  This is the strongest single check that a kernel does not peek forward — add
  the new column to that test.

## How to judge the result

Compare `M6_wave` against `M5_full` on:

- **ΔOOS correlation** and **ΔOOS R²**, per fold, at 100 ms / 250 ms / 500 ms /
  1 s — not at 1 s alone.
- **Day-blocked bootstrap interval on the delta** (table `13`). With 8 test days
  the interval will be wide; that width is the result, not an inconvenience.
- **Sign consistency across folds** (table `09`), the same bar OFI cleared 8/8.
- **`07b` leave-one-group-out** with the wave group removed, which is the direct
  attributable test — though note it is the peak-memory stage.

Reference points from the baseline: M1 → M5 moved correlation 0.00728 → 0.02214
and R² −0.000032 → +0.000425. A wave layer that adds less than the spread/depth
group already contributes is not yet interesting.

**Do not judge `u_t` on P&L.** The taker path is dead by a factor of 100 and no
kernel closes that. Tradability is Phase 0's question, not stage 2's.
