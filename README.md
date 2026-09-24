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
| **1. Order-book / OFI baseline** | built and measured on two complete runs — see below |
| 1b. Passive (maker) tradability | measured; promising but **not settled** — exit costs unmodelled |
| 2. Wave / Duhamel response layer (`u_t`) | not started |
| 3. Kinetic / Fokker–Planck layer | not started |

## Layout

```
wave-length/
├── data/pilot/          20 days of INTC mbp-10 (dbn.zst + parquet), 4.7 GB
└── ofi_research/        stage 1 — see ofi_research/README.md for the module map
    ├── configs/         baseline_intc.json, passive_wf_intc.json
    ├── outputs/
    │   ├── baseline/    ORIGINAL M1-based run (superseded, kept for history)
    │   ├── baseline_m5/ first complete run — taker + Phase 0 screen
    │   ├── passive_wf/  current headline run — Phase 0 walk-forward
    │   └── pilot_audit/ one-day raw-to-feature audit (protocol gate)
    └── tests/           180 tests
```

`outputs/passive_wf/` is the current reference. `outputs/baseline/` predates the
`M5_full` reference model and mislabels M5 numbers as M1 — read it as history,
not as a result.

---

# Stage 1 — findings

**Dataset.** INTC, XNAS.ITCH `mbp-10`, 20 trading days (2026-07-13 → 2026-08-07),
**72.3M raw records → 68.6M canonical events → 2,158,686 decision rows** at a
200 ms decision clock. (Earlier revisions of this file said "~63M raw records";
that figure was wrong — the decision-row count is unchanged, so it is the same
dataset.)
Walk-forward: rolling 10/2/1 train/validation/test → **8 folds, 8 distinct test
days** (2026-07-29 → 2026-08-07). Primary horizon **1 s**, selected by a fixed
outcome-independent rule.

## 1. OFI predicts — weakly, but real

Headline numbers are for **`M5_full`**, the reference model (not `M1_ofi`):

| | value |
|---|---|
| Mean OOS correlation | **0.02214** |
| Day-blocked 95% CI | **[0.01476, 0.02845]** |
| Day-clustered t | 5.71 |
| Folds with positive correlation | 8 / 8 |
| Folds with positive β₁ | **8 / 8** |

The interval excludes zero and the coefficient never changes sign.

## 2. The full specification is 3× better than OFI alone

| model | OOS corr | OOS R² | dir. acc |
|---|---|---|---|
| M0 constant | 0 | −0.000060 | 49.95% |
| M1 `OFI` | 0.00728 | **−0.000032** | **49.75%** |
| M4 `+intensity` | 0.00956 | +0.000015 | 49.61% |
| M3 `+signed volume` | 0.01114 | +0.000060 | 49.88% |
| M2 `+spread/depth` | 0.01894 | +0.000296 | 51.42% |
| **M5 full** | **0.02214** | **+0.000425** | **51.66%** |

M1 has **negative** out-of-sample R² and directional accuracy **below a coin
flip**. Most of M5's gain comes from spread/depth (M2), not from intensity or
signed volume. **This is the baseline `u_t` must beat.**

## 3. The aggressive taker path is dead

A 1-SD move in the reference feature predicts **0.0145 ticks** against a
**2.53-tick** mean spread — 0.57% of it, so ~174 SD would be needed to cover one
round trip. The 20-day M5 run (`outputs/baseline_m5/`) executed **1 trade** from
857,967 eligible rows: 72 cleared the cost gate, 1 cleared the z gate.

That single trade won 4 ticks, and the decision gate duly reported "net-positive
out of sample" beside a concentration FAIL saying 100% of P&L came from one day.
Both sentences described the same trade. See *Work completed (2026-09-13)*.

**Conclusion: OFI alone cannot support an aggressive taker strategy at this
horizon on this symbol.** The taker study is off by default
(`evaluation.run_taker_backtest = False`) — it is a settled conclusion, not a
bug, and re-deriving it costs ~25 min per run.

---

# Stage 1b — the passive (maker) path

The strategy question moved here, because a maker earns the spread rather than
paying it. Two studies, in increasing order of honesty.

## Phase 0 — front-of-queue screen (a ceiling)

Assumes we are first in queue and fill on every trade at our price. Real queue
position is strictly worse, so an unprofitable ceiling would be decisive.

**Unconditional** front-of-queue markout at 1 s is **−0.0537 ticks** (t = −1.99)
— the default maker business loses money at the best possible queue position.

**Conditioned on fill-aligned OFI**, fills separate cleanly:

| | taker deciles | **passive deciles** |
|---|---|---|
| Spearman monotonicity | 0.556 | **0.879** |
| Bottom → top spread | — | **0.183 ticks** |

Worst decile −0.150 ticks, best **+0.089** — at zero rebate. So the edge is not
in quoting, it is in **not** quoting into toxic flow, which makes the filter the
strategy.

Decile edges here are cut pooled over the whole sample and therefore peek. That
is why the walk-forward below exists.

## Phase 0 walk-forward — the fitted gate (`outputs/passive_wf/`)

Strictly harder than the screen on four axes: the gate threshold is fitted on
**train days only**, queue position is a swept dial rather than a free
assumption, cancels are assumed to leave from behind us, and both quoting and
cancelling pay latency. Fills simulate at event resolution; the gate stays on
the 200 ms decision clock.

All five passive criteria PASS and the report says **Proceed: YES**.
**Do not act on that verdict yet** — three problems, in order of severity.

### 1. Exit costs are omitted, and they dominate

Table `23` carries `markout_after_crossing_out_ticks` — markout minus half the
spread, i.e. the cost of flattening the position — and **no gate criterion
consults it**:

| gate | queue | markout | after crossing out |
|---|---|---|---|
| 0.3 | 0.00 | +0.272 | **−0.980** |
| 0.3 | 0.25 | +0.121 | **−1.138** |
| 0.3 | 1.00 | +0.203 | **−1.007** |

**All 20 zero-rebate cells fall in −1.156 … −0.945 ticks.** The headline assumes
inventory unwinds at mid for free; a typical 0.20–0.30 tick add rebate still
leaves roughly −0.7. The honest statement is a **bracket of [−1.0, +0.25] ticks
per fill**, whose width is set by how much inventory unwinds passively — and
inventory is not modelled at all (no limit, no unwind policy).

The same column exists in the Phase-0 screen (table `20`): −1.39 ticks at 1 s.

### 2. Two criteria pass on noise, chosen from 100 cells

* `filter_beats_unconditional` — 0.2716 vs 0.2551 is **+0.017 ticks against
  SE 0.035**, about half a standard error. Down the gate column at front of
  queue: 0.255 → 0.272 → 0.264 → 0.257 → 0.242. Flat and non-monotonic. The OFI
  gate adds nothing measurable there; it only cuts fills 115k → 12k.
* `survives_back_of_queue` — quotes the queue=1.0 / gate=0.9 cell: 1,111 fills,
  SE 0.336, **t = 0.75**. The gate picks the largest point estimate across 100
  cells with no multiplicity adjustment, and that cell is the noisiest.

The back-of-queue claim is nonetheless defensible via a better cell:
queue=1.0 / gate=0.3 gives **+0.203 ticks at t = 5.48**.

### 3. Unexplained sign flip against Phase 0

Same 20 days, both nominally unconditional and front-of-queue:

| | markout @ 1 s | fills/day |
|---|---|---|
| Phase 0, table `20` | **−0.0537** | 2,961 |
| Policy grid, gate=0 / queue=0 | **+0.2551** | 14,376 |

`gq ≤ 0` sets `thr = −inf`, so both sides really are quoted always, and sides are
pooled. The likely cause is that Phase 0 infers fills on the decision clock
while the policy simulates at event resolution with queue mechanics — but until
that is reconciled, **one of the two is mismeasuring**.

### What is actually supported

Gated passive markout is positive and significant across queue positions —
t ≈ 3–5.5 in well-populated cells — **before exit costs**. That is a real signal
and worth pursuing. It is **not** yet evidence that the maker business makes
money.

---

# Work completed (2026-09-13)

## Recovered the repository

A `git revert` of the initial commit on 2026-08-14 deleted all 165 tracked files
(−14,006 lines); three later `git reset --hard` cycles rebuilt `main` from that
emptied state, leaving only the `passive_m5` outputs and orphaning each
intervening commit. The tree was restored from `a857d85`, a strict superset of
what `main` held. Recovered commits are tagged `recovered-a857d85`,
`dangling-789bb54`, `dangling-a78580d`, `dangling-465a258`.

## The decision gate no longer scores a ledger too thin to score

`net_positive`, `not_one_day` and `survives_stress` read `NOT_EVALUABLE` unless
the ledger clears **30 trades across ≥2 distinct days**
(`evaluation.min_trades_for_pnl_gate`, `min_trade_days_for_pnl_gate`; `0`
restores the old behaviour). The three-valued gate already existed to stop an
*empty* ledger reading as a failure — that reasoning did not extend far enough,
because a one-trade ledger supports a verdict no better than an empty one.

The day floor earns its place independently: overlapping targets make same-day
rows dependent, so a one-day ledger carries one effective observation however
many trades it holds, and `not_one_day` reads 1.000 by arithmetic rather than by
evidence.

Two verdict strings also asserted more than the gate had established — one
claimed the result "survives the tested costs and stress checks" while
`survives_stress` was `NOT_EVALUABLE`; the other opened "predictive and
net-positive out of sample" unconditionally. Both now condition on the criterion
having been scored, and a thin ledger reads differently from "never traded".
`pnl_gate_evaluable` is a pure function so the rule is testable without a run
context. 8 tests added (**180 total**).

## First Phase 0 walk-forward on real data

`configs/passive_wf_intc.json`, 20 days, 4h55m wall clock. Findings and caveats
above. The maker rebate is deliberately **not** set to a single value: the config
sweeps `[0, 0.0010, 0.0020, 0.0025, 0.0030]` and the gate reports a break-even,
so the output is a bracket rather than an invented venue number.

## Reporting bug confirmed fixed

`outputs/baseline_m5/` labels its uncertainty row `oos_pred_corr_M1` while
carrying M5_full's value — those outputs predate the current reporting layer.
The new run emits `oos_pred_corr_M5_full` correctly.

---

# Known issues

- **Exit costs are unmodelled in every passive result.** The headline
  `markout_ticks` assumes free unwinding at mid; `markout_after_crossing_out`
  assumes crossing on every fill. Reality is between, and no criterion reports
  the bracket. This is the single largest gap.
- **The passive decision gate selects the best of 100 cells** without a
  multiplicity adjustment, and `survives_back_of_queue` currently passes on a
  t = 0.75 cell.
- **Phase 0 and the policy grid disagree in sign** on the unconditional
  front-of-queue markout (−0.054 vs +0.255). Unreconciled.
- **Queue position cannot be measured from MBP-10** — it is swept, and the
  bracket is the result.
- **`07b_leave_one_group_out` cost 54 minutes** for one 7-row table and is the
  peak-memory stage. Off by default.
- One symbol only. `18_tick_regime` records the preregistration requirement:
  a large-tick name must be a **separate** replication, never pooled.
- `outputs/passive_wf/passive_tapes/` is 556 MB of regenerable cache. It should
  be gitignored rather than committed.

## Memory — measured, not estimated

The earlier runbook figure of ~9 GB/day was pessimistic, and one of its claims
was wrong.

| | runbook | measured 2026-09-13 |
|---|---|---|
| Peak RSS, largest day | ~8.5 GB | **6.71 GB** |
| Accumulation across days | "does not accumulate" | **≈0.22 GB per session** |
| Full 20-day run | ~3 h | **4 h 55 m** |

Memory **does** accumulate: the per-day event frame is released, but the sampled
200 ms rows persist, so RSS climbs (1.98 GB at day 10 → 5.36 GB at day 20) and
the pace degrades from ~5 to ~15 min/day. The largest sessions are in the back
half of the sample, which compounds it.

With 113 GB of free disk macOS keeps extending the swap file, so the failure mode
is slowness rather than the OOM kill seen in August. Free several GB before
starting anyway.

---

# Next steps

1. **Close the exit-cost gap.** Add a criterion that reports the
   [cross-out, no-cross-out] bracket instead of one end, and decide what
   fraction of inventory realistically unwinds passively. Until this is done the
   maker path has no verdict — the current PASS is the optimistic end of its own
   bracket.
2. **Reconcile Phase 0 against the policy grid**, and require significance
   rather than the largest point estimate for `survives_back_of_queue`.
3. **Then build `u_t`** — see the brief below.

Note that (1)–(2) and (3) answer different questions. The wave layer can be
validated as a **predictive improvement** even if the signal is never tradable —
just don't let a positive Δcorr be read as "now it trades."

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
kernel closes that. Tradability is the passive path's question, not stage 2's —
and that question is still open on exit costs (see *Stage 1b*), so there is no
P&L number for a kernel to improve on yet.
