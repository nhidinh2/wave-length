# wave-length

A staged research programme building toward a PDE model of price formation from
order-book dynamics.

The eventual model expresses the drift of the midprice as

```
μ_t = θ₁·OFI_t + θ₂·intensity_t + θ₃·Δspread_t + θ₄·u_t
```

where `u_t` is the output of a wave / Duhamel response layer. A **first
version of `u_t` now exists** (`M6_wave`, 2026-09-24): OFI convolved with a small
basis of causal response kernels, weights fitted on train days only. It is
scored against the measured baseline `M5_full` — see *Stage 2 — first result*.

## Roadmap

| stage | status |
|---|---|
| **1. Order-book / OFI baseline** | built and measured on three complete runs — see below |
| 1b. Passive (maker) tradability | **settled negative** (2026-09-24) — loses at every queue position, before and after exit costs |
| **2. Wave / Duhamel response layer (`u_t`)** | first version built; **+15% OOS correlation at 100 ms, 8/8 days, CI excludes zero**; not significant at 1 s |
| 3. Kinetic / Fokker–Planck layer | not started |

## Layout

```
wave-length/
├── data/pilot/          20 days of INTC mbp-10, 4.7 GB — NOT in git (see below)
└── ofi_research/        stages 1-2 — see ofi_research/README.md for the module map
    ├── configs/         baseline_intc.json, passive_wf_intc.json
    ├── outputs/
    │   ├── baseline/       ORIGINAL M1-based run (superseded, kept for history)
    │   ├── baseline_m5/    first complete run — taker + Phase 0 screen
    │   ├── passive_wf/     2026-09-13 run — maker PASS was a simulator bug; VOID
    │   ├── passive_wf_v2/  current reference — corrected maker study + M6_wave
    │   └── pilot_audit/    one-day raw-to-feature audit (protocol gate)
    └── tests/           200 tests
```

`outputs/passive_wf_v2/` is the current reference. `outputs/passive_wf/` is kept
only as the record of the bug described under *Stage 1b*. `outputs/baseline/`
predates the `M5_full` reference model and mislabels M5 numbers as M1.

**Data is not in the repository.** The raw files exceed GitHub's 100 MB limit, so
`data/` was removed from history and is gitignored. Copy `data/pilot/` from a
machine that has it, or re-buy it one day at a time with
`python3 -m ofi_research.fetch_pilot_day --symbol INTC --date YYYY-MM-DD`
(needs `DATABENTO_API_KEY`; this is a paid download). A run also writes `features.parquet` (the sampled decision frame) and
`passive_tapes/` into its output directory; both are regenerable caches and
gitignored. `run --features <run>/features.parquet --output <run>` reuses them
and skips the ~4.5 h data prep.

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

That screen infers fills on the 200 ms decision clock and prices them at the
PREVIOUS decision row's quote, so it measures something different from the
event-level simulator below. Its decile ordering is still the reason the filter
was worth testing; its unconditional number is superseded.

## Walk-forward maker (`outputs/passive_wf_v2/`) — settled negative

The gate threshold is fitted on **train days only**, queue position is swept,
cancels are assumed to leave from behind us, quoting and cancelling pay latency,
and fills simulate at event resolution. Every result below is at zero rebate,
1 s markout, on the 8 held-out test days.

### The earlier PASS was a simulator bug

The 2026-09-13 run (`outputs/passive_wf/`) passed every maker criterion with
**+0.255 ticks** per unconditional front-of-queue fill. That number was an
artifact. Each tape row carries the book **after** its event, and on INTC 65% of
sell prints (69% of sell volume) arrive on a row whose bid has already dropped —
the print cleared the level. The simulator charged each print to its own row, so
a level-clearing fill was credited one level lower, a tick better than it
printed, or lost outright when the price change ended the queue episode first.
Precisely the most adverse fills disappeared. The same mechanism explains both
open puzzles from that run: the sign flip against Phase 0, and back-of-queue
cells looking *better* than mid-queue ones.

`precompute_side` now charges each print to the book it traded against; two
tests pin it and fail on the old code. Fill counts roughly triple once the
level-clearing fills are counted.

### Corrected result: negative everywhere

| filter (gate quantile) | front of queue | 0.25 | 0.5 | behind all depth |
|---|---|---|---|---|
| none (quote always) | −0.260 | −0.391 | −0.424 | −0.289 |
| 0.5 | −0.232 | −0.376 | −0.408 | −0.275 |
| 0.9 (most selective) | −0.214 | −0.329 | −0.375 | −0.266 |

Markout in ticks per fill, which still assumes the position can be sold at mid
for free. **All 20 zero-rebate cells are negative**, t between −2.4 and −12.9.

### Exit costs, now simulated

Each fill is flattened by a pegged passive order at the opposite touch — same
queue position as the entry, re-pegging when the touch moves — and crossed if it
has not filled by the timeout (`passive.unwind_timeouts_ms`). This replaces the
old bracket with a number inside it:

| filter | queue | free exit at mid | 1 s timeout | 5 s timeout | crossing every exit |
|---|---|---|---|---|---|
| none | front | −0.260 | −0.864 | −0.561 | ≈ −1.56 |
| 0.9 | front | −0.214 | −0.758 | **−0.472** | |
| 0.9 | behind all | −0.266 | −1.202 | −0.811 | −1.478 |

The best round trip anywhere in the sweep is **−0.47 ticks**. Break-even needs
**+0.505 ticks of rebate per share** (paid on the entry and on the 60% of exits
that fill passively), against a typical US add tier of 0.20–0.30.

### Does the filter help at all?

Yes, a little. Paired day by day against quoting always at the same queue
position, the best filter adds **+0.029 ticks per fill (t = 4.01)** — just short
of the Bonferroni bar of 4.41 for the 16 comparisons searched, and about 1/20th
of what would be needed. The OFI signal is real; it is too small to pay for the
adverse selection a maker takes on.

### Gate (table `25`)

| criterion | status | basis |
|---|---|---|
| `markout_positive_significant` | FAIL | best cell −0.266 ticks |
| `filter_beats_unconditional` | FAIL | +0.029 ticks, t = 4.01 vs 4.41 |
| `survives_back_of_queue` | FAIL | −0.266 ticks |
| `positive_after_exit_costs` | FAIL | −0.81 ticks, t = −5.43 (best by t) |
| `rebate_not_load_bearing` | FAIL | needs +0.505 ticks |

**Conclusion: OFI is not tradable on INTC at this horizon, as taker or as
maker.** Both negative verdicts are about effect size, not tuning. Any future
trading claim needs a different signal, a different symbol (a large-tick name,
run as a separate replication), or both.

---

# Stage 2 — first result

`M6_wave` = `M5_full` + the wave group and nothing else, so any difference is
attributable to `u_t`. The wave group is level-1 OFI increments convolved, in
continuous time and per segment, with five causal kernels
(`features.wave_kernels`): `exp(−Δt/τ)` for τ = 100 ms, 400 ms and 1.6 s, and
the damped-oscillator pair `exp(−Δt/1 s)·sin/cos(2πΔt/2 s)`, i.e. the Green's
function of `u'' + 2γu' + ω²u = OFI`. The kernel `G = Σ w_k φ_k` is fitted by
the walk-forward regression on train days only. The timescales were read off the
pooled stage-1 response table — a mild hyperparameter peek, disclosed.

| horizon | M5_full corr | M6_wave corr | Δcorr | 95% CI (day bootstrap) | days Δ>0 | ΔR² | days ΔR²>0 |
|---|---|---|---|---|---|---|---|
| **100 ms** | 0.04774 | 0.05509 | **+0.00735 (+15%)** | **[+0.00591, +0.00927]** | **8/8** | +0.000670 | 8/8 |
| 250 ms | | | *pending* | | | | |
| 500 ms | | | *pending* | | | | |
| 1 s | 0.02214 | 0.02419 | +0.00205 (+9%) | [−0.00038, +0.00422] | 7/8 | +0.000078 | 7/8 |

250 ms and 500 ms were still running when this was written; they land in
`outputs/passive_wf_v2/26_wave_vs_baseline.csv` and the run's report.

`M5_full` reproduces the 2026-09-13 number exactly (0.02214), so the
comparison is against the same baseline.

**Reading it.** Where the stage-1 response is strongest, 100 ms, the wave layer
improves on the full baseline on **every one of the 8 test days** and the
interval excludes zero. At 1 s, the tail of the response, the gain is a third
the size and not significant. That is the shape the brief predicted: `u_t`
captures the sub-second response body that the box-window features blur. For
scale, the spread/depth group moved 1 s correlation by +0.011 (M1 → M2).

A positive Δ is a predictive improvement only — it does not make anything
tradable (see *Stage 1b*).

---

# Work completed (2026-09-24)

## Repository pushed to GitHub

`data/` (40 files, 30 over 100 MB) was stripped from history with
`git filter-branch` and gitignored. That checkout deleted `data/` from the
working tree; it was restored from the local branch `backup/pre-strip-data`,
which still holds the original history. **Keep that branch** until the data is
backed up somewhere else.

## Passive simulator and gate

* Fills are charged to the pre-trade book (the bug above).
* `simulate_unwind`: pegged passive exit with a crossing timeout.
* `policy_gate`: every criterion picks its cell by day-clustered t and must
  clear a Bonferroni critical value over the family it searched
  (`passive.gate_family_alpha`); `filter_beats_unconditional` is a day-paired
  test; the verdict rests on `positive_after_exit_costs`; criteria can read
  `NOT_EVALUABLE`.
* The report section adds a *What does getting out cost?* table and picks
  cells by t rather than by the largest point estimate.

## Stage 2

`add_wave_features` / `_causal_kernel_filter` in `features.py` (exact, chunked,
checked against an O(n²) definition and added to the truncation leakage test),
the `wave` feature group, `M6_wave` in the ladder, and `wave_comparison` —
table `26_wave_vs_baseline`, scored at `evaluation.wave_horizons`.

## Feature cache

`evaluation.save_feature_frame` writes `features.parquet`; `run --features`
loads it and skips data prep.

20 tests added (**200 total**). Full run: 20 days, ~6 h to the maker and 1 s results.

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

`configs/passive_wf_intc.json`, 20 days, 4h55m wall clock. Its maker PASS was
later traced to a simulator bug and is void — see *Stage 1b*. The maker rebate is deliberately **not** set to a single value: the config
sweeps `[0, 0.0010, 0.0020, 0.0025, 0.0030]` and the gate reports a break-even,
so the output is a bracket rather than an invented venue number.

## Reporting bug confirmed fixed

`outputs/baseline_m5/` labels its uncertainty row `oos_pred_corr_M1` while
carrying M5_full's value — those outputs predate the current reporting layer.
The new run emits `oos_pred_corr_M5_full` correctly.

---

# Known issues

- **Queue position cannot be measured from MBP-10** — it is swept, and the
  bracket is the result.
- **Each fill is unwound independently.** No inventory limit, and a long fill
  is not netted against a later short one. Netting would cut crossing costs
  somewhat; it cannot close a −0.47 tick gap.
- **Taker fees on crossed exits are not charged**, which flatters every unwind
  number slightly.
- **Phase 0 prices fills at the previous decision row's quote** and so is not
  comparable to the event-level simulator; kept for its decile ordering only.
- **`07b_leave_one_group_out` cost 54 minutes** for one 7-row table and is the
  peak-memory stage. Off by default.
- One symbol only. `18_tick_regime` records the preregistration requirement:
  a large-tick name must be a **separate** replication, never pooled.
- `outputs/passive_wf/passive_tapes/` (556 MB) is still tracked from before the
  ignore rule; new runs' tapes are ignored.

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

1. **Push `u_t` where the response lives.** The brief below predicts the wave
   layer matters most at 100–500 ms; table `26` says whether it does. If it
   helps there, iterate on the kernel basis using the feature cache — a kernel
   change still needs event-resolution OFI, so it needs one prep run, but model
   and horizon changes do not.
2. **Significance, not direction.** With 8 test days the fold-bootstrap
   interval is wide. More test days (extend the data) is the cheapest way to
   tighten it.
3. **Tradability is closed on this symbol.** Reopen it only with a new signal or
   a large-tick replication; do not rerun the maker grid hoping for a sign.

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
