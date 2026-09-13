# ofi_research — Order-Flow Imbalance microstructure baseline

A reproducible, leakage-audited research pipeline that tests whether
**order-flow imbalance (OFI)** predicts **future midprice movement** out of
sample, and whether any edge survives **realistic execution**.

This stage deliberately stops at a linear microstructure baseline. It does
**not** implement the wave equation, Hamiltonian system, or Fokker–Planck PDE.
The code is modular so an impulse-response / Green's-function layer can be added
later without changing the feature, target, split, or backtest contracts.

The pipeline answers three **separate** questions and never conflates them:

| claim | means |
|---|---|
| Implementation validity | the code computes what it says |
| Predictive evidence | OFI forecasts future mid moves out of sample |
| Net trading evidence | that edge survives execution |

Predictive evidence does not imply net trading evidence, and neither follows
from implementation validity.

> **No data is bundled.** Point the loader at your own extract, or use
> `--synthetic` to prove the pipeline runs. The synthetic generator is a *test
> fixture*; every artifact it produces is stamped **SYNTHETIC — NOT A REAL
> RESULT**.

## Repository layout

```
ofi_research/
├── configs/
│   └── baseline_intc.json          # the reference run: INTC, 20 days, 200ms clock
├── outputs/
│   ├── baseline/                   # main experiment: tables 01–22, plots, report
│   └── pilot_audit/                # one-day raw-to-feature audit (protocol gate)
├── tests/                          # 149 tests — see "Tests" below
│
├── config.py                       # typed dataclass config: columns, windows,
│                                   #   horizons, costs, splits, execution
│
│   # ingestion
├── dbn.py                          # DBN flags, fixed-point price conversion (P0.6)
├── databento_loader.py             # mbp-10 raw records -> canonical native events
├── data_loader.py                  # generic CSV/parquet loader + synthetic generator
├── fetch_pilot_day.py              # CLI: download ONE day from Databento
├── validation.py                   # schema inspection, cleaning, removal report
│
│   # signal construction
├── features.py                     # midprice/spread/depth, OFI L1/L2, normalized,
│                                   #   signed volume, trade intensities
├── targets.py                      # future-mid targets, clock & event horizons
├── sampling.py                     # memory compaction + the DECISION CLOCK
│
│   # evaluation
├── splits.py                       # day-contiguous walk-forward, never shuffled
├── models.py                       # M0–M5 linear (train-only scaling, HAC t-stats)
├── costs.py                        # explicit entry+exit taker execution, depth walking
├── backtest.py                     # signal/position rules, latency, ledger, signal funnel
├── passive.py                      # Phase 0: front-of-queue passive markout screen
├── diagnostics.py                  # HAC correlations, deciles, regimes, FDR
├── plots.py                        # labeled, train/test-annotated figures
│
└── run_experiment.py               # orchestrator + CLI + report generation
```

## Install

```bash
pip install -r ofi_research/requirements.txt
```

pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, pytest.
Python 3.9+ (avoids `X | Y` union syntax).

## CLI

```bash
# 0. Look at your schema first — no modeling, no guessing.
python -m ofi_research.run_experiment inspect --data book.parquet

# 1. Buy and audit ONE day before a multi-day purchase.
export DATABENTO_API_KEY=db-...
python -m ofi_research.fetch_pilot_day --symbol INTC --date 2026-08-05
python -m ofi_research.run_experiment pilot --data data/pilot/INTC_20260805.parquet

# 2. The full walk-forward study.
python -m ofi_research.run_experiment run --databento \
    --data data/pilot/*.parquet \
    --config ofi_research/configs/baseline_intc.json \
    --output ofi_research/outputs/baseline

# 3. Passive screen only (skips the walk-forward).
python -m ofi_research.run_experiment phase0 --databento \
    --data data/pilot/*.parquet \
    --config ofi_research/configs/baseline_intc.json \
    --output ofi_research/outputs/phase0

# Prove it runs end to end on labeled synthetic data:
python -m ofi_research.run_experiment run --synthetic --synthetic-days 15

# Tests:
python -m ofi_research.run_experiment test        # or: pytest ofi_research/tests
```

On a 20-day single-symbol study, feature preparation dominates runtime
(~2h30m); the walk-forward itself is ~25 minutes.

## Column mapping

The pipeline never guesses essential columns. Map them in a JSON config:

```json
{
  "columns": {
    "timestamp": "ts", "instrument": "symbol",
    "bid_price_1": "bp1", "ask_price_1": "ap1",
    "bid_size_1": "bs1", "ask_size_1": "as1",
    "trade_price": "tp", "trade_size": "tsz", "trade_side": "side"
  },
  "data": { "timestamp_is_epoch": true, "timestamp_unit": "ns" }
}
```

Required: `timestamp`, `bid_price_1`, `ask_price_1`, `bid_size_1`,
`ask_size_1`. Optional families disable themselves (and say so) when absent:
L2 columns → L2 OFI; trade columns → signed volume (falling back to a causal
Lee–Ready classifier when `trade_side` is missing); `event_type` → intensity
imbalance. `--databento` applies the mbp-10 mapping automatically.

## Leakage controls

Enforced and tested in `tests/test_no_leakage.py`:

- **Truncation causality** — rebuilding features on a strict prefix reproduces
  identical values for retained rows. The strongest single proof nothing peeks
  forward.
- Targets use only rows with `timestamp > t`, never crossing a segment.
- Rolling normalizations are trailing (right-closed), never centered.
- Scalers, winsorization limits, and decile edges are fit on **train only**.
- Hyperparameters and z-thresholds are chosen on **validation only**; the test
  fold is evaluated once.
- No shuffled split, and no future-day information in a prior-day model.

Execution adds its own causality rules (`tests/test_execution.py`): a fill is
never booked before its signal, nor against a record sharing the signal's
`ts_recv` — every record in one captured packet became available at the same
instant (P0.5).

## Outputs

| table | contents |
|---|---|
| `01`, `01b` | data quality; what cleaning removed and why |
| `02`, `02b` | feature summary; realized vs requested horizons |
| `03` | OFI correlation by horizon (HAC, FDR-adjusted) |
| `04` | decile response + monotonicity |
| `05` | **per-fold walk-forward — the core evidence file** |
| `06`, `07`, `07b` | model comparison, ablation, leave-one-group-out |
| `08` | regime results (only written when trades exist) |
| `09` | coefficient stability across folds |
| `10`, `10b` | gross vs net P&L; per-day P&L |
| `11`, `12` | transaction-cost and latency sensitivity |
| `13` | **day-blocked bootstrap — the interval IS the result** |
| `14` | machine-readable summary of the decision gate |
| `15` | contemporaneous CKS replication (sanity check, NOT predictive) |
| `16`, `17` | L2 aggregation; normalization selection (validation only) |
| `18` | tick regime — small/large-tick classification and preregistration note |
| `19` | **execution accounting — the signal funnel** |
| `20`–`22` | Phase 0 passive markout: unconditional, by OFI decile, by side |

Plus labeled `p_*.png` plots and the decision report
`OFI_RESEARCH_REPORT.md`.

### Reading the report

The decision gate is **three-valued**: `PASS`, `FAIL`, `NOT_EVALUABLE`. A
criterion that was never exercised is not a failure. If the policy takes no
trades, the P&L criteria are `NOT_EVALUABLE` and every cost and latency figure
is **undefined, not zero** — absence of trading evidence is not evidence of
unprofitability.

Table `19` makes that legible by printing the whole funnel: rows with a usable
prediction → clearing the cost gate → clearing the z gate → signals → fills,
plus the mean forecast as a fraction of the round-trip cost it must clear.

Effect sizes are reported in **ticks**, not just correlation. Because models
standardize on the training fold, `beta_1` is already "price move per 1-SD of
OFI" and converts directly into the unit the spread is quoted in — the only
comparison that decides tradability.

## Statistical discipline

Overlapping targets ⇒ **Newey–West (HAC)** standard errors everywhere; iid SEs
are never trusted, and headline numbers carry a whole-day block bootstrap with
day-clustered SEs. Many horizon/window combinations ⇒ **FDR-adjusted**
(Benjamini–Hochberg) p-values, treated as exploratory. **Conclusions come from
out-of-sample walk-forward performance, not in-sample p-values.**

Queue position cannot be modeled from top-of-book data, so aggressive taker
execution is assumed throughout and reported as a limitation. XNAS.ITCH is a
venue-**local** book: its mid and spread are not the national NBBO.

Statistical and economic significance are reported separately. A correlation
whose interval excludes zero can still imply a predicted move far below one
tick.

## Phase 0 — the passive screen

`passive.py` bounds market-making performance using the MBP-10 data already on
disk, assuming our order sits at the **front of the queue** and so fills on
every trade at our price. Real queue position is always worse, which makes the
bound decisive in one direction: if it is unprofitable, every realistic queue
position is too, and no market-by-order data need be purchased.

Fills are conditioned on **fill-aligned OFI** — OFI signed by the position the
fill leaves us in. Adverse selection is symmetric (negative OFI leaves us long
into falling prices, positive leaves us short into rising ones), so markout
against *raw* OFI is U-shaped and its monotonicity reads ~0 even when OFI
predicts toxicity perfectly. Tables `21b`/`21c` keep the raw and |OFI| views for
comparison.

Set `costs.maker_rebate_per_unit` from the venue schedule before quoting any
passive number: at a $0.01 tick a typical add rebate is 0.2–0.3 ticks, which is
large relative to the edges being measured.

## Tests

149 tests, ~6 seconds. The failure mode this codebase guards against is not a
crash — it is a plausible-looking wrong number in a report you act on.

| file | protects |
|---|---|
| `test_no_leakage` | future information reaching a feature |
| `test_prices` | the converted-exactly-once assertion (P0.6) |
| `test_execution` | fills booked before their signal or inside its packet |
| `test_targets` | target causality across clock and event horizons |
| `test_ofi` | OFI increments against hand-computed books |
| `test_databento_events` | `F_LAST` event grouping and ingestion duplicates |
| `test_segmentation` | chain breaks at integrity boundaries |
| `test_sampling` | that the decision clock does not corrupt features |
| `test_costs` | spread-crossing and depth-walking arithmetic |
| `test_pilot_findings` | vendor assumptions in `PILOT_ASSUMPTIONS` |
| `test_passive` | fill-side conventions and pre-fill causality |

## Extension point: impulse-response / Green's-function layer

The signal enters the backtest as a single `predictions` array aligned to rows
(`backtest.run_backtest`). To add a response-kernel model:

1. Map the OFI/event increment series to a predicted midprice-response path via
   a causal kernel `G`, producing the same `predictions` contract.
2. Reuse `splits`, `costs`, `backtest`, `diagnostics`, and the report unchanged.

Proceed only if the decision gate passes on all six criteria: stable OFI
coefficient sign, consistently nonzero OOS correlation, economically sensible
deciles, net-positive after realistic costs, not dominated by one day, and
survival of latency/slippage stress. `NOT_EVALUABLE` is not a pass.
