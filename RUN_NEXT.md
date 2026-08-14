# Relaunch after restart

The 2026-08-12 M5 run was killed by the OS during data prep (day 5 of 20).
Cause was machine-wide memory exhaustion, not a code fault:

| | |
|---|---|
| RAM | 17.2 GB, 66% in use |
| **Swap** | **13.6 GB used of 14.3 — 702 MB free** |
| Per-day working frame | 5.9–7.2 GB (largest day ~8.5 GB) |

A restart clears the swap. The pipeline completed all 20 days of prep earlier
the same night when the machine was less loaded, so this is a headroom problem
rather than a scaling one.

## Before relaunching

Close memory-heavy applications (browsers, IDEs, VMs). The run needs roughly
**9 GB of headroom** for the largest session, and it holds one day at a time —
memory does not accumulate across days.

Check you have room:

```bash
sysctl vm.swapusage          # want several GB free
memory_pressure | tail -3    # want free percentage well above 34%
```

## The command

```bash
cd /Users/nemo/Documents/GitHub/wave-length
python3 -W ignore::FutureWarning -m ofi_research.run_experiment run --databento \
    --data data/pilot/*.parquet \
    --config ofi_research/configs/baseline_intc.json \
    --output ofi_research/outputs/baseline_m5 \
    > /tmp/m5_run.log 2>&1 &
```

Writes to a **new** directory, so the existing `outputs/baseline/` report
survives if this run dies again.

Expect **~3 hours**: data prep dominates (~2h30m), the walk-forward is ~25 min.

Watch it:

```bash
tail -f /tmp/m5_run.log | grep -E "Saved table|Wrote report|Killed|Error"
```

## What is already configured

| setting | value | why |
|---|---|---|
| `evaluation.reference_model` | `M5_full` | the real baseline; M1 has negative OOS R² |
| `evaluation.run_leave_one_group_out` | `False` | cost 54 min for one 7-row table |
| `evaluation.run_cks_replication` | `False` | construction audit, not the model |

## If it dies again

Fallback that cuts ~14% of the frame (~1 GB/day). Roughly half the 249 columns
are targets — 13 horizons × ~9 columns. Dropping the two longest clock horizons
costs nothing analytically, because the horizon scan shows OFI's signal is
statistically dead beyond 5 s (5 s p=0.11, 10 s and 30 s negative and
insignificant).

Add to `ofi_research/configs/baseline_intc.json`:

```json
"targets": { "clock_horizons_ms": [100, 250, 500, 1000, 2000, 5000],
             "event_horizons": [10, 25, 50] }
```

## What this run will settle

1. **Whether M5 trades.** Its predictions are ~3× M1's, whose maximum was
   1.37 ticks against a 2.38-tick threshold. If it trades, the execution path
   runs end-to-end on real data for the first time — inspect it deliberately.
2. **Phase 0 on real INTC** — tables `20`–`22`, the front-of-queue passive
   markout by fill-aligned OFI. This decides whether market-by-order data is
   worth buying.

See `README.md` for full project status and the stage-2 brief.
