Files from the 2026-08-11 run, moved here on 2026-08-12.

The 2026-08-12 re-run was killed by the OS at 06:21 before regenerating them,
so they do NOT correspond to the tables or the report now in the parent
directory. Kept rather than deleted, but any number in them predates the
reporting fixes and must not be quoted alongside the current report.

Regenerate by completing a full run:
  python3 -m ofi_research.run_experiment run --databento \
      --data data/pilot/*.parquet \
      --config ofi_research/configs/baseline_intc.json \
      --output ofi_research/outputs/baseline
