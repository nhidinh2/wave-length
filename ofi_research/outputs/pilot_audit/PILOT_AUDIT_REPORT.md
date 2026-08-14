# One-Day Databento Pilot Audit

> **Not a result.** This report exists to falsify the vendor assumptions below against real records before more data is bought. No profitability claim can be made from one day.

- Files: `['data/pilot/INTC_20260805.parquet']`
- Canonical rows: 3500096
- Raw records: 3636579
- Records per event (mean/max): 1.04/16
- Events spanning >1 sequence: 36703
- Book clears: 0  |  inferred halts: 0
- Trades with unspecified aggressor side: 37847
- Inferred quote increment: 0.010000 (configured 0.01)

## Assumptions this day must confirm

- [ ] F_LAST (128) marks the final record of one captured ts_recv PACKET, which may span several `sequence` values. CONFIRMED on XNAS.ITCH INTC 2026-08-05: max records/event (16) equals max ts_recv group size (16), no two canonical events share a ts_recv, and 36,703 events span >1 sequence while sequence groups never exceed 2 records. Re-confirm per dataset.
- [ ] Records of one native event are CONSECUTIVE in native file order.
- [ ] action=='T' records carry the AGGRESSOR side in `side` ('B' buy, 'A' sell).
- [x] action=='T' does not itself change the displayed book state. Only PART of the assumed Trade->Fill/Cancel pairing was observed on INTC 2026-08-05: 69,079 'TC' groups vs 99,464 standalone 'T' records, so 59% of trades carry no same-sequence book update. Aggregating at packet level is what makes this harmless — do not narrow it back to sequence level.
- [ ] The last record of an event carries the post-event book for all levels.
- [ ] Prices are fixed-point int64 nanodollars unless price_mode='float'.
- [ ] UNDEF_PRICE (INT64_MAX) marks an absent price level.
- [ ] action=='R' clears the book and invalidates cross-event differencing.
- [ ] Records sharing a ts_recv were captured in ONE packet, so none of them can be executed against another (execution.no_fill_within_signal_batch).
- [ ] A halt/auction reopen is INFERRED from a one-sided book lasting at least integrity.halt_min_outage_ms; mbp-10 carries no halt action. Verify the inferred count against the day's known halts.
- [ ] Auction/cross prints are not separately identifiable in mbp-10; they are handled only by the continuous-session filter. Verify against the day's opening/closing cross.



## Feature availability notes

- Signed volume taken from vendor `signed_trade_volume` (net per event); 72.28% of trades resolved from the vendor aggressor field, 22.02% inferred via 'lee_ready', 5.70% unresolved.



## Manual checks that a human must still perform

- [ ] Read `pilot_event_trace.csv`: for at least 100 consecutive native events (several containing trades) confirm record grouping, the post-event book state, the OFI contribution, signed trade volume, and target indices.
- [ ] Confirm `pilot_audit_consecutive_actions.csv` patterns match the assumed Trade -> Fill/Cancel normalization.
- [ ] Confirm the inferred halt count against the day's known halts.
- [ ] Confirm opening/closing cross handling at the session edges.

Only after these pass should 25-30 days be ingested.