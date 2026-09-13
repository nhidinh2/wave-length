"""Download ONE trading day of Databento ``mbp-10`` for the pilot audit.

Deliberately one day: the pilot protocol exists to falsify the vendor
assumptions in :data:`ofi_research.databento_loader.PILOT_ASSUMPTIONS` against
real records *before* a multi-day purchase. Buying 25-30 days first only makes
a wrong assumption more expensive.

Usage::

    export DATABENTO_API_KEY=db-...
    python3 -m ofi_research.fetch_pilot_day --symbol INTC --date 2026-08-05

Writes ``<symbol>_<date>.dbn.zst`` (the raw vendor archive, never modified —
re-deriving the tabular file is free, re-buying the data is not) and
``<symbol>_<date>.parquet`` (what the loader reads). Parquet over CSV because a
liquid Nasdaq name produces millions of records per session, which as CSV is
gigabytes of text that pandas widens further with object dtypes.

Prices are written as fixed-point int64 nanodollars (``pretty_px=False``) to
match the ``--databento`` path's ``price_mode='fixed'``. A dollar-valued export
would be caught by :func:`ofi_research.dbn.assert_prices_converted_once` rather
than silently misread, but there is no reason to trip it on purpose.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("ofi_research.fetch")

#: Exchange-local continuous session, padded so the open/close auctions and the
#: session edges are visible in the audit rather than trimmed away.
LOCAL_START = "09:25"
LOCAL_END = "16:05"
EXCHANGE_TZ = "America/New_York"


def utc_window(date_str: str, local_start: str = LOCAL_START,
               local_end: str = LOCAL_END) -> tuple:
    """Exchange-local session window -> UTC ISO strings.

    Derived from the exchange calendar rather than hardcoded, so the request
    does not silently shift by an hour across a DST boundary — the same reason
    ``session_date`` is computed in exchange-local time (P0.7).
    """
    import pandas as pd

    def to_utc(hhmm: str) -> str:
        local = pd.Timestamp(f"{date_str} {hhmm}", tz=EXCHANGE_TZ)
        return local.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")

    return to_utc(local_start), to_utc(local_end)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbol", required=True, help="e.g. INTC")
    p.add_argument("--date", required=True,
                   help="one trading day, YYYY-MM-DD (avoid half-days)")
    p.add_argument("--dataset", default="XNAS.ITCH")
    p.add_argument("--schema", default="mbp-10")
    p.add_argument("--outdir", default="data/pilot")
    p.add_argument("--format", choices=("parquet", "csv"), default="parquet",
                   help="loader input format (parquet is far smaller)")
    p.add_argument("--local-start", default=LOCAL_START,
                   help=f"exchange-local start, default {LOCAL_START}")
    p.add_argument("--local-end", default=LOCAL_END,
                   help=f"exchange-local end, default {LOCAL_END}")
    p.add_argument("--estimate-only", action="store_true",
                   help="print the cost estimate and exit without buying")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    if not os.environ.get("DATABENTO_API_KEY"):
        logger.error("DATABENTO_API_KEY is not set.")
        return 2

    import databento as db

    start, end = utc_window(args.date, args.local_start, args.local_end)
    logger.info("%s %s %s  |  %s-%s %s -> %s..%s UTC", args.dataset,
                args.symbol, args.schema, args.local_start, args.local_end,
                EXCHANGE_TZ, start, end)

    client = db.Historical()
    request = dict(dataset=args.dataset, symbols=args.symbol,
                   schema=args.schema, stype_in="raw_symbol",
                   start=start, end=end)

    cost = client.metadata.get_cost(**request)
    size = client.metadata.get_record_count(**request)
    # an mbp-10 record is 368 bytes uncompressed; billing is per uncompressed GB
    logger.info("Estimated cost $%.4f for %s records (~%.2f GB uncompressed)",
                float(cost), f"{size:,}", size * 368 / 1e9)
    if args.estimate_only:
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.symbol}_{args.date.replace('-', '')}"

    data = client.timeseries.get_range(**request)

    raw_path = outdir / f"{stem}.dbn.zst"
    data.to_file(raw_path)
    logger.info("Raw archive: %s (%.1f MB)", raw_path,
                raw_path.stat().st_size / 1e6)

    # fixed-point int64 nanodollars either way, matching price_mode='fixed'
    if args.format == "parquet":
        out_path = outdir / f"{stem}.parquet"
        data.to_parquet(out_path, price_type="fixed")
    else:
        out_path = outdir / f"{stem}.csv"
        data.to_csv(out_path, pretty_px=False)
    logger.info("Loader input: %s (%.1f MB)", out_path,
                out_path.stat().st_size / 1e6)

    print(f"\nNext:\n  python3 -m ofi_research.run_experiment pilot "
          f"--data {out_path} --output ofi_research/outputs/pilot\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
