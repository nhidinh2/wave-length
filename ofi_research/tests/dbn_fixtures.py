"""Hand-built Databento ``mbp-10`` records for the vendor-ingestion tests.

Not a test module (pytest only collects ``test_*.py``). Everything here builds
raw records in the shape :func:`ofi_research.databento_loader.read_raw` would
produce, so the loader can be exercised without a real DBN file.

Prices are fixed-point int64 nanodollars, exactly as the vendor sends them —
converting them here would defeat the purpose of the price tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ofi_research.config import Config
from ofi_research.dbn import FIXED_PRICE_SCALE, F_LAST

#: 09:30:00 exchange-local on a normal (EST) trading day, as epoch nanoseconds.
SESSION_OPEN_NS = int(pd.Timestamp("2024-03-05T14:30:00Z").value)

MS = 1_000_000
SEC = 1_000 * MS

Level = Tuple[float, float]  # (price in dollars, size)


def fixed(dollars: float) -> int:
    """Dollars -> Databento fixed-point nanodollars."""
    return int(round(dollars * FIXED_PRICE_SCALE))


def record(*, ts_recv: int, action: str = "A", side: str = "B",
           price: float = 100.00, size: float = 10.0,
           flags: int = F_LAST, sequence: int = 1,
           bid: Sequence[Level] = ((99.99, 100.0),),
           ask: Sequence[Level] = ((100.01, 100.0),),
           ts_event: Optional[int] = None, symbol: str = "TEST",
           instrument_id: int = 1) -> Dict:
    """One raw mbp-10 record. ``bid``/``ask`` are best-first level ladders."""
    row: Dict = {
        "ts_recv": ts_recv,
        "ts_event": ts_recv - MS if ts_event is None else ts_event,
        "rtype": 10,
        "publisher_id": 2,
        "instrument_id": instrument_id,
        "action": action,
        "side": side,
        "depth": 0,
        "price": fixed(price),
        "size": size,
        "flags": flags,
        "sequence": sequence,
        "symbol": symbol,
    }
    for i, (px, sz) in enumerate(bid):
        row[f"bid_px_{i:02d}"] = fixed(px)
        row[f"bid_sz_{i:02d}"] = sz
    for i, (px, sz) in enumerate(ask):
        row[f"ask_px_{i:02d}"] = fixed(px)
        row[f"ask_sz_{i:02d}"] = sz
    return row


def raw_frame(records: List[Dict]) -> pd.DataFrame:
    """Assemble records the way ``read_raw`` does, including ``raw_row_id``."""
    df = pd.DataFrame(records)
    df["_source_file"] = "memory"
    df["_source_index"] = 0
    df["raw_row_id"] = np.arange(len(df), dtype="int64")
    return df


def write_csv(records: List[Dict], path) -> str:
    """Write records to a CSV that ``read_raw`` can ingest."""
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def databento_config() -> Config:
    """Config for the Databento path: fixed-point prices, deep-level patterns."""
    cfg = Config()
    cfg.data.price_mode = "fixed"
    cfg.columns.deep_bid_price = "bid_px_{i0:02d}"
    cfg.columns.deep_ask_price = "ask_px_{i0:02d}"
    cfg.columns.deep_bid_size = "bid_sz_{i0:02d}"
    cfg.columns.deep_ask_size = "ask_sz_{i0:02d}"
    return cfg
