"""Configuration objects for the OFI research pipeline.

Everything that is a *choice* (column names, window sizes, horizons, cost
assumptions, split geometry, regime boundaries) lives here as a typed,
serializable dataclass. No magic numbers should be buried inside the analysis
modules; they should read from a ``Config`` instance.

Python 3.9 compatible: uses ``typing.Optional`` / ``typing.List`` rather than
``X | Y`` union syntax.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Column mapping ---
@dataclass
class ColumnMapping:
    """Map canonical field names -> column names in the raw dataset.

    Only ``timestamp``, the L1 price/size fields are strictly required for a
    minimal OFI run. Everything else is optional; when a source column is
    ``None`` (unmapped / not present) the dependent feature family is disabled
    and reported, never silently faked.
    """

    # required
    timestamp: str = "timestamp"
    bid_price_1: str = "bid_price_1"
    ask_price_1: str = "ask_price_1"
    bid_size_1: str = "bid_size_1"
    ask_size_1: str = "ask_size_1"

    # optional identity / ordering
    instrument: Optional[str] = "instrument"
    sequence: Optional[str] = "sequence"

    # optional level 2
    bid_price_2: Optional[str] = "bid_price_2"
    ask_price_2: Optional[str] = "ask_price_2"
    bid_size_2: Optional[str] = "bid_size_2"
    ask_size_2: Optional[str] = "ask_size_2"

    # optional trades
    trade_price: Optional[str] = "trade_price"
    trade_size: Optional[str] = "trade_size"
    trade_side: Optional[str] = "trade_side"  # 'B'/'S' or +1/-1
    event_type: Optional[str] = "event_type"

    # Pattern-based mapping for DEEP levels, used for taker execution only
    # (P0.9). ``{i0}`` is the 0-based level index, ``{i1}`` the 1-based one.
    # Databento mbp-10: "bid_px_{i0:02d}" etc. Signals still use levels 1-2.
    deep_bid_price: Optional[str] = None
    deep_ask_price: Optional[str] = None
    deep_bid_size: Optional[str] = None
    deep_ask_size: Optional[str] = None
    n_depth_levels: int = 10

    def has_depth_pattern(self) -> bool:
        return all(p is not None for p in (self.deep_bid_price,
                                           self.deep_ask_price,
                                           self.deep_bid_size,
                                           self.deep_ask_size))

    def depth_source(self, field_pattern: str, level_1_based: int) -> str:
        """Resolve a deep-level source column name for a 1-based level."""
        return field_pattern.format(i0=level_1_based - 1, i1=level_1_based)

    # canonical -> is-required flags used by validation
    REQUIRED_FIELDS = (
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "bid_size_1",
        "ask_size_1",
    )

    def required_source_columns(self) -> List[str]:
        return [getattr(self, f) for f in self.REQUIRED_FIELDS]

    def has_level2(self) -> bool:
        return all(
            getattr(self, f) is not None
            for f in ("bid_price_2", "ask_price_2", "bid_size_2", "ask_size_2")
        )

    def has_trades(self) -> bool:
        return self.trade_size is not None

    def has_trade_side(self) -> bool:
        return self.trade_side is not None

    def has_event_type(self) -> bool:
        return self.event_type is not None


# --- Data handling / cleaning ---
@dataclass
class DataConfig:
    data_path: str = "<REPLACE_WITH_DATA_PATH>"
    file_format: str = "auto"  # auto|csv|parquet|dbn
    timestamp_unit: Optional[str] = None  # None=let pandas parse; or 'ns'/'us'/'ms'/'s'
    timestamp_is_epoch: bool = False
    timezone: str = "UTC"

    # ---------------- session / day handling ----------------
    session_boundary: str = "calendar_day"  # 'calendar_day' or 'none'
    drop_cross_day_features: bool = True

    # P0.7: session_date is derived in EXCHANGE-LOCAL time, not the UTC calendar
    # date, so an early close or DST shift cannot split or merge a trading day.
    session_timezone: str = "America/New_York"
    # Continuous-trading window, exchange-local. Extended hours is a SEPARATE
    # experiment, not a wider default.
    continuous_session_start: Optional[str] = "09:30"
    continuous_session_end: Optional[str] = "16:00"
    restrict_to_continuous_session: bool = True

    # A quiet market is NOT a data fault: a gap is recorded as a feature
    # (gap_ms, long_quiet_gap), never breaking the OFI chain unless you have
    # evidence that gaps this size mean feed loss.
    gap_reset_ms: float = 1000.0        # labeling threshold for long_quiet_gap
    use_gap_as_integrity_break: bool = False

    # ---------------- price representation (P0.6) ----------------
    # 'float'  -> prices already in dollars; NEVER divided again
    # 'fixed'  -> prices are integer scaled by ``price_scale``; dollar columns
    #             are derived explicitly and quote comparisons stay integral
    price_mode: str = "float"
    price_scale: int = 1_000_000_000  # Databento fixed-point: 1e-9 dollars
    # Databento sentinel for "no price"; must become NaN before arithmetic.
    undef_price_sentinel: int = 9_223_372_036_854_775_807
    # Plausible dollar range used to catch accidental double conversion.
    plausible_price_min: float = 0.01
    plausible_price_max: float = 100_000.0
    # Untouched fixed-point integers kept alongside dollars (suffix ``_int``) so
    # quote comparisons stay integral. Only used when price_mode == 'fixed'.
    retain_fixed_price_integers: bool = True

    # P0.5. F_BAD_TS_RECV marks an untrusted ts_recv; ALWAYS reported. Excluding
    # is opt-in because dropping records also breaks the book-state chain.
    exclude_bad_ts_recv: bool = False

    # ---------------- cleaning toggles ----------------
    drop_crossed_books: bool = True   # bid > ask
    drop_locked_books: bool = False   # bid == ask (kept by default; flagged)
    drop_nonpositive_sizes: bool = True
    stale_quote_max_repeat: Optional[int] = None  # drop runs longer than this if set

    # Open/close trimming (minutes from session start/end); None = keep all.
    # Non-zero by default: mbp-10 carries the opening/closing crosses inline and
    # does not label them. On XNAS.ITCH INTC 2026-08-05 the session's final
    # second printed 5.48M shares (28% of the day's 19.2M) as one auction burst.
    # Left in, those prints dominate every volume feature at the session edges
    # via a mechanism the continuous-trading OFI model does not describe.
    # clean_data breaks the OFI chain at the boundary rather than differencing a
    # continuous quote against an auction print. None = study auctions.
    trim_open_minutes: Optional[float] = 1.0
    trim_close_minutes: Optional[float] = 1.0

    epsilon: float = 1e-12


# --- Data-integrity segmentation (P0.3 / P0.4) ---
@dataclass
class IntegrityConfig:
    """What constitutes a HARD break in the observable book-state chain.

    A hard break forces the next retained observation to start a new segment,
    so its OFI increment is zero and no rolling window or target spans the
    discontinuity. Ordinary market inactivity is deliberately NOT in this list.
    """

    break_on_instrument_change: bool = True
    break_on_session_change: bool = True
    break_on_book_clear: bool = True        # Databento action 'R'
    break_on_maybe_bad_book: bool = True    # F_MAYBE_BAD_BOOK
    break_on_snapshot: bool = True          # F_SNAPSHOT restates, not deltas
    break_on_invalid_book: bool = True      # missing/crossed/nonpositive state
    break_after_nonbenign_removal: bool = True
    # A trading halt resumes with a book that has no differencing relationship
    # to the pre-halt book. Set by the loader as ``halt_boundary``.
    break_on_halt: bool = True
    # HEURISTIC — check against a one-day sample. mbp-10 has no halt action, so
    # a halt is inferred from a quote outage (one side absent this long, then
    # restored). Conservative by design: it can only ADD breaks, so a false
    # positive costs a little data and a false negative is already covered by
    # the book-clear / snapshot / invalid-book breaks.
    halt_min_outage_ms: float = 5000.0

    # Labeling only — never a break by itself.
    long_quiet_gap_ms: float = 1000.0
    # Optional staleness cap for EVENT-count windows: if an N-event window spans
    # more clock time than this, the aggregate is marked stale (not dropped).
    max_event_window_age_ms: Optional[float] = None


# --- Latency composition (P0.5) ---
@dataclass
class LatencyConfig:
    """Total decision-to-execution latency, decomposed and named honestly.

    ``ts_recv`` is when DATABENTO's capture server received the packet, not
    when your process did. Zero total latency is therefore a theoretical
    upper bound, never an achievable live baseline.

    The defaults are left at zero DELIBERATELY. Any non-zero default would be
    a number nobody measured, silently baked into every P&L figure. Zero is at
    least honestly wrong in a known direction — it is the optimistic bound —
    and :meth:`is_upper_bound` exists so that a run which never configured it
    says so out loud instead of presenting the bound as a forecast.
    """

    capture_to_user_ms: float = 0.0   # Databento capture -> your process
    decode_decide_ms: float = 0.0     # your decode + feature + model time
    user_to_venue_ms: float = 0.0     # your order -> matching engine

    def total_ms(self) -> float:
        return (self.capture_to_user_ms + self.decode_decide_ms
                + self.user_to_venue_ms)

    def is_upper_bound(self) -> bool:
        """True when no latency was configured, so results are the bound."""
        return self.total_ms() <= 0.0


# --- Aggressive-taker execution (P0.9) ---
@dataclass
class ExecutionConfig:
    """Taker execution against displayed depth.

    Queue position is irrelevant for a taker, but displayed size and price
    walking are not: an order larger than L1 consumes deeper levels at worse
    prices.
    """

    depth_levels: int = 10          # levels available for walking
    max_participation: float = 0.25  # cap size at this fraction of displayed
    allow_partial_fill: bool = True  # else reject when depth is insufficient
    routing_venue: str = "XNAS"      # reported; XNAS.ITCH is a LOCAL book

    # Records sharing a ts_recv arrived in ONE captured packet (P0.5), so a
    # decision from any of them cannot execute against another member of the
    # same batch. Set: earliest fill is the first STRICTLY later ts_recv.
    # Unset permits intra-batch fills — not defensible for a live claim, and
    # exists only to quantify what the assumption is worth.
    no_fill_within_signal_batch: bool = True


# --- Features ---
@dataclass
class FeatureConfig:
    # aggregation windows
    event_windows: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    time_windows_ms: List[float] = field(
        default_factory=lambda: [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    )

    # L1:L2 combination weights (NOT tuned on test set)
    l2_weights: Dict[str, float] = field(
        default_factory=lambda: {"w1": 1.0, "w2": 0.5}
    )

    # trailing normalization window (in events) for zOFI and trailing vol
    trailing_stat_window: int = 500

    # floor on depth denominators so a depleted queue cannot produce a
    # near-infinite normalized OFI (P1.C)
    depth_denominator_floor: float = 1.0

    # winsorization quantiles (estimated on training window only)
    winsor_lower_q: float = 0.005
    winsor_upper_q: float = 0.995

    # trade classification when trade_side missing
    trade_classifier: str = "lee_ready"  # 'quote' | 'tick' | 'lee_ready'

    tick_size: float = 0.01


# --- Targets ---
@dataclass
class TargetConfig:
    clock_horizons_ms: List[float] = field(
        default_factory=lambda: [100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0,
                                 10000.0, 30000.0]
    )
    event_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 25, 50])

    # a clock target is accepted only if realized horizon is within tolerance
    horizon_tolerance_frac: float = 0.5  # |realized-H| <= tol*H
    horizon_abs_tolerance_ms: float = 50.0

    # 'unchanged' band for 3-class label, in ticks
    unchanged_band_ticks: float = 0.5

    use_log_return: bool = False  # primary target uses raw mid change


# --- Splits / walk-forward ---
@dataclass
class SplitConfig:
    scheme: str = "rolling"  # 'rolling' | 'expanding'
    train_days: int = 10
    validation_days: int = 2
    test_days: int = 1
    step_days: int = 1
    min_train_days: int = 5  # for expanding


# --- Costs ---
@dataclass
class CostConfig:
    fee_per_unit: float = 0.0
    fee_bps: float = 0.0            # per side, in basis points of notional
    slippage_ticks: float = 0.0    # additional adverse ticks per side
    tick_size: float = 0.01
    # Legacy/extra latency term, kept so sensitivity sweeps can vary a single
    # scalar. TOTAL latency = latency_ms + latency.total_ms(); see P0.5.
    latency_ms: float = 0.0
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    impact_coefficient: float = 0.0  # linear impact per unit size
    minimum_edge_buffer: float = 0.0  # extra edge (price units) required to trade
    # Maker rebate paid TO the liquidity provider, price units per share, as a
    # POSITIVE number; distinct from `fee_per_unit`, the taker charge. Zero by
    # default (inventing a fee schedule is worse than reporting no-rebate), but
    # that understates a maker's edge by a material fraction of a tick, so set
    # it from the venue schedule before quoting passive P&L.
    maker_rebate_per_unit: float = 0.0

    def total_latency_ms(self) -> float:
        """Full simulated decision-to-execution delay, in milliseconds."""
        return float(self.latency_ms) + self.latency.total_ms()


# --- Evaluation / uncertainty (overlapping horizons) ---
@dataclass
class EvaluationConfig:
    """How uncertainty is estimated when target windows overlap.

    Rows whose target windows overlap are not independent draws, so an iid
    standard error understates the true sampling error — often by a large
    factor at long horizons on high-frequency data. Inference is therefore
    reported at the level of whole trading days, which are the largest blocks
    the design can treat as approximately independent.
    """

    #: Resample whole trading days with replacement (moving-block bootstrap
    #: with the block = one session).
    n_day_bootstrap: int = 1000
    bootstrap_seed: int = 11
    #: Report day-clustered standard errors alongside the bootstrap.
    report_day_clustered: bool = True

    #: The model every headline number is reported for: the verdict, effect
    #: size, decision gate, day-blocked interval, and the ledger the backtest
    #: runs on. Defaults to the FULL specification rather than OFI-alone,
    #: because a bare OFI model is the weakest rung of the ladder (negative
    #: out-of-sample R^2 on INTC) and reporting it as the headline understates
    #: the baseline that any later wave/PDE term must actually beat.
    reference_model: str = "M5_full"

    #: The contemporaneous CKS regression is a data/formula sanity check, not a
    #: strategy and not part of the model. Off by default so no run spends
    #: effort on it or invites confusion with the predictive result; set True
    #: when specifically auditing OFI construction.
    run_cks_replication: bool = False

    #: Leave-one-group-out is a SECOND full walk-forward over 7 more models. On
    #: a 20-day INTC study it cost 54 minutes to produce one 7-row table and was
    #: the peak-memory stage that got a run OOM-killed. Off by default; turn it
    #: on deliberately when the attribution of a specific feature group is the
    #: question being asked.
    run_leave_one_group_out: bool = False

    #: The aggressive-taker execution study (tables 10-12, 19). Settled and
    #: negative on INTC: a 1-SD OFI move predicts ~0.015 ticks against a
    #: 2.53-tick spread, so ~174 SD would be needed to cover one round trip,
    #: and the 20-day M5 run executed 1 trade from 857,967 eligible rows. That
    #: is a conclusion, not a bug, and re-deriving it costs ~25 minutes per
    #: run. Off by default; the strategy question moved to the PASSIVE path,
    #: where the spread is earned rather than paid. Turn it back on when a
    #: feature change could plausibly overturn the taker verdict.
    run_taker_backtest: bool = False

    #: Trades the ledger needs before the P&L criteria (``net_positive``,
    #: ``not_one_day``, ``survives_stress``) may read PASS or FAIL rather than
    #: NOT_EVALUABLE. A ledger of one trade is an anecdote: the 20-day M5 run
    #: executed a single trade, won 4 ticks on it, and the gate duly reported
    #: "net-positive out of sample" beside a concentration FAIL saying that all
    #: of the P&L came from one day. Both statements were about the same trade.
    #: Zero restores the old behaviour, where any non-empty ledger is scored.
    min_trades_for_pnl_gate: int = 30

    #: Distinct trading days the ledger must span before those same criteria
    #: are scored. Overlapping targets make rows within a day dependent, so
    #: every interval in the report is day-blocked; a one-day ledger therefore
    #: carries one effective observation no matter how many trades it holds.
    #: ``not_one_day`` in particular cannot discriminate below two days — it
    #: reads 1.000 by construction, which is arithmetic rather than evidence.
    min_trade_days_for_pnl_gate: int = 2


# --- Passive (maker) evaluation ---
@dataclass
class PassiveConfig:
    """Walk-forward passive quoting, gated by a model prediction.

    Phase 0 established the shape of the problem on INTC: an unconditional
    front-of-queue maker LOSES 0.054 ticks per fill at ms1000, while fills
    sorted by fill-aligned OFI run from -0.150 (worst decile) to +0.089 (best),
    a 0.183-tick spread with Spearman 0.879. So the edge is not in quoting, it
    is in NOT quoting into toxic flow — which makes the filter the strategy.

    Three things separate this from that screen, and each can only move the
    result in the pessimistic direction:

    * the gate threshold is fitted on TRAIN days and applied to held-out days,
      instead of decile edges cut over the whole sample;
    * queue position is a dial (:attr:`queue_ahead_fractions`) rather than the
      free front-of-queue assumption;
    * quoting and cancelling both pay latency, so a withdrawal decision leaves
      us exposed for the round trip rather than taking effect instantly.
    """

    #: Fraction of displayed depth at our price assumed to sit AHEAD of us in
    #: the queue. 0.0 reproduces the Phase-0 front-of-queue ceiling; 1.0 is
    #: joining behind every displayed share (the pessimistic end); 0.5 is the
    #: naive midpoint. MBP-10 shows aggregate size at a level and never our own
    #: position in it, so this cannot be measured from the data on disk — it is
    #: swept, and the honest result is the bracket, not any single value.
    queue_ahead_fractions: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 1.0])

    #: Maker rebates to sweep, price units per share (POSITIVE = paid to us).
    #: Defaults span no-rebate through a typical US equity add tier at a $0.01
    #: tick: 0.0020-0.0030 per share is 0.20-0.30 ticks, which is larger than
    #: the entire measured decile spread and therefore decides the business.
    #: These are ILLUSTRATIVE tiers, not a venue schedule — replace them with
    #: the real one before quoting a number to anybody.
    rebate_sweep: List[float] = field(
        default_factory=lambda: [0.0, 0.0010, 0.0020, 0.0025, 0.0030])

    #: Quote only when the fill-aligned prediction sits above this quantile of
    #: its TRAIN-day distribution. 0.0 quotes always (the unconditional
    #: business); 0.5 quotes on the better half of expected flow; 0.9 is the
    #: highly selective end that trades fill count for fill quality.
    gate_quantiles: List[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9])

    #: Cancels are not free either: a withdrawal decided at t reaches the venue
    #: at t + this, and any trade in between still fills us. Defaults to the
    #: same total latency as the taker path when None.
    cancel_latency_ms: Optional[float] = None

    #: Assume queue-ahead shares that CANCEL leave from behind us (True) or
    #: in front of us (False). True is pessimistic and is the right default:
    #: with aggregate depth we cannot tell which, and assuming cancels help us
    #: is exactly the kind of free lunch that makes a maker backtest lie.
    cancels_leave_from_behind: bool = True

    #: Minimum fills a (queue, rebate, gate) cell needs before its markout is
    #: reported. Selective gates at deep queue positions can fill so rarely
    #: that the mean is one day's noise.
    min_fills_per_cell: int = 200


# --- Memory footprint / decision clock ---
@dataclass
class SamplingConfig:
    """How much of a real month has to fit in memory at once.

    One session of a high-message-rate name is ~3.5M canonical events and ~5 GB
    of features; the walk-forward holds train+validation+test days resident
    together. The first three switches are pure representation and change no
    result. ``decision_interval_ms`` is different in kind — it defines when the
    strategy may act, and belongs in the write-up.
    """

    #: Retain one row per interval AFTER features and targets are computed on
    #: every event. None = act on every event (exact, but memory-hungry).
    #: This is a strategy parameter, not a performance knob: see
    #: :func:`ofi_research.sampling.sample_decision_rows`.
    decision_interval_ms: Optional[float] = None

    prune_columns: bool = True
    downcast_float32: bool = True
    categorical_strings: bool = True
    #: The loader keeps fixed-point integer copies of every price so that the
    #: converted-exactly-once assertion can run (P0.6). That proof completes at
    #: load; keeping them through the pipeline costs ~0.6 GB per session.
    drop_fixed_price_integers: bool = True


# --- Tick-regime replication (P1.E) ---
@dataclass
class TickRegimeConfig:
    """Classify a symbol as large-tick or small-tick from quoted spread.

    The regime must be measured on TRAINING data (or a fixed prior reference
    period) and preregistered — never chosen after seeing performance. Large-
    and small-tick symbols have different mechanics and must not be pooled
    into a single performance statistic without interactions or stratification.
    """

    #: Median quoted spread at or below this many ticks => large-tick regime.
    large_tick_max_median_spread_ticks: float = 1.5
    #: Median quoted spread at or above this => small-tick regime.
    small_tick_min_median_spread_ticks: float = 3.0
    #: Fraction of observations quoted at exactly one tick, reported alongside.
    report_pct_at_one_tick: bool = True


# --- Signal / position / backtest ---
@dataclass
class SignalConfig:
    position_mode: str = "fixed"  # 'fixed' | 'vol_scaled' | 'capped_continuous'
    z_thresholds: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    max_position: float = 1.0
    prevent_overlap: bool = True  # no new trade while one is open


# --- Top-level config ---
@dataclass
class Config:
    columns: ColumnMapping = field(default_factory=ColumnMapping)
    data: DataConfig = field(default_factory=DataConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tick_regime: TickRegimeConfig = field(default_factory=TickRegimeConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    passive: PassiveConfig = field(default_factory=PassiveConfig)

    output_dir: str = "ofi_research/outputs"
    random_seed: int = 7  # only used for synthetic data / bootstrap ordering
    label: str = "default"

    # provenance flag: set True whenever data came from the synthetic generator
    synthetic_data: bool = False

    def to_json(self, path: Optional[str] = None) -> str:
        payload = json.dumps(asdict(self), indent=2, default=str)
        if path:
            Path(path).write_text(payload)
            logger.info("Wrote config to %s", path)
        return payload

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        """Build a Config from a nested dict, filling defaults for missing keys."""
        def build(klass, sub):
            sub = sub or {}
            valid = {f: sub[f] for f in sub if f in klass.__dataclass_fields__}
            return klass(**valid)

        def build_costs(sub):
            sub = dict(sub or {})
            lat = sub.pop("latency", None)
            cfg = build(CostConfig, sub)
            if lat is not None:
                cfg.latency = build(LatencyConfig, lat)
            return cfg

        return cls(
            columns=build(ColumnMapping, d.get("columns")),
            data=build(DataConfig, d.get("data")),
            integrity=build(IntegrityConfig, d.get("integrity")),
            features=build(FeatureConfig, d.get("features")),
            targets=build(TargetConfig, d.get("targets")),
            splits=build(SplitConfig, d.get("splits")),
            costs=build_costs(d.get("costs")),
            execution=build(ExecutionConfig, d.get("execution")),
            signal=build(SignalConfig, d.get("signal")),
            evaluation=build(EvaluationConfig, d.get("evaluation")),
            tick_regime=build(TickRegimeConfig, d.get("tick_regime")),
            sampling=build(SamplingConfig, d.get("sampling")),
            passive=build(PassiveConfig, d.get("passive")),
            **{k: d[k] for k in ("output_dir", "random_seed", "label",
                                 "synthetic_data") if k in d},
        )

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a concise format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
