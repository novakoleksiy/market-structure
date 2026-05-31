"""Backtesting engine for market-structure cluster signals.

Takes the long/short signals produced by ``signal_engine.run_cluster`` and
simulates trades with fixed ATR-based risk targets:

    SL = entry - sl_mult * ATR(period)   (mirror for shorts)
    TP = entry + tp_mult * ATR(period)

A position exits at whichever comes first: stop-loss, take-profit, or an
opposite-direction cluster signal.  ATR is computed on the lowest cluster
timeframe (the execution timeframe).  Across the universe every signal is
taken at an equal 1R risk, so the pooled equity curve is denominated in R.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from clusters import ALL_CLUSTERS, get_cluster
from ms_engine import compute_atr
from signal_engine import WARMUP_BARS, cluster_frames, fetch_all, run_cluster
from universe import UNIVERSE, Symbol

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "backtests"


@dataclass
class Trade:
    symbol: str
    cluster: str
    direction: str  # "long" | "short"
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "opposite" | "open"
    r_multiple: float
    return_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series  # cumulative R, indexed by exit_time
    metrics: dict


def _make_trade(
    symbol: str,
    cluster_name: str,
    direction: str,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    sl: float,
    tp: float,
    exit_price: float,
    exit_reason: str,
    index: pd.DatetimeIndex,
) -> Trade:
    """Build a Trade, computing R-multiple and percentage return."""
    risk = abs(entry_price - sl)
    sign = 1.0 if direction == "long" else -1.0
    pnl_price = sign * (exit_price - entry_price)
    r_multiple = pnl_price / risk if risk else 0.0
    return_pct = sign * (exit_price - entry_price) / entry_price * 100.0
    return Trade(
        symbol=symbol,
        cluster=cluster_name,
        direction=direction,
        entry_time=index[entry_idx].to_pydatetime(),
        entry_price=entry_price,
        sl=sl,
        tp=tp,
        exit_time=index[exit_idx].to_pydatetime(),
        exit_price=exit_price,
        exit_reason=exit_reason,
        r_multiple=r_multiple,
        return_pct=return_pct,
        bars_held=exit_idx - entry_idx,
    )


def backtest_symbol_cluster(
    df_l: pd.DataFrame,
    df_m: pd.DataFrame,
    df_h: pd.DataFrame,
    cluster: dict[str, str],
    cluster_name: str,
    symbol: str,
    *,
    atr_period: int = 14,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    opposite_exit: bool = True,
) -> list[Trade]:
    """Simulate trades for one symbol + cluster on the low timeframe.

    One open position at a time (no pyramiding).  Entries fill at the signal
    bar's close; SL/TP are checked intrabar from the next bar onward.  When SL
    and TP are both touched in the same bar, the stop is assumed to fill first.
    """
    df, longs, shorts = run_cluster(df_l, df_m, df_h, cluster, show_length=None)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = compute_atr(high, low, close, atr_period)
    index = df.index

    trades: list[Trade] = []
    n = len(df)

    direction = ""  # "" means flat
    entry_idx = 0
    entry_price = sl = tp = 0.0

    for i in range(n):
        # --- Manage an open position (never on its own entry bar) -----------
        if direction and i > entry_idx:
            exit_price = None
            exit_reason = ""
            if direction == "long":
                hit_sl = low[i] <= sl
                hit_tp = high[i] >= tp
                if hit_sl:  # conservative: stop fills before target
                    exit_price, exit_reason = sl, "sl"
                elif hit_tp:
                    exit_price, exit_reason = tp, "tp"
                elif opposite_exit and shorts[i]:
                    exit_price, exit_reason = close[i], "opposite"
            else:  # short
                hit_sl = high[i] >= sl
                hit_tp = low[i] <= tp
                if hit_sl:
                    exit_price, exit_reason = sl, "sl"
                elif hit_tp:
                    exit_price, exit_reason = tp, "tp"
                elif opposite_exit and longs[i]:
                    exit_price, exit_reason = close[i], "opposite"

            if exit_price is not None:
                trades.append(
                    _make_trade(
                        symbol, cluster_name, direction, entry_idx, i,
                        entry_price, sl, tp, exit_price, exit_reason, index,
                    )
                )
                direction = ""

        # --- Open a new position on a fresh signal --------------------------
        if not direction and (longs[i] or shorts[i]) and not pd.isna(atr[i]):
            entry_idx = i
            entry_price = float(close[i])
            risk = atr[i] * sl_mult
            reward = atr[i] * tp_mult
            if longs[i]:
                direction = "long"
                sl = entry_price - risk
                tp = entry_price + reward
            else:
                direction = "short"
                sl = entry_price + risk
                tp = entry_price - reward

    # --- Mark any still-open position to the last close --------------------
    if direction:
        last = n - 1
        trades.append(
            _make_trade(
                symbol, cluster_name, direction, entry_idx, last,
                entry_price, sl, tp, float(close[last]), "open", index,
            )
        )

    return trades


def _max_drawdown(equity: pd.Series) -> float:
    """Return the largest peak-to-trough drop on a cumulative-R curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    return float((equity - running_max).min())


def _breakdown(trades: list[Trade], key: str) -> dict[str, dict]:
    """Group closed trades by a Trade attribute and summarise each group."""
    groups: dict[str, list[Trade]] = {}
    for t in trades:
        groups.setdefault(getattr(t, key), []).append(t)
    out: dict[str, dict] = {}
    for name, group in sorted(groups.items()):
        wins = sum(1 for t in group if t.r_multiple > 0)
        out[name] = {
            "n_trades": len(group),
            "win_rate": wins / len(group) if group else 0.0,
            "total_R": sum(t.r_multiple for t in group),
        }
    return out


def compute_metrics(trades: list[Trade]) -> dict:
    """Aggregate performance metrics over closed trades (excludes 'open')."""
    closed = [t for t in trades if t.exit_reason != "open"]
    n = len(closed)
    if n == 0:
        return {"n_trades": 0}

    wins = [t for t in closed if t.r_multiple > 0]
    losses = [t for t in closed if t.r_multiple < 0]
    gross_win = sum(t.r_multiple for t in wins)
    gross_loss = abs(sum(t.r_multiple for t in losses))
    total_r = sum(t.r_multiple for t in closed)

    ordered = sorted(closed, key=lambda t: t.exit_time)
    equity = pd.Series(
        [t.r_multiple for t in ordered],
        index=pd.DatetimeIndex([t.exit_time for t in ordered]),
    ).cumsum()

    return {
        "n_trades": n,
        "n_open": len(trades) - n,
        "win_rate": len(wins) / n,
        "total_R": total_r,
        "avg_R": total_r / n,
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "max_drawdown_R": _max_drawdown(equity),
        "avg_bars_held": sum(t.bars_held for t in closed) / n,
        "exit_counts": {
            r: sum(1 for t in closed if t.exit_reason == r)
            for r in ("tp", "sl", "opposite")
        },
        "by_symbol": _breakdown(closed, "symbol"),
        "by_cluster": _breakdown(closed, "cluster"),
    }


def run_backtest(
    universe: list[Symbol] | None = None,
    clusters: list[str] | None = None,
    n_bars: int = 3000,
    *,
    atr_period: int = 14,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    opposite_exit: bool = True,
) -> BacktestResult:
    """Backtest selected clusters across the universe, pooling trades at 1R each.

    Fetches all required timeframes once (reusing the parquet cache) and runs
    every (symbol, cluster) pair.  Symbols missing data for a cluster are
    skipped.  The equity curve is cumulative R ordered by exit time.
    """
    universe = universe or UNIVERSE
    cluster_names = clusters or list(ALL_CLUSTERS)
    cluster_defs = {name: get_cluster(name) for name in cluster_names}

    data = fetch_all(universe, n_bars=n_bars + WARMUP_BARS)

    trades: list[Trade] = []
    for sym in universe:
        for name, cluster in cluster_defs.items():
            try:
                df_l, df_m, df_h = cluster_frames(data, sym.name, cluster)
            except KeyError:
                continue  # data missing for this symbol/timeframe
            if df_l.empty or df_m.empty or df_h.empty:
                continue
            trades.extend(
                backtest_symbol_cluster(
                    df_l, df_m, df_h, cluster, name, sym.name,
                    atr_period=atr_period,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    opposite_exit=opposite_exit,
                )
            )

    closed = sorted(
        (t for t in trades if t.exit_reason != "open"), key=lambda t: t.exit_time
    )
    equity_curve = pd.Series(
        [t.r_multiple for t in closed],
        index=pd.DatetimeIndex([t.exit_time for t in closed]),
        name="equity_R",
    ).cumsum()

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=compute_metrics(trades),
    )


def save_trades(trades: list[Trade], path: Path | None = None) -> Path:
    """Write trades to CSV under .cache/backtests and return the path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CACHE_DIR / f"trades_{stamp}.csv"
    fields = list(Trade.__dataclass_fields__)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow(asdict(t))
    return path


def print_report(result: BacktestResult, save: bool = True) -> None:
    """Print a formatted summary of a backtest to stdout."""
    m = result.metrics
    print("=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    if m.get("n_trades", 0) == 0:
        print("No trades generated.")
        return

    pf = m["profit_factor"]
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"Closed trades : {m['n_trades']}  (open at end: {m['n_open']})")
    print(f"Win rate      : {m['win_rate'] * 100:.1f}%")
    print(f"Total R       : {m['total_R']:+.1f}R")
    print(f"Avg / expectancy: {m['avg_R']:+.3f}R per trade")
    print(f"Profit factor : {pf_str}")
    print(f"Max drawdown  : {m['max_drawdown_R']:.1f}R")
    print(f"Avg bars held : {m['avg_bars_held']:.1f}")
    print(f"Exits         : {m['exit_counts']}")

    print("\nBy cluster:")
    for name, s in m["by_cluster"].items():
        print(
            f"  {name:4s}  n={s['n_trades']:4d}  win={s['win_rate'] * 100:5.1f}%  "
            f"total={s['total_R']:+.1f}R"
        )

    print("\nBy symbol (top by total R):")
    ranked = sorted(
        m["by_symbol"].items(), key=lambda kv: kv[1]["total_R"], reverse=True
    )
    for name, s in ranked:
        print(
            f"  {name:12s}  n={s['n_trades']:4d}  win={s['win_rate'] * 100:5.1f}%  "
            f"total={s['total_R']:+.1f}R"
        )

    if save:
        path = save_trades(result.trades)
        print(f"\nTrades written to {path}")
