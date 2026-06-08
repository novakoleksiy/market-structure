"""Interactive animated playback of a backtest using lightweight-charts.

Renders all three cluster timeframes (low / medium / high) stacked in a single
window. Candles are colored by market-structure trend and pivots are marked as
they get confirmed (mirroring ``chart.py``). The execution (low) timeframe also
shows trade entries/exits and the active trade's stop-loss / take-profit lines.

The higher timeframes advance off the low-timeframe clock: a higher-TF candle
appears once playback reaches its timestamp. Playback can be paused/resumed with
the topbar button or the spacebar.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import sleep

import numpy as np
import pandas as pd

from backtest import Trade, backtest_symbol_cluster
from clusters import get_cluster
from ms_engine import compute_market_structure, detect_pivots
from signal_engine import PIVOT_LENGTH, WARMUP_BARS, cluster_frames, fetch_all
from universe import UNIVERSE, Symbol

# Trend palette mirrors chart.py (fill = body, border = outline + wick).
_TREND_FILL = {1: "#80cbc4", -1: "#ff8a80", 0: "#888888"}
_TREND_BORDER = {1: "#00897b", -1: "#e53935", 0: "#888888"}

_ENTRY_COLOR = {"long": "#26a69a", "short": "#ef5350"}
_EXIT_COLOR = {"tp": "#26a69a", "sl": "#ef5350", "opposite": "#ffa726", "open": "#90a4ae"}


def _find_symbol(name: str) -> Symbol:
    """Return the universe Symbol matching ``name`` (by provider symbol)."""
    for sym in UNIVERSE:
        if sym.name == name:
            return sym
    raise ValueError(
        f"Unknown symbol '{name}'. Choose from: {sorted(s.name for s in UNIVERSE)}"
    )


def _style_pane(chart, label: str, size: int = 12) -> None:
    """Apply the shared per-pane styling: OHLC legend, top-right label, and a
    left-aligned price axis with a right-side buffer before the border.

    The small label uses the TradingView watermark directly (the wrapper's
    ``watermark`` helper hardcodes center alignment) so we can top-right align
    and shrink it. The OHLC legend sits top-left, so the two don't collide.
    """
    chart.legend(visible=True)
    chart.run_script(f"""
      {chart.id}.chart.applyOptions({{
          watermark: {{
              visible: true,
              horzAlign: 'right',
              vertAlign: 'top',
              text: "{label}",
              fontSize: {size},
              color: 'rgba(200, 200, 220, 0.75)'
          }},
          leftPriceScale: {{visible: true, borderVisible: false}},
          rightPriceScale: {{visible: false}}
      }});
      {chart.id}.series.applyOptions({{priceScaleId: 'left'}});
    """)
    chart.time_scale(right_offset=8)  # buffer between latest candle and border


def _feed_frame(df: pd.DataFrame, trend: np.ndarray) -> pd.DataFrame:
    """Build a lightweight-charts feed with a 'time' column and per-bar colors.

    The camelCase ``borderColor``/``wickColor`` keys are preserved because the
    wrapper only lowercases columns when no 'time' column is present.
    """
    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: "time"})
    out = out[["time", "open", "high", "low", "close"]].copy()
    # lightweight-charts' bulk set() does ``time.astype('int64') // 1e9`` assuming
    # nanosecond resolution; a datetime64[us] index would collapse many bars onto
    # the same second, so normalize to ns here.
    out["time"] = out["time"].dt.as_unit("ns")
    out["color"] = [_TREND_FILL[int(t)] for t in trend]
    out["borderColor"] = [_TREND_BORDER[int(t)] for t in trend]
    out["wickColor"] = [_TREND_BORDER[int(t)] for t in trend]
    return out


@dataclass
class _TFView:
    """Precomputed playback data for one timeframe."""

    feed: pd.DataFrame
    times: list
    pivots: dict[int, list[tuple[int, str]]]  # confirm_idx -> [(actual_idx, kind)]
    ptr: int = 0  # next bar index to push


def _prep_tf(full_df: pd.DataFrame, n_display: int, pivot_length: int) -> _TFView:
    """Compute trend (on full history) and pivots, then slice to a display window."""
    trend_full = compute_market_structure(
        full_df["high"].values, full_df["low"].values, full_df["close"].values,
        pivot_length,
    )
    n_display = max(1, min(n_display, len(full_df)))
    disp = full_df.iloc[-n_display:]
    feed = _feed_frame(disp, trend_full[-n_display:])
    times = list(feed["time"].dt.to_pydatetime())

    ph, pl = detect_pivots(disp["high"].values, disp["low"].values, pivot_length)
    pivots: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for ci in np.where(~np.isnan(ph))[0]:
        pivots[int(ci)].append((int(ci) - pivot_length, "high"))
    for ci in np.where(~np.isnan(pl))[0]:
        pivots[int(ci)].append((int(ci) - pivot_length, "low"))
    return _TFView(feed=feed, times=times, pivots=pivots)


class _ChartFeed:
    """Wraps a (sub)chart with its marker buffer and reveal helpers."""

    def __init__(self, chart, view: _TFView):
        self.chart = chart
        self.view = view
        self.markers: list[dict] = []

    def refresh(self) -> None:
        # lightweight-charts requires markers ascending by time, so re-render
        # the full set sorted whenever anything changes.
        self.chart.clear_markers()
        if self.markers:
            self.chart.marker_list(sorted(self.markers, key=lambda m: m["time"]))

    def add_pivot(self, actual_idx: int, kind: str) -> None:
        if actual_idx < 0:
            return
        self.markers.append({
            "time": self.view.times[actual_idx],
            "position": "above" if kind == "high" else "below",
            "shape": "circle",
            "color": _TREND_BORDER[-1] if kind == "high" else _TREND_BORDER[1],
            "text": "",
        })

    def set_initial(self, count: int) -> None:
        """Seed the chart with the first ``count`` bars and reveal their pivots."""
        count = max(1, min(count, len(self.view.feed)))
        self.chart.set(self.view.feed.iloc[:count])
        self.view.ptr = count
        for confirm_idx in range(count):
            for actual_idx, kind in self.view.pivots.get(confirm_idx, []):
                self.add_pivot(actual_idx, kind)

    def advance_to(self, t) -> None:
        """Stream all not-yet-shown bars whose time is <= ``t`` (clock = low TF)."""
        changed = False
        v = self.view
        while v.ptr < len(v.feed) and v.times[v.ptr] <= t:
            self.chart.update(v.feed.iloc[v.ptr])
            for actual_idx, kind in v.pivots.get(v.ptr, []):
                self.add_pivot(actual_idx, kind)
                changed = True
            v.ptr += 1
        if changed:
            self.refresh()


def _count_until(times: list, t) -> int:
    """Number of leading timestamps that are <= ``t``."""
    count = 0
    while count < len(times) and times[count] <= t:
        count += 1
    return count


def play_backtest(
    symbol: str = "BTCUSDT",
    cluster_name: str = "C1",
    n_bars: int = 2000,
    speed: float = 20.0,
    warmup: int = 200,
    pivot_length: int = PIVOT_LENGTH,
    *,
    atr_period: int = 14,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    opposite_exit: bool = True,
    final: bool = False,
) -> None:
    """Animate the backtest for one symbol + cluster across all three timeframes.

    ``speed`` is bars per second (low TF); ``warmup`` is how many low-TF bars are
    shown before streaming begins. Pause/resume with the topbar button or space.

    With ``final=True`` the whole window is rendered at once (every candle, pivot,
    and trade marker) and the streaming loop is skipped — a static view of the
    completed backtest instead of the candle-by-candle playback.
    """
    import asyncio

    from lightweight_charts import Chart
    from lightweight_charts.util import parse_event_message

    cluster = get_cluster(cluster_name)
    sym = _find_symbol(symbol)
    data = fetch_all([sym], n_bars=n_bars + WARMUP_BARS)
    df_l, df_m, df_h = cluster_frames(data, sym.name, cluster)

    trades = backtest_symbol_cluster(
        df_l, df_m, df_h, cluster, cluster_name, symbol,
        atr_period=atr_period, sl_mult=sl_mult, tp_mult=tp_mult,
        opposite_exit=opposite_exit,
    )

    n_low = max(1, min(n_bars, len(df_l)))
    window_start = df_l.index[-n_low]
    n_med = int((df_m.index >= window_start).sum())
    n_high = int((df_h.index >= window_start).sum())

    low_view = _prep_tf(df_l, n_low, pivot_length)
    med_view = _prep_tf(df_m, n_med, pivot_length)
    high_view = _prep_tf(df_h, n_high, pivot_length)

    # --- Build the three-pane window -------------------------------------
    # Low TF fills the left 60% full-height; medium/high stack in the right 40%.
    chart = Chart(
        inner_width=0.6, inner_height=1.0, toolbox=False,
        title=f"{symbol} {cluster_name} backtest",
    )
    med_chart = chart.create_subchart(position="right", width=0.4, height=0.5)
    high_chart = chart.create_subchart(position="right", width=0.4, height=0.5)
    for c, tf, tag in (
        (chart, cluster["low"], "execution"),
        (med_chart, cluster["med"], "medium"),
        (high_chart, cluster["high"], "high"),
    ):
        _style_pane(c, f"{symbol}  {tf}  ({tag})")

    low_cf = _ChartFeed(chart, low_view)
    med_cf = _ChartFeed(med_chart, med_view)
    high_cf = _ChartFeed(high_chart, high_view)

    entries: dict = defaultdict(list)
    exits: dict = defaultdict(list)
    for t in trades:
        entries[t.entry_time].append(t)
        exits[t.exit_time].append(t)
    active_lines: list = []

    def add_entry(t: Trade) -> None:
        low_cf.markers.append({
            "time": t.entry_time,
            "position": "below" if t.direction == "long" else "above",
            "shape": "arrow_up" if t.direction == "long" else "arrow_down",
            "color": _ENTRY_COLOR[t.direction],
            "text": f"{t.direction.upper()} @ {t.entry_price:.4f}",
        })
        active_lines.append(chart.horizontal_line(t.sl, color="#ef5350", text="SL"))
        active_lines.append(chart.horizontal_line(t.tp, color="#26a69a", text="TP"))

    def add_exit(t: Trade) -> None:
        low_cf.markers.append({
            "time": t.exit_time,
            "position": "above" if t.direction == "long" else "below",
            "shape": "square",
            "color": _EXIT_COLOR.get(t.exit_reason, "#90a4ae"),
            "text": f"{t.exit_reason.upper()} {t.r_multiple:+.1f}R",
        })
        while active_lines:
            active_lines.pop().delete()

    # --- Pause / resume controls -----------------------------------------
    state = {"paused": False}

    def toggle_pause(*_) -> None:
        state["paused"] = not state["paused"]
        chart.topbar["pause"].set("▶ Resume" if state["paused"] else "⏸ Pause")

    chart.topbar.button("pause", "⏸ Pause", func=toggle_pause)
    chart.hotkey(None, " ", toggle_pause)

    def pump_events() -> None:
        # Dispatch GUI callbacks (button/hotkey) ourselves so they fire while
        # our playback loop owns the main thread.
        wv = Chart.WV
        while not wv.emit_queue.empty():
            msg = wv.emit_queue.get()
            if msg == "exit":
                chart.is_alive = False
                return
            func, args = parse_event_message(chart.win, msg)
            asyncio.run(func(*args)) if asyncio.iscoroutinefunction(func) else func(*args)

    # --- Initial reveal (the warmup window, or all bars in final mode) ----
    # Seeds candles/pivots in bulk and replays trade markers in chronological
    # order so SL/TP lines settle to the correct state (only open trades left).
    reveal = n_low if final else max(1, min(warmup, n_low))
    cutoff = low_view.times[reveal - 1]
    low_cf.set_initial(reveal)
    med_cf.set_initial(_count_until(med_view.times, cutoff))
    high_cf.set_initial(_count_until(high_view.times, cutoff))
    for i in range(reveal):
        for t in exits.get(low_view.times[i], []):
            add_exit(t)
        for t in entries.get(low_view.times[i], []):
            add_entry(t)
    for cf in (low_cf, med_cf, high_cf):
        cf.refresh()

    if final:
        chart.show(block=True)
        return

    chart.show(block=False)

    # --- Stream the low timeframe; higher TFs follow its clock ------------
    delay = 1.0 / speed if speed > 0 else 0.0
    i = reveal
    while i < n_low and chart.is_alive:
        if state["paused"]:
            pump_events()
            sleep(0.03)
            continue

        chart.update(low_view.feed.iloc[i])
        t = low_view.times[i]

        changed = False
        for actual_idx, kind in low_view.pivots.get(i, []):
            low_cf.add_pivot(actual_idx, kind)
            changed = True
        for tr in exits.get(t, []):  # exits before entries (handles reversals)
            add_exit(tr)
            changed = True
        for tr in entries.get(t, []):
            add_entry(tr)
            changed = True
        if changed:
            low_cf.refresh()

        med_cf.advance_to(t)
        high_cf.advance_to(t)

        pump_events()
        if delay:
            sleep(delay)
        i += 1

    if chart.is_alive:
        chart.show(block=True)
