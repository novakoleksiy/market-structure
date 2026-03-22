"""Diagnose where signal divergence comes from across different limit values."""

import numpy as np

from binance_data import fetch_klines_full
from ms_engine import compute_cluster_signals, get_mtf_trend

PIVOT_LENGTH = 2
CLUSTER = {"low": "4h", "med": "1D", "high": "1W"}
LIMITS = [500, 1000, 1500]


def run_debug(limit):
    df = fetch_klines_full("BTCUSDT", CLUSTER["low"], n_bars=limit, market="futures-usdt")
    df_m = fetch_klines_full("BTCUSDT", CLUSTER["med"], n_bars=5000, market="futures-usdt")
    df_h = fetch_klines_full("BTCUSDT", CLUSTER["high"], n_bars=5000, market="futures-usdt")

    trend_l = get_mtf_trend(df, CLUSTER["low"], PIVOT_LENGTH, higher_tf_df=df)
    trend_m = get_mtf_trend(df, CLUSTER["med"], PIVOT_LENGTH, higher_tf_df=df_m)
    trend_h = get_mtf_trend(df, CLUSTER["high"], PIVOT_LENGTH, higher_tf_df=df_h)

    longs, shorts = compute_cluster_signals(trend_h.values, trend_m.values, trend_l.values)
    return df, trend_l, trend_m, trend_h, longs, shorts


results = {lim: run_debug(lim) for lim in LIMITS}

# Compare on overlapping tail (last 500 bars = smallest limit)
tail = min(LIMITS)
ref_limit = LIMITS[0]
ref_df, ref_tl, ref_tm, ref_th, ref_longs, ref_shorts = results[ref_limit]
ref_idx = ref_df.index

for lim in LIMITS[1:]:
    df, tl, tm, th, longs, shorts = results[lim]

    # Align by timestamp
    overlap = ref_idx.intersection(df.index)
    mask_ref = ref_idx.isin(overlap)
    mask_other = df.index.isin(overlap)

    tl_ref, tl_other = ref_tl[mask_ref].values, tl[mask_other].values
    tm_ref, tm_other = ref_tm[mask_ref].values, tm[mask_other].values
    th_ref, th_other = ref_th[mask_ref].values, th[mask_other].values
    l_ref, l_other = ref_longs[mask_ref], longs[mask_other]
    s_ref, s_other = ref_shorts[mask_ref], shorts[mask_other]

    print(f"\n=== limit={ref_limit} vs limit={lim} ({len(overlap)} overlapping bars) ===")
    print(f"  trend_l mismatches: {np.sum(tl_ref != tl_other)}")
    print(f"  trend_m mismatches: {np.sum(tm_ref != tm_other)}")
    print(f"  trend_h mismatches: {np.sum(th_ref != th_other)}")
    print(f"  long signal mismatches: {np.sum(l_ref != l_other)}")
    print(f"  short signal mismatches: {np.sum(s_ref != s_other)}")

    if np.sum(tl_ref != tl_other) > 0:
        idxs = np.where(tl_ref != tl_other)[0]
        print(f"  trend_l mismatch positions (0=oldest): {idxs.tolist()}")
        print(f"  trend_l mismatch dates (all): {[str(overlap[i]) for i in idxs]}")
        print(f"  trend_l values ref: {tl_ref[idxs].tolist()}")
        print(f"  trend_l values other: {tl_other[idxs].tolist()}")
        last_mm = idxs[-1]
        print(f"  last mismatch at position {last_mm}/{len(overlap)-1} "
              f"({overlap[last_mm]})")
        print(f"  trend_l agrees on last {len(overlap) - last_mm - 1} bars")

    if np.sum(l_ref != l_other) > 0 or np.sum(s_ref != s_other) > 0:
        sig_idxs = np.where((l_ref != l_other) | (s_ref != s_other))[0]
        print(f"  signal mismatch dates: {[str(overlap[i]) for i in sig_idxs]}")

    # Re-run cluster signals on the overlapping window using identical
    # trend inputs from the longer run — this isolates whether the signal
    # mismatch is from trend divergence or cluster state machine cold start.
    longs_rerun, shorts_rerun = compute_cluster_signals(
        th_other, tm_other, tl_other
    )
    # Compare: ref signals (from short run's full pipeline) vs rerun
    # (same trend data as longer run, but cluster SM starts fresh at overlap)
    l_rerun_diff = np.sum(l_other != longs_rerun)
    s_rerun_diff = np.sum(s_other != shorts_rerun)
    print(f"\n  Re-run cluster on overlapping trends (fresh state):")
    print(f"    vs longer run's signals: longs={l_rerun_diff}, shorts={s_rerun_diff}")

    # Now compare rerun vs ref (both start cluster SM fresh on same window)
    l_fresh = np.sum(l_ref != longs_rerun)
    s_fresh = np.sum(s_ref != shorts_rerun)
    print(f"    vs shorter run's signals: longs={l_fresh}, shorts={s_fresh}")
    print(f"  -> If both are 0, the cluster SM cold start is the sole cause.")