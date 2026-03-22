# Test Documentation

## Overview

71 tests covering `ms_engine.py` (100%), `binance_data.py` (95% — only `__main__` block excluded), plus repainting and determinism tests.

Run tests: `uv run pytest tests/ --cov=ms_engine --cov=binance_data --cov-report=term-missing`

---

## test_ms_engine.py

### detect_pivots

| Test | Description |
|------|-------------|
| `test_detect_pivots_simple_high` | Pivot high is placed at the confirmation bar (pivot_length bars after the extremum) |
| `test_detect_pivots_simple_low` | Pivot low is placed at the confirmation bar |
| `test_detect_pivots_end_not_confirmed` | Pivot at the last bar is not confirmed (insufficient bars ahead) |
| `test_detect_pivots_no_pivots_monotonic` | Monotonically increasing data produces no pivots |
| `test_detect_pivots_length_2` | Pivot detection works correctly with `pivot_length=2` |

### update_market_structure

| Test | Description |
|------|-------------|
| `test_ums_initial_state_is_neutral` | Default state has trend=0 |
| `test_ums_first_pivot_low_starts_uptrend` | First pivot low from neutral sets trend=1 and support |
| `test_ums_first_pivot_high_starts_downtrend` | First pivot high from neutral sets trend=-1 and resistance |
| `test_ums_both_pivots_neutral_no_change` | Both pivots arriving simultaneously in neutral trend causes no change |
| `test_ums_uptrend_break_support_flips_down` | Close below support flips trend from 1 to -1, sets resistance |
| `test_ums_uptrend_higher_low_updates_support` | Higher pivot low in uptrend raises support level |
| `test_ums_uptrend_lower_low_keeps_support` | Lower pivot low in uptrend does not lower support |
| `test_ums_uptrend_nan_support_sets_first_low` | First pivot low in uptrend with NaN support sets support |
| `test_ums_downtrend_break_resistance_flips_up` | Close above resistance flips trend from -1 to 1, sets support |
| `test_ums_downtrend_lower_high_updates_resistance` | Lower pivot high in downtrend lowers resistance level |
| `test_ums_downtrend_higher_high_keeps_resistance` | Higher pivot high in downtrend does not raise resistance |
| `test_ums_downtrend_nan_resistance_sets_first_high` | First pivot high in downtrend with NaN resistance sets resistance |
| `test_ums_both_pivots_uptrend_prefers_ph` | When both pivots arrive in uptrend, pivot high is preferred |
| `test_ums_both_pivots_downtrend_prefers_pl` | When both pivots arrive in downtrend, pivot low is preferred |

### compute_market_structure

| Test | Description |
|------|-------------|
| `test_cms_returns_correct_length_and_dtype` | Output array matches input length and has int dtype |
| `test_cms_values_are_valid` | All output values are in {-1, 0, 1} on random data |

### resample_ohlc

| Test | Description |
|------|-------------|
| `test_resample_5min_to_30min` | 30 five-minute bars resample to 5 thirty-minute bars |
| `test_resample_preserves_ohlc_semantics` | Open=first, high=max, low=min, close=last are correct |

### get_mtf_trend

| Test | Description |
|------|-------------|
| `test_mtf_aligned_to_base_index` | Returned series has same index as the input base DataFrame |
| `test_mtf_values_are_valid` | All forward-filled trend values are in {-1, 0, 1} |

### _reset_long

| Test | Description |
|------|-------------|
| `test_reset_long_when_high_not_up` | All long state fields (including fired) reset when high trend != 1 |
| `test_reset_long_partial_m_rec_and_medium_down` | m_dip/m_rec/l_dip reset but fired preserved when high=1 and m_rec with medium=-1 |
| `test_reset_long_no_reset_m_rec_false` | No reset when m_rec is False (second branch condition not met) |
| `test_reset_long_no_reset_high_up_medium_up` | No reset when high=1 and medium=1 (no reset condition met) |

### _reset_short

| Test | Description |
|------|-------------|
| `test_reset_short_when_high_not_down` | All short state fields reset when high trend != -1 |
| `test_reset_short_partial_m_rec_s_and_medium_up` | m_dip_s/m_rec_s/l_dip_s reset but fired_s preserved when high=-1 and m_rec_s with medium=1 |
| `test_reset_short_no_reset_m_rec_s_false` | No reset when m_rec_s is False |
| `test_reset_short_no_reset_high_down_medium_down` | No reset when high=-1 and medium=-1 |

### _advance_long

| Test | Description |
|------|-------------|
| `test_advance_long_medium_dip` | Medium trend going to -1 sets m_dip |
| `test_advance_long_recovery_after_dip` | Medium recovering to 1 after dip sets m_rec, clears l_dip |
| `test_advance_long_low_dip_after_recovery` | Low trend going to -1 after recovery sets l_dip |
| `test_advance_long_no_advance_when_fired` | No state changes when fired=True |
| `test_advance_long_no_advance_when_high_not_up` | No state changes when high trend != 1 |

### _advance_short

| Test | Description |
|------|-------------|
| `test_advance_short_medium_dip` | Medium trend going to 1 sets m_dip_s |
| `test_advance_short_recovery_after_dip` | Medium recovering to -1 after dip sets m_rec_s, clears l_dip_s |
| `test_advance_short_low_dip_after_recovery` | Low trend going to 1 after recovery sets l_dip_s |
| `test_advance_short_no_advance_when_fired_s` | No state changes when fired_s=True |
| `test_advance_short_no_advance_when_high_not_down` | No state changes when high trend != -1 |

### compute_cluster_signals

| Test | Description |
|------|-------------|
| `test_cluster_shape_and_dtype` | Output arrays match input length and have bool dtype |
| `test_cluster_long_signal_full_sequence` | Full long setup (high=up, med dips, recovers, low dips, recovers) fires signal |
| `test_cluster_short_signal_full_sequence` | Full short setup (mirror of long) fires signal |
| `test_cluster_fires_only_once` | Signal does not re-fire after initial trigger (fired flag) |
| `test_cluster_empty_input` | Empty arrays return empty output |
| `test_cluster_single_bar` | Single bar produces no signals (loop starts at index 1) |
| `test_cluster_reset_allows_new_long_signal` | High trend breaking and returning resets fired, allowing a second long signal |
| `test_cluster_reset_allows_new_short_signal` | High trend breaking and returning resets fired_s, allowing a second short signal |

---

## test_binance_data.py

### _to_binance_interval

| Test | Description |
|------|-------------|
| `test_to_binance_interval_known` | All mapped intervals convert correctly (5min→5m, 1h→1h, 1D→1d, etc.) |
| `test_to_binance_interval_unsupported` | Unsupported interval raises ValueError |

### fetch_klines

| Test | Description |
|------|-------------|
| `test_fetch_klines_basic` | Returns DataFrame with correct columns from mocked API response |
| `test_fetch_klines_futures_usdt` | Uses fapi.binance.com URL for futures-usdt market |
| `test_fetch_klines_futures_coin` | Uses dapi.binance.com URL for futures-coin market |
| `test_fetch_klines_unknown_market` | Unknown market string raises ValueError |
| `test_fetch_klines_start_and_end_time` | startTime and endTime params are included in URL when provided |
| `test_fetch_klines_closed_only_false` | Bar with future close_time is kept when closed_only=False |
| `test_fetch_klines_closed_only_drops_open_bar` | Bar with future close_time is dropped when closed_only=True |

### fetch_klines_full

| Test | Description |
|------|-------------|
| `test_fetch_klines_full_single_page` | Single page of results returned correctly |
| `test_fetch_klines_full_pagination` | Multiple pages are fetched and concatenated when first chunk fills limit |
| `test_fetch_klines_full_empty` | Empty first response returns empty DataFrame |
| `test_fetch_klines_full_partial_page_stops` | Partial page (fewer bars than limit) stops pagination after one call |

---

## test_repainting.py

Tests that validate the repainting behavior of `compute_market_structure` when history length changes, and that a warm-up buffer eliminates divergence.

| Test | Description |
|------|-------------|
| `test_trend_changes_with_history_length` (parametrized: pivot_length=2,3,5) | Demonstrates that trend values on the tail differ when computed on full vs truncated history (repainting) |
| `test_trend_stable_with_warmup` (parametrized: pivot_length=2,3,5) | With a 200-bar warm-up buffer, trend values on the output window are identical regardless of total history length |

---

## test_determinism.py

Network test that verifies cluster signals are deterministic across different data fetch sizes.

| Test | Description |
|------|-------------|
| `test_signals_deterministic_across_limits` | Fetches live Binance data with limit=1000 and limit=1500, asserts that signals on overlapping bars are identical (marked `@pytest.mark.network`) |
