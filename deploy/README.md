# Deployment — systemd timers

Scheduled signal generation runs as a set of systemd timers, one per
`(source, cluster)` pair. Each timer fires shortly **after** the relevant
candle closes, then triggers a templated oneshot service that runs the
`run-source-cluster` CLI (persist new signals + Telegram alert).

## Why candle-aligned timers

The job for a cluster is only meaningful once its lowest timeframe's candle has
closed. The lowest TF per cluster (`clusters.py`) sets the cadence:

| Cluster | low TF | binance close (UTC) | OANDA close |
|---------|--------|---------------------|-------------|
| C1 | 5min | every :00/:05/:10… | every :00/:05/:10… (UTC) |
| C2 | 30min | every :00/:30 | every :00/:30 (UTC) |
| C3 | 4h | 00,04,08,12,16,20 | 01,05,09,13,17,21 (America/New_York) |
| C4 | 1D | 00:00 | 17:00 (America/New_York) |

**Alignment facts** (from `binance_data.py` / `tradfi_data.py`):

- Binance: every timeframe is UTC-aligned.
- OANDA: M5/M30 are UTC-aligned, but H4 and D inherit OANDA's
  `dailyAlignment` default of **17:00 America/New_York** because
  `tradfi_data.py` sends no `dailyAlignment`/`alignmentTimezone`. That is why
  OANDA C3/C4 use a `America/New_York` `OnCalendar` (DST-safe) rather than UTC.

Timers fire with a grace offset after close — 30s intraday, 2min daily — so the
exchange/broker has settled the just-closed bar before we fetch it. All timers
set `Persistent=true` (catch up missed runs after downtime) and
`AccuracySec=30s`.

> `OnCalendar` timezone suffixes (e.g. `… America/New_York`) require
> **systemd ≥ 252** (Debian 12 / Ubuntu 24.04 ship this). On older systemd,
> replace OANDA C3/C4 with UTC equivalents — but note these are **not**
> DST-safe and must be flipped twice a year:
> EDT: C3 `*-*-* 01,05,09,13,17,21:00:30 UTC`, C4 `*-*-* 21:02:00 UTC`;
> EST: C3 `*-*-* 02,06,10,14,18,22:00:30 UTC`, C4 `*-*-* 22:02:00 UTC`.

## Components

- `systemd/market-structure@.service` — oneshot template; `%i` is
  `<source>-<cluster>` (e.g. `oanda-c1`). Runs as the `market-structure` user,
  `WorkingDirectory=/opt/market-structure`, `EnvironmentFile=/etc/market-structure.env`.
- `systemd/run-market-structure-job.sh` — parses `%i`, takes a shared
  `flock` (the sqlite db + parquet cache are not concurrency-safe), then runs:
  `uv run main.py run-source-cluster --source <source> --cluster <C#> --db-path /var/lib/market-structure/signals.sqlite3`
  (`latest_only=False` and Telegram notify are applied inside the CLI).
- `systemd/market-structure-{binance,oanda}-c{1,2,3,4}.timer` — the 8 timers.
- `market-structure.env.example` — template for `/etc/market-structure.env`.
- `install-systemd.sh` — creates the service user + state dir, installs units,
  enables and starts all timers.

## Install

```bash
# Repo checked out at /opt/market-structure, uv installed system-wide.
sudo deploy/install-systemd.sh
# Then edit secrets:
sudoedit /etc/market-structure.env
```

## Verify

```bash
# NEXT column should land on a candle boundary + grace (OANDA C4 = 21:02 UTC under EDT).
systemctl list-timers 'market-structure-*' --all --no-pager

# After the next OANDA 4h boundary, confirm a freshly closed bar was persisted:
journalctl -u market-structure-oanda-c3.service -b

# No new failures (pre-existing unprivileged-LXC mount failures are unrelated):
systemctl --failed

# Sanity-check a single calendar expression's next fire times:
systemd-analyze calendar '*-*-* 17:02:00 America/New_York' --iterations=3

# Run one job by hand:
sudo systemctl start market-structure-oanda-c1.service
```

## Run a job manually (without systemd)

```bash
cd /opt/market-structure
uv run main.py run-source-cluster --source oanda --cluster C1 \
  --db-path /var/lib/market-structure/signals.sqlite3
```
