# Deployment — systemd timers

Scheduled signal generation runs as 8 concrete systemd units, one per
`(source, cluster)` pair. Each timer fires shortly **after** the relevant
candle closes and triggers a oneshot service that runs the `run-source-cluster`
CLI (persist new signals + Telegram alert) behind a shared `flock`.

`install-systemd.sh` is the **single source of truth**: it embeds the schedule
table, removes any existing `market-structure-*` units, regenerates all 8
service+timer pairs, and enables them. It is idempotent — rerun it to refresh
the schedule or to provision a fresh LXC.

## Why candle-aligned timers

A cluster's job is only meaningful once its lowest timeframe's candle has
closed. The lowest TF per cluster (`clusters.py`) sets the cadence:

| Cluster | low TF | binance close (UTC) | OANDA close |
|---------|--------|---------------------|-------------|
| C1 | 5min | every :00/:05/:10… | every :00/:05/:10… (UTC) |
| C2 | 30min | every :00/:30 | every :00/:30 (UTC) |
| C3 | 4h | 00,04,08,12,16,20 | 01,05,09,13,17,21 (America/New_York) |
| C4 | 1D | 00:00 | 17:00 (America/New_York) |

**Alignment facts** (from `binance_data.py` / `tradfi_data.py`):

- Binance: every timeframe is UTC-aligned.
- OANDA: M5/M30 are UTC-aligned, but H4 and D inherit OANDA's `dailyAlignment`
  default of **17:00 America/New_York** because `tradfi_data.py` sends no
  `dailyAlignment`/`alignmentTimezone`. That is why OANDA C3/C4 use a
  `America/New_York` `OnCalendar` (DST-safe) rather than UTC.

Grace offset after close: 30s intraday, 2min daily, so the exchange/broker has
settled the just-closed bar. All timers set `Persistent=true` (catch up missed
runs after downtime) and a tight `AccuracySec=30s` (fire right at the boundary).

This replaces the original `OnBootSec=`+`OnUnitActiveSec=` schedule, which fired
on a boot-relative interval unrelated to candle boundaries — leaving C4 (daily)
running on the previous day's bar.

> `OnCalendar` timezone suffixes (`… America/New_York`) require
> **systemd ≥ 252** (Debian 12 / Ubuntu 24.04 ship this). Check with
> `systemctl --version`. On older systemd, edit the two OANDA rows in the
> `SCHEDULE` table to UTC — but those are **not** DST-safe and must be flipped
> twice a year:
> EDT: C3 `*-*-* 01,05,09,13,17,21:00:30 UTC`, C4 `*-*-* 21:02:00 UTC`;
> EST: C3 `*-*-* 02,06,10,14,18,22:00:30 UTC`, C4 `*-*-* 22:02:00 UTC`.

## Install / refresh

```bash
# Prereqs: app at /opt/market-structure, uv at /usr/local/bin/uv,
# /etc/market-structure.env populated with secrets.
sudo deploy/install-systemd.sh

# Fresh LXC: also create the service user, state dir, and env file:
sudo deploy/install-systemd.sh --bootstrap
sudoedit /etc/market-structure.env   # fill in real secrets, then rerun without --bootstrap
```

The script generates, for each `(source, cluster)`:

- `/etc/systemd/system/market-structure-<source>-<cluster>.service` — oneshot,
  `User=market-structure`, `WorkingDirectory=/opt/market-structure`,
  `EnvironmentFile=/etc/market-structure.env`,
  `ExecStart=/usr/bin/flock -w 900 /var/lib/market-structure/job.lock /usr/local/bin/uv run main.py run-source-cluster --source <s> --cluster <C#> --db-path /var/lib/market-structure/signals.sqlite3`.
- `/etc/systemd/system/market-structure-<source>-<cluster>.timer` — the
  `OnCalendar` schedule above.

To change schedules, sources, or clusters, edit the `SCHEDULE` table at the top
of `install-systemd.sh` and rerun it.

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

Also ground-truth OANDA alignment before fully trusting the NY values: pull one
OANDA H4 candle and confirm closes land on 01/05/09/13/17/21 UTC under EDT. If
OANDA returns UTC-aligned H4/D instead, switch the OANDA C3/C4 rows in the
`SCHEDULE` table to the binance UTC expressions and rerun.

## Run a job manually (without systemd)

```bash
cd /opt/market-structure
uv run main.py run-source-cluster --source oanda --cluster C1 \
  --db-path /var/lib/market-structure/signals.sqlite3
```
