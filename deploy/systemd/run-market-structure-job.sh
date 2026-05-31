#!/usr/bin/env bash
# Wrapper invoked by market-structure@<source>-<cluster>.service.
#
# Parses the systemd instance name ("<source>-<cluster>", e.g. "oanda-c1"),
# serialises all signal jobs behind a single flock (the shared sqlite db and
# parquet cache are not safe to write concurrently), and runs the
# run-source-cluster CLI which persists new signals and sends Telegram alerts.
set -euo pipefail

INSTANCE="${1:?usage: run-market-structure-job.sh <source>-<cluster>}"

# Split "<source>-<cluster>" on the last '-' so a source name is free to
# contain hyphens while the cluster suffix (c1..c4) stays unambiguous.
SOURCE="${INSTANCE%-*}"
CLUSTER_RAW="${INSTANCE##*-}"

case "${SOURCE}" in
  binance | oanda) ;;
  *)
    echo "unknown source '${SOURCE}' (expected binance|oanda)" >&2
    exit 64
    ;;
esac

# main.py expects an upper-case cluster id (C1..C4); the instance uses lower case.
CLUSTER="${CLUSTER_RAW^^}"
case "${CLUSTER}" in
  C1 | C2 | C3 | C4) ;;
  *)
    echo "unknown cluster '${CLUSTER_RAW}' (expected c1..c4)" >&2
    exit 64
    ;;
esac

WORKDIR="${MARKET_STRUCTURE_HOME:-/opt/market-structure}"
DB_PATH="${MARKET_STRUCTURE_DB:-/var/lib/market-structure/signals.sqlite3}"
LOCK_FILE="${MARKET_STRUCTURE_LOCK:-/var/lib/market-structure/job.lock}"
UV="${UV_BIN:-uv}"

cd "${WORKDIR}"

# -w 900: wait up to 15 min for a peer job to finish rather than failing fast,
# so a 5-min cluster does not silently skip a candle while a slow daily run
# holds the lock. fd 9 stays open for the duration of the run.
exec 9>"${LOCK_FILE}"
flock -w 900 9 || {
  echo "could not acquire ${LOCK_FILE} within 900s; skipping ${INSTANCE}" >&2
  exit 75
}

exec "${UV}" run main.py run-source-cluster \
  --source "${SOURCE}" \
  --cluster "${CLUSTER}" \
  --db-path "${DB_PATH}"
