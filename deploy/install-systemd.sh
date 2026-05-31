#!/usr/bin/env bash
# Single source of truth for the market-structure systemd schedule.
#
# Idempotent + rerunnable: removes any existing market-structure-* units
# (including the old OnBootSec/OnUnitActiveSec timers and stray drop-ins),
# regenerates 8 concrete service+timer pairs from the SCHEDULE table below,
# and enables them. Run it on a fresh LXC or to replace a broken schedule:
#
#   sudo deploy/install-systemd.sh            # install/refresh units
#   sudo deploy/install-systemd.sh --bootstrap # also create user/dirs/env
#
# Prerequisites (the script does NOT install these):
#   - the app checked out at /opt/market-structure
#   - uv at /usr/local/bin/uv
#   - /etc/market-structure.env with real secrets (see --bootstrap)
#
# Timers fire just after the relevant candle closes. The lowest timeframe per
# cluster sets the cadence (C1 5min, C2 30min, C3 4h, C4 1D). Binance is
# UTC-aligned on every TF; OANDA M5/M30 are UTC but H4/D inherit OANDA's 17:00
# America/New_York dailyAlignment default (tradfi_data.py sends no
# dailyAlignment), so OANDA C3/C4 use a NY OnCalendar to stay DST-correct.
# NOTE: timezone suffixes in OnCalendar require systemd >= 252 (Debian 12+).
set -euo pipefail

HOME_DIR="${MARKET_STRUCTURE_HOME:-/opt/market-structure}"
STATE_DIR="${MARKET_STRUCTURE_STATE:-/var/lib/market-structure}"
DB_PATH="${MARKET_STRUCTURE_DB:-${STATE_DIR}/signals.sqlite3}"
LOCK_FILE="${MARKET_STRUCTURE_LOCK:-${STATE_DIR}/job.lock}"
CACHE_DIR="${STATE_DIR}/uv-cache"
ENV_FILE="${MARKET_STRUCTURE_ENV:-/etc/market-structure.env}"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
SERVICE_USER="market-structure"
UNIT_DIR="/etc/systemd/system"

# source | cluster | OnCalendar | AccuracySec
# (tight AccuracySec so the run lands right after close, not minutes later)
SCHEDULE="
binance|c1|*-*-* *:00/5:30 UTC|30s
binance|c2|*-*-* *:00/30:30 UTC|30s
binance|c3|*-*-* 00,04,08,12,16,20:00:30 UTC|30s
binance|c4|*-*-* 00:02:00 UTC|30s
oanda|c1|*-*-* *:00/5:30 UTC|30s
oanda|c2|*-*-* *:00/30:30 UTC|30s
oanda|c3|*-*-* 01,05,09,13,17,21:00:30 America/New_York|30s
oanda|c4|*-*-* 17:02:00 America/New_York|30s
"

if [[ ${EUID} -ne 0 ]]; then
  echo "must run as root" >&2
  exit 1
fi

# --- optional fresh-LXC bootstrap -----------------------------------------
if [[ "${1:-}" == "--bootstrap" ]]; then
  if ! id "${SERVICE_USER}" &>/dev/null; then
    useradd --system --home-dir "${STATE_DIR}" --shell /usr/sbin/nologin "${SERVICE_USER}"
  fi
  install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${STATE_DIR}"
  install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${CACHE_DIR}"
  if [[ ! -f ${ENV_FILE} && -f ${HOME_DIR}/.env.example ]]; then
    install -o root -g "${SERVICE_USER}" -m 0640 \
      "${HOME_DIR}/.env.example" "${ENV_FILE}"
    echo "wrote ${ENV_FILE} from .env.example — edit it with real secrets before the first run."
  fi
  # The service user must own the checkout so `uv run` can create/sync .venv there.
  chown -R "${SERVICE_USER}:${SERVICE_USER}" "${HOME_DIR}"
  # Pre-build the venv as the service user so the first timer run isn't a cold sync.
  sudo -u "${SERVICE_USER}" env HOME="${STATE_DIR}" UV_CACHE_DIR="${CACHE_DIR}" \
    "${UV_BIN}" --directory "${HOME_DIR}" sync
fi

# --- clean slate: drop every existing market-structure unit ----------------
mapfile -t OLD < <(
  systemctl list-unit-files 'market-structure-*' 'market-structure@*' \
    --no-legend --all 2>/dev/null | awk '{print $1}'
)
for u in "${OLD[@]:-}"; do
  [[ -n ${u} ]] || continue
  systemctl disable --now "${u}" 2>/dev/null || true
done
rm -f "${UNIT_DIR}"/market-structure-*.service "${UNIT_DIR}"/market-structure-*.timer
rm -f "${UNIT_DIR}"/market-structure@*.service "${UNIT_DIR}"/run-market-structure-job.sh
rm -rf "${UNIT_DIR}"/market-structure-*.timer.d
systemctl daemon-reload

# --- generate concrete units from the schedule table -----------------------
TIMERS=()
while IFS='|' read -r src cl cal acc; do
  [[ -n ${src} ]] || continue
  CL_UP="${cl^^}"
  SRC_UP="${src^^}"
  base="market-structure-${src}-${cl}"

  cat >"${UNIT_DIR}/${base}.service" <<EOF
[Unit]
Description=Market Structure ${SRC_UP} ${CL_UP} signal job
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${HOME_DIR}
EnvironmentFile=${ENV_FILE}
Environment=HOME=${STATE_DIR}
Environment=UV_CACHE_DIR=${CACHE_DIR}
ExecStart=/usr/bin/flock -w 900 ${LOCK_FILE} ${UV_BIN} run main.py run-source-cluster --source ${src} --cluster ${CL_UP} --db-path ${DB_PATH}
EOF

  cat >"${UNIT_DIR}/${base}.timer" <<EOF
[Unit]
Description=Run Market Structure ${SRC_UP} ${CL_UP} at candle close

[Timer]
OnCalendar=${cal}
AccuracySec=${acc}
Persistent=true
Unit=${base}.service

[Install]
WantedBy=timers.target
EOF

  TIMERS+=("${base}.timer")
done <<<"${SCHEDULE}"

systemctl daemon-reload
systemctl enable --now "${TIMERS[@]}"

echo
echo "Installed ${#TIMERS[@]} timers. Next fire times:"
systemctl list-timers 'market-structure-*' --all --no-pager
