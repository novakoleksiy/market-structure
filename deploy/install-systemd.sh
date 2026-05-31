#!/usr/bin/env bash
# Install the market-structure systemd template + timers on a Linux host.
#
# Idempotent: safe to re-run after editing units. Run as root.
#   sudo deploy/install-systemd.sh
#
# Assumes the repo is checked out at /opt/market-structure (override with
# MARKET_STRUCTURE_HOME) and that uv is installed system-wide.
set -euo pipefail

HOME_DIR="${MARKET_STRUCTURE_HOME:-/opt/market-structure}"
STATE_DIR="/var/lib/market-structure"
ENV_FILE="/etc/market-structure.env"
SERVICE_USER="market-structure"
UNIT_SRC="${HOME_DIR}/deploy/systemd"
UNIT_DST="/etc/systemd/system"

if [[ ${EUID} -ne 0 ]]; then
  echo "must run as root" >&2
  exit 1
fi

if [[ ! -d ${UNIT_SRC} ]]; then
  echo "unit source ${UNIT_SRC} not found; is the repo at ${HOME_DIR}?" >&2
  exit 1
fi

# Service account (no login shell, owns the state dir).
if ! id "${SERVICE_USER}" &>/dev/null; then
  useradd --system --home-dir "${STATE_DIR}" --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${STATE_DIR}"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${STATE_DIR}/uv-cache"

if [[ ! -f ${ENV_FILE} ]]; then
  install -o root -g "${SERVICE_USER}" -m 0640 \
    "${HOME_DIR}/deploy/market-structure.env.example" "${ENV_FILE}"
  echo "wrote ${ENV_FILE} from example — edit it with real secrets before the first run."
fi

chmod +x "${UNIT_SRC}/run-market-structure-job.sh"

# Install the template service and all timers.
install -m 0644 "${UNIT_SRC}/market-structure@.service" "${UNIT_DST}/"
install -m 0644 "${UNIT_SRC}"/market-structure-*.timer "${UNIT_DST}/"

systemctl daemon-reload
# Enable + start each timer by unit name (systemctl needs names, not paths).
for t in "${UNIT_SRC}"/market-structure-*.timer; do
  systemctl enable --now "$(basename "$t")"
done

echo
echo "Installed. Scheduled fire times:"
systemctl list-timers 'market-structure-*' --all --no-pager
