#!/usr/bin/env bash
# Build and start RapCluster with Docker, then print the links where it can be reached.
#
#   ./start.sh            build + start (docker compose up --build -d)
#   ./start.sh --no-build start without rebuilding
#
# Environment overrides:
#   RAPCLUSTER_PORT   host port published in compose.yaml (default 8088)
#   DOCKER            docker command to use, e.g. DOCKER="sudo docker"
set -euo pipefail

cd "$(dirname "$0")"

PORT="${RAPCLUSTER_PORT:-8088}"
APP_PATH="/rapcluster/"
# The lab's official instance. Only a reference: this script runs RapCluster on THIS machine
# and does not deploy to that server. Set RAPCLUSTER_PUBLIC_URL="" to hide the line.
PUBLIC_URL="${RAPCLUSTER_PUBLIC_URL-https://rapplab.bioanalytik.tu-berlin.de/rapcluster/}"
DOCKER="${DOCKER:-docker}"

BUILD_FLAG="--build"
if [[ "${1:-}" == "--no-build" ]]; then
  BUILD_FLAG=""
fi

# shellcheck disable=SC2086
$DOCKER compose up $BUILD_FLAG -d

# Collect the machine's LAN IPv4 addresses (the ones other devices on the same network can reach).
lan_ips() {
  # WSL2: Docker Desktop publishes ports on the Windows host, so ask Windows for its addresses.
  if grep -qi microsoft /proc/version 2>/dev/null && command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command \
      "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.IPAddress -notmatch '^(127\.|169\.254\.)' -and \$_.InterfaceAlias -notmatch 'vEthernet|WSL|Loopback|Docker' } | ForEach-Object { \$_.IPAddress }" \
      2>/dev/null | tr -d '\r'
    return
  fi
  # Linux
  if command -v hostname >/dev/null 2>&1 && hostname -I >/dev/null 2>&1; then
    hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' \
      | grep -vE '^(127\.|169\.254\.|172\.(1[7-9]|2[0-9]|3[01])\.)'
    return
  fi
  # macOS
  if command -v ipconfig >/dev/null 2>&1; then
    for iface in en0 en1 en2; do ipconfig getifaddr "$iface" 2>/dev/null || true; done
  fi
}

mapfile -t IPS < <(lan_ips | sed '/^$/d' | sort -u)

echo
echo "RapCluster is running:"
echo "  Local:           http://localhost:${PORT}${APP_PATH}"
if ((${#IPS[@]})); then
  for ip in "${IPS[@]}"; do
    echo "  On your network: http://${ip}:${PORT}${APP_PATH}"
  done
else
  echo "  On your network: (could not detect a LAN IP address)"
fi
if [[ -n "$PUBLIC_URL" ]]; then
  echo
  echo "Official public instance (separate server, not this machine): ${PUBLIC_URL}"
fi
echo
echo "If other devices cannot open the network link, allow inbound TCP port ${PORT} in your firewall."
echo "Stop with: $DOCKER compose down"
