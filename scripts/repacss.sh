#!/usr/bin/env bash
# Connect to the REPACSS cluster at TTU (8 nodes x 4 NVIDIA H100 NVL 94 GB), https://guide.repacss.org
#
#   scripts/repacss.sh                 login shell
#   scripts/repacss.sh gpu [CPUS]      login and start an interactive session on the h100 partition
#   scripts/repacss.sh 'squeue -u $USER'   run one command on the login node
#   scripts/repacss.sh --check         network check only (VPN, DNS, port 22)
#   scripts/repacss.sh --close         close the shared connection
#
# You type the password (and approve MFA) yourself; nothing secret is stored here. The first login opens a shared
# connection that later calls reuse for 8 hours, so you authenticate once per day, not once per command.
#
# Off campus, connect TTU GlobalProtect VPN first: repacss.ttu.edu is not in public DNS and only answers on TTUnet.
# Inside a GPU session, run GPU programs with:  srun --gres=gpu:nvidia_h100_nvl:1 <program>
set -euo pipefail
HOST=${REPACSS_HOST:-repacss.ttu.edu}
USER_=${REPACSS_USER:-mahmedmohsin}
CTL="$HOME/.ssh/cm-repacss-%r@%h:%p"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6
     -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=8h
     -o PreferredAuthentications=publickey,keyboard-interactive,password
     -o IdentitiesOnly=yes)  # the guide's troubleshooting: too many offered keys fail the login
[[ -f "$HOME/.ssh/id_ed25519" ]] && SSH+=(-i "$HOME/.ssh/id_ed25519")

check() {
  if ! host_ip=$(dscacheutil -q host -a name "$HOST" 2>/dev/null | awk '/ip_address/{print $2; exit}') || [[ -z "$host_ip" ]]; then
    host_ip=$(getent hosts "$HOST" 2>/dev/null | awk '{print $1; exit}' || true)
  fi
  if [[ -z "${host_ip:-}" ]]; then
    echo "cannot resolve $HOST: connect TTU GlobalProtect VPN (and keep your default DNS, not 8.8.8.8)" >&2; return 1
  fi
  if ! nc -z -G 5 "$HOST" 22 2>/dev/null && ! nc -z -w 5 "$HOST" 22 2>/dev/null; then
    echo "$HOST resolves to $host_ip but port 22 does not answer: is the VPN connected?" >&2; return 1
  fi
  echo "$HOST ($host_ip) reachable on port 22"
}

case "${1:-}" in
  --check) check ;;
  --close) exec ssh -o "ControlPath=$CTL" -O exit "$USER_@$HOST" ;;
  gpu)
    check >/dev/null
    echo "starting: interactive -c ${2:-8} -p h100   (then: srun --gres=gpu:nvidia_h100_nvl:1 <program>)"
    exec "${SSH[@]}" -t "$USER_@$HOST" "interactive -c ${2:-8} -p h100" ;;
  "") check >/dev/null; exec "${SSH[@]}" "$USER_@$HOST" ;;
  *) check >/dev/null; exec "${SSH[@]}" "$USER_@$HOST" "$@" ;;
esac
