#!/usr/bin/env bash
# Connect to the REPACSS cluster at TTU (8 GPU nodes x 4 NVIDIA H100 NVL 94 GB), per https://guide.repacss.org
#
#   scripts/repacss.sh                     login node shell (light work only: files, builds, sbatch)
#   scripts/repacss.sh gpu [CPUS] [HOURS]  interactive session on the h100 partition (default 8 CPUs, 4 h)
#   scripts/repacss.sh cpu [CPUS] [HOURS]  interactive session on the zen4 CPU partition
#   scripts/repacss.sh 'squeue --me'       run one command on the login node
#   scripts/repacss.sh --vpn               open GlobalProtect (portal vpn.ttu.edu) and wait until REPACSS answers
#   scripts/repacss.sh --check             network check only (VPN, DNS, port 22)
#   scripts/repacss.sh --key               one time: create ~/.ssh/repacss and register its public key
#   scripts/repacss.sh --node rpc-91-7     point the `repacss-compute` SSH alias (VS Code Remote-SSH) at your node
#   scripts/repacss.sh --close             close the shared connection
#
# Off campus (and on TTUguest/EduRoam) the TTU GlobalProtect VPN must be connected: repacss.ttu.edu is not in public DNS.
# You type your password and approve Microsoft MFA yourself; nothing secret is stored here. After the first login,
# calls reuse one shared connection for 8 hours.
# Inside a GPU session, launch GPU programs with:  srun --gres=gpu:nvidia_h100_nvl:<N> <program>
set -euo pipefail
HOST=${REPACSS_HOST:-repacss.ttu.edu}
USER_=${REPACSS_USER:-mahmedmohsin}   # your TTU eRaider username (the VPN uses the same account)
KEY=${REPACSS_KEY:-$HOME/.ssh/repacss}
PORTAL=vpn.ttu.edu
CTL="$HOME/.ssh/cm-repacss-%r@%h:%p"
GP_APP=/Applications/GlobalProtect.app

SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6
     -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=8h -o IdentitiesOnly=yes)
[[ -f "$KEY" ]] && SSH+=(-i "$KEY")

resolves() {
  dscacheutil -q host -a name "$HOST" 2>/dev/null | grep -q ip_address || getent hosts "$HOST" >/dev/null 2>&1
}
port_open() { nc -z -G 5 "$HOST" 22 2>/dev/null || nc -z -w 5 "$HOST" 22 2>/dev/null; }

vpn_help() {
  cat >&2 <<EOF
REPACSS is only reachable on TTUnet or through the TTU GlobalProtect VPN.
EOF
  if [[ ! -d "$GP_APP" ]]; then
    cat >&2 <<EOF
GlobalProtect is not installed on this Mac. One-time setup (https://guide.repacss.org/connecting/vpn.html):
  1. Make sure your TTU eRaider account has Microsoft MFA enabled (https://guide.repacss.org/connecting/mfa.html).
  2. On TTU's VPN software page, submit the VPN assistance request form:
       Type of Assistance: Enable
       Justification:      Need TTUnet VPN to access REPACSS TTU Cluster
  3. Download the macOS GlobalProtect .pkg from the TTU Software Distribution Portal, install it, grant the permissions.
  4. Run: scripts/repacss.sh --vpn
Problems with the eRaider account or MFA: ithelpcentral@ttu.edu, 806-742-4357.
EOF
  else
    echo "Open GlobalProtect from the menu bar, portal: $PORTAL, sign in with eRaider + MFA (or run: scripts/repacss.sh --vpn)." >&2
  fi
  echo "Keep your default DNS (not 8.8.8.8): $HOST is only in TTU's DNS." >&2
}

check() {
  if ! resolves; then echo "cannot resolve $HOST" >&2; vpn_help; return 1; fi
  if ! port_open; then echo "$HOST resolves but port 22 does not answer: is the VPN connected?" >&2; vpn_help; return 1; fi
  echo "$HOST reachable on port 22"
}

vpn() {
  if resolves && port_open; then echo "already reachable: $HOST"; return 0; fi
  [[ -d "$GP_APP" ]] || { vpn_help; return 1; }
  open -a GlobalProtect
  echo "GlobalProtect opened. Portal: $PORTAL. Sign in with your eRaider account and approve the MFA prompt."
  printf "waiting for %s" "$HOST"
  for _ in $(seq 1 60); do
    if resolves && port_open; then echo; echo "connected: $HOST reachable"; return 0; fi
    printf "."; sleep 3
  done
  echo; echo "still not reachable after 3 minutes" >&2; vpn_help; return 1
}

interactive_session() {  # $1 partition, $2 cpus, $3 hours
  local hours=${3:-4}
  check >/dev/null || exit 1
  echo "requesting: interactive -p $1 -c ${2:-8} -t $(printf '%02d' "$hours"):00:00 --immediate=600"
  [[ $1 == h100 ]] && echo "on the node: srun --gres=gpu:nvidia_h100_nvl:1 <program>   (hostname gives the node for --node)"
  exec "${SSH[@]}" -t "$USER_@$HOST" "interactive -p $1 -c ${2:-8} -t $(printf '%02d' "$hours"):00:00 --immediate=600"
}

setup_key() {
  if [[ ! -f "$KEY" ]]; then
    echo "creating $KEY (choose a passphrase when asked)"
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    ssh-keygen -t ed25519 -f "$KEY" -C "repacss-$USER_"
  fi
  check >/dev/null || exit 1
  echo "registering $KEY.pub on $HOST (asks for your REPACSS password and MFA)"
  ssh-copy-id -i "$KEY.pub" -o IdentitiesOnly=yes "$USER_@$HOST" || \
    "${SSH[@]}" "$USER_@$HOST" 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys' < "$KEY.pub"
  ssh-add --apple-use-keychain "$KEY" 2>/dev/null || ssh-add "$KEY" || true
  echo "test: ssh -i $KEY $USER_@$HOST"
}

set_node() {  # managed block in ~/.ssh/config: repacss (login, jump host) + repacss-compute (the allocated node)
  local node=$1 cfg="$HOME/.ssh/config"
  [[ $node =~ ^rpc-[0-9]+-[0-9]+$ ]] || { echo "node names look like rpc-91-7 (run 'hostname' in your interactive session)" >&2; exit 1; }
  touch "$cfg" && cp "$cfg" "$cfg.bak.$(date +%Y%m%d%H%M%S)"
  local tmp; tmp=$(mktemp)
  awk '/^# >>> repacss \(scripts\/repacss.sh\)/{skip=1} !skip{print} /^# <<< repacss/{skip=0}' "$cfg" > "$tmp"
  cat >> "$tmp" <<EOF
# >>> repacss (scripts/repacss.sh) >>>
Host repacss
    HostName $HOST
    User $USER_
    IdentityFile $KEY
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath $CTL
    ControlPersist 8h

Host repacss-compute
    HostName $node
    User $USER_
    IdentityFile $KEY
    IdentitiesOnly yes
    ProxyJump repacss
# <<< repacss <<<
EOF
  mv "$tmp" "$cfg" && chmod 600 "$cfg"
  echo "repacss-compute -> $node. Test from a second terminal: ssh repacss-compute hostname"
  echo "VS Code: Remote-SSH: Connect to Host... -> repacss-compute. Keep the interactive session's terminal open."
}

case "${1:-}" in
  --check) check ;;
  --vpn) vpn ;;
  --key) setup_key ;;
  --node) set_node "${2:?usage: scripts/repacss.sh --node rpc-91-7}" ;;
  --close) exec ssh -o "ControlPath=$CTL" -O exit "$USER_@$HOST" ;;
  gpu) interactive_session h100 "${2:-8}" "${3:-4}" ;;
  cpu) interactive_session zen4 "${2:-8}" "${3:-4}" ;;
  -h|--help) sed -n '2,20p' "$0" ;;
  "") check >/dev/null || exit 1; exec "${SSH[@]}" "$USER_@$HOST" ;;
  *) check >/dev/null || exit 1; exec "${SSH[@]}" "$USER_@$HOST" "$@" ;;
esac
