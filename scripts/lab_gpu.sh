#!/usr/bin/env bash
# Connect to the lab RTX 5090 node over Tailscale (key-based SSH).
#
#   scripts/lab_gpu.sh                 interactive shell
#   scripts/lab_gpu.sh 'nvidia-smi'    run one command
#   scripts/lab_gpu.sh --setup         one-time: install ~/.ssh/id_ed25519.pub (asks for the password)
#   scripts/lab_gpu.sh --check         GPU / toolchain summary
#
# Requires Tailscale to be up on this machine. No password is stored here.
set -euo pipefail
HOST=${LAB_GPU_HOST:-100.100.101.36}       # stanford.tailedb112.ts.net
USER_=${LAB_GPU_USER:-ahmed}
KEY=${LAB_GPU_KEY:-$HOME/.ssh/id_ed25519}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$USER_@$HOST")

if ! nc -z -G 5 "$HOST" 22 2>/dev/null && ! nc -z -w 5 "$HOST" 22 2>/dev/null; then
  echo "cannot reach $HOST:22 - is Tailscale connected?" >&2; exit 1
fi
case "${1:-}" in
  --setup) exec ssh-copy-id -i "$KEY.pub" -o StrictHostKeyChecking=accept-new "$USER_@$HOST" ;;
  --check) exec "${SSH[@]}" 'hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nproc; free -g | sed -n 2p; df -h ~ | tail -1; for t in docker nvcc python3 clang++-18; do printf "%-10s %s\n" "$t" "$(command -v $t || echo missing)"; done' ;;
  "") exec "${SSH[@]}" ;;
  *) exec "${SSH[@]}" "$@" ;;
esac
