#!/usr/bin/env bash
# Container entrypoint. With $PUBLIC_KEY and sshd available, serve SSH; otherwise idle
# so that the harness can `docker exec` into the sandbox.
set -e
cat > /etc/profile.d/sandbox.sh <<'ENV'
export PATH=/usr/local/cuda/bin:$PATH
ENV
if [ -n "${PUBLIC_KEY:-}" ] && command -v sshd >/dev/null 2>&1; then
  mkdir -p /root/.ssh && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1
  exec /usr/sbin/sshd -D -e
fi
exec sleep infinity
