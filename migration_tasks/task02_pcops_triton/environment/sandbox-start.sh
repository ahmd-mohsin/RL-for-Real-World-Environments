#!/usr/bin/env bash
# Container entrypoint: authorize $PUBLIC_KEY (Runpod sets it from the account's
# SSH key) and keep sshd in the foreground. Without a key, idle so `docker exec` works.
set -e
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null 2>&1
# Make the toolchain environment visible to SSH sessions too.
cat > /etc/profile.d/sandbox.sh <<'ENV'
export PATH=/usr/lib/llvm-18/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib CUDA_HOME=/usr/local/cuda OMP_TARGET_OFFLOAD=MANDATORY
ENV
exec /usr/sbin/sshd -D -e
