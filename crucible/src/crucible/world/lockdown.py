"""The lockdown probe for Worlds on a shared-uid host (DeltaAI: no subuid range, code inside runs as the invoking user;
doc 14-deltaai §3).

``PROBE`` is a bash script run *inside* a container exactly as an episode's World is started (through the Apptainer
``docker`` shim, ``scripts/deltaai/docker_shim.py``). It prints one ``key=value`` per line: what the policy could reach.
``verdict(report)`` lists every problem; ``require(report)`` raises ``LockdownFailed`` with them, and the DeltaAI episode
runner calls it before the first model episode (no lockdown, no episodes).

What it checks, and why each matters for model-written code running as the user:
- **network:** a TCP connect to two public addresses and a DNS lookup must fail, and the only interface is loopback
  (exfiltration, downloading answers);
- **secrets:** the host home, ``~/.ssh``, ``~/.hf_token``, ``/u``, ``/projects``, ``/work`` and ``/sw`` must not be
  visible, nor the munge socket (``/run/munge``: with it, code could forge Slurm credentials and submit jobs on the
  account);
- **environment:** no ``SLURM_*``, token, Kerberos, ssh-agent or munge variables;
- **devices:** no Slingshot (``cxi*``) or InfiniBand device (a network path that ``--network none`` does not cover), and
  no GPU unless the World asked for one;
- **processes:** a private PID namespace (no ``slurmstepd`` or other host process visible);
- **mounts:** no Lustre, NFS or GPFS mount, nothing under the host's project or work trees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

PROBE = r"""#!/bin/bash
H="$1"; GPU_OK="$2"
kv() { printf '%s=%s\n' "$1" "$2"; }
tcp() { timeout 5 bash -c "exec 3<>/dev/tcp/$1/$2" >/dev/null 2>&1 && echo open || echo blocked; }
kv net_tcp_1_1_1_1_443 "$(tcp 1.1.1.1 443)"
kv net_tcp_8_8_8_8_53 "$(tcp 8.8.8.8 53)"
kv net_dns "$( (getent hosts huggingface.co || getent hosts github.com) >/dev/null 2>&1 && echo resolves || echo blocked)"
kv net_ifaces "$(awk -F: 'NR>2{gsub(/ /,"",$1); printf "%s,", $1}' /proc/net/dev)"
for p in "$H" "$H/.ssh" "$H/.hf_token" /u /projects /work /sw /run/munge /var/run/munge /run/munge/munge.socket.2 "$H/.apptainer"; do
  if [ -e "$p" ]; then
    if ls "$p" >/dev/null 2>&1 || cat "$p" >/dev/null 2>&1; then kv "path:$p" readable; else kv "path:$p" exists; fi
  else kv "path:$p" absent; fi
done
kv env_sensitive "$(env | grep -Ec '^(SLURM_|HF_TOKEN|HUGGING|TINKER|WANDB|SSH_AUTH|KRB5|MUNGE|AWS_|GH_TOKEN|GITHUB_TOKEN)')"
kv dev_fabric "$(ls /dev 2>/dev/null | grep -Ec '^(cxi|infiniband|hfi|uverbs)')"
kv dev_gpu "$(ls /dev 2>/dev/null | grep -Ec '^nvidia[0-9]')"
kv gpu_ok "$GPU_OK"
kv pids_visible "$(ls /proc | grep -Ec '^[0-9]+$')"
kv host_procs "$(cat /proc/[0-9]*/comm 2>/dev/null | grep -Ec '^(slurmstepd|slurmd|sshd|systemd|munged)$')"
kv mounts_shared "$(grep -Ec ' (lustre|nfs|nfs4|gpfs|cifs|fuse.sshfs) ' /proc/mounts)"
kv mounts_host_trees "$(awk '{print $2}' /proc/mounts | grep -Ec '^/(u|projects|work|sw)(/|$)')"
kv uid "$(id -u)"
kv uid_map "$(tr -s ' ' < /proc/self/uid_map | tr '\n' ';')"
kv root_writable "$(touch /.crucible_probe 2>/dev/null && rm -f /.crucible_probe && echo yes || echo no)"
kv tmp_writable "$(t=/tmp/.crucible_probe_$$; dd if=/dev/zero of=$t bs=1M count=128 2>/dev/null && rm -f $t && echo yes || echo no)"
echo probe_done=1
"""

SECRET_PATHS = ("/.ssh", "/.hf_token", "/u", "/projects", "/work", "/sw", "/run/munge", "/var/run/munge", "/run/munge/munge.socket.2", "/.apptainer")


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def verdict(report: Mapping[str, str], *, home: str) -> list[str]:
    """Every lockdown problem in a parsed probe report (empty: locked down)."""
    r = dict(report)
    bad: list[str] = []
    if r.get("probe_done") != "1":
        return ["the probe did not finish (no report): the World could not be checked"]
    for k in ("net_tcp_1_1_1_1_443", "net_tcp_8_8_8_8_53"):
        if r.get(k) != "blocked":
            bad.append(f"network: {k} is {r.get(k)!r}")
    if r.get("net_dns") != "blocked":
        bad.append("network: DNS resolves")
    ifaces = [i for i in (r.get("net_ifaces") or "").split(",") if i]
    if any(i != "lo" for i in ifaces):
        bad.append(f"network: interfaces besides loopback: {ifaces}")
    for k, v in r.items():
        if k.startswith("path:") and v != "absent":
            p = k[5:]
            if p == home or p.startswith(home + "/") or p in SECRET_PATHS or any(p.endswith(s) for s in SECRET_PATHS):
                bad.append(f"secret path visible: {p} ({v})")
    if r.get("env_sensitive", "0") != "0":
        bad.append(f"environment: {r.get('env_sensitive')} sensitive variables visible")
    if r.get("dev_fabric", "0") != "0":
        bad.append(f"devices: {r.get('dev_fabric')} fabric devices (Slingshot/InfiniBand) visible")
    if r.get("gpu_ok") != "1" and r.get("dev_gpu", "0") != "0":
        bad.append("devices: a GPU is visible in a World that did not ask for one")
    if r.get("host_procs", "0") != "0":
        bad.append(f"processes: {r.get('host_procs')} host daemons visible (no private PID namespace)")
    if r.get("mounts_shared", "0") != "0":
        bad.append(f"mounts: {r.get('mounts_shared')} shared filesystem mounts (Lustre/NFS/GPFS) visible")
    if r.get("mounts_host_trees", "0") != "0":
        bad.append(f"mounts: {r.get('mounts_host_trees')} mounts under /u, /projects, /work or /sw")
    if r.get("tmp_writable") != "yes":
        bad.append("workspace: /tmp is not writable for 128 MiB (builds would fail: infrastructure, not the policy)")
    return bad


class LockdownFailed(RuntimeError):
    def __init__(self, problems: Sequence[str]):
        super().__init__("the World lockdown failed: " + "; ".join(problems))
        self.problems = list(problems)


def require(report: Mapping[str, str], *, home: str) -> None:
    probs = verdict(report, home=home)
    if probs:
        raise LockdownFailed(probs)
