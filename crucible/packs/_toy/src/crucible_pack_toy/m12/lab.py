"""Stage / run / collect the M12 GPU gate on the shared lab RTX 5090 (polite: one GPU job at a time, waits for an idle
GPU and for no other crucible container, never pushes another user's job out).

    python -m crucible_pack_toy.m12.lab stage      # rsync the source, write run.sh, start it (it waits for an idle GPU)
    python -m crucible_pack_toy.m12.lab status
    python -m crucible_pack_toy.m12.lab collect    # pull m12_gpu_gate.json into docs/modules/reports/
"""

from __future__ import annotations

import subprocess
import sys

from crucible_pack_toy import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
REPO = CRUCIBLE.parent
LAB = [str(REPO / "scripts" / "lab_gpu.sh")]
HOST = "ahmed@100.100.101.36"
REMOTE = "rlenv/crucible_m12"
IMAGE = "vllm/vllm-openai:latest"
RUN = r"""#!/bin/bash
# Idle = GPU <= 10 % and <= 3000 MiB with no crucible container, on 3 consecutive checks 2 min apart (a single quiet
# sample can be a pause in another user's job). Exit 75 from the gate = the GPU was taken while vLLM started: wait again.
cd ~/REMOTE
echo $$ > runner.pid
mkdir -p out hf
for ATTEMPT in $(seq 1 30); do
  QUIET=0
  while [ "$QUIET" -lt 3 ]; do
    read U M <<< $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
    OTHERS=$(docker ps --format '{{.Names}}' | grep -c '^crucible' || true)
    if [ "$U" -le 10 ] && [ "$M" -le 3000 ] && [ "$OTHERS" = 0 ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
    echo "$(date -Is) waiting (attempt $ATTEMPT, quiet $QUIET/3): gpu ${U}% ${M}MiB crucible-containers $OTHERS" > status.txt
    [ "$QUIET" -lt 3 ] && sleep 120
  done
  echo "$(date -Is) running (attempt $ATTEMPT)" > status.txt
  docker run --rm --name crucible_m12_gate --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network host --ipc host \
    -v $HOME/REMOTE/src:/crucible -v $HOME/REMOTE/out:/work/out -v $HOME/REMOTE/hf:/root/.cache/huggingface \
    --entrypoint bash IMAGE -c 'pip install -q -e /crucible -e /crucible/packs/_toy > /work/out/pip.log 2>&1 && python3 -m crucible_pack_toy.m12.gate ARGS --out /work/out' \
    > out/console.log 2>&1
  RC=$?  # captured before anything else runs ($(date) in the echo would reset $?)
  [ "$RC" != 75 ] && break
done
echo "$(date -Is) done rc=$RC" > status.txt
"""


def lab(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run([*LAB, cmd], capture_output=True, text=True, check=False)


def stage(args: str = "--steps 20") -> None:
    lab(f"mkdir -p ~/{REMOTE}/src")
    subprocess.run(["rsync", "-a", "--delete", "--exclude", ".venv", "--exclude", ".envs", "--exclude", "data", "--exclude", "__pycache__", "--exclude", ".git",
                    f"{CRUCIBLE}/", f"{HOST}:{REMOTE}/src/"], check=True)
    run = RUN.replace("REMOTE", REMOTE).replace("IMAGE", IMAGE).replace("ARGS", args)
    subprocess.run([*LAB, f"cat > ~/{REMOTE}/run.sh && chmod +x ~/{REMOTE}/run.sh"], input=run, text=True, check=True)
    print(lab(f"cd ~/{REMOTE} && (nohup ./run.sh > runner.log 2>&1 &) ; sleep 2; cat status.txt 2>/dev/null").stdout)


def status() -> None:
    print(lab(f"cat ~/{REMOTE}/status.txt; tail -3 ~/{REMOTE}/out/console.log 2>/dev/null").stdout)


def collect() -> None:
    dest = CRUCIBLE / "docs" / "modules" / "reports" / "m12_gpu_gate.json"
    subprocess.run(["rsync", "-a", f"{HOST}:{REMOTE}/out/m12_gpu_gate.json", str(dest)], check=True)
    print(dest.read_text()[:2000])


if __name__ == "__main__":
    {"stage": lambda: stage(" ".join(sys.argv[2:]) or "--steps 20"), "status": status, "collect": collect}[sys.argv[1]]()
