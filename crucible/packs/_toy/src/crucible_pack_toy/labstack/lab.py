"""Stage / status / collect the lab full-stack campaign on the shared RTX 5090 (polite: waits for 3 consecutive quiet
GPU checks and no other crucible container; exit 75 from the campaign = GPU taken during vLLM start → wait again).

    python -m crucible_pack_toy.labstack.lab stage [--model Qwen/Qwen3-4B-Instruct-2507] [--only ...] [--skip ...]
    python -m crucible_pack_toy.labstack.lab status
    python -m crucible_pack_toy.labstack.lab collect      # → docs/modules/reports/labstack/
"""

from __future__ import annotations

import subprocess
import sys

from crucible_pack_toy import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
REPO = CRUCIBLE.parent
HARBOR = REPO / "RL-Environments" / "06-frameworks-and-env-code" / "harbor-framework-harbor"
LAB = [str(REPO / "scripts" / "lab_gpu.sh")]
HOST = "ahmed@100.100.101.36"
REMOTE = "rlenv/crucible_labstack"
HF_CACHE = "rlenv/crucible_m12/hf"  # shared with the M12 gate: Qwen3-0.6B already cached there
IMAGE = "vllm/vllm-openai:latest"
RUN = r"""#!/bin/bash
cd ~/REMOTE
echo $$ > runner.pid
mkdir -p out ~/HFCACHE
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
  docker run --rm --name crucible_labstack --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network host --ipc host --init \
    -v $HOME/REMOTE/src:/crucible -v $HOME/REMOTE/deps:/deps -v $HOME/REMOTE/out:/work/out -v $HOME/HFCACHE:/root/.cache/huggingface \
    --entrypoint bash IMAGE -c 'python3 -m pip freeze | grep -viE "^(mcp|crucible|harbor)|@|^-e" > /work/constraints.txt; python3 -m venv --system-site-packages /work/venv && /work/venv/bin/pip install -q -c /work/constraints.txt -e /deps/harbor -e "/crucible[mcp,llm,openenv-awm,dev]" -e /crucible/packs/_toy -e /crucible/packs/coding > /work/out/pip.log 2>&1 && /work/venv/bin/python -m crucible_pack_toy.labstack.run ARGS' \
    > out/console.log 2>&1
  RC=$?
  [ "$RC" != 75 ] && break
done
echo "$(date -Is) done rc=$RC" > status.txt
"""


def lab(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run([*LAB, cmd], capture_output=True, text=True, check=False)


def stage(args: str) -> None:
    lab(f"mkdir -p ~/{REMOTE}/src ~/{REMOTE}/deps/harbor ~/{REMOTE}/out")
    # "/data" (the repo's top-level cache), never "data": that also dropped packs/_toy/data and packs/coding/data (first run)
    ex = ["--exclude", ".venv", "--exclude", ".envs", "--exclude", "/data", "--exclude", "__pycache__", "--exclude", ".git"]
    subprocess.run(["rsync", "-a", "--delete", *ex, f"{CRUCIBLE}/", f"{HOST}:{REMOTE}/src/"], check=True)
    for part in ("src", "pyproject.toml", "README.md"):
        subprocess.run(["rsync", "-a", "--delete", *ex, str(HARBOR / part), f"{HOST}:{REMOTE}/deps/harbor/"], check=True)
    run = RUN.replace("REMOTE", REMOTE).replace("HFCACHE", HF_CACHE).replace("IMAGE", IMAGE).replace("ARGS", args)
    subprocess.run([*LAB, f"cat > ~/{REMOTE}/run.sh && chmod +x ~/{REMOTE}/run.sh"], input=run, text=True, check=True)
    print(lab(f"cd ~/{REMOTE} && (kill -0 $(cat runner.pid 2>/dev/null) 2>/dev/null && echo already-running) || (nohup setsid ./run.sh > runner.log 2>&1 < /dev/null &) ; sleep 2; cat status.txt 2>/dev/null").stdout)


def status() -> None:
    print(lab(f"cat ~/{REMOTE}/status.txt; grep '\\[labstack\\]' ~/{REMOTE}/out/console.log 2>/dev/null | tail -20").stdout)


def collect() -> None:
    dest = CRUCIBLE / "docs" / "modules" / "reports" / "labstack"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include", "*.json", "--include", "phase_*.log", "--include", "*/", "--exclude", "*",
                    f"{HOST}:{REMOTE}/out/", str(dest)], check=True)
    print((dest / "labstack.json").read_text()[:3000] if (dest / "labstack.json").exists() else "no report yet")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage":
        stage(" ".join(sys.argv[2:]))
    else:
        {"status": status, "collect": collect}[cmd]()
