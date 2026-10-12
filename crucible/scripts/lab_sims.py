"""Stage and run the live simulator checks on the lab node (CPU containers; polite: our own containers and images only).

    python scripts/lab_sims.py stage            # rsync crucible + the OpenEnv Unity server, build images
    python scripts/lab_sims.py mujoco [--ppo-steps 1000000]
    python scripts/lab_sims.py unity
    python scripts/lab_sims.py collect          # pull sims_*.json into docs/research/m12-infra/

Images: ``crucible-sim:v1`` (python 3.12, MuJoCo, Gymnasium, CPU torch, Crucible) and ``crucible-unity:v1`` (OpenEnv's
Unity ML-Agents server, Python 3.10). Containers are named ``crucible_sims_*`` and removed by name.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CRUCIBLE = Path(__file__).resolve().parents[1]
REPO = CRUCIBLE.parent
LAB = [str(REPO / "scripts" / "lab_gpu.sh")]
HOST = "ahmed@100.100.101.36"
REMOTE = "rlenv/crucible_sims"
UNITY_SRC = REPO / "RL-Environments" / "06-frameworks-and-env-code" / "meta-pytorch-openenv" / "envs" / "unity_env"
SIM_DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglfw3 libosmesa6 git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "gymnasium[mujoco]>=1.0" numpy websockets
COPY src /crucible
RUN pip install --no-cache-dir -e /crucible
ENV MUJOCO_GL=osmesa
"""


def sh(cmd: str, check: bool = True) -> str:
    p = subprocess.run([*LAB, cmd], capture_output=True, text=True, check=False)
    if check and p.returncode:
        raise SystemExit(f"lab command failed ({p.returncode}): {cmd}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def stage() -> None:
    sh(f"mkdir -p ~/{REMOTE}/src ~/{REMOTE}/unity ~/{REMOTE}/out")
    subprocess.run(["rsync", "-a", "--delete", "--exclude", ".venv", "--exclude", "data", "--exclude", "__pycache__", "--exclude", ".git", "--exclude", "site",
                    f"{CRUCIBLE}/", f"{HOST}:{REMOTE}/src/"], check=True)
    subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", f"{UNITY_SRC}/", f"{HOST}:{REMOTE}/unity/"], check=True)
    sh(f"cat > ~/{REMOTE}/Dockerfile.sim <<'EOF'\n{SIM_DOCKERFILE}EOF")
    print(sh(f"cd ~/{REMOTE} && docker build -q -f Dockerfile.sim -t crucible-sim:v1 . 2>&1 | tail -3"))
    print(sh(f"cd ~/{REMOTE}/unity && docker build -q -f server/Dockerfile -t crucible-unity:v1 . 2>&1 | tail -3"))
    print(sh("docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep crucible-"))


def run_check(which: str, extra: str = "") -> None:
    name = f"crucible_sims_{which}"
    sh(f"docker rm -f {name} >/dev/null 2>&1 || true", check=False)
    cmd = (f"docker run --rm --name {name} --network host -v $HOME/{REMOTE}/out:/work/out -v $HOME/{REMOTE}/src:/crucible crucible-sim:v1 "
           f"python /crucible/tests/live/sims_check.py {which} --out /work/out {extra}")
    if which == "unity":
        sh("docker rm -f crucible_sims_unity_server >/dev/null 2>&1 || true", check=False)
        sh("docker run -d --rm --name crucible_sims_unity_server -p 8010:8000 -v $HOME/rlenv/crucible_sims/unity_cache:/root/.mlagents-cache crucible-unity:v1")
        sh("for i in $(seq 1 90); do curl -sf http://127.0.0.1:8010/health && break; sleep 2; done")
        try:
            print(sh(cmd, check=False)[-4000:])
        finally:
            print(sh("docker logs --tail 30 crucible_sims_unity_server 2>&1", check=False))
            sh("docker rm -f crucible_sims_unity_server >/dev/null 2>&1 || true", check=False)
    else:
        print(sh(cmd, check=False)[-4000:])


def collect() -> None:
    dest = CRUCIBLE / "docs" / "research" / "m12-infra" / "live"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{HOST}:{REMOTE}/out/", f"{dest}/"], check=True)
    print(sorted(p.name for p in dest.iterdir()))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "stage"
    if what == "stage":
        stage()
    elif what in ("mujoco", "unity"):
        run_check(what, " ".join(sys.argv[2:]))
    elif what == "collect":
        collect()
    else:
        raise SystemExit(__doc__)
