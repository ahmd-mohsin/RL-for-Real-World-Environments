"""One Crucible rollout through the Apptainer World, graded in a fresh container, with a receipt (Marlowe smoke).

    python rollout_smoke.py --policy scripted --out $RUN                                 # login node, CPU: the World path
    python rollout_smoke.py --policy vllm --base-url http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct --out $RUN                               # the GPU job (serve_and_rollout.sh)

The task is a toy repository (``calc.py``: ``add`` subtracts, ``mean([])`` raises) seeded as the window lane seeds a World:
a ``common/git_repo`` App on the host (one commit tagged ``base``) whose shell, file and git tools run in an Apptainer
instance (``--containall --cleanenv --no-home --net --network none``) binding the workspace at its host path
(``world.backends.apptainer.open_workspace_world``). The slime ``Sandbox`` view (``ExecSandbox``) of the same instance
runs the egress probe. The patch is collected from the host tree (``git diff base``) and graded by hidden tests in a fresh
container of the same image (``run_once``, no network, workspace copy bound read-only). Receipt: ``<out>/rollout.receipt.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CALC = "def add(a, b):\n    return a - b\n\n\ndef mean(xs):\n    return sum(xs) / len(xs)\n"
VISIBLE = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
HIDDEN = ("from calc import add, mean\n\n\ndef test_add():\n    assert add(2, 3) == 5 and add(-1, 1) == 0\n\n\n"
          "def test_mean():\n    assert mean([1, 2, 3]) == 2\n\n\ndef test_mean_empty():\n    assert mean([]) == 0.0\n")
INSTRUCTION = ("The repository in your workspace has a small module calc.py. add(2, 3) must return 5, and mean([]) must return 0.0 "
               "instead of raising. Fix calc.py (do not edit the tests), run `python -m pytest -q` to check, then reply with a one-line summary.")


def seed(base: Path) -> None:
    (base / "tests").mkdir(parents=True, exist_ok=True)
    (base / "calc.py").write_text(CALC)
    (base / "tests" / "test_calc.py").write_text(VISIBLE)


def scripted_policy():
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Done, ToolCall

    return ScriptedPolicy("agent", [
        ToolCall(tool_call_id="c1", function_name="repo.read_file", arguments={"path": "calc.py"}),
        ToolCall(tool_call_id="c2", function_name="repo.edit_file", arguments={"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}),
        ToolCall(tool_call_id="c3", function_name="repo.edit_file",
                 arguments={"path": "calc.py", "old_string": "return sum(xs) / len(xs)", "new_string": "return sum(xs) / len(xs) if xs else 0.0"}),
        ToolCall(tool_call_id="c4", function_name="repo.shell", arguments={"command": "python -m pytest -q -p no:cacheprovider"}),
        Done(reason="fixed add and mean"),
    ])


def vllm_policy(base_url: str, model: str, data: Path):
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import LLMToolPolicy
    from crucible.core.config import Config, LLMProvider

    cfg = Config(data_dir=str(data), llm_providers={"marlowe": LLMProvider(base_url=base_url, api_key="EMPTY")}, models={"policy": f"marlowe:{model}"})
    client = LLMClient("policy", config=cfg, cache_mode="off", timeout_s=300)
    return LLMToolPolicy("agent", client, max_tokens=1024, temperature=0.0), client


def sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


async def run(a: argparse.Namespace) -> dict:
    from crucible.actors.policy import EpisodeLimits, run_episode
    from crucible.factory.lanes.windows_world import world_spec
    from crucible.runtime.harness.sandbox import ExecSandbox
    from crucible.world.backends.apptainer import open_workspace_world, run_once

    out = Path(a.out).resolve()
    work = out / "rollout"
    shutil.rmtree(work, ignore_errors=True)
    base, ws = work / "base", work / "ws"
    seed(base)
    image = str(Path(a.image).resolve()) if not a.image.startswith("docker://") else a.image
    receipt: dict = {"task": "marlowe/toy-calc", "policy": a.policy, "model": a.model if a.policy == "vllm" else "scripted", "image": image,
                     "image_sha256": Path(image).with_suffix(".sha256").read_text().strip() if Path(image).with_suffix(".sha256").exists() else None,
                     "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    t0 = time.time()
    world = await open_workspace_world(world_spec(ws, base), image=image, workspace=ws, workdir=work / "world", instruction=INSTRUCTION,
                                       env={"PYTHONPATH": str(ws)}, configdir=os.environ.get("APPTAINER_CONFIGDIR"), limits={"pids": 1024})
    sb = world._crucible_sandbox
    receipt["sandbox"] = sb.description
    receipt["open_s"] = round(time.time() - t0, 2)
    client = None
    try:
        receipt["isolation"] = await sb.isolation_report(must_hide=[str(ROOT), sys.prefix, os.environ.get("HF_HOME", "/nonexistent")])
        box = ExecSandbox(sb.exec, root=str(ws), sandbox_id=sb.name, alive=sb.alive)
        receipt["egress_probe"] = await box.egress_probe()
        _, git_log, _ = await box.exec("git log --oneline --all | head -3", timeout=60)
        receipt["seeded_history"] = git_log.strip().splitlines()
        if a.policy == "scripted":
            policy = scripted_policy()
        else:
            policy, client = vllm_policy(a.base_url, a.model, out / "llm-data")
        t1 = time.time()
        ep = await run_episode(world, {"agent": policy}, limits=EpisodeLimits(max_steps=a.max_steps, timeout_s=a.timeout_s))
        receipt["episode"] = {"steps": ep.steps, "stop": ep.stop_condition, "masked": ep.masked, "failures": ep.failures[:5], "seconds": round(time.time() - t1, 1),
                              "events": [f"{e.kind}:{e.type}" for e in ep.trace.events][:60]}
        if client is not None:
            receipt["llm"] = {"model_calls": sum(len(v) for v in ep.model_calls.values()), "spend": vars(client.spend) if hasattr(client.spend, "__dict__") else str(client.spend)}
        receipt["sandbox_alive_after_episode"] = await sb.alive()
    finally:
        await world.close()
    receipt["sandbox_stopped"] = not await sb.alive()
    patch = (await asyncio.to_thread(subprocess.run, ["git", "-C", str(ws), "diff", "base", "--", ".", ":(exclude)tests"], capture_output=True, text=True,
                                     check=False)).stdout
    receipt["patch"] = patch[-4000:]
    receipt["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
    # grading: a pristine copy of the base, the patch applied on the host, the hidden tests added; run in a fresh container
    with tempfile.TemporaryDirectory(prefix="crucible-grade-", dir=str(work)) as td:
        g = Path(td) / "g"
        shutil.copytree(base, g)
        ok_apply = True
        if patch.strip():
            ap_ = await asyncio.to_thread(subprocess.run, ["git", "apply", "--whitespace=nowarn", "-"], input=patch, text=True, cwd=g, capture_output=True,
                                        check=False)
            ok_apply = ap_.returncode == 0
        (g / "test_hidden.py").write_text(HIDDEN)
        t2 = time.time()
        gr = await run_once(image, "python -m pytest -q -p no:cacheprovider test_hidden.py", binds=[(str(g), "/g", "ro")], cwd="/g",
                            env={"PYTHONPATH": "/g", "PYTHONDONTWRITEBYTECODE": "1"}, timeout_s=300)
        receipt["grade"] = {"patch_applied": ok_apply, "rc": gr.rc, "tail": (gr.stdout + gr.stderr)[-800:], "seconds": round(time.time() - t2, 2)}
    receipt["reward"] = 1.0 if (ok_apply and gr.rc == 0) else 0.0
    receipt["seconds"] = round(time.time() - t0, 1)
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["scripted", "vllm"], default="scripted")
    ap.add_argument("--base-url", dest="base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--image", default=os.path.join(os.environ.get("CRU_SIF", "/scratch/m000159/muahmed/sif"), "world-py312.sif"))
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=24)
    ap.add_argument("--timeout-s", dest="timeout_s", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-reward", dest="require_reward", type=float, default=None, help="exit 1 below this reward (default: scripted 1.0, vllm none)")
    a = ap.parse_args()
    receipt = asyncio.run(run(a))
    path = Path(a.out) / f"rollout-{a.policy}.receipt.json"
    path.write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({k: receipt.get(k) for k in ("policy", "model", "reward", "seconds", "sandbox", "episode", "grade")}, indent=1, default=str))
    need = a.require_reward if a.require_reward is not None else (1.0 if a.policy == "scripted" else None)
    iso_ok = receipt.get("isolation", {}).get("ok") and receipt.get("egress_probe", {}).get("ok")
    if not iso_ok or not receipt.get("sandbox_stopped") or (need is not None and receipt["reward"] < need):
        sys.exit(1)


if __name__ == "__main__":
    main()
