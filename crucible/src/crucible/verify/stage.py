"""Separate verifier environment (DESIGN §8.3 layer 4): stage the kit runtime and a plan into a verifier image, grade,
and read the result back (host side).

``bundle(dest, plans)`` writes a self-contained directory: stub ``crucible/__init__.py`` and ``crucible/verify/__init__.py``
(so importing the runtime never pulls pydantic or the rest of Crucible), the standard-library runtime, and each plan
module at its package path. It is mounted read-only into the verifier container, which runs ``run.py`` as root with
``--network none``; hidden material (tests, references) is mounted only into this container, after the agent finished.

``docker_grade`` runs one grade and returns the parsed ``reward.json`` and ``report.json``. ``reward_json`` validates
the result against ``crucible.core.schema.reward.RewardJson``.

Built on: Harbor's separate verifier environment (``environment_mode="separate"``), verifiers ``IsolatedVerifierEnv``,
NeMo-Gym's second clean grading sandbox (``swemer_oml``); and our seed tasks' ``run_grade.sh``.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from crucible.core.schema.reward import RewardJson

RUNTIME = Path(__file__).resolve().parent / "runtime"
MOUNT = "/crucible_verify"


def bundle(dest: Path, plans: dict[str, Path] | None = None) -> Path:
    """``plans`` maps a dotted module name (``crucible_pack_coding.verify.task04``) to its source file."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "crucible" / "verify").mkdir(parents=True)
    (dest / "crucible" / "__init__.py").write_text("")
    (dest / "crucible" / "verify" / "__init__.py").write_text("")
    shutil.copytree(RUNTIME, dest / "crucible" / "verify" / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for mod, src in (plans or {}).items():
        parts = mod.split(".")
        d = dest.joinpath(*parts[:-1])
        d.mkdir(parents=True, exist_ok=True)
        for k in range(1, len(parts)):
            init = dest.joinpath(*parts[:k]) / "__init__.py"
            if not init.exists():
                init.write_text("")
        shutil.copy(src, d / f"{parts[-1]}.py")
    return dest


@dataclass
class Grade:
    reward: dict
    report: dict
    console: str
    rc: int

    @property
    def gates(self) -> dict[str, int]:
        return self.reward.get("gates", {})

    def failing(self) -> list[str]:
        return [g for g, v in self.gates.items() if not v]


def docker_grade(*, image: str, bundle_dir: Path, plan: str, out_dir: Path, mounts: dict[str, str] | None = None, params: dict | None = None,
                 base: str | None = None, patch: str | None = None, candidate: str | None = None, tier: str = "full", gpu: bool = False,
                 docker_host: str | None = None, name: str | None = None, timeout: float = 7200, extra_args: list[str] | None = None) -> Grade:
    """``mounts`` maps host paths to container paths (read-only); ``out_dir`` is mounted at ``/logs/verifier``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = ["docker", "run", "--rm", "--network", "none", "--init", "--shm-size", "8g", "-v", f"{Path(bundle_dir).resolve()}:{MOUNT}:ro",
            "-v", f"{out_dir.resolve()}:/logs/verifier"]
    if name:
        argv += ["--name", name]
    if gpu:
        argv += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    for h, c in (mounts or {}).items():
        argv += ["-v", f"{h}:{c}:ro"]
    argv += ["--entrypoint", "python3", image, "-I", f"{MOUNT}/crucible/verify/runtime/run.py", "--plan", plan, "--out", "/logs/verifier",
             "--tier", tier, "--require-root"]
    if base:
        argv += ["--base", base]
    if patch:
        argv += ["--patch", patch]
    if candidate:
        argv += ["--candidate", candidate]
    for k, v in (params or {}).items():
        argv += ["--param", f"{k}={json.dumps(v) if not isinstance(v, str) else v}"]
    argv += extra_args or []
    env = None
    if docker_host:
        import os

        env = {**os.environ, "DOCKER_HOST": docker_host}
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    rw = out_dir / "crucible_reward.json"
    reward = json.loads(rw.read_text()) if rw.exists() else {"reward": 0.0, "gates": {}, "infra": {"status": "masked", "failure_kind": "no_reward_json"}}
    rp = out_dir / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    return Grade(reward, report, (p.stdout + p.stderr)[-20000:], p.returncode)


def reward_json(g: Grade) -> RewardJson:
    r = dict(g.reward)
    r.setdefault("gates", {})
    return RewardJson.model_validate({k: v for k, v in r.items() if k in RewardJson.model_fields})


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
