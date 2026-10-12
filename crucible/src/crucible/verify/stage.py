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


def _masked(kind: str, reason: str) -> dict:
    # the same document the in-container grader writes when masked: ``reward`` null, never a 0 (M7 audit)
    return {"reward": None, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": kind, "reason": reason[:500]}}


def image_id(image: str, docker_host: str | None = None) -> str | None:
    import os

    env = {**os.environ, "DOCKER_HOST": docker_host} if docker_host else None
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, env=env, check=False)
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def docker_grade(*, image: str, bundle_dir: Path, plan: str, out_dir: Path, mounts: dict[str, str] | None = None, params: dict | None = None,
                 base: str | None = None, patch: str | None = None, candidate: str | None = None, tier: str = "full", gpu: bool = False,
                 docker_host: str | None = None, name: str | None = None, timeout: float = 7200, extra_args: list[str] | None = None,
                 expected_image: str | None = None, pre_patch: str | None = None) -> Grade:
    """``mounts`` maps host paths to container paths (read-only); ``out_dir`` is mounted at ``/logs/verifier``.

    ``expected_image``: the image id the task pins. A different id is a stale verifier image (Appendix B.5): the grade is
    masked as ``stale image`` before any grading process starts. A grade that exceeds ``timeout`` kills its (named)
    container and is masked as ``verifier_timeout``; a run that leaves no reward document is masked ``no_reward_json``.
    """
    import os
    import uuid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "DOCKER_HOST": docker_host} if docker_host else None
    if expected_image is not None:
        got = image_id(image, docker_host)
        if got != expected_image:
            return Grade(_masked("check_infra", f"stale image: {image} is {got}, the task pins {expected_image}"), {}, "", -1)
    dh = docker_host or os.environ.get("DOCKER_HOST", "")
    if dh and dh not in ("unix:///var/run/docker.sock", "unix:///run/docker.sock"):
        # a remote daemon resolves bind-mount sources on ITS host: unless the bundle was staged there at the same path, the
        # verifier sees empty mounts and leaves no reward (found live: 22 red-team grades masked as no_reward_json, a
        # misleading reason). Probe once and name the real cause.
        probe = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{Path(bundle_dir).resolve()}:/probe:ro", "--entrypoint", "sh", image, "-c",
                                "test -e /probe/crucible/verify/runtime/run.py"], env=env, capture_output=True, text=True, timeout=300, check=False)
        if probe.returncode != 0:
            return Grade(_masked("check_infra", f"bind-mount sources are not on the docker host {dh}: stage the bundle there at the same path "
                                 f"(or grade on that host's local daemon)"), {}, probe.stderr[-500:], -1)
    name = name or f"crucible-grade-{uuid.uuid4().hex[:12]}"
    argv = ["docker", "run", "--rm", "--network", "none", "--init", "--shm-size", "8g", "--name", name, "--label", "crucible.sandbox=1",
            "-v", f"{Path(bundle_dir).resolve()}:{MOUNT}:ro", "-v", f"{out_dir.resolve()}:/logs/verifier"]
    if gpu:
        argv += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    for h, c in (mounts or {}).items():
        argv += ["-v", f"{h}:{c}:ro"]
    argv += ["--entrypoint", "python3", image, "-I", f"{MOUNT}/crucible/verify/runtime/run.py", "--plan", plan, "--out", "/logs/verifier",
             "--tier", tier, "--require-root"]
    if base:
        argv += ["--base", base]
    if pre_patch:
        argv += ["--pre-patch", pre_patch]
    if patch:
        argv += ["--patch", patch]
    if candidate:
        argv += ["--candidate", candidate]
    for k, v in (params or {}).items():
        argv += ["--param", f"{k}={json.dumps(v)}"]  # always JSON: "true" stays a string (M7 audit V21)
    argv += extra_args or []
    from crucible.verify.runtime.plan import REWARD_FILES

    for f in REWARD_FILES:  # a previous grade's files must never be read as this one's (M7 audit V9)
        (out_dir / f).unlink(missing_ok=True)
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired as ex:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=env, check=False)
        out = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        return Grade(_masked("verifier_timeout", f"grade exceeded {timeout:.0f}s; container {name} removed"), {}, out[-20000:], -9)
    finally:  # a dropped connection to a remote daemon can leave the container running and holding the GPU (M7 audit)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=env, check=False, timeout=120)
    rw = out_dir / "crucible_reward.json"
    try:
        reward = json.loads(rw.read_text()) if rw.exists() else _masked(
            "verifier_not_started" if p.returncode in (125, 126, 127) else "no_reward_json", "the verifier left no reward document")
    except (ValueError, OSError) as e:
        reward = _masked("unreadable_reward_json", f"crucible_reward.json could not be read: {e}"[:300])
    rp = out_dir / "report.json"
    try:
        report = json.loads(rp.read_text()) if rp.exists() else {}
    except (ValueError, OSError):
        report = {}
    return Grade(reward, report, (p.stdout + p.stderr)[-20000:], p.returncode)


def reward_json(g: Grade) -> RewardJson:
    r = dict(g.reward)
    r.setdefault("gates", {})
    if r.get("reward") is None:  # masked: no reward (RewardJson.reward is None exactly when masked)
        r["reward"] = None
    return RewardJson.model_validate({k: v for k, v in r.items() if k in RewardJson.model_fields})


def shell_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
