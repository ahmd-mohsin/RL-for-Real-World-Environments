"""Builder Φ: the checkpoint sidecar on a builder's session (swarm-pipeline/13 §3; the student's Φ is factory/checkpoint.py).

A builder session (the oracle builder, the verifier writer, the prompt writer) is a long-horizon episode like a student's,
on the same RLM harness. Its potential is the share of its artifact's milestones that already pass **the gate the artifact
will face**, computed on a snapshot of its worktree out of band, so the session journal shows which builder turns made
progress, and the per-call ΔΦ is the same credit format a student episode has (``progress.jsonl``, ``assign="close"``):

| potential | the builder | a milestone passes when | the gate it mirrors |
|---|---|---|---|
| ``prompt_draft`` | prompt writer | its requirement line exists in the draft JSON, says enough (``min_chars``) and passes the lane's lint (no faulty names, hidden case names, hashes or versions) | S7 (``generators.debug.lint_debug``) |
| ``hidden_tests`` | verifier writer | its ``crucible_hidden/test_<mid>.py`` exists, parses, defines a test and reads no source structure | S5's static half (``generators.swarm.test_problems``) |
| ``oracle_commits`` | oracle builder | the commit for it exists, in plan order, subject ``mK: ...`` | S4's structural half (``generators.swarm.oracle_problems``) |

These are the artifact's *static* gates: the executed ones (hidden tests on A and on the oracle, the World, S6/S5/S8)
still decide afterwards (V6), and Φ never reaches the builder (the sidecar's rule: nothing measured travels back). A
milestone's status is computed on the base tree plus the snapshot's patch in a scratch copy, never in the live worktree.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _git(cwd: Path, *args: str, input: str | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, input=input, capture_output=True, text=True, check=False, timeout=300)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.strip()[:300]}")
    return r.stdout


class ArtifactPotential:
    """A ``factory.checkpoint.MilestoneTask`` over a builder's worktree. ``status(tree) -> {milestone id: bool}`` runs on a
    scratch copy of the base commit with the snapshot's patch applied. ``snapshot(world)`` (optional) replaces the
    sidecar's default snapshot (the worktree's diff against ``base``)."""

    def __init__(self, task_id: str, workspace: Path | str, milestone_ids: Sequence[str], status: Callable[[Path, str], Mapping[str, bool]], *,
                 requires: Mapping[str, Sequence[str]] | None = None, base: str = "base"):
        self.id, self.root, self.base = task_id, Path(workspace), base
        self.milestones = [{"id": m, "requires": list((requires or {}).get(m) or [])} for m in milestone_ids]
        self.requirement_ids = list(milestone_ids)  # STATUS.json claims are scored per milestone (honesty)
        self._status = status

    def workspace(self, world: Any) -> Path:
        return self.root

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        return await asyncio.to_thread(self._sync_status, patch)

    def _sync_status(self, patch: str) -> dict[str, bool]:
        tmp = Path(tempfile.mkdtemp(prefix="crucible-phi-"))
        try:
            _git(tmp, "init", "-q")
            _git(tmp, "fetch", "-q", str(self.root), f"refs/tags/{self.base}:refs/tags/{self.base}")
            _git(tmp, "checkout", "-q", self.base)
            if patch.strip():
                _git(tmp, "apply", "--whitespace=nowarn", "-", input=patch if patch.endswith("\n") else patch + "\n")
            return {k: bool(v) for k, v in self._status(tmp, patch).items()}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---- the three potentials ---------------------------------------------------------------------------------------------------------
def prompt_draft(task_id: str, workspace: Path | str, milestone_ids: Sequence[str], *, lint: Callable[[str], Sequence[str]],
                 path: str = "prompt.json", min_chars: int = 40) -> ArtifactPotential:
    """The prompt writer's draft ``{opening, requirements: {mid: text}, closing}``: a milestone passes when its requirement
    line is there, at least ``min_chars`` long, and the lint of it (with the opening and closing) has no finding. A
    milestone id that is a top-level field of the draft (a single ``opening``) is that field."""

    def status(tree: Path, patch: str) -> dict[str, bool]:
        f = tree / path
        try:
            d = json.loads(f.read_text()) if f.is_file() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            d = {}
        d = d if isinstance(d, dict) else {}
        reqs = d.get("requirements") if isinstance(d.get("requirements"), dict) else {}
        frame = " ".join(str(d.get(k) or "") for k in ("opening", "closing") if k not in milestone_ids)
        frame_ok = not lint(frame) if frame.strip() else True
        out = {}
        for m in milestone_ids:
            t = " ".join(str((d.get(m) if m in ("opening", "closing") else reqs.get(m)) or "").split())
            out[m] = bool(len(t) >= min_chars and frame_ok and not lint(t))
        return out

    return ArtifactPotential(task_id, workspace, milestone_ids, status)


def hidden_tests(task_id: str, workspace: Path | str, plan: Mapping[str, Any]) -> ArtifactPotential:
    """The verifier writer's files: a milestone passes ``generators.swarm.test_problems`` on its own file."""
    from crucible.factory.generators.swarm import HIDDEN_DIR, test_problems
    from crucible.factory.generators.swarm import hidden_tests as read_tests

    ids = [m["id"] for m in plan["milestones"]]

    def status(tree: Path, patch: str) -> dict[str, bool]:
        tests = read_tests(tree / HIDDEN_DIR, plan)
        return {m["id"]: not test_problems(tests, {"milestones": [m]}) for m in plan["milestones"]}

    return ArtifactPotential(task_id, workspace, ids, status)


SUBJECT_RX = re.compile(r"^Subject: (?:\[PATCH[^\]]*\] )?(.*)$", re.MULTILINE)


def oracle_commits(task_id: str, workspace: Path | str, plan: Mapping[str, Any]) -> ArtifactPotential:
    """The oracle builder's commits (``git format-patch base..HEAD``, the shape its capture takes): milestone k passes when
    commits 1..k exist with subjects ``m1: ...`` .. ``mk: ...`` in order (``generators.swarm.oracle_problems`` per prefix).
    Each milestone requires the one before it, so a commit out of order credits nothing after it."""
    ids = [m["id"] for m in plan["milestones"]]
    root = Path(workspace)

    def status(tree: Path, patch: str) -> dict[str, bool]:
        subjects = [s.strip() for s in SUBJECT_RX.findall(patch)]
        out, ok = {}, True
        for k, mid in enumerate(ids):
            ok = ok and k < len(subjects) and bool(re.match(rf"^\s*{re.escape(mid)}\b", subjects[k]))
            out[mid] = ok
        return out

    pot = ArtifactPotential(task_id, root, ids, status, requires={ids[i]: [ids[i - 1]] for i in range(1, len(ids))})

    async def snapshot(world: Any) -> str:
        return await asyncio.to_thread(_git, root, "format-patch", "--stdout", "base..HEAD")

    pot.snapshot_patch = snapshot  # type: ignore[attr-defined]
    pot._sync_status = lambda patch: {k: bool(v) for k, v in status(root, patch).items()}  # type: ignore[method-assign]
    return pot


POTENTIALS: dict[str, str] = {"prompt_draft": "the prompt writer", "hidden_tests": "the verifier writer", "oracle_commits": "the oracle builder"}


def _task_id(ctx: Any) -> str:
    return str((getattr(ctx, "params", None) or {}).get("task_id") or getattr(ctx, "run_id", "task"))


# the potentials a ``SessionPool`` builds by itself when the machine declares them (``session.potential``) and the pool was
# given none by that name: the two that need only the state's ``plan`` input. ``prompt_draft`` needs the lane's lint, so
# the lane passes it (``factory.swarm_prompts``).
DEFAULTS: dict[str, Callable[[Any, Any], ArtifactPotential]] = {
    "oracle_commits": lambda ctx, ws: oracle_commits(_task_id(ctx), ws.path, ctx.inputs["plan"]),
    "hidden_tests": lambda ctx, ws: hidden_tests(_task_id(ctx), ws.path, ctx.inputs["plan"]),
}
