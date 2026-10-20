"""The Crucible taskset as a verifiers v1 plugin: ``vf-eval crucible-vf --taskset.path <dir>`` (prime-rl loads it the
same way). verifiers' loader imports the module the id names and takes the one ``Taskset`` subclass exported in
``__all__`` (M11 audit G2 D2: the classes were built at call time, so no loader could find them). PORT of the packaging of
research-environments ``apex_agents`` (Apache-2.0): a local ``path`` beside the Hub ``dataset`` and the task's own
timeouts kept (``ignore_timeouts = False``).

M11 lab audit R-3, against verifiers dca5562 (MIT; pinned by ``scripts/build_verifiers_env.sh``):

- **Loads against current verifiers.** ``HarborEnv`` is gone upstream: isolated scoring of a ``[verifier].environment_mode
  = "separate"`` task is verifiers' own rollout lifecycle now (``verifiers/v1/grading.py``, ``tasksets/harbor/taskset.py:231-245``,
  ``parse_task`` selects it at ``:770``). This module imported it, so a rebuilt env raised ``ModuleNotFoundError`` and
  neither ``vf-eval`` nor prime-rl could load the taskset. Only ``CrucibleTaskset`` is exported.
- **A separate-verifier task is never graded in the solver's box** (rule 4). ``CrucibleConfig`` refuses
  ``ignore_separate_verifier`` (verifiers' opt-in to grade every task in the agent's box), and ``CrucibleTask.grading_task``
  refuses a resolved ``scoring.mode = "shared"`` (an agent-level override, ``resolve_scoring``, ``rollout.py:122``) for a
  task whose own verifier is separate: in shared mode verifiers drops the verifier (``taskset.py:235-236``), so its own
  ``solved`` guard never fires.
- **Shared grading is quiesced first** (P23, as Crucible's in-place Harbor grade, ``harbor/quiesce.py``). verifiers grades
  a shared task in the agent's box with no host-side kill (cleanup runs after scoring), and a surviving policy process could
  write ``reward.json``, ``mask.json`` or ``steps.jsonl`` while ``test.sh`` ran. ``setup`` records the box's processes
  with their start times (``trace.info["crucible_baseline"]``, host-side: the policy cannot edit it); before the tests are
  staged, ``box.QUIESCE`` kills every other process (the agent, its tools, anything they left), and its verdict decides as
  Crucible's in-place quiesce does: ``clean`` grades; ``survivors`` (a policy process outlived three kill rounds) scores 0
  without running ``test.sh``, with the flag ``policy_process_survived`` (``trace.info["crucible_flags"]`` and the metric
  ``crucible/policy_process_survived``): a mask there would let a policy drop any failing episode by leaving a hopper;
  ``respawned``, ``uninterruptible`` and a script that did not finish mask (``TaskError``). The receipt is
  ``trace.info["crucible_quiesce"]``. Refused (a mask) where the quiesce cannot be trusted: no baseline from ``setup``, or
  verifiers' ``subprocess`` runtime (its ``/proc`` is the host's: the kill would reach every process started since setup,
  other rollouts included). The quiesce shell runs with the scoring env's names blanked, so a policy process still alive
  cannot read a judge key from its ``environ``. Processes the harness starts after ``setup`` (the agent and its colocated
  tools) are the policy's. ``box``'s residual: a root policy can replace ``sh`` in its own box; the isolated lifecycle does
  not depend on the box.
- **Gold is checkable.** ``CrucibleTask.validate`` (``vf-validate``'s gold leg) records a baseline, runs
  ``solution/solve.sh`` with Harbor's ``[solution].env`` (``harbor/agents/oracle.py``), quiesces like a rollout (a
  solution's server is killed before the grade there too), and grades: valid iff the reward is 1. The noop leg is verifiers'
  own (``task.score`` on the untouched box): valid iff it does not pass, and a mask raises (an invalid empty control, never
  a pass). Not gold-checked here (``None``): a separate-verifier task (its grade is the fresh-box lifecycle, with artifacts
  and collect hooks, which ``validate(runtime)`` cannot drive; verifiers' noop leg errors for it too, ``solved`` refusing an
  unstaged separate verifier) and a task with a dedicated verifier image. ``finalize`` and collect hooks are not run.
- **Provenance.** Every grade records the verifiers distribution's version and commit (``direct_url.json``) in
  ``trace.info["crucible_provenance"]``.

Imports verifiers and Harbor at module level: it lives in verifiers' own environment (``crucible/.envs/verifiers``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import verifiers.v1 as vf
from pydantic import model_validator
from verifiers.v1.errors import TaskError
from verifiers.v1.tasksets.harbor.taskset import MAX_REWARD_BYTES, HarborConfig, HarborTask, dataset_dir, make_tar, parse_task, resolve_env

from crucible.interop.verifiers import box

# files the grader alone may write; anything already there was planted by the agent or shipped in the image (G2 D1, H2 V8:
# verifiers removes only reward.json and reward.txt, so a planted mask.json masked the episode and planted steps rode along)
GRADER_FILES = ("/logs/verifier/mask.json", "/logs/verifier/steps.jsonl", "/logs/verifier/crucible_reward.json")

def provenance() -> dict[str, Any]:
    """The verifiers distribution this process runs: version and, from ``direct_url.json``, the VCS commit or the editable
    checkout's ``HEAD`` (an editable shallow clone reports version 0.0.0, verifiers ``pyproject.toml:169``)."""
    from crucible.runtime.identity import dist_source

    return {"verifiers": dist_source("verifiers")}


class CrucibleConfig(HarborConfig):
    path: Path | None = None
    """A local Harbor/Crucible export instead of a Hub download."""
    ignore_timeouts: bool = False
    """Keep each task's agent and verifier deadlines (G2 D11: verifiers drops them by default)."""

    @model_validator(mode="after")
    def _separate_stays_separate(self) -> CrucibleConfig:
        if self.ignore_separate_verifier:
            raise ValueError("crucible-vf: ignore_separate_verifier would grade a task whose [verifier] is separate in the agent's own box, "
                             "where the agent can reach the grader (M11 lab audit R-3); it is refused for Crucible tasks")
        return self


class CrucibleTask(HarborTask):
    async def setup(self, runtime, deployment=None, trace=None) -> None:
        await super().setup(runtime, deployment)
        if trace is not None and self.data.verifier is None:  # a shared grade quiesces against this baseline
            trace.info["crucible_baseline"] = await self._baseline(runtime)

    @staticmethod
    async def _baseline(runtime) -> list[str]:
        res = await runtime.run(["sh", "-c", box.LIST_PIDS], {})
        return box.baseline(res.stdout) if not getattr(res, "exit_code", 0) else []

    def grading_task(self, runtime, *, scoring):
        if self.data.verifier is not None and scoring.mode == "shared":
            raise TaskError(f"crucible: task {self.data.name!r} declares a separate verifier ([verifier].environment_mode = \"separate\"); "
                            "scoring.mode=shared would grade it in the agent's box (M11 lab audit R-3); refused")
        return super().grading_task(runtime, scoring=scoring)

    async def _quiesce(self, info: dict, runtime) -> str:
        """Kill what the policy left in the box before a shared grade (module doc). Returns the verdict (``clean`` or
        ``survivors``); raises ``TaskError`` (a mask) for the others and where the quiesce cannot be trusted. Idempotent per
        rollout (``info`` is the trace's: a copy of this task made by ``with_data`` shares nothing)."""
        if info.get("crucible_quiesce"):
            return info["crucible_quiesce"]["verdict"]
        rtype = getattr(getattr(runtime, "config", None), "type", None)
        if rtype == "subprocess":
            raise TaskError("crucible: shared grading on verifiers' subprocess runtime is refused: its /proc is the host's, so the "
                            "pre-grade quiesce would kill unrelated processes (M11 lab audit R-3)")
        base = info.get("crucible_baseline")
        if not base:
            raise TaskError("crucible: no process baseline from setup, so the agent's box cannot be quiesced before a shared grade; refused")
        env = {k: "" for k in (self.data.verifier_env or {})}  # a live policy process could read the shell's environ
        res = await runtime.run(["sh", "-c", box.QUIESCE], {**env, "CRUCIBLE_BASE": " ".join(base), "CRUCIBLE_KILL": "kill"})
        killed, left, ran = box.parse(res.stdout)
        v = box.verdict(left, base, ran and not getattr(res, "exit_code", 0))
        info["crucible_quiesce"] = {"verdict": v, "killed": killed, "left": left, "baseline": len(base)}
        if v == "error":
            raise TaskError(f"crucible: the agent's box could not be quiesced before a shared grade: {(res.stderr or res.stdout or '').strip()[-300:]}")
        if v in ("respawned", "uninterruptible"):
            raise TaskError(f"crucible:masked:quiesce_{v}: processes left after the pre-grade quiesce: {left[:10]}")
        if v == "survivors":
            info.setdefault("crucible_flags", []).append({"rule": "policy_process_survived", "category": "grader_tampering", "seq": None,
                                                         "evidence": json.dumps(left[:20])[:600]})
        return v

    async def stage_verifier(self, trace, runtime) -> None:
        if self.data.verifier is None and await self._quiesce(trace.info, runtime) == "survivors":
            return  # nothing is staged: ``solved`` scores 0 without running test.sh beside a live policy process
        await super().stage_verifier(trace, runtime)

    @vf.reward(weight=1.0)
    async def solved(self, runtime, trace, deployment=None):
        if self.data.verifier is None and await self._quiesce(trace.info, runtime) == "survivors":  # (vf-validate's noop leg too)
            trace.record_metrics({"crucible/policy_process_survived": 1.0})
            return {"reward": 0.0}
        return await super().solved(runtime, trace, deployment)

    async def run_verifier(self, runtime, trace):
        trace.info["crucible_provenance"] = provenance()
        res = await runtime.run(["sh", "-c", "rm -f " + " ".join(GRADER_FILES)], {})
        if getattr(res, "exit_code", 0):
            raise TaskError("crucible: could not clear the grader's files before grading")
        await runtime.run(["bash", "/tests/test.sh"], resolve_env(self.data.verifier_env))  # the scoring env (grading_task merges it)
        try:
            mask = json.loads((await runtime.read("/logs/verifier/mask.json", max_bytes=MAX_REWARD_BYTES)).decode())
        except Exception:  # noqa: BLE001 -- no mask file: a real grade
            mask = None
        if isinstance(mask, dict) and mask.get("mask_sample"):
            raise TaskError(f"crucible:masked:{mask.get('failure_kind') or 'unknown'}: {(mask.get('reason') or '')[:200]}")
        scores = await self.read_reward_json(runtime)
        if isinstance(scores, dict) and "reward" in scores:
            trace.record_metrics({f"crucible/{k}": v for k, v in scores.items() if k != "reward"})
            try:
                steps = (await runtime.read("/logs/verifier/steps.jsonl", max_bytes=MAX_REWARD_BYTES)).decode()
                trace.info["crucible_steps"] = [json.loads(x) for x in steps.splitlines() if x.strip()]
            except Exception:  # noqa: BLE001, S110 -- the dense channel is optional
                pass
            return {"reward": scores["reward"]}
        try:  # a Harbor task whose grader writes only reward.txt (G2 D10), parsed as Harbor does
            r = float((await runtime.read("/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES)).decode().strip())
        except Exception as ex:
            raise TaskError("crucible: no reward.json with a 'reward' key, no reward.txt and no mask.json") from ex
        if not math.isfinite(r):
            raise TaskError(f"crucible: non-finite reward.txt ({r})")
        return {"reward": r}

    async def validate(self, runtime) -> bool | None:
        """``vf-validate``'s gold leg: a baseline, ``solution/solve.sh`` with ``[solution].env``, the quiesce a rollout runs,
        then the task's own grade; valid iff it scores 1. None for a separate-verifier task, a dedicated verifier image or no
        solution (module doc)."""
        import tomllib

        sol = Path(self.data.task_dir) / "solution"
        if not (sol / "solve.sh").is_file() or self.data.verifier is not None or self.data.verifier_image is not None:
            return None
        info: dict[str, Any] = {"crucible_baseline": await self._baseline(runtime)}
        declared = (tomllib.loads((Path(self.data.task_dir) / "task.toml").read_text()).get("solution") or {}).get("env") or {}
        env = {**resolve_env(self.data.env), **resolve_env({str(k): str(v) for k, v in declared.items()}), "DEBIAN_FRONTEND": "noninteractive"}
        await runtime.write("/tmp/solution.tgz", make_tar(sol))
        res = await runtime.run(["sh", "-c", "mkdir -p /solution && tar -xzf /tmp/solution.tgz -C /solution && bash /solution/solve.sh"], env)
        if getattr(res, "exit_code", 0):
            raise TaskError(f"crucible: solution/solve.sh failed (exit {res.exit_code}): {(res.stderr or res.stdout or '').strip()[-300:]}")
        if await self._quiesce(info, runtime) == "survivors":
            return False  # the reference solution leaves a process no kill stops: a rollout doing the same scores 0
        await self.stage_tests(runtime)
        scores = await self.run_verifier(runtime, SimpleNamespace(info=info, record_metrics=lambda m: None))
        return float(scores["reward"]) >= 1.0


class CrucibleTaskset(vf.Taskset[CrucibleTask, CrucibleConfig]):
    def load(self) -> Iterator[CrucibleTask]:
        root = self.config.path or dataset_dir(self.config)
        dirs = [p.parent for p in sorted(Path(root).rglob("task.toml")) if (p.parent / "instruction.md").is_file()
                and (self.config.tasks is None or p.parent.name in self.config.tasks)]
        if self.config.tasks is not None:
            missing = set(self.config.tasks) - {d.name for d in dirs}
            if missing:
                raise ValueError(f"unknown Crucible tasks: {sorted(missing)}")
        if not dirs:
            raise ValueError(f"no Crucible/Harbor tasks found in {root}")
        for i, d in enumerate(dirs):
            yield CrucibleTask(parse_task(d, i, self.config), self.config.task)


__all__ = ["CrucibleTaskset"]
