"""The Crucible taskset as a verifiers v1 plugin: ``vf-eval crucible.interop.verifiers.vf_taskset --taskset.path <dir>``
(prime-rl loads it the same way). verifiers' loader imports the module the id names and takes the one ``Taskset``
subclass exported in ``__all__``; exporting ``HarborEnv`` makes tasks with a separate verifier grade in a fresh box
(M11 audit G2 D2: the classes were built at call time, so no loader could find them). PORT of the packaging of
research-environments ``apex_agents`` (Apache-2.0): a local ``path`` beside the Hub ``dataset`` and the task's own
timeouts kept (``ignore_timeouts = False``).

Imports verifiers and Harbor at module level: it lives in verifiers' own environment (``crucible/.envs/verifiers``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.errors import TaskError
from verifiers.v1.tasksets.harbor.env import HarborEnv
from verifiers.v1.tasksets.harbor.taskset import MAX_REWARD_BYTES, HarborConfig, HarborTask, dataset_dir, parse_task, resolve_env

# files the grader alone may write; anything already there was planted by the agent or shipped in the image (G2 D1, H2 V8:
# verifiers removes only reward.json and reward.txt, so a planted mask.json masked the episode and planted steps rode along)
GRADER_FILES = ("/logs/verifier/mask.json", "/logs/verifier/steps.jsonl", "/logs/verifier/crucible_reward.json")


class CrucibleConfig(HarborConfig):
    path: Path | None = None
    """A local Harbor/Crucible export instead of a Hub download."""
    ignore_timeouts: bool = False
    """Keep each task's agent and verifier deadlines (G2 D11: verifiers drops them by default)."""


class CrucibleTask(HarborTask):
    async def run_verifier(self, runtime, trace):
        res = await runtime.run(["sh", "-c", "rm -f " + " ".join(GRADER_FILES)], {})
        if getattr(res, "exit_code", 0):
            raise TaskError("crucible: could not clear the grader's files before grading")
        await runtime.run(["bash", "/tests/test.sh"], resolve_env(self.data.verifier_env))
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


__all__ = ["CrucibleTaskset", "HarborEnv"]
