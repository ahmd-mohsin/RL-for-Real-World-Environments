"""Crucible tasks as a verifiers v1 taskset → prime-rl (export; research m11-m12/05 §3.3). DEPENDs on verifiers'
``HarborTaskset`` / ``HarborTask`` (``verifiers.v1.tasksets.harbor``) and lives in verifiers' own environment
(``crucible/.envs/verifiers``: verifiers pins mcp 2, agentenv mcp<2), so the import is lazy.

Crucible task directories are Harbor task directories (DESIGN §4.3), so verifiers' own Harbor machinery runs them. What
``CrucibleTask`` changes in ``run_verifier``:

- a masked grade (Crucible writes ``mask.json`` and no ``reward.json``) raises ``TaskError("crucible:masked:<kind>")``
  with the grader's failure kind → an errored trace, which prime-rl excludes from training (never a 0). Without this
  override verifiers would raise a generic "no valid reward" error: still excluded, but the kind is lost;
- the reward is ``reward.json["reward"]`` (the plan's gated reward, Crucible's single producer); the other keys are
  recorded as metrics under a ``crucible/`` prefix (unweighted: verifiers sums only rewards);
- ``steps.jsonl`` (the dense channel), when present, is attached as ``trace.info["crucible_steps"]`` for the M12 prime-rl
  algorithm that maps rows to sampled nodes by ``call_id``.
"""

from __future__ import annotations

import json


def build():
    from verifiers.v1.errors import TaskError
    from verifiers.v1.tasksets.harbor.taskset import (
        MAX_REWARD_BYTES,
        HarborTask,
        HarborTaskset,
        dataset_dir,
        parse_task,
        resolve_env,
    )

    class CrucibleTask(HarborTask):
        async def run_verifier(self, runtime, trace):
            await runtime.run(["bash", "/tests/test.sh"], resolve_env(self.data.verifier_env))
            try:
                mask = json.loads((await runtime.read("/logs/verifier/mask.json", max_bytes=MAX_REWARD_BYTES)).decode())
            except Exception:  # noqa: BLE001 -- no mask file: a real grade
                mask = None
            if mask and mask.get("mask_sample"):
                raise TaskError(f"crucible:masked:{mask.get('failure_kind') or 'unknown'}: {(mask.get('reason') or '')[:200]}")
            scores = await self.read_reward_json(runtime)
            if not isinstance(scores, dict) or "reward" not in scores:
                raise TaskError("crucible: no reward.json with a 'reward' key (and no mask.json)")
            trace.record_metrics({f"crucible/{k}": v for k, v in scores.items() if k != "reward"})
            try:
                steps = (await runtime.read("/logs/verifier/steps.jsonl", max_bytes=MAX_REWARD_BYTES)).decode()
                trace.info["crucible_steps"] = [json.loads(x) for x in steps.splitlines() if x.strip()]
            except Exception:  # noqa: BLE001, S110 -- the dense channel is optional
                pass
            return {"reward": scores["reward"]}

    class CrucibleTaskset(HarborTaskset):
        def load(self):
            root = dataset_dir(self.config)
            dirs = [p.parent for p in sorted(root.rglob("task.toml")) if (p.parent / "instruction.md").is_file()
                    and (self.config.tasks is None or p.parent.name in self.config.tasks)]
            if not dirs:
                raise ValueError(f"no Crucible/Harbor tasks found in {root}")
            for i, d in enumerate(dirs):
                yield CrucibleTask(parse_task(d, i, self.config), self.config.task)

    return CrucibleTask, CrucibleTaskset
