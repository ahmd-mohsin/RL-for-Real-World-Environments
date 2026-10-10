"""Imported OpenEnv environments (``import_env.OpenEnvTask`` configurations), starting with the FineEnvs simulation
catalogue (https://huggingface.co/spaces/FineEnvs/simulation-rl-environments; research m11-m12/05 §4).

``portsim``: PortSimEnv v1 (FineEnvs ``07-simulation-environments/portsim-v1``, data CC BY-SA 4.0): re-plan a disrupted
week of container berthings at a Port of Barcelona quay. Tools ``get_situation``, ``check_plan`` (10 per episode),
``submit_plan`` (ends the episode). Reward = the environment's own rubric, graded once, deterministically, against the
CP-SAT optimum: 0.2 + 0.8·e^(−gap/0.5) for a valid plan, ≤ 0.2 when a rule is broken, 0 with no submission (and 0 when
its 24-call budget runs out). Requires ``berth-core`` + ``berth-openenv`` installed next to OpenEnv.
"""

from __future__ import annotations

import json

from crucible.interop.openenv.import_env import OpenEnvTask

PORTSIM = "berth_openenv.environment:BerthPlanningEnvironment"


def _feasible_check(j: dict) -> bool:
    for s in j.get("steps", []):
        if s["tool"] == "check_plan":
            try:
                if json.loads(s.get("text") or "{}").get("feasible"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _submitted_feasible(j: dict) -> bool:
    return bool(j.get("done") and (j.get("final_metadata") or {}).get("grade", {}).get("feasible"))


def portsim(split: str = "train", index: int = 0, *, task_id: str | None = None) -> OpenEnvTask:
    reset = {"task_id": task_id} if task_id else {"split": split, "index": index}
    name = task_id or f"{split}-{index}"
    return OpenEnvTask(f"openenv/portsim/{name}@1", PORTSIM, reset=reset,
                       instruction=lambda obs: obs.metadata["instructions"],
                       progress={"feasible_plan_checked": _feasible_check, "submitted_feasible": _submitted_feasible})
