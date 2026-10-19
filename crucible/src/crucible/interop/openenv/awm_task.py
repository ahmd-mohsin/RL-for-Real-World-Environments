"""AWM-1K tasks with AWM's own deterministic verifiers (m13 E; audit 03 G12). ``awm.py`` turns an AgentWorldModel scenario
into a World (its generated FastAPI service as an App over a SQLite session DB); this module adds the *tasks* and the
*grader*, so the 1,000 scenarios × ~10 tasks produce rewards.

- ``AwmTask(scenario, task_idx)`` (an M10 RunTask): the instruction is the dataset's task text. ``open`` builds the
  World and copies the freshly seeded session DB to ``initial.db``, the verifier's reference state.
- **Grade**: AWM's *code-mode* verifier for that task (``gen_verifier.pure_code.jsonl``:
  ``verify_task_completion(initial_db_path, final_db_path, final_answer) -> {"result": "complete" | "others"}``) runs in
  AWM's own out-of-process runner (``_vendor/openenv_awm/verifier_runner.py``: import allowlist, CPU / memory / file /
  process limits) against the initial DB and the episode's final DB, both read-only. ``complete`` -> 1, ``others`` -> 0,
  and a runner error is masked only when the verifier also fails on the untouched database (a broken verifier); one
  that runs there failed on the episode's own output, which is scored 0. ``final_answer`` is the policy's final
  message, cut to ``ANSWER_CAP`` characters; the function is the code's first ``def verify_...`` (upstream's rule). The LLM-judged SQL mode is not used: the code mode is deterministic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from crucible.runtime.identity import config_digest

RUNNER = Path(__file__).resolve().parents[2] / "_vendor" / "openenv_awm" / "verifier_runner.py"


def verifier_for(scenario: str, task_idx: int, *, cache_dir: Path | None = None) -> dict:
    from crucible._vendor.openenv_awm.data_loader import normalize_scenario_name
    from crucible.interop.openenv.awm import loader

    key = normalize_scenario_name(scenario)
    for v in loader(cache_dir)._build_verifiers("code").get(key, []):
        if int(v.get("task_idx", -1)) == int(task_idx):
            return v
    raise KeyError(f"no code verifier for {key} task {task_idx}")


ANSWER_CAP = 4000  # characters of the final answer a verifier sees (H3 V16: a long answer pushed regex verifiers past their CPU limit)


def function_name(code: str, default: str = "verify_task_completion") -> str:
    """Upstream's rule (``agent_world_model_env/server/verifier.py``): the first ``def verify_...`` in the code (H3 V17: one
    verifier names its function ``verify_task_complete`` and was masked on every sample)."""
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def verify_") and "(" in line:
            return line.split("(")[0].replace("def ", "").strip()
    return default


def run_verifier(code: str, initial_db: Path, final_db: Path, final_answer: str | None, *, timeout_s: float = 30.0) -> dict:
    """AWM's runner, out of process; returns its JSON (``result``, ``execution_status``, ``error_message``?)."""
    payload = {"verifier_code": code, "function_name": function_name(code), "initial_db_path": str(initial_db), "final_db_path": str(final_db),
               "final_answer": final_answer, "mode": "code"}
    modes = {}
    for p in (initial_db, final_db):  # read-only for the verifier (AWM's own verifier.py does the same)
        modes[p] = p.stat().st_mode
        p.chmod(0o444)
    try:
        r = subprocess.run([sys.executable, str(RUNNER)], input=json.dumps(payload), capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"result": "others", "execution_status": "error", "error_message": f"verifier timed out after {timeout_s}s"}
    finally:
        for p, m in modes.items():
            p.chmod(m)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"result": "others", "execution_status": "error", "error_message": f"runner rc {r.returncode}: {(r.stderr or r.stdout)[-500:]}"}


class AwmTask:
    def __init__(self, scenario: str, task_idx: int, *, cache_dir: Path | None = None):
        from crucible._vendor.openenv_awm.data_loader import normalize_scenario_name
        from crucible.interop.openenv.awm import awm_world

        self.scenario, self.task_idx, self.cache_dir = normalize_scenario_name(scenario), int(task_idx), cache_dir
        self.spec_doc, tasks = awm_world(scenario, cache_dir=cache_dir)
        self.instruction = tasks[self.task_idx]
        self.verifier = verifier_for(scenario, task_idx, cache_dir=cache_dir)
        self.id = f"openenv-awm/{self.scenario}/t{self.task_idx}"
        self.world_id = self.spec_doc.world

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the scenario and task, the instruction, AWM's own verifier (its code is a text: digested by
        ``describe`` when long) and the service the policy works against: its generated code, DB schema and sample rows, which the
        App reads from the cache at start and the World spec only names (``cache_dir`` is a path: operational, ``*_dir``)."""
        from crucible.interop.openenv.awm import loader

        ld = loader(self.cache_dir)
        service = {"env_code": config_digest(ld.get_env_code(self.scenario)), "db_schema": config_digest(ld.get_db_schema(self.scenario)),
                   "sample_data": config_digest(ld.get_sample_data(self.scenario))}
        return {"scenario": self.scenario, "task_idx": self.task_idx, "instruction": self.instruction, "verifier": self.verifier, "service": service,
                "spec": self.spec_doc}

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec_doc, instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        await w.start()
        await w.reset()
        db = Path(w.apps["api"].state_params["path"])
        shutil.copyfile(db, workdir / "initial.db")
        w._awm_initial = workdir / "initial.db"  # the verifier's reference state, taken before any policy action
        return w

    @staticmethod
    def final_answer(world: Any) -> str | None:
        done = [e for e in world.to_trace().events if e.type == "done" and e.actor == "agent" and e.text not in (None, "environment_terminal")]
        return done[-1].text if done else None

    def plan(self):
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict
        from crucible.verify.runtime.trusted import InfraError

        async def awm_verifier(ctx) -> Verdict:
            world = ctx.params["world"]
            final = Path(world.apps["api"].state_params["path"])
            snap = ctx.trusted.work / "final.db"  # a copy: the live service keeps its handle on the session DB
            shutil.copyfile(final, snap)
            code = self.verifier["verification"]["code"]
            answer = self.final_answer(world)
            out = run_verifier(code, world._awm_initial, snap, answer[:ANSWER_CAP] if answer else answer)
            if out.get("execution_status") != "success":
                # a verifier that also fails on the untouched database with no answer is broken (masked); one that runs
                # there failed on what the episode produced, which is the policy's outcome (H3 V16)
                selfcheck = run_verifier(code, world._awm_initial, world._awm_initial, None)
                if selfcheck.get("execution_status") != "success":
                    raise InfraError(f"AWM verifier failed: {out.get('error_message', 'unknown')}"[:500])
                return Verdict(False, {"result": "others", "policy_error": str(out.get("error_message"))[:500]})
            ok = out.get("result") == "complete"
            return Verdict(ok, {"result": out.get("result"), "raw": str(out.get("raw_result"))[:500]})

        return Plan(self.id, [Gate("awm_verifier", kind="outcome", description="AWM's own deterministic code verifier")],
                    [Check("awm_verifier", "awm_verifier", awm_verifier, tier="fast", provenance="awm:gen_verifier.pure_code")])

    async def grade(self, world: Any, workdir: Path):
        from crucible.verify.runtime.plan import Context
        from crucible.verify.runtime.trusted import Trusted

        return await self.plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view(), "world": world}))
