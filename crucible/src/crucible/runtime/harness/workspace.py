"""The workspace harness: a policy works in a shell workspace World; its work is *collected as a patch* and graded by
the task from a pristine tree (the coding seam; the shape of a Harbor trial with a separate verifier).

Phases: seed (open) -> agent (run_episode; broken-environment check) -> collect (tolerant or legacy; an empty patch from
a changed tree is ``crucible:empty_collection``) -> verification (``task.grade_patch``) -> cleanup.

A collection failure preserves the workspace in the attempt directory. The next attempt of the same sample then
**re-collects only** (tolerant hook, preserved tree) instead of re-running the agent: the agent's work was fine, the
harness lost it (research m9-m10/02 §3.3 row 1).
"""

from __future__ import annotations

import json
import shutil
import time

from crucible.actors.policy import run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected, classify, collect
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import LIMIT_STOPS, NativeHarness
from crucible.runtime.infra import StageFailure, boundary, classify_verifier, failure


class WorkspaceHarness(NativeHarness):
    def __init__(self, name: str = "native:workspace", version: str = "1", *, collect_mode: str = "tolerant", **kw):
        super().__init__(name, version, **kw)
        self.collect_mode = collect_mode

    def _preserved(self, attempt: Attempt):
        prev = attempt.dir.parent / f"a{attempt.n - 1}"
        rec = prev / "record.json"
        if attempt.n > 1 and (prev / "preserved_ws").exists() and rec.exists() and \
                (json.loads(rec.read_text()).get("failure") or {}).get("kind") == "crucible:empty_collection":
            return prev / "preserved_ws"
        return None

    async def _grade(self, attempt: Attempt, c: Collected, res: HarnessResult, t: dict) -> None:
        (attempt.dir / "artifacts").mkdir(exist_ok=True)
        (attempt.dir / "artifacts" / "model.patch").write_text(c.patch)
        res.artifacts["patch"] = "artifacts/model.patch"
        res.tree_changed = c.tree_changed
        t0 = time.monotonic()
        with boundary("verification"):
            rep = await attempt.task.grade_patch(c.patch, attempt.dir / "grade")
        t["verification"] = time.monotonic() - t0
        res.grade = {"reward": rep.reward, "gates": rep.gate_bits(), "infra": rep.infra}
        if rep.infra.get("status") == "masked":
            raise StageFailure(classify_verifier(rep.infra))
        res.reward, res.gates = rep.reward, rep.gate_bits()

    async def run(self, attempt: Attempt) -> HarnessResult:
        t: dict[str, float] = {}
        res = HarnessResult()
        world = None
        spend0 = attempt.policy.spend()
        try:
            preserved = self._preserved(attempt)
            if preserved is not None:  # re-collect only: never re-run an agent whose work the harness lost
                c = collect(preserved, base="base", mode="tolerant")
                res.stop = "recollected"
                f = classify(c)
                if f:
                    raise StageFailure(f)
                await self._grade(attempt, c, res, t)
                return res
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                for p in policies.values():
                    if hasattr(p, "begin_episode"):
                        p.begin_episode(attempt.id)
                ep = await run_episode(world, policies, limits=self.limits)
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            (attempt.dir / "trace.json").write_text(ep.trace.model_dump_json())
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            if broken:
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            ws = attempt.task.workspace(world)
            t0 = time.monotonic()
            c = collect(ws, base="base", mode=self.collect_mode)
            t["collect"] = time.monotonic() - t0
            f = classify(c)
            if f:  # keep the agent's tree as it is (a rename: unreadable files survive) for a re-collect-only retry
                shutil.move(str(ws), str(attempt.dir / "preserved_ws"))
                raise StageFailure(f)
            await self._grade(attempt, c, res, t)
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if world is not None:
                try:
                    await world.close()
                except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                    pass
            usd, unc = attempt.policy.spend()
            res.cost_usd, res.cost_uncertain_usd = max(0.0, usd - spend0[0]), max(0.0, unc - spend0[1])
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
