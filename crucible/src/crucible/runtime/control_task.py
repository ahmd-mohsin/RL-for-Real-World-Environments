"""Control tasks (M1 item 10): a ``RunTask`` over one control App (``world/apps/control.py``), graded by the environment's
own return.

Grade: one ``score`` gate over the journal. The episode return (the sum of per-tick rewards) is normalized by
``return_range`` (min, max) to [0, 1] (clipped); ``success_return`` (optional) sets the outcome. An episode the policy
left before the simulator ended it still scores its return so far: control returns accumulate per tick, unlike a
final answer. Per-tick rewards stay in the journal for dense credit (M6) and in each tool result's ``data``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.verify.runtime.plan import CheckResult, GradeReport

ACTOR = "agent"


class ControlTask:
    def __init__(self, id: str, *, env: str | None = None, factory: str | None = None, client: dict | None = None, env_kwargs: dict | None = None, app: str = "sim",
                 return_range: tuple[float, float] = (0.0, 1.0), success_return: float | None = None, instruction: str = "", seed: int = 0,
                 clock: bool = True, **params: Any):
        if sum(bool(x) for x in (env, factory, client)) != 1:
            raise ValueError("a control task needs exactly one of env, factory or client")
        lo, hi = return_range
        if not hi > lo:
            raise ValueError("return_range needs max > min")
        self.id, self.app, self.return_range, self.success_return = id, app, (float(lo), float(hi)), success_return
        self.world_id = id.rsplit("@", 1)[0]
        self.instruction, self.seed, self.clock = instruction or "Control the system to maximize its return.", seed, clock
        self.params = {**({"env": env} if env else {}), **({"factory": factory} if factory else {}), **({"client": dict(client)} if client else {}),
                       "env_kwargs": env_kwargs or {}, **params}

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: every constructor parameter this task stores (the environment, factory or client and its kwargs
        live in ``params``: plain data by construction), so a changed return range, seed or environment is refused by name."""
        return {"id": self.id, "params": self.params, "app": self.app, "return_range": self.return_range, "success_return": self.success_return,
                "instruction": self.instruction, "seed": self.seed, "clock": self.clock}

    def spec(self):
        from crucible.core.schema.world import WorldSpec

        d = {"world": self.world_id, "seed": self.seed,
             "apps": [{"id": self.app, "template": "common/control", "state": {"adapter": "json_doc", "params": {"path": f"{self.app}.json"}}, "params": self.params}],
             "actors": [{"id": ACTOR, "kind": "policy", "tools": [f"{self.app}.*"]}]}
        if self.clock:
            d["clock"] = {"mode": "sim", "t0": "2026-01-05T09:00:00Z"}
        return WorldSpec.model_validate(d)

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world import lifecycle
        from crucible.world.world import World

        w = World(self.spec(), instructions={ACTOR: self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        async with lifecycle.opening(w):  # M10 lab audit R-6: a failure after start closes the World
            await w.reset()
        return w

    def score(self, ret: float) -> float:
        """The return mapped onto [0, 1]. A non-finite return raises: ``min(1, nan)`` is 1.0, so a simulator that blew up
        scored full reward (M10 audit V13)."""
        import math

        if not math.isfinite(ret):
            raise ValueError(f"non-finite return {ret}")
        lo, hi = self.return_range
        return max(0.0, min(1.0, (ret - lo) / (hi - lo)))

    async def grade(self, world, workdir: Path) -> GradeReport:
        import math

        j = world.apps[self.app].journal
        ret = float(j.get("return", 0.0))
        if not math.isfinite(ret):  # the simulator (or a remote environment) produced it: the environment's fault, masked
            return GradeReport(self.id, "full", [], {}, 0.0, 0.0, {"status": "masked", "failure_kind": "verifier_error",
                                                                  "reason": f"non-finite return {ret}"}, True, 0.0, meta={"return": str(ret)})
        s = self.score(ret)
        ok = self.success_return is None or ret >= self.success_return
        detail = {"return": round(ret, 6), "normalized_reward": round(s, 6), "ticks": j.get("tick", 0), "done": j.get("done", False), "score": ret}
        return GradeReport(self.id, "full", [CheckResult("return", "return", "pass" if s > 0 else "fail", detail)], {"return": "pass" if s > 0 else "fail"},
                           round(s, 6), float(ok), {"status": "ok"}, True, 0.0, meta={"kinds": {"return": "score"}, **detail})
