"""A pack's tasks as M9 ``AcceptTask`` s (M14 audit G2 D3 / pick 3): the Acceptance Contract runs on every pack, not only
on packs with a ``forge`` hook (``finance_ap``'s tasks never reached A5 or A3).

Which tasks: each fixture task (``hooks.fixtures`` cases grouped by task id: the case at the best end without a
``fails_gate`` is the oracle, the cases at the worst end are designed negatives, other best-end cases are valid
alternatives), each ``hooks.tasks`` anchor with its gold (and the fixture negatives of the same task id), and ``n_fresh``
generated items with their own oracle.

What the adapter provides: ``grade`` (a fresh World per solution, scripted actions, the task's own grader; the end-state
fingerprint from the State Adapters), ``start_fingerprint``, ``noop``, ``solution_of``, ``key_inputs``, ``grade_env`` and
``runtime_task``. Leak material defaults to:
- ``agent_visible``: the instruction and every text file under the World's ``data/`` directory (the pack's convention for
  what its Apps seed from);
- ``hidden_strings``: each oracle action serialized (a pasted answer key).
A task that knows better defines ``agent_visible()`` / ``hidden_strings()`` itself. Every other capability (stated
``properties``, ``checks``, ``attacks``, ``facts``, ...) is the task's own attribute or absent; absence is never a pass
(M9 reports ``MissingCapability`` and the verdict says so)."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from crucible.forge.schema import NegativeSpec

DELEGATED = ("properties", "checks", "attacks", "tamper_attempts", "isolation_audit", "facts", "declared_env", "perf_bars", "fresh_gates",
             "mutants", "generated_alternatives", "canary", "solver_trajectories", "audit_checks", "property_audit", "calibration_runs",
             "plan_tiers", "grade_tier", "split", "verifier_timeout_s", "attack_surface", "images", "query", "trap", "provenance", "horizon")


class PackAcceptTask:
    def __init__(self, task: Any, oracle: Any, negatives: list[NegativeSpec], *, pack_dir: Path, source: str):
        from crucible.studio.pack import as_dicts

        self._t = task
        self.source = source  # fixture | anchor | generated
        self.pack_dir = Path(pack_dir)
        self.task_id = str(getattr(task, "id", None) or getattr(task, "task_id", "task"))
        self.instruction = str(getattr(task, "instruction", "") or "")
        self.oracle = as_dicts(oracle)
        self.negatives = list(getattr(task, "negatives", None) or negatives)
        self.horizon = str(getattr(task, "horizon", None) or "H1")
        self.trap = getattr(task, "trap", None)
        self.provenance = list(getattr(task, "provenance", None) or [])
        for k in ("properties", "checks"):
            if hasattr(task, k):
                setattr(self, k, getattr(task, k))
        self._visible: dict[str, str] | None = None
        h = hashlib.sha256(json.dumps([self.task_id, self.instruction, self.oracle, [n.model_dump(mode="json") for n in self.negatives]],
                                      sort_keys=True, default=str).encode())
        self._plan_files = [p for p in (_source_file(type(task)),) if p is not None]
        for p in self._plan_files:
            h.update(p.read_bytes())
        self.task_hash = h.hexdigest()[:24]

    def __getattr__(self, k: str) -> Any:
        if k in DELEGATED and "_t" in self.__dict__:
            return getattr(self.__dict__["_t"], k)
        raise AttributeError(k)

    async def prepare(self) -> None:
        """Read what the agent can see once (A5 calls ``agent_visible`` synchronously)."""
        own = getattr(self._t, "agent_visible", None)
        if callable(own):
            self._visible = dict(own())
            return
        vis = {"instruction": self.instruction}
        d = Path(tempfile.mkdtemp(prefix="pack-accept-vis-"))
        try:
            w = await self._t.open(d / "work")
            try:
                root = getattr(w, "pack_root", None)
            finally:
                await w.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        data = Path(root) / "data" if root else None
        if data is not None and data.is_dir():
            for p in sorted(x for x in data.rglob("*") if x.is_file()):
                try:
                    vis[f"data/{p.relative_to(data)}"] = p.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
        self._visible = vis

    def agent_visible(self) -> dict[str, str]:
        if self._visible is None:
            raise RuntimeError("PackAcceptTask.prepare() was not awaited")
        return self._visible

    def hidden_strings(self) -> list[str]:
        own = getattr(self._t, "hidden_strings", None)
        if callable(own):
            return list(own())
        out: list[str] = []
        for a in self.oracle:
            for s in (json.dumps(a), json.dumps(a, sort_keys=True)):
                if len(s) >= 24 and s not in out:
                    out.append(s)
        return out

    async def grade(self, solution: Any, *, run: int = 0) -> tuple[Any, str]:
        from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
        from crucible.studio.pack import as_actions

        acts = as_actions(solution)
        d = Path(tempfile.mkdtemp(prefix="pack-accept-"))
        try:
            w = await self._t.open(d / "work")
            try:
                await run_episode(w, {"agent": ScriptedPolicy("agent", acts)}, limits=EpisodeLimits(max_steps=max(50, 2 * len(acts))), reset=False)
                rep = await self._t.grade(w, d / "grade")
                try:
                    fps = await w.verifier_view().fingerprints()
                except Exception:  # noqa: BLE001 - no state adapters: the verdict stands in for the state
                    fps = {"reward": repr(rep.reward), "gates": sorted((rep.gates or {}).items())}
            finally:
                await w.close()
            return rep, hashlib.sha256(json.dumps(fps, sort_keys=True, default=str).encode()).hexdigest()[:24]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def start_fingerprint(self) -> str:
        return (await self.grade([]))[1]

    def noop(self) -> list:
        return []

    def solution_of(self, n: NegativeSpec) -> list:
        return list(n.actions)

    def key_inputs(self) -> dict:
        return {"overlay_hash": self.task_hash, "plan_files": [str(p) for p in self._plan_files]}

    def grade_env(self) -> str:
        return hashlib.sha256(f"{self.task_hash}|{self.source}".encode()).hexdigest()[:24]

    def runtime_task(self) -> Any:
        return self._t

    def cleanup(self) -> None:
        c = getattr(self._t, "cleanup", None)
        if callable(c):
            with contextlib.suppress(Exception):
                c()


def _source_file(cls: type) -> Path | None:
    try:
        return Path(inspect.getfile(cls)).resolve()
    except (TypeError, OSError):
        return None


def _negatives_from_cases(tid: str, cases: list[dict], best: float, worst: float, oracle_name: str | None) -> list[NegativeSpec]:
    from crucible.studio.pack import _num, as_dicts

    out = []
    for c in cases:
        name = str(c.get("name"))
        if name == oracle_name or "actions" not in c:
            continue
        r = _num(c.get("reward"))
        acts = as_dicts(c["actions"])
        if r is not None and abs(r - worst) <= 1e-6:
            out.append(NegativeSpec(id=name, task_id=tid, kind="floor" if not acts else "cheat", actions=acts, expected_gate=c.get("fails_gate"),
                                    expected="fail", provenance="pack fixtures", description=f"designed negative {name}"))
        elif r is not None and abs(r - best) <= 1e-6 and not c.get("fails_gate"):
            out.append(NegativeSpec(id=name, task_id=tid, kind="valid_alternative", actions=acts, expected="pass", provenance="pack fixtures",
                                    description=f"valid alternative {name}"))
    return out


async def pack_accept_tasks(pack_dir: Path, m: Any, *, n_fresh: int = 1) -> list[PackAcceptTask]:
    from crucible.bench.run import _oracle_actions
    from crucible.studio.pack import GEN_SEEDS, _num, _Range, _split_cases, call_hook, resolve

    rng = _Range(m)
    groups: dict[str, list[dict]] = {}
    tasks_of: dict[str, Any] = {}
    if m.hooks.fixtures:
        cases, _ = _split_cases(await call_hook(pack_dir, m.hooks.fixtures))
        for c in cases:
            if isinstance(c, dict) and "task" in c and "actions" in c:
                tid = str(getattr(c["task"], "id", ""))
                groups.setdefault(tid, []).append(c)
                tasks_of.setdefault(tid, c["task"])
    out: list[PackAcceptTask] = []
    seen: set[str] = set()
    if m.hooks.tasks:
        for t, gold in list(await call_hook(pack_dir, m.hooks.tasks) or []):
            tid = str(getattr(t, "id", ""))
            out.append(PackAcceptTask(t, gold, _negatives_from_cases(tid, groups.get(tid, []), rng.best, rng.worst, None), pack_dir=pack_dir, source="anchor"))
            seen.add(tid)
    for tid, cs in groups.items():
        if tid in seen:
            continue
        gold = next((c for c in cs if (r := _num(c.get("reward"))) is not None and abs(r - rng.best) <= 1e-6 and not c.get("fails_gate")), None)
        if gold is None:
            continue
        out.append(PackAcceptTask(tasks_of[tid], gold["actions"], _negatives_from_cases(tid, cs, rng.best, rng.worst, str(gold.get("name"))),
                                  pack_dir=pack_dir, source="fixture"))
    if m.hooks.generator:
        gen = resolve(pack_dir, m.hooks.generator)
        for s in GEN_SEEDS[: max(0, n_fresh)]:
            t = gen(s)
            acts = _oracle_actions(t)
            if acts is None:
                continue  # semantic.generator reports it
            out.append(PackAcceptTask(t, acts, [], pack_dir=pack_dir, source="generated"))
    for t in out:
        await t.prepare()
    return out
