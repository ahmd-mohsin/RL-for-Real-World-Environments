"""The forge orchestrator (DESIGN §7.2, ARCHITECTURE M8): a deterministic DAG of role nodes, with a journal, budgets,
retries that carry feedback, and asymmetric information enforced by what is staged.

- A **plan** is one forged task: nodes, each a role run with named inputs. An input is a store key, or ``@<node>`` for
  an upstream node's output. ``visible`` lists the inputs the role may see. Roles never get a store handle: they receive
  only the staged values, so a role cannot read what it must not (the Test writer cannot open the oracle), whatever its
  prompt says (Repo2RLEnv: a prompt instruction is a nudge, not a control).
- Nodes run in topological order, sorted by id within a level (deterministic). Each node's **input hash** is
  ``H(role, role.version, visible input keys, knobs)``. A journal hit with status ``done`` reuses the stored output, so
  an interrupted forge resumes where it stopped and re-running costs nothing (RTS ``publish.py`` pattern).
- A role's output goes through deterministic **validators**. Their findings become **feedback** for the next attempt
  (at most ``max_attempts``, default 3). Infrastructure errors (``RoleInfraError``, an exhausted budget) never count as
  content attempts. Malformed role output (``LLMOutputError``: not JSON, truncated; a pydantic ``ValidationError``)
  counts as a content attempt whose feedback is the parse error: one bad completion never aborts the plan.
- **Budgets**: a reserve/settle ledger per forge run (``budget.py``); a node that cannot reserve its estimate is
  ``abandoned(budget)``, never half-run.
- **Canaries**: oracle material can be planted with a canary string; any later node whose output contains it fails
  (an asymmetric-information breach is a hard error, not a warning).
- **Task states** (journaled): ``draft -> constructed -> audited -> accepted | rejected(reason) | abandoned(reason)``.
  ``accepted`` is bound to the content hash of the assembled task (any edit sends it back).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from crucible.actors.llm import LLMOutputError
from crucible.core.store import Store
from crucible.forge.budget import BudgetExhausted, Ledger


class RoleInfraError(RuntimeError):
    """The role could not run (provider down, sandbox failed): retried as infra, never a content attempt."""


class BreachError(RuntimeError):
    """An asymmetric-information breach: a role's output contains material it was not allowed to see."""


class Role(Protocol):
    name: str
    version: str

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> BaseModel: ...


Validator = Callable[[BaseModel, dict[str, Any]], list[str]]  # (output, staged inputs) -> findings (empty = ok)


@dataclass
class Node:
    id: str
    role: str
    inputs: dict[str, str] = field(default_factory=dict)  # name -> store key | "@node_id" | literal JSON via "=..."
    visible: list[str] | None = None  # None = every input
    validators: list[Validator] = field(default_factory=list)
    max_attempts: int = 3
    estimate_usd: float = 0.0
    canary_out: bool = False  # this node's output carries oracle material: plant and track a canary

    def needs(self) -> list[str]:
        return sorted({v[1:] for v in self.inputs.values() if v.startswith("@")})


@dataclass
class Plan:
    task_id: str
    generator: str
    nodes: list[Node]
    knobs: dict[str, Any] = field(default_factory=dict)

    def order(self) -> list[Node]:
        return [n for level in self.levels() for n in level]

    def levels(self) -> list[list[Node]]:
        """Dependency levels: every node of a level depends only on earlier levels (those run concurrently)."""
        by = {n.id: n for n in self.nodes}
        missing = {d for n in self.nodes for d in n.needs()} - set(by)
        if missing:
            raise ValueError(f"plan {self.task_id}: unknown upstream nodes {sorted(missing)}")
        levels: list[list[Node]] = []
        seen: set[str] = set()
        while len(seen) < len(self.nodes):
            level = sorted((n for n in self.nodes if n.id not in seen and set(n.needs()) <= seen), key=lambda n: n.id)
            if not level:
                raise ValueError(f"plan {self.task_id}: cycle among {sorted(set(by) - seen)}")
            levels.append(level)
            seen |= {n.id for n in level}
        return levels


@dataclass
class NodeResult:
    node: str
    status: str  # done | failed | breach | abandoned (budget/infra) | skipped (an upstream node did not finish)
    output_key: str | None = None
    attempts: int = 0
    findings: list[str] = field(default_factory=list)
    reused: bool = False
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass
class ForgeResult:
    task_id: str
    state: str
    nodes: dict[str, NodeResult]
    outputs: dict[str, Any]
    reason: str | None = None


class Journal:
    """Append-only ``events.jsonl``; the latest ``done`` event for an input hash is reusable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[str, dict] = {}
        self._states: dict[str, list[dict]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._index(json.loads(line))

    def _index(self, ev: dict) -> None:
        if ev.get("event") == "node" and ev.get("status") == "done":
            self._done[ev["input_hash"]] = ev
        elif ev.get("event") == "node" and ev.get("status") == "breach" and ev.get("input_hash"):
            self._done.pop(ev["input_hash"], None)  # a late breach revokes the cached output: a resume never reuses it
        if ev.get("event") == "state":
            self._states.setdefault(ev["task_id"], []).append(ev)

    def append(self, ev: dict) -> None:
        ev = {"ts": round(time.time(), 3), **ev}
        with self.path.open("a") as f:
            f.write(json.dumps(ev, default=str) + "\n")
            f.flush()
        self._index(ev)

    def reusable(self, input_hash: str) -> dict | None:
        return self._done.get(input_hash)

    def state(self, task_id: str) -> dict | None:
        evs = self._states.get(task_id)
        return evs[-1] if evs else None

    def history(self, task_id: str) -> list[dict]:
        return list(self._states.get(task_id, []))


def _hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def content_hash(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return _hash(obj)


class Orchestrator:
    def __init__(self, store: Store, journal: Journal, ledger: Ledger, roles: dict[str, Role], *, run_id: str = "forge", node_concurrency: int = 8):
        self.store, self.journal, self.ledger, self.roles, self.run_id = store, journal, ledger, roles, run_id
        # per plan (task id): node id -> canary planted in its output / upstream nodes reaching it through visible inputs.
        # Keyed per plan: concurrent plans reuse node ids ("oracle", "tests"), and one plan's canary must never be checked
        # (or overwritten) in another's
        self._canaries: dict[str, dict[str, str]] = {}
        self._sees: dict[str, dict[str, set[str]]] = {}
        self._node_slots = asyncio.Semaphore(node_concurrency)

    @property
    def canaries(self) -> dict[str, str]:
        """All canaries (node id -> canary) of the plans run so far, flattened (last plan wins on a reused node id)."""
        return {k: v for c in self._canaries.values() for k, v in c.items()}

    def set_state(self, task_id: str, state: str, *, reason: str | None = None, task_hash: str | None = None) -> None:
        self.journal.append({"event": "state", "task_id": task_id, "state": state, "reason": reason, "task_hash": task_hash})

    def _resolve(self, plan: Plan, node: Node, outputs: dict[str, str]) -> dict[str, str]:
        keys = {}
        for name, ref in node.inputs.items():
            if ref.startswith("@"):
                keys[name] = outputs[ref[1:]]
            elif ref.startswith("="):
                keys[name] = self.store.put_bytes(ref[1:].encode())
            else:
                keys[name] = ref
        return keys

    def _stage(self, node: Node, keys: dict[str, str]) -> dict[str, Any]:
        visible = node.inputs.keys() if node.visible is None else node.visible
        unknown = set(visible) - set(node.inputs)
        if unknown:
            raise ValueError(f"node {node.id}: visible names {sorted(unknown)} are not inputs")
        staged = {}
        for name in sorted(visible):
            raw = self.store.get_bytes(keys[name])
            try:
                staged[name] = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                staged[name] = raw.decode("utf-8", "replace")
        return staged

    def _sees_of(self, plan: Plan, node: Node) -> set[str]:
        visible = set(node.inputs.keys() if node.visible is None else node.visible)
        direct = {ref[1:] for name, ref in node.inputs.items() if ref.startswith("@") and name in visible}
        out = set(direct)
        sees = self._sees.setdefault(plan.task_id, {})
        for d in direct:
            out |= sees.get(d, set())
        return out

    def _breach(self, plan: Plan, node: Node, text: str) -> list[str]:
        """Canaries of upstream outputs that cannot reach this node through visible inputs must not appear in it."""
        allowed = self._sees_of(plan, node)
        return sorted(nid for nid, c in self._canaries.get(plan.task_id, {}).items() if nid not in allowed and nid != node.id and c in text)

    async def run_plan(self, plan: Plan) -> ForgeResult:
        outputs: dict[str, str] = {}
        results: dict[str, NodeResult] = {}
        prior = self.journal.state(plan.task_id)
        if prior is None:
            self.set_state(plan.task_id, "draft")
        for level in plan.levels():  # a level's nodes are independent: they run concurrently (bounded by node_concurrency)
            runnable = []
            for node in level:
                failed_up = [d for d in node.needs() if results.get(d) and results[d].status != "done"]
                if failed_up:
                    results[node.id] = NodeResult(node.id, "skipped", findings=[f"upstream did not finish: {failed_up}"])
                else:
                    runnable.append(node)

            async def one(node: Node) -> tuple[Node, NodeResult]:
                async with self._node_slots:
                    return node, await self._run_node(plan, node, outputs)

            for node, res in await asyncio.gather(*(one(n) for n in runnable)):
                if res.status == "done" and not res.reused:
                    # a same-level node's canary may exist only now: re-scan the output against every canary this plan knows
                    breached = self._breach(plan, node, self.store.get_bytes(res.output_key).decode(errors="replace"))
                    if breached:
                        ih = next((h for h, ev in self.journal._done.items() if ev.get("task_id") == plan.task_id and ev.get("node") == node.id
                                   and ev.get("output_key") == res.output_key), None)
                        self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "breach", "sources": breached,
                                             "late": True, "input_hash": ih})
                        res = NodeResult(node.id, "breach", findings=[f"output contains material from {breached} it was not allowed to see"],
                                         attempts=res.attempts)
                results[node.id] = res
                if res.status == "done":
                    outputs[node.id] = res.output_key
        bad = [r for r in results.values() if r.status != "done"]
        # a content failure or breach rejects the task; budget or infra exhaustion abandons it (it may be retried later)
        state = "constructed" if not bad else ("rejected" if any(r.status in ("failed", "breach") for r in bad) else "abandoned")
        reason = None if not bad else "; ".join(f"{r.node}: {r.status} {r.findings[:2]}" for r in bad)[:800]
        self.set_state(plan.task_id, state, reason=reason)
        docs = {nid: self.store.get_json(k) for nid, k in outputs.items()}
        return ForgeResult(plan.task_id, state, results, docs, reason)

    async def run_plans(self, plans: list[Plan], *, concurrency: int = 5) -> list[ForgeResult]:
        """Many plans at once (the swarm knob C; GitSwarm's default 5), each still level-parallel inside; results in plan
        order. The ledger is shared, so concurrent reservations still never exceed the cap."""
        sem = asyncio.Semaphore(concurrency)

        async def one(p: Plan) -> ForgeResult:
            async with sem:
                return await self.run_plan(p)

        return list(await asyncio.gather(*(one(p) for p in plans)))

    async def _run_node(self, plan: Plan, node: Node, outputs: dict[str, str]) -> NodeResult:
        role = self.roles[node.role]
        self._sees.setdefault(plan.task_id, {})[node.id] = self._sees_of(plan, node)
        keys = self._resolve(plan, node, outputs)
        visible = node.inputs.keys() if node.visible is None else node.visible
        ih = _hash(node.role, getattr(role, "version", "0"), {k: keys[k] for k in sorted(visible)}, plan.knobs, node.id)
        hit = self.journal.reusable(ih)
        if hit is not None and self.store.has(hit["output_key"]):
            if node.canary_out and hit.get("canary"):
                self._canaries.setdefault(plan.task_id, {})[node.id] = hit["canary"]
            return NodeResult(node.id, "done", hit["output_key"], 0, reused=True)
        staged = self._stage(node, keys)
        feedback: list[str] = []
        t0 = time.time()
        attempts = infra = 0
        while attempts < node.max_attempts:
            try:
                self.ledger.reserve(f"{plan.task_id}/{node.id}", node.estimate_usd)
            except BudgetExhausted as ex:
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "abandoned", "reason": str(ex)})
                return NodeResult(node.id, "abandoned", findings=[str(ex)], attempts=attempts)
            spent0 = self.ledger.observed()
            try:
                out = await role.run(staged, feedback=list(feedback), knobs=dict(plan.knobs))
            except RoleInfraError as ex:
                self.ledger.settle(f"{plan.task_id}/{node.id}", self.ledger.observed() - spent0)
                infra += 1
                if infra >= 3:
                    return NodeResult(node.id, "abandoned", findings=[f"infra: {ex}"], attempts=attempts)
                await asyncio.sleep(0)
                continue
            except BudgetExhausted as ex:
                return NodeResult(node.id, "abandoned", findings=[str(ex)], attempts=attempts)
            except (LLMOutputError, ValidationError) as ex:
                self.ledger.settle(f"{plan.task_id}/{node.id}", self.ledger.observed() - spent0)
                attempts += 1
                feedback = [f"your previous output could not be parsed: {str(ex)[:500]}"]
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "retry", "attempt": attempts, "findings": feedback})
                continue
            cost = self.ledger.observed() - spent0
            self.ledger.settle(f"{plan.task_id}/{node.id}", cost)
            attempts += 1
            text = out.model_dump_json()
            breached = self._breach(plan, node, text)
            if breached:
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "breach", "sources": breached})
                return NodeResult(node.id, "breach", findings=[f"output contains material from {breached} it was not allowed to see"], attempts=attempts)
            findings = [f for v in node.validators for f in v(out, staged)]
            if not findings:
                payload = out.model_dump(mode="json")
                canary = None
                if node.canary_out:
                    canary = "CRUCIBLE-CANARY-" + _hash(plan.task_id, node.id, ih)[:16]
                    payload = {**payload, "_canary": canary}
                    self._canaries.setdefault(plan.task_id, {})[node.id] = canary
                key = self.store.put_bytes(json.dumps(payload, sort_keys=True).encode())
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "role": node.role, "status": "done", "input_hash": ih,
                                     "output_key": key, "attempts": attempts, "cost_usd": round(cost, 6), "canary": canary,
                                     "visible": sorted(visible), "input_keys": {k: keys[k] for k in sorted(visible)}})
                return NodeResult(node.id, "done", key, attempts, cost_usd=cost, seconds=round(time.time() - t0, 3))
            feedback = findings
            self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "retry", "attempt": attempts, "findings": findings[:10]})
        return NodeResult(node.id, "failed", findings=feedback, attempts=attempts, seconds=round(time.time() - t0, 3))
