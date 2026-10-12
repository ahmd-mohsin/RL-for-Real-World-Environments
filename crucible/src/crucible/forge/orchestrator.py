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
- **Economics** (m13 G-6, ``economy.py``): with a ``YieldBook``, every plan run is recorded (generator, operator, seed,
  family, construction state, spend), and a plan whose generator the stop rules paused is not run (state ``paused``).
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

from crucible.actors.llm import LLMOutputError, SpendScope
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
    timeout_s: float | None = 1800.0  # a role run past this is an infra attempt (sweep G4: a hung role held the plan forever)
    backoff_s: float = 0.5  # infra retries wait backoff_s * 2**k (sweep G2/G4: retries were immediate)

    def needs(self) -> list[str]:
        return sorted({v[1:] for v in self.inputs.values() if v.startswith("@")})


@dataclass
class Plan:
    task_id: str
    generator: str
    nodes: list[Node]
    knobs: dict[str, Any] = field(default_factory=dict)
    operator: str = ""  # the composition operator / derivation that made it (yield, quota)
    seed: str = ""  # the seed task it derives from (quota)
    family: str = ""  # the task family (cost cap, quota)

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
        self.torn = 0
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    self._index(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # a torn tail from a crash mid-append (M8-M10 sweep G1 D3: the forge journal could not be opened at
                    # all): skip it, and cut it off below so the next append starts on a fresh line
                    self.torn += 1
            if raw and not raw.endswith(b"\n"):
                with self.path.open("ab") as f:
                    f.write(b"\n")

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


def _fingerprint(v: Any) -> str:
    """A validator's identity for the input hash: its qualified name and an optional ``version`` attribute."""
    f = getattr(v, "func", v)  # functools.partial
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}@{getattr(v, 'version', getattr(f, 'version', 0))}"


def content_hash(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return _hash(obj)


class Orchestrator:
    def __init__(self, store: Store, journal: Journal, ledger: Ledger, roles: dict[str, Role], *, run_id: str = "forge", node_concurrency: int = 8,
                 book: Any = None):
        self.store, self.journal, self.ledger, self.roles, self.run_id = store, journal, ledger, roles, run_id
        self.book = book  # forge.economy.YieldBook (m13 G-6)
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
        why = self.book.paused(plan.generator) if self.book is not None else None
        if why:
            return ForgeResult(plan.task_id, "paused", {}, {}, f"generator {plan.generator} paused: {why}")
        r = await self._run_plan(plan)
        if self.book is not None:
            self.book.record_construction(plan.task_id, plan.generator, operator=plan.operator, seed=plan.seed, family=plan.family, state=r.state,
                                          cost_usd=sum(n.cost_usd for n in r.nodes.values() if not n.reused))
        return r

    async def _run_plan(self, plan: Plan) -> ForgeResult:
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
                    try:
                        return node, await self._run_node(plan, node, outputs)
                    except Exception as ex:  # noqa: BLE001 -- staging or store errors end this node, never the run
                        return node, NodeResult(node.id, "abandoned", findings=[f"orchestrator: {type(ex).__name__}: {ex}"[:500]])

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
        if not bad and prior is not None and prior.get("state") in ("audited", "accepted") and results and all(r.reused for r in results.values()):
            # nothing changed: every output is the journaled one, so the assembled task is the one already judged (M8
            # audit V18: a no-op re-run reset an accepted task to constructed)
            state, reason = prior["state"], prior.get("reason")
        else:
            self.set_state(plan.task_id, state, reason=reason)
        docs = {nid: self.store.get_json(k) for nid, k in outputs.items()}
        return ForgeResult(plan.task_id, state, results, docs, reason)

    async def run_plans(self, plans: list[Plan], *, concurrency: int = 5) -> list[ForgeResult]:
        """Many plans at once (the swarm knob C; GitSwarm's default 5), each still level-parallel inside; results in plan
        order. The ledger is shared, so concurrent reservations still never exceed the cap."""
        sem = asyncio.Semaphore(concurrency)

        async def one(p: Plan) -> ForgeResult:
            async with sem:
                try:
                    return await self.run_plan(p)
                except Exception as ex:  # noqa: BLE001 -- one plan's failure never aborts the others (M8 audit V4)
                    why = f"{type(ex).__name__}: {ex}"[:500]
                    self.set_state(p.task_id, "abandoned", reason=why)
                    return ForgeResult(p.task_id, "abandoned", {}, {}, why)

        return list(await asyncio.gather(*(one(p) for p in plans)))

    async def _run_node(self, plan: Plan, node: Node, outputs: dict[str, str]) -> NodeResult:
        role = self.roles[node.role]
        self._sees.setdefault(plan.task_id, {})[node.id] = self._sees_of(plan, node)
        keys = self._resolve(plan, node, outputs)
        visible = node.inputs.keys() if node.visible is None else node.visible
        # the validators and the canary flag are part of the key (M8 audit V5, V6): a stricter validator or a newly planted
        # canary must not reuse an output that was never checked against it
        ih = _hash(node.role, getattr(role, "version", "0"), {k: keys[k] for k in sorted(visible)}, plan.knobs, node.id,
                   [_fingerprint(v) for v in node.validators], node.canary_out)
        hit = self.journal.reusable(ih)
        if hit is not None and self.store.has(hit["output_key"]) and (hit.get("canary") or not node.canary_out):
            if node.canary_out:
                self._canaries.setdefault(plan.task_id, {})[node.id] = hit["canary"]
            return NodeResult(node.id, "done", hit["output_key"], 0, reused=True)
        staged = self._stage(node, keys)
        feedback: list[str] = []
        t0 = time.time()
        attempts = infra = 0
        total = 0.0  # every attempt's spend, retries and failures included (G-6: $ per accepted task)
        who = f"{plan.task_id}/{node.id}"
        while attempts < node.max_attempts:
            try:
                self.ledger.reserve(who, node.estimate_usd)
            except BudgetExhausted as ex:
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "abandoned", "reason": str(ex)})
                return NodeResult(node.id, "abandoned", findings=[str(ex)], attempts=attempts, cost_usd=total)
            spent0 = self.ledger.observed()
            err: BaseException | None = None
            out = None
            with SpendScope() as scope:
                try:
                    coro = role.run(staged, feedback=list(feedback), knobs=dict(plan.knobs))
                    out = await (asyncio.wait_for(coro, node.timeout_s) if node.timeout_s else coro)
                except TimeoutError:
                    err = RoleInfraError(f"role {node.role} timed out after {node.timeout_s:.0f}s")
                except Exception as ex:  # noqa: BLE001 -- classified below; the spend is settled whatever happened
                    err = ex
            # this node's own spend: what its role booked in its scope (M8 audit V3: the global delta also held every
            # concurrent node's calls); a role with no scope falls back to the delta, clamped to the observed total
            cost = self.ledger.settle(who, scope.usd, clamp_to_observed=False) if scope.booked else \
                self.ledger.settle(who, self.ledger.observed() - spent0, clamp_to_observed=True)
            total += cost
            if isinstance(err, RoleInfraError):
                infra += 1
                if infra >= 3:
                    return NodeResult(node.id, "abandoned", findings=[f"infra: {err}"], attempts=attempts, cost_usd=total)
                await asyncio.sleep(node.backoff_s * 2 ** (infra - 1))
                continue
            if isinstance(err, BudgetExhausted):  # the role's own client cap: settled above, never a leaked reservation (V4)
                return NodeResult(node.id, "abandoned", findings=[str(err)], attempts=attempts, cost_usd=total)
            if isinstance(err, (LLMOutputError, ValidationError)):
                attempts += 1
                feedback = [f"your previous output could not be parsed: {str(err)[:500]}"]
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "retry", "attempt": attempts, "findings": feedback})
                continue
            if err is not None:  # a bug in the role (V4: a KeyError aborted every plan of the run): this node only
                why = f"role error: {type(err).__name__}: {err}"[:500]
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "abandoned", "reason": why})
                return NodeResult(node.id, "abandoned", findings=[why], attempts=attempts, cost_usd=total)
            attempts += 1
            text = out.model_dump_json()
            breached = self._breach(plan, node, text)
            if breached:
                self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "breach", "sources": breached})
                return NodeResult(node.id, "breach", findings=[f"output contains material from {breached} it was not allowed to see"], attempts=attempts, cost_usd=total)
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
                                     "output_key": key, "attempts": attempts, "cost_usd": round(total, 6), "canary": canary,
                                     "visible": sorted(visible), "input_keys": {k: keys[k] for k in sorted(visible)}})
                return NodeResult(node.id, "done", key, attempts, cost_usd=total, seconds=round(time.time() - t0, 3))
            feedback = findings
            self.journal.append({"event": "node", "task_id": plan.task_id, "node": node.id, "status": "retry", "attempt": attempts, "findings": findings[:10]})
        return NodeResult(node.id, "failed", findings=feedback, attempts=attempts, cost_usd=total, seconds=round(time.time() - t0, 3))
