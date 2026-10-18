"""The ``LongTask`` contract (swarm-pipeline/00 §1): one contract every task generator emits (mined, stitched and
swarm-built windows, rebreak, open), so the machine (§4), the verifier gates (§5), the targeting probes (§7) and the RL
wrappers (§11) are written once.

| member | content |
|---|---|
| ``task_id`` | the task's id |
| ``world`` | the World at the start state: ``{kind, repos, image, digest, base_dir, network: "none", history: "cut", ...}`` |
| ``prompt``, ``requirements`` | the user-voice request; the two-way ledger requirement <-> milestone (``requirement["milestone"]``) |
| ``milestones`` | hidden checks on the end state: ``{id, tests, requires}``; every ``requires`` edge **executed** (fails alone, passes with the prerequisite) |
| ``path_rules`` | ``factory.pathcheck.PathRules``: protected paths, forbidden operations, required actions, step budget |
| ``grade_patch(patch, workdir=None, *, trail=None)`` | the grader owns the build and the run; reward = phi_weight * Phi + (1 - phi_weight) * [every gate, path gates included]; a hack-severity path violation zeroes it. A ``GradeReport`` with per-milestone and per-path results |
| ``milestone_status(patch)`` | milestone id -> passed, for the checkpoint sidecar (raises when unmeasured) |
| ``oracle`` | the executed oracle solution (an attribute, as the accept contract reads it: ``AcceptTask.oracle``) |
| ``oracle_patch()`` | the oracle as the patch an agent would submit (what ``grade_patch`` takes) |
| ``floors()``, ``alternatives()`` | ``GroundCase`` lists: must score 0 (Phi 0) / 1 |
| ``mutants()`` | must fail their milestone: each has ``.id``, ``.solution`` and ``.milestone`` |
| ``simplify(kind, **kw)`` | the difficulty-removed variant for the counterfactual control (§7): a ``LongTask`` |
| ``provenance`` | observation ids binding the checks to real observations |
| ``card(**kw)`` | the ``cards.TaskCard`` |

**Deviation from the spec table:** the oracle is the attribute ``oracle`` (a solution value) plus ``oracle_patch()``, not
a method ``oracle()``: every accept-contract check reads ``task.oracle`` as a value, and a method of that name would break
them.

``check_longtask`` is the runtime conformance checker (static: members, types, milestone DAG, executed edges, the
ledger); ``ground_truth`` executes oracle, floors, mutants and alternatives; ``null_agent_score`` and
``clean_world_solvable`` are the S8 and S5 checks of §4 and §5.3, callable on any ``LongTask``. ``LongTaskRef`` is
the serialisable pointer (the slime bridge's ``crucible_task_ref``).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from crucible.factory.pathcheck import PathRules

SIMPLIFY_KINDS = ("prerequisites", "host_change", "faulty_function", "baseline_bar")  # windows / stitch / rebreak / open (§7)
MEMBERS = ("task_id", "world", "prompt", "requirements", "milestones", "path_rules", "oracle", "provenance")
METHODS = ("grade_patch", "milestone_status", "oracle_patch", "floors", "mutants", "alternatives", "simplify", "card")
ASYNC_METHODS = ("grade_patch", "milestone_status")


@dataclass(frozen=True)
class GroundCase:
    """One executed ground-truth end state: ``kind`` oracle (must score 1) | floor (0, Phi 0) | alternative (1) |
    mutant (must fail ``milestone``). ``solution`` is what the task's ``grade_patch`` takes (a patch)."""

    id: str
    kind: Literal["oracle", "floor", "alternative", "mutant"]
    solution: Any
    milestone: str | None = None
    note: str = ""


@runtime_checkable
class LongTask(Protocol):
    task_id: str
    world: Mapping[str, Any]
    prompt: str
    requirements: Sequence[Mapping[str, Any]]
    milestones: Sequence[Mapping[str, Any]]
    path_rules: PathRules
    oracle: Any
    provenance: Sequence[str]

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None) -> Any: ...

    async def milestone_status(self, patch: str) -> dict[str, bool]: ...

    def oracle_patch(self) -> str: ...

    def floors(self) -> list[GroundCase]: ...

    def mutants(self) -> list[Any]: ...

    def alternatives(self) -> list[GroundCase]: ...

    def simplify(self, kind: str, **kw: Any) -> LongTask: ...

    def card(self, **kw: Any) -> Any: ...


@dataclass(frozen=True)
class LongTaskRef:
    """A serialisable pointer to one task: the lane, the store, the source (window) id, and hashes of what makes the task
    (prompt, check set, verifier version) so a ref resolved later is known to be the same task. ``simplified`` names the
    ``simplify`` variant (kind and pre-applied milestones) when the task is one."""

    lane: str
    task_id: str
    source_id: str
    store: str
    split: str
    prompt_sha256: str
    checkset_sha256: str
    verifier_version: str
    simplified: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> LongTaskRef:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)


# ---- conformance ---------------------------------------------------------------------------------------------------------
def _ids(task: Any) -> list[str]:
    return [str(m.get("id")) for m in task.milestones]


def _executed(m: Mapping[str, Any], r: str, by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    """Is the edge m -> r backed by an executed probe? ``requires_evidence[r] == "executed"`` (generic) or, for windows,
    the prober's label of r's unit in ``m["edges"]``."""
    ev = m.get("requires_evidence")
    if isinstance(ev, Mapping):
        return ev.get(r) == "executed"
    edges = m.get("edges")
    unit = (by_id.get(r) or {}).get("unit")
    return isinstance(edges, Mapping) and unit is not None and edges.get(unit) == "executed"


def _cycle(by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(i: str) -> list[str]:
        state[i] = 1
        stack.append(i)
        for r in by_id[i].get("requires") or []:
            if r not in by_id:
                continue
            if state.get(r) == 1:
                return stack[stack.index(r):] + [r]
            if r not in state and (c := visit(r)):
                return c
        state[i] = 2
        stack.pop()
        return []

    for i in by_id:
        if i not in state and (c := visit(i)):
            return c
    return []


def check_longtask(task: Any, *, long_horizon: bool = True, need_prompt: bool = True) -> list[str]:
    """Every way ``task`` breaks the ``LongTask`` contract (empty = conforms). Static only: nothing is graded.

    ``long_horizon``: at least one executed ``requires`` edge (§2: long-horizon means intrinsic dependency); a task without
    one is labelled short and routed out (S6). ``need_prompt``: the task is authored (a prompt, requirements, a two-way
    ledger); off for an unauthored window graded for its ground truth."""
    from crucible.factory.cards import TaskCard

    out = [f"member: missing {m}" for m in MEMBERS if not hasattr(task, m)]
    out += [f"method: missing {m}()" for m in METHODS if not callable(getattr(task, m, None))]
    if out:
        return out
    out += [f"method: {m}() must be async" for m in ASYNC_METHODS if not inspect.iscoroutinefunction(getattr(task, m))]
    if "trail" not in inspect.signature(task.grade_patch).parameters:
        out.append("method: grade_patch() takes no trail= (the path checks read the trail)")
    w = task.world
    if not isinstance(w, Mapping):
        out.append("world: not a mapping")
    else:
        out += [f"world: no {k}" for k in ("kind", "image", "base_dir") if not w.get(k)]
        if w.get("network") != "none":
            out.append(f"world: network is {w.get('network')!r}, not 'none'")
        if w.get("history") not in ("cut", "fresh"):
            out.append(f"world: history is {w.get('history')!r}, not cut")
    if not isinstance(task.path_rules, PathRules):
        out.append("path_rules: not a pathcheck.PathRules")
    if task.oracle is None:
        out.append("oracle: none")
    if not list(task.provenance or []):
        out.append("provenance: no observation")
    ms = list(task.milestones or [])
    ids = _ids(task)
    if not ms:
        out.append("milestones: none")
    if len(set(ids)) != len(ids):
        out.append(f"milestones: duplicate ids {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {str(m.get("id")): m for m in ms}
    edges = 0
    for m in ms:
        mid = str(m.get("id"))
        if not list(m.get("tests") or []):
            out.append(f"milestone {mid}: no hidden check")
        for r in m.get("requires") or []:
            if r == mid or r not in by_id:
                out.append(f"milestone {mid}: requires {r!r}, not another milestone")
            elif not _executed(m, r, by_id):
                out.append(f"milestone {mid}: requires {r} without an executed edge")
            else:
                edges += 1
    if (c := _cycle(by_id)):
        out.append(f"milestones: requires cycle {' -> '.join(c)}")
    if long_horizon and ms and edges == 0:
        out.append("short: no executed requires edge (long, not long-horizon)")
    reqs = list(task.requirements or [])
    if need_prompt:
        if not str(task.prompt or "").strip():
            out.append("prompt: empty")
        mapped = {r.get("milestone") for r in reqs if r.get("milestone")}
        out += [f"ledger: requirement {r.get('id')} maps to unknown milestone {r.get('milestone')}" for r in reqs
                if r.get("milestone") and r.get("milestone") not in by_id]
        out += [f"ledger: milestone {i} has no requirement" for i in ids if i not in mapped]
    try:
        cases = [*task.floors(), *task.alternatives()]
        out += [f"ground truth: {getattr(c, 'id', c)!r} is not a GroundCase" for c in cases if not isinstance(c, GroundCase)]
        if not any(c.kind == "floor" for c in cases if isinstance(c, GroundCase)):
            out.append("floors: none")
    except Exception as e:  # noqa: BLE001 -- a broken member is a conformance finding, not a crash
        out.append(f"ground truth: floors()/alternatives() raised {type(e).__name__}: {e}"[:300])
    try:
        for mu in task.mutants():
            if getattr(mu, "milestone", None) not in by_id:
                out.append(f"mutant {getattr(mu, 'id', mu)}: no target milestone")
    except Exception as e:  # noqa: BLE001
        out.append(f"mutants() raised {type(e).__name__}: {e}"[:300])
    if need_prompt:
        try:
            card = task.card()
            if not isinstance(card, TaskCard):
                out.append("card: not a cards.TaskCard")
        except Exception as e:  # noqa: BLE001
            out.append(f"card: {type(e).__name__}: {e}"[:300])
    return out


# ---- executed checks (S5, S8) ------------------------------------------------------------------------------------------
def _row(rep: Any) -> dict[str, Any]:
    m = rep.meta or {}
    return {"reward": rep.reward, "phi": m.get("phi"), "full_pass": m.get("full_pass"), "milestones": m.get("milestones"),
            "infra": (rep.infra or {}).get("status"), "path_violations": (m.get("path") or {}).get("violations", []),
            "hack_suspect": bool(m.get("hack_suspect"))}


def new_file_patch(path: str, text: str) -> str:
    """A git patch that adds one text file."""
    lines = text.splitlines()
    body = "".join(f"+{ln}\n" for ln in lines)
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"
            + ("" if text.endswith("\n") or not text else "\\ No newline at end of file\n"))


def status_claim_patch(task: Any) -> str:
    """The null agent that claims everything: STATUS.json with every requirement at 1.0 and no other change."""
    ids = [str(r.get("id")) for r in task.requirements or [] if r.get("milestone")] or _ids(task)
    return new_file_patch("STATUS.json", json.dumps(dict.fromkeys(ids, 1.0), indent=1) + "\n")


async def null_agent_score(task: Any) -> dict[str, Any]:
    """S8's null agent (§5.3): the empty patch and a patch that only claims success (STATUS.json) must both score reward 0
    with Phi 0. ``ok`` is False when either scores above 0 or is unmeasured (unmeasured never passes)."""
    rows = {"empty": _row(await task.grade_patch("", trail=[])), "claims_only": _row(await task.grade_patch(status_claim_patch(task), trail=[]))}
    bad = [k for k, r in rows.items() if r["infra"] != "ok" or r["reward"] != 0.0 or (r["phi"] or 0.0) != 0.0]
    return {"ok": not bad, "rows": rows, "failed": bad}


async def clean_world_solvable(task: Any, *, in_world: bool = False, workdir: Path | None = None) -> dict[str, Any]:
    """S5's solvability (§5.3; ``EVALS`` L1: 2 of 20 example tasks were unsolvable and no pass rate showed it): the oracle,
    submitted as a patch with a clean trail, scores reward 1 with Phi 1 and no path violation.

    ``in_world``: the patch is applied inside a freshly opened World (``task.open``), collected back with the harness's
    own snapshot (``task.snapshot_patch``) and graded: the oracle survives the agent's real round trip. Needs ``workdir``
    and a task with ``open``, ``snapshot_patch`` and ``apply_patch_in_world``."""
    patch = task.oracle_patch()
    out: dict[str, Any] = {"patch_bytes": len(patch)}
    if in_world:
        if workdir is None or not all(callable(getattr(task, a, None)) for a in ("open", "snapshot_patch", "apply_patch_in_world")):
            raise TypeError("in_world needs a workdir and a task with open(), snapshot_patch() and apply_patch_in_world()")
        world = await task.open(Path(workdir))
        try:
            await task.apply_patch_in_world(world, patch)
            patch = await task.snapshot_patch(world)
        finally:
            await world.close()
        out["collected_bytes"] = len(patch)
    row = _row(await task.grade_patch(patch, trail=[]))
    out.update(row)
    out["ok"] = row["infra"] == "ok" and row["reward"] == 1.0 and row["phi"] == 1.0 and not row["path_violations"]
    return out


async def ground_truth(task: Any) -> dict[str, Any]:
    """Every executed ground truth of a task on the RL path (``grade_patch``): the oracle and alternatives score 1, the
    floors 0 with Phi 0, each mutant fails its milestone (and never earns the full-pass bonus)."""
    import asyncio

    cases = [GroundCase("oracle_patch", "oracle", task.oracle_patch()), *task.floors(), *task.alternatives()]
    cases += [GroundCase(str(m.id), "mutant", m.solution, milestone=getattr(m, "milestone", None)) for m in task.mutants()]
    reps = await asyncio.gather(*[task.grade_patch(c.solution, trail=[]) for c in cases])
    rows: dict[str, Any] = {}
    bad: list[str] = []
    for c, rep in zip(cases, reps, strict=True):
        r = {"kind": c.kind, **_row(rep)}
        if r["infra"] != "ok":
            ok = False
        elif c.kind in ("oracle", "alternative"):
            ok = r["reward"] == 1.0 and r["phi"] == 1.0
        elif c.kind == "floor":
            ok = r["reward"] == 0.0 and (r["phi"] or 0.0) == 0.0
        else:
            ok = (r["milestones"] or {}).get(c.milestone) is False and not r["full_pass"]
        r["ok"] = ok
        rows[c.id] = r
        if not ok:
            bad.append(c.id)
    return {"ok": not bad, "failed": bad, "rows": rows}
