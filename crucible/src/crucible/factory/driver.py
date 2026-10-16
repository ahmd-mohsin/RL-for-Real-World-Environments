"""The per-task machine of M17 (05 §5-§7, 03 §2): one (intent, surface) cell in, one gate-accepted TaskCard out, or an
escalation.

The machine is data (``driver_machine()``), run by ``factory/machine.py``; this module adds the state functions. Order is
the funnel of 05 §11 (cheap deterministic checks before solvers, solvers before the audit):

    spec -> author -> realism -> oracle -> verifier -> ledger -> floors -> probes -> hackers -> second_verifier
         -> solvers -> differential -> audit -> flake -> horizon -> calibrate -> gate -> select -> accepted | dropped

- **Generation and verification never share a context.** ``verifier`` and ``second_verifier`` are ``isolated_from`` the
  oracle (the validator refuses any input path from it; findings carried back into them are screened for the oracle's
  canary and quoted shingles), and the second verifier is also isolated from the first. The oracle carries a canary.
- **Cross-family by config**: realism, the second verifier, the auditor and the hacker declare ``different_family_from``;
  ``ModelBook.check`` (or ``MachineRun(models=...)``) refuses an assignment that breaks it.
- **Every check stage settles one ``verdict``** ``{"ok", "owner", "findings", "measured"}``; failures loop back to the
  owner (``spec``, ``author``, ``oracle``, ``verifier``) with the findings. An owner nobody routes (an unexplained gate
  failure) and every exhausted loop escalate: they are decisions for a person, never ``done``.
- **Validation stages are adapters** onto agent R's modules: ``floors.run_floors``, ``probes.probe_verifier`` (the
  mutant-builder client as ``LLMProbeAuthor``), ``differential.differential`` (the second checkset materialised as a
  second task), ``audit.audit`` (the auditor client as ``Auditor``, seeded controls from the env), ``flake.measure_flake``,
  and ``horizon`` (milestone, concatenation and shortcut tests, for tasks the spec marks long). Hackers reuse
  ``forge/specialists.SpecialistHacker`` + ``forge/hacker.red_team``; the gate is ``accept/run.accept_task`` + ``verdict``;
  select is ``forge/judge.select``. Every adapter is injectable (``Stages``), and the LLM roles take injectable clients,
  so a run on fakes costs nothing. ``LANE_TASK`` lists what the lane's materialised task must provide.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.actors.llm import LLMOutputError
from crucible.factory.machine import MachineRun, Settle, StateContext

RECIPE_VERSION = "m17-v1"
OWNERS = ("spec", "author", "oracle", "verifier")
CAUSES = ("skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake")
# accept-contract checks whose failure has one owner (03 §2 "Gate: owner of the failed check"); any other blocker escalates
GATE_OWNERS = {"A1": "oracle", "A2": "verifier", "A3": "verifier", "A7": "verifier", "A13": "verifier", "A16": "verifier"}
LEDGER_AUTHOR_CODES = ("span_out_of_prompt", "span_mismatch")

# What ``FactoryEnv.materialize(draft)`` must return (the lane's job, agent S): an accept-contract task. The adapters use:
LANE_TASK = {
    "grade(solution)": "-> (GradeReport, fingerprint) | Graded-like, graded in a fresh World (floors, probes, hackers, differential, flake)",
    "noop() / oracle / instruction / negatives / solution_of(n)": "the floor set (floors.floors_for_task)",
    "property_audit(actions) -> bool": "optional: confirms hacks (forge/hacker.red_team); without it hacks are unconfirmable",
    "milestones_of(prefix) -> {milestone: bool}": "optional: long tasks (horizon.series_by_replay)",
}

StageVerdict = dict[str, Any]  # {"ok": bool, "owner": str | None, "findings": [str], "measured": {...}}
StageFn = Callable[[dict[str, Any], "FactoryEnv", StateContext], Awaitable[StageVerdict]]


# ---- the machine ------------------------------------------------------------------------------------------------------
def _in(name: str, src: str, typ: str = "json", optional: bool = False) -> dict[str, Any]:
    return {"name": name, "type": typ, "from": src, **({"optional": True} if optional else {})}


_V = [{"name": "verdict", "type": "json"}]
_INTENT = _in("intent", "params.intent")
_CELL = _in("cell", "params.cell")
_OBS = _in("observations", "params.observations", optional=True)
_SPEC = _in("spec", "spec.spec")
_WORLD = _in("world", "author.world")
_PROMPT = _in("prompt", "author.prompt", "text")
_SPANS = _in("spans", "author.spans")
_ORACLE = _in("oracle", "oracle.oracle")
_CHECKS = _in("checkset", "verifier.checkset")
_DRAFT = [_INTENT, _CELL, _OBS, _SPEC, _WORLD, _PROMPT, _SPANS, _ORACLE, _CHECKS]


def _routes(sid: str, nxt: str, default_owner: str | None) -> list[dict[str, Any]]:
    """ok -> next; a named owner -> that owner (findings carried); not ok -> the stage's default owner (or escalate)."""
    out = [{"from": sid, "to": nxt, "when": "verdict.ok == true"}]
    out += [{"from": sid, "to": o, "when": f"verdict.owner == '{o}'", "carry": "verdict.findings"} for o in OWNERS]
    if default_owner:
        out.append({"from": sid, "to": default_owner, "when": "verdict.ok == false", "carry": "verdict.findings"})
    return out


def driver_machine(*, max_spec: int = 2, max_author: int = 3, max_verifier: int = 4, max_transitions: int = 400,
                   budget_usd: float | None = None, budget_s: float | None = None) -> dict[str, Any]:
    """The per-task machine of 05 §5-§6 as data. Loops are bounded by the owners' ``max_entries``; downstream states
    may be re-entered once per verifier entry, so they share its bound."""
    down = max(max_verifier, max_author)
    llm = {"retries": 2, "timeout_s": 900.0}
    states: list[dict[str, Any]] = [
        {"id": "spec", "entry": True, "role": "planner", "max_entries": max_spec, **llm, "inputs": [_INTENT, _CELL, _OBS],
         "outputs": [{"name": "spec", "type": "json"}], "doc": "requirements, verifier plan, expected causes, milestones, surface"},
        {"id": "author", "role": "author", "max_entries": max_author, **llm, "inputs": [_INTENT, _CELL, _SPEC],
         "outputs": [{"name": "world", "type": "json"}, {"name": "prompt", "type": "text"}, {"name": "spans", "type": "json"}]},
        {"id": "realism", "role": "realism", "different_family_from": ["author"], "max_entries": max_author, **llm,
         "inputs": [_INTENT, _SPEC, _PROMPT], "outputs": _V},
        {"id": "oracle", "role": "oracle_author", "canary": True, "max_entries": down, **llm, "inputs": [_SPEC, _WORLD, _PROMPT],
         "outputs": [{"name": "oracle", "type": "json"}]},
        {"id": "verifier", "role": "verifier_writer", "isolated_from": ["oracle"], "max_entries": max_verifier, **llm,
         "inputs": [_INTENT, _OBS, _SPEC, _WORLD, _PROMPT], "outputs": [{"name": "checkset", "type": "json"}]},
        {"id": "ledger", "fn": "ledger", "max_entries": down, "inputs": [_OBS, _SPEC, _PROMPT, _SPANS, _CHECKS], "outputs": _V},
        {"id": "floors", "fn": "floors", "max_entries": down, "inputs": _DRAFT, "outputs": _V},
        {"id": "probes", "role": "mutant_builder", "max_entries": down, "retries": 1, "inputs": _DRAFT, "outputs": _V},
        {"id": "hackers", "role": "hacker", "different_family_from": ["verifier"], "max_entries": down, "retries": 1, "inputs": _DRAFT,
         "outputs": _V},
        {"id": "second_verifier", "role": "second_verifier", "different_family_from": ["verifier", "author"],
         "isolated_from": ["oracle", "verifier"], "max_entries": down, **llm, "inputs": [_INTENT, _OBS, _SPEC, _WORLD, _PROMPT],
         "outputs": [{"name": "checkset", "type": "json"}]},
        {"id": "solvers", "fn": "solvers", "max_entries": down, "retries": 1, "inputs": _DRAFT, "outputs": [{"name": "runs", "type": "json"}]},
        {"id": "differential", "fn": "differential", "max_entries": down,  # second checkset, runs
         "inputs": [*_DRAFT, _in("checkset2", "second_verifier.checkset"), _in("runs", "solvers.runs")], "outputs": _V},
        {"id": "audit", "role": "auditor", "different_family_from": ["author", "verifier"], "max_entries": down, "retries": 1,
         "inputs": [*_DRAFT, _in("runs", "solvers.runs"), _in("differential", "differential.verdict")], "outputs": _V},
        {"id": "flake", "fn": "flake", "max_entries": down, "inputs": _DRAFT, "outputs": _V},
        {"id": "horizon", "fn": "horizon", "max_entries": down, "inputs": [*_DRAFT, _in("runs", "solvers.runs")], "outputs": _V},
        {"id": "calibrate", "fn": "calibrate", "max_entries": down,
         "inputs": [*_DRAFT, _in("runs", "solvers.runs"), _in("audit", "audit.verdict")], "outputs": _V},
        {"id": "gate", "fn": "gate", "max_entries": down, "inputs": _DRAFT, "outputs": _V},
        {"id": "select", "fn": "select", "max_entries": down,
         "inputs": [*_DRAFT, *[_in(s, f"{s}.verdict") for s in ("floors", "probes", "hackers", "differential", "audit", "flake",
                                                                 "horizon", "calibrate", "gate")], _in("runs", "solvers.runs")],
         "outputs": [{"name": "verdict", "type": "json"}, {"name": "card", "type": "json"}]},
        {"id": "accepted", "terminal": "done"},
        {"id": "dropped", "terminal": "rejected"},
    ]
    transitions: list[dict[str, Any]] = [
        {"from": "spec", "to": "author"},
        {"from": "author", "to": "realism"},
        {"from": "realism", "to": "oracle", "when": "verdict.ok == true"},
        {"from": "realism", "to": "author", "when": "verdict.ok == false", "carry": "verdict.findings"},
        {"from": "oracle", "to": "verifier"},
        {"from": "verifier", "to": "ledger"},
        *_routes("ledger", "floors", "verifier"),
        *_routes("floors", "probes", "verifier"),
        *_routes("probes", "hackers", "verifier"),
        *_routes("hackers", "second_verifier", "verifier"),
        {"from": "second_verifier", "to": "solvers"},
        {"from": "solvers", "to": "differential"},
        {"from": "differential", "to": "audit"},  # disagreements are defects whose owner the auditor decides (05 §6.4)
        *_routes("audit", "flake", None),
        *_routes("flake", "horizon", "author"),
        *_routes("horizon", "calibrate", "spec"),
        *_routes("calibrate", "gate", "author"),
        *_routes("gate", "select", None),  # an unexplained gate failure escalates
        {"from": "select", "to": "accepted", "when": "verdict.keep == true"},
        {"from": "select", "to": "dropped", "when": "verdict.keep == false"},
    ]
    run: dict[str, Any] = {"max_transitions": max_transitions, "failure_policy": "escalate"}
    if budget_usd is not None:
        run["budget_usd"] = budget_usd
    if budget_s is not None:
        run["budget_s"] = budget_s
    return {"run": run, "states": states, "transitions": transitions}


# ---- the environment the state functions use ---------------------------------------------------------------------------
@dataclass
class Calibration:
    window: tuple[float, float] = (0.1, 0.8)  # 05 §7 v1: the target policy's pass rate
    min_skill_gap: float = 0.7  # 05 §7 v1: share of failures that are skill gaps
    min_policies: int = 2  # 05 §6.5
    min_attempts: int = 4
    target: str | None = None  # the policy under study (default: the first solver rung)
    harness: str = "native"


@dataclass
class Stages:
    """The validation adapters; ``None`` = the default (R's modules, forge, accept). Each is
    ``async (draft, env, ctx) -> StageVerdict``."""

    floors: StageFn | None = None
    probes: StageFn | None = None
    hackers: StageFn | None = None
    solvers: StageFn | None = None
    differential: StageFn | None = None
    audit: StageFn | None = None
    flake: StageFn | None = None
    horizon: StageFn | None = None
    gate: StageFn | None = None


@dataclass
class FactoryEnv:
    """What the per-task state functions may use. ``materialize(draft)`` builds the accept-contract task (the lane's job:
    ``factory/lanes/*``); ``solvers`` are ``(policy name, runner)`` rungs where ``runner(task, n) -> [trace]``."""

    stages: Stages = field(default_factory=Stages)
    materialize: Callable[[dict[str, Any]], Any] | None = None
    solvers: Sequence[tuple[str, Any]] = ()
    attempts: int = 4
    calibration: Calibration = field(default_factory=Calibration)
    specialists: Any = None  # root.Specialists (family hacker memory, World experts)
    tie_break: Any = None  # the tie-break client (z-ai/glm-4.7-flash by default)
    accept_cache: Path | None = None
    accept_needs: Sequence[Any] = ()
    allow_provisional: bool = False  # a gate verdict "provisional" (pending needs) counts as passing only when True
    variants: Callable[[str], Sequence[Any]] | None = None  # accepted judge.Variant siblings of a cell (root)
    keep_per_cell: int | None = None  # judge selection size per cell; None keeps every accepted task
    probe_context: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None  # extra probes.ProbeContext fields (tolerances, ...)
    seeded_controls: Callable[[dict[str, Any]], Any] | None = None  # audit.SeededCase list (an untested auditor cannot gate)
    isolated_rates: Callable[[dict[str, Any]], Any] | None = None  # long tasks: milestone -> (solved, attempts) as short tasks
    expert_minutes: float | None = None
    flake_config: Any = None  # flake.FlakeConfig
    horizon_config: Any = None  # horizon.HorizonConfig
    recipe_version: str = RECIPE_VERSION
    family: str | None = None


# ---- helpers ----------------------------------------------------------------------------------------------------------
async def _call(fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    out = fn(*a, **kw)
    return await out if inspect.isawaitable(out) else out


def _jsonable(x: Any) -> Any:
    """Finite JSON: NaN/inf -> None, tuples -> lists, dataclasses / models -> dicts."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return _jsonable(x.model_dump(mode="json"))
    if hasattr(x, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(x))
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


def verdict(ok: bool, *, owner: str | None = None, findings: Sequence[str] = (), measured: Mapping[str, Any] | None = None) -> StageVerdict:
    return {"ok": bool(ok), "owner": None if ok else owner, "findings": [str(f) for f in findings][:50], "measured": _jsonable(dict(measured or {}))}


def _verdict_of(rep: Any, *, owner: str | None) -> StageVerdict:
    """Normalise a stage report: a StageVerdict dict, or an object with ``ok`` and ``reasons()``/``findings`` and
    ``card()``/``measured``/``to_dict()``."""
    if isinstance(rep, Mapping):
        ok = bool(rep.get("ok"))
        return verdict(ok, owner=rep.get("owner", owner), findings=rep.get("findings") or [], measured=rep.get("measured") or {})
    ok = bool(getattr(rep, "ok", False))
    r = getattr(rep, "reasons", None)
    findings = r() if callable(r) else (getattr(rep, "findings", None) or [])
    m = getattr(rep, "card", None)
    measured = m() if callable(m) else (getattr(rep, "measured", None) or (rep.to_dict() if hasattr(rep, "to_dict") else {}))
    return verdict(ok, owner=getattr(rep, "owner", None) or owner, findings=findings, measured=measured if isinstance(measured, Mapping) else {"report": measured})


def _draft(ctx: StateContext) -> dict[str, Any]:
    d = dict(ctx.inputs)
    d["task_id"] = ctx.run_id
    d["observations"] = d.get("observations") or []
    return d


def _hash(x: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def _env(ctx: StateContext) -> FactoryEnv:
    env = ctx.services
    if not isinstance(env, FactoryEnv):
        raise TypeError("the driver's states need a FactoryEnv as the run's services")
    return env


async def _ask(ctx: StateContext, role_brief: str, payload: Mapping[str, Any], *, max_tokens: int = 4000) -> dict[str, Any]:
    if ctx.client is None:
        raise RuntimeError(f"state {ctx.state_id}: no client for role {ctx.role!r}")
    user = dict(payload)
    if ctx.findings:
        user["findings_to_address"] = ctx.findings
    msgs = [{"role": "system", "content": f"ROLE: {ctx.role}\n{role_brief}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(_jsonable(user), sort_keys=True)}]
    # (M17 e2e) a retry must be a new sample: with the LLM record cache, a retry of the same messages replayed the first
    # attempt's malformed reply ('{": ": ", "}' from deepseek-v4-flash), so every retry failed identically and the state paused
    kw: dict[str, Any] = {"sample_key": f"{ctx.state_id}#{ctx.entry}.{ctx.attempt}"} if ctx.attempt > 1 else {}
    out = await ctx.client.complete_json(msgs, max_tokens=max_tokens, temperature=0.0, **kw)
    if not isinstance(out, dict):
        raise LLMOutputError(f"{ctx.role}: expected a JSON object")
    return out


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise LLMOutputError(what)


def ledger_of(draft: Mapping[str, Any], checkset: Mapping[str, Any] | None = None) -> Any:
    """The ``cards.RequirementLedger`` of a draft: the spec's requirements with the author's spans, the checkset's checks."""
    from crucible.factory.cards import CheckRef, Requirement, RequirementLedger

    spans = draft.get("spans") or {}
    reqs = [Requirement(id=r["id"], text=r["text"], kind=r.get("kind") or "behaviour", prompt_span=tuple(spans.get(r["id"]) or (0, 1)))
            for r in (draft.get("spec") or {}).get("requirements") or []]
    checks = [CheckRef(id=c["id"], requirement_ids=list(c.get("requirement_ids") or []), observation_ids=list(c.get("observation_ids") or []),
                       gating=bool(c.get("gating", True))) for c in (checkset if checkset is not None else draft.get("checkset") or {}).get("checks") or []]
    return RequirementLedger(requirements=reqs, checks=checks)


def observations_of(draft: Mapping[str, Any]) -> dict[str, Any]:
    from crucible.factory.cards import ObservationRecord

    out = {}
    for o in draft.get("observations") or []:
        rec = o if isinstance(o, ObservationRecord) else ObservationRecord.model_validate(o)
        out[rec.id] = rec
    return out


async def _task(draft: dict[str, Any], env: FactoryEnv) -> Any:
    if env.materialize is None:
        raise RuntimeError("no lane: FactoryEnv.materialize(draft) is needed to build a gradeable task")
    return await _call(env.materialize, draft)


# ---- LLM role states --------------------------------------------------------------------------------------------------
async def s_spec(ctx: StateContext) -> Settle:
    out = await _ask(ctx, "You are the planner of an RL task. From the intent card and the cell, write the task spec: "
                          "requirements [{id, text, kind}] (each one prompt sentence; kind keys the probe library: laziness, "
                          "tolerance, compiled_path, behaviour, idempotence, minimality, side_effect_scope), verifier_plan, "
                          "expected_causes, milestones (for long tasks), surface {domain, world, tools, data_shape, prompt_style}, long (bool).",
                     {"intent": ctx.inputs["intent"], "cell": ctx.inputs["cell"]})
    reqs = out.get("requirements")
    _need(isinstance(reqs, list) and reqs and all(isinstance(r, dict) and r.get("id") and r.get("text") for r in reqs),
          "spec.requirements must be a non-empty list of {id, text, kind}")
    _need(len({r["id"] for r in reqs}) == len(reqs), "spec.requirements ids must be unique")
    spec = {"requirements": [{"id": str(r["id"]), "text": str(r["text"]), "kind": str(r.get("kind") or "behaviour")} for r in reqs],
            "verifier_plan": str(out.get("verifier_plan") or ""), "expected_causes": [c for c in out.get("expected_causes") or [] if c in CAUSES],
            "milestones": list(out.get("milestones") or []), "surface": dict(out.get("surface") or {}), "long": bool(out.get("long", False))}
    return Settle({"spec": spec})


async def s_author(ctx: StateContext) -> Settle:
    spec = ctx.inputs["spec"]
    out = await _ask(ctx, "You are the task author. Build the World (start state, tools, data) and the user prompt. The prompt "
                          "states every requirement in plain words a real user would use, and no hints a real user would not give. "
                          "Return world (object), prompt (string), spans {requirement id: [start, end]} (character offsets of the "
                          "prompt text stating each requirement).", {"intent": ctx.inputs["intent"], "cell": ctx.inputs["cell"], "spec": spec})
    prompt, world, spans = out.get("prompt"), out.get("world"), out.get("spans") or {}
    _need(isinstance(prompt, str) and prompt.strip() != "", "author.prompt must be a non-empty string")
    _need(isinstance(world, dict), "author.world must be an object")
    want = [r["id"] for r in spec["requirements"]]
    missing = [r for r in want if not (isinstance(spans.get(r), list) and len(spans[r]) == 2)]
    _need(not missing, f"author.spans must give [start, end] for every requirement; missing {missing}")
    return Settle({"world": world, "prompt": prompt, "spans": {r: [int(spans[r][0]), int(spans[r][1])] for r in want}})


async def s_realism(ctx: StateContext) -> Settle:
    out = await _ask(ctx, "You review a task prompt for realism. Reject hints a real user would not give, unrealistic requirement "
                          "lists, and rules that appear only in hidden checks. Return {ok: bool, findings: [one sentence each]}.",
                     {"intent": ctx.inputs["intent"], "spec": ctx.inputs["spec"], "prompt": ctx.inputs["prompt"]})
    _need(isinstance(out.get("ok"), bool), "realism.ok must be a boolean")
    v = verdict(out["ok"], owner="author", findings=[f"realism: {f}" for f in out.get("findings") or []])
    return Settle({"verdict": v}, v["findings"])


async def s_oracle(ctx: StateContext) -> Settle:
    out = await _ask(ctx, "You write the reference solution of this task, as the solution object the World accepts (tool calls "
                          "or a patch). Return {solution: ...}.", {"spec": ctx.inputs["spec"], "world": ctx.inputs["world"], "prompt": ctx.inputs["prompt"]})
    _need("solution" in out, "oracle.solution is required")
    return Settle({"oracle": {"solution": out["solution"], "notes": str(out.get("notes") or "")}})


_VERIFIER_BRIEF = ("You write the verifier of this task. You never see a solution. Write one or more checks per requirement id: "
                   "checks [{id, requirement_ids, observation_ids, gating, spec}] where spec is the check in the World's check grammar "
                   "and observation_ids name the ground-truth records the check rests on (from the observations given).")


def _checkset(out: Mapping[str, Any], role: str) -> dict[str, Any]:
    checks = out.get("checks")
    _need(isinstance(checks, list) and checks and all(isinstance(c, dict) and c.get("id") for c in checks),
          f"{role}.checks must be a non-empty list of checks with ids")
    return {"checks": [{"id": str(c["id"]), "requirement_ids": [str(x) for x in c.get("requirement_ids") or []],
                        "observation_ids": [str(x) for x in c.get("observation_ids") or []], "gating": bool(c.get("gating", True)),
                        "spec": c.get("spec")} for c in checks]}


async def s_verifier(ctx: StateContext) -> Settle:
    out = await _ask(ctx, _VERIFIER_BRIEF, {k: ctx.inputs.get(k) for k in ("intent", "spec", "world", "prompt", "observations")})
    return Settle({"checkset": _checkset(out, "verifier")})


async def s_second_verifier(ctx: StateContext) -> Settle:
    out = await _ask(ctx, _VERIFIER_BRIEF + " Work independently: your checks are compared with another verifier's on real end states.",
                     {k: ctx.inputs.get(k) for k in ("intent", "spec", "world", "prompt", "observations")})
    return Settle({"checkset": _checkset(out, "second_verifier")})


# ---- deterministic and adapter states ----------------------------------------------------------------------------------
async def s_ledger(ctx: StateContext) -> Settle:
    """05 §5 step 5: every requirement has a check, every check a requirement, every gating check a first-class
    observation (``cards.RequirementLedger.problems``); span problems belong to the author, the rest to the verifier."""
    from crucible.factory.ledger import unmapped_candidates

    d = _draft(ctx)
    led = ledger_of(d)
    probs = led.problems(observations_of(d), prompt=d["prompt"])
    author = [p for p in probs if p.split(":", 1)[0] in LEDGER_AUTHOR_CODES]
    owner = "author" if author and len(author) == len(probs) else "verifier"
    loose = [f"unmapped prompt sentence (flag, not a gate): {c.text if hasattr(c, 'text') else c}"
             for c in unmapped_candidates(d["prompt"], led.requirements)][:10]
    v = verdict(not probs, owner=owner, findings=probs or loose, measured={"requirements": len(led.requirements), "checks": len(led.checks),
                                                                          "problems": len(probs), "unmapped": len(loose)})
    return Settle({"verdict": v}, v["findings"])


async def _default_floors(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    from crucible.factory.floors import floors_for_task, grader, run_floors

    task = await _task(draft, env)
    rep = await run_floors(floors_for_task(task, prompt=draft["prompt"]), grader(task))
    return verdict(rep.ok, owner="verifier", findings=rep.reasons(), measured={"floors": rep.card()})


def _traces(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list((d.get("runs") or {}).get("traces") or [])


def _oracle_solution(d: Mapping[str, Any]) -> Any:
    return (d.get("oracle") or {}).get("solution")


async def _default_probes(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.2: per-requirement mutants and valid alternatives, executed (``probes.probe_verifier``)."""
    from crucible.factory.floors import grader
    from crucible.factory.probes import LLMProbeAuthor, ProbeContext, probe_verifier

    task = await _task(draft, env)
    led = ledger_of(draft)
    extra = dict(await _call(env.probe_context, draft)) if env.probe_context else {}
    pctx = ProbeContext(oracle=_oracle_solution(draft), prompt=draft["prompt"], requirements=list(led.requirements), **extra)
    rep = await probe_verifier(led, pctx, grader(task), author=LLMProbeAuthor(ctx.client) if ctx.client is not None else None)
    owner = "oracle" if rep.oracle_ok is False else "verifier"
    return verdict(rep.ok, owner=owner, findings=rep.reasons(),
                   measured={"mutant_kill": rep.mutant_kill, "alternatives_pass": rep.alternatives_pass, "weak": rep.weak_requirements,
                             "strict": rep.strict_requirements, "notes": rep.notes})


async def _default_differential(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.4: both checksets judge every solver end state; disagreements are defects whose owner the audit decides."""
    from crucible.factory.differential import EndState, differential
    from crucible.factory.floors import grader

    task_a = await _task(draft, env)
    task_b = await _task({**draft, "checkset": draft["checkset2"]}, env)
    states = [EndState(t["id"], t["solution"], {"policy": t.get("policy")}) for t in _traces(draft) if "solution" in t]
    rep = await differential(states, grader(task_a), grader(task_b), ledger_a=ledger_of(draft), ledger_b=ledger_of(draft, draft["checkset2"]))
    findings = [f"verifiers disagree on {x.item}: first {'pass' if x.primary else 'fail'}, second {'pass' if x.secondary else 'fail'}"
                f" (requirements {x.requirements})" for x in rep.disagreements]
    findings += [f"state mismatch (environment nondeterminism) on {i}" for i in rep.state_mismatch]
    return verdict(rep.ok, owner=None, findings=findings, measured={"differential_disagreements": rep.card, **rep.to_dict()})


# FailureCause owners (audit.OWNER_OF) -> the machine state that owns the fix
_AUDIT_OWNER = {"spec": "spec", "world": "author", "prompt": "author", "verifier": "verifier"}


async def _default_audit(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.5: a FailureCause per failing trace, FP/FN candidates, the cause mix (``audit.audit``, seeded controls
    included). An unresolved differential disagreement with a clean audit is a tie for the tie-break."""
    from crucible.factory.audit import Auditor, Trace, audit

    traces = [Trace(t["id"], str(t.get("text") or json.dumps(t.get("solution", t.get("end_state")), default=str)), t.get("passed"),
                    policy=str(t.get("policy", "")), failing_checks=list(t.get("failing_checks") or []), meta=dict(t.get("meta") or {}))
              for t in _traces(draft)]
    seeded = list(await _call(env.seeded_controls, draft)) if env.seeded_controls else []
    model = getattr(ctx.model, "ref", "") if ctx.model is not None else ""
    rep = await audit(traces, Auditor(ctx.client, model=model), prompt=draft["prompt"], requirements=list(ledger_of(draft).requirements),
                      seeded=seeded)
    send = rep.send_back
    owner = next((_AUDIT_OWNER[k] for k in ("spec", "world", "prompt", "verifier") if k in send), None)
    if owner is None and (rep.fp_candidates or rep.fn_candidates):
        owner = "verifier"
    sg = rep.skill_gap_share
    if owner is None and rep.auditor_trusted and sg.n and (sg.rate or 0.0) < rep.config.skill_gap_min:
        owner = "author"
    passing = sum(1 for t in traces if t.verifier_pass)
    failing = sum(1 for t in traces if t.verifier_pass is False)
    measured = {"causes": [{"trace": tid, "cause": lab.cause} for tid, lab in rep.labels.items() if lab.cause],
                "fp": (len(rep.fp_candidates), passing), "fn": (len(rep.fn_candidates), failing), **rep.to_dict()}
    ok, findings = rep.ok, rep.reasons()
    diff = draft.get("differential") or {}
    if ok and diff.get("ok") is False:
        ok, owner, findings = False, "tie", list(diff.get("findings") or ["the two verifiers disagree"])
    return verdict(ok, owner=owner, findings=findings, measured=measured)


async def _default_flake(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.6: oracle reruns at two concurrency levels, the Clopper-Pearson upper bound (``flake.measure_flake``)."""
    from crucible.factory.flake import FlakeConfig, measure_flake

    task = await _task(draft, env)
    sol = _oracle_solution(draft)

    async def run(i: int) -> Any:
        return await task.grade(sol)

    rep = await measure_flake(run, config=env.flake_config or FlakeConfig())
    return verdict(rep.ok, owner="verifier" if rep.verifier_nondeterministic else "author", findings=rep.reasons(),
                   measured={"flake_upper": rep.upper, **rep.to_dict()})


async def _default_horizon(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.7, for tasks the spec marks long: milestones from World state, order and non-redundancy, the concatenation
    test against isolated milestone rates, the shortcut probe (``factory/horizon.py``)."""
    from crucible.factory.horizon import HorizonConfig, concatenation_test, horizon_card, milestone_test, series_by_replay, shortcut_probe

    cfg = env.horizon_config or HorizonConfig()
    task = await _task(draft, env)
    oracle = _oracle_solution(draft) or []
    order = [str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in (draft.get("spec") or {}).get("milestones") or []]
    read = getattr(task, "milestones_of", None)
    series = await series_by_replay(oracle, read) if read is not None else []
    ms = milestone_test(series, order=order or None, config=cfg)
    iso = dict(await _call(env.isolated_rates, draft)) if env.isolated_rates else {}
    traces = _traces(draft)
    if order and iso:
        conc = concatenation_test([frozenset(t.get("milestones") or []) for t in traces], iso, order=order,
                                  full_success=[bool(t.get("passed")) for t in traces], config=cfg)
    else:
        conc = {"ok": False, "reasons": ["concatenation test not run: needs the milestone order and isolated milestone rates"]}
    sc = await shortcut_probe(oracle, task.grade, config=cfg)
    card = horizon_card(milestones=ms, concatenation=conc, shortcut=sc, expert_minutes=env.expert_minutes)
    return verdict(card["ok"], owner="spec", findings=card["reasons"], measured={"horizon": card})


async def _default_hackers(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    """05 §6.3: the family hacker (``forge/specialists.SpecialistHacker``: remembered strategies first, then its fallback)
    proposes, ``forge/hacker.red_team`` grades and audits. Confirmed hacks fail the task and are remembered for the family."""
    from crucible.forge.hacker import red_team

    task = await _task(draft, env)
    family = env.family or (draft.get("intent") or {}).get("id", "family")
    hacker = env.specialists.hacker(family, client=ctx.client) if env.specialists is not None else None
    if hacker is None:
        from crucible.forge.hacker import MutationHacker

        hacker = MutationHacker()
    sol = (draft.get("oracle") or {}).get("solution")
    view = {"prompt": draft["prompt"], "intent": draft.get("intent"), "checkset": draft.get("checkset"),
            "solver_trajectories": [sol] if isinstance(sol, list) else []}
    attempts = await _call(hacker.propose, view)
    audit = getattr(task, "property_audit", None)
    if audit is None:
        return verdict(True, measured={"hacks_confirmed": 0, "attempts": len(attempts), "audited": False},
                       findings=["no property audit on this task: hack attempts graded but not confirmable"] if attempts else [])
    from types import SimpleNamespace

    from crucible.factory.floors import graded_from

    async def grade(attempt: Any) -> Any:  # red_team grades HackAttempts and reads GradeReport.reward / .infra
        g = graded_from(await task.grade(attempt.actions))
        return SimpleNamespace(reward=g.reward, infra={"status": "masked"} if g.masked else {})

    rep = await red_team(attempts, grade, audit)
    hacks = rep.hacks
    if hacks and env.specialists is not None:
        env.specialists.remember_hacks(family, draft["task_id"], [h.attempt.strategy for h in hacks])
    return verdict(not hacks, owner="verifier", findings=[f"hack: {h.attempt.strategy} earned {h.reward} without the stated properties" for h in hacks],
                   measured={"hacks_confirmed": len(hacks), "attempts": len(rep.attempts)})


async def _default_solvers(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    task = await _task(draft, env)
    traces = []
    for name, runner in env.solvers:
        for i, tr in enumerate(await _call(runner, task, env.attempts)):
            traces.append({"id": f"{name}#{i}", "policy": name, **dict(tr)})
    return {"traces": traces}


async def _default_gate(draft: dict[str, Any], env: FactoryEnv, ctx: StateContext) -> StageVerdict:
    from crucible.accept.run import accept_task
    from crucible.accept.run import verdict as accept_verdict

    task = await _task(draft, env)
    cache = env.accept_cache or ((ctx.run_dir or Path(".")) / "accept")
    rep = await accept_task(task, cache_root=Path(cache), needs=list(env.accept_needs))
    v = accept_verdict(rep)
    ok = v["state"] == "accepted" or (env.allow_provisional and v["state"] == "provisional")
    owners = {GATE_OWNERS.get(c) for c in v["blocking"]}
    owner = owners.pop() if len(owners) == 1 and None not in owners else None
    return verdict(ok, owner=owner, findings=[f"gate {c}: {s}" for c, s in v["blocking"].items()], measured={"accept": v})


def _stage(name: str) -> Callable[[StateContext], Awaitable[Settle]]:
    defaults = {"floors": _default_floors, "probes": _default_probes, "hackers": _default_hackers, "differential": _default_differential,
                "flake": _default_flake, "horizon": _default_horizon, "gate": _default_gate}

    async def run(ctx: StateContext) -> Settle:
        env = _env(ctx)
        d = _draft(ctx)
        if name == "horizon" and not (d.get("spec") or {}).get("long"):
            v = verdict(True, measured={"horizon": None, "applies": False})
            return Settle({"verdict": v})
        fn = getattr(env.stages, name) or defaults[name]
        v = _verdict_of(await fn(d, env, ctx), owner=None)
        return Settle({"verdict": v}, v["findings"])

    run.__name__ = f"s_{name}"
    return run


async def s_solvers(ctx: StateContext) -> Settle:
    env = _env(ctx)
    fn = env.stages.solvers or _default_solvers
    runs = await fn(_draft(ctx), env, ctx)
    traces = list((runs or {}).get("traces") or [])
    per: dict[str, int] = {}
    for t in traces:
        per[t["policy"]] = per.get(t["policy"], 0) + 1
    cal = env.calibration
    if len(per) < cal.min_policies or any(n < cal.min_attempts for n in per.values()):
        raise RuntimeError(f"solvers: need >= {cal.min_policies} policies x >= {cal.min_attempts} attempts (05 §6.5), got {per}")
    return Settle({"runs": _jsonable({"traces": traces})})


async def s_audit(ctx: StateContext) -> Settle:
    env = _env(ctx)
    d = _draft(ctx)
    fn = env.stages.audit or _default_audit
    v = _verdict_of(await fn(d, env, ctx), owner=None)
    if v["owner"] == "tie":
        # the auditor could not name the defect's owner: one tie-break call on another family (05 §12)
        msgs = [{"role": "system", "content": "ROLE: tie_break\nA verifier defect needs an owner. Choose one of spec, author, oracle, "
                                              "verifier, from the evidence. Answer with one JSON object {owner, reason}."},
                {"role": "user", "content": json.dumps(_jsonable({"prompt": d["prompt"], "spec": d["spec"], "audit": v,
                                                                   "differential": d.get("differential")}), sort_keys=True)}]
        owner = None
        if env.tie_break is not None:
            tb = await env.tie_break.complete_json(msgs, max_tokens=800, temperature=0.0)
            owner = tb.get("owner") if isinstance(tb, dict) and tb.get("owner") in OWNERS else None
        v = {**v, "owner": owner, "findings": [*v["findings"], f"tie-break owner: {owner}"]}
    return Settle({"verdict": v}, v["findings"])


def _rate(outcomes: Sequence[Any], policy: str, harness: str) -> dict[str, Any]:
    from crucible.forge.learnability import estimate, learnability

    r = estimate([None if o is None else bool(o) for o in outcomes], model=policy, harness=harness)
    return {"policy": policy, "harness": harness, "k": r.k, "n": r.n, "p": round(r.p, 4), "wilson": [r.lo, r.hi], "masked": r.masked,
            "learnability": round(learnability(r.p), 4)}


async def s_calibrate(ctx: StateContext) -> Settle:
    """05 §7: pass rate per (policy, harness) with Wilson intervals; the target policy in the window; >= 70% of
    failures are skill gaps; difficulty moves only along the intent's axes."""
    from crucible.forge.learnability import Band, advise, estimate

    env = _env(ctx)
    d = _draft(ctx)
    cal = env.calibration
    traces = (d.get("runs") or {}).get("traces") or []
    by: dict[str, list[Any]] = {}
    for t in traces:
        by.setdefault(t["policy"], []).append(t.get("passed"))
    rates = {p: _rate(o, p, cal.harness) for p, o in by.items()}
    target = cal.target or (next(iter(by)) if by else None)
    causes = [c.get("cause") for c in ((d.get("audit") or {}).get("measured") or {}).get("causes") or []]
    failed = [t for t in traces if t.get("passed") is False]
    skill = sum(1 for c in causes if c == "skill_gap")
    mix = (skill / len(causes)) if causes else (1.0 if not failed else 0.0)
    passes = [all(t.get("passed") for t in traces if t["policy"] == p) for p in by]
    batch = "all_pass" if all(passes) and by else ("all_fail" if not any(t.get("passed") for t in traces) else "mixed")
    findings: list[str] = []
    ok = True
    if target not in rates:
        ok = False
        findings.append(f"no solver runs for the target policy {target!r}")
    else:
        tr = rates[target]
        lo, hi = cal.window
        if not lo <= tr["p"] <= hi:
            ok = False
            axes = ", ".join((d.get("intent") or {}).get("difficulty_axes") or []) or "the intent's difficulty axes"
            for m in advise(estimate([None if o is None else bool(o) for o in by[target]], model=target, harness=cal.harness), Band(lo, hi)):
                findings.append(f"calibrate: target {target} p={tr['p']:.2f} outside [{lo}, {hi}]: make the task {m.direction} "
                                f"along {axes} only ({m.reason})")
    if causes and mix < cal.min_skill_gap:
        ok = False
        findings.append(f"calibrate: only {mix:.0%} of failures are skill gaps (needs {cal.min_skill_gap:.0%}): hard for the wrong reason")
    stats = {"rates": rates, "target": target, "cause_mix": {c: causes.count(c) for c in sorted(set(causes))}, "skill_gap_share": round(mix, 4),
             "batch": batch, "window": list(cal.window)}
    v = verdict(ok, owner="author", findings=findings, measured={"stats": stats})
    return Settle({"verdict": v}, v["findings"])


def _pair(x: Any) -> tuple[int, int]:
    if isinstance(x, Mapping):
        return int(x.get("k", 0)), int(x.get("n", 0))
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    return (0, 0)


def task_card(d: Mapping[str, Any], *, recipe_version: str, family: str | None = None) -> dict[str, Any]:
    """The ``cards.TaskCard`` of an accepted draft (validated), as JSON."""
    from crucible.factory.cards import TaskCard, VerifierCard

    m = {s: ((d.get(s) or {}).get("measured") or {}) for s in ("floors", "probes", "hackers", "differential", "audit", "flake", "horizon",
                                                                 "calibrate", "gate")}
    floors = {k: float(v) for k, v in (m["floors"].get("floors") or {}).items() if isinstance(v, (int, float))}
    vc = VerifierCard(verifier_version=_hash(d.get("checkset"))[:23], floors=floors, mutant_kill=_pair(m["probes"].get("mutant_kill")),
                      alternatives_pass=_pair(m["probes"].get("alternatives_pass")), hacks_confirmed=int(m["hackers"].get("hacks_confirmed") or 0),
                      differential_disagreements=_pair(m["differential"].get("differential_disagreements")), fp=_pair(m["audit"].get("fp")),
                      fn=_pair(m["audit"].get("fn")), flake_upper=m["flake"].get("flake_upper"), horizon=m["horizon"].get("horizon"))
    intent = d.get("intent") or {}
    cell = d.get("cell") or {}
    card = TaskCard(id=str(d["task_id"]), family=str(family or intent.get("id") or "family"), intent=str(intent.get("id") or "intent"),
                    surface={k: str(v) for k, v in ((d.get("spec") or {}).get("surface") or cell.get("surface_spec") or {}).items()},
                    prompt=d["prompt"], ledger=ledger_of(d), world_ref=_hash(d.get("world")), checkset_ref=_hash(d.get("checkset")),
                    oracle_ref=_hash({k: v for k, v in (d.get("oracle") or {}).items() if k != "_canary"}), verifier=vc,
                    stats=_jsonable(m["calibrate"].get("stats") or {}),
                    lineage={"cell": cell, "recipe_version": recipe_version, "accept": m["gate"].get("accept")},
                    recipe_version=recipe_version, status="accepted")
    return card.model_dump(mode="json")


async def s_select(ctx: StateContext) -> Settle:
    """The Judge (``forge/judge.select``: a selector, never a synthesizer) keeps the best accepted variants of a cell."""
    from crucible.forge.judge import Variant, select

    env = _env(ctx)
    d = _draft(ctx)
    cal_stats = ((d.get("calibrate") or {}).get("measured") or {}).get("stats") or {}
    target = (cal_stats.get("rates") or {}).get(cal_stats.get("target")) or {}
    cell_key = str((d.get("cell") or {}).get("key") or ctx.run_id)
    mine = Variant(task_id=ctx.run_id, seed=cell_key, operator=env.recipe_version, accepted=True, l2=target.get("p"),
                   gates=len((d.get("checkset") or {}).get("checks") or []))
    keep = True
    if env.keep_per_cell is not None:
        siblings = list(env.variants(cell_key)) if env.variants else []
        keep = ctx.run_id in select([*siblings, mine], env.keep_per_cell)
    card = task_card(d, recipe_version=env.recipe_version, family=env.family)
    v = verdict(True, measured={"variant": {"task_id": mine.task_id, "seed": mine.seed, "l2": mine.l2, "gates": mine.gates}})
    v["keep"] = keep
    return Settle({"verdict": v, "card": card})


def handlers() -> dict[str, Callable[[StateContext], Awaitable[Settle]]]:
    return {"spec": s_spec, "author": s_author, "realism": s_realism, "oracle": s_oracle, "verifier": s_verifier, "ledger": s_ledger,
            "floors": _stage("floors"), "probes": _stage("probes"), "hackers": _stage("hackers"), "second_verifier": s_second_verifier,
            "solvers": s_solvers, "differential": _stage("differential"), "audit": s_audit, "flake": _stage("flake"),
            "horizon": _stage("horizon"), "calibrate": s_calibrate, "gate": _stage("gate"), "select": s_select}


def task_run(*, run_dir: Path, intent: Mapping[str, Any], cell: Mapping[str, Any], env: FactoryEnv, clients: Mapping[str, Any],
             models: Mapping[str, Any] | None = None, observations: Sequence[Any] = (), ledger: Any = None, run_id: str | None = None,
             machine: Mapping[str, Any] | None = None) -> MachineRun:
    """A journaled per-task machine run for one cell (resumes when ``run_dir`` already holds its journal). ``clients`` maps
    role -> client (``ModelBook.clients()``, or fakes); ``models`` (role -> ModelSpec; default the configured ``ModelBook``)
    is what the cross-family rule is checked against."""
    if models is None:  # the cross-family rule is always checked: the configured assignment (05 §12 defaults)
        from crucible.factory.models import ModelBook

        models = ModelBook.from_config().roles
    params = {"intent": _jsonable(dict(intent)), "cell": _jsonable(dict(cell)),
              "observations": [_jsonable(o) for o in observations]}
    return MachineRun(machine or driver_machine(), run_dir=run_dir, handlers=handlers(), run_id=run_id, params=params, models=models,
                      clients=clients, ledger=ledger, services=env)
