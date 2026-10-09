"""World audit: the checks every World must pass before anyone trains or evaluates on it.

Check ids follow OpenEnv RFC 008 (environment auto-validation) where one exists; a check that cannot run (no
gold trajectory, no verifier predicate, no isolation) is reported as a named SKIP, never as a PASS. Sources:
OpenEnv ``validation/`` (ids, severities, report shape), NeMo-Gym ``verifier_fixture.py`` (full / zero reward,
determinism), verifiers ``vf-validate`` (setup / gold / noop), Repo2RLEnv ``control_evidence`` (nop and oracle
runs bound to the bundle hash) and its isolation probe.

The report is bound to the World's ``spec_hash``: a changed spec needs a new audit.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from crucible.core.schema.actions import Action, Done, ToolCall
from crucible.world.world import VerifierView, World

Status = Literal["pass", "fail", "skip"]
Severity = Literal["critical", "major", "minor"]
Verifier = Callable[[VerifierView], Awaitable[bool]]


@dataclass
class CheckResult:
    id: str
    status: Status
    severity: Severity
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    world: str
    spec_hash: str
    checks: list[CheckResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" and c.severity in ("critical", "major") for c in self.checks)

    def summary(self) -> str:
        bad = [f"{c.id} ({c.severity})" for c in self.checks if c.status == "fail"]
        skipped = [c.id for c in self.checks if c.status == "skip"]
        return (f"{self.world}: {'PASS' if self.ok else 'FAIL'} - {sum(c.status == 'pass' for c in self.checks)} passed"
                + (f", failed: {', '.join(bad)}" if bad else "") + (f", skipped: {', '.join(skipped)}" if skipped else ""))

    def to_dict(self) -> dict[str, Any]:
        return {"world": self.world, "spec_hash": self.spec_hash, "ok": self.ok, "seconds": round(self.seconds, 1), "checks": [asdict(c) for c in self.checks]}


async def _run(world: World, actor: str, actions: list[Action]) -> None:
    for i, a in enumerate(actions):
        if isinstance(a, ToolCall):
            a = a.model_copy(update={"tool_call_id": f"audit-{i}"})
        if isinstance(a, Done):
            break
        await world.step(actor, a)


async def audit_world(world: World, *, gold: list[Action] | None = None, verifier: Verifier | None = None,
                      forbidden: list[str] | None = None, actor: str | None = None) -> AuditReport:
    """Audit a started World. ``gold``: an oracle trajectory for the policy actor; ``verifier``: a predicate over
    the verifier view that the gold end state satisfies and the initial state does not; ``forbidden``: strings that
    must never reach the policy (oracle snippets, hidden test names, answers)."""
    from crucible.core.schema.actions import openai_name
    from crucible.world.leaks import leak_tokens

    t0 = time.monotonic()
    rep = AuditReport(world=world.spec.world, spec_hash=world.spec_hash)
    add = rep.checks.append
    actor = actor or world.spec.policy_actors[0].id

    # --- runtime.reset_stable / runtime.state_contract
    await world.reset()
    ep1, f1 = world.episode_id, (await world.snapshot()).fingerprint
    await world.reset()
    ep2, f2 = world.episode_id, (await world.snapshot()).fingerprint
    add(CheckResult("runtime.reset_stable", "pass" if f1 == f2 else "fail", "major", {"fingerprints": [f1[:16], f2[:16]]}))
    add(CheckResult("runtime.state_contract", "pass" if ep1 != ep2 and world.step_index == 0 else "fail", "major",
                    {"episode_ids_differ": ep1 != ep2, "step_after_reset": world.step_index}))

    # --- tool_declaration_accuracy (schemas valid, names unique and API-safe, descriptions present)
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    tools = world.all_tools()
    problems = []
    names = [openai_name(t.qualified) for t in tools]
    if len(set(names)) != len(names):
        problems.append("function names collide after sanitizing")
    for t in tools:
        try:
            Draft202012Validator.check_schema(t.input_schema or {"type": "object"})
        except SchemaError as e:
            problems.append(f"{t.qualified}: invalid schema ({e.message[:120]})")
        if not (t.description or "").strip():
            problems.append(f"{t.qualified}: no description")
    add(CheckResult("tool_declaration_accuracy", "fail" if problems else "pass", "minor", {"tools": len(tools), "problems": problems[:20]}))

    # --- tool_smoke: read-only tools without required arguments never crash (internal_error / infra fault)
    crashed, called = [], 0
    for t in tools:
        if not t.read_only or (t.input_schema or {}).get("required"):
            continue
        r = await world._call_tool(actor, ToolCall(tool_call_id="smoke", function_name=t.qualified, arguments={}), system=True)
        called += 1
        if r.fault == "infra":
            crashed.append(f"{t.qualified}: {r.error}")
    await world.reset()
    add(CheckResult("tool_smoke", "fail" if crashed else "pass", "major", {"called": called, "crashed": crashed}))

    # --- network.answer_sources (static): an allowlist that opens a code host or a public package index hands the policy
    # the published fix (Repo2RLEnv: GitHub, PyPI wheels and sdists, mirrors). A world that needs packages serves a
    # date-frozen mirror under its own name instead; a public-network world is flagged for the same reason.
    from crucible.world.guard.guard import ANSWER_SOURCES, allowed

    if world.spec.network == "allowlist":
        opened = sorted(h for h in ANSWER_SOURCES if allowed(h, world.spec.allowed_hosts))
        add(CheckResult("network.answer_sources", "fail" if opened else "pass", "critical", {"allowed": list(world.spec.allowed_hosts), "answer_sources_opened": opened}))
    elif world.spec.network == "public":
        add(CheckResult("network.answer_sources", "fail", "major", {"reason": "public network: published fixes, wheels and mirrors are reachable"}))
    else:
        add(CheckResult("network.answer_sources", "pass", "critical", {"network": world.spec.network}))

    # --- no_solution_leakage (static): fix pointers in what the policy reads before acting
    texts = {"instruction": world.instructions.get(actor, "")} | {f"tool:{t.qualified}": t.description or "" for t in world.tools_for(actor)}
    leaks = {k: leak_tokens(v) for k, v in texts.items() if leak_tokens(v)}
    hits = {k: [s for s in forbidden or [] if s and s in v] for k, v in texts.items()}
    hits = {k: v for k, v in hits.items() if v}
    add(CheckResult("no_solution_leakage.static", "fail" if leaks or hits else "pass", "critical", {"leak_tokens": leaks, "forbidden": hits}))

    # --- episode isolation, determinism, oracle / floor gap (need a gold trajectory)
    initial = world.initial.fingerprint if world.initial else f1
    if gold:
        await world.reset()
        nop_ok = await verifier(world.verifier_view()) if verifier else None
        await _run(world, actor, gold)
        g1 = await world.snapshot(quiesce=True)
        gold_ok = await verifier(world.verifier_view()) if verifier else None
        observed = " ".join(e.text or "" for e in world.log if e.kind in ("observation", "message") and e.actor in (actor, None) or (e.data or {}).get("to") == actor)
        await world.reset()
        back = (await world.snapshot()).fingerprint
        await _run(world, actor, gold)
        g2 = await world.snapshot(quiesce=True)
        await world.reset()
        add(CheckResult("runtime.episode_isolation", "pass" if g1.fingerprint != initial and back == initial else "fail", "major",
                        {"gold_changed_state": g1.fingerprint != initial, "reset_restored_initial": back == initial}))
        add(CheckResult("episode_determinism", "pass" if g1.fingerprint == g2.fingerprint else "fail", "major",
                        {"apps_differing": sorted(a for a in g1.apps if g1.apps[a].fingerprint != g2.apps[a].fingerprint)}))
        leaked = [s for s in forbidden or [] if s and s in observed]
        add(CheckResult("no_solution_leakage.dynamic", "fail" if leaked else "pass", "critical", {"forbidden_seen": leaked}))
        if verifier:
            add(CheckResult("oracle_passes", "pass" if gold_ok else "fail", "critical", {}))
            add(CheckResult("floor_gap.nop_fails", "pass" if not nop_ok else "fail", "critical", {}))
        else:
            for cid in ("oracle_passes", "floor_gap.nop_fails"):
                add(CheckResult(cid, "skip", "critical", {"reason": "no verifier predicate given"}))
    else:
        for cid in ("runtime.episode_isolation", "episode_determinism", "no_solution_leakage.dynamic", "oracle_passes", "floor_gap.nop_fails"):
            add(CheckResult(cid, "skip", "major", {"reason": "no gold trajectory given"}))

    # --- containment (isolated container worlds): egress, planted anomalies, root-only paths
    iso = world.backend.isolation
    if iso is None:
        add(CheckResult("runtime.host_containment", "skip", "critical", {"reason": "world is not isolated (no policy_user / container backend)"}))
    else:
        eg = iso.egress or {}
        add(CheckResult("network_policy", "pass" if (eg.get("isolated", True) or world.spec.network == "public") else "fail", "critical",
                        {"tested": eg.get("tested"), "reachable": eg.get("reachable")}))
        roots = list(iso.writable)
        a = await iso.guard.call("audit", {"uid": iso.uid, "roots": roots})
        planted = a.get("setuid_by_uid") or []
        add(CheckResult("runtime.host_containment", "fail" if planted else "pass", "critical",
                        {"setuid_by_policy": planted, "owned_outside_roots": len(a.get("owned_outside_roots") or []), "specials": len(a.get("specials") or [])}))
        secret_paths = list(world.backend.params.get("root_only", [])) + ["/var/lib/crucible"]
        readable = []
        for p in secret_paths:
            out = await iso.policy_exec.run(f"ls {p} >/dev/null 2>&1 && echo READABLE || echo denied", timeout_s=30)
            if "READABLE" in out.stdout:
                readable.append(p)
        add(CheckResult("oracle_containment", "fail" if readable else "pass", "critical", {"checked": secret_paths, "readable_by_policy": readable}))
    rep.seconds = time.monotonic() - t0
    return rep
