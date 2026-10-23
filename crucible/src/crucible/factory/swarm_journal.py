"""The swarm's journal, its credit assignment and its self-diagnosis (swarm-pipeline/13).

A factory run is many episodes: builder sessions (RLM agents on the M16 harness, with the checkpoint sidecar's Φ),
model calls (planner, reviewer) and deterministic gates. The journal keeps one row per session and one per gate verdict,
each with the provenance of the artifact it is about (``factory.provenance``), so a failure is blamed on the role that
produced the artifact that failed, and the swarm can be debugged like a policy.

**Rows** (JSON lines under the journal directory):
- ``sessions.jsonl``: a session attempt (``sessions.SessionPool``'s record: role, family, model, turns, tokens, USD, stop,
  failure kind, the Φ summary and its directory, where ``progress.jsonl`` holds the per-call ΔΦ).
- ``gates.jsonl``: a state's verdict (ok, findings, the owner it sends back to, retry) with the provenance of what it
  judged, and every state failure (``StateFailed`` = task, ``InfraError`` = infra) with the role that raised it.
- ``runs.jsonl``: a machine run's end (status, terminal, reason, USD).

**Blame.** A failed verdict blames, in order: the role its ``owner`` names (the state it sends the work back to), else
the builder producers of the judged inputs (provenance), else the state's own role. A session's own failure (gave up,
budget, stall) blames its role; infra is never blamed on a role (V8), it is counted per model and per host.

**Credit.** The attempts of one (run, state, entry) are a group: an attempt's reward is 1 when the next verdict on its
artifact passed, 0 when it failed, None (masked) when nothing judged it. Each attempt's per-turn Φ (from its
``progress.jsonl``) and that reward go through ``credit.compose.turn_credit`` (centred within the episode, bounded by the
outcome advantage), so ``builder_credit`` says which builder turns made the progress that the gates later confirmed.

**Diagnosis** (``diagnose``) ranks the swarm's bottlenecks with the evidence (counts, dollars, example findings and
trace directories) and proposes fixes, each marked ``{"proposal": true, "applied": false}``: a proposal never changes a
gate or a machine (V6); a person or a later run decides.
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from crucible.factory.provenance import Provenance, ProvenanceLedger


class SwarmJournal:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, row: Mapping[str, Any]) -> None:
        with (self.root / name).open("a") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), **row}, default=str) + "\n")

    # ---- writers ------------------------------------------------------------------------------------------------------------
    def session(self, row: Mapping[str, Any]) -> None:
        """``SessionPool(on_session=journal.session)``: a session's row (re-written when its capture settles: the last row
        of a directory wins)."""
        self._append("sessions.jsonl", {k: row.get(k) for k in ("run", "state", "entry", "attempt", "role", "family", "model", "stop", "turns", "tokens",
                                                                 "usd", "failure_kind", "phi_final", "capture_commit", "calls", "steps", "masked", "dir",
                                                                 "stop_reason")}
                     | {"progress": _progress_brief(row.get("progress"))})

    def gate(self, *, run: str, state: str, entry: int, attempt: int, role: str | None, ok: bool | None, findings: Iterable[str] = (), owner: str = "",
             retry: bool = False, provenance: Provenance | None = None, kind: str = "verdict", error: str = "") -> None:
        self._append("gates.jsonl", {"run": run, "state": state, "entry": entry, "attempt": attempt, "role": role, "ok": ok, "findings": list(findings)[:20],
                                     "owner": owner, "retry": retry, "kind": kind, "error": error[:500],
                                     "provenance": (provenance or Provenance()).to_json()})

    def run_end(self, *, run: str, status: str, reason: str | None, usd: float, task: str = "", accepted: bool = False) -> None:
        self._append("runs.jsonl", {"run": run, "status": status, "reason": (reason or "")[:500], "usd": round(float(usd), 6), "task": task, "accepted": accepted})

    # ---- the handler guard: provenance check, verdict and failure rows --------------------------------------------------------
    def guard(self, handlers: Mapping[str, Callable[[Any], Awaitable[Any]]], ledger: ProvenanceLedger) -> dict[str, Callable[[Any], Awaitable[Any]]]:
        """``handlers`` checked by ``ledger`` (the producer-family rule) and journaled: each settled verdict and each failure
        becomes a ``gates.jsonl`` row carrying the provenance of what the state read."""
        from crucible.factory.machine import InfraError, StateFailed

        def wrap(fn: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[Any]]:
            async def run(ctx: Any) -> Any:
                prov = ledger.check(ctx.state_id, ctx.role)
                base = {"run": ctx.run_id, "state": ctx.state_id, "entry": ctx.entry, "attempt": ctx.attempt, "role": ctx.role, "provenance": prov}
                try:
                    res = await fn(ctx)
                except InfraError as e:
                    self.gate(**base, ok=None, kind="infra", error=f"{getattr(e, 'kind', '')}: {e}")
                    raise
                except StateFailed as e:
                    self.gate(**base, ok=False, kind="harness" if any(k in str(e) for k in HARNESS_STOPS) else "state_failed", error=str(e))
                    raise
                ledger.record(ctx.state_id, ctx.role)
                outs = getattr(res, "outputs", res) if res is not None else {}
                v = outs.get("verdict") if isinstance(outs, Mapping) else None
                if isinstance(v, Mapping) and "ok" in v:
                    self.gate(**base, ok=bool(v["ok"]), findings=v.get("findings") or [], owner=str(v.get("owner") or ""), retry=bool(v.get("retry")))
                return res

            return run

        return {k: wrap(v) for k, v in handlers.items()}


def _progress_brief(p: Any) -> Any:
    if not isinstance(p, Mapping):
        return None
    return {k: p.get(k) for k in ("phi0", "phi_final", "checkpoints", "graded", "unmeasured", "compaction_turns", "tool_calls", "satisfied", "subgoals",
                                  "series", "honesty", "faults")}


def read_rows(root: Path | str, name: str) -> list[dict[str, Any]]:
    f = Path(root) / name
    if not f.exists():
        return []
    rows = []
    for ln in f.read_text().splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def sessions(root: Path | str) -> list[dict[str, Any]]:
    """The session rows, one per attempt directory (its last row)."""
    by: dict[str, dict[str, Any]] = {}
    for r in read_rows(root, "sessions.jsonl"):
        by[str(r.get("dir") or f"{r.get('run')}/{r.get('state')}/{r.get('entry')}.{r.get('attempt')}")] = r
    return list(by.values())


# ---- blame ------------------------------------------------------------------------------------------------------------------------
INFRA_KINDS = ("infra",)
# session stops that are the harness's configuration, not the builder's skill: blamed on "harness", never on the role
HARNESS_STOPS = ("compaction_failed", "crucible:harness")


def blame_of(gate: Mapping[str, Any], role_of_state: Mapping[str, str] | None = None) -> list[str]:
    """The roles a failed gate row blames: its owner (a state id or a role), else the builder producers of what it judged,
    else the state's own role. Infra and passing rows blame nobody."""
    if gate.get("ok") is not False or gate.get("kind") in INFRA_KINDS:
        return []
    if gate.get("kind") == "harness":
        return ["harness"]
    owner = str(gate.get("owner") or "")
    if owner:
        return [str((role_of_state or {}).get(owner) or owner)]
    if gate.get("kind") == "state_failed" and gate.get("role"):
        return [str(gate["role"])]
    prods = (gate.get("provenance") or {}).get("producers") or []
    if prods:
        return sorted({str(p["role"]) for p in prods})
    return [str(gate.get("role") or gate.get("state"))]


# ---- credit -----------------------------------------------------------------------------------------------------------------------
def _phis(session_dir: str | None) -> tuple[float, list[float]]:
    """Φ0 and Φ after each World step from an attempt's ``progress.jsonl`` (the sidecar's rows; the step=-1 orphan row is
    folded into the last step)."""
    if not session_dir or not (Path(session_dir) / "progress.jsonl").exists():
        return 0.0, []
    rows = [json.loads(x) for x in (Path(session_dir) / "progress.jsonl").read_text().splitlines() if x.strip()]
    phi0 = float(rows[0]["phi"]) if rows else 0.0
    steps = [r for r in rows[1:] if int(r.get("step", 0)) >= 0]
    phis = [float(r["phi"]) for r in steps]
    orphan = [r for r in rows if int(r.get("step", 0)) < 0]
    if orphan and phis:
        phis[-1] = float(orphan[-1]["phi"])
    return phi0, phis


def attempt_rewards(sess_rows: Iterable[Mapping[str, Any]], gate_rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """dir -> reward: 1 / 0 by the first verdict after the attempt on an artifact its state produced (the gate whose
    provenance names the attempt's state, in the same run, later in time), None when nothing judged it."""
    gates = sorted(gate_rows, key=lambda g: g.get("ts", 0))
    out: dict[str, float | None] = {}
    for s in sess_rows:
        r: float | None = None
        for g in gates:
            if g.get("run") != s.get("run") or g.get("ts", 0) < s.get("ts", 0) or g.get("kind") != "verdict":
                continue
            states = {p.get("state") for p in (g.get("provenance") or {}).get("producers") or []}
            if s.get("state") in states:
                r = 1.0 if g.get("ok") else 0.0
                break
        if s.get("stop") not in (None, "agent_done") and not s.get("masked"):
            r = 0.0  # gave up, budget, stall: nothing to judge, the attempt failed
        out[str(s.get("dir"))] = r
    return out


def builder_credit(root: Path | str) -> list[dict[str, Any]]:
    """Per-turn credit of every builder attempt: groups = (run, state): every attempt of one builder step, retries and
    loop-backs alike (a loop-back is the same step with the gate's findings); reward from ``attempt_rewards``, Φ from the
    attempt's ``progress.jsonl``, through ``credit.compose.turn_credit``. One row per attempt with its per-turn ΔΦ and
    credit, and the turns that made the progress (positive ΔΦ)."""
    from crucible.credit.compose import Episode, turn_credit

    srows = sessions(root)
    rewards = attempt_rewards(srows, read_rows(root, "gates.jsonl"))
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = collections.defaultdict(list)
    for s in srows:
        if s.get("progress"):
            groups[(s.get("run"), s.get("state"))].append(s)
    out = []
    for (run, state), rows in groups.items():
        eps, metas = [], []
        for s in rows:
            phi0, phis = _phis(s.get("dir"))
            if not phis:
                continue
            eps.append(Episode(reward=rewards.get(str(s.get("dir"))), phis=phis, phi0=phi0))
            metas.append((s, phi0, phis))
        if not eps:
            continue
        credit = turn_credit(eps)
        for (s, phi0, phis), c in zip(metas, credit, strict=True):
            deltas = [round(b - a, 6) for a, b in zip([phi0, *phis[:-1]], phis, strict=True)]
            out.append({"run": run, "state": state, "entry": s.get("entry"), "attempt": s.get("attempt"), "role": s.get("role"), "model": s.get("model"),
                        "reward": rewards.get(str(s.get("dir"))), "phi0": phi0, "phi_final": phis[-1], "delta_phi": deltas, "credit": [round(x, 6) for x in c],
                        "progress_turns": [i + 1 for i, d in enumerate(deltas) if d > 0], "dir": s.get("dir")})
    return out


# ---- yield, cost, bottlenecks ----------------------------------------------------------------------------------------------------
_NUM = re.compile(r"\b[0-9a-f]{7,40}\b|\d+(\.\d+)?|'[^']*'|\"[^\"]*\"")


def finding_kind(text: str) -> str:
    """A finding with its specifics removed (ids, numbers, quoted names), cut to its head: findings of one kind count
    together."""
    t = _NUM.sub("#", str(text))
    t = re.sub(r"\bm#\b", "m#", t)
    # a requirement-level finding ("m03: requirement leaks the fix by naming ...") is grouped by what it says, not by the
    # milestone id: every finding of a report started with "m#" and counted as one kind
    t = re.sub(r"^\s*(?:m#|R#)\s*:\s*", "", t)
    t = re.sub(r"\([^)]*\)", "", t)  # quoted names and examples are specifics
    return t.split(":")[0].strip()[:80] if ":" in t[:60] else " ".join(t.split()[:8]).strip()[:80]


def tables(root: Path | str) -> dict[str, Any]:
    """Per role and per model: sessions, attempts > 1, stops, infra vs task failures, tokens, USD, mean Φ_final, the gate
    pass rate of the artifacts it produced, and the blamed failures; per run: accepted tasks and dollars."""
    srows, grows, runs = sessions(root), read_rows(root, "gates.jsonl"), read_rows(root, "runs.jsonl")
    role_of_state = {s["state"]: s["role"] for s in srows if s.get("state") and s.get("role")}
    role_of_state |= {g["state"]: g["role"] for g in grows if g.get("state") and g.get("role")}
    per_role: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"sessions": 0, "retries": 0, "usd": 0.0, "tokens": 0, "phi": [], "stops": collections.Counter(),
                                                                          "infra": 0, "task_failures": 0, "judged": 0, "passed": 0, "blamed": 0,
                                                                          "findings": collections.Counter(), "examples": [], "models": set()})
    per_model: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"sessions": 0, "usd": 0.0, "tokens": 0, "infra": 0, "kinds": collections.Counter()})
    for s in srows:
        r = per_role[str(s.get("role"))]
        r["sessions"] += 1
        r["retries"] += int((s.get("attempt") or 1) > 1)
        r["usd"] += float(s.get("usd") or 0)
        r["tokens"] += int(s.get("tokens") or 0)
        r["stops"][str(s.get("stop"))] += 1
        r["models"].add(str(s.get("model")))
        if s.get("phi_final") is not None:
            r["phi"].append(float(s["phi_final"]))
        m = per_model[str(s.get("model"))]
        m["sessions"] += 1
        m["usd"] += float(s.get("usd") or 0)
        m["tokens"] += int(s.get("tokens") or 0)
        if s.get("failure_kind"):
            m["kinds"][str(s["failure_kind"])] += 1
    for g in grows:
        if g.get("kind") == "infra":
            per_role[str(g.get("role") or g.get("state"))]["infra"] += 1
            continue
        for p in {str(x["role"]) for x in (g.get("provenance") or {}).get("producers") or []}:
            per_role[p]["judged"] += 1
            per_role[p]["passed"] += int(bool(g.get("ok")))
        for who in blame_of(g, role_of_state):
            b = per_role[who]
            b["blamed"] += 1
            b["task_failures"] += int(g.get("kind") == "state_failed")
            for f in g.get("findings") or [g.get("error") or ""]:
                if f:
                    b["findings"][finding_kind(f)] += 1
            if len(b["examples"]) < 5:
                b["examples"].append({"run": g.get("run"), "state": g.get("state"), "entry": g.get("entry"), "finding": (list(g.get("findings") or []) or [g.get("error")])[0]})
    roles = {}
    for name, r in per_role.items():
        roles[name] = {"sessions": r["sessions"], "retries": r["retries"], "usd": round(r["usd"], 6), "tokens": r["tokens"], "stops": dict(r["stops"]),
                       "mean_phi_final": round(statistics.fmean(r["phi"]), 4) if r["phi"] else None, "infra": r["infra"], "task_failures": r["task_failures"],
                       "artifacts_judged": r["judged"], "gate_pass_rate": round(r["passed"] / r["judged"], 4) if r["judged"] else None, "blamed": r["blamed"],
                       "top_findings": dict(r["findings"].most_common(5)), "examples": r["examples"], "models": sorted(r["models"] - {"None"})}
    usd = sum(float(x.get("usd") or 0) for x in runs) or sum(v["usd"] for v in roles.values())
    accepted = sum(1 for x in runs if x.get("accepted"))
    return {"roles": roles, "models": {k: {**v, "usd": round(v["usd"], 6), "kinds": dict(v["kinds"])} for k, v in per_model.items()},
            "runs": len(runs), "accepted": accepted, "usd": round(usd, 6), "accepted_per_usd": round(accepted / usd, 4) if usd else None}


# ---- diagnosis and proposals ------------------------------------------------------------------------------------------------------
# (finding kind or stop pattern, the proposal's kind, its text). Proposals only: nothing here edits a gate or a machine (V6).
RULES: tuple[tuple[str, str, str], ...] = (
    (r"lint (hidden|leak|version)", "tool", "give the prompt writer the lane's lint as a tool it can run on its draft (the gate it fails is static and cheap)"),
    (r"lint (hidden|leak|version)", "prompt", "put the forbidden strings (faulty names, changed files, hidden case names) into the writer's brief as a list"),
    (r"unmapped_milestone|unknown_requirement_id", "prompt", "state the requirement ids the draft must use, one line per milestone, in the brief"),
    (r"no file|does not parse|defines no test", "prompt", "the verifier writer's brief should name each file path it must write and ask it to run pytest --collect-only"),
    (r"commits for|subject must start", "prompt", "the oracle builder's brief should show the exact commit subject format with an example"),
    (r"reads source or code structure", "prompt", "forbid inspect/ast/getsource in the verifier brief with the reason (behaviour, not structure)"),
    (r"^gave up|agent_gave_up", "harness", "read the give-up reasons: a missing tool or a wrong World is a harness fix, not a model fix"),
    (r"budget|state_budget|max_turns|tool limit", "harness", "the role runs out of turns or dollars before its gate: raise its state budget or compact earlier, after checking Φ was still rising"),
    (r"stall", "harness", "the stall detector stopped sessions: inspect the repeated cells in the ledger (a tool that fails silently?)"),
    (r"compaction_failed", "harness", ("the session's window cannot hold its own hand-off after a compaction: raise window_tokens (sessions.window_floor) "
     "or shorten the hand-off")),
    (r"missing declared output|capture", "prompt", ("sessions end without the file their gate reads: the brief should end with the exact path to write, "
     "and a finish without it should be refused by the harness")),
)


def diagnose(root: Path | str, *, min_artifacts: int = 3, low_pass: float = 0.3) -> dict[str, Any]:
    """The bottlenecks of the swarm, ranked by the dollars spent on work a gate later refused (blamed failures × the
    role's mean dollars per session), each with its evidence and its proposals. Also: models whose sessions end in infra
    more than in task failures (a provider or harness problem, not a model problem), and roles below ``low_pass`` gate
    pass rate over ``min_artifacts`` artifacts (a role-assignment proposal, checked against the family rule by whoever
    applies it)."""
    t = tables(root)
    credit = builder_credit(root)
    out: list[dict[str, Any]] = []
    for role, r in t["roles"].items():
        per = (r["usd"] / r["sessions"]) if r["sessions"] else 0.0
        cost = r["blamed"] * per
        evidence = {"blamed": r["blamed"], "gate_pass_rate": r["gate_pass_rate"], "sessions": r["sessions"], "retries": r["retries"], "usd": r["usd"],
                    "stops": r["stops"], "top_findings": r["top_findings"], "examples": r["examples"], "mean_phi_final": r["mean_phi_final"]}
        props = []
        text = " | ".join([*r["top_findings"], *(k for k in r["stops"] if k not in ("agent_done", "None"))])
        for rx, kind, msg in RULES:
            if re.search(rx, text, re.IGNORECASE):
                props.append({"kind": kind, "text": msg, "proposal": True, "applied": False})
        if r["gate_pass_rate"] is not None and r["artifacts_judged"] >= min_artifacts and r["gate_pass_rate"] < low_pass:
            props.append({"kind": "role", "text": f"{role} passes {r['gate_pass_rate']:.0%} of its gates over {r['artifacts_judged']} artifacts: try another "
                          f"family on it (models {r['models']}), within the provenance family rule", "proposal": True, "applied": False})
        if r["mean_phi_final"] is not None and r["mean_phi_final"] >= 0.9 and r["gate_pass_rate"] is not None and r["gate_pass_rate"] < 0.5:
            props.append({"kind": "harness", "text": f"{role}'s Φ (the static half of its gate) ends near 1 but its gates fail: the executed half fails, so give "
                          "the session a way to run that check (a tool or a World command)", "proposal": True, "applied": False})
        if r["blamed"] or props:
            out.append({"role": role, "score_usd": round(cost, 6), "evidence": evidence, "proposals": props})
    out.sort(key=lambda x: (-x["score_usd"], -x["evidence"]["blamed"]))
    infra = [{"model": m, "sessions": v["sessions"], "failure_kinds": v["kinds"]} for m, v in t["models"].items() if v["kinds"]]
    stuck = [c for c in credit if c["reward"] == 0.0 and c["phi_final"] < 1.0]
    return {"tables": t, "bottlenecks": out, "models_with_failures": infra,
            "builder_credit": {"attempts": len(credit), "with_progress": sum(1 for c in credit if c["progress_turns"]),
                               "failed_below_phi_1": [{"role": c["role"], "state": c["state"], "phi_final": c["phi_final"], "dir": c["dir"]} for c in stuck][:20]}}


def render(d: Mapping[str, Any]) -> str:
    """The diagnosis as markdown (``scripts/swarm.py diagnose``)."""
    t = d["tables"]
    lines = ["# Swarm diagnosis", "", f"runs {t['runs']}, accepted {t['accepted']}, USD {t['usd']}, accepted per USD {t['accepted_per_usd']}", "",
             "| role | sessions | retries | USD | mean Φ end | gate pass | blamed | infra | stops |", "|---|---|---|---|---|---|---|---|---|"]
    for role, r in sorted(t["roles"].items()):
        lines.append(f"| {role} | {r['sessions']} | {r['retries']} | {r['usd']:.4f} | {r['mean_phi_final']} | {r['gate_pass_rate']} | {r['blamed']} | {r['infra']} | "
                     f"{', '.join(f'{k} {v}' for k, v in r['stops'].items())} |")
    lines += ["", "## Bottlenecks (ranked by dollars spent on refused work)", ""]
    for i, b in enumerate(d["bottlenecks"], 1):
        ev = b["evidence"]
        lines.append(f"{i}. **{b['role']}**: {ev['blamed']} blamed failures, ${b['score_usd']:.4f}; top findings {ev['top_findings']}")
        for p in b["proposals"]:
            lines.append(f"   - proposal ({p['kind']}, not applied): {p['text']}")
        for e in ev["examples"][:2]:
            lines.append(f"   - e.g. run {e['run']} state {e['state']} #{e['entry']}: {str(e['finding'])[:160]}")
    if d["models_with_failures"]:
        lines += ["", "## Session failure kinds per model", ""] + [f"- {m['model']}: {m['failure_kinds']}" for m in d["models_with_failures"]]
    bc = d["builder_credit"]
    lines += ["", f"## Builder credit: {bc['attempts']} attempts with Φ, {bc['with_progress']} with progress turns", ""]
    lines += [f"- {c['role']} at {c['state']}: failed at Φ {c['phi_final']} ({c['dir']})" for c in bc["failed_below_phi_1"][:10]]
    return "\n".join(lines) + "\n"
