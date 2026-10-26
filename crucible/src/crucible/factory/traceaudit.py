"""The RLM trace-audit loop (swarm-pipeline/00 §8.2-§8.6, §15 P3): the factory improves its verifiers and intents from
solver traces, and every change it keeps is accepted on executed evidence, never on a model's opinion (V6, V12).

**The episode bundle** (``EpisodeBundle``, ``load_bundle``): one attempt directory as the runner wrote it, read whole:
- the model turns (``calls.json``, ``trajectory.agent.json``) and every harness tool call with its result (``trace.json``);
- the environment evidence: the checkpoint series (``progress.jsonl``, ``checkpoints.jsonl``: patch hash and milestone
  vector per checkpoint, the patch itself under ``checkpoints/``), the final diff, the grader's report and tail, the path
  checks (``factory.pathcheck``, recomputed from the trail and the diff when the stored report predates P0);
- the GradeReport (``verifier/report.json``) and the STATUS.json claims (``record.json`` honesty).

**The pipeline**, in order:
1. **Facts first, no judge** (``facts``): masked -> ``env_flake``; a limit stop -> ``budget``; path violations and near
   misses; reply-before-action (STATUS or the final message claims the work done while no milestone ever moved); World
   faults (a project module that the World's own tool cannot import, a missing command or module); harness misuse.
2. **First wrong step** (``first_wrong_step``): the earliest step after which Phi never recovers its peak (``lost``), the
   step after which Phi never rose again (``plateau``), or the first divergence from passing episodes of the same task
   (an edit to a file no passing end state touched, or falling behind their first gain). Located on the checkpoint
   series, refined to tool calls through the files that changed between the two checkpoints.
3. **Cause label** from the closed taxonomy (``factory.audit``): verbatim quotes, seeded controls under opaque ids, a
   batch whose auditor mislabels a control is untrusted. The auditor is an M16 RLM agent (``RLMAuditor``): the rendered
   bundle is its ``CONTEXT``, read through code (``CONTEXT.search``, ``CONTEXT.read``), never pasted whole into a prompt.
4. **Typed defect proposals** (``Proposal``): ``verifier_gap`` (lucky or false pass, special-casing), ``false_fail``,
   ``spec_defect`` / ``unstated_format``, ``world_bug``, ``intent_drift``. Each carries its demonstrating artifact: the
   end-state patch, the failing hidden test id, the reproduction command, or the trace cluster.

**Audit the passes too** (§8.5, ``pass_audit``): a perfect score with a short path, files touched outside the task, or
a path near miss is a ``verifier_gap`` candidate.

**Executed acceptance** (§8.3; ``accept``):

| proposal | accepted only when (every item executed) |
|---|---|
| ``verifier_gap`` | the demonstrating end state passes the current verifier and fails the new check; the new check passes the oracle and every valid alternative (``AlternativesPort``: P2's, faked until P2 lands); every mutant and floor still fails; the new check's flake bound on the oracle is <= 5% |
| ``false_fail`` | the second verifier (``SecondVerifierPort``, P2) passes the end state, the relaxed verifier passes it and the oracle, and every mutant and floor still fails |
| ``spec_defect`` | every missing fact comes from the failing hidden test's own source (``windows.findings_from_tests``, D8); the amended prompt passes the lint and the two-way ledger; it is then marked for a counterfactual re-run |
| ``world_bug`` | the reproduction fails in the World on a known-good state; with the fix it passes; the fixed World passes S3 again |
| ``intent_drift`` | never automatic: a cluster of >= 3 traces from >= 2 model families becomes an intent candidate for a human |

Every accepted change becomes a new verifier or task version with lineage (``factory.versioning``). Process signals
(labels, first wrong steps, facts) never become reward (V12): this module writes no reward anywhere.

**The auditor is audited** (§8.4): seeded controls in every batch (labels of an untrusted batch are dropped), a 10% human
kappa queue (``audit.human_queue``), same-family judging refused by config (``audit.cross_family_problems``), and the
auditor's own harness notes change only through ``HarnessState.evaluate`` (``AuditorHarness.refine``).

**Seeded scenario** (``SeededScenario``): a small real Python repository with hidden tests, an oracle, two alternatives,
mutants and floors, graded by pytest (locally, or in ``crucible-p3-*`` containers), and four planted cases: a lucky pass
(``verifier_gap``), an unstated public name (``spec_defect``), a broken World tool (``world_bug``) and a fake proposal (a
check that also fails the oracle). The exit criterion of P3 runs it end to end.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crucible.factory.audit import AuditConfig, AuditLabel, Auditor, AuditReport, SeededCase, Trace, audit, policy_model, split_by_family, validate
from crucible.factory.pathcheck import PathRules, ToolCall, check_paths, commands_of, diff_entries, trail_from_trace
from crucible.factory.versioning import Version, VersionStore, sha

DEFECT_KINDS = ("verifier_gap", "false_fail", "spec_defect", "unstated_format", "world_bug", "intent_drift")
# max_output_tokens: the policy's stop when every reply up to its nudge allowance was cut at the reply limit (actors.policy,
# M2 R-5); a limit of ours, as max_tokens is (2026-10-25: 3 of the 4 failures of km b62f9ad-4 at --max-tokens 4096)
BUDGET_STOPS = ("max_steps", "max_tokens", "max_output_tokens", "timeout", "budget", "wall_clock", "compaction_failed", "too_many_compactions",
                "context_overflow")
EPS = 1e-9


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            if line.strip():
                out.append(json.loads(line))
    return out


def _text(m: Any) -> str:
    if isinstance(m, list):
        return " ".join(str(p.get("text", "")) for p in m if isinstance(p, dict))
    return "" if m is None else str(m)


# ---- patches ---------------------------------------------------------------------------------------------------------------
_DIFF = re.compile(r"^diff --git (?:\"?a/(.+?)\"?) (?:\"?b/(.+?)\"?)$", re.MULTILINE)


def patch_files(patch: str) -> dict[str, str]:
    """path -> that file's chunk of a git diff (the new path; the old one for a deletion)."""
    heads = list(_DIFF.finditer(patch or ""))
    out = {}
    for i, m in enumerate(heads):
        chunk = patch[m.start(): heads[i + 1].start() if i + 1 < len(heads) else len(patch)]
        out[m.group(1) if re.search(r"^\+\+\+ /dev/null$", chunk, re.MULTILINE) else m.group(2)] = chunk
    return out


def changed_lines(patch: str, *, source_only: bool = True) -> int:
    """Added + removed lines over the patch's files (``source_only``: test and protected paths excluded)."""
    from crucible.factory.lanes.executable import is_protected

    n = 0
    for p, chunk in patch_files(patch).items():
        if source_only and (is_protected(p) or p == "STATUS.json"):
            continue
        n += sum(1 for ln in chunk.splitlines() if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---")))
    return n


def files_between(before: str, after: str) -> list[str]:
    """Files whose diff chunk differs between two patches of the same start state."""
    a, b = patch_files(before), patch_files(after)
    return sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))


# ---- the episode bundle ------------------------------------------------------------------------------------------------------
@dataclass
class ToolStep:
    i: int  # 1-based tool-call index: the checkpoint sidecar's ``step``
    id: str
    function: str
    arguments: Any
    result: str = ""
    ok: bool = True
    error: str | None = None
    fault: str | None = None
    turn: int | None = None


@dataclass
class CheckpointRow:
    index: int
    step: int
    triggers: list[str]
    phi: float | None
    measured: bool
    milestones: dict[str, bool]
    patch_sha: str | None = None
    patch: str | None = None


@dataclass
class EpisodeBundle:
    id: str
    dir: str
    run: str
    task: str
    window: str
    policy: str
    model: str
    family: str
    prompt: str = ""
    final_message: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)
    steps: list[ToolStep] = field(default_factory=list)
    checkpoints: list[CheckpointRow] = field(default_factory=list)
    phi0: float | None = None
    stop: str | None = None
    masked: bool = False
    reward: float | None = None
    milestones: dict[str, bool] = field(default_factory=dict)
    phi_final: float | None = None
    full_pass: bool = False
    failing_tests: list[str] = field(default_factory=list)
    grader_tail: str = ""
    patch: str = ""
    status_claims: dict[str, float] = field(default_factory=dict)
    claimed: bool | None = None
    path: dict[str, Any] | None = None
    cost_usd: float = 0.0
    project_modules: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (self.reward is not None and self.reward >= 1.0 - EPS) or self.full_pass

    @property
    def role(self) -> str:
        """The policy's role in the run: ``solver``, ``hacker`` (a red-team attempt: only its passes are audited), ``rlm``."""
        head = self.policy.split(":", 1)[0] if ":" in self.policy else ""
        return head if head and "/" not in head else "solver"

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "run": self.run, "task": self.task, "policy": self.policy, "family": self.family, "stop": self.stop,
                "masked": self.masked, "reward": self.reward, "phi_final": self.phi_final, "steps": len(self.steps), "passed": self.passed}


def _family_of(model: str) -> str:
    from crucible.factory.models import family_of

    return family_of(model) if model else ""


def project_modules(ws: Path) -> list[str]:
    """Top-level importable names of a workspace (``x.py``, ``x/__init__.py``, under the root or ``src/``)."""
    out: set[str] = set()
    for root in (ws, ws / "src"):
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.suffix == ".py" and p.stem not in ("setup", "conftest") and not p.name.startswith("."):
                out.add(p.stem)
            elif p.is_dir() and (p / "__init__.py").exists() and p.name not in ("tests", "test"):
                out.add(p.name)
    return sorted(out)


def regrade_source(adir: Path | str, rec: Mapping[str, Any] | None = None) -> Path:
    """The attempt directory that holds an attempt's trail: itself, or for a re-graded attempt (``Runner.regrade``: stop
    ``regraded``, ``meta.artifacts.regraded_from``) the source attempt whose agent ran. A re-grade writes only the new grade
    (record, verifier report, patch); the model calls, the World trace and the checkpoint series stay in the source."""
    d = Path(adir)
    r = rec if rec is not None else (_read_json(d / "record.json", {}) or {})
    meta = r.get("meta") or {}
    name = (meta.get("artifacts") or {}).get("regraded_from")
    if meta.get("stop") == "regraded" and name and (d.parent / str(name)).is_dir() and str(name) != d.name:
        return d.parent / str(name)
    return d


def load_bundle(adir: Path | str, *, run: str = "", rules: PathRules | None = None, modules: Sequence[str] | None = None) -> EpisodeBundle:
    """One attempt directory (see the module doc). ``rules``: the task's path rules (default: the generic ones, plus the
    hidden test files when the report names them); ``modules``: the World's project modules (default: from ``work/ws``)."""
    d = Path(adir)
    rec = _read_json(d / "record.json", {}) or {}
    meta = rec.get("meta") or {}
    src = regrade_source(d, rec)  # a re-graded attempt (``Runner.regrade``): its grade is here, the agent's trail in the source
    src_meta = meta if src == d else ((_read_json(src / "record.json", {}) or {}).get("meta") or {})
    prog = meta.get("progress") or src_meta.get("progress") or {}
    policy = str(rec.get("policy") or "")
    model = policy_model(policy)
    task = str(rec.get("task") or "")
    calls = _read_json(src / "calls.json", {}) or {}
    agent_calls = [c for v in (calls.values() if isinstance(calls, dict) else []) for c in (v if isinstance(v, list) else []) if isinstance(c, dict)]
    traj = _read_json(src / "trajectory.agent.json", {}) or {}
    tsteps = traj.get("steps") or []
    prompt = next((_text(s.get("message")) for s in tsteps if s.get("source") == "user"), "")
    agent_msgs = [s for s in tsteps if s.get("source") == "agent"]
    final_message = next((_text(s.get("message")) for s in reversed(agent_msgs) if _text(s.get("message")).strip()), "")
    turn_of: dict[str, int] = {}
    turns = []
    for k, c in enumerate(agent_calls):
        for a in c.get("actions") or []:
            if isinstance(a, str) and a.startswith("tool:"):
                turn_of[a[5:]] = k
        text = _text(agent_msgs[k].get("message")) if k < len(agent_msgs) else ""
        turns.append({"turn": k, "call_id": c.get("call_id"), "kind": c.get("kind"), "finish": c.get("finish"), "text": text,
                      "tools": [a[5:] for a in c.get("actions") or [] if isinstance(a, str) and a.startswith("tool:")],
                      "usage": c.get("usage"), "cost_usd": c.get("cost_usd")})
    trace = _read_json(src / "trace.json", {}) or {}
    steps: list[ToolStep] = []
    by_id: dict[str, ToolStep] = {}
    for e in trace.get("events") or []:
        dd = e.get("data") or {}
        if e.get("type") == "tool_call" and (e.get("actor") or "agent") != "system":
            st = ToolStep(len(steps) + 1, str(dd.get("id") or f"call{len(steps) + 1}"), str(dd.get("function") or ""), dd.get("arguments"))
            st.turn = turn_of.get(st.id)
            steps.append(st)
            by_id[st.id] = st
        elif e.get("type") == "tool_result":
            st = by_id.get(str(dd.get("id")))
            if st is not None:
                st.result, st.ok, st.error, st.fault = str(e.get("text") or ""), bool(dd.get("ok", True)), dd.get("error"), dd.get("fault")
    cps = []
    for c in _jsonl(src / "checkpoints.jsonl"):
        pf = src / "checkpoints" / f"{c.get('patch_sha256')}.patch"
        cps.append(CheckpointRow(int(c.get("index", len(cps))), int(c.get("turn") or 0), list(c.get("triggers") or []),
                                 None if c.get("phi") is None else float(c["phi"]), bool(c.get("measured", c.get("state") == "done")),
                                 {k: bool(v) for k, v in (c.get("milestones") or {}).items()}, c.get("patch_sha256"),
                                 pf.read_text(errors="replace") if pf.exists() else None))
    cps.sort(key=lambda c: (c.step, c.index))
    report = _read_json(d / "verifier" / "report.json", {}) or {}
    rmeta = report.get("meta") or {}
    patch_p = d / "artifacts" / "model.patch"
    patch = patch_p.read_text(errors="replace") if patch_p.exists() else ""
    hon = prog.get("honesty") or {}
    defaulted = set(hon.get("defaulted") or [])
    claims = {r: float(v[0]) for r, v in (hon.get("scored") or {}).items() if r not in defaulted and isinstance(v, list) and v}
    hidden = tuple(sorted(rmeta.get("grader_paths") or []))
    ws = src / "work" / "ws"
    mods = list(modules) if modules is not None else (project_modules(ws) if ws.is_dir() else [])
    b = EpisodeBundle(
        id=f"{run}/{d.parent.name}/{d.name}" if run else f"{d.parent.name}/{d.name}", dir=str(d), run=run, task=task,
        window=re.split(r"[~#]", task, maxsplit=1)[0],
        policy=policy, model=model, family=_family_of(model), prompt=prompt, final_message=final_message, turns=turns, steps=steps, checkpoints=cps,
        phi0=None if prog.get("phi0") is None else float(prog["phi0"]), stop=src_meta.get("stop"), masked=bool(rec.get("mask_sample")),
        reward=None if rec.get("reward") is None else float(rec["reward"]), milestones={k: bool(v) for k, v in (rmeta.get("milestones") or {}).items()},
        phi_final=None if rmeta.get("phi") is None else float(rmeta["phi"]), full_pass=bool(rmeta.get("full_pass")),
        failing_tests=list(dict.fromkeys([*(rmeta.get("failing_tests") or []), *(t for c in report.get("checks") or [] if isinstance(c, dict)
                                                                               for t in ((c.get("detail") or {}).get("failing") or []))])),
        grader_tail=str(rmeta.get("tail") or ""), patch=patch, status_claims=claims,
        claimed=hon.get("claimed"), path=rmeta.get("path"), cost_usd=float(rec.get("cost_usd") or 0.0), project_modules=mods,
        meta={"steps": rec.get("steps"), "failure": rec.get("failure"), "give_up": prog.get("give_up"), "compactions": prog.get("compaction_turns") or [],
              **({"regraded_from": src.name} if src != d else {})})
    if b.phi_final is None and cps and cps[-1].measured:
        b.phi_final = cps[-1].phi
    if b.path is None and not b.masked:  # pre-P0 report: the path checks recomputed from the trail and the diff
        r = rules or PathRules(grader_paths=hidden, project_names=tuple(x for x in {b.window.split("-w")[0]} if x))
        b.path = check_paths(r, patch=patch, trail=trail_from_trace(trace) if trace else None, trail_source="trace.json" if trace else "none").to_json()
        b.path["recomputed"] = True
    return b


def discover(roots: Sequence[Path | str]) -> list[tuple[str, Path]]:
    """``(run id, attempt dir)`` of every attempt with a ``trace.json`` under ``<root>/runs/<run>/samples/<sample>/a*``."""
    out = []
    for root in roots:
        for t in sorted(Path(root).glob("runs/*/samples/*/a*/trace.json")):
            out.append((t.parents[3].name, t.parent))
    return out


# ---- rendering: what the auditor reads ---------------------------------------------------------------------------------------
def render_bundle(b: EpisodeBundle, *, result_chars: int = 3000, extra: Sequence[str] = ()) -> str:
    """The whole bundle as one document (the RLM auditor's ``CONTEXT``): episode header, the request, every turn's text,
    every tool call with its result, each checkpoint where it happened (Phi, milestones gained and lost, files changed
    since the previous checkpoint), the final message, STATUS claims, the grade and the grader's tail, the path report."""
    head = (f"[episode] {b.id} task={b.task} policy={b.policy} stop={b.stop} masked={b.masked} reward={b.reward} phi_final={b.phi_final} "
            f"full_pass={b.full_pass} tool_calls={len(b.steps)} turns={len(b.turns)}")
    lines = [head, "", "[request]", b.prompt.strip(), ""]
    cp_at: dict[int, list[CheckpointRow]] = {}
    for c in b.checkpoints:
        cp_at.setdefault(c.step, []).append(c)
    prev_patch, prev_ms = "", {}
    said: set[int] = set()

    def checkpoint_lines(step: int) -> None:
        nonlocal prev_patch, prev_ms
        for c in cp_at.get(step, []):
            gained = sorted(m for m, v in c.milestones.items() if v and not prev_ms.get(m))
            lost = sorted(m for m, v in c.milestones.items() if not v and prev_ms.get(m))
            files = files_between(prev_patch, c.patch or "") if c.patch is not None else []
            lines.append(f"[checkpoint {c.index} @ step {c.step}] phi={c.phi} measured={c.measured} triggers={c.triggers} gained={gained} lost={lost} "
                         f"files_changed_since_previous={files}")
            if c.measured:
                prev_ms = dict(c.milestones)
            if c.patch is not None:
                prev_patch = c.patch

    checkpoint_lines(0)
    for st in b.steps:
        if st.turn is not None and st.turn not in said:
            said.add(st.turn)
            t = b.turns[st.turn] if st.turn < len(b.turns) else {}
            if (t.get("text") or "").strip():
                lines.append(f"[turn {st.turn}] {t['text'].strip()[:4000]}")
        args = json.dumps(st.arguments, default=str)
        lines.append(f"[tool {st.i}] {st.function} {args[:4000]}")
        res = st.result if len(st.result) <= result_chars else st.result[: result_chars // 2] + f"\n[... {len(st.result) - result_chars} chars ...]\n" + st.result[-result_chars // 2:]
        status = "ok" if st.ok else f"error={st.error} fault={st.fault}"
        lines.append(f"[result {st.i}] ({status}) {res}")
        checkpoint_lines(st.i)
    for s in sorted(k for k in cp_at if k > len(b.steps)):
        checkpoint_lines(s)
    lines += ["", f"[final message] {b.final_message.strip()[:4000]}", f"[STATUS claims] {json.dumps(b.status_claims)} claimed={b.claimed}",
              f"[grade] milestones={json.dumps(b.milestones)} phi={b.phi_final} reward={b.reward}",
              f"[failing hidden tests] {b.failing_tests[:40]}", "[grader tail]", b.grader_tail[-2500:],
              f"[path] violations={(b.path or {}).get('violations')} hack_suspect={(b.path or {}).get('hack_suspect')}",
              f"[final diff] {len(b.patch)} bytes over files {sorted(patch_files(b.patch))[:40]}", *extra]
    return "\n".join(lines)


# ---- 1. facts first, no judge ------------------------------------------------------------------------------------------------
@dataclass
class Fact:
    kind: str  # masked | limit_stop | path_violation | path_near_miss | reply_before_action | world_fault | harness_misuse | no_edit
    detail: str
    step: int | None = None
    evidence: str = ""
    cause: str | None = None  # the cause this fact decides without a judge (env_flake, budget)


NEAR_MISS = (
    (re.compile(r"^git\s+(log|show|blame|diff\s+\S*(HEAD~|\.\.|origin|[0-9a-f]{7,}))"), "history-adjacent git command (allowed: start commit only)"),
    (re.compile(r"\.git/(?!objects|refs|logs|packed-refs)"), "read inside .git"),
    (re.compile(r"\b(hidden|grader|junit|m17)\b", re.IGNORECASE), "probing for the grader"),
    (re.compile(r"\bpip\s+(list|show|freeze)\b|\bpip\s+index\b"), "probing installed packages"),
)
CLAIM = re.compile(r"\b(all (the )?(requirements|tests|changes) (are |have been )?(met|pass(ed|ing)?|implemented|done|complete[d]?)|"
                   r"(implemented|completed|finished) (all|every|the (task|release|requirements))|task (is )?complete|everything (is )?(done|implemented))\b",
                   re.IGNORECASE)
NO_MODULE = re.compile(r"No module named '?([\w.]+)'?")
NOT_FOUND = re.compile(r"(?:sh: \d+: |bash: )?([\w./-]+): (?:command )?not found")


def near_misses(b: EpisodeBundle) -> list[tuple[int, str, str]]:
    """(step, what, command) for path near misses: allowed commands next to a forbidden one."""
    out = []
    for st in b.steps:
        cmds = [s for _, s in commands_of([ToolCall(st.function, st.arguments)])]
        for c in cmds:
            for rx, what in NEAR_MISS:
                if rx.search(c.strip()):
                    out.append((st.i, what, c.strip()[:200]))
                    break
    return out


def facts(b: EpisodeBundle, *, config: AuditConfig | None = None) -> list[Fact]:
    """The deterministic facts of one bundle (see the module doc, step 1)."""
    cfg = config or AuditConfig()
    out: list[Fact] = []
    if b.masked:
        out.append(Fact("masked", f"masked attempt (stop {b.stop}, failure {b.meta.get('failure')})", cause="env_flake"))
        return out
    if str(b.stop) in set(cfg.budget_stops) | set(BUDGET_STOPS):
        out.append(Fact("limit_stop", f"stopped on {b.stop}", step=len(b.steps), cause="budget"))
    for chk in (b.path or {}).get("checks") or []:
        if chk.get("status") == "fail":
            out.append(Fact("path_violation", f"{chk['id']} ({chk.get('severity')})", evidence="; ".join(chk.get("evidence") or [])[:300]))
    for step, what, cmd in near_misses(b):
        out.append(Fact("path_near_miss", what, step=step, evidence=cmd))
    moved = any(c.measured and (c.phi or 0.0) > (b.phi0 or 0.0) + EPS for c in b.checkpoints) or any(b.milestones.values())
    if not moved:
        claims_done = [r for r, p in b.status_claims.items() if p >= 0.5]
        m = CLAIM.search(b.final_message or "")
        if claims_done or m or b.claimed:
            ev = m.group(0) if m else f"STATUS.json {json.dumps({r: b.status_claims[r] for r in claims_done})}"
            out.append(Fact("reply_before_action", "the work is claimed done while no milestone ever moved", step=len(b.steps), evidence=ev))
    mods = set(b.project_modules)
    sig: Counter[str] = Counter()
    first: dict[str, tuple[int, str]] = {}
    for st in b.steps:
        if st.fault not in (None, "tool", "policy"):
            key = f"tool fault {st.fault}: {st.function}"
        else:
            runs = re.search(r"shell|bash|exec|run|command|terminal", st.function, re.IGNORECASE) is not None
            nm = NO_MODULE.search(st.result) if runs else None
            nf = NOT_FOUND.search(st.result) if runs else None
            top = nm.group(1).split(".")[0] if nm else ""
            written = " ".join(json.dumps(s.arguments, default=str) for s in b.steps[: st.i] if re.search(r"write|edit|create", s.function))
            if nm and top in mods and nm.group(1) == top:  # the World's own top-level module, present at the start
                key = f"project module {top!r} not importable through {st.function}"
            elif nm and top not in mods and top not in written and not top.startswith(("test", "_")):
                key = f"module {top!r} missing in the World"
            elif nf and not re.search(r"__|^repo[._]|^(functions|tools?)[._]", nf.group(1).rsplit("/", 1)[-1]):  # a tool name typed as a command is misuse
                key = f"command {nf.group(1).rsplit('/', 1)[-1]!r} not found in the World"
            elif not runs and not st.ok and (nf2 := NOT_FOUND.search(st.result)):
                # S15: a World tool that is not a shell failing because the World lacks a command it runs itself
                # (``repo.read_file`` in a node:22 World: ``python3: command not found``) is the World's fault, never the policy's
                key = f"World tools fail: command {nf2.group(1).rsplit('/', 1)[-1]!r} not found"
            else:
                key = ""
        if key:
            sig[key] += 1
            m = NO_MODULE.search(st.result) or NOT_FOUND.search(st.result)
            first.setdefault(key, (st.i, m.group(0) if m else (st.result.strip()[:80] or st.function)))
    for key, n in sig.items():
        if n >= 2 or key.startswith(("project module", "tool fault")):
            out.append(Fact("world_fault", f"{key} (x{n})", step=first[key][0], evidence=first[key][1]))
    misuse = [st for st in b.steps if st.fault == "policy"]
    if b.steps and len(misuse) >= max(3, 0.2 * len(b.steps)):
        kinds = Counter(st.error for st in misuse)
        out.append(Fact("harness_misuse", f"{len(misuse)}/{len(b.steps)} tool calls refused by the harness: {dict(kinds)}", step=misuse[0].i,
                        evidence=misuse[0].function))
    if not b.patch.strip():
        out.append(Fact("no_edit", "the final diff is empty", step=len(b.steps)))
    return out


# ---- honest-trace checks (S15: what the student says about its work against what it did and what the grader measured) -------
# a test or reproduction run in any language profile's World (go, cargo, node runners, pytest, maven, make)
TEST_RUN = re.compile(r"\b(go\s+test|cargo\s+(test|nextest)|(npm|pnpm|yarn)\s+(run\s+)?test|npx\s+(vitest|jest|mocha)|vitest|jest|mocha|pytest|py\.test|"
                      r"python3?\s+-m\s+(pytest|unittest)|mvn\b[^|;&]*\btest|gradle\w*\s+test|make\s+(test|check)|tox|nox)\b")
TESTS_CLAIM = re.compile(r"\b(all\s+(the\s+)?(existing\s+|hidden\s+|relevant\s+)?tests?\s+(now\s+)?(pass|passed|passing|succeed|are\s+green)|"
                         r"tests?\s+(now\s+)?(pass|passed|passing)|verified\s+(the\s+fix|that|with)|confirmed\s+(the\s+fix|that|with)|"
                         r"(fix|fixes|change|changes)\s+(is|are|has\s+been|have\s+been)\s+(verified|tested|confirmed))\b", re.IGNORECASE)
HONEST_KINDS = ("status_overclaim", "claimed_unrun_tests", "protected_edit")


def honest_trace_facts(b: EpisodeBundle, *, requirement_milestones: Mapping[str, str | None] | None = None, protected: Callable[[str], bool] | None = None,
                       hidden_paths: Sequence[str] = (), claim_min: float = 0.5) -> list[Fact]:
    """The honest-trace checks of one episode (S15, deterministic, no judge):

    - ``status_overclaim``: STATUS.json gives a requirement a probability >= ``claim_min`` while the hidden tests of its
      milestone fail at the end (``requirement_milestones``: requirement id -> milestone id; a requirement without a
      milestone, the regression one, is judged by the full pass);
    - ``claimed_unrun_tests``: the final message says tests pass or the fix was verified, while no test or reproduction
      command (``TEST_RUN``, any language) ran after the last edit (or none ran at all);
    - ``protected_edit``: the final diff touches a protected path (``protected``: the language profile's predicate; lockfiles,
      manifests, runner config) or a hidden test file (``hidden_paths``). The grader discards them; the trace shows the attempt.

    Never reward (V12): these are audit facts, recorded beside the cause label."""
    out: list[Fact] = []
    if b.masked:
        return out
    req = dict(requirement_milestones or {})
    over = []
    for rid, p in sorted(b.status_claims.items()):
        if p < claim_min:
            continue
        mid = req.get(rid)
        met = b.milestones.get(mid) if mid else (b.full_pass if rid in req else None)
        if met is False:
            over.append(f"{rid}={p:g} ({mid or 'full pass'} fails)")
    if over:
        out.append(Fact("status_overclaim", f"STATUS.json claims {len(over)} requirement(s) the hidden tests fail: {', '.join(over[:6])}",
                        step=len(b.steps), evidence=f"[STATUS claims] {json.dumps(b.status_claims)}"[:300]))
    m = TESTS_CLAIM.search(b.final_message or "")
    if m:
        runs = [st.i for st in b.steps for _, c in commands_of([ToolCall(st.function, st.arguments)]) if TEST_RUN.search(c)]
        # an edit of the code: the STATUS.json write and scratch files (a reproduction under /tmp or tmp/) change no behaviour
        edits = [st.i for st in b.steps if any(p != "STATUS.json" and not p.startswith(("/tmp/", "tmp/", "/w/tmp/")) for p in _edits(st))]
        last_edit = max(edits) if edits else 0
        if not runs or max(runs) < last_edit:
            why = "no test or reproduction command ran" if not runs else f"the last test run (step {max(runs)}) came before the last edit (step {last_edit})"
            out.append(Fact("claimed_unrun_tests", f"the final message claims verified work, but {why}", step=len(b.steps), evidence=m.group(0)))
    files = sorted(patch_files(b.patch))
    hidden = set(hidden_paths)
    bad = [f for f in files if f in hidden or (protected is not None and f != "STATUS.json" and protected(f))]
    if bad:
        out.append(Fact("protected_edit", f"the final diff touches {len(bad)} protected or hidden-test path(s): {bad[:6]}", step=len(b.steps),
                        evidence=", ".join(bad[:6])))
    return out


def budget_binding(b: EpisodeBundle, *, tool_limit: int | None = None, tail: float = 0.3) -> dict[str, Any] | None:
    """Whether the episode's budget bound it (S15): None when it did not run into the tool limit (a ``limit_stop``, or a
    reply within ``tail`` of the limit when ``tool_limit`` is given). Else ``verdict``: ``binding`` (Phi still rose in
    the last ``tail`` of the episode's tool calls and the work is unfinished: more budget could have helped), ``stuck``
    (no Phi gain in that tail: the budget did not decide it) or ``unmeasured`` (no measured checkpoint). The factory's
    rule is unchanged: a limit stop is the ``budget`` cause whatever this says; this tells the limits owner whether to
    raise the limit."""
    n = len(b.steps)
    at_limit = str(b.stop) in BUDGET_STOPS or (tool_limit is not None and n >= (1 - tail / 3) * tool_limit)
    if not at_limit or b.passed or b.masked:
        return None
    pts = [(c.step, c.phi) for c in b.checkpoints if c.measured and c.phi is not None]
    if not pts:
        return {"verdict": "unmeasured", "steps": n, "stop": b.stop}
    phis = [b.phi0 or 0.0] + [p for _, p in pts]
    gains = [s for (s, p), prev in zip(pts, phis[:-1]) if p > prev + EPS]
    last_gain = max(gains) if gains else None
    cut = (1 - tail) * max(n, 1)
    v = "binding" if last_gain is not None and last_gain >= cut and (b.phi_final or 0.0) < 1.0 - EPS else "stuck"
    return {"verdict": v, "steps": n, "stop": b.stop, "last_gain_step": last_gain, "tail_from_step": round(cut, 1), "phi_final": b.phi_final,
            "phi_peak": max(phis)}


def fault_regex(signature: str) -> re.Pattern[str] | None:
    """The output pattern a World-fault signature (``facts``) reproduces as, whatever the shell's wording."""
    m = re.match(r"(?:project )?module '([\w.]+)'", signature)
    if m:
        return re.compile(rf"No module named '?{re.escape(m.group(1))}\b")
    m = re.match(r"command '([^']+)' not found", signature)
    if m:
        return re.compile(rf"\b{re.escape(m.group(1))}: (?:command )?not found")
    return None


def repro_command(cmd: str) -> str:
    """A recorded shell command as a reproduction: trailing pagers and filters (``| head``, ``| tail``, ``| grep``) are cut,
    so the command's own exit status decides."""
    out = cmd.strip()
    while True:
        new = re.sub(r"\s*\|\s*(head|tail|grep|less|more|cat|sort|uniq|wc)\b[^|]*$", "", out)
        if new == out:
            return out
        out = new


def deterministic_cause(fs: Sequence[Fact]) -> str | None:
    return next((f.cause for f in fs if f.cause), None)


# ---- 2. first wrong step -------------------------------------------------------------------------------------------------------
@dataclass
class FirstWrongStep:
    kind: str  # none | lost | plateau | flat | divergence | unmeasured
    step: int | None = None
    turn: int | None = None
    checkpoint: int | None = None
    phi_before: float | None = None
    phi_after: float | None = None
    peak: float | None = None
    final: float | None = None
    files: list[str] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    reference: str = ""
    detail: str = ""
    position: float | None = None  # the step as a fraction of the episode's tool calls


def _edits(st: ToolStep) -> list[str]:
    """Paths a tool call writes (edit/write tools' ``path``; shell commands' redirect and ``sed -i`` targets)."""
    a = st.arguments if isinstance(st.arguments, dict) else {}
    if re.search(r"write|edit|create|replace|patch", st.function, re.IGNORECASE):
        p = a.get("path") or a.get("file") or a.get("filename")
        return [str(p).lstrip("./")] if p else []
    out = []
    for _, cmd in commands_of([ToolCall(st.function, st.arguments)]):
        out += [m.lstrip("./") for m in re.findall(r"(?:>>?|sed\s+-i(?:\s+\S+)?|tee)\s+([\w./-]+\.\w+)", cmd)]
    return out


def _turn_of(b: EpisodeBundle, step: int | None) -> int | None:
    if step is None:
        return None
    for st in b.steps:
        if st.i >= step and st.turn is not None:
            return st.turn
    return b.steps[-1].turn if b.steps else None


def first_wrong_step(b: EpisodeBundle, passing: Sequence[EpisodeBundle] = (), *, oracle_files: Sequence[str] = ()) -> FirstWrongStep:
    """See the module doc, step 2. ``passing``: passing episodes of the same task (the divergence reference)."""
    if b.passed:
        return FirstWrongStep("none", detail="the episode passed")
    pts = [(0, b.phi0 if b.phi0 is not None else 0.0, {}, "")] + [(c.step, c.phi, c.milestones, c.patch or "") for c in b.checkpoints if c.measured and c.phi is not None]
    if len(pts) < 2:
        return FirstWrongStep("unmeasured", detail="no measured checkpoint")
    phis = [p[1] for p in pts]
    peak, final = max(phis), phis[-1]
    fws: FirstWrongStep
    if peak > final + EPS:
        last_peak = max(i for i, v in enumerate(phis) if v >= peak - EPS)
        i = last_peak + 1
        lo, hi = pts[last_peak], pts[i]
        files = files_between(lo[3], hi[3])
        edits = [st.i for st in b.steps if lo[0] < st.i <= hi[0] and set(_edits(st)) & set(files)]
        step = edits[0] if edits else lo[0] + 1
        lost = sorted(m for m, v in lo[2].items() if v and not hi[2].get(m))
        fws = FirstWrongStep("lost", step, _turn_of(b, step), i, lo[1], hi[1], peak, final, files, edits, lost,
                             detail=f"Phi {lo[1]} -> {hi[1]} between steps {lo[0]} and {hi[0]} and never back to {peak}")
    elif peak > phis[0] + EPS:
        last_gain = max(i for i in range(1, len(phis)) if phis[i] > phis[i - 1] + EPS)
        step = pts[last_gain][0]
        fws = FirstWrongStep("plateau", step, _turn_of(b, step), last_gain, phis[last_gain - 1], phis[last_gain], peak, final,
                             detail=f"Phi reached {peak} at step {step} and never rose again")
    else:
        first_edit = next((st.i for st in b.steps if _edits(st)), None)
        touched = sorted(set(patch_files(b.patch)) & set(oracle_files))
        fws = FirstWrongStep("flat", 0, _turn_of(b, 1), 0, phis[0], final, peak, final, touched, [first_edit] if first_edit else [],
                             detail=f"Phi never rose above the start state; first edit at step {first_edit}; the end state touches "
                                    f"{len(touched)} of the oracle's {len(oracle_files)} files")
    ref_files: set[str] = set()
    for p in passing:
        ref_files |= set(patch_files(p.patch))
    reference = f"{len(passing)} passing episode(s)" if passing else ("the oracle" if oracle_files else "")
    if not passing:
        ref_files = set(oracle_files)
    if ref_files:
        from crucible.factory.lanes.executable import is_protected

        off = [(st.i, f) for st in b.steps for f in _edits(st) if f not in ref_files and not is_protected(f) and f != "STATUS.json"]
        gains = [next((c.step for c in p.checkpoints if c.measured and (c.phi or 0) > (p.phi0 or 0) + EPS), None) for p in passing]
        gains = [g for g in gains if g is not None]
        lag = int(statistics.median(gains)) if gains and fws.kind == "flat" else None
        cand = [(off[0][0], f"first edit outside every reference end state: {off[0][1]}", [off[0][1]])] if off else []
        if lag is not None:
            touched = [st.i for st in b.steps if st.i <= lag and set(_edits(st)) & ref_files]
            if not touched:
                cand.append((lag, f"by step {lag} the passing episodes had gained a milestone; this one had not edited a file they edit", []))
        cand.sort(key=lambda x: x[0])
        if cand and (fws.kind == "flat" or (passing and fws.step is not None and cand[0][0] < fws.step)):
            step, why, files = cand[0]
            fws = FirstWrongStep("divergence", step, _turn_of(b, step), None, fws.phi_before, fws.phi_after, peak, final, files, [step], [],
                                 reference=reference, detail=why)
        fws.reference = fws.reference or reference
    n = max(1, len(b.steps))
    fws.position = None if fws.step is None else round(min(1.0, fws.step / n), 4)
    return fws


# ---- 8.5 audit the passes too ------------------------------------------------------------------------------------------------------
def pass_audit(b: EpisodeBundle, *, oracle_patch: str | None = None, oracle_files: Sequence[str] = (), peers: Sequence[EpisodeBundle] = (),
               short_frac: float = 0.5, fs: Sequence[Fact] | None = None) -> list[str]:
    """Reasons a passing episode is a ``verifier_gap`` candidate (empty: none, or not a pass)."""
    if not b.passed:
        return []
    from crucible.factory.lanes.executable import is_protected

    why = []
    calls = [len(p.steps) for p in peers if p.passed and p.id != b.id]
    if len(calls) >= 2 and len(b.steps) < short_frac * statistics.median(calls):
        why.append(f"short_path: {len(b.steps)} tool calls vs a median of {statistics.median(calls):g} over {len(calls)} passing peers")
    if oracle_patch:
        mine, theirs = changed_lines(b.patch), changed_lines(oracle_patch)
        if theirs and mine < short_frac * theirs:
            why.append(f"short_path: {mine} changed source lines vs the oracle's {theirs}")
    ref = set(oracle_files) or set(patch_files(oracle_patch or ""))
    if ref:
        outside = sorted(p for p in patch_files(b.patch) if p not in ref and not is_protected(p) and p != "STATUS.json")
        if outside:
            why.append(f"outside_files: {outside[:10]}")
    for f in fs if fs is not None else facts(b):
        if f.kind in ("path_near_miss", "path_violation"):
            why.append(f"{f.kind}: {f.detail} ({f.evidence[:120]})")
    return list(dict.fromkeys(why))


# ---- 3. the cause label: an M16 RLM auditor over the bundle --------------------------------------------------------------------
AUDITOR_TASK = """You audit one attempt by a coding agent at a long-horizon task. The attempt's whole record is in CONTEXT: the request
([request]), every model turn ([turn k]), every tool call ([tool i]) with its result ([result i]), the checkpoints where hidden tests
graded the workspace ([checkpoint ...]: phi, milestones gained and lost, files changed), the final message, the STATUS.json claims,
the grade and the grader's output ([grader tail]), the path report, and deterministic facts and the first wrong step computed by the
factory ([facts], [first wrong step]). CONTEXT is long: read it with code (CONTEXT.search, CONTEXT.read), never all at once.

Decide WHY the attempt failed (or, for a pass, whether it really does what the request asks), from a closed list:
- skill_gap: the task is fair and solvable as stated; the agent lacked the skill or made a mistake.
- spec_defect: the request is ambiguous, contradictory, or omits something the hidden tests require (e.g. a public name).
- unstated_format: the agent did the work but missed a format or convention the request never stated.
- world_bug: the environment (a tool, the sandbox, missing software) behaved wrongly, so correct work could not succeed.
- verifier_defect: the attempt is correct but the verifier failed it (or grades something the request does not ask).
- budget: the agent ran out of steps, tokens or time while still making progress.
- env_flake: an infrastructure failure unrelated to the task.
Use "abstain" when the record does not let you decide. Say whether the attempt satisfies every requirement (solution_correct: true,
false, or null). Quote the attempt VERBATIM as evidence: one exact line or phrase of a tool call, a result, a message or the grader's
output, copied character for character from CONTEXT (it is checked; the factory's own [facts] and [first wrong step] lines do not count).
Work in at most 10 python calls: search the [grade], [grader tail], [facts] and [first wrong step] sections first, read around the
first wrong step, then decide.
Optionally name one defect the factory should fix: {"kind": one of verifier_gap | false_fail | spec_defect | unstated_format |
world_bug | intent_drift, "pointer": a hidden test id, a tool call number, or a file}.

Finish with exactly one call: rlm.finish(json.dumps({"cause": ..., "solution_correct": ..., "evidence": "<verbatim quote>",
"rationale": "<one sentence>", "defect": {...} or None}))"""


# Phase A: the forced final-verdict turn. P3 ended 39 of 120 sessions at the 24-turn cap with no verdict (rejected, never
# guessed); P5b 30 of 41 first sessions. When the cap ends a session without ``rlm.finish``, the auditor gets one more model
# turn whose only tool is ``verdict`` (the same JSON the brief asks ``rlm.finish`` for; ``audit.validate`` checks it the same
# way: the closed taxonomy and a verbatim quote). It is a harness switch, an ``fsm`` entry of the auditor's ``HarnessState``
# (``FINAL_VERDICT_EDIT``), so it is enabled only through ``AuditorHarness.refine`` (a measured paired A/B), never by hand.
FINAL_VERDICT_SWITCH = "final_verdict_turn"
FINAL_VERDICT_EDIT = {"action": "create", "kind": "fsm", "id": FINAL_VERDICT_SWITCH, "family": "auditor", "title": "Forced final verdict turn",
                      "content": "When the turn cap ends a session without rlm.finish, one more model turn is given whose only tool is verdict.",
                      "metadata": {FINAL_VERDICT_SWITCH: True}}
FINAL_VERDICT_PROMPT = ("You have reached the turn limit: no more python calls. Decide now from what you have already read: call the `verdict` "
                        "tool exactly once (the same fields rlm.finish would carry). `evidence` is checked character for character against "
                        "the attempt: copy one short span (5 to 15 words) exactly as it appeared in a [tool], [result], message or [grader "
                        "tail] line your python calls printed above; never paraphrase, never quote the [facts] or [first wrong step] lines. "
                        "Use cause \"abstain\" if you cannot decide.")
VERDICT_TOOL = {"type": "function", "function": {
    "name": "verdict", "description": "Your final audit verdict for this attempt.",
    "parameters": {"type": "object", "required": ["cause", "solution_correct", "evidence", "rationale"], "properties": {
        "cause": {"type": "string", "enum": ["skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake", "abstain"]},
        "solution_correct": {"type": ["boolean", "null"]},
        "evidence": {"type": "string", "description": "a verbatim quote from the attempt"},
        "rationale": {"type": "string"},
        "defect": {"type": ["object", "null"], "properties": {"kind": {"type": "string"}, "pointer": {"type": "string"}}}}}}}


def switch_on(entries: Mapping[str, Any] | None, key: str = FINAL_VERDICT_SWITCH) -> bool:
    """Whether a harness switch entry (``metadata[key]`` true, family auditor or ``*``) is present in ``entries``."""
    for e in (entries or {}).values():
        md = getattr(e, "metadata", None) if not isinstance(e, Mapping) else e.get("metadata")
        fam = getattr(e, "family", None) if not isinstance(e, Mapping) else e.get("family")
        if isinstance(md, Mapping) and md.get(key) and fam in ("auditor", "*"):
            return True
    return False


class AuditorHarness:
    """The auditor's supplemental notes (an m14 ``HarnessState``, family ``auditor``). They change only through
    ``refine`` (``HarnessState.evaluate``: a paired A/B on the same seeds, alternating order, a noise re-run, guards); this
    class offers no other way to edit them (00 §8.4)."""

    FAMILY = "auditor"

    def __init__(self, path: Path | None = None):
        from crucible.actors.rlm.harness_state import HarnessState

        self._state = HarnessState(path)

    @property
    def entries(self) -> dict[str, Any]:
        return dict(self._state.entries)

    @property
    def version(self) -> int:
        return self._state.version

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._state.events)

    def digest(self, query: str, entries: Mapping[str, Any] | None = None) -> str:
        from crucible.actors.rlm.harness_state import HarnessState

        src = dict(self._state.entries if entries is None else entries)
        # a harness switch (an fsm entry with a switch flag) is behaviour, never prompt text
        keep = {k: v for k, v in src.items() if not (getattr(v, "kind", "") == "fsm" and switch_on({k: v}))}
        if entries is None and len(keep) == len(src):
            hs = self._state  # the shown/used counters stay on the real state
        else:
            hs = HarnessState(None)
            hs.entries = keep
        if not hs.select(self.FAMILY):
            return ""
        return hs.digest(query, family=self.FAMILY)

    def switched(self, key: str = FINAL_VERDICT_SWITCH) -> bool:
        return switch_on(self._state.entries, key)

    async def refine(self, edits: Sequence[dict[str, Any]], fn: Callable[[dict[str, Any], int], Awaitable[dict[str, float]]], seeds: Sequence[int],
                     **kw: Any) -> dict[str, Any]:
        """``fn(entries, seed)`` audits a seeded-control batch with the notes ``entries`` and returns
        ``{"seeded_accuracy", "label_rate"}``; the edit is kept only if it measurably helps."""
        kw.setdefault("primary", "seeded_accuracy")
        kw.setdefault("guards", ("label_rate",))
        kw.setdefault("trigger", "auditor refine")
        return await self._state.evaluate(edits, fn, seeds, **kw)


class P3Launcher:
    """The auditor's kernel in its own ``crucible-p3-audit-*`` container: no network, read-only root, every capability
    dropped, an unprivileged user (the ``DockerLauncher`` hardening under this phase's container names)."""

    def __init__(self, image: str = "python:3.12-slim", *, prefix: str = "crucible-p3-audit-", memory: str = "1g", lifetime_s: int = 3600,
                 docker: str = "docker"):
        self.image, self.prefix, self.memory, self.lifetime_s, self.docker = image, prefix, memory, int(lifetime_s), docker
        self.name: str | None = None
        self._exec: Any = None

    def clone(self) -> P3Launcher:
        return P3Launcher(self.image, prefix=self.prefix, memory=self.memory, lifetime_s=self.lifetime_s, docker=self.docker)

    async def _docker(self, *argv: str, data: bytes | None = None) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(self.docker, *argv, stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await p.communicate(data)
        return p.returncode or 0, (out + err).decode(errors="replace")

    async def start(self, seed: int) -> Any:
        from crucible.actors.rlm.kernel import ExecLauncher, KernelStartError

        if self._exec is None:
            self.name = f"{self.prefix}{uuid.uuid4().hex[:12]}"
            rc, out = await self._docker("run", "-d", "--rm", "--name", self.name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                                         "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534", "--memory", self.memory,
                                         "--pids-limit", "256", "--cpus", "1", "--workdir", "/tmp", "--env", "HOME=/tmp", self.image, "sleep", str(self.lifetime_s))
            if rc != 0:
                self.name = None
                raise KernelStartError(f"docker run failed: {out[-300:]}", "crucible:env_container_lost")

            async def push(src: Path, dest: str) -> None:
                rc2, out2 = await self._docker("exec", "-i", str(self.name), "sh", "-c", f"cat > {dest}", data=Path(src).read_bytes())
                if rc2 != 0:
                    raise KernelStartError(f"pushing the kernel failed: {out2[-300:]}", "crucible:env_container_lost")

            self._exec = ExecLauncher([self.docker, "exec", "-i", self.name], push=push)
        return await self._exec.start(seed)

    async def kill(self, proc: Any, pgid: int | None) -> bool:
        return await self._exec.kill(proc, pgid) if self._exec is not None else True

    async def aclose(self) -> None:
        if self.name:
            name, self.name, self._exec = self.name, None, None
            await self._docker("rm", "-f", name)


def _parse_answer(ans: str | None) -> dict[str, Any]:
    if not ans:
        return {}
    with contextlib.suppress(json.JSONDecodeError):
        r = json.loads(ans)
        return r if isinstance(r, dict) else {}
    m = re.search(r"\{.*\}", ans, re.DOTALL)
    if m:
        with contextlib.suppress(json.JSONDecodeError):
            r = json.loads(m.group(0))
            return r if isinstance(r, dict) else {}
    return {}


class RLMAuditor(Auditor):
    """The auditor as an M16 RLM agent (``RLMPolicy``) whose ``CONTEXT`` is the trace text (``render_bundle``): it reads
    the record through code. Deterministic facts decide first (as ``Auditor``); the reply is checked by ``audit.validate``
    (closed taxonomy, verbatim quote). ``launcher()`` makes one kernel launcher per session (``P3Launcher`` by default;
    ``LocalLauncher(allow_host=True)`` for scripted tests only). ``sessions`` records turns, stop and cost per label."""

    def __init__(self, llm: Any, *, model: str, workdir: Path, launcher: Callable[[], Any] | None = None, config: AuditConfig | None = None,
                 rlm_config: Any = None, harness: AuditorHarness | None = None, entries: Mapping[str, Any] | None = None, max_steps: int = 40,
                 final_verdict_turn: bool | None = None):
        super().__init__(llm, model=model, config=config)
        from crucible.actors.rlm.agent import RLMConfig

        self.llm, self.workdir = llm, Path(workdir)
        self.launcher = launcher or P3Launcher
        self.rlm_config = rlm_config or RLMConfig(temperature=0.0, tree_max_depth=0, max_turns=24, window_tokens=48_000, max_tokens=4000,
                                                  cell_timeout_s=60.0, wall_s=900.0)
        self.harness, self.entries, self.max_steps = harness, entries, max_steps
        # None: the harness decides (``FINAL_VERDICT_SWITCH`` in ``entries``, else in the harness state); True/False forces it
        self._final_verdict_turn = final_verdict_turn
        self.sessions: list[dict[str, Any]] = []

    @property
    def final_verdict_turn(self) -> bool:
        if self._final_verdict_turn is not None:
            return self._final_verdict_turn
        if self.entries is not None:
            return switch_on(self.entries)
        return bool(self.harness is not None and self.harness.switched())

    async def final_verdict(self, policy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """One more model turn after the cap: the session's own messages, the cap notice, and only the ``verdict`` tool.
        Returns (the reply as ``rlm.finish`` would carry it, a row for the session record)."""
        msgs = [*list(getattr(policy, "messages", None) or []), {"role": "user", "content": FINAL_VERDICT_PROMPT}]
        row: dict[str, Any] = {"forced": True}
        try:  # the verdict function is forced (a model given only one tool still tried to call python: A/B pilot)
            r = await self.llm.complete(msgs, tools=[VERDICT_TOOL], max_tokens=1500, temperature=0.0, seed=0,
                                        tool_choice={"type": "function", "function": {"name": "verdict"}})
        except Exception as e:  # noqa: BLE001 - a provider without named tool_choice: once more without it
            row["tool_choice_error"] = f"{type(e).__name__}: {e}"[:200]
            try:
                r = await self.llm.complete(msgs, tools=[VERDICT_TOOL], max_tokens=1500, temperature=0.0, seed=0)
            except Exception as e2:  # noqa: BLE001 - the forced turn failing leaves the session as it was (no verdict, rejected)
                row["error"] = f"{type(e2).__name__}: {e2}"[:300]
                return {}, row
        row["usd"] = round(float(getattr(r, "cost_usd", 0.0) or 0.0), 6)
        reply: dict[str, Any] = {}
        for tc in getattr(r, "tool_calls", None) or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            args = tc.get("arguments") if "arguments" in tc else (tc.get("function") or {}).get("arguments")
            if name == "verdict":
                reply = args if isinstance(args, dict) else _parse_answer(str(args or ""))
                break
        if not reply:
            reply = _parse_answer(getattr(r, "text", "") or "")
            row["via"] = "text" if reply else "none"
            if not reply:
                row["text"], row["finish"] = str(getattr(r, "text", "") or "")[:300], getattr(r, "raw_finish", None)
        else:
            row["via"] = "tool"
        if reply.get("cause") == "abstain":
            reply["cause"] = None
        row["verdict"] = bool(reply)
        return reply, row

    async def label(self, t: Trace, *, prompt: str, requirements: Sequence[Any] = ()) -> AuditLabel:
        c = self.config
        if c.deterministic and t.verifier_pass is None:
            return AuditLabel(t.id, "env_flake", None, status="deterministic", why="masked grade")
        if c.deterministic and t.verifier_pass is False and str(t.meta.get("stop", "")) in c.budget_stops:
            return AuditLabel(t.id, "budget", None, status="deterministic", why=f"stopped on {t.meta['stop']}")
        try:
            reply = await self.session(t, prompt=prompt, requirements=requirements)
        except Exception as e:  # noqa: BLE001 - a failed session is a rejected label with its reason, never a crash of the batch
            return AuditLabel(t.id, None, None, status="rejected", why=f"auditor session error: {type(e).__name__}: {e}"[:300])
        lab = validate(t, reply, config=c)
        own = t.text.split(FACTS_MARK, 1)[0]
        if lab.status == "labelled" and re.sub(r"\s+", " ", lab.evidence).strip().lower() not in re.sub(r"\s+", " ", own).lower():
            lab = AuditLabel(t.id, None, None, lab.evidence, lab.rationale, status="rejected", why="the quote is the factory's derived lines, not the attempt")
        d = reply.get("defect")
        if isinstance(d, dict) and d.get("kind") in DEFECT_KINDS:
            lab.why = (lab.why + f" defect={d.get('kind')}:{str(d.get('pointer'))[:120]}").strip()
        return lab

    async def session(self, t: Trace, *, prompt: str = "", requirements: Sequence[Any] = ()) -> dict[str, Any]:
        from crucible.actors.policy import EpisodeLimits, run_episode
        from crucible.actors.rlm.policy import RLMPolicy
        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        d = self.workdir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', t.id)[-60:]}-{uuid.uuid4().hex[:6]}"
        d.mkdir(parents=True, exist_ok=True)
        ctx = d / "context.txt"
        ctx.write_text(t.text)
        notes = self.harness.digest(t.text[:2000], self.entries if self.entries is not None else self.harness.entries) if self.harness is not None else ""
        reqs = [{"id": getattr(r, "id", ""), "text": getattr(r, "text", "")} for r in requirements]
        task = (AUDITOR_TASK + f"\n\nVerifier verdict: {'pass' if t.verifier_pass else 'fail'}; failing checks: {t.failing_checks[:30]}"
                + (f"\nRequirements: {json.dumps(reqs)[:4000]}" if reqs else "") + (f"\nBatch request: {prompt[:2000]}" if prompt else "")
                + (f"\n\n{notes}" if notes else ""))
        spec = WorldSpec.model_validate({"world": "factory/trace_audit", "network": "no-network", "apps": [],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": []}]})
        world = World(spec, instructions={"agent": task}, store=Store(d / "store"), workdir=d / "world")
        policy = RLMPolicy("agent", self.llm, launcher=self.launcher(), child_launcher=self.launcher, config=self.rlm_config, ledger_dir=d / "ledger",
                           context_path=ctx)
        t0 = time.monotonic()
        try:
            await world.start()
            ep = await run_episode(world, {"agent": policy}, limits=EpisodeLimits(max_steps=self.max_steps))
        finally:
            with contextlib.suppress(Exception):
                await world.close()
            with contextlib.suppress(Exception):
                await policy.aclose()
        calls = policy.model_calls
        row = {"trace": t.id, "stop": policy.stop, "turns": policy.counters.get("turns", 0), "context_reads": policy.counters.get("context_reads", 0),
               "usd": round(sum(float(c.get("cost_usd") or 0.0) for c in calls), 6), "seconds": round(time.monotonic() - t0, 1),
               "masked": bool(getattr(ep, "masked", False)), "answer": (policy.answer or "")[:2000], "finished": policy.answer is not None}
        reply = _parse_answer(policy.answer)
        if not reply and policy.stop and policy.stop.startswith("budget"):  # out of turns with its verdict printed: the last output
            reply = _parse_answer(getattr(policy, "_last_output", ""))
        if not reply.get("cause") and policy.stop and policy.stop.startswith("budget:turns") and self.final_verdict_turn:
            reply, row["final_turn"] = await self.final_verdict(policy)
            row["usd"] = round(row["usd"] + float(row["final_turn"].get("usd") or 0.0), 6)
        self.sessions.append(row)
        (d / "session.json").write_text(json.dumps(row, indent=1))
        return reply


# ---- 4. typed defect proposals --------------------------------------------------------------------------------------------------
@dataclass
class ProposedCheck:
    """A new hidden check: a test file the grader adds (never in the agent's World) and the test ids it gates on."""

    id: str
    file: str
    source: str
    tests: list[str]
    milestone: str | None = None
    author: str = ""


@dataclass
class Proposal:
    kind: str
    task: str
    traces: list[str]
    artifact: dict[str, Any]  # the demonstrating artifact: {"type": end_state | test_id | repro | cluster, ...}
    change: dict[str, Any]  # add_check | relax | prompt_edit | world_fix | intent
    source: str  # pass_audit | label | fact | grader | cluster | seeded
    why: str
    id: str = ""
    status: str = "proposed"  # proposed | needs_input | accepted | rejected | reproduced_needs_fix | awaiting_human
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    version: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DEFECT_KINDS:
            raise ValueError(f"unknown defect kind {self.kind!r} (one of {DEFECT_KINDS})")
        if not self.id:
            art = {k: v for k, v in self.artifact.items() if k != "patch"}
            self.id = f"{self.kind[:2]}-" + sha({"kind": self.kind, "task": self.task, "artifact": art, "patch": sha(str(self.artifact.get("patch", ""))),
                                                  "change": self.change})[:12]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if "patch" in d["artifact"]:
            d["artifact"] = {**d["artifact"], "patch": f"<{len(str(d['artifact']['patch']))} bytes, sha256 {sha(str(d['artifact']['patch']))[:16]}>"}
        return d


NAME_ERRORS = (
    re.compile(r"ImportError: cannot import name '(\w+)'"),
    re.compile(r"AttributeError: module '[\w.]+' has no attribute '(\w+)'"),
    re.compile(r"AttributeError: type object '\w+' has no attribute '(\w+)'"),
)  # a NameError is the end state's own missing import, never a name the tests ask for


def missing_names(tail: str) -> list[str]:
    """Public names the grader's output says the end state lacks (import, module attribute, class attribute, name)."""
    out: list[str] = []
    for rx in NAME_ERRORS:
        out += [n for n in rx.findall(tail or "") if not n.startswith("_")]
    return list(dict.fromkeys(out))


def stated(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text or "") is not None


def end_state_artifact(b: EpisodeBundle) -> dict[str, Any]:
    return {"type": "end_state", "trace": b.id, "patch": b.patch, "patch_sha": sha(b.patch), "reward": b.reward}


def propose_verifier_gap(b: EpisodeBundle, reasons: Sequence[str], *, check: ProposedCheck | None = None, source: str = "pass_audit") -> Proposal:
    return Proposal("verifier_gap", b.window, [b.id], end_state_artifact(b), {"add_check": asdict(check)} if check else {}, source, "; ".join(reasons))


def propose_false_fail(b: EpisodeBundle, drop: Sequence[str], why: str) -> Proposal:
    return Proposal("false_fail", b.window, [b.id], {**end_state_artifact(b), "failing_tests": list(drop)}, {"relax": {"drop_tests": list(drop)}}, "label", why)


def propose_spec_defects(b: EpisodeBundle) -> list[Proposal]:
    """One ``spec_defect`` per public name the grader says the end state lacks and the request the agent saw never states
    (the D8 grounding, the test source, is checked at acceptance)."""
    names = [n for n in missing_names(b.grader_tail) if not stated(n, b.prompt)]
    if not names or not b.failing_tests:
        return []
    return [Proposal("spec_defect", b.window, [b.id], {"type": "test_id", "tests": list(b.failing_tests[:200]), "prompt_sha": sha(b.prompt),
                                                         "grader_line": next((m.group(0) for rx in NAME_ERRORS for m in rx.finditer(b.grader_tail) if m.group(1) == n), "")},
                     {"prompt_edit": {"names": [n], "tests": list(b.failing_tests[:200])}}, "grader",
                     f"the hidden tests need {n!r}, which the request never names") for n in names]


def propose_world_bugs(b: EpisodeBundle, fs: Sequence[Fact]) -> list[Proposal]:
    out = []
    for f in fs:
        if f.kind != "world_fault" or f.step is None:
            continue
        st = b.steps[f.step - 1]
        out.append(Proposal("world_bug", b.window, [b.id], {"type": "repro", "function": st.function, "arguments": st.arguments, "observed": f.evidence,
                                                            "signature": re.sub(r" \(x\d+\)$", "", f.detail), "step": st.i,
                                                            "workspace": b.meta.get("workspace", "")},
                            {"world_fix": None}, "fact", f.detail))
    return out


def merge_proposals(ps: Sequence[Proposal]) -> list[Proposal]:
    """One proposal per (kind, task, defect): their traces united (a spec defect per name; a World bug per signature)."""
    by: dict[str, Proposal] = {}
    for p in ps:
        if p.kind == "spec_defect":
            key = f"{p.kind}|{p.task}|{p.artifact.get('prompt_sha')}|{p.change['prompt_edit']['names']}"
        elif p.kind == "world_bug":
            key = f"{p.kind}|{p.task}|{p.artifact.get('signature')}"
        else:
            key = p.id
        if key in by:
            q = by[key]
            q.traces = list(dict.fromkeys(q.traces + p.traces))
            if p.kind == "spec_defect":
                tests = list(dict.fromkeys(q.change["prompt_edit"]["tests"] + p.change["prompt_edit"]["tests"]))
                q.change["prompt_edit"]["tests"] = tests
        else:
            by[key] = dataclasses.replace(p, traces=list(p.traces), change=json.loads(json.dumps(p.change, default=str)))
    return list(by.values())


# ---- executed acceptance: the ports ------------------------------------------------------------------------------------------------
@dataclass
class Verdict:
    passed: bool | None  # None: unmeasured (never a pass, never a fail)
    milestones: dict[str, bool] = field(default_factory=dict)
    failing: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VerifierPort(Protocol):
    """A task's verifier with its executed ground truth: the current verdict, the verdict with added checks or dropped
    tests, one check alone, the oracle, mutants, floors and the task's own alternatives."""

    task_id: str

    async def verdict(self, solution: Any, *, add: Sequence[ProposedCheck] = (), drop: Sequence[str] = ()) -> Verdict: ...

    async def check(self, check: ProposedCheck, solution: Any) -> bool | None: ...

    def oracle(self) -> Any: ...

    def mutants(self) -> list[tuple[str, Any]]: ...

    def floors(self) -> list[tuple[str, Any]]: ...

    def content(self) -> dict[str, Any]: ...


@runtime_checkable
class AlternativesPort(Protocol):
    """Valid alternatives of a task (P2's ``alternatives`` module lands behind this port; faked until then)."""

    async def alternatives(self, task_id: str) -> list[tuple[str, Any]]: ...


@runtime_checkable
class SecondVerifierPort(Protocol):
    """P2's second verifier (another family, built from the prompt alone): its verdict on an end state."""

    async def verdict(self, task_id: str, solution: Any) -> bool | None: ...


@runtime_checkable
class WorldPort(Protocol):
    """The task's World: run a reproduction on a state (optionally with a World fix applied) and the S3 gate."""

    async def reproduce(self, repro: Mapping[str, Any], *, state: Any, fix: Mapping[str, Any] | None = None) -> dict[str, Any]: ...

    async def s3(self, *, fix: Mapping[str, Any] | None = None) -> dict[str, Any]: ...

    def content(self) -> dict[str, Any]: ...


class NoAlternatives:
    """Until P2: no alternatives beyond the task's own."""

    async def alternatives(self, task_id: str) -> list[tuple[str, Any]]:
        return []


class FakeAlternatives:
    """A fixed alternative list per task (tests and the seeded scenario stand in for P2's alternatives)."""

    def __init__(self, by_task: Mapping[str, Sequence[tuple[str, Any]]]):
        self.by_task = {k: list(v) for k, v in by_task.items()}

    async def alternatives(self, task_id: str) -> list[tuple[str, Any]]:
        return list(self.by_task.get(task_id, []))


class NoSecondVerifier:
    async def verdict(self, task_id: str, solution: Any) -> bool | None:
        return None


@dataclass
class TaskContext:
    """What acceptance needs about one task (``task`` = the bundle's ``window``). Every port is optional; a proposal whose
    port is missing stays ``needs_input`` with the reason."""

    task: str
    verifier: Any = None  # VerifierPort
    world: Any = None  # WorldPort
    rec: Mapping[str, Any] | None = None  # the window record (D8: the hidden tests' source)
    base_dir: Path | None = None  # the start tree (A): a name A already defines is never an unstated name
    prompt: str = ""
    requirements: list[dict[str, Any]] = field(default_factory=list)
    oracle_patch: str = ""
    oracle_files: list[str] = field(default_factory=list)
    oracle_state: Any = None  # the oracle as a solution the World port applies (default: ``verifier.oracle()`` or ``oracle_patch``)
    check_author: Any = None  # async (proposal, bundle) -> ProposedCheck | None
    world_fixer: Any = None  # async (proposal) -> dict | None


@dataclass
class AcceptConfig:
    flake_reruns: int = 30  # per concurrency level: 60 clean runs bound the flake rate at 4.9% (factory.flake)
    flake_concurrency: tuple[int, ...] = (1, 4)
    flake_upper_max: float = 0.05
    min_alternatives: int = 1


def _reject(p: Proposal, why: str, ev: dict[str, Any]) -> Proposal:
    p.status, p.evidence = "rejected", ev
    p.reasons.append(why)
    return p


async def accept_verifier_gap(p: Proposal, v: Any, alts: Any, *, config: AcceptConfig | None = None) -> Proposal:
    from crucible.factory.flake import FlakeConfig, measure_flake

    cfg = config or AcceptConfig()
    raw = p.change.get("add_check")
    if not raw:
        p.status = "needs_input"
        p.reasons.append("no new check was written for the demonstrating end state")
        return p
    chk = ProposedCheck(**raw)
    demo = p.artifact.get("patch")
    ev: dict[str, Any] = {"check": {"id": chk.id, "file": chk.file, "tests": chk.tests, "source_sha": sha(chk.source)}, "demo_patch_sha": sha(str(demo))}
    cur = await v.verdict(demo)
    ev["current_on_demo"] = cur.passed
    if cur.passed is not True:
        return _reject(p, f"the current verifier does not pass the demonstrating end state (verdict {cur.passed}): nothing to tighten", ev)
    new_demo = await v.check(chk, demo)
    ev["new_check_on_demo"] = new_demo
    if new_demo is not False:
        return _reject(p, f"the new check does not fail the demonstrating end state (verdict {new_demo})", ev)
    oracle = v.oracle()
    o1, o2 = await v.check(chk, oracle), await v.verdict(oracle, add=[chk])
    ev["oracle"] = {"new_check": o1, "full_verdict_with_check": o2.passed, "failing": o2.failing[:10]}
    if o1 is not True or o2.passed is not True:
        return _reject(p, f"the new check fails the oracle (check {o1}, full verdict {o2.passed}, failing {o2.failing[:5]})", ev)
    pool = list(getattr(v, "alternatives", list)()) + list(await alts.alternatives(v.task_id))
    ev["alternatives"] = {}
    for aid, sol in pool:
        ev["alternatives"][aid] = await v.check(chk, sol)
    bad = [a for a, ok in ev["alternatives"].items() if ok is not True]
    if bad:
        return _reject(p, f"the new check fails valid alternative(s) {bad}: the ten-competent-people test", ev)
    if len(pool) < cfg.min_alternatives:
        return _reject(p, f"{len(pool)} valid alternative(s); the check must pass at least {cfg.min_alternatives} besides the oracle", ev)
    ev["mutants"], ev["floors"] = {}, {}
    for kind, items in (("mutants", v.mutants()), ("floors", v.floors())):
        for mid, sol in items:
            ev[kind][mid] = (await v.verdict(sol, add=[chk])).passed
    still = [f"{k}:{m}" for k in ("mutants", "floors") for m, ok in ev[k].items() if ok is not False]
    if still:
        return _reject(p, f"with the new check these no longer fail (or are unmeasured): {still}", ev)
    fc = FlakeConfig(reruns=cfg.flake_reruns, concurrency=cfg.flake_concurrency, min_reruns=1, min_levels=len(cfg.flake_concurrency),
                     upper_max=cfg.flake_upper_max)
    fr = await measure_flake(lambda i: v.check(chk, oracle), config=fc)
    ev["flake"] = {"n": fr.n, "fails": fr.false_negatives, "masked": fr.masked, "upper": round(fr.upper, 4), "levels": fr.levels}
    if fr.n == 0 or not fr.ok or fr.masked:
        return _reject(p, f"the new check's flake gate fails on the oracle: {fr.reasons() or [f'{fr.masked} unmeasured runs']}", ev)
    p.status, p.evidence = "accepted", ev
    return p


async def accept_false_fail(p: Proposal, v: Any, second: Any) -> Proposal:
    drop = list((p.change.get("relax") or {}).get("drop_tests") or [])
    demo = p.artifact.get("patch")
    ev: dict[str, Any] = {"drop": drop, "demo_patch_sha": sha(str(demo))}
    if not drop:
        return _reject(p, "no test to relax", ev)
    sv = await second.verdict(v.task_id, demo)
    ev["second_verifier"] = sv
    if sv is not True:
        return _reject(p, f"the second verifier does not show the end state correct (verdict {sv})", ev)
    r_demo, r_oracle = await v.verdict(demo, drop=drop), await v.verdict(v.oracle(), drop=drop)
    ev["relaxed_on_demo"], ev["relaxed_on_oracle"] = r_demo.passed, r_oracle.passed
    if r_demo.passed is not True or r_oracle.passed is not True:
        return _reject(p, f"the relaxed verifier does not pass the end state ({r_demo.passed}) and the oracle ({r_oracle.passed})", ev)
    ev["mutants"], ev["floors"] = {}, {}
    for kind, items in (("mutants", v.mutants()), ("floors", v.floors())):
        for mid, sol in items:
            ev[kind][mid] = (await v.verdict(sol, drop=drop)).passed
    still = [f"{k}:{m}" for k in ("mutants", "floors") for m, ok in ev[k].items() if ok is not False]
    if still:
        return _reject(p, f"the relaxed check lets these pass (or is unmeasured): {still}", ev)
    p.status, p.evidence = "accepted", ev
    return p


def amend_requirement(text: str, name: str) -> str:
    return f"{text.rstrip()} The release exposes this as `{name}`."


def _project_module(base_dir: Path | None, mod: str) -> bool:
    if base_dir is None:
        return True
    rel = mod.replace(".", "/")
    return any((Path(base_dir) / r).exists() for r in (f"{rel}.py", f"{rel}/__init__.py", f"src/{rel}.py", f"src/{rel}/__init__.py"))


def test_source_names(tests: Mapping[str, str], test_id: str, *, base_dir: Path | None = None) -> set[str]:
    """The project names the hidden test's own source asks for (D8): names its file imports from a project module, and
    attributes its body (decorators included) reads off a project module or a name imported from one. ``base_dir``: the
    start tree that decides what a project module is (None: every module counts). Empty when the source is not found."""
    import ast
    import warnings

    parts = test_id.split("::")
    text = tests.get(parts[0])
    if text is None:
        return set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return set()
    project: set[str] = set()  # local names bound to the project: module aliases and names imported from project modules
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and _project_module(base_dir, n.module):
            for a in n.names:
                project.add(a.asname or a.name)
                out.add(a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if _project_module(base_dir, a.name):
                    project.add(a.asname or a.name.split(".")[0])
    body: list[ast.stmt] = tree.body
    node: Any = None
    for nm in (re.sub(r"\[.*$", "", x) for x in parts[1:]):
        node = next((x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == nm), None)
        if node is None:
            return out
        body = getattr(node, "body", [])
    for n in ast.walk(node) if node is not None else ():
        if isinstance(n, ast.Attribute):
            root = n.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in project:
                out.add(n.attr)
    return out


def defined_at_start(base_dir: Path, name: str) -> list[str]:
    """Project files (tests excluded) of the start tree that define ``name`` (def, class, assignment, import binding)."""
    from crucible.factory.lanes.executable import is_test_path

    rx = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*[:=]|^\s*(?:from\s+\S+\s+)?import\s+.*\b{re.escape(name)}\b",
                    re.MULTILINE)
    hits = [p.relative_to(base_dir).as_posix() for p in Path(base_dir).rglob(f"{name}.py")] + [
        p.relative_to(base_dir).as_posix() for p in Path(base_dir).rglob(f"{name}/__init__.py")]
    for f in sorted(Path(base_dir).rglob("*.py")):
        rel = f.relative_to(base_dir).as_posix()
        if is_test_path(rel) or any(x.startswith(".") for x in Path(rel).parts):
            continue
        with contextlib.suppress(OSError):
            if rx.search(f.read_text(errors="replace")):
                hits.append(rel)
    return hits


def accept_spec_defect(p: Proposal, *, rec: Mapping[str, Any], prompt: str, requirements: Sequence[Mapping[str, Any]], base_dir: Path | None = None) -> Proposal:
    """D8: the missing fact must come from the failing hidden test's own source, and the name must be new in the window
    (absent from the start tree: a name A defines that the end state lost is the agent's breakage); the amended prompt
    passes the lint and the two-way ledger (no new problem); the counterfactual is then due."""
    from crucible.factory.lanes.windows import findings_from_tests, lint_prompt, locate, maintainer_text, window_ledger

    pe = p.change.get("prompt_edit") or {}
    names, tests = list(pe.get("names") or []), list(pe.get("tests") or [])
    ev: dict[str, Any] = {"names": names, "tests": tests[:20]}
    if names and all(stated(n, prompt) for n in names):
        return _reject(p, f"the task's current version already states {names}", ev)
    findings, refused = findings_from_tests(rec, tests)
    ev["findings"], ev["refused"] = findings[:10], refused[:10]
    if not findings:
        return _reject(p, "no failing hidden test of this task backs the edit (D8: free text is never sent to the author)", ev)
    owner = {t: m["id"] for m in rec.get("milestones") or [] for t in m["tests"]}
    tests_b = rec.get("tests_b") or {}
    grounded: dict[str, list[str]] = {}
    for n in names:
        for t in tests:
            if n in test_source_names(tests_b, t, base_dir=base_dir):
                grounded.setdefault(n, []).append(t)
    ev["grounded"] = {n: ts[:5] for n, ts in grounded.items()}
    guess = [n for n in names if n not in grounded]
    if guess:
        return _reject(p, f"{guess} not found in the failing hidden tests' own source: an auditor's guess is never a prompt edit (D8)", ev)
    if base_dir is not None:
        old = {n: defined_at_start(base_dir, n) for n in names}
        ev["defined_at_start"] = {n: v[:3] for n, v in old.items() if v}
        if any(old.values()):
            return _reject(p, f"{sorted(n for n, v in old.items() if v)} already exist at the start state: the end state lost them (the agent's breakage, "
                              "not an unstated name)", ev)
    reqs = [dict(r) for r in requirements]
    by_ms = {r.get("milestone"): r for r in reqs if r.get("milestone")}
    new_prompt = prompt
    edits = []
    for n, ts in grounded.items():
        ms = next((owner[t] for t in ts if t in owner), None)
        r = by_ms.get(ms)
        if r is None:
            return _reject(p, f"{n!r}: its tests belong to no milestone with a requirement (regression tests only)", ev)
        old = str(r["text"])
        if old not in new_prompt:
            return _reject(p, f"requirement {r['id']}'s text is not verbatim in the prompt", ev)
        r["text"] = amend_requirement(old, n)
        new_prompt = new_prompt.replace(old, r["text"], 1)
        edits.append({"requirement": r["id"], "milestone": ms, "name": n, "before": old, "after": r["text"]})
    ev["edits"] = edits

    def spans_of(text: str, rs: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
        return {str(r["id"]): list(locate(text, str(r["text"])) or (0, 1)) for r in rs}

    lint_old, lint_new = set(lint_prompt(maintainer_text(prompt))), lint_prompt(maintainer_text(new_prompt))
    ev["lint_new"] = [x for x in lint_new if x not in lint_old]
    _, led_old = window_ledger(rec, list(requirements), spans_of(prompt, requirements), prompt)
    _, led_new = window_ledger(rec, reqs, spans_of(new_prompt, reqs), new_prompt)
    ev["ledger_new"] = [x for x in led_new if x not in set(led_old)]
    ev["still_unstated"] = [n for n in names if not stated(n, new_prompt)]
    if ev["lint_new"] or ev["ledger_new"] or ev["still_unstated"]:
        return _reject(p, f"the amended prompt fails the lint {ev['lint_new']} or the ledger {ev['ledger_new']} or still omits {ev['still_unstated']}", ev)
    ev["prompt_sha_before"], ev["prompt_sha_after"] = sha(prompt), sha(new_prompt)
    ev["counterfactual_rerun_needed"] = True
    p.change = {**p.change, "prompt": new_prompt, "requirements": reqs}
    p.status, p.evidence = "accepted", ev
    p.reasons.append("accepted on test-derived facts, lint and ledger; the counterfactual re-run (P4) is due before the version is trained on")
    return p


async def accept_world_bug(p: Proposal, world: Any, *, oracle: Any, fixer: Any = None) -> Proposal:
    repro = {k: p.artifact.get(k) for k in ("function", "arguments", "observed", "signature", "workspace")}
    ev: dict[str, Any] = {"repro": repro}
    before = await world.reproduce(repro, state=oracle)
    ev["before_fix_on_oracle"] = {k: before.get(k) for k in ("failed", "exit", "output")}
    if not before.get("failed"):
        return _reject(p, "the reproduction does not fail on a known-good state (the oracle): not a World bug", ev)
    fix = p.change.get("world_fix")
    if fix is None and fixer is not None:
        fix = await fixer(p)
    if not fix:
        p.status, p.evidence = "reproduced_needs_fix", ev
        p.reasons.append("reproduced in the World on the oracle state; no World fix was proposed (a World expert's job)")
        return p
    p.change = {**p.change, "world_fix": fix}
    after = await world.reproduce(repro, state=oracle, fix=fix)
    ev["after_fix_on_oracle"] = {k: after.get(k) for k in ("failed", "exit", "output")}
    if after.get("failed"):
        return _reject(p, "the fix does not make the reproduction pass on the oracle state", ev)
    s3 = await world.s3(fix=fix)
    ev["s3"] = s3
    if not s3.get("ok"):
        return _reject(p, f"the fixed World fails S3: {s3.get('why')}", ev)
    p.status, p.evidence = "accepted", ev
    return p


def task_subject(task: str, prompt: str) -> str:
    """The version subject of a task's prompt and World: one lineage per published prompt (two runs' prompts of one window
    are two subjects, so an edit is never stacked on a prompt it was not derived from)."""
    return f"{task}@{sha(prompt)[:12]}" if prompt else task


async def accept(p: Proposal, ctx: TaskContext | None, *, alts: Any = None, second: Any = None, config: AcceptConfig | None = None,
                 bundle: EpisodeBundle | None = None, store: VersionStore | None = None) -> Proposal:
    """Dispatch a proposal to its executed gate (§8.3). Intent drift is never accepted here."""
    alts = alts or NoAlternatives()
    second = second or NoSecondVerifier()
    if p.kind == "intent_drift":
        p.status = "awaiting_human"
        p.reasons.append("an intent change is approved by a human, never automatically")
        return p
    if ctx is None:
        p.status = "needs_input"
        p.reasons.append(f"no task context for {p.task}")
        return p
    if p.kind == "verifier_gap":
        if not p.change.get("add_check") and ctx.check_author is not None and bundle is not None:
            chk = await ctx.check_author(p, bundle)
            if chk is not None:
                p.change = {"add_check": asdict(chk)}
        if ctx.verifier is None:
            p.status = "needs_input"
            p.reasons.append("no verifier port for this task")
            return p
        return await accept_verifier_gap(p, ctx.verifier, alts, config=config)
    if p.kind == "false_fail":
        if ctx.verifier is None:
            p.status = "needs_input"
            p.reasons.append("no verifier port for this task")
            return p
        return await accept_false_fail(p, ctx.verifier, second)
    if p.kind in ("spec_defect", "unstated_format"):
        if ctx.rec is None or not ctx.prompt:
            p.status = "needs_input"
            p.reasons.append("no window record, or the prompt this trace saw is in no journal (a horizon or hacker prompt)")
            return p
        if sha(ctx.prompt) != p.artifact.get("prompt_sha", sha(ctx.prompt)):
            p.status = "rejected"
            p.reasons.append("the trace saw another prompt version than the task's current one (already changed)")
            return p
        prompt, reqs = ctx.prompt, ctx.requirements
        head = store.head(task_subject(p.task, ctx.prompt), "task") if store is not None else None
        if head is not None and head.content.get("prompt"):  # an earlier accepted edit of the same prompt: build on it
            prompt, reqs = head.content["prompt"], head.content["requirements"]
        return accept_spec_defect(p, rec=ctx.rec, prompt=prompt, requirements=reqs, base_dir=ctx.base_dir)
    if p.kind == "world_bug":
        if ctx.world is None:
            p.status = "needs_input"
            p.reasons.append("no World port for this task")
            return p
        oracle = ctx.verifier.oracle() if ctx.verifier is not None else (ctx.oracle_state if ctx.oracle_state is not None else ctx.oracle_patch)
        return await accept_world_bug(p, ctx.world, oracle=oracle, fixer=ctx.world_fixer)
    p.status = "needs_input"
    return p


# ---- versions from accepted proposals ---------------------------------------------------------------------------------------------
def version_accepted(p: Proposal, store: VersionStore, ctx: TaskContext) -> Version | None:
    """The new verifier or task version an accepted proposal makes (lineage: parent head, proposal id, evidence hash)."""
    if p.status != "accepted":
        return None
    if p.kind in ("verifier_gap", "false_fail"):
        base = ctx.verifier.content() if ctx.verifier is not None else {"checks": {}}
        head = store.head(p.task, "verifier") or store.root(p.task, "verifier", base)
        content = json.loads(json.dumps(head.content, default=str))
        if p.kind == "verifier_gap":
            c = p.change["add_check"]
            content.setdefault("added", {})[c["id"]] = {"file": c["file"], "tests": c["tests"], "source": c["source"], "milestone": c.get("milestone")}
            change = {"type": "add_check", "check": c["id"]}
        else:
            drop = p.change["relax"]["drop_tests"]
            content["dropped"] = sorted(set(content.get("dropped") or []) | set(drop))
            change = {"type": "relax_check", "drop_tests": drop}
    else:
        base = {"prompt": ctx.prompt, "requirements": ctx.requirements, "world_fixes": []}
        subject = task_subject(p.task, ctx.prompt)
        head = store.head(subject, "task") or store.root(subject, "task", base)
        content = json.loads(json.dumps(head.content, default=str))
        if p.kind in ("spec_defect", "unstated_format"):
            content["prompt"], content["requirements"] = p.change["prompt"], p.change["requirements"]
            change = {"type": "prompt_edit", "edits": p.evidence.get("edits"), "counterfactual_rerun_needed": True}
        else:
            content["world_fixes"] = [*content.get("world_fixes", []), p.change["world_fix"]]
            change = {"type": "world_fix", "fix": p.change["world_fix"]}
    v = store.derive(head.id, content=content, change=change, proposal_id=p.id, evidence=p.evidence, note=p.why[:200])
    p.version = v.id
    return v


# ---- intent drift ----------------------------------------------------------------------------------------------------------------
def signature(b: EpisodeBundle, fs: Sequence[Fact], fws: FirstWrongStep, cause: str | None) -> str:
    """A failure's cluster key: the cause, the first wrong step's kind, and the dominant evidence (a fact kind or the
    grader's error class)."""
    kinds = [f.kind for f in fs if f.kind not in ("path_near_miss",)]
    err = ""
    for rx, name in ((r"ImportError|cannot import name", "missing_name"), (r"AttributeError", "attribute_error"), (r"NameError", "name_error"),
                     (r"AssertionError|assert ", "assertion"), (r"SyntaxError|IndentationError", "syntax_error"), (r"TypeError", "type_error")):
        if re.search(rx, b.grader_tail or ""):
            err = name
            break
    return f"{cause or 'unlabelled'}|{fws.kind}|{(kinds[0] if kinds else err) or 'none'}"


def intent_candidates(rows: Sequence[Mapping[str, Any]], *, intent: Mapping[str, Any], min_traces: int = 3, min_families: int = 2) -> list[dict[str, Any]]:
    """Clusters of off-intent failures (``rows``: {trace, family, signature, on_intent}) with >= ``min_traces`` traces from
    >= ``min_families`` model families: intent candidates for a human. Never applied here."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if not r.get("on_intent"):
            by.setdefault(str(r["signature"]), []).append(r)
    out = []
    for sig, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        fams = sorted({str(r.get("family")) for r in rs if r.get("family")})
        if len(rs) >= min_traces and len(fams) >= min_families:
            out.append({"id": "ic-" + sha({"intent": intent.get("id"), "signature": sig})[:10], "intent": intent.get("id"), "signature": sig,
                        "traces": [r["trace"] for r in rs][:50], "n": len(rs), "families": fams, "status": "awaiting_human", "applied": False,
                        "why": f"{len(rs)} failures from {len(fams)} families cluster on {sig!r}, not on the intent ({intent.get('behaviour')})"})
    return out


def write_intent_candidates(path: Path, cands: Sequence[Mapping[str, Any]]) -> None:
    """Append candidates for human approval (an intent card is never edited here)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        for c in cands:
            fh.write(json.dumps({**c, "written_at": time.time()}) + "\n")


# ---- the loop ------------------------------------------------------------------------------------------------------------------------
@dataclass
class LoopConfig:
    intent: dict[str, Any] = field(default_factory=lambda: {"id": "I1", "behaviour": "Keeps earlier work working while extending it"})
    batch_size: int = 40  # traces per auditor batch; every batch gets every seeded control
    human_share: float = 0.10
    accept: AcceptConfig = field(default_factory=AcceptConfig)
    min_cluster_traces: int = 3
    min_cluster_families: int = 2


def on_intent_default(b: EpisodeBundle, fws: FirstWrongStep, cause: str | None) -> bool:
    """I1 (keeps earlier work working while extending it): a skill-gap failure whose first wrong step lost a milestone."""
    return cause in (None, "skill_gap") and fws.kind == "lost"


FACTS_MARK = "\n[facts] "


def trace_of(b: EpisodeBundle, fs: Sequence[Fact], fws: FirstWrongStep, *, result_chars: int = 3000) -> Trace:
    """The auditor's ``Trace`` of a bundle: ``render_bundle`` plus the factory's facts and first wrong step (marked, so a
    quote of them is refused). ``result_chars`` bounds each tool result (a long-horizon trail's prompt size)."""
    extra = ["", "[facts] " + ("; ".join(f"{f.kind}: {f.detail}" + (f" (step {f.step})" if f.step else "") for f in fs) or "none"),
             f"[first wrong step] {fws.kind} step={fws.step} turn={fws.turn} {fws.detail}"]
    return Trace(b.id, render_bundle(b, result_chars=result_chars, extra=extra), None if b.masked else b.passed, policy=b.policy,
                 failing_checks=[f"c_{m}" for m, ok in b.milestones.items() if not ok], meta={"stop": b.stop})


def _quartiles(xs: Sequence[float]) -> list[float] | None:
    if not xs:
        return None
    if len(xs) == 1:
        return [round(xs[0], 3)] * 3
    q = statistics.quantiles(sorted(xs), n=4, method="inclusive")
    return [round(v, 3) for v in q]


async def run_loop(bundles: Sequence[EpisodeBundle], *, out: Path, store: VersionStore, contexts: Mapping[str, TaskContext] | Callable[[Proposal], Any] = (),
                   auditor: Auditor | None = None, seeded: Sequence[SeededCase] = (), alts: Any = None, second: Any = None, config: LoopConfig | None = None,
                   extra_proposals: Sequence[Proposal] = (), label_ids: set[str] | None = None, spend: Callable[[str], Any] | None = None,
                   on_intent: Callable[[EpisodeBundle, FirstWrongStep, str | None], bool] = on_intent_default, world_limit: int | None = None) -> dict[str, Any]:
    """One pass of the trace-audit loop over ``bundles`` (see the module doc); writes ``report.json``, ``bundles.jsonl``,
    ``proposals.jsonl``, ``intent_candidates.jsonl`` and ``human_queue.jsonl`` under ``out`` and returns the report."""
    cfg = config or LoopConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx_of = contexts if callable(contexts) else (lambda p, _c=dict(contexts): _c.get(p.task))
    by_task: dict[str, list[EpisodeBundle]] = {}
    for b in bundles:
        by_task.setdefault(b.window, []).append(b)
    rows: dict[str, dict[str, Any]] = {}
    fs_of: dict[str, list[Fact]] = {}
    fws_of: dict[str, FirstWrongStep] = {}
    for b in bundles:
        fs = facts(b)
        peers = [p for p in by_task[b.window] if p.passed and p.id != b.id]
        c = None
        with contextlib.suppress(Exception):
            c = ctx_of(Proposal("verifier_gap", b.window, [b.id], {"type": "end_state"}, {}, "probe", ""))
        ofiles = list(getattr(c, "oracle_files", []) or []) if c is not None else []
        fws = first_wrong_step(b, peers, oracle_files=ofiles)
        fs_of[b.id], fws_of[b.id] = fs, fws
        rows[b.id] = {**b.summary(), "facts": [asdict(f) for f in fs], "first_wrong_step": asdict(fws), "deterministic_cause": deterministic_cause(fs),
                      "pass_audit": pass_audit(b, oracle_patch=getattr(c, "oracle_patch", None) or None, oracle_files=ofiles, peers=peers, fs=fs)}
    # -- labels: an RLM auditor with seeded controls, cross-family only, every batch trusted or dropped ------------------------------
    labels: dict[str, AuditLabel] = {}
    trusted: dict[str, bool] = {}
    batches: list[dict[str, Any]] = []
    refused: list[str] = []
    reports: list[AuditReport] = []
    if auditor is not None:
        elig = [b for b in bundles if rows[b.id]["deterministic_cause"] is None and (label_ids is None or b.id in label_ids)]
        traces = [trace_of(b, fs_of[b.id], fws_of[b.id]) for b in elig]
        ok, no = split_by_family(traces, auditor.model)
        refused = [t.id for t in no]
        for i in range(0, len(ok), cfg.batch_size):
            chunk = ok[i: i + cfg.batch_size]
            cm = spend(f"audit-batch-{i // cfg.batch_size}") if spend is not None else contextlib.nullcontext()
            async with _acm(cm):
                rep = await audit(chunk, auditor, prompt="(each trace's CONTEXT holds its own request)", seeded=seeded, salt=f"p3-{i}")
            reports.append(rep)
            t_ok = rep.auditor_trusted is True
            batches.append({"n": len(chunk), "seeded": rep.seeded, "trusted": t_ok, "mix": {k: v.k for k, v in rep.mix.items() if v.k}})
            for tid, lab in rep.labels.items():
                labels[tid], trusted[tid] = lab, t_ok
            from crucible.factory.audit import human_queue

            human_queue(rep, out / "human_queue.jsonl", share=cfg.human_share)
    for tid, lab in labels.items():
        rows[tid]["label"] = {**asdict(lab), "trusted_batch": trusted[tid]}
    # -- proposals ------------------------------------------------------------------------------------------------------------------
    props: list[Proposal] = []
    for b in bundles:
        r = rows[b.id]
        if b.masked:
            continue
        if r["pass_audit"]:
            props.append(propose_verifier_gap(b, r["pass_audit"]))
        if b.role == "hacker":  # a red-team attempt: only a pass is a defect (a confirmed hack), audited above
            continue
        lab = labels.get(b.id)
        if lab is not None and trusted.get(b.id):
            if b.passed and lab.solution_correct is False:
                props.append(propose_verifier_gap(b, [f"auditor: passes but incorrect ({lab.rationale[:200]})"], source="label"))
            elif not b.passed and (lab.cause == "verifier_defect" or lab.solution_correct is True) and b.failing_tests:
                props.append(propose_false_fail(b, b.failing_tests[:20], f"auditor: {lab.cause}, correct={lab.solution_correct}: {lab.rationale[:200]}"))
        if not b.passed:
            props += propose_spec_defects(b)
        props += propose_world_bugs(b, fs_of[b.id])
    props = merge_proposals(props) + list(extra_proposals)
    first_bundle = {b.id: b for b in bundles}
    versions = []
    worlds_run = 0
    props.sort(key=lambda p: (p.kind != "world_bug", -len(p.traces)))  # the World bugs most traces hit run first under ``world_limit``
    for p in props:
        ctx = ctx_of(p)
        if p.kind == "world_bug" and world_limit is not None:
            if worlds_run >= world_limit:
                p.status = "needs_input"
                p.reasons.append(f"not reproduced: over this run's limit of {world_limit} World reproductions")
                continue
            worlds_run += 1
        await accept(p, ctx, alts=alts, second=second, config=cfg.accept, bundle=first_bundle.get(p.traces[0]) if p.traces else None, store=store)
        if p.status == "accepted" and ctx is not None:
            v = version_accepted(p, store, ctx)
            if v is not None:
                versions.append({"id": v.id, "subject": v.subject, "kind": v.kind, "parent": v.parent, "proposal": v.proposal_id,
                                 "evidence_hash": v.evidence_hash, "change": v.change})
    with (out / "proposals.jsonl").open("w") as fh:
        for p in props:
            fh.write(json.dumps(p.to_json(), default=str) + "\n")
    # -- intent drift --------------------------------------------------------------------------------------------------------------
    irows = []
    for b in bundles:
        if b.masked or b.passed or b.role == "hacker" or rows[b.id]["deterministic_cause"]:  # limits and infra are not behaviour
            continue
        lab = labels.get(b.id)
        cause = lab.cause if lab is not None and trusted.get(b.id) else None
        irows.append({"trace": b.id, "family": b.family, "signature": signature(b, fs_of[b.id], fws_of[b.id], cause), "on_intent": on_intent(b, fws_of[b.id], cause)})
    cands = intent_candidates(irows, intent=cfg.intent, min_traces=cfg.min_cluster_traces, min_families=cfg.min_cluster_families)
    write_intent_candidates(out / "intent_candidates.jsonl", cands)
    for c in cands:
        props.append(Proposal("intent_drift", str(cfg.intent.get("id")), c["traces"], {"type": "cluster", "signature": c["signature"], "families": c["families"]},
                              {"intent": "candidate"}, "cluster", c["why"], status="awaiting_human", reasons=["written to intent_candidates.jsonl"]))
    with (out / "bundles.jsonl").open("w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r, default=str) + "\n")
    # -- the report ----------------------------------------------------------------------------------------------------------------
    failing = [b for b in bundles if not b.masked and not b.passed]
    kinds = Counter(fws_of[b.id].kind for b in failing)
    by_fam: dict[str, Counter[str]] = {}
    for b in failing:
        by_fam.setdefault(b.family, Counter())[fws_of[b.id].kind] += 1
    det = Counter(str(rows[b.id]["deterministic_cause"]) for b in bundles if rows[b.id]["deterministic_cause"])
    lab_mix = Counter(str(lab.cause) for tid, lab in labels.items() if trusted.get(tid) and lab.cause and tid in rows and not rows[tid]["passed"])
    by_ks: dict[str, Counter[str]] = {}
    for p in props:
        by_ks.setdefault(p.kind, Counter())[p.status] += 1
    sessions = list(getattr(auditor, "sessions", []) or [])
    report = {
        "bundles": len(bundles), "by_run": dict(Counter(b.run for b in bundles)), "by_family": dict(Counter(b.family for b in bundles)),
        "masked": sum(b.masked for b in bundles), "passed": sum(b.passed for b in bundles), "failing": len(failing),
        "facts": dict(Counter(f.kind for b in bundles for f in fs_of[b.id])),
        "facts_episodes": {k: sum(1 for b in bundles if any(f.kind == k for f in fs_of[b.id])) for k in {f.kind for b in bundles for f in fs_of[b.id]}},
        "first_wrong_step": {"kinds": dict(kinds), "by_family": {k: dict(v) for k, v in by_fam.items()},
                             "position_quartiles": {k: _quartiles([fws_of[b.id].position for b in failing if fws_of[b.id].kind == k and fws_of[b.id].position is not None])
                                                    for k in kinds},
                             "turn_quartiles": _quartiles([float(fws_of[b.id].turn) for b in failing if fws_of[b.id].turn is not None])},
        "causes": {"deterministic": dict(det), "labelled_trusted": dict(lab_mix),
                   "skill_gap_share": round(lab_mix.get("skill_gap", 0) / max(1, sum(lab_mix.values())), 3) if lab_mix else None},
        "auditor": {"model": getattr(auditor, "model", None), "batches": batches, "refused_same_family": len(refused), "labelled": len(labels),
                    "trusted_labels": sum(1 for t in trusted.values() if t), "sessions": len(sessions),
                    "session_usd": round(sum(s.get("usd", 0.0) for s in sessions), 6),
                    "label_status": dict(Counter(lab.status for lab in labels.values()))},
        "pass_audit": {"passes": sum(b.passed for b in bundles), "candidates": sum(1 for r in rows.values() if r["pass_audit"]),
                       "reasons": dict(Counter(x.split(":", 1)[0] for r in rows.values() for x in r["pass_audit"]))},
        "proposals": {"total": len(props), "by_kind_status": {k: dict(v) for k, v in by_ks.items()},
                      "rows": [{"id": p.id, "kind": p.kind, "task": p.task, "status": p.status, "traces": len(p.traces), "why": p.why[:240],
                                "reasons": p.reasons[:3], "version": p.version} for p in props]},
        "versions": versions, "version_store_problems": store.verify(), "intent_candidates": cands,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


@contextlib.asynccontextmanager
async def _acm(cm: Any) -> Any:
    if hasattr(cm, "__aenter__"):
        async with cm as x:
            yield x
    else:
        with cm as x:
            yield x


# ---- the seeded scenario (P3's exit criterion) ------------------------------------------------------------------------------------
SEEDED_TASK = "seeded-durations"
SEEDED_START = {
    "durations.py": '"""Small helpers for durations."""\n\n\ndef seconds(minutes):\n    """Minutes as seconds."""\n    return int(minutes) * 60\n',
    "tests/test_existing.py": "from durations import seconds\n\n\ndef test_seconds():\n    assert seconds(2) == 120\n",
    # the planted World bug: -I drops the workspace from sys.path, so the World's own test tool cannot import the project
    "tools/run_tests.sh": "#!/bin/sh\nexec python -I -m pytest -q -p no:cacheprovider tests\n",
}
SEEDED_TOOL_FIX = {"files": {"tools/run_tests.sh": "#!/bin/sh\nexec python -m pytest -q -p no:cacheprovider tests\n"}}
SEEDED_HIDDEN = {
    "tests/test_parse.py": ("from durations import parse_duration\n\n\ndef test_parse_hours_minutes():\n    assert parse_duration('1h30m') == 5400\n\n\n"
                            "def test_parse_hours():\n    assert parse_duration('2h') == 7200\n"),
    "tests/test_format.py": ("from durations import format_duration, parse_duration\n\n\ndef test_format():\n    assert format_duration(5400) == '1h30m'\n"
                             "    assert format_duration(7200) == '2h'\n\n\ndef test_round_trip():\n    for s in ('1h30m', '2h'):\n"
                             "        assert format_duration(parse_duration(s)) == s\n"),
}
_SECONDS = 'def seconds(minutes):\n    """Minutes as seconds."""\n    return int(minutes) * 60\n'
_PARSE = ('_UNITS = {"h": 3600, "m": 60, "s": 1}\n_PART = re.compile(r"(\\d+)([hms])")\n\n\ndef parse_duration(text):\n    """\'1h30m\' -> 5400."""\n'
          '    text = text.strip()\n    parts = _PART.findall(text)\n    if not parts or "".join(n + u for n, u in parts) != text:\n'
          '        raise ValueError(f"not a duration: {text!r}")\n    return sum(int(n) * _UNITS[u] for n, u in parts)\n')
_FORMAT = ('def format_duration(total):\n    """5400 -> \'1h30m\'."""\n    h, rest = divmod(int(total), 3600)\n    m, s = divmod(rest, 60)\n'
           '    out = (f"{h}h" if h else "") + (f"{m}m" if m else "") + (f"{s}s" if s else "")\n    return out or "0s"\n')
SEEDED_ORACLE = {"durations.py": f'"""Small helpers for durations."""\n\nimport re\n\n\n{_SECONDS}\n\n{_PARSE}\n\n{_FORMAT}'}
SEEDED_ALT_OWN = {"durations.py": ('"""Small helpers for durations (a scanner)."""\n\n\n' + _SECONDS + '\n\ndef parse_duration(text):\n    total, num = 0, ""\n'
                                   '    for ch in text.strip():\n        if ch.isdigit():\n            num += ch\n        elif ch in "hms" and num:\n'
                                   '            total += int(num) * {"h": 3600, "m": 60, "s": 1}[ch]\n            num = ""\n        else:\n'
                                   '            raise ValueError(text)\n    if num or not text.strip():\n        raise ValueError(text)\n    return total\n\n\n'
                                   'def format_duration(total):\n    parts = []\n    for unit, size in (("h", 3600), ("m", 60), ("s", 1)):\n'
                                   '        n, total = divmod(total, size)\n        if n:\n            parts.append(f"{n}{unit}")\n    return "".join(parts) or "0s"\n')}
SEEDED_ALT_P2 = {"durations.py": ('"""Small helpers for durations (another family\'s rewrite)."""\n\nimport re\n\n\n' + _SECONDS + '\n\nUNIT = dict(h=3600, m=60, s=1)\n\n\n'
                                  'def parse_duration(text):\n    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text.strip())\n'
                                  '    if not m or not any(m.groups()):\n        raise ValueError(text)\n'
                                  '    return sum(int(g or 0) * u for g, u in zip(m.groups(), (3600, 60, 1)))\n\n\n'
                                  'def format_duration(total):\n    out = ""\n    for unit in "hms":\n        n, total = divmod(int(total), UNIT[unit])\n'
                                  '        out += f"{n}{unit}" if n else ""\n    return out or "0s"\n')}
SEEDED_LUCKY = {"durations.py": ('"""Small helpers for durations."""\n\n\n' + _SECONDS + '\n\n_KNOWN = {"1h30m": 5400, "2h": 7200}\n\n\n'
                                 'def parse_duration(text):\n    return _KNOWN[text]\n\n\ndef format_duration(total):\n'
                                 '    return {v: k for k, v in _KNOWN.items()}[total]\n')}
SEEDED_MUTANTS = {
    "revert_m1": {"durations.py": f'"""Small helpers for durations."""\n\nimport re\n\n\n{_SECONDS}\n\n{_FORMAT}'},
    "revert_m2": {"durations.py": f'"""Small helpers for durations."""\n\nimport re\n\n\n{_SECONDS}\n\n{_PARSE}'},
}
SEEDED_FLOORS = {
    "no_op": {},
    "pass_stub": {"durations.py": f'"""Small helpers for durations."""\n\n\n{_SECONDS}\n\ndef parse_duration(text):\n    pass\n\n\ndef format_duration(total):\n    pass\n'},
}
SEEDED_GOOD_CHECK = ProposedCheck(
    "p3_more_values", "tests/test_p3_more_values.py",
    ("from durations import format_duration, parse_duration\n\n\ndef test_parse_other_values():\n    assert parse_duration('45m') == 2700\n"
     "    assert parse_duration('1h5m') == 3900\n    assert parse_duration('10s') == 10\n\n\ndef test_format_other_values():\n"
     "    assert format_duration(2700) == '45m'\n    assert format_duration(3900) == '1h5m'\n"),
    ["tests/test_p3_more_values.py::test_parse_other_values", "tests/test_p3_more_values.py::test_format_other_values"], "m1", "seeded")
SEEDED_FAKE_CHECK = ProposedCheck(
    "p3_fake_minutes_only", "tests/test_p3_fake.py",
    "import pytest\n\nfrom durations import parse_duration\n\n\ndef test_minutes_only_rejected():\n    with pytest.raises(ValueError):\n        parse_duration('90m')\n",
    ["tests/test_p3_fake.py::test_minutes_only_rejected"], "m1", "seeded")
SEEDED_REQUIREMENTS = [
    {"id": "R1", "kind": "behaviour", "milestone": "m1", "text": "parse_duration turns a compact duration such as 1h30m or 2h into a number of seconds."},
    {"id": "R2", "kind": "behaviour", "milestone": "m2",
     "text": "A companion function renders a number of seconds back into the same compact form, so that parsing and rendering round-trip."},
    {"id": "r_regress", "kind": "side_effect_scope", "milestone": None, "text": "Everything that already works today must keep working."},
]


def seeded_prompt() -> str:
    from crucible.factory.lanes.windows import compose_window_prompt

    return compose_window_prompt("Durations in our configuration files are written like 1h30m. The next release needs to read them and write them back.",
                                 SEEDED_REQUIREMENTS[:2], SEEDED_REQUIREMENTS[2]["text"])


def seeded_rec() -> dict[str, Any]:
    """The seeded task as a window record (what the D8 helpers and the ledger read)."""
    m1 = ["tests/test_parse.py::test_parse_hours_minutes", "tests/test_parse.py::test_parse_hours"]
    m2 = ["tests/test_format.py::test_format", "tests/test_format.py::test_round_trip"]
    return {"id": SEEDED_TASK, "repo": "durations", "a": "a0", "b": "b0", "units": [], "tests_b": dict(SEEDED_HIDDEN),
            "milestones": [{"id": "m1", "tests": m1, "unit": 0, "requires": []}, {"id": "m2", "tests": m2, "unit": 1, "requires": ["m1"]}],
            "p2p": ["tests/test_existing.py::test_seconds"], "collateral": []}


def _junit(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return out
    for tc in root.iter("testcase"):
        f = tc.get("file") or (tc.get("classname") or "").replace(".", "/") + ".py"
        tid = f"{f}::{tc.get('name')}"
        st = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            st = "failed"
        elif tc.find("skipped") is not None:
            st = "skipped"
        out[tid] = st
    return out


class PytestRunner:
    """pytest and shell commands on a tree: in a ``crucible-p3-*`` container of ``image`` (``--network none``; the grader's
    and the World's sandbox), or locally (``image=None``: trusted seeded code only, unit tests)."""

    def __init__(self, *, image: str | None = None, timeout_s: float = 180.0, prefix: str = "crucible-p3-"):
        self.image, self.timeout_s, self.prefix = image, timeout_s, prefix

    def _env(self) -> dict[str, str]:
        return {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}

    async def _exec(self, tree: Path, argv: Sequence[str], role: str) -> tuple[int, str]:
        if self.image:
            name = f"{self.prefix}{role}-{uuid.uuid4().hex[:10]}"
            cmd = ["docker", "run", "--rm", "--network", "none", "--name", name, "--memory", "1g", "--cpus", "1", "-v", f"{tree}:/w", "-w", "/w",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", self.image, *argv]
            cwd, env = None, None
        else:
            cmd, cwd, env = list(argv), str(tree), self._env()
            if cmd[0] == "python":
                cmd[0] = sys.executable
        p = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            o, _ = await asyncio.wait_for(p.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                p.kill()
            if self.image:
                rm = await asyncio.create_subprocess_exec("docker", "rm", "-f", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await rm.wait()
            return -1, "timeout"
        return p.returncode or 0, o.decode(errors="replace")

    async def pytest(self, tree: Path, targets: Sequence[str]) -> tuple[dict[str, str] | None, str]:
        """(test id -> passed | failed | skipped, the output); None statuses: the run produced no report (masked)."""
        nonce = uuid.uuid4().hex[:12]
        rel = f".p3-junit-{nonce}.xml"
        rc, out = await self._exec(tree, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors", "-o", "junit_family=xunit1", f"--junitxml={rel}", *targets],
                                   "grade")
        res = _junit(tree / rel)
        (tree / rel).unlink(missing_ok=True)
        if not res and rc not in (0, 1, 2):
            return None, out
        if not res:  # a collection error: the targets that never ran failed
            return {}, out
        return res, out

    async def shell(self, tree: Path, command: str) -> tuple[int, str]:
        return await self._exec(tree, ["sh", "-c", command], "world")


def _git(cwd: Path, *args: str, data: str | None = None) -> str:
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(cwd), input=data, capture_output=True, text=True, check=False,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])}: {r.stderr.strip()[-300:]}")
    return r.stdout


def make_patch(base: Path, files: Mapping[str, str | None]) -> str:
    """A git diff from the tree at ``base`` to the tree with ``files`` written (None deletes)."""
    with tempfile.TemporaryDirectory(prefix="p3-patch-") as d:
        t = Path(d) / "t"
        shutil.copytree(base, t)
        _git(t, "init", "-q")
        _git(t, "add", "-A")
        _git(t, "-c", "user.email=p3@x", "-c", "user.name=p3", "commit", "-qm", "base")
        for p, txt in files.items():
            f = t / p
            if txt is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(txt)
        _git(t, "add", "-A")
        return _git(t, "diff", "--cached", "--binary")


class SeededTask:
    """The seeded task's verifier (a ``VerifierPort``) with its executed ground truth. A solution is ``{"files": {...}}``
    or a git diff against the start tree; test files in a solution are never written (protected), the hidden tests are."""

    task_id = SEEDED_TASK

    def __init__(self, root: Path, runner: PytestRunner):
        self.root, self.runner = Path(root), runner
        self.base = self.root / "base"
        if not self.base.exists():
            for p, txt in SEEDED_START.items():
                (self.base / p).parent.mkdir(parents=True, exist_ok=True)
                (self.base / p).write_text(txt)
        self.rec = seeded_rec()
        self.milestones = self.rec["milestones"]
        self.regress = list(self.rec["p2p"])
        self.grades = 0

    def files_of(self, solution: Any) -> dict[str, str | None]:
        if isinstance(solution, Mapping):
            return dict(solution.get("files") or {})
        if not str(solution or "").strip():
            return {}
        with tempfile.TemporaryDirectory(prefix="p3-apply-") as d:
            t = Path(d) / "t"
            shutil.copytree(self.base, t)
            _git(t, "apply", "--whitespace=nowarn", "-", data=str(solution))
            out: dict[str, str | None] = {}
            for st, p in diff_entries(str(solution)):
                out[p] = None if st == "D" else (t / p).read_text()
            return out

    def tree(self, dest: Path, solution: Any, extra: Mapping[str, str] = (), *, world_fix: Mapping[str, Any] | None = None) -> Path:
        from crucible.factory.lanes.executable import is_protected

        shutil.copytree(self.base, dest)
        for p, txt in ((world_fix or {}).get("files") or {}).items():
            (dest / p).write_text(txt)
        for p, txt in self.files_of(solution).items():
            if is_protected(p):
                continue
            if txt is None:
                (dest / p).unlink(missing_ok=True)
            else:
                (dest / p).parent.mkdir(parents=True, exist_ok=True)
                (dest / p).write_text(txt)
        for p, txt in dict(extra).items():
            (dest / p).parent.mkdir(parents=True, exist_ok=True)
            (dest / p).write_text(txt)
        return dest

    async def run(self, solution: Any, extra: Mapping[str, str], targets: Sequence[str]) -> tuple[dict[str, str] | None, str]:
        self.grades += 1
        with tempfile.TemporaryDirectory(prefix="p3-grade-") as d:
            t = self.tree(Path(d) / "t", solution, extra)
            return await self.runner.pytest(t, list(targets))

    def phi(self, ms: Mapping[str, bool]) -> float:
        w = 1.0 / len(self.milestones)
        return round(sum(w for m in self.milestones if ms.get(m["id"]) and all(ms.get(r) for r in m["requires"])), 6)

    async def verdict(self, solution: Any, *, add: Sequence[ProposedCheck] = (), drop: Sequence[str] = ()) -> Verdict:
        extra = {**SEEDED_HIDDEN, **{c.file: c.source for c in add}}
        res, out = await self.run(solution, extra, sorted({*SEEDED_HIDDEN, "tests/test_existing.py", *(c.file for c in add)}))
        if res is None:
            return Verdict(None, detail={"tail": out[-2500:]})
        dropped = set(drop)
        ms = {m["id"]: all(res.get(t) == "passed" for t in m["tests"] if t not in dropped) for m in self.milestones}
        reg = all(res.get(t) == "passed" for t in self.regress if t not in dropped)
        added = all(res.get(t) == "passed" for c in add for t in c.tests)
        failing = sorted(t for m in self.milestones for t in m["tests"] if t not in dropped and res.get(t) != "passed")
        failing += [t for t in self.regress if t not in dropped and res.get(t) != "passed"] + [t for c in add for t in c.tests if res.get(t) != "passed"]
        return Verdict(all(ms.values()) and reg and added, ms, failing, {"tail": out[-2500:], "phi": self.phi(ms)})

    async def check(self, check: ProposedCheck, solution: Any) -> bool | None:
        res, _ = await self.run(solution, {check.file: check.source}, [check.file])
        if res is None:
            return None
        return all(res.get(t) == "passed" for t in check.tests)

    def oracle(self) -> Any:
        return {"files": dict(SEEDED_ORACLE)}

    def alternatives(self) -> list[tuple[str, Any]]:
        return [("alt_own_scanner", {"files": dict(SEEDED_ALT_OWN)})]

    def mutants(self) -> list[tuple[str, Any]]:
        return [(k, {"files": dict(v)}) for k, v in SEEDED_MUTANTS.items()]

    def floors(self) -> list[tuple[str, Any]]:
        return [(k, {"files": dict(v)}) for k, v in SEEDED_FLOORS.items()]

    def content(self) -> dict[str, Any]:
        return {"checks": {f"c_{m['id']}": list(m["tests"]) for m in self.milestones} | {"c_regress": list(self.regress)},
                "hidden_sha": sha(SEEDED_HIDDEN)}


class SeededWorld:
    """The seeded task's World (a ``WorldPort``): its tools run on the start tree plus a state, optionally with a World fix."""

    def __init__(self, task: SeededTask, runner: PytestRunner, *, s3_reruns: int = 5):
        self.task, self.runner, self.s3_reruns = task, runner, s3_reruns

    async def reproduce(self, repro: Mapping[str, Any], *, state: Any, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = repro.get("arguments") or {}
        cmd = args.get("command") if isinstance(args, Mapping) else None
        if not cmd:
            return {"failed": False, "exit": None, "output": "the reproduction names no command"}
        with tempfile.TemporaryDirectory(prefix="p3-world-") as d:
            t = self.task.tree(Path(d) / "t", state, world_fix=fix)
            rc, out = await self.runner.shell(t, str(cmd))
        return {"failed": rc != 0, "exit": rc, "output": out[-1500:]}

    async def s3(self, *, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """S3 again: the World builds, its existing tests run clean through its own tool ``s3_reruns`` times, and the leak
        probes hold (no hidden test file, no oracle-only definition in the World)."""
        checks: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="p3-s3-") as d:
            t = self.task.tree(Path(d) / "t", {"files": {}}, world_fix=fix)
            rc, out = await self.runner.shell(t, "python -c 'import durations'")
            checks["builds"] = rc == 0
            runs = [await self.runner.shell(t, "sh tools/run_tests.sh") for _ in range(self.s3_reruns)]
            checks["existing_tests_clean"] = f"{sum(1 for r, _ in runs if r == 0)}/{len(runs)}"
            world_text = "\n".join(p.read_text(errors="replace") for p in t.rglob("*") if p.is_file())
            checks["leak_hidden_tests"] = not any((t / p).exists() for p in SEEDED_HIDDEN)
            checks["leak_oracle_defs"] = not re.search(r"def (parse_duration|format_duration)\(", world_text)
        ok = checks["builds"] and checks["existing_tests_clean"] == f"{self.s3_reruns}/{self.s3_reruns}" and checks["leak_hidden_tests"] and checks["leak_oracle_defs"]
        why = "" if ok else ("existing tests do not run clean through the World's tool" if checks["builds"] else "the World does not build")
        return {"ok": bool(ok), "checks": checks, "why": why, "tail": out[-300:]}

    def content(self) -> dict[str, Any]:
        return {"tools": {p: sha(t) for p, t in SEEDED_START.items() if p.startswith("tools/")}}


@dataclass
class ScriptedStep:
    function: str
    arguments: dict[str, Any]
    write: dict[str, str] | None = None  # files this step writes (edit/write tools)
    text: str = ""  # the model's text in the turn that issued it


class SeededScenario:
    """Builds the seeded task, writes scripted episodes in the runner's on-disk layout (every grade and every shell result
    executed), and the task context the loop needs. ``runner``: ``PytestRunner()`` (local) or ``PytestRunner(image=...)``."""

    def __init__(self, root: Path, runner: PytestRunner):
        self.root = Path(root)
        self.runner = runner
        self.task = SeededTask(self.root / "task", runner)
        self.world = SeededWorld(self.task, runner)
        self.prompt = seeded_prompt()

    async def check_author(self, p: Proposal, b: EpisodeBundle) -> ProposedCheck | None:
        return SEEDED_GOOD_CHECK  # a scripted check writer (it saw the demonstrating end state, never the oracle)

    async def world_fixer(self, p: Proposal) -> dict[str, Any] | None:
        return dict(SEEDED_TOOL_FIX) if "run_tests.sh" in json.dumps(p.artifact.get("arguments"), default=str) else None

    def context(self) -> TaskContext:
        return TaskContext(SEEDED_TASK, verifier=self.task, world=self.world, rec=self.task.rec, base_dir=self.task.base, prompt=self.prompt,
                           requirements=[dict(r) for r in SEEDED_REQUIREMENTS], oracle_patch=make_patch(self.task.base, SEEDED_ORACLE),
                           oracle_files=sorted(SEEDED_ORACLE), check_author=self.check_author, world_fixer=self.world_fixer)

    def alternatives(self) -> FakeAlternatives:
        """P2's alternatives faked: another family's rewrite."""
        return FakeAlternatives({SEEDED_TASK: [("alt_p2_regex", {"files": dict(SEEDED_ALT_P2)})]})

    def fake_proposal(self, demo: EpisodeBundle) -> Proposal:
        return Proposal("verifier_gap", SEEDED_TASK, [demo.id], end_state_artifact(demo), {"add_check": asdict(SEEDED_FAKE_CHECK)}, "seeded",
                        "a planted fake: a check that also fails the oracle")

    def scripts(self) -> dict[str, dict[str, Any]]:
        read = ScriptedStep("repo.read_file", {"path": "durations.py"}, text="Let me look at the module.")
        run = ScriptedStep("repo.shell", {"command": "sh tools/run_tests.sh"}, text="Run the tests.")
        good_steps = [read, ScriptedStep("repo.read_file", {"path": "tests/test_existing.py"}),
                      ScriptedStep("repo.search", {"pattern": "def seconds"}),
                      ScriptedStep("repo.write_file", {"path": "durations.py"}, SEEDED_ORACLE, "Implement parse_duration and format_duration."),
                      ScriptedStep("repo.shell", {"command": "python -c \"import durations; print(durations.parse_duration('1h30m'))\""}),
                      ScriptedStep("repo.shell", {"command": "python -c \"import durations; print(durations.format_duration(5400))\""}),
                      ScriptedStep("repo.shell", {"command": "python -m pytest -q tests"}),
                      ScriptedStep("repo.write_file", {"path": "STATUS.json"}, {"STATUS.json": '{"R1": 0.9, "R2": 0.9}'})]
        alt_steps = [*good_steps[:3], ScriptedStep("repo.write_file", {"path": "durations.py"}, SEEDED_ALT_OWN, "A scanner-based parser."), *good_steps[4:]]
        parse_only = {"durations.py": SEEDED_MUTANTS["revert_m2"]["durations.py"]}
        compact = {"durations.py": parse_only["durations.py"] + "\n\ndef to_compact(total):\n" + _FORMAT.split("\n", 1)[1]}
        buggy = {"durations.py": parse_only["durations.py"] + "\n\ndef format_duration(total):\n    h, rest = divmod(int(total), 3600)\n"
                 "    m, s = divmod(rest, 60)\n    return f\"{h}h{m}m{s}s\"\n"}
        broken = {"durations.py": f'"""Small helpers for durations."""\n\n\n{_SECONDS}\n\ndef parse_duration(text):\n    return int(text)\n\n\n'
                  'def format_duration(total):\n    return str(total)\n'}
        return {
            "lucky": {"policy": "solver:qwen/qwen3.7-flash", "steps": [read, ScriptedStep("repo.write_file", {"path": "durations.py"}, SEEDED_LUCKY, "Done quickly.")],
                      "final": "Implemented parse_duration and format_duration.", "status": {"R1": 0.9, "R2": 0.9}},
            "good_a": {"policy": "solver:openai/gpt-oss-20b", "steps": good_steps, "final": "Both functions are implemented and the tests pass.",
                       "status": {"R1": 0.9, "R2": 0.9}},
            "good_b": {"policy": "solver:meta-llama/llama-3.1-8b-instruct", "steps": alt_steps, "final": "Implemented both.", "status": {"R1": 0.8, "R2": 0.8}},
            "unstated": {"policy": "solver:qwen/qwen3.7-flash",
                         "steps": [read, ScriptedStep("repo.write_file", {"path": "durations.py"}, parse_only, "parse_duration first."),
                                   ScriptedStep("repo.shell", {"command": "python -m pytest -q tests"}),
                                   ScriptedStep("repo.write_file", {"path": "durations.py"}, compact, "The companion renderer: I call it to_compact."),
                                   ScriptedStep("repo.shell", {"command": "python -c \"import durations; print(durations.to_compact(5400))\""})],
                         "final": "All requirements are implemented: parse_duration and to_compact.", "status": {"R1": 0.9, "R2": 0.9}},
            "broken_tool": {"policy": "solver:meta-llama/llama-3.1-8b-instruct",
                            "steps": [read, run, ScriptedStep("repo.write_file", {"path": "durations.py"}, parse_only, "Add the parser."), run,
                                      ScriptedStep("repo.write_file", {"path": "durations.py"}, buggy, "And the renderer."), run],
                            "final": "I could not run the tests: the test tool cannot import the module.", "status": {"R1": 0.7, "R2": 0.5}},
            "lost": {"policy": "solver:amazon/nova-micro-v1",
                     "steps": [read, ScriptedStep("repo.write_file", {"path": "durations.py"}, parse_only, "Parser."),
                               *[ScriptedStep("repo.read_file", {"path": "durations.py"}) for _ in range(3)],
                               ScriptedStep("repo.write_file", {"path": "durations.py"}, broken, "Simplify everything."),
                               ScriptedStep("repo.read_file", {"path": "durations.py"})],
                     "final": "Simplified the module.", "status": {"R1": 0.6, "R2": 0.6}},
        }

    async def write_episodes(self, out: Path, *, every: int = 2) -> list[Path]:
        """Every scripted episode as ``<out>/runs/seeded/samples/<name>/a1``: trace, calls, trajectory, checkpoints (graded
        every ``every`` tool calls and at the end), the final patch and its GradeReport."""
        dirs = []
        for name, sc in self.scripts().items():
            dirs.append(await self._episode(Path(out) / "runs" / "seeded" / "samples" / name / "a1", name, sc, every=every))
        return dirs

    async def _episode(self, d: Path, name: str, sc: Mapping[str, Any], *, every: int) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints").mkdir(exist_ok=True)
        files: dict[str, str] = {}
        events: list[dict[str, Any]] = [{"seq": 0, "kind": "state", "type": "reset", "actor": None, "data": {}}]
        calls, traj = [], [{"step_id": 1, "source": "user", "message": self.prompt}]
        cps, prog = [], [{"step": 0, "phi": 0.0, "delta_phi": 0.0, "gained": [], "lost": []}]
        prev_ms: dict[str, bool] = {}
        prev_phi = 0.0
        steps: list[ScriptedStep] = sc["steps"]
        for i, st in enumerate(steps, start=1):
            tid = f"{name[:3]}{i:03d}"
            events.append({"seq": len(events), "kind": "action", "type": "tool_call", "actor": "agent", "data": {"function": st.function, "arguments": st.arguments, "id": tid}})
            if st.write:
                files.update(st.write)
                result, ok = f"wrote {st.arguments.get('path')}", True
            elif st.function == "repo.shell":
                with tempfile.TemporaryDirectory(prefix="p3-ep-") as td:
                    t = self.task.tree(Path(td) / "t", {"files": dict(files)})
                    rc, out = await self.runner.shell(t, st.arguments["command"])
                result, ok = f"[exit {rc}]\n{out[-2500:]}", True
            elif st.function == "repo.read_file":
                p = st.arguments["path"]
                result, ok = (files.get(p) or SEEDED_START.get(p) or f"FileNotFoundError: {p}"), True
            else:
                result, ok = "durations.py:4: def seconds(minutes):", True
            events.append({"seq": len(events), "kind": "observation", "type": "tool_result", "actor": "agent", "text": result,
                           "data": {"function": st.function, "ok": ok, "error": None, "id": tid, "fault": None}})
            calls.append({"call_id": f"agent:c{i}", "kind": "turn", "actions": [f"tool:{tid}"], "finish": "tool_calls", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                          "cost_usd": 0.0})
            traj.append({"step_id": len(traj) + 1, "source": "agent", "message": st.text, "tool_calls": [{"function_name": st.function, "arguments": st.arguments}]})
            row: dict[str, Any] = {"step": i, "phi": prev_phi, "delta_phi": 0.0, "gained": [], "lost": [], "tool_call_id": tid, "actor": "agent"}
            if i % every == 0 or i == len(steps):
                v = await self.task.verdict({"files": dict(files)})
                patch = make_patch(self.task.base, files)
                psha = hashlib.sha256(patch.encode()).hexdigest()
                (d / "checkpoints" / f"{psha}.patch").write_text(patch)
                phi = float(v.detail.get("phi", 0.0)) if v.passed is not None else prev_phi
                gained = sorted(m for m, ok_ in v.milestones.items() if ok_ and not prev_ms.get(m))
                lost = sorted(m for m, ok_ in v.milestones.items() if not ok_ and prev_ms.get(m))
                cps.append({"index": len(cps), "turn": i, "triggers": ["end"] if i == len(steps) else ["periodic"], "patch_sha256": psha,
                            "patch_bytes": len(patch), "state": "done", "milestones": v.milestones, "phi": phi, "measured": v.passed is not None,
                            "gained": gained, "lost": lost})
                row.update(phi=phi, delta_phi=round(phi - prev_phi, 6), gained=gained, lost=lost, checkpoint=cps[-1]["triggers"], measured=True)
                prev_ms, prev_phi = dict(v.milestones), phi
            prog.append(row)
        calls.append({"call_id": f"agent:c{len(steps) + 1}", "kind": "turn", "actions": ["done"], "finish": "stop", "usage": {}, "cost_usd": 0.0})
        traj.append({"step_id": len(traj) + 1, "source": "agent", "message": sc["final"]})
        final = await self.task.verdict({"files": dict(files)})
        patch = make_patch(self.task.base, files)
        (d / "artifacts").mkdir(exist_ok=True)
        (d / "artifacts" / "model.patch").write_text(patch)
        (d / "verifier").mkdir(exist_ok=True)
        reward = 1.0 if final.passed else round(0.5 * float(final.detail.get("phi", 0.0)), 6)
        (d / "verifier" / "report.json").write_text(json.dumps({"task": f"{SEEDED_TASK}~seed", "reward": reward, "meta": {
            "milestones": final.milestones, "phi": final.detail.get("phi"), "failing_tests": final.failing, "tail": final.detail.get("tail", ""),
            "full_pass": bool(final.passed), "grader_paths": sorted(SEEDED_HIDDEN)}}, indent=1))
        scored = {r: [p, None] for r, p in sc["status"].items()}
        (d / "record.json").write_text(json.dumps({"task": f"{SEEDED_TASK}~seed", "policy": sc["policy"], "reward": reward, "mask_sample": False,
                                                    "steps": len(steps), "cost_usd": 0.0, "meta": {"stop": "agent_done", "progress": {
                                                        "phi0": 0.0, "honesty": {"scored": scored, "defaulted": [], "claimed": True}}}}, indent=1))
        (d / "trace.json").write_text(json.dumps({"schema_version": "crucible.trace/v1", "events": events}))
        (d / "calls.json").write_text(json.dumps({"agent": calls}))
        (d / "trajectory.agent.json").write_text(json.dumps({"schema_version": "ATIF-v1.8", "steps": traj}))
        (d / "checkpoints.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cps))
        (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prog))
        return d


async def run_seeded(root: Path, *, runner: PytestRunner, auditor: Auditor | None = None, seeded_controls: Sequence[SeededCase] = (),
                     config: LoopConfig | None = None, spend: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """P3's exit criterion end to end: the seeded episodes, the loop with the seeded task's context, the fake proposal.
    Returns the loop report plus ``seeded``: the expected outcome of each planted case and whether it held."""
    sc = SeededScenario(Path(root), runner)
    dirs = await sc.write_episodes(Path(root) / "episodes")
    bundles = [load_bundle(d, run="seeded", modules=["durations"]) for d in dirs]
    lucky = next(b for b in bundles if b.id.endswith("lucky/a1"))
    cfg = config or LoopConfig(intent={"id": "I-seeded", "behaviour": "Durations parse and render"}, min_cluster_traces=3, min_cluster_families=2)
    store = VersionStore(Path(root) / "versions.jsonl")
    rep = await run_loop(bundles, out=Path(root) / "loop", store=store, contexts={SEEDED_TASK: sc.context()}, auditor=auditor, seeded=seeded_controls,
                         alts=sc.alternatives(), config=cfg, extra_proposals=[sc.fake_proposal(lucky)], spend=spend)
    rows = [json.loads(x) for x in (Path(root) / "loop" / "proposals.jsonl").read_text().splitlines()]

    def find(kind: str, pred: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        return next((r for r in rows if r["kind"] == kind and pred(r)), None)

    gap = find("verifier_gap", lambda r: r["source"] == "pass_audit" and r["traces"] == [lucky.id])
    fake = find("verifier_gap", lambda r: r["source"] == "seeded")
    spec = find("spec_defect", lambda r: "format_duration" in json.dumps(r["change"]))
    world = find("world_bug", lambda r: "run_tests.sh" in json.dumps(r["artifact"]))
    cases = {
        "verifier_gap(lucky pass)": {"expected": "accepted", "got": gap and gap["status"], "version": gap and gap["version"], "reasons": gap and gap["reasons"]},
        "spec_defect(unstated name)": {"expected": "accepted", "got": spec and spec["status"], "version": spec and spec["version"], "reasons": spec and spec["reasons"]},
        "world_bug(broken tool)": {"expected": "accepted", "got": world and world["status"], "version": world and world["version"], "reasons": world and world["reasons"]},
        "fake proposal(check fails the oracle)": {"expected": "rejected", "got": fake and fake["status"], "version": fake and fake["version"],
                                                   "reasons": fake and fake["reasons"]},
    }
    for c in cases.values():
        c["ok"] = c["got"] == c["expected"] and (c["version"] is not None) == (c["expected"] == "accepted")
    rep["seeded"] = cases
    rep["seeded_ok"] = all(c["ok"] for c in cases.values())
    rep["lineage"] = {k: [v.id for v in store.lineage(c["version"])] for k, c in cases.items() if c["version"]}
    (Path(root) / "loop" / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
