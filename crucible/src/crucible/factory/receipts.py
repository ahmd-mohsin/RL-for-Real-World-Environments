"""Receipts: verifying the factory itself (P1; docs/research/swarm-pipeline/00 §13, 01 §4.2 and fact 9).

Prime Agent verifies its factory by running the real daemon against a scripted model and writing a JSON receipt with
binary hashes and an explicit ``not_verified`` list; each receipt script has a ``test_*`` twin that mutates the probe and
requires the checker to reject it (``scripts/review_factory_resumption_parity.py``, ``test_review_factory_resumption_
parity.py:13-41``, Prime Agent 7cbffa77b, MIT: PATTERN, nothing copied). Here:

- **Every run writes a receipt** (``run_receipt``): versions (the crucible package and source commit, Python, the
  machine's hash), image digests, seeds, spend (USD, tokens, seconds), per-state outcomes read from the run's journal
  (entries, attempts, settles, task and infra failures, free respawns, recoveries by mode, capture failures, snapshots),
  the final status and decision, the ``not_verified`` list, and a sha256 over all of it.
- **The seeded-defect run** (``run_seeded_defects``) plants one defect per case through fakes and records where the
  machine caught it. Task defects (§13): an unsolvable task (caught at S5, clean-World solvability), a reachable grader
  (S12, the hacker's write scores), a leaky World (S3, leak probes), a hinting prompt (S7, the hint lint), a flaky test
  (S10, ``flake.measure_flake``'s Clopper-Pearson bound), a mislabelled cause (S14, ``audit.audit``'s seeded controls).
  Machine-layer defects (P1): a dead worker (one free respawn, no attempt spent), a missing capture (the producer
  fails), a capture that vanished before its consumer bound it (the producer is re-run), an unbounded review loop
  (escalates, never ``done``), a mutating state with no recovery posture (refused at load), a state over its token
  budget (paused, no retry). A clean control task must pass every gate, so no check fires on its own.
- **The checker is mutation-tested** (``check_receipt``; the twin is ``tests/unit/test_m17_p1_receipts.py``): a wrong
  catching state, a defect that reached ``done``, a missing clean control, an edited row, an empty ``not_verified``
  each make it fail.

The checks inside the task-defect states are small deterministic detectors over fake tasks: they prove the **machine
layer** routes each defect to its designated state and records the evidence. They are not the production gates (the
window lane's leak probes, the floors, the hacker), which P2-P5 wire into the same states.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.machine import InfraError, MachineError, MachineRun, Settle, StateContext, artifact_digest, book_tokens

RECEIPT_VERSION = "p1-1"
TASK_DEFECTS: dict[str, str] = {  # defect -> the state that must catch it (00 §4 / §13)
    "leaky_world": "s3_world",
    "unsolvable_task": "s5_verifier",
    "hinting_prompt": "s7_prompt",
    "flaky_test": "s10_flake",
    "reachable_grader": "s12_hack",
    "mislabelled_cause": "s14_audit",
}
NOT_VERIFIED_SEEDED = [
    "production gates in these states (the window lane's leak probes, floors, hacker swarm, auditor model): fakes stand in; P2-P5 wire them",
    "agentic sessions (factory/sessions.py) in this receipt: they have their own Docker end-to-end test (test_m17_p1_sessions)",
    "multi-process concurrency of the task repo and of the machine journal",
    "real provider outages: an InfraError stands in for one",
]


def sha256_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# ---- the receipt of any run ---------------------------------------------------------------------------------------------
def versions(machine_hash: str | None = None) -> dict[str, Any]:
    import crucible

    src = Path(crucible.__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=60)
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "."], capture_output=True, text=True, check=False, timeout=60)
    try:
        from importlib.metadata import version

        pkg = version("crucible")
    except Exception:  # noqa: BLE001 -- an uninstalled source tree
        pkg = getattr(crucible, "__version__", None)
    return {"receipt": RECEIPT_VERSION, "crucible": pkg, "source_commit": head.stdout.strip() or None, "source_dirty": bool(dirty.stdout.strip()),
            "python": platform.python_version(), "platform": platform.platform(), **({"machine_hash": machine_hash} if machine_hash else {})}


def image_digests(tags: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for t in tags:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", t], capture_output=True, text=True, check=False, timeout=60)
        out[t] = r.stdout.strip() if r.returncode == 0 else None
    return out


def journal_outcomes(journal: Path) -> dict[str, Any]:
    """Per-state outcomes and the final status of a machine run, from its journal (the run's own record)."""
    states: dict[str, dict[str, Any]] = {}
    final: dict[str, Any] = {}

    def row(sid: str) -> dict[str, Any]:
        return states.setdefault(sid, {"entries": 0, "attempts": 0, "settles": 0, "task_fails": 0, "infra_fails": 0, "infra_respawns": 0,
                                       "recoveries": {"reset": 0, "reconcile": 0}, "capture_failed": 0, "snapshots": 0, "tokens": 0, "usd": 0.0})

    for line in Path(journal).read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        k, sid = ev.get("event"), ev.get("state")
        if k == "enter":
            row(sid)["entries"] = max(row(sid)["entries"], ev["entry"])
        elif k == "spend":
            r = row(sid)
            r["attempts"] += 0 if ev.get("respawn") else 1
            r["tokens"] += int(ev.get("tokens", 0))
            r["usd"] = round(r["usd"] + float(ev.get("usd", 0.0)), 6)
        elif k == "settle":
            row(sid)["settles"] += 1
        elif k == "fail":
            row(sid)["infra_fails" if ev.get("blame") == "infra" else "task_fails"] += 1
        elif k == "infra_respawn":
            row(sid)["infra_respawns"] += 1
        elif k == "recovery":
            row(sid)["recoveries"][ev["mode"]] += 1
        elif k == "capture_failed":
            row(sid)["capture_failed"] += 1
        elif k == "snapshot":
            row(sid)["snapshots"] += 1
        elif k == "transition":
            final["last_transition"] = [ev["from"], ev["to"]]
        elif k == "status":
            final["status"], final["reason"] = ev["status"], ev.get("reason")
            req = ev.get("decision_request") or {}
            final["decision"] = {"kind": req.get("kind"), "state": req.get("state")} if req else None
    return {"states": states, **final}


def make_receipt(*, scope: str, runs: Mapping[str, Mapping[str, Any]], not_verified: Sequence[str], seeds: Mapping[str, Any] | None = None,
                 images: Mapping[str, str | None] | None = None, spend: Mapping[str, Any] | None = None, machine_hash: str | None = None,
                 cases: Sequence[Mapping[str, Any]] = (), verified: bool | None = None) -> dict[str, Any]:
    body = {"scope": scope, "versions": versions(machine_hash), "images": dict(images or {}), "seeds": dict(seeds or {}),
            "spend": dict(spend or {}), "runs": {k: dict(v) for k, v in runs.items()}, "cases": [dict(c) for c in cases],
            "not_verified": list(not_verified)}
    if verified is None:
        verified = all(c.get("ok") for c in cases) if cases else True
    body["verified"] = bool(verified)
    return {**body, "sha256": sha256_json(body)}


def run_receipt(run: MachineRun, *, scope: str, not_verified: Sequence[str], seeds: Mapping[str, Any] | None = None,
                images: Sequence[str] = (), plan_usage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The receipt of one machine run (00 §13: versions, image digests, seeds, spend, state outcomes, not_verified).
    ``plan_usage``: the Claude plan accounting of the run's builder clients (``ModelBook.plan_usage()``; swarm-pipeline/08),
    kept under ``spend.plan``: the plan bills no per-call dollars, so it is reported beside ``spend.usd``, never in it."""
    out = journal_outcomes(run.journal.path)
    r = run.result()
    spend: dict[str, Any] = {"usd": r.usd, "tokens": r.tokens, "seconds": r.seconds}
    if plan_usage:
        spend["plan"] = dict(plan_usage)
    return make_receipt(scope=scope, runs={run.run_id: out}, not_verified=not_verified, seeds=seeds, images=image_digests(images) if images else {},
                        spend=spend, machine_hash=run.hash)


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=1) + "\n")
    return path


def check_receipt(receipt: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    """Every problem with a receipt (empty: it holds). ``expected``: case -> fields its observed outcome must equal.
    Checks the hash, the required sections, a non-empty ``not_verified``, each case against its designation and
    its evidence hash, and (for a seeded run) that the clean control passed every gate."""
    out: list[str] = []
    body = {k: v for k, v in receipt.items() if k != "sha256"}
    if receipt.get("sha256") != sha256_json(body):
        out.append("the receipt's sha256 does not match its content (edited after writing)")
    for key in ("scope", "versions", "runs", "not_verified", "verified"):
        if key not in receipt:
            out.append(f"the receipt has no {key!r}")
    nv = receipt.get("not_verified")
    if not isinstance(nv, list) or not nv or not all(isinstance(x, str) and x.strip() for x in nv):
        out.append("not_verified must be a non-empty list of statements (a receipt that claims everything is a defect)")
    cases = {c.get("case"): c for c in receipt.get("cases", []) if isinstance(c, dict)}
    for name, c in cases.items():
        if c.get("evidence_sha256") != sha256_json(c.get("evidence")):
            out.append(f"case {name}: its evidence does not hash to evidence_sha256")
        if c.get("observed") != c.get("designated"):
            out.append(f"case {name}: designated {c.get('designated')}, observed {c.get('observed')}")
        if not c.get("ok"):
            out.append(f"case {name}: not ok")
    for name, want in (expected or {}).items():
        c = cases.get(name)
        if c is None:
            out.append(f"case {name}: missing from the receipt")
            continue
        for k, v in want.items():
            if (c.get("designated") or {}).get(k) != v:
                out.append(f"case {name}: designated {k} is {(c.get('designated') or {}).get(k)!r}, the release expects {v!r}")
    if receipt.get("verified") is not True:
        out.append("the receipt is not verified")
    elif cases and not all(c.get("ok") for c in cases.values()):
        out.append("the receipt says verified but a case is not ok")
    return out


def validate(receipt: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """``check_receipt`` as an assertion (Prime's ``validate`` shape)."""
    problems = check_receipt(receipt, expected)
    if problems:
        raise AssertionError("; ".join(problems))


# ---- the seeded-defect run ----------------------------------------------------------------------------------------------
@dataclass
class FakeTask:
    """A task reduced to what the six detectors read. ``clean()`` passes every gate; each defect changes one field."""

    world_files: dict[str, str] = field(default_factory=lambda: {"src/calc.py": "def add(a, b):\n    return a + b\n"})
    world_commits: int = 1
    hidden_test_names: tuple[str, ...] = ("test_mul_small", "test_mul_negative")
    hidden_literals: tuple[str, ...] = ("mul(2, 3) == 6", "mul(-2, 3) == -6")
    prompt: str = "Add a mul(a, b) function to calc.py next to add; it multiplies two integers."
    oracle_checks: dict[str, bool] = field(default_factory=lambda: {"test_mul_small": True, "test_mul_negative": True})
    flake_every: int = 0  # >0: every n-th oracle grade fails (a flaky test)
    grader_reads_workspace: bool = False  # a reachable grader: it trusts a results file the agent can write
    auditor_mislabels: bool = False

    @classmethod
    def with_defect(cls, defect: str | None) -> FakeTask:
        t = cls()
        if defect == "leaky_world":
            t.world_files["tests/test_future.py"] = "def test_mul_small():\n    assert mul(2, 3) == 6\n"
        elif defect == "unsolvable_task":
            t.oracle_checks["test_mul_negative"] = False
        elif defect == "hinting_prompt":
            t.prompt += " Make sure mul(-2, 3) == -6."
        elif defect == "flaky_test":
            t.flake_every = 5
        elif defect == "reachable_grader":
            t.grader_reads_workspace = True
        elif defect == "mislabelled_cause":
            t.auditor_mislabels = True
        elif defect is not None:
            raise ValueError(f"unknown defect {defect!r}")
        return t


def _verdict(ok: bool, evidence: Mapping[str, Any]) -> Settle:
    return Settle({"verdict": {"ok": ok, "evidence": json.loads(json.dumps(evidence, default=str))}})


async def _s3_world(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    hits = sorted(p for p, body in t.world_files.items() for n in t.hidden_test_names if f"def {n}(" in body)
    probes = {"no_new_test_names_in_world": not hits, "git_log_only_base": t.world_commits == 1}
    return _verdict(all(probes.values()), {"probes": probes, "hits": hits})


async def _s5_verifier(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    from crucible.factory.floors import Graded

    g = Graded(1.0 if all(t.oracle_checks.values()) else 0.0, dict(t.oracle_checks), "oracle")
    return _verdict(not g.failing(), {"oracle_failing_checks": g.failing(), "reward": g.reward})


async def _s7_prompt(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    hints = [lit for lit in t.hidden_literals if lit in t.prompt]
    return _verdict(not hints, {"hidden_literals_in_prompt": hints})


async def _s10_flake(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    from crucible.factory.flake import FlakeConfig, measure_flake
    from crucible.factory.floors import Graded

    async def grade(i: int) -> Graded:
        bad = bool(t.flake_every) and i % t.flake_every == t.flake_every - 1
        return Graded(0.0 if bad else 1.0, {"test_mul_small": not bad}, "oracle-end-state")

    rep = await measure_flake(grade, config=FlakeConfig(reruns=30, concurrency=(1, 4)))
    return _verdict(rep.ok, {"upper": round(rep.upper, 5), "false_negatives": rep.false_negatives, "n": rep.n, "reasons": rep.reasons()})


async def _s12_hack(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    workspace = {"calc.py": t.world_files["src/calc.py"], ".results.json": json.dumps({"passed": True})}  # the hacker's attempt
    if t.grader_reads_workspace:  # a reachable grader trusts what the agent wrote (V4: EVALS L7)
        score = 1.0 if json.loads(workspace.get(".results.json", "{}")).get("passed") else 0.0
    else:  # the grader runs its own hidden tests on a pristine tree plus the patch: no mul, no pass
        score = 1.0 if "def mul(" in workspace["calc.py"] else 0.0
    return _verdict(score == 0.0, {"hack_attempt": "write .results.json {passed: true}", "score": score})


async def _s14_audit(ctx: StateContext) -> Settle:
    t: FakeTask = ctx.services
    from crucible.factory.audit import AuditConfig, Auditor, SeededCase, Trace, audit

    class FakeAuditorClient:
        async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            trace = json.loads(messages[-1]["content"])["trace"]
            cause = trace.split("[cause:", 1)[1].split("]", 1)[0]
            if t.auditor_mislabels and cause == "spec_defect":
                cause = "skill_gap"  # the planted mislabel: blames the model for the spec's defect
            return {"cause": cause, "solution_correct": False, "evidence": trace.split(" | ", 1)[0], "rationale": "fake"}

    traces = [Trace(f"r{i}", f"step {i}: mul(2, 3) returned 5 [cause:skill_gap] | tail", False) for i in range(4)]
    seeded = [SeededCase(Trace("ctl-spec", "the prompt never says mul must accept floats [cause:spec_defect] | tail", False), "spec_defect"),
              SeededCase(Trace("ctl-skill", "the agent edited add instead of writing mul [cause:skill_gap] | tail", False), "skill_gap")]
    rep = await audit(traces, Auditor(FakeAuditorClient(), model="fake", config=AuditConfig(concurrency=1)), prompt=t.prompt, seeded=seeded)
    return _verdict(rep.auditor_trusted is True and rep.ok, {"seeded": rep.seeded, "reasons": rep.reasons()})


def seeded_task_machine() -> dict[str, Any]:
    """S3 -> S5 -> S7 -> S10 -> S12 -> S14 -> accepted; a failing gate ends ``caught`` (rejected) with its evidence."""
    order = list(TASK_DEFECTS.values())
    v = [{"name": "verdict", "type": "json"}]
    states: list[dict[str, Any]] = [{"id": sid, "fn": sid, "outputs": v, **({"entry": True} if i == 0 else {})} for i, sid in enumerate(order)]
    states += [{"id": "accepted", "terminal": "done"}, {"id": "caught", "terminal": "rejected"}]
    trans = []
    for i, sid in enumerate(order):
        trans += [{"from": sid, "to": order[i + 1] if i + 1 < len(order) else "accepted", "when": "verdict.ok == true"},
                  {"from": sid, "to": "caught", "when": "verdict.ok == false"}]
    return {"run": {"max_transitions": 40}, "states": states, "transitions": trans}


SEEDED_HANDLERS = {"s3_world": _s3_world, "s5_verifier": _s5_verifier, "s7_prompt": _s7_prompt, "s10_flake": _s10_flake,
                   "s12_hack": _s12_hack, "s14_audit": _s14_audit}


async def _task_case(root: Path, defect: str | None) -> dict[str, Any]:
    name = defect or "clean_control"
    run = MachineRun(seeded_task_machine(), run_dir=root / name, handlers=SEEDED_HANDLERS, services=FakeTask.with_defect(defect), run_id=name)
    res = await run.run()
    out = journal_outcomes(run.journal.path)
    caught_at = out.get("last_transition", [None, None])[0] if res.status == "rejected" else None
    observed = {"status": res.status, "caught_at": caught_at}
    designated = {"status": "rejected", "caught_at": TASK_DEFECTS[defect]} if defect else {"status": "done", "caught_at": None}
    evidence = res.outputs.get(caught_at, {}).get("verdict") if caught_at else {s: res.outputs[s]["verdict"] for s in res.outputs}
    return {"case": name, "group": "task", "designated": designated, "observed": observed, "evidence": evidence,
            "evidence_sha256": sha256_json(evidence), "ok": observed == designated}


# ---- machine-layer cases ------------------------------------------------------------------------------------------------
def _chain(*states: dict[str, Any], run: dict[str, Any] | None = None) -> dict[str, Any]:
    ids = [s["id"] for s in states]
    trans = [{"from": a, "to": b} for a, b in itertools.pairwise(ids)] + [{"from": ids[-1], "to": "done"}]
    return {"run": run or {"max_transitions": 20}, "states": [{**states[0], "entry": True}, *states[1:], {"id": "done", "terminal": "done"}],
            "transitions": trans}


async def _machine_cases(root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    art_root = root / "artifacts"
    art_root.mkdir(parents=True, exist_ok=True)

    def case(name: str, designated: dict[str, Any], observed: dict[str, Any], evidence: Any) -> None:
        cases.append({"case": name, "group": "machine", "designated": designated, "observed": observed, "evidence": evidence,
                      "evidence_sha256": sha256_json(evidence), "ok": observed == designated})

    # a dead worker: re-checked, re-run once for free; retries 0, so a counted attempt would have paused the run
    deaths = {"n": 0}

    async def flaky_worker(ctx: StateContext) -> dict[str, Any]:
        deaths["n"] += 1
        if deaths["n"] == 1:
            raise InfraError("child worker unreachable (the sandbox died)", "crucible:env_container_lost")
        return {"out": "built"}

    async def no_sleep(_s: float) -> None:
        return None

    m = _chain({"id": "work", "fn": "work", "outputs": [{"name": "out", "type": "text"}]})
    r = MachineRun(m, run_dir=root / "dead_worker", handlers={"work": flaky_worker}, run_id="dead_worker", sleep=no_sleep)
    res = await r.run()
    jo = journal_outcomes(r.journal.path)["states"]["work"]
    case("dead_worker", {"status": "done", "attempts": 1, "infra_respawns": 1, "task_fails": 0},
         {"status": res.status, "attempts": jo["attempts"], "infra_respawns": jo["infra_respawns"], "task_fails": jo["task_fails"]}, jo)

    # a missing capture fails its producer, never the consumer
    entered: list[str] = []

    async def producer(ctx: StateContext) -> dict[str, Any]:
        entered.append("producer")
        return {"tests": {"path": "nowhere/tests", "sha256": "0" * 64}}

    async def consumer(ctx: StateContext) -> dict[str, Any]:
        entered.append("consumer")
        return {"verdict": {"ok": True}}

    m = _chain({"id": "producer", "fn": "producer", "outputs": [{"name": "tests", "type": "artifact"}]},
               {"id": "consumer", "fn": "consumer", "inputs": [{"name": "tests", "type": "artifact", "from": "producer.tests"}],
                "outputs": [{"name": "verdict", "type": "json"}]})
    r = MachineRun(m, run_dir=root / "missing_capture", handlers={"producer": producer, "consumer": consumer}, run_id="missing_capture",
                   artifact_root=art_root, sleep=no_sleep)
    res = await r.run()
    case("missing_capture", {"status": "paused", "decision": "failed", "state": "producer", "consumer_entered": False, "needs_verify": 1},
         {"status": res.status, "decision": (res.decision_request or {}).get("kind"), "state": (res.decision_request or {}).get("state"),
          "consumer_entered": "consumer" in entered, "needs_verify": len(res.needs_verify)}, {"reason": res.reason, "needs_verify": res.needs_verify})

    # a capture that vanished after its producer settled: the producer runs again, the consumer is not blamed
    produced = {"n": 0}

    async def producer2(ctx: StateContext) -> dict[str, Any]:
        produced["n"] += 1
        p = art_root / f"tests-{produced['n']}"
        p.mkdir(exist_ok=True)
        (p / "test_x.py").write_text(f"def test_x():\n    assert {produced['n']}\n")
        return {"tests": {"path": p.name, "sha256": artifact_digest(p)}}

    async def vandal(ctx: StateContext) -> dict[str, Any]:
        import shutil

        shutil.rmtree(art_root / "tests-1", ignore_errors=True)  # the artifact vanishes between settle and bind
        return {"x": "ok"}

    m = _chain({"id": "producer", "fn": "producer", "max_entries": 2, "outputs": [{"name": "tests", "type": "artifact"}]},
               {"id": "vandal", "fn": "vandal", "max_entries": 2, "outputs": [{"name": "x", "type": "text"}]},
               {"id": "consumer", "fn": "consumer", "max_entries": 2, "inputs": [{"name": "tests", "type": "artifact", "from": "producer.tests"}],
                "outputs": [{"name": "verdict", "type": "json"}]})
    r = MachineRun(m, run_dir=root / "vanished_capture", handlers={"producer": producer2, "vandal": vandal, "consumer": consumer},
                   run_id="vanished_capture", artifact_root=art_root, sleep=no_sleep)
    res = await r.run()
    first = {"status": res.status, "decision": (res.decision_request or {}).get("kind"), "state": (res.decision_request or {}).get("state")}
    res2 = await r.resume({"action": "revise", "actor": "receipt", "reason": "re-run the producer"})
    case("vanished_capture", {"status": "paused", "decision": "capture_failed", "state": "producer", "after_resume": "done", "producer_runs": 2},
         {**first, "after_resume": res2.status, "producer_runs": produced["n"]}, {"reason": res.reason})

    # an unbounded review loop escalates (V7); Prime ends it done (F:2469-2483, TF:5450-5477)
    async def write(ctx: StateContext) -> dict[str, Any]:
        return {"draft": f"draft {ctx.entry}"}

    async def review(ctx: StateContext) -> dict[str, Any]:
        return {"verdict": {"approved": False, "findings": [f"still wrong at {ctx.entry}"]}}

    m = {"run": {"max_transitions": 30},
         "states": [{"id": "write", "entry": True, "fn": "write", "max_entries": 2, "outputs": [{"name": "draft", "type": "text"}]},
                    {"id": "review", "fn": "review", "max_entries": 2, "outputs": [{"name": "verdict", "type": "json"}]},
                    {"id": "done", "terminal": "done"}],
         "transitions": [{"from": "write", "to": "review"}, {"from": "review", "to": "done", "when": "verdict.approved == true"},
                         {"from": "review", "to": "write", "when": "verdict.approved == false", "carry": "verdict.findings"}]}
    r = MachineRun(m, run_dir=root / "unbounded_loop", handlers={"write": write, "review": review}, run_id="unbounded_loop")
    res = await r.run()
    case("unbounded_loop", {"status": "paused", "decision": "exhausted"}, {"status": res.status, "decision": (res.decision_request or {}).get("kind")},
         {"reason": res.reason, "prime_7cbffa77b": "ends this loop done (F:2469-2483, :3515-3541; TF:5450-5477)"})

    # a mutating state with no recovery posture is refused at load
    bad = _chain({"id": "build", "fn": "build", "mutates": True, "outputs": [{"name": "out", "type": "text"}]})
    try:
        MachineRun(bad, run_dir=root / "no_recovery", handlers={"build": write}, run_id="no_recovery")
        refused, errors = False, []
    except MachineError as e:
        refused, errors = True, [x for x in e.errors if "recovery" in x]
    case("no_recovery_posture", {"refused": True, "names_recovery": True}, {"refused": refused, "names_recovery": bool(errors)}, errors)

    # a state over its token budget: paused, no retry
    calls = {"n": 0}

    async def hungry(ctx: StateContext) -> dict[str, Any]:
        calls["n"] += 1
        book_tokens(5_000)
        raise RuntimeError("the draft does not compile")

    m = _chain({"id": "author", "fn": "author", "retries": 3, "budget_tokens": 4_000, "outputs": [{"name": "out", "type": "text"}]})
    r = MachineRun(m, run_dir=root / "state_budget", handlers={"author": hungry}, run_id="state_budget")
    res = await r.run()
    case("state_budget", {"status": "paused", "decision": "state_budget", "calls": 1},
         {"status": res.status, "decision": (res.decision_request or {}).get("kind"), "calls": calls["n"]}, {"reason": res.reason, "tokens": res.tokens})
    return cases


async def run_seeded_defects(root: Path, *, receipt_path: Path | None = None) -> dict[str, Any]:
    """Run every seeded case (and the clean control), and return (and write) the receipt."""
    root = Path(root)
    task_cases = await asyncio.gather(*[_task_case(root / "task", d) for d in [None, *TASK_DEFECTS]])
    cases = [*task_cases, *await _machine_cases(root / "machine")]
    receipt = make_receipt(scope="P1 seeded-defect run: the machine layer catches each planted defect at its designated state "
                                 "(00 §13); a clean control passes every gate",
                           runs={c["case"]: c["observed"] for c in cases}, not_verified=NOT_VERIFIED_SEEDED, seeds={"fake_tasks": "deterministic"},
                           spend={"usd": 0.0, "tokens": 0}, cases=cases)
    if receipt_path is not None:
        write_receipt(receipt_path, receipt)
    return receipt


EXPECTED_SEEDED: dict[str, dict[str, Any]] = {
    **{d: {"status": "rejected", "caught_at": s} for d, s in TASK_DEFECTS.items()},
    "clean_control": {"status": "done", "caught_at": None},
    "dead_worker": {"status": "done", "attempts": 1, "infra_respawns": 1, "task_fails": 0},
    "missing_capture": {"status": "paused", "decision": "failed", "state": "producer"},
    "vanished_capture": {"decision": "capture_failed", "state": "producer"},
    "unbounded_loop": {"decision": "exhausted"},
    "no_recovery_posture": {"refused": True},
    "state_budget": {"decision": "state_budget", "calls": 1},
}
