"""P1 agentic-session states (factory/sessions.py; docs/research/swarm-pipeline/00 §3.2, §4, §15 P1), with scripted models
($0) and real Docker sandboxes.

- ``GitWorkspace``: a private repository at the base commit (no other refs or objects: a verifier's ``git log --all``
  cannot see the oracle builder's commits), snapshot / restore / diff for the recovery posture.
- The end-to-end run: lease -> planner -> World expert session -> oracle builder session -> verifier writer session ->
  review loop (a scripted reviewer rejects, then approves) -> accept, every session an M16 RLM agent whose kernel and shell
  run in its own offline ``crucible-p1-*`` container over its own worktree; hand-offs typed with the full spec and stored
  in the task repo; outputs captured as files and committed with ``informed_by``; the oracle builder's first attempt leaves
  no commit, so its capture fails the producer and the retry reconciles from the diff; the receipt of the run checks.
- The exhaustion variant (a reviewer that never approves) pauses ``exhausted``, never ``done``; the World expert is a
  persistent specialist resumed by id across the two runs (its ledger is its context); a third run whose scope overlaps a
  live announcement abstains before any session starts.
- A sandbox killed mid-session is infra: verified dead, the zombie removed, the session re-run once for free on a fresh
  container after a ``reset`` recovery."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crucible.actors.llm import LLMResponse
from crucible.factory.handoff import TaskRepo, accept_state, lease_state
from crucible.factory.machine import MachineRun, Settle, StateContext
from crucible.factory.receipts import check_receipt, run_receipt
from crucible.factory.sessions import PREFIX, SESSION_IMAGE, GitWorkspace, SessionPool, SpecialistBook, ensure_image, own_containers

CALC = "def add(a, b):\n    return a + b\n"


def make_source(root: Path) -> Path:
    src = root / "calc-src"
    src.mkdir(parents=True)
    (src / "calc.py").write_text(CALC)
    subprocess.run("git init -q -b main && git add -A && git -c user.email=a@b -c user.name=a commit -qm 'release A' && git tag A "
                   "&& echo 'def mul(a, b):\n    return a * b' >> calc.py && git -c user.email=a@b -c user.name=a commit -qam 'future: mul'",
                   shell=True, cwd=src, check=True, capture_output=True)
    return src


def git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=ws, capture_output=True, text=True, check=True).stdout


# ---- the worktree (no Docker) -----------------------------------------------------------------------------------------------
async def test_workspace_is_a_private_clone_with_snapshot_restore_diff(tmp_path: Path) -> None:
    src = make_source(tmp_path)
    ws = await GitWorkspace(tmp_path / "ws", source=src, base="A").create()
    assert git(ws.path, "log", "--all", "--format=%s").split("\n")[0] == "release A"  # B's commit is not in this store
    assert "future: mul" not in git(ws.path, "log", "--all", "--format=%s")
    assert git(ws.path, "tag").split() == ["base"]
    (ws.path / "calc.py").write_text(CALC + "# wip\n")
    (ws.path / "new.py").write_text("x = 1\n")
    tok = await ws.snapshot("t1")
    (ws.path / "calc.py").write_text("broken\n")
    (ws.path / "junk.py").write_text("junk\n")
    git(ws.path, "add", "-A")
    git(ws.path, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "half done")
    d = await ws.diff(tok)
    assert "half done" in d and "+broken" in d and "junk.py" in d
    await ws.restore(tok)
    assert (ws.path / "calc.py").read_text() == CALC + "# wip\n" and (ws.path / "new.py").exists() and not (ws.path / "junk.py").exists()
    assert git(ws.path, "log", "-1", "--format=%s").strip() == "release A"  # HEAD is back too
    assert "+# wip" in await ws.patch("base") and "new.py" in await ws.patch("base")


def test_specialist_book_resumes_by_id(tmp_path: Path) -> None:
    book = SpecialistBook(tmp_path / "sp")
    a = book.resume("world_expert:calc", "world_expert")
    book.record(a.id, {"run": "r1"})
    b = SpecialistBook(tmp_path / "sp").resume("world_expert:calc", "world_expert")  # another process, same id
    assert b.ledger_dir == a.ledger_dir and b.sessions == [{"run": "r1"}]


async def test_a_handoff_that_breaks_its_schema_sends_the_producer_back(tmp_path: Path) -> None:
    """The planner settles a plan with no milestones; the World expert's typed hand-off refuses it before any sandbox starts,
    and the run pauses on the producer (``capture_failed``), never on the session that received it."""
    make_source(tmp_path)
    repo = TaskRepo(tmp_path / "fam", "calc")
    m = {"states": [{"id": "plan", "entry": True, "fn": "plan", "max_entries": 2, "outputs": [{"name": "plan", "type": "json"}]},
                    {"id": "world", "role": "world_expert", "mutates": True, "recovery": "reset", "max_entries": 2,
                     "inputs": [{"name": "plan", "type": "json", "from": "plan.plan"}],
                     "session": {"captures": [{"output": "world", "kind": "json", "path": "world.json"}]}, "outputs": [{"name": "world", "type": "json"}]},
                    {"id": "done", "terminal": "done"}], "transitions": [{"from": "plan", "to": "world"}, {"from": "world", "to": "done"}]}
    prm = {"spec": {"request": "x"}, "repo": str(tmp_path / "calc-src"), "base": "A"}
    pool = SessionPool(tmp_path / "factory", repo=repo, machine=m, run_id="r", params=prm, kernel="local", llm_for=lambda role: None)

    async def plan(ctx: StateContext) -> dict:
        return {"plan": {"steps": ["mul"]}}

    run = MachineRun(m, run_dir=tmp_path / "run", run_id="r", params=prm, handlers={"session": pool.handler, "plan": plan}, **pool.machine_kwargs())
    res = await run.run()
    assert res.status == "paused" and res.decision_request["kind"] == "capture_failed" and res.decision_request["state"] == "plan"
    assert "milestones" in res.reason and pool.records == [] and pool.sandboxes == {}


# ---- Docker end to end ------------------------------------------------------------------------------------------------------
def _docker_ok() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False).returncode == 0


docker = pytest.mark.skipif(not _docker_ok(), reason="needs a local Docker daemon")


def cell(code: str, i: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}],
                       usage={"prompt_tokens": 200, "completion_tokens": 20})


def task_of(messages: list[dict[str, Any]]) -> str:
    return next((m["content"] for m in messages if m.get("role") == "user"), "")


class SessionScript:
    """Scripted session models: ``scripts[role](task, k)`` -> the k-th cell (k = tool results so far). Records every TASK
    and every tool output per role; ``on_turn(role, k)`` may inject a fault."""

    model = "scripted"

    def __init__(self, scripts: dict[str, Any], on_turn: Any = None):
        self.scripts, self.on_turn = scripts, on_turn
        self.tasks: dict[str, list[str]] = {}
        self.outputs: dict[str, list[str]] = {}

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        task = task_of(messages)
        role = re.search(r"^ROLE: (\S+)", task, re.MULTILINE).group(1)  # type: ignore[union-attr]
        k = sum(1 for m in messages if m.get("role") == "tool")
        if k == 0:
            self.tasks.setdefault(role, []).append(task)
        else:
            self.outputs.setdefault(role, []).append(str(next(m["content"] for m in reversed(messages) if m.get("role") == "tool")))
        if self.on_turn:
            self.on_turn(role, k)
        r = cell(self.scripts[role](task, k), k)
        r.model = self.model
        return r


def W(call: str) -> str:
    return f"r = await world.call({call}); print(r['ok'], str(r['content'])[:3000])"


def world_expert(task: str, k: int) -> str:
    return [
        "import os, json; print('IN_SANDBOX', os.path.exists('/handoff/handoff.json'), sorted(json.load(open('/handoff/handoff.json'))['ports']))",
        W("'repo.shell', command='python3 -c \"import calc; assert calc.add(1, 2) == 3\" && echo SUITE_OK; "
          "python3 -c \"import socket; socket.create_connection((\\'1.1.1.1\\', 53), 2)\" 2>/dev/null || echo OFFLINE'"),
        W("'repo.write_file', path='world.json', content='{\"test_cmd\": \"python3 -m unittest discover -s tests/hidden\"}'"),
        W("'repo.write_file', path='Dockerfile', content='FROM crucible-p1-session:py312\\nCOPY . /w\\n'"),
        "rlm.finish('World ready')",
    ][min(k, 4)]


def oracle_builder(task: str, k: int) -> str:
    if "[recovery: reconcile]" not in task:  # attempt 1: does the work, forgets to commit (its capture finds no commit)
        return [W("'repo.edit_file', path='calc.py', old_string='    return a + b\\n', "
                  "new_string='    return a + b\\n\\n\\ndef mul(a, b):\\n    return a * b\\n'"), "rlm.finish('done')"][min(k, 1)]
    return [W("'repo.shell', command='git status --short && git add -A && git commit -qm \"m1: mul\" && echo C1'"),
            W("'repo.shell', command='printf \"\\\\n\\\\ndef pow_(a, b):\\\\n    return a ** b\\\\n\" >> calc.py && git commit -qam \"m2: pow\" && echo C2'"),
            "rlm.finish('two milestone commits')"][min(k, 2)]


def verifier_writer(task: str, k: int) -> str:
    neg = "negative" in task
    body = ("import unittest\\nfrom calc import mul\\n\\n\\nclass T(unittest.TestCase):\\n    def test_mul(self):\\n        self.assertEqual(mul(2, 3), 6)\\n"
            + ("\\n    def test_mul_negative(self):\\n        self.assertEqual(mul(-2, 3), -6)\\n" if neg else ""))
    return [W("'repo.shell', command='git log --all --format=%s'"),
            W(f"'repo.write_file', path='tests/hidden/test_mul.py', content='{body}'"),
            "rlm.finish('tests written')"][min(k, 2)]


def script_llm(**kw: Any) -> SessionScript:
    return SessionScript({"world_expert": world_expert, "oracle_builder": oracle_builder, "verifier_writer": verifier_writer}, **kw)


PLAN = {"milestones": [{"id": "m1", "text": "mul(a, b)"}, {"id": "m2", "text": "pow_(a, b)", "requires": ["m1"]}]}


def e2e_machine(*, verifier_max: int = 3) -> dict[str, Any]:
    j = "json"
    return {
        "run": {"max_transitions": 60, "budget_usd": 1.0, "budget_tokens": 2_000_000},
        "states": [
            {"id": "lease", "entry": True, "fn": "lease", "outputs": [{"name": "lease", "type": j}]},
            {"id": "plan", "role": "planner", "handler": "plan", "inputs": [{"name": "spec", "type": j, "from": "params.spec"}],
             "outputs": [{"name": "plan", "type": j}]},
            {"id": "world", "role": "world_expert", "mutates": True, "recovery": "reset", "retries": 1, "budget_tokens": 200_000,
             "inputs": [{"name": "plan", "type": j, "from": "plan.plan"}],
             "session": {"brief": "Make the repo at A buildable and testable offline; state the test command.", "specialist": "world_expert:{repo_name}",
                         "captures": [{"output": "world", "kind": "json", "path": "world.json"}, {"output": "dockerfile", "kind": "file", "path": "Dockerfile"}]},
             "outputs": [{"name": "world", "type": j}, {"name": "dockerfile", "type": "artifact"}]},
            {"id": "oracle", "role": "oracle_builder", "canary": True, "mutates": True, "recovery": "reconcile", "retries": 1,
             "inputs": [{"name": "plan", "type": j, "from": "plan.plan"}, {"name": "world", "type": j, "from": "world.world"}],
             "session": {"brief": "Build the oracle: one commit per milestone.", "captures": [{"output": "oracle", "kind": "commits"}]},
             "outputs": [{"name": "oracle", "type": "artifact"}]},
            {"id": "verifier", "role": "verifier_writer", "isolated_from": ["oracle"], "mutates": True, "recovery": "reset", "max_entries": verifier_max,
             "inputs": [{"name": "plan", "type": j, "from": "plan.plan"}, {"name": "world", "type": j, "from": "world.world"},
                        {"name": "dockerfile", "type": "artifact", "from": "world.dockerfile"}],
             "session": {"brief": "Write hidden tests per milestone, from the spec only.",
                         "captures": [{"output": "tests", "kind": "dir", "path": "tests/hidden"}]},
             "outputs": [{"name": "tests", "type": "artifact"}]},
            {"id": "review", "role": "reviewer", "handler": "review", "different_family_from": ["verifier"], "max_entries": verifier_max,
             "inputs": [{"name": "tests", "type": "artifact", "from": "verifier.tests"}], "outputs": [{"name": "verdict", "type": j}]},
            {"id": "accept", "fn": "accept", "inputs": [{"name": "lease", "type": j, "from": "lease.lease"},
                                                        {"name": "parent", "type": "text", "from": "params.parent"}],
             "outputs": [{"name": "accept", "type": j}]},
            {"id": "accepted", "terminal": "done"}, {"id": "abstained", "terminal": "rejected"}, {"id": "herded", "terminal": "rejected"}],
        "transitions": [
            {"from": "lease", "to": "plan", "when": "lease.ok == true"}, {"from": "lease", "to": "abstained", "when": "lease.ok == false"},
            {"from": "plan", "to": "world"}, {"from": "world", "to": "oracle"}, {"from": "oracle", "to": "verifier"},
            {"from": "verifier", "to": "review"},
            {"from": "review", "to": "accept", "when": "verdict.approved == true"},
            {"from": "review", "to": "verifier", "when": "verdict.approved == false", "carry": "verdict.findings"},
            {"from": "accept", "to": "accepted", "when": "accept.ok == true"}, {"from": "accept", "to": "herded", "when": "accept.ok == false"}]}


MODELS = {"planner": "qwen", "world_expert": "qwen", "oracle_builder": "qwen", "verifier_writer": "qwen", "reviewer": "deepseek"}


def reviewer(repo: TaskRepo, *, demand: str, seen: list[str]) -> Any:
    """A scripted reviewer that reads the captured tests (files, from the task repo) and rejects until ``demand`` is covered."""

    async def review(ctx: StateContext) -> Settle:
        text = "".join(p.read_text() for p in sorted((repo.root / ctx.inputs["tests"]["path"]).rglob("*.py")))
        ok = demand in text
        seen.append("approve" if ok else "reject")
        return Settle({"verdict": {"approved": ok, "findings": [] if ok else [f"no test covers {demand} numbers"]}})

    return review


async def plan(ctx: StateContext) -> dict:
    return {"plan": PLAN}


def params(n: int, intent: str) -> dict[str, Any]:
    return {"spec": {"request": "Add mul(a, b) and pow_(a, b) to calc.py.", "intent": "dependent milestones", "source": "issue #7"},
            "scope": {"repo": "calc", "intent": intent}, "abort_criteria": ["upstream ships mul"], "parent": f"issue-{n}", "task_id": f"calc-{intent}",
            "repo_name": "calc", "holder": f"root/{n}"}


async def run_machine(tmp_path: Path, run_id: str, prm: dict[str, Any], llm: SessionScript, repo: TaskRepo, review_demand: str,
                      seen: list[str], *, verifier_max: int = 3) -> tuple[MachineRun, Any, SessionPool]:
    m = e2e_machine(verifier_max=verifier_max)
    pool = SessionPool(tmp_path / "factory", repo=repo, machine=m, run_id=run_id, params={**prm, "repo": str(tmp_path / "calc-src"), "base": "A"},
                       llm_for=lambda role: llm)
    run = MachineRun(m, run_dir=tmp_path / "runs" / run_id, run_id=run_id, models=MODELS, params=pool.params,
                     handlers={"session": pool.handler, "lease": lease_state(repo), "plan": plan, "accept": accept_state(repo),
                               "review": reviewer(repo, demand=review_demand, seen=seen)},
                     **pool.machine_kwargs())
    try:
        res = await run.run()
    finally:
        await pool.close()
    return run, res, pool


@docker
@pytest.mark.docker
async def test_end_to_end_sessions_review_loop_exhaustion_and_specialist(tmp_path: Path) -> None:
    ensure_image()
    before = set(own_containers())
    make_source(tmp_path)
    repo = TaskRepo(tmp_path / "family-calc", "calc")
    llm = script_llm()
    seen: list[str] = []
    run, res, pool = await run_machine(tmp_path, "run-1", params(1, "mul"), llm, repo, "negative", seen)

    assert res.status == "done", (res.reason, res.decision_request)
    assert res.entries == {"lease": 1, "plan": 1, "world": 1, "oracle": 1, "verifier": 2, "review": 2, "accept": 1}
    assert seen == ["reject", "approve"]  # the scripted reviewer rejected, the verifier writer fixed it, the reviewer approved
    # the World expert's kernel and shell ran in its own offline sandbox and saw the typed hand-off
    out = "\n".join(llm.outputs["world_expert"])
    assert "IN_SANDBOX True ['plan']" in out and "SUITE_OK" in out and "OFFLINE" in out
    # every hand-off carries the full original spec; work products are files committed with informed_by
    ho = json.loads(next(repo.root.glob("runs/run-1/world/*/handoff.json")).read_text())
    world_rec = next(r for r in pool.records if r.state == "world")
    assert repo.trailers(world_rec.handoff_commit)["handoff"] == [ho["id"]]
    assert ho["spec"] == pool.params["spec"] and ho["ports"]["plan"]["milestones"][0]["id"] == "m1"
    world_art = res.outputs["world"]["dockerfile"]
    assert (repo.root / world_art["path"]).read_text().startswith("FROM crucible-p1-session") and repo.trailers(world_art["commit"])["informed_by"]
    # the oracle builder: attempt 1 left no commit (capture failure of the producer), attempt 2 reconciled from its diff
    assert len(llm.tasks["oracle_builder"]) == 2 and "[recovery: reconcile]" in llm.tasks["oracle_builder"][1]
    assert "+def mul(a, b):" in llm.tasks["oracle_builder"][1] and "missing declared output 'oracle'" in llm.tasks["oracle_builder"][1]
    assert res.outputs["oracle"]["oracle"]["subjects"] == ["m1: mul", "m2: pow"] and len(res.needs_verify) == 1
    # generation and verification never shared a context: the verifier's store has no oracle commit, its hand-off no canary
    vws = pool.workspaces["verifier"].path
    assert "m1: mul" not in git(vws, "log", "--all", "--format=%s") and "m1: mul" not in "\n".join(llm.outputs["verifier_writer"])
    canary = res.outputs["oracle"]["oracle"]["_canary"]
    assert all(canary not in p.read_text() for p in repo.root.glob("runs/run-1/verifier/*/handoff.json"))
    assert "no test covers negative numbers" in llm.tasks["verifier_writer"][1]  # the review finding reached the second entry
    assert (repo.root / res.outputs["verifier"]["tests"]["path"] / "test_mul.py").read_text().count("def test_") == 2
    assert res.outputs["accept"]["accept"]["ok"] is True and [a["task_id"] for a in repo.accepted()] == ["calc-mul"]
    rec = run_receipt(run, scope="P1 e2e sessions", not_verified=["real models", "multi-host"], images=[SESSION_IMAGE])
    assert check_receipt(rec) == [] and rec["images"][SESSION_IMAGE] and rec["runs"]["run-1"]["states"]["oracle"]["recoveries"]["reconcile"] == 1
    assert rec["runs"]["run-1"]["states"]["oracle"]["capture_failed"] == 1 and rec["spend"]["tokens"] > 0

    # the exhaustion variant: a reviewer that never approves pauses the run; the World expert resumes as the same specialist
    seen2: list[str] = []
    _, res2, _ = await run_machine(tmp_path, "run-2", params(2, "pow"), llm, repo, "float", seen2, verifier_max=2)
    assert res2.status == "paused" and res2.decision_request["kind"] == "exhausted" and seen2 == ["reject", "reject"]
    assert "[memory] You are the persistent specialist `world_expert:calc`" in llm.tasks["world_expert"][1]
    sp = SpecialistBook(tmp_path / "factory" / "specialists").get("world_expert:calc")
    assert sp is not None and [s["run"] for s in sp.sessions] == ["run-1", "run-2"]
    windows = [json.loads(x) for x in (Path(sp.ledger_dir) / "root" / "events.jsonl").read_text().splitlines() if '"window"' in x]
    assert any((w.get("meta") or {}).get("kind") == "specialist_session" and (w.get("meta") or {}).get("run") == "run-2" for w in windows)

    # run 2 is paused, its announcement live: a third root on the same scope abstains before any session starts
    _, res3, _ = await run_machine(tmp_path, "run-3", params(3, "pow"), llm, repo, "negative", [])
    assert res3.status == "rejected" and res3.reason.endswith("abstained") and res3.entries == {"lease": 1}
    assert set(own_containers()) - before == set()  # every crucible-p1-* container of these runs is gone


@docker
@pytest.mark.docker
async def test_killed_sandbox_is_infra_respawned_once_with_reset(tmp_path: Path) -> None:
    ensure_image()
    make_source(tmp_path)
    repo = TaskRepo(tmp_path / "family-calc", "calc")
    killed: list[str] = []

    def kill_once(role: str, k: int) -> None:
        if k == 2 and not killed:  # the sandbox dies after the World expert's first write
            for c in own_containers():
                if c.startswith(PREFIX) and c in {sb.name for sb in pool.sandboxes.values()}:
                    subprocess.run(["docker", "rm", "-f", c], capture_output=True, check=False)
                    killed.append(c)

    llm = script_llm(on_turn=kill_once)
    m = {"states": [{"id": "world", "entry": True, "role": "world_expert", "mutates": True, "recovery": "reset", "retries": 0,
                     "inputs": [{"name": "plan", "type": "json", "from": "params.plan"}],
                     "session": {"captures": [{"output": "world", "kind": "json", "path": "world.json"}]}, "outputs": [{"name": "world", "type": "json"}]},
                    {"id": "done", "terminal": "done"}], "transitions": [{"from": "world", "to": "done"}]}
    prm = {"spec": {"request": "x"}, "plan": PLAN, "repo": str(tmp_path / "calc-src"), "base": "A"}
    pool = SessionPool(tmp_path / "factory", repo=repo, machine=m, run_id="r", params=prm, llm_for=lambda role: llm)
    run = MachineRun(m, run_dir=tmp_path / "run", run_id="r", params=prm, handlers={"session": pool.handler}, infra_quiet_s=0.1, **pool.machine_kwargs())
    try:
        res = await run.run()
    finally:
        await pool.close()
    assert killed and res.status == "done" and res.outputs["world"]["world"]["test_cmd"].startswith("python3")
    evs = [json.loads(x) for x in run.journal.path.read_text().splitlines()]
    resp = [e for e in evs if e["event"] == "infra_respawn"]
    assert len(resp) == 1 and resp[0]["verdict"] == "dead" and "No such container" in resp[0]["error"]
    assert [e["mode"] for e in evs if e["event"] == "recovery"] == ["reset"] and not [e for e in evs if e["event"] == "fail"]
    assert len({r.container for r in pool.records}) == 2 and killed[0] == pool.records[0].container
