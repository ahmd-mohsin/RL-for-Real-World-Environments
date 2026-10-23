"""The Tinker swarm (swarm-pipeline/13): the ``swarm`` preset and its family rules, artifact provenance, builder Φ on the
RLM harness, the prompt machine end to end with scripted models ($0, local kernel), the journal's blame and credit, the
diagnosis, and the difficulty interface."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crucible.actors.llm import LLMResponse, book_scoped
from crucible.factory import models
from crucible.factory.builder_potential import hidden_tests, oracle_commits, prompt_draft
from crucible.factory.difficulty_hooks import GateResult, HeldOutRefused, KnobAction, Revision, revise_and_gate
from crucible.factory.generators.swarm import SERVICE_READS, swarm_machine
from crucible.factory.models import ModelBook, SameFamilyError, family_of
from crucible.factory.provenance import Producer, Provenance, ProvenanceLedger, static_problems, upstream, v2_problems
from crucible.factory.sessions import GitWorkspace, StateBudgetClient, StateBudgetExceeded, context_window
from crucible.factory.swarm_journal import SwarmJournal, attempt_rewards, blame_of, builder_credit, diagnose, finding_kind, render, tables
from crucible.factory.swarm_prompts import PromptJob, prompt_machine, write_prompt


# ---- the preset -----------------------------------------------------------------------------------------------------------------------
def test_swarm_preset_has_no_claude_and_three_builder_families() -> None:
    b = ModelBook(preset="swarm")
    assert not [r for r, s in b.roles.items() if s.provider == models.CLAUDE_CODE]
    assert set(b.builder_families()) == {"deepseek", "glm", "kimi"}
    assert b.spec("oracle_builder").family == "deepseek" and b.spec("verifier_writer").family == "glm" and b.spec("prompt_author").family == "kimi"
    checkers = {b.spec(r).family for r in models.CHECKER_ROLES if r in b.roles}
    assert checkers.isdisjoint({"deepseek", "glm", "kimi"}) and "qwen" not in {b.spec(r).family for r in models.INDEPENDENT_ROLES if r in b.roles}
    assert [s.family for s in b.frontier] == ["qwen", "nemotron"] and b.family_problems() == []
    assert b.spec("prompt_author").model.endswith(":peft:131072")  # the 32K Kimi cannot hold a session
    assert family_of("tinker:moonshotai/Kimi-K2.6") == "kimi"


@pytest.mark.parametrize("override, why", [
    ({"verifier_writer": models.DEEPSEEK}, "V2"),  # the first verifier writer on the oracle builder's family
    ({"reviewer": models.GLM}, "shares family 'glm'"),  # a checker on a builder's family
    ({"rewriter_qwen": models.DEEPSEEK}, "rewriter_qwen"),  # an alternative by the oracle builder's family
    ({"auditor": models.QWEN_FRONTIER}, "student"),  # the Qwen student audited by Qwen
    ({"hacker_2": models.NEMOTRON}, "2 families"),
    ({"planner": models.CLAUDE_BUILDER}, "Claude Code"),
])
def test_swarm_family_rules_refuse(override: dict[str, str], why: str) -> None:
    with pytest.raises(SameFamilyError, match=re.escape(why) if why != "2 families" else ">= 2 families"):
        ModelBook(override, preset="swarm")


def test_frontier_of_a_builder_family_is_refused() -> None:
    with pytest.raises(SameFamilyError, match="frontier solver"):
        ModelBook(preset="swarm", frontier=[models.DEEPSEEK])
    ModelBook()  # the default map (Claude builds) keeps its DeepSeek fairness solver


# ---- provenance -----------------------------------------------------------------------------------------------------------------------
def test_static_provenance_on_the_swarm_machine() -> None:
    m = swarm_machine()
    fams = ModelBook(preset="swarm").families()
    assert static_problems(m, fams, reads=SERVICE_READS) == []
    assert {"plan", "oracle", "verifier", "build"} <= upstream(m, "hack", SERVICE_READS)
    bad = {**fams, "hacker": "deepseek"}  # the hacker on the oracle builder's family reads the build (through services)
    assert any("hack" in p and "oracle" in p for p in static_problems(m, bad, reads=SERVICE_READS))
    assert v2_problems({"oracle_builder": "glm", "verifier_writer": "glm"}) and not v2_problems(fams)


def test_ledger_checks_the_family_of_the_client_that_runs() -> None:
    m = {"states": [{"id": "plan", "role": "planner", "outputs": []}, {"id": "tests", "role": "verifier_writer", "inputs": [{"name": "p", "from": "plan.plan"}]},
                    {"id": "review", "role": "reviewer", "inputs": [{"name": "t", "from": "tests.tests"}]}]}
    fam = {"planner": "glm", "verifier_writer": "kimi", "reviewer": "nemotron"}
    led = ProvenanceLedger(m, family_for=fam.get)
    led.record("plan", "planner")
    led.record("tests", "verifier_writer")
    assert led.states["tests"].families == {"glm", "kimi"}  # transitive: the tests were written from the plan
    assert led.check("review", "reviewer").families == {"glm", "kimi"}
    fam["reviewer"] = "glm"  # a later override puts the reviewer on the planner's family: refused at the check
    with pytest.raises(SameFamilyError, match="planner@plan"):
        led.check("review", "reviewer")
    p = Provenance((Producer("a", "x"),)).merge(Provenance((Producer("a", "x"), Producer("b", "y"))))
    assert Provenance.from_json(p.to_json()) == p and p.families == {"x", "y"}


# ---- builder Φ ------------------------------------------------------------------------------------------------------------------------
def _ws(tmp: Path) -> Path:
    src = tmp / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    subprocess.run("git init -q && git add -A && git -c user.email=a@b -c user.name=a commit -qm A && git tag A", shell=True, cwd=src, check=True)
    return src


async def test_prompt_draft_potential_per_requirement(tmp_path: Path) -> None:
    ws = await GitWorkspace(tmp_path / "ws", source=_ws(tmp_path), base="A").create()
    pot = prompt_draft("t", ws.path, ["m1", "m2"], lint=lambda t: ["lint hidden: 'secret_fn'"] if "secret_fn" in t else [])
    draft = {"opening": "We use the library daily.", "requirements": {"m1": "Calling mean on [2, 4] returns 2 but must return 3, the plain average.",
                                                                       "m2": "secret_fn breaks"}}
    (ws.path / "prompt.json").write_text(json.dumps(draft))
    patch = await ws.patch("base")
    assert await pot.milestone_status(patch) == {"m1": True, "m2": False}  # m2: too short and names a hidden string
    assert await pot.milestone_status("") == {"m1": False, "m2": False}
    assert not (tmp_path / "ws" / "STATUS.json").exists() and pot.requirement_ids == ["m1", "m2"]


async def test_hidden_tests_and_oracle_commit_potentials(tmp_path: Path) -> None:
    ws = await GitWorkspace(tmp_path / "ws", source=_ws(tmp_path), base="A").create()
    plan = {"milestones": [{"id": "m1"}, {"id": "m2"}]}
    (ws.path / "crucible_hidden").mkdir()
    (ws.path / "crucible_hidden" / "test_m1.py").write_text("def test_one():\n    assert 1\n")
    (ws.path / "crucible_hidden" / "test_m2.py").write_text("def test_two(:\n")
    assert await hidden_tests("t", ws.path, plan).milestone_status(await ws.patch("base")) == {"m1": True, "m2": False}
    oc = oracle_commits("t", ws.path, plan)
    git = ["git", "-c", "user.email=a@b", "-c", "user.name=a"]
    subprocess.run([*git, "commit", "-qam", "m2: wrong order", "--allow-empty"], cwd=ws.path, check=True)  # noqa: ASYNC221
    assert await oc.milestone_status(await oc.snapshot_patch(None)) == {"m1": False, "m2": False}
    assert oc.milestones[1]["requires"] == ["m1"]


# ---- the session plumbing ---------------------------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self, ctx: int):
        self.ctx = ctx

    def lock_settings(self) -> dict[str, Any]:
        return {"context": self.ctx}


def test_context_window_fits_the_model() -> None:
    from crucible.actors.rlm.agent import RLMConfig

    cfg = RLMConfig(max_tokens=2000)
    assert context_window(_Ctx(131_072), cfg) == 131_072 - 2000 - 4096  # Kimi's 128K: below the 128K default window
    assert context_window(_Ctx(262_144), cfg) is None  # the default already fits
    assert context_window(object(), cfg) is None
    from crucible.factory.sessions import window_floor

    assert window_floor(9_000, cfg) > 10_000  # the smoke's 10K window and 3K-token hand-off: compaction_failed
    assert window_floor(9_000, cfg) < 16_000


async def test_state_budget_client_stops_mid_session() -> None:
    class Paid:
        model, family = "m", "x"

        async def complete(self, *a: Any, **k: Any) -> Any:
            book_scoped(0.4)
            return "ok"

    from crucible.actors.llm import SpendScope

    c = StateBudgetClient(Paid(), 0.5)
    with SpendScope() as outer:  # the machine's scope around the state still sees every dollar (smoke bug: it saw 0)
        assert await c.complete([]) == "ok" and await c.complete([]) == "ok"
        with pytest.raises(StateBudgetExceeded):
            await c.complete([])
    assert outer.usd == pytest.approx(0.8) and c.family == "x" and StateBudgetExceeded.failure_kind == "crucible:state_budget"


def test_text_protocol_reads_kimi_native_calls() -> None:
    from crucible.train.bridges.tinker.codec import PARSERS

    native = ('Let me look.<|tool_calls_section_begin|><|tool_call_begin|>functions.python:2<|tool_call_argument_begin|>{"code": "print(1)"}'
              "<|tool_call_end|><|tool_calls_section_end|>")
    content, calls = PARSERS["text"](native)
    assert content == "Let me look." and [(c["name"], c["arguments"]) for c in calls] == [("python", {"code": "print(1)"})]
    _, bad = PARSERS["text"]("<|tool_calls_section_begin|><|tool_call_begin|>functions.python:0<|tool_call_argument_begin|>{\"code\": ")
    assert bad and "__unparseable__" in bad[0]["arguments"]  # never a silent final answer
    assert PARSERS["text"]('<tool_call>{"name": "python", "arguments": {"code": "x"}}</tool_call>')[1][0]["name"] == "python"


# ---- the prompt machine end to end (scripted models, local kernel) -------------------------------------------------------------------------
def _cell(code: str, i: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}], usage={"prompt_tokens": 300, "completion_tokens": 30})


def _W(call: str) -> str:
    return f"r = await world.call({call}); print(r['ok'], str(r['content'])[:2000])"


class Writer:
    """A scripted prompt writer: a first draft with one leaky line, then (with the lint's findings in its TASK) a clean one."""

    model, family = "scripted-writer", "kimi"

    def __init__(self) -> None:
        self.tasks: list[str] = []

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        task = next(m["content"] for m in messages if m.get("role") == "user")
        k = sum(1 for m in messages if m.get("role") == "tool")
        if k == 0:
            self.tasks.append(task)
        fixed = "[findings]" in task
        m2 = "Clamping 5 into the range 0 to 10 gives 0, but it must give 5 back unchanged." if fixed else "min(x, lo) is wrong"
        draft = json.dumps({"opening": "Our team uses the calc library for reports.",
                            "requirements": {"m1": "The mean of [2, 4] comes out as 2, but it must be 3, the ordinary average.", "m2": m2}})
        steps = [_W(f"'repo.write_file', path='prompt.json', content={draft!r}"),
                 _W("'repo.write_file', path='STATUS.json', content='{\"m1\": 0.9, \"m2\": 0.8}'"), "rlm.finish('written')"]
        return _cell(steps[min(k, 2)], k)


class Reviewer:
    model, family = "scripted-reviewer", "nemotron"

    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {"ok": True, "findings": []}


class Book:
    def __init__(self, clients: dict[str, Any]):
        self.clients = clients

    def client(self, role: str) -> Any:
        return self.clients[role]

    def spec(self, role: str) -> Any:
        return models.parse_spec(f"tinker:{self.clients[role].model}", default_provider="tinker")


async def test_prompt_machine_end_to_end_with_phi_journal_and_credit(tmp_path: Path) -> None:
    from crucible.actors.rlm.agent import RLMConfig

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "calc.py").write_text("def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n")
    lint_text = lambda t: [f"lint hidden: {s!r}" for s in ("min(x, lo)",) if s in t]
    lint = lambda d: lint_text(json.dumps(d)) + [f"unmapped_milestone: {m}" for m in ("m1", "m2") if not (d.get("requirements") or {}).get(m)]
    job = PromptJob("calc-1", ["m1", "m2"], {"task": "calc-1", "milestone_ids": ["m1", "m2"], "milestones": [], "forbidden_strings": ["min(x, lo)"]}, tree,
                    lint, lint_text, tmp_path / "prompts" / "calc-1.json")
    writer, reviewer = Writer(), Reviewer()
    journal = SwarmJournal(tmp_path / "journal")
    cfg = RLMConfig(temperature=0.0, kernel="local", allow_host=True, tree_max_depth=0, max_turns=8, window_tokens=12_000, max_tokens=2000)
    res = await write_prompt(job, book=Book({"prompt_author": writer, "reviewer": reviewer}), root=tmp_path / "swarm", journal=journal, kernel="local",
                             rlm_config=cfg)
    assert res["status"] == "done" and res["sessions"] == 2 and reviewer.calls == 1
    assert res["phi_final"] == [0.5, 1.0]  # the leaky line held m2 at 0 in the first draft; the lint's findings fixed it
    assert "[findings]" in writer.tasks[1] and "min(x, lo)" in writer.tasks[1]
    out = json.loads(job.out.read_text())
    assert out["builder"].endswith("scripted-writer") and out["swarm"]["provenance"]["families"] == ["kimi"]
    sess = [json.loads(x) for x in (tmp_path / "journal" / "sessions.jsonl").read_text().splitlines()]
    assert {s["family"] for s in sess} == {"kimi"} and all(s["progress"]["honesty"] for s in sess)
    gates = [json.loads(x) for x in (tmp_path / "journal" / "gates.jsonl").read_text().splitlines()]
    lint_rows = [g for g in gates if g["state"] == "lint"]
    assert [g["ok"] for g in lint_rows] == [False, True] and blame_of(lint_rows[0], {"write": "prompt_author"}) == ["prompt_author"]
    credit = builder_credit(tmp_path / "journal")
    by_attempt = {(c["entry"], c["reward"]) for c in credit}
    assert by_attempt == {(1, 0.0), (2, 1.0)}
    assert all(c["progress_turns"] for c in credit)  # the write_file turn made the progress
    t = tables(tmp_path / "journal")
    assert t["accepted"] == 1 and t["roles"]["prompt_author"]["blamed"] == 1 and t["roles"]["prompt_author"]["gate_pass_rate"] == pytest.approx(2 / 3, abs=1e-3)
    d = diagnose(tmp_path / "journal")
    props = [p for b in d["bottlenecks"] for p in b["proposals"]]
    assert props and all(p["proposal"] and not p["applied"] for p in props) and any(p["kind"] == "tool" for p in props)
    assert "Bottlenecks" in render(d)


class Lingerer(Writer):
    """Writes a clean report, then keeps reading files and never calls finish (the smoke's Kimi at turn 9..24)."""

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        k = sum(1 for m in messages if m.get("role") == "tool")
        if k == 0:
            draft = json.dumps({"opening": "Our team uses the calc library for reports.",
                                "requirements": {"m1": "The mean of [2, 4] comes out as 2, but it must be 3, the ordinary average.",
                                                 "m2": "Clamping 5 into the range 0 to 10 gives 0, but it must give 5 back unchanged."}})
            return _cell(_W(f"'repo.write_file', path='prompt.json', content={draft!r}"), k)
        return _cell(_W("'repo.read_file', path='calc.py'"), k)


async def test_a_budget_stop_still_goes_to_the_gate(tmp_path: Path) -> None:
    from crucible.actors.rlm.agent import RLMConfig

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "calc.py").write_text("x = 1\n")
    job = PromptJob("calc-2", ["m1", "m2"], {"task": "calc-2"}, tree, lambda d: [], lambda t: [], tmp_path / "out.json")
    cfg = RLMConfig(temperature=0.0, kernel="local", allow_host=True, tree_max_depth=0, max_turns=3, window_tokens=12_000, max_tokens=2000)
    res = await write_prompt(job, book=Book({"prompt_author": Lingerer(), "reviewer": Reviewer()}), root=tmp_path / "swarm",
                             journal=SwarmJournal(tmp_path / "j"), kernel="local", rlm_config=cfg)
    assert res["status"] == "done" and res["phi_final"] == [1.0] and json.loads(job.out.read_text())["requirements"]["m2"].startswith("Clamping")
    rows = [json.loads(x) for x in (tmp_path / "j" / "sessions.jsonl").read_text().splitlines()]
    assert rows[-1]["stop"] == "budget:turns"


def test_prompt_machine_validates() -> None:
    from crucible.factory.machine import validate_machine

    assert validate_machine(prompt_machine(), models={"prompt_author": "kimi", "reviewer": "nemotron"}) == []
    assert validate_machine(prompt_machine(), models={"prompt_author": "kimi", "reviewer": "kimi"})  # different_family_from


# ---- journal pieces ------------------------------------------------------------------------------------------------------------------------
def test_rewards_blame_and_finding_kinds() -> None:
    s = [{"run": "r", "state": "write", "dir": "d1", "ts": 1, "stop": "agent_done"}, {"run": "r", "state": "write", "dir": "d2", "ts": 5, "stop": "agent_gave_up"}]
    g = [{"run": "r", "state": "lint", "ts": 2, "kind": "verdict", "ok": True, "provenance": {"producers": [{"role": "prompt_author", "state": "write"}]}}]
    assert attempt_rewards(s, g) == {"d1": 1.0, "d2": 0.0}
    assert blame_of({"ok": False, "kind": "infra"}) == []
    assert blame_of({"ok": False, "kind": "verdict", "provenance": {"producers": [{"role": "verifier_writer"}]}}) == ["verifier_writer"]
    assert finding_kind("m3: test_m3.py does not parse (invalid syntax, line 4)") == finding_kind("m1: test_m1.py does not parse (x, line 9)")


# ---- the difficulty interface ---------------------------------------------------------------------------------------------------------
async def test_revise_and_gate_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        KnobAction("ease", "H1", "wrong table")

    class R:
        async def revise(self, task: Any, action: KnobAction) -> Revision:
            return Revision(f"{task}-k1", task, action, task=f"{task}-k1")

    class G:
        def __init__(self, ok: bool):
            self.ok = ok

        async def run(self, rev: Revision) -> GateResult:
            return GateResult(self.ok, {"S5": self.ok}, [] if self.ok else ["S5: oracle 0.5"])

    act = KnobAction("ease", "K1", "p̂ 0.0 over 8", milestone="m3", evidence={"p": 0.0})
    j = SwarmJournal(tmp_path / "j")
    ok = await revise_and_gate("t", act, reviser=R(), gates=G(True), journal=j)
    bad = await revise_and_gate("t", act, reviser=R(), gates=G(False), journal=j)
    assert ok["ok"] and ok["revision"].task == "t-k1" and not bad["ok"] and bad["revision"] is None
    assert tables(tmp_path / "j")["roles"]["reviser"]["blamed"] == 1
    with pytest.raises(HeldOutRefused):
        await revise_and_gate("t", act, reviser=R(), gates=G(True), heldout=True)


def test_a_stale_prompt_run_is_archived_never_resumed(tmp_path: Path) -> None:
    import json as _json

    from crucible.factory.machine import canonicalize, machine_hash
    from crucible.factory.swarm_prompts import archive_stale_run, prompt_machine

    m = prompt_machine()
    for d in ("runs", "sessions", "family/runs"):
        (tmp_path / d / "prompt-x").mkdir(parents=True)
    j = tmp_path / "runs" / "prompt-x" / "journal.jsonl"
    j.write_text(_json.dumps({"event": "start", "machine_hash": machine_hash(canonicalize(m)), "params": {"spec": {"a": 1}}}) + "\n")
    assert archive_stale_run(tmp_path, "prompt-x", m, {"spec": {"a": 1}}) == []  # same machine, same hand-off: resumable
    moved = archive_stale_run(tmp_path, "prompt-x", m, {"spec": {"a": 2}})  # a new hand-off under the same id
    assert sorted(p.name for p in moved) == ["prompt-x.prev-1"] * 3 and not (tmp_path / "sessions" / "prompt-x").exists()
    (tmp_path / "runs" / "prompt-x").mkdir()
    (tmp_path / "runs" / "prompt-x" / "journal.jsonl").write_text(_json.dumps({"event": "start", "machine_hash": "0" * 16}) + "\n")
    assert [p.name for p in archive_stale_run(tmp_path, "prompt-x", m)] == ["prompt-x.prev-2"]  # another machine
