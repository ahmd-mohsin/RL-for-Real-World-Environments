"""The Claude Code builder client (actors/claude_code.py) and its registry rules (factory/models.py, generators/swarm.py;
docs/research/swarm-pipeline/08). Every call goes to a fake ``claude`` binary (a Python script in tmp): no network, no plan.

- JSON roles: the CLI flags (print mode, JSON output, no tools, safe mode, no session persistence, the system prompt, the
  schema), stdin carries the conversation, a schema-breaking reply is retried with the error fed back, exhausted retries
  raise ``LLMOutputError``, a plan limit is ``BudgetExceeded``, the API-equivalent budget stops the next call, the child
  never sees ``ANTHROPIC_API_KEY``, receipts hold the plan accounting and land in a JSONL file.
- Agentic roles: ``run_session`` runs in the given directory with the restricted, no-prompt permission flags and an
  allowed list confined to file tools, read/commit git and ``docker exec <the sandbox>``; a ``SessionPool`` state backed by
  this client captures the commits the agent made in its private worktree.
- The registry: ``builder=`` puts the builder roles on Claude Code; a Claude Code client (or the builder's family) on a
  reviewer, auditor, hacker, second verifier, realism or tie-break role raises ``SameFamilyError``; Claude Code is never a
  solver rung; the plan usage reaches ``describe()`` and a run receipt."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.actors.claude_code import MODEL_ID, ClaudeCodeClient, render_messages
from crucible.actors.llm import BudgetExceeded, LLMError, LLMOutputError
from crucible.factory import models
from crucible.factory.models import CLAUDE_BUILDER, ModelBook, SameFamilyError, family_of, parse_spec

FAKE = r'''#!{python}
import json, os, subprocess, sys
log = os.environ["FAKE_CLAUDE_LOG"]
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
stdin = sys.stdin.read()
n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
with open(log, "a") as fh:
    fh.write(json.dumps({{"argv": sys.argv[1:], "stdin": stdin, "cwd": os.getcwd(), "api_key": "ANTHROPIC_API_KEY" in os.environ}}) + "\n")
base = {{"type": "result", "subtype": "success", "is_error": False, "duration_ms": 1200, "duration_api_ms": 1100, "num_turns": 2,
        "session_id": "s-%d" % n, "total_cost_usd": 0.02, "usage": {{"input_tokens": 10, "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 50, "output_tokens": 20}}, "modelUsage": {{"claude-opus-5-5": {{"costUSD": 0.02}}}}}}
if mode == "ok":
    base["result"] = json.dumps({{"milestones": 3, "ok": True}})
elif mode == "bad_then_ok":
    base["result"] = json.dumps({{"milestones": "three"}}) if n == 0 else json.dumps({{"milestones": 3, "ok": True}})
elif mode == "always_bad":
    base["result"] = "I think the answer is three."
elif mode == "plan_limit":
    base.update(is_error=True, subtype="error_during_execution", result="Claude AI usage limit reached|1792500000")
elif mode == "crash":
    sys.stderr.write("boom\n"); sys.exit(3)
elif mode == "session":
    with open("feature.py", "w") as fh:
        fh.write("def mul(a, b):\n    return a * b\n")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "m1: mul"], check=True)
    base.update(result="Added mul in one commit.", num_turns=7)
elif mode == "session_give_up":
    base.update(result="I looked around.\nGIVE UP: the plan names a module that does not exist")
print(json.dumps(base))
'''


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("CRUCIBLE_CLAUDE_BIN", str(exe))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-key")
    return log


def calls(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


SCHEMA = {"type": "object", "properties": {"milestones": {"type": "integer"}, "ok": {"type": "boolean"}}, "required": ["milestones", "ok"]}


# ---- JSON roles -----------------------------------------------------------------------------------------------------------
async def test_json_call_flags_stdin_receipt_and_no_dollars(fake: Path, tmp_path: Path) -> None:
    c = ClaudeCodeClient(receipts_path=tmp_path / "receipts.jsonl", role="planner")
    val = await c.complete_json([{"role": "system", "content": "You plan."}, {"role": "user", "content": "Plan 3 milestones."}], schema=SCHEMA)
    assert val == {"milestones": 3, "ok": True}
    (call,) = calls(fake)
    a = call["argv"]
    assert a[:3] == ["-p", "--output-format", "json"] and a[a.index("--model") + 1] == "opus"
    for flag in ("--no-session-persistence", "--safe-mode", "--strict-mcp-config"):
        assert flag in a
    assert a[a.index("--tools") + 1] == "" and a[a.index("--system-prompt") + 1] == "You plan."
    assert json.loads(a[a.index("--json-schema") + 1]) == SCHEMA
    assert call["stdin"] == "Plan 3 milestones." and call["api_key"] is False  # the plan pays: the API key never reaches the child
    assert c.model == MODEL_ID == "claude-code:opus-5.5" and c.family == "anthropic"
    assert c.spend.usd == 0.0 and c.spend.calls == 1 and c.spend.prompt_tokens == 160 and c.spend.completion_tokens == 20
    r = c.receipts[0]
    assert (r["num_turns"], r["duration_ms"], r["total_cost_usd"], r["session_id"], r["role"]) == (2, 1200, 0.02, "s-0", "planner")
    assert r["model_usage"] == {"claude-opus-5-5": {"costUSD": 0.02}}
    assert json.loads((tmp_path / "receipts.jsonl").read_text())["session_id"] == "s-0"
    pu = c.plan_usage()
    assert pu["calls"] == 1 and pu["turns"] == 2 and pu["cost_equiv_usd"] == pytest.approx(0.02) and pu["cache_write_tokens"] == 100


async def test_schema_breaking_reply_is_retried_with_the_error(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "bad_then_ok")
    c = ClaudeCodeClient(json_retries=2)
    assert await c.complete_json([{"role": "user", "content": "Plan."}], schema=SCHEMA) == {"milestones": 3, "ok": True}
    first, second = calls(fake)
    assert "breaks the schema" in second["stdin"] and "[assistant]" in second["stdin"] and '"three"' in second["stdin"]
    assert first["argv"][first["argv"].index("--system-prompt") + 1] == "Reply with one JSON object and nothing else."


async def test_exhausted_retries_raise_output_error(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "always_bad")
    c = ClaudeCodeClient(json_retries=1)
    with pytest.raises(LLMOutputError, match="no valid JSON after 2 tries"):
        await c.complete_json([{"role": "user", "content": "Plan."}])
    assert len(calls(fake)) == 2


async def test_plan_limit_is_budget_and_a_crash_is_infra(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "plan_limit")
    with pytest.raises(BudgetExceeded, match="plan limit"):
        await ClaudeCodeClient().complete_json([{"role": "user", "content": "x"}])
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "crash")
    with pytest.raises(LLMError, match="exit 3 without a JSON result") as ei:
        await ClaudeCodeClient().complete([{"role": "user", "content": "x"}])
    assert ei.value.failure_kind == "provider_unavailable"


async def test_api_equivalent_budget_caps_calls(fake: Path) -> None:
    c = ClaudeCodeClient(budget_usd=0.03)
    await c.complete_json([{"role": "user", "content": "x"}])
    assert calls(fake)[0]["argv"][calls(fake)[0]["argv"].index("--max-budget-usd") + 1] == "0.0300"
    await c.complete_json([{"role": "user", "content": "x"}])  # 0.02 spent: 0.01 left
    assert calls(fake)[1]["argv"][calls(fake)[1]["argv"].index("--max-budget-usd") + 1] == "0.0100"
    with pytest.raises(BudgetExceeded):
        await c.complete_json([{"role": "user", "content": "x"}])
    assert len(calls(fake)) == 2


async def test_openai_tools_are_refused(fake: Path) -> None:
    with pytest.raises(LLMError, match="run_session"):
        await ClaudeCodeClient().complete([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "f"}}])
    assert calls(fake) == []


def test_render_messages_keeps_a_retry_transcript() -> None:
    sysm, prompt = render_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"},
                                    {"role": "user", "content": [{"type": "text", "text": "again"}]}])
    assert sysm == "S" and prompt == "[user]\nU\n\n[assistant]\nA\n\n[user]\nagain"


# ---- agentic roles --------------------------------------------------------------------------------------------------------
def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run("git init -q -b main && echo 'x = 1' > a.py && git add -A && git -c user.email=a@b -c user.name=a commit -qm A && git tag A",
                   shell=True, cwd=path, check=True, capture_output=True)
    return path


async def test_run_session_flags_cwd_and_result(fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "session")
    ws = _repo(tmp_path / "ws")
    c = ClaudeCodeClient()
    res = await c.run_session("Add mul.", cwd=ws, add_dirs=[tmp_path / "handoff"], sandbox="crucible-p1-abc")
    (call,) = calls(fake)
    a = call["argv"]
    assert call["cwd"] == str(ws.resolve()) and call["stdin"] == "Add mul."
    for flag in ("--restricted", "--safe-mode", "--no-session-persistence"):
        assert flag in a
    assert a[a.index("--permission-mode") + 1] == "acceptEdits" and a[a.index("--permission-prompts") + 1] == "none"
    assert a[a.index("--tools") + 1] == "Read,Edit,Write,Glob,Grep,Bash" and a[a.index("--add-dir") + 1] == str(tmp_path / "handoff")
    allowed = a[a.index("--allowed-tools") + 1: a.index("--add-dir")]
    assert "Bash(docker exec crucible-p1-abc *)" in allowed and "Bash(git commit *)" in allowed
    assert not [x for x in allowed if x.startswith("Bash(") and not x.startswith(("Bash(git ", "Bash(docker exec crucible-p1-abc "))]
    assert res.stop == "agent_done" and res.turns == 7 and res.tokens == 180 and c.plan.sessions == 1
    assert git_out(ws, "log", "--format=%s", "-1").strip() == "m1: mul"


def git_out(ws: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=ws, capture_output=True, text=True, check=True).stdout


async def test_a_session_pool_state_runs_the_claude_builder_and_captures_its_commits(fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crucible.factory.handoff import TaskRepo
    from crucible.factory.machine import MachineRun
    from crucible.factory.sessions import SessionPool

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "session")
    src = _repo(tmp_path / "src")
    m = {"states": [{"id": "oracle", "entry": True, "role": "oracle_builder", "mutates": True, "recovery": "reconcile", "max_entries": 1,
                     "inputs": [{"name": "spec", "type": "json", "from": "params.spec"}],
                     "session": {"brief": "Build the oracle, one commit per milestone.", "captures": [{"output": "oracle", "kind": "commits"}]},
                     "outputs": [{"name": "oracle", "type": "artifact"}]},
                    {"id": "done", "terminal": "done"}], "transitions": [{"from": "oracle", "to": "done"}]}
    prm = {"spec": {"request": "Add mul."}, "repo": str(src), "base": "A"}
    book = ModelBook(builder=CLAUDE_BUILDER)
    repo = TaskRepo(tmp_path / "fam", "calc")
    pool = SessionPool(tmp_path / "factory", repo=repo, machine=m, run_id="r", params=prm, kernel="local", llm_for=book.client)
    run = MachineRun(m, run_dir=tmp_path / "run", run_id="r", params=prm, handlers={"session": pool.handler}, models=book.roles, **pool.machine_kwargs())
    res = await run.run()
    assert res.status == "done", (res.reason, res.decision_request)
    assert res.outputs["oracle"]["oracle"]["subjects"] == ["m1: mul"]
    (call,) = calls(fake)
    assert call["cwd"] == str(pool.workspaces["oracle"].path) and "--restricted" in call["argv"]
    assert "Build the oracle" in call["stdin"] and "GIVE UP:" in call["stdin"] and "rlm.finish" not in call["stdin"]
    rec = json.loads(next((pool.dir / "oracle").glob("*/session.json")).read_text())
    assert rec["agent"] == "claude_code" and rec["plan"]["num_turns"] == 7 and rec["usd"] == 0.0
    assert book.plan_usage()["total"]["sessions"] == 1 and book.describe()["plan_usage"]["by_role"]["oracle_builder"]["calls"] == 1


async def test_a_give_up_reply_fails_the_state(fake: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from crucible.factory.handoff import TaskRepo
    from crucible.factory.machine import MachineRun
    from crucible.factory.sessions import SessionPool

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "session_give_up")
    src = _repo(tmp_path / "src")
    m = {"states": [{"id": "oracle", "entry": True, "role": "oracle_builder", "mutates": True, "recovery": "reconcile", "max_entries": 1, "retries": 0,
                     "inputs": [{"name": "spec", "type": "json", "from": "params.spec"}],
                     "session": {"captures": [{"output": "oracle", "kind": "commits"}]}, "outputs": [{"name": "oracle", "type": "artifact"}]},
                    {"id": "done", "terminal": "done"}], "transitions": [{"from": "oracle", "to": "done"}]}
    prm = {"spec": {"request": "Add mul."}, "repo": str(src), "base": "A"}
    pool = SessionPool(tmp_path / "factory", repo=TaskRepo(tmp_path / "fam", "calc"), machine=m, run_id="r", params=prm, kernel="local",
                       llm_for=lambda role: ClaudeCodeClient(role=role))
    res = await MachineRun(m, run_dir=tmp_path / "run", run_id="r", params=prm, handlers={"session": pool.handler}, **pool.machine_kwargs()).run()
    assert res.status != "done" and "gave up: the plan names a module that does not exist" in (res.reason or "")


# ---- the registry ---------------------------------------------------------------------------------------------------------
def test_builder_roles_go_to_claude_code_and_reviewers_stay_cross_family() -> None:
    b = ModelBook(builder=CLAUDE_BUILDER, preset="openrouter")  # builder= reassigns the builder roles of any preset
    for r in ("planner", "author", "oracle_author", "verifier_writer"):
        assert b.spec(r).ref == "claude-code:opus-5.5" and b.spec(r).family == "anthropic"
    assert b.spec("reviewer").family == "deepseek" and b.spec("tie_break").family == "glm"
    c = b.client("planner")
    assert isinstance(c, ClaudeCodeClient) and c.role == "planner" and b.client("planner") is c
    assert parse_spec("claude-code:opus-5.5").family == "anthropic" == family_of("anthropic/claude-sonnet-5.5") == family_of("claude-code:opus-5.5")
    from crucible.factory import driver

    assert b.check(driver.driver_machine()) == []  # the machine's own cross-family rule holds under the builder assignment


@pytest.mark.parametrize("role", list(models.INDEPENDENT_ROLES))
def test_claude_code_on_an_independent_role_raises(role: str) -> None:
    with pytest.raises(SameFamilyError, match=role):
        ModelBook({role: CLAUDE_BUILDER})
    with pytest.raises(SameFamilyError, match="shares family 'anthropic'"):
        ModelBook({role: "openrouter:anthropic/claude-sonnet-5.5"})  # the default builders are Claude
    with pytest.raises(SameFamilyError, match="shares family 'qwen' with the student"):
        ModelBook({role: "tinker:Qwen/Qwen3.5-397B-A17B:peft:262144"})  # never a Qwen model on the Qwen student's work
    # the v1 preset (Qwen authors, no Claude builder): a Claude reviewer is allowed
    assert ModelBook({role: "openrouter:anthropic/claude-sonnet-5.5"}, preset="openrouter").spec(role).family == "anthropic"


def test_the_default_map_is_claude_plus_tinker_and_holds_the_family_rule() -> None:
    b = ModelBook()
    fam = b.families()
    assert {fam[r] for r in ("planner", "oracle_builder", "verifier_writer", "prompt_author", "author")} == {"anthropic"}
    assert (fam["second_verifier"], fam["reviewer"], fam["auditor"], fam["judge"]) == ("deepseek", "glm", "glm", "glm")
    assert {fam["hacker"], fam["hacker_2"]} == {"deepseek", "nemotron"} and {fam["rewriter_deepseek"], fam["rewriter_glm"]} == {"deepseek", "glm"}
    assert b.spec("reviewer").ref == "tinker:zai-org/GLM-5.3:peft:262144" and b.spec("judge").model == "zai-org/GLM-5.3-Flash:peft:262144"
    assert [s.ref for s in b.solvers] == ["tinker:Qwen/Qwen3.5-4B", "tinker:Qwen/Qwen3.5-9B", "tinker:Qwen/Qwen3.8-27B", "tinker:Qwen/Qwen3.6-35B-A3B"]
    assert [s.ref for s in b.frontier] == ["tinker:Qwen/Qwen3.5-397B-A17B:peft:262144", "tinker:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"]
    assert all(s.provider in ("claude-code", "tinker") for s in [*b.roles.values(), *b.solvers, *b.frontier])  # OpenRouter unused
    assert b.family_problems() == []
    from crucible.factory import driver

    assert b.check(driver.driver_machine()) == []
    with pytest.raises(SameFamilyError, match="span >= 2 families"):
        ModelBook({"hacker_2": models.DEEPSEEK})


def test_tinker_roles_build_through_the_registered_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "_BACKENDS", {})
    b = ModelBook()
    with pytest.raises(KeyError, match="train/bridges/tinker"):
        b.client("reviewer")  # nothing registered: a clear error, and no call
    made = []

    class Fake:
        def __init__(self, model: str, **kw):
            made.append((model, kw))
            self.model, self.spend = model, type("S", (), {"usd": 0.0})()

    models.register_backend("tinker", Fake, final=True, doc="fake TinkerSampler")
    c = b.client("reviewer")
    assert isinstance(c, Fake) and made[-1] == ("zai-org/GLM-5.3:peft:262144", {"role": "reviewer"})
    (spec, _sc), *_ = b.solver_clients()
    assert spec.ref == "tinker:Qwen/Qwen3.5-4B" and made[1] == ("Qwen/Qwen3.5-4B", {"role": "solver", "thinking": False})
    assert made[-1][0] == "Qwen/Qwen3.6-35B-A3B"
    assert len(b.frontier_clients()) == 2 and b.observed() == 0.0


def test_client_refuses_a_later_misassignment_and_claude_is_never_a_solver() -> None:
    b = ModelBook(builder=CLAUDE_BUILDER)
    b.roles["auditor"] = parse_spec(CLAUDE_BUILDER)
    with pytest.raises(SameFamilyError):
        b.client("auditor")
    with pytest.raises(ValueError, match="never a solver"):
        ModelBook(solvers=[CLAUDE_BUILDER])


def test_from_config_reads_the_builder() -> None:
    from crucible.core.config import Config

    b = ModelBook.from_config(Config(backends={"factory": {"builder": CLAUDE_BUILDER}}), builder_kwargs={"effort": "high"})
    assert b.spec("verifier_writer").provider == "claude-code" and b.client("verifier_writer").effort == "high"


def test_swarm_overrides_refuse_claude_on_reviewers() -> None:
    from crucible.factory.generators.swarm import checked_overrides

    claude = ClaudeCodeClient()

    class Deep:
        model = "deepseek/deepseek-v4-flash"

    cur = {"reviewer": Deep(), "hacker": Deep(), "second_verifier": Deep(), "planner": Deep()}
    assert checked_overrides(cur, {"planner": claude, "oracle_builder": claude}) == {"planner": claude, "oracle_builder": claude}
    with pytest.raises(SameFamilyError, match="reviewer"):
        checked_overrides(cur, {"planner": claude, "reviewer": claude})


def test_run_receipt_keeps_plan_usage_beside_dollars(tmp_path: Path) -> None:
    from crucible.factory.receipts import run_receipt

    (tmp_path / "journal.jsonl").write_text("")

    class R:
        usd, tokens, seconds = 0.0, 10, 1.0

    class Run:
        run_id, hash = "r", "h"

        class journal:
            path = tmp_path / "journal.jsonl"

        def result(self) -> R:
            return R()

    rec = run_receipt(Run(), scope="s", not_verified=["x"], plan_usage={"total": {"calls": 2}})  # type: ignore[arg-type]
    assert rec["spend"]["plan"] == {"total": {"calls": 2}} and rec["spend"]["usd"] == 0.0
