"""P1 machine v2 (factory/machine.py; docs/research/swarm-pipeline/00 §4, 01 facts 5-7): the infra/task split with one
verified free respawn, capture failures owned by the producer, the declared recovery posture (validated; reset restores,
reconcile hands over the diff and findings), V7 against Prime at 7cbffa77b, and dollar + token budgets per state and run."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.actors.llm import LLMError, LLMOutputError, book_scoped
from crucible.factory.driver import driver_machine
from crucible.factory.machine import (
    InfraError,
    MachineError,
    MachineRun,
    Settle,
    StateContext,
    artifact_digest,
    book_tokens,
    canonicalize,
    is_infra_error,
    machine_hash,
    validate_machine,
)


async def no_sleep(_s: float) -> None:
    return None


def chain(*states: dict[str, Any], run: dict[str, Any] | None = None) -> dict[str, Any]:
    ids = [s["id"] for s in states]
    trans = [{"from": a, "to": b} for a, b in itertools.pairwise(ids)] + [{"from": ids[-1], "to": "done"}]
    return {"run": run or {"max_transitions": 20}, "states": [{**states[0], "entry": True}, *states[1:], {"id": "done", "terminal": "done"}],
            "transitions": trans}


class MemTree:
    """An in-memory ``MutableTree``: files as a dict."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {"a.py": "x = 1\n"}
        self.snaps: dict[str, dict[str, str]] = {}
        self.restores = 0

    async def snapshot(self, label: str) -> str:
        tok = f"s{len(self.snaps)}"
        self.snaps[tok] = dict(self.files)
        return tok

    async def restore(self, token: str) -> None:
        self.files = dict(self.snaps[token])
        self.restores += 1

    async def diff(self, token: str) -> str:
        old = self.snaps[token]
        return "\n".join(f"{k}: {old.get(k)!r} -> {v!r}" for k, v in sorted(self.files.items()) if old.get(k) != v)


# ---- the validator ---------------------------------------------------------------------------------------------------------
def test_mutating_state_must_declare_recovery_and_sessions_must_declare_mutation() -> None:
    m = chain({"id": "build", "fn": "build", "mutates": True, "outputs": [{"name": "out", "type": "text"}]})
    assert any("must declare recovery" in e for e in validate_machine(m))
    m["states"][0]["recovery"] = "rebuild"
    assert any("recovery must be one of" in e for e in validate_machine(m))
    m["states"][0]["recovery"] = "reset"
    assert validate_machine(m) == []
    sess = chain({"id": "w", "role": "world_expert", "session": {"captures": [{"output": "world", "kind": "json", "path": "w.json"}]},
                  "outputs": [{"name": "world", "type": "json"}]})
    assert any("declare mutates: true" in e for e in validate_machine(sess))
    sess["states"][0].update(mutates=True, recovery="reset")
    assert validate_machine(sess) == []
    sess["states"][0]["session"]["read_only"] = True  # a fresh tree per attempt needs no posture...
    sess["states"][0].pop("mutates")
    sess["states"][0].pop("recovery")
    assert validate_machine(sess) == []


def test_session_block_rules() -> None:
    def m(**sess: Any) -> dict[str, Any]:
        return chain({"id": "v", "role": "verifier_writer", "mutates": True, "recovery": "reset", "session": sess,
                      "outputs": [{"name": "tests", "type": "artifact"}, {"name": "note", "type": "json"}]})

    ok = m(captures=[{"output": "tests", "kind": "dir", "path": "tests/hidden"}, {"output": "note", "kind": "json", "path": "n.json"}])
    assert validate_machine(ok) == []
    errs = validate_machine(m(captures=[{"output": "tests", "kind": "json", "path": "t.json"}]))
    assert any("fills a json port" in e for e in errs) and any("['note'] have no capture" in e for e in errs)
    assert any("unknown keys" in e for e in validate_machine(m(captures=ok["states"][0]["session"]["captures"], cwd="/")))
    assert any("sessions are offline" in e for e in validate_machine(m(captures=ok["states"][0]["session"]["captures"], network=True)))
    assert any("relative to the worktree" in e for e in validate_machine(
        m(captures=[{"output": "tests", "kind": "dir", "path": "../x"}, {"output": "note", "kind": "json", "path": "n.json"}])))
    fn = chain({"id": "v", "fn": "v", "session": {}, "outputs": []})
    assert any("has no role" in e for e in validate_machine(fn))


def test_budget_fields_validated_and_v1_hash_stable() -> None:
    m = chain({"id": "a", "fn": "a", "budget_usd": 0, "budget_tokens": 1.5, "outputs": []}, run={"budget_tokens": -1})
    errs = validate_machine(m)
    assert any("budget_usd must be a positive" in e for e in errs) and any("budget_tokens must be a positive integer" in e for e in errs)
    assert any("run budget_tokens" in e for e in errs)
    c = canonicalize(driver_machine())  # a v1 machine gains no P1 keys: its hash (and its journals) survive the upgrade
    assert not any("mutates" in s or "recovery" in s for s in c["states"])
    assert machine_hash(c) == machine_hash(canonicalize(driver_machine()))


# ---- infra vs task --------------------------------------------------------------------------------------------------------
def test_infra_classifier() -> None:
    assert is_infra_error(InfraError("x"))
    assert is_infra_error(RuntimeError("Child worker unreachable after 3 probes"))  # Prime's marker, verbatim
    assert is_infra_error(RuntimeError("Error response from daemon: No such container: crucible-p1-x"))
    assert is_infra_error(LLMError("provider 503"))  # provider_unavailable
    assert not is_infra_error(LLMOutputError("not JSON"))  # the model's malformed reply is the task's
    assert not is_infra_error(ValueError("assertion failed"))
    assert not is_infra_error(None)


async def test_verified_dead_worker_respawns_once_for_free(tmp_path: Path) -> None:
    probes: list[str] = []
    calls: list[StateContext] = []

    async def probe(sid: str, err: BaseException) -> str:
        probes.append(sid)
        return "dead"

    async def work(ctx: StateContext) -> dict:
        calls.append(ctx)
        if len(calls) == 1:
            raise InfraError("child worker unreachable", "crucible:env_container_lost")
        return {"out": "ok"}

    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    m = chain({"id": "work", "fn": "work", "max_entries": 1, "retries": 0, "outputs": [{"name": "out", "type": "text"}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"work": work}, infra_probe=probe, sleep=sleep)
    res = await run.run()
    assert res.status == "done" and res.entries == {"work": 1}  # retries 0 and max_entries 1: a counted attempt would have paused
    assert probes == ["work"] and slept == [run.infra_quiet_s]  # re-checked after the quiet period (F:2990-3017)
    assert [c.attempt for c in calls] == [1, 1] and [c.respawned for c in calls] == [False, True]
    evs = [json.loads(x) for x in run.journal.path.read_text().splitlines()]
    assert [e["verdict"] for e in evs if e["event"] == "infra_respawn"] == ["dead"]
    assert not [e for e in evs if e["event"] == "fail"]  # the dead attempt is no task failure
    assert [e.get("respawn", False) for e in evs if e["event"] == "spend"] == [False, True]  # its spend row stays


async def test_second_infra_death_takes_retries_then_pauses_as_infra_even_under_fail_fast(tmp_path: Path) -> None:
    n = {"calls": 0}

    async def work(ctx: StateContext) -> dict:
        n["calls"] += 1
        raise InfraError("supervisor link lost")

    m = chain({"id": "work", "fn": "work", "retries": 1, "outputs": [{"name": "out", "type": "text"}]}, run={"failure_policy": "fail_fast"})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"work": work}, sleep=no_sleep)
    res = await run.run()
    assert n["calls"] == 3  # attempt 1, its free respawn, attempt 2 (the respawn is once per entry run)
    assert res.status == "paused" and res.decision_request["kind"] == "infra"  # V8: masked, a decision; fail_fast is for task failures
    evs = [json.loads(x) for x in run.journal.path.read_text().splitlines()]
    assert [e["blame"] for e in evs if e["event"] == "fail"] == ["infra", "infra"]
    res = await run.resume()  # resume re-runs the held entry with fresh retries and a fresh free respawn
    assert n["calls"] == 6 and res.decision_request["kind"] == "infra" and res.entries == {"work": 1}


async def test_alive_probe_reruns_the_same_session(tmp_path: Path) -> None:
    seen: list[bool] = []

    async def probe(sid: str, err: BaseException) -> str:
        return "alive"

    async def work(ctx: StateContext) -> dict:
        seen.append(ctx.respawned)
        if len(seen) == 1:
            raise LLMError("upstream connection reset")
        return {"out": "ok"}

    m = chain({"id": "work", "fn": "work", "outputs": [{"name": "out", "type": "text"}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"work": work}, infra_probe=probe, sleep=no_sleep)
    assert (await run.run()).status == "done"
    evs = [json.loads(x) for x in run.journal.path.read_text().splitlines()]
    assert [e["verdict"] for e in evs if e["event"] == "infra_respawn"] == ["alive"] and seen == [False, True]


async def test_task_failure_is_never_respawned(tmp_path: Path) -> None:
    n = {"calls": 0}

    async def work(ctx: StateContext) -> dict:
        n["calls"] += 1
        raise LLMOutputError("output is not valid JSON")

    m = chain({"id": "work", "fn": "work", "outputs": [{"name": "out", "type": "text"}]})
    res = await MachineRun(m, run_dir=tmp_path / "r", handlers={"work": work}, sleep=no_sleep).run()
    assert n["calls"] == 1 and res.decision_request["kind"] == "failed"


# ---- capture failures belong to the producer ------------------------------------------------------------------------------
async def test_missing_artifact_fails_the_producer_after_one_recapture_and_keeps_needs_verify(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    entered: list[str] = []
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)
        if len(slept) == 2:  # the second attempt's file lands inside the re-capture window
            (root / "t2.py").write_text("ok")

    async def producer(ctx: StateContext) -> dict:
        entered.append(f"producer{ctx.attempt}")
        name = f"t{ctx.attempt}.py"
        digest = "0" * 64 if ctx.attempt == 1 else __import__("hashlib").sha256(b"ok").hexdigest()
        return {"tests": {"path": name, "sha256": digest}}

    async def consumer(ctx: StateContext) -> dict:
        entered.append("consumer")
        assert (root / ctx.inputs["tests"]["path"]).read_text() == "ok"
        return {"v": "ok"}

    m = chain({"id": "producer", "fn": "producer", "retries": 1, "outputs": [{"name": "tests", "type": "artifact"}]},
              {"id": "consumer", "fn": "consumer", "inputs": [{"name": "tests", "type": "artifact", "from": "producer.tests"}],
               "outputs": [{"name": "v", "type": "text"}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"producer": producer, "consumer": consumer}, artifact_root=root, sleep=sleep)
    res = await run.run()
    assert res.status == "done" and entered == ["producer1", "producer2", "consumer"]
    assert len(slept) == 2  # one bounded re-capture window per attempt
    assert len(res.needs_verify) == 1 and res.needs_verify[0]["attempt"] == 1  # the failed capture stays provisional


async def test_missing_port_fails_producer_not_consumer(tmp_path: Path) -> None:
    async def producer(ctx: StateContext) -> dict:
        return {}

    async def consumer(ctx: StateContext) -> dict:
        raise AssertionError("never entered")

    m = chain({"id": "producer", "fn": "producer", "outputs": [{"name": "plan", "type": "json"}]},
              {"id": "consumer", "fn": "consumer", "inputs": [{"name": "plan", "type": "json", "from": "producer.plan"}], "outputs": []})
    res = await MachineRun(m, run_dir=tmp_path / "r", handlers={"producer": producer, "consumer": consumer}, sleep=no_sleep).run()
    assert res.decision_request["kind"] == "failed" and res.decision_request["state"] == "producer" and "consumer" not in res.entries
    assert res.needs_verify[0]["error"].startswith("missing declared output")


async def test_artifact_changed_after_capture_pauses_on_the_producer(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    runs = {"p": 0}

    async def producer(ctx: StateContext) -> dict:
        runs["p"] += 1
        f = root / f"out{runs['p']}.txt"
        f.write_text(f"v{runs['p']}")
        return {"a": {"path": f.name, "sha256": artifact_digest(f)}}

    async def tamper(ctx: StateContext) -> dict:
        (root / "out1.txt").write_text("edited after capture")
        return {}

    async def consumer(ctx: StateContext) -> dict:
        return {"v": (root / ctx.inputs["a"]["path"]).read_text()}

    m = chain({"id": "producer", "fn": "producer", "max_entries": 2, "outputs": [{"name": "a", "type": "artifact"}]},
              {"id": "tamper", "fn": "tamper", "max_entries": 2, "outputs": []},
              {"id": "consumer", "fn": "consumer", "max_entries": 2, "inputs": [{"name": "a", "type": "artifact", "from": "producer.a"}],
               "outputs": [{"name": "v", "type": "text"}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"producer": producer, "tamper": tamper, "consumer": consumer}, artifact_root=root,
                     sleep=no_sleep)
    res = await run.run()
    assert res.decision_request["kind"] == "capture_failed" and res.decision_request["state"] == "producer"
    assert "does not hash" in res.reason
    res = await run.resume()
    assert res.status == "done" and res.outputs["consumer"]["v"] == "v2" and runs["p"] == 2


# ---- recovery posture ----------------------------------------------------------------------------------------------------
async def test_reset_restores_the_snapshot_before_every_rerun(tmp_path: Path) -> None:
    tree = MemTree()
    seen: list[dict[str, Any] | None] = []

    async def build(ctx: StateContext) -> dict:
        seen.append(ctx.recovery)
        assert tree.files["a.py"] == "x = 1\n"  # every attempt starts from the entry's snapshot
        tree.files["a.py"] = "x = 2  # half done\n"
        if ctx.attempt < 3:
            raise ValueError("tests fail")
        return {"out": "built"}

    m = chain({"id": "build", "fn": "build", "mutates": True, "recovery": "reset", "retries": 2, "outputs": [{"name": "out", "type": "text"}]})
    res = await MachineRun(m, run_dir=tmp_path / "r", handlers={"build": build}, trees={"build": tree}).run()
    assert res.status == "done" and tree.restores == 2 and seen[0] is None
    assert seen[1]["mode"] == "reset" and seen[1]["findings"] == ["ValueError: tests fail"]


async def test_reconcile_hands_the_diff_and_findings_to_the_retry_and_to_a_resumed_entry(tmp_path: Path) -> None:
    tree = MemTree()
    seen: list[dict[str, Any] | None] = []

    async def build(ctx: StateContext) -> dict:
        seen.append(ctx.recovery)
        tree.files["a.py"] += f"# attempt {len(seen)}\n"
        raise ValueError(f"failure {len(seen)}")

    m = chain({"id": "build", "fn": "build", "mutates": True, "recovery": "reconcile", "retries": 1, "outputs": [{"name": "out", "type": "text"}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"build": build}, trees=lambda sid: tree)
    res = await run.run()
    assert res.decision_request["kind"] == "failed" and tree.restores == 0  # reconcile never resets
    assert seen[0] is None and seen[1]["mode"] == "reconcile" and "attempt 1" in seen[1]["diff"]
    assert seen[1]["findings"] == ["ValueError: failure 1"]
    # a resumed run (a new process: replayed from the journal) keeps the entry's snapshot and its earlier failures
    run2 = MachineRun(m, run_dir=tmp_path / "r", handlers={"build": build}, trees=lambda sid: tree)
    await run2.resume()
    assert seen[2]["mode"] == "reconcile" and "attempt 2" in seen[2]["diff"] and "ValueError: failure 2" in seen[2]["findings"]
    snaps = [json.loads(x) for x in run2.journal.path.read_text().splitlines() if '"snapshot"' in x and '"event": "snapshot"' in x]
    assert len(snaps) == 1  # one snapshot per entry, journaled


async def test_mutating_state_without_a_tree_fails_loudly(tmp_path: Path) -> None:
    async def build(ctx: StateContext) -> dict:
        return {"out": "x"}

    m = chain({"id": "build", "fn": "build", "mutates": True, "recovery": "reset", "outputs": [{"name": "out", "type": "text"}]})
    res = await MachineRun(m, run_dir=tmp_path / "r", handlers={"build": build}).run()
    assert res.decision_request["kind"] == "failed" and "gives it no tree" in res.reason


# ---- V7: exhaustion escalates; Prime ends it done ---------------------------------------------------------------------------
PRIME_7CBFFA77B_EXHAUSTED_LOOP = "done"  # TF:5450-5477 "blocked transition quiesces done"; F:2469-2483 + _finalize F:3515-3541


async def test_exhausted_review_loop_escalates_where_prime_ends_done(tmp_path: Path) -> None:
    """Prime's pr-manager shape (reviewing <-> fixing, MX/pr-manager/MACHINE.md): a reviewer that never approves. At
    7cbffa77b Prime records ``transition_blocked`` and finalises ``done`` with no approval; ours pauses with an
    ``exhausted`` decision and never reaches the done terminal, also after the transition cap is raised."""

    async def review(ctx: StateContext) -> Settle:
        return Settle({"verdict": {"approved": False, "findings": [f"r{ctx.entry}"]}})

    async def fix(ctx: StateContext) -> dict:
        return {"report": {"fixed": ctx.findings}}

    m = {"run": {"max_transitions": 24},
         "states": [{"id": "reviewing", "entry": True, "fn": "review", "max_entries": 4, "outputs": [{"name": "verdict", "type": "json"}]},
                    {"id": "fixing", "fn": "fix", "max_entries": 3, "outputs": [{"name": "report", "type": "json"}]},
                    {"id": "done", "terminal": "done"}],
         "transitions": [{"from": "reviewing", "to": "done", "when": "verdict.approved == true"},
                         {"from": "reviewing", "to": "fixing", "when": "verdict.approved == false", "carry": "verdict.findings"},
                         {"from": "fixing", "to": "reviewing"}]}
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"review": review, "fix": fix})
    res = await run.run()
    assert res.status != PRIME_7CBFFA77B_EXHAUSTED_LOOP
    assert res.status == "paused" and res.decision_request["kind"] == "exhausted" and res.entries == {"reviewing": 4, "fixing": 3}
    assert res.transitions < 24  # exhaustion, not the transition cap, stopped it (Prime's seed claims the cap would)
    res = await run.resume({"action": "approve", "actor": "person", "extra_entries": 1})
    assert res.status == "paused" and res.decision_request["kind"] == "exhausted"  # still no silent done


# ---- budgets --------------------------------------------------------------------------------------------------------------
async def test_state_token_budget_stops_retries_and_resume_grants_more(tmp_path: Path) -> None:
    n = {"calls": 0}
    budgets: list[dict[str, Any]] = []

    async def author(ctx: StateContext) -> dict:
        n["calls"] += 1
        budgets.append(dict(ctx.budget))
        book_tokens(600)
        if n["calls"] < 3:
            raise ValueError("draft fails its lint")
        return {"out": "x"}

    m = chain({"id": "author", "fn": "author", "retries": 5, "budget_tokens": 1000, "outputs": [{"name": "out", "type": "text"}]},
              run={"budget_tokens": 100_000})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"author": author})
    res = await run.run()
    assert n["calls"] == 2 and res.decision_request["kind"] == "state_budget" and res.tokens == 1200
    assert budgets[0] == {"usd": None, "tokens": 1000} and budgets[1]["tokens"] == 400
    res = await run.resume({"action": "revise", "actor": "person", "extra_tokens": 2000})
    assert res.status == "done" and n["calls"] == 3


async def test_state_usd_budget_and_run_token_cap(tmp_path: Path) -> None:
    async def spend(ctx: StateContext) -> Settle:
        book_scoped(0.30)
        if ctx.state_id == "a":
            raise ValueError("again")
        return Settle({"out": "x"}, tokens=900)

    m = chain({"id": "a", "fn": "a", "retries": 9, "budget_usd": 0.5, "outputs": [{"name": "out", "type": "text"}]})
    res = await MachineRun(m, run_dir=tmp_path / "a", handlers={"a": spend}).run()
    assert res.decision_request["kind"] == "state_budget" and abs(res.usd - 0.6) < 1e-9

    m = chain({"id": "b", "fn": "b", "outputs": [{"name": "out", "type": "text"}]}, {"id": "c", "fn": "b", "outputs": [{"name": "out", "type": "text"}]},
              run={"budget_tokens": 500})
    res = await MachineRun(m, run_dir=tmp_path / "b", handlers={"b": spend}).run()
    assert res.status == "stopped" and "token cap" in res.reason and res.entries == {"b": 1}  # Settle.tokens count too


def test_mutating_machine_refused_by_machine_run(tmp_path: Path) -> None:
    m = chain({"id": "build", "fn": "build", "mutates": True, "outputs": []})
    with pytest.raises(MachineError, match="recovery"):
        MachineRun(m, run_dir=tmp_path / "r", handlers={"build": None})  # type: ignore[dict-item]


async def test_a_declared_port_schema_is_checked_at_the_producer(tmp_path: Path) -> None:
    bad = chain({"id": "p", "fn": "p", "outputs": [{"name": "plan", "type": "json", "schema": {"type": "nonsense"}}]})
    assert any("is not a JSON schema" in e for e in validate_machine(bad))
    n = {"calls": 0}

    async def p(ctx: StateContext) -> dict:
        n["calls"] += 1
        return {"plan": {"milestones": []} if n["calls"] == 1 else {"milestones": [{"id": "m1"}]}}

    schema = {"type": "object", "required": ["milestones"], "properties": {"milestones": {"type": "array", "minItems": 1}}}
    m = chain({"id": "p", "fn": "p", "retries": 1, "outputs": [{"name": "plan", "type": "json", "schema": schema}]})
    run = MachineRun(m, run_dir=tmp_path / "r", handlers={"p": p})
    res = await run.run()
    assert res.status == "done" and n["calls"] == 2
    fails = [json.loads(x) for x in run.journal.path.read_text().splitlines() if '"event": "fail"' in x]
    assert "breaks its schema" in fails[0]["error"] and "should be non-empty" in fails[0]["error"]
