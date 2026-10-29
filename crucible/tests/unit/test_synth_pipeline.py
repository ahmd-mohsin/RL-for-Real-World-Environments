"""crucible.synth.pipeline: one end-to-end run with scripted roles on the fixture helpdesk World (M4). No model is called."""

from __future__ import annotations

import json

from crucible.synth.canon import Normalizers
from crucible.synth.pipeline import DomainEnv, RunConfig, Runner, RunRoles, cx_view
from crucible.synth.task import WorldRef
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, const, section, seq

T_RESOLVE = "Resolve ticket T-3 with the id of the knowledge base article on refund policy as the resolution."
T_PRIO = "Set the priority of ticket T-4 to high."
PROPOSALS = {"tasks": [{"skill": "answer from the kb", "instruction": T_RESOLVE, "derive": ["KB-2"]},
                       {"skill": "triage", "instruction": T_PRIO, "derive": []}]}
ORACLES = {T_RESOLVE: {"steps": [{"id": "s1", "tool": "desk.search_kb", "args": {"query": "refund policy"}},
                                 {"id": "s2", "tool": "desk.resolve_ticket", "args": {"ticket_id": "T-3", "resolution": {"$ref": "s1.articles.0.id"}}}]},
           T_PRIO: {"steps": [{"id": "s1", "tool": "desk.set_priority", "args": {"ticket_id": "T-4", "priority": "high"}}]}}
DEAD = {"steps": [{"id": "s1", "tool": "desk.set_priority", "args": {"ticket_id": "T-4", "priority": "urgent"}}]}


def oracle_script():
    tried: set[str] = set()

    def f(msgs, _n):
        task = section(msgs, "task")
        if task == T_PRIO and T_PRIO not in tried:  # the first program is dead: it gets its execution errors back
            tried.add(T_PRIO)
            return DEAD
        return ORACLES[task]

    return f


def alt_script(msgs, _n):
    hist = section(msgs, "history")
    ins = section(msgs, "instruction")
    steps = [{"tool": s["tool"], "args": {k: (v if not isinstance(v, dict) else "KB-2") for k, v in s["args"].items()}} for s in ORACLES[ins]["steps"]]
    done = 0 if isinstance(hist, str) else len(hist)
    return {"call": steps[done]} if done < len(steps) else {"done": True, "final": None}


GOAL = {T_RESOLVE: ("desk.resolve_ticket", {"ticket_id": "T-3", "resolution": "KB-2"}), T_PRIO: ("desk.set_priority", {"ticket_id": "T-4", "priority": "high"}),
        "Ticket T-7": ("desk.resolve_ticket", {"ticket_id": "T-7", "resolution": "KB-3"})}


def panel_script(msgs, _n):
    """A judge that knows each task's goal: valid iff the trace makes exactly that write (and no other write)."""
    tr, task = section(msgs, "trace"), section(msgs, "task")
    goal = next((g for k, g in GOAL.items() if task.startswith(k)), None)
    writes = [x for x in tr if x["tool"] in ("desk.resolve_ticket", "desk.set_priority", "mail.send", "desk.admin_update")] if isinstance(tr, list) else []
    ok = goal is not None and any(x["tool"] == goal[0] and x["args"] == goal[1] and "error" not in x for x in writes) and \
        all(x["tool"] == goal[0] or x["tool"] == "mail.send" for x in writes)
    return {"valid": ok, "confidence": 0.9, "why": ""}


def roles() -> RunRoles:
    return RunRoles(proposer=FakeClient("kimi", const(PROPOSALS)), oracle_programmer=FakeClient("deepseek", oracle_script()),
                    contract_writer=FakeClient("glm", const({"edits": []})), repairer=FakeClient("deepseek", const({"give_up": True})),
                    alt=[("alt-n", FakeClient("nemotron", alt_script)), ("alt-o", FakeClient("openai", alt_script))],
                    atk=[("atk-n", FakeClient("nemotron", seq({"done": True, "final": None})))],
                    panel=[("p-n", FakeClient("nemotron", panel_script)), ("p-o", FakeClient("openai", panel_script))],
                    abstractor=FakeClient("openai", const({"lemmas": []})))


def env() -> DomainEnv:
    return DomainEnv(name="synth-test-m4", description="An IT helpdesk with a knowledge base and mail.", seeds=(0,), normalizers=fx.HD_N,
                     world=lambda s: WorldRef(factory=f"{fx.MOD}:helpdesk_world"))


async def test_end_to_end_run_with_scripted_roles(tmp_path):
    rl = roles()
    r = Runner(env(), rl, RunConfig(n_tasks=2, batch=2, concurrency=2), tmp_path / "run", tmp_path / "archive", run_id="r1")
    rep = await r.run()
    by = {t.id.rsplit("/", 1)[1]: t for t in rep.tasks}
    assert set(by) == {"t01", "t02"}
    assert by["t02"].oracle_attempts == 2 and by["t01"].oracle_attempts == 1  # the dead program got one execution-feedback retry
    assert "returned an error" in rl.oracle_programmer.calls[-1][-1]["content"] or any(
        "returned an error" in m[-1]["content"] for m in rl.oracle_programmer.calls[1:])
    for t in rep.tasks:
        assert t.stage == "refined" and t.status in ("certified", "provisional"), (t.status, t.reasons)
        assert t.classes["ALT"]["passes"] == 2 and t.classes["ATK"]["c"] == 0
        assert all(x.startswith("MUT:upper=") for x in t.reasons)  # short programs: MUT has n < 59, so provisional
    f = rep.funnel
    assert f["proposed"] == 2 and f["oracle_executed"] == 2 and f["contract"] == 2 and f["refined"] == 2 and f["status"] == {"provisional": 2}
    assert rep.calibration and rep.calibration["calibrated"]
    for name in ("proposals.json", "report.json", "report.md", "calibration.json", "tasks/t01/oracle.json", "tasks/t01/contract.json", "tasks/t01/cegis.json"):
        assert (tmp_path / "run" / name).exists(), name
    assert "| t01 |" in (tmp_path / "run" / "report.md").read_text()
    tags = [t for c in (rl.proposer, rl.oracle_programmer, rl.contract_writer) for t in c.tags]
    assert any(t.startswith("propose:") for t in tags) and any(t.startswith("oracle:") for t in tags) and any(t.startswith("contract:") for t in tags)
    # the contract writer saw the skeleton and the effects, never the gate
    cw = rl.contract_writer.calls[0][1]["content"]
    assert "## skeleton" in cw and "## effects" in cw and "grade(" not in cw


async def test_a_defective_contract_is_falsified_and_repaired(tmp_path):
    """The contract writer drops the resolution check (D1): MUT finds it, the repairer restores it, the task ends clean."""
    rl = roles()
    rl.proposer = FakeClient("kimi", const({"tasks": [PROPOSALS["tasks"][0]]}))

    def lax(msgs, _n):
        sk = section(msgs, "skeleton")
        rid = next(p["id"] for p in sk["required"] if p.get("field") == "resolution")
        return {"edits": [{"op": "relax", "target": rid, "pred": {"op": "any"}}]}

    rl.contract_writer = FakeClient("glm", lax)

    def fix(msgs, _n):
        c = section(msgs, "contract")
        for p in c["required"]:
            if p.get("field") == "resolution":
                p["value"] = {"op": "eq", "v": "KB-2"}
        assert section(msgs, "counterexamples")  # it sees the counterexamples (and their reason codes)
        return {"contract": c, "notes": "the resolution must be the refund article"}

    rl.repairer = FakeClient("deepseek", fix)
    r = Runner(env(), rl, RunConfig(n_tasks=1, batch=1, calibrate=False), tmp_path / "run", tmp_path / "archive", run_id="r2")
    rep = await r.run()
    t = rep.tasks[0]
    assert t.first_round_confirmed.get("MUT", 0) > 0 and t.repairs_accepted == 1 and t.status == "provisional", (t.first_round_confirmed, t.reasons)
    assert any(d in t.defects for d in ("D1", "D3", "D9"))
    cx = json.loads((tmp_path / "run" / "tasks" / "t01" / "cegis.json").read_text())
    assert cx["rounds"][0]["repair"] == "accepted" and cx["task"]["version"] == 2
    assert "gate" not in json.dumps(rl.repairer.calls[0][1]["content"]).lower().replace("gate code", "")


def test_counterexample_view_names_the_defect_not_the_code(tmp_path):
    from crucible.synth.falsify.base import Counterexample, Probe

    cx = Counterexample(id="cx-0123456789abcdef", domain="d", task_id="t", task_version=1, cls="TAINT", op="literal", site="literal:x", defect="D5",
                        probe=Probe(cls="TAINT", op="literal", site="literal:x", expect="property", data={"value": "INV-9", "loc": "s1.id"}),
                        reasons_seen=[])
    v = cx_view(cx)
    assert v["defect"] == "D5 HIDDEN" and v["detail"] == {"value": "INV-9", "loc": "s1.id"} and "property" in v["should"]
    assert Normalizers().empty


T_MAIL = ("Ticket T-7 asks how to do a password reset. Find the knowledge base article on password reset, resolve the ticket with the article id "
          "as the resolution, and email the customer the article title.")


async def test_normalizer_edits_apply_first_and_the_skeleton_is_reinferred(tmp_path):
    """An id_canon on the outbox re-keys the inserted mail: the writer's relax edit, written against the first skeleton's
    pattern id, still lands on the re-inferred insert pattern, and the oracle passes the result."""
    rl = roles()
    rl.proposer = FakeClient("kimi", const({"tasks": [{"skill": "kb answer", "instruction": T_MAIL, "derive": ["KB-3"]}]}))
    fx_prog = fx.t3_resolve_email().oracle
    from crucible.synth.roles import program_doc

    rl.oracle_programmer = FakeClient("deepseek", const(program_doc(fx_prog)))

    def writer(msgs, _n):
        sk = section(msgs, "skeleton")
        ins = next(p for p in sk["required"] if p.get("kind") == "insert")
        assert ins["key"] != "*"  # the first skeleton keys the mail by its autoincrement id
        return {"edits": [{"op": "normalize", "normalizers": {"id_canon": [{"app": "mail", "coll": "outbox"}]}},
                          {"op": "relax", "target": ins["id"], "field": "body", "pred": {"op": "regex", "pattern": "(?i).*password reset.*"}}]}

    rl.contract_writer = FakeClient("glm", writer)
    e = env()
    e.normalizers = None
    r = Runner(e, rl, RunConfig(n_tasks=1, batch=1, calibrate=False, max_rounds=1), tmp_path / "run", tmp_path / "archive", run_id="r3")
    await r.survey()
    recs = await r.propose()
    await r.build(recs[0])
    rec, task, _ = r.ready[0]
    ins = next(p for p in task.contract.required if p.kind == "insert")
    assert ins.key == "*" and ins.where["body"].op == "regex" and rec.edits_removed == 0 and task.contract.normalizers.id_canon


def test_remap_patterns_by_address():
    from crucible.synth.contract import Contract, EffectPattern
    from crucible.synth.pipeline import remap_patterns

    old = Contract(required=[EffectPattern(id="r0", app="a", coll="t", key="1", field="s"), EffectPattern(id="r1", app="m", coll="o", key="5", field="*",
                                                                                                          kind="insert")])
    new = Contract(required=[EffectPattern(id="r0", app="m", coll="o", key="*", field="*", kind="insert"), EffectPattern(id="r1", app="a", coll="t", key="1",
                                                                                                                         field="s")])
    assert remap_patterns(old, new) == {"r0": "r1", "r1": "r0"}
