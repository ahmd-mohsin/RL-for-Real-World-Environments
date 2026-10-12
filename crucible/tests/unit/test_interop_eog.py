"""M11: EnterpriseOps-Gym task importer (``interop/eog``; ServiceNow EnterpriseOps-Gym, Apache-2.0).

Synthetic: schema inference from a data-only dump (integer ids become ``INTEGER PRIMARY KEY``, verifier-only tables are
created), the verifier parser and the derived witness, refusal and unsupported-verifier handling. Real (skipped when the
local clone is absent): every task of ``data/revised`` imports except the one whose seed is missing from
``gym_dbs.zip``; each World starts and lists its tools, doing nothing scores 0, and on every task whose verifiers are
single-table the derived ``sql_execute`` witness scores 1, once end to end through the Runner; an HF dataset row
imports like its file; a refusal variant of a real task scores 1 for doing nothing and 0 for touching anything.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import Done, ToolCall
from crucible.interop.eog.importer import app_id_for, import_eog_dir, import_eog_task, infer_schema, load_task, referenced_columns
from crucible.interop.eog.oracle import Underivable, derive, parse_verifier

EOG = Path(__file__).resolve().parents[3] / "RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym"
need_clone = pytest.mark.skipif(not (EOG / "gym_dbs.zip").is_file(), reason="EnterpriseOps-Gym checkout not present")
CSM = "task_20251205_153330_906_a8eea1c0_8c7a6205"  # single-table verifiers: derivable
ITSM = "task_20251212_172511_458_e6427839_47076c15"  # a verifier-only table (incident_sla has no seed rows)
MISSING_SEED = "task_20251209_132736_542_99ba2325_6705caf4"

SEED = """-- Database Seed Data for: t
-- Created: 2025-11-04 11:06:36.586221+00:00
INSERT INTO customer_case (case_id, state, priority, sys_created_on) VALUES
(1, 'new', 'low', datetime('now')), (2, 'closed', 'high', '2025-01-01 00:00:00');
INSERT INTO org (org_id, name) VALUES ('ORG_001', 'TechCorp');
INSERT INTO customer_case (case_id, assigned_to) VALUES (3, 7);
"""


def test_infer_schema_parser_and_derivation():
    extra = {"case_sla": ["case_id", "sla_def_id", "stage"], "customer_case": ["sys_updated_on"]}  # what the import collects from verifiers
    ddl, pk = infer_schema(SEED, extra=extra, now="2025-12-05 15:33:30")
    assert pk == {"customer_case": "case_id", "org": "org_id"}
    assert '"case_id" INTEGER PRIMARY KEY' in ddl and '"org_id" PRIMARY KEY' in ddl and '"assigned_to"' in ddl  # union of inserted columns
    assert 'CREATE TABLE IF NOT EXISTS "case_sla" ("case_id", "sla_def_id", "stage")' in ddl  # queried, never seeded
    assert referenced_columns("SELECT COUNT(*) FROM case_sla cs JOIN customer_case cc ON cs.case_id = cc.case_id WHERE cs.stage = 'x'") == \
        {"case_sla": ["case_id", "stage"], "customer_case": ["case_id"]}
    assert referenced_columns("SELECT state FROM customer_case WHERE case_id = 4 AND note = 'AND x'") == {"customer_case": ["state", "case_id", "note"]}
    assert app_id_for("sn-csm-server") == "csm" and app_id_for("gym-google-drive-mcp") == "google-drive" and app_id_for("sn-hr-internal") == "hr"

    w = parse_verifier("SELECT COUNT(*) AS n FROM customer_case c WHERE c.case_id = 4 AND c.title LIKE '%VPN%' AND c.closed_on IS NULL "
                       "AND sys_updated_on > sys_created_on AND c.prio IN ('high', 'critical');", 1)
    assert w.kind == "exists" and [(p.col, p.op, p.value) for p in w.preds] == [
        ("case_id", "=", 4), ("title", "=", "VPN"), ("closed_on", "=", None), ("sys_updated_on", "col>", None), ("prio", "=", "high")]
    assert parse_verifier("SELECT state FROM t WHERE id = 'it''s'", "closed").preds[-1].value == "closed"
    for bad in ("SELECT COUNT(*) FROM a JOIN b ON a.x = b.x WHERE a.y = 1", "SELECT COUNT(*) FROM t WHERE a = 1 OR b = 2",
                "SELECT COUNT(*) FROM t WHERE a != 1", "SELECT COUNT(*) FROM t WHERE id = (SELECT MAX(id) FROM t)"):
        with pytest.raises(Underivable):
            parse_verifier(bad, 1)
    with pytest.raises(Underivable):
        parse_verifier("SELECT COUNT(*) FROM t WHERE a = 1", 3)

    con = sqlite3.connect(":memory:")
    con.executescript(ddl + "INSERT INTO customer_case (case_id, state, priority) VALUES (1, 'new', 'low'), (2, 'closed', 'high');")
    checks = [("SELECT COUNT(*) FROM customer_case WHERE case_id = 4 AND state = 'in_progress' AND sys_updated_on IS NOT NULL", 1, "equals"),
              ("SELECT priority FROM customer_case WHERE case_id = 4", "critical", "equals"),  # same row: merged with the line above
              ("SELECT COUNT(*) FROM case_sla WHERE case_id = 4 AND sla_def_id = 1 AND stage = 'in_progress'", 1, "equals"),
              ("SELECT COUNT(*) FROM case_sla WHERE case_id = 4 AND sla_def_id = 6", 1, "equals"),  # conflicts on sla_def_id: a second row
              ("SELECT COUNT(*) FROM customer_case WHERE state = 'closed'", 0, "equals"),
              ("SELECT COUNT(*) FROM customer_case WHERE case_id = 1 AND sys_updated_on > '2025-11-18'", 1, "equals")]
    stmts = derive(checks, pk={"customer_case": "case_id"}, now="2025-12-05 15:33:30")
    for _ in range(2):  # self-deciding statements: running the witness twice changes nothing
        for sql, params in stmts:
            con.execute(sql, params)
    for q, expected, _cmp in checks:
        assert con.execute(q).fetchone()[0] == expected, q
    assert con.execute("SELECT COUNT(*) FROM case_sla").fetchone()[0] == 2


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    if not (EOG / "gym_dbs.zip").is_file():
        pytest.skip("EnterpriseOps-Gym checkout not present")
    cache = tmp_path_factory.mktemp("eog")
    tasks, skipped = import_eog_dir(EOG / "data" / "revised", seeds=EOG / "gym_dbs.zip", cache_dir=cache)
    return {t.meta["task_id"]: t for t in tasks}, skipped, cache


async def _play(task, actions, tmp: Path):
    w = await task.open(tmp / "w")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=80), reset=False)
        bad = [e.data for e in w.to_trace().events if e.type == "tool_result" and not (e.data or {}).get("ok", True)]
        return await task.grade(w, tmp / "g"), bad
    finally:
        await w.close()


@need_clone
async def test_real_tasks_import_start_and_list_tools(imported, tmp_path):
    tasks, skipped, _ = imported
    files = sorted((EOG / "data" / "revised").glob("*/*.json"))
    assert len(tasks) + len(skipped) == len(files) and len(tasks) >= 12
    assert list(skipped) == [str(next(f for f in files if f.stem == MISSING_SEED))] and "is not in" in next(iter(skipped.values()))
    t = tasks[CSM]
    assert t.spec_doc.clock.t0 == "2025-12-05T15:33:30Z" and t.meta["seeds"] == {"csm": "db_1762254390925_u5icw4thh.sql"}
    assert len(t.assertions) == 11 and {a.app for a in t.assertions} == {"csm"} and not t.refusal
    agent = t.spec_doc.actors[0]
    assert agent.identity.email == agent.identity.http_headers()["x-user-email"] and f"You are signed in as {agent.identity.email}." in t.instruction
    assert t.user_prompt in t.instruction and t.system_prompt.splitlines()[0] in t.instruction and t.meta["selected_tools"]
    w = await t.open(tmp_path / "w")
    try:
        assert sorted(x.qualified for x in w.tools_for("agent")) == ["csm.sql_execute", "csm.sql_query"]
        r = await w.step("agent", ToolCall(tool_call_id="q1", function_name="csm.sql_query", arguments={"sql": "SELECT MAX(case_id) FROM customer_case"}))
        assert json.loads(r.observation.tool_results[0].content)["rows"] == [[1232]]  # the verifiers expect the next case to be 1233
    finally:
        await w.close()
    it = tasks[ITSM]
    ident = it.spec_doc.actors[0].identity
    assert (ident.user_id, ident.name, ident.headers) == ("USER_001", "Marcus Thompson", {"x-itsm-user-token": "admin_token_marcus_2024_secure"})
    assert "You are signed in as Marcus Thompson <marcus.thompson@techcorp.com> (user id USER_001)." in it.instruction
    assert "incident_sla" in Path(it.spec_doc.apps[0].params["schema_sql"]).read_text() and it.pk["itsm"]["incident"] == "incident_id"


@need_clone
async def test_real_noop_scores_zero_and_witness_scores_one(imported, tmp_path):
    tasks, _, _ = imported
    derivable = []
    for tid, t in sorted(tasks.items()):
        rep, _ = await _play(t, [Done(reason="nothing to do")], tmp_path / tid / "noop")
        assert rep.reward == 0.0 and rep.infra["status"] == "ok", tid
        assert rep.meta["components"]["eog/verifier_pass_rate"] < 1.0
        try:
            acts = t.oracle_actions()
        except Underivable:
            continue
        rep, bad = await _play(t, acts, tmp_path / tid / "oracle")
        assert not bad and rep.reward == 1.0 and rep.meta["components"]["eog/verifier_pass_rate"] == 1.0, (tid, [c.detail for c in rep.checks if c.status != "pass"])
        derivable.append(tid)
    assert len(derivable) >= 6 and CSM in derivable and ITSM in derivable


@need_clone
async def test_real_task_through_the_runner(imported, tmp_path):
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec

    t = imported[0][ITSM]
    for name, acts in (("oracle", t.oracle_actions()), ("noop", [Done(reason="no")])):
        ref = PolicyRef(id=f"scripted:{name}", make=lambda attempt, a=acts: {"agent": ScriptedPolicy("agent", list(a))})
        await Runner(tmp_path / "runs").run(RunSpec(run_id=name, tasks=[t], policies=[ref], harnesses=[NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20))], k=1))
        rec = json.loads(next((tmp_path / "runs" / name / "samples").glob("*/a1/record.json")).read_text())
        assert rec["failure"] is None and rec["reward"] == (1.0 if name == "oracle" else 0.0), rec


@need_clone
def test_hf_row_and_unsupported_verifiers(imported, tmp_path):
    tasks, _, cache = imported
    cfg, _ = load_task(next((EOG / "data" / "revised").glob(f"*/{CSM}.json")))
    row = {**cfg, "task_id": CSM, "domain": "csm", "gym_servers_config": json.dumps(cfg["gym_servers_config"]), "verifiers": json.dumps(cfg["verifiers"])}
    t = import_eog_task(row, seeds=EOG / "gym_dbs.zip", cache_dir=cache)
    assert t.id == "eog/csm/" + CSM and [a.query for a in t.assertions] == [a.query for a in tasks[CSM].assertions]
    judged = {**cfg, "verifiers": [*cfg["verifiers"], {"verifier_type": "response_check", "validation_config": {"sql_query": "SELECT 1", "comparison_prompt": "?"}}]}
    with pytest.raises(ValueError, match="response_check"):
        import_eog_task({**judged, "task_id": "judged"}, seeds=EOG / "gym_dbs.zip", cache_dir=cache)
    t = import_eog_task({**judged, "task_id": "judged"}, seeds=EOG / "gym_dbs.zip", cache_dir=cache, strict=False)
    assert len(t.assertions) == 11 and t.meta["unported_verifiers"] == ["LLM-judged response_check verifiers are not ported"]


@need_clone
async def test_refusal_variant_leaves_apps_unchanged(imported, tmp_path):
    """An infeasible request (no case may be opened for a customer the policy forbids serving): its verifiers all expect a
    zero count, so the task is a refusal task; doing nothing scores 1, touching any row scores 0."""
    _, _, cache = imported
    cfg, _ = load_task(next((EOG / "data" / "revised").glob(f"*/{CSM}.json")))
    gym = cfg["gym_servers_config"][0]["mcp_server_name"]
    cfg = {**cfg, "task_id": "refuse", "user_prompt": "Open a case for an account that does not exist.",
           "verifiers": [{"verifier_type": "database_state", "name": "no new case", "gym_name": gym,
                          "validation_config": {"query": "SELECT COUNT(*) FROM customer_case WHERE case_id = 1233;", "expected_value": 0, "comparison_type": "equals"}}]}
    t = import_eog_task(cfg, seeds=EOG / "gym_dbs.zip", cache_dir=cache)
    assert t.refusal and [g.id for g in t.plan().gates] == ["eog/state", "unchanged"]
    rep, _ = await _play(t, t.oracle_actions(), tmp_path / "refuse")
    assert rep.reward == 1.0 and rep.gates == {"eog/state": "pass", "unchanged": "pass"}
    sneaky = [ToolCall(tool_call_id="s1", function_name="csm.sql_execute", arguments={"sql": "UPDATE customer_case SET priority = 'low' WHERE case_id = 1"}), Done(reason="x")]
    rep, _ = await _play(t, sneaky, tmp_path / "sneaky")
    assert rep.reward == 0.0 and rep.gates == {"eog/state": "pass", "unchanged": "fail"}  # EnterpriseOps' own check alone would pass it
    bare = import_eog_task({**cfg, "task_id": "bare", "verifiers": []}, seeds=EOG / "gym_dbs.zip", cache_dir=cache)
    assert bare.refusal and [g.id for g in bare.plan().gates] == ["unchanged"]
    assert not import_eog_task(cfg, seeds=EOG / "gym_dbs.zip", cache_dir=cache, infeasible=False).refusal
