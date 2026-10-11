"""M1 item 3: enterprise record-world primitives (ports EnterpriseOps-Gym; DESIGN §4.4).

A small service-desk World: a SQLite system of record seeded from a schema and an EnterpriseOps-style data dump that
calls ``datetime('now')``, a tools module, and one policy actor with an identity, a role and a tool catalog. Checked:
seeding is byte-identical across provisions with ``now`` pinned to the clock's t0; the catalog shows the oracle tools
plus seeded distractors and hides the rest; RBAC refuses a denied tool as the policy's error; Apps see the caller's
identity; SQL assertions and the collateral check ("every other row unchanged") grade the end state; EnterpriseOps-Gym
verifiers convert.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import IdentitySpec, WorldSpec
from crucible.core.store import Store
from crucible.verify.runtime.plan import Context, Gate, Plan
from crucible.verify.runtime.trusted import Trusted
from crucible.verify.sql_checks import Allow, Collateral, SqlAssertion, collateral_from, compare, from_enterpriseops, reduce_result
from crucible.world.apps.sqlite_app import pin_now
from crucible.world.world import World

SCHEMA = """
CREATE TABLE users (user_id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, active INTEGER, created_at TEXT);
CREATE TABLE cases (case_id INTEGER PRIMARY KEY, title TEXT, priority TEXT, state TEXT, assigned_to INTEGER, opened_by TEXT, created_at TEXT);
"""
SEED = """
INSERT INTO users (user_id, email, name, active, created_at) VALUES
(7, 'agent@corp.test', 'Ana Agent', 1, datetime('now')),
(8, 'cory@corp.test', 'Cory Vargas', 1, datetime('now', '-30 days')),
(9, 'old@corp.test', 'Old Account', 0, date('now'));
INSERT INTO cases (case_id, title, priority, state, assigned_to, opened_by, created_at) VALUES
(100, 'VPN down', 'high', 'new', NULL, 'seed', strftime('%Y-%m-%dT%H:%M:%S', 'now')),
(101, 'Printer jam', 'low', 'new', 8, 'seed', datetime('now'));
"""
TOOLS = '''"""Service-desk tools (plain functions over SQLite)."""
import sqlite3
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "db" / "itsm.db"


def _con():
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    return con


def find_user(email: str) -> dict:
    """Find a user by email."""
    with _con() as c:
        r = c.execute("SELECT user_id, email, name, active FROM users WHERE email = ?", (email,)).fetchone()
    if r is None:
        raise ValueError(f"no user {email}")
    return dict(r)


def create_case(title: str, priority: str, assigned_to: int) -> dict:
    """Open a case."""
    with _con() as c:
        cur = c.execute("INSERT INTO cases (title, priority, state, assigned_to, opened_by, created_at) VALUES (?, ?, 'new', ?, 'agent', '2026-03-02 10:00:00')",
                        (title, priority, assigned_to))
        return {"case_id": cur.lastrowid}


def update_case(case_id: int, state: str) -> dict:
    """Change a case's state."""
    with _con() as c:
        n = c.execute("UPDATE cases SET state = ? WHERE case_id = ?", (state, case_id)).rowcount
    return {"updated": n}


def delete_user(user_id: int) -> dict:
    """Delete a user account."""
    with _con() as c:
        return {"deleted": c.execute("DELETE FROM users WHERE user_id = ?", (user_id,)).rowcount}


def merge_users(keep: int, drop: int) -> dict:
    """Merge two user accounts."""
    return {"merged": [keep, drop]}


def list_cases() -> list:
    """List every case."""
    with _con() as c:
        return [dict(r) for r in c.execute("SELECT case_id, title, state FROM cases ORDER BY case_id")]


def export_report() -> str:
    """Export a CSV report."""
    return "case_id,state"


def archive_case(case_id: int) -> dict:
    """Archive a case."""
    return {"archived": case_id}


def reindex() -> str:
    """Rebuild search indexes."""
    return "ok"
'''
ORACLE = ["itsm.find_user", "itsm.create_case", "itsm.update_case"]


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    (root / "data" / "svc" / "tools").mkdir(parents=True)
    (root / "data" / "schema.sql").write_text(SCHEMA)
    (root / "data" / "seed.sql").write_text(SEED)
    (root / "data" / "svc" / "tools" / "itsm.py").write_text(TOOLS)
    return root


def spec(*, catalog: dict | None = None, roles: list[str] | None = None, rbac: dict | None = None, seed: int = 0, identity: bool = True) -> WorldSpec:
    agent = {"id": "agent", "kind": "policy", "tools": ["itsm.*"]}
    if identity:
        agent["identity"] = {"user_id": 7, "email": "agent@corp.test", "name": "Ana Agent", "roles": roles if roles is not None else (["agent"] if rbac is not False else [])}
    if catalog is not None:
        agent["catalog"] = catalog
    d = {"world": "t/itsm", "seed": seed, "clock": {"t0": "2026-03-02T10:00:00Z"},
         "backend_params": {"bundles": [{"src": "data/svc", "dest": "svc"}]},
         "apps": [{"id": "itsm", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "svc/db/itsm.db"}},
                   "params": {"schema_sql": "data/schema.sql", "seed_sql": ["data/seed.sql"], "tools_module": "svc/tools/itsm.py"}}],
         "actors": [agent, {"id": "admin", "kind": "policy", "tools": ["itsm.*"]}],
         "rbac": rbac if rbac is not None else {"agent": {"allow": ["itsm.*"], "deny": ["itsm.delete_user", "itsm.merge_users"]}}}
    if rbac is False:
        d.pop("rbac")
    return WorldSpec.model_validate(d)


async def _world(pack, tmp, s: WorldSpec) -> World:
    w = World(s, store=Store(tmp / "s"), workdir=tmp / "w", pack_root=pack)
    await w.start()
    await w.reset()
    return w


async def _call(w: World, name: str, i: int = 1, actor: str = "agent", **args):
    r = await w.step(actor, ToolCall(tool_call_id=f"c{i}", function_name=name, arguments=args))
    return r.observation.tool_results[0]


# ----------------------------------------------------------------------------------------------- seeding
def test_pin_now():
    con = sqlite3.connect(":memory:")
    pin_now(con, "2026-03-02 10:00:00")
    q = lambda sql: con.execute(sql).fetchone()[0]
    assert q("SELECT datetime('now')") == "2026-03-02 10:00:00" and q("SELECT datetime()") == "2026-03-02 10:00:00"
    assert q("SELECT date('now', '+1 day')") == "2026-03-03" and q("SELECT strftime('%Y', 'now')") == "2026"
    assert q("SELECT datetime('2020-01-01 00:00:00', '+1 hour')") == "2020-01-01 01:00:00"  # explicit times are untouched
    assert q("SELECT unixepoch('now')") == 1772445600


async def test_seeding_is_deterministic(pack, tmp_path):
    w1 = await _world(pack, tmp_path / "a", spec())
    w2 = await _world(pack, tmp_path / "b", spec())
    try:
        assert (await w1.verifier_view().fingerprints()) == (await w2.verifier_view().fingerprints())
        r = await w1.verifier_view().query("itsm", {"op": "sql", "args": {"sql": "SELECT created_at FROM users ORDER BY user_id"}})
        assert [x[0] for x in r.value["rows"]] == ["2026-03-02 10:00:00", "2026-01-31 10:00:00", "2026-03-02"]
    finally:
        await w1.close()
        await w2.close()


# ----------------------------------------------------------------------------------------------- catalog
async def test_catalog_oracle_plus_distractors(pack, tmp_path):
    w = await _world(pack, tmp_path / "a", spec(catalog={"oracle": ORACLE, "distractors": 2}, rbac=False, identity=False))
    w2 = await _world(pack, tmp_path / "b", spec(catalog={"oracle": ORACLE, "distractors": 2}, rbac=False, identity=False))
    try:
        shown = sorted(t.qualified for t in w.tools_for("agent"))
        assert len(shown) == 5 and set(ORACLE) <= set(shown)
        assert shown == sorted(t.qualified for t in w2.tools_for("agent"))  # reproducible from the World seed
        hidden = sorted({t.qualified for t in w.all_tools()} - set(shown))
        r = await _call(w, hidden[0])
        assert not r.ok and r.error == "unknown_tool"  # outside the catalog the tool does not exist
        r = await _call(w, hidden[0], 2, actor="admin")
        assert r.error != "unknown_tool"  # an actor without a catalog sees every tool it may call
    finally:
        await w.close()
        await w2.close()
    w = await _world(pack, tmp_path / "c", spec(catalog={"oracle": ORACLE, "distractors": -1, "hide": ["itsm.delete_*"]}, rbac=False, identity=False))
    try:
        shown = {t.qualified for t in w.tools_for("agent")}
        assert "itsm.delete_user" not in shown and len(shown) == len(w.all_tools()) - 1
    finally:
        await w.close()
    w = await _world(pack, tmp_path / "d", spec(catalog={"oracle": ORACLE}, rbac=False, identity=False))
    try:
        assert sorted(t.qualified for t in w.tools_for("agent")) == sorted(ORACLE)  # oracle mode
    finally:
        await w.close()


# ----------------------------------------------------------------------------------------------- identity and rbac
async def test_rbac_refuses_denied_tools_as_policy_error(pack, tmp_path):
    w = await _world(pack, tmp_path, spec())
    try:
        assert "itsm.delete_user" in {t.qualified for t in w.tools_for("agent")}  # visible: refused when called, like a 403
        r = await _call(w, "itsm.delete_user", user_id=9)
        assert not r.ok and r.error == "permission_denied" and r.fault == "policy" and "role(s) agent" in r.content
        assert (await w.verifier_view().query("itsm", {"op": "count", "args": {"table": "users"}})).value == 3  # nothing ran
        assert (await _call(w, "itsm.update_case", 2, case_id=101, state="closed")).ok
        assert (await _call(w, "itsm.delete_user", 3, actor="admin", user_id=9)).ok  # no identity: RBAC does not apply
        assert w.apps["itsm"].identity("agent").email == "agent@corp.test" and w.apps["itsm"].identity("admin") is None
    finally:
        await w.close()


def test_identity_validation_and_headers():
    with pytest.raises(ValueError, match="unknown roles"):
        spec(roles=["root"])
    with pytest.raises(ValueError, match="defines no rbac"):
        spec(roles=["agent"], rbac=False)
    with pytest.raises(ValueError, match="unknown app"):
        spec(rbac={"agent": {"allow": ["crm.*"]}})
    h = IdentitySpec(user_id=32, email="jose@corp.test", roles=["agent", "manager"], headers={"tenant_id": "t1", "X-Org": "o"}).http_headers()
    assert h == {"x-user-id": "32", "x-user-email": "jose@corp.test", "x-user-roles": "agent,manager", "x-tenant-id": "t1", "X-Org": "o"}


# ----------------------------------------------------------------------------------------------- state checks
def test_reduce_and_compare():
    assert reduce_result({"columns": ["count"], "rows": [[1]]}) == 1
    assert reduce_result({"columns": ["a", "b"], "rows": [[1, 2]]}) == {"a": 1, "b": 2}
    assert reduce_result({"columns": ["a"], "rows": [[1], [2]]}) == [{"a": 1}, {"a": 2}]
    assert compare(1, 1.0, "equals") and compare("1", 1, "equals") and not compare("1", "1.0", "equals")
    assert compare(5, 3, "gt") and compare("abc", "b", "contains") and compare(2, [1, 2], "in")
    assert compare([{"a": 2}, {"a": 1}], [{"a": 1}, {"a": 2}], "rows_equal")
    with pytest.raises(ValueError):
        SqlAssertion("x", "itsm", "SELECT 1", 1, compare="approx")


def _plan(*checks):
    return Plan("t/itsm", [Gate("state", kind="outcome"), Gate("no_collateral")], list(checks))


async def test_sql_assertions_and_collateral(pack, tmp_path):
    asserts = [SqlAssertion("case_opened", "itsm", "SELECT COUNT(*) FROM cases WHERE title = 'Laptop broken' AND priority = 'high' AND assigned_to = 8", 1).check("state"),
               SqlAssertion("vpn_closed", "itsm", "SELECT state FROM cases WHERE case_id = 100", "closed").check("state")]
    # the task: open the laptop case and close the VPN case; nothing else may change (the data form a task file holds)
    scope = collateral_from({"id": "collateral", "app": "itsm", "key": {"cases": "case_id"},
                             "allow": {"cases": [{"match": {"title": "Laptop broken"}, "ops": ["insert"]}, {"match": {"case_id": 100}, "ops": ["update"]}]}})
    insert_only = Collateral("insert_only", "itsm", allow={"cases": (Allow(match={"title": "Laptop broken"}, ops=("insert",)),)}, key={"cases": ("case_id",)})
    update_only = Collateral("update_only", "itsm", allow={"cases": (Allow(match={"case_id": 100}, ops=("update",)),)}, key={"cases": ("case_id",)})

    async def run(actions, name):
        w = await _world(pack, tmp_path / name, spec())
        try:
            for i, (tool, args) in enumerate(actions):
                assert (await _call(w, tool, i, **args)).ok
            view = w.verifier_view()
            verdicts = [await c.evaluate(view) for c in (insert_only, update_only)]
            rep = await _plan(*asserts, scope.check("no_collateral")).arun(Context(Trusted(tmp_path / name / "g"), params={"view": view}))
            return verdicts, rep
        finally:
            await w.close()

    good = [("itsm.create_case", {"title": "Laptop broken", "priority": "high", "assigned_to": 8}), ("itsm.update_case", {"case_id": 100, "state": "closed"})]
    (v_ins, v_upd), rep = await run(good, "good")
    assert rep.gates == {"state": "pass", "no_collateral": "pass"} and rep.reward == 1.0
    assert not v_ins.passed and any("unexpected update" in p for p in v_ins.detail["unexpected"])  # each allowance alone is too narrow
    assert not v_upd.passed and any("unexpected insert" in p for p in v_upd.detail["unexpected"])

    sloppy = [*good, ("itsm.update_case", {"case_id": 101, "state": "closed"})]  # also closes an unrelated case
    _, rep = await run(sloppy, "sloppy")
    assert rep.gates["state"] == "pass"  # every assertion about what should happen still holds
    assert rep.gates["no_collateral"] == "fail" and rep.reward == 0.0
    detail = next(c.detail for c in rep.checks if c.id == "collateral")
    assert detail["unexpected_count"] == 1 and '"case_id": 101' in detail["unexpected"][0]  # only the collateral check sees it


ENTERPRISEOPS = Path(__file__).resolve().parents[3] / "RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym/data/revised"


@pytest.mark.skipif(not ENTERPRISEOPS.is_dir(), reason="EnterpriseOps-Gym checkout not present")
def test_enterpriseops_verifiers_convert():
    n = 0
    for f in sorted(ENTERPRISEOPS.glob("*/*.json")):
        task = json.loads(f.read_text())
        app_of = {g["mcp_server_name"]: g["mcp_server_name"].replace("sn-", "").replace("-server", "") for g in task["gym_servers_config"]}
        for i, v in enumerate(task["verifiers"]):
            a = from_enterpriseops(v, app_of=app_of, index=i)
            assert a.compare == "equals" and a.query.lstrip().upper().startswith("SELECT") and a.app in app_of.values()
            n += 1
    assert n >= 100


WHO_SERVER = '''
import sys
import uvicorn
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("who")


@mcp.tool()
def whoami(ctx: Context) -> dict:
    """The x- headers of the HTTP request that carried this call."""
    return {k: v for k, v in ctx.request_context.request.headers.items() if k.startswith("x-")}


uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
'''


async def test_identity_headers_reach_an_http_mcp_server(tmp_path):
    """EnterpriseOps-Gym sends the acting user as ``x-*`` headers so the server applies its own permissions. Each
    identity gets its own session; an actor without one uses the App's session (with the App's static headers)."""
    import asyncio
    import socket
    import subprocess
    import sys

    pytest.importorskip("mcp")
    pytest.importorskip("uvicorn")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp_path / "who.py").write_text(WHO_SERVER)
    proc = subprocess.Popen([sys.executable, str(tmp_path / "who.py"), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: ASYNC220
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                await asyncio.sleep(0.1)
        s = WorldSpec.model_validate({
            "world": "t/who", "rbac": {"agent": {"allow": ["who.*"]}},
            "apps": [{"id": "who", "template": "common/mcp_server", "state": {"adapter": "json_doc", "params": {"path": str(tmp_path / "who.json")}},
                      "params": {"url": f"http://127.0.0.1:{port}/mcp", "headers": {"x-service": "crucible"}}}],
            "actors": [{"id": "agent", "kind": "policy", "tools": ["who.*"],
                        "identity": {"user_id": 32, "email": "jose@corp.test", "roles": ["agent"], "headers": {"tenant": "t1"}}},
                       {"id": "admin", "kind": "policy", "tools": ["who.*"]}]})
        w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
        await w.start()
        try:
            await w.reset()
            got = json.loads((await _call(w, "who.whoami")).content)
            assert got == {"x-service": "crucible", "x-user-id": "32", "x-user-email": "jose@corp.test", "x-user-roles": "agent", "x-tenant": "t1"}
            got = json.loads((await _call(w, "who.whoami", 2, actor="admin")).content)
            assert got == {"x-service": "crucible"}
            assert len(w.apps["who"]._as) == 1  # one session per distinct identity
        finally:
            await w.close()
    finally:
        proc.terminate()
        proc.wait(5)
