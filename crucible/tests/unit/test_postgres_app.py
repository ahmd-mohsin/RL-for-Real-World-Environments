"""Postgres App (database server as a system of record): tools, MCP tool module on DATABASE_URL, canonical
fingerprints incl. sequences, diff, and snapshot -> restore into a fresh world's own container."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = pytest.mark.docker

SCHEMA = """
CREATE TABLE vendors (id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE invoices (id SERIAL PRIMARY KEY, vendor_id INT NOT NULL REFERENCES vendors(id), amount NUMERIC(12,2) NOT NULL,
                       status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','matched','paid','on_hold')));
"""
SEED = """
INSERT INTO vendors(name) VALUES ('Acme Parts'), ('Globex');
INSERT INTO invoices(vendor_id, amount) VALUES (1, 1200.50), (2, 89.99), (1, 15.00);
"""
TOOLS = textwrap.dedent('''
    """AP tools on a real Postgres (psycopg2 + DATABASE_URL, the agentenv MCP-server convention)."""
    import os
    import psycopg2

    def _con():
        c = psycopg2.connect(os.environ["DATABASE_URL"])
        c.autocommit = True
        return c

    def hold_invoice(invoice_id: int, reason: str) -> dict:
        """Put an invoice on hold."""
        with _con() as c, c.cursor() as cur:
            cur.execute("UPDATE invoices SET status = 'on_hold' WHERE id = %s AND status = 'open'", (invoice_id,))
            if cur.rowcount != 1:
                raise ValueError(f"invoice {invoice_id} is not open")
        return {"id": invoice_id, "status": "on_hold", "reason": reason}

    def open_invoices() -> list:
        """List open invoices."""
        with _con() as c, c.cursor() as cur:
            cur.execute("SELECT id, vendor_id, amount::text FROM invoices WHERE status = 'open' ORDER BY id")
            return [{"id": r[0], "vendor_id": r[1], "amount": r[2]} for r in cur.fetchall()]
''')


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack"
    p.mkdir()
    (p / "schema.sql").write_text(SCHEMA)
    (p / "seed.sql").write_text(SEED)
    (p / "ap_tools.py").write_text(TOOLS)
    return p


def spec(tools_module: bool) -> WorldSpec:
    params = {"schema_sql": "schema.sql", "seed_sql": "seed.sql", "sql_write": True}
    if tools_module:
        params["tools_module"] = "ap_tools.py"
    return WorldSpec.model_validate({
        "world": "test/erp-postgres",
        "apps": [{"id": "erp", "template": "common/postgres_app", "state": {"adapter": "postgres", "params": {"database": "erp", "schema": "ap"}}, "params": params}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*"]}],
    })


async def test_postgres_world(pack: Path, tmp_path: Path):
    store = Store(tmp_path / "s")
    w = World(spec(False), store=store, workdir=tmp_path / "w1", pack_root=pack)
    await w.start()
    try:
        await w.reset()
        r = await w.step("agent", ToolCall(tool_call_id="1", function_name="erp.sql_query", arguments={"sql": "SELECT id, amount FROM ap.invoices ORDER BY id"}))
        assert r.observation.tool_results[0].ok and "1200.50" in r.observation.tool_results[0].content
        r = await w.step("agent", ToolCall(tool_call_id="2", function_name="erp.sql_query", arguments={"sql": "DELETE FROM ap.invoices"}))
        assert not r.observation.tool_results[0].ok  # read-only transaction: sql_query can never write
        s0 = await w.snapshot()
        r = await w.step("agent", ToolCall(tool_call_id="3", function_name="erp.sql_execute",
                                            arguments={"sql": "UPDATE invoices SET status='matched' WHERE id=2; INSERT INTO vendors(name) VALUES ('Initech');"}))
        assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        s1 = await w.snapshot()
        assert s1.fingerprint != s0.fingerprint
        d = await w.apps["erp"].state.diff(s0.apps["erp"], s1.apps["erp"])
        assert set(d.tables_changed) == {"invoices", "vendors"} and len(d.rows) == 3
        # sequences are state: restoring s0 resets vendors_id_seq, so the next insert gets id 3 again
        await w.restore(s0)
        await w.step("agent", ToolCall(tool_call_id="4", function_name="erp.sql_execute", arguments={"sql": "INSERT INTO vendors(name) VALUES ('Initech')"}))
        assert (await w.verifier_view().query("erp", {"op": "scalar", "args": {"sql": "SELECT max(id) FROM ap.vendors"}})).value == 3
        await w.restore(s1)
        assert (await w.snapshot()).fingerprint == s1.fingerprint
    finally:
        await w.close()
    # a fresh world (a different Postgres container) restores s1 exactly
    w2 = World(spec(False), store=store, workdir=tmp_path / "w2", pack_root=pack)
    await w2.start()
    try:
        assert (await w2.snapshot()).fingerprint == s0.fingerprint  # same seed -> same canonical state
        await w2.restore(s1)
        assert (await w2.snapshot()).fingerprint == s1.fingerprint
    finally:
        await w2.close()


async def test_postgres_tools_module_over_mcp(pack: Path, tmp_path: Path):
    pytest.importorskip("psycopg2")
    w = World(spec(True), store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=pack)
    await w.start()
    try:
        obs = (await w.reset())["agent"]
        assert {t.name for t in obs.tools} == {"hold_invoice", "open_invoices"}
        r = await w.step("agent", ToolCall(tool_call_id="1", function_name="erp.hold_invoice", arguments={"invoice_id": 1, "reason": "PO mismatch"}))
        assert r.observation.tool_results[0].ok
        r = await w.step("agent", ToolCall(tool_call_id="2", function_name="erp.hold_invoice", arguments={"invoice_id": 1, "reason": "again"}))
        assert not r.observation.tool_results[0].ok and "not open" in r.observation.tool_results[0].content
        v = await w.verifier_view().query("erp", {"op": "sql", "args": {"sql": "SELECT id, status FROM ap.invoices ORDER BY id"}})
        assert v.value["rows"] == [[1, "on_hold"], [2, "open"], [3, "open"]]
    finally:
        await w.close()
