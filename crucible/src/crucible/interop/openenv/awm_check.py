"""Generality check of the World runtime against OpenEnv AgentWorldModel-1K (ARCHITECTURE M1 gate, part 2b).

Per scenario: the World starts (DB from the scenario's DDL + samples, generated FastAPI service with
FastApiMCP); tools are listed over MCP; snapshot -> reset is stable; a direct DB mutation changes the
fingerprint and ``reset`` restores it exactly; read-style tools round-trip through MCP.
Run: ``python -m crucible.interop.openenv.awm_check <n> [out.json]`` (n scenarios, evenly spaced).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crucible.core.schema.actions import ToolCall
from crucible.core.store import Store
from crucible.interop.openenv.awm import awm_world, loader
from crucible.world.world import World


@dataclass
class AwmCheck:
    scenario: str
    ok: bool = False
    tools: int = 0
    tables: int = 0
    tasks: int = 0
    checks: dict[str, bool] = field(default_factory=dict)
    calls_ok: int = 0
    calls_err: int = 0
    seconds: float = 0.0
    error: str | None = None


async def check(scenario: str, store: Store, root: Path) -> AwmCheck:
    r = AwmCheck(scenario=scenario)
    t0 = time.monotonic()
    w = None
    try:
        spec, tasks = awm_world(scenario)
        r.tasks = len(tasks)
        w = World(spec, instructions={"agent": tasks[0] if tasks else ""}, store=store, workdir=root / scenario)
        await w.start()
        tools = w._tools["api"]
        r.tools = len(tools)
        r.checks["has_tools"] = r.tools > 0
        obs = (await w.reset())["agent"]
        r.checks["instruction_and_tools_delivered"] = bool(obs.tools) and len(obs.tools) == r.tools
        s0 = await w.snapshot()
        r.tables = len(s0.apps["api"].meta.get("tables", {}))
        r.checks["db_has_tables"] = r.tables > 0
        await w.reset()
        r.checks["reset_stable"] = (await w.snapshot()).fingerprint == s0.fingerprint
        db = Path(w.apps["api"].state_params["path"])
        con = sqlite3.connect(db)
        tbl = next((t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
                    if con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]), None)
        if tbl:
            con.execute(f'DELETE FROM "{tbl}" WHERE rowid = (SELECT MIN(rowid) FROM "{tbl}")')
            con.commit()
        con.close()
        r.checks["mutation_visible"] = tbl is None or (await w.snapshot()).fingerprint != s0.fingerprint
        await w.reset()
        r.checks["reset_undoes_mutation"] = (await w.snapshot()).fingerprint == s0.fingerprint
        transport_ok = True
        reads = [t for t in tools if t.name.lower().startswith(("get", "list", "search", "read"))][:3]
        for t in reads:
            res = await w.step("agent", ToolCall(tool_call_id=f"c-{t.name}", function_name=t.qualified, arguments={}))
            tr = res.observation.tool_results[0]
            if tr.ok:
                r.calls_ok += 1
            else:
                r.calls_err += 1
                transport_ok = transport_ok and tr.error not in ("transport", "unknown_tool")
        r.checks["transport_ok"] = transport_ok
        r.ok = all(r.checks.values())
    except Exception as e:  # noqa: BLE001 - report and continue the sweep
        r.error = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if w is not None:
            await w.close()
        r.seconds = round(time.monotonic() - t0, 2)
    return r


async def sweep(n: int, *, concurrency: int = 4) -> list[AwmCheck]:
    names = [s["name"] for s in loader().list_scenarios()]
    step = max(1, len(names) // n)
    picked = names[::step][:n]
    sem = asyncio.Semaphore(concurrency)
    with tempfile.TemporaryDirectory() as d:
        store = Store(Path(d) / "store")

        async def one(name: str) -> AwmCheck:
            async with sem:
                return await check(name, store, Path(d) / "worlds")

        return list(await asyncio.gather(*(one(x) for x in picked)))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    res = asyncio.run(sweep(n))
    ok = sum(r.ok for r in res)
    print(f"{ok}/{len(res)} AWM scenarios pass all world checks; tools={sum(r.tools for r in res)} tables={sum(r.tables for r in res)} "
          f"calls ok/err={sum(r.calls_ok for r in res)}/{sum(r.calls_err for r in res)}")
    for r in res:
        if not r.ok:
            print("FAIL", r.scenario, r.error or {k: v for k, v in r.checks.items() if not v})
    if out:
        out.write_text(json.dumps([asdict(r) for r in res], indent=1))


if __name__ == "__main__":
    main()
