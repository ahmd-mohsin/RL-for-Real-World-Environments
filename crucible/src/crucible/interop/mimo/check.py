"""Generality check of the World runtime against MiMo envs (ARCHITECTURE M1 exit gate, part 2).

For each env: the imported World starts; App ids equal MiMo's own tool discovery and the
manifest's ``mcp_servers``; every App's tool names equal the public functions of its tools file
(counted independently with ``ast``); the verifier files are absent from the policy's world;
snapshot -> reset -> snapshot is stable; a mutation (DB row + answer.md) changes the fingerprint and
``reset`` undoes it exactly; every read-style tool call round-trips through the transport.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crucible.core.schema.actions import ToolCall
from crucible.core.store import Store
from crucible.interop.mimo.importer import _app_id, import_mimo_env
from crucible.world.world import World


@dataclass
class EnvCheck:
    env: str
    ok: bool = False
    apps: int = 0
    tools: int = 0
    tier: int | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    calls_ok: int = 0
    calls_err: int = 0
    seconds: float = 0.0
    error: str | None = None


def _public_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    # set(): a module may define the same function twice (upstream s3k_0151); the last definition wins.
    return sorted({n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")})


async def check_env(env: Path, store: Store, workroot: Path, *, transport: str = "inprocess", backend: str = "local", calls_per_app: int = 2) -> EnvCheck:
    r = EnvCheck(env=env.name)
    t0 = time.monotonic()
    w: World | None = None
    try:
        m = import_mimo_env(env, transport=transport)
        r.tier = m.meta.get("tier")
        w = World(m.world, instructions={"agent": m.instruction}, store=store, workdir=workroot / env.name, backend=backend)
        await w.start()
        app_ids = [a.id for a in m.world.apps if a.id != "workspace"]
        r.apps = len(app_ids)
        r.checks["apps_match_discovery"] = app_ids == [_app_id(s) for s in m.tool_files]
        r.checks["apps_match_manifest"] = sorted(app_ids) == sorted(_app_id(s["name"]) for s in m.manifest.get("mcp_servers", []))
        tools_ok = True
        for stem, app in zip(m.tool_files, app_ids, strict=True):
            names = sorted(t.name for t in w._tools[app])
            r.tools += len(names)
            if names != _public_functions(env / "tools" / f"{stem}.py"):
                tools_ok = False
        r.checks["tools_match_source"] = tools_ok
        world_files = {p.name for p in (w.workdir / "apps" / "mimo").rglob("*")}
        r.checks["verifier_files_hidden"] = not ({"verify.py", "verifier_meta.json", "run_verify.py"} & world_files)
        obs = (await w.reset())["agent"]
        r.checks["instruction_delivered"] = bool(obs.instruction) and len(obs.tools) == r.tools + 3 + 1  # + doc tools (3) + write_doc
        s0 = await w.snapshot()
        await w.reset()
        r.checks["reset_stable"] = (await w.snapshot()).fingerprint == s0.fingerprint == w.initial.fingerprint
        # mutate like an agent would: write a row and the answer file
        db = next((w.workdir / "apps" / "mimo" / "system").glob("*/state.db"), None)
        if db is not None:
            con = sqlite3.connect(db)
            tbl = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchone()[0]
            cols = [c[1] for c in con.execute(f'PRAGMA table_info("{tbl}")')]
            con.execute(f'DELETE FROM "{tbl}" WHERE rowid = (SELECT MIN(rowid) FROM "{tbl}")')
            con.commit()
            con.close()
            assert cols
        await w.step("agent", ToolCall(tool_call_id="w", function_name="workspace.write_doc", arguments={"path": "answer.md", "content": "draft"}))
        s1 = await w.snapshot()
        r.checks["mutation_visible"] = s1.fingerprint != s0.fingerprint
        await w.reset()
        r.checks["reset_undoes_mutation"] = (await w.snapshot()).fingerprint == s0.fingerprint
        # transport round-trips: call list/search/get-style tools with no args (errors are fine, crashes are not)
        transport_ok = True
        for app in app_ids:
            cands = [t for t in w._tools[app] if t.name.split("_")[0] in ("list", "search", "get", "query", "find", "lookup")][:calls_per_app]
            for t in cands:
                res = await w.step("agent", ToolCall(tool_call_id=f"c-{app}-{t.name}", function_name=t.qualified, arguments={}))
                tr = res.observation.tool_results[0]
                if tr.ok:
                    r.calls_ok += 1
                else:
                    r.calls_err += 1
                    if tr.error in ("transport", "unknown_tool", "permission_denied"):
                        transport_ok = False
        r.checks["transport_ok"] = transport_ok
        r.ok = all(r.checks.values())
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        r.error = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if w is not None:
            await w.close()
        r.seconds = round(time.monotonic() - t0, 2)
    return r


async def sweep(envs: list[Path], store: Store, workroot: Path, *, transport: str = "inprocess", backend: str = "local", concurrency: int = 4) -> list[EnvCheck]:
    sem = asyncio.Semaphore(concurrency)

    async def one(e: Path) -> EnvCheck:
        async with sem:
            return await check_env(e, store, workroot, transport=transport, backend=backend)

    return list(await asyncio.gather(*(one(e) for e in envs)))


def main() -> None:  # python -m crucible.interop.mimo.check <envs_root> [out.json] [transport] [backend] [stride]
    import sys
    import tempfile

    from crucible.interop.mimo.importer import is_complete, list_mimo_envs

    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    transport = sys.argv[3] if len(sys.argv) > 3 else "inprocess"
    backend = sys.argv[4] if len(sys.argv) > 4 else "local"
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    envs = [e for e in list_mimo_envs(root) if is_complete(e)][::stride]
    with tempfile.TemporaryDirectory() as d:
        res = asyncio.run(sweep(envs, Store(Path(d) / "store"), Path(d) / "worlds", transport=transport, backend=backend))
    passed = sum(r.ok for r in res)
    print(f"{passed}/{len(res)} MiMo envs pass all world checks ({len(list_mimo_envs(root)) - len(envs)} incomplete skipped)")
    for r in res:
        if not r.ok:
            print("FAIL", r.env, r.error or {k: v for k, v in r.checks.items() if not v})
    if out:
        out.write_text(json.dumps([asdict(r) for r in res], indent=1))


if __name__ == "__main__":
    main()
