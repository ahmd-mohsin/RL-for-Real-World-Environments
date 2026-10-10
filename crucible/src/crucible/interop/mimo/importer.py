"""Import Xiaomi MiMo v2.6 "general" environments as Crucible Worlds (DATA + TARGET for MiMo format).

One MiMo env (``general/envs/s3k_<id>_<domain>_<lang>_t<tier>_rl_<n>/``) becomes:

- one ``common/sqlite_app`` per ``tools/<app>.py`` (state: ``system/<app>/state.db``), tools loaded with
  MiMo's own loader (in-process) or served by MiMo's ``mcp_http`` (transport="mcp");
- a ``workspace`` ``common/doc_library`` App (writable: the agent writes ``answer.md`` there, which is
  what MiMo's verifier grades);
- one policy actor with every tool (MiMo agents see all MCP servers plus file tools);
- the instruction from ``instruction.md``; MiMo metadata (tier, domain, language, verifier items) kept
  for M7 (MiMo's ``verify.py`` / ``verifier_meta.json`` are graded there).

The env directory is copied per world (``bundles``), because MiMo tools write to their DBs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.core.schema.world import WorldSpec

_NAME = re.compile(r"^s3k_(?P<id>\d+)_(?P<domain>.+)_(?P<lang>en|zh)_t(?P<tier>\d)_rl_(?P<n>\d+)$")
DEST = "mimo"


@dataclass
class MimoEnv:
    path: Path
    world: WorldSpec
    instruction: str
    manifest: dict[str, Any]
    meta: dict[str, Any]
    tool_files: list[str]
    files: dict[str, str] | None = None  # relpath -> sha256 of the source env at import (byte-identical export + provenance)


def mimo_tool_stems(env: Path) -> list[str]:
    """Exactly MiMo's discovery rule (sidecar_entrypoint.py): *.py, not _-prefixed, not tools_test.py, sorted."""
    tools = env / "tools"
    if not tools.is_dir():
        return []
    return sorted(p.stem for p in tools.iterdir() if p.is_file() and p.suffix == ".py" and not p.name.startswith("_") and p.name != "tools_test.py")


def _app_id(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def is_complete(env: Path) -> bool:
    return (env / "manifest.json").is_file() and (env / "instruction.md").is_file() and bool(mimo_tool_stems(env)) and (env / "workspace").is_dir()


def import_mimo_env(env: str | Path, *, transport: str = "inprocess") -> MimoEnv:
    env = Path(env).resolve()
    if not is_complete(env):
        raise ValueError(f"{env.name}: incomplete MiMo env (manifest/instruction/tools/workspace missing)")
    manifest = json.loads((env / "manifest.json").read_text())
    stems = mimo_tool_stems(env)
    m = _NAME.match(env.name)
    meta: dict[str, Any] = {"source": "xiaomi-mimo-v2.6-rl-oss", "env": env.name}
    if m:
        meta.update(domain=m["domain"], language=m["lang"], tier=int(m["tier"]))
    vm = env / "verifier_meta.json"
    if vm.is_file():
        v = json.loads(vm.read_text())
        items = v.get("items", [])
        meta.update(verifier_items=len(items), verifier_llm_items=sum(1 for i in items if i.get("method") == "llm"),
                    act=v.get("act"), input=v.get("input"))
    apps: list[dict[str, Any]] = []
    for stem in stems:
        apps.append({
            "id": _app_id(stem),
            "template": "common/sqlite_app",
            "state": {"adapter": "sql", "params": {"path": f"{DEST}/system/{stem}/state.db"}},
            "params": {"tools_module": f"{DEST}/tools/{stem}.py", "transport": transport, "server_name": stem},
            "description": f"MiMo app {stem}",
        })
    apps.append({
        "id": "workspace",
        "template": "common/doc_library",
        "state": {"adapter": "fs", "params": {"root": f"{DEST}/workspace"}},
        "params": {"writable": True},
        "description": "MiMo workspace (business documents; the final answer goes to answer.md)",
    })
    spec = WorldSpec.model_validate({
        "world": f"mimo/{env.name}",
        "description": f"Imported MiMo env ({meta.get('domain', '?')}, tier {meta.get('tier', '?')}, {len(stems)} apps)",
        "backend": "local",
        "backend_params": {"bundles": [{"src": str(env), "dest": DEST, "ignore": ["__pycache__", ".cache", "verify.py", "verifier_meta.json", "run_verify.py", "_helpers.py"]}]},
        "apps": apps,
        "actors": [{"id": "agent", "kind": "policy", "tools": ["*"]}],
    })
    return MimoEnv(path=env, world=spec, instruction=(env / "instruction.md").read_text(), manifest=manifest, meta=meta, tool_files=stems,
                   files=tree_hashes(env))


def tree_hashes(root: Path) -> dict[str, str]:
    import hashlib

    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and ".cache" not in p.parts}


def list_mimo_envs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("s3k_"))
