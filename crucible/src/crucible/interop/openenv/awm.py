"""OpenEnv AgentWorldModel-1K scenarios as Crucible Worlds (DATA + PORT from OpenEnv, BSD-3).

Each AWM scenario is a generated FastAPI + SQLAlchemy service over SQLite. Using OpenEnv's own code
(``_vendor/openenv_awm``): the DB is built from the scenario's DDL + sample INSERTs (``db_manager``),
the generated code is patched to use that DB and to mount ``FastApiMCP`` (``patch``), and the service
runs as a subprocess. Crucible wraps it as an App (template ``openenv/awm_service``): tools come from
the service's MCP endpoint, state is the ``sql`` adapter over the session DB. So 1,000 synthetic
"enterprise" services become Worlds with the same snapshot/restore/verifier-view contract as
everything else. AWM's own verifiers belong to M7.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.core.schema.world import WorldSpec
from crucible.world.apps.base import App
from crucible.world.exec import spawn_logged
from crucible.world.packaging.sidecar import McpClient, free_port, port_open

DEFAULT_CACHE = Path(os.environ.get("AWM_DATA_DIR", "data/cache/awm"))


def loader(cache_dir: Path | None = None):
    from crucible._vendor.openenv_awm.data_loader import AWMDataLoader

    return AWMDataLoader(str(cache_dir or DEFAULT_CACHE))


class AwmServiceApp(App):
    template = "openenv/awm_service"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.scenario = spec.params["scenario"]
        self.cache_dir = Path(spec.params.get("cache_dir", DEFAULT_CACHE))
        self.proc: asyncio.subprocess.Process | None = None
        self.port = 0
        self.client: McpClient | None = None
        self._tools: list[ToolSpec] = []

    async def provision(self) -> None:
        if not self.ctx.exec.is_local:
            raise RuntimeError("openenv/awm_service runs locally (local or agentenv backend)")
        from crucible._vendor.openenv_awm.db_manager import create_database

        ld = loader(self.cache_dir)
        create_database(self.state_params["path"], ld.get_db_schema(self.scenario), ld.get_sample_data(self.scenario))

    async def start(self) -> None:
        from crucible._vendor.openenv_awm.patch import _patch_env_code

        ld = loader(self.cache_dir)
        self.port = free_port()
        code = _patch_env_code(ld.get_env_code(self.scenario), os.path.abspath(self.state_params["path"]), "127.0.0.1", self.port)
        script = self.ctx.host_workdir / f"awm_{self.id}.py"
        script.write_text(code)
        self.proc = await spawn_logged(sys.executable, str(script), log=self.ctx.host_workdir / f"awm_{self.id}.log")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                tail = (self.ctx.host_workdir / f"awm_{self.id}.log").read_text()[-1500:]
                raise RuntimeError(f"AWM service {self.scenario} exited ({self.proc.returncode}): {tail}")
            if port_open(self.port):
                break
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError(f"AWM service {self.scenario} did not start")
        self.client = McpClient(f"http://127.0.0.1:{self.port}/mcp", timeout_s=30)
        for _ in range(50):  # the MCP mount may lag the port by a moment
            try:
                self._tools = await self.client.list_tools(self.id)
                break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError(f"AWM service {self.scenario}: MCP endpoint never answered")

    async def stop(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, 15)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.proc.wait(), 5)

    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        assert self.client is not None
        return await self.client.call(self.id, name, args, call_id=call_id)


def awm_world(scenario: str, *, cache_dir: Path | None = None) -> tuple[WorldSpec, list[str]]:
    """World spec for one AWM scenario and its natural-language tasks."""
    from crucible._vendor.openenv_awm.data_loader import normalize_scenario_name

    key = normalize_scenario_name(scenario)
    ld = loader(cache_dir)
    tasks = ld._build_tasks().get(key, {}).get("tasks", [])
    spec = WorldSpec.model_validate({
        "world": f"openenv-awm/{key}",
        "description": (ld._build_scenarios().get(key, {}).get("description") or "")[:500],
        "backend": "local",
        "apps": [{"id": "api", "template": "openenv/awm_service", "state": {"adapter": "sql", "params": {"path": f"awm/{key}.db"}},
                  "params": {"scenario": key, "cache_dir": str(cache_dir or DEFAULT_CACHE)}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["api.*"]}],
    })
    return spec, [t if isinstance(t, str) else str(t) for t in tasks]
