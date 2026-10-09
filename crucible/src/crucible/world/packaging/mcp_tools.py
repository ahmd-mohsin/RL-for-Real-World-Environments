"""Python tool modules (MiMo style: ``tools/<app>.py`` with plain public functions) as tool surfaces.

Two transports with identical semantics:

- **in-process**: the module is loaded with MiMo's own loader and size-cap wrapper
  (``_vendor/mimo_sidecar/mcp_http.py``) and every function becomes a FastMCP ``Tool`` — the same
  object the HTTP server builds, so input schemas, argument validation, result conversion and
  error text are byte-identical to the served version. Fast; used by the local backend and tests.
- **mcp**: the module is served over streamable HTTP by the vendored ``mcp_http`` script and called
  through an MCP client (``crucible.world.packaging.sidecar``). Used when an external harness or the
  agentenv gateway must reach the tools over the network.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec


def _vendored():
    from crucible._vendor.mimo_sidecar import mcp_http

    return mcp_http


def _content_text(blocks: Any) -> str:
    parts: list[str] = []
    for b in blocks or []:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text is not None:
            parts.append(text)
        else:
            dump = getattr(b, "model_dump", None)
            parts.append(json.dumps(dump() if dump else b, default=str))
    return "\n".join(parts)


def envelope_error(structured: Any, content: str) -> str | None:
    """The error code of a MiMo-style error envelope (``{"error": {"code": "NOT_FOUND", ...}}``), else None."""
    obj = structured
    if isinstance(obj, dict) and set(obj) == {"result"}:
        obj = obj["result"]
    if obj is None and content.lstrip().startswith("{"):
        try:
            obj = json.loads(content)
        except ValueError:
            return None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict | str) and not obj.get("ok", False) and not obj.get("success", False):
        err = obj["error"]
        return str(err.get("code") or err.get("type") or "error") if isinstance(err, dict) else "error"
    return None


class InProcessToolModule:
    """All public functions of one tools file, as FastMCP Tools, called in this process."""

    def __init__(self, app: str, tools_path: str | Path, *, server_name: str | None = None, env: Any = None):
        from mcp.server.fastmcp.tools.base import Tool

        self.app = app
        self.path = Path(tools_path).resolve()
        mh = _vendored()
        # Unique module name per world copy: two worlds in one process must not share module state.
        before = set(sys.modules)
        module = mh._load_tools_module(self.path)
        self._module_names = set(sys.modules) - before
        self.virtualized: list[str] = []
        if env is not None:  # the World's virtual clock and seed instead of wall time and os.urandom (virtual_env.py)
            from crucible.world.packaging.virtual_env import patch_module, shims

            s = shims(env)
            for n in sorted(self._module_names | {module.__name__}):
                if sys.modules.get(n) is not None:
                    self.virtualized += [f"{n}.{k}" for k in patch_module(sys.modules[n], env, s)]
        self.tools: dict[str, Any] = {}
        for name, fn in mh._public_functions(module):
            self.tools[name] = Tool.from_function(mh._wrap_capped(fn), name=name, description=mh._first_line(fn.__doc__) or name)
        if not self.tools:
            raise ValueError(f"no public tool functions in {self.path}")
        self.server_name = server_name or self.path.stem

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(app=self.app, name=n, description=t.description or n, input_schema=t.parameters) for n, t in sorted(self.tools.items())]

    async def call(self, name: str, args: dict[str, Any], *, call_id: str) -> ToolResult:
        t0 = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        try:
            result = await tool.run(args, convert_result=True)
        except Exception as e:  # noqa: BLE001 - FastMCP surfaces tool exceptions as isError text; mirror that exactly
            return ToolResult(tool_call_id=call_id, function_name=f"{self.app}.{name}", ok=False, content=str(e), error=type(e).__name__,
                              fault="tool", duration_s=time.monotonic() - t0)
        unstructured = result[0] if isinstance(result, tuple) else result
        structured = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        content = _content_text(unstructured)
        code = envelope_error(structured, content)
        return ToolResult(
            tool_call_id=call_id,
            function_name=f"{self.app}.{name}",
            ok=code is None,  # MiMo convention: {"error": {"code": ...}} returned as a value is still a failed call
            content=content,
            data=structured,
            error=f"tool:{code}" if code else None,
            fault="tool" if code else None,
            duration_s=time.monotonic() - t0,
        )

    def unload(self) -> None:
        for n in self._module_names:
            sys.modules.pop(n, None)
