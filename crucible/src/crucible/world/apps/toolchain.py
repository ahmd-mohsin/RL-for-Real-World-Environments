"""Toolchain App: compilers, GPUs, test runners — state is the ``process`` adapter (builds, tests,
benchmarks, GPU probes run through it on the verifier side). The policy gets no tools from it unless
``params.shell`` is set (then a ``shell`` tool in ``params.cwd``); usually the policy's shell lives in
the repo App and this App exists so verifiers and milestones can run builds/benchmarks by name.
"""

from __future__ import annotations

from typing import Any

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, ToolError


class ToolchainApp(App):
    template = "common/toolchain"
    PATH_PARAMS = ("cwd",)

    async def list_tools(self) -> list[ToolSpec]:
        if not self.spec.params.get("shell"):
            return []
        return [ToolSpec(app=self.id, name="shell", description="Run a bash command with the toolchain; returns exit code and output.",
                         input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, "required": ["command"]})]

    async def tool_shell(self, command: str, timeout_s: int = 600) -> Any:
        if self.policy_exec.is_local and not self.spec.params.get("allow_unsafe_local_shell"):
            raise ToolError("shell refused: the local backend has no sandbox")
        out = await self.policy_exec.run(command, cwd=self.state_params.get("cwd"), timeout_s=self.deadline(timeout_s, self.spec.params.get("max_timeout_s")))
        text = out.stdout + (("\n[stderr]\n" + out.stderr) if out.stderr.strip() else "")
        return f"[{'TIMEOUT' if out.timed_out else f'exit {out.rc}'}]\n{text[-30000:]}"
