"""``BashLoopPolicy``: Crucible's native agent harness, a port of the mini-swe-agent v2 loop (PORT; MIT,
``minisweagent/agents/default.py`` + ``config/mini.yaml``) onto a World's shell tool (research 06: "seam W").

Why this loop: it is the harness behind every recorded task04 frontier run (0 format errors in 680 calls), it has one
tool, and every command runs in a fresh subshell, so a ``WorldRef`` snapshot is a *complete* state (no hidden tmux pane,
cwd or env vars): the property C3 branching needs. Every action goes through ``World._call_tool`` (P25), so each step
is traced, checked for answer-key probes (P27) and snapshot-able.

Protocol:
- each reply must contain exactly one fenced ```bash block (or one ``bash`` tool call); its command runs via the
  World's shell tool (default ``<app>.shell``);
- the observation is ``<returncode>`` + ``<output>`` (head/tail truncated at ``max_output`` characters, as
  mini-swe-agent does) so long builds do not flood the context;
- the agent ends the attempt by running ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (an explicit submit; "no
  command" is never treated as done);
- a reply with no block, several blocks, or cut off at the output limit is a *format error*: the agent is told
  exactly what was wrong; ``max_format_errors`` consecutive errors end the episode (stop ``format_error``, scored);
- step and cost limits come from ``run_episode`` / the LLM client's hard budget; context overflow ends the episode as
  truncated and scored, credits exhausted masks it.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMClient, LLMError
from crucible.actors.policy import Policy
from crucible.core.schema.actions import Action, Done, Observation, ToolCall

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)\n```", re.DOTALL)

SYSTEM = f"""You are a helpful assistant that can interact with a computer shell to solve programming tasks.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command explaining your reasoning.

Format:
THOUGHT: your reasoning
```bash
your_command_here
```

Rules:
- Every command runs in a new subshell: `cd` and environment variables do not persist; use `cd /path && ...`.
- Do not run interactive programs (editors, pagers). Use `sed`, `cat <<'EOF' > file`, or python to edit files.
- Redirect long output to a file and inspect it with `grep`/`tail`.
- When you are done, run exactly: `echo {SUBMIT}` as your only command. You cannot continue after that."""

FORMAT_ERROR = f"""Please always provide EXACTLY ONE action in triple backticks, found {{n}} actions.

If you have completed your assignment, please consult the first message about how to submit your solution (you will not
be able to continue working on this task after that): run `echo {SUBMIT}`.
Else, please format your response exactly as follows:

THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
<action>
```"""

CUT_OFF = "Your output was cut off at the output limit before a complete bash block. Keep the THOUGHT short and give exactly one ```bash block."


def format_observation(rc: int | None, output: str, max_output: int = 10000) -> str:
    if len(output) <= max_output:
        return f"<returncode>{rc}</returncode>\n<output>\n{output}</output>"
    head, tail = output[: max_output // 2], output[-max_output // 2:]
    return (f"<returncode>{rc}</returncode>\n<warning>\nThe output of your last command was too long ({len(output)} characters). Please try a "
            f"different command that produces less output, e.g. redirect to a file and use head/tail/grep.\n</warning>\n"
            f"<output_head>\n{head}\n</output_head>\n<elided_chars>\n{len(output) - max_output} characters elided\n</elided_chars>\n"
            f"<output_tail>\n{tail}\n</output_tail>")


_RC = re.compile(r"^\[exit (-?\d+)\]")


class BashLoopPolicy(Policy):
    def __init__(self, actor_id: str, llm: LLMClient, *, shell_tool: str = "ws.shell", temperature: float = 1.0, max_tokens: int = 4000,
                 max_output: int = 10000, max_format_errors: int = 3, system: str = SYSTEM, timeout_s: int | None = None,
                 prefix: list[dict[str, Any]] | None = None, harness_id: str = "crucible-bash-loop@1"):
        self.actor_id, self.llm, self.shell_tool = actor_id, llm, shell_tool
        self.temperature, self.max_tokens, self.max_output, self.max_format_errors = temperature, max_tokens, max_output, max_format_errors
        self.timeout_s = timeout_s
        self.harness_id = harness_id
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *(prefix or [])]  # prefix: C3 "prefix" mode
        self.format_errors = 0
        self.error: str | None = None
        self.failure_kind: str | None = None
        self.stop: str | None = None
        self.sample_key: str | None = None
        self.model_calls: list[dict[str, Any]] = []
        self._last_call: str | None = None

    def begin_episode(self, episode_id: str) -> None:
        self.sample_key = episode_id

    def _ingest(self, obs: Observation) -> None:
        if obs.instruction:
            self.messages.append({"role": "user", "content": f"Please solve this task:\n\n{obs.instruction}\n\nYou can run shell commands. "
                                                              f"Submit with `echo {SUBMIT}` when you are done."})
        for tr in obs.tool_results:
            if tr.tool_call_id != self._last_call:
                continue
            text = tr.content or ""
            m = _RC.match(text)
            rc = int(m.group(1)) if m else (0 if tr.ok else 1)
            body = text[m.end():].lstrip("\n") if m else text
            if SUBMIT in body.splitlines()[0:1] or body.strip().startswith(SUBMIT):
                self.stop = "agent_done"
            self.messages.append({"role": "user", "content": format_observation(rc, body, self.max_output)})

    async def act(self, obs: Observation) -> Action:
        self._ingest(obs)
        if self.stop == "agent_done":
            return Done(reason="submitted")
        while True:
            try:
                r = await self.llm.complete(self.messages, max_tokens=self.max_tokens, temperature=self.temperature,
                                            seed=0 if self.temperature == 0 else None, sample_key=self.sample_key)
            except ContextOverflow as e:
                self.stop = "context_overflow"
                return Done(reason=f"context overflow: {e}")
            except (LLMError, BudgetExceeded) as e:
                self.failure_kind = getattr(e, "failure_kind", None) or ("crucible:credits_exhausted" if isinstance(e, BudgetExceeded) else "provider_unavailable")
                self.error = f"{self.failure_kind}: {e}"
                return Done(reason=f"policy error: {e}")
            call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "step": obs.step, "model": r.model, "finish": r.raw_finish,
                    "usage": r.usage, "cost_usd": r.cost_usd, "cached": r.cached, "harness": self.harness_id, "actions": [], "tokens": r.tokens}
            self.model_calls.append(call)  # a format-error call keeps ``actions`` empty: its own ATIF step (runtime.trajectories)
            text = r.text or ""
            blocks = BLOCK.findall(text)
            if r.tool_calls and not blocks:  # a model that answers with a tool call instead of a block
                args = r.tool_calls[0].get("arguments") or {}
                blocks = [str(args.get("command") or args.get("cmd") or "")] if len(r.tool_calls) == 1 else []
            self.messages.append({**(r.raw_message or {}), "role": "assistant", "content": text})
            if len(blocks) == 1 and blocks[0].strip():
                self.format_errors = 0
                cmd = blocks[0].strip()
                self._last_call = f"bash-{uuid.uuid4().hex[:10]}"
                call["actions"] = [f"tool:{self._last_call}"]
                args: dict[str, Any] = {"command": cmd}
                if self.timeout_s:
                    args["timeout_s"] = self.timeout_s
                return ToolCall(tool_call_id=self._last_call, function_name=self.shell_tool, arguments=args)
            self.format_errors += 1
            if self.format_errors >= self.max_format_errors:
                self.stop = "format_error"
                call["actions"] = ["done"]
                return Done(reason=f"{self.format_errors} consecutive format errors")
            self.messages.append({"role": "user", "content": CUT_OFF if r.raw_finish == "length" else FORMAT_ERROR.format(n=len(blocks))})
