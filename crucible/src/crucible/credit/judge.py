"""The LLM-judge outcome baseline (M6 gate S2): given the task and a trajectory prefix, how likely is the attempt to
succeed? Φ at the same point must beat it, or dense state credit adds nothing a judge reading the log could not.

Protocol (GitSwarm's judging, MiMo's strict judge, prime-environments' fail-closed parsers):
- the judge sees the task statement and the prefix as the agent saw it (messages, tool calls, truncated tool
  outputs; the most recent part is kept when the prefix is long), never the final outcome or any later step;
- it answers with ``<probability>0.xx</probability>`` after a short reasoning; the probability is parsed with an
  anchored regex and clamped to [0, 1];
- ``passes`` independent samples (temperature > 0, distinct ``sample_key``) are averaged and their spread reported;
- an unparseable pass is retried; if every pass stays unparseable the judgement is ``None`` (masked, never 0);
- reasoning models spend hidden tokens before answering: ``max_tokens`` must leave room (400 gave empty answers);
- the judge model should come from a different family than the agent, and runs under the LLM client's hard budget.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PROB = re.compile(r"<probability>\s*([01](?:\.\d+)?|\.\d+)\s*</probability>", re.IGNORECASE)

SYSTEM = """You assess software-engineering agents. You will see a task and the first part of one agent's attempt \
(its messages, the commands it ran and truncated outputs). The attempt has not finished. Estimate the probability that \
the attempt will finally succeed, i.e. that the agent's final code change will make the task's hidden tests pass.

Judge only from the evidence shown: has the agent located the right code, understood the root cause, reproduced the \
problem, made a plausible fix, avoided breaking other behaviour? Do not reward confident claims the evidence does not \
support. Think briefly (at most 120 words), then end with exactly one line: <probability>P</probability> with P \
between 0 and 1."""


@dataclass
class Judgement:
    p_success: float | None
    passes: list[float | None] = field(default_factory=list)
    spread: float | None = None
    cost_usd: float = 0.0
    error: str | None = None


def render_prefix(messages: Sequence[dict[str, Any]], *, max_chars: int = 24000, tool_output_chars: int = 600, task_chars: int = 5000) -> str:
    """Compact text of a chat-format prefix; the task (first user message) is always kept, then the most recent turns."""
    task = ""
    parts: list[str] = []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            continue
        if role == "user" and not task:
            task = content[:task_chars]
            continue
        if role == "assistant":
            if content.strip():
                parts.append(f"AGENT: {content.strip()[:1500]}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append(f"CALL {fn.get('name')}: {str(fn.get('arguments'))[:1200]}")
        elif role == "tool":
            out = content.strip()
            parts.append(f"OUTPUT: {out[:tool_output_chars]}{' …' if len(out) > tool_output_chars else ''}")
        elif role == "user":
            parts.append(f"USER: {content[:1500]}")
    body = "\n".join(parts)
    if len(body) > max_chars:
        body = "[… earlier steps omitted …]\n" + body[-max_chars:]
    return f"TASK:\n{task}\n\nATTEMPT SO FAR:\n{body}"


def parse(text: str) -> float | None:
    found = PROB.findall(text or "")
    if not found:
        return None
    return min(1.0, max(0.0, float(found[-1])))


async def judge(client: Any, prefix_text: str, *, passes: int = 3, temperature: float = 0.7, retries: int = 1, key: str = "",
                max_tokens: int = 2500) -> Judgement:
    """``client``: an ``LLMClient`` (budgeted, cached). Returns the mean of the parsed passes."""
    vals: list[float | None] = []
    cost = 0.0
    for k in range(passes):
        v = None
        for r in range(retries + 1):
            resp = await client.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prefix_text}],
                                         temperature=temperature, max_tokens=max_tokens, seed=None, sample_key=f"{key}:pass{k}:try{r}")
            cost += resp.cost_usd
            v = parse(resp.text)
            if v is not None:
                break
        vals.append(v)
    got = [v for v in vals if v is not None]
    if not got:
        return Judgement(None, vals, None, cost, "unparseable in every pass")
    return Judgement(statistics.fmean(got), vals, (max(got) - min(got)) if len(got) > 1 else 0.0, cost)
