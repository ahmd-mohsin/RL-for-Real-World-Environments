"""The RLM harness (m14): a policy's model inside the Recursive-Language-Model scaffold (``actors.rlm.RLMPolicy``), as a
harness for evaluation and RL rollouts. The harness is the action space, so the same model under ``native:llm-tool`` and
under ``rlm:<digest>`` gives different rollouts; the harness id is part of the SampleKey, so one GRPO group never mixes them
(docs/research/m5-m6/06-harnesses.md, items 6 and 11).

What it adds to ``NativeHarness`` (the seed -> agent -> verification -> cleanup phases, checkpoint/resume and progress are
inherited):

- **The policy's model is wrapped.** ``attempt.policy.make(attempt)`` may return, per actor, an LLM client (anything with
  ``complete``) or an LLM policy (anything with ``.llm``); the actor gets an ``RLMPolicy`` over that client, configured by
  ``RLMConfig``. Scripted actors and RLM policies pass through unchanged.
- **Context for long horizons lives in the kernel.** The prompt holds the system prompt, the task, the RLM's own turns
  (output capped) and, after a compaction, the RLM's summary. ``OBS``, ``INBOX`` and ``HISTORY`` (every compacted segment
  verbatim) stay in the kernel. Compaction is the recorded summary call of DESIGN §10.4, and each one starts a segment:
  a fresh exact-prefix token sequence (docs/research/long-horizon/01-frameworks.md §(a)).
- **The config is the harness id.** ``rlm:<sha of RLMConfig>@<version>`` records the compaction mode, budgets and output
  cap, so groups and comparisons stay well-defined.
- **Evidence:** ``rlm.<actor>.json`` per RLM actor, holding the config, counters, segments, notes and the kernel journal
  (every cell, with its recorded host replies). The journal is what restores the kernel exactly, and what an auditor reads
  to see what the code did between World steps.

Training: every root, sub and compaction call keeps its token record (``LLMToolPolicy``'s schema, with ``step`` and
linear capture). Sub-calls are tree nodes that share their root turn's advantage and turn credit (``runtime.record.build``,
``train.credit.turn_credit_map``). A sub-call answered by another model (``sub_llm``) is recorded with that model and is
not trainable (``Turn.trainable``). A rollout with a missing token record is masked ``capture_incomplete`` by the runner,
never trained on.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from crucible.actors.policy import ContextConfig, EpisodeLimits
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.infra import LimitPolicy


@dataclass(frozen=True)
class RLMConfig:
    max_output_chars: int = 8192  # the RLM default: large data stays in variables
    cell_timeout_s: float = 120.0
    max_tokens: int = 2000
    sub_max_tokens: int = 1000
    temperature: float = 1.0  # RL samples; set 0 for a deterministic evaluation
    max_turns: int = 500
    max_continuations: int = 2
    max_stalls: int = 3
    tool_index: bool = True  # a one-line-per-tool index after the task (schemas stay in TOOLS)
    # where model-written code runs: "docker" (a no-network, read-only, capability-free container; the default for any
    # rollout) or "local" (a host subprocess: trusted code and unit tests only, refused unless ``allow_host``; M14 audit H3 V1)
    kernel: str = "docker"
    kernel_image: str = "python:3.12-slim"
    allow_host: bool = False
    # the sub-LLM budget (H3 V4, G1 D10; VF's RLM caps concurrent subagents at 4): per episode, per python call, in flight,
    # and prompt + completion tokens per episode; ``sub_max_tokens`` caps each call
    max_sub_calls: int = 256
    max_sub_calls_per_cell: int = 64
    sub_concurrency: int = 4
    max_sub_tokens: int = 2_000_000
    # compaction (ContextConfig, mode "compact", strategy "summary"); 0 disables it (an overflow ends the episode, scored)
    budget_tokens: int = 128_000
    trigger: float = 0.8
    keep_recent_turns: int = 3
    summary_max_tokens: int = 2000

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def launcher(self) -> Any:
        from crucible.actors.rlm.kernel import DockerLauncher, LocalLauncher

        if self.kernel == "docker":
            return DockerLauncher(self.kernel_image)
        if self.kernel == "local":
            if not self.allow_host:
                raise ValueError("RLMConfig.kernel='local' runs model-written code on the host: refused unless allow_host=True "
                                 "(trusted code only; rollouts and evaluations use kernel='docker')")
            return LocalLauncher(allow_host=True)
        raise ValueError(f"RLMConfig.kernel: docker | local (got {self.kernel!r})")

    def policy_kwargs(self) -> dict[str, Any]:
        return {"max_output_chars": self.max_output_chars, "cell_timeout_s": self.cell_timeout_s, "max_tokens": self.max_tokens,
                "sub_max_tokens": self.sub_max_tokens, "context": self.context(), "max_continuations": self.max_continuations,
                "max_stalls": self.max_stalls, "max_turns": self.max_turns, "tool_index": self.tool_index, "max_sub_calls": self.max_sub_calls,
                "max_sub_calls_per_cell": self.max_sub_calls_per_cell, "sub_concurrency": self.sub_concurrency, "max_sub_tokens": self.max_sub_tokens}

    def context(self) -> ContextConfig:
        if not self.budget_tokens:
            return ContextConfig(mode="none")
        return ContextConfig(budget_tokens=self.budget_tokens, trigger=self.trigger, keep_recent_turns=self.keep_recent_turns,
                             summary_max_tokens=self.summary_max_tokens)


class RLMHarness(NativeHarness):
    def __init__(self, config: RLMConfig | None = None, *, version: str = "1", launcher: Callable[[Attempt, str], Any] | None = None,
                 limits: EpisodeLimits | None = None, limit_policy: LimitPolicy | None = None, checkpoint_every: int = 0,
                 progress_full_every: int = 250):
        """``launcher(attempt, actor)`` places the kernel (default: ``RLMConfig.kernel``, a no-network container; ``ExecLauncher``
        puts it inside the World's own sandbox)."""
        self.config = config or RLMConfig()
        super().__init__(f"rlm:{self.config.digest()}", version, limits=limits or EpisodeLimits(max_steps=10_000), limit_policy=limit_policy,
                         checkpoint_every=checkpoint_every, progress_full_every=progress_full_every)
        self.launcher = launcher
        if launcher is None:
            # M14 campaign B9: kernel='local' without allow_host was refused only inside each attempt, where it became an
            # unknown-blame mask (retried, and counted toward the run's stop rule). A config that cannot run is refused here
            self.config.launcher()

    def wrap(self, attempt: Attempt, actor: str, x: Any) -> Any:
        from crucible.actors.rlm.policy import RLMPolicy

        if isinstance(x, RLMPolicy) or hasattr(x, "act") and not hasattr(x, "llm"):
            return x  # already an RLM, or a scripted / responder policy
        llm = x if hasattr(x, "complete") and not hasattr(x, "act") else x.llm
        c = self.config
        # a sub_llm other than the policy's own model answers sub-calls that are recorded with their model and never trained (G1 D1)
        return RLMPolicy(actor, llm, sub_llm=getattr(x, "sub_llm", None), launcher=self.launcher(attempt, actor) if self.launcher else c.launcher(),
                         seed=zlib.crc32(f"{attempt.sample}:{actor}".encode()), temperature=c.temperature, **c.policy_kwargs())

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.actors.rlm.policy import RLMPolicy

        made: dict[str, Any] = {}
        inner = attempt.policy

        def make(a: Attempt) -> dict[str, Any]:
            out = {actor: self.wrap(a, actor, x) for actor, x in inner.make(a).items()}
            made.update(out)
            return out

        ref = PolicyRef(id=inner.id, make=make, spend=inner.spend, estimate_usd=getattr(inner, "estimate_usd", 0.0))
        try:
            res = await super().run(dataclasses.replace(attempt, policy=ref))
        finally:  # every kernel and container closed, even when the attempt is cancelled mid-flight (M14 campaign B4)
            closes = [asyncio.ensure_future(p.aclose()) for p in made.values() if isinstance(p, RLMPolicy)]
            if closes:
                try:
                    await asyncio.shield(asyncio.gather(*closes, return_exceptions=True))
                except asyncio.CancelledError:
                    with contextlib.suppress(BaseException):
                        await asyncio.wait_for(asyncio.gather(*closes, return_exceptions=True), 120)
                    raise
        for actor, p in made.items():
            if not isinstance(p, RLMPolicy):
                continue
            kernel = p.kernel.state() if p.kernel is not None else None
            doc = {"harness": self.id, "config": asdict(self.config), "counters": p.counters, "segments": p.segments, "notes": p.notes,
                   "stop": p.stop, "kernel": kernel}
            name = f"rlm.{actor}.json"
            (attempt.dir / name).write_text(json.dumps(doc, default=str))
            res.artifacts[f"rlm.{actor}"] = name
        return res
