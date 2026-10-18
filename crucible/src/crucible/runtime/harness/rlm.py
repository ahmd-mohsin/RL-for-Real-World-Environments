"""The RLM harness (m14, rebuilt in M16): a policy's model inside the Recursive-Language-Model scaffold
(``actors.rlm.RLMPolicy``, the very-long-horizon agent loop of ``actors.rlm.agent``), as a harness for evaluation and RL
rollouts. The harness is the action space, so the same model under ``native:llm-tool`` and under ``rlm:<digest>`` gives
different rollouts; the harness id is part of the SampleKey, so one GRPO group never mixes them
(docs/research/m5-m6/06-harnesses.md, items 6 and 11).

What it adds to ``NativeHarness`` (the seed -> agent -> verification -> cleanup phases, checkpoint/resume and progress are
inherited):

- **The policy's model is wrapped.** ``attempt.policy.make(attempt)`` may return, per actor, an LLM client (anything with
  ``complete``) or an LLM policy (anything with ``.llm``); the actor gets an ``RLMPolicy`` over that client, configured by
  ``RLMConfig``. Scripted actors and RLM policies pass through unchanged.
- **Three tiers per agent** (docs/research/long-horizon-rlm/04-design.md §1): the prompt (bounded, compacted at
  ``ContextSpec.threshold()`` on usage-anchored tokens), the kernel (variables survive every compaction; snapshotted), and
  the ledger in ``<attempt>/rlm/<actor>/`` (every message, the full output of every cell, every World result; the kernel
  reads it through ``history`` / ``OUT``). A task with ``context_path`` (and ``delivery="context"``) gets it as ``CONTEXT``.
  Each compaction starts a segment: a fresh exact-prefix token sequence.
- **The config is the harness id.** ``rlm:<sha of RLMConfig>@<version>`` records the window, compaction, budgets, tree
  limits and output cap, so groups and comparisons stay well-defined.
- **Container lifetime** follows the episode's wall budget (``RLMConfig.wall_s``, ``kernel.lifetime_for_episode``) and is
  renewed while the episode runs (M16 B9: a fixed 6 h ended long episodes).
- **Evidence:** ``rlm.<actor>.json`` per RLM actor: the config, counters (root and tree), segments, context-health
  metrics (compactions, turns between them, characters dropped against summary characters, failures, history re-reads
  after compaction, delegations, stall verdicts, restores by mode, largest prompt against the threshold), the ledger's
  stats, the tree's usage, and the kernel journal. Never in the reward.

Training: every root, child, sub and compaction call keeps its token record (``agent``, ``depth``, ``edge``, ``segment``,
``window``, ``ledger_index``: design §3.5). Sub-calls and compaction attempts are tree nodes; a child's calls form their own
chain and share the credit of the root turn that spawned it (``runtime.record.build``). A sub-call answered by another
model (``sub_llm``) is recorded with that model and is not trainable. A rollout with a missing token record is masked
``capture_incomplete`` by the runner, never trained on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import zlib
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from crucible.actors.policy import EpisodeLimits
from crucible.actors.rlm.agent import RLMConfig, truncating_stop
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.infra import LimitPolicy

__all__ = ["RLMConfig", "RLMHarness"]


def context_path(task: Any) -> Any:
    """The host file a task offers as ``CONTEXT``: its ``context_path`` when it delivers by context (a task that serves the
    same document through World tools keeps the World as the way in)."""
    p = getattr(task, "context_path", None)
    if p is None:
        return None
    return p if getattr(task, "delivery", "context") == "context" else None


class RLMHarness(NativeHarness):
    def __init__(self, config: RLMConfig | None = None, *, version: str = "1", launcher: Callable[[Attempt, str], Any] | None = None,
                 limits: EpisodeLimits | None = None, limit_policy: LimitPolicy | None = None, checkpoint_every: int = 0,
                 progress_full_every: int = 250):
        """``launcher(attempt, actor)`` places a kernel (default: ``RLMConfig.kernel``, a no-network container; ``ExecLauncher``
        puts it inside the World's own sandbox). It is called once for the root and once per child agent."""
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
        factory = (lambda: self.launcher(attempt, actor)) if self.launcher else c.launcher
        # a sub_llm other than the policy's own model answers sub-calls that are recorded with their model and never trained (G1 D1)
        return RLMPolicy(actor, llm, sub_llm=getattr(x, "sub_llm", None), launcher=factory(), child_launcher=factory, config=c,
                         seed=zlib.crc32(f"{attempt.sample}:{actor}".encode()), ledger_dir=attempt.dir / "rlm" / actor, stream_dir=attempt.dir,
                         context_path=context_path(attempt.task))

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.actors.rlm.policy import RLMPolicy

        made: dict[str, Any] = {}
        inner = attempt.policy

        def make(a: Attempt) -> dict[str, Any]:
            out = {actor: self.wrap(a, actor, x) for actor, x in inner.make(a).items()}
            made.update(out)
            return out

        ref = PolicyRef(id=inner.id, make=make, spend=inner.spend, estimate_usd=getattr(inner, "estimate_usd", 0.0))
        # M10 lab audit R-3 review round 2: NativeHarness keeps Attempt.phase / Attempt.regradable current on the Attempt it
        # runs, opens the grading window on Attempt.clock, and the Runner reads phase and regradable from its own Attempt when
        # its clock cancels the harness (run.py). A copy (dataclasses.replace) hid them, so a clock firing during the grade,
        # its re-grade or the control read as the agent's and re-sampled the policy: the policy is swapped in place on the
        # Runner's Attempt and restored on the way out (the Runner records with its own policy, never attempt.policy)
        attempt.policy = ref
        try:
            res = await super().run(attempt)
        finally:  # every kernel and container closed, even when the attempt is cancelled mid-flight (M14 campaign B4)
            attempt.policy = inner
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
            if truncating_stop(p.stop):  # compaction_failed, budget:<kind>: the episode ran out of something (M16 §3.6)
                res.truncated = True
            doc = {"harness": self.id, "config": asdict(self.config), **p.evidence()}
            name = f"rlm.{actor}.json"
            (attempt.dir / name).write_text(json.dumps(doc, default=str))
            res.artifacts[f"rlm.{actor}"] = name
        return res
