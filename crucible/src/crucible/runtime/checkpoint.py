"""Mid-episode checkpoints and resume (DESIGN §10.4; research long-horizon/01 (c)).

A million-token episode that dies at step 3,000 must not start again from step 0: that re-spends the whole
trajectory and samples a different one. Every ``every`` steps the harness writes ``checkpoint.json`` atomically:

- the ``WorldRef`` (app states in the attempt's store, clock, pending events, responders, inboxes, ACL);
- the length of the streamed trace (``trace.jsonl``) at that point;
- each policy's ``state_dict()`` (its *current segment's* context, tools, queued actions, model calls, segments);
- each observer's state (dense progress) and the length of its stream.

A retry of a masked attempt (infrastructure blame) resumes from the previous attempt's checkpoint: the World is
restored in the previous attempt's work directory (where the snapshot blobs live), the trace and progress streams are
carried over up to the checkpoint, and the episode continues with ``run_episode(reset=False, start_steps=k)``. The
policy re-prefills one segment, never the trajectory. Steps after the checkpoint that the crash interrupted are
re-sampled; the record says where it resumed (``resumed_from``), so a trainer can mask the resumed segment if it
requires an exact token prefix across the resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.core.schema.state import WorldRef
from crucible.runtime.journal import atomic_write

NAME = "checkpoint.json"


class Checkpointer:
    def __init__(self, attempt_dir: Path, world: Any, policies: dict[str, Any], observers: list[Any], *, every: int = 200):
        self.dir, self.world, self.policies, self.observers, self.every = Path(attempt_dir), world, policies, observers, every
        self.saved: list[int] = []

    async def __call__(self, step: int) -> None:
        if self.every and step % self.every == 0:
            await self.save(step)

    async def save(self, step: int) -> None:
        ref = await self.world.snapshot()
        doc = {"step": step, "world_ref": ref.model_dump(mode="json"), "log_len": len(self.world.log),
               "policies": {a: p.state_dict() for a, p in self.policies.items() if hasattr(p, "state_dict")},
               "observers": [o.state_dict() if hasattr(o, "state_dict") else None for o in self.observers]}
        atomic_write(self.dir / NAME, json.dumps(doc, default=str))
        self.saved.append(step)


def load(attempt_dir: Path) -> dict | None:
    p = Path(attempt_dir) / NAME
    return json.loads(p.read_text()) if p.exists() else None


async def resume(world: Any, policies: dict[str, Any], observers: list[Any], ckpt: dict, *, prev_dir: Path, new_dir: Path) -> int:
    """Restore a started World and its policies/observers to a checkpoint; returns the step to continue from."""
    from crucible.world.world import World

    await world.restore(WorldRef.model_validate(ckpt["world_ref"]))
    prefix = World.read_trace(Path(prev_dir) / "trace.jsonl", upto=ckpt["log_len"]) if (Path(prev_dir) / "trace.jsonl").exists() else []
    world.stream_trace(Path(new_dir) / "trace.jsonl", prefix=prefix)
    for a, st in ckpt.get("policies", {}).items():
        if a in policies and hasattr(policies[a], "load_state_dict"):
            policies[a].load_state_dict(st)
    for o, st in zip(observers, ckpt.get("observers") or [], strict=False):
        if st is not None and hasattr(o, "load_state_dict"):
            o.load_state_dict(st, prev_dir=Path(prev_dir), log_len=ckpt["log_len"])
    return int(ckpt["step"])
