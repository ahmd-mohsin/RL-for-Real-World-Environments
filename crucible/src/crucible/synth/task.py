"""The synthesized task τ = (W, seed, ι, actors, C, π*, I) of part2-paper-core §3.3, as data.

``world`` names the World by a ``WorldRef``: an import path ``module:callable`` (called with ``params``, returning a
``SynthWorld`` or an awaitable of one) or a name the caller's ``FalsifyContext`` resolves. The intent card says which
literals the agent is *given* (outside the instruction, e.g. by an actor) and which it must *derive*: TAINT reads both
(a ``derive`` value found in ι is a leak, D7). ``actors`` is a placeholder for the simulated people of later milestones.
A repair makes a new ``version`` of the same ``id``; the archive keys counterexamples by both.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash
from crucible.synth.contract import Contract
from crucible.synth.iface import SynthWorld
from crucible.synth.program import Program


class WorldRef(CrucibleModel):
    factory: str  # "module:callable", or a name a FalsifyContext resolves
    params: dict[str, Any] = Field(default_factory=dict)

    async def build(self) -> SynthWorld:
        mod, _, attr = self.factory.partition(":")
        if not attr:
            raise ValueError(f"world factory {self.factory!r} is not 'module:callable'")
        f = getattr(importlib.import_module(mod), attr)
        w = f(**self.params)
        return await w if inspect.isawaitable(w) else w


class IntentCard(CrucibleModel):
    skill: str = ""
    given: list[Any] = Field(default_factory=list)
    derive: list[Any] = Field(default_factory=list)
    notes: str = ""


class Task(CrucibleModel):
    id: str
    domain: str
    version: int = 1
    world: WorldRef
    seed: int = 0
    instruction: str
    actors: list[dict[str, Any]] = Field(default_factory=list)
    contract: Contract
    oracle: Program
    intent: IntentCard = Field(default_factory=IntentCard)

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 32)

    def next_version(self, **changes: Any) -> Task:
        return self.model_copy(update={"version": self.version + 1, **changes}, deep=True)
