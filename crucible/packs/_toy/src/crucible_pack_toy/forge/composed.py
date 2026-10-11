"""Composite toy tasks (m13 G-2/G-3): ``forge.compose`` composites of forged ``toy/helpdesk_v2`` tasks, as acceptance tasks.
A composite runs in its components' World (same pack, world file and scenario); ``composite`` exposes the component map
to A14 / A17, ``component_tasks()`` the parts to A15."""

from __future__ import annotations

from typing import Any

from crucible.forge.compose import Composite
from crucible_pack_toy.forge.task import ToyForgedTask


class ToyComposedTask(ToyForgedTask):
    def __init__(self, composite: Composite, sources: list[ToyForgedTask], *, calibration: list[dict] | None = None):
        s0 = sources[0]
        if any(s.world_file != s0.world_file for s in sources):
            raise ValueError("toy composites need components from one World (fuse is a separate operator)")
        super().__init__(composite.spec, composite.checks, composite.oracle, composite.negatives, pack_root=s0.pack_root, world_file=s0.world_file,
                         scenario=s0.sc)
        self.composite, self._sources, self._calibration = composite, list(sources), calibration

    def component_tasks(self) -> list[Any]:
        return list(self._sources)

    def calibration_runs(self) -> list[dict]:
        return list(self._calibration) if self._calibration is not None else super().calibration_runs()
