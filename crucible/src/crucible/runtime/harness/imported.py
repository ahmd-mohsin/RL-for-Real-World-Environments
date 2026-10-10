"""The imported harness: an agent run that already happened elsewhere (a historical frontier run, a dataset's recorded
submission) enters the runtime as an attempt whose agent phase is the recorded patch. Grading, classification,
journaling, re-grading and rollout records are then exactly those of a live run (research m9-m10/02 §6.1, R1).

``ImportedHarness({policy_id: patch_path})``; the policy id names the recorded run (e.g. ``import:gpt-5.6-sol-run1``).
"""

from __future__ import annotations

import time
from pathlib import Path

from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import StageFailure


class ImportedHarness(WorkspaceHarness):
    def __init__(self, patches: dict[str, Path], name: str = "import:patch", version: str = "1", **kw):
        super().__init__(name, version, **kw)
        self.patches = {k: Path(v) for k, v in patches.items()}

    async def run(self, attempt: Attempt) -> HarnessResult:
        prior = self._prior(attempt)
        if prior is not None and prior[0] == "regrade":
            return await self.regrade(attempt, prior[1])
        res = HarnessResult(stop="imported")
        t: dict[str, float] = {}
        src = self.patches.get(attempt.policy.id)
        try:
            if src is None or not src.exists():
                raise StageFailure(Failure(kind="crucible:capture_incomplete", stage="collect", reason=f"no recorded patch for {attempt.policy.id}"))
            res.artifacts["imported_from"] = str(src)
            await self._grade(attempt, Collected(patch=src.read_text(), mode="imported", tree_changed=True), res, t)
        except StageFailure as sf:
            res.failure = sf.failure
        res.timings = {k: round(v, 3) for k, v in t.items()} | {"imported_at": round(time.time(), 0)}
        return res
