"""The imported harness: an agent run that already happened elsewhere (a historical frontier run, a dataset's recorded
submission) enters the runtime as an attempt whose agent phase is the recorded patch. Grading, classification,
journaling, re-grading and rollout records are then exactly those of a live run (research m9-m10/02 §6.1, R1).

``ImportedHarness({policy_id: patch_path})``; the policy id names the recorded run (e.g. ``import:gpt-5.6-sol-run1``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.identity import file_digest
from crucible.runtime.infra import StageFailure
from crucible.verify.patchio import read_patch


class ImportedHarness(WorkspaceHarness):
    def __init__(self, patches: dict[str, Path], name: str = "import:patch", version: str = "1", **kw):
        super().__init__(name, version, **kw)
        self.patches = {k: Path(v) for k, v in patches.items()}
        # M10 lab audit R-4: the recorded runs ARE the harness (they are what is graded), so their content is in its id and its
        # identity document, not their paths: an edited patch is a different run under the same policy id. Taken once, here
        self._patch_digests = {k: file_digest(v) for k, v in self.patches.items()}

    def _id_config(self) -> dict[str, Any]:
        return {**super()._id_config(), "patches": self._patch_digests}

    async def run(self, attempt: Attempt) -> HarnessResult:
        prior = self._prior(attempt)
        if prior is not None and prior[0] == "regrade":
            return await self.regrade(attempt, prior[1])
        res = HarnessResult(stop="imported")
        t: dict[str, float] = {}
        src = self.patches.get(attempt.policy.id)
        try:
            if src is None or not src.exists():
                # a configuration error (no such recorded run): masked once, never retried (M10 audit V17)
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="collect", retryable=False,
                                           reason=f"no recorded patch for {attempt.policy.id}"))
            res.artifacts["imported_from"] = str(src)
            await self._grade(attempt, Collected(patch=read_patch(src), mode="imported", tree_changed=True), res, t)
        except StageFailure as sf:
            res.failure = sf.failure
        res.timings = {k: round(v, 3) for k, v in t.items()}  # no wall-clock stamp: records stay byte-reproducible (V17)
        return res
