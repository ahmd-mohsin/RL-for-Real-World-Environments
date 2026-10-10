"""C2, defect -> check (DESIGN §6.3): every negative expert signal with a (wrong, corrected) pair becomes a candidate
hidden check plus a negative mutant.

1. **Locate** the defect: the file and line span the signal targets, and the correction diff (wrong -> corrected).
2. **Synthesize** a check that fails on the wrong state and passes on the corrected one. Behavioural checks
   (differential tests, sanitizer runs, edge inputs) are produced by the Task Forge (M8) from the located defect; here
   the operator derives the *static* fallback from the correction itself: constructs the correction removed must be
   absent, constructs it added must be present, scoped to the corrected file (``milestones.mine.from_diff``).
3. **Validate**: the check must kill the wrong version, pass the corrected version and the expert end state, and (for
   behavioural checks) not be flaky (A4). A check that does not separate wrong from corrected is dropped.
4. **Register** the check and the mutant: the wrong version, i.e. the correction reversed (SWE-smith's "PR mirror"),
   becomes a negative that every reward function must score 0 (R3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.milestone import Milestone
from crucible.workflow.filestate import FileState, parse_patch
from crucible.workflow.milestones.evaluate import FileStateView, evaluate
from crucible.workflow.milestones.mine import from_diff


@dataclass
class CheckCandidate:
    signal: str
    checks: list[Milestone]
    mutant_patch: str  # applying it to the corrected state reproduces the wrong version
    validated: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)


def reverse_patch(patch: str) -> str:
    """The inverse of a unified diff (swap '-' and '+' lines and the hunk ranges): the PR-mirror mutant."""
    out = []
    for fp in parse_patch(patch):
        old = fp.old or "/dev/null"
        new = fp.new or "/dev/null"
        out.append(f"--- {new}\n+++ {old}\n")
        for h in fp.hunks:
            out.append(f"@@ -{h.new_start},{h.new_len} +{h.old_start},{h.old_len} @@\n")
            for ln in h.lines:
                tag, body = ln[:1], ln[1:]
                out.append(("+" if tag == "-" else "-" if tag == "+" else tag) + body + "\n" if tag in "+- " else ln + "\n")
    return "".join(out)


async def from_correction(signal_id: str, correction_patch: str, wrong: Callable[[str], str | None], *, app: str = "repo",
                          end_state: Callable[[str], str | None] | None = None) -> CheckCandidate:
    """``wrong``: file reader for the state before the correction; ``end_state``: the expert's final state (must pass)."""
    checks = [c.milestone.model_copy(update={"id": f"c2/{signal_id}/{c.milestone.id.split('/', 1)[1]}"})
              for c in from_diff(correction_patch, wrong, app=app, source=f"signal:{signal_id}") if c.evidence.get("change") != "located"]
    before = FileState(base=wrong)
    after = FileState(base=wrong)
    after.apply_patch(correction_patch)
    kill = await evaluate(checks, FileStateView(before))
    keep = await evaluate(checks, FileStateView(after))
    end = await evaluate(checks, FileStateView(FileState(base=end_state))) if end_state else keep
    good = [m for m in checks if kill.get(m.id) == 0.0 and keep.get(m.id) == 1.0 and end.get(m.id) == 1.0]
    return CheckCandidate(signal=signal_id, checks=good, mutant_patch=reverse_patch(correction_patch), validated=bool(good),
                          evidence={"proposed": len(checks), "separating": len(good), "dropped": [m.id for m in checks if m not in good]})
