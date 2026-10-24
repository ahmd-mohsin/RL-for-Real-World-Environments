"""Crucible tasks as a verifiers v1 taskset → prime-rl (export; research m11-m12/05 §3.3). DEPENDs on verifiers'
``HarborTaskset`` / ``HarborTask`` (``verifiers.v1.tasksets.harbor``) and lives in verifiers' own environment
(``crucible/.envs/verifiers``: verifiers pins mcp 2, agentenv mcp<2), so the import is lazy.

The loadable plugin is ``crucible.interop.verifiers.vf_taskset`` (``__all__`` exports ``CrucibleTaskset``; a local
``path``; task timeouts kept; separate-verifier tasks graded only in verifiers' fresh box, shared ones after a quiesce:
M11 lab audit R-3). Crucible task directories are Harbor task directories (DESIGN §4.3), so verifiers' own Harbor machinery runs them. What
``CrucibleTask`` changes in ``run_verifier``:

- a masked grade (Crucible writes ``mask.json`` and no ``reward.json``) raises ``TaskError("crucible:masked:<kind>")``
  with the grader's failure kind → an errored trace, which prime-rl excludes from training (never a 0). Without this
  override verifiers would raise a generic "no valid reward" error: still excluded, but the kind is lost;
- the reward is ``reward.json["reward"]`` (the plan's gated reward, Crucible's single producer); the other keys are
  recorded as metrics under a ``crucible/`` prefix (unweighted: verifiers sums only rewards);
- before grading, the grader's own files (``mask.json``, ``steps.jsonl``, ``crucible_reward.json``) are removed: verifiers
  clears only the reward files, so an agent could plant a mask or dense rows (M11 audit);
- ``steps.jsonl`` (the dense channel), when the grader writes it, is attached as ``trace.info["crucible_steps"]`` for the M12 prime-rl
  algorithm that maps rows to sampled nodes by ``call_id``.
"""

from __future__ import annotations


def build():
    """``(CrucibleTask, CrucibleTaskset)`` from the loadable plugin module ``vf_taskset`` (kept for callers of the old API)."""
    from crucible.interop.verifiers.vf_taskset import CrucibleTask, CrucibleTaskset

    return CrucibleTask, CrucibleTaskset
