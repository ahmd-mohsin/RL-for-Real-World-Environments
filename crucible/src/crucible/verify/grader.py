"""Host-side grading of an episode: the plan's verdicts composed into ``reward.json`` and ``steps.jsonl`` (DESIGN §4.3).

``grade_episode`` runs a plan against a live World (or any state view in ``ctx.params``), then:

- composes the terminal reward with ``credit.compose``: ``R = G * [O (1 + a*rubric) + k (1 - O) Phi]``, so a failed
  attempt that reached milestones keeps bounded partial credit, and a masked grade is ``None`` for trainers;
  record-style plans (score gates) keep the plan's normalized score as the reward;
- writes the reward files through the single producer ``write_reward_files`` (``reward.json`` = the plan's gated reward, as in the
  container; the composed training scalar with κ(1−O)Φ in ``crucible_reward.json["training_reward"]``; masked = no reward file);
- writes ``steps.jsonl`` when the episode's per-step milestone values are given (the dense channel; M6).

Inside verifier images the standard-library ``runtime/run.py`` does the terminal part; this module is the host path
(toy worlds, CPU tasks, and post-processing of a verifier image's ``crucible_reward.json``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from crucible.core.schema.milestone import Milestone
from crucible.core.schema.reward import InfraStatus, RewardJson, StepRow
from crucible.credit import steps as steps_io
from crucible.credit.compose import compose, scalar
from crucible.verify.runtime.plan import Context, GradeReport, Plan, write_reward_files
from crucible.workflow.potential import potential, trajectory
from crucible.workflow.steps import Step


@dataclass
class GradeOutcome:
    report: GradeReport
    reward: RewardJson
    steps: list[StepRow]


def to_reward(report: GradeReport, *, milestones: Sequence[Milestone] = (), final_phis: Mapping[str, float | None] | None = None,
              rubric: float = 0.0) -> RewardJson:
    from crucible.credit.operators.infra_mask import mask_for

    rj = report.reward_json()
    finals = dict(final_phis or {})
    measured = [m for m in milestones if finals.get(m.id) is not None and all(finals.get(p) is not None for p in m.requires)]
    # Φ over the milestones that could be measured: an unmeasured one is not a failed one (M6 audit: it counted as 0)
    phi = potential(measured, finals)[0] if measured else 0.0
    masked = InfraStatus(**report.infra) if report.infra.get("status") == "masked" else None
    if masked is None and milestones:  # C6, wired (it had no production caller): nothing measurable at the end masks
        c6 = mask_for(terminal_phis={m.id: finals.get(m.id) for m in milestones})
        masked = c6 if c6.status == "masked" else None
    out = compose(gates={g: bool(v) for g, v in rj["gates"].items()}, outcome=report.outcome == 1.0, phi=phi, rubric=rubric, masked=masked,
                  components={**{k: v for k, v in rj["components"].items() if k.startswith("check/")},
                              **({"progress/unmeasured": float(len(milestones) - len(measured))} if milestones and len(measured) < len(milestones) else {})})
    if "score" in report.meta or "abstained" in report.meta:  # record-style or abstention: the plan's reward is the reward
        out = out.model_copy(update={"reward": report.reward if masked is None else None,
                                     "components": {**out.components, "plan_reward": report.reward}})
    return out


async def grade_episode(plan: Plan, ctx: Context, *, out_dir: Path | None = None, tier: str = "full", milestones: Sequence[Milestone] = (),
                        series: Sequence[Mapping[str, float | None]] | None = None, episode_steps: Sequence[Step] | None = None,
                        call_ids: Sequence[str | None] | None = None, harness: str | None = None, rubric: float = 0.0,
                        base_phis: Mapping[str, float | None] | None = None) -> GradeOutcome:
    """``base_phis``: φ at the state before the first step, so step 0 is not credited with what already held (M6 audit)."""
    report = await plan.arun(ctx, tier=tier)
    final = series[-1] if series else None
    rj = to_reward(report, milestones=milestones, final_phis=final, rubric=rubric)
    rows: list[StepRow] = []
    if milestones and series and episode_steps is not None and len(series) == len(episode_steps):
        rows = steps_io.rows_for(episode_steps, trajectory(list(milestones), list(series), initial=base_phis), call_ids=call_ids, harness=harness)
    if out_dir is not None:
        write(out_dir, report, rj, rows)
    return GradeOutcome(report, rj, rows)


def write(out_dir: Path, report: GradeReport, rj: RewardJson, rows: Sequence[StepRow] = ()) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # one producer (verify.runtime.plan.write_reward_files): reward.json is the plan's gated reward, as the in-container
    # grader writes it; the composed training scalar (with κ(1−O)Φ partial credit) goes to crucible_reward.json only
    plan_rj = report.reward_json()
    write_reward_files(out_dir, plan_rj, report.to_dict(),
                       extra={"training_reward": scalar(rj), "composed": json.loads(rj.model_dump_json())})
    if rows:
        steps_io.write(list(rows), out_dir / "steps.jsonl")
