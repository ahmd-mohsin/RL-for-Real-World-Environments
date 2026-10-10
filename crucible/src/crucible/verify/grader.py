"""Host-side grading of an episode: the plan's verdicts composed into ``reward.json`` and ``steps.jsonl`` (DESIGN §4.3).

``grade_episode`` runs a plan against a live World (or any state view in ``ctx.params``), then:

- composes the terminal reward with ``credit.compose``: ``R = G * [O (1 + a*rubric) + k (1 - O) Phi]``, so a failed
  attempt that reached milestones keeps bounded partial credit, and a masked grade is ``None`` for trainers;
  record-style plans (score gates) keep the plan's normalized score as the reward;
- writes ``crucible_reward.json`` (full), ``reward.json`` (flat numeric, Harbor), ``reward.txt`` and ``report.json``;
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
from crucible.credit.compose import compose
from crucible.credit.operators.potential import potential, trajectory
from crucible.verify.runtime.plan import Context, GradeReport, Plan
from crucible.workflow.steps import Step


@dataclass
class GradeOutcome:
    report: GradeReport
    reward: RewardJson
    steps: list[StepRow]


def to_reward(report: GradeReport, *, milestones: Sequence[Milestone] = (), final_phis: Mapping[str, float | None] | None = None,
              rubric: float = 0.0) -> RewardJson:
    rj = report.reward_json()
    phi = potential(list(milestones), final_phis or {})[0] if milestones else 0.0
    masked = InfraStatus(**report.infra) if report.infra.get("status") == "masked" else None
    out = compose(gates={g: bool(v) for g, v in rj["gates"].items()}, outcome=report.outcome == 1.0, phi=phi, rubric=rubric, masked=masked,
                  components={k: v for k, v in rj["components"].items() if k.startswith("check/")})
    if "score" in report.meta or "abstained" in report.meta:  # record-style or abstention: the plan's reward is the reward
        out = out.model_copy(update={"reward": report.reward if masked is None else 0.0,
                                     "components": {**out.components, "plan_reward": report.reward}})
    return out


async def grade_episode(plan: Plan, ctx: Context, *, out_dir: Path | None = None, tier: str = "full", milestones: Sequence[Milestone] = (),
                        series: Sequence[Mapping[str, float | None]] | None = None, episode_steps: Sequence[Step] | None = None,
                        call_ids: Sequence[str | None] | None = None, harness: str | None = None, rubric: float = 0.0) -> GradeOutcome:
    report = await plan.arun(ctx, tier=tier)
    final = series[-1] if series else None
    rj = to_reward(report, milestones=milestones, final_phis=final, rubric=rubric)
    rows: list[StepRow] = []
    if milestones and series and episode_steps is not None and len(series) == len(episode_steps):
        rows = steps_io.rows_for(episode_steps, trajectory(list(milestones), list(series)), call_ids=call_ids, harness=harness)
    if out_dir is not None:
        write(out_dir, report, rj, rows)
    return GradeOutcome(report, rj, rows)


def write(out_dir: Path, report: GradeReport, rj: RewardJson, rows: Sequence[StepRow] = ()) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crucible_reward.json").write_text(rj.model_dump_json(indent=1) + "\n")
    (out_dir / "reward.json").write_text(json.dumps(rj.flat(), indent=1) + "\n")
    (out_dir / "reward.txt").write_text(f"{rj.reward:g}\n")
    (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=1, default=str) + "\n")
    if rows:
        steps_io.write(list(rows), out_dir / "steps.jsonl")
