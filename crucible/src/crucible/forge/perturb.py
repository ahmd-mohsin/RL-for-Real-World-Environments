"""The perturb operator (m13; audit 06 §4.3): change the *task* (its World or start state), not the solution.

Kinds (SWE-smith operators on the start state, and the enterprise analogues): ``distractor`` (more entities the task must
leave alone), ``decoy`` (plausible but wrong material), ``policy_amendment`` (a rule changes), ``clock_event`` (something
happens mid-episode), ``injected_bug`` (coding: a defect in the start state).

- A **cosmetic** perturbation must not change what a correct solution is: the unperturbed oracle still passes the
  perturbed task (A18, with A17's non-interference intent).
- A **semantic** perturbation changes it: the Builder re-derives the oracle (which must pass A1 again), the instruction
  gets a new property sentence (A7: every graded property is stated), and the unperturbed oracle must *fail* the
  perturbed task, at the perturbation's gate when it names one (A18). A "semantic" perturbation the old oracle still
  passes changed nothing the grader sees: mislabelled, or the grader misses it.

The pack applies the perturbation to its World (``crucible_pack_toy.forge.perturbed``); this module carries the record
and the spec change.
"""

from __future__ import annotations

from dataclasses import dataclass

from crucible.forge.schema import StatedProperty, TaskSpec

KINDS = ("distractor", "decoy", "policy_amendment", "clock_event", "injected_bug")


@dataclass
class Perturbation:
    id: str
    kind: str
    semantic: bool
    sentence: str | None = None  # semantic: the new property sentence
    gate: str | None = None  # semantic: the gate the unperturbed oracle must now fail
    description: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"perturbation kind must be one of {KINDS}")
        if self.semantic and not self.sentence:
            raise ValueError("a semantic perturbation states its new property (A7)")


def perturbed_spec(spec: TaskSpec, p: Perturbation, *, task_id: str) -> TaskSpec:
    props = list(spec.properties)
    instruction = spec.instruction
    if p.semantic:
        props.append(StatedProperty(id=f"perturb_{p.id}", text=p.sentence, kind=p.kind))
        instruction = instruction.rstrip() + " " + p.sentence
    return spec.model_copy(update={"task_id": task_id, "instruction": instruction, "properties": props,
                                   "difficulty": {**spec.difficulty, "perturbation": {"id": p.id, "kind": p.kind, "semantic": p.semantic}}})
