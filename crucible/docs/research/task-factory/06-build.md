# M17 build: modules, owners, fixed interfaces

Package `src/crucible/factory/` (new). It builds on `forge/` (orchestrator, swarm, repo, specialists, hacker, judge,
ladder, learnability, curator, dedup, economy, tighten / perturb / stitch) and `accept/` (A1–A18, `mutation/`). Reuse,
don't duplicate. Each owner writes only its files and `tests/unit/test_m17_<owner>_*.py`; requests for other files go in
the report.

| Owner | Files |
|---|---|
| P | `factory/cards.py`, `factory/ledger.py`, `factory/labels.py`, `interop/agentenv/plugin/labels.py` (explorer label routes), `factory/publish.py` (agentenv Task + pinned Eval) |
| Q | `factory/machine.py` (declarative cyclic machine), `factory/models.py` (role → model, cross-family rule, OpenRouter), `factory/driver.py` (per-task machine), `factory/root.py` (root curator, coverage matrix, leases, budget) |
| R | `factory/floors.py`, `factory/probes.py` (mutants + valid alternatives, probe library), `factory/differential.py`, `factory/audit.py` (FailureCause auditor, FP/FN), `factory/flake.py`, `factory/horizon.py`, `factory/ortho.py`, `factory/transfer.py`, `factory/goldset.py` |
| S (after P, Q, R) | `factory/lanes/executable.py` (mine real fix + test commits), `scripts/factory_run.py`, the end-to-end CPU validation and `docs/research/task-factory/07-validation.md` |

## Fixed interfaces (`factory/cards.py`, P implements exactly these names; pydantic `CrucibleModel`)

```python
class ObservationRecord:  # R3
    id: str; type: Literal["real_pr_tests", "expert_end_state", "human_graded_outcome", "production_trace",
                           "recorded_session", "observed_model_failure", "reference_derived", "simulator_state"]
    pointer: str; sha256: str; created: str; by: str            # by: "human:<id>" | "model:<id>" | "miner:<id>"
    first_class: bool  (property: type not in {"reference_derived", "simulator_state"})

class IntentCard:
    id: str; behaviour: str; failure_mode: str; evidence: list[str]   # ObservationRecord ids (observed_model_failure)
    difficulty_axes: list[str]; coverage: dict[str, str]              # {"behaviour": ..., "surface": ...}
    probe_set: str | None; status: Literal["candidate", "approved", "rejected", "retired"]; approved_by: str | None

class Requirement:  id: str; text: str; prompt_span: tuple[int, int]; kind: str  # kind keys the probe library
class CheckRef:     id: str; requirement_ids: list[str]; observation_ids: list[str]; gating: bool

class RequirementLedger:  # R1, enforced both ways
    requirements: list[Requirement]; checks: list[CheckRef]
    def problems(self, observations: dict[str, ObservationRecord]) -> list[str]  # unchecked req, unstated check,
                                                                                 # gating check w/o first-class obs

FailureCause = Literal["skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake"]

class VerifierCard:  # §6 of 05: every number with its denominator
    verifier_version: str; floors: dict[str, float]; mutant_kill: tuple[int, int]; alternatives_pass: tuple[int, int]
    hacks_confirmed: int; differential_disagreements: tuple[int, int]; fp: tuple[int, int]; fn: tuple[int, int]
    flake_upper: float | None; kappa: float | None; horizon: dict | None

class TaskCard:
    id: str; family: str; intent: str; surface: dict[str, str]; prompt: str; ledger: RequirementLedger
    world_ref: str; checkset_ref: str; oracle_ref: str; verifier: VerifierCard
    stats: dict  # pass rates with Wilson intervals per (policy, harness, k); cause mix; all-pass/mixed/all-fail
    lineage: dict; recipe_version: str; status: Literal["draft", "accepted", "rejected", "quarantined", "retired"]

class Label:     # append-only (labels.py); never edited, only superseded
    id: str; item: str; item_sha256: str; labeller: str; value: str; reason: str; at: str; supersedes: str | None
class Decision:  # the decision store (outer loop)
    id: str; subject: str; actor: str; action: Literal["keep", "drop", "revise", "approve", "reject"]; reason: str
    criteria: list[str]; recipe_version: str; at: str
```

## Machine (Q): `factory/machine.py`

PORT (MIT) of Prime Agent's factory validator, guards and canonicalizer (`docs/research/task-factory/01` §1). Rules:
- a machine is a dict/YAML of states, each with a `role` (or deterministic `fn`) and typed ports;
- transitions are guarded; cycles are allowed with `max_entries`; exhaustion is `escalate` (pause plus a decision
  request), never `done`;
- loop-back carries `findings` to the owning state;
- the run state is journaled to disk (Prime keeps it in kernel memory: the defect to avoid) and resumable;
- run caps: USD (the forge ledger), transitions, wall-clock.
Cross-family rule: a state may declare `different_family_from: [state]`, and the validator refuses a model assignment
that breaks it.
