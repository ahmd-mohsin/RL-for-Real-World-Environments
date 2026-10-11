# M8 `forge` — agent-swarm task synthesis

**Status: built (2026-10-09).**
- **Toy gate ✅:** the forged variant world's 5 tasks pass the M9 subset.
- **Coding gate, CPU part ✅:** 4 span tasks from task04 v3; traps for 4 of 5 seeds.
- **Coding gate, behavioural part ✅ (2026-10-11):** 17/17 grades on the lab RTX 5090 (`reports/m8_coding_gpu.json`). The 3 spans (people, ncv, kinfu_family): every solution scores 1.0 with all gates green; every start, unported mutant and trap scores 0.0 at its expected gate. The kinfu_family solution, masked earlier for contaminated perf rounds, regraded clean.
- **task01 has no trap:** its only frontier run passed, so there is no failure signal to ground one on.

- **Code:**
  - `src/crucible/forge/` (orchestrator, budget, schema, learnability, dedup);
  - `src/crucible/accept/contract.py` (the M9 subset);
  - toy pack `forge/` (world_synth roles, task adapter, pipeline);
  - coding pack `forge/` (`derive.py`, `forge_gpu.py`);
  - M4 mappers: toy `signals.py`, coding `AgentRunSignalMapper`.
- **Tests:** `tests/unit/test_forge.py` (7 tests: DAG order, staging, resume, feedback retries, infra vs content, budget, canary breach, learnability, dedup).
- **Reports:** `reports/m8_toy_world_synth.json`, `reports/m8_toy_world_synth_llm.json`, `reports/m8_coding_derive.json`, `reports/m8_coding_gpu.json` (written by `forge_gpu collect`), `reports/smoke_cheap.json`.
- **Research:** `docs/research/m7-m8/02-forge-synthesis.md` (8 sources; no framework ships an M8).

## What it is

| Piece | What | Built on |
|---|---|---|
| `forge/orchestrator.py` | A deterministic DAG of role nodes: topological order, sorted within a level.<br>• **Journal:** append-only `events.jsonl`; a node is reused when its input hash matches, hash = role + version + visible input keys + knobs.<br>• **Retries:** deterministic validators feed findings back to the next attempt (at most 3).<br>• **Infra errors** never count as content attempts.<br>• **Task states:** constructed / rejected / abandoned / accepted, with `accepted` bound to the task hash.<br>• **Asymmetric information by staging:** a role receives only its `visible` inputs, never the store.<br>• **Canaries:** the oracle's output carries one. Any node that cannot reach the oracle through visible inputs (transitively) and emits it is a **breach**, a hard failure | RTS `publish.py` (journal resume); agentenv DAG rules; FrontierSmith hash-bound states; Repo2RLEnv ("a prompt instruction is a nudge, not a control") |
| `forge/budget.py` | Reserve/settle ledger over the roles' observed LLM spend; a node that cannot reserve is abandoned, never half-run | Repo2RLEnv `campaigns/budget.py` |
| `forge/schema.py` | Role I/O documents: `Architecture`, `TaskSpec` + `StatedProperty`, `EnvBuild`, `OracleBuild`, `CheckSet` (`CheckSpec` with `states` and `provenance`), `NegativeSet` (`floor`, `trap_reversion`, `defect_mutant`, `cheat`, `valid_alternative`), `AuditReport` | research §12.4 |
| `accept/contract.py` | The M9 subset. **A1:** oracle = 1 on 3 fresh grades, identical replay end states, gold check. **A2:** no-op and naive floors = 0. **A3:** each negative fails its intended gate, valid alternatives pass, every gate family has a negative, mutation score ≥ 0.9. **A4-lite:** replays agree. **A5:** hidden strings vs everything agent-visible, plus instruction leak tokens. **A7:** two-way property ↔ check coverage. **A8:** ≥ 80% provenance, every trap cites a resolvable M4 signal, and trap negatives cite it too | NeMo-Gym `verifier_fixture`; `vf-validate`; OpenEnv RFC 008; R2E probes (valid alternatives) |
| `forge/learnability.py` | 4p(1−p) on the binary outcome per (policy, harness), Wilson interval, sampling 4 → 8 → 16 until the band is decided. Bands: train p ∈ [0.1, 0.7]; bench frontier upper bound ≤ 0.2. `advise` moves generator knobs, never instruction obscurity | verifiers `proposer_solver` (MIT) |
| `forge/dedup.py` | Layers: L0 content, L1 instruction prefix + oracle footprint, L2 start/end state fingerprints, L3 check-signature Jaccard ≥ 0.8. The earliest candidate wins | R2E `_oss_instruct.py`; general_agent `DB.get_hash` |
| toy `forge/` (world_synth) | **Architect:** scenario from a seed, tasks before data, properties stated as instruction sentences, the trap stated as a property. **Env builder:** seed rows that serve every literal, distractors, an archived policy as a decoy, frozen time. **Oracle builder:** privileged, so it knows the supervisor. **Test writer:** never sees the oracle; checks are compiled from the properties and policies. **Adversary:** floors, trap reversions, procedural mutants, valid alternatives. **Auditor:** deterministic. LLM variants of the Architect (narrative) and the Test writer, both validated | AWM stage order; MiMo superseded-document decoys; EnterpriseOps-Gym infeasible requests; SWE-smith procedural modifiers |
| coding `forge/derive.py` | **G-span:** task04's oracle split by module. The start = base + the oracle's other modules, **with sibling forks excluded**. Hidden gates = the scoped task04 kit suites; mined milestones are used for Φ only. **G-trap:** M4 gate-level frontier failures → C2 on (frontier version, oracle version) of the shared area → separating static checks, plus the frontier's code as the trap reversion | M5 `mine.from_diff`; M6 C2 `from_correction`; SWE-smith PR mirror |

## Exit gate

| Requirement (ARCHITECTURE M8) | Result |
|---|---|
| The toy variant world and its tasks pass M9 | ✅ `toy/helpdesk_v2` (seed 7; 5 customers, 19 orders with 7 distractors, an archived-policy decoy, a month-end freeze event). **5/5 tasks accepted:** A1 (3/3 fresh grades = 1.0, identical end states), A2, A3 (6–10 negatives per task, mutation score 1.0, valid alternatives pass), A4-lite, A5, A7, A8 (100% provenance). 0 paid calls; replayed from the journal at no cost |
| Every derived span task passes M9 | **CPU ✅, behavioural ✅.** 4 spans from task04 v3: people (H2, 8 files), NCV (H2, 3 files), KinFu (H3, 20 files), KinFu-LS (H3, 21 files). Each passes the CPU subset: static gate (oracle passes, start and unported mutants fail), A5 (clean instruction, no oracle lines in the start, sibling overlap ≤ 20%), A7, A8. Behavioural ✅: 17/17 grades in the verifier image (scoped kit grader, `--pre-patch`) on the lab RTX 5090 |
| ≥ 1 trap task per seed, citing its signal | **4 of 5 seeds.** Each trap's C2 checks pass the oracle and kill the frontier's version, and each cites a resolvable M4 signal:<br>• task04: sol v3 failed `correctness/ncv` → NCV area, 27 separating checks;<br>• task03: sol failed `api` → 51;<br>• task02: sol failed `ops` → 25;<br>• task01v2: sol failed `tests_gpu` → 264.<br>**task01: none.** Its only frontier run (Fable 5.1) passed, so no failure signal exists. A construction-transcript mapper (M4) would be the source |
| G-hindsight on recorded workflow data (toy) | ✅ **6/6 tasks accepted** (`m8_toy_hindsight.json`). The recorded refund cases T1001, T1003, T1013, T1015, T1018 and T1021 become World tasks graded on the outcome that **held up** after corrections. Four cases carry the expert's real first response as a trap negative, each citing its M4 signal (violation, revert or rework). A rule that a correction revealed ("used add-ons are not refundable", from T1021's revert) is compiled into a policy invariant with that signal as provenance. T1003's fee rule rests on one supervisor correction (240.00 → 230.00); the report says so |
| Calibrator (learnability steering) | ✅ `forge/calibrate.py` on the 5 forged toy tasks, with the cheapest model as the stand-in "policy being trained" (k = 4, $0.003). Results:<br>• p = 0 on T1, T2, T4 and T5;<br>• p = 0.25 on T3 (interval 0.05–0.70, learnability 0.75).<br>The advice is "sample more" for every task, because k = 4 never decides the band: the 0/4 interval [0, 0.49] overlaps p ∈ [0.1, 0.7]. Knobs move only once the 4 → 8 → 16 schedule decides the band or is spent. Failure modes recorded per attempt: outcome checks (no note, wrong status) and the policy gate (unapproved refunds) |
| The LLM-role path | The cheapest model (`mistral-nemo`) as Test writer wrote parameterized SQL three times despite the feedback. The node **failed closed** and the plan was rejected, never accepted with weak checks. The LLM Architect narrative passed its validators (literals and graded sentences verbatim) |

## Findings

1. **The contract caught the forge's own gaps.**
   - The Adversary first had no negative failing T1's policy gate (a ≤ 50 USD refund has no approval trap). A3's per-gate coverage rejected T1 until a generic double-refund mutant was added.
   - The journal then reused the old Adversary output, which is correct behaviour: **role logic changes must bump the role version**, which is part of the input hash.
2. **Span starts can leak through sibling modules.** KinFu-LS is a fork of KinFu, so 67–70% of a KinFu span's new lines were already in the start, inside the expert's port of the sibling. Spans now exclude sibling forks, and A5 measures the overlap: 6.5% (KinFu) and 8.3% (KinFu-LS) after the fix.
3. **Span acceptance is behavioural by nature.** The CPU tier (texture-reference ban, C2 static checks) cannot kill a naive port. Following M6's finding that gold-token milestones are brittle, mined milestones are Φ only and never gates. A1/A2/A3 for coding spans therefore run in the verifier image.
4. **Frontier failures are trap sources the seeds already have.** The M3 run traces kept each grade's gate events, but no M4 mapper read them. The new `coding/agent_run` mapper makes each failed hidden gate a signal that traps cite (A8).
5. **The seed instruction trips the leak scrubber.** "upstream commit `c47198828`" matches Repo2RLEnv's commit-SHA pattern. Derived tasks drop it; the seed keeps it, because the base commit is not a fix.
6. **Corrections reveal policy.** In G-hindsight, a case whose first refund was reversed ("the add-on was used 40 times") teaches a rule the written policy never stated. The contract caught that the checks did not enforce it: refunding the used add-on failed only the outcome. The rule is now an invariant grounded on the correction's signal. This is the mechanism for grounding tasks on an organization's own records: the workflow data supplies both the answer and the traps.
7. **Weak models fail the Test-writer role.** Deterministic validators plus fail-closed rejection keep that from becoming weak tasks. The role needs a stronger model in practice; this is recorded as a knob (`--llm`).

## Not done

- G-transplant (OOD tasks on unseen repos), G-tighten. (m13 G-1..G-5 added the concurrent orchestrator, chain/fan-in composition with A14/A15/A17, the solver ladder and the A16 red-team hacker; see DESIGN.md.) G-hindsight exists for the toy logs only; for coding it needs per-step expert states (M5 replay).
- Frontier calibration (A10: ≥ 2 frontier runs per task; paid) and the Curator (splits, difficulty ladder).
- Harbor-run sandboxed builder roles (`role_runner.py`) and the agentenv human-review step.
- A task01 trap, via a construction-transcript signal mapper (M4).
- Harbor `task.toml` emission for forged tasks (M11).
