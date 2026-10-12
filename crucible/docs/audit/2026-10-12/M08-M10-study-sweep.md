# M8-M10 study: every local source, read for forge, acceptance and runtime, and the picks

Five sweeps read every local source for M8 forge, M9 accept and M10 runtime, at code level, with probes. Three defect
hunts audited the modules themselves. Each sweep is its own document:

| Group | Sources | Document |
|---|---|---|
| G1 | Harbor, repo2rlenv, SWE-smith, mini-swe-agent, FineEnvs and its HF spaces, the RL environments guide, OpenEnv | `M08-M10-sweep-G1.md` |
| G2 | verifiers, prime-environments, research-environments, prime-agent, prime-rl, GitSwarm, autoresearch | `M08-M10-sweep-G2.md` |
| G3 | NeMo-Gym, NeMo-RL, EnterpriseOps-Gym, Karotte, Xiaomi's verl fork with mimoagent, the MiMo v2.6 RL release | `M08-M10-sweep-G3.md` |
| G4 | agentenv (both runners), OpenRLHF, ROLL, slime, SkyRL, checkpoint-engine, AReaL, TRL, verl, rLLM | `M08-M10-sweep-G4.md` |
| G5 | every article in `07-articles` (with the GitSwarm and Prime Agent papers), all 33 datasets in `08-hf-rl-env-datasets` | `M08-M10-sweep-G5.md` |
| hunts | the M8, M9 and M10 code | `M08-study-defects.md`, `M09-study-defects.md`, `M10-study-defects.md` |

Every sweep found the same few Crucible defects independently. Those are the most certain findings:

- One exception from a harness or adapter aborts a whole `Runner.run`, and its budget reservations stay open (G1 D1, G2
  D1, G3 D1, G4 D1, G5, M10 V11).
- The re-collect retry reads the recorded base commit from the new attempt's directory, so it diffs against the
  agent-movable `base` tag (G1 D2, G2 D5, G3 D3, M10 V2). This is a hole in the M7 audit's own fix.
- A3 passes with zero mutants (G1 D5, G3, M9 V3).
- A torn journal line loses the next result (G1 D3, G2 D4, G3 D2).
- Masked outcomes count as failures in calibration, learnability and the ladder (G2 D10, G3, G4).
- The forge ledger double-counts concurrent spend and aborts on an over-cap settle (G1 D4, G2 D6-D7, G4 D20, M8 V3-V4).
- `accept/imports/` (I0-I10) and `runtime/harness/harbor.py` do not exist, although ARCHITECTURE lists them (G1, G5, M9
  L24, M10 L21).

## The picks

DEPEND = use the framework, PORT = copy code (licence noted in the group document), PATTERN = rebuild the idea.
Effort: S < 1 day, M a few days, L more. "Built" means built in this audit round.

### M8 forge

| # | Pick | Source | Kind, effort | Status |
|---|---|---|---|---|
| 1 | Masked rollouts leave learnability, ladder and calibration (outcomes `bool | None`) | MiMo DAPO filter (`replay_buffer.py:257-268`), SkyRL zero-variance test, AReaL `min_usable_group_size` | PATTERN, S | built |
| 2 | Contain node and worker failures; settle in every path; per-node timeout and backoff | prime-rl dispatcher, agentenv `RetryConfig`, GitSwarm charged episodes | PATTERN, S | built |
| 3 | Validators and canary flag in the journal key; a torn journal tail is skipped | repo2rlenv `publish.py`, NeMo-Gym torn-tail truncation | PATTERN, S | built |
| 4 | Decontamination at curation, failing closed for bench | research-environments `eval_hashes.py`; `bench/contamination.py` | PATTERN, S | built (hook; the bench layer wires the corpus) |
| 5 | Shingle-based dedup L1 | Repo2RLEnv `_oss_instruct.py`, GitSwarm redundancy data | PATTERN, S | built |
| 6 | Information-flow zones and shingle quote scans | GitSwarm provenance, our G-7 | PATTERN, M | built |
| 7 | Durable forge spend with per-call receipts | Repo2RLEnv `campaigns/budget.py`, `llm.py` | PORT, M | queued (scope attribution built) |
| 8 | Code-level bug generator: SWE-smith operators kept only when an execution check passes | SWE-smith `bug_gen/`, Repo2RLEnv validation | PORT, L | queued |
| 9 | Per-prompt pass rates from training back to the forge | SkyRL, slime, verl refill caps | PATTERN, L | queued |
| 10 | Red team until clean (repeat A16 after each fix) | G5 articles | PATTERN, S | queued |
| 11 | Imported tasks as seeds (termgrade, Tasksmith, TB-science, Terminal-Lego) | G5 datasets | DEPEND, M | queued (needs M9 imports) |

### M9 accept

| # | Pick | Source | Kind, effort |
|---|---|---|---|
| 1 | Imports as acceptance adapters over `HarborInPlaceTask` and `NemoGymTask`; refuse unknown grading keys; honour separate verifiers | G5 datasets, PI taskset `validate()`/`apply_gold_patch()` | PATTERN, M |
| 2 | A kill counts only when the tests ran and failed; zero mutants fails A3 | SWE-smith, Repo2RLEnv | PATTERN, S |
| 3 | A2: malformed submission and finite in-range reward; each floor declares its gate | NeMo-Gym verifier fixtures | PATTERN, S |
| 4 | One bench band (Wilson upper bound) for A10 and the ladder; success by the task's own predicate | SkyRL, prime-rl pools | PATTERN, S |
| 5 | Atomic, parse-guarded cache and grade files; the key covers every grading module | Repo2RLEnv journal | PATTERN, S |
| 6 | Backfill quarantines on error; detector ids tested against the registry | MiMo, NeMo-Gym | PATTERN, S |
| 7 | Karotte's credential scan and attack list in A5/A13 | Karotte (MIT) | PORT, S |
| 8 | Repair commits only after the full contract | Prime Agent merge rule | PATTERN, S |
| 9 | A19: World snapshot → restore → snapshot round trip | agentenv universe round trip | PATTERN, S |
| 10 | Harbor quality checker as advisory A7b | Harbor | DEPEND, S |

### M10 runtime

| # | Pick | Source | Kind, effort |
|---|---|---|---|
| 1 | Contain every attempt: catch, classify, settle, per-attempt deadline, drain | agentenv `_run_safe`, rLLM, OpenEnv, prime-rl | PATTERN, S |
| 2 | Re-collect from the base recorded by the first attempt | our M7 fix, completed | S |
| 3 | Map foreign error classes (Harbor, validators); unmapped names are `unknown` blame | Harbor, NeMo-Gym | PATTERN, S |
| 4 | Retryable verdict per failure kind; backoff outside the semaphore; no retry of a masked limit stop | prime-agent `error_classify.rs`, verifiers `retries.py`, Karotte | PATTERN, S |
| 5 | Journal and ledger hardening: torn tail, single writer, duplicate refusal | NeMo-Gym, prime-agent `lease.rs`, Repo2RLEnv | PORT/PATTERN, S |
| 6 | Secret scrubbing at the record boundary | our M3 redaction | S |
| 7 | Checkpoint resume after a process crash; no re-charge of restored calls | prime-agent append-only history | PATTERN, S |
| 8 | Sandbox hygiene: liveness, start-failure cap, TTL, orphan sweep | rLLM WarmQueue, ROLL, agentenv `teardown_run` | PATTERN, M |
| 9 | `harness/harbor.py` and a verifiers harness | Harbor, verifiers | DEPEND, M |

## Corrections to earlier Crucible documents (from the sweeps)

- NeMo-Gym has no circuit breaker (`rollout_health.py` is a post-run audit) and no per-rollout timeout (G3).
- `vf-validate`'s gold check is "unchecked" for every verifiers taskset except gsm8k (G2).
- research-environments contains no synthesis code to port; OpenEnv's AWM environment downloads a pre-built dataset
  (G1, G2).
- agentenv has a second runner (`bundle/`) the earlier studies missed, and its review step has no 10% sampling (G4).
- rLLM CompactFiltering is off by default; verl ignores `max_num_gen_batches`; TRL has no `experimental/openenv` (G4).
- Karotte's `ExecutableJudge` scores a grader crash as 0: port its runner split, not its judges (G3).
- GitSwarm's "1 build in 6 wasted" is about 6-7% of tokens (G5).
