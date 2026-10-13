# M14 audit: packs and the RLM agent (2026-10-13)

Scope: the pack framework (`core/schema/pack.py`, `studio/pack.py`, `studio/accept_adapter.py`, ingest and redaction,
the registry, `pack new` / `check` / `export`), the four packs (`_template`, `toy`, `coding`, `finance_ap`), the core
tool layer the packs expose (`world/packaging/mcp_tools.py`), and the RLM agent (`actors/rlm/*`,
`runtime/harness/rlm.py`, `accept/repair.py`, the RLM record and credit paths). Base `f2d9e18` (M13).

Study: two sweeps (G1: the RLM against Prime Agent's code, papers and blog, verifiers and the prime environments; G2:
packaging and validation in Harbor, verifiers, NeMo-Gym, OpenEnv, agentenv, EnterpriseOps-Gym, FineEnvs, MiMo), three
hunts (H1 the framework, with 25 mutant packs; H2 the packs' domain code as a red team; H3 the RLM kernel, policy,
continual harness and manager), and a validation campaign on the fixed tree (`M14-campaign.md`). Regression tests:
`tests/unit/test_m14_audit_{framework,packs,rlm}.py` (24 + 26 + 26 functions); each batch ran them against a pristine
checkout of `f2d9e18` (framework: 62 of 72 cases fail there, the other 10 are the control packs that must pass and
mutants the old check already caught; packs: 31 of 31; RLM: 26 of 26).

## 1. Requirement ledger

| # | Requirement (source) | Before | After |
|---|---|---|---|
| L1 | A pack is accepted only if its grader can fail (H1 V1–V4, V8, V21; G2 D1, D2, D6, D7) | 19 of 25 mutant packs and 17 of 21 broken finance_ap copies passed `pack check` with exit 0: always-pay graders once fixtures were missing, empty case lists, NaN and out-of-range rewards, `fails_gate` naming a gate never emitted, a no-op at 0.999 | missing or empty fixtures and tasks fail; every reward finite and in `reward.range`; a fixture case at each end of the range; gold at the best end, no-op at the worst; `fails_gate` must name an emitted gate |
| L2 | Adversaries are part of a pack (G2 pick 2, H2) | the pack's own `approve_all` was never run | each pack declares `{"policy", "expect_max"}` adversaries, run on every anchor and generated item at the semantic level: finance_ap 5, toy 2, coding 2, template 1 |
| L3 | Determinism is measured, not declared (H1 V9, G2 D6) | each case graded once; a 15–30% flaky grader passed 3–4 of 11–12 runs | seeded packs re-grade every case 3× on fresh Worlds and rebuild tasks and generator items |
| L4 | Requirements are verified and skips are loud (H1 V6, G2 D4) | a self-declared `network` turned semantic into a silent pass; coding's semantic could never run | docker / gpu probed (10 s timeout); a skip is reported, exit 2 unless `--allow-skip`; coding has CPU dense-replay tasks |
| L5 | Checking a pack checks that pack (H1 V7, V11, V15, G2 D5) | module caching let a broken copy pass after the real pack; hooks could point at another pack; a pack's `numpy.py` shadowed numpy | each check isolated (its own `src/`, `sys.path` and `sys.modules` restored; `--all` one subprocess per pack); hooks must live in the pack's package; the pyproject entry point must match and `register()` must register this pack |
| L6 | Secrets never leak through ingest (H1 V5, V17, V20) | a misspelled `redaction_env` was a no-op and `crucible label` stored the secret; `../`, absolute roots and symlinks were ingested | a missing redaction file fails static and label refuses to store; ingest paths confined after following symlinks |
| L7 | Metadata is real (H1 V10, V18, V19, G2 D10) | `TBD` licences, raw template copies (`__PACK_NAME__`) and `version: banana` passed; names collided | SPDX licences agreeing with pyproject; placeholders scanned across all files; name and PEP 440 version validated; unique names with registry lookup; `--publish` stricter |
| L8 | Packs go through the M9 acceptance contract (G2 D3) | `crucible accept finance_ap` crashed; finance_ap tasks never met A3 / A5 | `pack check --level accept` runs M9 on fixture tasks, anchors and a generated item; toy, finance_ap and a fresh scaffold are provisional on every task (only A10 pending: it needs paid calibration) |
| L9 | A policy cannot hide its own failure (H2 V1) | one out-of-range integer argument raised `OverflowError` inside sqlite, counted as infra: approve-everything + that call went from 0 to "3 of 3 masked" in all three SQLite packs | argument-binding errors (int64 range, non-finite, schema) are the policy's scored `invalid_arguments`; a bug in a tool body on clean arguments stays infra (core `mcp_tools.py`) |
| L10 | Graders grade what the policy says (H2 V2, V3, V11, V13) | toy: any typed supervisor name passed (`LIKE '%Sam Lee%'`), 3 × 43.33 dodged the 50 USD approval threshold, a note `"x"` passed, a premature close was erased by a re-close | approvals requested through the World and matched exactly (the supervisor seeded outside agent-visible files), the threshold on the refunded total, notes must say why, the first resolution counts |
| L11 | Dense credit cannot be gamed (H2 V4, V5, G2 D4) | coding: comment stuffing reached Φ 0.92–1.00, above every honest failing run; finance_ap: a control breach cost no Φ; coding fixtures were vacuous (`compose(outcome=True)`) | coding Φ on code only, gated by positive milestones (stuffing and deletion-only now 0); finance_ap Φ zeroed by a breach; fixtures exercise Φ |
| L12 | The thesis domain needs the policy to solve (H2 V6–V8, V10, G2 D8) | finance_ap: one invoice of each kind per world, no invoice firing two rules, byte-identical duplicates; a never-read-the-policy agent solved 100% and a memorized 2% tolerance 73% | generator v2: per-world tolerance and rule order stated only in the policy, compound, near-miss, disguised-duplicate and infeasible cases, varying n and base rates; generator v3 after the validation campaign; a memorized 2% solves 15%, every fixed rule order ≤ 18%; all-or-nothing reward (partial credit let every grader mutant survive A3) |
| L13 | Packs are distributable (G2 D9) | wheels carried no `pack.yaml` or data; `PACK_ROOT` pointed into site-packages | data force-included under `<pkg>/_pack/`, `PACK_ROOT` falls back to it, crucible version bounds declared |
| L14 | Model code never runs with the host's credentials (H3 V1, G1 D8) | `LocalLauncher` was the default for `RLMPolicy`, `RLMAgent` and the manager: it read the parent's API keys and the repo `.env`, reached the network and left processes behind | the isolated Docker kernel (no network, read-only root, uid 65534) by default; host only with `allow_host=True` and an allow-listed environment; no isolation → refused in train and eval |
| L15 | The kernel protocol cannot be forged (H3 V2, G1 D12) | cell code shared fd 1 with the protocol: a forged frame turned the policy's own failure into an infra mask; a 70 MB line crashed the host | protocol on a private fd (Prime Agent `repl.py` port, MIT), validated frames, `protocol_error` replies, caps with markers, interrupt, owner watchdog |
| L16 | Only the policy's own tokens are trained, and faults mask (G1 D1, D2, D4, D14, H3 V5, V6) | a sub-LLM answered by another model was trained as the policy; sub-call and sandbox faults scored 0; sub turns diluted root credit (+0.500 → +0.045 with 10 subs) and were chained into the critic's root sequence | `Turn.model` / `Turn.trainable` honoured by every trainer path; faults masked; sub turns take their parent's credit and are their own critic sequences |
| L17 | Budgets and evidence are enforced (H3 V3, V4, V7, V8, V11) | killed kernels kept running in containers; sub-calls unbounded; the harness A/B kept 58 of 100 no-effect edits; commit trusted a subset verification; spawns unbounded in breadth | in-container kill; per-episode / per-cell / concurrency / token budgets; paired, interleaved, ≥ 3-seed A/B with a noise gate; commit needs the full verification; `TreeBudget` |

## 1a. The validation campaign on the fixed tree (`M14-campaign.md`)

≈1,550 runner episodes and ≈3,200 fuzzed ones, CPU only: every oracle 1.0, every no-op 0, every declared adversary at or
under its bound, no NaN, nothing masked counted; Qwen2.5-0.5B rollouts (native and inside the Docker RLM kernel) train-valid
with a ≤ 1e-4 logprob gap; GRPO smokes on toy and finance_ap with 0 masked rollouts. It found 14 more defects, all fixed
before the commit (`tests/unit/test_m14_audit_validation.py`, 25 tests, and 44 pack tests):

| id | defect | fix |
|---|---|---|
| B1 P0 | a lone surrogate in any tool argument masked the episode (unknown blame) and the health rule aborted the bench | refused at `World._call_tool` as the policy's `invalid_arguments`; trace writing surrogate-safe; residual encode errors map to `policy_misbehavior` |
| B2 P0 | coding Φ 1.0 on all four seeds from oracle tokens in string literals or dead code | a lexer / parser code view (`packs/coding/.../codeonly.py`): literals blanked, `#if` evaluated, constant-false and unreachable code and unused new functions dropped; both attacks declared as adversaries (Φ 0) |
| B3 | a fixed host seed-tar path: concurrent Worlds corrupted each other | one tarball per World, removed; failures are infra |
| B4 | RLM containers leaked when a process ended mid-rollout | owner and expiry labels, bounded lifetime, shielded start / remove, reaping at start and exit |
| B5 | toy and the scaffold failed `--level accept` | M9 capabilities on every toy and template task; strict backlog items all-or-nothing; accept provisional everywhere |
| B6 | `bench run coding` crashed on an async hook; `requires` ignored | hooks awaited, `requires` refused unless `--allow`, gold / no-op when a pack has no policies hook |
| B7 | finance_ap shortcuts: ignoring hold order 57.8%, whole-unit prices 68%, amount-matched duplicates 54.5%, no aliases | generator v3 (0.3.0): two compound cases on the drawn order, a restated duplicate, a one-cent-over probe, a stopped payee (an alias in 85% of worlds) — fixed order ≤ 18.2% (the 1/6 floor), whole units 0%, amount-matched 0%, ignore-alias 15.25% |
| B8 | the toy backlog accepted any note and unbound approvals | approvals bound to a ticket / order / amount and consumed once; notes must say why |
| B9–B14 | host kernel refused only per attempt; forged `done` frames; misleading key-reuse message; policy `ValueError`s labelled tool faults and unknown args dropped; a truncated oracle reported as broken | refused at construction; HMAC-signed kernel frames (protocol 3); the clash names the other benchmark; tool-layer attribution and unknown-arg refusal; controls sized to the oracle, `oracle_truncated` reported |

Residual (documented): a cell can still reach the frame signer through interpreter internals (its effect stays inside its
own scored episode); MCP-over-HTTP tools cannot separate domain errors; decorated or exported Python functions count as
live code in the coding view.

## 2. Corrections to earlier claims

- `docs/modules/M14-packs.md`: "the coding semantic level runs with `--allow gpu`" — it could not run (no tasks hook);
  now CPU tasks exist. "`pack new demo` passes every level" was true and meaningless: its printed next step failed.
- `docs/modules/M14-rlm.md`: `actors/rlm/manager.py` is `accept/repair.py`; `LocalLauncher` was described as
  credential-free (it inherited SSH, proxy and database credentials); `docs/research/m14-rlm/01-prime-agent-study.md:209`
  claimed a virtual clock and seeded ids for exact restore — restore is exact only for deterministic cells (now a
  lenient replay tells the model which cells diverged).

## 3. Gates and numbers

- `pack check --all` at static / fixture / semantic: toy, coding (now on CPU: 29 fixture cases, 4 adversaries), finance_ap
  ok; `--level accept`: toy, finance_ap and a fresh `pack new demo` provisional on every task (A10 pending paid calibration).
- Validation campaign on the fixed tree: `M14-campaign.md`.
- Full suite: in the commit message. lint-imports 4 kept; ruff clean.

## 4. Open

- NeMo-Gym `manifest.yaml`, Harbor `dataset.toml` and verifiers package exports (G2 pick 8); `pack digest` / upgrade
  (pick 7); the finance_ap "blocked alias" variant (two vendor ids for one payee).
- coding dense Φ: dead code or string literals carrying tokens can still earn credit; only the GPU outcome gate stops it.
- Host-mode RLM is opt-in, not isolation; `ExecLauncher` cannot kill a double-forked `setsid` escapee; `Kernel.orphaned`
  is recorded but not acted on.
- `uv.lock` not refreshed for the packs' new `crucible` dependency.
