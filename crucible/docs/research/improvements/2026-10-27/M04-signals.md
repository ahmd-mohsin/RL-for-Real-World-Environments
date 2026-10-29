# M04 signals: audit against the lab RL repos (2026-10-27)

Crucible HEAD `b477006b` (`uzair` = `main` = `dev`). Citations: Crucible source under `crucible/src/crucible/`
(`signals/pivots.py:16`); other Crucible files relative to `crucible/` (`DESIGN.md`, `docs/…`, `tests/…`), with `CP/` =
`packs/coding/src/crucible_pack_coding/`, `TP/` = `packs/_toy/src/crucible_pack_toy/`; references as `<folder>/<path>:line`,
the folder dropped inside the repo's own §5 entry; commits in §4. **[ran: probe]** = reproduced in this run (`notes/`).
Work folder (notes, probes): `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-27/M04-signals/`.

## 0. Summary

- **Where M4 matters today.** `crucible label` turns M4 signals into M15 rows (pivots → DPO; graded-wrong turns kept out
  of SFT/KTO), the pack gate scripts fit reliability and score the exit gate, and M8 forge reads agent-run gate signals.
  M6 and the trainer read no grades, polarity or reliability (`credit/signed.py:29-30` has no M4 producer), and
  `crucible label` never applies reliability, so `Signal.reliability` is None in production.
- **Established gaps.**
  - Customer-rejected and escalated replies are positive targets: the toy `crucible label` run writes 2 of 10 SFT rows
    and 2 of 10 KTO-desirable rows from them **[ran: v3_mv1_label_trains_rejected]** (R-1, must).
  - A trace whose `extract` raised still becomes rows with no M4 verdict; A15's isolation fails open (R-2).
  - PR check runs that `timed_out` or hit `startup_failure` grade their commit -1, and a later hotfix counts that as a
    confirmed failure in the gate's reliability row **[ran: v2_probes]** (R-3).
  - Gate reports name only the labels file: an edited reply moves type accuracy 1.0 → 0.988 under the same hash
    **[ran: v3_gate_inputs_drift]** (R-4). A pass-then-pass of one CI check "contradicts" the first pass (R-5).
  - Open from earlier: comment rejections are still circular (0/10 confirmed), verdicts one-sided (NEW-1), no held-out
    coding labels, second labeller or kappa; neither gate `confident` (73 straight outcome agreements reach 0.95, not
    51: NEW-2).
- **Recommendations.** 14 items: 1 must (R-1), 4 should (R-2..R-5), 9 later (L-1..L-9). Top: R-1, R-2, R-3, R-4.
- **Ledger.** 99 items: 1 done, 22 partly, 73 not done, 1 superseded, 2 wrong. The 67 M4 tests pass at HEAD.
- **Against the earlier audit.** None exists in this series (no dated `M04-*.md`). Against the 2026-10-12 module audit
  (fixes in `b65754b2`): 3 items were introduced by its fixes (R-2 by A15, R-5 by R9, L-1 by its event-key signal ids);
  11 were present then, 9 of them on its open or study lists (R-3, R-4, L-2..L-6, L-8, L-9) and 2 missed (R-1, which
  the M15 V8 fix `775f30f1` closed for corrective signals only, and L-7).
- **Coverage.** 33 reference folders: 2 primary, 27 secondary, 4 not relevant. No sweep cluster failed and no verify
  batch failed. 27 candidates were verified in 3 batches and merged into the 14 items; none was rejected outright, and
  five proposed designs were replaced (§7).

## 1. What the module does

M4 turns a trace's raw events (a review comment, a revert, a CI run, a ticket reopen, a user interrupt) into typed
signals from one closed taxonomy of 13 types (`signals/taxonomy.py:24-38`). It decides which earlier step each signal
grades (a pack rule such as blame, else the last writer; session rules grade the proposing call), labels the trace's
outcome through pack labelers, estimates per (family, type, actor class) how often a signal agrees with later
independent evidence, marks pivots (steps graded wrong and then corrected), and is gated against hand labels.

| Contract | |
|---|---|
| Inputs | `crucible.trace/v1` from M3 connectors and World episodes (`core/schema/trace.py:71-97`): `event_key` (`:52-62`), `data.writes` (`core/agent_events.py:120-171`), connector facts (`CP/connectors/github_pr.py:232-333`) |
| Outputs | `Trace.signals` (`core/schema/signal.py:39-49`), `Trace.outcome`, `outcome.details.pivots` (`signals/pivots.py:33-36`), reliability rows (`signals/reliability.py:124-137`), gate reports (`docs/modules/reports/m4_gate_*.json`) |
| Called by | `crucible label` (`studio/label.py:145,168-172`), `studio/datasets.py:620-647`, M8 forge (`CP/forge/derive.py:133-145`), the store carry (`connectors/trace_store.py:43-66`), the pack gates (`CP/m4/gate.py`, `TP/m4/gate.py`) |
| Plugins | registry kinds `signal_mapper` / `outcome_labeler` (`core/registry.py:37-38,156-222`); coding registers 3 mappers and the PR labeler (`CP/__init__.py:34-37`) |
| Layer | depends on M0 and M3 (`ARCHITECTURE.md:544`); upstream of M6, M8 and M15 |
| Principles | P12, P16, P27 and P28 honoured (`DESIGN.md:79-112`); rule 3 (no LLM-decided gate) held; rule 4 partly: the report names only the labels file (`signals/gate.py:166-169`) |

**Gate** (`ARCHITECTURE.md:540-543`: on 50 hand-labelled events per pack, type ≥ 0.9, attribution precision ≥ 0.9,
outcome ≥ 0.95; status "in progress", `:41`). Coding dev (2026-10-12): 1.0 / 1.0 / 1.0 on 50 events, 32 attributed and
14 outcomes, lower bounds 0.929 / 0.893 / 0.785; not re-runnable here (`data/cache` absent). Toy test: 1.0 / 1.0 / 1.0 on
84 events, 17 attributed (0.816) and 12 outcomes (0.757), equal to the committed report **[ran: p02_toy_gate_now]**.

| Consumer | Uses M4 output? | Evidence |
|---|---|---|
| M15 rows via `crucible label` | yes: pivots → DPO, `wrong` → no SFT, KTO False | `studio/datasets.py:620-629,786-795,905` |
| `Signal.reliability` in rows | field only; None in production | `studio/label.py:168-172`; `studio/datasets.py:645,886` |
| Pack gate scripts | yes: `estimate` / `apply`, gate scores | `CP/m4/gate.py:13,29-31`; `TP/m4/gate.py:17,40-41` |
| M8 forge | agent-run signals with `meta.gate` only | `CP/forge/derive.py:133-145` |
| PR `check_result` in training | no: the label path ingests PRs from git with no check runs | `packs/coding/pack.yaml:19-23`; `CP/connectors/github_pr.py:161` |
| M6 credit, trainer | no reader of grades, polarity or reliability | `credit/signed.py:29-30`; `train/trainer.py:547-561` |
| `from_signals` (milestones) | no caller | `workflow/milestones/mine.py:224-247` |

So R-1 changes training rows today and R-2 closes a fail-open on the same path; R-3 and R-5 change the gate's
reliability table, R-4 the exit evidence; most later items wait for an M6 or RL consumer.

## 2. Status ledger

99 items (`02-ledger.md`): 1 done, 22 partly, 73 not done, 1 superseded, 2 wrong; NEW-1..NEW-6 were found in this run.

| Item | State | Evidence |
|---|---|---|
| AUD-M08-V11 | done | A closed grammar resolves `sig_*` ids (`accept/contract.py:210-231`); `CP/forge/derive.py:455` still counts any string. |
| IMP-S4 | superseded | Outcome labelers are pack plugins now (`ARCHITECTURE.md:535`; `core/registry.py:37-38`); no `labels.py` is named. |
| HO-M02-1, NEW-2 | wrong | Prompt-fit admission is M10's, not M4's; 51/51 has Wilson bound 0.930, 73 are needed for 0.95 (`signals/gate.py:40`) **[ran: p03_ledger]**. |
| AUD-FIXED | partly | 67 M4 tests pass and the toy gate equals its report **[ran: p02_toy_gate_now]**; R8's one-sided verdicts remain (NEW-1). |
| IMP-S1 | partly | A comment rejection exists only when `acted_on` is False (`CP/signals.py:151-153`) and is contradicted on every good trace (`signals/reliability.py:106-110`): 0/10. |
| IMP-M4-1, IMP-M4-2, SWP-TA-F7 | partly | Revert evidence and actor-class keys exist (`signals/reliability.py:35,93-99`), judgements never grade their author (`signals/attribute.py:30,84`); no CI-failure confirmation, held-out run, actor id, role key or shrinkage. |
| IMP-M4-3, HO-M03-2, AUD-F2 | partly | CI-log infra unscored and in-trace rerun judging (`CP/signals.py:286-287`; `signals/reliability.py:103-105`); PR flaky is a head-only boolean (`CP/signals.py:172-174`). |
| STUDY-REL-F6, STUDY-REL-F7 | partly | No row gives None, but an n = 0 row writes its prior 0.5 (`signals/reliability.py:145`); no test of a revert or hotfix inside the window. |
| AUD-F1, SWP-TB-F1, SWP-TA-F4, AUD-F8, STUDY-REL-F4, SWP-TB-F2, SWP-TB-F3 | partly | Operators and fields exist (`credit/signed.py:15-30`; `credit/compose.py:104`; `studio/datasets.py:886,891-922`) but no M4 producer or reader → R-1, L-9. |
| AUD-F6, STUDY-MAP-F5 | partly | Unattributed signals carry a reason (`signals/attribute.py:76-91`); nothing records why a mapper emitted nothing (`CP/signals.py:145,152-153`). |
| SWP-H-G1, HO-M03-1 | partly | The rejection string and exit check are done (`core/agent_events.py:25-27`; `CP/signals.py:205-240`); takeover, new task and text tool calls are not → L-2. |
| HO-M00a-1 | partly | Head compare-and-set is done (`connectors/trace_store.py:76-89`); `event_key` has no NFC (`core/schema/trace.py:62`) **[ran: p03_ledger]**. |
| NEW-1, DRIFT-3, SWP-C9, SWP-H-F8, IMP-S5, SWP-TB-F8, STUDY-REL-F11, SWP-C8 | not done | Verdicts one-sided, reverts get None not their prior, double counting, no revival or `known_at`, fixed priors (`signals/reliability.py:13,100-121`; `signals/taxonomy.py:24-38`). |
| SWP-TB-D1, AUD-R12-A25, DOC-open-4, AUD-M06-L19 | not done | `crucible label` calls `extract` only (`studio/label.py:168-172`); nothing in `credit/` reads grades; `StepRow.violations` unfilled (`verify/grader.py:71`). |
| SWP-TB-F4, SWP-TA-F2, SWP-TB-F5, SWP-TA-F5 | not done | No SDPO rows, single-step pivot rows, attributable filter or branch-local attribution; no consumer asks. |
| AUD-M5, DOC-open-1, AUD-H4, SWP-C2 | not done | Head-commit check runs only, default filter (`CP/connectors/github_pr.py:174,185`); the corpus `flaky()` list reaches no mapper. |
| SWP-C3 | not done | `FAILED` holds `timed_out` and `startup_failure` (`CP/signals.py:63`) → R-3. |
| AUD-M7, DOC-open-2, AUD-M20, AUD-M22 | not done | A review grades the commit it names (`CP/signals.py:161-167`); DISMISSED gives no signal; one signal per root comment **[ran: p03_ledger]**. |
| AUD-R6, SWP-C11, NEW-6 | not done | Closed-unmerged is always bad (`CP/outcomes.py:41-42`); revert PRs are ordinary; "Fix typo" is fix-like **[ran: p04_fixlike]** → L-3. |
| STUDY-MAP-F9, STUDY-MAP-F10 | not done | Suggestion match is a whole-file substring (`CP/connectors/github_pr.py:333`); `flaky()` groups raw test names (`CP/connectors/ci_logs.py:321`). |
| SWP-TA-F3, NEW-3 | not done | No "polarity undetermined" signal; a pre-PR line escapes last-writer only by a `file:` prefix mismatch (`signals/attribute.py:38-41`) **[ran: p03_ledger]**. |
| AUD-F3, AUD-NONENG, DOC-open-3, IMP-M4-6 | not done | Intent is regex only (`CP/signals.py:73-99`); German, Chinese and Spanish requests give `discussion` **[ran: p03_ledger]** → L-5. |
| SWP-C7, IMP-M4-7, AUD-F10, AUD-FIN, DOC-open-5 | not done | No evidence span, policy_docs mapper or `target` producer; `finance_ap` registers no mapper or labeler (`CP/__init__.py:34-37`). |
| AUD-H3, NEW-5, AUD-H8, SWP-H-F5, SWP-TA-F6 | not done | Session rules cover interrupt and rejection only (`signals/session.py:47-54`); human commands record no writes (`connectors/chat_messages.py:61-62`). |
| AUD-H6, AUD-F7, SWP-H-F4 | not done | Frontier gate failures grade only the final patch (`CP/signals.py:235,244-246`) **[ran: p03_ledger]** → L-4. |
| HO-M01-1, HO-M00b-1 | not done | R-35 was dropped, so World writes stay heuristic (`world/world.py:1172-1173`); the gate rebuilds `set(ts.ids())` per id (`CP/m4/gate.py:31`). |
| IMP-S2, IMP-M4-5 | not done | `TraceSource` has no raw hash or connector version (`core/schema/trace.py:22-27`); `Signal` has no `subject_hash` or `mapper_version`. |
| SWP-C5, AUD-F4 | not done | Labels bind by `event_key` and the labels-file sha256 only (`signals/gate.py:85-88,166-169`) → R-4, L-8. |
| AUD-R23, AUD-R24, DOC-gate-1, SWP-H-F6 | not done | No code reads `CP/m4/test_prs.json`; both label files are "Claude (the builder)"; no kappa in `signals/gate.py`. |
| DOC-gate-2, AUD-R19 | not done | Both reports `confident: false`; 247 coding signals sit on unlabelled events (`signals/gate.py:148`). |
| IMP-M4-4, AUD-F5, SWP-TA-F1, AUD-F9, SWP-C4 | not done | Pivots come only from corrective signals and record ids, no category (`signals/pivots.py:16-36`). |
| STUDY-ATTR-F4, STUDY-ATTR-F8, STUDY-ATTR-F10 | not done | No reasoning-placement rule (`studio/datasets.py:596-597`), no `informed_by` in signals, no weight-0 fallback (`signals/attribute.py:89-91`). |
| DRIFT-1, SWP-H-F7 | not done | `core/agent_events.py:184-185` says M4 turns probes into violations; no code does, and P27 forbids it (`DESIGN.md:110`). |
| DRIFT-2, NEW-4 | not done | `ARCHITECTURE.md:529` omits `check_result`; `docs/modules/M04-signals.md:62-67` lists 5 of 15 open items. |

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | Which earlier step, turn or span does later evidence grade, and how are object identities (paths, records, subagents) matched? | correctness |
| Q2 | How is a derived label keyed so a grown or reordered source gives the same id, and is text normalised before hashing? | identity and reproducibility |
| Q3 | How is a label source's trust estimated (per type, annotator, human or automated), from which evidence, prior and interval, disjoint from the rule? | correctness |
| Q4 | How is "cannot judge" (infra, abort, limit stop, flaky, unknown acted-on, contaminated pass, judge parse failure) kept apart from a negative through to training? | correctness |
| Q5 | How are CI results aggregated over reruns, attempts and commits (rate or boolean, inconclusive class), and which commit does each grade? | correctness |
| Q6 | How are hand labels bound to the exact data they were written on, and how is drift detected and migrated? | evolution and versioning |
| Q7 | How is a labeler evaluated: per-class metrics, coverage, intervals, minimum sizes, held-out splits, agreement? | operability |
| Q8 | How do step signals and their reliability reach training (marks, advantages, pairs with a noise rate, KTO), on the policy's own tokens? | interop |
| Q9 | How do packs contribute mappers and labelers, how is the plugin chosen for a trace, and how is its output validated? | extensibility |
| Q10 | How are outcomes known only later (reverts, hotfixes, reopens) labelled, with the observation window recorded? | correctness |
| Q11 | How are human interventions (interrupt, rejection, takeover, new task, abort, abandoned branch) recorded and mapped to a step? | interop |
| Q12 | How does extraction scale with trace length and signal count? | performance and scale |

## 4. Repo triage

33 reference folders: 2 primary, 27 secondary, 4 not relevant. Every commit was read with `git rev-parse --short HEAD`
in this run; folders that are not git repositories say so.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| nvidia-nemo-gym | primary | Cannot-judge contract, inconclusive CI verdicts, input digests, pivot verifiers | 7ece89985 | Apache-2.0 |
| infra-inclusionai-areal | primary (partial) | PRM scorers: validated output, reject on None, turn-id attribution | aedc21ec | Apache-2.0 |
| harbor-framework-harbor | secondary | M4's ATIF input (DEPEND); converters lose mini-swe-agent interventions; lock digests | a2fafbcf | Apache-2.0 |
| agentenv-framework | secondary | Judge findings with an unchecked step index; human review gate | 041afd7 | Apache-2.0 |
| primeintellect-ai-verifiers | secondary | Attempt identity, content-hash reuse, fail-closed judge verdicts | e22533430 | MIT |
| primeintellect-ai-prime-environments | secondary | Judge incidents per turn; parse failures scored 0.5 (anti-pattern) | cf220ee | Apache-2.0 |
| primeintellect-ai-research-environments | secondary | Pinned labeller benchmark; invalid trials scored 0 | f63f07c8a | Apache-2.0 |
| primeintellect-ai-prime-agent | secondary | Typed aborts, user-run commands, abandoned branches; no Crucible importer | 3e0893260 | MIT |
| servicenow-enterpriseops-gym | secondary | Trace-level verifiers; infra counted as failure (anti-patterns) | 271f2c3 | Apache-2.0 |
| meta-pytorch-openenv | secondary | Native-id turn identity, duplicates refused; source digests in reports | 8d3455ab | BSD-3-Clause |
| adithya-s-k-fineenvs | secondary | Tutorials; infra gives None; order-agreeing pairwise judge | 5e0b46e | Apache-2.0 (BSD-3 file headers) |
| hf-space-fineenvs-multi-harness-rl | secondary | Article; comment anchors by quote with prefix and suffix | not a git repo | CC-BY-4.0 (code unclear: PATTERN) |
| 08-hf-rl-env-datasets | secondary | Dataset dumps: flake as attempt counts, three-state verdicts | not a git repo | per dataset |
| swe-bench-swe-smith | secondary | Gold-rerun flake repair; timed-out eval labelled unresolved | 9b74ac0 | MIT |
| swe-agent-mini-swe-agent | secondary | Typed human interventions; format of the coding agent_runs connector | 04d809ce | MIT |
| huggingface-repo2rlenv | secondary | Labels bound to bundle hashes; two-stage bug-fix classifier | 27d04ee | Apache-2.0 |
| preferencemodel-karotte | secondary | Refuses a suite without calibration rows; judges score failures 0 | 31c693e | MIT |
| karpathy-autoresearch | secondary | Keep/discard ledger per commit; crash logged as best value | 228791f | none (README says MIT): PATTERN |
| xiaomimimo-verl | secondary | adv_signed channels; marked spans never reinforced; infra vocabulary | a2ad9f61 | Apache-2.0 |
| xiaomi-mimo-v2.6-rl-oss | secondary | Release verifiers mask judge outages; rule timeouts score 0 | not a git repo (HF 639865fd) | none (README apache-2.0): PATTERN |
| infra-volcengine-verl | secondary | Per-turn scores carried, never read; reward errors score 0.0 | 2035f58d | Apache-2.0 |
| infra-primeintellect-ai-prime-rl | secondary | Length-checked credit routing onto sampled tokens; versioned trace facts | 1e5318108 | Apache-2.0 |
| infra-nvidia-nemo-rl | secondary | One infra/data failure classifier; timeouts never scored | 925f70d55 | Apache-2.0 |
| infra-openrlhf-openrlhf | secondary | Step rewards summed to the last token; one cDPO epsilon per run | dc2a7ad | Apache-2.0 |
| infra-huggingface-trl | secondary | Robust-DPO epsilon per run; default DPO ignores label smoothing | d9b9f39b | Apache-2.0 |
| infra-thudm-slime | secondary | Segment rollout ids; removed samples stay in group stats | 05e3b017 | Apache-2.0 |
| infra-alibaba-roll | secondary | GiGPO state groups with no evidence gate; timeouts scored 0 | 0f37fe13 | Apache-2.0 |
| infra-novasky-ai-skyrl | secondary | Masks infra and agent faults alike; masked rewards stay in baselines | a1e34dbf | Apache-2.0 |
| infra-rllm-org-rllm | secondary | ATIF continuation chains followed; positional trace matching | 4e828c81 | Apache-2.0 |
| hf-space-fineenvs-simulation-rl-environments | not relevant | PortSimEnv article; no evidence labels or signal mechanism | not a git repo | CC-BY-4.0 |
| hf-space-rl-environments-guide | not relevant | Survey article of six frameworks; conceptual prose only | not a git repo | CC-BY-4.0 |
| veds12-git-swarm | not relevant | Project page only; no grading, label or credit code | 9f66af0 | none |
| infra-moonshotai-checkpoint-engine | not relevant | Weight-update middleware; no reward, label or signal code | 84631a2 | MIT |

## 5. Findings per repo

### nvidia-nemo-gym
- Cannot judge: `mask_sample` stays apart from `failure_kind` (`nemo_gym/base_resources_server.py:107-148`), yet an
  unparseable verdict is wrong (`nemo_gym/judge.py:25-27`).
- Reruns are bucketed supported/flaky/broken/inconclusive (`resources_servers/swemer_v2/aggregate_golden_patch.py:51-64`);
  inputs are pinned by `expected_sha256` (raise on mismatch) (`nemo_gym/atif_reverification.py:101-108,225-229`).
- swe_pivot grades by action match (`resources_servers/swe_pivot/app.py:295-449`, against P12), its path suffix crosses
  packages **[ran: p_nemo_paths]**, and subagents are refused (`nemo_gym/atif_reverification.py:392-395`).
- Since `3ef478d`: 139 commits; here only a verdict-marker whitespace fix (`judge.py`) and docstrings. Corrections:
  invalid verdicts are scored rows, not "never scored"; M03-M05's raw-hash lines are now `atif_reverification.py:134-150`.
- Verdicts: better: the inconclusive class (→ R-3), digest pins (→ R-4, L-8); of interest: the judge-failure split (→ L-5);
  Crucible better: path identity, subagents, reliability, intervals; reject: action-match step rewards (P12).

### infra-inclusionai-areal
- PRM output is validated before commit (`areal/reward/prm/runner.py:464-608`); None or an exception rejects the trajectory
  by default (`areal/api/cli_args.py:2701-2711`) (→ R-2); rewards name their turn by interaction id
  (`areal/experimental/openai/proxy/proxy_rollout_server.py:701-739`).
- Weak: a reward timeout or an unrewarded last turn gives 0 (`areal/api/reward_api.py:154-161`); late rewards are absorbed
  (`areal/v2/inference_service/data_proxy/session.py:207-218`); GVPO does not conserve mass
  (`areal/trainer/ppo/actor.py:108-139`).
- Since `298412a`: 3 commits; `aedc21ec` makes PPO loss aggregation configurable (`actor.py`, `cli_args.py`), no PRM
  change. Corrections (`docs/audit/2026-10-12/M04-study-sweep.md:315,327`): "never 0" is only the default `error_policy`;
  orphan removal is opt-in (`areal/experimental/openai/cache.py:568`).
- Verdicts: better: the reject default (→ R-2), the mark commit contract (→ L-9); Crucible better: never 0, the outcome
  window, reliability, mass conservation, retry collapse on evidence.

### Secondary patterns
- **harbor-framework-harbor**: the mini-swe-agent converter folds later user turns into the previous step and drops `extra`
  (`src/harbor/agents/installed/mini_swe_agent.py:337-354`): 1 of 5 interventions survive
  **[ran: h1_mini_swe_interventions]** (→ L-2); lock digests (→ R-4). Correction: -1 is DeepSWE's crash sentinel.
- **agentenv-framework**: a finding's `step` is asked 0-based, logged 1-based, copied unchecked
  (`src/agent_env/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:192-194,1124`)
  **[ran: p_ae1_agentenv_judge]** (→ L-4). Correction: the training block is prompt text (`:175-177`).
- **primeintellect-ai-verifiers**: attempts keyed by request id, a duplicate resolved identity raises
  (`verifiers/v1/trace.py:565-592`) (→ R-5, L-7). Since `ae819de`: a decision judge and a resume planner keyed by task hash
  (`verifiers/v1/utils/eval.py:10-36`). Correction: the refusal check moved to `verifiers/v1/judge.py:214-221`.
- **primeintellect-ai-prime-environments**: a judge parse failure scores 0.5
  (`environments/reward_bench/reward_bench/reward_bench.py:12-19`). Correction: a misspelled key scores a good spiral_bench
  judgment 50.0, not 86.5 (`environments/spiral_bench/spiral_bench/utils.py:217-251`) **[ran: pp01_prime_sweep]**.
- **primeintellect-ai-research-environments**: a labeller benchmark pinned to revision and detector version
  (`environments/judges/trace_cheating_recall_500/trace_cheating_recall_500/taskset.py:11-14,131-136`). Correction:
  senior_swe_bench scores an invalid trial 0 (`environments/swe/senior_swe_bench/senior_swe_bench/taskset.py:122-138`).
- **primeintellect-ai-prime-agent**: typed aborts (`crates/pa-agent/src/agent_loop/abort.rs:71-97`), user-run `bashExecution`
  and `branch_summary` (`crates/pa-types/src/session.rs:69-87,211-223`); no Crucible importer (→ L-6).
- **servicenow-enterpriseops-gym**: anti-patterns: infra counted as failure (`benchmark/verifier.py:75-79`), a vacuous
  `all()` (`benchmark/executor.py:371,442-450`). Corrections: `tool_execution` is routed (`verifier.py:71-74`); errored files
  are flagged but averaged (`compute_score.py:35-42`); subtask success means "no exception".
- **meta-pytorch-openenv**: turn identity by `node_id` and index, duplicates refused
  (`src/openenv/core/harness/capture/graph.py:158-189`) (→ L-1); source digests (`src/openenv/validation/runner.py:36-57`) (→
  R-4). Corrections: the 0.5 draw is a docstring; F10's fallback is test-only.
- **adithya-s-k-fineenvs**: infra gives None (`05-multi-harness-rl/train/multi_harness.py:77-85`); a pairwise verdict counts
  only when both orders agree (`02-watercolour/envs/watercolour/core/pairwise_judge.py:436-459`). Corrections: the monotone
  rubric check is a docstring number (`core/quality.py:22-32`); some files are BSD-3-Clause.
- **hf-space-fineenvs-multi-harness-rl**: review comments anchored by quote plus 80-character prefix and suffix
  (`review/server.py:228-238`).
- **08-hf-rl-env-datasets**: flake as an attempt count (`PrimeIntellect__R2E-Gym-Subset-Verified/README.md:27-38`) and
  three-state verdicts (`PrimeIntellect__Scale-SWE-Verified/README.md:28-38`) support R-3. Correction: replaced on 2026-10-14
  (41 folders; NatureBench and procedural-pile gone).
- **swe-bench-swe-smith**: flake = fails one gold rerun, removed from F2P/P2P (`swesmith/harness/repair.py:138-215`);
  `traj_id` hashes the directory path (`swesmith/train/traj_mgr/collect_trajs.py:73`). Correction: timed-out runs are dropped
  in validation only; collection labels them unresolved (`collect_trajs.py:57-61`).
- **swe-agent-mini-swe-agent**: typed `extra.interrupt_type` and human `extra.actions`
  (`src/minisweagent/agents/interactive.py:40-41,58-71,144-182`); Crucible keeps 1 signal via agent_runs, 2 via
  chat_messages, no human writes **[ran: sws_mini_interactive]** (→ L-2). Correction: `/u` at exit (`:154-155`) is missed.
- **huggingface-repo2rlenv**: labels bound to a bundle hash (`src/repo2rlenv/quality/labels.py:29-139`); its two-stage fix
  filter trades errors (`src/repo2rlenv/pipelines/commit_runtime.py:89-104`) **[ran: v_sws4_tradeoff]** (→ L-3). Correction:
  no held-out split or kappa.
- **preferencemodel-karotte**: refuses a suite missing a calibration row
  (`templates/default/src/environment/suite.py:40-58`); a grader crash scores 0
  (`src/karotte/judges/executable_judge.py:79-84`). Correction: `_composite_judge.py:104-194` is And/Or composition.
- **karpathy-autoresearch**: credit as the delta against the last kept commit (`analysis.ipynb` cell 9); a crash is logged as
  val_bpb 0.000000, the best value (`program.md:75-76`). No consumer: `target` has no producer (`signals/taxonomy.py:37`).
- **xiaomimimo-verl**: adv_signed channels merged by max kappa (`verl/trainer/ppo/v1/trainer_base.py:2104-2134`); a marked
  span of a positive sample is never reinforced (`verl/trainer/ppo/signed_rebalance.py:14-22,142`) (→ R-1, L-9). Correction:
  `apply_tool_penalty` raises (4 arguments, 3 parameters) **[ran: s04_arvo_call_signature]**.
- **xiaomi-mimo-v2.6-rl-oss**: no usable judge vote raises `_JudgeUnavailable` and masks
  (`general/envs/*/verify.py:113-114,311-340`) (→ L-5); rule timeouts score 0 (`:34-48`). Correction: 925 general envs
  locally, not 267; no F2P/P2P or flaky lists **[ran: s02_mimo_instance_keys]**.
- **infra-volcengine-verl**: `turn_scores` and `tool_rewards` are carried, never read; a reward timeout or exception scores
  0.0 (`verl/experimental/reward_loop/reward_manager/limited.py:451-467`).
- **infra-primeintellect-ai-prime-rl**: length-checked credit onto sampled tokens of trainable branches, shared forks
  credited once (`src/prime_rl/orchestrator/algo/routing.py:11-40`; `src/prime_rl/orchestrator/trajectories.py:105-128`) (M6
  handoff).
- **infra-nvidia-nemo-rl**: one public infra/data classifier; timeouts are infra, re-dispatched, never scored
  (`nemo_rl/experience/failures.py:96-97,146,192-202`) (→ R-3). Correction: per-turn reward summing moved to
  `nemo_rl/experience/rollouts.py:996,1083-1099,1429-1437`.
- **infra-openrlhf-openrlhf**: step rewards summed onto the last action token (`openrlhf/utils/agent.py:122`); one cDPO
  `label_smoothing` per run (`openrlhf/models/loss.py:329-358`).
- **infra-huggingface-trl**: robust-DPO epsilon is one float per run in [0, 0.5) (`trl/trainer/dpo_config.py:284-292`).
  Corrections: the default sigmoid loss ignores `label_smoothing` (`trl/trainer/dpo_trainer.py:1366-1367`; `:1426-1443` is
  aot); GRPO nansums partly-None rows as 0 (`trl/trainer/grpo_trainer.py:2692`).
- **infra-thudm-slime**: `rollout_id` groups segments and refuses mixed outcomes (`slime/utils/types.py:111-119`);
  `remove_sample` stays in GRPO group stats (`slime/data/batch_builder.py:210-257`). Corrections: the coding example disables
  rewrite merge (`examples/coding_agent_rl/generate.py:82`); five grader-infra paths give 0.
- **infra-alibaba-roll**: GiGPO groups steps by state hash, no evidence gate (`roll/pipeline/agentic/utils.py:155-211`);
  TEST_TIMEOUT is a failure (`roll/pipeline/agentic/env/rock/sanbox_manager.py:1334-1339`). Corrections: M15's evidence rule
  partly supersedes F5; reset placeholders score 0.
- **infra-novasky-ai-skyrl**: infra errors and agent faults are masked alike and keep their rewards in the LOO baseline
  (`skyrl-agent/skyrl_agent/agents/base.py:320-329,485-488`). Correction: the earlier row missed the copy of another
  trajectory's result into an empty one (`base.py:281-319`).
- **infra-rllm-org-rllm**: the ATIF bridge follows `continued_trajectory_ref`
  (`rllm/integrations/harbor/atif_trajectory_bridge.py:43-77`); Crucible does not **[ran: tc01_atif_continuation_ids]**.
  Correction: precomputed advantages are fail-soft (`rllm/trainer/algorithms/advantage.py:147-163`).

## 6. Gap matrix

Legend: ● built and tested, ◐ built with a stated weakness, ○ absent.

| Q | Crucible | nvidia-nemo-gym | infra-inclusionai-areal |
|---|---|---|---|
| Q1 | ◐ last writer, blame, session; human writes missing | ◐ action match vs expert; suffix paths mismatch | ◐ explicit interaction id or last turn |
| Q2 | ◐ content event keys; no NFC; session ids collide | ◐ content hash; positional task-id fallback | ◐ provider ids; sha1 digest, no NFC |
| Q3 | ◐ Beta per key; circular rejections; never applied | ○ none (swap re-judge only) | ○ none |
| Q4 | ◐ unscored kept; PR timeout -1; failed extract trains | ◐ mask contract; unparseable verdict scored wrong | ◐ PRM reject default; timeouts give 0.0 |
| Q5 | ◐ in-trace flaky boolean; head-only fetch | ● supported, flaky, broken, inconclusive classes | ○ none |
| Q6 | ◐ event keys plus labels-file sha only | ● expected_sha256 pins, raise on mismatch | ○ none |
| Q7 | ◐ Wilson, minimums; one labeller, no held-out | ◐ coverage reported; t-interval collapses at extremes | ○ metrics without intervals or agreement |
| Q8 | ◐ pivots to DPO; non-corrective negatives trained | ○ reward components only; token-capture checks | ● token rewards shaped, finite, asserted |
| Q9 | ● registry, family prefix, validated output | ◐ folder discovery, pydantic responses | ◐ dotted-path scorers with validate hook |
| Q10 | ● 30-day window recorded; open is unknown | ○ none | ○ late rewards absorbed after timeout |
| Q11 | ◐ interrupt and rejection only; mini path lossy | ○ none | ○ online user set_reward only |
| Q12 | ● 11,000 events extracted in 0.10 s | ● capped SequenceMatcher, linear passes | ◐ per-insert sort; quadratic topological order |

## 7. Recommendations, ranked

27 candidates from three verify batches (`verify-1.md`..`verify-3.md`) became 14 items; each R-item's Status names its
candidates. Later items: L-1 openenv-hf:OE-1 + harbor:C3; L-2 harbor:C2 + swe-safety:SWS-3; L-3 SWS-4; L-4 agentenv:AE-C2;
L-5 prime-envs:PI-3 + nemo-gym:NG-3 + mimo-verl:MV-3; L-6 PI-4; L-7 PI-1 (mapper side); L-8 PI-2 + SWS-2 + the pin halves
of NG-1, OE-2, harbor:C1, agentenv:AE-C1; L-9 MV-4 + trainers-b:T-B3. Two verifier tiers disagreed and the code decided:
NG-2 (verify-1: later) joins R-3 at should, because the gate's reliability fit reads PR polarity (probe v1b;
`signals/reliability.py:100-102`); the C1/AE-C1 label pin (verify-2: should) stays later with verify-3: no label set is
pinned, and the pin hashes salted actors (`connectors/redact.py:97-109`; 8 of 10 labelled events change with the salt, g4).
Order: R-1 and R-2 touch `crucible label`; R-3 and R-5 share `CP/signals.py`; both pairs and R-4 precede DOC-gate-1's run.

| R | Title | Tier | Kind | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| R-1 | Negatively graded policy turns are never positive targets | must | defect | M | L | M |
| R-2 | A trace whose extraction raised yields no training rows | should | defect | M | L | L |
| R-3 | PR check timeout or startup failure is polarity 0, not -1 | should | defect | M | L | L |
| R-4 | Gate reports record the traces and code they scored | should | production gap | M | L | L |
| R-5 | One CI-check identity in `verdict`; a pass never contradicts a pass | should | defect | L | L | L |

### R-1. A policy turn graded wrong by a non-contradicted negative step signal is never a positive target
- **Kind** defect · **Tier** must · **Status** correction of the M15 V8 fix (`775f30f1`), which excludes corrective-graded
  turns only; mimo-verl:MV-1 · **Origin** present at `b65754b2`, when SFT excluded no graded turn
- **Problem.** `from_trace` takes `Labelled.wrong` only from `pivots()` (`studio/datasets.py:620-629`): correction,
  suggestion_accepted, revert (`signals/pivots.py:16,27`). A good trace's turn graded by a rejection, escalation, violation,
  rework, interrupt or failing check stays an SFT row (`datasets.py:786-795`), KTO True (`:905`) and a possible DPO
  chosen turn (`:836`); `verdict` gives these signals None, so nothing contradicts them (`signals/reliability.py:105-115`).
  Today `crucible label --pack _toy` trains T1005 seq 2 ("The carrier shows it at your building's parcel room.",
  answered "There is no parcel room in my building") and T1007 seq 2 ("Please regenerate your API key and retry.",
  rejected, then escalated): 2 of 10 SFT and 2 of 10 KTO-desirable rows; 23 such signals, 0 contradicted.
- **Reference.** `xiaomimimo-verl/verl/trainer/ppo/signed_rebalance.py:14-22,142` zeroes a marked span of a positive
  sample, every channel, fix known or not; max-kappa merge: `xiaomimimo-verl/verl/trainer/ppo/v1/trainer_base.py:2104-2134`.
- **Recommendation.** Invariant: such a turn is never a positive target. Contract: `signals.pivots.graded_wrong(trace)
  -> dict[int, list[str]]` (graded seq → signal ids) beside `pivots`: signals with step grades, polarity < 0, non-empty
  grades, no `all_commits` attribution and `verdict(trace, s) is not False` (pivots' filters without the corrective
  set). `pivots()` still drives pairing. `from_trace` fills `wrong` from it through `turn_of_seq`, so non-policy events
  (the toy `sla_breach`) drop out; summary.json counts `graded_wrong`. SFT skips the turn, KTO labels it False, DPO
  best-vs-worst never takes it as chosen. No schema change; reliability values are not read (None on this path).
- **Reasoning.** A type list in M15 would drift from M4's verdicts; a reliability threshold excludes nothing while
  reliability is None (SWP-TB-D1); skipping the KTO row gives `wrong` two meanings (see SWP-TB-F3); token marks need RL
  signals (L-9). Keeps V8 and P11. Trade-off: fewer positives; correct late replies graded by `sla_breach` drop too.
- **Established by.** Code read (`studio/datasets.py:537-651,760-906`; `signals/pivots.py:16-36`; `TP/signals.py:107-111`;
  `signals/session.py:47-54`; `git show b65754b2` of `datasets.py`); **[ran: v3_mv1_label_trains_rejected]** sft 10, KTO
  10/2; **[ran: v3_mv1_counterfactual]** sft 8, KTO 8/4, `sft_graded_wrong` 2, dpo 0.
- **Impact / Effort / Risk.** M: 20% of toy SFT rows, and any pack with session rejections on policy turns leaks alike ·
  L: one helper, one call site, one counter, one test · M: rows change (toy SFT 10 → 8, KTO 10/2 → 8/4).
- **Mode.** PATTERN (Apache-2.0, nothing copied; the operator is already PORTed as `credit/signed.py`).
- **Acceptance.** `crucible label` on the toy pack: neither reply is an SFT or KTO-desirable completion, both are
  KTO-undesirable, `graded_wrong` ≥ 2, sft == 8, KTO desirable == 8; the V8 test passes; a `check_result` -1 that the
  same check later passed marks nothing.

### R-2. A trace whose extraction raised yields no training rows
- **Kind** defect · **Tier** should · **Status** new, narrows AUD-FIXED A15; trainers-b:T-B1 · **Origin** introduced by
  `b65754b2` (A15; blame of `studio/label.py:166-174`): before it a failing `extract` aborted the run
- **Problem.** `crucible label` catches any `extract` error, records it and keeps the unextracted trace
  (`studio/label.py:167-174`): the pack credit hook gives it an outcome (`:175-179`), `from_trace` makes items
  (`:191-193`), and with no signals `wrong` is empty (`studio/datasets.py:620-629`), so a corrected step becomes an SFT
  target (`:778,786-788`) and KTO-desirable (`:905`); only `summary["extract_failed"]` shows it. Triggers: a mapper bug,
  a carried signal that no longer validates (`signals/extract.py:76-93,130`), or L-1's duplicate session ids. A good
  trace with a correction of "rm -rf /tmp/*" gives `wrong=[1]`, 1 SFT row and KTO False when extracted; with extract
  raised, `wrong=[]`, 2 SFT rows (one is "rm -rf /tmp/*") and KTO True. Committed toy and coding data fail nothing
  today (`extract_failed=[]`); the A15 test checks only the error text (`tests/unit/test_m4_audit.py:166-171`).
- **Reference.** A PRM scorer that raises or returns None rejects the trajectory by default (`error_policy="reject"`,
  `infra-inclusionai-areal/areal/api/cli_args.py:2701-2711`; `infra-inclusionai-areal/areal/reward/prm/runner.py:426-445`);
  `keep_original` is a logged opt-in that the v2 proxy refuses
  (`infra-inclusionai-areal/areal/v2/inference_service/data_proxy/config.py:37-39`).
- **Recommendation.** Invariant: a trace whose `extract()` raised has no M4 verdict and yields no SFT, KTO or DPO row.
  Contract: `label` lists it in `summary["extract_failed"]`, counts it in `skipped["extract_failed"]` and never passes
  it to `from_trace`; it may still be written to `traces.jsonl` and `rewards.jsonl` as evidence. No flag; the other
  traces' rows are unchanged. The contract is M4's; the edit lands in M15's `studio/label.py`.
- **Reasoning.** Keeps A15's goal (one bad trace does not stop a run) and fails closed for that trace, as AReaL does.
  Aborting the run is the defect A15 fixed; keeping only negative rows adds a mode for no consumer. Trade-off: a mapper
  bug costs that trace's rows until fixed. Should, not must: no committed mapper raises today.
- **Established by.** Code read (`studio/label.py:166-200`; `studio/datasets.py:620-629,770-792,895-910`;
  `signals/extract.py:76-134`); AReaL at the lines cited; **[ran: v2_probes]** v3 (numbers above);
  **[ran: t_b1_extract_failure_still_trains]** the failed toy T1005 keeps its 2 SFT and 2 KTO rows.
- **Impact / Effort / Risk.** M: a corrected step stops becoming a positive target · L: about five lines and one test ·
  L: only traces whose extraction already failed change.
- **Mode.** PATTERN (Apache-2.0, idea only).
- **Acceptance.** A mapper raises for one of two traces, the one with a correction: 0 SFT, KTO and DPO rows from it,
  `skipped["extract_failed"] == 1`, and the other trace's rows equal those of a run without the failure.

### R-3. A PR check run that timed out or never started is "cannot judge" (polarity 0), not -1
- **Kind** defect · **Tier** should · **Status** carry-over (SWP-C3, HO-M03-2, part of AUD-F2); trainers-a:TRA-C1,
  trainers-b:T-B2, trainers-c:TC-1, nemo-gym:NG-2 · **Origin** present at `b65754b2` (`CP/signals.py:62` there)
- **Problem.** `FAILED = {failure, timed_out, startup_failure, error}` (`CP/signals.py:63`): each such PR check run grades
  its commit -1 (`:168-176`) and counts toward `flaky` (`:173`). The same pack masks GitHub's runner time-limit line as
  infra in CI logs (`CP/connectors/ci_logs.py:15-17,49`), where nothing is emitted (`CP/signals.py:286-287`); P16
  (`DESIGN.md:93`) masks infra, never scores it. `timed_out`, `startup_failure` and `failure` on one commit each give -1
  (v1a), the CI-log form gives infra and no signal (v1c), and after a later hotfix `_trouble` confirms the timeout's -1
  in the gate's `check_result/automated` row (v1b; `signals/reliability.py:100-102`). Hurt: the M4 gate's PR traces
  (`CP/m4/build_traces.py:46-55`) and reliability table; `crucible label --pack coding` ingests PRs from git with no check
  runs (`packs/coding/pack.yaml:19-23`; `CP/connectors/github_pr.py:161`). Corpus frequency is not established.
- **Reference.** A timeout is infra (`infra-nvidia-nemo-rl/nemo_rl/experience/failures.py:96-97,146`), re-dispatched and
  never scored (`infra-nvidia-nemo-rl/nemo_rl/experience/rollout_manager.py:2032-2060`); one public classifier
  (`infra-nvidia-nemo-rl/nemo_rl/experience/failures.py:192-202`) keeps two copies from drifting; a verdict-less pass is
  `inconclusive` (`nvidia-nemo-gym/resources_servers/swemer_v2/aggregate_golden_patch.py:15-25,51-64`).
- **Recommendation.** Invariant: a CI conclusion that does not say whether the code is right never grades a commit, and
  the PR and CI-log mappers agree on a runner timeout. Contract: `INCONCLUSIVE = {timed_out, startup_failure}` leaves
  `FAILED`; such a run still emits `check_result` (the labelled type) with polarity 0, `meta.inconclusive=<conclusion>`
  and `meta.reason` (P16): the flaky shape, which `judgeable` already keeps out of reliability and pivots
  (`signals/reliability.py:74-75`). Flaky compares only PASSED with FAILED. `failure` and `error` stay -1.
- **Reasoning.** One pack, one rule for one GitHub event, and P16. Polarity 0 beats dropping the signal (T-B2): every CI
  run is labelled `check_result`, so a drop is a type miss. `startup_failure` moves on the same reasoning (no job ran),
  its GitHub cause not established in code. Trade-off: a hang the PR caused is no longer -1, as on the CI-log path.
- **Established by.** Code read (`CP/signals.py:63,117-123,168-176,286-287`; `CP/connectors/ci_logs.py:45-53,286-302`;
  `signals/reliability.py:74-75,100-105`); **[ran: v2_probes]** v1a-v1c; **[ran: t_a1_ci_timeout]**.
- **Impact / Effort / Risk.** M: a wrong sign leaves stored PR traces and the gate's reliability row · L: two strings,
  one meta branch, one test · L: nothing outside the gate reads PR check polarity.
- **Mode.** PATTERN (Apache-2.0, nothing copied).
- **Acceptance.** One PR trace with `timed_out`, `startup_failure`, `failure` and `success` on one commit gives [0, 0, -1,
  +1]; the timeout's verdict is None after a hotfix; a CI-log timeout still gives infra and no signal; 67 M4 tests pass.

### R-4. M4 gate reports record the traces and the code they scored
- **Kind** production gap · **Tier** should · **Status** carry-over (report side of AUD-F4, SWP-C5; IMP-S2); SWS-1, MV-2,
  NG-1, OE-2, report halves of C1 and AE-C1 · **Origin** present at `b65754b2` (`signals/gate.py:168` hashed only labels)
- **Problem.** The exit reports (`docs/modules/reports/m4_gate_*.json`; `ARCHITECTURE.md:41`) name only `labels_sha256`
  (`signals/gate.py:166-169`; `CP/m4/gate.py:33`; `TP/m4/gate.py:43`). The coding gate rebuilds by default through a
  read-through cache of the live API that is absent here (`CP/m4/gate.py:21-24`; `CP/github_reviews.py:44-69`), with only
  the outcome window pinned (`CP/m4/build_traces.py:16`), so its next run (DOC-gate-1) scores whatever changed after
  2026-10-09 (open PR #6293 holds 8 of 50 labels); an edited comment keeps its row id (`CP/connectors/github_pr.py:301`).
  One edited toy reply moves type accuracy 1.0 → 0.9881 with `label_errors` 0 and the same labels hash (g2); an edited
  or re-pushed labelled comment scores type 0.0, `label_errors` 0 (v4). Rule 4 (`ARCHITECTURE.md:835`) is not met.
- **Reference.** A raise on an input digest mismatch (`nvidia-nemo-gym/nemo_gym/atif_reverification.py:101-108,225-229`);
  a source digest before and after validation (`meta-pytorch-openenv/src/openenv/validation/runner.py:36-57,401-425`);
  a report bound to its bundle hash (`huggingface-repo2rlenv/src/repo2rlenv/quality/report.py:64-74`).
- **Recommendation.** Invariant: a gate report names the trace content and code its numbers came from. Contract: both
  gate scripts call `signals.gate.inputs(traces)` → `{"traces": {trace_id: digest}, "traces_sha256": combined}`, with
  digest = `content_hash()` of the pre-`extract` trace, `signals` cleared, and combined = sha256 of sorted `id\0digest\n`
  lines. Reports gain `inputs`, `code` = {git_sha, git_dirty, source_sha256} (`core.build.build_info`) plus the pack hash
  (`studio.runs.pack_hash`), and `redaction` = {version, salt_id = `redact.pseudonym("salt", "id", salt)`} for redacted
  inputs, so other tenants' digests read as not comparable. Metrics, thresholds and labels stay; nothing is refused.
- **Reasoning.** The smallest change that meets rule 4; a moved number is explained by diffing two reports' `inputs` and
  `code`. Refusing needs pins written with the labels (L-8); the TraceStore head key (`connectors/trace_store.py:91-93`)
  is coding-only and includes carried signals; committing PR data breaks rule 5. Trade-off: digests are tenant-scoped.
- **Established by.** Code read (`signals/gate.py:85-169`; `CP/m4/build_traces.py:15-55`; `connectors/redact.py:97-109`;
  `core/build.py:31-38`); `git show 74ba5f04` (agent runs only); **[ran: v3_gate_inputs_drift]** g1-g4 (digests stable
  24/24, only T1013 moves, salts differ); **[ran: v2_probes]** v4.
- **Impact / Effort / Risk.** M: a moved gate number traces to data or code · L: one helper, ~6 lines per script, one
  test · L: report fields only.
- **Mode.** PATTERN (Apache-2.0 and BSD-3-Clause ideas; reuses `content_hash`, `build_info`, `pack_hash`).
- **Acceptance.** Two toy runs give equal `inputs.traces_sha256`; editing one ticket event moves that trace's and the
  combined digest; reports carry `code.git_sha`, the pack hash and (coding, fake TraceStore) `redaction.salt_id`.

### R-5. One CI-check identity in `verdict`; a later pass never contradicts a pass
**Kind** defect · **Tier** should · **Status** new (PI-1) · **Origin** introduced by `b65754b2` (R9) · **Mode** PATTERN (MIT)
· **Impact / Effort / Risk** L/L/L. **Problem.** `verdict` matches a check by name, the mapper by (app, name, sha)
(`CP/signals.py:117-123`), and any later same-named run contradicts a pass (`signals/reliability.py:115-120`). **Reference.**
Attempt ids (`primeintellect-ai-verifiers/verifiers/v1/trace.py:565-589`). **Recommendation.** Match (`meta.app`, check); only
a later negative contradicts a pass. **Reasoning.** Restores the pre-R9 rule for passes. **Established by.** **[ran: v2_probes]**
v2b pass-then-pass verdicts [False, True]; v2c. **Acceptance.** Pass then pass gives [True, True]; Azure -1, Actions +1: None.

### Later (not recommended now)

| L | Title | Why not now | What would make it due |
|---|---|---|---|
| L-1 | Distinct ids for repeated identical session events: an occurrence ordinal in `SessionSignalMapper` (`signals/session.py:44`) | No extract caller sees session traces: 11 agent runs and 4 CI logs pass, 0 session signals **[ran: v_oe1_current_paths]** | A pack, `crucible signals`, forge or M5 extracts `claude_code_sessions`, capture or `chat_messages` traces |
| L-2 | mini-swe-agent interventions and human commands survive the Harbor import (tags from raw messages); fix the `signals/session.py:1-4` docstring now | All 11 recorded runs are `yolo`; nothing is lost today **[ran: h1_mini_swe_interventions]** | The first confirm- or human-mode log, or a consumer of the DESIGN §5.2 rows (`DESIGN.md:399-400`) |
| L-3 | Measure the hotfix rule on the PCL corpus; exclude cosmetic subjects if they flip labels; fix the `CP/connectors/github_pr.py:53` docstring | No wrong label seen (14/14 agree), cache absent, PR traces give no training rows **[ran: p04_fixlike]** | The cache returns (a confirmed flip makes it must), held-out labels contradict a hotfix, or PR outcomes feed training |
| L-4 | Automated judge findings as step-graded `check_result`, bound to event keys, not positional steps (`CP/signals.py:244-246`) | Nothing reads agent-run step grades | An M6 operator consumes M4 step marks, or agent-run pivots are wanted |
| L-5 | The planned intent judge abstains with a reason (no signal) on a missing, unparseable or out-of-set verdict | No judge exists | The judge behind the intent rules (DOC-open-3) is built |
| L-6 | Session rules for prime-agent aborts, `bashExecution` and `branch_summary` | No prime-agent importer or harness | One is planned in M3 or M10 |
| L-7 | Check-suite identity: two workflows' same-named jobs are two checks, not one flaky check (`CP/signals.py:117-118`) | The result is an abstention, not a wrong sign (v2a) | The `filter=all` rerun fetch (DOC-open-1) is built, or a corpus shows a lost -1 |
| L-8 | Hand labels pin the content they were written on (salt-free); a mismatch is a `label_drift` error | No label set is pinned, the 2026-10-09 data is gone, actor digests differ across salts (g4) | The held-out or second-labeller labels are written (DOC-gate-1), or dev labels are re-scored on a re-fetch |
| L-9 | M4 negative step signals as `adv_signed` marks on RL rollouts (`credit/signed.py:29-30`) | No rollout carries M4 signals; the trainer marks format errors only (`train/trainer.py:547-561`) | M11 rollouts go through `signals.extract`, or a World emits step signals |

### Rejected in verification

No candidate group was rejected. Proposed designs replaced in verification (the problem kept):

| Ids | Reason |
|---|---|
| swe-safety:SWS-4 (adopt Repo2RLEnv's classifier) | Trades errors (4 false and 1 missed against 2 and 3) and keeps "Fix typo" fix-like **[ran: v_sws4_tradeoff]** |
| openenv-hf:OE-1, harbor:C3 (unique `event_key`; call-id extra) | The key change moves M0 identity for no producer; the call-id extra still fails untimed rejections **[ran: v_oe1_ordinal_extra]** |
| trainers-b:T-B2 (no signal for a timeout) | Every CI run is labelled `check_result`, so dropping the signal makes a type miss |
| harbor:C1, agentenv:AE-C1 (event-key digest; full-event digest) | The first misses edited comments (v4); the second flags connector fixes as drift |

## 8. What we already do better (keep)

- **"Cannot judge" never becomes a number**, limit stops included (`CP/signals.py:189-214,286-295`; `train/trainer.py:398`);
  19 references score it 0, 0.5, 1.0 or a fail somewhere (`harbor-framework-harbor/src/harbor/metrics/base.py:23-35`; §5).
- **Evidence-based attribution, stable ids**: writes, blame, session rules, path identity, subagent splice, content keys
  (`signals/attribute.py:33-93`; `core/schema/trace.py:52-62`) vs `nvidia-nemo-gym/resources_servers/swe_pivot/app.py:327-332`.
- **Reliability from later evidence; a gate with Wilson intervals and minimums** (`signals/reliability.py:35-147`;
  `signals/gate.py:36-82`): no reference estimates trust; t-intervals collapse (`nvidia-nemo-gym/nemo_gym/reward_profile.py:251-265`),
  and DPO noise is a user-set constant (`infra-huggingface-trl/trl/trainer/dpo_config.py:284-292`).
- **Validated plugin output, no vacuous pass** (`signals/extract.py:76-107`; `CP/signals.py:249-261`), against `all()` over
  nothing (`servicenow-enterpriseops-gym/benchmark/executor.py:371,442-450`) and trusted script JSON in Harbor.
- **CI flakes, outcome windows, interventions** (`CP/connectors/ci_logs.py:313-321`; `CP/outcomes.py:20-60`;
  `signals/session.py:47-54`): no reference keeps a window or maps interventions to steps; the last line wins in
  `primeintellect-ai-verifiers/verifiers/v1/utils/score.py:153-177`.
- **Studio and credit**: DPO cuts need evidence, KTO conflicts drop, signed marks conserve mass, retries collapse only on
  evidence (`studio/datasets.py:805-848,910-921`; `credit/signed.py:70-107`; `connectors/capture_proxy.py:653-690`), vs
  `infra-alibaba-roll/roll/pipeline/agentic/utils.py:155-211` and `infra-inclusionai-areal/areal/trainer/ppo/actor.py:108-139`.

## 9. Open questions

| Question | Tried | Missing (who can supply it) |
|---|---|---|
| Do Claude Code's refused parallel calls share one refusal sentence (L-1's tier)? | 212 local sessions: 9 refusals, none in a multi-call turn **[ran: h3c_cc_refusal_siblings]**; only single-call fixtures exist | A session log with a refused parallel batch, or Claude Code's permission code (the user; Anthropic) |
| How do Codex aborts, Claude Code `!` commands and "[Request interrupted by user]" arrive through Harbor, and do the construction transcripts hold `!` commands (L-6, NEW-5)? | Read `harbor-framework-harbor/src/harbor/agents/installed/codex.py:1050-1066` and `harbor-framework-harbor/src/harbor/agents/installed/claude_code.py:954-978`; grep for `bash-input` found nothing | Real Codex and Claude Code logs with these events (the user; the frozen construction transcripts) |
| Does Scale's hub drop runs with high or critical findings from training? | grep of agentenv `src/` and `tst/`: prompt text only (`agentenv-framework/src/agent_env/task_step/task_steps/verifiers/judge_utils/judge_output_format.py:175-177`) | The hub's consumer code (Scale) |
| How often do PR checks end `timed_out` or `startup_failure`, infra or PR-caused (R-3's size)? | `labels_dev.json` (6 checks, all success); report rows; MiMo's vocabulary is split | `data/cache`, `data/m4` (a re-fetch on the builder's machine); GitHub's conclusion docs (not opened) |
| Do PCL PRs hold two same-named jobs on one commit (L-7) or body-only fix words that flip outcomes (L-3)? | Synthetic probes q1a and `sws_fixlike_compare` | The same PCL cache |
| Do traces hold one file under two absolute roots, or retried actions, that `last_writer` would misgrade? | Probe `p_nemo_paths` case 4; read `signals/attribute.py:33-49`, the capture retry collapse, `otel_import.py:98` | A multi-mount trace; `sql_audit_log` or `file_drop` data with retries |
| Can `grade_report.json` carry `pass: null`, and are hidden perf gates stable on oracle re-runs? | Grep: readers only; 4 reports, one boolean per gate | The graders' schema; GPU oracle re-runs (task authors) |
| Should the gate score `check_result` polarity? A wrong sign passes it today | Read `signals/gate.py:91-163`: type, attribution, outcome only | Hand labels with polarity (the labeller) |
| Does `crucible label` into the same `out` keep stale carried signals for a failed extraction? | Probe t_b1 with a fresh `out` | A re-run into one `out` (not run under the read-only rule) |
| What did THIRD_PARTY port from OpenEnv `55347f62e2`; how does MiMo run SWE tests and flakes? | `git cat-file` (absent); searched xiaomimimo-verl recipes | That commit; the empty `third_party/mimoagent-osr` and `uni_agent` submodules |

## 10. Handoff: findings outside this module

| To | Finding |
|---|---|
| M0 core | L-1's wider fix changes `event_key` for row-less events (native call id, then an ordinal; `core/schema/trace.py:52-62`); the importers that stamp shared times are M3's (`connectors/chat_messages.py:52-86`). Meanwhile the store carry maps a duplicated key to its last seq and moves a hand signal from event 2 to 4 (`connectors/trace_store.py:52`) **[ran: p_oe3_carry_identical]**. NFC is still missing (HO-M00a-1). |
| M3 connectors (importers) | L-2's change site is `interop/harbor/atif_import.py:77-121`: Harbor's mini converter drops `extra` (`harbor-framework-harbor/src/harbor/agents/installed/mini_swe_agent.py:337-351`), and the comment at `atif_import.py:81` is wrong for that path. Harbor's Codex converter discards `turn_aborted` (`harbor-framework-harbor/src/harbor/agents/installed/codex.py:1061-1062`); a prime-agent importer precedes L-6. |
| M3 connectors (ATIF chains) | `continued_trajectory_ref` is not followed: Terminus-2 segments sharing a `session_id` collide on one trace id and drop the earlier version's signals (`interop/harbor/atif_import.py:163-169`; `connectors/trace_store.py:46-90`) **[ran: tc01_atif_continuation_ids]**; `infra-rllm-org-rllm/rllm/integrations/harbor/atif_trajectory_bridge.py:43-77` follows it. Later. |
| M3 connectors (PR fetch) | Check runs are head-only with the default filter (`CP/connectors/github_pr.py:174,185`; AUD-M5), so reruns never reach the flaky rule; record `check_suite.id` (L-7, `:248-250`) and keep each check-run id as an attempt under (app, name, sha), as `infra-nvidia-nemo-rl/nemo_rl/experience/rollout_recovery.py:121-128` keeps attempts under a logical id. |
| M3 connectors (other) | A gate with `pass` null becomes a failed gate (`CP/connectors/agent_runs.py:78-81`; `CP/connectors/ci_logs.py:230-237`), where OpenEnv excludes it (`meta-pytorch-openenv/src/openenv/harbor/rollout.py:516-544`). Revert, backport and cherry-pick PRs are ingested as ordinary (SWP-C11; `huggingface-repo2rlenv/src/repo2rlenv/pipelines/pr_runtime.py:318-335`). AReaL's orphan rule could back clients that send no retry header (`infra-inclusionai-areal/areal/experimental/openai/cache.py:258-338`). |
| M6 credit | `credit/signed.py:29-30` names an M4 mark producer that does not exist. When L-9 adds it: mark only policy-sampled calls, keep unmarked apart from 0, refuse misaligned streams and FATAL capture findings, credit a forked node once, keep per-role baselines (`primeintellect-ai-verifiers/verifiers/v1/trace.py:229-266`; `infra-primeintellect-ai-prime-rl/src/prime_rl/orchestrator/algo/routing.py:11-40`; `meta-pytorch-openenv/src/openenv/harbor/contract.py:68-74`). |
| M6 credit (judge items) | rewardkit scores a judge timeout 0.0 (`harbor-framework-harbor/packages/rewardkit/src/rewardkit/judges.py:360-380`); mask it when judge items adopt rewardkit (`ARCHITECTURE.md:214`). |
| M7 verify (docs) | `THIRD_PARTY.md:157` still says "pre-write reward 0"; `verify/runtime/plan.py:384-396` removes reward files instead. |
| M8 forge / M7 verify | Failing hidden gates of frontier runs become `check_result` -1 and trap seeds with no oracle re-run (`CP/forge/derive.py:134-145`); `swe-bench-swe-smith/swesmith/harness/repair.py:138-215` drops tests that fail a gold re-run. |
| M11 interop (agentenv) | agentenv scores an agent-reported `error_type` as a 0 row that `aggregate_verifiers` counts (`agentenv-framework/src/agent_env/task_step/task_steps/verifiers/rubrics_verifier.py:381-393`); do not aggregate it with `crucible_verify`. User-sim and human turns stay in agentenv's conversation store and never reach M4 (`interop/agentenv/plugin/steps.py:1-6`). |
| M11/M12 verifiers import | `interop/verifiers/v1_import.py:83-95` cites `ae819de` lines, which hold at `e22533430`; the pinned `dca5562ee6` is not in the clone. |
| M11 train | Keep and test the exclusion of masked or reward-None rollouts from grouped baselines (`train/trainer.py:398`); slime misses it (`infra-thudm-slime/slime/data/batch_builder.py:210-257`). |
| M12 train | `train/trainer.py:558` calls `signed_rebalance` without row weights, so under `seq_mean_token_mean` or `const_max_len` mass is conserved on raw advantages, not gradient mass; MiMo passes prompt-mean weights (`xiaomimimo-verl/verl/trainer/ppo/v1/trainer_base.py:2121-2125`). M4 marks must land on policy tokens; NeMo-Gym masks ambiguous chains (`nvidia-nemo-gym/nemo_gym/token_id_capture/consumer.py:180-200`). |
| M15 studio | R-1 and R-2 land in `studio/datasets.py:621-628` and `studio/label.py:166-179`; row-count tests change. `dpo_pivots` treats 1 - reliability as a per-row noise rate (`studio/datasets.py:884-886`), but TRL takes one epsilon per run and its default loss ignores it (`infra-huggingface-trl/trl/trainer/dpo_trainer.py:1366-1367`). Later: reviewer text as SDPO or OPSD teacher context (`infra-huggingface-trl/trl/experimental/sdpo/sdpo_trainer.py:658`). |
| M17 task factory | agentenv's `rubric_evidence_with_mistakes` as trace auditor (`docs/research/task-factory/04-agentenv-curation.md:149`) would inherit its unchecked 0-based step and "unavailable = fail"; keep `factory/traceaudit.py:146,634`. |
| M1 world / THIRD_PARTY | `THIRD_PARTY.md:190` records 267 MiMo general envs; the local release has 925, so the M1 generality gate's denominator may be stale. |

## 11. Sources

- **References**: the 33 folders at the commits in §4. Earlier pins that differ (read where cited): nvidia-nemo-gym
  `3ef478d`, primeintellect-ai-verifiers `ae819de`, primeintellect-ai-research-environments `d4fb80a`, meta-pytorch-openenv
  `49aa302`, huggingface-repo2rlenv `5555443`, primeintellect-ai-prime-agent `2ee7e623a`, infra-volcengine-verl `5eb4826`,
  infra-primeintellect-ai-prime-rl `f7e8ab6`, infra-nvidia-nemo-rl `c9193bc`, infra-inclusionai-areal `298412a`,
  infra-thudm-slime `c933787`, infra-alibaba-roll `581046a`, infra-huggingface-trl `72cf37d`, infra-novasky-ai-skyrl
  `3d0ae22`. Not in local history: `dca5562ee6`, `4330be260a`, `55347f62e2`, `1f555b97a1`, `7cbffa77b`, `859938d`,
  `6cad8710bb`, `e42f222fcf`.
- **Crucible documents**: `DESIGN.md` (§3 P11-P29, §5.2); `ARCHITECTURE.md` (:20-60, §2.2, M4 card :526-543, §6-§8);
  `THIRD_PARTY.md`; `docs/modules/M04-signals.md`; `docs/audit/2026-10-12/M04-signals.md` and its four M04 studies;
  `docs/audit/2026-10-13/M15-study-defects-H1.md`; `docs/research/improvements/M03-M05.md`, `00-index.md` and the
  2026-10-14..20 M00-M03 handoff rows; `docs/modules/reports/m4_gate_*.json`; `CP/m4/labels_dev.json`, `TP/m4/labels_test.json`.
- **Crucible code**: `signals/` (all modules); `core/schema/{trace,signal}.py`, `core/agent_events.py`, `core/registry.py`;
  `connectors/{trace_store,redact,capture_proxy,chat_messages}.py`; `interop/harbor/atif_import.py`;
  `studio/{label,datasets,runs}.py`; `credit/signed.py`; `train/trainer.py`; `CP/{signals,outcomes,__init__}.py`,
  `CP/connectors/{github_pr,ci_logs,agent_runs}.py`, `CP/m4/`; `TP/signals.py`, `TP/m4/`; `tests/unit/test_m4_audit*.py`,
  `tests/unit/test_agent_run_exits.py` (67 M4 tests passed at HEAD).
- **Probes** (`probes/`, outputs in `notes/`): scout `p01_scout_baseline`, `p02_toy_gate_now`; ledger `p03_ledger`,
  `p04_fixlike`; harbor `h1`-`h3c`; openenv `p_oe1`-`p_oe3`; nemo-gym `p_nemo_paths`; agentenv `p_ae1_agentenv_judge`;
  prime `pp01_prime_sweep`; swe-safety `sws_*`; mimo `s01`-`s04`; trainers `t_a1_ci_timeout`,
  `t_b1_extract_failure_still_trains`, `tc01_atif_continuation_ids`; verify `v_oe1_*`, `v_sws4_tradeoff`, `v2_probes`, `v3_*`.
- **Work folder**: `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-27/M04-signals` (`01-brief.md`,
  `02-ledger.md`, `sweep-*.md`, `verify-1.md`..`verify-3.md`, `probes/`, `notes/`).
