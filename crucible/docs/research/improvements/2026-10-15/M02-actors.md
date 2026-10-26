# M02 actors: audit against the lab RL repos (2026-10-15)

Crucible HEAD `c27f5ca`. Citations: Crucible source under `crucible/src/crucible/` (`actors/policy.py:456`); other
Crucible files relative to `crucible/` (`packs/...`, `DESIGN.md`); reference code as `<folder>/<path>:line`, the folder
dropped in that repo's own §5 subsection. Commits are in §4. **[ran: probe]** marks a claim reproduced by a probe. Notes
and probes: `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-15/M02-actors/` (`02-ledger.md`, `sweep-*.md`,
`verify-*.md`, `probes/`; ledger probes in `probes/ledger/`).

## Triage (added 2026-10-20)

The 40 recommendations were written before audits tiered their items. They were triaged by the rule audits now use: a
recommendation is planned only if it prevents a realistic failure today, and the smallest version is taken. What runs
today (section 1): the training loop drives `LLMToolPolicy` (on the vLLM path a context refusal comes back as an empty
`length` reply, `train/vllm_http.py:108-112`), Runner attempts share one `LLMClient`, and M6 `branch_value` restores and
continues. Simulated users run only in the toy M2 gate, humans and peers only in tests, the episode clock only in the toy
ladder, and no pack attaches the check runner, so items about those wait. At `d282251` R-1, R-2, R-3, R-5, R-6, R-14,
R-15 and R-24 still hold; R-11 is done. Section 7 now holds only the 8 planned recommendations; the other 32 are named
there without detail.

## 0. Summary

- **Where it matters (§1).** `LLMToolPolicy` and `run_episode` set the trajectory, stop label, `truncated` flag and
  masks of every training rollout, single-turn run and Runner attempt; the overlong filter, `LimitPolicy`, the run stop
  rules and M6 `branch_value` read them. Simulated users run only in the toy M2 gate; no pack attaches a human or peer.
- **Ledger (§2).** 162 items (done = code plus a passing test or probe): 59 done, 16 partly, 86 not done, 1 wrong; four
  done items are corrected (AUD-L6, L9, L15, H4). The ≤1% fabrication gate holds only as a point estimate (95% bound
  1.29%; the re-run judged 0 sentences).
- **Gaps, with numbers.** An engine-path window refusal ends `agent_done` after 3 calls **[ran: tb02]**; a third cut-off
  reply's call executes **[ran: ss05]**; after one 0.20 USD call, 5 of 16 cheap calls fail `credits_exhausted` with 91%
  of the budget unused **[ran: v3_ss1]**; one responder fault masks all 3 `branch_value` continuations
  **[ran: v3_ng7]**; a restore under another scenario is accepted **[ran: ng02]**; 6 silent humans in 100 samples stop a
  run **[ran: v2_ng2]**.
- **Planned (triage 2026-10-20).** 4 must: R-1, R-5, R-2, R-3; 4 should: R-14, R-15, R-24, R-6. R-11 is done; the
  other 31 are not planned (section 7).
- **Coverage.** 33 reference folders (12 primary, 12 secondary, 9 not relevant); no failed cluster and no unverified
  candidate. 93 candidates verified: 10 rejected, 5 partly rejected, the rest merged into 40 recommendations.

## 1. What the module does

M2 is everyone in a World who is not the policy under training, plus the policy loop: grounded responders that answer
only from real records under a citation checker; scenario simulators checked by a judge from another model family; relay
and CI bots; humans and peers behind callable, file-queue and A2A transports; the coding reviewer; `LLMToolPolicy`,
`BashLoopPolicy` and `run_episode`; the LLM client (USD budget, replay cache, error classes, TITO path); and the gate's
fidelity measures.

| Aspect | Contract |
|---|---|
| Inputs | A started `World` with responders attached (`world/world.py:417-423`); `ActorTurn`s; records; `Scenario`s; `LLMClient`s; policies, `EpisodeLimits`, `sample_key` (`actors/policy.py:476-508`) |
| Outputs | Actions; `EpisodeResult` with stop, `truncated`, masks, `model_calls` (`actors/policy.py:490-503`); actor state in `WorldRef.actors` (`world/world.py:986`); `Spend`; gate reports |
| Called by | The consumers below; `runtime/single_turn.py:200`; 20 pack files |
| Layer | Imports only `core` and `world` (`pyproject.toml:153-171`) |
| Principles | P16 masks by kind and blame (`actors/policy.py:676-681`); P21 read-only simulator tools; P25 one choke point; P28 actor memory in the `WorldRef` |

**Gate: passed** (`ARCHITECTURE.md:39`). Reviewer precision 0.9683 (214/221), offline only (LED-6). Fabrication: coding
0/231 sentences (0 judged, all calls cached), toy 0/34; 95% upper bounds 1.29% and 8.43% against a 1% bar. 69 core unit
tests pass.

| Consumer | Uses it? | Evidence (recommendations it decides) |
|---|---|---|
| Training loop | yes; stops set `truncated` | `train/world_env.py:254,308` (R-1, R-5, R-13) |
| Runner attempts | yes, every native attempt | `runtime/harness/native.py:331-336` (R-2, R-7, R-11, R-12) |
| M6 `branch_value` | yes, restore then continue | `credit/operators/branch_value.py:56-58` (R-3) |
| Simulated users | toy M2 gate only | `packs/_toy/src/crucible_pack_toy/m2_gate.py:96` (R-20, R-33, R-34 latent) |
| Humans, peers | tests only | grep (R-21, R-31, R-38, R-39 latent) |
| Episode clock | toy long-horizon ladder only | `packs/_toy/src/crucible_pack_toy/longhorizon/ladder.py:110` (R-40 latent) |

## 2. Status ledger

| Item | State | Evidence |
|---|---|---|
| AUD-D1–D8 | done | Fact tokens, polarity, ≤2 citations, Unicode tokens, time rules (`actors/grounding.py:30,120-176`); `test_m2_audit.py` passed. |
| AUD-D9–D14 | done | Question ids, relay state, holds, atomic cache (`actors/llm.py:271-299,388-393`); judge sees tool results **[ran: l04]**. |
| AUD-L1–L5, L7, L8, L10–L14, L16 | done | `actors/policy.py:188-643`; `actors/llm.py:156-323`; `test_m2_audit_loop.py` passed; L11, L16 **[ran: l01]**. |
| AUD-L6, AUD-L15 | done, corrected | Hold only while the 2-nudge allowance lasts (`actors/policy.py:405-425`) **[ran: ss05]**; R-5. |
| AUD-L9 | done, corrected | The zero-token rule is only in `_complete_paid` (`actors/llm.py:379-381`) **[ran: tb02]**; R-1. |
| AUD-S2–S15 | done | `actors/simulated.py:64-262`; `test_m2_audit_sim.py` passed. |
| AUD-H2, H3, H5, H7, H8, H10–H12 | done | `actors/scripted.py:101-169`; `world/world.py:818-825`; `test_m2_audit_people.py` passed; H5 **[ran: l03]**. |
| AUD-H4 | done, corrected | Bounded only if `answer_timeout_s` is set (`actors/human.py:31,46`) **[ran: v1_ng8]**; R-38. |
| AUD-G1–G3, IMP-S1-actors, AUD-M04-M2, AUD-M03-D20 | done | `actors/grounding.py:88-137` **[ran: l02]**, untested; layering **[ran: l09]**; bots dropped **[ran: l07]**; M3 normalizes times. |
| IMP-M2-1 | partly | `invalid_arguments` works (`world/world.py:52-57`); Hermes only, uuid4 ids (`train/chat.py:55,58`); R-6. |
| IMP-M2-4 | partly | Only `Turn.trainable` exists (`core/schema/rollout.py:74-76`). |
| IDX-FIX-1 | partly | `cost_usd` right (`runtime/harness/native.py:384-387`); uncertain spend double-counts **[ran: l06]**. |
| AUD-H9, LED-7 | partly | Loud paged reads (`packs/coding/src/crucible_pack_coding/reviewer.py:99-126`); files over 2,800 lines mask. |
| AUD-R21 | partly | Card fixed; §3 tree stale (`ARCHITECTURE.md:353`). |
| STU-PL-F1, STU-PL-F3, STU-PL-F5 | partly | No window discovery, five stop lists (`actors/policy.py:659-660`), 9 overflow markers; R-14, R-15, R-25, R-32. |
| STU-SU-F4b, STU-SU-F14 | partly | Actor writes unjudged; `<think>` delivered (`actors/simulated.py:259-262`) **[ran: l04]**. |
| RES-HARN-1, RES-HARN-4, RES-TAX-7, RES-VFP-298, RES-LHR-B2 | partly | Status text ends `agent_done` (`actors/policy.py:421-425`); opt-in checkpoints; no transfer; sim calls unrecorded; kind `None` masks **[ran: l05]**. |
| IMP-M2-2, IMP-M2-3, STU-PL-F6, SWP-G4-119, SWP-M57-729 | not done | No TITO retry, limiter, latch or phase deadline (`actors/llm.py:324-347,396-426`) **[ran: l01]**; R-23, R-26. |
| IMP-M2-5, M00-R1 | not done | No traits, persona hash or RunLock (`actors/simulated.py:41-52`); R-4. |
| IMP-M2-6, AUD-F2, RES-HARN-5, HO-M00-1, M00-R19 | not done | Non-streaming chat only; cached tokens at the input rate (`actors/llm.py:252`) **[ran: l01]**; R-22. |
| IMP-M2-7, IMP-S4, NEW-6, AUD-NF-PARA, AUD-NF-COV | not done | Ad hoc judges (`actors/fidelity.py:80-97`); no labelled set; bounds 1.29%, 8.43%; paraphrase gate stale; no fidelity tests. |
| LED-1, LED-2, LED-4, LED-5, LED-6 | not done | Dropped negation passes (`actors/grounding.py:124-126`); `BudgetExceeded` escapes the audit **[ran: l02]**; doc says 0/225; verbatim answers unanswerables; offline reviewer gate. |
| AUD-F1, M01-LEDGER-F1F6, AUD-F5, AUD-F10 | not done | Responders under `_step_lock` (`world/world.py:654`); no remote snapshot; latest-message retrieval. |
| AUD-F3, AUD-F4, AUD-F11, RES-TAX-108, AUD-NF-S11 | not done | No transfer, cap, feedback or metrics (`actors/simulated.py:176-212`) **[ran: l04]**; R-20, R-29, R-33, R-34. |
| AUD-F6, AUD-F7, AUD-F8, AUD-F9, AUD-NF-H6, LED-8 | not done | No `source_kind`, token caps, pools or question ids (`actors/transports.py:55`) **[ran: l03]**; R-19, R-27, R-30, R-39. |
| AUD-NF-D12a, AUD-NF-D12b, STU-HP-F10, STU-HP-F11, STU-HP-D12c, NEW-9 | not done | Relay trusts `from` (`interop/agentenv/a2a.py:92-108`); last sender only; no probe; new id per send; unsaved counters **[ran: l03, l05]**; R-16, R-37. |
| STU-PL-F10, STU-PL-F11, STU-PL-F12, STU-PL-D14b, NEW-7, RES-HARN-2, RES-HARN-3 | not done | Duplicate ids; one action per step; seed 0 at T 0 (`actors/policy.py:158,376`) **[ran: l05]**; tokenless cache hits; no format cap; R-6, R-12. |
| STU-SU-F11, STU-SU-F15, STU-SU-F16, RES-ART-1, RES-ART-2 | not done | No reactions, examples, locale, verifying user, template or `abstain` (`actors/simulated.py:41-67`). |
| HO-M00-2, HO-M00-3, M00-R3, M00-R8, M00-R15, M00-R26 | not done | Long tool names, re-sent paid calls, unrecorded sampling, literal headers, unchecked cache mode **[ran: l01, l04]**; ad hoc MIME (`actors/policy.py:118-120`). |
| HO-M01-1, HO-M01-2, HO-M01-3, HO-M01-4, HO-M01-5, M01-R21, M01-R5 | not done | Unpinned media (`actors/policy.py:116-117`); no network prompt, pruning, counters or hints; no responder triggers (`world/world.py:691`); double reset. |
| DES-LH-1, DES-LH-2, DES-LH-3, DES-LH-4, DES-44-1, LED-9, LED-10 | not done | No group compaction, `calls.jsonl`, offloading or real-model run; `ActorSpec.grounded_on` unread (`core/schema/world.py:154-169`). |
| NEW-1, NEW-2, NEW-3, NEW-4, NEW-5 | not done | Reviewer masks >2,800 lines, 3 kinds unregistered, `finished` lost **[ran: rerun_p01-p03]**; validation needs `jsonschema` (`world/world.py:60-63`); registry kinds unused. |
| HO-M00-4, HO-M01-6, ARCH-M2-1, ARCH-M2-2, ARCH-M2-3, LED-3 | not done | Doc errors (`docs/research/improvements/M00-M02.md:55`; `ARCHITECTURE.md:39,353,483`); unquoted mkdir (`world/apps/mailbox.py:90`). |
| NEW-8 | wrong | No `LimitPolicy` rule covers the stop (`runtime/infra.py:132-140`), so no effect; R-32. |

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | How is a simulated person kept in character and factual, and how is fidelity measured? | correctness (fidelity) |
| Q2 | How does a simulated user end (stop, transfer, out of scope), and how are bad stops caught? | correctness (termination) |
| Q3 | How are actor and provider failures classified, masked or scored? | operability (failure taxonomy) |
| Q4 | How are calls reproducible yet distinct across k samples (seeds, cache keys, ids)? | identity and reproducibility |
| Q5 | How is non-policy actor state captured and restored on a branch or resume? | correctness (state and branching) |
| Q6 | How are spend, concurrency and rate limits bounded across episodes? | performance and scale |
| Q7 | How is context growth managed and compaction recorded for training? | correctness (long horizon) |
| Q8 | How are actors declared and wired, and how do agents and humans join? | interop and extensibility |
| Q9 | What can an actor do and see, and how is the policy protected from it? | security |
| Q10 | How do slow actors interact with the step loop, and how are shared humans kept apart? | operability (concurrency and latency) |
| Q11 | How is malformed model output handled, and which stop labels result? | correctness (policy harness) |
| Q12 | How are actor configurations identified and versioned? | evolution and versioning |

## 4. Repo triage

33 reference folders (Crucible and two loose download files excluded): 12 primary, 12 secondary, 9 not relevant.
Commit = HEAD read here, the earlier sweep's pin in parentheses where it differs. Absent: tau2-bench, nano-rlm, NeMo's
UserSim, MiMo's `mimoagent-osr`.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| harbor-framework-harbor | primary | User agent, terminus-2, pools, TrialLock | dc933be7 (a2fafbc) | Apache-2.0 |
| agentenv-framework | primary | A2A (our DEPEND), user sim, human peers | 015c89b (041afd7) | Apache-2.0 |
| primeintellect-ai-verifiers | primary | UserSimEnv, judges, limits, compaction | e22533430 (ae819de) | MIT |
| nvidia-nemo-gym | primary | CTU, nemo_user_sim, model servers | 7ece89985 (3ef478d) | Apache-2.0 |
| meta-pytorch-openenv | primary | tau2 and thinkingbox users, harness | 8d3455ab (49aa302) | BSD-3-Clause |
| swe-agent-mini-swe-agent | primary | BashLoopPolicy source; cost limits | 04d809ce | MIT |
| preferencemodel-karotte | primary | Message loop, retrying streaming source | 31c693e | MIT |
| huggingface-repo2rlenv | primary | Spend ledger; explicit-finish solver | 27d04ee (5555443) | Apache-2.0 |
| xiaomimimo-verl | primary | Infra blame, prompt fit, grader limits | a2ad9f61 | Apache-2.0 |
| infra-inclusionai-areal | primary | Policy proxy, seeds, tau2 wiring | aedc21ec (298412a) | Apache-2.0 |
| infra-rllm-org-rllm | primary | Policy half: stop enum, gateway | 4e828c81 | Apache-2.0 |
| infra-novasky-ai-skyrl | primary | Policy half: finish tool, skycap | a1e34dbf (3d0ae22) | Apache-2.0 |
| primeintellect-ai-prime-agent | secondary | Overflow table, retry policy | 3e0893260 (2ee7e623a) | MIT |
| primeintellect-ai-research-environments | secondary | Pinned tau2 simulator per row | f63f07c8a (d4fb80a) | Apache-2.0 |
| primeintellect-ai-prime-environments | secondary | User-sim envs; injection defences | cf220ee | Apache-2.0 |
| adithya-s-k-fineenvs | secondary | Parameter preflight; prompt SHA pin | 5e0b46e | Apache-2.0 |
| swe-bench-swe-smith | secondary | USD caps, cache_control | 9b74ac0 | MIT |
| infra-volcengine-verl | secondary | Parser registry; per-process limiter | 2035f58d (5eb4826) | Apache-2.0 |
| infra-primeintellect-ai-prime-rl | secondary | Source-owned sampling | 1e5318108 (f7e8ab6) | Apache-2.0 |
| infra-nvidia-nemo-rl | secondary | Status taxonomy, pausable deadlines | 925f70d55 (c9193bc) | Apache-2.0 |
| infra-openrlhf-openrlhf | secondary | Agent executor; per-engine seeds | dc2a7ad | Apache-2.0 |
| infra-huggingface-trl | secondary | Per-request seeds; retries | d9b9f39b (72cf37d) | Apache-2.0 |
| infra-thudm-slime | secondary | Window clamp; duplicate-id check | 05e3b017 (c933787) | Apache-2.0 |
| infra-alibaba-roll | secondary | Length clamp; Ray limiter | 0f37fe13 (581046a) | Apache-2.0 |
| servicenow-enterpriseops-gym | not relevant | Static user prompt; nothing better | 271f2c3 | Apache-2.0 |
| hf-space-fineenvs-multi-harness-rl | not relevant | Article Space; no actors | n/a (not git) | CC-BY-4.0 |
| hf-space-fineenvs-simulation-rl-environments | not relevant | Article Space; pin 46bb2be unverifiable | n/a (not git) | CC-BY-4.0 |
| hf-space-rl-environments-guide | not relevant | Comparison article | n/a (not git) | CC-BY-4.0 |
| 08-hf-rl-env-datasets | not relevant | Data; frozen persona turns | per dataset | per dataset |
| veds12-git-swarm | not relevant | Project page, prose only | 9f66af0 | none |
| karpathy-autoresearch | not relevant | Training scripts; no client | 228791f | none (README: MIT) |
| xiaomi-mimo-v2.6-rl-oss | not relevant | Dataset dump; no actors | HF rev 639865f | Apache-2.0 (README) |
| infra-moonshotai-checkpoint-engine | not relevant | Weight-update middleware | 84631a2 | MIT |

## 5. Findings per repo

### harbor-framework-harbor
- Approve-all ACP user agent (`src/harbor/bridges/acp.py:67`); terminus-2 ends its summary ladder with a fabricated
  reply (`src/harbor/agents/terminus_2/terminus_2.py:1203-1207`); TrialLock digests persona and prompts
  (`src/harbor/models/job/lock.py:175-244`); keyed pools (`src/harbor/trial/queue.py:112-192`).
- 45 commits since the pin, none here. Corrections: `src/harbor/trial/trial.py:651-662` is a wall-clock wait, not a
  parked turn (AUD-F1); terminus-2 does mark compaction in ATIF.
- Verdicts: better on R-4, R-14, R-30, R-39; Crucible better on judging, masking, budget, state; HB-C3, HB-C9, HB-C10
  rejected.
### agentenv-framework
- Dead peers are detected in ~60 s only given `sandbox_id` (`src/agent_env/a2a_agent/protocol.py:145-204`); the user
  sim's `done` is unjudged and its failure graded (`src/agent_env/task_step/task_steps/prompt_agent.py:609-645`); judges
  are re-asked with the error.
- 62 commits since the pin; 02b7259 lets personas provoke each other. Corrections: snapshot/v1 moved to
  `packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py:1054-1101`; Crucible imports none of the
  task steps its card calls DEPEND.
- Verdicts: DEPEND R-21; patterns R-10, R-17, R-34, R-39; Crucible better on masking, stops, memory; AE-2, AE-7
  rejected.
### primeintellect-ai-verifiers
- Unjudged user marker, 8-turn cap (`verifiers/v1/envs/user_sim/env.py:24-42`); window discovery and compact-and-retry
  (`verifiers/v1/harnesses/utils/compaction.py:63-256`); accepted attempts only trained
  (`verifiers/v1/trace.py:516-553`).
- 25 commits since the pin, none here. Corrections: the chat program reads with a 600 s timeout; the judge moved to
  `verifiers/v1/judge.py:121-248`.
- Verdicts: better on R-13, R-14, R-25, R-27, R-40; Crucible better on judging, masking, cache, budget; PV-C8 rejected.
### nvidia-nemo-gym
- CTU judges afterwards with the user's own model
  (`resources_servers/conversational_tool_use_simulation/app.py:1783-1826`) and scores transfer mismatches 0
  (`:1316-1330`); `nemo_user_sim` refuses row-digest mismatches (`resources_servers/nemo_user_sim/app.py:196-251`);
  error latch (`nemo_gym/openai_utils.py:1594-1666`).
- 139 commits since the pin. Corrections: indian_banking scores a failed simulator as-is, not 0; vLLM overflow can
  propagate.
- Verdicts: better on R-4, R-6, R-7, R-26, R-29, R-38; Crucible better on online judging, grounding, masking, memory.
### meta-pytorch-openenv
- New `tau2_env` raises on user failure (`envs/tau2_env/server/tau2_environment.py:312-321`) yet lets STOP reach the
  policy; thinkingbox hashes a config naming its models (`envs/thinkingbox_env/server/config.py:21-48`); capture
  negotiates parameters.
- 179 commits since the pin. Corrections: `src/openenv/validation/signature.py:12-26` lists RFC 008 categories, not a
  ban on simulated users; `src/openenv/core/harness/environment.py:254-296` blocks rather than parks; a WS human channel
  exists.
- Verdicts: patterns R-4, R-20, R-29, R-36; Crucible better on fidelity, masking, client, harness; OH-6 rejected.
### swe-agent-mini-swe-agent
- No tool call is a FormatError and 3 end the run (`src/minisweagent/agents/default.py:95-114`); limits are checked
  before each query; every exception is scored (`src/minisweagent/run/benchmarks/swebench.py:157-160`).
- HEAD equals the pin. Corrections: the global cost cap overshoots by the calls in flight; `THIRD_PARTY.md:44` names the
  wrong config files.
- Verdicts: patterns R-5, R-16, R-27, R-28, R-39; Crucible better on masking, cache, compaction, holds.
### preferencemodel-karotte
- Retry-After up to 300 s from headers or body (`src/karotte/agents/builtin_source.py:146-169`); per-family cache hints
  (`src/karotte/providers.py:102-142`); malformed arguments replaced in history
  (`src/karotte/agents/message_loop.py:240-255`).
- HEAD equals the pin. Correction: `actors/llm.py:160-162` credits Karotte, yet Crucible caps at 60 s from headers.
- Verdicts: PORT R-35, pattern R-22; Crucible better on cut-offs, which Karotte executes, and masking.
### huggingface-repo2rlenv
- SQLite ledger with uncertain status and denial reasons (`src/repo2rlenv/campaigns/budget.py:27-257`); waits only on
  pressure (`src/repo2rlenv/quality/loop/client.py:45-100`); a commit tool ends the solver.
- 6 commits since the pin, none here. Correction: USD caps with holds existed at the pin, contrary to the 2026-10-12
  audit.
- Verdicts: patterns R-2, R-24, R-28; Crucible better on pricing (repo2rlenv books unpriced calls at $0).
### xiaomimimo-verl
- Trajectory timeouts are infra (`recipes/general/agent_loop.py:78-87,931-942`); prompt fit checked first (`:506-587`);
  `call_{n}_{i}` ids (`:375`); grader AIMD and shared 429 pause
  (`recipes/design/grader_service/src/design_grader/service/vision.py:104-388`).
- HEAD equals the pin; the user-agent submodules are empty, so those claims are not established. Corrections: verl has
  nine parsers; the fork masks infra.
- Verdicts: patterns R-6, R-8, R-26, R-40; MV-6 rejected; Crucible better on compaction, state, masking.
### infra-inclusionai-areal
- Hashed per-request seeds (`areal/utils/seeding.py:22-27`); exact window clamp
  (`areal/experimental/openai/client.py:1204-1236`); zero-interaction overflows rejected
  (`areal/experimental/openai/proxy/workflow.py:598-617`).
- 3 commits since the pin. Corrections: call ids are uuid4 (`areal/experimental/openai/tool_call_parser.py:219-227`);
  only the control plane retries.
- Verdicts: patterns R-1, R-8, R-12, R-37; Crucible better on masking (tau2 scores simulator errors 0).
### infra-rllm-org-rllm
- Termination enum with per-reason masks, off by default (`rllm/trainer/algorithms/config.py:110-162`); only never-sent
  requests retried (`rllm-model-gateway/src/rllm_model_gateway/proxy.py:531-566`); `max_tokens` clamped, 4xx retried.
- HEAD equals the pin. Corrections: `rllm-model-gateway/src/rllm_model_gateway/proxy.py:747-768` retries only
  `ConnectError` (IMP-M2-2's source); HO-M00-3's skycap cite concerns file copies.
- Verdicts: patterns R-1, R-11, R-23, R-24, R-32; Crucible better on masking, cut-offs, error classes, budget.
### infra-novasky-ai-skyrl
- A finish tool ends skyrl-agent (`skyrl-agent/skyrl_agent/agents/react/react_agent.py:345-380`), which masks infra with
  policy outcomes; skycap hashes sampling into identity (`skycap/src/skycap/hashing.py:27-30`).
- 1 commit since the pin. Corrections: `skyrl/train/utils/rate_limiter.py:82-178` limits trajectories, not calls
  (IMP-M2-3's source); SkyRL does mask.
- Verdicts: patterns R-11, R-23, R-28; TC-7 rejected; Crucible better on compaction and nudges.
### Secondary patterns
Paths are relative to the folder in bold.
- **primeintellect-ai-prime-agent**: 24 overflow patterns (`crates/pa-ai/src/utils_inner/overflow.rs:8-100`; R-15).
  Correction: Retry-After over 60 s ends the chain.
- **primeintellect-ai-research-environments**: simulator revision per row
  (`packages/tau_common/tau_common/taskset.py:55-56`).
- **primeintellect-ai-prime-environments**: delimited untrusted text (`environments/agent_dojo/agent_dojo.py:48-57`).
- **adithya-s-k-fineenvs**: parameter preflight (`tooling/mimo-rl-explorer/app/runner/core.py:172-191`; R-36).
- **swe-bench-swe-smith**: per-instance USD caps, `cache_control` (`agent/swesmith_gen_claude.yaml:63-71`).
- **infra-volcengine-verl**: nine-format parser registry (`verl/experimental/agent_loop/tool_parser.py:96-816`); its
  limiter is per process (correction).
- **infra-primeintellect-ai-prime-rl**: sampling stamped per sample (`src/prime_rl/orchestrator/train_sink.py:256-262`);
  Crucible's RunLock cite names a missing path (correction).
- **infra-nvidia-nemo-rl**: markup flags invalid calls (`nemo_rl/environments/nemo_gym.py:369-431`; R-9).
- **infra-openrlhf-openrlhf**: per-engine seeds (`openrlhf/trainer/ray/vllm_engine.py:269`); nothing to adopt.
- **infra-huggingface-trl**: random per-request seeds (`trl/experimental/async_grpo/async_rollout_worker.py:997-1000`).
- **infra-thudm-slime**: clamp minus draft slots (`slime/agent/adapters/common.py:534-558`; R-1); user strategies only
  named in a comment (correction).
- **infra-alibaba-roll**: pre-call clamp (`roll/pipeline/agentic/env_manager/traj_env_manager.py:219-232`; R-1).

## 6. Gap matrix

● built and tested, ◐ built with a stated weakness, ○ absent.

| Q | Crucible | Harbor | agentenv | verifiers | NeMo-Gym | OpenEnv | mini-swe | Karotte | repo2rlenv | MiMo | AReaL | rLLM | SkyRL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | ◐ online checker, judge; no metrics | ○ persona only | ○ no checks | ○ persona rules | ◐ post-hoc, same model | ○ delegated | ○ none | ○ none | ○ none | ○ none | ○ sim unchecked | ○ none | ○ none |
| Q2 | ◐ judged markers; no transfer, cap | ◐ agent-settable reasons | ◐ unjudged done | ◐ unjudged; 8-turn cap | ● transfer vs scope | ◐ TRANSFER; turn budget | ● submit sentinel | ◐ text ends | ● finish tool | ○ policy only | ○ external | ○ policy only | ◐ finish tool |
| Q3 | ◐ per-kind blame; 3 unregistered | ◐ regex; missing reward 0 | ◐ liveness; sim graded | ◐ user failure scored | ● validated kinds | ○ timeout scores 0 | ○ all scored | ◐ typed limits | ○ not established | ◐ tool crash scored | ◐ tau2 scores 0 | ○ all ERROR | ○ infra mixed in |
| Q4 | ◐ attempt cache; uuid ids; no seeds | ◐ derived ids | ○ uuid4 | ◐ sampling per call | ◐ seed per rollout | ◐ sent sampling kept | ○ none | ○ none | ◐ request-hash receipts | ◐ positional ids | ● hashed seeds | ◐ request as sent | ◐ sampling in identity |
| Q5 | ◐ WorldRef state; finished lost | ○ not resumable | ◐ remote snapshot/v1 | ○ episode resume | ○ in-memory | ○ none | ○ no resume | ○ none | ○ n/a | ○ actor killed | ○ none | ○ none | ○ none |
| Q6 | ◐ hard budget; inflated holds | ◐ no budget | ○ none | ◐ token caps | ● slots, latch | ◐ param negotiation | ◐ run cost cap | ◐ no budget | ● uncertain ledger | ● AIMD, shared pause | ◐ $0 unpriced | ◐ never-sent retries | ◐ token bucket |
| Q7 | ◐ 5 strategies; fixed window | ◐ declared window | ○ none | ● discovery, retry | ◐ recency only | ○ none | ○ aborts | ○ stops | ○ stops | ◐ prompt-fit check | ◐ clamp only | ◐ clamp only | ◐ length stop |
| Q8 | ◐ spec unused; code wiring | ● factory, validated | ● typed steps | ● roles by name | ● roles in config | ◐ task data; WS human | ◐ human as policy | ○ n/a | ○ n/a | ◐ YAML per row | ◐ sim from config | ◐ by name | ◐ registry |
| Q9 | ◐ read-only, scrub; no provenance | ○ approve-all | ◐ caps; unframed | ◐ tool-less user | ◐ grants; no scrub | ○ names only | ○ n/a | ○ n/a | ◐ untrusted framing | ○ n/a | ◐ HMAC tokens | ○ none | ○ none |
| Q10 | ◐ blocks under lock | ◐ pools; 3600 s | ◐ CAS; fails open | ● per-segment time | ● 300 s deadline | ◐ blocking, capped | ◐ TTY check | ◐ timed turns | ○ n/a | ◐ timeouts | ◐ 600 s episode | ○ n/a | ○ n/a |
| Q11 | ◐ nudges lapse; text is done | ◐ double confirm | ○ JSON scan | ◐ cut-off runs | ◐ id checks | ○ text is done | ● 3-strike format | ◐ cut-off runs | ● tool required | ◐ repetition check | ◐ bad JSON as text | ○ cut-off runs | ◐ uncounted guidance |
| Q12 | ○ no actor identity | ● lock digests | ● versioned tasks | ● config on trace | ● pinned, row digest | ● config sha256 | ● config per trajectory | ◐ model only | ● requests saved | ● grader version | ◐ tau2 unpinned | ◐ request as sent | ◐ sampling only |

## 7. Recommendations, ranked

Triaged on 2026-10-20 (see the Triage section): 8 of the audit's 40 recommendations are planned, 4 must and 4 should;
R-11 was done meanwhile by M0's A14-R3. The other 31 are listed by title only, under Not planned; their detail stays in
the work folder. Order: R-14 before R-1's clamp; R-1 and R-5 together (both relabel the nudge path).

| R | Title | Tier | Kind | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| R-1 | One window rule at every call boundary | must | defect | H | L | L |
| R-5 | Nudge path never runs cut-off calls | must | defect | M | L | L |
| R-2 | Budget holds price the call; pressure waits | must | defect | H | L | L |
| R-3 | `World.finished` and `failures` follow restore | must | defect | H | L | L |
| R-14 | Declared or discovered context window | should | production gap | M | L | L |
| R-15 | One overflow wording table | should | production gap | M | L | L |
| R-24 | Failed paid calls booked as uncertain | should | production gap | M | L | M |
| R-6 | Unique deterministic call ids | should | defect | M | L | L |

### R-1. One window rule at every model-call boundary
- **Kind** defect · **Status** correction of AUD-L9 (hosted path only); the clamp is new. TB-2, TC-1.
- **Problem.** vLLM's context-length 400 becomes an empty `length` reply on the training path
  (`train/vllm_http.py:108-112`; `train/world_env.py:186-229`); `LLMToolPolicy` nudges twice and ends `agent_done`,
  untruncated (`actors/policy.py:400-425`), outside `LimitPolicy` and the overlong filter (`train/trainer.py:568`).
  **[ran: tb02]** window 400, prompt 228, `max_tokens` 300: three empty calls, then `agent_done`. Nothing clamps
  `max_tokens` (`actors/policy.py:375`): **[ran: tc01]** a 26,172-token prompt in a 32,768 window is refused with 6,596
  tokens free.
- **Reference.** `infra-inclusionai-areal/areal/experimental/openai/client.py:1204-1236` sends `min(requested, window -
  len(prompt_ids))` and raises before the engine at ≤0; slime, rLLM and ROLL do the same.
- **Recommendation.** Smallest first (triage): rule (4) on every path; the clamp (1-3) once R-14 gives a window. One `fit_request` for `_complete_paid`, `_tito_complete` and `EngineLLM.complete`: (1) window =
  R-14's window minus a margin (speculative reserve); (2) too little room raises `ContextOverflow` before any hold or
  spend; (3) else send `min(requested, room)` and record both values (R-11); (4) on every path a zero-token `length`
  finish raises `ContextOverflow`; (5) unknown window: rule 4 only. `GenResult` is unchanged; the wrong comment at
  `train/vllm_http.py:110-111` is fixed.
- **Reasoning.** Rule 4 alone wastes the refused call; the clamp alone needs a window. P16; ARCH §7 rule 4.
- **Established by.** Code on both sides; tb02, tc01 (`probes/v4_rerun_sweep_probes.out`).
- **Impact / Effort / Risk.** H: window-edge episodes train as voluntary stops · L: one function · L: edge calls only.
- **Mode.** PATTERN (AReaL, slime, rLLM, ROLL; Apache-2.0).
- **Acceptance.** tb02 A makes one call with 172 tokens sent; tb02 C ends `context_overflow`, truncated, no nudges; tc01
  A records 6,596 sent; `tests/unit/test_llm_tito.py` passes.

### R-5. Nudge path: no cut-off call runs, no stall is `agent_done`
**Kind** defect · **Status** correction of AUD-L15, AUD-L6; PV-C5, SS-3, OH-4 · **Impact / Effort / Risk** M/L/L · **Mode** PATTERN (verifiers, MIT) \
**Problem.** After 2 nudges a cut-off reply's calls run and reset the allowance, and a blank one ends `agent_done`, untruncated (`actors/policy.py:400-425`). **Reference.** `primeintellect-ai-verifiers/verifiers/v1/trace.py:620-632` truncates on a last `length` finish. **Recommendation.** Never run `length` calls; a spent allowance ends `max_output_tokens` (truncating) or `format_error`; label nudged calls. **Reasoning.** P16. **Established by.** **[ran: ss05, pv05, v1_ss3]**. **Acceptance.** ss05 makes no World call and is truncated.

### R-2. Budget holds price each call's own worst case; pressure waits instead of failing
- **Kind** defect · **Status** new, a side effect of the AUD-D12 fix. SS-1.
- **Problem.** `_reserve` holds `max(priced worst case, _max_seen)` (`actors/llm.py:290-299,369`), though its docstring
  (`:293`) limits `_max_seen` to unpriced models; an over-budget hold raises `crucible:credits_exhausted`, which stops
  the run (`runtime/run.py:192-196`); packs share one client. **[ran: v3_ss1]** after one 0.2005 USD call, 16 concurrent
  ~1k-token calls each hold 0.2005 USD, not 0.014; 5 fail with 91% of 2.5 USD unused. **[ran: ss02]** holds alone refuse
  21 of 30 calls.
- **Reference.** `huggingface-repo2rlenv/src/repo2rlenv/campaigns/budget.py:27-53` classifies each denial and waits only
  on pressure from holds that can still settle.
- **Recommendation.** Invariant: `credits_exhausted` only when booked money plus this call's estimate exceeds the
  budget. (1) Hold = priced worst case × observed overrun ratio, no `_max_seen` floor. (2) Pressure waits on a condition
  notified at each settle; after `timeout_s` it raises `crucible:budget_pressure` (masked, retryable, never
  run-stopping). (3) `BudgetExceeded.source` (`local_cap` or `provider`) is named in the stop reason; waits are
  recorded.
- **Reasoning.** Dropping holds re-opens AUD-D12 (8× overspend); a retryable refusal still wastes the attempt. P16.
  Trade-off: waits count on the episode clock.
- **Established by.** Code on both sides; v3_ss1, ss02.
- **Impact / Effort / Risk.** H: a shared client stops the run, mislabelled · L: `_reserve` and `complete` · L: the cap
  and D12's test stay.
- **Mode.** PATTERN (repo2rlenv, Apache-2.0).
- **Acceptance.** v3_ss1 completes 16 of 16 and books ≤0.45 USD; ss02 completes 30 of 30; true exhaustion raises at
  once; D12's test passes.

### R-3. `World.finished` and `World.failures` follow restore
- **Kind** defect · **Status** carry-over NEW-4, widened to `World.failures`. NG-7; owned jointly with M1.
- **Problem.** Both are cleared only by `reset` (`world/world.py:270-271,598-599`), left out of `snapshot` and untouched
  by `restore` (`:974-1035`), while responders restore their own copies (`actors/base.py:31-37`). `run_episode` ends and
  stops on them (`actors/policy.py:606-613`); M6 `branch_value` restores then continues
  (`credit/operators/branch_value.py:56-58`). **[ran: v3_ng7]** k=3: a user DONE at reply 2 cuts continuations 2 and 3
  to 1 step; one responder fault in continuation 1 masks all three. **[ran: p03]** a World restored after a DONE runs to
  `max_steps`.
- **Reference.** `nvidia-nemo-gym/resources_servers/conversational_tool_use_simulation/app.py:830-847,890-927` derives
  terminal state and next actor from the replayed log.
- **Recommendation.** Invariant: after start, step, restore or reset, `World.finished` equals the finished responders'
  reasons and `World.failures` holds only this branch's faults. (1) `ActorBase` saves `finished_reason`. (2)
  `World.finished` becomes a derived view, the only one `run_episode` reads. (3) `WorldRef.failures` (absent reads `[]`)
  is restored. M1 owns `world.py`; M2 owns `ActorBase` and `run_episode`. Since M1 R-1, `World.failures` also collects
  `crucible:env_container_lost`, so a branch's lost container is carried into the branches after it.
- **Reasoning.** A second copy can disagree; clearing on restore breaks resumes. P28; P16.
- **Established by.** Code on both sides; v3_ng7, p03.
- **Impact / Effort / Risk.** H: `branch_value` estimates truncated or masked · L: one field, one view · L: additive.
- **Mode.** PATTERN (NeMo-Gym, Apache-2.0).
- **Acceptance.** v3_ng7 case 1 gives three 2-step continuations; case 2 masks 1 of 3; p03 B ends `user_stop`.

### R-14. Declare or discover the context window
**Kind** production gap · **Status** carry-over STU-PL-F1; HB-C1, PV-C3 · **Impact / Effort / Risk** M/L/L · **Mode** PATTERN (verifiers, MIT) \
**Problem.** Budgets default to 128,000 (`actors/policy.py:87`); providers have no window (`core/config.py:24-38`). **Reference.** `primeintellect-ai-verifiers/verifiers/v1/harnesses/utils/compaction.py:63-106`. **Recommendation.** Config, else `/models`, else an error for self-hosted; record it. **Reasoning.** No silent mismatch. **Established by.** **[ran: hb01]**: overflow, 0 compactions. **Acceptance.** A fake 32,768-token model card makes hb01 compact.

### R-15. One overflow wording table
**Kind** production gap · **Status** carry-over STU-PL-F5; PV-C4 · **Impact / Effort / Risk** M/L/L · **Mode** PORT (prime-agent, MIT, with notice) \
**Problem.** 9 wordings, one on TITO (`actors/llm.py:105-115,415-418`); missed overflows are masked. **Reference.** `primeintellect-ai-prime-agent/crates/pa-ai/src/utils_inner/overflow.rs:8-100`. **Recommendation.** One shared classifier with rate-limit exclusions. **Reasoning.** Overflows are scored on purpose. **Established by.** **[ran: v2_pvc4]**: 2 of 8 caught. **Acceptance.** A row per pattern passes.

### R-24. Failed paid calls booked as uncertain
**Kind** production gap · **Status** carry-over HO-M00-3, AUD-F7; SS-4, TC-8 · **Impact / Effort / Risk** M/L/M · **Mode** PATTERN (repo2rlenv, Apache-2.0) \
**Problem.** Possibly billed failures are re-sent and unbooked (`actors/llm.py:324-372`). **Reference.** `huggingface-repo2rlenv/src/repo2rlenv/campaigns/llm.py:36-100`. **Recommendation.** Book them uncertain; one keyed re-send; a failed-call record. **Reasoning.** Bounded double billing. **Established by.** **[ran: ss04]**. **Acceptance.** ss04 books both holds as uncertain.

### R-6. Unique, provider-safe, deterministic tool-call ids
**Kind** defect · **Status** carry-over STU-PL-F10, NEW-7, IMP-M2-1; NG-6, MV-2, HB-C8 · **Impact / Effort / Risk** M/L/L · **Mode** PATTERN (NeMo-Gym, Apache-2.0) \
**Problem.** Ids pass unchecked and empty ones get uuid4 (`actors/policy.py:456`; `actors/harness.py:142`), though they key penalties (`credit/shaping.py:27-37`) and the cache. **Reference.** `nvidia-nemo-gym/environment_servers/nemo_user_sim/app.py:805-829`. **Recommendation.** Keep valid unused ids, else `c` + 8 hex of sha256(actor:n:i); one helper, `train/chat.py` too. **Reasoning.** Repair, not penalty. **Established by.** **[ran: v1_ng6, v2_mv2]**: wrong call charged; replay masks. **Acceptance.** v1_ng6 charges only `agent:c2`.

### Not planned

Done since the audit:

- R-11 Record the sampling each call sent (M0 A14-R3: `Turn.sampling` is each call record's sent sampling, `runtime/record.py:14`)

Later, each due when the feature or setup it concerns is used:

- R-4 Actor configuration identity, recorded and checked on restore
- R-7 Register the actor failure kinds
- R-8 An opening prompt that cannot fit is a counted mask
- R-9 Tool-call markup in a text reply is an unexecuted call
- R-10 Repair unparseable judge verdicts
- R-12 Derived seeds for sampled policy calls
- R-13 Summary attempts as `compaction_attempt` nodes
- R-16 A checked state contract
- R-18 No message to an unanswered actor is lost
- R-19 Quote forged sender lines; add `source_kind`
- R-20 Simulator turn cap
- R-21 Dead A2A peer detected in a minute
- R-22 Prompt-cache hints; cached tokens priced apart
- R-23 Bounded TITO retry
- R-25 `LLMToolPolicy` on the RLM context engine
- R-26 Process-wide provider gate
- R-27 Per-episode token, cost and call caps
- R-28 Opt-in `submit` tool
- R-29 TRANSFER end and observed handoff
- R-30 Responders built from `ActorSpec`
- R-31 Transport failures classified
- R-33 Per-episode responder metrics
- R-34 Regeneration with the rejection reason
- R-36 Rejected parameters fail at admission
- R-38 World deadline per responder turn

Dropped:

- R-17 Bounded relay rule matching
- R-32 One stop-condition enum
- R-35 Malformed arguments resent as valid JSON
- R-37 Relay accepts only registered envelopes
- R-39 Transport pools, attribution, attendance
- R-40 Episode clock blames actor time

### Rejected in verification

| Ids | Reason |
|---|---|
| MV-6 | No repetition shown; 1,200-token replies end through the cut-off path (`actors/policy.py:157`), which R-5 relabels. |
| TC-7 | Benefit not established; it changes the observation; HO-M01-4 tracks it. |
| AE-2 | Its harm is R-20's; World-state stops are graded (P12) and baseless DONE is judged (`actors/simulated.py:98-100`); STU-SU-F11 stays open. |
| AE-3, NG-2, OH-3, OH-4, MV-2 (parts) | Post-marker text is harmless (`actors/simulated.py:82-84`); infra blame for unknown kinds drops the safety net; `can_end` shows no failure (50 turns, not 25); not training nudged blanks drops failure signal for unmeasured harm; `:` ids break Bedrock. |
| AE-7 | No World attaches a stateful peer outside tests; snapshot/v1 is runtime-owned; AUD-F5 waits. |
| HB-C3 | Needs a multi-replica router; none is deployed (`train/gateway.py:1-30`). |
| PV-C8 | Needs ~32k-token replies; the largest default is 4,096 (`runtime/single_turn.py:192`); AUD-F2 stays open. |
| OH-6 | Provider behaviour rests on a code comment; the diagnostic moved to R-11. |
| HB-C10 | ACP drives the agent under test, an M10 role, not an M2 peer. |
| TB-6 | No code wraps a stateful remote service; A2A peers replay (`actors/human.py:66-72`). |
| HB-C9 | The judge prompt already delimits inputs (`actors/simulated.py:87-100`); no injection attempted. |

## 8. What we already do better (keep)

- **Masking by kind and blame** (`actors/policy.py:663-681`); Harbor scores a missing reward 0
  (`harbor-framework-harbor/src/harbor/metrics/base.py:14-35`), as nine other repos score infra faults.
- **Online judge, grounded citations** (`actors/simulated.py:175-250`); CTU judges afterwards with the user's own model
  (`nvidia-nemo-gym/resources_servers/conversational_tool_use_simulation/app.py:1783-1826`).
- **Actor memory in the `WorldRef`** (`world/world.py:986`); Harbor's user agent cannot resume
  (`harbor-framework-harbor/src/harbor/models/trial/config.py:519-525`); five others keep none.
- **LLM client**: hard budget, fallback prices, classified retries, replay cache (`actors/llm.py:157-347`); Harbor books
  unknown cost as $0 (`harbor-framework-harbor/src/harbor/llms/lite_llm.py:545-566`).
- **Harness**: capped nudges, dropped cut-off calls, five compaction strategies (`actors/policy.py:265-473`); Harbor
  fabricates a reply (`harbor-framework-harbor/src/harbor/agents/terminus_2/terminus_2.py:1203-1207`).
- **Security, determinism, scope**: read-only tools, deterministic ids (`actors/simulated.py:171-190`); Harbor's bridge
  approves all (`harbor-framework-harbor/src/harbor/bridges/acp.py:67`); seven trainers lack actors.

## 9. Open questions

- **Not on disk.** tau2-bench, tau-bench, NeMo's UserSim, MiMo's agent submodules and prime-rl's submodules hold logic
  seen only through wrappers; MiMo's parquet needs pyarrow. Vendoring them answers.
- **Providers.** Rejection of duplicate ids or malformed history, honouring of cache hints and seeds, billing of
  abandoned reads, overflow wordings (R-6, R-35, R-22, R-12, R-24, R-15): billed live calls.
- **Engines.** vLLM's refusal (`train/vllm_http.py:108-111`), SGLang, unseeded DP decoding, verl determinism: a GPU run.
- **Deployment.** Routers (HB-C3), cross-process clients (R-26), TITO via the train gateway (R-23): M10/M12 owners.
- **Frequencies.** Judge parse failures, premature finishes, markup-as-content, judge injection, A2A message dedupe,
  snapshot/v1 adoption, prime-rl's user-failure handling: a gate or bench run with R-33.

## 10. Handoff: findings outside this module

| To | Finding |
|---|---|
| M0 core | Register R-7's and R-2's kinds; the RunLock embeds R-4's cards, after MiMo's manifest (`xiaomimimo-verl/recipes/write_run_manifest.py:14-201`) and agentenv's task versions; resolve headers from env (`primeintellect-ai-verifiers/verifiers/v1/configs/client.py:122-145`); reject URL credentials, as OpenEnv does; fix prime-rl cites (`docs/research/improvements/M00-M02.md:25,29`). |
| M1 world | Parts of R-3, R-18, R-19, R-38, R-40; responder calls fire no triggers (`world/world.py:691`) while agentenv guards persona chains; unbounded regexes (`core/triggers.py:31`); NEW-2 fallback binding as in repo2rlenv; unquoted mkdir (`world/apps/mailbox.py:90`). |
| M1 world (MCP) | Typeless union schemas drop every MCP tool and failing servers vanish (`xiaomimimo-verl/recipes/general/agent_loop.py:729-778`); MiMo ships 925 envs, not 267 (`THIRD_PARTY.md:135`); OpenEnv has mode-specific tools. |
| M1/M11 agentenv backend | A failed nl verify leaks its envelope to every policy, and `_deliver` trusts the sender (`interop/agentenv/world_backend.py:441-451,516-525`). |
| M3 capture | Unmarked retries are kept (`connectors/capture_proxy.py:11-13`); AReaL adopts the extended completion (`infra-inclusionai-areal/areal/experimental/openai/cache.py:258-367`); skycap forks a context graph. |
| M4/M10 admission | Check prompt fit before attempts: 2.67% of MiMo's tasks exceed 4,096 tokens (`xiaomimimo-verl/recipes/general/agent_loop.py:506-534`). |
| M5/M7 judges | Strict schema and correction re-asks (`agentenv-framework/src/agent_env/task_step/task_steps/verifiers/rubrics_verifier.py:170-195`); Harbor's median-of-N agreement; MiMo's head-and-tail drop reasons; OpenEnv scores bad verdicts 0.0 (`meta-pytorch-openenv/src/openenv/core/rubrics/llm_judge.py:91-99`); FineEnvs pins prompt SHAs. |
| M7 verify | Transfer truth, indian_banking floors, tri-state scorer (`nvidia-nemo-gym/resources_servers/indian_banking/README.md:54-61`); verifiers purges planted verdicts. |
| M6/M12 credit | Invalid-call advantages (`infra-nvidia-nemo-rl/nemo_rl/algorithms/single_controller_utils/advantage_stage.py:451-468`); MiMo's group-mean invalid rows; prime-rl's trainable-only RAE; rLLM's per-role groups (IMP-M2-4). |
| M8 forge | `RoleTrialFailed` collapses Harbor's typed failures, with `max_retries=0` (`interop/harbor/role_runner.py:81-87`); SWE-smith conditions issues on real ones. |
| M10 runtime | Dead A2A policy agents graded as timeouts (`interop/agentenv/harness.py:101-117`); non-transactional ledger writes (`runtime/budget.py:48-54`); policy-only cost (`runtime/harness/native.py:382-389`); silent checkpoint skips; no default wall clock. |
| M10 runtime (policy) | Per-agent retries (`primeintellect-ai-verifiers/verifiers/v1/utils/retries.py:77-158`); NeMo-RL's infra and data retry budgets; prime-rl's per-mode limits and AIMD; NeMo's episode lifecycle; SkyRL's finish-reason ratios; rLLM's trace clearing; FineEnvs' judge relay. |
| M11 interop | ATIF lacks compaction markers (`harbor-framework-harbor/rfcs/0001-trajectory-format.md:629-662`); OpenEnv's TrainingTrace and RFC 008; NeMo's tool grants; verifiers' ACP runner. |
| M11 engines | Salt the prefix cache per weight version (`infra-novasky-ai-skyrl/skyrl/train/generators/skyrl_gym_generator.py:322-338`) rather than resetting it (`train/vllm_http.py:178`). |
| M12 train | Hermes-only parser with uuid4 ids (`train/chat.py:31,49-58`); declared-sampling gate (`train/trainer.py:745-748`); unused seed; SGLang overflow masks (`train/sglang_http.py:72`); aborting pause (`train/vllm_http.py:60-66`); MiMo's infra metrics and raw dump; OpenEnv keeps missing-logprob turns; NeMo's role-tagged logs. |
| M13 bench | Group by actor digest (R-4); OpenEnv's tau2 recipe drops errored tasks (`meta-pytorch-openenv/examples/claude_code_harness_eval/run_eval.py:61-63`). |
| Docs | `THIRD_PARTY.md:44` cites the wrong mini-swe-agent config; fix ARCH-M2-1 to ARCH-M2-3, HO-M00-4, HO-M01-6, LED-4, `runtime/keys.py:5-6`. |

## 11. Sources

- **Repos.** The 33 folders of §4 at the commits there; agentenv also at its pin 041afd7.
- **Crucible code.** `actors/` (all, with `rlm/`); `world/world.py`; `core/schema/`, `core/config.py`,
  `core/registry.py`; `runtime/` (`run.py`, `infra.py`, `record.py`, `keys.py`, `checkpoint.py`, `budget.py`,
  `harness/native.py`); `train/` (`world_env.py`, `vllm_http.py`, `chat.py`, `trainer.py`, `runtime.py`); `credit/`;
  `interop/agentenv/`; the `_toy` and `coding` actor, gate and bench files; `tests/unit/test_actors*.py`,
  `test_m2_audit*.py`.
- **Crucible documents.** `DESIGN.md` §3, §4.4, §10.4; `ARCHITECTURE.md` §0, §3, §7 and the M2 card; `THIRD_PARTY.md`;
  `docs/modules/M02-actors.md` and its reports; `docs/audit/2026-10-12/M02-*.md`; `docs/research/improvements/`
  (`M00-M02.md`, `00-index.md`, `2026-10-14/`); research notes in `m5-m6/`, `m1-taxonomy/`, `long-horizon-rlm/`.
- **Probes** (`probes/`). Scout p01-p04; ledger l01-l09 and rerun_p01-p03; sweep ae, hb, mv, ng, oh, pv, ss, ta, tb, tc;
  verify v1_* to v4_*.
- **Work folder.** The path in the header; drafts in `writer/`.
