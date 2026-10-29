# Crucible, Part 1: the async RL framework and the World. What exists, and the improvements it needs

*Written 2026-10-29 against `main` at `f07ff594`. Paths are relative to `crucible/`.*

## Scope

Crucible is an RL post-training framework for long-horizon agents. It has two parts. This document covers **Part 1 only**:
- **The async RL framework.** RLM harnesses for long-horizon work, credit assignment, staleness control,
  stabilisation, score centering, reward-hacking defences, task and trace checks, and alignment during training.
- **The World.** Simulated databases, websites, people, time and tools, from which RL tasks and environments are
  extracted. The goal is a World dense and realistic enough that synthetic environments closely mimic real enterprise
  work: CUA tools through Harbor, Slack-, Notion-, mail-, calendar-, ticket- and CRM-like apps, and enterprise data
  feeding new tools and workflows.

Part 2, the automatic and self-improving task and environment factory, is a separate document ("the Part 2 plan"). Stage
names such as "F5" or "F6" and measurements such as "M13" refer to that plan. (G1–G7 in this document are clock gaps.)

**Evidence base.**
- A code map of the framework, built from the source with file:line references.
- Four research notes: ScaleEnv (arXiv 2602.06820) and environment synthesis; task generation; verifiers, traces and
  self-improving loops; enterprise Worlds, CUA and simulated people.
- Two code audits: async RL and harness/reward.

More than 350 papers and repos were checked against arXiv or their sources. Paper claims carry their arXiv id. Flags
are kept from the notes:
- **[uncertain]**, **UNVERIFIED**, **[secondary]** and **[web]** qualify the source of a claim.
- **[inferred]** marks our own reasoning.
- **[derived]** marks arithmetic on reported numbers.

The status words are:
- **live**: used by a real run;
- **partial**: wired but switched off;
- **built, unused**: tested but used by no real task;
- **toy-only**;
- **schema-only**;
- **absent**.

**Current facts.**
- **Held-out benchmark:** 10 sealed tasks.
- **Training pool:** 13 tasks.
- **Training so far:** the 35B MoE student (Qwen3.6-35B-A3B, LoRA) has taken 2 real gradient steps, on a toy task.
- **Queued on DeltaAI:** a 35B GPU verification job for score centering, routing replay, resume and async.
- **Being prepared:** an offline baseline on the sealed benchmark.

---

## 0. Summary: the improvements that matter most, ranked

**The diagnosis in four lines.**
1. The RL framework is deep but barely exercised. Most stabilisers and credit machinery are built; only 2 student
   gradient steps exist.
2. Several defences are off on the live training path. One is a real gap: the trace-level hack rules never run there.
3. The World framework is broad: 20 App templates, six backends, snapshots, a clock, actors and RBAC. The World in
   use is thin: **one App (a git repo in a container), no people, no time**.
4. Simulated people, the clock and enterprise Apps exist only in toy packs. Messaging, calendar and CRM are absent.
   Enterprise data never seeds a World.

| # | Improvement | Area | Why (evidence) | Effect |
|---|---|---|---|---|
| 1 | **Wire the L1 hack rules and format-error capture into `WorkspaceHarness`**, record-only first | RL / reward | Confirmed in code: `runtime/harness/workspace.py:189-307` overrides `run` and never calls `rule_flags` or `format_errors_by_call`. So on the live path no hack is ever confirmed, GAR zeroing and the turn-credit cancel never fire, and the format penalty has no data (§1.1). | The live training path gets the hack signal it was designed to have |
| 2 | **Read back real RL before turning more on:** the queued 35B job, then a synchronous sparse run on the 13 tasks, then one switch at a time (Φ-dense, τ=1 async) at equal compute | RL | Only 2 student gradient steps exist. Several switches changed at once on an unread path cannot be debugged (§2(a), R0–R4). | Each later change is measured against a baseline |
| 3 | **framework_health as a trainer stop condition;** per-run monitors for hack-flag rate, shaping-only gradient share and mismatch KL per staleness bucket | RL / ops | framework_health is an operator tool the trainer never reads (§1, table). | Collapse or hacking pauses a run instead of being found afterwards |
| 4 | **Treat every verifier as a classifier with a measured false-pass rate:** side-effect negatives, a per-task attack score from a different-family attacker, a gold-sanity gate on generated tests, exploit scans per language, a `report_task_defect` exit | Reward | 28.5% and 25.0% of sampled SWE-bench Verified and R2E-Gym tasks accept a wrong patch (2606.16062). Every false positive in 2610.09142 came from an unrequested change. An exit action cut GPT-5's cheating from 54% to 9% (2510.20270). | "Hard to hack" becomes a number per task |
| 5 | **A dense T1 App kit:** executable DB-backed Apps with generated CRUD for the verifier, curated business tools for the policy, and policies as code | World | Executable Worlds beat LLM-simulated ones at every model size, by up to 13.4 BFCLv3 points (2602.10090). The reference Worlds hold 9–32 apps and 100–600 tools; ours holds 1 App. | Enterprise workflows become buildable and gradeable by state |
| 6 | **A clock bus with notifications, an event DAG, time tools and timing checks** | World | GPT-5 (high) scores 0.0 on Gaia2's Time split (2509.17158, 2602.11964). Our clock has no notification channel and no event dependencies (§1.1). | Asynchronous, deadline-driven work, as real offices run, becomes trainable |
| 7 | **An org model and declarative people:** an `ActorSpec` the runtime reads, sampled profiles, people with their own tools, simulator rotation, audits | World | Actor fields are schema-only today. LLM users are over-cooperative, which inflates scores (2603.11245). Tool-grounded users err least: 6% against 12–13% (2506.07982). One fixed simulator made the policy collapse against another (2603.00540). | People become data, consistent across Apps, and realistic enough not to inflate reward |
| 8 | **Office core Apps** (chat, calendar, docs or wiki, tickets, CRM, plus the existing mailbox) as T1, and a **T2 realism pair** of real OSS apps (Mattermost or Rocket.Chat, Plane, Gitea) through official MCP servers | World | These cover most enterprise benchmarks (note 04 §8.2). TheAgentCompany uses Rocket.Chat and Plane, and Harbor already has its adapter. | Slack-, Notion-, Jira- and CRM-shaped workflows, plus a transfer check against real software |
| 9 | **Harbor as the CUA and isolation layer:** export Worlds as Harbor tasks with Apps as MCP sidecars, run the live lane on the harbor backend, CUA via `computer-1` on our desktop App | World / ops | Harbor's environments expose tools only as MCP, skills, APIs or CLIs. Our tested isolation stack runs only on the harbor and agentenv backends; the live lane's local backend has a no-op quiesce. | Isolation on the live path, and CUA on the same Worlds |
| 10 | **Enterprise data → World:** redact (HMAC pseudonyms, which exist, plus Presidio), synthesise (SDV or TabularARGN, DP options), gate on fidelity, build Apps and people, rebuild state from logs | World | Connectors ingest traces but never seed App state (§1.1). No paper gives a full trace-to-environment redaction protocol, so this is our design. | Customer schemas and workflows become Worlds, without raw data entering training |
| 11 | **A density card per World:** breadth, per-task use, cross-app coupling, asynchrony, hard negatives, health, realism against customer logs, transfer | World / eval | No paper defines a standard realism metric (note 04 §7). | "How dense is our World" gets a measured answer |
| 12 | **OTel GenAI export, counterfactual first-wrong-step labels, calibrated per-mode judges, task-clustered pass@k/pass^k with cost per pass** | Traces / eval | Import exists, export is absent. Step attribution is 14.2% on Who&When (2505.00212). One judge returned a constant score in 9 of 37 settings (2610.27465). | Failures are checkable for the intended reason with standard viewers; judges are trusted only when measured |

**Long-horizon training items** cover RLM and compaction in the RL loop, cross-turn credit, MoE and async stabilisation
defaults, and alignment during RL. §2(a) A9–A10 cover them at the code level. A literature addendum follows in §3.

---

## 1. Where Part 1 stands today

Status words follow the map:

| Word | Meaning |
|---|---|
| **live** | Used by a real run |
| **partial** | Wired but switched off on the live path |
| **built, unused** | Tested, but no real task or run uses it. Includes toy-only. |
| **stub / schema-only** | A shim, or a field nothing reads |
| **absent** | No code |

**The RL path.**

| Component | Status | Evidence |
|---|---|---|
| Async control loop (torch runtime; colocated, partial, disaggregated, elastic) | live on Modal (M12) | `train/runtime.py`, `train/scheduler.py:157`, `train/pool.py:74` |
| Async on the student path (Tinker, prime-rl) | **partial**: synchronous defaults | `scripts/tinker_rl.py:429-430` (`--max-staleness 0`, `--async-groups 0`; checked) |
| Staleness bound | torch live at τ=4; Tinker on the path at τ=0 | `train/scheduler.py:72-91`; `train/bridges/tinker/trainer.py:528-562` |
| Weight sync (NCCL buckets, Tinker `save_for_sampler`, prime-rl LoRA broadcast) | live | `train/weights.py`, `train/weightsync.py:170` |
| Remote leased rollout workers, env pool | built, unused on the student path | `train/workers.py`, `train/envpool.py` |
| Score centering: Dr.GRPO R − mean, no std; zero-spread groups dropped | live | `train/credit.py:88-99` (checked), `tinker/trainer.py:518-525` |
| Group ops (GRS, GAR, Bradley-Terry, tournament, zero_hacks) | built, unused | `train/group_ops.py`, `credit/gar.py` |
| Mismatch / MIS corrections (11 variants) | torch live; Tinker uses importance sampling only | `train/mismatch.py:138`; `tinker/trainer.py:142` |
| SC+MIS loss and MoE routing replay (prime-rl) | built; **the GPU verification job is queued on DeltaAI** | `train/bridges/primerl/loss.py`, `primerl/routing.py` |
| Φ milestone potential and turn credit (η=0.1) | live in the smoke; cancelled for hacked or gate-failed episodes | `factory/checkpoint.py`, `train/credit.py:52,122` |
| Predictive-validity gate for Φ weights | built, **no caller** | `credit/validity.py:133`; `checkpoint.py:128` |
| Honesty (STATUS.json Brier, GIVE_UP) | live | `factory/honesty.py`, `bridges/tinker/datums.py:50-62` |
| Format penalty | **off** (default 0), and **no data on the live harness** | `tinker/trainer.py:178` (checked); §1.1 |
| L1 hack rules (`rule_flags`) | **not on the live path** | `runtime/harness/workspace.py:189-307` (checked, §1.1) |
| L2 hack judge and GAR zeroing | built; never fed on the live path, since L1 never fires there | `credit/hack_detector.py:131-153`, `credit/gar.py:111` |
| Path rules, protected paths, forbidden ops | live; a hack-severity violation zeroes the reward | `factory/pathcheck.py` |
| Protected hidden tests in a fresh `--network none` container | live | `factory/lanes/windows_world.py:698-754` |
| Filter-driver neutralisation (snapshots never trust the agent's tree) | live | `verify/collect.py:61-92` |
| Loss masks, untrainable-rollout drops, masked-share stop | live | `train/bridges/common.py:70-125`; `tinker/trainer.py:183-184` |
| Training timeouts masked, agent clock counts own time | live (fixed 2026-10-28, R2) | `factory/envs.py:296-322` (checked) |
| Quarantine | live | `train/quarantine.py`; `tinker/trainer.py:407` |
| Trace audit (facts → first_wrong_step → RLM auditor → typed proposals) | live in the factory, offline; not in the training loop | `factory/traceaudit.py` |
| framework_health (about 20 deterministic detectors) | operator tool; the trainer never reads it | `factory/framework_health.py`; `scripts/train_monitor.py:129-181` |
| Curriculum (online p̂ sampler) | opt-in; default is round-robin | `train/curriculum.py` |
| Admission, teachers (OPD), critic (GAE) | built, unused | `train/admission.py`, `teachers.py`, `critic.py` |
| Compaction | built, **off in training**; harness audit R7 still open; one report of "2 compactions" contradicts "off" | `actors/policy.py:330-333`; `runs/handoff-deltaai-inference.md` |
| RLM harness (M16) | live for factory swarm sessions; **never trained**; audit C1 (critical if trained), H1–H3 open | `actors/rlm/*`; `runtime/harness/rlm.py` |
| Trainer bridges | Tinker live (2 steps); prime-rl smoke at 0.6B; torch on Modal at 4 GPUs; verl, SkyRL, rLLM and TRL never trained; slime not installed | map §2.3 |
| Async-RL audit R1 (overlap sampling with training) | open by design | `async-rl-audit.md` §4 |

**What has trained:** 2 gradient steps of Qwen3.6-35B-A3B on a toy 3-milestone window, through the Tinker smoke on
2026-10-20. There were also 3 prime-rl steps at 0.6B and 20 torch steps at 0.6B on Modal (map §2.1). About 790 student
rollouts measured difficulty; none took a gradient step.

**The World.**

| Component | Status | Evidence |
|---|---|---|
| `World` runtime: start, reset, step, snapshot, restore, verifier view | live | `world/world.py` |
| `crucible.world/v1` schema (Apps, actors, RBAC, catalogs, clock, events, isolation, turns, custody) | live | `core/schema/world.py` |
| One tool choke point `_call_tool` (RBAC, catalogs, quotas, interceptors, output bounds) | live | `world/world.py:505-610` |
| File system and git repo Apps (`common/workspace`, `common/git_repo`) | **live** (the only App the live lane uses) | `world/apps/workspace.py` |
| Shell through `DockerExec` | live | `world/exec.py` |
| SQLite App with a pack `tools_module` | built, unused (toy and finance_ap only) | `world/apps/sqlite_app.py` |
| Postgres, website (ASGI), browser (Playwright MCP), JSON API, CI runner | built, unused | `world/apps/*` |
| Desktop (Xvfb, xdotool, one `computer` tool) and VM (QEMU `loadvm`) | built, unused | `world/apps/desktop.py`, `world/apps/vm.py` |
| MCP-server App (stdio or HTTP, per-identity sessions) | built, unused: no pack mounts one | `world/apps/mcp_server.py` |
| Mailbox, doc library (folder of files) | toy-only | `world/apps/mailbox.py`, `doc_library.py` |
| Messaging (Slack-like), calendar, CRM | **absent** | map §1.2 |
| Clock (step, wall, sim), triggers, events | toy-only | `world/clock.py`, `core/triggers.py` |
| People: grounded responder, scenario persona, fidelity audit | toy-only | `actors/grounded.py`, `actors/simulated.py`, `actors/fidelity.py` |
| Declarative actor fields (`persona`, `grounded_on`, `policy_doc`, `transport`) | **schema-only**: built in Python, never from a spec | `core/schema/world.py:157-169`; actor kinds registered at `core/registry.py:330-338` but never resolved (checked) |
| Isolation stack (`guard/isolate`, custody, separate verifier, sandbox pool) | built; called only by the harbor and agentenv backends | map §1.4 |
| Harbor World backend | built; its only user was the deleted GPU lane | `interop/harbor/world_backend.py` |
| agentenv backend, MCP envs, A2A | tests only; `RowTask.BACKENDS = ("local",)` refuses export | `factory/envs.py:677` |
| Connectors (file_drop, sql_audit_log, otel_import, capture_proxy, chat_messages, atif, plus coding ones) | live as trace ingest; they **never seed App state** | `connectors/*`, `core/registry.py:340-345` |
| Trace redaction (credentials, e-mails and hosts to HMAC pseudonyms, canary gate) | built | `connectors/redact.py` (checked) |
| AWM and OpenEnv importers | partial: no AWM worlds in the repo | `interop/openenv/awm.py`, `_vendor/openenv_awm/` |

**Observability and evaluation.**

| Component | Status | Evidence |
|---|---|---|
| OTel GenAI **import** (OTLP/JSON spans → Trace) | built | `connectors/otel_import.py` (checked) |
| OTel GenAI **export** | absent | grep over `src/crucible` (checked) |
| Weave-shaped offline spans, with an upload once a key exists | built | `observe/weave_sync.py` (checked) |
| pass@1, pass^k (τ-bench estimator), mean final Φ | built | `bench/lh_metrics.py:94-129` (checked) |
| Run-to-run comparison clustered by task; power planning | built | `bench/compare.py`, `bench/power.py` (checked) |
| IRT (2PL) | used by the M11 benchmark runner, not the factory | `bench/irt.py` |
| Summaries with cost per solved task, hacks, calibration | live in measurement | `factory/tinker.py:11,177-179` |

### 1.1 Code checks of load-bearing claims

| Claim (source) | Check | Result |
|---|---|---|
| **`WorkspaceHarness` never fills the hack rules** (map §2.4, item 1) | Read `runtime/harness/workspace.py` (328 lines) and `runtime/harness/native.py:880-940`. Traced `factory/envs.py:326-328`. | **Confirmed.** `WorkspaceHarness` subclasses `NativeHarness` but overrides `run` (`workspace.py:189-307`). The override calls `run_episode` and `save_episode`, then collects and grades the patch. It never calls `rule_flags(trace)` or `format_errors_by_call(...)`. Only `NativeHarness.run` calls them (`native.py:917-926`), and so does the torch World env (`train/world_env.py:356-363`). `make_env(mode="workspace")` builds `WorkspaceHarness` (`factory/envs.py:326-328`), and that is the Tinker student path (`harness-reward-audit.md` §0). So on the live path, `res.flags` holds only the grader's own findings (`_take_findings`, `workspace.py:183`). The L1 trace rules (intercepted calls, tampering rules) never run. `hack/confirmed` is never set, so `credit/gar.py:111` and `train/credit.py:136-141` never see a hack. `Turn.format_errors` stays 0, so the format penalty has nothing to act on even if turned on. Path rules and protected paths still apply through the grade. |
| Tinker student path is synchronous by default (map §2.2) | `scripts/tinker_rl.py:425-431` | Confirmed: `--max-staleness 0`, `--async-groups 0`. |
| Format penalty off by default | `train/bridges/tinker/trainer.py:176-178` | Confirmed: `tool_format_penalty: float = 0.0`. |
| Keep-rule bars | `factory/targeting.py:467-496` | Confirmed: `SKILL_MIN = 0.5`, `BENCH_HI = 0.5`, train band [0.1, 0.6]. |
| Score centering is R − mean | `train/credit.py:88-99` | Confirmed (Dr.GRPO, masked rollouts get 0). |
| Task contract is patch-shaped | `factory/longtask.py:60-86` | Confirmed: `grade_patch`, `milestone_status(patch)`, `oracle_patch`, `mutants`. |
| Actor registry kinds never resolved | `core/registry.py:330-338`; grep for resolutions | Confirmed: registered, never resolved outside the registry. |
| Clock has no notification channel or event dependencies | grep `depends_on`, `notification` in `world/clock.py`, `core/schema/world.py` | Confirmed: no matches. |
| Redaction and OTel | `connectors/redact.py`, `connectors/otel_import.py`, `observe/weave_sync.py` | Redaction and OTel import exist; there is no OTel export. The map lists only the connectors, so this refines it. |

## 2. The improvements, area by area

Part 1 is an async RL post-training framework with a deeply integrated simulated World. Tasks and environments are
extracted from the World. This section covers six things in turn:
- (a) what to turn on along the training path;
- (b) how to make the World dense;
- (c) how to harden reward and verifiers;
- (d) how to observe traces;
- (e) how to evaluate;
- §3 adds the long-horizon RL training literature, and §4 the smoke-test results.

### 2(a) The RL path: what is off and should be turned on

**Principle.** Turn things on in the order that keeps each run interpretable. First comes one synchronous, sparse,
on-policy baseline that is read back. Then each switch is changed alone against it, on the same tasks and seeds. Several
switches at once on an unread path cannot be debugged.

| # | Switch | Today | Turn on? When | Risk | Acceptance test |
|---|---|---|---|---|---|
| A1 | **L1 hack rules and format errors in `WorkspaceHarness`** | Not called (§1.1) | **Now.** Port the 8 lines of `native.py:917-926` into `workspace.py` after `run_episode`. Start in record-only mode: flags are stored and reported but zero nothing until the L2 judge is calibrated (§2(e)). | False positives would zero honest rewards through the L2 → GAR path once enforcement is on. | A planted intercepted call in a workspace episode shows in `res.flags`. A re-scan of stored episodes (the `repath.scan` pattern) reports the flag rate per rule. A unit test fails on today's code. |
| A2 | Format penalty | 0 by default; and no data (A1) | **Not yet.** First measure the malformed-call rate per episode with A1's data. Turn it on only if malformed calls are a material share of turns. | It penalises exploration. Qwen3.6's interleaved template re-renders at harness turns (harness audit §0). | The rate is reported per run in `train_monitor`. If turned on: an ablation at equal compute shows no drop in pass rate. |
| A3 | Async on the student path (`--max-staleness`, `--async-groups`) | τ=0 | **After the sync baseline (R2 below).** Step to τ=1. Tinker has importance sampling only, no MIS, so keep τ small there. Larger τ belongs on prime-rl, where SC+MIS exists. | Off-policy bias. The audit's R1 overlap of sampling and training stays open. | τ=1 against τ=0 on the same 13 tasks and seeds: mismatch KL logged per staleness bucket (R2 is fixed in the audit). The reward curve lies inside the sync run's task-clustered CI. Throughput gain reported. |
| A4 | SC+MIS loss and MoE routing replay (prime-rl) | Built; not run live | **On the queued DeltaAI 35B job's outcome.** | Routing replay across weight swaps is refused by design (async audit R5). Coverage below 1 means no replay for that step. | The job's own checks (score centering, routing replay, resume, async), read back and recorded in DESIGN. |
| A5 | Curriculum (p(1−p) sampling, online p̂) | Opt-in; round-robin | **When the pool passes about 50 tasks.** With 13 tasks, round-robin is fine. | Too narrow a focus on boundary tasks. Over-sampling flaky tasks that look "medium" (SSR's randomly failing test, 2512.18552). | p̂ is tracked per (task, student, checkpoint). The sampler weight is p(1−p) (Online Difficulty Filtering, 2504.03380). Zero-variance groups are already dropped (DAPO's dynamic sampling, 2503.14476). A flaky-task guard reads S10's flake bound before upweighting. |
| A6 | Predictive-validity gate for Φ weights | `weights_from_validity` has no caller | **Wire it as a gate:** Φ weights are used only when validity passes. Otherwise train sparse. | DESIGN S2 failed at the midpoint: Φ AUC 0.50 against the judge's 0.75. The revision (measure where state has started changing; within-task AUC) is pending. | The S2 revision decided and measured. The gate's verdict recorded per task family. A dense run uses only validated weights. |
| A7 | framework_health read by the trainer | Operator tool; exit code 3 on a critical defect | **Now, as a stop condition.** A critical detector (masked_failure, credit_grade_mismatch, trainer_numeric…) pauses the run. It never edits it. | False stops waste a queue slot. | Each critical detector has a planted-defect test. Pause, then resume from the last checkpoint. |
| A8 | GAR `zero_hacks` and the other group ops | Built, unused | **After A1 has data** and the L2 judge passes calibration. | Zeroing on a weak judge removes honest signal. | The L2 judge passes the §2(e) protocol on the frozen anchor set. The zeroed share is reported per step. |
| A9 | Compaction in training | Off; harness audit R7 open; one contradicting report of "2 compactions" | **Not for the debug lane.** The audit measured median 110 turns, p90 279, no `context_overflow` at 64k/256k. **Needed for long multi-App Worlds.** Fix R7 first: pass a `ContextConfig` and save `_compactions_in_row` in the policy `state_dict` (`actors/policy.py:421,657-662`). Resolve the contradicting report. | Summary tokens train with the episode's advantage. Compaction can hide evidence the grader or honesty check needs (audit §3.4). | R7 fixed with a test. A long-World smoke where compaction fires, the trained token stream matches what the model saw, and honesty still reads its evidence. |
| A10 | RLM harness in training | Never trained; C1 critical open (`repl.py` pushed into a directory the kernel user can write, `actors/rlm/kernel.py:407,439-449`) | **Not until C1 and H1–H3 are fixed.** | The policy could rewrite its own REPL. | C1 red-team test: the agent cannot modify `repl.py`. Then a 2-step RLM smoke. |
| A11 | Remote rollout workers and the env pool | Built, unused on the student path | **Before scale.** CPU contention on one 14-CPU host kills candidates (axios, lo) and slows S3 and S10. | Isolation differs by provider. | The same task's S10 flake bound holds on the remote provider. Throughput per dollar reported. |
| A12 | Teachers (OPD) and critic (GAE) | Built, unused | P2 | Distillation hides whether RL itself teaches. | An ablation only, after R2–R4. |

**Score centering and stabilisation, as built and as planned.**
- **Centering.** Dr.GRPO subtracts the group mean, divides by nothing, and gives masked rollouts 0 (`train/credit.py:88-99`).
  Groups whose composed advantages are all equal are dropped (harness audit R8: keep groups whose composed advantages
  differ). With a band of [0.1, 0.6] and k=8, most kept tasks give mixed groups. Honesty and Φ can still separate
  members of a flat group. Watch the share of steps whose signal comes only from shaping: if most of the gradient is
  shaping, the dense-vs-sparse comparison measures the shaping, not the outcome.
- **Credit.** Turn credit follows `A_t = A + η(ΔΦ_t − c)·σ_R/σ_ΔΦ` with η = 0.1. It is cancelled for hacked or
  gate-failed episodes (`train/credit.py:136-141`). Until A1 lands, "hacked" never fires on the live path, so the cancel
  never fires either.
- **Stabilisers already live:** loss masks, the masked-share stop (above 50% of the last 64), quarantine, the per-bucket
  trust region (async R2), neutral sampling parameters (async B1), and the temperature guard (async B4).
- **Monitors to add to every run:** the hack-flag rate (A1), the impossible-twin cheat rate on an eval slice (§2(c)), the
  share of gradient from shaping only, and mismatch KL per staleness bucket.

**The run order this implies.**

| Run | What | Purpose | Paper measurement |
|---|---|---|---|
| R0 | Offline baseline (being prepared): student p̂ per task at step 0, from the targeting runs | Reference for every later run | Pre-training band and on-intent per task |
| R1 | The queued DeltaAI 35B verification job | Trust the prime-rl path: score centering, routing replay, resume, async | Infrastructure validity |
| R2 | Tinker, synchronous, sparse reward, the 13 training tasks | First read-back of RL on debug tasks | **Does the keep rule select tasks that teach?** Per-task learning progress against band position and on-intent share |
| R3 | R2 with Φ-dense credit (only if A6 passes) | DESIGN S5 | Dense vs sparse at equal compute, task-clustered bootstrap CI |
| R4 | R2 at τ=1 | Async cost and benefit | Throughput and stability against off-policy bias |
| Eval | 10-task sealed benchmark, plus every training task's untrained twin where one exists | Transfer | pass@1, pass^k, cost per pass. The CIs will be wide with 10 tasks (`bench/power.py` gives the plan). Held-out *families* do not exist yet; the Part 2 plan (§3.10) adds them. |

### 2(b) The World: how dense it is, and how to make it dense and realistic

#### How dense the World really is

The World *framework* is broad: 15 core App templates plus 5 interop templates, six backends, snapshot and restore, a
clock, events, actors and RBAC. The World as *used* is thin:

| World | Apps | Tools | People | Clock / events | Cross-app flows | Source |
|---|---|---|---|---|---|---|
| **Live debug lane** | 1 (`common/git_repo`) | Workspace tools: list, read, write, edit, search, shell, git status and diff | 0 | none | none | map §1.2, §1.5 |
| Toy helpdesk (most complete assembled World) | 3: tickets, kb, mail | pack tools | 2 grounded customers, a supervisor, an ops system actor (attached in Python) | step clock, 3 events | ticket ↔ mail | map §1.6 |
| finance_ap invoices | 2: ERP sqlite_app, kb doc_library | 130-line AP tools module | 1 policy actor | none | none | map §1.6 |
| DESIGN §4.4 AP-invoice example | 4 Apps, 4 actors, a clock and an event | — | — | — | — | **Fails validation with 5 errors**; two templates do not exist (map §1.6) |

For scale, the reference Worlds in note 04 §7:
- **AppWorld** (2407.18901): 9 apps, 457 APIs, 101 tables, about 370K rows and 106 people.
- **EnterpriseOps-Gym** (2603.13594): 8 domains, 512 tools, 164 tables.
- **Gaia2** (2602.11964): 12 apps and 101 tools.
- **Toolathlon** (2510.25726): 32 apps, 604 tools, about 70 tools offered per task.

So the answer to the lead's question is direct: **today's training World has one App, no people and no time.**

**The density card.** Note 04 §7 proposes it [inferred]. Build it as `world/density.py`, computed from the `WorldSpec`,
the State Adapters and the oracle traces, and stored in the World manifest:

| Group | Fields |
|---|---|
| Breadth | Apps; tools (visible to the policy against total); tables, rows, documents; simulated people, roles, RBAC rules |
| Per-task use (median and p90) | Apps touched, unique tools, calls, tables written, actors contacted, events fired, virtual time elapsed |
| Coupling | Share of tasks needing 2 or more Apps; share with a value read in one App and written in another; share needing a person's reply |
| Asynchrony | Share of tasks with a timed event or a deadline |
| Hard negatives | Share of infeasible, confidential and policy-violating requests |
| Health | Env-error rate in rollouts, oracle solve rate, null-agent rate, pass^k spread |
| Realism, when customer data exists | Divergence of the World's tool-call and entity distributions from the customer's logs; SDNist k-marginal on synthetic tables; gap between simulated and logged reply times |
| Transfer | Held-out real benchmark deltas after training |

No paper defines a standard cross-app workflow metric (note 04 §7), so the card is also a contribution. The card is a
**measurement first**. A family may later set floors on it (for example "coupling ≥ x"), but only after we measure
whether coupling predicts transfer (Part 2 plan, M13).

#### The App tiers and their priority

Note 04 §8.1 proposes four tiers behind one choke point. Each tier is an App kind with one contract: a tool surface, a
State Adapter, `identity(actor)`, `healthcheck`, `reset_hook`, snapshot and restore, and clock injection. All four
already fit `World._call_tool` and the State Adapter protocol (`world/apps/base.py:103`, `world/state/base.py`).

| Tier | What | Reset | Use | Crucible pieces that exist |
|---|---|---|---|---|
| **T1 native executable Apps** | SQLite or Postgres schema + generated CRUD + curated business tools (AppWorld, EnterpriseOps-Gym, AWM style) | DB copy, under 0.5 s in AppWorld | Bulk RL | `sqlite_app` with `tools_module`; `postgres_app`; `json_api`; `verify/sql_checks.py` (`SqlAssertion`, `Collateral`); agentenv server SDK `@tool`/`@reset_data`/`@add_data`/`@get_data` (map §1.7) |
| **T2 real OSS stand-ins** | Mattermost, Plane, Gitea, Stalwart or Mailpit + Radicale, Outline, Zammad, Twenty or ERPNext, unmodified in containers, with official MCP where present | Per-episode identity and row scoping (WorkArena 2403.07718), diff-and-revert (SCUBA 2509.26506), copy-on-write containers (WEBSERV 2510.16252) | Realism checks, CUA, held-out eval, sim-to-real checks of T1 | `mcp_server` App (built, unused); harbor and agentenv backends |
| **T3 LLM-simulated tools** | Long-tail read-only APIs, grounded on a worldbook (Trace2Env 2610.06100), cached (StableToolBench 2403.07714); writes mirrored into a T1 store | Cache + journal replay | Coverage only; **never a verifier input** | none |
| **T4 desktop, VM, browser** | Pixels over T1 or T2 state | `loadvm` about 1.1 s (DESIGN, lab); clean browser profile | CUA. The verifier reads T1 or T2 state, never pixels | `desktop`, `vm`, `browser` Apps (built, unused) |

**Why executable first.** AWM's matched ablation favours executable Worlds at every model size. The simulated version
lost up to 13.4 BFCLv3 points and 8.9 τ² pass@1 points (2602.10090). Simia's one win, OfficeBench RL 64.7 simulated
against 60.8 real (2511.01824), was later criticised as overfit (2604.17739). And when the component that produces
observations also grades, hacking follows (note 04 §4.2).

**Priority, adapted to the lead's no-detour rule.** Build the generic machinery first. Domain Apps then come out of
`world_synth` (Part 2 plan) as DomainSpecs, instead of being hand-built one by one.

| Order | Item | Why it comes here |
|---|---|---|
| W1 | **App kit for T1**: a generated CRUD layer + curated business-verb tools + policies as code, minted through the agentenv server SDK pattern (DEPEND), with column policies declared in the schema | The generated CRUD layer is the verifier's and State Adapter's view; the curated tools are the policy's view (EnterpriseOps-Gym and τ make this split, note 04 §4.3). Every later App reuses the kit. |
| W2 | **Clock bus, notifications and event DAG** (next subsection) | Without them, chat and calendar are just databases to poll. |
| W3 | **Org model and declarative people** | One person has the same identity in chat, mail, CRM and tickets. Generators must be able to emit "a vendor who disputes the invoice" as data. |
| W4 | **Office core as `world_synth` templates**: calendar, chat, docs or wiki, tickets, CRM (plus the existing mailbox) | These five cover most of EnterpriseOps-Gym, ClawsBench (2604.05172), OfficeBench (2407.19056) and Gaia2 Mobile (note 04 §8.2). CRM has the most enterprise demand: CRMArena-Pro (2505.18878) shows large headroom. |
| W5 | **T2 realism pair**: Mattermost (or Rocket.Chat) + Plane, plus Gitea | All have official or ready MCP. TheAgentCompany uses Rocket.Chat and Plane, and Harbor has its adapter (note 04 §3.1, §4.1). They check that T1 training transfers to real software. |
| W6 | **CUA path** through Harbor `computer-1` against T4 desktops showing T2 web UIs | See "Harbor and CUA" below. |
| W7 | Customer-data importer (redaction → synthesis → fidelity gate → T1) | See "Enterprise data ingest" below. |
| W8 | T3 long-tail tool simulator | Last, and only for read-only coverage. |

#### Clock, events and notifications

Crucible's clock (`world/clock.py`) has wall, step and sim modes, plus `at`, `after`, `every` and `on_tool` triggers with
nl, tool and permission actions. ARE / Gaia2 (2509.17158, 2602.11964) is the reference design. Gaia2's Time split is the
least-solved axis: GPT-5 (high) scores 0.0. The gaps (note 04 §5.2):

| Gap | Design | Schema or code touched |
|---|---|---|
| G1 Notification channel and policy | Arrivals (mail, chat, calendar reminders, ticket updates) reach the policy as notifications injected before its next step, at a verbosity level: LOW user only, MEDIUM mail/chat/calendar, HIGH all (ARE). Today they only sit in inboxes to be polled. | `World.step` (`world.py:875-916`); a `notifications` block in `crucible.world/v1` |
| G2 Event DAG and condition events | `depends_on(parent, delay)` ("vendor replies 2 h after the clerk's e-mail"); condition events over state | `core/schema/world.py:213-309`, `core/triggers.py` |
| G3 Time tools | `wait_for_notification(timeout)` jumps virtual time to the next event; `get_current_time()` | `world/clock.py`, a system App |
| G4 Timing in the verifier | A write must land inside a window relative to its causal parent (Gaia2), so deadline tasks ("reply within 1 business day") can be graded | new `TimingCheck` in the task contract (Part 2 plan §3.3) |
| G5 One time source for every App | T1 reads the World clock. T2 containers get libfaketime or an app-level "now". libfaketime does not reach static binaries such as Go built without cgo [inferred]. T4 guests get their clock set at restore. Snapshots fingerprint timestamps. | App contract `clock_injection` |
| G6 Business calendars | Working hours, weekends, holidays and time zones per person. Note 04 found no paper that does this well. | org model |
| G7 Time cost per action | Each tool may cost simulated time (Vending-Bench 2502.15840: 5 min to 5 h per tool, with overnight events) | `ToolSpec` gets an optional `duration` |

**Determinism checklist** (OSWorld-Verified's fixes for timing, fonts and dates, and AndroidWorld's pinned clock;
note 04 §2–3): fixed clock, fixed fonts and resolution, no network, pinned app versions, images pinned by digest.

#### People as data

**Today.** Responders are Python wiring. The `persona`, `grounded_on`, `policy_doc` and `transport` fields are
schema-only. The `actor` registry kind is registered but never resolved (checked).

**Design.**
1. **An `ActorSpec` the runtime reads.** It holds a kind, a profile, grounding records, the actor's own tools and a
   transport.
   - Resolve `actor` kinds from the registry at `World.start`.
   - Keep `World.attach` for Python-built actors.
   - Validate grounding references against the org model.
2. **Profiles sampled per person.** Fields: responsiveness, patience, whether they know the answer, typical delay,
   error rate and knowledge boundary.
   - LLM users are over-cooperative and uniform in style, which inflates agent success: a 451-person study (2603.11245).
   - A knowledge boundary cut simulator hallucination from 12.4% to 6.97% (ToolSandbox, 2408.04682).
   - Non-collaborative behaviours break tool agents (2509.23124).
   - Revealing a request in shards costs 39% on average (2505.06120).
3. **People bounded by tools and records.** Under τ² dual control (2506.07982), the user has its own tools and state. The
   tool-grounded telecom user had the lowest critical-error rate: 6%, against 13% for airline and 12% for retail.
4. **Audits that gate any World with LLM people:**
   - fabrication rate (`actors/fidelity.py`, exists);
   - a τ²-style critical-vs-benign error audit on a sample;
   - GAUGE rank agreement against the verifiable reward (2609.12191). GAUGE found 57.5% of "satisfied" conversations
     had failed the task.
5. **Simulator diversity in training, a different family at evaluation.**
   - LOGIGEN (2603.00540) trained against one low-temperature simulator and was evaluated against another. The policy
     rose and then collapsed.
   - AutoForge's τ²-Telecom score moved from 76.3 to 90.4 with only the user simulator changed (2512.22857).
   - Rotate simulator models and temperatures during training. Always evaluate against a simulator from a different
     family than training.
6. **Never a satisfaction judge as reward.** Every reward keeps a state bit, GAUGE's judge-free completion bit.

#### Enterprise data ingest with redaction

**Today.** The `Connector` protocol (discover, fetch, normalize, redact; `connectors/base.py`) ingests traces. Redaction
runs inside the tenant before storage. It removes credentials, maps e-mails, hosts and actor handles to HMAC pseudonyms
under a per-tenant secret, and is gated by planted canaries (`connectors/redact.py`, checked). **Connectors never
populate App state.** Seeding a World from an export is manual (map §1.3).

**Design** (note 04 §6.3 [inferred], note 02 Q5):

| Step | What | Tool or method | Gate |
|---|---|---|---|
| D1 Metadata first | Schemas, API specs (OpenAPI), policy docs, ticket and mail samples, logs with timestamps | `Connector.discover` | Human review of scope |
| D2 Redact | Structured and free-text PII and secrets. The same real ID always maps to the same synthetic persona, so relational structure survives. | Existing `connectors/redact.py` HMAC pseudonyms, plus Presidio recognizers for structured fields and custom IDs | Canary gate (exists). A human reviews the recognizer config. |
| D3 Synthesise | Tables: SDV HMA (tuned for about 5 tables and 1 level of depth), or TabularARGN (2501.12012), with a DP option (AIM 2201.12677). Text: LLM grounded on the synthetic records (EnterpriseBench 2510.27287), or Aug-PE (2403.01749) when it must be DP. Hidden latent traits per person and customer (CRMArena 2411.02305). | new `connectors/seed.py` producing a `SeedBundle` | SDNist-style fidelity and privacy metrics; referential integrity; every task feasible (AWM) |
| D4 Build Apps | Generated CRUD + curated tools + policies as code, from the schema | App kit (W1) | Tool tests (Part 2 plan, F1) |
| D5 Build people | From the org chart and role docs. Reply-time and cooperativeness distributions from the logs. | `ActorSpec` | People audits above |
| D6 Rebuild state from logs | Replay logged writes against the synthetic base, keyed by entity. This is missing today. | new | Re-solve check (Part 2 plan §3.5) |

**Hard rule.** Raw rows and raw text never enter a training World. Language models can regurgitate training sequences
verbatim (2012.07805), so any trace text that reaches a trained model can leak. This matches DESIGN's "nothing leaves
the building". No paper gives a complete trace-to-environment redaction protocol (note 02 gaps), so D2–D3 are design,
not literature.

#### Harbor and CUA integration

**What Harbor is** (note 04 §3.1, repo, Apache-2.0):
- A task is a directory: instruction, `task.toml`, an environment (Dockerfile, compose or Apptainer), a solution and a
  test that writes a reward.
- `[[environment.mcp_servers]]` declares MCP servers, with `${VAR}` templating for credentials.
- **Tools belong to the agent.** An environment offers tools only as MCP servers, skills, APIs or CLIs.
- About 35 sandbox providers; compose works on 17 of them.
- The `computer-1` agent runs Xvfb + XFCE + noVNC + Chromium inside the task sandbox through `exec`.
- Adapters include osworld, theagentcompany, crmarena, tau3-bench and gaia2.
- Harbor has no trainer. SkyRL's `harbor_generator.py` is the verified integration.

**Integration plan.**
1. **Export a World as a Harbor task.**
   - T1 and T2 Apps become MCP sidecars in the task's compose file. That is the only route by which Harbor agents can
     see Slack- or CRM-like tools [inferred from the docs].
   - The verifier runs Crucible's State-Adapter checks in Harbor's **separate verifier sandbox**. P23 holds: Crucible
     never trusts a reward file the policy's sandbox can write.
2. **Run the live lane on the harbor backend, not raw `docker run`.**
   - `guard/isolate` (policy uid, git scrub, firewall, quiesce, egress probe) runs only on the harbor and agentenv
     backends (map §1.4).
   - The local backend's quiesce is a no-op, so harness-audit B4 (survivors of the quiesce score 0) only applies when
     the World is isolated.
   - Moving the lane brings the tested isolation stack onto the live path.
3. **CUA.**
   - Use Harbor's `computer-1` against Crucible's desktop App, whose `computer` tool already covers the Anthropic and
     OSWorld action union.
   - Add three things: batched actions per call (OpenAI style), an accessibility-tree observation, and a
     `report_infeasible` terminal action.
   - Record HAR for browser Apps, so runs can be re-scored offline (WebArena-Verified, note 04 §2).
4. **CUA throughput is bound by environment utilisation, not VM count.** DART-GUI decoupled actors and raised env
   utilisation from 12.2% to 67.7% (2509.23866). So CUA RL needs the remote workers of A11.

#### MCP sidecars

- **Mount, don't write.** `common/mcp_server` mounts any stdio or HTTP MCP server with per-identity sessions. It needs no
  code, and no pack uses it yet (map §1.3).
- **Official MCP servers.** Mattermost (Agents plugin), Plane (`plane-mcp-server`, MIT), Gitea (`gitea-mcp`), GitLab CE
  (built-in), Outline (built-in `/mcp`, but BSL), Twenty (built-in, alpha) and Frappe (official) all ship one
  (note 04 §4.1).
- **Licensing** [inferred]. Run AGPL apps unmodified in containers and do all adaptation in an MCP sidecar. Avoid BSL
  Outline for hosted offers (Docmost is the AGPL alternative).
- **Shared services.** Give each episode a fresh identity (WorkArena's throwaway user) and scope its rows, or mark the
  World `serial` (exists).
- **Upstream churn.**
  - Daytona's public repo has been frozen since June 2026.
  - AppFlowy-Cloud and LocalStack OSS are archived.
  - Pin images by digest. Prefer apps with active OSS and an official MCP server.

### 2(c) Reward and verifier hardening

**The field's position** (note 03 §0). Verifiers are attack surfaces.

| Finding | Number | Source |
|---|---|---|
| Docker-verified tasks that accept a wrong patch | 28.5% of a SWE-bench Verified sample; 25.0% of an R2E-Gym sample | 2606.16062 |
| Previously passing patches rejected after hardening | 19.71% | SWE-ABS, 2603.00520 |
| Harness flaws across 10 benchmarks | 219 | BenchJack, 2605.12673 |
| Full scores with zero capability | 100% on SWE-bench Verified, SWE-bench Pro and Terminal-Bench, by an exploit agent | Berkeley [web] |
| Score drop under a strict harness | 87.1% to 73.0%; 63% of one model's resolutions retrieved the fix | Cursor [web] |

Crucible's debug lane already holds several defences: hidden tests in a fresh container, protected paths, history and
network rules at hack severity, filter-driver neutralisation, null-agent and oracle gates, commit-revert mutants and
Clopper–Pearson flake bounds. The additions below are ranked by cost and value. "Gate" names the Part 2 plan stage (its §3.2) where
each one runs.

| # | Defence | Today | Add | Evidence | Gate |
|---|---|---|---|---|---|
| V1 | **Side-effect negatives**: the work done plus an unrequested change | None. Mutants are commit reverts (`lanes/windows.py:686-713`). | Code: correct fix + a behaviour change in a neighbouring function; correct fix + a weakened visible test elsewhere. State Worlds: a duplicated write, an extra field, an edited distractor row. Each must fail. If one passes, add a collateral check (a regression subset or a count assertion). | Every false positive in 2610.09142 came from an unrequested change. AppWorld passed duplicated writes in two templates, and WorkArena passed extra fields in all 36 pairs. ABC O.g.2 (2507.02825). | F5, F6 |
| V2 | **Per-task hackability H(t)** | S12 hackers exist but are off in debug | K wrong solutions from a family other than the builder's, the suite's and the student's, 3 rounds with the failure logs fed back, applied in the real World. Any pass means fix before keeping. Attack transcripts become negatives. | 2606.16062 (audit cost $6.04 for 49 tasks). SWE-ABS (2603.00520). | F6 |
| V3 | **Gold-sanity gate on every generated test or check** | "Oracle 1, start 0" covers the task, not later additions | Every test the swarm or an augmenter adds must pass on the oracle and fail on the start state before any judge sees it. Record the defect rate per generator as a swarm-quality metric. | 61.9% of decisive augmentations failed that gate, and an LLM judge alone missed them (2606.16062). | F5 |
| V4 | **Harness exploit scan per LangProfile and per World recipe** | Python conftest, sitecustomize and `*.pth` are hack severity (harness audit §4.3) | Go `_test.go` with `TestMain` or `init()`; Node jest `setupFiles` or a vitest config override; Rust `build.rs` or a `#[cfg(test)]` module; Java surefire or `pom.xml`. Run null, random, prompt-injection, state-tampering and parser-overwrite agents on every new profile. Any nonzero score blocks it. | BenchJack's 8 flaw classes (2605.12673) [inferred mapping] | Profile CI |
| V5 | **Mutation-kill floor with a held-out mutant set** | `mutation_score` recorded | Kill 100% of crafted negatives and at least X% of generic mutants (X set by measurement). Keep a held-out mutant set that is never used during repair. | GateTruth (2608.12635): 72% of RTLLM v2.0 below a 95% floor, and a Goodhart effect on testbenches revised to pass it | F6 |
| V6 | **Isomorphic re-grade of suspicious passes** | none | Re-run a passing solution under renamed identifiers or permuted test data. Literal special-casing fails. | IPT (2604.15149): shortcuts were specific to RLVR-trained models | F6 sample, eval |
| V7 | **Cross-unit separation test for warm pools** | S10 reruns at concurrency 4 | A seeded run leaves a delayed effect (a background process, a file outside the workspace, a cache). The next pooled sandbox's score must not change. | 2608.14940 | Pool CI |
| V8 | **L1 rules on the live harness** | absent (§1.1) | A1 above | — | Training |
| V9 | **`report_task_defect` action** | none | Ends the episode with reward 0 (not negative) and a structured note. The flag rate per task becomes a framework_health detector. Watch that the student's flag rate on valid tasks stays near the frontier's. | ImpossibleBench's flag exit cut GPT-5's cheating from 54% to 9% and o3's from 49% to 12% (2510.20270) | Training, eval |
| V10 | **Impossible twins and honeypots, eval-only** | none | For 5–10% of training tasks, build a conflicting-check twin where any pass is a cheat. Plant a detectable decoy answer source in some eval Worlds. Plot both cheat rates against the RL step. **Never train on them.** | ImpossibleBench (2510.20270); HVE / HVTB (2605.20744, 2608.22103); BaitBench (2608.30724): 57.1% hack | Eval |
| V11 | **Closed-loop shortcut discovery** | `repath.scan` re-applies rules (321 attempts, 17 flips) | Each training interval, a different-family auditor reads passing and flagged rollouts and proposes new patterns. A pattern enters `path_rules` only after it holds on a frozen labelled set built from Crucible's own audited episodes. Then bump `RULES_VERSION` and repath. | Verification Horizon (2606.26300). Monitors trained on prompted hacks generalise poorly to training-time hacks (2604.23488). METR: patch the scorer rather than train against detected hacks [web]. | Training loop |
| V12 | **Never a CoT monitor in the reward** | none | Policy. If an action-level penalty is ever used, keep a second monitor that is never in the reward. | Optimising against a CoT monitor teaches obfuscated hacking (2503.11926). Monitorability is fragile (2507.11473). | Policy |
| V13 | **Inoculation framing only with an ablation** | none | Use it only for a known-exploitable environment that cannot be fixed at once. Record it on the task card. Repairing the World is preferred. | 2511.18397; inoculation works where synthetic documents do not (2609.14998) | Policy |
| V14 | **Swarm hygiene** | Independent review for tasks | Every swarm-written test, oracle or check passes an independent reviewer of another family before it enters the pool. | Hacks spread between agents; review reduces them (2610.12822) | F5 |

**A correction carried from note 03.** The Verification Horizon figure (hacked-resolved falling from 28.57% to 0.56%,
2606.26300, Table 3) measures a trajectory-level behaviour monitor during RL on top of static hardening. It does not
measure "verification out of the agent's reach" alone. The out-of-reach argument rests on ImpossibleBench's
hidden-test result, BenchJack's isolation class and Cursor's strict harness.

**State-based verifiers for non-coding Worlds.** These are needed by the generic task contract of the Part 2 plan (its §3.3):
1. **Column policies declared at synthesis time.** Each column is exempt, exact or semantic. Generated IDs and
   wall-clock columns are exempt by rule. ScaleEnv leaves both how columns are assigned and its fuzzy matcher
   unspecified (note 01 §1.9). Its tools stamp `datetime.now()`, and 531 of AWM's 1,000 scenarios seed rows with
   `datetime('now')`. Crucible pins `now` in `interop/openenv/awm.py`.
2. **A frozen clock** for every App (§2(b), G5).
3. **Answer checks for read-only intents** (`verify/answers.py` exists). In ScaleEnv a no-op passes "tell me my
   conversion rate" (note 01 §2.1).
4. **A whole-World collateral check.** Every change from start to end must be explained by the expected or allowed
   change set (AppWorld; `Collateral` in `verify/sql_checks.py:160`).
5. **Timing checks** with causal order and windows (Gaia2).
6. **An LLM judge only for free text,** always next to a state bit. Use validated judge items (`verify/judge_item.py`).
7. **Measure the rule vs judge question; do not assume it.** ScaleEnv found rule-based beat an LLM judge by 2 Avg@4
   points (2602.06820). AWM found a code-augmented judge beat code-only by about 9 BFCL points, because code-only fails
   on imperfect environments (2602.10090). The likely reconciliation is a deterministic check where state is clean and a
   judge only on imperfections (note 01 §3.4). Run all three modes on Crucible's gold sets (`factory/goldset.py`) and
   report false pass and false fail for each.

### 2(d) Trace observability

**Goal.** Check that failures happen for the intended reason, at scale, with off-the-shelf viewers, and without sending
data out of the building.

#### The OTel GenAI export

The OpenTelemetry GenAI semantic conventions live in `open-telemetry/semantic-conventions-genai`, all at status
Development [web]. Crucible already **imports** OTel GenAI spans (`connectors/otel_import.py`) and writes Weave-shaped
spans offline (`observe/weave_sync.py`). Add an **export**, as a view of Crucible's own store, never as the store.

| Crucible entity | OTel span or event | Key attributes |
|---|---|---|
| Episode (attempt) | `invoke_agent {gen_ai.agent.name}` span | episode, task, family, World, student, checkpoint, run ids; stop condition; reward |
| Model call | `chat` span | model, usage tokens, call id. Content is opt-in through `gen_ai.input.messages_ref` / `output.messages_ref` URIs into the store. |
| Tool call through `_call_tool` | `execute_tool` span | `gen_ai.tool.name`, call id, App, RBAC verdict, interceptor hits, probe tags (P27) |
| Simulated person's turn | `invoke_agent` child span | actor id, profile id |
| Clock event, notification | span event | event id, virtual time, `depends_on` |
| Gate verdict, grader, judge, path rule | `gen_ai.evaluation.result` event | `evaluation.name` = gate id (F6.G5, S10, path:history…); `score.value`; `score.label`; explanation |
| Snapshot | span event | `WorldRef` digest |

**Rules.**
- **Content stays opt-in.** Message content goes out only as `_ref` URIs pointing into Crucible's store.
- **Held-out redaction is "don't emit the ref".** Sealed and held-out episodes emit spans without content refs.
- **Use OpenInference span kinds too.** Set `openinference.span.kind` (AGENT, LLM, TOOL, EVALUATOR, GUARDRAIL…) so
  Arize Phoenix renders the spans natively [web]. Phoenix's licence is UNVERIFIED.
- **Self-hosted viewers only.** Phoenix or Langfuse act as viewers with no new code [web]. Langfuse stores yes/no
  decision models as probabilities, so thresholds can be chosen afterwards.

#### First wrong step

Step-level attribution is hard. On Who&When, the best automatic method finds the responsible agent 53.5% of the time
but the decisive step only 14.2% (2505.00212). On TRAIL, the best model localised issues 11% of the time (2505.08638).
Crucible's `first_wrong_step` is an LLM judgement in the trace-audit pipeline today.

1. **Counterfactual labels from snapshots** (AgenTracer, 2509.03312, which beat large proprietary models by up to 18.18%
   on Who&When).
   - For a failed episode, restore the `WorldRef` (or the M16 kernel snapshot) before candidate step t.
   - Swap in the oracle's next action, or a fixed frontier policy's continuation, and re-run a few times.
   - The earliest t at which success recovers is the decisive step.
   - Do this on a sample, and use it to **calibrate** the cheap judge and to train a small tracer.
   - On the debug lane, the existing Φ checkpoints (`factory/lanes/windows.py:2075-2097`) give the restore points. In
     state Worlds, World snapshots do (`world.py:1069-1159`).
2. **Allow sets of first wrong steps** with confidences (MP-Bench, 2603.25001), and score the judge on set overlap.
3. **Long traces:** an event-dependency graph plus a failure-propagation graph (DeFA, 2610.01256) for segmentation.
4. **A cheap outcome model as a Φ check.** Classifiers on intermediate behaviour predict outcomes at 89–97% accuracy
   (EarlyEval, 2609.02783). Use one as a baseline that Φ must beat, as DESIGN S2 already does with the judge.

#### Failure taxonomies

| Layer | Today | Add |
|---|---|---|
| Cause | Closed taxonomy with quoted evidence: skill_gap, spec_defect, unstated_format, world_bug, verifier_defect, budget, env_flake, abstain (`factory/cards.py:33`, `factory/audit.py:45-59`) | Keep it |
| Fault side | framework_health labels agent vs framework per detector | A mandatory fault side on every failure cluster: model, harness, World, grader or task spec ("Model or Harness?", 2607.28802, 41 modes, judge κ = 0.76). "Student failed on skill" and "the World made the skill unreachable" are never pooled. |
| Mode seed | Lane-specific `failure_skill` (`debug.py:717-755`) | MAST's 14 modes (2503.13657, κ = 0.88) and AgentDebug's modules (memory, reflection, planning, action, system; 2509.25370) as a generic seed. Each domain adds its own leaves as data. |
| Process quality | `lucky_pass` class in S9b | A lucky-pass rate per task over student passes: no reproduction before the fix, no test after the edit, repeated edit-revert cycles (AgentLens, 2605.12925: 10.7% lucky). |

#### Clustering at scale

Run the Clio / Docent pattern weekly as a framework_health input. Each episode gets a facet summary and an embedding.
Then cluster hierarchically, let an LLM name each cluster, and have a human review the top clusters (Clio 2412.13678;
Docent [web]).

The key view **contrasts student failures with frontier successes on the same task**. That is exactly the on-intent
question: what did the solver do that the student did not? HAL (2510.11977) used LLM-aided log inspection over 21,730
rollouts and found agents searching for the benchmark instead of solving it.

### 2(e) Evaluation methodology

| Topic | Protocol | Crucible today | Evidence |
|---|---|---|---|
| **pass@k and pass^k** | Report pass@1, pass@k and pass^k (all k of k pass, unbiased estimator) per task, averaged over tasks with n ≥ k | `bench/lh_metrics.py:94-129` | τ-bench (2406.12045) |
| **Task-clustered CIs** | Clustered standard errors or a case-clustered paired bootstrap. Paired differences for A/B. Power planning before spend. | `bench/compare.py`, `bench/power.py` | 2411.00640 |
| **Null and oracle agents** | Null (empty and "claims everything") must score 0; oracle must score 1; every release. | S5 gate (`longtask.py:262-298`) | ABC T.9 (2507.02825) |
| **Scripted positive controls** | Per lane: the oracle's action sequence replayed by a scripted agent, and a deliberately skill-less scripted agent. Show the metric separates them. | none | Validity contract (2609.01519) |
| **Cost per pass** | Dollars and tokens per solved task, per model, beside every score | `factory/tinker.py` summaries | — |
| **Judge-free completion bit** | The World's deterministic outcome reported next to any judge-derived score | partial (state grades) | GAUGE (2609.12191) |
| **Reliability** | Consistency across seeds (pass^k), robustness to prompt mutation (instruction rewrites), predictability (STATUS Brier) | pass^k, Brier | Rabanser (2602.16666); 2510.08996 |
| **Held-out families** | Transfer measured on families never trained on, not only on held-out instances | none (one family today) | ScaleEnv's OOD setup (2602.06820) |
| **Simulator family** | Evaluation user simulator from a different family than training | n/a yet | LOGIGEN (2603.00540) |
| **Contamination and sealing** | Date rule (post-cutoff only), disjoint owners, git-grep and 13-gram probes, hashed manifest, HMAC seeds | `benchaudit.py`, `benchseal.py`, `sealguard.py`. 10 sealed tasks, 4 exposed; accepted because the eval World is offline. | SWE-rebench's date rule (2505.20411) |
| **Item selection** | IRT to pick informative items once the benchmark grows | `bench/irt.py` (M11 only) | tinyBenchmarks (2402.14992), Fluid (2509.11106) |

**Judge calibration protocol.** It covers every trace judge: the hack auditor, the skill or on-intent judge, the
first-wrong-step judge and the honesty checks (note 03 §6.3; Husain & Saha [web]).
1. **One binary judge per failure mode.** Each has its own versioned rubric. Criteria drift is real (EvalGen,
   2404.12272).
2. **Store probabilities, not labels.** Ask for a probability or use logprobs.
3. **Thresholds by task-grouped CV.** Use GroupKFold by task or repo, because episodes of one task are correlated
   [inferred adaptation of Husain & Saha's stratified 5-fold].
   - Set precision and recall minimums per mode: hack detection favours recall, the skill gate favours precision.
   - Choose the highest-F1 threshold that meets both on the training folds. Judge it on the held-out folds.
   - Record infeasible splits. Husain & Saha saw 36/100 and 29/100 infeasible for two models and 0 for the best.
4. **Calibrate.** Draw a reliability diagram. Fit Platt scaling out of fold, and keep it only if held-out Brier
   improves (it did not always: 0.159 to 0.164 in one case [web]). Isotonic regression overfits at about 100 labels
   [inferred].
5. **Report against baselines.** Put F1 next to the always-positive baseline 2π/(1+π), which beat 5 of 21 models on
   R-Judge (2607.28685). Also report κ against humans and cost per 100 judgements. Draw a cost-F1 Pareto frontier.
6. **Test for constant output.** Score a blank or shuffled trace and a known-good/known-bad pair. Identical outputs mean
   the judge is broken. One judge gave the same score regardless of input in 9 of 37 settings (2610.27465). Make this
   the framework_health detector `judge_constant`.
7. **Test for shortcuts in the judge.**
   - Run the four tests of 2610.17747: input restriction, cue following, residual accuracy, aggregation.
   - Hold out task groups when estimating accuracy (2610.22243).
8. **Use PPI for rates.** Estimate S7 and hack rates with Prediction-Powered Inference (2301.09633): the judge on every
   rollout, humans on a random sample. The interval stays valid when the judge is biased.
9. **Cascade.** A cheap judge decides confident cases and escalates the rest to a stronger judge of a different family,
   or to the human queue, with a target agreement level (Trust or Escalate, 2407.18370).
10. **Re-query, don't just read.** Where a judge reads only a transcript, add an independent referee that re-queries the
    World: re-run the claimed tests, re-read the files (2608.03744). `traceaudit.honest_trace_facts` already does this.

---

## 3. Long-horizon RL training: literature addendum

*Being added.* A sixth research sweep is running. It covers:
- RLM and compaction inside RL training;
- credit assignment across turns;
- stabilisation of MoE and async RL: training/inference mismatch, entropy control, LoRA and bf16 precision;
- async defaults for a 35B MoE on few GPUs;
- agent rewards beyond pass/fail;
- alignment during agentic RL;
- early-warning metrics.

Its findings will be listed here as improvements, in the format of §2(a).

## 4. Smoke-test results

*Being added.* An end-to-end CPU smoke test of every Part 1 path is running:
- the full unit suite;
- World episodes with null and oracle checks;
- the World's simulated parts;
- the RLM harness and compaction;
- a tiny async RL loop with score centering and resume;
- trace audit, credit and reward;
- agentenv interop.

Results and any defects found will be listed here.

---

## 5. References (shared with the Part 2 plan)

Merged and deduplicated from notes 01–04. Every arXiv id was checked against arXiv by the note that cites it (on
2026-10-29). Flags are carried over. Each id appears once, under its main topic.

**Environment and tool-world synthesis**
- 2602.06820 ScaleEnv (ICML 2026; no code released)
- 2602.10090 Agent World Model (AWM)
- 2512.22857 AutoForge
- 2601.05808 EnvScaler
- 2509.13311 AgentScaler
- 2510.01179 Toucan
- 2601.08276 ToolACE-MCP / ACE-Router
- 2409.00920 ToolACE
- 2511.03773 DreamGym (headline numbers UNVERIFIED in note 04)
- 2511.01824 Simia
- 2508.08791 FTRL
- 2504.03601 APIGen-MT
- 2507.20534 Kimi K2
- 2510.10197 Environment Tuning
- 2511.19304 AutoEnv
- 2510.01051 GEM
- 2505.24760 Reasoning Gym
- 2409.15637 Synatra
- 2410.12952 BUTTON
- 2603.00540 LOGIGEN
- 2604.18292 Agent-World
- 2610.37648 SynthAPI
- 2609.33665 CompoWorld
- 2605.18703 EnvFactory
- 2609.05576 EnvCraft
- 2607.10891 SETA
- 2608.05466 RST
- 2610.04906 WorkForge
- 2610.14163 AutoEnvScaling
- 2610.16523 EAU
- 2606.19047 RODS
- 2610.21656 Data World
- 2610.18868 Tev
- 2609.27321 VHD-Play
- 2512.19682 GenEnv
- 2601.16443 Endless Terminals
- 2601.04126 InfiniteWeb
- 2604.17739 TRUSTEE
- 2403.07714 StableToolBench
- 2503.20527 MirrorAPI
- 2512.04535 GTM (numbers not verified)
- 2606.24597 Qwen-AgentWorld
- 2411.06559 WebDreamer
- 2510.14969 UI-Simulator
- 2609.04148 Terminal-Universe
- 2610.06100 Trace2Env
- 2606.12191 Agentic Environment Engineering survey
- 2610.26828 WildAWM
- 2608.20634 AgentMercury
- 2601.17722 EntWorld [search]
- Older and adjacent: 2509.08755 AgentGym-RL; 2509.17158 ARE; 2501.10893 Learn-by-Interact; 2503.07826 Magnet;
  2506.11045 RandomWorld; 2511.16043 Agent0
- ScaleEnv citers whose mechanisms were not read: 2610.33568, 2610.33817, 2610.33149, 2610.27781, 2610.23194,
  2610.19686, 2610.17953, 2610.16397, 2610.14774, 2610.14118, 2610.08048, 2609.36887, 2609.32750, 2609.08404, 2609.05837,
  2608.27260, 2608.06197, 2607.05943, 2606.09032, 2605.27209, 2604.05336
- UNVERIFIED names seen only in other papers' tables: TermiGen, Tmax, RLVE-Gym, EnvGen, a LongCat "EnvFactory"

**SWE and code task pipelines**
- 2504.21798 SWE-smith
- 2412.21139 SWE-Gym
- 2504.07164 R2E-Gym
- 2505.20411 SWE-rebench
- 2506.10954 SWE-Factory
- 2509.08724 SWE-Mirror
- 2502.13681 Repo2Run
- 2505.23419 SWE-bench-Live
- 2502.18449 SWE-RL
- 2512.18552 Self-play SWE-RL (SSR)
- 2510.19898 BugPilot
- 2509.16941 SWE-Bench Pro
- 2610.24206 SWE-Bench Pro V2
- 2609.08149 SWE-Bench Pro Verified
- 2410.06992 SWE-Bench+
- 2506.12286 SWE-Bench Illusion
- 2510.08996 Saving SWE-Bench
- 2601.11868 Terminal-Bench 2.0
- 2602.21193 Nemotron-Terminal (domain count **[uncertain]**)
- 2606.24855 OpenThoughts-Agent
- 2506.04178 OpenThoughts
- 2605.26494 MiniMax-M2 (scale **[uncertain]**)
- 2512.02556 DeepSeek-V3.2
- 2512.20856 Nemotron 3
- 2502.12115 SWE-Lancer
- 2604.20779 SWE-chat

**Diversity and open-endedness**
- 1504.04909 MAP-Elites
- 2310.13032 QDAIF
- 2402.16822 Rainbow Teaming
- 1901.01753 POET
- 2003.08536 Enhanced POET
- 2306.01711 OMNI
- 2405.15568 OMNI-EPIC
- 2309.16797 Promptbreeder
- 2402.13064 GLAN
- 2406.20094 Persona Hub
- 2212.10560 Self-Instruct
- 2304.12244 WizardLM / Evol-Instruct
- 2406.08464 Magpie
- 2308.07074 InsTag
- 2210.02410 Vendi Score
- 2505.20161 Prismatic Synthesis (G-Vendi)
- 2410.15226 Diversity of synthetic data
- 2303.09540 SemDeDup
- 2504.05228 NoveltyBench
- 2510.22954 Artificial Hivemind
- 2510.01171 Verbalized Sampling
- 2010.03934 Prioritized Level Replay
- 2012.02096 PAIRED
- 2203.01302 ACCEL
- 2502.07709 MAGELLAN
- 2411.01775 Eurekaverse

**Difficulty, curriculum and self-play**
- 2402.03300 DeepSeekMath (GRPO)
- 2503.14476 DAPO
- 2504.03380 Online Difficulty Filtering
- 2502.12272 Learning at the Frontier of Learnability
- 2501.12599 Kimi k1.5
- 2505.14970 SEC
- 2504.09710 DUMP
- 2504.05520 AdaRFT
- 2411.02337 WebRL
- 2505.03335 Absolute Zero
- 2508.05004 R-Zero (headline numbers differ from its tables **[uncertain]**)
- 2506.01716 Self-Challenging Agents (50 vs 200 annotated trajectories inconsistent in the paper **[uncertain]**)
- 2506.24119 SPIRAL
- 2505.17063 Synthetic Data RL
- 2506.08989 SwS
- 2508.03682 Self-Questioning LMs
- 2509.07414 Language Self-Play
- 2510.24695 AgentFrontier
- 2511.10395 AgentEvolver
- 2507.13266 QuestA
- 2402.05808 R³ reverse curriculum
- 2505.24864 ProRL
- 2412.02674 Mind the Gap
- 2506.14205 AgentSynth
- 2506.10055 TaskCraft (affiliation UNVERIFIED; 32K vs 36K tasks **[uncertain]**)
- 2507.15061 WebShaper (scores **[uncertain]**)
- 2507.02592 WebSailor
- 2505.22648 WebDancer
- 2509.24726 Socratic-Zero
- 2510.24684 SPICE
- 2510.18821 Search Self-play
- 2506.03136 CURE
- 2406.13542 AutoIF
- 2509.22624 SPARK
- 2511.14460 Agent-R1 (a training framework, not a curriculum method)

**Traces to tasks and privacy**
- 2412.09605 AgentTrek
- 2502.11357 Explorer
- 2412.19723 OS-Genesis
- 2410.02907 NNetNav
- 2403.08140 BAGEL
- 2502.06776 InSTA
- 2506.03533 Go-Browse
- 2308.06259 Instruction Backtranslation
- 2412.13678 Clio
- 2403.01749 Aug-PE
- 2012.07805 Extracting Training Data from LMs
- 2606.23654 EnterpriseClawBench
- 2201.12677 AIM
- 2501.12012 TabularARGN; 2508.06647 its DP variant
- 2210.14348 DP fine-tuning for text

**Verifier testing and harness integrity**
- 2507.02825 ABC checklist
- 2411.12990 BetterBench
- 2605.12673 BenchJack / Agent-Eval Checklist
- 2606.16062 Auditing Reward Hackability
- 2603.00520 SWE-ABS
- 2506.09289 UTBoost
- 2503.15223 PatchDiff
- 2610.09142 Finding Side-Effect Loopholes
- 2608.12635 GateTruth
- 2501.12862 Mutation-guided test generation (Meta)
- 2604.15149 Isomorphic Perturbation Testing
- 2604.21579 Metamorphic testing for memorisation
- 2505.24098 HardTests
- 2603.12698 EvolveCoder
- 2207.10397 CodeT
- 2502.01619 UTGen
- 2502.13820 Scoring Verifiers
- 2505.22203 Rule- vs model-based math verifiers
- 2507.08794 One Token to Fool LLM-as-a-Judge
- 2609.06780 Shortcutting the Fix
- 2608.14940 Outcome finality and cross-unit separation
- 2609.11028 BenchShield
- 2607.26819 (git-history leak mechanism, cited via note 03)

**Intent and construct validity**
- 2511.04703 Construct-validity review
- 2607.22368 Protocol validity / HackDetect
- 2609.01519 Construct-validity contract
- 2609.12191 GAUGE
- 2606.26300 The Verification Horizon
- 2605.26079 ABA
- 1803.02324 Annotation artifacts (partial-input baselines)
- 2610.17747 Auditing failure attribution
- 2610.22243 Shortcut sensitivity
- 2605.12925 AgentLens
- 2607.28802 Model or Harness?

**Trace analysis and attribution**
- 2503.13657 MAST
- 2505.08638 TRAIL
- 2509.25370 AgentDebug
- 2505.00212 Who&When
- 2509.03312 AgenTracer
- 2610.01256 DeFA
- 2608.29646 DuoTrace
- 2603.25001 MP-Bench
- 2502.10325 AgentPRM
- 2601.12294 ToolPRMBench
- 2609.02783 EarlyEval
- 2510.11977 HAL
- 2602.16666 Towards a Science of AI Agent Reliability

**Reward hacking**
- 2209.13085 Defining reward hacking
- 2310.09144 Goodhart's Law in RL
- 2210.10760 Reward-model over-optimisation scaling laws
- 2511.18397 Natural emergent misalignment from reward hacking
- 2609.14998 Shallow Beliefs
- 2510.05024, 2510.04340 Inoculation prompting
- 2508.17511 School of Reward Hacks
- 2406.10162 Sycophancy to Subterfuge
- 2510.20270 ImpossibleBench
- 2511.21654 EvilGenie
- 2605.21384 SpecBench
- 2605.20744 Hack-Verifiable Environments; 2608.22103 HVTB
- 2608.30724 BaitBench
- 2503.11926 CoT monitoring and obfuscation
- 2507.11473 CoT monitorability
- 2505.05410 Reasoning models don't always say what they think
- 2610.03055 hacktrace
- 2604.23488 Trace-and-Amplify
- 2606.09711 PRIME
- 2610.12822 Transmissible hacking
- 2605.02964 Reward Hacking Benchmark

**Self-improving loops**
- 2408.08435 ADAS
- 2410.04444 Gödel Agent
- 2310.02304 STOP
- 2505.22954 Darwin Gödel Machine
- 2506.13131 AlphaEvolve
- 2605.13821 AEvo (the abstract gives "26 relative improvement" with no unit)
- 2609.24972 RRSI
- 2606.17546 SEAGym
- 2606.04455 Meta-Agent Challenge
- 2610.11464 Who Verifies the Verifier?
- 2610.27465 Recoverability and Verifier Strength

**Judges, calibration and statistics**
- 1706.04599 Calibration of modern nets
- 2301.09633 Prediction-Powered Inference
- 2407.18370 Trust or Escalate
- 2411.00640 Adding Error Bars to Evals
- 2402.14992 tinyBenchmarks
- 2509.11106 Fluid Benchmarking
- 2410.10934 Agent-as-a-Judge
- 2504.08942 AgentRewardBench
- 2406.18403 JUDGE-BENCH
- 2404.12272 EvalGen
- 2609.12439 TraceJudgeBench
- 2607.28685 Safety, or Just Capability? (trivial-policy F1)
- 2608.03744 Agents Catching Agents
- 2507.17746 Rubrics as Rewards

**Enterprise, workplace and tool benchmarks**
- 2412.14161 TheAgentCompany
- 2510.19286 TheMCPCompany
- 2403.07718 WorkArena; 2407.05291 WorkArena++
- 2411.02305 CRMArena; 2505.18878 CRMArena-Pro
- 2509.26506 SCUBA
- 2603.13594 EnterpriseOps-Gym
- 2510.27287 EnterpriseBench (Fujitsu)
- 2602.16179 EnterpriseBench Corecraft (scale UNVERIFIED)
- 2609.37658 EnterpriseBench (strategic reasoning)
- 2407.18901 AppWorld
- 2406.12045 τ-bench; 2506.07982 τ²-bench
- 2408.04682 ToolSandbox
- 2407.19056 OfficeBench
- 2405.00823 WorkBench
- 2604.05172 ClawsBench
- 2604.10866 OccuBench
- 2510.04374 GDPval
- 2509.26490 VitaBench
- 2510.25726 Toolathlon
- 2509.24002 MCPMark
- 2508.14704 MCP-Universe
- 2508.20453 MCP-Bench
- 2508.01780 LiveMCPBench
- 2602.00933 MCP-Atlas
- 2506.07672 MCPWorld
- 2508.06600 BrowseComp-Plus
- ITBench, OdysseyBench: ids UNVERIFIED

**Web, OS and CUA**
- 2307.13854 WebArena
- 2408.06327 VisualAgentBench / WebArena-Lite
- 2401.13649 VisualWebArena
- 2510.16252 WEBSERV
- 2412.05467 BrowserGym
- 2504.11543 REAL
- 2504.01382 Online-Mind2Web
- 2601.02439 WebGym
- 2602.14093 GUI-GENESIS [search]
- 2603.10505 Recreated websites
- 2604.10988 WebForge
- 2404.07972 OSWorld
- 2606.29537 OSWorld 2.0
- 2510.24563 OSWorld-MCP
- 2409.08264 Windows Agent Arena
- 2405.14573 AndroidWorld
- 2501.12326 UI-TARS
- 2310.11441 Set-of-Mark
- 2508.14040 ComputerRL
- 2509.02544 UI-TARS-2
- 2505.16282 ARPO
- 2509.23866 DART-GUI
- 2505.23762 ZeroGUI
- 2508.09123 OpenCUA

**People and time**
- 2603.11245 Mind the Sim2Real Gap
- 2601.17087 Lost in Simulation
- 2509.23124 Non-Collaborative Users
- 2510.05444 SimulatorArena
- 2601.08118 MirrorBench
- 2505.06120 LLMs Get Lost in Multi-Turn
- 2507.22034 UserBench
- 2509.19736 UserRL
- 2312.03664 Concordia
- 2310.11667 SOTOPIA
- 2602.11964 Gaia2
- 2502.15840 Vending-Bench
- 2402.05733 TimeArena

**Non-arXiv sources.** Each is used as the notes used it.
- **Infrastructure and observability:**
  - Harbor (github.com/harbor-framework/harbor; Apache-2.0) and SkyRL's Harbor integration;
  - the OTel GenAI semantic conventions (`open-telemetry/semantic-conventions-genai`, status Development);
  - OpenInference; Arize Phoenix (licence UNVERIFIED); Langfuse and its decision models; W&B Weave;
  - LangSmith and Braintrust [secondary]; Inspect AI log viewer; Docent (ingestion format UNVERIFIED).
- **Field reports:**
  - METR reward-hacking report (2025-06-05);
  - METR MALT (monitor AUROCs UNVERIFIED);
  - Berkeley RDI benchmark-exploit post;
  - Cursor reward-hacking audit (2026-06-25);
  - OpenAI's SWE-bench Verified retirement (2026-02-23) [secondary];
  - Husain & Saha, "How To Use Decision Models for Evals" (2026-10-20);
  - Claude system card hack rates [secondary, UNVERIFIED].
- **Data and privacy tools:** Microsoft Presidio; SDV (HMASynthesizer); NIST CRC / SDNist; libfaketime.
- **Sandboxes:** E2B (resume latency UNVERIFIED); Daytona (public repo unmaintained since June 2026); Modal snapshots;
  Runloop; trycua/cua; Playwright MCP; Steel.
- **MCP and API tooling:** FastMCP `from_openapi`; openapi-mcp-generator; Google MCP Toolbox for Databases; PostgREST;
  Frappe MCP.
- **Open-source app stand-ins:** Mattermost, Rocket.Chat, Zulip, Outline, AFFiNE, Docmost, AppFlowy, Mailpit, Stalwart,
  GreenMail, Poste.io, MailHog, Radicale, Baikal, Plane, OpenProject, Redmine, Taiga, Gitea, Forgejo, GitLab CE, Twenty,
  SuiteCRM, EspoCRM, ERPNext/Frappe, Odoo, Zammad, GLPI, Nextcloud, oCIS, ONLYOFFICE, Collabora, BookStack, Wiki.js,
  moto, Prism, WireMock, Mockoon.
