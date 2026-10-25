# 13 The Tinker swarm: no Claude, the whole framework, and a swarm that debugs itself

**Status (2026-10-23).**
- **Built:** the framework code, with tests.
- **Smoke:** $1.07 of Tinker, inside the $5 cap.
- **Not done yet:** no task-generation pilot. The $300 pilot starts after this branch and the difficulty branch are
  merged.

**The user's decisions:**
- "all next steps should be done by my swarm intelligence not claude use tinker api".
- On the builder: "all three [DeepSeek, GLM, Kimi] but for different things".
- "exploit my complete framework, credit assignment and whatever we did, and keep debugging the framework as well on
  how to make the swarm intelligence itself better".

## 1. What changed, in one table

| piece | where | what it does |
|---|---|---|
| the `swarm` preset | `factory/models.py` (`SWARM_ROLES`, `STRICT_PRESETS`) | every role on Tinker, builders over DeepSeek / GLM / Kimi, checkers on Nemotron / gpt-oss, fairness on Qwen3.5-397B / Nemotron-Ultra; no Claude Code role (refused) |
| artifact provenance | `factory/provenance.py` | every artifact carries the families that produced it; a checker refuses an input its own family produced |
| builder Φ | `factory/builder_potential.py`, `factory/sessions.py` (`session.potential`) | the checkpoint sidecar grades a builder's worktree out of band against the static half of the gate its artifact faces |
| session plumbing | `factory/sessions.py` | window from the model's context, a window floor, a mid-session state USD cap, settle-on-budget, the producer's family on every session record, a journal hook |
| journal, blame, credit, diagnosis | `factory/swarm_journal.py`, `scripts/swarm.py` | per-role and per-model yield and cost, blame by provenance, per-turn builder credit, ranked bottlenecks and proposals (never applied) |
| the lanes on the swarm | `factory/swarm_prompts.py`; `write-prompts` in `lh_debug.py`, `lh_kernel_debug.py`, `lh_kernel_debug_migrate.py`; `lh_swarm.py --preset swarm`; `lh_debug_target.py --preset swarm` | the S7 report is written by a `prompt_author` session, decided by the lane's own lint and ledger, read once by the reviewer; the swarm-built windows run on the preset and are journaled |
| difficulty interface | `factory/difficulty_hooks.py` | a typed knob action in, a gated revised task out (the difficulty subsystem implements the policy) |

## 2. Provenance: the family rule on artifacts, not only roles

`ModelBook.family_problems` checks roles: no checker on a builder's family, never Qwen on the Qwen student, hackers over
two families. With three builder families that is necessary but not enough, because what matters is what each checker
actually reads.

- **Producer stamp.** A state whose role is a builder adds `Producer(role, family, model, state)` to the provenance of
  its inputs. Provenance is transitive: hidden tests written from a plan carry the planner's family too. Any other state
  passes its inputs' provenance through, so a gate's verdict carries what it judged.
- **Reads.** Provenance follows `inputs[].from`. It also follows declared service reads, for handlers that read another
  state's results without a port: the swarm machine's `prompt`, `gates`, `second` and `hack` read the build
  (`generators.swarm.SERVICE_READS`).
- **Two checks.**
  - `static_problems(machine, families, reads)` checks a machine before any run.
  - `ProvenanceLedger.check` checks again at every activation, using the family of the client that will actually run
    (`family_of_client`), so a later override cannot slip past.
- **V2 by family** (strict presets): the oracle builder and the first verifier writer are two families.
- **Fairness** (strict presets): no fairness solver is a builder's family. The legacy maps keep their own rules, so a
  qwen-built P5 task can still be judged by Qwen3.5-397B for fairness; the strict rule applies to new runs only.

## 3. Builder sessions are episodes: Φ, credit, honesty

A builder session is a long-horizon episode on the same M16 harness the student uses (`actors/rlm`: ledger, context
compaction, snapshots, stall detection). A session state may declare `potential`. The pool then attaches the checkpoint
sidecar (`factory/checkpoint.py`) to the session, so it is graded at commits, every K calls, at each compaction and at
the end. Nothing measured travels back to the builder.

| potential | builder | Φ counts a milestone when | it mirrors |
|---|---|---|---|
| `prompt_draft` | prompt writer | its requirement line exists in `prompt.json`, has at least 40 characters, and passes the lane's lint | S7 |
| `hidden_tests` | verifier writer | `crucible_hidden/test_<mid>.py` passes `test_problems` | S5's static half |
| `oracle_commits` | oracle builder | commits 1..k exist in plan order, subjects `mK: ...` | S4's structural half |

- **The executed halves still decide afterwards** (V6): hidden tests on A and on the oracle, the World, and S6, S5 and
  S8.
- **Defaults:** `oracle_commits` and `hidden_tests` are built by the pool itself (`builder_potential.DEFAULTS`), so the
  swarm-built-window machine gets them with no wiring. The P5 end-to-end test now checks Φ = 1 for both builders.
- **Honesty:** the builder keeps `STATUS.json` like a student, and the sidecar scores its claims (Brier) against Φ's
  milestones.
- **Per-turn credit** (`swarm_journal.builder_credit`):
  - **Group:** one builder step, every attempt of `(run, state)`, retries and loop-backs alike.
  - **Reward:** 1 or 0 by the first verdict on the attempt's artifact, or None when nothing judged it.
  - **Φ:** per World step, from `progress.jsonl`.
  - **Credit:** through `credit.compose.turn_credit`: centred within the episode, bounded by the outcome advantage, and
    it never flips sign.

  The rows say which builder turns made progress that the gates later confirmed. This is the data to post-train a
  builder on, if we ever do.

## 4. The lanes run on the swarm

The S7 reports of the rebreak lanes were the one step a person (the Claude builder) did by hand. Now
`write-prompts --preset swarm` runs one machine per window. The machine is in `factory/swarm_prompts.py`:

1. **`write`**: a `prompt_author` session in a private clone of A', in an offline sandbox, with potential
   `prompt_draft`. The hand-off holds the maintainers' notes per milestone, the strings the lint refuses, and the
   calibration block (training windows only). `settle_on_budget` is on: a writer that runs out of turns still hands its
   file to the gate.
2. **`lint`**: deterministic: the lane's own check (`author_debug`, `kernel_debug.author`, `migration_lint`). Its
   findings go back to `write`, then the window is `dropped`.
3. **`review`**: the reviewer, a family nothing built, provenance-checked. It blocks once; after that its verdict is
   recorded next to the prompt but no longer gates (V6).

The lane's `author` step then runs unchanged. Training and held-out windows use separate roots and journals (V9).
`lh_swarm.py --preset swarm` fills `SwarmRunner`'s slots from the preset; every slot goes through the run's guarded
clients, so there is one ledger. `lh_debug_target.py --preset swarm` uses the swarm's auditor, judge and fairness
solvers.

**Not model steps in these lanes:**
- The mutants of the rebreak lanes are the real single-commit reverts.
- The hidden tests are the real fixes' tests, so there is no verifier writer.

Those roles act only in the swarm-built windows (`generators/swarm.py`), which now run on the preset too.

## 5. The difficulty interface

`factory/difficulty_hooks.py` is the contract for the difficulty subsystem, which is built separately: online p̂
filtering, a p ≈ 0.5 controller, a cheap difficulty predictor and checkpoint-conditioned starts.

- **`KnobAction`:** `ease` K1-K6 (the easing doc) or `harden` H1-H4, an optional milestone, a reason, and the evidence
  (p̂ and its interval, Φ per milestone).
- **The three sides:**
  - `Controller.propose`: the difficulty side picks the action.
  - `Reviser.revise`: the swarm side makes the change; a model may draft it.
  - `Gates.run`: the lane's executed gates decide.
- **`revise_and_gate`:** the one call a generator makes. A refused revision leaves the original task untouched and is
  journaled with blame on the reviser.
- **Held-out tasks are refused** (`HeldOutRefused`): a benchmark task's difficulty is measured, not tuned.

## 6. What the smoke found, and the fixes (each with a test)

The smoke (`scripts/swarm.py smoke`) made two kinds of runs:
- one JSON call per distinct model: 7 models, $0.0024, all valid JSON;
- one `write-prompts` machine per builder family, on a tiny repository, with a long README to push the context past the
  compaction threshold.

| # | found | cause | fix | test |
|---|---|---|---|---|
| 1 | **Kimi** "finished" without writing anything (`missing declared output 'prompt'`) | after a few turns Kimi answers in its own tool-call section (`<\|tool_calls_section_begin\|>...`) instead of the text protocol; the codec read it as a final answer | `codec.parse_text`: Hermes blocks, else Kimi's native section; an unterminated call is an unparseable call the World answers with an error | `test_text_protocol_reads_kimi_native_calls` |
| 2 | **Every session** booked $0 to the machine's ledger (V10 caps were blind to sessions) | the mid-session state cap wrapped each call in its own `SpendScope` and never re-booked it outward | `StateBudgetClient` re-books each call's dollars into the enclosing scope | `test_state_budget_client_stops_mid_session` |
| 3 | A session that compacted showed **no compaction point** in its Φ series | `SessionPool` called `run_episode` without the sidecar probe (`runtime.harness.native.probed`), so `before_action` never ran | the pool probes the policy when a sidecar is on | smoke: Kimi `compaction_turns [9]` after, `[]` before |
| 4 | **DeepSeek** ended `compaction_failed` at a 10K window (rebuilt 4,894 tokens > threshold 6,750 with the summary request) | a window too small for its own hand-off; the RLM refuses a compaction it cannot fit | `sessions.window_floor`: the window is raised to what the rebuilt context needs (TASK, system ~3K, two summaries, one reply) | `test_context_window_fits_the_model` |
| 5 | **Kimi** wrote a correct report by turn 9 (Φ = 1), never called finish, and was thrown away at the turn limit | a budget stop failed the state even with the artifact captured | `session.settle_on_budget`: whatever was captured goes to the gate (V6) | `test_a_budget_stop_still_goes_to_the_gate` |
| 6 | The hand-off was refused (`no original spec`) | the prompt machine passed its brief as `params.brief`; a hand-off always needs `params.spec` | the brief is the run's spec | the end-to-end test |
| 7 | (design) a harness stop was blamed on the builder | `compaction_failed` is the harness's configuration | the journal tags it `harness`; blame goes to `harness`, never to the role | `diagnose` on the smoke's failing journal |

**Final smoke** (after the fixes): all three families wrote a lint-clean report, with Φ = 1 and Brier 0.0025.
- **DeepSeek:** compacted at turn 7. 46K tokens, $0.024.
- **GLM-5.3:** never compacted, because it printed only the first 3,000 characters of the long file. 23K tokens, $0.039.
- **Kimi:** compacted at turn 9 (two attempts). 127K tokens, $0.134.

So compaction ran live on two of the three families. GLM's sessions stay small: it reads economically.

**Cost per role.** Kimi was the most expensive builder per session: 5.7× DeepSeek's dollars for the same artifact, and
it uses the most turns. If the pilot confirms this, the diagnosis proposes moving the prompt writer to DeepSeek or GLM.
That proposal stays within the family rule, because the prompt writer may share the oracle builder's family; only
checkers may not.

## 7. Self-diagnosis: `scripts/swarm.py diagnose`

`diagnose --root <journal> [--md]` produces the following:
- **Per-role and per-model tables:**
  - sessions and retries, USD and tokens;
  - stops, and infra versus task failures;
  - mean Φ at the end, the gate pass rate of the role's artifacts, and blamed failures;
  - accepted tasks per dollar.
- **Ranked bottlenecks:** blamed failures × the role's mean session cost, i.e. the dollars spent on work a gate later
  refused. Each comes with its top finding kinds (specifics removed) and example runs.
- **Proposals:** kind `prompt` | `tool` | `harness` | `role`, each `{"proposal": true, "applied": false}`. A proposal
  never edits a gate or a machine. Examples:
  - a static lint failure: give the writer the lint as a tool;
  - a high Φ with failing gates: the executed half fails, so give the session a way to run that check;
  - a pass rate below 30% over three or more artifacts: try another family, within the provenance rule.
- **Builder credit:** attempts that failed below Φ = 1, with their trace directories.

On the smoke's failing journal it ranked the Kimi budget stop first, with the turn-limit proposal. It put the DeepSeek
`compaction_failed` under `harness` with the window proposal, and listed the DeepSeek attempt that ended at Φ = 0.

## 8. Before the pilot ($300 approved)

- **Merge:** this branch, then the difficulty branch, then wire its controller into `revise_and_gate`.
- **Measure per role:** run the pilot with `--cap` per lane and run `diagnose` after each batch. Watch Kimi's cost per
  accepted report; move the writer if the numbers say so.
- **Not run live yet:**
  - the swarm-built-window machine (`lh_swarm.py --preset swarm`), whose hacker is an agent episode in the solver's
    World;
  - Nemotron-Ultra as a reviewer on real prompts;
  - gpt-oss-120b as an agentic hacker.

  The first pilot batch should be small.
- **DeltaAI grading of builder code:** the Apptainer grader there runs as the user (`trusted: false`). That is fine for
  gates on our own patches, but not for grading model-written code at scale. See `10` §8.3.

## 9. The first pilot (2026-10-23, $300 approved): no Claude wrote any task content

**What ran:** `scripts/lh_difficulty_pilot.py`, in three steps:
- the controller (`difficulty_bridge.ControlController`) picks one knob per measured training task;
- the swarm's `prompt_author` writes the variant's report (Kimi-K2.6, reviewed by Nemotron-3-Ultra), and executed gates
  decide (`revise_debug.DebugGates`);
- the 35B student re-measures it at 300 calls, k = 8 (`lh_debug_target.py --preset swarm`). A fair task also gets
  DeepSeek-V4.1-Flash k = 2, and the swarm's auditor (Nemotron-3-Ultra) and judge (gpt-oss-120b) label the failures.

Held-out tasks were never touched. Each task got at most two revisions.

**Step 1: the difficulty controller on measured debug training tasks**

| task (parent: student 300-call rate) | round 1 (knob: result) | round 2 (knob: result) | verdict |
|---|---|---|---|
| python-black-ae5830b7 (0/8; 6/8 budget stops) | C1, m01+m02 pre-applied: **1/8**, 0 budget stops, m03 7/8, m04 2/8; DeepSeek 1/2 | — | **kept (train)**: rate 0.125, on-intent 0.86, a frontier solver passes |
| go-mvdan-sh-04056a74 (0/8; 4/8 budget stops) | K1 (drop m01): **1/8**, 4/8 budget stops; DeepSeek 1/4 | — | not kept: fair, but on-intent share 0.43 < 0.7 (4 of 7 failures are budget stops) |
| go-go-toml-a02114e3 (0/8; m04 given m03 = 0.67) | C1, m01+m02 pre-applied: 0/8 (free rewrite lost a fact, m03 0/8); restored: 0/8, m03 8/8, m04 0/8 | — | too hard: one test fails in every episode (12 §7) |
| go-mvdan-sh-dc951aeb (0/8) | K2: 0/8, m03 1/8 | K3: 0/8, m03 0/8 | too hard (m03 and m04 never move) |
| go-mvdan-sh-5e983709 (0/8) | K2: 0/8, 7/8 budget stops | K1 (drop m02): 0/8, 1/8 budget stops, m04 0/8 | too hard (m04 0/24 over all three measurements) |

- **Accepted training tasks: 1.** It is the first task the framework kept, and the first written end to end by the
  swarm: the controller's knob, the swarm's report, the executed gates.
- **Spend:** Step 1 cost **$129.7** of Tinker, about $130 per accepted task:

  | item | cost |
  |---|---|
  | student episodes (64 at 300 calls) | $101.6 |
  | DeepSeek fairness (8 episodes) | $15.4 |
  | auditor Nemotron-3-Ultra | $7.6 |
  | judge gpt-oss-120b | $1.5 |
  | report writing (Kimi sessions and Nemotron reviews) | $3.5 |

**What the pilot taught the controller** (12 §7):
- **Budget-bound tasks get a shorter horizon.** C1 and K1 removed every budget stop on black (6/8 to 0/8) and 6 of 7
  on 5e983709.
- **The checkpoint estimate is a prior, not a prediction.** It predicted go-toml at 0.5 and black at 0.5; it measured
  0 and 0.125.
- **The frame knobs moved nothing on mvdan-sh's interacting faults.** K2 (announce the structure) and K3 (point to the
  area) left the never-solved milestones at 0 to 1 in 8. The next knobs there would be K6 on the stuck milestone or a
  C1 that keeps it.

### 9.1 mvdan-sh-04056a74's K1 variant

The student passed it 1/8. DeepSeek passed neither of its first 2 attempts; it did pass the parent after the
install-rule regrade. Two fresh attempts were run separately (`data/targeting/swarm-pilot/r3f`; the phase's identity
guard refuses changing k on an existing run), with the result 1/2. So a frontier solver passes it (1/4 in all), and it is fair. Its failures are 4 budget stops and 3 skill
gaps, so the on-intent share (0.43) misses 0.7 either way. It is not kept.

**Step 2: kernel tasks by the swarm, for our GPUs** (no Tinker spent, no GPU used)
- **Mining** with the hardware-reachability filter: llama.cpp kept 60 changes (42 bug, 18 speed-up) and dropped 442.
  Of the 60, 20 are used by no stored record; 11 of those are after the cut.
- **Building:** the 4 groups they form all fail synthesis ("fewer than two faults revert together"): each group's
  reverts conflict at its B.
- **The other sources have nothing new.** HeCBench has no new commits since 2026-10-13, so the CUDA migration family
  has no new supply. ik_llama.cpp has no second held-out owner (10 §3.1).
- **New tasks: 0.** No GPU gates or student runs were needed.
- **What it needs:** the kernel lane's harness (`test-backend-ops`) is llama.cpp's own. The other cloned CUDA
  repositories (flash-attention, flashinfer, cutlass, cccl, cuCollections, TransformerEngine, CTranslate2, faiss,
  xformers, vllm) need a harness per repository before the swarm can mine them. That is a framework task; it is
  recorded here as the main blocker for more GPU tasks.

### 9.2 What `diagnose` and the runs found, and the fixes (each with a test)

| finding | fix |
|---|---|
| Every `write-prompts` session stopped at the shared Tinker meter's 1-USD default, whatever `--cap` said | `sampler.ensure_cap`, called by every lane's `write-prompts` and the pilot |
| A free rewrite of the report lost a fact the hidden tests need (go-toml m03: 6/8 to 0/8) | knob scope: frame knobs keep every requirement verbatim; K3 appends an area; K6 rewrites one milestone |
| K1 picked a fault the student fixed 8/8 | K1 only drops a fault at rate <= 0.7 |
| Every task was told "checkpoint start first" | a root milestone's rate is no conditional signal; above-band rates are not in band |
| Budget-bound tasks were "not a difficulty problem", with no action | `budget_bound` carries C1/K1 (the horizon shrinks, the budget never moves) |
| A revision refused because its writer hit the meter was blamed on the reviser | an infra row, no blame (V8) |
| Every report finding was one kind, `m#` | findings group by what they say |
| A missing candidate raised inside the phase's gather and killed the episodes in flight | unknown ids are refused before any episode |
| Report before mine read the kept black variant as off_intent 0.00 | `unmined`, never off_intent; the source names the preset's real judge and auditor |
| A refused `git clone` made the install verdict say "the World was not offline" | the verdict names the attempted fetch |

**Yield:**
- Report writing accepted 7 of 10 machine runs, about 2 per dollar. The 3 that wrote nothing all hit the meter cap
  (fixed).
- The reviewer blocked two leaky rewrites (named internal functions). With the scope rule, the requirement texts are
  the parent's measured ones.

**Commits:** listed in the pilot report to the operator. Data is in `data/targeting/swarm-pilot/` (`actions.jsonl`,
`revisions.jsonl`, and `r2`, `r3` and `r3f` per round) and in `data/debug/swarm/train/journal`.

## 10. Pilot round 2 (2026-10-23): diagnose first, then the knob

### 10.1 A: the four debug tasks

**go-toml-a02114e3 (its C1 variant).**
- `disclosure_check` on the parent: every reverted fix has a milestone.
- **The failing test is a skill gap, not a spec defect.** Applying a student patch reproduced it:
  `TestMarshalTextMarshalerStringKindOmitempty` prints `s = 'enc:'` for an empty named string. The student decides
  emptiness on MarshalText's output, and the report states the opposite rule in one clause.
- **K6, first attempt:** the swarm's rewrite quoted an "observed" output the program does not print. It claimed
  `a = "hello"`; the program prints `a = 'hello'`.
- **K6, second attempt,** with the executed-repro rule below: its repro failed at the oracle (S7r `fails_at_oracle`), so
  the gate refused it.
- **Verdict:** too hard. Two rounds are used (C1, K6).

**mvdan-sh-dc951aeb.**
- **m04 is tied to m03 by a shared test.** m04's own test set holds the whole `TestPrintBinaryNextLine`, which fails
  whenever m03 does. So m04 cannot pass before m03, and m03 is the blocker: 1 pass in 24 episodes.
- **K6 on m03:** the writer used its turns verifying the claim in the sandbox and wrote no report (twice; the second
  time with 120 turns).
- **Verdict:** too hard.

**mvdan-sh-5e983709.**
- **m04 fails on one case.** `TestPrintFiles/bash/#127`, the trailing-backslash case, failed in every one of 24
  episodes, also when m03 passed. The report states the behaviour (idempotent printing of a trailing backslash).
- **K6 on m04:** the writer wrote no report (twice).
- **Verdict:** too hard.

**mvdan-sh-04056a74 (its K1 variant).**
- **C1, pre-applying m02,** cleared every gate, and the student's patches then fixed both remaining faults in 4 of 8
  episodes.
- **Every one of those 4 lost the full pass on `c_regress`.** The regression test `TestParseRecoverErrors/#10` fails at
  the new start: m02's real fix breaks it, and only a later fix repairs it. So it was never pass-to-pass for this
  variant; it asked for something the report does not state.
- **After the repair** (it leaves the regression check, recorded), the World gates pass again, and the 8 stored
  patches regraded offline give **4/8**.
- **Fresh re-measurement** (r5, k = 8): student **3/8**, DeepSeek 2/2.
- **Verdict: not kept, off_intent.** 4 of the 5 failures are budget stops at 300 calls, so the on-intent share is 0.20.
  Even with two faults, this mvdan-sh variant takes the student about 300 calls. Two rounds are used (K1, C1).

### 10.2 B: GPU tasks

| source | result |
|---|---|
| 1. 5090 rerun (the kernel-grader worker) and the GH200 run (3440735–38) | **not finished**: the 5090 is running the worker's `pipeline3` (ik held-out and the near misses); the GH200 jobs are still queued (estimated start 2026-10-24 13:14 CT) |
| 2. llama.cpp at HEAD f5027fb21 with the hardware filter, re-grouped (`build --subsets`: every connected pair and triple) | **1 new record**, `kd-bug-llamacpp-12fa0914` (2 faults: FLASH_ATTN_EXT and ARGSORT/FLASH_ATTN_EXT). It passed the CPU gates and the hardware reach (2/2). The report is the swarm's (Kimi), rewritten once after the new cause lint; lint is clean. GPU gates queued on DeltaAI (job 3443829, 1 GPU, 3 h, `bjhg-dtai-gh`, own directory `crucible-pilot2`) |
| ik_llama.cpp at HEAD 0cd49c59, re-grouped | **0**: its only new group does not revert together |
| 3. CUDA 12→13 migrations from more repositories | **0**: HeCBench has no commits since 2026-10-13. A GitHub commit search (CUDA 13, after 2026-05-01) found no suite with HeCBench's Makefile-per-program layout. The migration lane's builder reads that layout only, so another repository (for example NVIDIA's CUDA samples, which use CMake) needs a layout adapter first |
| 4. a swarm-written harness for a second CUDA repository | **not started**. It needs a test-driver role, a GPU reference and a tolerance per op, and the lane must take a harness other than `test-backend-ops`. Estimated at 2-3 days of framework work before the swarm can mine such a repository |

**`kd-bug-llamacpp-12fa0914` on the 5090** (run after the kernel-grader worker's rerun ended, own directory
`~/crucible-lh-k-p2`, about 1 GPU-hour): **not gradable on sm_120.**
- **m1:** the full sweep found 75 of 7,284 cases failing at the start, but none entered the pool. At the start, every
  pool case and every sanitizer check passes.
- **m2** (the large-size overflow) failed no case.
- **Next:** the same task is queued on the GH200 (96 GB, job 3443829), where m2's sizes may fit. Results are in
  `data/kernel_debug/gpu_results_5090_p2`.

**GPU tasks passing every CPU and GPU gate: still 5, all training.**

| task | source |
|---|---|
| `kd-bug-llamacpp-1b8b8fb1`, `-ca21f3a1`, `-efe7d4b1` | earlier rounds |
| `km-cuda13-hecbench-591f311-4`, `-ba3a30a-3` | earlier rounds |

**Pending GPU:**
- 1 new training task, `kd-bug-llamacpp-12fa0914`;
- the worker's rescues: near misses cef66d4a and a45699aa (training), its migration fixes, and 6 held-out ik tasks (000b1a1e, 1ef4f2f7, 35d83401, 44b81230, 5b8c07ff, 89213aa6) in its `pipeline3`;
- the 30-task GH200 run.

**Student difficulty runs:**
- No kernel task has been through one.
- The kernel lane's episode World (`kernel_debug_world`) is a CPU-only Docker container. Its GPU grading
  (`GpuBackend`) runs on a CUDA host out of band, and `lh_debug_target` drives the debug lane only.
- Measuring the student needs:
  - episodes run on the GPU host (World and grader side by side);
  - the checkpoint sidecar's grades sent to the GPU backend;
  - a kernel targeting stage.
- **Estimate:** 1-2 days of framework work. Each episode would then cost about one GPU grade per checkpoint (up to
  15) on top of the student's Tinker tokens.
- **The agent cannot run CUDA in its World.** Today it can only compile and run the CPU harness. A GPU in the World
  would be more realistic, but it needs the sandbox rules for an untrusted agent on a GPU host.

### 10.3 What round 2 fixed (each with a test)

| finding | fix |
|---|---|
| K6 quoted an "observed" output the program does not print | K6's sharpened requirement must carry a repro whose S7r verdict for that milestone is `verified` |
| A pre-applied fix broke a regression test, so every agent lost its full pass | a checkpoint start's pass-to-pass tests are graded at the new start; a failing one leaves the regression check, recorded |
| The first swarm kernel report diagnosed the faults ("incorrectly includes out-of-bounds values", "integer overflow in ...") and passed lint and review | kernel lint `CAUSE`; the writer brief forbids diagnosing |
| A `--redo` after a framework change crashed (`MachineError`: the journal belongs to another machine) | a stale run is moved aside and written fresh |
| A revision under an old id was handed its old `prompt.json` back (the run, its session state and its hand-off branch were resumed) | a new hand-off under the same id is stale as well; all three are moved aside together (sessions live under `sessions/runs/<id>`) |
| Two K6 writers ran out of turns verifying their claims | K4/K6 sessions get 120 turns and $2.5 |
| The greedy kernel groups use each change once, so a group with one conflicting fault lost its other faults too | `build --subsets` (`kdm.connected_subsets`) |
| A knob the diagnosis picks, not the controller's ranking | `revise --override task=KNOB[:milestone]@why`, journaled; C1 takes an explicit prerequisite-closed set |

## 11. The training monitor: general failures, before the baseline and the RL run (2026-10-25)

`scripts/train_monitor.py` (doc 11 §4.4) separates **general** failures (a behaviour seen in >= 3 tasks over >= 2
languages or lanes) from **task-specific** ones (a task's milestone its failures keep missing), tracks them over training
steps, and turns the persistent general ones into framework proposals and swarm intents. It reads stored episodes only;
it never grades, rewards or gates.

**Validation on the data we have** (no new episodes): the 150- and 300-call targeting runs, the swarm pilot's rounds and
the GPU smoke; `--policy Qwen3.6-35B-A3B`, training split only (235 student episodes, 3 masked; Go 178, Python 48,
Rust 8, CUDA 1). The co-authors' runs are on their machines and were not read.

| general mode | student episodes | prevalence (95% CI) | tasks | languages |
|---|---|---|---|---|
| overclaims_status | 94/232 | 0.41 (0.34–0.47) | 18 | go, python, rust |
| stalls_after_partial_fix | 81/209 failed | 0.39 (0.32–0.46) | 17 | go, python |
| slow_first_edit | 68/209 failed | 0.33 (0.27–0.39) | 15 | go, python, rust |
| breaks_regressions | 70/232 | 0.30 (0.25–0.36) | 14 | go, python, rust |
| budget_flat | 43/209 failed | 0.21 (0.16–0.27) | 13 | cuda, go, python, rust |
| lost_after_compaction | 27/173 compacted | 0.16 (0.11–0.22) | 12 | go, python, rust |
| env_misuse | 25/232 | 0.11 (0.07–0.15) | 16 | go, python, rust |
| claims_unverified | 11/232 | 0.05 (0.03–0.08) | 9 | go, python, rust |

- **Recovers what we already knew:**
  - the stall after the first fault (`stalls_after_partial_fix`, 0.39);
  - overclaiming (`overclaims_status`, 0.41);
  - budget stops (47% of the student's episodes end on the limit; 21% of its failures do so with Φ flat for the last
    30%);
  - the install false positive. Re-pathed under the current rules (228 student and 50 DeepSeek episodes),
    `path:install` failed 6 times for the student and 5 for DeepSeek under the stored rules, and 1 and 0 under the
    current ones.
- **Task-specific, reported apart:**
  - `loops_failing_command` (3 episodes, 2 tasks);
  - `symptom_only` (1 of the 30 judged failures, one task);
  - each task's stuck milestones, for example m04 of mvdan-sh-5e983709 and black-ae5830b7, m03 and m04 of
    mvdan-sh-dc951aeb, m02 of go-toml-a02114e3.
- **By model (training split):** the frontier models share most modes. Qwen3.5-397B stalls after a partial fix in 0.59
  of its failures and DeepSeek in 0.39, so the stall is partly the tasks' shape (late faults are hard) and partly the
  agent. The student stands out on `slow_first_edit` (0.33 against 0.22 and 0.17). On `breaks_regressions` it sits at
  0.30, against 0.36 for Qwen3.5-397B and 0.12 for DeepSeek. DeepSeek's `env_misuse` (0.46) is mostly not installs (5
  stored install fails in 50 episodes).
- **The judge pass:** 36 student training failures sampled across tasks, judged by `openai/gpt-oss-120b` (never Qwen).
  30 were labelled and 6 rejected (a quote not in the trace, or a label outside the closed lists), for $0.045. 16 of the
  labelled already had a deterministic cause (budget), which a judge never overrides; the other 14 came out 7 skill_gap
  and 7 spec_defect.
- **No trend yet:** these runs are arms of one measurement, not training steps, so the monitor gives no
  learned/regressing verdict and no trend alert. The first training run's `steps.jsonl` turns them on.

**Proposals it makes today.** None is applied. Each needs an A/B on training tasks where the mode falls and the reward
does not.

| general mode | proposals |
|---|---|
| stall after a partial fix | checkpoint starts before the milestone that failures keep missing; an "interacting faults" swarm intent |
| overclaims | a higher honesty weight; a clearer STATUS.json line |
| slow first edit | a fast code-search affordance; a "localize fast" intent |
| regressions | a regression term in Φ; a one-command full-suite runner |
| lost after compaction | carry STATUS.json and the milestone vector into the compaction hand-off |

`scripts/swarm.py diagnose --monitor <report.json>` now lists the general modes and their intent candidates for the
builders.

