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
