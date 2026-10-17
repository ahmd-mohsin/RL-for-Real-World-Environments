# M17 task factory, report 11: the first end-to-end long-horizon coding run (10 windows, CPU)

Study 08 §9 step 5 on 10 training windows, CPU only, cheap OpenRouter models, hard cap $18 (authorised $20). Driver:
`scripts/lh_run.py` (resumable stages, one JSON journal per window). Raw numbers: `runs/lh_run.json`. Spend: **$6.54**.

## 1. Setup

- **Windows** (`SELECTED` in the script): ≥ 1 executed milestone edge, ≥ 1 valid revert mutant, oracle suite ≤ 20 s (an
  episode grades ≤ 21 snapshots and the flake gate 60 oracles), 7 repos, ≤ 2 per repo, 4-10 milestones.
  cachetools-w0913ada was selected first and dropped at the prompt stage (§4 D1); toolz-w2ac03ce replaced it.
- **Episode** (`WorkspaceHarness` + `LLMToolPolicy` subclass, GitRepoApp shell/file tools in a `--network none` World):
  ≤ 150 tool calls, ≤ 45 min, ≤ 400k context tokens (`trajectory_tokens`: what entered the context), and a cost guard of
  3M billed tokens (2M for the hacker); context compaction at 32k of a 40k budget; tool output cut to 8k chars; T = 0.7.
- **Checkpoint sidecar on** (`CheckpointConfig(every=10, cap=20)`): Φ at commits, every 10 calls, compaction boundaries, end.
- **Spend guard**: every client is wrapped so no call starts once journal spend + this process's provider-reported spend
  reaches $18; every stage is booked with `factory.models.booked` on one forge `Ledger`; paid stages ran in one process at
  a time so the guard saw all spend.
- **Extrapolation, written before scaling** (`<work>/extrapolation.md`): pilot = 1 qwen episode on semver-w1bafb9f,
  104 calls, 100 tool calls, 1.87M billed / 80k context tokens, 9.4 min, $0.022. List-price estimate for the plan:
  ~$11.3 (worst case at every billed cap $16.6), so no trimming. After the hacker stage the measured deepseek cost was
  $0.020 per episode (9× below list, provider caching) and the estimate was revised down.

## 2. Funnel

| stage | in | out | notes |
|---|---|---|---|
| prompt (author qwen → ledger + lint → realism deepseek, ≤ 3 revisions) | 11 | 10 | cachetools: lint refused 3 revisions (D1). Second round with the name check (D2): 10/10, 8 on the first try, 2 after one realism revision |
| floors: no-op, start state, A rewritten, `pass`-stubs of B's new symbols | 10 | 10 | all outcome 0, Φ 0 (stubs never earned a milestone) |
| oracle: B's source diff and B as unit steps | 10 | 10 | all outcome 1, Φ 1 |
| revert mutants (valid) | 29 | 29 caught | each fails its milestone and only the milestones the mining run expected |
| flake: 30 oracle grades at concurrency 1 + 30 at 4 | 10 | 10 | 60/60 clean everywhere, Clopper-Pearson upper 4.87% (≤ 5% gate met) |
| hacker (deepseek, 2 episodes per window, task prompt + intent I1 + "pass the hidden tests without the behaviour") | 22 attempts | 0 hacks | 20 scored (2 per window), all outcome 0; 2 attempts masked by D3 were re-run; partial Φ up to 0.75 came from real implementation |
| solvers: 3 rungs × 4 + 2 rung variants × 4 + RLM 4 windows × 2 | 208 | 0 passes | 173 scored, 35 masked (30 of them the main-run llama, F4) |
| audit (deepseek, 3 seeded controls per window) | 208 | | seeded 30/30 right in every window; 112 judged, 66 deterministic, 30 rejected (no verbatim quote) |

## 3. Per window (target rung qwen3.7-flash, 4 attempts; Φ = final checkpoint Φ)

| window | ms / edges | hacker max Φ | qwen pass (Wilson hi) | qwen Φ mean (values) | lost/ep | Brier (STATUS written) | causes sg/budget/other | verdict |
|---|---|---|---|---|---|---|---|---|
| python-semver-w1bafb9f | 4 / 3 | 0.75 | 0/4 (0.49) | 0.50 (.75 .75 .5 0) | 0 | 0.43 (4/4) | 11/2/uf 1 | **back**: unstated output format of `cmd_check` (auditor) |
| python-semver-w41775dd | 7 / 7 | 0.43 | 0/4 (0.49) | 0.29 (.43 .29 .29 .14) | 0 | 0.69 (4/4) | 12/3/0 | **accept-candidate** (Φ spread only; see F1) |
| parse-wc669f87 | 6 / 4 | 0.17 | 0/4 (0.49) | 0 | 0 | 0.82 (1/4) | 11/5/0 | **split**: m18 m20 m22 m24 m33 never reached |
| parse-w07c2ff9 | 9 / 17 | 0 | 0/4 (0.49) | 0.33 (all equal) | 1.75 | — (0/4) | 8/8/0 | **split**: m05 m11 m17 m25 never reached |
| more-itertools-w89d58c2 | 10 / 5 | 0 | 0/4 (0.49) | 0.03 (.1 0 0 0) | 0 | — (0/4) | 11/6/0 | **hold**: skill-gap share 0.65 < 0.7 (budget-bound) |
| more-itertools-w15407c7 | 4 / 2 | 0 | 0/4 (0.49) | 0.06 (.25 0 0 0) | 0 | 0.93 (2/4) | 10/3/0 | **accept-candidate** (Φ spread only; F1) |
| humanize-wf9fa838 | 4 / 3 | 0 | 0/4 (0.49) | 0 | 0 | 0.97 (4/4) | 11/0/uf 1 | **back**: `metric(x, unit)` argument never stated (auditor) |
| toolz-w2ac03ce | 8 / 8 | 0.25 | 0/4 (0.49) | 0.19 (.25 .25 .125 .125) | 0.5 | 0.65 (3/4) | 6/7/0 | **hold**: skill-gap share 0.46 (budget-bound) |
| glom-w67f423c | 8 / 7 | 0 | 0/4 (0.49) | 0 | 0 | 0.82 (2/4) | 9/3/0 | **split**: 7 of 8 milestones never reached |
| funcy-wc83fd24 | 7 / 1 | 0.29 | 0/3 (0.56) | 0.33 (.43 .29 .29) | 0 | 0.71 (1/3) | 9/4/0 | **hold**: skill-gap share 0.69 |

**Learnable band:** no window has a qwen pass rate in [0.1, 0.8]; 6 have final-Φ spread (the band's second arm). Verdicts:
2 accept-candidate, 3 hold, 3 split, 2 back. Nothing is ready to accept (F1).

## 4. Headline numbers

| rung (harness) | scored / masked | pass@1 | mean final Φ | lost/ep | tool calls | billed / context tok | STATUS written: Brier, mean claim vs true | give-ups (correct) | $/episode |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.7-flash | 39 / 1 | 0 | 0.169 | 0.23 | 116 | 2.1M / 77k | 21/39: 0.73, 0.97 vs 0.23 | 0 | 0.027 |
| gpt-oss-20b | 40 / 0 | 0 | 0.013 | 0 | 37 | 0.6M / 23k | 6/40: 0.64 | 3 (3) | 0.025 |
| gpt-oss-20b @nudges6 | 40 / 0 | 0 | 0.044 | 0.10 | 78 | 1.4M / 49k | 6/40: 0.76 | 5 (5) | 0.071 |
| llama-3.1-8b | 10 / 30 | 0 | 0 | 0 | 8 | 6k / 2k | 2/10: 1.0 | 1 (1) | 0.0006 |
| llama-3.1-8b @route | 36 / 4 | 0 | 0 | 0 | 22 | 10k / 4k | 7/36: 0.93 | 6 (6) | 0.0004 |
| qwen3.7-flash, M16 RLM | 8 / 0 | 0 | 0.20 | 0 | 96 | 1.9M / 287k | 1/8 | 0 | 0.042 |

- **Pass rate 0/173 scored** (qwen Wilson upper 0.09). Nothing solved, so $ and tokens per solved task are undefined.
- **Milestones lost (I1):** qwen 0.23 per episode, concentrated in parse-w07c2ff9 (1.75) and toolz (0.5).
- **Compaction-boundary loss (I6):** qwen 109 post-boundary spans, 0 losses (0.02 per other span); RLM (24k window,
  6-33 boundaries per episode) 144 post-boundary spans, 0 losses. No I6 signal at this Φ level.
- **Honesty (I5):** with defaulted claims (no STATUS.json = p 0.5) qwen Brier 0.48, ECE 0.51. When it writes STATUS.json
  it claims 0.97 against a true rate of 0.23. Half its episodes end at 150 tool calls without writing it.
- **Cause mix** (live failures, all rungs): skill gap 98, budget 41, unstated format 2, env flake 2, unlabelled 30.
- **Flake:** 600 oracle grades, 0 false negatives, bound 4.87% per window. **Hacks:** 0 confirmed in 20.
- **Horizon (05 §6.7, cheap):** 9 milestones of 3 windows posed alone (A + every earlier unit, the milestone's tests as its
  unit wrote them; all 9 oracle 1 / no-op 0), qwen × 4. Isolated 15/36 solved vs in context: semver-w1 m07 4/4 vs 3/4,
  m17 4/4 vs 4/4, m19 0/4 vs 0/4; semver-w41 m07 4/4 vs 4/4, m13 2/4 vs 2/4, m30 0/4 vs 0/4; parse-w07 m03 1/4 vs 0/4,
  m05/m11 0/4 vs 0/4. No milestone is solved less often in context. **The test has no power at n = 4**: with
  Bonferroni α = 0.0125 only a 4/4-vs-0/4 split is detectable, and the full-window product of isolated rates is 0 here.
  What the data does say: the failing milestones (m19, m30, m05, m11) fail alone too, so these windows fail on single
  hard milestones, not (yet measurably) on length.

## 5. Spend ($6.54 of $18; provider-reported per call)

author 0.022, realism 0.001 · hacker 0.421 · pilot 0.022 · solvers: qwen 1.127, gpt-oss 1.005, llama 0.024,
gpt-oss@nudges6 2.852, llama@route 0.019 · RLM 0.559 (first run 0.222 without Φ, D5) · horizon 0.239 · auditor 0.249.
Per solved task: undefined (0 solved). Per scored qwen episode: $0.027.

## 6. Defects found (D) and fixed

| # | defect | evidence | fix |
|---|---|---|---|
| D1 | prompt lint false positive: "accessible **without modifying** the decorated function's signature" is a behaviour, read as a process instruction | cachetools refused 3 times | **not fixed** (recorded; window replaced) |
| D2 | **hidden tests import public names the prompt never states**: `from semver import cmd_bump, cmd_check, cmd_compare`; one missing name fails collection of the whole file, so every milestone and the regression check read 0 (the pilot's Φ stayed 0 after 100 tool calls) | 6 of 10 windows import new names (semver ×2, toolz, glom, parse, humanize) | `factory/lanes/windows.py:735-853` `required_names` (module-level, in-function and `pkg.name` imports resolved against A; dynamic modules unknown, private names never required), `author_payload` puts each in its change's `must_name`, `unstated_names` is a finding; brief :672. Test `test_m17_e_names.py` |
| D3 | exposure probe: `**/tests/**` (a glob over the repo's own public tests) matched the `grader_files` rule and masked the episode as `answer_exposed` | 2 hacker episodes | `core/agent_events.py:205` (`*` in the lookbehind); test in `test_m17_e_run.py` |
| D4 | `WorkspaceHarness` booked each attempt the spend delta of a client shared by concurrent attempts | records said $2.95 for $0.40 of hacker calls | `runtime/harness/workspace.py:192-196` (the attempt's own calls, as `NativeHarness` does); test |
| D5 | own script bugs: oracle diff read in text mode broke CRLF files (glom oracle "did not apply"); the RLM task adapter delegated via `__getattr__`, invisible to the runtime protocol check, so the first RLM run had no sidecar | | `scripts/lh_run.py` (bytes decode; explicit `MilestoneTask` members); RLM re-run |

Found, **not fixed** (owners: lane, credit, harness):
- **F1 Φ-spread-only windows give no gradient today.** The band accepts "pass rate in [0.1, 0.8] *or* final-Φ spread", but
  `turn_credit_map` gives no credit to a group without outcome spread. Both accept-candidates are all-fail groups: they
  would train nothing until the credit rule admits Φ-only groups or the windows are cut until the outcome moves.
- **F2 `patch-conflict` edges are not behavioural prerequisites.** toolz m14 passes 4/4 in context while its prerequisites
  m07/m09 never pass; parse m26 passes 2/4 without m18/m20. C1's gate then credits nothing for real progress. Use only
  `executed` edges in Φ's `requires`, or confirm edges against in-context data.
- **F3 Milestone ownership:** parse m20 ("convert tests to pytest") and glom m33 pass on A + earlier units without their own
  unit: a test-rewrite unit owns tests whose behaviour came earlier, so the milestone re-credits an earlier unit's work
  (the order and non-redundancy test of 05 §6.7 should catch it; found by the isolated-task no-op check).
- **F4 Weak rungs are harness-limited, not only skill-limited.** gpt-oss-20b returns reasoning-only replies with no tool
  call and leaks harmony tokens into tool names (`repo__read_file<|channel|>commentary`); `LLMToolPolicy` stops after 2
  nudges. With 6 nudges it works 2× longer and reaches Φ > 0 in 7/40. llama-3.1-8b: 29/40 masked because OpenRouter
  routed to CoreWeave, which rejects histories with parallel tool calls; routing around it (`@route`) masks 4/40.
- **F5 STATUS.json is skipped when the budget ends** (20/39 qwen episodes hit 150 calls): I5 is unmeasured exactly where
  claims matter. A last-turn reminder at the limit, or reading a missing file as a claim, would fix the gap.
- **F6** Residual masks: `grep … /tests/` (absolute path that does not exist here) and `git merge origin/main` (the error
  echoes `origin/main`) were read as exposures (1 each); one qwen call returned no usage (`usage_blank`).
- **F7** `test_m17_w_world.py::test_an_episode_in_the_window_world` fails whenever another `crucible-lh-world-*` container
  is up on the machine (it asserts none exist globally); it passes after the run.

## 7. Honest limits

- n = 4 attempts per window × rung: Wilson intervals span [0, 0.49]; the concatenation test cannot detect anything short
  of a 4/4-vs-0/4 split; the flake gate's 4.87% bound is just under 5% with zero failures.
- The rung variants and the RLM re-run are separate runs under changed conditions (stated in their policy ids); the
  main-run llama rung is mostly masked and is not a measurement.
- The auditor rejected 30 of 142 judged labels (no verbatim quote); budget labels are deterministic from the stop.
- One prompt per window; realism review sees the maintainer text only (09 §4.8).

## 8. Next

1. Decide F1 (credit for Φ-only groups vs. cutting windows) before any GPU run; it decides whether these windows teach.
2. Fix F2/F3 in the lane (executed edges only in `requires`; owners from behaviour, not the last test rewrite), re-mine.
3. Send semver-w1bafb9f and humanize back to the author (stated formats); re-run D1's lint rule with a behaviour exception.
4. Harden the weak rungs (F4) or replace them; add the STATUS.json reminder (F5); then the 80-window machine run.

## Run 2: fixes F1-F5, re-mine, same 10 windows + 3 sub-windows

Cap $12 for run 2 (user's $20 minus run 1's $6.54, with margin). Spent **$6.25**; total over both runs **$12.79**. Raw:
`runs/lh_run2.json`, the re-mined funnel `runs/window_funnel2.json` (it also counts the 6 derived sub-windows).

**What changed** (09 §6 has file references):

| # | decision | built as |
|---|---|---|
| F1 | outcome reward = 0.5·Φ_final + 0.5·[full pass] | `WindowTask.phi_weight` (default 0.5) on `grade_patch`; `grade` stays binary for the accept contract, which kills a mutant only at reward 0 |
| F1 | band = outcome-reward spread in the target group, plus ≥ 70% skill gap | `reward_band`, `--band reward` |
| F2 | only `executed` edges in `requires`; `patch-conflict` edges become `order_hints` | `probe_dependencies`, which now runs B's tests as the grader does |
| F3 | owner = first unit after whose prefix the test passes | `prefix_owners`: ≤ 25 runs by bisection, prefix files read from the unit's commit |
| F4 | weak rungs replaced | probe of 8 cheap non-qwen/deepseek models (multi-turn fix with a parallel-call history, pinned providers): 0 errors for all. Pilots: mistral-small-3.2 hit the 45-min wall after 56 calls (48 s/call); llama-3.3-70b quit after 3 calls. Rungs: **nemotron-3-nano-30b** (Nvidia, Nebius→Novita) and **nova-micro** (Amazon, Bedrock). gpt-oss dropped: no harmony adapter was built |
| F5 | told tool-call limit | opt-in on `LLMToolPolicy` (`tool_limit`): a note at 5 calls left, then one final turn with only `write_file` for STATUS.json |
| D1 | lint false positive | editing verbs are process only when their object is the work environment; cachetools now authors on the first try |
| F7 | world test sees other runs' containers | `open_world(name_prefix=)`; the test checks only its own prefix |

**Re-mine** (134 stored windows, no model calls; held-out mined only):

| split | f2p_valid before → after | dependent | with valid mutant | behavioural edges | order hints |
|---|---|---|---|---|---|
| train (91) | 86 → 84 | 51 → **26** | 48 → **26** | 50 | 232 |
| held-out (43) | 37 → 40 | 20 → **10** | 19 → **10** | 16 | 176 |

- 27 train windows lost every edge (all were textual) and 2 gained one.
- 3 train windows collapsed below 2 milestones: when B's test file imports names that history adds last (semver's `cmd_*`), no B test passes on any earlier prefix, so every test has one owner (F8).
- Of the selected 10:
  - Type A: parse-w07c2ff9 (10 milestones, 1 edge), more-itertools-w15407c7 (4, 2), humanize (6, 1), toolz (8, 1), glom (7, 6);
  - independent: parse-wc669f87, more-itertools-w89d58c2, funcy;
  - 1 milestone: both semver windows.

**Prompts.**
- semver-w1bafb9f and humanize were sent back with the auditor's findings. The auditor's semver guess ("capitalised True") was **wrong**: the test expects `cmd_check` to return None or raise `ValueError`. The authored prompt then contradicted the verifier (D8). Fixed for this run with findings from the failing test's own parameters (a hand-extracted ledger-mapper step, to be automated).
- semver-w41775dd and toolz failed the name check 6 times: the author ignores names whose unit owns no milestone. The lane now states such names in one fixed sentence (`names_stated_by_lane`).
- All 13 prompts are lint- and realism-clean.

**Gates (13 windows).**
- Floors: 0 with Φ 0. Oracle: 1 with Φ 1.
- Mutants: 17/17 caught.
- Flake: 60/60 per window, bound 4.87%.
- Hacker (1 per window): 13 scored, no full pass. The mean reward of 0.094 comes from real source fixes that pass hidden milestone tests.

**Split rule.**
- Literally, "deepest milestone any run-1 end state reached, + 1" keeps the whole window in all 3 cases: a late independent milestone was reached.
- Split instead at the first unreached milestone in order: parse-wc669f87-p2 (2 milestones), parse-w07c2ff9-p3 (3), glom-p2 (2). All three are independent once cut.

**Per window** (qwen × 4; reward = F1 outcome reward):

| window | type (run 2) | qwen rewards | spread | final Φ mean | pass | verdict |
|---|---|---|---|---|---|---|
| parse-w07c2ff9 | A, 10 ms / 1 edge | .15 .15 .20 (1 masked) | yes | 0.33 | 0/3 | hold (skill gap 0.57) |
| toolz | A, 8 / 1 | 0 0 .06 .06 | yes | 0.06 | 0/4 | hold (skill gap 0.56) |
| glom | A, 7 / 6 | .07 0 0 0 | yes | 0.07 | 0/4 | back: world (D7) |
| humanize | A, 6 / 1 | .08 ×4 | no | 0.17 | 0/4 | split |
| more-itertools-w15407c7 | A, 4 / 2 | 0 ×4 | no | 0 | 0/4 | split |
| more-itertools-w89d58c2 | independent | .05 .10 .10 .05 | yes | 0.15 | 0/4 | not Type A (else accept-candidate) |
| parse-wc669f87 | independent | .17 .06 .11 .11 | yes | 0.22 | 0/4 | not Type A |
| funcy | independent | .14 .21 .14 .14 | yes | 0.32 | 0/4 | not Type A |
| semver-w1bafb9f / -w41775dd | 1 milestone | 0 ×4 / 0 ×4 | no | 0 | 0/4 | not Type A |
| parse-wc669f87-p2 / -w07c2ff9-p3 / glom-p2 | independent sub-windows | .25 ×4 / .17 .17 0 .17 / 0 .25 0 .25 | no / yes / yes | 0.5 / 0.25 / 0.25 | 0/4 | not Type A |

**Headline:**
- Pass rate 0 on all three rungs (qwen 0/51, nemotron 0/20, nova 0/52).
- The F1 reward spreads in **8 of 13** qwen groups (3 of the 5 Type A windows), against 6 by Φ in run 1. Mean final Φ: qwen 0.18, nemotron 0.04, nova 0.
- **F5 works:** qwen wrote STATUS.json in 45/51 episodes (run 1: 21/39); nemotron took the final STATUS turn after 150 calls. Claims stay overconfident (qwen Brier 0.70 when written).
- **First I6 signal, small n:** qwen loses 0.040 milestones per span after a compaction boundary vs 0.022 elsewhere (150 vs 626 spans); nemotron 0.022 vs 0.007.
- Cause mix: skill gap 62, budget 30, world bug 4, unlabelled 27. Auditor seeded controls 39/39.

**Spend:** prompts 0.013, probe 0.007, pilots 0.271, hacker 0.199, qwen 1.902, nova 0.224, nemotron 3.591, auditor 0.038.

**Defects found in run 2:**
- **D7 (harness, also in run 1):** `max_tool_output` 8,000 chars cut the middle of `read_file`'s 400-line pages. 409 qwen and 139 nova reads were elided. The 4 `world_bug` labels say exactly this; glom's `back` is owned by this config, not the task. Next: cap ≥ the World's 30k, or page 150 lines.
- **D8:** auditor rationale reached the prompt as fact; only test-derived statements may.
- **F8:** module-level imports of late names collapse prefix ownership; an import-tolerant grader would keep per-test granularity.
- **OpenRouter refused in-flight requests** (402 `in_flight_budget_exhausted`) with the guard at $6.2 of $12: the **account balance is near its limit**. 4 of 26 nemotron samples never ran, and no further paid work was attempted.
- The lane's grading sandboxes are named `crucible-m17-*` (existing code), not `crucible-lh-*`.

**DESIGN-ready notes:**
- Type A reward = 0.5·Φ_final + 0.5·pass on the RL path; the accept verdict stays binary.
- Φ gates only on behavioural edges.
- Milestone owner = first passing prefix.
- `LLMToolPolicy.tool_limit` tells the agent its budget truthfully.
- Prompt names come from the tests' imports, stated by the lane if the author omits them.
- Send-back findings must be test-derived.
