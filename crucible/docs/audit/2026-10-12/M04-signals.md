# M4 signals: audit (2026-10-12)

Scope:
- `src/crucible/signals/` (taxonomy, extract, attribute, reliability, pivots, gate: 233 statements);
- the pack plugins: coding `signals.py` (PR, frontier-run mappers), `outcomes.py`, `m4/` (gate, labels, trace build);
  toy `signals.py` and `m4/labels_test.json`;
- the facts M4 consumes from M3 (`github_pr`, `ci_logs`, `agent_runs`, `core/agent_events.py`, ATIF import) and the
  World's trace export;
- consumers of signals: `studio/datasets.py` (decision-point pairs), `studio/label.py`, the forge.

Checked against ARCHITECTURE §M4 and DESIGN §5.2. There was no `docs/modules/M04-*.md`; this audit adds one. Method:
the M0 protocol (requirement ledger, coverage, import contracts, one executable probe per suspected defect). Every
finding below was reproduced. 34 regression tests (`tests/unit/test_m4_audit.py`, `test_m4_audit_packs.py`) all
fail on the unmodified `HEAD` (`90f2539`) and pass after the fixes.

Seven read-only studies compared M4 with the local frameworks:
- `M04-study-mapping.md` (how events become signals), `M04-study-attribution.md` (step attribution, extraction,
  decision points), `M04-study-reliability.md` (reliability, outcomes, gate statistics);
- `M04-study-sweep.md`: at the user's request, four more agents read every other locally cloned repo and dataset
  (30 framework repos, agentenv, deepswe, the 33 HF dataset folders, the MiMo RL release).

## 1. Requirement ledger

| # | Requirement (source) | Status before | Evidence |
|---|---|---|---|
| R1 | Exit gate: type accuracy ≥ 0.9, attribution precision ≥ 0.9, outcome agreement ≥ 0.95 on 50 hand labels **per pack** (ARCH) | **failing, unnoticed** | <ul><li>Coding: re-run on HEAD gave 0.78 / 0.44 / 0.21. The committed report (`passed: true`) described traces the M3 audit had since rebuilt.</li><li>Toy: 84 labels existed, nothing scored them; forced through, 0.87 / 0.91 / 0.08.</li></ul> |
| R2 | Typed signals per pack: coding review defects, accepted suggestions, reverts and hotfixes, CI regressions, perf targets; toy reopen, escalation, CSAT (ARCH) | partial | intent rules misread 25 of 29 probe comments (M1); CI logs had no mapper (M6); customer replies unmapped and CSAT mistyped (M17); no `target` emitter |
| R3 | Attribution: the event that last wrote the object; coding: blame within the PR (ARCH) | **wrong** | `.` made `a.py.bak` cover `a.py` (A2); absolute vs relative paths never matched (A1); the App was ignored (A3); World business-App calls recorded no writes, so enterprise episodes had no step credit (A6) |
| R4 | Reliability: per-type agreement with later outcomes (ARCH) | **wrong** | verdicts came from the mapper's own rule and the trace outcome (R7); one CI job's failure "contradicted" every other job's pass (R9); families and AI reviewers pooled (R10, M2); fitted and scored on the same 14 traces (R13) |
| R5 | Outcome labels supplied by packs (ARCH `labels.py`) | **wrong** | coding: every merged PR "unknown" after M3 (R1-R3); toy: no labeler registered (R27); `labels.py` never existed (spec drift) |
| R6 | DESIGN §5.2 session rows: interrupt or rejected edit → negative on that call | missing | no mapper for any agent-session family; `extract` raised `LookupError` (A16); mini-swe-agent's rejections typed as talk (H1) |
| R7 | DESIGN P28: credit follows writes, to the agent or sub-agent that made the change | partial | ATIF subagent events landed after the user's later correction (A5); see A6 |
| R8 | Signal ids deterministic and idempotent (extract docstring) | **wrong** | ids keyed on position: one earlier row changed every id; the M3 store carried ids a fresh extract no longer made (A11, M21) |
| R9 | Decision points: graded wrong, then corrected (pivots docstring) | partial | a revert became the "chosen" side of a DPO pair (A20); duplicate rows (A21); contradicted signals kept (A22) |
| R10 | Signal reliability used by M6 (DESIGN) | missing | nothing read `Signal.reliability` (R12, A25) |
| R11 | Import contracts | live | `lint-imports`: 4 kept |

## 2. Defects (reproduced)

Ids follow the studies: M (mapping), A (attribution), R (reliability), H (harness sweep). Severity: P0 means wrong
labels, credit or gate verdicts in normal use; P1 a common case or a stated promise unmet; P2 an edge case.

### 2.1 The gate and outcomes (the regression M3 caused)

| Id | Sev | Defect | Fix |
|---|---|---|---|
| R1, R2, R3 | P0 | M3 took the fetch time out of PR traces (it changed the hash on every read); `PullRequestOutcome` read it, so every merged PR was "unknown | merged less than 30 days ago", even one merged 280 days earlier. A naive fetch time raised against Z times; a missing merge time raised | the connector records `outcome.details.window` (`checked`, `until`, `closed`: past the window **and** the clone holds a later commit), a fact that changes once; times parsed as instants |
| R4, R5 | P1 | a revert 199 days after the merge made the PR bad; a merge whose history was never read (merge commit not in the clone, or a PR that only deleted lines, which returned before looking for reverts) was "good" | the labeler filters to the window and needs `checked`; deletion-only PRs are scanned for reverts |
| R15 | P0 | labels keyed by `(trace, seq)`: M3's instant sort moved events, and labels pointed at other comments | every PR event carries `row_id` (`comment:`, `review:`, `check:<run id>`, `commit:<sha>`, `post:<sha>`, `pr:`, `merged`); labels name `event_key`s; both label files migrated by natural id against the versions they were written on (no label changed by the migration) |
| R16, R17, R18, R21 | P1 | a missing trace raised `KeyError`; only the first signal at an event was seen; a step graded where the label says "none" never counted; one label passed | fails closed: label errors are misses, the set of types at an event is compared, precision and recall count both kinds of miss, ≥ 50 events and ≥ 10 outcomes required |
| R20, R22, R25 | P1 | no per-class view (a mapper emitting nothing scored 0.73 on toy types), no uncertainty, "unknown" agreement mixed with decisions | per-type recall and macro recall, Wilson 95% intervals with a `confident` flag, outcome coverage, accuracy per human / automated source |
| R13 | P1 | reliability fitted and applied on the gate's own traces | fitted on stored traces the gate does not score (coding: 14 other PRs; toy: the dev cases), with a 90% interval |
| R26-R29, M17 | P1 | toy: labels named `toy/helpdesk_ticket` (no mapper); no runner; no outcome labeler; CSAT score dropped at ingest (`extra=["team"]`), typed `check_result`; customer replies unmapped; escalations graded on the queue's routing | `crucible_pack_toy.m4.gate`; labels migrated to `toy/support_ticket` and row ids; `TicketOutcome` registered (the rule moved from the M5 pipeline, which now imports it); ingest keeps every column the mapper reads; customer-reply and CSAT-as-outcome rules; escalation grades the last answer |

### 2.2 Extraction and attribution

| Id | Sev | Defect | Fix |
|---|---|---|---|
| A11, M21 | P0 | ids keyed on seq: toy case T1003 plus one earlier row kept 0 of 3 ids; forge provenance and the "trap signal resolves" check cite ids | `signal_id(trace, type, event)` keys on `event_key` |
| A6 | P0 | `tickets.update_ticket`, `erp.approve_invoice` wrote nothing in World traces: a later violation graded no step | `agent_events.app_writes`: `<entity>_id` (or JSON-API `collection`+`id`) names the record, other scalar arguments are fields (`ticket:T7.status`); read-only tools and read verbs write nothing; failed calls record no writes |
| A1, A2, A4 | P1 | absolute vs relative and `./` paths never matched; `.` as hierarchy for files; `rm`, `git mv`, `git rm` were not writes; a directory write did not cover its files | `file:` objects are normalised paths with `/` hierarchy and absolute/relative suffix matching (NeMo-Gym `swe_pivot` pattern); records keep `.` fields; `shell_writes` adds deletions and moves |
| A3 | P1 | a peer's write in another workspace took the grade | the App must match when both sides name one |
| A5 | P1 | ATIF subagent events appended after the whole main trajectory, so a later user correction could not grade them | spliced after the call whose result carries Harbor's `subagent_trajectory_ref` (DEPEND); the call records `delegated_to` |
| A7, A8, A9 | P2 | `exclude_self` with no author excluded nothing; a reviewer's own fixup could be the graded step; "unattributed" looked like "trace-level" | anonymous exclusion; rejections, violations and approvals never grade their author's write (reverts, reworks and escalations may: actors undo their own steps); `meta.attribution` + `reason` |
| A12 | P1 | re-extraction dropped hand-added signals and computed reliabilities | mapper output replaces mapper output by id; `meta.by` producers stay; known reliability is kept |
| A13, A14, A15 | P1 | bad mapper output failed later, in whichever stage loaded the stored trace; wrong polarity accepted; a duplicate id aborted a batch without naming the trace | `validate` at extraction (`ExtractError` names producer and trace); grades after the signal (clock skew in real data: the pre-M3 traces had CI runs before their commits) are dropped and flagged; Studio labelling isolates a failing trace |
| A16, H1, H2 | P1 | agent-session families had no mapper; mini-swe-agent's "Commands not executed. The user rejected ..." was a plain user message | `signals/session.py` runs on every trace: interrupt → the agent's last action; rejected tool use → the proposing call (it never ran, so last-writer would grade an unrelated edit) |
| A19 | P2 | a family with a labeler but no mapper could not be labelled | extraction runs without a pack mapper |
| A20, A21, A22, A24 | P1 | a revert was the "chosen" action ("reverse the refund" at the state before the refund); two graded calls in one turn gave two identical rows; a contradicted signal stayed a pivot | only corrections and applied suggestions pair; one row per (turn, fix turn); contradicted signals and reverts that touched no line are not pivots; rows carry the signal's reliability (a label-noise rate for robust DPO) and `state_shift` |

### 2.3 Reliability

| Id | Sev | Defect | Fix |
|---|---|---|---|
| R7, R8 | P0 | corrections and applied suggestions could never be contradicted; a revert defines the bad outcome and so confirmed itself; any bad trace confirmed every negative on any step | verdicts only from evidence the mapping rule did not use: a revert or hotfix of the graded step; the fix itself reverted; the trace ending well without the request; reverts and hotfixes keep their prior |
| R9 | P0 | 22 of 60 contradicted passing checks were "contradicted" by a different job failing on that commit | a check is judged against the same check only |
| R10, M2 | P0 | estimates pooled families and people with AI reviewers: Copilot produced 107 of 214 non-CI signals in the corpus, and was two actors of two kinds | rows per (family, type, human or automated); one identity per AI reviewer (alias table); gate accuracy per source |
| R11, R14 | P2 | flaky and ungraded signals got the type mean; seq 0 read as "no seq" | only judgeable signals get a value; explicit `None` checks |

### 2.4 The coding and toy mappers

| Id | Sev | Defect | Fix |
|---|---|---|---|
| M1 | P1 | intent rules: negations were requests ("No need to change this"), any cue beat a question mark, praise had to be a whole comment in one spelling, plain imperatives and defect statements were missed; 25 of 29 probe comments wrong | sentence-level rules: negated cues, questions (clarify unless a suggestion: "can you", "why not", "should we add"; a hedged "I think X is nowhere ...?" asserts a defect), praise heads with a clean tail, imperatives, defect statements, inline code masked; non-English abstains |
| M3 | P1 | `suggestion_applied` was a substring test over the whole final file (`pcl#6364` ev 4: an unchanged line whose suggestion matched line 271) | applied only when the commented span changed |
| M4 | P1 | an unknown "acted on" (LEFT-side or file-level comment) read as "not acted on": a rejection | no signal when the connector could not tell |
| M6 | P1 | `ci_logs` traces had no mapper | `CIRunSignalMapper`: one check per test, infra-failed and cut-off runs unscored, runner-marked flaky tests polarity 0, a corpus flaky list can be passed in |
| M8, M9 | P1 | `Revert "Revert ..."`, "Reapply", "do not revert #N" were reverts; `refactor:` and `docs:` commits that mention a fix were hotfixes | an odd number of `Revert "` or git's `This reverts commit <sha>`; conventional-commit types are not fixes (repo2rlenv's rule) |
| M10, M11 | P2 | a reviewer's reply under the author's own comment was dropped; a suggestion with praise words became an approval | the first reviewer reply speaks for an author-rooted thread; a concrete fix is a request |
| M12, M13 | P2 | `startup_failure`, `FAILURE`, `error` gave polarity 0; two workflows' `build` jobs were "flaky"; a revert touching no PR line graded every commit (the normal path for squash merges) | lowercased conclusions, jobs keyed by (app, name, sha); squash reverts map to the PR commits that wrote the touched files; otherwise the revert grades nothing |
| M14, M15, M16, H5 | P1 | a contaminated green frontier run got +1; a graded run with every gate passing got nothing; `"1.0"` raised, NaN and the -1 "not graded" sentinel were -1, 0.5 was 0, `"pass": "false"` passed, a repeated gate aborted extraction; runs stopped by limits were failures | finite non-negative rewards only, partial rewards polarity 0, gates deduped, pass parsed, contaminated and limit-stopped runs unscored |
| M18, M19 | P2 | CSAT `"3"` was -1, `"4.5"` raised, the text fallback took the first digit; rule and SLA text fallbacks mislabelled | numeric coercion on a declared 1-5 scale, explicit "N/5" or "rated N" only; wider rule and SLA patterns |

One hand label was revised: `pcl#6309` comment 2316983555 graded nothing, but `git blame -L 373,373 7286269` names the
PR commit `2556e89` (the line is a `+` line of the diff hunk); the null was left when its type was revised earlier.
The labels file records it (5 revisions).

## 3. Coverage and contracts

- Signals core (`crucible.signals`): 51% on HEAD (`gate.py` and `reliability.py` 0%; the coding mapper and
  `outcomes.py` were never imported by a test). After: 93% over the core and the pack mappers and labelers (core 96%,
  coding `signals.py` 88%, `outcomes.py` 85%, toy 97-100%).
- `lint-imports`: 4 contracts kept (signals still depend only on core).
- `ruff check`: clean.

## 4. Comparison with the local frameworks

No repo or dataset turns real later evidence (reverts, hotfixes, reopened tickets, review comments, real PR CI history)
into typed, step-attributed signals, and none measures annotator agreement (`M04-study-sweep.md`).

### 4.1 Ranked features (not built in this audit)

| # | Feature | Source | Mode | Where |
|---|---|---|---|---|
| 1 | Signed step credit from graded signals: zero flagged tokens on positive rows, scale them on negative rows, conserve each sign's total; κ from reliability | xiaomimimo-verl `verl/trainer/ppo/signed_rebalance.py:46-215` | PORT | M6 (the consumer M4 lacks: A25, R12) |
| 2 | CI verdicts over all attempts with an "inconclusive" class; per-check flake rate across the corpus | NeMo-Gym `swemer_v2/aggregate_golden_patch.py:51-64`; SWE-smith `harness/repair.py:122-171` | PORT / PATTERN | `github_pr` fetch with `filter=all` per commit (M5), `CIRunSignalMapper` |
| 3 | A closed-choice intent judge behind the rules, reasoning first, abstaining on no verdict | verifiers `judges/rubric.py:24-35`, `utils/score.py:51-81` | PATTERN (DEPEND from the pack) | `CodingSignalMapper` |
| 4 | Held-out labels, a second labeller and Cohen's κ; labels bound to a trace hash | repo2rlenv `quality/labels.py`; Crucible's own κ in the toy M5 pipeline | PATTERN | `m4/labels_test.json` (14 PRs listed in `test_prs.json`, unlabelled) |
| 5 | Decision points from outcomes alone (steps that reach the same state across traces), profiled with ≥ 8 on-policy rollouts | ROLL GiGPO `agentic/utils.py:155-211`; NeMo-Gym pivot-dataset skill `SKILL.md:94-110` | PATTERN | `pivots.py`, `studio/datasets.py` |
| 6 | "Cannot judge" kept apart from 0 everywhere a score is made | AReaL `prm/runner.py:26-37`; TRL GRPO; MiMo | PATTERN | gate and reliability already; mapper reason codes next |
| 7 | Judge findings with severity and step index as step signals | agentenv `judge_output_format.py:142-200` | PATTERN | a `review_finding` mapper |
| 8 | Label-noise-aware preference losses fed by reliability | TRL Robust DPO `dpo_trainer.py:1407-1410`; OpenRLHF cDPO `models/loss.py:334-358` | PATTERN | M12 (rows now carry `reliability`) |
| 9 | Typed failure categories for decision points | NeMo-Gym `StepRewardCategory` (`verification_utils.py:60-75`) | PATTERN | `pivots.py` |
| 10 | An experiment-ledger family that emits `target` with a noise band | karpathy autoresearch | PATTERN | a pack mapper |

### 4.2 Where Crucible is stronger

- A signal layer at all: no framework maps human evidence (review, revert, reopen, policy flag) to the step it grades.
- Step attribution from git (blame at the comment's commit, followed through pushes, rebases and renames), and
  "acted on" decided from the code, not the text.
- Post-merge history as signal, with an explicit `unknown` while the 30-day window is open.
- One taxonomy across domains, with per-type priors; packs map onto it.
- Infra failures masked, never scored 0; frontier gate failures feed trap synthesis.
- Reliability as a posterior per source with intervals, and decision points that carry the evidence that resolved them.

## 5. Gate re-run

| Gate | Before (HEAD) | After |
|---|---|---|
| Coding dev (`python -m crucible_pack_coding.m4.gate dev --report`) | type 0.78, attribution 0.44, outcome 0.21: **failed** | type 1.0 (95% CI 0.93-1.0), attribution precision 1.0 over 32 events (0.89-1.0), recall 1.0, outcome 1.0 over 14 (0.79-1.0), coverage 0.79: passed, not `confident` |
| Toy test (`python -m crucible_pack_toy.m4.gate test --report`) | not runnable; forced: 0.87 / 0.91 / 0.08 | type 1.0 (0.96-1.0), attribution 1.0 over 17 (0.82-1.0), outcome 1.0 over 12 (0.76-1.0): passed, not `confident` |

Read these with care:
- The coding labels are a dev split written by the builder; two intent rules (`should we add ...?`, a hedged defect
  question) came from dev errors in this audit. The toy labels were written before the mapper, but the toy fixes were
  made with the label errors visible. Neither is held-out evidence any more.
- 247 signals on the labelled coding traces sit on unlabelled events and are not scored.
- Of the 35 labelled coding events that carry a signal, 20 come from automated reviewers.
- At these label counts the lower bounds cannot clear the bars: 51 straight agreements are needed for outcome ≥ 0.95.

## 6. Resolution (2026-10-12)

Everything in §2 is fixed and covered by the 34 tests (all 34 fail on `HEAD`, all pass after).

Re-runs after the fixes:
- M4 gates: §5.
- M3 gate: pass, 106 traces with the same counts, 0 canaries leaked, idempotent. Post-merge "fixes" touching PR lines
  fell from 8 to 6: two merges of "modernize" PRs had been counted because the commits they brought in mentioned fixes
  (only a commit's own subject and body count now).
- M8 toy hindsight: 6/6 accepted; M8 toy world synthesis: 5/5; M8 coding derivation: unchanged (4 of 5 seeds with a
  trap, task01 without). The signal ids these reports cite changed once (stable keys); two hindsight traps now also cite
  the customer's rejection or a low CSAT. `m8_toy_world_synth_llm.json` (a paid run) keeps the old ids.
- M5/M6 toy pipeline (now on the shared ticket outcome rule): identical gates (S2 was already below its bar).
- Suite (`tests/unit`, `tests/toy_e2e`): 924 passed, 3 skipped (890 before, plus the 34 new tests); `ruff` clean; `lint-imports` 4 kept. A first run alongside the gate and forge re-runs had 7 failures from the shared data directories; each passed alone, and the clean run passed.

Open:
- M5: CI reruns and per-commit CI need a new API fetch (`filter=all` for every PR commit); the cached corpus has the
  latest run of each check on the head only, so the PR flaky branch cannot fire on it.
- M7: a review verdict grades the reviewed head commit, not the commits it is about.
- M20 (dismissed reviews kept as evidence), M22 (one comment, several requests), R6 (closed-unmerged is always bad,
  as the guideline says).
- R19, R23, R24: unlabelled signals are counted, not scored; no held-out coding labels; one labeller (the builder).
- R12, A25: M6 does not read `grades` or `reliability` yet (feature 1; the M6 audit).
- H3, H4, H6, H8: human-mode commands as human writes; the corpus flaky list wired into the PR mapper; frontier
  failures grading only the final patch; prime-agent aborts and abandoned branches.
- `finance_ap` has no mapper or outcome labeler (no recorded AP data yet).
- Non-English review comments abstain (feature 3).
