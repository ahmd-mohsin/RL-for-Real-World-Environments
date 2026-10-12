# M5 workflow: audit (2026-10-12)

Scope: `src/crucible/workflow/` (steps, actions, shellops, filestate, replay, logstate, milestones/{mine, filters,
match, evaluate}, stages, graph, tracker, potential: 2,021 lines), the live fs adapter M5 milestones run against
(`world/state/fs.py` and its snippet), the shared write detector (`core/agent_events.py`), and the pack pipelines
(coding `m56/` rediscovery, rebench Φ and gate; toy `m56/pipeline.py`).

Checked against ARCHITECTURE §M5, DESIGN §5.3 and `docs/modules/M05-workflow.md`. Method: the M0 protocol. Before the
audit, at the user's request, every locally cloned framework and dataset was read again for M5, M6 and M7 and the
best of them picked (`M05-M07-study-sweep.md`, six agents); a seventh agent hunted defects in Crucible's M5 code
(`M05-study-defects.md`). 24 regression tests (`tests/unit/test_m5_audit.py`) all fail on the unmodified `HEAD`
(`b65754b`) and pass after the fixes.

## 1. Requirement ledger

| # | Requirement (source) | Before | Evidence |
|---|---|---|---|
| R1 | Segmentation: one step per state-changing call, read-only runs merged; PRs by commit / review / CI (DESIGN §5.3.1) | partial | PR steps re-sorted by the time *string* (W4); a write the M4 detector saw classed "investigate" and merged away (W14); a subagent's message folded into the main agent's step (W23); one merged step kept one model call's credit row |
| R2 | Replay test and quarantine (DESIGN §5.3.2) | **wrong** | shell replay dropped edits silently (`sed -Ei`, `perl -pi`, `tee`, `dd`, `mv ... 2>/dev/null`; W6) and applied wrong ones (`git restore --staged` reset the file, `mv f dir` made a file `dir`, `false && rm` deleted; W7); refused edits replayed (W8); renames, binaries, unprefixed diffs broke the parser (W9); per-step Φ was computed on files an unreplayed command had written |
| R3 | Exit gate S1: replay ≥ 90% on ≥ 30 traces | live, unreproducible | 0.9485 from cached, unversioned rows; any written source file a call did not name was excused as a "byproduct" (W20) |
| R4 | Diff, CI and signal candidates (DESIGN §5.3.3; SWE-smith F2P per step) | partial | C pointer lines (`*dst = tex1Dfetch(...)`) skipped as comments (W5); changelog prose mined (W10); anchors bound to forward declarations (W11); signal candidates were `exists(path)`, true at the base (W12); CI mining missed added tests and skip -> pass, mined a single flaky flip, and passed test ids through `repr()` into a shell |
| R5 | Filters: executable, monotone, non-trivial, predictive | partial | coding SWE-rebench milestones skipped every filter (W10); the toy κ gate holds only with `min_monotone=0.6` (W3) |
| R6 | Milestones measurable on a live World (`evaluate.py`: "the same milestone gives the same φ on both views") | **wrong** | the live fs adapter's grep rejected `paths` and had no `block_changed`: every mined coding milestone was unmeasurable live (W1) |
| R7 | Stages and tracker (DESIGN §5.3.4-5) | partial | layering depended on input order when prerequisites contradicted the data (W16); stage names collided (W18); a validly achieved milestone reported out of order (W17); Φ(s0) taken as 0 (W19); invalid graphs accepted, the tracker crashed on a stage/milestone mismatch (W15) |
| R8 | Tracker κ ≥ 0.7 vs manual stage labels | live, fragile | κ 0.8685 at the pipeline's `min_monotone=0.6`; 0.2756 at the module default (W3) |
| R9 | Rediscovery: mined milestones cover all six grader check families | **wrong as reported** | "5 of 6, hand-audited": only 2 families were audited; API-removal hits included APIs being added; 34 of 40 edge-semantics hits were block *names*; the committed report predated a mining fix (W2) |
| R10 | Event-log state for World episodes (M4 App writes) | **wrong** | `logstate` stored the event type, not the written value (W13) |
| R11 | Coding F1 graph; Harbor prefix seeding; agentenv snapshots (ARCH Takes) | missing | not built (picks 6-8 queued) |

## 2. Defects (reproduced) and fixes

| Id | Sev | Defect | Fix |
|---|---|---|---|
| W1 | P1 | mined milestones unmeasurable on a live World | the fs snippet's grep takes `paths`; `FsGitAdapter.q_block_changed` (git show through the trusted git dir); `block()` moved to `core/textblock.py` so both views share it; a parity test runs a mined set on both |
| W2 | P1 | rediscovery over-credited | only removed/added constructs count, API removal needs a removal, block names never count; every family hand-audited and the audit stored in the report: **4 of 6** (build, API removal, concurrency from code; perf from review comments only); edge semantics has 1 genuine PR (`#6460`'s `isfinite`) |
| W3 | P1 | κ hinges on an unrecorded threshold | `MIN_MONOTONE = 0.6` named and recorded in the graph; κ at the module default reported beside it (0.2756) |
| W4 | P1 | PR steps sorted by time string | keep the connector's instant order |
| W5 | P1 | `*` lines dropped from mining | `*` is a comment margin only inside `/* ... */` |
| W6, W7 | P1 | shell replay dropped or invented effects | quote-aware parsing with subshells, fd redirects and word-internal `#`; every path the shared write detector names is applied or reported; `git restore --staged`, `mv f dir`, `printf` formats, unquoted here-docs, writes behind `||` or behind `&&` after a command of unknown outcome are reported, not guessed |
| W8 | P2 | refused editor calls replayed | skipped when their result is an error (`meta.skipped_failed`), as M4 records no writes for them |
| W9 | P2 | renames, binaries, unprefixed diffs | `rename from/to` and `copy from/to` parsed; binaries recorded as opaque (`skip_binary`); the `a/` `b/` prefix detected |
| W10 | P1 | unfiltered coding milestones, prose mined | `from_diff` skips prose files by default; the SWE-rebench pipeline filters for non-triviality (base false, expert end true) |
| W11 | P2 | anchors on forward declarations | anchors refuse a line ending in `;` |
| W12 | P2 | signal candidates trivially true | `from_signals(signals, base)` checks `block_changed` on the definition enclosing the flagged line |
| W13 | P2 | App writes stored the event type | the value comes from the call's arguments |
| W14, W23 | P2 | writes merged into investigations; another actor's message folded in | a call with writes is an edit and never merged; another actor's pending events are their own step |
| W15 | P2 | invalid graphs | `WorkflowGraph` validates weights, prerequisites and stage members; the tracker looks prerequisites up by id |
| W16, W17, W18 | P2 | order-dependent layering, false out-of-order, colliding names | contradicting prerequisites dropped and reported (`StageModel.conflicts`), fixed iteration order; out-of-order decided after the row settles; unique stage names |
| W19 | P2 | Φ(s0) = 0 | `WorkflowTracker.start(base_view)` |
| W20 | P2 | any unnamed file excused as a byproduct | only run output (logs, caches, build dirs, `.pyc`, `.coverage`) |
| W21 | P2 | S1 rows not tied to code | `replay_version()` stamps rows; S1 re-run with the current code (`reports/m5_s1_replay.json`); the gate uses it when the versions match |
| W22 | P2 | toy replay checks tautological | the final-status check dropped (it compared a value with itself); a converse check added for the rule the state decides alone (refund over 50 without approval) |
| S (sweep) | P1 | CI mining (tests the change adds, skip -> pass, flaky flips, shell injection through test ids) | as listed in R4; `shlex.quote` |
| S (sweep) | P1 | per-step Φ on stale files | `FileState.stale`: paths an unreplayed command wrote are unknown until rewritten whole; `FileStateView` answers unmeasured for them |
| S (sweep) | P2 | no state key, no per-call credit row, mini-swe-agent traces quarantined | `Step.state_ref` from the event or its result; `Step.data.call_ids`; `EditReplayer(scope="submitted")` |
| M4 follow-up | P1 | the M4 write detector split shell text on `;` and `>` before parsing quotes, so `python -c "x > 0"` "wrote" files named `=` and `0:` | quote-aware segmentation in `core/agent_events.shell_writes`; 62 false unreplayable reports gone from the SWE-rebench re-run |

## 3. Coverage and contracts

- `crucible.workflow`: 78% before (59 tests); 85% after over the full suite (`shellops.py` 64% -> 83%, `mine.py` 87% -> 98%, `steps.py` 68% -> 80%).
- Suite (`tests/unit`, `tests/toy_e2e`): 948 passed, 3 skipped (924 before, plus the 24 new tests); `ruff` clean.
- `lint-imports`: 4 kept (`core.textblock` is core; the world adapter imports only core).

## 4. Comparison with the local frameworks

The picks, with what this audit built and what is queued: `M05-M07-study-sweep.md` ("The picks", M5 table). Built
here: fail-closed replay (NeMo-Gym pattern), per-step state keys (ROLL), per-call credit rows (NeMo-Gym pivot rules),
submitted-scope replay (mini-swe-agent), SWE-smith-style CI mining. Queued with sources: agentenv per-call state
capture, Harbor prefix seeding and live trajectory following, P2P guardrail milestones, native-progress milestones,
the valid-alternative filter, an information-gain potential.

Where Crucible is stronger (unchanged by the sweep): no framework mines milestones from expert diffs, induces stages
from data, or tracks a workflow state online; replay of recorded shell and editor calls against a final patch has no
counterpart; NeMo-Gym's re-verification re-runs nothing.

## 5. Gate re-run and resolution

| Gate | Before | After |
|---|---|---|
| S1 replay (coding, 913 SWE-rebench attempts on 80 issues) | 0.9485 agent-state, 0.9113 strict (cached rows) | **0.9189** agent-state (839), **0.9102** strict (831), current code (`m5_s1_replay.json`); 378 calls reported unreplayable (440 with the old write detector); 167 attempts have stale intermediate states, now unmeasured |
| S1 replay (toy) | 24/24 on 31 checks (24 tautological) | 24/24 on 7 checks |
| Tracker κ (toy) | 0.8685 | 0.8685 at `min_monotone` 0.6; 0.2756 at the module default, both reported |
| Rediscovery | "5 of 6" | **4 of 6**, every family hand-audited (`m5_rediscovery.json`) |

The S1 drop comes from W20: 27 attempts whose submitted source files no call named were counted as replayed.

Open:
- the queued picks above;
- coding stage labels for the tracker κ; a coding F1 graph; behavioural (CI-derived) milestones on SWE-rebench, which
  need each issue's image;
- the S2 rows (`rebench_phi`) were computed before the non-triviality filter and the stale-state rule; re-running
  them re-evaluates Φ on every state of 913 attempts (cheap, offline) and the judge comparison (paid) — not run;
- `rm -r` of a directory whose files were never read cannot delete what the replay never loaded (reported by neither
  side; rare).
