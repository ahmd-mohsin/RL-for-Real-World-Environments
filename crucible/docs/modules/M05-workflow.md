# M5 `workflow` — workflow model

**Status: built (2026-10-09); gate 2 of 3 met, rediscovery 5 of 6 families.**

- **Code:** `src/crucible/workflow/`; coding pack `m56/` and `corpora/swe_rebench.py`; toy pack `m56/`.
- **Tests:** `tests/unit/test_workflow.py`, `tests/unit/test_workflow_stages.py`, `tests/unit/test_probes.py`.
- **Reports:** `reports/m56_gate_coding.json`, `reports/m56_toy.json`, `reports/m5_rediscovery.json`.
- **Research behind the design:** `docs/research/m5-m6/` (nine reports over every framework and article we pulled).

## What it is

| Piece | What | Built on |
|---|---|---|
| `actions.py` | What a tool call does: investigate, edit, build, test, run, communicate, act, submit. Shell commands are parsed with quotes understood, so `python -c "print(1 >= 2)"` is not a redirect. Scratch work in `/tmp` is not an edit. A chained command takes its most consequential kind. Truncated JSON arguments are recovered | — |
| `steps.py` | Segmentation. Agent sessions: one step per state-changing call, consecutive read-only calls merged into one "investigate" step. PRs: commits, review rounds, CI runs. Event logs: one step per writing event. Families can register their own `segmenter` | NeMo-Gym segmentation rules (one model call per step; reasoning does not start a new step) |
| `filestate.py` | File state rebuilt from recorded actions, with exact editor semantics: OpenHands/SWE-agent `str_replace_editor` (including tab expansion and `undo_edit`), Claude Code Edit/Write/MultiEdit, and a strict unified-diff applier. The applier passes 80/80 randomized round-trips against real `git diff`, covering end-of-file newlines and empty files | — |
| `shellops.py` | The file effects of shell commands, replayed from their text: `cd` across calls, `rm` (including globs and `-r`), `mv`, `cp`, `touch`, here-doc and `echo` writes, `git checkout/restore`. Effects that depend on running a program (`sed -i`, scripts that write files, `git stash`) are reported as unreplayable rather than guessed | — |
| `replay.py` | The replay test (S1). `EditReplayer` re-applies every edit to the base snapshot and compares the result with the recorded final patch. A file no recorded call ever names (test logs, caches) is an execution byproduct: strict replay counts it, agent-state replay does not. A trace that does not replay is quarantined | GitSwarm exact-source repeats (a tolerance band per recorded metric) |
| `logstate.py` | Object state rebuilt from event logs (the horizontal path for tickets, ERP and CRM exports): `writes` plus value fields, plus pack-supplied effects. Queried like a World (`get`, `exists`, `match`, `count`, `{case}` templates) | — |
| `milestones/evaluate.py` | φ for each milestone on any state view: a live World's `VerifierView`, a rebuilt `FileState` (`read`, `exists`, `list`, `grep`, `block_changed`), or a `LogStateView`. Cost policy: cheap checks every step, medium ones at checkpoints, expensive ones at the end; a value not re-evaluated is carried forward. A check that cannot run gives `None`, never 0 | — |
| `milestones/mine.py` | Candidates from three sources. **Diff:** constructs removed or added (tokens, so a different implementation can still satisfy them), the expert's block changed ("located"; brace-aware for C/C++), files created or deleted. **CI:** fail-to-pass per step. **Signal:** regions a reviewer flagged | SWE-smith F2P logic, applied per step |
| `milestones/filters.py` | Executable, non-trivial (false at base, true at the expert's end state), monotone on good traces; the predictive filter is `credit.validity` | — |
| `milestones/match.py` | Milestone → the step that achieved it, respecting prerequisites. A milestone that holds before its prerequisites is reported as out of order, not credited | NeMo-Gym ToolSandbox milestone-DAG matcher |
| `stages.py` | A → B is a precedence edge when A comes first in ≥ 90% of traces reaching both. Stages are layers of that graph. Names come from a default namer or an LLM; membership comes only from the data | — |
| `graph.py` | The family `crucible.workflow/v1` document: milestones, stages, edges, filter verdicts, sources | — |
| `tracker.py` | The online Workflow State Tracker. Each step yields stage, Φ, milestones gained and lost, the next expected milestones and deviations (regressions, answer-key probes). The same code labels recorded traces offline | — |

## Exit gate

| Requirement (ARCHITECTURE M5) | Result |
|---|---|
| S1: replay ≥ 90% on ≥ 30 traces | ✅ **Coding:** 866/913 OpenHands attempts on 80 SWE-rebench issues (94.9%) reproduce their recorded final state; strict 832/913 (91.1%). 47 are quarantined: generated files, Python-written files, `git stash`. **Toy:** 24/24 ticket logs; every recorded check (31) is reproduced from the rebuilt state: unapproved refund over 50, closure without retention offer, deletion without identity check, first-response and reopen-response SLAs |
| Tracker vs manual stage labels, Cohen's κ ≥ 0.7 | ✅ **Toy: κ = 0.87, 90.5% agreement** on 84 test-ticket events. The labels were written before any toy tracker code existed. The disagreements are reassignments after a rejected answer, which a human calls "triage" and object state calls "respond". Caveat: the builder wrote both logs and labels. Coding stage labels are not written yet |
| Rediscovery: mined milestones cover every grader check family (build, API removal, bit-exactness, edge semantics, concurrency, perf) | ⚠️ **5 of 6**, on 55 PCL GPU-migration PRs, hand-audited. **From code:** build, API removal, edge semantics, concurrency. **From review comments only:** perf. **Not rediscovered:** bit-exactness, whose candidates were tolerance parameters and NaN guards. In tasks 01–04 bit-exactness came from differential checks against a reference build (DESIGN §7.1), a source upstream PRs do not contain |

## Findings

1. **A first keyword pass "rediscovered" all six families. It was wrong.** "round" matched "stereo_g**round**_segmentation", "perform" matched `performShift`, "stream" matched `PointStream`. Families are now matched only against a milestone's construct, and verdicts are audited (`m5_rediscovery.json["audit"]`).
2. **Checking R1–R6 on the real tasks found mining bugs.** Function bodies in brace languages were cut at the anchor line. Comment lines ("Software License Agreement (BSD License)") anchored blocks. A token counted as new because the tokenizer skipped comments while the grep check did not. The oracle's own Φ was 0.69 on task04 until these were fixed; it is 1.0 now.
3. **Agents edit through the shell as much as through editors.** The first replay pilot reproduced 21/100 attempts. Replaying `rm`/`mv`/`cp`/here-docs/`git checkout` from command text raised it to 78/100, and separating harness-written files and execution byproducts to 99/100.
4. **OpenHands writes `issue.md` and `.openhands/TASKS.md` itself.** They appear in the recorded final patch but come from no agent action; `swe_rebench.harness_file` lists them.

## Not done

- Coding stage labels for the tracker κ.
- A behavioural (CI-derived) milestone run on SWE-rebench, which needs each issue's Docker image.
- PCL formatting-CI replay (designed, not run).
- LLM stage naming (the default namer is used).
