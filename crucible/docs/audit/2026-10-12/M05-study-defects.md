# M5 study: defects in the workflow module (replay, mining, stages, tracker, gates)

Scope: `C/workflow/` (`actions.py`, `steps.py`, `filestate.py`, `shellops.py`, `replay.py`, `logstate.py`, `milestones/{mine,filters,match,evaluate}.py`, `stages.py`, `graph.py`, `tracker.py`, `potential.py`); pack usage `TP/m56/pipeline.py` (+ `stage_labels_test.json`), `CP/m56/{gate,rediscovery,rebench_phi,task_properties}.py`, `CP/forge/derive.py` (milestone writer); live-state side `C/world/state/{base,fs}.py`, `C/world/world.py:to_trace`; M4 inputs `C/core/agent_events.py`, `TP/outcomes.py`. Spec: ARCHITECTURE "### M5 `workflow`", DESIGN §5.3 (`DESIGN.md:403-437`), `docs/modules/M05-workflow.md`, `docs/research/improvements/M03-M05.md` (M5 rows). Reports: `docs/modules/reports/{m56_toy,m56_gate_coding,m5_rediscovery}.json`. Read-only study, 2026-10-12, HEAD `b65754b`.

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Every candidate marked **[ran]** was reproduced with `crucible/.venv` on HEAD `b65754b`. Probes are in the session scratchpad `m5hunt/p1..p10_*.py`; each input is quoted below so it can be retyped. No committed file was modified: the toy pipeline and rediscovery were re-run in memory or into the scratchpad, and the coding gate's report was copied, re-generated and restored with `git checkout`.

Not repeated here (already in `M03-M05.md`): missing `fidelity.py` / `segment.py` / `states`, missing held-out next-milestone accuracy, Provenance without input hashes, no automated nop/oracle control gate. W12 restates the dead signal source only because it bears on the rediscovery claim.

## 0. Gate re-runs

| Gate | Command | Result on HEAD | Note |
|---|---|---|---|
| Toy (S1, κ, R1-R6, S2) | `TP/m56/pipeline.run()` in memory (probe p3) | replay 24/24, κ 0.8685, agreement 0.9048, R1-R6 pass, S2 mid AUC 0.583: identical to the committed `m56_toy.json` | The M4 outcome rule move (`TP/outcomes.py`) changed nothing. Graph is deterministic across two runs. κ depends on a non-default threshold (W3) |
| Coding assembly | `python -m crucible_pack_coding.m56.gate` (report restored) | S1 agent-state 866/913 = 0.9485, strict 832/913 = 0.9113; rediscovery false; κ(toy) true | Assembled from cached, untracked `data/m56/*.jsonl`; nothing recomputed (W21) |
| Rediscovery | `CP/m56/rediscovery.run()` with `REPORT` redirected to the scratchpad (probe p7) | `pass=False`, same verdicts, but **build drops from 29 hits / 11 PRs (committed) to 3 hits / 2 PRs** | Committed report predates the `code_file` fix in `7e3e5b1` (W2) |
| Unit tests | see §4 | 59 passed | — |

## 1. Requirement ledger

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| R1 | Step segmentation: one step per state-changing call, read-only runs merged; PRs by commit/review/CI (DESIGN §5.3.1) | partial | Built (`steps.py`). PR steps are re-sorted by raw time string, undoing M3's instant sort (W4); writes recorded by M4 are classified `investigate` and dropped on merge (W14); another actor's message is folded into this actor's step (W23) |
| R2 | State reconstruction + replay test; quarantine traces that fail (DESIGN §5.3.2) | partial | `EditReplayer` + `shellops` built. DESIGN says replay "must reproduce the recorded CI outcomes"; S1 compares final files with the final patch, never CI (W21). Shell replay silently drops (W6) or misapplies (W7) common edits; failed calls are applied (W8); renames/binary/no-prefix patches break it (W9) |
| R3 | Exit gate S1: replay ≥ 90% on ≥ 30 traces (ARCH) | partial | Coding 0.9485 agent-state on 913 (≥ 30) from cached rows of untracked data; strict 0.9113. Agent-state rate excuses authored files no argument names (W20). Toy 24/24 is below 30 traces and its `final_status` check is tautological (state built from the same events) |
| R4 | Diff-derived candidates (DESIGN §5.3.3) | partial | Built. Tokenizer skips C pointer-deref lines (W5); changelog prose mined (W10); `located` anchors can bind to a forward declaration (W11) |
| R5 | CI-derived candidates, SWE-smith F2P per step (ARCH PORT) | partial | `from_ci` exists and is unit-tested; never called by any pipeline (`grep from_ci(` hits only `tests/unit/test_workflow.py:163`). Its `run` milestones cannot be evaluated offline (FileStateView has no `run`) |
| R6 | Signal-derived candidates (DESIGN §5.3.3) | wrong | `from_signals` never called; its check `exists(path)` is true at base (W12). The doc still lists "Signal" as a source |
| R7 | Filters (a) executable (b) monotone (c) non-trivial (d) predictive (DESIGN §5.3.3, ARCH) | partial | (a)-(c) in `filters.py`, applied only by the toy pipeline with a relaxed `min_monotone=0.6` (W3). Coding SWE-rebench milestones are used unfiltered (W10). (d) is not applied to any M5 graph (M6 fits type weights separately) |
| R8 | Stage induction: ≥ 90% precedence + LLM names, membership from data (DESIGN §5.3.4) | partial | Built; deterministic for a fixed input order. Cycles from declared prerequisites make the layering depend on input order (W16); default names collide (W18); no LLM namer |
| R9 | Tracker: stage, Φ, satisfied / regressed / next-expected, deviations (forbidden actions, policy violations); cost policy (DESIGN §5.3.5) | partial | Offline tracker works on the toy. Deviations cover regressions and probes only, no policy violations. No runtime or World calls `WorkflowTracker` (only `TP/m56/pipeline.py:193`). On a live World every mined coding milestone is unmeasurable (W1). Crashes on a stage/milestone mismatch (W15) |
| R10 | Tracker vs manual stage labels κ ≥ 0.7 (ARCH) | partial | Toy κ 0.8685 reproduces, but only with the pipeline's `min_monotone=0.6`; the module default gives κ 0.2756 (W3). Labels, tracker and pipeline land in one commit (`bf95267`), so "labels written before the code" is not checkable from history; builder-labelled. No coding labels |
| R11 | Rediscovery: mined F1 milestones cover build, API removal, bit-exactness, edge semantics, concurrency, perf (ARCH) | wrong (as reported) | Gate fails (bit-exactness) as documented, but the "5 of 6, hand-audited" claim is overstated: only 2 families were audited; under the same standard edge semantics has 1 PR; the committed report is stale for build (W2) |
| R12 | Coding: F1 graph from upstream PRs and our transcripts (ARCH) | missing | No coding `WorkflowGraph`, stages or tracker run exists (`grep WorkflowGraph.build` hits only the toy pipeline and tests) |
| R13 | Toy: helpdesk graph triage → diagnose → resolve → confirm (ARCH) | partial | Mined graph: `assignee` → `resolution` → `status=resolved` (3 stages named after fields) |
| R14 | DEPEND Harbor `trajectory.json` prefix seeding, agentenv `snapshot_env` / `snapshot_agent_state` (ARCH Takes) | missing | Nothing in `C/workflow/` imports or calls them |
| R15 | Φ (§6.2) in [0,1], prerequisites gate credit, unmeasured ≠ 0 | partial | `potential.py` correct for valid inputs; `WorkflowGraph` skips the `MilestoneSet` checks, so weight ≤ 0 gives Φ = -1 and an unknown prerequisite zeroes a milestone forever (W15). First ΔΦ is measured from 0, not Φ(s0) (W19) |
| R16 | M4 fit: stable signal ids, shared toy outcome rule, App writes on World traces | partial | Toy outcome shared and gates unchanged. Signal ids only reach the dead `from_signals`. App writes reach `Step.writes`, but `logstate` stores the event type instead of the written value (W13); M4 drops failed calls' writes, M5 replays them (W8) |

## 2. Defect candidates

### W1 (P1) [ran] Mined milestones cannot be measured on a live World
- **Where:** `C/workflow/milestones/mine.py:111-115,141-145` emit `grep {pattern, paths:[file]}` and `block_changed`; `C/world/state/fs.py:152` `q_grep(pattern, glob, max_hits)` has no `paths`; no `q_block_changed`; `C/world/state/base.py:65` calls `handler(**q.args)` and catches only `ExecError/OSError/ValueError/KeyError`, so the `TypeError` escapes to `evaluate()`, which turns it into `None`.
- **Input (p8):** base `k.cu: dst[0] = tex1Dfetch(t, 0);` plus `other.cu` with another `tex1Dfetch`; patch → `__ldg(src)`; evaluate the 3 mined milestones on `FileStateView(end)` and on `FsAdapter("repo", LocalExec(), {"root": tmp})` holding the same end files (what `VerifierView.query` forwards to).
- **Observed:** `live grep with paths raises: FsAdapter.q_grep() got an unexpected keyword argument 'paths'`; `located`, `removed_call:tex1Dfetch`, `added_call:__ldg`: offline φ = 1.0, live φ = None.
- **Expected:** the same milestone gives the same φ on both views (`evaluate.py:1-9` promises this). Without `paths`, the live grep would also count `other.cu`, so `removed` would read 0 and `added` could hold at base.
- **Impact:** forged tasks write these milestones to `milestones.yaml` (`CP/forge/derive.py:227,415`). Any live Φ, online tracker or dense channel over them is all-None (Φ 0, `partial`). Latent today because nothing evaluates them live.
- **Fix:** add `paths` to the fs snippet's `grep` and a `block_changed` op (needs the base content: a `git show HEAD:path` in `fs_git`), or make the miner emit only ops the adapter supports. Add a parity test that runs a mined set on both views.

### W2 (P1) [ran] Rediscovery is stale and over-credited
- **Where:** `CP/m56/rediscovery.py:105-111` (families from any candidate, ignoring `change`), `:132-146` (audit covers only bit_exactness and perf; perf hard-coded `True`); report `docs/modules/reports/m5_rediscovery.json` (commit `bf95267`, before `7e3e5b1` added `code_file` at `mine.py:140`).
- **Input (p7):** `rediscovery.run()` with `REPORT` pointed at the scratchpad, plus a per-PR re-mining that keeps `evidence.change`.
- **Observed:**
  - build: committed 29 hits / 11 PRs → re-run **3 hits / 2 PRs** (`#5953` CMake `check_language(`, `CMAKE_CUDA_COMPILER_VERSION`; `#4402` `CUDA_ARCH_BIN`). The 26 lost hits were `located` anchors in `CMakeLists.txt`, which mining no longer emits.
  - api_removal: 10 of 47 hits are **added** constructs (`cudaBindTexture(`, `cudaBindTexture2D(`, `cudaUnbindTexture(` added in `#6010`; `deprecated` added in 5 files). Adding a deprecated API is not API removal.
  - edge_semantics: 34 of 40 hits are `located` blocks whose *names* contain Bound/Edge (`getBoundarySmoothing`, `OrganizedEdgeFromRGBNormals`, `setHalfEdgeDataCloud`...) from refactor PRs (`#2728` "use override", `#3006` "Use nullptr", `#3121` "using instead of typedef", `#3180` "Remove else after return"). The removed `fov_*_bound_(` (`#5722`) is a member-init refactor. Only `isfinite(` added in `#6460` is behavioural: **1 PR**.
  - concurrency: 34 of 35 are `__ballot_sync/__all_sync/__any_sync` renames (also counted as api_removal for the removed `__ballot/__all`) and `lock_guard`; plausible, unaudited.
  - perf: one diff hit; the verdict rests on review-comment keyword matches (`text_families`), not on mined milestones.
- **Expected:** the module doc's "5 of 6, hand-audited" holds only if all six families were audited at the bit_exactness standard. Under that standard edge semantics falls below 2 PRs: **4 of 6** (build, API removal, concurrency from code; perf from review text).
- **Fix:** regenerate the report on HEAD; count only `removed`/`located` for api_removal; match families against the changed lines, not block names; audit every family's evidence the same way and store the audit next to each hit; correct `M05-workflow.md` and DESIGN §5.3 "As built".

### W3 (P1) [ran] Toy κ gate holds only under a relaxed, undocumented filter threshold
- **Where:** `TP/m56/pipeline.py:166` `screen(..., min_monotone=0.6)`; module default `filters.py:57` `min_monotone=0.9`; stage → label mapping by position `pipeline.py:199-200`.
- **Input (p3):** re-run `P.run()` with `screen` wrapped to drop the `min_monotone` override.
- **Observed:** `log/status=resolved` is 0.875 monotone over good tickets, so it is dropped; kept `['log/assignee','log/resolution']`; **κ 0.2756** (vs 0.8685).
- **Expected:** a gate value that does not hinge on a threshold set in the same commit as the labels (`bf95267`) without a recorded reason. The label file says it was written "before any toy milestone/stage/tracker code existed", which git history cannot show.
- **Fix:** record the threshold choice (chosen on dev only) in the graph's stats; report κ at the module default too; have the 4-word vocabulary map to stages through an explicit, dev-fitted mapping instead of "number of stages == len(vocab) - 1".

### W4 (P1) [ran] PR steps are ordered by the time string
- **Where:** `C/workflow/steps.py:119` `sorted(trace.events, key=lambda x: (x.time or "", x.seq))`. `CP/connectors/github_pr.py:262-266` sorts by instant because commit times keep the committer offset (`%cI`) and API times are `Z`.
- **Input (p5):** commit c1 `2024-01-01T10:30:00+02:00` (08:30Z), review `09:00:00Z`, commit c2 `09:30:00Z`.
- **Observed:** `[('review', ...09:00Z), ('commit','c2', 09:30Z), ('commit','c1', 10:30+02:00)]`; an untimed commit sorts first (`['c2-untimed','c1']`).
- **Expected:** c1, review, c2 (instant order); untimed events keep their seq position.
- **Impact:** commit/review steps (and the review rounds merged by actor) are reordered for any committer east or west of UTC, which moves step credit across review rounds.
- **Fix:** keep the connector's order (events are already sorted), or sort by a parsed instant with seq as tie-break.

### W5 (P1) [ran] Lines that start with `*` are skipped as comments, so the canonical texture migration is not mined
- **Where:** `C/workflow/milestones/mine.py:62` skips any stripped line starting with `*`.
- **Input (p6 §5):** `k.cu` line `    *dst = tex1Dfetch(t, 0);` changed to `    *dst = __ldg(src);`.
- **Observed:** candidates `['diff/located/k.cu:k']` only. Control (`dst[0] = ...`): `located`, `removed_call:tex1Dfetch`, `added_call:__ldg`.
- **Expected:** the removal of `tex1Dfetch` (API removal) and the added `__ldg` (perf) mined in both spellings. Pointer writes (`*out = ...`, `*p++ = ...`) are common in CUDA kernels; Markdown/RST bullets are also dropped.
- **Fix:** treat `*` as a comment only inside a `/* ... */` block (track block-comment state per file) or when followed by space and the previous line is a comment.

### W6 (P1) [ran] Shell replay silently drops edits (no `unreplayable` reason)
- **Where:** `C/workflow/shellops.py:227` (`a.startswith("-i") or a == "--in-place"`), `:62-66` (shlex splits `2>` into `2`,`>`; `commenters="#"`), `:174` (mv/cp need exactly 2 non-flag args), `:139` (`(cd` is not `cd`), `:230` (only `python`/`python3`), `:160-173` (`rm -r` deletes only files already loaded), no `tee`/`dd`/`truncate` handling.
- **Input (p1):** each command alone on `src/a.py`, `src/b.py`, `README`, cwd `/repo`.
- **Observed:** `applied=[] unreplayable=[]` for `sed -Ei 's/x/z/' src/a.py`, `sed --in-place=.bak ...`, `perl -pi -e ...`, `echo hi | tee src/new.py`, `mv src/a.py src/c.py 2>/dev/null`, `cp ... 2>/dev/null`, `(cd src && rm a.py)`, `python3.11 -c "open('src/a.py','w')..."`, `dd ... of=src/a.py`, `truncate -s 0 src/a.py`, `rm -rf src`, `echo a#b > src/h.txt`. Plain `sed -i` is reported. `actions.classify` calls most of these `edit` (so the classifier and the replayer disagree); `tee` and `echo a#b >` are `investigate`.
- **Expected:** either the effect is applied or the call is listed as unreplayable. Today intermediate states are wrong without a trace (`n_unreplayable` undercounts; p2 §2: `sed -Ei` then `Write` → `ok=True`, `problems=[]`), and Φ at mid-trace fractions (`rebench_phi.py:61-65`) is computed on those states.
- **Fix:** reuse M4's `shell_writes` (`core/agent_events.py:35`) as the authority for "this command writes X": any written path not applied by `apply_shell` becomes an unreplayable reason. Match `-[a-zA-Z]*i` and `--in-place=*`; handle `N>` tokens; strip `(`/`)`; match `python[0-9.]*`.

### W7 (P1) [ran] Shell replay applies wrong effects
- **Where:** `shellops.py:209-223` (`git restore` with any flags restores the working file), `:181-185` (`mv x dir` without trailing `/`), `:147-152` (printf treated like echo), `:109` (every segment applied regardless of `&&`/`||`), `:145-146` (unquoted here-doc taken verbatim).
- **Input (p1) → observed:**
  - `git restore --staged src/a.py` → `src/a.py` reset to base (`--staged` only unstages);
  - `mkdir -p pkg && mv src/a.py pkg` → a *file* `pkg` with a.py's content;
  - `printf '%s\n' hello > src/p.txt` → content `'%s\n hello'` (bash: `hello\n`);
  - `false && rm src/a.py`, `test -f src/zz.py && rm src/a.py` → `src/a.py` deleted;
  - `cat > src/x.py <<EOF\nv = $HOME\nEOF` → `v = $HOME` written literally.
- **Expected:** `--staged`/`-S` without `-W` leaves files; a known directory target gets `dir/<name>`; printf with format arguments, conditionals and unquoted here-docs with `$` are reported unreplayable.
- **Fix:** as listed; track directories created by `mkdir` and seen in paths; treat any segment after `&&`/`||` whose guard cannot be decided as unreplayable.

### W8 (P2) [ran] Failed tool calls are replayed as if they succeeded
- **Where:** `C/workflow/replay.py:95-146` never reads the `tool_result`. `C/world/world.py:997-1000` (M4) excludes calls whose result has `ok: False` from `writes`.
- **Input (p2 §1):** `Write /a.py "x = 99"` whose result is `{"ok": false, "is_error": true, "output": "File has not been read yet..."}`; final patch empty.
- **Observed:** `ok=False`, `final:a.py` differs; trace quarantined.
- **Expected:** a refused call changes nothing (Claude Code refuses Write/Edit before a Read; OpenHands returns errors for bad `insert_line`). M4 and M5 should agree on what a call wrote.
- **Fix:** skip calls whose linked result is an error (`ok: False`, `is_error`, `exit_code != 0` for editor tools) and list them in `meta["skipped_failed"]`.

### W9 (P2) [ran] Diff parser: renames ignored, one binary file aborts everything, no-prefix diffs fail
- **Where:** `C/workflow/filestate.py:168-178` (only `new/deleted file mode` header-only entries), `:186-188` (`/dev/null` kept as a path), `:115-116` (binary raises), `strip=1` hard-coded in `replay.py:154` and `mine.py:122`.
- **Input (p2 §3-4, p6 §1):**
  - `diff --git a/a.py b/c.py / similarity index 100% / rename from a.py / rename to c.py` → `parse_patch` returns `[]`; an agent `mv a.py c.py` then "differs" on both files;
  - `Binary files /dev/null and b/img.png differ` + a text hunk → `('/dev/null','b/img.png',True)`; replay quarantined with `binary patch for img.png` **even with `ignore=*.png`**; a binary deletion errors as `binary patch for dev/null`. One rediscovery PR is lost to this (`per_pr` error `binary patch for dev/null`);
  - `--- src/mod.py / +++ src/mod.py` (no prefix) → `EditError: hunk @@ -1,6 does not apply to mod.py`.
- **Fix:** parse `rename from/to` (and `copy`); map `/dev/null` to None in the binary branch; record binary files as opaque changes (or honour `ignore` before raising); detect the prefix (`a/`,`b/`) instead of always stripping one component.

### W10 (P1) [ran] Coding milestones skip every filter; changelog prose becomes milestones
- **Where:** `CP/m56/rebench_phi.py:83` `from_diff(..., include=lambda p: not TEST.search(p))` with no `screen`; `mine.py:140` excludes docs only from `located`, not from tokens.
- **Input:** `data/m56/rebench_milestones_n80_s7.jsonl` (660 milestones); p6 §2b patch adding `Fix tz handling. Thanks to Marcin Sulikowski in PR #433.` to `CHANGELOG.rst`.
- **Observed:** 101 of 660 milestones sit on `.rst/.md/.txt`/CHANGELOG files; the probe yields `added/CHANGELOG.rst:{Sulikowski, handling, Marcin, Thanks}`. The first cached row (`adamchainz__time-machine-433`) carries `diff/added/CHANGELOG.rst:Sulikowski`.
- **Expected:** filters (a)-(c) (and a docs/prose exclusion) before milestones feed Φ, S2 and the M6 type weights. A correct alternative fix never names the contributor.
- **Fix:** run `screen` (base, expert end, attempts as series) in `rebench_phi`; skip tokens from prose files by default.

### W11 (P2) [ran] A `located` anchor can bind to a forward declaration
- **Where:** `mine.py:99` anchor = escaped definition line; `evaluate.py:162-186` `block()` takes the first match.
- **Input (p9):** `a.c` = `int helper(int x);` … `int helper(int x)\n{\n    return x + 1;\n}`; patch changes `+ 1` to `+ 2`.
- **Observed:** anchor `^\s*int\ helper\(int\ x\)`; `block()` returns `'int helper(int x);'`; φ at the **expert's own end state = 0.0**.
- **Expected:** 1.0 at the expert end state. Unfiltered pipelines (W10) keep such milestones and every attempt loses that weight.
- **Fix:** anchor on the definition (skip matches ending in `;`), or anchor by occurrence index.

### W12 (P2) [ran] Signal-derived mining is dead and trivially true (still)
- **Where:** `mine.py:195-209`; no caller.
- **Input (p6 §3):** a `correction` signal on `file:src/mod.py`.
- **Observed:** check `exists {'path': 'src/mod.py'}`, φ at base 1.0, `non_trivial → False`.
- **Impact here:** the rediscovery "perf from review signals" verdict does not come from this source but from keyword matching review text (W2).
- **Fix:** as `M03-M05.md` M5 #1 (`block_changed(path, anchor=target.span)`), then route M4 signals into mining.

### W13 (P2) [ran] Event-log state stores the event type for World App writes
- **Where:** `C/workflow/logstate.py:59-60,72-73` read the value from top-level `data` fields; World `tool_call` events keep values under `data.arguments` (`core/agent_events.py:110-137` derives `ticket:T7.status` from them).
- **Input (p5 §4):** `tool_call` `tickets.update_ticket {"ticket_id":"T7","status":"resolved"}` with `writes = app_writes(...)`.
- **Observed:** `['ticket:T7.status']` → logstate objects `{'ticket:T7.status': 'tool_call'}`.
- **Expected:** `'resolved'`, so `log/status=resolved` can hold on a recorded World episode.
- **Fix:** for `tool_call` events take the value from `arguments[<field>]` when the write names `<obj>.<field>`.

### W14 (P2) [ran] Writes classified `investigate` and lost on merge
- **Where:** `actions.py:115-127` (`tee` not an edit; `#` comment cut by shlex in `_redirect_targets`), `steps.py:93-99` (the merge keeps `prev.writes`, drops the call's).
- **Input (p10):** `cat src/a.py`, then `echo 'x = 2' | tee src/a.py` with M4 `writes = ['file:src/a.py']`.
- **Observed:** one step `(0, 'investigate', [0, 1], [])`.
- **Expected:** an `edit` step carrying `file:src/a.py` (M4's `shell_writes` already counts `tee`).
- **Fix:** classify as `edit` whenever `data.writes` is non-empty (or `writes_of` returns paths); never merge a call that writes.

### W15 (P2) [ran] `WorkflowGraph` accepts invalid milestone sets; the tracker crashes on a stage mismatch
- **Where:** `graph.py:40-43` (no `MilestoneSet` validation), `potential.py:37-44`, `tracker.py:64-65` (`next(...)` inside a list comprehension).
- **Input (p4 §4-5):** milestones `p` (weight -1), `q`, `r` (requires `dropped`); a stage listing `["a", "ghost"]` with only `a` in the graph.
- **Observed:** Φ(all satisfied) = **-1.0**; `r` can never count; `observe()` raises `RuntimeError: coroutine raised StopIteration`.
- **Fix:** validate in `WorkflowGraph` (weights > 0, prerequisites and stage members ⊆ milestones), as `MilestoneSet` does (`core/schema/milestone.py:62-68`).

### W16 (P2) [ran] Stage layering depends on milestone order when prerequisites contradict the data
- **Where:** `stages.py:83-99` (declared `requires` edges added unconditionally; a cycle is cut where the DFS first meets it).
- **Input (p4 §2):** data `x` at 0.1, `y` at 0.9 in 5 traces; `x.requires = ["y"]`.
- **Observed:** order `[x, y]` → stages `[['y'], ['x']]`; order `[y, x]` → `[['x'], ['y']]`.
- **Fix:** detect the contradiction and report it (or drop the weaker edge by support) before layering; sort ids before the DFS.

### W17 (P2) [ran] `match()` reports a validly achieved milestone as out of order
- **Where:** `match.py:36-43` (`out_of_order.setdefault` before the prerequisite is processed in the same row).
- **Input (p4 §1):** `b.requires=[a]`, list order `[b, a]`, both hold at state 0.
- **Observed:** `achieved {'log/a': 0, 'log/b': 0}`, `out_of_order {'log/b': 0}`.
- **Fix:** record out-of-order only after the fixed-point loop for that row ends, for milestones still unachieved.

### W18 (P2) [ran] Default stage names collide
- **Where:** `stages.py:123-131`.
- **Input (p4 §3):** 4 `diff/added/a.py:tok{i}` milestones achieved at 0.1/0.2/0.7/0.8 in 4 traces.
- **Observed:** 4 stages, all named `added a.py`.
- **Fix:** suffix the index or the dominant token; names must be unique within a graph.

### W19 (P2) [ran] Φ(s0) is never observed
- **Where:** `tracker.py:48,70` (`prev_phi = 0`), `potential.py:57,74` (`initial` defaults to 0); toy non-triviality uses the state **after** the first event of one good ticket (`pipeline.py:163`) and `end_phi = max` over good tickets (`:165`).
- **Input (p4 §7):** milestones `a, b`; first observed state already has `a`.
- **Observed:** `delta_phi = 0.5`, `gained ['a']` at step 0.
- **Expected:** the tracker seeds from the base state (or reports Φ(s0)), so a milestone true before any action is not credited to the first step.
- **Fix:** `WorkflowTracker.start(view)` evaluating s0; non-triviality on every training trace's base.

### W20 (P2) [ran] Agent-state replay excuses authored files nobody named
- **Where:** `replay.py:166-172` (basename substring over all arguments).
- **Input (p2 §5):** final patch creates `gen_out.py`; the only call is `python tools/codegen.py`.
- **Observed:** `ok=False`, `ok_agent_state=True`, `byproducts=['final:gen_out.py']`.
- **Expected:** a source file in the submitted patch counts against agent-state replay (byproducts are logs and caches). The headline 94.9% uses agent-state.
- **Fix:** treat only non-source paths (logs, caches, `__pycache__`, build dirs, harness files) as byproducts; report how many passing traces relied on the exemption.

### W21 (P2) [ran] S1 evidence is not reproducible from the repository and measures files, not CI
- **Where:** `CP/m56/gate.py:26-35` reads `data/m56/rebench_phi_n80_s7.jsonl` (untracked: `git log` on it is empty; mtime 2026-10-09 18:22) and recomputes nothing.
- **Observed:** re-running the gate reproduces 866/913 and 832/913 from the cache; the rows have no code version, so they cannot be tied to the current `replay.py`/`shellops.py`.
- **Expected:** DESIGN §5.3.2 defines replay as reproducing recorded CI outcomes; S1 here is final-file equality. That is a defensible proxy, but the doc should say so and the rows should carry the replayer version.
- **Fix:** stamp rows with a code hash; add a small checked-in replay fixture set (≥ 30 traces) so S1 re-runs offline.

### W22 (P2) [ran] Toy replay checks are partly tautological
- **Where:** `pipeline.py:118-120`: the rebuilt status is compared with the last recorded status, both read from the same events.
- **Observed:** cannot fail by construction. The policy checks are one-sided (only where a flag was recorded); a probe for unflagged firings (refund over 50 without approval and no flag) found none in the 24 logs, so no false negatives today.
- **Fix:** add the converse check (no rule fires on the rebuilt state where none was recorded) and drop the status tautology from the count.

### W23 (P2) [ran] Another actor's message is folded into this actor's step
- **Where:** `steps.py:84-100` (pending messages attach to the next `tool_call` regardless of actor).
- **Input (p5 §2):** `agent_message` by `subagent`, then an `Edit` by `main`.
- **Observed:** one step `('main', 'edit', [0, 1])`.
- **Fix:** flush pending events of a different actor as their own step (as `_log` does at `steps.py:149-150`).

## 3. Severity summary

- **P1 (8):** W1 live/offline milestone mismatch; W2 rediscovery stale and over-credited (4 of 6, not 5 of 6); W3 κ depends on `min_monotone=0.6`; W4 PR step order by time string; W5 deref lines dropped from mining; W6 silent shell drops; W7 wrong shell effects; W10 unfiltered coding milestones.
- **P2 (15):** W8, W9, W11-W23.
- **No P0:** no defect found that corrupts a training reward today, because the live paths in W1 and W13 are not wired yet.

## 4. Coverage

Command (from `crucible/`, coverage data written to the scratchpad so the tracked `.coverage` is untouched):
`.venv/bin/python -m pytest tests/unit/test_workflow.py tests/unit/test_workflow_stages.py tests/unit/test_probes.py tests/unit/test_credit.py tests/unit/test_runtime_records.py -q -p no:cacheprovider --cov=crucible.workflow --cov-report=term` → **59 passed**, total **78%**.

| File | Cover | | File | Cover |
|---|---|---|---|---|
| `shellops.py` | 64% | | `filestate.py` | 79% |
| `milestones/evaluate.py` | 66% | | `logstate.py` | 82% |
| `steps.py` | 68% | | `stages.py` | 88% |
| `replay.py` | 79% | | `milestones/mine.py` | 87% |
| `actions.py` | 90% | | `potential.py`, `filters.py`, `match.py`, `tracker.py` | 88-91% |

The least-covered files are where most defects sit: `shellops.py` (W6, W7: no test for `-Ei`, `perl -pi`, `tee`, `2>`, `git restore --staged`, conditionals), `steps.py` (W4 and W23: no PR test with mixed offsets or interleaved actors), and `evaluate.py` (W1: no live/offline parity test). No test covers renames, binary files or no-prefix diffs (W9), failed tool calls (W8), C deref lines (W5) or invalid graphs (W15).
