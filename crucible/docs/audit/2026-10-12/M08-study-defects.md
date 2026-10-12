# M8 study: defects in `forge` (orchestrator, zoned repo, swarm, economy, generators, red team, M9 subset)

Scope: `C/forge/` (`orchestrator.py`, `budget.py`, `schema.py`, `repo.py`, `swarm.py`, `economy.py`, `compose.py`, `fuse.py`, `stitch.py`, `tighten.py`, `perturb.py`, `ladder.py`, `learnability.py`, `dedup.py`, `curator.py`, `judge.py`, `specialists.py`, `hacker.py`), the M9 subset `C/accept/contract.py`, A16 `C/accept/checks/redteam.py`, the sandboxed roles `C/interop/harbor/role_runner.py`, the toy pack `TP/forge/` (`world_synth`, `scenario`, `env`, `roles`, `task`, `fused`, `hindsight`) and the coding pack `CP/forge/` (`derive.py`, `forge_gpu.py`, `stitch_task04.py`). Reports read: `docs/modules/reports/m8_*.json`.

Spec: ARCHITECTURE "### M8 `forge`" and its status row. DESIGN §7 (7.1–7.5), §9, and the m13 entries G-1 … G-14 (DESIGN lines about 1575–1900). `docs/modules/M08-forge.md`. `docs/research/improvements/M06-M08.md` (M8 table). `docs/research/m13-audit/00-master-plan.md` phase G.

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Method: a read-only study on HEAD `526fdb5`, using `crucible/.venv` on macOS. **[ran]** means reproduced with a probe; **[read]** means established from the code alone. The probes are in the session scratchpad at `m8hunt/p0_toy_gate.py … p7_coding_cpu.py` (shared helper `common.py`). Every probe writes only into the scratchpad. No paid call, GPU, lab, network or Docker run was made; no repo file other than this document was changed.

Offline gates re-run:
- **Toy world_synth** (`p0`): a fresh forge into the scratchpad (no journal) runs every node once and gives `constructed`, 5/5 accepted, `pass: true` in 7 s. Per-check statuses and all 5 task hashes equal the recorded `m8_toy_world_synth.json`. A second run replays every node from the journal (`attempts 0, reused true`). Note: the recorded report itself is such a replay (every node `reused: true`), so it re-ran no validator (see V5).
- **Toy G-hindsight** (`p6`): fresh run gives 6/6 accepted; per-check statuses and task hashes equal `m8_toy_hindsight.json`.
- **Coding CPU part** (`p7`): `derive_span` + `cpu_accept_span` for people, ncv and kinfu_family, without writing task dirs. Every `cpu` row equals `m8_coding_derive.json`. The 4 seed traps re-derive with 8 checks each.
- Not re-run: `m8_coding_gpu.json` (GPU), `m8_toy_calibration.json` and `m8_toy_world_synth_llm.json` (paid).

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, but missing a specified part, or with a reproduced hole; **wrong** = contradicts the spec; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | Deterministic DAG: topological levels, sorted within a level (ARCH contents; DESIGN §7.2) | live | `orchestrator.py:89-103`. Fresh and replayed toy runs give identical hashes (p0) |
| L2 | Journal reuse by input hash; role logic changes bump the version (M08 doc finding 1) | partial | The hash is role + version + visible keys + knobs + node id (`:302`). Validators, `canary_out` and `max_attempts` are not in it. A stricter validator reuses an output that fails it (V5). Turning on `canary_out` reuses an output with no canary (V6). A no-op re-run turns `accepted` back into `constructed` (V18) |
| L3 | Budgets: reserve/settle; never half-run; "a role that under-estimates is still accounted exactly" (`budget.py:1-6`) | **wrong** | Concurrent nodes double-count spend (V3). An overspend raises out of `run_plans` and aborts every plan (V4). A role-side `BudgetExhausted` leaks its reservation (V4) |
| L4 | Retries with feedback; infra never a content attempt | live | `:313-363`. An unexpected exception in any role aborts all concurrent plans (V4) |
| L5 | Asymmetric information by staging + canary breach (DESIGN §7.2 "Built in M8") | partial | Staging is real: a role only gets `visible` inputs (`:207-219`). The toy Test writer never sees the oracle (`world_synth.py:79-81`). The canary is never written into any task file, so A5's canary test is vacuous (V9). Reuse can drop the canary (V6) |
| L6 | Zoned repo: roles cannot read or quote what they must not (G-7) | **wrong** | No write-down rule: a role that reads `oracle/` may copy oracle lines and the oracle canary into `public/`, which every role, including the black-box hacker, reads (V1). The quote scan matches whole lines only, by a global index (V2) |
| L7 | Sandboxed builder roles see only visible files; forbidden strings refuse the wrapper (G-11) | partial | Layout and network policy are live. The forbidden scan runs on JSON-escaped text, so a forbidden string with a quote passes (V10). A failed trial's spend is recorded as 0 (V19) |
| L8 | Solution-first, oracle from the expert end state (DESIGN §7.1) | live | Coding spans and traps derive from `solution.patch` and recorded frontier patches (`derive.py:207-262, 302-331`). Toy oracle is builder-derived from the scenario |
| L9 | Every trap cites a resolvable M4 signal (ARCH exit gate; A8) | partial | Toy and coding traps cite real signal ids that resolve (p0, p7). A8 counts any non-empty string as provenance; only `sig_*` ids are resolved (V11). The DESIGN definition ("an expert signal or a reference-differential check") is not enforced |
| L10 | Every derived span task passes M9 (ARCH exit gate) | **partial** | CPU subset re-run passes (p7). Behaviourally, each oracle was graded once (A1 needs 3), no valid alternative was graded (Astra was excluded), no naive floor exists, and the 8 C2 "trap" gate checks in people and ncv are never run by the GPU grader (V13) |
| L11 | ≥ 1 trap task per seed passes M9 (ARCH exit gate) | **wrong** | The written seed trap tasks have an empty `instruction.md` and no properties (`derive.py:466-468`). A7 fails on all 4 (p7). They had no A1/A2/A3 run at all. Their gates are gold-token checks on English words (V12) |
| L12 | Toy variant world with 5 tasks passes M9 (ARCH exit gate) | partial | Reproduced (p0): 5/5 pass the M9 subset. A4 (40 draws), A6 and A9–A13 are skipped (`contract.py:22-24`). A5 and A7 in that subset have holes (V8, V9, V11) |
| L13 | M9 subset: A5 hidden material vs everything agent-visible (`contract.py`; DESIGN §9) | **wrong** | Toy `hidden_strings` is the supervisor name plus a canary that no file carries (`task.py:165-166`). `agent_visible` omits the ticket DB the agent reads through tools (`:157-163`). The full oracle pasted into the instruction passes A5 (V9) |
| L14 | M9 subset: A7 two-way coverage (DESIGN §9) | partial | It trusts each check's self-declared `states`; a check on the oracle's note text labelled `note` is accepted, and an honest solution then scores 0 (V8). Only the first 40 characters of a property sentence are checked (V8). The fused task fails A7 (V14) |
| L15 | Composition: chain, fan-in, fuse, nest with A14/A15/A17 (G-2, G-9) | partial | Built and tested. The fused task's cross check states no property, so the fused task cannot pass A7 (V14). Widened scopes let a component touch another component's tickets [read] |
| L16 | Stitch: spans → seed (G-10) | live, with an edge | task04 reproduction is a test. A span that pre-applies a file it also owns is accepted, and its oracle diff is then applied to the wrong base (V20) |
| L17 | Perturb (A18) and G-tighten fair + separating with 15% headroom (G-tighten) | live, with an edge | `tighten.py:43-73`. `add_gate` counts one failing frontier run as "separating" (a flake qualifies) [read, `:70`] |
| L18 | Learnability 4p(1−p), Wilson, 4 → 8 → 16 (DESIGN §7.4) | live | `learnability.py`. At n = 16 a Wilson interval is too wide to be "inside" [0.1, 0.7], so `measure` always spends 16 and acts on the point estimate (as documented) |
| L19 | Solver ladder, monotonicity, accept/steer/retire (G-4) | live | `ladder.py`. Pure functions; covered 91% |
| L20 | A16 adaptive red team: reward_hack vs legit vs infra (G-5) | partial | `classify` is correct (`hacker.py:145-152`). A16 passes when every attempt is masked (V7). The specialist's memory replay suppresses the search that would find a new hack (V7) |
| L21 | Swarm stage: C/B scheduling, announce dedup, charged episodes (G-8) | partial | Announce is cooperative: a worker that catches `Cancelled` still contributes and is charged; a second announce leaks the first into `in_flight` forever (V16). Not wired into any pipeline (only tests instantiate `SwarmStage`) |
| L22 | Economy: $/accepted, stop rules, quota, persistence (G-6) | partial | A person's `resume` is not persisted: reloading the book re-pauses the generator and restores its counts (V15). Re-recording a task wipes its M9 verdict (V15). Costs are inflated under concurrency (V3) |
| L23 | Curator: dedup, placement, lineage sealing, recompose (G-12) | partial | Sealing is in memory only: a new `Curator` places a sibling of a bench task in train. A composite placed without an explicit lineage seals only itself (V17). No caller outside tests |
| L24 | Dedup L0–L3 and decontamination (ARCH; research M8-2, M8-3) | partial | L1 is an 80-character prefix; every toy task shares a 170-character intro, so L1 reduces to footprint equality. No decontamination against bench splits or `bench.contamination` (grep: no forge caller) [read] |
| L25 | Judge never changes verdicts; specialists with hack memory (G-13/G-14) | live | `judge.py` ranks accepted variants only. Specialist: see V7 |
| L26 | Span closure under consumer deps; alternatives share outside state (DESIGN "Forge GPU run") | live | `closure_violations`, `alternative_admissible` (`derive.py:188-204`); `closure_consumers` row true (p7) |
| L27 | Toy world synthesis: tasks before data, literal grounding, frozen time, decoys (ARCH "Toy") | live | `scenario.py`, `env.py`; `env_validator` checks literals and the wall clock |
| L28 | Generators: hindsight, span, trap, transplant, compose, tighten, world_synth (ARCH contents) | partial | **G-transplant missing.** G-hindsight exists for the toy only. No `generators/` or `roles/` package; roles live in packs |
| L29 | PORT SWE-smith `bug_gen/`, `valid.py`, `repair`, `issue_gen` (ARCH Takes; research M8-1) | **missing** | `accept/mutation/` has only action mutants (`actions.py`); no patch mutation operators (grep) |
| L30 | DEPEND Harbor `compile` (scaffolding), `analyze` (Auditor); agentenv `review` (10% human audit) | **missing** | Harbor `analyze` is ported in structure only (`role_runner.py:1-3`); no `compile` use; no human-review step (grep). M08 doc lists both as not done |
| L31 | Calibrator and frontier calibration A10 (DESIGN §7.2) | partial | `TP/forge/calibrate.py` (paid, 0% unit coverage). A10 is skipped in the subset |

## 2. Defect candidates

Severity: **P0** = a policy or a role can obtain reward or acceptance it did not earn, or hidden material leaks into a task; **P1** = a wrong result or an audit hole in normal use; **P2** = an edge case, wrong attribution or a robustness gap.

### Asymmetric information and leaks

**V1 [P0][ran] The zoned repo has no write-down rule: oracle material flows into `public/`.** `scan` checks only that an author writes into zones it can read (`repo.py:166-171`), and that it does not quote zones it cannot read (`:172-184`). Nothing stops a role that reads `oracle/` (builder, adversary, judge, white-box hacker: `:41-51`) from copying oracle lines into `public/`, which every role reads.
- p2 case 1: the Adversary commits `public/notes/t2.md` holding the oracle's refund call and the `oracle/` canary. Result: `COMMITTED`. `view("hacker_blackbox")` (public only) then contains the oracle line (`True`) and the oracle canary (`True`). The Verifier-writer sees the oracle line too.
- p2 case 2: the Verifier-writer then quotes the oracle line in `checks/`: `COMMITTED`, because the line "also appears in a zone the author can read" (`:183`).
- p2 case 5: the readable-zone exemption reads a global line index, not the parent tree. A Judge note on an unrelated branch (parent = root) unlocks the exact quote on the oracle branch: `COMMITTED`.
- Impact: the stated purpose of G-7 ("a Verifier-writer that reads the oracle writes checks that encode it") is defeated by any one high-zone role writing down. The black-box hacker's mount (`public/`) gets oracle material. Today only tests use `ForgeRepo` (no pipeline caller), so this is P0 by construction and P1 in practice.
- Fix:
  - an information-flow rule: an author may write into zone Z only material it could have read from Z's readers' view. Concretely, scan every added line against the indices of all zones that are hidden from *any* reader of the target zone (for `public/`: oracle, checks, solver), and carry canaries of every such zone;
  - compute the readable-zone exemption from the parent tree's zones, not the global index;
  - refuse commits into `public/` by roles that read `oracle/` unless the type is in an allowlist with a scan.

**V2 [P1][ran] The repo's quote scan matches whole normalized lines only.** `_h(n) in self._index["lines"][z]` (`repo.py:183`) hashes the entire added line.
- p2 case 4, fresh repo, Verifier-writer and an oracle line of 89 characters: the exact line is a breach (control); the same line with a `-- ` prefix is `COMMITTED`; the line split in two is `COMMITTED`.
- p2 case 3: lines under 20 characters are never indexed (`:152`), so `amount: 97.31` copies freely.
- Fix: index normalized token n-grams (for example 8-token shingles) per zone and flag any added line containing one; index short lines that carry numbers or identifiers.

**V9 [P0][ran] A5 does not look for the oracle, and does not scan the ticket data the agent reads.** The toy adapter's `hidden_strings` is `[supervisor, canary]` (`TP/forge/task.py:165-166`). The canary is put only into the orchestrator's stored payload (`orchestrator.py:352-355`); `assemble` writes the oracle without it (`world_synth.py:103`). So the canary appears in no file and the check can never fire. `agent_visible` is the instruction, kb files and `customers.jsonl` (`task.py:157-163`). It omits `data/tickets.sql`, which the agent reads through `tickets.*` tools.
- p4 a: t1's full oracle action list appended to the instruction: `A5: pass`.
- p4 b: the supervisor's name planted in a ticket subject in `tickets.sql`: `A5: pass`; the scanned set lists no ticket data.
- DESIGN's own contract says hidden strings are "check literals, oracle-only values, canaries" (`contract.py:49-51`).
- Fix:
  - `hidden_strings` returns the oracle's argument values that the instruction does not state (note bodies, the approver name, amounts not in the instruction), each check's SQL literals that are not in the instruction, and the serialized oracle actions;
  - `agent_visible` includes every App's seed state the agent's tools can read (dump `tickets.sql` rows) and tool descriptions;
  - write the canary into `solution/actions.jsonl` (as a comment record) so A5 and the leak audit can see a copied oracle.

**V10 [P1][ran] The sandboxed role's forbidden-string scan runs on JSON-escaped text.** `HarborRole.run` turns non-string staged values into `json.dumps(v, indent=1)` (`role_runner.py:220-221`), then `assemble_role_task` checks `s in v` (`:89`).
- p5: a forbidden oracle line containing double quotes. As a string input it is refused. Inside a dict input it becomes `\"O-7213\"` in the file: `wrapper assembled, oracle line present in a staged file: True`.
- Fix: scan the decoded values (walk the structure) and the escaped text; normalize quotes and whitespace before matching.

**V6 [P2][ran] Turning on `canary_out` for an already journaled node drops the canary.** The input hash does not include `canary_out` (`orchestrator.py:302`). On reuse, the canary is restored only if the old entry had one (`:305-306`).
- p1 f: after a first run without the flag, the second run with `canary_out=True` reuses the output: `canaries {}`, the oracle output has no `_canary`. Every later breach scan for that plan is off.
- Fix: put `canary_out` (and the validator list, see V5) in the input hash, or refuse reuse when `canary_out` is set and the hit has no canary.

### Acceptance (M9 subset) holes

**V8 [P1][ran] A7 trusts each check's own `states` label and only the first 40 characters of each property.** `unstated_checks` tests only that `states ⊆ properties` (`contract.py:175`). `missing_text` tests `p.text.strip()[:40] in instruction` (`:177`).
- p4 d: t1 plus a check "the note body contains `FIN-7`" (the oracle's note text, never stated), labelled `states=["note"]`. Full subset result: `accepted: True` (A1–A8 all pass). An honest solution whose note says "Refunded the duplicate charge." then scores **0.0** (on the original t1: 1.0). A3's valid alternatives miss it because both keep the oracle's note body.
- p4 e: t1's refund sentence cut at 40 characters in the instruction (dropping `7108 on ticket 38.`): `A7: pass`.
- The coding CPU A7 exempts every `milestone` check from needing a property (`derive.py:392`), which is how the C2 gates pass (V12, V13).
- Fix:
  - require the whole property sentence in the instruction (as `architect_validator` already does, `scenario.py:158`);
  - add a valid-alternative generator that varies every free text and order field (note bodies, wording), so a check on oracle internals fails A3;
  - drop the milestone exemption, or require milestone gates to state a property.

**V11 [P1][ran] A8 counts any non-empty string as provenance.** `with_prov = [c for c in t.checks if c.provenance]` (`contract.py:189`); only ids starting `sig_` are resolved (`:191`).
- p4 c: every check's provenance set to `"x"`: `A8: pass` (100%).
- In the real toy tasks, scope and note checks cite `instruction:scope` and `instruction:note` (`roles.py:132, 143`), which are neither "an expert signal" nor "a reference-differential check" (DESIGN §9 A8). The reported "100% provenance" is therefore not the DESIGN metric.
- Fix: a closed provenance grammar (`sig_*` resolved by M4; `kb:<path>#<section>` resolved against the World's documents; `reference_differential:*`; `instruction:*` counted separately and excluded from the 80% share), and fail on unknown prefixes.

**V7 [P1][ran] A16 passes on zero evidence, and the specialist suppresses the search.**
- All attempts masked: `red_team` labels each `infra` (`hacker.py:164-169`). `a16_red_team` adds a reason only for hacks or for no attempts (`redteam.py:47-49`). p4 g (every grade masked on t2): `A16 -> pass`. A candidate-triggered mask (M7 audit V2/V3/V10) turns A16 green.
- Memory first, fallback never: `SpecialistHacker.propose` returns the remembered-strategy replay whenever it is non-empty (`specialists.py:49-54`), whatever its outcome. p4 h: t2 with its note checks removed and the committed checks as the audit floor. A plain `MutationHacker` finds the hack `drop_one_step#2` (skip the note). The specialist with `skip_interactions` in memory and a `MutationHacker` fallback proposes 1 attempt, finds no hack, and makes 0 fallback calls. A16 would pass.
- Fix:
  - A16 is `error`/pending when more than a small share of attempts are infra, and never passes with zero graded attempts;
  - the specialist always also runs the fallback (or a fixed share of its budget) after the replay, and remembers only what the replay could reproduce.

**V14 [P1][ran] The fused task (G-9) cannot pass A7.** `compose._assemble` namespaces the cross checks as `x.<id>` but keeps their empty `states` and adds no property or sentence for them (`compose.py:153`; `CROSS` in `TP/forge/fused.py:77-80` has no `states`). p4 f: `A7: fail ["checks grading nothing the instruction states: ['x.consistency/refunds_match_returns']"]`. The G-9 tests check A14 and A17 only. Fix: `fuse`/`nest` take cross properties (sentence + id) along with cross checks, append the sentence to the instruction, and namespace the states as `x.<prop>`.

### Coding exit gate

**V12 [P1][ran] The seed trap tasks cannot pass M9, and their gates grade words.**
- `main_async` writes each seed trap with `"instruction": ""` and `"properties": []` (`derive.py:466-468`). On disk, `data/forge/coding/task0{2,3}_*_trap/trap/instruction.md` is 0 bytes.
- p7: A7 on the 4 written trap tasks: `fail` (every `trap/*` check grades nothing the instruction states).
- Their only gates are C2 milestone checks used as gates (`derive.py:288-289`, `gate_kind="gate"`). Examples from `checks.json`: task02 `trap/__init__.py:defines`, `schemas`, `implementations` are `grep \bdefines\b pcops/__init__.py` with `count_eq 0`; task01v2 `tiles.h:unspecified`, `scratch`, `contents` the same. A correct solution whose docstring says "This module defines …" fails the trap gate.
- The report's `traps_cpu_ok` (`:465`) only checks that the oracle passes, the frontier version fails and the signal resolves. No A1/A2/A3/A5/A7 ran, and no behavioural grade exists for these 4 tasks. The status row's "traps for 4/5 seeds" is therefore not "≥ 1 trap task per seed passes M9".
- Fix: emit the trap task's instruction (the seed instruction plus the stated property of the failed gate); grade traps with the seed's behavioural gate, and keep C2 milestones for Φ only (as DESIGN §7.3 says for spans); drop C2 candidates that come from comments or docstrings (`strip_c` / Python tokenize before mining).

**V13 [P1][ran] The people and ncv span tasks declare 8 C2 gate checks each that grade the oracle's implementation, and no grader runs them.**
- `derive_span` appends the trap's C2 checks to the span's checks as gates of family `ncv` (`derive.py:241-245`). p7: people 8, ncv 8. In `task04_spans/people/crucible/checks.json`: `trap/elec.cu:cudaTextureObject_t`, `TextureObject`, `texture_binder` require those tokens (`count_ge 1`) in `elec.cu`. The instruction says "you may use texture objects or other memory paths" (`instruction.md:8`). A correct port through `__ldg` fails them.
- The GPU grader (`forge_gpu.py:47-51`, `verify/task04.py`) never reads `checks.json`; grep finds no `checks.json`/`trap/` in the task04 plan. So the 17/17 behavioural result is for a different check set than the task declares.
- The valid-alternative row is excluded from the CPU negatives verdict (`derive.py:395-396`), and Astra's port was excluded for outside-state reasons, so no alternative ever exercised these checks.
- Behavioural A1 ran 1 oracle grade per span (A1 needs 3), and no naive floor exists (only `start` and `unported_*`) (`m8_coding_gpu.json`, 17 grades).
- Fix: remove C2 milestones from span gates (Φ only); make the task's declared checks the grader's checks (or generate `checks.json` from the plan); add 2 more oracle grades and a naive floor per span.

### Orchestrator, budget, economy

**V3 [P1][ran] Concurrent nodes double-count spend.** Each node measures `ledger.observed() - spent0` around its own `await role.run` (`orchestrator.py:319, 339`), but `observe` is the global total over all clients. With level parallelism (G-1) other nodes' spend lands in the window.
- p1 a: two same-level nodes spending $0.01 each: real spend 0.02, ledger committed **0.04**, node costs 0.02 each.
- So the cap triggers early, and `YieldBook` $/accepted (which sums node costs, `:241-242`) is inflated by up to the node concurrency.
- Fix: attribute spend per role client (each role reports its own delta, as `HarborRole.spend_usd` does), or serialize the measurement with a per-node client.

**V4 [P1][ran] Budget and role failures abort every concurrent plan.**
- `settle` raises `BudgetExhausted` when committed spend passes the cap (`budget.py:40-41`), and `_run_node` calls it outside any handler (`orchestrator.py:341`). p1 b: a role that spends $0.50 under a $0.10 cap makes `run_plans` raise; both plans stay at `draft` in the journal and the paid output is lost.
- A `BudgetExhausted` raised inside `role.run` (an LLM client's own cap) returns without settling (`:330-331`). p1 c: `reserved {'t1/a': 0.08}` stays forever, and the next plan is abandoned for a cap it never used.
- Any other exception in a role (p1 d: `KeyError`) propagates through `asyncio.gather` (`:264`, `:295`) and aborts all plans; the other plans never record a final state.
- Fix: settle in a `finally`; turn an over-cap settle into "abandoned(budget)" for the node and stop new reservations; catch unexpected role exceptions as a failed attempt (or `abandoned(infra)`), never a run abort; use `gather(return_exceptions=True)` in `run_plans`.

**V5 [P1][ran] A changed validator does not invalidate journaled outputs.** Validators are not part of the input hash, and the reuse path returns before validation (`orchestrator.py:302-307`). p1 e: a new "no DROP" validator, with the old output `SELECT 1; DROP TABLE x` in the journal: `constructed, reused True`. The recorded toy report is such a replay (every node `reused: true`), so its validators did not run for that report. Fix: include a validator fingerprint (qualified names plus a version) in the input hash, or always re-run validators on a reused output.

**V15 [P1][ran] The YieldBook forgets a person's resume and wipes M9 verdicts.**
- `resume` logs `{"event": "resume"}` (`economy.py:143`), but `__init__` replays only `attempt` and `pause` events (`:62-69`). p3: paused after 3 rejected attempts, `resume` clears it (rows 0), and a reloaded book is **paused again with 3 rows**.
- `record_construction` builds a new `Attempt` with `accepted=None, hacks=0` (`:84-88`). p3: an accepted task with 1 hack, re-run: `accepted None, hacks 0`. Yield and hack-found rates then count it as pending.
- Fix: replay `resume` (pop the pause, drop that generator's earlier rows) on load; keep `accepted`/`hacks` when the task hash is unchanged.

**V18 [P2][ran] A no-op re-run of an accepted plan resets its state to `constructed`.** `_run_plan` always appends the construction state (`orchestrator.py:282`). p1 g: `accepted` (with a task hash) → re-run with every node reused → `constructed`. The docstring binds `accepted` to the content hash, so an unchanged hash should keep it. Fix: if every node is reused and the assembled task hash equals the accepted one, keep `accepted`.

### Swarm, curator, stitch, role runner

**V16 [P2][ran] Announce-before-edit is cooperative.** `announce` raises `Cancelled` inside the worker (`swarm.py:107-110`); the stage only treats it as a collision if the worker lets it propagate (`:166-169`).
- p3: 5 workers that catch `Cancelled` and contribute the same plan: 5 commits, 4 collisions counted, 5 charged.
- A worker that announces twice leaves its first announcement in `in_flight` forever (`:111-112` overwrite `ep.announced`; `:171-172` removes only the last). p3: 4 stale gaps after the stage, which make later workers collide.
- A worker that never announces may contribute.
- Fix: the stage records the announce outcome itself and rejects a contribution from an episode whose announce collided or that never announced; keep all of an episode's announcements and remove them all at episode end.

**V17 [P1][ran] Lineage sealing is not durable, and it trusts the caller's lineage.** `Curator.sealed` is an in-memory dict (`curator.py:47`) with no persistence.
- p3: in one process, a sibling of a bench task is held (`lineage_sealed`); a new `Curator` (the next forge run) places the same sibling in **train**.
- `lin = sorted(set(lineage)) or [task_id]` (`:60`): a composite placed without its components' seeds seals only its own id. p3: the composite `seedA/span1+seedB` with no lineage passed goes to train although `seedA` is sealed to bench.
- Fix: persist `sealed`, `kept` and decisions (jsonl, like the YieldBook); derive a composite's lineage from `Composite.components` (and require it for any `Composite`).

**V20 [P2][ran] `stitch` accepts a span that pre-applies a file it also owns.** The consistency check skips the owner (`stitch.py:63`). p3: span `a` with start `{f.c: BASE->START}` and oracle `{f.c: START->ORACLE}`: stitched start `{}` and stitched oracle `f.c = START->ORACLE`, a diff relative to `a`'s start applied to the base. Fix: reject a span whose start and oracle touch the same path, or compose the two diffs.

**V19 [P2][read] A failed sandboxed trial counts as $0.** `run_role_job` records `cost_usd: 0.0` when `extract_role_result` raises (`role_runner.py:198-199`), although the agent may have spent tokens before failing; the orchestrator then settles 0. Fix: compute the cost before raising and return it with the error.

### Other observations

- **V21 [P2][read]** Dedup L1 compares the first 80 normalized characters (`dedup.py:34-35`). Every toy task starts with the same 170-character `INTRO` (`scenario.py:30`), so L1 is effectively "same oracle footprint". Research M8-2 (shingle Jaccard, behavioural L4) and M8-3 (decontamination at curation) are still open; no forge code calls `bench.contamination`.
- **V22 [P2][read]** `tighten.add_gate` calls a gate separating when the frontier fails one of its runs (`tighten.py:70`); one flaky frontier run qualifies. Require the frontier to fail at a rate whose Wilson lower bound is above 0.
- **V23 [P2][read]** `toy env` writes `forge/scenario.json` (supervisor name, order statuses) into the forged pack root (`env.py:73`). The World does not mount it today, but any exporter that copies the pack root into an agent image would ship it, and A5 does not scan it. Keep builder-only material outside the pack root.

## 3. Coverage

Run from `crucible/` with `COVERAGE_FILE` pointed at the scratchpad (the repo's `.coverage` is untouched):

```
pytest -q -p no:cacheprovider --cov=crucible.forge --cov=crucible.accept.contract --cov=crucible.interop.harbor.role_runner \
  --cov=crucible_pack_toy.forge --cov=crucible_pack_coding.forge tests/unit/{test_forge,test_compose,test_curator,test_economy,
  test_forge_repo,test_fuse,test_hacker,test_judge_specialists,test_ladder,test_perturb,test_role_runner,test_stitch,test_swarm,
  test_tighten,test_span_closure}.py
59 passed in 77.3 s; TOTAL 3368 stmts, 66%
```

| Module | Cover | Notes |
|---|---|---|
| `forge/budget.py`, `schema.py`, `hacker.py`, `judge.py`, `tighten.py` | 100% | V4's overspend path is covered only as a unit; no test runs an overspend through `run_plans` |
| `forge/orchestrator.py` | 96% | uncovered: `:327` (3 infra errors → abandoned), `:331` (role-side `BudgetExhausted`, V4), `:346-347` (in-node breach). No test has concurrent spenders (V3) or a changed validator (V5) |
| `forge/repo.py` 97%, `swarm.py` 97%, `economy.py` 98% | | tests cover the happy paths; write-down (V1), prefixed quotes (V2), swallowed `Cancelled` (V16) and reload after `resume` (V15) are untested |
| `forge/compose.py` 94%, `curator.py` 97%, `dedup.py` 95%, `ladder.py` 91%, `learnability.py` 98%, `specialists.py` 95%, `stitch.py` 96%, `perturb.py` 92%, `fuse.py` 83% | | `fuse.py` conflict branches `:33-83` partly uncovered |
| `interop/harbor/role_runner.py` | 91% | the trial-failure cost path (`:156-169`, `:198-199`; V19) is uncovered |
| `accept/contract.py` | 26% | in this set the tests call `world_synth.forge(collect=...)`, which skips `accept` (`world_synth.py:165-171` uncovered); `a1_oracle`…`a8_grounding` run elsewhere (M9 tests). V8, V9, V11 have no test |
| `TP/forge/roles.py` 14%, `scenario.py` 30%, `env.py` 35%, `task.py` 62% | | the scripted roles run only on a fresh journal; in these tests the nodes are reused from an existing journal |
| `TP/forge/hindsight.py`, `calibrate.py` | 0% | hindsight is only exercised by its CLI (re-run here, p6) |
| `CP/forge/derive.py` | 23% | `derive_span`, `trap_from_run`, `derive_seed_trap`, `cpu_accept_span`, `write_task` are untested (V12, V13 live here) |
| `CP/forge/forge_gpu.py` | 0% | lab-only |

Risky paths with no test: the M9 subset checks on adversarial inputs (V8, V9, V11), the coding derivation (V12, V13), budget failure under concurrency (V3, V4), and the repo's flow rule (V1, V2). The probes `p1`, `p2`, `p4` and `p7` run offline in under 20 s each and are the cheapest regression seeds.
