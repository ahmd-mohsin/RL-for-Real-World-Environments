# compose-1: composition, dependency depth and difficulty knobs on dense Worlds (2026-10-29)

Code: `src/crucible/synth/compose.py` (DESIGN §7.7 item 18). Data: `crucible/data/synth/{finance_ap,it_helpdesk}/*compose*`
(gitignored). The swarm wrote every task, link and instruction: proposer, oracle programmer and repairer DeepSeek-V4.1-Flash;
contract writer GLM-5.3; instruction writer DeepSeek; ALT Nemotron-3-Ultra + Nemotron-3-Super + gpt-oss-120b; ATK
Nemotron-3-Super + gpt-oss; panel Nemotron-3-Ultra + gpt-oss. No Qwen model ran in any role, and the student was never
called. Worlds: finance_ap (`--invoices 30`, seeds 0–3), and it_helpdesk on the RSI dense World
(`it_helpdesk/20261029-165124-rsi-dense/world`: 40 tickets, 18 agents).

Method (the lead's): real calls and rollouts, then fix what breaks and run again. Each row below is a run, what it
showed, and the fix.

## Runs and fixes

| run | what | result | fix |
|---|---|---|---|
| compose-smoke1, finance_ap | 3 links, 2 candidates each | 2 links kept, 13 calls, dep 1. Link 3 refused: its candidates depend on link 2's *reads*, and link 2 wrote nothing | **carry** edges (information the agent keeps from earlier observations: a context dependency, never counted in dep); fallback selection (state edge > write > depth > carry > length); dropped candidates logged |
| compose-r1, finance_ap | 2 chains × 8 links | c0: 45 calls, **1 write**, dep 1 (read-only "re-check" links). c1: **212 calls** (one link alone made 135, a vendor sweep), dep 2. 16 of 42 candidates had dead oracles: a ref inside `select`'s literal filter 6, acting on invoices earlier links had decided 6, a list index out of range 4 | `max_horizon` = 48 (a link past it is refused); prefer links that write; `select_eq(list, field, value)` in the library and in the parse error's hint |
| compose-r1, it_helpdesk | 2 chains × 8 links | **c0: 49 calls, 20 writes, dep 2**, carry 3. c1: 37 calls, 13 writes, dep 1 | — |
| deterministic population, offline (4 composites) | MUT cap 240 | 42–52 distinct MUT effects (n < 59); at cap 600: **173–189** distinct. 28–106 s per task | MUT cap = 12 × horizon in `certify_composed` |
| same | supersession | MUT DropStep D1 on the composite: a part's note-insert requirement was gone. (1) a later part's glob that *matched* the change was taken as its owner; (2) the merged `id_canon` re-keys inserts, so the part's raw-keyed pattern matched nothing; (3) two parts insert equal-content notes, which share one content key in their segments | superseded only if a later part wrote that very address; patterns matched under the part's own normalizers and mapped to the merged key space; inserts never superseded |
| same | TWIN across part boundaries | false D9s: (1) σ onto a sibling a later part acts on; (2) "approved", a status named for a read, rewritten in another part's write pattern; (3) "blocked", a vendor name: σ leaves C unchanged; (4) "critical" also steered a later part's read filter (π*′ escalated a ticket the oracle never touches); (5) "critical"→"high" where the ticket already is "high" | sibling ∉ entities π* writes or names; a cell value moves through C only if a write carried it; uncarried with C′ = C → suspect; π*′ failing only on addresses neither π* nor σ touch → suspect; twin value = start value of a written cell → invalid. The fixture's planted D9 stays confirmed (a literal naming an entity π* writes always reaches C) |
| compose-r2, finance_ap | 2 chains × 8 links, after the fixes | **c0: 46 calls, 5 writes, dep 2**, carry 2. c1: 45 calls, 2 writes, dep 1. 5 candidates refused at the cap (50–56 calls). Dead oracles: index out of range 13 (finance_ap's `list_invoices` pages its results, and authors filter page 1), decided invoices 5, select refs **1** (was 6) | — (pagination is a property of the World's tools: noted for the World-families work) |
| compose-cert1, it_helpdesk | full certification of r1 c0 and c1 | both rejected (table below). Composition-introduced findings: a false TAINT **D7** ("critical" is derived by one part and stated by another) and exact rebases turning relaxed free text into equality | derive values that another part's instruction states are dropped from the composite card; a rebased pattern keeps the part's own predicates, re-keyed, when they still match |
| compose-cert2, finance_ap | full certification of r2 c0 | rejected in round 1, all counterexamples inherited from the parts. **The repairer failed 3 of 3**: it re-emits the whole 46-step program and breaks it | open (below) |
| compose-ladder-dry / ladder1, it_helpdesk | rungs 2, 4, 8 parts + 4 knobs | all 7 rungs move the signature; actor first refused ("no literal from the instruction alone") | the actor knob looks at the literal's *first* use only |
| rebase on main 94234191 (§7.7 item 20) | read mutants in MUT, layered TAINT leak matching | runtime unchanged (bounded by the cap); a D7 from the exact-only derive merge | one statement test (`_stated`: exact or the layered `LeakMatcher`) for every compose check |

## Long-horizon compositions (after the fixes)

| World | composite | parts | oracle calls | writes | dep | carry | MUT distinct n (U) | certificate | reasons |
|---|---|---|---|---|---|---|---|---|---|
| it_helpdesk | cert1/c0 (r1 chain 0) | 8 | **49** | 20 | **2** | 3 | 185 (0.049) | rejected | MUT 4 (WrongAnswer D3, DupStep D3 ×2, DropStep D1: the restoration gap), TAINT 9 (exact note bodies in the parts), TWIN 2 (before the divergence fix), PERT D11 1 (an order-dependent agent pick), ALT no pass |
| it_helpdesk | cert1/c1 (r1 chain 1) | 8 | 37 | 13 | 1 | 1 | 166 (0.143) | rejected | MUT 16 (ExtraWrite D2 after the repair), TAINT 9, ALT 0/3 passes |
| finance_ap | cert2/c0 (r2 chain 0) | 7 | **46** | 5 | **2** | 2 | 122 (**0.038**) | rejected | TAINT 3 (answer-format literals of the parts), MUT WrongAnswer D3 (a part's loose answer check), ALT no pass; repair failed |
| finance_ap | r1 c1 | 8 | 212 | 15 | 2 | 4 | — | not certified | over any useful horizon; led to the cap |

The table shows two things:
- **The certificate machinery behaves correctly on long programs.** REP, RED and NULL stay clean (0 confirmed on
  every composite), and TAINT sources literals across boundaries: the cross-part literal is no D5, and an earlier
  result the instruction states is a D7.
- **MUT is the binding class.** With the scaled cap its U bound reaches the admission level (0.038, 0.049). Effect
  dedup keeps 122–212 trials out of 560–612 proposals.

What rejects these composites is defects inherited from the parts. These tasks were built without per-link
certification, so their parts were never cleaned. The binding rule is that no ALT solver passes a 37–51-call task.

## Deterministic re-run after every fix (no model; the stored writer text replayed and re-checked)

| composite | superseded | rebased (relaxations kept) | confirmed | from the composition |
|---|---|---|---|---|
| it_helpdesk r1 c0 (49 calls) | 3 | 10 (`x.t2.r0` keeps part 2's own pattern) | TAINT D5 8, MUT 2 (WrongAnswer, DropStep), PERT 1; TWIN 0 (2 suspects) | DropStep D1 = restoration gap only |
| finance_ap r2 c0 (46 calls) | 0 | 0 | TAINT D5 3, MUT WrongAnswer 1 | none |

## Difficulty ladder (it_helpdesk r1 chain 0; each rung a new task id, certified again)

| rung | knob | parts | calls | writes | dep | distractors | observed records | pages | asks | derived literals | MUT n | certificate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | 2 | 13 | 5 | 1 | 0 | 15 | 0 | 0 | 17 | 203 | rejected (NULL 2, RED 2, TAINT 1; ALT no pass) |
| 1 | horizon | 4 | 27 | 10 | 1 | 0 | 36 | 0 | 0 | 30 | 179 | rejected (NULL 1, TAINT 6) |
| 2 | depth | 8 | 49 | 20 | 2 | 0 | 53 | 0 | 0 | 57 | 193 | rejected (MUT 5, TAINT 8, PERT 1) |
| 3 | distractors (3) | 8 | 49 | 20 | 2 | 3 | 56 | 0 | 0 | 57 | 193 | rejected (MUT 4, TAINT 8, PERT 1) |
| 4 | paginate (5) | 8 | 50 | 20 | 2 | 3 | 56 | 1 | 0 | 58 | 212 | rejected (MUT 2, TAINT 4, PERT 1) |
| 5 | actor | 8 | 51 | 20 | 2 | 3 | 56 | 1 | 1 | 66 | 183 | rejected (MUT 5, TAINT 4, PERT 1) |
| 6 | derive | 8 | 51 | 20 | 2 | 3 | 56 | 1 | 1 | 71 | 183 | rejected (MUT 5, TAINT 5, PERT 1) |

Every knob raised the features it declares, so every rung was kept. The rungs' defects are the base chain's own, and
the ALT rule binds on all of them. The student check is a hook only: no p̂ was measured.

## Writer role (the Tinker check)

The instruction writer wrote one instruction for all 10 compositions it was asked for: 9 accepted on the first attempt
and 1 on the second (smoke1). It cost $0.006–0.007 per run, and the stored texts pass the checks again after the rebase.
In the fixture test, feedback names a stated cross-boundary value and an instruction that lost a needed fact (TAINT D5).
Knob rewrites (actor, derive) were accepted on the first attempt.

## Runtime and the rebase on main (MUT read mutants, layered leak matching)

Before the rebase, the full deterministic population took 20–106 s per composite at cap 240 and 92–280 s at cap 600
(it_helpdesk runs through the sandbox child). One certification with paid classes took 222–702 s per composite.

After the rebase (no model; the stored writer text replayed and re-checked under the layered TAINT):

| composite | calls | MUT proposals | MUT distinct n | MUT s | all classes s | writer text | confirmed |
|---|---|---|---|---|---|---|---|
| it_helpdesk r1 c0 | 49 | 600 | 165 (was 169) | 50.6 | 63.2 | passes | TAINT D5 8, MUT 2, PERT 1; TWIN 2 suspects |
| it_helpdesk r1 c1 | 37 | 600 | 154 (was 173) | 47.1 | 58.7 | passes | TAINT D5 11, MUT 1 |
| finance_ap r2 c0 | 46 | 560 | 122 | 55.6 | 62.3 | passes | TAINT D5 3, MUT 1 (a D7 first, fixed below) |
| finance_ap r2 c1 | 45 | 560 | 95 | 47.7 | 52.5 | passes | MUT 1 |

- **MUT's read mutants do not raise the work.** The scaled cap bounds MUT's proposals, so read mutants take part of
  the same 560–600 slots, and distinct n falls slightly. Runtime stays near one minute per 37–49-call composite.
- **The layered leak matcher found one composition defect.** The derive merge dropped only exact-token statements, so
  "missing_receipt", stated as "missing receipt" by another part, stayed in the derive list and TAINT confirmed a D7.
  Every compose check now uses one test (`_stated`: exact tokens or a counted `LeakMatcher` hit) for the writer's
  must-not-state, the derive merge and the added cross-boundary values. After that, finance_ap r2 c0 shows TAINT 3 (the
  parts' D5s) and no D7.
- **The new LEAK class (L2, a model) was not run on the composites.** It is paid, and it is not part of `cheap()`.

## Cost

| run | $ |
|---|---|
| compose-smoke1 (finance_ap) | 0.156 |
| compose-r1 (finance_ap; it_helpdesk) | 0.985; 1.151 |
| compose-r2 (finance_ap) | 1.132 |
| compose-cert1 (it_helpdesk, 2 composites) | 1.067 (ALT 0.44, adjudication 0.46) |
| compose-cert2 (finance_ap, 1 composite) | 0.096 |
| ladder dry; ladder1 (7 rungs certified) | 0.009; 1.964 (adjudication 0.87, ALT 0.68, repair 0.29) |
| **total** | **6.56 of the 10 cap** |

## Open issues

1. **Repairs on long composites fail.** The repairer re-emits the whole 40–50-step program and breaks it (3 of 3 on
   cert2). What to build next:
   - part-level repair: map each counterexample to its part, repair the part on its prefix World, then re-compose;
   - or certify each link before it is composed (cheap, and it removes the inherited defects that reject every
     composite here).
2. **The ALT rule binds.** No ALT solver passes a 37–51-call composite (0 of 3 per task, even with budgets scaled to
   1.5 × horizon + 6 calls). Admission of long-horizon tasks under the "ALT has a pass" rule needs stronger solvers, a
   rule scaled to horizon, or the lead's decision.
3. **Restoration gap.** Part 1 clears a field and part 5 restores it, so the composite has no net change there, and a
   merged *allowed* pattern still permits the cleared value. MUT finds this as DropStep D1, so certification rejects
   correctly, but the merge itself does not close it.
4. **TWIN suspects need a judge.** The pipeline adjudicates MUT inline only, so a TWIN suspect stays open and blocks
   admission.
5. **Distractors need a Perturbable World.** The finance_ap pack adapter has no `perturb`, so the distractor knob and
   PERT do not apply there.
6. **The repairer sees base-World tools.** `certify_composed` surveys the base World, so on a knob variant the
   repairer's tool catalogue lacks `actors.ask` and the `page` parameter (ALT, ATK and the falsifiers see the variant).
7. **The locked-set FR/FA matrix was not re-measured.** The TWIN changes are sound on the fixtures (every synth test
   passes), but their effect on TWIN's 3/6 D9 recall in RSI-2 is unmeasured.
