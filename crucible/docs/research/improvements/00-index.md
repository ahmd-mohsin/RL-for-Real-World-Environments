# Improvement study across every local RL library and framework (2026-10-11)

Six read-only studies, one per module group. Each one maps what Crucible has against the frameworks under
`RL-Environments/06-frameworks-and-env-code/`, plus agentenv, the articles and the dataset cards. They end in a ranked
list with path:line sources, impact and effort, and DEPEND/PORT/PATTERN plus the license.

| File | Modules |
|---|---|
| `M00-M02.md` | core, world, actors |
| `M03-M05.md` | connectors, signals, workflow |
| `M06-M08.md` | credit, verify, forge |
| `M09-M11.md` | accept, runtime, interop |
| `M12-train.md` | train |
| `M13-M15-RLM.md` | bench, packs, studio, the RLM agent and harness |

## Correctness problems the studies found (fix before features)

| Area | Problem | File |
|---|---|---|
| M7 | A candidate can turn a perf failure into a masked grade (candidate-side noise → `infra`) | `M06-M08.md` V1 |
| M6 | Dense credit ignores gates and hacks: a gate-failing episode still earns shaping. Live evidence: on finance_ap, approve-everything scores 0 but has progress AUC 0.15 | `M06-M08.md` V3 |
| M7 | LLM judges read agent text unfenced (prompt injection into the hack and quality judges) | `M06-M08.md` V4 |
| M4 | Signal reliability is circular (labels decide their own confirmation) | `M03-M05.md` S1 |
| M10 | Harness exceptions are not contained (one escape aborts a run); no per-attempt deadline; no process lock | `M09-M11.md` |
| M0/M11 | Layering is broken: world → connectors, actors → train/accept/forge, runtime → Harbor. No CI runs the import contracts | `M00-M02.md` S1, `M09-M11.md` 1 |
| M1 | `SandboxPool` loses start errors and `warm()` can loop forever | `M00-M02.md` |
| M5 | Signal-derived milestones are dead code; no automated nop/oracle control | `M03-M05.md` |
| RLM | No cap on sub-calls, batch fan-out or tree tokens (unbounded RL cost) | `M13-M15-RLM.md` RLM #1 |
| M13 | The seal hashes ids, not content: a task edited after sealing is not detected | `M13-M15-RLM.md` M13 #1 |

**Already fixed on 2026-10-11**, while the studies ran:
- concurrent attempts sharing an LLM client were charged each other's spend (`NativeHarness` now charges its own calls);
- SGLang weights released at a sleep were not restored on resume without a publish (the engine reloads them; found by
  the live Modal check);
- the trainer now logs the pre-clip grad norm and the lr.

## Highest-leverage features

1. **M12 learning:**
   - difficulty pools plus dynamic sampling that cannot deadlock;
   - on-policy distillation (already built);
   - larger groups;
   - periodic eval.
2. **M6:** GiGPO anchor-state step credit, which credits exploration where Φ is blind (the S2 finding).
3. **M9 / M8:** SWE-smith patch mutation operators (real behavioural negatives for coding tasks).
4. **M1:** baked world images and incremental snapshots (the restore cost dominates long-horizon branching).
5. **M13 / M14:**
   - harness profiles (the same model moves 13 points across harnesses);
   - content-addressed `pack lock` / release.
