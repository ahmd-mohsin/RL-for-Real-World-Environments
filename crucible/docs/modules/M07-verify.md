# M7 `verify` — verifier kit

**Status (2026-10-11; audited 2026-10-12, `docs/audit/2026-10-12/M07-verify.md`).**
- **Audit:** 22 reproduced defects fixed, among them a P0: a `pytest11` entry point in the candidate's install forged
  every hidden-test outcome as root. Masks the candidate could trigger (perf jitter, a missing row, a dropped table, a
  FIFO, a certain fail beside an infra check) are now fails. The CPU and toy goldens pass after the fixes; the GPU
  goldens were not re-run.
- **Passing:** the toy golden, the A13 attack suite, the CPU golden, and **the task04 GPU golden: 8/8 graded, `pass: true`**
  (`reports/m7_golden_gpu.json`).
  - Every grade reproduces its recorded gate profile.
  - The last one, `astra_v3b`, was first masked because another user's job held the GPU, then regraded on an idle GPU on
    2026-10-11. It reproduces exactly: all gates pass except `perf`, the intended failure.
- **Open:** the task03 GPU golden (`neg_floatgrid`, `cheat_tamper` diffs), and tasks 01/01v2/02 are not re-expressed.

- **Code:** `src/crucible/verify/`; coding pack `verify/` (`task04.py`, `task03.py`, `golden_cpu.py`, `golden_gpu.py`); toy pack `verify.py`.
- **Tests:**
  - `tests/unit/test_verify.py` (21 tests), `tests/unit/test_m7_audit.py` (29 audit regressions);
  - `tests/redteam/test_grader_attacks.py` (18 docker tests: honest, 11 cheats, 6 controls).
- **Reports:** `reports/m7_toy.json`, `reports/m7_golden_cpu.json`, `reports/m7_golden_gpu.json` (written by `golden_gpu collect`).
- **Research:** `docs/research/m7-m8/01-verifiers-isolation.md` (12 frameworks and our 4 seed graders).

## What it is

Two layers:
- **`runtime/` (standard library only)** is staged read-only into any verifier image and run as root (`run.py`).
- **The host side** (pydantic allowed) composes rewards, calibrates and audits.

| Piece | What | Built on |
|---|---|---|
| `runtime/trusted.py` | Candidate commands run as a fresh uid: new session, no-new-privs, no groups. Each gets a fresh HOME/TMPDIR, and its cohort is killed at exit. IPC is cleaned, outputs are sealed, copies are root-owned. `python()` runs `-I` with grader paths first and the candidate after the stdlib. Wall and CPU time are measured by the grader | our task03/04 graders (`as_unpriv`, `kill_unpriv`, `seal`); the M1 guard; Karotte `_demote`/`process_utils` |
| `runtime/plan.py` | `Plan` = gates (`gate` / `outcome` / `score`) of checks, run with explicit verdict semantics:<br>• a candidate crash while judged → fail;<br>• `InfraError` → masked;<br>• a failed prerequisite → skip;<br>• fast/full tiers; abstention.<br>`write_outputs` writes `reward.txt`, a flat `reward.json`, `crucible_reward.json` and `report.json`; `prewrite_zero` runs first | Repo2RLEnv `grade.py` (pre-write 0); NeMo-Gym `mask_sample`; Harbor reward files |
| `patterns/static.py` | Comment-stripped rules over the **pruned tree that gets built**. Also: macro expansion, token pasting, dynamic imports, native binaries under any suffix, runner-tamper references | task03/04 `static_checks`; A13 |
| `patterns/differential.py` | Reference (root, 0700, cached per grade, optional self-consistency) vs candidate (unprivileged, sealed) with task comparators. Also concurrency equivalence against **solo** runs, and fresh draws from a grade-time seed revealed after the run | task04 `gate_kinfu`/`gate_concurrency`/`gate_fresh`; Appendix B.2/B.3 |
| `patterns/compare.py` | `bitwise_tree`, `json_close` (per-key tolerances, rejects NaN), `TwoTier` (strict, then relaxed only when the reference bifurcates; the tier used is reported) | task04 KinFu-LS two-tier compare |
| `patterns/hidden_tests.py` | pytest via a plugin writing to a grader-opened fd; doctest via its XML reporter into an fd. Verdict needs a finish record, the expected count, all passed and no skips. `count_pytest` builds the manifest against the original | Repo2RLEnv `test_driver.py` (exact expected tests); SWE-smith F2P/P2P |
| `patterns/perf.py` | Best-of-k alternating rounds, one re-measurement, load guard (load average, GPU utilization), round-spread check (contaminated → infra), whole-process wall ratio, end-state sanity via `CandidateError` | task03/04 `gate_perf`; Appendix B.5 |
| `patterns/invariants.py`, `verify/state.py` | Violating-row invariants and state predicates on any state view. They reuse M5's `phi`, so a check and a milestone never disagree | EnterpriseOps-Gym `database_state` (turned into "no row breaks the policy") |
| `patterns/record.py` | Record-relative score gates (no ceiling) and `AbstainPolicy` (P29, DESIGN §7.5) | Preference Model (elliptic-curve rank); EnterpriseOps-Gym infeasible tasks |
| `stage.py` | `bundle()`: stub `crucible` packages + runtime + plan modules. `docker_grade()`: separate verifier container, `--network none`, hidden mounts | Harbor separate verifier env; verifiers `IsolatedVerifierEnv`; our `run_grade.sh` |
| `grader.py` | Host path: the plan composed with `credit.compose` (Φ partial credit, masks) into the reward files and `steps.jsonl` | M6 |
| `calibrate.py` | Clopper-Pearson flake bound (A4), tolerance tiers from natural plus induced draws (must still kill negatives), bifurcation split, perf headroom (A1 ≥ 15%), reference self-consistency | DESIGN §8.2 |
| `tiers.py` | Fast/full Spearman agreement (A12), fast-pass/full-fail count, a deterministic 5% audit sample | DESIGN §10.2 |
| `leak_audit.py` | Hidden strings from tests and oracle lines, canaries, hidden artifact names; over directories and streamed `docker export`. Git history must stay within the base's ancestry | OpenEnv RFC 008 canaries; SWE-smith's branch leak |
| `collect.py` | Tolerant collect hook: never aborts on one path, includes untracked files, drops derived artifacts. `COLLECT_SH` for Harbor `[[verifier.collect]]` | Appendix B.5 (empty-patch incident) |
| `judge_item.py` | Only **validated** rubric items are judged, into the `rubric` channel, never a gate | M6 C4; engineering rule 3 |

## Exit gate

| Requirement (ARCHITECTURE M7) | Result |
|---|---|
| Toy checks pass and fail as designed | ✅ The oracle scores 1 and the no-op 0 (all three outcome gates fail). **7/7 negatives fail their intended gate:** unapproved refund, in-transit refund and double refund fail `refund_policy`; a fabricated approver fails `approvals`; a partial refund fails `t2`; closing instead of escalating fails `t3`; no notes fails `t1`–`t3` |
| Seed graders re-expressed with the kit reproduce history: every oracle 1.0, every negative fails at its gate, Astra/sol reproduce their profiles | **CPU part ✅:** the kit's decision logic re-scores the recorded history.<br>• task04: 15/15 static verdicts (patches applied to the pinned PCL tree), 7/7 perf verdicts from the recorded rounds, 14/14 sanitizer verdicts.<br>• task03: 10/10 static verdicts.<br>• One divergence is documented: `astra_v3` was graded during another user's GPU job. The kit masks it, and its re-grade `astra_v3b` is reproduced.<br>**GPU part ✅ for task04 (8/8, 2026-10-11; earlier note follows):** the kit task04 grader in the real verifier image reproduced the empty patch's full profile (static 0, build 1, binary 0, consumers 0, the rest 0), and **scored the v3 oracle 1.0 with all 11 gates passing**, perf included, under the load guard, spread check and whole-process wall bar. The oracle, Astra v3, sol v3 and the global/oob/tex/warp negatives are queued and run when the GPU is idle. The task03 GPU golden comes after them. tasks 01/01v2/02 are not re-expressed yet |
| Grader attack suite (A13) | ✅ Honest 1. **11/11 cheats 0** at their intended gate: `os._exit(0)` under pytest, fake stdout rows, writing reward files, detached output fix-up, memoizing in HOME, shadowing stdlib `json`, shadowing the test runner, delegating to the reference, tampering with hidden tests, dynamic banned import, `pytest.skip` from library code. **6 controls:** the same cheats beat a grader with the seed graders' habits |

## Findings

1. **Attack tests need controls.** The first suite passed 12/12, yet 4 attacks did not work even against a weak grader, so their "pass" proved nothing. Worse, `background_fixup` "failed" only because the grader crashed: `poll()` reaped the child and then `wait4` raised. Once the attacks were made real, the kit lost to `background_fixup`, because the cohort was killed after a 1 s pipe-drain grace. Both are fixed (B.12).
2. **"Consumers first on `PYTHONPATH`" (P26) is not enough.** `PYTHONPATH` precedes the standard library, so a candidate `json.py` shadows the grader's own imports. The kit runs Python isolated and appends the candidate after the stdlib.
3. **Harbor cannot read our `reward.json`.** Non-numeric values raise. The file is now flat; the full document is `crucible_reward.json`.
4. **A historical perf verdict was noise.** `astra_v3` was scored 0 while another user's job slowed KinFu 37×. The kit's spread check would have masked it and asked for a re-grade.
5. **task04 v3's oracle has 12% headroom on the KinFu bar**, below A1's 15%. This is recorded for re-versioning.
6. **M5 `phi` could not measure float-valued SQL milestones** (the adapter's exact float encoding). Found by the toy golden; fixed in `evaluate.decode`.

7. **Record mode shows what a bar hides (P29).** task04's perf gate can run as a score gate (`perf_mode=record`). Each bar is the baseline (0), the v3 oracle's recorded ratio is the best known (1.0), and a faster port scores above 1. Re-scored from the recorded measurements (`reports/m7_record_mode_task04.json`):
   - **Astra v3 beats our oracle** on marching cubes and the people detector (new records), while failing three KinFu bars;
   - **oracle v2 beats oracle v3** on per-frame KinFu, cloud extraction and people, because v3 traded single-tracker speed for concurrency.

   The bar collapses all of that to 0 or 1. With the default all-bars floor Astra still scores 0; without it, 0.46.

## Corrections to ARCHITECTURE's "Takes" (from research 01)

- MiMo has no `/verify` endpoint. It is a `run_verify.py` command with the rule "no parseable reward → masked", which the kit follows.
- verifiers' `IsolatedVerifierEnv` belongs to the M11 export adapter, not the M7 runtime.
- mimoagent `antihack.py` is off by default and easy to bypass. It is used only as a probe-tagging list (M3 already covers that).
- agentenv's verifier runs in the agent's container with injected keys, and it scores a NaN reward file as 1.0. Crucible's separate-verifier step (M11) must not reuse it as is.

## Not done

- The GPU golden for task04 (7 grades queued) and task03.
- Kit plans for tasks 01, 01v2 and 02. Their B.10 holes are recorded; they need grader-owned builds of the doctest suites (task01) and root-free candidate execution (task02).
- Seed re-versioning: the task04 consumer `PCLConfig.cmake` flag injection, the KinFu headroom, and the task03/04 instruction sentences that are unenforced or looser than the grader (A7).
- A Harbor-run end-to-end check of the flat `reward.json` (M11).
- From the audit: the patch is still decoded as text (non-UTF-8 context lines), two-level `##` macro pastes,
  `kill_cohort` survivors still mask, no `adversarial_inputs` pattern, `judge_item` has no production caller, and the
  masked-mean metric (`interop/harbor/metrics.py`) is not yet wired into the M11 export. Re-run the task04 GPU golden
  after the perf changes.
