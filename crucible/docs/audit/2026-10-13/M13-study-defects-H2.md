# M13 study, hunt H2: splits, IRT, contamination, dynamic

Scope: `C/bench/{splits,irt,contamination,dynamic}.py`, their CLI wiring (`C/cli/studio.py:222-296`, `C/studio/build.py`),
and how M8 forge, M9 accept and M14 packs feed tasks into them. Spec: `docs/modules/M13-bench.md`, DESIGN §11.2 to
§11.3. Read-only study, 2026-10-13, HEAD `b6d89fc`.

Paths: `C` = `crucible/src/crucible/`. Probes are in the session scratchpad
`m13/h2/{r_splits,r_contam,r_irt,r_irt_anchor,r_irt_stuck,r_irt_stationary,r_dynamic,r_vacuity}.py`, with their
outputs (`*.out`). Run them from `crucible/` with `PYTHONPATH=src` (`r_irt*.py` need scipy: `.envs/nemo/bin/python`).
`tests/unit/test_bench.py`: 4 passed on HEAD.

## Summary

| ID | Sev | Where | Defect |
|---|---|---|---|
| V1 | P0 | `splits.py:54-77` | `assign` puts one lineage in both train and held-out (OOD or time). 499/500 fuzzed forests violate `check()` |
| V2 | P0 | `splits.py:33`, `forge/curator.py:40,86-136` | A lineage is one string. Composites and forks of forks leak, and `check()` cannot see it |
| V3 | P0 | `dynamic.py:56,90`, every pack `generate` | The published manifest carries the item seeds inside the ids. The test that guards it is vacuous |
| V4 | P0 | `cli/studio.py:259`, `studio/build.py:24` | The default item keys are public constants, and `sha256(K)` gives them away. K goes on argv |
| V5 | P0 | `irt.py:82-111` | The 2PL item step falls into a 2-cycle and returns non-stationary items. Anchored fit: 28 nats off the MAP, b off by 1.0 |
| V6 | P0 | `irt.py:180-193` | `drift` quarantines clean anchors. False-positive rate per anchor: 17% (2PL), 48% (shifted cohort), 89% with 3 contaminated anchors |
| V7 | P1 | `splits.py:85-90` | The seal ignores labels and train or guard membership. Showcase to bench and guard to train both verify. `\n` in an id collides |
| V8 | P1 | `splits.py:93-105` | `check()` has no time, guard or showcase-lineage invariants. A non-showcase sibling of a showcase task lands in bench |
| V9 | P1 | `splits.py:108-114` | `eligible` fails open. An unknown model cutoff admits every dated public item. An item that never set `source_kind` counts as synthetic |
| V10 | P1 | `splits.py:63` | An undated public-derived item under a time split goes to train |
| V11 | P1 | `dynamic.py:28-29` | The item seed does not bind bench name or version. Reusing K across versions reproduces the same items. `/` is ambiguous |
| V12 | P1 | `studio/build.py:89-91`, `cli/studio.py:240` | Production builds pass no validation checks, and no IRT anchors. Fresh items are never validated or linked |
| V13 | P1 | (wiring) | No production path uses `splits` or `contamination`. Curator and splits are two split authorities that can disagree |
| V14 | P1 | `irt.py:119` | Below 15 respondents, Rasch is forced even against 2PL anchors. Fresh items get a = 1 on a 2PL scale |
| V15 | P1 | `contamination.py:115` | Canary detection is an exact, case-sensitive substring. Lower-cased or reflowed corpora and a bare GUID are missed |
| V16 | P1 | `contamination.py:28` | The leak regex misses `org/repo#1234`, "pull request 1234", bare SHAs and MRs. It flags `#123456` colours and the toy pack's own `ticket #39` |
| V17 | P1 | `tests/unit/test_bench.py:70-89,40-67` | Vacuous tests. Mutants that break `check`, `verify_seal`, lineage and `drift` all pass |
| V18 | P2 | `splits.py:44-45,113-114` | Naive and aware timestamps crash `assign` and `eligible` |
| V19 | P2 | `splits.py:48-51,59,103` | `ood={"repo": "pcl"}` becomes `{"p","c","l"}`. Values are compared raw, so `PCL` and `pcl ` train. An empty salt is accepted. Duplicate ids collapse |
| V20 | P2 | `contamination.py:27,36-37` | Non-ASCII text tokenizes to fragments, and CJK to nothing. `strip_canary` drops the whole line holding an inline canary, and a wrapped canary leaves residue |
| V21 | P2 | `contamination.py:124` | The "50-character" code scan only guarantees copies of 74 characters or more |
| V22 | P2 | `contamination.py:105-127` | `scan(n=...)` never matches (bench grams are always 13). Mixed `num_perm` silently misses identical texts |
| V23 | P2 | `contamination.py:77-85` | `tree_fingerprint` hashes `.git/` and follows symlinks: "the same files hash the same anywhere" breaks |
| V24 | P2 | `irt.py:114-149,152-166` | No input validation (s > n, s < 0, NaN give finite garbage, such as θ = ±18 from NaN). No convergence flag. Post-hoc standardization is not the MAP |
| V25 | P2 | `dynamic.py:73-94` | A short benchmark returns silently. Duplicate ids are admitted. A check that raises aborts the build. A failing family is starved without a report |

## Findings

### V1 (P0) `assign` itself leaks a lineage across train and held-out
`C/bench/splits.py:59-70` sends an item to `ood:<axis>`, `time` or `guard` by its *own* axis value or time. The lineage
reconciliation at `:72-77` only touches items already in `train`/`bench`, so siblings of an OOD or time-held item stay in
train. The module claims "siblings of one seed never straddle train and bench" (`:5`).

Reproducer (`r_splits.py`):
```
items: seed-a (lineage A, repo=pcl), fork-a1/fork-a2 (lineage A, repo=other); ood={"repo":{"pcl"}}
{'fork-a1': 'train', 'fork-a2': 'train', 'seed-a': 'ood:repo'}  check: ['lineage A straddles train and held-out']
pr-old (P, 2026-01) / pr-new-fork (P, 2026-08), cutoff 2026-06:
{'pr-new-fork': 'time', 'pr-old': 'train'}  check: ['lineage P straddles train and held-out']
fuzz (30 items, 8 lineages, 4 repos, monthly times): 499/500 assignments violate check()
```
Fix: decide the split per *group*. Compute every member's rule (showcase > ood > time > guard > HMAC). If any member is
held out, the whole group takes the strongest held label; showcase wins. Run `check()` at the end of `assign` and raise if
it returns anything.
Test: `assign` on random lineage forests (hypothesis), then `assert not check(...)`. It fails now (499/500).

### V2 (P0) One lineage string cannot express composites or forks of forks
`SplitItem.lineage: str` (`splits.py:33`). The curator's lineage is a *list* of roots: it is transitive, and a composite
carries every component's lineage (`forge/curator.py:10-13,86-99`). No adapter maps one to the other. A caller has to
pick one root, and the other root's material crosses splits. `check()` then groups by the same single string, so it
passes.
```
composite of a train root and a bench root:
lineage=<train root>: {'composite': 'train', 'seed-bench': 'bench', 'seed-train': 'train'} check: []
fork of fork with lineage = parent: {'child': 'train', 'grandchild': 'bench', 'root': 'train'} check: []
```
Fix: `lineage: tuple[str, ...]`. Union-find over ids and lineage roots, so the group is the connected component. HMAC on
the component's smallest root. Add `SplitItem.from_card(card, curator)`, which takes `card.splits.showcase`,
`forge.seed_ref` and `Curator.lineage_of`.
Test: three items, where a composite links roots R1 and R2. Assert that all three get the same split for 200 salts.

### V3 (P0) Manifest ids carry the fresh items' seeds
`Benchmark.manifest()` publishes `id_of(task)` (`dynamic.py:56`), and the rejection log publishes it too (`:90`). Every
pack's generator puts the seed into the id: toy `toy/helpdesk_backlog/s{seed}-n12@1`
(`packs/_toy/.../longhorizon/task.py:106`), finance_ap `finance_ap/invoices/s{seed}-n{n}@1`, and the `_template`
`__PACK_NAME__/greeting/s{seed}@1`. The seed space is also `seed % 100_000` (toy, finance_ap), so the whole fresh pool can
be enumerated without K. The guard test checks for the word `"seed"` (`test_bench.py:160`), so it cannot fail.
```
[{"id": "toy/helpdesk_backlog/s67852-n12@1", ...}, {"id": ".../s45987-n12@1"}, {"id": ".../s74428-n12@1"}]
'seed' substring in manifest (the test's check): False
seeds recovered from published ids: [67852, 45987, 74428] true: [67852, 45987, 74428]
```
Fix: the manifest publishes `HMAC(K, id)[:16]`, or a fresh opaque id that `build` assigns (`<bench>/<version>/f<n>`).
Keep the id-to-task map private. Packs must not derive an id from the seed. A lint in `pack check` should refuse an id
that contains `str(seed)`. Use the full 64-bit seed, not `% 100_000`.
Test: build with a generator whose id embeds the seed. Assert that no published id or rejection entry contains
`str(item_seed(...))` or `str(seed % 10**5)`.

### V4 (P0) The default keys are public, and the commitment reveals them
`crucible bench run --key` defaults to `"crucible-bench-v1"` (`cli/studio.py:259`), and `crucible build` to
`b"crucible-build-v1"` (`studio/build.py:24`). `commitment = sha256(K)` has no salt (`dynamic.py:32-33`), so a dictionary
recovers any low-entropy K from the published commitment. K is passed on the command line (`ps`, shell history).
```
dictionary attack recovers K: b'crucible-bench-v1'
```
Fix: no default key. Read K from a file or an env var. Require ≥ 32 random bytes, and refuse a known default.
Commitment = `sha256(nonce || K)`, with the nonce published (or K random by construction). Raise if `len(key) < 32`.
Test: `bench run` without a key exits non-zero, and `dynamic.build(key=b"short")` raises.

### V5 (P0) The 2PL item step oscillates; anchored fits return non-stationary items
`_item_step` (`irt.py:82-111`) has no line search. It alternates between a Newton step (when the Hessian is negative
definite) and a "damped" gradient step `0.1·g`, which is not damped at all: |g| ≈ 25 gives a clipped full step of 1.
The pair forms a 2-cycle that ends after 20 iterations at an arbitrary point. The outer loop's convergence test reads only
θ and b deltas, and both repeat each sweep, so `fit` reports convergence.
```
it4: b=-1.7176 la=0.4165 grad=(23.582,-12.457) det=-105.41 grad   step=(-2.358,1.246)
it5: b=-0.7176 la=-0.0835 grad=(-26.253,19.463) det=261.63 newton step=(3.079,-4.110)
it6: b=-1.7176 ... (repeats)
anchored 2PL, 24x30, 10 anchors: neg log posterior crucible 2540.205 vs scipy L-BFGS MAP 2512.237 (same priors)
worst item i12: crucible (b=-2.0, a=1.30), MAP (-0.999, 2.056), truth (-1.046, 1.958)
MAE b vs truth: crucible 0.286, MAP 0.155; MAE θ 0.138 vs 0.100
20 seeds: 42/400 fresh items returned with |∇| > 0.01 (worst 85.5)
```
The free-fit gate test (corr > 0.9) still passes. Fix: backtracking line search on the item's log posterior (accept a
step only if it increases). In the indefinite case, a Levenberg-Marquardt shift `(H − λI)` instead of `0.1·g`. Use a
convergence test on the gradient norm, and include `a` in the outer delta.
Test: on simulated anchored data, assert |∇ log posterior| < 1e-4 for every fresh item, and that the objective is within
1e-3 of a scipy reference. It fails now (28 nats).

### V6 (P0) `drift` quarantines clean anchors
`drift` (`irt.py:180-193`):
- It refits freely, which standardizes θ on the *new* cohort.
- It aligns to the anchors by a mean shift of b only. There is no scale (A) term, so a 2PL refit on a different cohort is
  on another scale.
- The mean includes the contaminated anchors.
- The test uses only the refit's conditional `se_b`, ignoring the anchor's own SE and θ uncertainty.
- There is no multiplicity control.

Anchors are "quarantined, never rotated", so each false positive permanently shrinks the comparability set.
```
no drift, anchors = truth:
 2PL, N(0,1) cohort, 24 resp:        runs with ≥1 false quarantine 33/40; per-anchor FP 0.168
 2PL, N(1,1.5) cohort:               40/40; per-anchor FP 0.480
 Rasch, 8 resp:                      10/40; per-anchor FP 0.035
real drift (Δb = 1.5, one anchor), 2PL: detected 38/40, but 91 clean anchors quarantined alongside
3 anchors contaminated (Δb = 3), Rasch: 124 of 140 clean anchors quarantined over 20 runs
```
Fix: link by mean-sigma or Stocking-Lord (b* = A·b + B, a* = a/A) on a robust subset: iterate, dropping flagged anchors
and recomputing the link, or use the median and MAD. Use SE = √(se_refit² + se_anchor²). Apply a Holm or Bonferroni
correction over the anchors.
Test: 40 simulated no-drift runs with 10 anchors. Assert that the runs with any quarantine number ≤ 4 (5% FWER plus
slack). A Δb = 1.5 anchor is still detected in ≥ 80% of runs, and no clean anchor is flagged when 3 of 10 are
contaminated.

### V7 (P1) The seal binds the held set only, not the split
`seal(held_ids, salt)` hashes the sorted ids with `split ∉ {train, guard}` (`splits.py:78,85-90`). Labels, train
membership, guard membership and the assignment parameters are outside it. Ids are joined with `\n` and no escaping, and
are not NFC.
```
showcase->bench verifies: True      guard->train verifies: True     bench->time verifies: True
added train item verifies: True     remove bench item verifies: False
seal(['a\nb']) == seal(['a','b']): True
```
"guard → train" means training on the guard gap passes both the seal and `check()` (V8). Fix: seal the canonical JSON
(`core/canonical.py`, NFC) of `{id: split}` for *every* item, plus the parameters (`bench_frac`, `ood`, `time_cutoff`,
`guard_days`). Keep a separate public commitment of the held set if one is needed.
Test: each of the four relabels above must make `verify_seal` false, and `seal(["a\nb"]) != seal(["a","b"])`.

### V8 (P1) `check()` is incomplete
`check` (`splits.py:93-105`) cannot see a train item with `source_time ≥ cutoff`, since it takes no `time_cutoff` and
no `guard_days`. It cannot see a guard item in train either. It treats `showcase` and `bench` as the same "held", so a
non-showcase sibling of a showcase task in **bench** (a published split derived from task 04) passes. It raises
`KeyError` on an unassigned item, and it ignores split ids that are not in the item list.
```
{'task04-1': 'showcase', 'task04-1-span': 'bench'} check: []
time->train check(): []      guard->train check(): []
check on unassigned item: KeyError 'missing'
```
Fix: `check(items, splits, *, ood, time_cutoff, guard_days)` re-derives each item's rule-forced split and compares it.
Showcase propagates to its lineage (derive.py already marks derived spans `showcase=True`). Report unknown or missing
ids as violations.
Test: each case above returns a non-empty list.

### V9 (P1) `eligible` fails open
`splits.py:112-113`: `model_cutoff is None` with a dated item returns `True`. A model whose cutoff is unknown is scored on
every public item, including those from 2020. `SplitItem.source_kind` defaults to `"synthetic"`, so a forged task whose
caller forgot the field is eligible for every model (`:110-111`).
```
pr from 2020, model cutoff None -> True
undated default-kind item, cutoff 2026 -> True
```
Fix: an unknown cutoff makes public items ineligible. Report them separately as "cutoff unknown". `source_kind` gets no
default (required), or defaults to `"unknown"` and is treated as public.
Test: `eligible(SplitItem("x", source_kind="pr", source_time=<2020>), None) is False`.

### V10 (P1) Undated public items train under a time split
`splits.py:63`: `if cut is not None and it.source_time` skips undated items, so they fall through to the HMAC and usually
to train. A PR trace with a lost timestamp may post-date T. The same module excludes undated public items from
eligibility, so the two rules disagree.
```
{'pr-undated': 'train'} (eligible for any model: False)
```
Fix: under a time split, an undated non-synthetic item goes to `guard` (never train, never scored).
Test: the case above asserts `guard`.

### V11 (P1) The item seed is not bound to the benchmark or version
`item_seed(key, family, seed_ref, idx)` hashes `f"{family}/{seed_ref}/{idx}"` (`dynamic.py:28-29`). The bench name and
version are not inputs, and both call sites hard-code `"v1"`. Reusing K for v2 (the CLI default, V4) regenerates v1's
items exactly. Once v1 retires and K is revealed, v2 is exposed. Fields are joined with `/`, which families or seed refs
may contain.
```
v1 ids == v2 ids: True      different bench name, same items: True
('a/b','c',0) == ('a','b/c',0): True
```
Fix: HMAC over the canonical JSON `[name, version, family, seed_ref, idx]`. `build` refuses a K whose commitment a
registry has already seen.
Test: same key, different version, so the seeds differ. The two ambiguous tuples hash differently.

### V12 (P1) Production builds skip validation and anchor linking
`studio/build.py:89-91` and `cli/studio.py:240` call `dynamic.build` with no `checks`. The `build` docstring promises
fresh items "validated (gold solves, no-op does not)" (`build.py:13-14`). DESIGN §11.3 requires fresh items placed on the
anchor scale. Neither path passes `anchors=` to `run_bench`, so `irt.fit` runs unanchored and the abilities of different
versions are not comparable. Only the unit test supplies a validation chain.
Fix: packs expose `hooks.validate`. Fail when it is missing and `n_fresh > 0`. Persist anchor `Item`s per bench version
and pass them to `report_run`.
Test: `build` on the toy pack with a generator that emits an unsolvable item. Assert that the item is rejected (it is
admitted now).

### V13 (P1) Splits and contamination are not wired into any pipeline
`grep` over `src/` and `packs/` finds `bench.splits` and `bench.contamination` only in `tests/unit/test_bench.py`.
- No adapter goes from `TaskCard.splits` / `ForgeRecord.seed_ref` / curator lineage to `SplitItem`.
- No command writes a seal "before any builder sees test traces" (DESIGN §11.2).
- `Curator(contamination=...)` documents "the bench layer wires `bench.contamination.scan`" (`forge/curator.py:50-51`),
  but no caller passes it.
- The curator seals lineages to train or bench by ladder placement (`curator.py:129-136`), and `splits.assign` places
  them by HMAC. These are two split authorities that never reconcile.

Fix: one split authority. `splits.assign` takes the curator's sealed map as a constraint, or the curator calls `assign`.
Add a `crucible bench split` command that writes and verifies the seal, and wire `contamination.scan` into
`Curator(contamination=...)` in the forge pipelines.
Test: an end-to-end toy build writes `splits.json` with a seal, and the curator holds a bench task found in the training
corpus.

### V14 (P1) Forced Rasch against 2PL anchors
`irt.py:119`: below `min_2pl` respondents (15), `rasch=True` even when `anchors` carry 2PL slopes, so every fresh item
is fit with a = 1.0. The anchors' slopes span 0.7 to 2.0. Fresh items are then on a different metric from the anchors
they are supposed to land on.
```
6 respondents, 10 2PL anchors: model 1pl, fresh a's [1.0], anchor a's [0.72, 0.83, 1.02, 1.22, ...]
```
Fix: with anchors, estimate fresh a under the log-a prior (MAP keeps it finite), or fix fresh a at the anchors' geometric
mean. Never a raw 1.0.

### V15 (P1) Canary detection is brittle
`scan` matches `canary in text` exactly (`contamination.py:115`). Corpora are often lower-cased, re-wrapped or
NBSP-normalized, and pages often carry the GUID without the prefix.
```
lower / upper / nbsp / newline / guid only / dashless: detected=False (all six)
```
Fix: search for the bare GUID case-insensitively, with or without dashes, over a whitespace-normalized text. Accept a
list of canaries (all versions).

### V16 (P1) Leak regex misses and false hits
`LEAK` (`contamination.py:28`):
- **Misses**:
  - `NVIDIA/cutlass#1234`: the lookbehind `(?<![\w/])` rejects the canonical GitHub cross-reference.
  - `pull request 1234`, `issue 1234`, `PR # 1234`.
  - A bare 40-hex SHA.
  - GitLab `!1234` and `/-/merge_requests/1234`.
  - Jira `PROJ-1234`.
- **False hits**: CSS `#123456`, `arr[#12]`, `line #42`, and the toy domain's `ticket #39` / `order #7270`, so every toy
  helpdesk instruction would be flagged.

Fix: one rule per reference form, with `org/repo#N` allowed. Exclude 6- and 8-hex colours, and let packs declare
in-domain `#N` namespaces.
Test: table-driven on the 19 strings in `r_contam.py`.

### V17 (P1) Vacuous tests
Mutants applied to `tests/unit/test_bench.py` (`r_vacuity.py`):
```
check() always []:                test PASSES
verify_seal ignores the split:    test PASSES
assign ignores lineage:           test PASSES
drift quarantines every anchor:   test PASSES
```
`test_bench.py:160` (`"seed" not in manifest`) cannot fail (V3). The splits fixture never puts siblings on different axis
values, so V1 is invisible. No test checks the IRT objective against a reference (V5), or `drift` under no drift (V6).

### V18 (P2) Naive and aware timestamps crash
`_t = datetime.fromisoformat` (`splits.py:44-45`). A naive cutoff (`"2026-06-01"`) against an aware `source_time`
(`...Z`) raises `TypeError: can't compare offset-naive and offset-aware datetimes`, in both `assign` and `eligible`. Fix:
parse with `core` time helpers, so that naive means UTC (or reject naive input).

### V19 (P2) API footguns in `assign`
- `ood={"repo": "pcl"}` is turned into `set("pcl")`, so `a (repo=pcl)` trains and `b (repo=p)` is held out. `check` has
  the same bug, so it passes.
- Values are compared raw, so `PCL` and `pcl ` train next to held-out `pcl`.
- `salt=b""` is accepted, which makes the bench ids public.
- Two items with one id collapse silently: `{'train': 1}` from two items.

Fix: reject a `str` value, casefold and strip axis values, require `len(salt) ≥ 32`, and raise on duplicate ids.

### V20 (P2) Tokenization and canary stripping
- `WORD = [a-z0-9_]+` (`contamination.py:27`) splits `código` into `c` and `digo`. A CJK instruction yields 0 tokens and 0
  13-grams, so a verbatim copy is not detected (`set()`).
- `strip_canary` (`:36-37`) drops the *whole line* holding an inline canary (the instruction text on it is lost). It
  leaves the GUID when the canary is wrapped across lines. It also normalizes line endings and drops the trailing newline.

Fix: `\w+` with `re.UNICODE`, plus character n-grams for scripts without spaces. Remove only the canary token span.

### V21 (P2) The code scan guarantees 74 characters, not 50
Windows over the oracle start every 25 characters (`contamination.py:124`), so a corpus copy of L characters is caught
only if it contains an aligned window: L ≥ 74 in general.
```
copied slice start=13 len=60: detected=False   (start=13 len=74: True)
```
Fix: stride 1 with a rolling hash, or index every oracle window and slide over the corpus.

### V22 (P2) `scan` parameters that silently disable layers
- `scan(n=8)` compares 8-grams from the corpus to the bench's fixed 13-grams (`fingerprint` always uses 13), so there are
  never any hits.
- Bench fingerprints with different `num_perm` share one index sized by `bench[0]`, so an identical text with
  `num_perm=64` is not found as a near duplicate.

Fix: store `n` and `num_perm` in `Fingerprint`, and assert they match in `scan`.

### V23 (P2) `tree_fingerprint` is not location-independent
It walks `.git/` (a change to `.git` alone changes the fingerprint) and follows symlinks (git hashes the link path). The
blob header `blob <len>\0` is correct and matches `git hash-object --no-filters`, CRLF included. Fix: skip VCS
directories, and hash a symlink as git does (`os.readlink`).

### V24 (P2) IRT input validation and reporting
- `fit` accepts s > n, s < 0 and NaN (`irt.py:125-128`). NaN gives θ = ±18 and b = −18, all finite and all wrong.
- `ability` with a NaN b returns `(-8.0, nan)`, and `next_item(θ=NaN)` returns the first id.
- A Guttman (perfectly separated) 2PL matrix runs the full 200 sweeps, with a ranging up to 6, and `Fit` has no converged
  flag.
- With no anchors, θ is standardized post hoc (`:137-143`). That is not a reparametrization under the stated priors: the
  returned point is 0.49 nats from the MAP, which has θ sd 1.07, not 1.0.

Ability SE matches the closed-form Fisher information (0.12711 both), and `next_item` agrees with a brute-force maximum.

### V25 (P2) `dynamic.build` edge cases
- With every check failing, `n_fresh=5` returns 0 fresh items and no error.
- An anchor and fresh items sharing the id `dup` are all admitted: `['dup','dup','dup']`.
- A check that raises (`docker down`) aborts the whole build instead of recording a rejection.
- A family that always fails is skipped indefinitely, while the others fill `n_fresh` (stratification is not reported).

The loop *is* bounded (`max_tries·|families|` consecutive failures). Fix: raise when fewer than `n_fresh` items pass,
dedupe ids, treat an exception as a rejection with reason `error: <type>`, and report per-family yields.

## Checked and clean

- HMAC assignment is stable when items are added (`moved: []`).
- Guard boundaries are half-open: `[cut − g, cut)` is guard, and `cut` is time.
- The MinHash band math is 16×8, with candidate probability 0.947 at J = 0.8. Simulated near-duplicate false negatives
  were 0/60 for each of J ∈ [0.80, 0.84], [0.85, 0.89] and [0.90, 0.94]. The blake2b salt-per-permutation family is
  independent.
- Python `hash()` of an int tuple is not salted per process.
- The git blob sha1 matches git.
- `ability` SE matches the closed form.
- The perfect-score rows (all correct, all wrong) stay finite under the priors.
- K is never written to disk by `build` (only `key_commitment`).
