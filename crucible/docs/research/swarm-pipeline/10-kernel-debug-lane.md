# 10 The kernel debugging lane (CUDA kernel bugs, kernel performance regressions, kernel migrations)

Spec: `00` (LongTask §1, generators §2, machine S0-S17 §4, verifier rigor §5 with HUD fairness, long horizon §6, targeting
§7, traces §8, sealing §10), `06` (the kernel-migration family this lane extends), `07` (the multi-language debugging lane
whose `LangProfile` protocol this lane builds on), `04` (build log; World-leak lessons). Student: **Qwen/Qwen3.6-35B-A3B**
(MoE, ~3B active, 262K context; released 2026-04-15; sampled and post-trained on Tinker). Targets measured later: the
35B in the 10-60% band on training tasks, below 30% on the held-out benchmark, failures on the intended skill, and a
strong solver (Qwen3.5-397B / DeepSeek-V4.1-Flash) solving each task (fairness). Builder of every prompt and intent
card: `claude-code:opus-5.5` (recorded on each record). No model was called and no GPU was used to build this lane.

Code: `factory/langs/cuda.py` (the C++/CUDA profile), `factory/lanes/kernel_harness.py` (the grader-owned op-test
harness), `kernel_debug_mine.py` + `kernel_debug_ops.py` (S1), `kernel_debug.py` (records, synthesis, the task, prompts,
intent), `kernel_debug_world.py` (CPU gates in Docker, the GPU backend), `kernel_debug_plan.py` (the GPU grader, a kit
plan), `kernel_debug_migrate.py` (the migration family); scripts `lh_kernel_debug.py` (funnel), `lh_kernel_debug_migrate.py`
(migrations), `lh_kernel_debug_gpu.py` (the GPU half, copied into the bundle as `run_gpu.py`). Tests
`tests/unit/test_m17_kd_{harness,mine,task,migrate}.py`. State: `crucible/data/kernel_debug/` (durable, gitignored).

## 1. Three families, one task shape

A task is a real fixed release B of a CUDA codebase with several real changes taken out (A'), hidden checks on the end
state, and a GPU grader that owns the build. Each fault is a real commit; nothing is synthesized (V1).

| family | intent | ground truth | the agent sees | a milestone passes when |
|---|---|---|---|---|
| `kernel-bug` | KD1 | 2-4 real CUDA kernel bug fixes reverted at B (races, divergent barriers, out-of-bounds access, index overflow, grid limits, strides, launch configurations, stream misuse, precision) | a user's symptom report per op | the op's cases match the CPU reference on fixed and fresh seeds (and stay clean under the class's sanitizer tool) |
| `kernel-perf` | KD2 | 2-3 real kernel speed-ups reverted at B: results stay correct, speed regresses | a user's slowdown report with the bar to meet | results stay correct **and** the grader's timing is within the per-GPU bar of the reference build |
| `kernel-migration` | KD3 | 2-4 real toolkit-migration commits reverted at B (CUDA 12 -> 13: CCCL 3 functors, `cudaMemLocation`, removed `cudaDeviceProp` fields; CUDA 11 -> 12: texture/surface references, mask-less warp intrinsics) | a maintainer's request: these programs no longer build on the new toolkit | each unit compiles for the GPU with no removed construct left, and each program builds and matches the original's outputs |

**Why long-horizon.** Several real faults across several files of a ~100k-line backend (`ggml/src/ggml-cuda`), connected
through shared ops: one op's cases observe two faults, so fixing one leaves the op failing (an interacting fault). Each
fault needs its own reproduction (the op test, compute-sanitizer), localisation, fix and re-check. The connection is
*claimed* on the CPU (shared op or file) and *executed* on the GPU (a case of milestone j also breaks under fault k's
single revert: `m_j requires m_k`). Migrations break several programs for different removed APIs; the executed compile
edges (a program builds only once its migrated headers do) run on the CPU.

## 2. The language layer: the `cuda` profile on the shared `LangProfile` protocol

`factory/langs/cuda.py` subclasses the debug lane's `CLikeProfile` (`07` §2) when that layer is present and is
self-sufficient otherwise (every method the kernel lane calls is defined there; the debug lane's branch was not merged
here: the merge was refused by the permission system, see §9). Paths: source = `.cu/.cuh` and the C++ hosting kernels,
outside tests/benchmarks/examples/vendored trees; protected = tests, every build file (`CMakeLists.txt`, `*.cmake`,
Makefiles, `cmake/`), CI; plumbing (hack severity) = build files (they choose architectures and flags). Definitions: a
comment-, string- and char-literal-aware brace matcher finds C++/CUDA functions (templates, `__global__`/`__device__`
qualifiers, `__launch_bounds__` between return type and name, methods), `kernels()` lists `__global__` ones;
`changed_defs` diffs bodies (the faulty-name lint and `simplify('locate')`). Tests: gtest `TEST*`, Catch2 `TEST_CASE`, and
ggml op-test `test_cases.emplace_back(new test_<op>(...))` lines (keyed by text; the grader enumerates cases by
executing the harness, never from text). To register it with the debug lane: `langs.NAMES += ("cuda",)`,
`_MODULES["cuda"] = "cuda"` (their file, left to them).

## 3. S1: sources, filters, fault classes, groups

**Sources.** Training: `ggml-org/llama.cpp` (MIT). Held-out benchmark: `ikawrakow/ik_llama.cpp` (MIT), a disjoint repository
and disjoint commits with the same harness type (ggml's op test, 2024 layout), sealed by the lease
(`kernel_debug.SEALED_REPOS`; `EvalLease.default(extra_repos=...)` refuses it to training). Migrations: `zjin-lcf/HeCBench`
(BSD-3, CUDA 13 ports) and `ArchaeaSoftware/cudahandbook` (BSD-2, CUDA 12 ports of July 2026). Surveyed and not used (no
C++ harness that runs without torch, or no recent CUDA fixes): vLLM csrc (python/torch tests), flash-attention (3 CUDA
commits since 2026-05), CUTLASS (11), xformers (1), FAISS (18, 2 fix-like), CTranslate2 (4), CCCL (885 CUDA commits but
Catch2 tests fetched over the network, heavy builds), TransformerEngine (cuDNN-bound builds), flashinfer (JIT/python,
Blackwell-specific).

**Cut.** Fixes on or after **2026-05-01** (after the student's release). Training supply after the cut is 17 bug fixes
and 4 speed-ups in llama.cpp, so the training side also mines back to **2025-09-01** as a marked fallback; every record
states `cutoff`, `contamination` (`post-release` | `fallback`) and every fault date. The held-out side uses the cut only.

**Filters** (`kernel_debug_mine`, every drop with its reason): non-merge commits that change the CUDA backend (the
backend directory plus, in the 2024 layout, `ggml/src/ggml-cuda.cu`) and nothing else but the op test; refused: platform
work the grading GPUs (sm_90, sm_120) cannot observe (an older architecture or Windows named; another vendor named without
CUDA: `CUDA/HIP: fix ...` is kept), warnings, build/compile/CI/docs work, reverts, multi-GPU-only code (`allreduce.cu`),
changes touching any other code (a CPU-backend change moves the reference), changes that reach no op the harness tests or
more than 10 ops (shared plumbing no milestone can own).

**Hardware reach** (`kernel_debug_reach`, from 2026-10-23; §3.1). A fault is mined only when a GPU we grade on can
grade it: the GH200 (`sm_90`, DeltaAI) or the RTX 5090 (`sm_120`, lab). Each record carries `targets` per fault.

**Op reach** (`kernel_debug_ops`): the backend's dispatch (`ggml_cuda_compute_forward`: `case GGML_OP_X: ...; break;`,
nested unary/GLU switches) plus the name-level reference graph of every function and function-like macro in the backend
(calls, `<<<>>>` launches, kernels passed as template arguments, `FATTN_VEC_CASE(...)`-style dispatch macros). A change
reaches an op when a function that existed before the change, and that it changed, is reachable from the op's entry
(functions the change adds are reached through their changed callers); the map is computed at each change's own tree.

**Fault classes** (message first, then the diff): `race` (racecheck + repeated seeds), `divergent_barrier` (synccheck +
repeated seeds), `bounds` (memcheck), `overflow`, `grid_limit`, `indexing`, `launch_config`, `stream_async` (repeated
seeds), `precision`, `logic`; `perf` for speed-ups (golden + perf bar).

**Groups** (`group_changes`): seed-and-absorb in date order within 120 days, connected by a shared op or file, 2-4 faults
(perf 2-3); post-cut changes are grouped first and only with each other.

### 3.1 Hardware reach: which of our GPUs can grade a fault (2026-10-23)

Seven tasks reached the GPU gates with faults no GPU of ours runs: an AMD-only kernel selection, a pre-Ampere bf16
path, an integrated-GPU buffer check, speed-only "fixes", a launch reshape that computes the same values, a precision
selection whose difference stays inside the case tolerance, and ops the fork's op test has no case class for. The
subject-only filter above let them through. `kernel_debug_reach` decides from the change itself, per target GPU:

- **Compiled and dispatched** (`changed_on`): the changed file before and after, as one GPU sees it. Preprocessor
  branches are resolved with that GPU's `__CUDA_ARCH__` (device pass) or without it (host pass, where kernel and
  `__device__` bodies are emptied and branches on `__CUDA_ARCH__` are device code), with HIP, MUSA and NCCL undefined and
  the backend's feature macros (`AMPERE_MMA_AVAILABLE`, ...) and `*_available(cc)` predicates read from `common.cuh` the
  same way. Runtime compute-capability tests (`cc >= GGML_CUDA_CC_AMPERE`, `GGML_CUDA_CC_IS_AMD(cc)`) fold with that
  GPU's `cc` (three-valued: what cannot be decided stays), code after a `return` in the same block is dropped, and `if`
  branches that became identical merge. The fault reaches the GPU when the two views still differ. A macro set by the
  toolkit or the build (CCCL version, `STRIDED_ITERATOR_AVAILABLE`) is undecided, so its code counts as reached: the
  filter only rejects on evidence, and S1 executed on the GPU stays the final word.
- **The message** (`message_reasons`): hardware we do not have named in the subject (pre-Ampere, Volta/Turing/Pascal,
  integrated GPUs, multi-GPU, split mode, ring reduction); for the bug family, a speed change that names no wrong result,
  a launch-shape change (only the grid/block numbers change, the launch's own arguments do not) that names no limit and no
  wrong result, and a precision-only selection between two kernels with no symptom named.
- **Testable** (`covered_ops`): at least one op the change reaches is built by the repository's op test at the change.

Mining drops what fails (reason `not gradable on our GPUs (...)`), `lh_kernel_debug.py reach` records the verdict per
milestone of stored records (`rec["reach"]`; `unmeasurable_on_our_gpus` on a milestone no GPU of ours can grade), and
`retire` moves a record out of the store to `data/kernel_debug/retired/<id>/` (`record/`, its `work/`, staged bundle
task and prompt, and `retired.json` with the reason) — nothing is deleted; moving `record/` back restores it.
`retire --auto` retires a record once every GPU it targets has run its gates and fewer than two milestones are both
kept by some GPU's calibration and qualified by the filter. The report lists retired records (`status: retired`).

On the stored tasks the filter rejects all five retired tasks and accepts the three that pass every 5090 gate
(1b8b8fb1, ca21f3a1, efe7d4b1); `tests/unit/test_m17_kd_reach.py` runs it on them when the clones are present.

| Retired (2026-10-23) | Why no GPU of ours grades two of its faults |
|---|---|
| kd-bug-llamacpp-7ac249cc | m1 a speed change (64-bit strides so the compiler unrolls; same results), m2 an AMD-only kernel selection |
| kd-bug-llamacpp-4b8c156e | m1 only reshapes the launch (the old one-thread blocks compute the same values) |
| kd-bug-llamacpp-b377258d | m1 moves FP16 activations to another kernel whose results stay inside the case tolerance |
| kd-bug-ikllamacpp-5b19ab0c (held out) | m1-m3 change ops (MASK_TOPK, INDEXER_TOPK) the fork's op test has no case class for |
| kd-bug-ikllamacpp-4da3f014 (held out) | m2 a pre-Ampere bf16 path (both of our GPUs take the Ampere branch) |

Marked, not retired (they are in the queued GH200 run; `retire --auto` after its ingest): kd-bug-llamacpp-7d70c81f m4
(integrated GPUs only) and kd-bug-llamacpp-b4a950d4 m1 (a PTX 7.0 / Volta selection: not dispatched on sm_90 or sm_120).
New kernel tasks are now generated by the agentic swarm (non-Claude models on Tinker), which mines through this filter;
Claude mined, authored and gated no new candidates under it (a filtered re-mine was started and stopped before it wrote
anything). The filter also flags kd-bug-llamacpp-fcf94bd1 m2 (an occupancy and speed change) and kd-bug-ikllamacpp-47f22c74 m1/m3
(ops with no case class, ring reduction), and kd-bug-ikllamacpp-5b8c07ff m3 (an MMQ path without int8 MMA, which both
GPUs have).

## 4. The record, the World and the grader

**Synthesis** (`kernel_debug.synthesize`): B = the newest fault; A' = B with each fault's backend change and its op-test
hunks reversed, newest first, 3-way through a private index (`generators.base.apply_chain`); a fault whose revert conflicts
is dropped and the chain retried. History A' -> fault 1 -> ... -> B' under `refs/crucible/kernel-debug/<id>` of a derived
clone; B' == B checked. Recorded per fault: its fix applies alone on A' (the GPU prober) and its revert applies alone at B
(the revert mutant). Mutants: `revert-mK` (B with only fault K reverted) and `partial-mK` (B with the first of the fix's
hunks reverted: a half fix must still fail). Floors: no-op, STATUS.json only, a test-only edit and a build-file edit (both
discarded by the grader), and `no_support_<op>` (the CUDA backend claims not to support the op: the harness would skip it;
the grader requires it to run). The oracle is the A' -> B diff of the backend only.

**The World**: `crucible-lh-k:base-12.8.1` (CUDA 12.8 devel, the kernel lane's base), the repository at A' as one commit
(`history: cut`), no network, the GPU at rollout time (`CRUCIBLE_CUDA_ARCH`). The agent's op test is A''s (the fixes'
own cases removed); it can build and run it and compute-sanitizer.

**The harness** (`kernel_harness`): the grader always uses B's `tests/test-backend-ops.cpp` patched (two dialects:
`ggml-2025`, `ggml-2024`): `CRUCIBLE_SEED` (every random draw from one counter-based stream; `std::random_device`
replaced; per-thread engines seeded from (seed, call, slice)), `CRUCIBLE_CASES` (exact case selection), `CRUCIBLE_LIST`
(name every case of a mode, run none), `CRUCIBLE_DUMP` (each compared node's GPU and CPU vectors, the case tolerance),
`CRUCIBLE_REPS` (exact timed repetitions), `CRUCIBLE_ALLOW_CPU` (the CPU gates run it on the CPU backend). A missing
anchor refuses the task (S5), never grades with an unpatched file.

**The grader** (`kernel_debug_plan`, a kit plan, root in the verifier image): the grader's tree = A' + the candidate's
files under the backend only + B's patched harness; CMake with the grader's flags (`CMAKE_CUDA_ARCHITECTURES=<arch>-real`:
SASS only, no PTX), built unprivileged, sealed; `cuobjdump` must show SASS for the arch only. Per milestone: `golden`
(calibrated cases on two fixed seeds, one fresh seed drawn now and revealed after the run, and `repeat` more fresh seeds for
nondeterministic faults; each node of each case within the case's own NMSE tolerance against the **reference build's** CPU
output, compared outside the candidate's process; the case must run; CUPTI launch evidence: at least one kernel launch per
case), `racecheck`/`synccheck`/`memcheck` (compute-sanitizer, every summary zero), `perf` (`patterns.perf.PerfBars`: the
grader's wall clock around whole harness runs at R and 3R repetitions so start-up cancels, alternating reference and
candidate, best of 3 rounds + a re-measure, load guard, spread check -> infra, launches must scale with repetitions; bars
per GPU model). No GPU -> masked `no_gpu`; uncalibrated -> masked; never a 0 (V8). Reward 0.5 Phi + 0.5 [all], the lane
convention.

## 5. CPU gates now, GPU gates bundled

| state | CPU (here, Docker, aarch64 CUDA devel images, no GPU) | GPU (bundle) |
|---|---|---|
| S1 | licence, cut, filters, classes, op reach, groups | real-fix check: B passes, each single revert breaks its milestone's cases or is caught by its sanitizer tool |
| S3 | A'..B' synthesised, B' == B; A' compiles (every unit a fault touches, both arches, CMake's own command); the start configures with CUDA in the World; one-commit workspace; leak probe (no B-only line in the image) | - |
| S4 | B compiles (same units, both arches) | the reference build of B |
| S5 | harness anchors present at B; the patched harness builds against B and A' on the CPU backend; each milestone's case pool is listed by executing it; the fixes' own cases (B list minus A' list); the seeding executed (same seed -> identical dumps, new seed -> different inputs, the CPU reference passes) | the oracle scores 1 |
| S6 | claimed edges (shared ops/files); migrations: executed compile edges | executed edges from the per-case results of each single revert |
| S7 | the builder's report: two-way ledger, lint (no faulty function/kernel/file names, no hidden case strings, no hashes/links/PR numbers/versions), fairness lint | - |
| S8 | floors apply; every mutant applies and compiles | floors 0, mutants fail their milestone, sanitizer clean on B and dirty on the revert |
| S10 | - | 60 oracle grades at concurrency 1 and 4, Clopper-Pearson <= 5% |
| S12-S15 | - | hackers, the 35B and the strong solver (Tinker), audit with the intent cards' trace evidence, targeting |

## 6. Intent cards and the S7 ledger

Each record carries `intent` (`kernel_debug.task_intent`): the skills (`kernel_fault_localization`, `gpu_concurrency`,
`index_arithmetic`, `interacting_faults`, `reproduce_with_op_tests`, `perf_regression_diagnosis`, `toolkit_migration`), why it
is long-horizon, the milestones, the expected failure mode per skill, the trace evidence that attributes a failure to the
skill (files edited vs the fault files; sanitizer and op-test runs before and after edits; milestones lost in the
checkpoint series; timing runs for perf), and the attribution rule (S14/S15: >= 70% of failures with their evidence; the
locate counterfactual must succeed significantly more often). Family-level cards: `kernel_debug.intent_cards()`
(KD1-KD3, status `candidate` until the student's failures are observed and a human approves). The prompt's provenance
records the builder and that it states only symptoms and goals; the lint enforces it.

## 7. GPU bundle

`crucible/data/kernel_debug/gpu_bundle/` (`lh_kernel_debug.py stage-gpu`): per task `a.tar.gz`, `b.tar.gz`, `kdspec.json`,
`oracle.patch`, `floor_*.patch`, `mutant_*.patch`, `milestones.json`; the kit (`kernel_debug_plan`, `kernel_harness`,
`kernel_plan` + the runtime), the CUPTI hook source, `Dockerfile`, `run_gpu.sh`, `run_gpu.py`, `PENDING.json`.

    # lab RTX 5090 (Docker with the NVIDIA runtime)
    cd crucible/data/kernel_debug/gpu_bundle && bash run_gpu.sh docker
    # Marlowe H100 (Apptainer; the image built from Dockerfile and pushed or converted first)
    cd crucible/data/kernel_debug/gpu_bundle && bash run_gpu.sh apptainer

`run_gpu.py` builds the reference, calibrates (case ownership, executed edges, sanitizer and perf calibration), then
grades oracle, floors, mutants and the flake run; results in `results/<id>.json`, the calibrated spec in
`tasks/<id>/kdspec.calibrated.json`. `lh_kernel_debug.py ingest-gpu --bundle <dir>` (§8) writes them back to the records.

## 8. Results

Run 2026-10-20 on this Mac (Docker, 2 CPUs and 3.8 GB per container, at most 2 at once; no GPU, no model call, $0).
Funnel: `docs/research/swarm-pipeline/runs/kernel_debug_funnel.json` (`lh_kernel_debug.py report`).

| family : split | mined commits | real changes | groups built | CPU structural pass | every CPU gate | authored lint-clean |
|---|---|---|---|---|---|---|
| kernel-bug : train (llama.cpp) | 78 | 46 | 12 | 11 | 11 | 11 |
| kernel-perf : train (llama.cpp) | 26 | 18 | 4 | 4 | 3 | 3 |
| kernel-migration : train (HeCBench, cudahandbook) | 24 | 24 | 5 | 5 | 5 | 5 |
| kernel-bug : held out (ik_llama.cpp) | 23 | 17 | 5 | 5 | pending x86 host | 5 |
| kernel-perf : held out (ik_llama.cpp) | 8 | 4 | 1 | 0 | 0 | 1 |

"Mined commits" are the commits that pass the message and diff filters into the per-change check; the commits refused
before it (llama.cpp since 2025-09-01: 314 not a behavioural fix or speed-up, 74 platform-specific, 6 reverts; ik since
2026-05-01: 79, 14, 1) are not counted there. Per-change drops: touches code outside the CUDA backend (19 + 8), reaches
no op the harness tests (7), reaches too many ops (>= 11; 13), no CUDA source changed (2), multi-GPU code only (1).

**19 candidates pass every CPU gate** (11 bug, 3 perf, 5 migration; all training), and **5 held-out bug candidates** pass
every structural CPU gate with their harness runs pending the x86 GPU host (the ik CPU backend needs AVX2; this Mac
emulates x86_64 without AVX, so the listing and the seeding run there, in `run_gpu.py`'s `fill_pools`).

| id | family | faults / milestones | interacting edges (claimed) | classes |
|---|---|---|---|---|
| kd-bug-llamacpp-1b8b8fb1 | bug | 3 | m3-m1, m3-m2 | race, logic, indexing |
| kd-bug-llamacpp-3b87408f | bug | 2 | m2-m1 | bounds, bounds |
| kd-bug-llamacpp-4b8c156e | bug | 2 | m2-m1 | launch_config, indexing |
| kd-bug-llamacpp-60ed515c | bug | 3 | all 3 pairs | overflow, indexing, overflow |
| kd-bug-llamacpp-7ac249cc | bug | 3 | m2-m1 | overflow, launch_config, logic |
| kd-bug-llamacpp-a45699aa | bug | 4 | all 6 pairs | grid_limit, launch_config, logic, stream_async |
| kd-bug-llamacpp-b2519561 | bug | 2 | m2-m1 | overflow, race |
| kd-bug-llamacpp-c434b3a5 | bug | 2 | m2-m1 | race, race |
| kd-bug-llamacpp-ca21f3a1 | bug | 2 | m2-m1 | race, grid_limit |
| kd-bug-llamacpp-eab11c90 | bug | 2 | m2-m1 | overflow, logic |
| kd-bug-llamacpp-efe7d4b1 | bug | 2 | m2-m1 | logic, divergent_barrier |
| kd-perf-llamacpp-4bc24fdb | perf | 2 | m2-m1 | perf |
| kd-perf-llamacpp-9d89aaa0 | perf | 2 | m2-m1 | perf |
| kd-perf-llamacpp-d2084ba6 | perf | 2 | m2-m1 | perf |
| km-cuda12-cudahandbook-19a4af6-3 | migration | 6 programs | L2-short | texref |
| km-cuda12-cudahandbook-b62f9ad-4 | migration | 4 programs | L2-short | surfref, texref |
| km-cuda13-hecbench-591f311-4 | migration | 4 programs | L2 (1 executed cross edge) | cccl3_functors, devprop_removed, memloc_api |
| km-cuda13-hecbench-8fcc759-4 | migration | 3 programs | L2 (1 executed cross edge) | cccl3_functors, devprop_removed |
| km-cuda13-hecbench-ba3a30a-3 | migration | 2 programs | L2-short | cccl3_functors, devprop_removed |
| kd-bug-ikllamacpp-000b1a1e (held out) | bug | 4 | all 6 pairs | precision, indexing, indexing, logic |
| kd-bug-ikllamacpp-5b19ab0c (held out) | bug | 4 | 5 pairs | race, indexing x3 |
| kd-bug-ikllamacpp-5b1dc162 (held out) | bug | 3 | all 3 pairs | stream_async x2, indexing |
| kd-bug-ikllamacpp-5b8c07ff (held out) | bug | 3 | all 3 pairs | logic, race, bounds |
| kd-bug-ikllamacpp-89213aa6 (held out) | bug | 2 | m2-m1 | logic, grid_limit |

Claimed edges are shared ops (static); executed edges are the GPU calibration's job (cases broken by two single reverts)
and every bug/perf candidate is pending it, with every gate in `PENDING.json`. Dropped at the CPU gates:
kd-bug-llamacpp-dd366156 (A' does not compile: one fault is a build-compatibility fix), kd-perf-llamacpp-3ad8a68c
(milestone m1 has no perf case in the harness), kd-perf-ikllamacpp-e898e200 (m1's revert mutant does not build; the
leak probe hit a CUDA toolkit sample that holds a generic B-only line). Case pools range from 8 to 10,657 cases per
milestone; the grader samples at most 3,000 per milestone (the fixes' own added cases first, then a stride).

Sealing (`report`): repositories disjoint (train llama.cpp, HeCBench, cudahandbook; held out ik_llama.cpp), 0 cutoff
violations, 5 time overlaps (the held-out and training periods overlap: the split is by repository, not time), 31
contamination hits (27 diff code-substrings, 4 prompt), all from ik being a fork of ggml (shared code, not shared fixes).

Fixed while running: Docker Desktop kept serving a deleted and recreated bind-mount source, so the seeding run after
the case listing saw no output directory and every seeding check failed; the runs now write into a fresh directory per
call (`kernel_debug_world.cpu_harness_runs`) and `lh_kernel_debug.py repool` re-ran the harness gates alone. Unit
selection skips a file the start tree does not have. A' compiles only the units that exist at A'.

Verified GPU-less: the staged GPU plan run in the verifier image without a GPU returns `masked`
(`no GPU visible (nvidia-smi)`, reward `None`), never a 0. Tests: `test_m17_kd_*` 28 functions, kernel lane total 53; m17
non-Docker 623 passed, 1 skipped; ruff on the lane files and lint-imports (4 kept) clean.

### 8.1 GPU gates on the lab RTX 5090 (2026-10-20/21, halted by the user before the perf family)

More supply first: llama.cpp mined back to 2024-06-01 (7 more training groups) and ik_llama.cpp back to 2025-06-01
(6 more held-out bug groups, 2 perf; contamination `fallback`). Results ingested from `data/kernel_debug/gpu_results_5090`
(funnel: `runs/kernel_debug_funnel.json`, field `gpu_failed_gate` per task).

| family : split | through the CPU gates | GPU: survive every gate | dropped at |
|---|---|---|---|
| kernel-bug : train | 18 | **2** (kd-bug-llamacpp-1b8b8fb1, kd-bug-llamacpp-ca21f3a1) | S1 executed 9, calibration 6 (one milestone left), S8 mutants 1 |
| kernel-bug : held out | 11 | 0 | host CPU gates 2 (no op-test case for a milestone's ops), S1 executed 3, calibration 6 |
| kernel-migration : train | 5 | **2** (km-cuda13-hecbench-591f311-4, km-cuda13-hecbench-ba3a30a-3) | S5 oracle 3 |
| kernel-perf : train / held out | 3 / 2 | pending (not run) | |

What the drops mean. S1 executed: the start (every fault in) passes every case of the repository's op test on sm_120,
swept in full on fixed seeds, and every sanitizer check: the fixes are real but not visible on this GPU (Hopper/Ampere
code paths, int32 overflow at sizes the tests never use, multi-GPU). Calibration: fewer than two milestones keep a stable
owned case (a revert that breaks no case, or B itself failing the case on a fresh seed). S8: kd-bug-llamacpp-3af2e58d's
race milestone is caught on 59 of 60 runs (S10 also fails). Migration: the cudahandbook originals (CUDA 11.8 PTX
JIT-compiled for sm_120) write zeros through legacy surface references, print timings, or copy >256 MB; one HeCBench
program's output differs run to run. Survivors have no executed cross edges (each revert breaks only its own cases).
The 5090 is not the GPU for most of this supply: the H100 (sm_90) calibration on Marlowe is the next lever.

### 8.2 Why the 5090 gates failed, and what was corrected (2026-10-22)

Every failed GPU result was re-read against the grader's own reports on the 5090. Six pipeline bugs explain part of the
drops; the rest are faults no case of the repository's op test reaches on sm_120, or that only an H100, an older GPU or
an AMD GPU can show. Classes: **(a)** task defect corrected, **(b)** pipeline bug fixed (code and a unit test),
**(c)** not fixable on sm_120 or not fixable honestly (evidence recorded).

Pipeline bugs (b), all in the GPU runner or grader (`DESIGN.md`, "GPU gate fixes"):

| bug | effect on the 2026-10-21 run |
|---|---|
| `pool_cap=0` reached the plan as the integer 0 and fell back to the 1,000-case stride | the S1 sweep "of every pool case" covered 1,000 of 1,620-10,632 cases on 17 tasks; 9 of the 14 "not real on this GPU" drops were swept on 10-20% of their pool |
| pool grades were reused by a key without the grader's code | sweeps graded by an older grader (one with a launch-count bug) were read back after it was fixed |
| the S1 start drop applied to perf milestones; pool grades ran the perf check, whose missing bar masked the grade | every perf task would have been dropped (`kd-perf-ikllamacpp-09e3d098` was) |
| a passing perf check's ratios were read from the wrong field; the candidate was timed with the CUPTI launch hook and the reference without it | every perf case read "unmeasured"; the same build timed 4-14% slower as the candidate |
| sanitizer checks ran on the pool's first four cases | races in kernels those cases never launch (transposed copy, flash attention's vector kernel, the MMQ stream-k fixup) were invisible; now the cases come from a CUPTI launch profile of the fault's kernels |
| `120-real` instead of `120a-real` | trees from late 2025 failed ptxas on the FP4 MMQ unit at sm_120 |

Also: S10 counted masked grades as passes; a stability check passed a q4_1 case B meets at 0.9-2x of its tolerance (the
`3af2e58d` S10 false negative); the runner waited for an idle GPU before every correctness calibration.

Per task (30 failures: 16 llama.cpp bug, 11 ik_llama.cpp bug held out, 3 migration). "rerun" = recalibrated on the 5090
with the fixes (results in `gpu_results_5090` once the queued run ends; ingest with `lh_kernel_debug.py ingest-gpu`).

| task | root cause | class | change |
|---|---|---|---|
| kd-bug-llamacpp-60ed515c, 44d1bb19, fcf94bd1, eab11c90, efe7d4b1, c434b3a5, 3b87408f | sweep covered 1,000 of 5,180-10,632 cases; race/barrier faults checked on cases that never launch their kernels | (b) | rerun: full sweep, targeted sanitizers. Likely (c) for the overflow-only faults (60ed515c m1/m3, 44d1bb19 m1/m2, eab11c90 m1: int32 overflow or fp16 overflow at sizes or input ranges the op test never uses; a case big enough exceeds the 200 MB dump limit or hours of CPU reference) and for 3b87408f (out-of-bounds reads inside the backend's memory pool: no sanitizer sees them) |
| kd-bug-llamacpp-cef66d4a, a45699aa | grid-limit faults (more than 65,535 rows, batches past 1,024 matrices) at sizes no repository case has | (a)+(b) | grader cases at the report's shapes (`harness_extra`, disclosed in the prompt); rerun |
| kd-bug-llamacpp-7d70c81f | m2 needs a long-context decode; m4 is integrated-GPU only; m1's race needs fully masked KV blocks | (a)/(c) | grader cases for m2; m4 (c) |
| kd-bug-llamacpp-b2519561, b4a950d4 | race milestones checked on cases that never launch the racy kernel; b2519561 m1 needs a >2 GB copy (dump limit), b4a950d4 m1 is a PTX 7.0 / Volta selection, m3 a partial-offload buffer the op test cannot build | (b)/(c) | rerun with targeted sanitizers |
| kd-bug-llamacpp-3af2e58d | m3's only owned case is a borderline q4_1 tolerance case (S10 false negative 1/60); its race was checked on cases that never launch the fixup kernel; m2 had been dropped the same way | (b) | margin rule, targeted racecheck; m2 restored to the rerun |
| kd-bug-llamacpp-7ac249cc | m1 is a speed change (64-bit strides for unrolling: same results), m2 an AMD-only kernel selection | (c) | none: one measurable milestone |
| kd-bug-llamacpp-4b8c156e | m1 changes only the launch shape (the old one-thread blocks compute the same values) | (c) | none |
| kd-bug-llamacpp-b377258d | m1 moves FP16 activations to another kernel whose results stay inside the case tolerance (pool fully swept) | (c) | none (a tighter tolerance would not be the repository's check) |
| kd-bug-ikllamacpp-000b1a1e, 1ef4f2f7, 35d83401, 44b81230, 5b8c07ff | every owned case is a q4_0 flash-attention case, and ik's CPU reference disagrees with B's own GPU output on them (nmse ~1.0 vs a 5e-4 tolerance; B fails its own op test there); several faults are Pascal/no-tensor-core paths or asymmetric K/V head sizes the 2024 harness cannot build | (c) mostly; rerun for the full 1,620-case sweep and targeted race checks (44b81230 m1, 5b8c07ff m2) | rerun |
| kd-bug-ikllamacpp-89213aa6 | m2 is a grid limit past 65,535 rows; m1 never differs on the swept cases | (a) | grader cases for m2; rerun |
| kd-bug-ikllamacpp-4da3f014 | m2 is a pre-Ampere bf16 path | (c) | none |
| kd-bug-ikllamacpp-5b1dc162 | m1/m2 are mixed-type graph-reduce additions the 2024 harness cannot construct | (c) | none |
| kd-bug-ikllamacpp-8dcd4428 | pool fully swept: neither fault changes a result on sm_120 | (c) | none |
| kd-bug-ikllamacpp-47f22c74, 5b19ab0c | ops with no case class in ik's op test (FUSED_UP_GATE, REDUCE, MASK_TOPK, INDEXER_TOPK); 47f22c74 m3 is multi-GPU | (c) | none |
| km-cuda13-hecbench-8fcc759-4 | scan3: a CCCL 3 scan adds floats in another order (bitwise compare); softmax_online copies 9.9 GB back to check itself | (a) | f32 compare within the program's own 1e-3; `d2h: none` (the program's own report); rerun |
| km-cuda12-cudahandbook-19a4af6-3 | tex1dfetch_big run without its size argument prints its usage and launches nothing | (a) | runs on 256 MB; rerun |
| km-cuda12-cudahandbook-b62f9ad-4 | surf1Dmemset's original writes zeros on sm_120 (legacy surface references, current driver); globalReadTex copies 738 MB; the shmoo prints only timings | (a)/(c) | globalReadTex graded on its own sum check; r01 and r04 dropped as not measurable; rerun |

Perf family. `kd-perf-llamacpp-4bc24fdb` calibrated with the fixed timing: B against itself within 0.98-1.02 on the
large cases, m2's revert 1.5x and 1.9x slower (bar 1.071, kept); m1's revert is 1.10x on the best case against a 1.07
bar plus the 0.10 headroom, so m1 drops and the task with it (one milestone). `kd-perf-llamacpp-9d89aaa0`: m1 kept (two
cases), m2's revert is at most 1.06x slower on any candidate case (noise-level) and drops. `kd-perf-llamacpp-d2084ba6`: m1 kept
(bar 1.42, revert 1.58x), m2's revert is at most 1.12x and drops. Both end with one milestone: not gradable on the 5090 (c;
the H100 calibrates its own bars). `kd-perf-llamacpp-3ad8a68c` (dropped at the CPU gates: no RMS_NORM perf case) passed them
with two grader RMS_NORM cases, but its start does not build on the GPU: B's `binbcast.cu`, a unit no fault touches, uses the
integer-division helpers m1 adds, so the CPU gate (which compiles only the touched units) missed it (c: a correct start would
change m1's fault; a CPU gate that also compiles units including the changed headers is open). The two held-out ik
perf tasks have no case of the speed-ups' quantization types (iq2_k/iq3_k/iq4_ks are not in the 2024 harness's type
list); grader cases at those types (512 tokens, 8 experts) hung the reference build on its first chunk (idle, 0% GPU,
over 30 minutes) and were withdrawn: (c) until that hang is understood.

### 8.3 GPU gates on DeltaAI GH200 (CIS262200, from 2026-10-22)

**Host.** NCSA DeltaAI, account `bjhg-dtai-gh` (CIS262200), partition `ghx4`: GH200 120GB (sm_90, aarch64 Grace), driver
595, Apptainer 1.4, no Docker, no subuid range. The bundle is the 5090 bundle's inputs (task trees, patches, specs,
`cpu_host.json` from the x86 host) with this host's own builds, calibration and gates; results go to
`data/kernel_debug/gpu_results_gh200` and are ingested with `ingest-gpu --hardware-only` (the 5090's calibration stays
the record's own). Job scripts: `data/kernel_debug/gpu_results_gh200/jobs`. The login environment exports
`SBATCH_ACCOUNT` for another project, which overrides `#SBATCH --account`; jobs go through `submit.sh` (unsets it and
passes `--account=bjhg-dtai-gh`), checked in `squeue -o %a`.

**Runtime.** SIFs `crucible-lh-k:base-12.8.1` and `base-13.0.2` built on a compute node from the bundle Dockerfiles
(`--build-sif`, about 3 minutes each). `CRUCIBLE_APPTAINER_ROOT=user`: the grader runs as the submitting user, so every
grade here is `trusted: false` (the candidate is not dropped to its own uid). That is acceptable for the gates (the lane's
own oracle, floors, mutants) and not for grading a policy. Two runtime bugs only an Apptainer host shows were fixed first:
the grader's rebuilt environment dropped the container's LD_LIBRARY_PATH, the only route to the driver's libraries under
`--nv`, so every CUDA program saw no device (the first migration oracle was masked with "Unified Memory not supported");
and a job step without its own `--mem` took the whole allocation, so a second stream waited.

**Parity smoke (job 3439700, 2 GPUs, 2 h 13 min).** `kd-bug-llamacpp-ca21f3a1`: both milestones kept, oracle 1, floors
0, both revert mutants fail their milestone, S10 0/60 false negatives (CP upper 0.049): passes every GPU gate on the
GH200 as on the 5090 (1 h 16 min cold). `km-cuda13-hecbench-591f311-4`: oracle 1, floors 0, S10 0/60 (no mutants in this
window): passes (53 min).

**Full run.** Four one-GPU stream jobs (perf alone on its GPU; three correctness streams, migrations first on one),
12 h each, queued 2026-10-23 01:00 (the exclusive four-GPU node was estimated a day out). Not run here, because their cause does
not depend on the GPU: `kd-bug-llamacpp-7ac249cc`, `-4b8c156e` (speed-only, AMD-only), `-b377258d` (within tolerance),
`kd-bug-ikllamacpp-4da3f014` (pre-Ampere), `-5b1dc162` (the 2024 op test cannot build its cases), `-47f22c74`, `-5b19ab0c`
(no op-test class, multi-GPU only), and `kd-perf-llamacpp-3ad8a68c` (its start does not build; below).

**3ad8a68c.** m1's fix adds the integer-division helpers to `common.cuh`; a later commit (no fault) uses them in
`binbcast.cu`, so reverting m1 at B gives a start that does not build. The S3 start build now also compiles
`start_dependents` (units no fault touches that use a name a fault's header change takes away), so the CPU gate fails
such a start. It flags `binbcast.cu` here; in the other records, it flags 3 units in `kd-bug-ikllamacpp-5b8c07ff` and 5 in
`kd-perf-ikllamacpp-09e3d098`, which the GPU sweeps built. The task itself is not corrected: keeping the helpers at A'
would change m1's fault, because the start would then carry the tool the fix introduces.

### 8.4 The remaining grader gaps on the existing tasks (2026-10-23, RTX 5090)

Grader fixes only; no task was made easier or harder and no new task was built. Evidence:
`data/kernel_debug/evidence/2026-10-23/`.

**1. A case's inputs depended on the cases before it (b).** The harness drew every input from one process-wide counter,
so a case's inputs changed with its position in a run: the calibration graded 96-case chunks, S8 graded each milestone's
few owned cases, and the same seed gave different inputs. This is why three revert mutants "passed" milestones the
calibration had them break: `cef66d4a` m2 (one break on one of seven seeds), `a45699aa` m3 (two breaks, fresh seed
only), held-out `35d83401` m1 (14-15 of 360 cases break per seed, a different set each run). A selected case now restarts
the stream from (seed, case name); the 2024 layout's static per-thread engines and the 2025 layout's first-tensor
thread count are rebuilt per call. On the 5090 the same cases dump bitwise identical outputs alone and inside a chunk, in
both dialects. Ownership and executed edges now read the fixed-seed verdicts only (`per_case_fixed`): a break seen only
on a fresh seed does not repeat at S8.

**2. ik_llama.cpp's own CPU reference is wrong on its flash-attention cases (b, held out).** B failed 237 of its own 455
FLASH_ATTN_EXT cases (`000b1a1e`), including f16 ones, so every ik flash-attention task lost its owned cases. The split is
by code path: cases with `max_bias = 0` take ik's IQK CPU kernel (`iqk_flash_attn_noalibi`) and fail; ALiBi cases take
ggml's generic CPU kernel and pass. An exact computation of ggml's attention semantics (in double, from the CPU graph's
own inputs) settles which side is wrong: B's GPU matches it within 0.001 of the tolerance on all 64 probed cases, and the
IQK kernel misses it by 340-450x (up to 1.9e6x) exactly where B failed; B's GPU is bitwise stable across processes. A
CPU reference without IQK does not build at B, and B's unpatched op test does not compile, so the grader now replaces
the CPU column with the exact result for exactly the cases the calibration's probe shows the CPU wrong on, with B's
GPU matching the exact result and the exact result matching the CPU on at least 8 cases where both kernels agree
(`calibration.fa_reference`). Non-deterministic cases are refused.

**3. Migration S8 never counted a killed mutant (b).** The runner read per-gate verdicts from the reward document's
`gates`, which `kernel_plan` leaves empty, so every fault mutant read "escaped". On the 5090's own grades:
`km-cuda13-hecbench-8fcc759-4`'s `shfl_down_to_up` mutant fails r02, and `km-cuda12-cudahandbook-b62f9ad-4`'s two
filter-mode mutants fail r03. The three that pass are equivalent on their program's inputs, with the reason on the
window (`equivalent_mutants`): `drop_syncwarp` and `mask_active` in `softmax_forward_online_kernel2` (every lane of each
warp reaches the full-mask shuffles; the only earlier returns are never taken or warp-uniform at B=8, T=1024, V=50257),
and `addr_clamp_to_border` in surf2Dmemset (all texture reads are in range: an 8x8 texture read at base (0, 0), step
(1, 1)).

**4. `kd-bug-llamacpp-3b87408f`: not measurable (c).** Both faults are out-of-bounds reads of padding columns in MMQ;
the values read are discarded, so no result changes, and compute-sanitizer memcheck is clean on the start, the start
plus fix 1, B with fault 2 reverted and B, with the VMM pool and with the legacy pool (`-DGGML_CUDA_NO_VMM=ON`) on 55
cases up to 1,024 experts: the over-read stays inside the pool's slack. Showing it would need an exact-size allocator
patched into the backend under test, which is not grading the repository's code.

**Rerun (5090, 2026-10-23 03:30-13:25 PDT, `dbg/pipeline3.sh` and `pipeline2.sh`).** Recalibration and every GPU gate
with the fixes. A first start with three bug-task streams left empty grades (concurrent nvcc builds, likely memory) and
was restarted with two; the exact-reference marker first went to stdout, where it split the console verdict line and the
probe never saw it (moved to stderr; restarted).

| task | split | calibration | gates | verdict |
|---|---|---|---|---|
| kd-bug-llamacpp-1b8b8fb1 | train | m1-m3 kept (32, 32 owned cases; m1 by racecheck) | S5 1, floors 0, 3/3 reverts fail, S10 0/60 | **pass** |
| kd-bug-llamacpp-ca21f3a1 | train | m1 (racecheck), m2 (4 cases) | S5, S8, S10 0/60 | **pass** |
| kd-bug-llamacpp-efe7d4b1 | train | m1 (1 case), m2 (synccheck) | S5, S8, S10 0/60 | **pass** |
| kd-bug-llamacpp-cef66d4a | train | m1, m3 (2 owned cases each); m2 dropped: no fixed-seed break, racecheck not dirty on the revert | S5 1, floors 0, revert-m1 and revert-m3 fail their milestones, partial-m1 fails m1 (partial-m3 equivalent), S10 0/60 | **pass** (was S8) |
| kd-bug-llamacpp-a45699aa | train | m1 only: m3's breaks were all on a fresh seed (none on the fixed seeds), m2 and m4 never break a case | - | not gradable: one milestone (c) |
| kd-bug-ikllamacpp-000b1a1e | held out | exact reference on 205 cases (agree 48, unmodelled 51); B then passes them, and no revert breaks any case on the fixed seeds | - | not gradable: the faults do not change a result on sm_120 (c) |
| kd-bug-ikllamacpp-1ef4f2f7 | held out | exact reference on 15 cases; no revert breaks a case | - | not gradable (c) |
| kd-bug-ikllamacpp-5b8c07ff | held out | exact reference on 205 cases; no owned case; m2 by racecheck, m3 kept on a memcheck dirty on B too | S5 0 | not gradable: under the corrected sanitizer rule m3 drops, one milestone left (c) |
| kd-bug-ikllamacpp-44b81230 | held out | exact reference on 19 cases; m1 kept on a racecheck dirty on B too | - | not gradable: none left under the corrected rule (c) |
| kd-bug-ikllamacpp-35d83401 | held out | no case switched: the CPU agrees with the exact result and B's GPU is 2,300-2,700x off on ALiBi + softcap cases (gpu_disagrees 14); m2 only | - | not gradable: one milestone (c) |
| kd-bug-ikllamacpp-89213aa6 | held out | m2 only (grid-limit grader cases); m1 never breaks a case | - | not gradable (c) |
| km-cuda12-cudahandbook-b62f9ad-4 | train | - | S5 1, floors 0, both filter-mode mutants fail r03 (clamp-to-border equivalent), S10 0/60 | **pass** (was S8) |
| km-cuda13-hecbench-8fcc759-4 | train | - | S5 1, floors 0, shfl_down_to_up fails r02 (two equivalent), S10 0 false negatives over 59 measured grades (one masked: the original program's launch timed out), CP upper 0.0495 | **pass** (was S8), at the S10 limit |

Two more grader rules came out of the rerun: a sanitizer tool keeps a milestone only when it is clean on B and dirty on
the revert (`sanitizer_discriminates`; before, dirty on both kept `44b81230` m1 and `5b8c07ff` m3), and the migration S10
leaves masked grades out of the denominator (as the bug lane has since 10-22; the run's own count was 1/60, CP 0.0766).
`8fcc759-4`'s S10 is recomputed from its sixty grades with that rule, not rerun.

**Totals passing every GPU gate on the 5090:** 8 tasks, all training: 4 kernel-bug (`1b8b8fb1`, `ca21f3a1`, `cef66d4a`,
`efe7d4b1`) and 4 migration (`591f311-4`, `ba3a30a-3`, `b62f9ad-4`, `8fcc759-4`); held out: 0. With the CPU reference
corrected, the ik flash-attention faults show no result change on sm_120: the held-out kernel set needs other sources or
another GPU.

## 9. Open issues

1. **Every GPU gate is pending** (`PENDING.json`): S1 executed (B passes, each single revert fails its milestone),
   case calibration, S6 executed edges, the oracle's reward 1, floors 0 and mutants failing on the GPU, sanitizer
   clean on B and dirty on the revert, launch evidence, perf bars, S10 flake (60 grades), then S12-S15. A candidate
   whose revert breaks no case on the GPU (a fix for hardware this host lacks) is dropped then, not now.
2. **Perf-bar noise is unmeasured.** Bars are `1 + max(3 x noise, 0.05)` with 10% headroom, from the GPU host's own
   repetitions; the 5090 and the H100 calibrate separately and a bar never moves between them.
3. **Held-out harness runs need the x86 host**: the ik CPU backend does not build on aarch64 and SIGILLs under
   emulation. The bundle lists the pools there; held-out ik harness repairs (the 2024 layout's stale cases and API
   drift) are anchored and recorded per record.
4. **Contamination.** Training uses the post-2026-05-01 llama.cpp fixes first but falls back to 2025-09-01 for supply
   (records say `fallback`); every held-out fix is post-release. HeCBench migration commits are 2026-04-02
   (pre-release); cudahandbook's are post-release. The student may have seen pre-cut fixes; the S13 baseline measures it.
5. **Op-map over-approximation.** Ops come from a static reference graph from the changed functions to the dispatch;
   it may name ops the fix never affects. Calibration keeps only the cases each revert breaks.
6. **Prompts are re-authored after calibration** drops a milestone (`author` drops calibration-dropped milestones).
7. **Alternatives** (a second correct fix) are not built yet; the golden-output check accepts any fix that matches
   the reference within tolerance, so they are expected to pass, but S8 has no explicit alternative.
8. **cudahandbook windows are L2-short** (programs share no executed edge); they test the migration skill breadth-first.
9. **Licences**: llama.cpp and ik_llama.cpp MIT, HeCBench BSD-3, cudahandbook BSD-2 (recorded per record).
