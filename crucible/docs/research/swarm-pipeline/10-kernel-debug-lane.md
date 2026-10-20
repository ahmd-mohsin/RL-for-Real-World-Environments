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

(filled at the end of the run: per family, mined -> real -> grouped -> candidates per CPU gate, with drop reasons.)

## 9. Open issues

(filled at the end of the run.)
