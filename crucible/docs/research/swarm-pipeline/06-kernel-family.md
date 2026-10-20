# 06 The kernel-migration family (MVP task family for 7B students)

Spec: `00` (every state S0-S17 applies), `04` (machinery as built), grader audit `docs/reviews/2026-10-09-articles-crosscheck.md`,
`migration_tasks/task04_pcl_gpu_cuda12` (the top rung). Student: Qwen2.5-7B-Instruct (OpenRouter `qwen/qwen-2.5-7b-instruct`;
Qwen2.5-Coder-7B is not served there). Training GPU: Marlowe H100 (sm_90). Validation GPU: lab RTX 5090 (sm_120).

## 1. Intent card K1 (behaviour, not topic)

"Migrates GPU code across toolkit and API generations (removed texture and surface references, warp-synchronous code to
`*_sync` intrinsics or cooperative groups, deprecated runtime calls, dropped architectures) while preserving exact results,
race-freedom, concurrency safety and speed."
- **Evidence (frontier, task01-04):** task04 v3: GPT 6 Astra correct but 3 perf bars missed; GPT 5.6 sol broke the
  out-of-range texture semantics of NCV's mirror-border filters while optimizing (the only failure on correctness). v1
  without speed bars was solved by both. So the frontier's failure axes are *texture addressing semantics* and *speed*.
- **Evidence (student):** filled by S13/S15 once the student is graded (section 7); until then the card is `proposed`.

## 2. The ladder (one behaviour; difficulty only along real axes)

| rung | scope (real axis) | gates beyond the rung below | pool |
|---|---|---|---|
| L1 | one source file with removed or warp-synchronous constructs | build for the target arch, golden outputs (fresh seed) | train (labelled `short`: no edge) |
| L2 | one module: several kernels + a shared dependency (header, helper) | executed `requires` edge (dependent fails without its prerequisite) | train |
| L3 | the module + a downstream consumer in the same repo | consumer milestone requires the module; determinism; concurrency equivalence where the API is per-call | train |
| L4 | L2/L3 scope | compute-sanitizer racecheck/synccheck/memcheck clean; perf bar vs the original with >= 15% oracle headroom | train + benchmark |
| L5 | task04 scale (whole repo, consumers, speed bars) | task04 v3 gates through the kit grader | benchmark only |

Never used as difficulty: vaguer prompts, stricter graders, hidden formats. The rung is computed from the window, never chosen.

## 3. Ground truth (never invented)

1. **Mined migration commits** (the main source). Scouts (`sources/github.py`, commit search, cached, unauthenticated) run
   per migration kind; clones are scanned with pickaxe (`git log -G`) for commits whose removed lines hold a legacy
   construct and whose added lines hold its modern form. Kinds: `texref` (texture/surface references -> objects),
   `warp_sync` (`__shfl*`/`__any`/`__all`/`__ballot` -> `*_sync`), `volatile_warp` (warp-synchronous volatile reductions ->
   `__syncwarp`/shuffles), `thread_api` (`cudaThread*` -> `cudaDevice*`), `arch` (dropped `sm_<50` targets),
   `lib_api` (removed cuSPARSE/cuBLAS legacy calls). Window: A = B's parent restricted to the repo; oracle = B's change to
   the CUDA sources. Permissive licences only (MIT, BSD, Apache-2.0, ISC, Zlib, Unlicense), read from the clone's licence file.
2. **Seeds:** task04 v3 (L5) and its M8 spans `people`, `ncv`, `kinfu_family` (L4/L5) through the kit grader
   `crucible_pack_coding.verify.task04`; listed with their open audit items (section 6).
3. **Rebreak** of a migration = reverting real migration commits in one tree (the window at A is exactly that).
4. **Swarm-built** migrations only on a real start repo with a real request (an issue asking for CUDA 12 support); not in
   this MVP's numbers.

## 4. The task

- **World** (`crucible-k-world:<window>`): `nvidia/cuda:12.8.1-devel-ubuntu24.04` (multi-arch: arm64 locally for compile
  gates, x86_64 on GPU hosts), the repo at A in `/workspace/<repo>` as a fresh one-commit history tagged `base`, offline,
  `CRUCIBLE_CUDA_ARCH` (90 Marlowe, 120 lab). Only CUDA 12.8 is installed. Leak probes: no file outside the workspace holds a
  line only B adds; no git object beyond `base`; no legacy toolkit.
- **Verifier image** (`crucible-k-verifier:<window>`): the World plus `/opt/legacy` (root, 0700): the original A built in a
  CUDA 11.8 builder stage (the last toolkit with texture references) as PTX (`compute_60`) run on the GPU by driver JIT,
  statically linked against cudart; plus the grader bundle (kit runtime + `lanes/kernel_plan.py`). Hosted by Docker
  (`--runtime=nvidia`) or Apptainer (`--nv`, same OCI image).
- **Harness** (grader-owned, never in the World): one driver per milestone calling the repo's host entry points that
  exist at A, inputs from a seed, outputs written to a grader-chosen sealed directory. Written by the verifier writer
  (author family, call) from A only (V2); qualified by execution: compiles against A (legacy) and B (modern); the original
  is deterministic over reruns; the oracle matches (S5). Where the repo has its own test or example program covering the
  unit, it is preferred over a written driver.
- **Milestones:** one per unit (file or module, consumer); passes when the unit builds for the target arch, every output
  matches the original's (bitwise, or a tolerance calibrated on >= 120 oracle runs and stated in the prompt), and its
  rung's gates pass. Reward = 0.5 Phi + 0.5 [all gates] (lane convention).

## 5. GPU grading (kit plan `lanes/kernel_plan.py`, stdlib + kit only)

| gate | rule (audit lesson it encodes) |
|---|---|
| static | no removed construct left (comment-stripped scan of the window's files) |
| build | the grader owns the nvcc/CMake command and flags: `-gencode arch=compute_<A>,code=sm_<A>`; run unprivileged; outputs sealed |
| binary | `cuobjdump`: SASS for sm_<A> only, no PTX for another arch (no JIT escape), no CUDA 11 runtime |
| golden | original (root) vs candidate (uid 65534) on a seed drawn at grading time, revealed after the candidate ran |
| launches | a grader CUPTI injection records kernel launches: every harness call must launch >= 1 kernel (no host port) |
| concurrency | L3+: N host threads equal the candidate's own solo runs bitwise |
| sanitizer | L4: racecheck, synccheck, memcheck; every summary line 0 and the sanitizer exit code 0 |
| perf | L4: grader wall clock around its own process launches, R vs 3R repetitions (startup cancels), fresh inputs per repetition, alternating original/candidate, best of 3 + one re-measure, load guard (`uptime`, `nvidia-smi`), spread check -> infra; bars per GPU model |

Never trusted: exit codes, stdout rows, the candidate's build files, timing inside its process (P26). Fresh HOME/TMPDIR per
run (no cache-and-replay). A grader exception masks the grade (V8). No GPU backend -> masked `no_gpu`, never a 0.

## 6. Gates per state (what runs where)

| state | kernel lane | where |
|---|---|---|
| S1 | licence, size, the window touches CUDA sources, A has the legacy construct | CPU |
| S3 | World builds; A *fails* to build for sm_90 and sm_120 on CUDA 12.8 (the migration is needed) or carries a warp/race defect; leak probes; the original builds in the legacy stage | CPU (compile) |
| S4 | oracle B builds for sm_90 and sm_120 | CPU (compile) |
| S5 | harness compiles against A and B; oracle golden-equal on fresh seeds | CPU / GPU |
| S6 | executed edges: the dependent unit fails to build or to match without its prerequisite | CPU (build edges) / GPU (output edges) |
| S7 | prompt author (qwen3.8-flash), realism, two-way ledger, lint, stated host-API names and headers | CPU, paid |
| S8 | floors: empty, claims-only, lower-the-arch, host-port, stub kernels = 0; revert mutants (CPU) + fault-class mutants (GPU) | CPU / GPU |
| S9 | second verifier (deepseek) writes shadow drivers from the prompt; differential on GPU | GPU |
| S10 | 60 oracle grades at concurrency 1 and 4, CP bound <= 5% | GPU |
| S12 | hacker (deepseek) in the World; executed attempts graded | GPU |
| S13/S15 | student (qwen-2.5-7b) >= 5 attempts per arm, counterfactual on `locate` (files + removed constructs named) and `prerequisites` | GPU |

**Fault-class mutants** (applied to B; each must fail its milestone): texture addressing clamp<->border/mirror, filter
point<->linear, normalized coordinates flipped, read mode element<->normalized, out-of-range fetch returning a clamped value
instead of 0 (task04's sol_v3 failure), shuffle mask `0xffffffff` -> `__activemask()` inside divergent code, a dropped
`__syncwarp`, a dropped `__syncthreads`, a per-call resource turned into a static global (task04's concurrency negative).

## 7. Compute, arch and budget

- Every build takes the arch from `CRUCIBLE_CUDA_ARCH`; perf bars are keyed by GPU model (`nvidia-smi --query-gpu=name`) and
  calibrated per model; a model without bars runs the perf gate as `uncalibrated` (L4 not graded there).
- CPU-side work runs locally: scouting, clones, images (arm64), compile gates, prompts, ledgers. GPU-side work is staged as
  one bundle (`scripts/lh_kernel.py stage-gpu`) for the lab or Marlowe.
- Spend: one USD ledger with a hard $15 cap; author qwen3.8-flash, reviewer/second verifier/hacker deepseek-v4-flash,
  student qwen-2.5-7b-instruct. No frontier runs.

## 8. As built and results (2026-10-19; `runs/kernel_funnel.json`, `runs/kernel_targets.json`)

**Code.** `factory/lanes/kernel_mine.py` (kinds, licences, pickaxe), `kernel.py` (program discovery, compile/run milestones,
rungs, `KernelTask`, prompt + ledger + lint, fault mutants, seeds), `kernel_plan.py` (the kit grader, observation hooks,
CUPTI launch counter), `kernel_world.py` (images, compile jobs, leak/history probes, `DockerBackend`), `kernel_gates.py`
(CPU gates); `scripts/lh_kernel.py` (scan, windows, gates, author, kitcheck, target, stage-gpu, report) and
`scripts/lh_kernel_gpu.py` (the GPU half, stdlib, Docker or Apptainer). Tests `tests/unit/test_m17_k_*.py` (24 + 2 Docker).

**Milestones as built.** `compile` (a unit compiles for the arch, its migrated files hold no graded construct: CPU) and
`run` (the program builds, carries sm_<arch> code only, launches kernels, matches the original's observation: GPU), the
run milestone `requires` its units. Observation = normalized stdout + the device-to-host byte stream (link-time wraps of
`cudaMemcpy*`) + files written, against the original; `srand` is mixed with the grade's secret seed.

**Funnel (CPU, $0).** 9 repos scanned, 51 migration commits with a permissive licence -> 40 windows (31 commits had no
buildable program: OpenGL, NPP, HIP, missing data, or a library with no consumer; 12 were deprecation only) -> 21 pass
every CPU gate: S3 (the start fails to build for sm_90 and sm_120, or holds semantic warp code), S4 (the oracle builds for
both), the original builds with CUDA 11.8 as compute_60 PTX (15 needed the enum repair: chLib names CUDA 12 error codes),
S6 (20 executed compile edges), S8 (no-op, claims-only and every revert mutant fail through the real kit grader; 22 fault
mutants build and wait for the GPU), S7 (21 prompts: 20 by qwen3.8-flash, 1 template; ledger and lint clean). Drops after
gating: 13 oracles that do not build on CUDA 12.8 with the start's build (other bit rot left in B), 4 static-only starts,
1 original that needs a removed cuSPARSE call. Defects found by executing: CRLF sources made unappliable oracle patches
(text-mode diff); oracles moved to driver-API calls (the grader now always links `libcuda`).

**Rungs (honest).** 20 L1 (cudahandbook 15, HeCBench 3, rodinia 1, SMs 1), 1 L2-short with an L3 shape (cub warp
reduce: migrated headers used by three `examples/` consumers; the start builds, so its edges execute only on the GPU;
the only L4 candidate). 0 L2/L3 with executed edges and 0 calibrated L4: real migration commits are local (one sample, one
file); the dependent ones (nbody, scan headers, cub tests) carry other bit rot that keeps B from building on CUDA 12.8.

**Benchmark seeds (L5/L4).** task04 v3 and its M8 spans people, ncv, kinfu_family, graded on the 5090 with the kit grader
(17/17 as expected, `docs/modules/reports/m8_coding_gpu.json`). Open before sealing: the consumers' `PCLConfig.cmake`
flag injection, sm_120 hard-coded (Marlowe is sm_90), bars calibrated on the 5090 only, KinFu headroom 12%, the KinFu-LS
relaxed tier unstated (P20). PCL is sealed from training (`SEALED_REPOS`).

**Student (qwen-2.5-7b-instruct, CPU World, compile milestones only).** 26 valid episodes on 3 L1 windows before
OpenRouter refused for credits (402): compile pass 0/26 (full 0/14, locate 0/12); counterfactual `not_attributed` on 2
windows (the student fails even with the lines located), so L1 is above the 7B student's band. 54 later episodes were
402-masked; a harness bug read them as 0 (fixed: `res.masked`), and they are discarded. Spend $0.36 of $15.

**Not done here (needs the GPU or credits).** Every run milestone: golden equality, launches, fresh seeds, S9 shadow
drivers, S10 60-grade flake, S12 hackers, alternatives, sanitizer and perf calibration, and full-reward targeting. The
lab 5090 was unreachable all session (Tailscale: machine not approved by the tailnet admin); `bash run_gpu.sh` in the
`stage-gpu` bundle (21 windows, 167 MB) runs them on the lab or Marlowe.
