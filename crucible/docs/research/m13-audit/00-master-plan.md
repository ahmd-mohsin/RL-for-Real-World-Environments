# M13 master plan: from the six audits to a framework that supports everything

**Sources:** `01-article-beyond-52.md` (61 requirements A2-*), `02-agentenv-full.md` (C1–C15), `03-corpus-coverage.md`
(46 families, gaps G1–G12), `04-sandboxes-engines-gpus.md`, `05-scaling.md`, `06-swarm-forge.md`. Started 2026-10-10.

## Status after the audits

The audits found real defects in code that was already shipped, so Phase A comes before any new capability.

- **Environment coverage:** 4 of 46 environment families run through Crucible today.
- **Harbor:** foreign Harbor tasks receive a silent wrong reward (G1).
- **agentenv plugin:** two P0 bugs. `gpu_docker` can never create a sandbox (C1), and `crucible_world` is missing its env card (C2).
- **M12 trainer:** quadratic in turns. Every call is an item with its full prompt (05 §vertical).
- **Scheduler pool:** admission is O(N²) and nothing is freed (05).
- **Hack metric:** the confirmed-hack metric is never written (01).

Already closed in this session:
- the M12 GPU-gate root cause (the transformers-5 chat codec), the runtime circuit breaker and failure accounting;
- critic mode;
- the run-level state plane (A2-1, checklist #1).

## Phases (in order; each item ships with tests + DESIGN.md + module doc in the same commit)

### A. Correctness of shipped code (P0)

| ID | Item | Source | Acceptance |
|---|---|---|---|
| A-1 ✅ | In-place Harbor task: the agent acts in the task container, then the verifier runs there; oracle mode (`solution/solve.sh`); network read from `allow_internet` / `[environment].network_mode`; multi-key `reward.json` | G1–G4 | oracle → native reward, no-op → native floor, offline network → masked, multi-key → no crash (local Docker on a Harbor-format fixture) |
| A-2 ✅ | agentenv: `gpu_docker` sandbox type; full `crucible_world` record (card, MCP URL, bind host); `crucible_verify` booleans, episode window, actor strip, task kwargs | C1–C3 | the provider returns its own type; stock `Task.run()` chain completes; two episodes on one World give disjoint step lists |
| A-3 ✅ | Allowlist worlds run on Docker (egress policy passed through) | 04 | an allowlisted host is reachable, every other host blocked, on local Docker |
| A-4 ✅ | vLLM colocated wake order (trainer offload before KV wake) + `VLLMServerEngine` tests against a fake server | 04 | call-order test; abort / pause / sleep / load sequence tested without a GPU |
| A-5 ✅ | Pool hot path: indexed untrained count (O(1) admission), trained groups freed (values and tasks) | 05 | admission cost flat in N (10⁵ groups); memory stays bounded over a long run |
| A-6 ✅ | Hack metric written (`hack/confirmed` from probes and the detector); audit sample wired into the runtime | 01 A2-29/31 | an exposed-answer rollout raises `confirmed_hack_rate` |

### B. Vertical scale (long horizons, P0)

| ID | Item | Acceptance |
|---|---|---|
| B-1 ✅ | Segment-level trainer items: one causal sequence per (actor, segment), with completion tokens trained and observation tokens masked (reuses `bridges/common.flatten`) | loss and gradients equal the per-call items; forward tokens drop from O(turns²) to O(turns) |
| B-2 ✅ | Chunked LM-head logprobs (no full-vocabulary fp32 logits) | equal to unchunked within 1e-6; peak memory lower by more than 5× at 8k tokens |
| B-3 ✅ | Linear token capture (delta prompts in policy captures and checkpoints) | capture memory is O(T) on a 2k-turn episode |

### C. Reward and training pipeline (P0/P1)

| ID | Item | Source |
|---|---|---|
| C-1 ✅ | Metric/record sinks (JSONL + optional TensorBoard/W&B) | A2-56 |
| C-2 ✅ | Group grading stage (the group-level reward hook: tournaments, hack zeroing, recompute the group mean) | A2-15 |
| C-3 ✅ | GAR (mass-conserving redistribution, λ cap) and GRS-style multiplicative rubric reward | A2-16/17 |
| C-4 ✅ | Online hack interception at `World._call_tool` (refusal or dummy result, episode continues); a two-level detector (rules for recall, then an LLM judge for precision) | A2-28/29 |
| C-5 ✅ | On-policy distillation: a teacher pool and a per-token OPD/MOPD loss (route ④) | A2-9/10 |
| C-6 ✅ | Multi-domain runtime: per-source `rollout_fn` / `reward_fn` / harness, plus per-source staleness and metrics | A2-53..55 |
| C-7 | Context modes: keep-recent-k, discard-all, tiered; compaction as a policy action; thinking-history modes in the chat codec | A2-46..48 |
| C-8 | Shaping: verbosity budget, speed term, reward-to-go, incomplete-trajectory and tool-format penalties | A2-20..26 |

### D. Horizontal scale (P0/P1)

| ID | Item | Acceptance (localhost, multi-process) |
|---|---|---|
| D-1 | Model gateway: engine health checks, sticky routing that queues instead of spilling, version stamping | a killed engine is drained and its sessions moved, with no masked spike |
| D-2 | Remote rollout workers: leased, with heartbeats; a lost worker's rollouts are re-leased | killing a worker mid-run leaves no rollout lost or double-trained |
| D-3 | Remote environment client (NeMo-Gym / OpenEnv HTTP protocols) with session affinity | 4 env-server processes × 64 sessions; rewards equal to in-process |
| D-4 | Launcher (Ray-free default, Ray optional) | one command starts a controller, gateway, workers and env servers |

### E. Environment coverage (P0/P1; at least one family per kind)

The corpus audit's ten picks:
- terminal-bench 2.1 / science; MiMo Harbor code + cyber; SmolDataEnvs; HF_ML_Tasksmith (all via A-1);
- EnterpriseOps-Gym (new `interop/eog`);
- verifiers v1 import (tau2/tau3, BrowseComp-Plus) and v0 import;
- NeMo-Gym import (workplace_assistant, ruler/mrcr, math);
- OpenEnv two-player games as multi-agent;
- geoguesser (vision); browsergym (GUI);
- the AWM grader (G12).

### F. Engines (P0/P1)

| ID | Item |
|---|---|
| F-1 | Engine protocol v2: `capabilities()`, per-call abort, pause modes (abort / keep), sleep/wake, a logprobs-mode probe |
| F-2 | HF generate engine: a real HF model on CPU/MPS for end-to-end runs on the Mac |
| F-3 | SGLang server engine (update weights, abort, pause, logprobs, routed experts) |
| F-4 | vLLM routed experts (R3) and top-p candidates over HTTP |

### G. Swarm Forge (P0 core on CPU first, then the swarm)

From `06`:
- a concurrent orchestrator;
- composition operators (chain, fan-in, nest, fuse, perturb, stitch, tighten);
- new acceptance checks: A14 (each component is necessary), A15 (the composite is measurably harder than its parts), A16 (an adversarial hacker agent), A17;
- a solver ladder for difficulty control;
- cost/yield stop rules;
- shared-repo visibility zones with canary scans;
- the G-tighten headroom fix (15%).

### H. Sandboxes (P1/P2)

- `ssh_exec` hosts (the Runpod pod);
- gVisor or Kata on the lab;
- warm pools;
- pause/fork through providers that have them (Smol `fork_batch`, OpenSandbox pause);
- agentenv C5–C10 (fallback chains, attribution, the native Env, the gateway inside the sandbox, the full trigger language, agentenv env images as Apps).

## Working rules

- **Every item:** a test that decides it, DESIGN.md plus the module doc updated in the same commit, and the guarded ship
  script (lint, then the full suite, then the privacy and secret checks, then commit and push).
- **GPU items:** run on the lab RTX 5090 through the polite runner.
- **Credentials:** cloud items that need them (Modal, E2B, Daytona) are staged with fakes and listed for the user.
