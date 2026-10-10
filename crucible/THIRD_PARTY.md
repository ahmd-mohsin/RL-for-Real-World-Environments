# Third-party code, frameworks and data

Crucible builds on open-source RL and agent frameworks. This file records exactly what is used, how, at which revision, and under which license. Paths are relative to the repository root.

## Frameworks we depend on (installed from the local clones, editable)

| Framework | Local path | Commit (date) | License | Used for (module) | Install |
|---|---|---|---|---|---|
| Harbor 0.23.0 | `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor` | a2fafbc (2026-10-03) | Apache-2.0 | task model + content hash (M0); sandbox environments, GPU subclass, remote Docker (M1) | extra `harbor` |
| agentenv-framework 0.9.1301 + agentenv-protocol 0.1.316 | `agentenv-framework`, `agentenv-framework/packages/agentenv-protocol` | 041afd7 (2026-10-08) | Apache-2.0 | gateway (role ACL, clock, triggers, trajectory) as a world backend (M1); A2A agents, peers, humans (M2) | extra `agentenv` |
| verifiers v1 | `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-verifiers` | ae819de (2026-10-04) | MIT | training path (M11/M12, later) | extra `verifiers`, **conflicts** with `agentenv`/`mcp` (it pins `mcp==2.0.0`; agentenv needs `mcp<2`), so it lives in a separate environment |
| MCP Python SDK 1.30 | PyPI | — | MIT | App tool serving/clients (M1) | extra `mcp` |
| fastapi-mcp 0.4, SQLAlchemy 2.1, huggingface-hub 2.2 | PyPI | — | MIT | running OpenEnv AWM services (M1 generality) | extra `openenv-awm` |

## Code ported into `src/crucible/_vendor/` (each folder has `ORIGIN.md`; changed lines are tagged `# [crucible]`)

| Folder | Source | Revision | License | Changes |
|---|---|---|---|---|
| `mimo_sidecar/` | Xiaomi MiMo v2.6 RL release, `general/envs/*/{mcp_http.py, mcp_bridge.py}` | md5 69484c5f…, 98d327a9… | Apache-2.0 | `mcp_http.py`: bind address parameter (default 127.0.0.1 instead of 0.0.0.0). `sidecar_entrypoint.py` not ported (Linux-only readiness probe, `/dump` and `/verify` not implemented upstream) |
| `openenv_awm/` | OpenEnv `envs/agent_world_model_env/server/{data_loader.py, db_manager.py}` + `_patch_env_code` from `scenario_manager.py` | 49aa302 | BSD-3-Clause | per-file lazy download (upstream fetches all 7 files, ~500 MB, on first access) |

## Code ported outside `_vendor/` (adapted into Crucible modules; the source is named in each module docstring)

| Crucible module | Source | Revision | License | What was taken |
|---|---|---|---|---|
| `world/guard/guard.py` | Preference Model Karotte, `src/karotte/{subprocess.py, process_utils.py, confinement.py, demoted.py}` | 31c693e (2026-10-08) | MIT (`LICENSES/MIT-karotte.txt`) | privilege demotion in `preexec_fn` (setsid, `oom_score_adj=1000`, setgroups/setgid/setuid); killing a uid's whole process cohort with `kill(-1)` from a forked child plus a `/proc` sweep before any verification; egress self-test as the policy uid; no-follow, non-blocking file collection that refuses symlinks, FIFOs and devices. Re-written as a standalone stdlib script (no Karotte runtime) |
| `world/state/snippets/fs.py` (`_open_parent`, `_entry`) | Karotte `src/karotte/{untrusted_paths.py, demoted.py}` | 31c693e | MIT | O_NOFOLLOW component walk, lstat-based special-file detection, size caps for sparse files, fstat re-check after open |
| `world/leaks.py` | Hugging Face Repo2RLEnv, `pipelines/cve_patches.py` (`_strip_fix_leaks`) and `pipelines/pr_runtime.py` (`_LEAK_PATTERNS`) | 5555443 (2026-09-30) | Apache-2.0 (`LICENSES/Apache-2.0.txt`) | leak regexes verbatim (marked in the file); Crucible adds short-SHA context, code-host URLs, package pins, version pointers and debris clean-up |
| `world/guard/guard.py` (`git_scrub`) | Repo2RLEnv `pipelines/_env_guard.py` git-history scrub | 5555443 | Apache-2.0 | the scrub sequence (checkout base, drop remotes, delete refs, expire reflog, gc/prune). Crucible adds the verification Repo2RLEnv lacks: `rev-list --all` ⊆ ancestry of base, `fsck --unreachable` empty, forbidden SHAs absent |

| `actors/llm.py` (`CONTEXT_OVERFLOW_MARKERS`, `is_context_overflow`) | Prime Intellect verifiers v1 `harnesses/utils/compaction.py` | ae819de | MIT | the provider wordings of a context overflow, verbatim; backoff curve with full jitter from `utils/retries.py` |
| `actors/harness.py` (`BashLoopPolicy`, `format_observation`, prompts) | SWE-agent mini-swe-agent v2, `minisweagent/agents/default.py`, `config/mini.yaml` | 04d809c (2026-09-03) | MIT | the loop protocol: one bash block per turn, the submit sentinel, the format-error and observation templates with head/tail truncation; re-written over World shell tools |
| `packs/coding/.../connectors/ci_logs.py` (`parse_ctest`, `parse_gtest`, `parse_pytest`) | SWE-bench SWE-smith `swesmith/profiles/cpp.py` (`parse_log_ctest`, `parse_log_gtest`, `parse_log_pytest`) | 9b74ac0 (2026-03-22) | MIT (`LICENSES/MIT-swe-smith.txt`) | the line patterns; changed: no synthetic test names from summary lines (counts kept apart), first failure message kept, CTest "Not Run" is a skip |

## Frameworks we call (DEPEND), added in the hardening pass

| Framework piece | Where | What for |
|---|---|---|
| agentenv `providers/sandbox_providers` (`ModalSandboxProvider(gpu=...)`, `build_sandbox_provider`, `NetworkPolicy`) | `interop/agentenv/sandbox.py` | cloud GPU sandboxes (Modal H100/A100/L4, ...) and any registered agentenv provider for the agentenv backend |
| Harbor `EnvironmentFactory.create_environment` + `NetworkPolicy` | `interop/harbor/world_backend.py` (`environment_type`) | any Harbor provider (Modal, Daytona, GKE, E2B, Runloop, ...) with `gpus`/`gpu_types` |
| Harbor ATIF models, `convert_mini_swe_agent_to_atif`, `ClaudeCode._convert_events_to_trajectory` | `interop/harbor/atif_import.py`, coding pack `agent_runs`, `claude_code_sessions` | agent transcripts → ATIF → traces (M3) |

## Attacks we test against (no code copied)

`tests/redteam/test_attacks.py` reproduces the reward hacks reported by Karotte's comparison of frameworks (pre-written reward files, leftover processes, fake interpreters, symlinked and FIFO answers, fork bombs, internet access) and by Repo2RLEnv (PR links, `git log --all`, `pip download` of the fixed version) against a Harbor-backed World with `isolation.policy_user` set.

## Patterns re-implemented (no code copied)

| Pattern | Source | Where in Crucible |
|---|---|---|
| failure vocabulary (names reused verbatim for compatibility) | NeMo-Gym `nemo_gym/failure_kinds.py` | `core/schema/run.py` |
| content hash of a task directory (`rel\0sha256\n`) | Harbor `publisher/packager.py` | `core/ids.py` |
| `module:Class` registries, `env:` secret references, fail-loud duplicate registration | agentenv `config/runtime.py` | `core/registry.py`, `core/config.py` |
| virtual clock (t0 + elapsed·rate, ≤ 86400) and nl / tool / permission trigger actions | agentenv `env/gateway/{clock,triggers}.py` | `world/clock.py` (step mode added) |
| GPU compose overlay appended to the compose stack | Harbor `environments/kata.py` | `interop/harbor/gpu_docker.py` |
| user-simulator persona and done marker | verifiers v1 `envs/user_sim/env.py` | `actors/simulated.py` |
| scenario fields, per-message consistency judge, simulator failures mask the sample | NeMo-Gym `resources_servers/conversational_tool_use_simulation` | `actors/simulated.py`, `core/schema/run.py` |
| "never hallucinate; unknown info is unavailable; disclose progressively; out-of-scope token" | tau2 guidelines (via NeMo-Gym `indian_banking`) | `actors/simulated.py` |
| check ids and report shape for environment validation; named SKIPs | OpenEnv RFC 008 + `src/openenv/validation/` | `world/audit.py` |
| full / zero reward and determinism fixtures; masking only infra failures; STOP not allowed as the first user message | NeMo-Gym `verifier_fixture.py`, `conversational_tool_use_simulation/app.py` | `world/audit.py`, `actors/simulated.py` |
| gold / noop validation | verifiers `cli/validate.py` | `world/audit.py` |
| settled trigger snapshots (re-poll while a trigger is firing / queued) | agentenv `task_step/task_steps/prompt_agent.py` | `interop/agentenv/world_backend.py` |
| process-group cleanup after a call; subreaper-style drain | NeMo-Gym `sandbox/process_supervisor.py` | `world/guard/guard.py` (`run`) |
| tool results appended together after one assistant turn; non-object arguments answered with a corrective error | verifiers `harnesses/utils/core.py` | `actors/policy.py`, `world/world.py` |
| reasoning content replayed verbatim | research-environments `tau_common/program.py` | `actors/policy.py` |
| error envelopes `{"error": {"code": ...}}`; virtual time for generated tool code (the problem, measured) | MiMo v2.6 release tool modules | `world/packaging/mcp_tools.py`, `virtual_env.py` |
| refusing infeasible requests = leaving apps unchanged | EnterpriseOps-Gym refusal subset, MiMo `verifier_meta.act` | `VerifierView.unchanged` |
| recording proxy: per-session capability keys, one server for many sessions, request digest for retries | verifiers v1 `interception/server.py` | `connectors/capture_proxy.py` (retries recorded and collapsed to the last attempt instead of dropped) |
| milestone DAG matched to states monotonically along prerequisites | NeMo-Gym ToolSandbox `tool_sandbox/common/evaluation.py` | `workflow/milestones/match.py` |
| rubric items with tier, method and `pass_anchor`; strict binary evidence judge | MiMo v2.6 `verifier_meta.json`, `verify.py` | `credit/operators/rubric_item.py` |
| zero-net-mass turn-level advantage shaping; masked rows excluded from group statistics | XiaomiMiMo verl fork `signed_rebalance.py`, `core_algos.py` | `credit/compose.py` |
| PR-mirror negative mutants (reverse a diff); F2P per step | SWE-smith `bug_gen`, `harness/grading.py` | `credit/operators/defect_to_check.py`, `workflow/milestones/mine.py` |
| judge passes averaged with spread; fail-closed anchored parsing | GitSwarm (arXiv 2610.04862) judging protocol; prime-environments `torch_ao`, `nextjs` judges | `credit/judge.py` |
| segmentation: one model call per step, reasoning not a break | NeMo-Gym pivot-dataset references | `workflow/steps.py` |
| Anthropic Messages SSE event sequence (`message_start`, `content_block_*`, `input_json_delta`, `thinking_delta`, `message_delta`) | NeMo-Gym `nemo_gym/anthropic_converter.py` | `connectors/capture_proxy.py` (`assemble_anthropic`) |
| GenAI span conventions (`gen_ai.operation.name`, `gen_ai.tool.*`, `gen_ai.usage.*`) | OpenTelemetry GenAI semantic conventions | `connectors/otel_import.py` |
| flaky tests are dropped from FAIL_TO_PASS / PASS_TO_PASS | SWE-smith `harness/repair.py` | `ci_logs.flaky` (listed for M4 instead of dropped) |

## Data used for tests and gates (never in training or benchmark splits)

| Data | Source | License / terms | Use |
|---|---|---|---|
| 267 MiMo "general" environments (266 complete) | `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs` | Apache-2.0 (release README) | M1 generality gate |
| AgentWorldModel-1K (1,000 scenarios) | HF `Snowflake/AgentWorldModel-1K` → `crucible/data/cache/awm` (gitignored) | see dataset card | M1 generality gate |
| Public review comments | GitHub `PointCloudLibrary/pcl`, `NVIDIA/cccl` (via the REST API + raw.githubusercontent.com) → `crucible/data/cache/github` (gitignored) | public repositories | M2 reviewer-fidelity and grounding gates |
| PCL git history and policy documents (CONTRIBUTING, style guide, `.clang-format`, `.clang-tidy`) | `git clone --filter=blob:none` of `PointCloudLibrary/pcl` → `crucible/data/cache/git` (gitignored) | BSD-3-Clause | M3 gate (55 PR traces from git, policy rules) |
| Our Claude Code construction transcript (tasks 01–04) | `~/.claude/projects/...` frozen to `crucible/data/cache/sessions` (gitignored, unredacted) | private | M3 gate; only redacted traces are stored |
| SWE-rebench OpenHands trajectories (67,074) and SWE-rebench instances | HF `nebius/SWE-rebench-openhands-trajectories`, `nebius/SWE-rebench` → `crucible/data/cache/hf/swe-rebench` (gitignored); base files from raw.githubusercontent.com | CC-BY-4.0 | M5 replay test, M6 S2 gate (held-out evaluation only; never training or benchmark splits) |
