# M11 sweep, group 2: verifiers, Prime Intellect environments, prime-rl, GitSwarm and autoresearch vs Crucible interop (2026-10-12)

Read-only, 2026-10-12, Crucible HEAD `af402b4`. Two helper agents read prime-rl (how it consumes verifiers traces), and
prime-environments with research-environments, prime-agent, GitSwarm and autoresearch (packaging and loading). I read
verifiers and all of `crucible/src/crucible/interop/verifiers/`, `interop/canonical.py`, `interop/replay.py`, the reward
writer, the gate tests, spot-checked the helpers' key citations, and ran the probes.

REF = `RL-Environments/06-frameworks-and-env-code/`. ART = `RL-Environments/07-articles/`. C = `crucible/src/crucible/`.
VF = `REF/primeintellect-ai-verifiers/verifiers/v1/`. PRL = `REF/infra-primeintellect-ai-prime-rl/src/prime_rl/`.

| Repo | Pin | Licence |
|---|---|---|
| `primeintellect-ai-verifiers` (VF) | `ae819de` (2026-10-04), clean | MIT |
| `primeintellect-ai-prime-environments` (PE, 109 envs) | `cf220ee` (2026-05-07) | Apache-2.0 |
| `primeintellect-ai-research-environments` (RE, 106 tasksets) | `d4fb80a` (2026-10-03) | Apache-2.0 |
| `infra-primeintellect-ai-prime-rl` (PRL) | `f7e8ab6` (2026-10-09) | Apache-2.0 |
| `primeintellect-ai-prime-agent` (PA, Rust) | `2ee7e623a` (2026-10-10) | MIT |
| `veds12-git-swarm` (GS) | `9f66af0` | **no licence**; a static project page |
| `karpathy-autoresearch` (AR) | `228791f` | **no licence** |

**The pinned verifiers.** `crucible/.envs/verifiers` holds an **editable install of the local checkout**
(`direct_url.json`: `file://…/primeintellect-ai-verifiers`, `editable: true`). So the "pin" is whatever that checkout's
HEAD is, today `ae819de`. Nothing records the commit. hatch-vcs finds no tag, so the installed version is `0.0.1.dev1`
(`crucible/pyproject.toml:42, 93` declare `verifiers` with no version) [ran P4]. Every line cited below is at `ae819de`.

**Import limits.** Core may not import `verifiers` (`crucible/pyproject.toml:130, 143`). verifiers lives in its own env
because it pins mcp 2 (`pyproject.toml:76-78`). prime-rl is not a dependency. So DEPEND always means "in
`crucible.interop.verifiers`, run in `.envs/verifiers`". Only MIT and Apache-2.0 code is marked PORT.

Probes are in `scratchpad/m11sweep/g2/` (index in §5). P1-P4 run with `crucible/.envs/verifiers/bin/python` from
`crucible/`, so they use verifiers' real classes, serializers and loader.

---

## 0. Corrections and confirmations of the spec and earlier studies

| # | Claim | Status | Evidence |
|---|---|---|---|
| 0.1 | ARCHITECTURE M11 verifiers: "dense credit written into `MessageNode.advantages`" (`ARCHITECTURE.md:660`) | **Cannot work from the environment side.** The env-server wire is append-only for nodes: a node is sent once, when appended, and later edits are not resent (VF `serve/delta.py:46-54`, `:212-228`). Every RL algorithm then overwrites `node.advantages` (PRL `orchestrator/algo/routing.py:17-40`). `ALGORITHM_CLASSES` is a closed dict (PRL `algo/__init__.py:43-53`), and the config says a new scheme is "a new named algorithm in code" (`packages/prime-rl-configs/.../algorithm.py:420-421`). What does reach the orchestrator is `trace.info` (`delta.py:56-62` lists it in `SCALAR_FIELDS`). opsd already reads a key from it (PRL `algo/opsd.py:55`). So the path is `trace.info["crucible_steps"]` plus an M12 algorithm inserted into `ALGORITHM_CLASSES`, as research 05 §3.3 says. Fix the ARCHITECTURE line | [read] |
| 0.2 | DESIGN "M11 part 1": "verifiers keeps the reward and raises a typed error on a mask" (`DESIGN.md:868`); "Gate (toy) ✅" | **Overstated.** The test builds `CrucibleTask` with `object.__new__`, a stub runtime and a stub trace (`tests/unit/test_interop_gate.py:134-170`). It never loads a taskset, runs a verifiers env, compares `reward.json`, or touches `steps.jsonl`. verifiers is not in the identity matrix at all. See D9 | [read] |
| 0.3 | Research 05 §3.3: prime-rl drops errored traces (`PR/orchestrator/algo/base.py:11-19`) | **Confirmed, and wider.** `iter_trainable_traces` skips on `not episode.ok or trace.has_error …` (PRL `algo/base.py:15`), so one failed trace drops every trace of its episode. `has_error` is `not ok` (VF `trace.py:476-478`). prime-rl never branches on the error type; grep finds no `TaskError` in PRL. Groups keep survivors and the GRPO baseline is their plain mean, no std (PRL `algo/grpo.py:29-52`; `train_sink.py:243-245`) | helper [read], spot-checked |
| 0.4 | Research 05 §3.3: "Missing reward → `TaskError`", lines `:370-403` | Confirmed (VF `tasksets/harbor/taskset.py:370-403`). The file grew to 874 lines; research line numbers `:327-355`, `:357-368`, `:454-475` still match. `stage_tests` removes only `reward.json` and `reward.txt` before grading (`:345-348`) | [read] |
| 0.5 | Research 05 §0.1 item 3 / B.17: verifiers sums every key of a reward.json that lacks `reward` | Confirmed (`taskset.py:380-386`; `task.py:264-273`). Crucible never writes such a file: a masked grade writes no `reward.json` (`C/verify/runtime/plan.py:376-381`) | [read] |
| 0.6 | Research 05 §3.3: verifiers' NeMo-Gym bridge ignores `mask_sample` | **Confirmed against Crucible's own export.** See D7 | [ran P3] |
| 0.7 | Research 05 §2 N2: "verifiers rounds per-token float streams to 4 decimals in `to_record`" | **It also rounds `traces.jsonl`.** `write_episode` uses `dump_json` (VF `utils/trace_store.py:22-28`); the node serializer defaults to 4 decimals when no context is given (`graph.py:75, 177-191`). Only the msgpack wire is full precision ("training never reads the record", `graph.py:76-80`). Crucible's importer trains from `traces.jsonl`: D3 | [ran P1a] |
| 0.8 | The verifiers article: v1 is a preview and "the legacy path is frozen" (ART `pi-verifiers-v1.md:3, 18`) | **Overtaken.** The checkout has removed v0: "The legacy v0 stack (`import verifiers as vf`) has been removed" (`REF/primeintellect-ai-verifiers/docs/overview.md:7`; `verifiers/__init__.py:1-9`). All 109 PE envs are v0 (111 dirs use `vf.SingleTurnEnv` / `ToolEnv` / `Rubric`, 0 use `verifiers.v1`). So `vf-eval` at this pin cannot run any PE env: D12 | helper [read], spot-checked |
| 0.9 | prime-agent's skill doc: v1 envs expose `load_environment(config)` / `load_taskset(config)` (PA `skills/prime-intellect/references/environments.md:35`) | **Stale.** The v1 loader imports the module named by the id (dashes to underscores, lower case) and takes the one subclass exported in `__all__` (VF `utils/loaders.py:68-126`). VF's own skill forbids loader functions (`skills/create-environments/SKILL.md:54-56`). This is the contract D2 breaks | [read] |
| 0.10 | M8-M10 sweep G2 0.5: a scoring timeout leaves `ok=True` and every reward `None` | Confirmed (VF `task.py:253-255`; `trace.py:407-409, 472-474`). `None` rewards survive into `traces.jsonl` even with `exclude_none=True` (dict values are kept) [ran]. Crucible's importer crashes on them: D4 | [ran P1b] |
| 0.11 | verifiers `IsolatedVerifierEnv` retries "the verifier box alone, on any exception" (M8-M10 sweep G2 0.4) | Confirmed: 2 extra fresh boxes by default, with backoff (VF `envs/isolated_verifier/env.py:36-38, 134-182`). `HarborEnv` builds the grader as `type(task)(verifier_box_data(data), config)` and calls **`run_verifier`**, not `solved` (`tasksets/harbor/env.py:122-138`). So `CrucibleTask.run_verifier` is the right hook in both modes | [read] |
| 0.12 | Research 05 §6.3: "verifiers GPU on the lab host: not verified" | Still open. Nothing in this sweep ran a verifiers container | — |

---

## 1. The M11 contracts, field by field

### 1.1 Export: `CrucibleTask` / `CrucibleTaskset` (`C/interop/verifiers/taskset.py`) vs verifiers v1

| Contract (VF source) | What verifiers requires | Crucible | Status |
|---|---|---|---|
| Loading by id (`utils/loaders.py:68-126, 168-191`) | A module named by the id, `__all__` exporting exactly one `Taskset` subclass; the env from an exported `Env` subclass, else `SingleAgentEnv` | `build()` returns two classes made at call time. No module exports them | **broken** (D2) [ran P2e] |
| Separate verifier (`tasksets/harbor/env.py:42-54, 115-138`; `taskset.py:356-365`) | Export `HarborEnv` so a `[verifier].environment_mode = "separate"` task grades in a fresh box | Not exported, so `SingleAgentEnv` runs and `solved` raises "declares a separate verifier" | **broken** (D2) |
| Task directory source (`taskset.py:489-565`) | `dataset_dir` downloads a Harbor Hub package or registry with the `harbor` CLI | `CrucibleTaskset.load` calls `dataset_dir` (`:55`); there is no local path option | **gap** (D2) |
| `parse_task` (`taskset.py:619-710`) | `harbor` importable; a pullable `docker_image` for `[environment]` and `[verifier.environment]` | `.envs/verifiers` has no `harbor`; task02 declares only Dockerfiles | **broken** (D2) [ran P2f] |
| Timeouts (`taskset.py:79-86, 667-680`) | `ignore_timeouts=True` by default drops the task's agent and verifier timeouts | Not overridden | **gap** (D11) [ran P2g] |
| Stale files (`taskset.py:327-354`) | `rm -f reward.json reward.txt` before `test.sh` | `mask.json` and `steps.jsonl` are read but never removed | **broken** (D1) [ran P2a, P2b] |
| Reward (`taskset.py:52-55, 370-417`) | strict `float | dict[str, float]`; `{"reward": r}` returns r, other keys become metrics; `reward.txt` fallback | Same for `reward.json` (`:42-45, 51`); metrics under `crucible/`; the `reward.txt` fallback is gone | honoured; fallback dropped (D10) [ran P2d] |
| Mask | none native; an exception in scoring becomes `TaskError` (`task.py:231`), the trace is not ok, prime-rl drops the episode | `TaskError("crucible:masked:<kind>: …")` (`:40-41`) | honoured; but retried twice by `IsolatedVerifierEnv` (D13) |
| `trace.info` (`trace.py:412`; `delta.py:56-62`) | free dict, on the wire | `trace.info["crucible_steps"]` (`:46-50`) | honoured; but the file is never written by Crucible and is capped at 1 MiB (D8) |
| Bounded reads (`runtimes/base.py:345-358`) | `max_bytes` raises `SandboxError` past the cap | used | honoured |

### 1.2 Import: `trace_to_record` / `vf_rollout` (`C/interop/verifiers/v1_import.py`) vs the v1 trace

| Contract (VF source) | verifiers | Crucible | Status |
|---|---|---|---|
| Success (`trace.py:476-478`; `agent.py:695-698`: "success is the `ok` stamp, never errors-emptiness") | `ok` | `masked = bool(errors) or reward is None` (`:77`) | **broken** (D5) [ran P1c, P1d] |
| Rewards (`trace.py:135-141, 407-409, 472-474`) | `dict[str, Reward \| None]`; `None` = scoring did not run | `v.get(...)` on every value (`:76`) | **crash** (D4) [ran P1b] |
| Logprobs (`graph.py:75-80, 135-137, 177-191`) | compact, full precision on the wire, 4 decimals in `traces.jsonl` | reads `traces.jsonl` (`:154-160`) | **lossy** (D3) [ran P1a] |
| Branches (`trace.py` `branches`, `:515-530`; `graph.py:88-100`) | N leaves = N samples; compaction attempts are dangling leaves; subagents are branches | follows only the parent chain of `nodes[-1]` (`:34-46`) | **lossy** (D6) [ran P1f] |
| Sampled nodes (`graph.py:103-125`) | `sampled` with a `mask` span; relay turns may carry no tokens | a sampled node with no trainable tokens is folded into the prompt (`:57-59`) | **lossy** (D6) [ran P1e] |
| Episode (`episode.py`) | an episode holds one trace per agent | `rows[0]["traces"][0]` (`:178`) | **lossy** (D6) |
| Truncation (`trace.py:620-632`) | `is_truncated`, `stop_condition`, `is_timeout` | only `stop` kept (`:80`) | gap (N4 not applied) |
| Error type (`trace.py:742-758`) | `Error{type, message, status_code}`, oldest first | `failure_kind = errors[0].type` (`:80`) | gap (D15) |

### 1.3 prime-rl consumption (what a Crucible export must survive)

| Behaviour | Source | Effect on Crucible |
|---|---|---|
| Reward = Σ score·weight over non-None rewards | VF `trace.py:472-474` | A Crucible task contributes exactly `reward.json["reward"]` (one reward, weight 1). OK |
| Error → whole episode excluded; survivors set the group baseline | PRL `algo/base.py:15`; `train_sink.py:243-255`; `algo/grpo.py:29-52` | A masked grade is excluded, never a 0. OK |
| Train-time timeout becomes an error | PRL `orchestrator/dispatcher.py:669-674` | Same as Crucible's `LimitPolicy` "mask" choice, not its default "score". Document per export |
| Zero-advantage tokens pruned; no reward clamp; MaxRL zeroes groups with mean ≤ 0 | PRL `train_sink.py:29-54`; `algo/max_rl.py:24-27` | Crucible's record-style score gates (reward > 1) pass through unclamped. OK |
| Truncated traces kept with full advantage | PRL `metrics.py:200, 231-233` | Crucible's N4 (truncation is a policy outcome, scored) holds |
| Env-set `node.advantages` never reach the orchestrator | VF `serve/delta.py:46-54`; PRL `algo/routing.py:17-40` | 0.1 |

### 1.4 Packaging that an import needs

- **PE** (v0 only): `environments/<slug>/pyproject.toml` with `verifiers>=0.1.x` lower bounds and `load_environment(**kwargs)`
  (e.g. `environments/synlogic/pyproject.toml:2, 9, 25-26`; `synlogic/synlogic.py:137`). Only 5 declare a
  `verifiers.environments` entry point. Importing a PE env needs a separate env with verifiers 0.1.x and a v0 runner.
  Crucible has neither.
- **RE** (v1): one package per taskset, `__all__` exporting the Taskset (and optionally an `Env` and a `Harness`), pins
  `verifiers>=0.3.1` (68 of 106) or `verifiers[harbor]>=0.3.1` (16) (e.g. `environments/science/gpqa/pyproject.toml:7`;
  `gpqa/__init__.py:1-3`; `gpqa/taskset.py:39-91`). `vf_rollout("<id>")` can run any of them once installed into
  `.envs/verifiers`, but the installed `0.0.1.dev1` satisfies none of those pins: D12.
- **Isolated verifier envs**: RE `apex_agents` is the closest analogue of Crucible's export and the model to copy
  (§2.1, pick 1).

---

## 2. Mechanism tables

Verdicts: DEPEND (in interop only), PORT (licence allows), PATTERN, nothing (Crucible has it or better), anti-pattern
(use as a negative fixture).

### 2.1 verifiers (MIT) and research-environments (Apache-2.0)

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| A Harbor taskset package: `__all__ = ["HarborEnv", "<Harness>", "<Taskset>"]`; a `path` field for a local export beside `dataset`; `ignore_timeouts = False`; the verifier `docker_image` injected into a cached copy of `task.toml`, the source kept immutable | RE `tool_use/apex_agents/src/apex_agents/__init__.py`; `taskset.py:22-27, 48-80` | `build()` in `interop/verifiers/taskset.py` (no package, no path, timeouts dropped, Dockerfile tasks rejected) | **PORT (S)**: fixes D2, D11 |
| `run_verifier` override: `test.sh` exit code is a grader failure; strict `reward.json` with a `reward` key; other keys as metrics | RE `apex_agents/taskset.py:35-45` | `CrucibleTask.run_verifier` (`taskset.py:34-51`) | nothing (taken; Crucible also carries the mask kind) |
| `HarborEnv`: separate verifier box, artifacts restored, `run_verifier` on a fresh box, retries with backoff | VF `tasksets/harbor/env.py:42-138`; `envs/isolated_verifier/env.py:105-182` | `verify/stage.py` (Crucible's own separate grader) | DEPEND (export it from the Crucible package) |
| Remove reward files before `test.sh`, and raise if removal fails | VF `taskset.py:327-354` | `plan.prewrite_zero` runs only if Crucible's `run.py` starts (`C/verify/runtime/run.py:87`) | PATTERN (S): extend it to `mask.json`, `steps.jsonl`, `crucible_reward.json` in `CrucibleTask` (D1) |
| Typed `WireEpisode` / `read_episodes` loader for `traces.jsonl` (task fields loose, harness not imported) | VF `utils/trace_store.py:32-45`; `trace.py:772-776` | dict poking in `trace_to_record` | DEPEND (S): parse with it, use `Trace.ok`, `Trace.reward`, `Trace.branches` (D4, D5, D6) |
| Full-precision trace over msgpack (ZMQ ROUTER/DEALER, delta stream) | VF `serve/server.py:28-87`; `serve/client.py:47-51, 213-227`; `serve/delta.py` | `vf-eval` subprocess per rollout, reading `traces.jsonl` | DEPEND (M): drive a served env, or keep the generate server's own logprobs (D3, D14) |
| `boundary(TaskError, …)` around scoring; any escape is a `TaskError` | VF `task.py:231`; `errors.py:80-95` | `infra.boundary` | nothing (Crucible already has it) |
| NeMo-Gym bridge: `float(result["reward"])`, every numeric field a metric, `mask_sample` ignored | VF `tasksets/nemo_gym/taskset.py:72-87` | Crucible's NeMo export sends `reward: 0.0` with `mask_sample: true` (`C/interop/nemo_gym/server.py:143-148`) | anti-pattern → fixture (D7) |
| OpenEnv bridge: sums `result.reward or 0.0` over steps | VF `tasksets/openenv/taskset.py:86, 129, 134` | Crucible's OpenEnv export sends Ψ deltas and `None` on a masked final step (`C/interop/openenv/export.py:110-122`) | anti-pattern → fixture (D7) |
| Retry rule by exact error type, status class, message; per-rule budgets; default 0 | VF `configs/retries.py:15-48`; `utils/retries.py:77-117` | `runtime/infra.py` | nothing new for M11 (ported in M10) |

### 2.2 prime-rl (Apache-2.0)

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| An algorithm reads task-specific data from `trace.info` | PRL `orchestrator/algo/opsd.py:55` | none (`trace.info["crucible_steps"]` has no reader) | PATTERN (M): the M12 dense-credit algorithm (D8) |
| `assign_advantages(trace, float \| list[float])` in compact sampled-token order | PRL `algo/routing.py:17-40` | `credit/` per-call credit | DEPEND (M12): call it with per-token lists built from `crucible_steps` by `call_id` |
| Episode exclusion on any error; baseline over survivors | PRL `algo/base.py:15`; `train_sink.py:243-255` | `infra` masks; independent samples | nothing (consistent with Crucible's mask) |
| Train timeouts become errors | PRL `dispatcher.py:669-674` | `LimitPolicy` default `score` | document in the export card (0.3) |

### 2.3 prime-environments, prime-agent, GitSwarm, autoresearch

| Repo | Mechanism | Source | Verdict |
|---|---|---|---|
| PE | v0 packaging, `load_environment(**kwargs)`, lower-bound pins | `AGENTS.md:73-78`; `environments/synlogic/pyproject.toml:9, 25-26` | DATA only, through a separate verifiers 0.1.x env if ever needed. Nothing to import at this pin (0.8, D12) |
| PE | 24 envs return 0 from an `except`; marker spoofs (`gpu_puzzles`) | M8-M10 sweep G2 §1.2 | anti-pattern (unchanged; I-check fixtures) |
| PA | verifiers runs prime-agent 0.9.5 over ACP as a harness; no task or reward interop in the Rust code | VF `harnesses/prime_agent/harness.py:1, 21-26` | nothing for M11 |
| PA | stale loader doc | `skills/prime-intellect/references/environments.md:35` | nothing (0.9) |
| GS | static project page | `README.md:1-40` | nothing |
| AR | `results.tsv` keep/discard/crash; no task or reward interop | `program.md:33, 66-84` | nothing |

---

## 3. Crucible defects and spec gaps

| # | Sev | Defect | Where | Evidence | Fix |
|---|---|---|---|---|---|
| D1 | **P0** | **The policy can forge the mask and the dense channel in shared-box grading.** verifiers grades in the agent's own box unless the task asks for a separate one (`HarborData.verifier` None, VF `taskset.py:181-183, 366-367`), and removes only `reward.json` / `reward.txt` (`:345-348`). `CrucibleTask` reads `mask.json` **first** (`taskset.py:36-41`) and attaches any `steps.jsonl` (`:46-50`). Crucible's in-container grader removes `mask.json` only if `run.py` starts (`C/verify/runtime/run.py:87`; `plan.py:318, 344-351`), never removes `steps.jsonl`, and never writes it (grep: no `steps.jsonl` in `C/verify/runtime/`). So an agent that plants `mask.json` and breaks the grader (or any task whose `test.sh` is not Crucible's, e.g. task02's) turns its failure into a `TaskError` mask that prime-rl drops. A planted `steps.jsonl` rides the trace as dense credit next to a real reward 0. Crucible tasks default to `verifier_mode = "shared"` (`C/core/schema/task.py:156`) | `C/interop/verifiers/taskset.py:34-51` | [ran P2a]: planted mask, grader never ran → `TaskError crucible:masked:node_contention: planted`. [ran P2b]: real reward 0 + planted file → `reward 0.0`, `crucible_steps=[{'phi': 1.0, 'delta_phi': 1.0}]` | Refuse shared mode in the export: `CrucibleTaskset.load` raises unless the task is `environment_mode = "separate"` (graded through `HarborEnv`). Override `stage_tests` to `rm -f` `mask.json`, `steps.jsonl`, `crucible_reward.json`, `report.json` and fail if removal fails. Read the mask only from `crucible_reward.json` written in this grade (carry a per-grade nonce passed to `test.sh`). Test: both P2a and P2b must end as a real 0 with no steps |
| D2 | **P0** | **The verifiers export cannot be loaded by vf-eval or prime-rl.** (a) No module exports the classes: `taskset_class("crucible")` raises "defines no `__all__`", and a dotted id fails with `ModuleNotFoundError` (VF `utils/loaders.py:68-126`). (b) Without an exported `HarborEnv`, `SingleAgentEnv` runs, so every separate-verifier task raises in `solved` (`taskset.py:358-365`). (c) `load` only takes Harbor Hub or registry downloads (`dataset_dir`, `:533-565`); there is no local path. (d) `.envs/verifiers` has no `harbor`, so `parse_task` cannot import (`:621-623`). (e) task02 declares only Dockerfiles for both boxes, which `resolve_image` rejects (`:568-616`; `data/runtime/tasks/task02_pcops_triton@5e53057/task.toml`) | `C/interop/verifiers/taskset.py:23-63`; `crucible/.envs/verifiers` | [ran P2e]: both loader ids raise. [ran P2f]: `parse_task` on the toy fixture and on task02 → `ModuleNotFoundError: No module named 'harbor'` | PORT RE `apex_agents` (pick 1): a real package `crucible_vf` (`__init__.py` with `__all__ = ["HarborEnv", "CrucibleTaskset"]`), `path: Path \| None` for a local export, `ignore_timeouts = False`, inject the pinned image digests from `verify/stage.py` into a cached copy of each `task.toml`. Install `verifiers[harbor]` in `.envs/verifiers`. Gate: `vf-eval crucible-vf --env.taskset.path <dir> --dry-run` resolves the env |
| D3 | **P1** | **The importer trains on logprobs rounded to 4 decimals.** `run_vf_eval` reads `traces.jsonl` (`v1_import.py:154-160`), which verifiers writes with `dump_json` and the 4-decimal default (VF `utils/trace_store.py:22-28`; `graph.py:75-80, 177-191`). The rounding moves the behaviour logprob by up to 5e-5 per token, which biases importance ratios over long completions. `RolloutGenerateServer` already holds the exact `res.logprobs` (`:110-117`) and discards them | `C/interop/verifiers/v1_import.py:62-68, 110-117, 154-160` | [ran P1a]: `[-0.123456789, -2.000049]` → `[-0.1235, -2.0]` in the record, `valid_for_training = True` | Keep each completion's ids and logprobs per route key in the server, and in `trace_to_record` replace a node's logprobs with the server's when the completion ids match (refuse otherwise). Or read the msgpack wire. The test should compare logprobs, not only ids (`tests/unit/test_verifiers_v1_import.py:73-77`) |
| D4 | **P1** | **The importer crashes on a trace whose scoring did not run.** verifiers seeds rewards as `None` (VF `task.py:253-255`), and `None` survives into `traces.jsonl`. `float(v.get(...))` on `None` raises (`v1_import.py:76`). This hits every errored trace and every scoring timeout (`ok=True`, rewards `None`) | `C/interop/verifiers/v1_import.py:74-80` | [ran P1b]: both cases → `AttributeError: 'NoneType' object has no attribute 'get'` | Skip `None` values; mask when any reward is `None`, with `failure_kind = verifier_timeout` when `is_timeout` and `stop_condition == "scoring_timeout"` |
| D5 | **P1** | **The importer decides success from the error list, not `ok`.** verifiers prepends earlier attempts' errors to the final trace, and says "success is the `ok` stamp, never errors-emptiness" (VF `agent.py:699-700`). So a rollout that succeeded on retry is masked, and a trace with `ok=False` and no error rows is trained | `C/interop/verifiers/v1_import.py:74-77` | [ran P1c]: retried, final attempt ok → masked `ProviderError`. [ran P1d]: `ok=False`, no errors → trained, reward 1.0 | `masked = not trace["ok"] or any(r is None ...)`; take `failure_kind` from the **last** error |
| D6 | **P1** | **The importer drops parts of the trace silently.** (a) It follows only the parent chain of `nodes[-1]` (`:34-46`). A subagent branch, a second sampled branch, or a dangling compaction leaf after the main leaf replaces the main branch. (b) A `sampled` node with no trainable tokens is folded into the prompt (`:57-59`), so the record loses a model call and still passes `valid_for_training`. (c) `rows[0]["traces"][0]` takes the first trace of a multi-agent episode (`:178`) | `C/interop/verifiers/v1_import.py:34-68, 178` | [ran P1f]: two branches → 1 turn with the second branch's logprob `-0.5`, valid. [ran P1e]: a token-less sampled node → still 1 turn, valid | Enumerate `Trace.branches` (VF `trace.py:515-530`) and emit one record per trainable branch, or refuse more than one. A sampled node without a full token record masks the rollout (DESIGN "never guessed"). Refuse episodes with more than one trace unless an actor map is given |
| D7 | **P1** | **A masked Crucible grade reaches prime-rl as a score through verifiers' own bridges.** Crucible's NeMo-Gym export answers a masked grade with `reward: 0.0, mask_sample: true` (`C/interop/nemo_gym/server.py:143-148`). verifiers' NeMo bridge returns `float(reward)` and records `mask_sample` as a metric (VF `tasksets/nemo_gym/taskset.py:81-87`). The OpenEnv export sends Ψ deltas and `None` on a masked final step (`C/interop/openenv/export.py:110-122`); verifiers' OpenEnv bridge adds `result.reward or 0.0` (`tasksets/openenv/taskset.py:129, 134`), so the trace scores Φ_{T-1}, the progress before the mask. Research 05 §6.2 item 10 requires every runtime to show a mask natively, never as a score | `C/interop/nemo_gym/server.py:143-151`; `C/interop/openenv/export.py:116-122` | [ran P3]: Crucible's masked body through the verifiers NeMo bridge → `reward = 0.0`, no error. OpenEnv: [read] | Add both bridges to the gate's negative controls. In the NeMo server, answer `/verify` with HTTP 503 and the failure kind when the caller is not NeMo-RL (a header or a `?strict=1` route), so verifiers' `raise_for_status` errors the trace. In the OpenEnv export, raise from `step` on a masked final grade when served to verifiers. State in each export card that verifiers must consume Crucible through `CrucibleTaskset`, not these bridges |
| D8 | **P1** | **The verifiers dense channel is empty or forged, never Crucible's.** The in-container grader does not write `steps.jsonl` (D1). The read is capped at 1 MiB and failures are swallowed (`taskset.py:46-50`), so an honest long channel vanishes. No reader of `trace.info["crucible_steps"]` exists in `src/` (grep). `replay.py` has no verifiers-trace parser (its list: `replay.py:10-13`), although research 05 §6.2 item 4 calls for one. Env-side `node.advantages` cannot work (0.1) | `C/interop/verifiers/taskset.py:46-50`; `C/interop/replay.py:10-13` | [ran P2c]: 7,600 rows (the xl helpdesk shift) = 1,617,690 bytes → `steps=0`, no error | Produce the rows host-side: add `from_vf_trace` (sampled nodes in branch order → actions, `call_id` from the sampled ordinal) and replay it, as the NeMo and agentenv paths do. Or have the grader write `steps.jsonl` from the collected transcript. Raise the cap or stream; log a dense-channel loss as a metric, never silently. M12: an algorithm that reads `trace.info["crucible_steps"]` and calls `assign_advantages(trace, per_token)` |
| D9 | **P1** | **The M11 exit gate is not met for verifiers, and the doc says ✅.** The gate test stubs the runtime and the trace (`tests/unit/test_interop_gate.py:134-170`). The export never ran through `vf-eval`, `HarborEnv` or a served env. There is no `steps.jsonl` identity for verifiers and no task02 row at all. A toy World task cannot go through a `HarborTaskset` (it is not a container task), so "the same toy task through verifiers" needs a World-backed `Taskset` that does not exist | `tests/unit/test_interop_gate.py:134-170`; `DESIGN.md:868` | [read] | After D2: run the toy oracle through a World-backed verifiers `Taskset` (the NeMo server's pattern) and a Harbor fixture through `HarborEnv` on Docker, then compare canonical `reward.json` and `steps.jsonl`. Mark DESIGN "verifiers: reward on a stub only" until then |
| D10 | P2 | **Harbor tasks that write only `reward.txt` error under `CrucibleTask`.** Stock `HarborTask` falls back to `reward.txt` (VF `taskset.py:387-403`). `CrucibleTask` requires a `reward.json` with a `reward` key (`:42-44`). task02's `test.sh` writes only `reward.txt` | `C/interop/verifiers/taskset.py:42-44` | [ran P2d]: `TaskError crucible: no reward.json with a 'reward' key` | Fall back to `super().run_verifier` when the task's `test.sh` is not Crucible's runtime, or migrate task02's `test.sh` to `run.py` |
| D11 | P2 | **The export drops the task's timeouts.** `HarborConfig.ignore_timeouts` defaults to `True` (VF `taskset.py:79-83, 667-680`), so the 12 h agent limit and the verifier limit of an exported task vanish, and a hung grader has no scoring deadline | `C/interop/verifiers/taskset.py:53-61` | [ran P2g]: `ignore_timeouts = True` | Default `ignore_timeouts = False` in the Crucible config class (as RE `apex_agents/taskset.py:27`) |
| D12 | P2 | **The verifiers env is unpinned and no published environment can install into it.** It is an editable install of a mutable checkout (no commit recorded). The version is `0.0.1.dev1`, which satisfies none of RE's `verifiers>=0.3.1` pins (68 of 106) nor PE's `>=0.1.x`. PE envs are all v0, which this verifiers no longer has (0.8). So `vf_rollout` can run only envs vendored with verifiers (glossary) or installed `--no-deps` | `crucible/pyproject.toml:42, 93`; `.envs/verifiers` | [ran P4]: `installed 0.0.1.dev1`, editable, HEAD `ae819de`; `'>=0.3.1' admits: False` | Pin a released verifiers (or a git rev with `SETUPTOOLS_SCM_PRETEND_VERSION`) in a lock for `.envs/verifiers`; record the commit in the import manifest; state in DESIGN that PE (v0) is out of scope at this pin |
| D13 | P2 | **A masked grade is graded three times.** `IsolatedVerifierEnv.grade` retries any exception in a fresh box, 2 extra times by default (VF `envs/isolated_verifier/env.py:36-38, 134-182`). A deterministic mask (`hidden_tests_unreadable`, a task02 GPU grade of up to 1 h) costs three grades | `C/interop/verifiers/taskset.py:40-41` | [read] | Set `verifier.retries` from Crucible's per-kind retryable verdict (M8-M10 sweep G2 pick M10.3), or raise non-retryable kinds as a type the env does not retry |
| D14 | P2 | **`run_vf_eval` leaks a process on timeout and runs one `vf-eval` per rollout.** `asyncio.wait_for(p.communicate())` cancels the wait but never kills the child (`:153`). The child gets the whole `os.environ` (`:151`). Each rollout pays a full `vf-eval` start and env load | `C/interop/verifiers/v1_import.py:146-160, 165-187` | [read] | Kill and reap on timeout; pass an allowlisted environment; for training, keep one served env (VF `serve/`) and send `RunRequest`s |
| D15 | P2 | **Imported failure kinds are raw verifiers class names, and every error masks.** `failure_kind = errors[0].type` (`:80`) is the oldest error and is not mapped to Crucible's taxonomy (the M10 audit maps unmapped names to `unknown`). A reward function that raises on a malformed agent answer is a `TaskError` (VF `task.py:231`) and is masked, so the policy's failure leaves training (the M10 rule says policy failures score 0) | `C/interop/verifiers/v1_import.py:74-80` | [read] | Map `ProviderError`, `SandboxError`, `HarnessError`, `ToolsetError`, `InterceptionError` to infra kinds; keep `TaskError` masked but count it per task, and quarantine a task whose `TaskError` share is high (it is a task bug, not noise) |

---

## 4. Best picks, ranked

1. **Package the export like RE `apex_agents` (PORT, Apache-2.0, S).** Source: RE `apex_agents/__init__.py`;
   `taskset.py:22-27, 35-80`. A real `crucible_vf` package in `interop/verifiers/` with `__all__ = ["HarborEnv",
   "CrucibleTaskset"]`, `path` for a local export, `ignore_timeouts = False`, pinned image digests injected into a cached
   `task.toml`. Fixes D2 and D11 and makes `vf-eval` and prime-rl able to load Crucible tasks at all.
2. **Separate box only, clean files first (PATTERN from VF `taskset.py:327-354`, S).** Refuse shared mode; remove
   `mask.json`, `steps.jsonl`, `crucible_reward.json` before `test.sh`; bind the mask to a per-grade nonce. Fixes D1.
3. **Parse traces with verifiers' own types (DEPEND, S).** `WireEpisode` / `read_episodes`
   (VF `utils/trace_store.py:32-45`), `Trace.ok`, `Trace.reward`, `Trace.branches`. Fixes D4, D5, D6.
4. **Exact logprobs from Crucible's own generate server (own fix, S).** Keep ids and logprobs per route key and
   substitute them by matching completion ids. Fixes D3.
5. **Dense credit through `trace.info`, produced host-side (PATTERN from PRL `algo/opsd.py:55`, M).** Add
   `replay.from_vf_trace`; write rows host-side; M12 algorithm reads `trace.info["crucible_steps"]` and calls
   `assign_advantages` (PRL `algo/routing.py:17-40`), inserted into `ALGORITHM_CLASSES`. Fixes D8 and corrects 0.1.
6. **Negative controls for foreign bridges (fixtures, S).** VF's NeMo and OpenEnv bridges as gate cases that must end
   errored on a Crucible mask. Fixes D7.
7. **A real verifiers row in the gate (M).** After picks 1-3: the toy oracle through a World-backed verifiers Taskset
   and a Harbor fixture through `HarborEnv`, canonical `reward.json` and `steps.jsonl` compared. Fixes D9.
8. **Pin verifiers and record it (S).** Fixes D12.

---

## 5. Probe index (`scratchpad/m11sweep/g2/`)

| Probe | Runs in | What it shows |
|---|---|---|
| `p1_vf_import.py` | `.envs/verifiers` | `trace_to_record` on nodes serialized by verifiers' own `MessageNode` JSON serializer: P1a rounding (D3), P1b `None` rewards crash (D4), P1c/P1d `ok` vs errors (D5), P1e token-less sampled node, P1f second branch (D6) |
| `p2_vf_export.py` | `.envs/verifiers` | `CrucibleTask.run_verifier` with a runtime stub that follows `Runtime.read`'s cap: P2a planted mask, P2b planted steps (D1), P2c 1 MiB cap (D8), P2d `reward.txt` only (D10); P2e verifiers' loader on Crucible ids, P2f `parse_task` on the toy fixture and task02 (D2), P2g `HarborConfig` defaults (D11) |
| `p3_vf_nemo_bridge.py` | `.envs/verifiers` | verifiers' `NeMoGymTask.nemo_gym` on Crucible's masked `/verify` body (D7) |
| `p4_vf_pin.py` | `.envs/verifiers` | installed verifiers version, editable source and HEAD, RE and PE pins (D12) |
