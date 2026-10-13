# M14 sweep G1: RLM against recursive-LM, REPL-agent and continual-harness sources

Auditor: G1 of six. Crucible HEAD f2d9e18. Scope: `actors/rlm/{repl,kernel,policy,harness_state}.py`,
`runtime/harness/rlm.py`, `train/world_env.py:rlm_world_rollout`, sub-call records (`runtime/record.py`), plus the
trainer path that consumes them. Nothing under `src/` or `tests/` was edited.

Probes live in `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m14/g1/`:
- `p1_kernel.py`: kernel protocol, environment, kill.
- `p2_train.py`: RLM rollout → `TorchTrainer.step`.
- `p3_policy.py`: `RLMPolicy` behaviours.
- `p4_prime_scenarios.py`: Prime Agent's own `test_repl.py` scenarios.
- `p5_records.py`: sub_llm records, credit, shared launcher.
- `p6_harness_state.py`: the continual harness.
- `p7_infra.py`: kernel start failure.

Every probe runs the real Crucible code. Run them with `PYTHONPATH=src:packs/_toy/src .venv/bin/python <probe>` from `crucible/`. The gate
`tests/unit/test_rlm_{kernel,policy,harness,manager}.py` passes: 21 passed, Docker test included.

**Source naming.** In `07-articles/prime-agent/`, `2605.09998.txt` is the **Continual Harness** paper (CH) and
`2608.23552.txt` is the **Prime Agent** paper (PA). The study doc maps them correctly (`01-prime-agent-study.md:9-10`).
Other abbreviations:
- RLM: the `rlm.txt` blog.
- RUST: `prime-agent-rust.txt`.
- PAK: the prime-agent repo (`06-frameworks-and-env-code/primeintellect-ai-prime-agent`).
- VF: primeintellect-ai-verifiers (v1 only; RLMEnv is not in this checkout).
- PRL: infra-primeintellect-ai-prime-rl.
- MH: hf-space multi-harness article.
- GS: GitSwarm paper.
- AR: karpathy-autoresearch.

## Headline defects

| # | Sev | Defect | Crucible | Evidence |
|---|---|---|---|---|
| D1 | **P0** | **A sub-LLM of another model is trained as the policy.** `RLMHarness.wrap` passes `x.sub_llm`. Its token records become `Turn(kind="sub")` with no model field, `valid_for_training` is True, and `TorchTrainer` makes items of them. VF/PRL run sub-agents on the same `ctx.model` (VF `harnesses/rlm/harness.py:225`). | `runtime/harness/rlm.py:101`, `actors/rlm/policy.py:273-279`, `runtime/record.py:67-71`, `core/schema/rollout.py:24-60` (no `model`) | [ran] p5 (a): a "frozen-helper-model" EngineLLM answered the sub-call. The record was train-valid (reward 1.0), its completion was `b'answer from OTHER-MODEL'` with `policy_version=7`, and `TorchTrainer.step` → items=4, records_invalid=0. `test_rlm_policy.py::test_train_rollout_complete_with_token_records` asserts this as correct: its sub_llm is a separate FakeLLM. |
| D2 | **P0** | **An infra fault in a sub-call is scored, not masked.** `_sub` exceptions (engine down, `EngineCallError(failure_kind=provider_unavailable)`, `BudgetExceeded`) become `{"__error__"}` replies, the model sees `RuntimeError`, and the episode is graded. The same fault on the root call masks. This breaks the M10/M11 rule that infra faults mask. In M12 the rollout has root turns, so it trains with the grader's reward. | `kernel.py:241-246` (catch-all), `policy.py:261-265` | [ran] p3 `subinfra`: the sub-call fault gave masked=False, stop=agent_done, reward=0.0, and the model saw `RuntimeError: EngineCallError: engine down`. The root fault gave masked=True, failure_kind=provider_unavailable. |
| D3 | P1 | **Every compacted rollout is masked on interleaved/strip thinking templates.** `_maybe_compact` ignores `tokens.new_segment`, but the turn path honours it (`policy.py:357-360`). The COMPACT user message makes `EngineLLM` re-render, the compaction call is flagged `prefix_break`, and `train/runtime.py:205-220` masks the rollout `capture_incomplete`. Only episodes that never compact train. That is a length-dependent survivorship bias, and long episodes are the RLM's purpose. | `policy.py:447-452` | [ran] p2 `interleaved`: 4/4 compaction calls were `prefix_break`, valid_for_training=False ("token record incomplete for 4/15 calls (first: agent:c4: prefix_break)"). The same run on the preserved codec is valid. |
| D4 | P1 | **Sub-calls break the credit map.** `record.build` copies the parent's ΔΦ onto every sub turn. `turn_credit_map` then treats sub turns as sequential turns of the Φ chain, so Φ is inflated, and a root turn's process credit depends on how many sub-calls its cell made. The doc claims sub-calls "share the parent turn's advantage". | `runtime/record.py:71`, `train/credit.py:66-69` | [ran] p5 `credit`: with the same root ΔΦ = 0.25, root-turn credit was +0.500 with 0 subs, +0.125 with 3 subs, and +0.045 with 10 subs. |
| D5 | P1 | **A restore after a timeout fails on any nondeterministic cell.** The ending is "policy error" and scored. Replay is digest-checked, but the kernel has no virtual clock and no seeded ids. The study doc claims both ("virtual clock; seeded ids", `01-prime-agent-study.md:209`): **WRONG**. One `time.time()` or `uuid4()` value in the namespace makes `SnapshotMismatch` on the next timeout, the driver dies, and `act()` returns `Done` with `crucible:policy_error`. The episode is reported `agent_done`, unmasked, and graded. | `policy.py:397-398`, `kernel.py:295-299`, `policy.py:202-207` | [ran] p3 `replay`: stop=agent_done, masked=False, failure_kind=crucible:policy_error, error `SnapshotMismatch: ... after cell 2`, reward 0.0. |
| D6 | P1 | **Kernel start failure is scored 0.** If `docker run` fails (daemon or image), `_ensure_kernel` raises, which becomes `crucible:policy_error`, unmasked. The same holds for any harness-side exception in the driver, e.g. a replay start. In M12 such a rollout is masked only by accident, because it has no turns. | `policy.py:283-290`, `202-207` | [ran] p7: DockerLauncher with a missing image gave masked=False, stop=agent_done, failure_kind=crucible:policy_error, reward=0.0. |
| D7 | P1 | **A "killed" kernel keeps running inside Docker/Exec sandboxes.** `Kernel.close()` kills the local `docker exec` client. The exec'd python keeps running; Crucible's own `world/exec.py:121-122` documents this. The timeout restore then starts a second kernel next to the wedged one, which shares `--cpus 1` and `/tmp`. With `ExecLauncher` into a World sandbox, the documented coding-World path (`M14-rlm.md:130`), the wedged cell keeps mutating the World after "restore" and into grading. | `kernel.py:174-187`, `141-156` | [ran] p1 `execkill` (real Docker): after the kill, 1 repl process was still in the container and the wedged cell was still writing (`/tmp/still_running` 35 → 45 bytes over 2 s). |
| D8 | P1 | **`LocalLauncher` is not credential-free** (the claim is at `kernel.py:4`). `safe_environ` is a name deny-list (`world/exec.py:109`). It lets through `SSH_AUTH_SOCK` (the operator's ssh-agent), URLs with embedded passwords (`DATABASE_URL`, `HTTP_PROXY`), `GH_PAT` and `HOME`, and `~/.ssh` and `~/.cache/huggingface` are readable. `LocalLauncher` is the **default** for `RLMPolicy(...)` (`policy.py:139`) and for `RLMAgent`/`RLMTaskManager`, which run LLM-written implementer code (`accept/repair.py:82,146`). The RL/eval harness defaults to Docker (`runtime/harness/rlm.py:56`). | as cited | [ran] p1 `env`: the kernel saw `['DATABASE_URL','GH_PAT','HOME','HTTP_PROXY','SSH_AUTH_SOCK']` and `['~/.ssh','~/.cache/huggingface','~/.docker/config.json']` exist. The test only checks names containing KEY/TOKEN (`test_rlm_kernel.py:29-30`), so it is vacuous for these. |
| D9 | P1 | **`rlm_world_rollout(launcher=X)` shares one launcher across all rollouts.** A `LocalLauncher` shares one cwd; a `DockerLauncher` shares one container, and the first `aclose` removes it under the others. One GRPO member can read another's files. `RLMHarness` takes a factory instead (`rlm.py:84-87`). | `train/world_env.py:348` | [ran] p5 `shared`: rollout B read `answer.txt` = `SECRET-FROM-A`, written by concurrent rollout A, in the same `crucible-rlm-*` dir. |
| D10 | P1 | **No sub-call budget.** There is no concurrency cap, no per-episode count or token budget, and `max_tokens` is model-controlled and unclamped. VF's RLM caps concurrent subagents at 4 and tree tokens at 10M (VF `harness.py:90-94,109-113`). Sub tokens are not in `trajectory_tokens` (`policy.py:275`), and sub-calls are not World steps, so `EpisodeLimits` never bound them. One rollout can flood a shared engine. | `policy.py:261-273` | [ran] p3 `batch`: `rlm.batch(['q']*300, max_tokens=10**9)` made 300 sub-calls, peak concurrency 300, and passed `max_tokens=1000000000` to the client. |
| D11 | P2 | **Continual-harness A/B gate holes.** (1) If a guard metric is missing in the candidate (NaN), the guard counts as "not regressed". (2) There is no minimum seed count and no variance or noise gate: CH uses ≥3 seeds and medians (CH:348-349), RUST uses a noise gate and alternating order (RUST:112-122). (3) Base runs all run before candidate runs, not interleaved. (4) An `update` with `kind="note"` on a skill skips the compile check and stays a skill. (5) `HarnessState` is wired into nothing: no `src/` caller, only a docstring in `accept/repair.py:19`. | `harness_state.py:170-189`, `131-137` | [ran] p6: guard NaN gave accepted=True. One seed at +0.01 gave accepted=True. Run order was base0, base1, cand0, cand1. The skill content `"this is not python((("` was stored with kind=skill. |
| D12 | P2 | **Kernel I/O is the protocol channel.** Prime dups the protocol fd and pipes fds 1/2 (PAK `repl.py:1702-1724`). Crucible writes events to `sys.__stdout__`, which is fd 1, so a cell can forge `done`/`host_request` frames, poison the journal digest, or crash the host. A >64 MiB line raises an uncaught `ValueError` from `readline`. `os.close(1)` kills the protocol. Background-task output between cells is dropped; Prime ships it as `id:null`. | `repl.py:35,44-61`, `kernel.py:194-206` | [ran] p1 `forge`: a forged done gave `finished='FORGED'`, stdout `forged`, digest `x` recorded in the journal. [ran] p1 `bigline`: a 70 MB write raised `ValueError: Separator is not found` in the host. [ran] p4 S4: background prints were lost. [ran] p4 S8: `os.close(1)` gave killed, "kernel process exited". |
| D13 | P2 | **Smaller kernel-protocol gaps against Prime.** No `interrupt`: a blocking cell costs a kill plus replay, and Prime preserves state. The result is cut at 1 MiB with **no marker**. The error keeps the last 20,000 chars, so `ValueError: xxx…` loses its type. A malformed line is silently ignored; with `"["*100000` the reader thread dies (`RecursionError` is not a `ValueError`) and the kernel goes deaf. There is no host-request payload cap (Prime: 16 MiB). Children started by a cell survive `close()` (`p.kill()` only, although `start_new_session=True` would allow `killpg`). | `repl.py:125,147,156-164`; `kernel.py:183-187` | [ran] p4: S3 result_len=1048576 marker=False; S3 evalue 20000 chars of `x`; S6 `[`×100000 gave killed, dt 6.0 s; S3 a 17 MB host payload was accepted. [ran] p1 `orphan`: a `sleep 321` child was still alive after close. |
| D14 | P2 | **The cell deadline counts time the policy does not control.** It runs while the host serves World steps (other actors' turns) and sub-LLM latency. For sub-calls this matches RLM:105 ("also limits its use of sub-LLM calls"). The model is not told the timeout or each call's duration (RLM:454 says it always is). | `repl.py:137`, `policy.py:57-64` (no timeout in `RLM_SYSTEM`) | [ran] p1 `clock`: 0 s of compute plus 3 s of host latency against a 2 s deadline gave timed_out. [ran] p3 `deadline`: two 0.6 s sub-calls against a 1 s deadline timed out and `X` was rolled back. |
| D15 | P2 | **Training-record shape.** Sub turns drop `version_spans` (`record.py:68-71`), so staleness is under-reported for a sub-call that straddles a weight update. In critic mode, sub-call tokens are chained into the root's GAE Bellman chain (`trainer.py:244`), which contradicts "tree node". `flatten` restarts the root chain after each sub-call (exact, but it re-encodes): 7 sequences for 5 chains plus 1 sub. A second `python` tool call in one reply is silently dropped, though its tokens stay in the trained completion and in the model's TITO context (`policy.py:366-381`). | as cited | [ran] p2: flatten made 7 sequences for 5 (actor, segment) chains + 1 sub (prompt tokens 31,217 vs completion+obs 19,341). [ran] p3 `twocalls`: the second call was not executed and the assistant message keeps 1 of 2. |
| D16 | P2 | **Smaller gaps and doc drift.** The output cap leaks: INBOX announcements are appended after `_format` (`policy.py:394-396`); [ran] 50 messages give ~10.9k chars against a 200-char cap. A failed `_archive` silently drops HISTORY (`policy.py:497`). `rlm_world_rollout` never passes a kernel seed (all rollouts use seed 0; `world_env.py:348`). The M14 doc lists `actors/rlm/manager.py`, which now lives at `accept/repair.py` (`M14-rlm.md:17`). The design's "skills must pass the task's checks" (`02-rlm-agent-design.md` §3) is compile-only. | as cited | |

**Training check (task 3).** The core passes.
- [ran] p2: two RLM rollouts went through the real path (`rlm_world_rollout` → `EngineLLM` byte codec → `runtime.record.build`), with behaviour log-probs computed by the same `TinyPolicy`.
- Rollout 1: reward 1.0, 15 calls (10 turn, 1 sub, 4 compaction), 4 segments, 0 prefix breaks, valid. Rollout 2: reward 0.083, valid.
- `TorchTrainer.step` gives loss 1.8161 and grad_norm 0.1793 with both `item_unit="segment"` (10 items, 60,682 forward tokens) and `"call"` (20 items, 118,126 forward tokens). The values are identical, so segment packing stays exact across sub-calls and compactions.
- A temperature mismatch (0.7) is refused.

The defects are around this core: D1–D4, D15.

## Requirement ledger

Status is done / partial / missing / WRONG. "[ran]" refers to the probes above.

| Req (source) | Crucible | Status |
|---|---|---|
| Prompt-as-variable: payload only via code (RLM:89, PA:204-206) | `TASK/TOOLS/OBS/INBOX` in the kernel, `policy.py:66-71`; the prompt holds the task + tool index (`policy.py:467-475`) | done. RLM keeps the prompt in context, Crucible duplicates `TASK` (fine). |
| Empty payload works like an LLM (RLM:151) | A plain reply gets a continuation, then Done (`policy.py:367-376`) | done |
| REPL output cap 8192, configurable (RLM:89) | `max_output_chars=8192`, `policy.py:323-327`, `RLMConfig:45` | partial: the INBOX tail is uncapped (D16) |
| Kernel stream caps with a marker (PAK repl.py:53-54, 534-565; host 65,536/stream) | 1 MiB/stream silent (`repl.py:34,59-60`), result slice with no marker (`:125`), error tail 20k (`:147`) | partial (D13) |
| Depth config {0,1,N} (RLM:602) | `RLMPolicy` is fixed at depth 1 (no depth-0 switch); `RLMAgent.max_depth=2` (`repair.py:81,124`) | partial |
| Tools only in sub-LLMs variant (RLM:59-63) | `sub_tools` designed (`02-…:59`), not implemented (`M14-rlm.md:72`) | missing (acknowledged) |
| Batch primitive, ordered results (RLM:55-57) | `llm.batch` → `asyncio.gather` (`policy.py:263-265`) | done. Note that `gather(rlm.query…)` in a cell is serialized by the host loop (`kernel.py:241-249`). |
| Sub-call budgets / concurrency (VF harness.py:90-116; PA:628-630 records peak) | none | missing (D10) |
| Sub-calls count against the cell deadline (RLM:105) | yes (`repl.py:137`) | done. World-step latency counts too (D14). |
| The model is told the timeout and each call's duration (RLM:454) | no | missing (D14) |
| Per-cell deadline 120 s (RLM:105, h1) | `cell_timeout_s=120` (`RLMConfig:46`) | done. Prime has none (PAK `manager/mod.rs:591-593`). |
| Kill wedged cell, kernel recovers (PAK abort_kernel_cell.rs) | kill + replay (`kernel.py:224-229`, `policy.py:397-398`) | partial: Exec/Docker kill leaves the process running (D7); replay breaks on nondeterminism (D5) |
| Interrupt preserves state (PAK repl.py:391-470) | none | missing (D13) |
| Determinism / seeding (RLM:246; RUST:48) | `random` seeded, `PYTHONHASHSEED` (`repl.py:200`, `kernel.py:87`); no virtual clock / uuid | partial; study claim WRONG (D5) |
| Compaction keeps events retrievable from the REPL (PA:190-192) | `HISTORY` verbatim archive (`policy.py:477-497`); [ran] the gate reads `HISTORY[0]` after 3 compactions | done (silent loss if the archive cell fails, D16) |
| `[python-state]` live names after compaction (PAK ipython_state.rs:119-196) | `ARCHIVED` + `kernel.names()` (`policy.py:457-459`) | done |
| Compaction branch trainable / independent sample (VF trace.py:515-549; VF:113-117) | the compaction call is `kind="compaction"`, a new segment (`policy.py:449-465`) | partial: masked on re-rendering templates (D3). No rejected-summary handling ("summary trainable only when good", open, `M14-rlm.md:72`). |
| Sandbox isolation, no network (RLM:75; GS:807; VF network_allow=[]) | `DockerLauncher`: no network, read-only, cap-drop ALL, uid 65534, mem/pids/cpus (`kernel.py:119-122`); [ran] gate Docker test | done for Docker. Prime has none (PAK `startup.rs:166-189`). |
| Credential-free env (MH openenv:53-56; VF interception secret `server.py:372-377`) | Docker: host env not forwarded (done). Local: deny-list (D8). | partial |
| Sub-LLM credentials stay on the host (VF server.py:952-955) | sub-calls run in the host (`policy.py:271-274`); the kernel only sees text | done |
| Host bridge surface enumerated (RLM:57-61; PA:207-210) | `world.call/send`, `rlm.query/batch/finish/note` (`policy.py:225-269`); unknown type gives an error | done |
| Bridge error → exception in the cell (PAK `__init__.py:144-150`) | `__error__` → RuntimeError (`repl.py:74-75`); [ran] p4 S5 | done (but infra errors too: D2) |
| Malformed protocol → error event, keep serving (PAK repl.py:1483-1590) | silently ignored; deep nesting kills the reader | partial (D13) |
| Payload cap on host requests (PAK repl.py:56) | none ([ran] 17 MB accepted) | missing |
| Answer variable with ready flag (RLM:77-85) | `rlm.finish(summary)` → `FINISHED` → Done (`policy.py:106-108,402-406`); the World state is graded | done (adapted) |
| Goal continuation / audit (PA:311-315) | `CONTINUE` up to `max_continuations` (`policy.py:129-130,369-372`) | done |
| Stall detection: CH failure signatures (CH:246-260, 1021) | identical cells / errors / World calls (`policy.py:409-431`) | done |
| Kill/cleanup: no leaked kernels (h7 GS:3740) | `aclose` after every episode (`rlm.py:118-123`); [ran] gate step-limit test | partial: orphan children (D13), wedged exec kernels (D7) |
| Infra fault masks, never 0 (MH openenv:317-319; PRL algo/base.py:11-18 drops errored) | root faults mask. Sub-call, kernel start and restore faults are scored. | WRONG (D2, D5, D6) |
| Sub-calls as separate samples, tree nodes (MH openenv:121-137; VF trace.py:192-201; PRL trajectories.py:150-196) | `kind="sub"` own sequence (`record.py:67-71`, `rollout.py:173-175`, `bridges/common.py:87`) | done for the policy loss; WRONG in critic GAE (D15) |
| One advantage per rollout broadcast to every branch (PRL routing.py:11-40) | sequence advantage on all turns (`credit.py:44-48`); sub copies parent ΔΦ | done for outcome; WRONG for turn credit (D4) |
| Sub-calls on the trained model only (VF harness.py:225) | `sub_llm` may differ, untracked | WRONG (D1) |
| Same harness per GRPO group (MH openenv:383) | `rlm:<digest>@1` (`rlm.py:64-65,90`) | done |
| Temperature matches the trainer (MH openenv:383) | the sub uses the root temperature (`policy.py:273`); the trainer refuses a mismatch; [ran] p2 | done |
| Token-in-token-out per call, versions (MH openenv:90-110) | root/sub/compaction carry engine records; [ran] p2 | done (sub `version_spans` dropped, D15) |
| Weight per rollout, not per row (MH training:195-196, 379-380) | M12 per-rollout row weights (`objectives.row_weights`) | done (M12) |
| Same output cap in train and eval (MH training:350-356) | one `RLMConfig` hashed into the harness id | done |
| Continual harness typed state: prompt notes, memory, skills, subagents (CH:149-165; PA:265-269) | `KINDS` (`harness_state.py:34`) + fsm | done |
| Immutable base prompt (PA:270-274) | `_validate` (`harness_state.py:125-126`) | done |
| Versions + rollback (PA:270-274; PA:636-638) | whole-state versions, `rollback_to` (`:93-112`); [ran] p6 | done |
| Same-seed A/B before keeping an edit (RUST:116-122; AR program.md:103-104; CH has none) | `evaluate` (`:163-199`) | partial: NaN guard, no min seeds or noise gate, not interleaved (D11) |
| Retire never-helping entries (CH:1873-1876) | `retire_candidates` (`:229-231`) | done (`digest()` counts are not saved until the next `_save`) |
| Cross-episode memory risk: validate before promotion (PA:632-638; CH:361-362) | policy-side harness off by default (`M14-rlm.md:131-133`); not wired anywhere (D11) | done by omission |
| Multi-agent spawn: handles, persistent children (PA:208-213) | `rlm.spawn` is synchronous, depth ≤2 (`repair.py:123-135`) | partial (design choice) |
| Fixed per-experiment budget and keep/discard by metric (AR program.md:23,108) | `evaluate(fn)` is caller-defined; no wall-clock budget in `HarnessState` | partial |

## Protocol comparison with Prime Agent (task 2)

PAK is protocol v3 (`repl.py:38`, `protocol.rs:14`). Crucible says `protocol: 1` (`repl.py:185`) and the host never checks it. Prime refuses a mismatch (`startup.rs:251-259`).

**Messages.** Prime's requests are `execute, interrupt, host_reply, snapshot, restore, list_names, mcp_status, bash_activity, factory_activity, shutdown`. Crucible has `execute, host_reply, digest, names, shutdown`. Prime sends separate `stdout/stderr/result/display/error/done` events; Crucible packs everything into one `done`. Prime rejects duplicate ids; Crucible has no check.

**Results of Prime's tests replayed on Crucible** ([ran] p4):

| Prime scenario | Prime (test_repl.py) | Crucible |
|---|---|---|
| S1 blocking loop + interrupt (518-534) | KeyboardInterrupt, state kept | killed; `KEEP` lost at kernel level (the policy replays) |
| S2 exception (642-662) | `<cell-N>` frame, no runtime frames | same: `<cell>` yes, `repl.py` no, last line `ValueError: nope` |
| S3 300k stdout (1086-1104) | 64 KiB frames, exact | exact, 300,000 |
| S3 3M repr | 1,048,576 + marker | 1,048,576, no marker |
| S3 3M evalue | newest kept + marker | 20,000 trailing chars, type lost |
| S3 17M host payload | ValueError in the cell | accepted |
| S4 background task across cells (274-327) | persists, output `id:null` | persists (N=11 after 0.5 s); output dropped |
| S5 host error reply (1595-1699) | RuntimeError | RuntimeError "handler exploded" |
| S6 `{not json` (385-423) | ProtocolError, serving | silent, serving |
| S6 `[`×100000 | ProtocolError, serving | reader dies, cell killed after 6 s |
| S8 stdio abuse (1939-1983) | done ok | `sys.stdout=None` ok; `os.close(1)` gives "kernel process exited" |

**Where Crucible is stronger:**
- per-cell deadline (Prime has none);
- sandbox and stripped env (Prime inherits the full host env);
- journal replay with recorded host replies (Prime uses best-effort dill);
- World calls as World steps;
- the trainable record contract.

## Picks (task 4)

| Source | Pick | What | Licence | Effort |
|---|---|---|---|---|
| PAK `prime-agent-runtime/src/rlm/repl.py` | **PORT** | **Kernel I/O:** a private dup of fd 1 for protocol events, fds 1/2 piped through pump threads with drain fences (`:227-243,1702-1724`). This closes D12. **Protocol hygiene:** ProtocolError replies, duplicate-id refusal (`:1483-1590`), a 16 MiB host payload cap (`:56,128-138`), a 1 MiB result/evalue cap with markers keeping the newest frames (`:534-565`). **Interrupt:** SIGINT to the main thread (`:391-470`) as a first resort before kill + replay. **Owner watchdog:** `:1598-1695`. **Host side:** a protocol line cap with repair (`manager/repair.rs:10-60`). The code is stdlib-only, so it stays pushable into sandboxes. | MIT (`LICENSE`; `Cargo.toml:24`) | 1–2 d |
| PAK dill snapshot (`repl.py:771-1170`) | PATTERN | A fallback when journal replay diverges (D5): snapshot after each ok cell, and restore the snapshot if replay fails. It needs dill in the kernel image. | MIT | 1 d |
| PAK `bash.py:332,1068-1080` (process groups killed on shutdown) | PATTERN | `killpg` on `LocalLauncher` close. Inside containers, run the kernel under its own pgid and kill it from inside the container (`exec.with_deadline` pattern) on wedge/close. This closes D7 and the orphan part of D13. | MIT | 0.5 d |
| VF `harnesses/rlm/harness.py:76-116` | PATTERN | `RLMConfig` knobs for D10: `max_depth`, `max_concurrent_subagents` (4), `max_subagent_calls`, a tree-wide `max_total_tokens`, and a clamp on sub `max_tokens`. Put them in the harness digest. | MIT | 0.5 d |
| VF `trace.py:192-201,515-553` + `graph.py:1-11` | PATTERN | Branches with semantic edges (`subagent_call/return`, `compaction_attempt`) and a `trainable` flag for rejected summaries. This is the open "summary trainable only when good" item. | MIT | 1 d |
| PRL `orchestrator/algo/routing.py:11-40`, `algo/base.py:11-18` | PATTERN | One scalar per trace on every trainable branch: it confirms Crucible's outcome path and is the fix shape for D4 (sub turns carry no own ΔΦ). Errored traces are dropped, never scored: the fix shape for D2/D6. | Apache-2.0 | 0.5 d |
| RUST:112-122, CH:348-349 | PATTERN | Fixes for `HarnessState.evaluate` (D11): A/B on the same sandbox in interleaved order, ≥3 seeds, a paired noise gate, NaN = reject. | text only | 0.5 d |
| nano-rlm (VF `harness.py:32`) | none | Not vendored, licence unknown. Crucible keeps its own kernel. | ? | — |
| AR (MIT), GS (paper/site), MH (CC-BY-4.0 text) | PATTERN only | Fixed experiment budget with keep/discard (AR); execute-only references (GS:3664-3667). The multi-harness evidence supports hashing the harness config into the group key, which Crucible already does. | — | — |

## Suggested fixes (ordered)

1. **D1:** record `model` on each call and on `Turn`. In train mode, refuse (or freeze as `frozen_actors`-style context) any sub-call whose model is not the policy's. Alternatively, forbid `sub_llm` under RLMHarness train mode.
2. **D2/D6/D5:**
   - Let an exception from `_sub` that carries a `failure_kind` abort the episode as a masked infra fault; don't reply `__error__`.
   - Classify a kernel-start failure as infra (`crucible:sandbox_error`).
   - Make a replay divergence `crucible:harness_unsupported` (masked), or fall back to a snapshot. Remove the "virtual clock; seeded ids" claim, or implement it (patch `time`/`uuid` in the bootstrap).
3. **D3:** in `_maybe_compact`, if `r.tokens.new_segment`, open the segment before recording, the same way the turn path does at `policy.py:357-360`.
4. **D4:** have `record.build` give sub turns `advantage=None` (or 0) in the ΔΦ chain, and make `turn_credit_map` skip `kind=="sub"` in the Φ accumulation and then copy the parent's credit.
5. **D7–D10, D12–D13:** as in the picks. Make `rlm_world_rollout` take a launcher factory (D9). Change `RLMAgent`/`RLMPolicy` to default to Docker, or make `LocalLauncher` allow-list the environment and use `HOME=<tmp>` (D8).
6. **Regression tests:** add one test per defect that fails on f2d9e18.
