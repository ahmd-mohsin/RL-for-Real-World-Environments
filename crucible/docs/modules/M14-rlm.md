# M14: the RLM agent (Recursive-Language-Model policy and task manager)

**Research:**
- `docs/research/m14-rlm/01-prime-agent-study.md`: Prime Agent's Rust rewrite, its Python runtime, the Prime Agent
  and Continual Harness papers, and the RLM and Rust-rewrite blogs.
- `02-rlm-agent-design.md`: the design.

**Code:** `src/crucible/actors/rlm/`

| File | What |
|---|---|
| `repl.py` | The kernel: a stdlib-only persistent Python REPL over NDJSON. Top-level `await`, per-cell deadline, one host bridge `await host(type, **payload)`, namespace digest, seeded `random`. It can be pushed into any sandbox. |
| `kernel.py` | Host side. `LocalLauncher` runs a subprocess with a credential-free environment. `ExecLauncher` runs any argv prefix, e.g. `docker exec -i` into the World's sandbox. It also holds the journal, **exact restore by replay** (digest-checked; divergence raises), and kills wedged cells. |
| `policy.py` | `RLMPolicy`: the RLM as a World policy. Prompt-as-variable (`TASK`, `TOOLS`, `OBS`, `INBOX`). **One World step per `world.call`.** Sub-LLMs `rlm.query/batch`, recorded as `kind="sub"` calls. `rlm.finish`, goal continuation, compaction with the kernel note, stall detection, timeout restore, mid-cell checkpoint/resume. |
| `harness_state.py` | The continual harness: typed entries (`note`, `memory`, `skill`, `subagent`, `fsm`), whole-state versions, `rollback_to`. **Same-seed A/B evaluation before any edit is kept.** Also a ranked digest, usage counters and retirement candidates. |
| `runtime/harness/rlm.py` | **`RLMHarness`**: the policy's model inside the RLM scaffold, as a harness for evaluation and RL rollouts. `RLMConfig` (output cap, budgets, compaction, temperature) is hashed into the harness id `rlm:<digest>@1`, so one GRPO group never mixes scaffolds. `train.world_env.rlm_world_rollout` is the M12 `rollout_fn`. |
| `manager.py` | `RLMAgent` (the RLM loop with no World, proxy objects, `rlm.spawn`, `obj.help()`, budget wrap-up), `ForgeWorkspace` (versioned tasks; the verifier is M9; commit needs a passing verification), `TaskFSM` (Planner → Implementer (child RLM) → Reviewer (a different model) → objective Verifier, journaled and resumable) and `RLMTaskManager`. |

## Training contract

- **`RLMPolicy.model_calls`** uses the `LLMToolPolicy` schema: `call_id`, `kind`, `segment`, `step`, `tokens`, `actions`.
- **Sub-LLM calls** carry `parent`. `runtime.record.build` makes each its own sequence: a tree node, not a link of the
  actor's linear chain. It shares the parent turn's advantage.
- **Train mode:** a rollout is trainable when every root and sub call has a token record (test:
  `test_train_rollout_complete_with_token_records`).

## The RLM harness for long-horizon RL trajectories

The context of a long episode is kept by the harness in the kernel, not in the prompt:
- **The prompt holds** the system prompt, the task, the RLM's own turns (output capped at 8192 characters) and, after a
  compaction, its summary and a pointer.
- **The kernel holds** `OBS` (every tool result), `INBOX` (every message) and `HISTORY` (every compacted segment, verbatim:
  the summary and each cell's code and output). The model searches its past with code after the summary replaced it.
- **Compaction** is the recorded summary call of DESIGN §10.4. Each one starts a segment, a fresh exact-prefix token
  sequence. It is skipped while there is nothing to drop (a long task alone can exceed the trigger).

Requirements taken from the harness docs (DESIGN §10.3–10.4, `docs/research/long-horizon/01-frameworks.md` §(a),
`docs/research/m5-m6/06-harnesses.md`):

| Requirement | Where |
|---|---|
| A token record on every call (TITO; never re-tokenized) | root, sub and compaction calls carry the engine's record; a missing one masks the rollout `capture_incomplete` |
| One exact-prefix sequence per segment; a prefix check inside a segment | linear capture (`prompt_delta`, `LLMToolPolicy._compress`); `runtime.record.build` |
| An engine re-render (thinking template) is a new segment, not a break | `tokens.new_segment` → segment `strategy: "rerender"` |
| Sub-calls are tree nodes | `kind="sub"` with `parent`; they share the root turn's advantage |
| Calls joined to World steps | `step` on every call |
| The compaction mode is recorded in `harness@version`; one harness per group | `rlm:<sha of RLMConfig>@1` |
| Evidence for audit and exact restore | `rlm.<actor>.json`: config, counters, segments, notes, and the kernel journal (every cell with its recorded host replies) |
| No leaked kernels | `RLMPolicy.aclose()` after every episode, including limit stops (container removed) |
| Model-written code is sandboxed | `DockerLauncher` by default: no network, read-only root, no capabilities, uid 65534 |

Gate: `tests/unit/test_rlm_harness.py`. A scripted byte-codec engine (exact TITO) drives the RLM through the backlog
with a 2.5k-token budget:
- three compactions, one sub-call, deltas within each segment, no prefix breaks, policy versions present, and reward
  1.0;
- the model reads the first archived cell back from `HISTORY` after the summaries;
- the same RLM runs as an M12 `rollout_fn`;
- a step-limit stop closes the kernel.

**Live (2026-10-11, Modal L4, vLLM 0.11 serving Qwen3-4B-Instruct-2507 over TITO, 2 train rollouts, under $0.20):**
  - Both rollouts were train-valid: real engine token ids and logprobs, `policy_version` stamped, 10–12 delta-chained
    turns, and no prefix breaks.
  - Reward was 0. The 4B model never made a World call: it searched the host's working directory for `refund_policy.md`
    and messaged the supervisor.
  - That surfaced two harness defects, both fixed:
    - **The kernel ran on the host.** The kernel is now a `DockerLauncher` container (no network, read-only root, all
      capabilities dropped, uid 65534, memory and pid limits), the `RLMConfig.kernel` default. `LocalLauncher` starts
      in an empty directory and is for trusted code only.
    - **The model never read `TOOLS`.** A one-line-per-tool index now follows the task (`tool_index`).
  - Also found: vLLM 0.11 breaks on transformers 5 (pinned in `scripts/modal_vllm.py`).

Still open: accepting a summary as trainable only when it is good (01-frameworks §(a)), cross-segment GAE, and
`sub_tools`.

## Gate

- `tests/unit/test_rlm_kernel.py`
- `test_rlm_policy.py`
- `test_rlm_manager.py` (6)
- `test_rlm_harness.py` (4)

## Live smoke, 2026-10-10 (GPT-6 Luna policy; Claude Haiku 5.5 reviewer)

**`RLMPolicy`** on the toy helpdesk:
- It works mechanically: 21–23 World calls through the bridge, $0.003–0.004 per episode.
- It scored 0, and so did native function calling with the same model (2/2 runs at 0). Both fail on the toy
  supervisor's strict approval phrasing, so the RLM harness is not the cause.

**`TaskFSM`** on the B.15 task. Six live runs, each one surfacing a harness defect, each fixed and covered by the
unit gate:
- planner and reviewer budgets were eaten by reasoning tokens (now 4000 tokens, one retry);
- a reviewer error was treated as a rejection (now re-reviewed);
- `forge.show` dropped each check's `expect`;
- verifier findings had no evidence (now the A16 hack's actions);
- the proxy API was opaque (now `obj.help()`);
- the vocabulary did not say that `approver` is an actor id;
- the reviewer lacked the policy documents, the rule semantics, and what each acceptance check proves (`CHECK_MEANING`);
- the implementer ran out of turns after reaching a correct candidate (now a 2-turns-left wrap-up, 20 turns);
- a candidate that dropped protective checks passed A16, because its audit used its own checks (now the committed
  version's checks are an audit floor);
- the implementer invented spec fields that the grader silently ignored (now vocabulary conformance in `add_check`).

- a candidate that dropped every scope check passed A1 and A16 (no hack makes collateral edits); A7 is now in the
  default verifier set.

**Seventh run:** verification and review rejected three bad candidates for correct reasons. The remaining weak link is
the implementer model, which over-edits (dropping scope checks) instead of adding the one approval check.

Live spend for the whole smoke: about $0.20.

**Stronger models (about $0.46 more), 2026-10-10:**
  - Implementer Claude Sonnet 5.5 (alias `implementer-strong`), reviewer GPT-6.1 Sol (`reviewer-strong`, a different
    family). Every candidate in both runs passed A1, A7 and A16; the implementer is no longer the weak link.
  - Run 1 (GPT-6 Luna reviewer): the reviewer rejected the candidates over ambiguities in the vocabulary. The text now
    says the rule only examines in-episode calls, that the supervisor only answers the agent, and that `approver` is an
    actor id.
  - Run 2 (Sol reviewer): it rejected all three candidates. Two of its reasons were real:
    - **`approval_grounded` matched approvals by amount alone.** An approval asked for another order at the same amount,
      or one approval reused for a second refund, grounded the refund. Now an approval is bound to the request it
      answers (the latest earlier message to the approver must name the refund's order or ticket id) and is consumed.
      Every oracle already names both. Test: `test_approval_is_bound_to_its_request_and_used_once`.
    - **The reviewer could not tell a final-state check from a delta check**, because it did not know the seed. The
      vocabulary's `start_facts` (scalar SQL) are evaluated on the seeded start state and shown in `forge.show` and the
      reviewer input as `start_state`.

The A2A path with a real agent (`LLMToolAgent`) is in DESIGN.md C7.

## Open

- A real coding World for `RLMPolicy`: the kernel running inside the task's Docker/gVisor sandbox (the
  `ExecLauncher` path is tested with a no-network container).
- The policy-side continual harness, which stays off for training and evaluation: cross-episode memory is the B.15
  threat.
- Training a model *in* the RLM scaffold: the RLM blog, the Prime Agent paper and Continual Harness all report that
  untrained models under-use it.
