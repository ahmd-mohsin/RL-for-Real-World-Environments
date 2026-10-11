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
| `manager.py` | `RLMAgent` (the RLM loop with no World, proxy objects, `rlm.spawn`, `obj.help()`, budget wrap-up), `ForgeWorkspace` (versioned tasks; the verifier is M9; commit needs a passing verification), `TaskFSM` (Planner → Implementer (child RLM) → Reviewer (a different model) → objective Verifier, journaled and resumable) and `RLMTaskManager`. |

## Training contract

- **`RLMPolicy.model_calls`** uses the `LLMToolPolicy` schema: `call_id`, `kind`, `segment`, `step`, `tokens`, `actions`.
- **Sub-LLM calls** carry `parent`. `runtime.record.build` makes each its own sequence: a tree node, not a link of the
  actor's linear chain. It shares the parent turn's advantage.
- **Train mode:** a rollout is trainable when every root and sub call has a token record (test:
  `test_train_rollout_complete_with_token_records`).

## Gate

- `tests/unit/test_rlm_kernel.py`
- `test_rlm_policy.py`
- `test_rlm_manager.py`

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

Live spend for the whole smoke: about $0.15.

## Open

- A real coding World for `RLMPolicy`: the kernel running inside the task's Docker/gVisor sandbox (the
  `ExecLauncher` path is tested with a no-network container).
- The policy-side continual harness, which stays off for training and evaluation: cross-episode memory is the B.15
  threat.
- Training a model *in* the RLM scaffold: the RLM blog, the Prime Agent paper and Continual Harness all report that
  untrained models under-use it.
