# M14 design 02: the Crucible RLM agent (policy + task manager)

Ask: "add an RLM agent for our workflow, very densely, that manages and improves the tasks, especially for very long
horizon". Study: `01-prime-agent-study.md`. Crucible gets **one RLM core** with two uses:

1. **`RLMPolicy`**: an RLM that *acts* in a World for very long episodes. It is trainable, so every root and sub-LLM
   call keeps its token record. It is checkpointable to the exact state.
2. **`RLMTaskManager`**: an RLM that *manages and improves tasks*. It is a long-running orchestrator over the Swarm
   Forge (M8/M13). It runs the Planner → Implementer → Reviewer → Verifier FSM from the Rust rewrite, and improves itself
   through a measured continual harness.

## 1. The kernel (`actors/rlm/repl.py` + `kernel.py`)

- **Process:** a stdlib-only Python process, so it can be pushed into any sandbox (`Exec.push_file`) and run with
  `python3`. The protocol is NDJSON over stdio, modeled on Prime's v3 (`repl.md`) but smaller:
  - Requests: `execute{id, code, timeout_s}`, `host_reply{id, data}`, `names`, `digest`, `shutdown`.
  - Events: `ready`, `host_request{id, cell, data}`, `done{id, stdout, stderr, result, error, timed_out, finished}`.
- **Cells:**
  - top-level `await`, a persistent namespace, one event loop (as in Prime);
  - stdout/stderr captured per cell;
  - the repr of a trailing expression;
  - **a per-cell deadline** (Prime has none). A timed-out cell is cancelled; if the kernel is wedged, the host kills it
    and restores (below).
- **One bridge:** `await host(type, payload)` (Prime's `host_request`). Everything that touches the world, models or the
  forge goes through it, so **the host decides and records every effect**.
- **Exact restore by journal replay** (Prime's dill snapshot is best-effort, and its factory state is lost):
  - The host keeps a journal: every executed cell and the host replies it received, in order.
  - **Restore** = fresh kernel, the same seed, re-execute the cells with recorded replies.
  - The namespace **digest** (sorted names plus type plus a hash of the repr for plain values) must equal the recorded
    one, or restore raises `SnapshotMismatch`, never a silent partial restore.
  - A partly executed cell is replayed from its recorded replies, then switches to live replies. That makes a
    mid-cell checkpoint resumable (`runtime.checkpoint`).
- **Isolation:** `LocalLauncher` (a subprocess with a scrubbed environment, `safe_environ`) or `ExecLauncher` (`docker
  exec -i` into the World's sandbox: Docker, the gVisor tier, pooled). Model-written code never runs in the harness
  process.

## 2. `RLMPolicy` (an `actors.policy.Policy`)

- **Prompt-as-variable, for real** (Prime only asks for it). The kernel is bootstrapped with:
  - `TASK`, the instruction;
  - `TOOLS`, the World tools with their JSON schemas;
  - `OBS`, an append-only list of every tool result;
  - `INBOX`, messages from other actors.

  The model's context holds the system prompt, the task, and its own turns: code plus output **capped at
  `max_output_chars` (8192, as in the RLM paper)**. Observations reach the model only as far as it prints them.
- **World calls are World steps.**
  - `await world.call("tickets.issue_refund", ticket_id=1, ...)` (or `await world.tickets.issue_refund(...)`) raises a
    host request.
  - The policy turns it into a `ToolCall` and returns it from `act()`. `run_episode` steps the World, and the next
    `act(obs)` delivers the result back into the awaiting cell. A coroutine bridge runs this: the model loop is one
    task, `act()` exchanges with it through futures.
  - So step limits, interception, the trace, the dense progress and the M9 checks all see one World action per call,
    exactly as for `LLMToolPolicy`. `world.send(to, text)` maps to `Message`.
- **Sub-LLMs**, RLM style (depth-1 synchronous recursion, creditable): `await rlm.query(prompt, context=...)` and `await
  rlm.batch([...])`, run by the host in parallel.
  - Each is recorded as a model call `kind="sub"` with `parent` = the root call that issued it, so token records and
    advantages flow (open problem in all sources: we credit sub-calls with their root turn's advantage).
  - `sub_tools=True` lets sub-LLMs call World tools, the RLMEnv "tools only in sub-LLMs" variant.
- **Answer-as-variable:** `rlm.finish(summary)` ends the episode (Done).
  - A reply without code and without `finish` gets a goal continuation (Prime's `[goal: continuation]` with "audit the
    current state against every requirement"), up to `max_continuations`, then Done.
- **Long horizon:**
  - **Compaction** (Prime pattern) at `ContextConfig.trigger`. The summary prompt carries the *kernel note* ("every
    variable stays available; record names worth remembering"). The new context gets a `[python-state]` line listing
    live names. Each compaction starts a segment; the summary is a recorded `kind="compaction"` call.
  - **Stall detection** (CH failure signatures): the same cell code k times, n consecutive cell errors, or the same
    World call with the same result k times. Each injects `[stall-detected: …]` with the signature, and the episode
    stops `too_many_errors` after `max_stalls`. Self-repair dies after about 500 turns of a stall (CH), so stopping
    early is cheaper.
  - **Timeout:** a timed-out cell restores the kernel to before the cell and tells the model so.
- **Checkpoint:** `state_dict()` holds messages, model calls, the journal (including the in-progress cell), counters
  and stall state. `load_state_dict()` restarts the kernel and replays.
- **Training records:** the same `model_calls` schema as `LLMToolPolicy` (`call_id`, `kind`, `segment`, `step`,
  `tokens`, `actions`), plus `parent` for sub-calls. `runtime.record` builds the rollout unchanged.

## 3. Continual harness (`actors/rlm/harness_state.py`)

- **Typed entries**, Prime's five kinds renamed: `note` (supplemental prompt), `memory`, `skill` (code, imported
  into the kernel), `subagent` (role spec), `fsm` (workflow spec).
  - Each entry is scoped by `(family, task|None)`.
  - Stored in an append-only version log; the state hash is per version.
- **Measured edits**, which Prime lacks:
  - `propose(edits)` returns a candidate.
  - `evaluate(candidate, evaluate_fn, seeds)` runs **the same seeds** on the current and candidate harness.
  - Accept only if the candidate improves the primary metric by at least `margin` and no guard metric regresses;
    otherwise reject and record the result. Every event stores its metrics.
- **Rollback:** `rollback_to(version)` restores a whole recorded state (stack-safe, unlike Prime's per-edit inverse).
- **Usage counters per entry** (shown, used, helped) and retirement of entries that never help (the create-and-forget
  finding in CH).
- **Safety:**
  - edits that persist skills must pass the task's checks;
  - **policy-side memory is off by default**, because cross-episode memory is the B.15 threat;
  - the manager's harness is local to a forge family.

## 4. `RLMTaskManager` and the task FSM (`actors/rlm/manager.py`, `fsm.py`)

- **The manager** is an RLM agent with its own kernel. Its namespace exposes a **forge facade** through the host
  bridge, so the operations run in the host and are journaled:
  - `forge.tasks()`, `forge.show(id)`, `forge.accept(id, checks)` (M9), `forge.hack(id)` (A16);
  - `forge.grade(id, actions)`;
  - `forge.add_check(id, spec)`, `forge.drop_check(id, check_id)`, `forge.add_negative(id, spec)`: these create a new
    *candidate version*;
  - `forge.commit(id, version)` (refused unless the verifier passed that version).

  Plus `rlm.query/batch` and `rlm.spawn(role, prompt)`, a synchronous child RLM agent with its own kernel, depth ≤ 2.
- **Task FSM** (data, journaled, resumable). For a task:
  - **Planner** (LLM): reads the verifier report and writes a repair plan.
  - **Implementer** (child RLM): applies edits through the facade, creating a candidate version.
  - **Reviewer** (LLM, **a different model**, separate context, adversarial): approve, or reject with findings.
  - **Verifier** (**objective, not an LLM**): M9 checks (A1, A3, A13, A16, …) on the candidate in fresh Worlds.
  - A failed review or verification loops back to the Implementer with the findings, up to `max_rounds`.
  - Commit happens only when both pass.
  - Generation and verification never share an agent; the verifier is the M9 contract.
- **Long-horizon manager loop:** `manage(task_ids)` iterates with a goal, per-task FSM state in the journal,
  checkpoint/resume, and the economics book (G-6) for cost and yield.
  - Lessons learned (e.g. "approval properties need event-grounded checks") become harness memories **only after**
    the measured evaluation shows they help on held-out tasks of the family.

## 5. Acceptance (tests)

- **Kernel:**
  - cells, await, the persistent namespace, output capture and caps;
  - deadline → restore;
  - replay restore with equal digests, and a mismatch refused;
  - the host bridge;
  - a scrubbed environment.
- **`RLMPolicy`** on the toy helpdesk with a scripted model:
  - code cells perform the gold actions → reward 1.0;
  - one World step per `world.call`;
  - `rlm.batch` sub-calls are recorded with their parent;
  - the output cap forces variable use (`OBS` holds the full results);
  - a goal continuation fires on a premature stop;
  - stalls are detected;
  - a mid-cell checkpoint resumes to the same final World fingerprint and reward;
  - compaction keeps kernel state;
  - train rollout records are complete with a token engine.
- **Harness:** a measured accept, a reject on regression, rollback to a version, counters.
- **Manager:**
  - on the toy forged task with B.15 re-introduced, the FSM finds the hack (A16), the implementer adds the in-episode
    approval check, the verifier passes (A1, A16 = 0 hacks), and the reviewer approves → commit;
  - a reviewer rejection loops back with findings;
  - the FSM state resumes from the journal.
