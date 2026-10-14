# Long-horizon RLM harness, study 03: where Crucible's current RLM harness breaks at thousands of steps

- **Question.** Where does Crucible's RLM harness (`RLMPolicy` in the Docker kernel, `RLMHarness`,
  `rlm_world_rollout`) break, or get slow, on very long episodes (thousands of World steps, about 1M tokens)? What does
  a new very-long-horizon RLM harness need?
- **Run:** 2026-10-13, on a CPU-only Mac (14 cores, 24 GB) with local Docker, against Crucible HEAD 775f30f. No paid
  API and no GPU. Nothing under `crucible/src`, `packs` or `tests` was edited. Every script, log and run directory is in
  `scratchpad/lh/stress/` (`stress.py`, `replay.py`, `probes.py`, `probe_overflow.py`, `analyze.py`, `qwen_run.py`,
  `runs/`). Every RLM run used the **Docker kernel** (`RLMConfig.kernel="docker"`, `python:3.12-slim`). The host
  kernel was not needed. No RLM container was left behind.
- **Method.** The episodes ran through the M10 `Runner` with `rollout_mode="train"`, as a training rollout would.
  - **Scripted model.** An `EngineLLM` over `HFChatCodec` with the **real Qwen2.5 tokenizer**, so it has exact TITO and
    real token counts. It emits correct RLM cells that replay the backlog's `diligent` plan, 4 World calls per cell,
    each result printed to 240 characters. The plan reads the policy, every listing page, every ticket and order, and
    re-reads the policy every 40 tickets.
  - **Lookups and sub-calls.** Every 40 cells it runs a lookup cell: it searches `OBS`, `INBOX` and `HISTORY` for a
    ticket handled long ago, and searches `HISTORY` for its own earlier reasoning text. Every 25 cells it makes an
    `rlm.query` sub-call. It writes a summary at every compaction. Each reply carries a short "thought" before the tool
    call.
  - **Measurements.** The script checks each answer from the next prompt, so retrieval is measured, not assumed.
    Instruments: per-call prompt and completion tokens, the wall-time gap between calls (cell plus World cost), kernel
    RSS (`/proc/<pid>/status` in the container), host RSS, journal and `state_dict` sizes and serialization time,
    checkpoint save time and size (`Checkpointer.save` wrapped), and the rollout record fed to the M12 `TorchTrainer`
    gate (`_gate` and `_items`).
  - **Native comparison.** The same plan through `LLMToolPolicy` with `ContextConfig` compaction, one tool call per
    model call.
  - **Horizons.** Backlog seed 0 with n = 8, 80, 315 and 1,200 tickets: 49, 446, 1,833 and 7,160 plan actions. The xl
    gate's 7,623 steps come from a different draw. Context budget 128k (the `RLMConfig` default), keep 3 recent turns,
    trigger 0.8. One extra run used a 32k budget.
- **Wall times** are from the scripted model, so they show harness and kernel overhead only. A served model adds
  seconds per call. Some runs shared the CPU with others, so treat their timings as approximate.

## 1. Measurement table

Per horizon. The RLM rows use the Docker kernel, 128k budget and 4 World calls per cell unless stated. "Prompt peak"
is the largest real Qwen-token prompt sent to the model. "Entered" counts each token once as it enters the root chain
(the deltas, plus the first prompt of each segment, plus completions).

| Measure | RLM n=8 | RLM n=80 | RLM n=315 | RLM n=1200 | RLM n=1200, 32k budget | Native n=1200 |
|---|---|---|---|---|---|---|
| World steps / reward | 50 / 1.0 | 447 / 1.0 | 1,834 / 1.0 | 7,161 / 1.0 | 7,161 / 1.0 | 7,161 / 1.0 |
| Model calls (turn / sub / compaction) | 16 / 0 / 0 | 122 / 4 / 0 | 491 / 19 / 1 | 1,907 / 76 / 6 | 1,907 / 76 / 26 | 7,161 / 0 / 15 |
| Wall s; ms per World step (harness only) | 2.0; 40 | 4.5; 10 | 16.3; 8.9 | 112; 15.6 | 87; 12.1 | 95; 13.3 |
| Median gap per work cell (4 World steps) | | | | 16 ms at the start, 65 ms at cell 1,500, 116 ms in the last 100 | | 3.6 ms → 25 ms per call |
| Prompt peak, root / compaction (real tokens) | 7.3k | 55.5k | **145.3k / 145.9k** | **147.5k / 148.1k** | **36.8k / 37.4k** | 102.5k / 103.0k |
| Prefill tokens, all calls | 70k | 3.5M | 30.9M | 139M | 37.9M | 378M |
| `trajectory_tokens` field vs entered | 5.8k vs 7.3k | 44k vs 56k | 180k vs 232k | **708k vs 920k** | 708k vs 987k | 1.05M vs 1.62M |
| Segments | 0 | 0 | 1 | 6 | 26 | 15 |
| Kernel RSS (MB) | 28.4 | 28.7 | 30.2 | 36.9 (41.3 peak) | 40.9 | n/a |
| Host RSS (MB) | 402 | 301 | 179 | 255 | 659 | 498 |
| Kernel journal (MB) / entries | 0.05 / 16 | 0.36 / 122 | 1.9 / 482 | **8.4 / 1,914** | 8.5 | n/a |
| Last checkpoint (MB) / save s | | 1.2 / 0.018 | 3.8 / 0.041 | **16.9 / 0.157** (133 MB written over 14 saves) | | 13.4 / 0.134 (106 MB) |
| `rollout.json` / `rlm.agent.json` (MB) | 0.05 / 0.05 | 0.42 / 0.36 | 1.7 / 1.9 | 6.9 / 8.4 | 7.2 / 8.5 | 10.1 / n/a |
| `valid_for_training`; trainer gate masked | yes; 0 | yes; 0 | yes; 0 | yes; 0 | yes; 0 | yes; 0 |
| Trainer segment items / forward tokens | 1 / 7.3k | 9 / 173k | 40 / 1.46M | **159 / 6.43M (7.0× entered)** | 179 / 2.43M | **16 / 1.62M (1.0×)** |
| Full kernel restore by replay | | 0.5 s (100 cells) | 2.5 s (493 cells) | **22.9 s (1,914 cells)** | | n/a |
| Trivial cell latency at the end (digest) | 0.7 ms | 1.2 ms | 4.8 ms | **18.6 ms** | | n/a |
| Lookup cell (OBS + `str(HISTORY)` scan) | | 3–8 ms | 8–9 ms | 4 ms at cell 40 → 29 ms at cell 1,840 | | n/a |

**Synthetic journals.** A bootstrap cell, then N cells each reading one document of S characters, replayed into a fresh
Docker kernel (`probes.py`):

| Cells | Observations | Journal | Replay | Trivial cell after | Kernel max RSS |
|---|---|---|---|---|---|
| 1,000 | 0.2M tokens (800 chars each) | 1.1 MB | 2.5 s | 2.3 ms | 31 MB |
| 5,000 | **1M tokens** | 5.5 MB | **26.8 s** | 7.6 ms | 49 MB |
| 5,000 | 5M tokens (4,000 chars each) | 21.5 MB | **66.1 s** | 22.7 ms | 109 MB |

**Variants on the same world** (details in §3):

| Run | Result |
|---|---|
| Kernel killed between cells (n=315, call 300) | The next cell found it dead. Restore by replay took 1.3 s (289 cells). The cell's code never ran, and the model was told "the cell did not finish in time". Reward 1.0, not masked. |
| Container removed at call 1,500 (n=1200, checkpoints every 500) | a1 masked `env_container_lost`. a2 resumed from step 5,000; the replay of 1,331 cells took **18.1 s**. a2 then finished the episode and was **masked `capture_incomplete` (prefix_break at the first resumed call)**. a3 restarted from step 0. |
| Engine failure at call 1,300 / 400 (native / RLM, n=315) | The same in both harnesses: a2 resumed from step 1,000 and finished, then was masked `capture_incomplete` (prefix_break). a3 restarted from step 0. |
| Container lifetime 6 s (n=315) | a1 lost its container at step 1,096 and a2 at step 1,821. a3 finished and was masked `capture_incomplete`. The episode never yielded a record. |
| `EpisodeLimits.timeout_s=5` (n=315) | Stopped at step 928, truncated, scored 0.517. The model was never told about the deadline. |
| Polling a pending ticket 3× in one cell, every 150 cells (n=315) | Three stalls spread over 450 cells ended the episode `too_many_errors` at step 1,698 of 1,834 (reward 0.908). |
| Sub-call every 5 cells (n=1200) | 256 sub-calls served, then **192 refused** for the last ~40% of the episode. 519 trainer items, 18.9M forward tokens (19.7× entered). |
| Engine window = budget (16k), RLM | The root call overflowed. The episode was **masked `agent_run_error`, retried 3×** (documented as scored). |
| Engine refuses only the compaction call (16k), RLM | The compaction prompt (18.9k) overflowed. **Masked `agent_run_error`, retried 3×.** |
| Qwen2.5-0.5B-Instruct, real model (MPS), n=4, k=5 | 5/5 episodes ended after 3 turns with 0 World calls and 0 cells. The model called `tickets.issue_refund` directly 9 times instead of `python` (10 named-tool calls in all). Records valid, reward 0. |

## 2. Break points, in the order an episode meets them

### B1. The prompt is not bounded: the RLM compaction trigger counts chars/4 (defect)

- **Where:** `RLMPolicy._maybe_compact` (`actors/rlm/policy.py:580`) compares `approx_tokens(self.messages)` with
  `trigger × budget`.
- **Native already fixed this.** `LLMToolPolicy` uses `context_tokens()`: the provider's measured prompt plus an
  estimate of the rest (`actors/policy.py:188-195`). That was the M2 audit fix ("chars/4 alone undercounted code ...
  so compaction fired after the provider had already refused the prompt"). The RLM never adopted it.
- **Measured.** RLM cells are JSON-escaped code with many digits. The Qwen tokenizer gives **1.42 real tokens per
  approx token** on them (stable from call 50 to call 1,900).
  - With the default 128k budget, every RLM segment peaks at **145–148k real tokens**: 6 of 6 segments at n=1200, and
    the first at n=315, about 940 World steps in.
  - Native stays at 102.5k.
  - With a 32k budget (the window of Qwen2.5-0.5B), **204 root calls and all 26 compaction calls exceed 32,768 real
    tokens.**
- **Consequence.** Every long RLM episode on a model whose window equals the budget overflows once per segment. Through
  B2 the overflow is masked, not scored.
- **Compaction makes it worse.** The compaction call is always the largest prompt of a segment (context +
  `COMPACT`), and it is sent only after the context has already passed the trigger.

### B2. A context overflow masks the episode instead of scoring it (defect)

- **Root call.** `_drive` catches `ContextOverflow` and sets `stop="context_overflow"`. It also sets
  `self.error = str(e)` with `failure_kind` None (`rlm/policy.py:463-466`). `run_episode` turns every policy with an
  `error` into a failure `{"kind": None}` (`actors/policy.py:625`). `_masks` masks any kind outside
  `POLICY_OUTCOME_KINDS` (`:676-681`), so the episode is masked `agent_run_error` "episode masked" and retried.
  - Measured: 3 attempts, each a full re-run.
  - Native returns `Done` without setting `error` (`actors/policy.py:377-379`), so the episode is scored and truncated,
    as `ContextOverflow`'s own docstring requires (`actors/llm.py:98-102`).
- **Compaction call.** `_maybe_compact` calls `llm.complete` with no `try` (`rlm/policy.py:586`). An overflow or a
  provider fault in the summary call escapes the driver, and `act()` labels it `agent_run_error` (`:227`).
  - Measured: masked and retried 3×.
  - A provider outage in the summary call should mask as `provider_unavailable`. An overflow should be scored.
- **Why it matters.** With B1 this decides the outcome. On a 128k model, each RLM episode past about 900 steps is
  masked, retried and masked again. Every long-horizon failure disappears from the data, which is what the
  `ContextOverflow` contract exists to prevent.

### B3. Resume never yields a trainable record, in either harness (defect, Crucible-wide)

- **The intent.** `runtime/checkpoint.py` says a crashed long episode resumes from its last checkpoint. "The record
  says where it resumed (`resumed_from`), so a trainer can mask the resumed segment."
- **What happens.**
  - The resumed policy keeps its segment number, and the new `EngineLLM` holds no conversation. So the first call
    after the resume re-renders the whole prompt, which does not extend the stored `_last_full`.
  - `runtime.record.build` marks it `prefix_break`. `Turn.missing()` counts a prefix break as an incomplete record
    (`core/schema/rollout.py:99`). The runner masks the whole rollout `crucible:capture_incomplete` at collect
    (`runtime/run.py:275-276`).
  - Because the mask comes after the episode, the next attempt is not resumable (`NativeHarness._resumable` only
    resumes a mid-episode mask), so **it restarts from step 0**.
- **Measured:** 4 of 4 resume scenarios.
  - RLM: container lost at step 5,000 of 7,161, and at step 1,441.
  - RLM: engine failure.
  - Native: engine failure.
  - The resumed attempt always finished the episode (reward computable) and was then masked.
- **Needed:** on resume, the policy opens a new segment (strategy `"resume"`), or only the resumed segment is masked.

### B4. Kernel state is durable only through full replay: O(cells) to O(cells²)

- **Restore is a full replay.** Timeout restore, kernel death and checkpoint resume all run `Kernel.replay`
  (`rlm/kernel.py:697-736`). It starts a fresh kernel and re-executes every journaled cell, answering each host request
  from the record.
- **Each replayed cell gets slower.** After every cell the kernel computes `digest()` (`rlm/repl.py:318-332`). It
  builds `repr()` of every plain global, including all of `OBS` and `HISTORY`, then truncates it. The cost of a cell
  therefore grows with everything the episode has seen.
- **Measured replay times:**
  - The n=1200 journal (1,914 cells): 500 cells in 2.8 s, 1,000 in 7.2 s, 1,500 in 14.6 s, 1,914 in 22.9 s
    (superlinear).
  - Synthetic 5,000 cells with 1M tokens of observations: 26.8 s. With 5M tokens: 66 s.
  - Checkpoint resume at step 5,000: 18 s before the first new call.
  - A timeout restore late in a 1M-token episode pays the same price (about 30–70 s) while the episode stalls.
- **Live cost:** a trivial cell costs 0.7 ms at the start and 18.6 ms at the end of n=1200. A work cell's gap grows
  16 → 116 ms. Most of that is the digest and the growing namespace.
- **Replay is exact only for deterministic cells.**
  - A cell that stores `time.time()` or `uuid4()` in a global fails the strict replay (`SnapshotMismatch` after cell
    1; `probes.py`). It then falls back to a lenient replay with 2 divergences.
  - Real models use the clock and random ids. Each restore in a long episode is then inexact, and the policy
    re-runs every earlier cell's side effects inside the kernel (files in `/tmp` included).
  - World actions are never re-issued: their replies come from the journal.
- **The digest is blind past 100k characters per variable.** `h.update(repr(v)...[:100_000])`. Changing `OBS[-1]` in a
  206k-character `OBS` leaves the digest unchanged; changing `OBS[0]` changes it (`probes.py`). In a long episode the
  integrity check covers only the first ~100 observations, but costs a full `repr` of all of them on every cell.

### B5. Checkpoints and evidence are O(episode), rewritten every time

- **What a checkpoint holds.** `RLMPolicy.state_dict` (`rlm/policy.py:646-653`) holds every message, **every model
  call with its token ids** (8.4 MB at n=1200) and the **whole kernel journal** (8.3 MB).
  - The journal holds a second copy of every tool result (the replies: 4.8 MB, a duplicate of `OBS` and the trace).
  - It also holds a third copy of every compacted transcript: each `_archive({...!r})` is a cell whose code is the
    dropped turns as a Python literal (6 cells, 2.3 MB, 28% of the journal).
- **Cost.** The checkpoint docstring promises "its current segment's context". In fact each save serializes the whole
  episode: 16.9 MB per save at step 7,000, and 133 MB written over the episode.
  - The same journal is written again into `rlm.agent.json` (8.4 MB), and the token records into `rollout.json`.
  - Native: 13.4 MB per save, 106 MB in all. That is the same O(N) shape, from `model_calls`.
- **Extrapolation.** At about 1M observation tokens per episode and a checkpoint every 500 steps, both harnesses write
  O(N²/every) bytes. Save time grows linearly: 0.157 s at 7k steps.

### B6. Episode-wide counters that never reset turn honest long episodes into failures

- **Stalls.** `counters["stalls"]` accumulates over the whole episode and `max_stalls=3` ends it
  (`rlm/policy.py:560-566`). Polling a pending ticket (three identical `get_ticket` results in one cell) is the honest
  way to wait for a customer's reply.
  - Measured: three polls 150 cells apart ended the episode `too_many_errors` at step 1,698 of 1,834.
  - A stall at hour 1 and another at hour 5 count as one loop.
- **Continuations.** `max_continuations=2` per episode (`:485-491`). The third reply without code ends the episode
  as `agent_done`.
  - Measured with the real model: 5/5 Qwen2.5-0.5B episodes ended after 3 turns.
  - Across thousands of turns, any model that sometimes replies in prose will end its episode early.
- **Turns.** `RLMConfig.max_turns=500` (`runtime/harness/rlm.py:54`). With 4 World calls per cell the default cap is
  about 2,000 steps. The 7,161-step episode needed 1,907 turns. One World call per cell, the common habit of models,
  caps the episode at 500 steps. Every run here raised it.
- **Sub-calls.** `max_sub_calls=256` and `max_sub_tokens=2M` are per episode (`rlm/policy.py:325-338`) and do not
  scale with the horizon.
  - Measured: with a sub-call every 5 cells, 192 calls were refused in the last ~40% of the episode.
  - Each refusal is a cell error the model sees. The budget is shown nowhere in the prompt, so the model learns of it
    only by failing.

### B7. Sub-calls fragment the trainer's segment packing (inefficiency, grows with horizon)

- **The mechanism.** `train/bridges/common.py:flatten` (`:84-91`) chains turns in record order. An interleaved
  `kind="sub"` turn starts its own sequence, so the next root turn cannot extend the previous root chain. It opens a
  new `TrainSeq` with its full prompt.
- **Measured.**
  - Native n=1200: 16 items, 1.62M forward tokens, equal to the entered tokens.
  - RLM n=1200 with 76 sub-calls: 159 items and **6.43M forward tokens (7.0×)**.
  - RLM with 256 sub-calls: 519 items, **18.9M (19.7×)**.
  - Each item restarts a ~12k–145k-token prefix.
- **Needed.** Sub turns go into their own list (they are tree nodes already), and the root chain is packed per segment
  regardless of where they fall.

### B8. Token accounting undercounts

- **What is counted.** `RLMPolicy.trajectory_tokens` mixes chars/4 estimates for observations and the task with real
  completion counts.
- **What is left out:** INBOX announcements, continuations, stall notices, the rebuilt context after compaction, and
  sub-calls.
- **Measured:** 708k reported against 920k entered root-chain tokens at n=1200 (−23%). Sub-calls add 14.5k, or 54k in
  the sub-heavy run.
- **Native also undercounts:** 1.05M against 1.62M (−35%), because chars/4 is applied to observations. The xl gate's
  "1,114,725 trajectory tokens" is an estimate in the same units.
- **Segment metadata uses the same estimate.** `context_before` is in approx units: 102.4k recorded against 146–148k
  real.

### B9. The container has a fixed 6-hour life

- **The limit.** `DockerLauncher` runs `sleep <lifetime_s>`, with `DEFAULT_LIFETIME_S = 6 h` (`rlm/kernel.py:286,
  437`). The container ends by itself at 6 h whatever the episode is doing, and expired containers are reaped.
- **Scale.** A 7,000-step episode at 2–3 s per real model call (a 100k-token prefill each, about 2,000 cells) runs
  past it.
- **Measured** with a 6 s lifetime: each loss masked `env_container_lost`. Each resume replayed the journal into a new
  container, and the final attempt was masked by B3.
- **No episode clock for the model.** `EpisodeLimits.timeout_s` truncates and scores (0.517 at step 928) without
  telling the model the time left or giving it a wrap-up turn.

### B10. Smaller harness defects found on the way

- **Kernel death between cells.** The next cell reports "the cell did not finish in time; the kernel state was
  restored" (`rlm/policy.py:531-533`), but that cell's code never ran. The model cannot tell a timeout from a kernel
  crash, or know that its last cell did nothing.
- **A named-tool call gets a generic continuation.** A reply that calls a World tool by name (what 0.5B models do, and
  what the tool index invites) is treated as "no code". It gets the generic `CONTINUE` notice, not "the only tool is
  `python`; call `world.call(...)` inside it", and it burns one of the episode's 2 continuations (B6).
- **The `Live names:` line is unbounded.** After every compaction it lists every global (`rlm/policy.py:604-605`), so
  a model that creates per-ticket variables puts thousands of names into the post-compaction prompt.
- **HISTORY outputs are capped.** HISTORY stores the cell output as the model saw it, already cut to 8,192
  characters, so the archive is not verbatim for large outputs. `OBS` still holds the full tool results.

## 3. What the model can and cannot get back from its past

Measured by the lookup cells at n=1200: 38 lookups, the oldest about 4,700 steps back.

| Item | Retrievable? | Evidence |
|---|---|---|
| Every tool result, in full, with its arguments | yes (`OBS`) | 38/38 lookups found the ticket's results |
| Every message from a person | yes (`INBOX`, 803 entries) | |
| Archived cells (code + output as shown) and every summary | yes (`HISTORY[i]`) | 32/32 lookups after the first compaction found the ticket's cells |
| Variables the model kept (`DONE_IDX`) | yes, also after a resume | the resumed script re-derived its position from `DONE_IDX` in the replayed context |
| **Its own reasoning text that came with a tool call** | **no** | 0/38. `_archive` keeps assistant text only when the message has no tool call (`rlm/policy.py:639`) |
| **Sub-LLM answers** not assigned to a variable | **no** | they exist only in the host's `model_calls` |
| **`rlm.note()` notes** | **no** | write-only, stored on the host (`self.notes`) |
| Cell output past the 8,192-character display cap | only through `OBS` / variables | HISTORY stores the capped text |
| Exact earlier values after an inexact restore | **no** | a lenient replay recomputes clock and random values |
| A searchable index | **no** | only linear scans; `str(HISTORY)` plus an `OBS` scan grew from 4 to 29 ms per lookup at 6.5k observations and 6 segments, and grows linearly |

## 4. The real-model run (Qwen2.5-0.5B-Instruct, `HFGenerateEngine` on MPS, Docker kernel)

- **Setup:** n = 4 tickets, budget 3k–6k, 5 samples (k = 1 and k = 4).
- **Result:** every episode ended after 3 turns.
  - In 12 replies the model made 10 tool calls naming a World tool directly (`tickets.issue_refund` 9×,
    `tickets.list_tickets` 1×) and wrote no `python` call. The other 2 replies had no tool call.
  - The harness answered with the generic continuation twice, then ended the episode `agent_done`: 0 cells, 0 World
    calls, reward 0.
  - Records were train-valid, with no prefix breaks. Per call: 1.2–1.9 s for about 1k-token prompts.
- **Interpretation.** The episode-wide continuation budget and the generic notice end an untrained model's episode
  before it ever writes code. This matches the M14 live smoke, where a 4B model never made a World call, and the
  "untrained models under-use the scaffold" finding. For RL from such a policy the harness has to steer, not stop.
  Measuring the long-horizon behaviour of a real model needs a bigger model on a GPU, which was not run here.
- **Other local models:** the Qwen3 checkpoints are tokenizer-only on this machine, so only Qwen2.5-0.5B could run.

## 5. Gap list: what a very-long-horizon RLM harness needs

| # | Requirement | Current state (evidence) | Gap |
|---|---|---|---|
| G1 | **Prompt bounded by construction** | The trigger counts chars/4. Peaks of 1.42× the budget (B1). The compaction call is the largest prompt. An overflow in it escapes (B2). `Live names` is unbounded | Count the engine's real tokens (`context_tokens()`). Reserve headroom for the summary call, or compact from a bounded view: the summary prompt is the summary state plus the last K turns, never the whole context. Bound every injected block (names, announcements, notices). Have a hard pre-send check that compacts again or truncates instead of overflowing |
| G2 | **Durable kernel state with ~O(1) restore** | Restore is a full replay: 23 s at 1.9k cells, 27–66 s at 5k cells / 1–5M tokens. Exact only for deterministic cells. The digest is O(namespace) per cell and blind past 100k characters (B4) | Periodic kernel snapshots (pickle / `dill` of the namespace, or a CRIU / container checkpoint) plus replay of the tail since the last snapshot. A content hash per variable, incremental and complete, instead of a truncated `repr`. Time and random values recorded as host requests so replays are exact |
| G3 | **Observations offloaded, not copied** | Each tool result is stored 3–4× (`OBS`, the journal's replies, the trace, checkpoints), plus each transcript as an `_archive` literal (28% of the journal) (B5) | One content-addressed store (World trace or blob store). `OBS` and `HISTORY` hold references, and the journal records hashes. A checkpoint holds the current segment plus a pointer to the store, as the checkpoint docstring already promises |
| G4 | **Searchable memory / history API** | Linear scans of Python lists. No index. Reasoning text, sub-answers and notes are not retrievable (§3) | `HISTORY.search(query, k)` / `OBS.find(**args)` / `MEMORY.get/put`, backed by an index kept outside the namespace. Archive the assistant text that comes with tool calls, the sub-call answers and the notes. Keep the full (uncapped) output by reference |
| G5 | **Persistent files and notes** | `/tmp` tmpfs (256 MB) inside a container that is lost on expiry or crash, rebuilt only by replay. `rlm.note` is host-side and write-only | A per-episode workspace volume included in snapshots and checkpoints. Notes are readable (`rlm.notes()`) and shown in the summary prompt |
| G6 | **Sub-agent recursion with budgets** | In the World harness, sub-calls are plain completions (depth 1, no tools, no kernel). Spawned child RLMs with `TreeBudget` exist only in `accept/repair.py` (`RLMAgent`), with no World. Budgets are per episode and hidden (B6). Sub turns fragment training packing 7–20× (B7) | `rlm.spawn(role, task, tools=..., budget=...)` with World access through the parent's step accounting, its own kernel and its own token record. Budgets scale with the horizon (per segment, or per N steps) and are shown in the prompt. Sub sequences are packed apart from the root chain |
| G7 | **Progress tracking** | Only model-side variables (`DONE_IDX`). Dense Φ exists host-side (`verify.progress`) but the model cannot see it | A harness-held progress ledger (todo / done / pending ids) that survives compaction and resume, is injected into every post-compaction prompt, and is checked against Φ for drift |
| G8 | **Checkpoint / resume that trains** | Every resume is masked `capture_incomplete` (prefix_break) and the next attempt restarts at step 0, in both harnesses (B3). Each checkpoint is O(episode) (B5). Resume costs O(cells²) replay (B4) | The resume opens a new segment. Checkpoints are incremental (an append-only journal plus a snapshot id). The resume restores the kernel snapshot. The engine conversation re-prefills one segment |
| G9 | **Episode-scaled limits** | Stalls and continuations are counted per episode. `max_turns=500` allows about 2,000 steps. A 6 h container life. The deadline is invisible (B6, B9) | Windowed stall detection (stalls per last N cells) that tells waiting from looping (a poll with a backoff is not a stall). Continuations per segment, with a specific notice for a wrong tool name. Turn and lifetime limits derived from the task's horizon. Time left shown to the model with a wrap-up turn |
| G10 | **Training records per segment with honest credit** | Records are train-valid (0 prefix breaks without a resume) and Φ credit telescopes. But an overflow is masked, not scored (B2). A resumed rollout is never trainable. Sub turns break packing. `trajectory_tokens` is low by 23–35% (B8). No cross-segment GAE | Score overflows. Segment-level masks instead of rollout-level ones. Count entered tokens from the engine's ids. Per-segment credit from Φ. Summary calls trained only when the summary was good (still open in M14) |
| G11 | **Observability** | `rlm.agent.json` is one 8.4 MB blob written at the end. No per-cell timing, kernel RSS, digest cost or replay time is recorded. A kill is reported as a timeout | Streamed per-cell events (time, host requests, bytes, RSS, digest ms), per-segment summaries, restore events with their kind (timeout / kill / resume) and cost, and budget dashboards (tokens, sub-calls, time left) |
| G12 | **Scaffold adoption by untrained models** | 5/5 Qwen2.5-0.5B episodes ended after 3 turns without writing code (§4) | Accept a named World tool call by translating it to `world.call` (or answer with a precise correction). Never end the episode on prose before a minimum turn budget. A few-shot first cell in the system prompt |

## 6. Reproduce

From `crucible/`, with
`PYTHONPATH=src:packs/coding/src:packs/_toy/src:packs/finance_ap/src:packs/_template/src` and `.venv/bin/python`. The
scripts are in `scratchpad/lh/stress/`. `run.sh`, `runr.sh`, `runp.sh`, `runa.sh` and `runq.sh` wrap them with these
paths.

```
stress.py --mode rlm|native --n {8,80,315,1200} [--budget 32000] [--poll-every 150] [--sub-every 5]
          [--crash-kernel-at K] [--rm-container-at K] [--raise-at K] [--lifetime S] [--timeout S] --out runs/<name>
replay.py "<run>/runner/r/samples/*/a1/rlm.agent.json" docker 500,1000,1500 out.json
probes.py runs/probes.json          # digest blindness, nondeterministic replay, synthetic 1M/5M-token journals
probe_overflow.py                   # overflow classification (root / compaction)
analyze.py runs/<name>              # token accounting, segment chains, TorchTrainer _gate/_items
N=4 K=4 BUDGET=3000 qwen_run.py runs/qwen_k4
```
