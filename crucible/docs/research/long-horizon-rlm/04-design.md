# The very-long-horizon RLM harness — design (M16)

Goal: an RLM harness that runs World episodes of thousands of steps and 1–10M tokens, keeps the model's work in a
persistent Python kernel, loses nothing at a compaction, bounds every cost, and produces training records that stay
correct across compactions and child agents.

Inputs: `01-frameworks.md` §7 (requirements R1–R20), `02-papers-articles.md` §9–§10 (H1–H16, P1–P12),
`03-crucible-stress.md` (when it lands). The requirement ids below refer to those lists.

Rules that hold everywhere (Crucible-wide, unchanged):
- Reuse before writing (DEPEND / PORT with licence and file:line / PATTERN). Ports from nano-rlm and Prime Agent are MIT;
  research-environments generators are Apache-2.0. Name the source line in a comment.
- Infra faults mask (never score 0); policy faults are scored. Every new stop has a typed kind.
- Grader state and anything the grader reads stay outside the kernel's write reach. In-episode feedback uses only what
  the policy itself observes (never Φ, never a hidden check).
- Every number the harness reports has a stated source (`usage` vs `estimate`).
- No paid or GPU run without the user's approval. Docker kernels on the local machine are fine.

## 1. Architecture

```
            ┌──────────────── RLMPolicy (root, depth 0) ─────────────────┐
 World ◄──► │ act() bridge ── AgentLoop ── ContextManager (context.py)   │
            │                    │  ▲          Compactor, meter, rebuild │
            │                    ▼  │                                    │
            │               Kernel (kernel.py / repl.py, Docker)         │
            │                 TASK TOOLS OBS INBOX CONTEXT history out() │
            │                    │ host requests                         │
            │   Ledger (ledger.py, on disk) ◄── every event, full text   │
            │   TreeSupervisor + ChildRegistry (tree.py)                 │
            │        └─ child AgentLoop (depth 1..2): own kernel, own    │
            │           ledger dir, own context, World only if granted   │
            └────────────────────────────────────────────────────────────┘
 records: model_calls (typed edges) → runtime/record.py → RolloutRecord (segments per agent)
```

- **L1 prompt:** system, task verbatim, the summary, a python-state row, a boundary marker, the recent tail.
- **L2 kernel:** variables. They survive compaction because the kernel keeps running.
- **L3 ledger:** append-only, on disk, outside the kernel; every message, the *full* cell output, every World result,
  every sub-call and child event. The kernel reads it through `history` host requests.

## 2. Modules and owners (phase 1 runs in parallel; phase 2 integrates)

| Module | Owner | Phase |
|---|---|---|
| `actors/rlm/ledger.py` (new) + kernel-side `HISTORY_API` string | A | 1 |
| `actors/rlm/context.py` (new), `actors/rlm/stall.py` (new) | B | 1 |
| `actors/rlm/tree.py` (new) | C | 1 |
| `runtime/record.py`, `core/schema/rollout.py`, the trainer's sequence keying | D | 1 |
| `bench/longctx/` (new), toy `longhorizon/ladder.py`, `runtime/inject.py` scripted RLM models, `scripts/lh_ladder.py` | E | 1 (runs after 2) |
| `actors/rlm/kernel.py`, `actors/rlm/repl.py`, `pyproject.toml` (dill) | G | 1 |
| `actors/rlm/agent.py` (new core loop), `actors/rlm/policy.py`, `runtime/harness/rlm.py`, `train/world_env.py` | F | 2 |
| `DESIGN.md`, `docs/modules/M16-long-horizon-rlm.md`, `ARCHITECTURE.md` row | me | 3 |

Each owner writes only its files plus its own tests `tests/unit/test_m16_<owner>_*.py`. Nobody edits another owner's
file; a needed change goes in the owner's report as a request.

## 3. Interfaces (fixed; owners implement exactly these names)

### 3.1 `ledger.py` (A) — R1, H1

```python
class Ledger:
    def __init__(self, root: Path, *, name: str = "root"): ...   # creates root/<name>/events.jsonl (+ blobs/)
    @classmethod
    def open(cls, root: Path, *, name: str = "root") -> "Ledger": ...  # reopen after a crash: a torn last line is dropped
    def append(self, kind: str, **fields: Any) -> int          # stable index, monotonic; text fields > 64 KiB go to blobs/<i>.txt
    def open_window(self, reason: str, **meta: Any) -> int     # reason: start | compaction | compaction_failed | restore | rerender | resume
    @property
    def window(self) -> int
    def get(self, i: int) -> dict                               # header + text fields (each capped at 65,536 chars, with "more")
    def read(self, i: int, field: str = "text", cursor: int = 0, max_chars: int = 65_536) -> dict  # {"text","next","total"}
    def search(self, pattern: str, *, regex: bool = False, kinds: list[str] | None = None, window: int | None = None,
               limit: int = 20, context_chars: int = 200) -> list[dict]   # [{i, kind, window, step, snippet}], newest first
    def windows(self) -> list[dict]                             # [{w, reason, start_index, end_index, meta}]
    def messages(self, *, window: int | None = None, start: int = 0, end: int | None = None,
                 kinds: list[str] | None = None, preview_chars: int = 300) -> list[dict]
    def user_messages(self) -> list[dict]                       # every user/supervisor/INBOX message, verbatim
    def since(self, step: int) -> list[dict]                    # previews of events at World step >= step
    def child(self, name: str) -> "Ledger"                      # root/<name>/<child>/..., same API
    def handle(self, request: dict) -> Any                      # serves kernel host requests "history.<op>"; replies ≤ 256 KiB
    def stats(self) -> dict                                     # events, bytes, windows, blobs
    def close(self) -> None
```

- Event kinds (field `kind`): `system user assistant tool world_call world_send inbox sub_call summary
  compaction_attempt notice window child_spawn child_event child_return snapshot restore`. Every event carries `i`,
  `kind`, `window`, `step` (World step), `t` (monotonic s), `agent`.
- `tool` events hold `text` = the **full** cell output (stdout, stderr, result, error, up to the kernel's 1 MiB per
  stream) and `shown_chars` = what reached the context. The context message names the index: `[full output: OUT(<i>)]`.
- `search` must be bounded: ≤ 2 s wall-clock and ≤ 64 MiB scanned per call. A pathological regex cannot hang the
  host (run regex searches in a worker process with a timeout, or another bounded method; substring search is the
  default).
- Writes are streamed (flushed per event) so a crash loses at most the event being written.
- `HISTORY_API`: a Python source string (the kernel half), appended to the RLM bootstrap by F:
  `history.get(i)`, `history.read(i, cursor=0, max_chars=65536)`, `history.search(pattern, regex=False, kinds=None,
  window=None, limit=20)`, `history.windows()`, `history.messages(window=-1, start=0, end=None, kinds=None)`,
  `history.user_messages()`, `history.since(step)`, `history.child(name)` → an object with the same API, and
  `OUT(i, cursor=0, max_chars=65536)` = `history.read(i)`. All are `async` (they are host requests): `await
  history.search("refund")`. Each wrapper sends `host("history.<op>", **kw)`.

### 3.2 `context.py` (B) — R2, R3, R4, R16, H3, H15

```python
@dataclass(frozen=True)
class ContextSpec:
    window_tokens: int = 128_000
    max_out: int = 2_000               # the turn's max_tokens
    seq_len: int | None = None         # the trainer's sequence length: threshold ≤ seq_len − max_out
    reserve: int = 16_384
    tail_tokens: int = 20_000          # recent tail kept verbatim after a compaction (0 = none)
    summary_max_tokens: int | None = None   # None: clamp(window // 64, 2_000, 16_000)
    max_compaction_attempts: int = 5
    max_compactions: int | None = None # None = unbounded (but counted)
    mode: str = "auto"                 # auto | action (policy compact() + hard threshold) | none
    footer_from: float = 0.8           # usage fraction from which every observation carries the <context_usage> footer
    tool_chars: int = 2_000            # per tool output in the summarizer's view (head) ...
    tool_tail: int = 500               # ... plus this many tail chars
    def threshold(self) -> int         # max(W//2, min(int(0.95·W), W − max_out − reserve)), then ≤ seq_len − max_out

class ContextMeter:                    # usage-anchored accounting (PA SE/compaction.rs:39-295, nano-rlm engine.py:1133-1146)
    def anchor(self, usage: dict, n_messages: int) -> None   # after an accepted call: prompt + completion tokens
    def tokens(self, messages: list[dict]) -> int            # anchor + chars/4 of messages[n_anchor:]; estimate if no anchor
    def reset(self) -> None                                  # after a compaction: pre-compaction usage never re-triggers
    source: str                                              # "usage" | "estimate"

def is_overflow(exc: BaseException | None = None, *, status: int | None = None, text: str = "") -> bool  # 9 markers, 400/413, ContextOverflow
def is_silent_overflow(usage: dict, finish: str | None, window: int, text: str, tool_calls: list) -> bool  # PA overflow.rs:8-98

def summary_view(messages: list[dict], spec: ContextSpec) -> list[dict]   # tool outputs head+tail capped; old images
                                                                          # (keep 3) and reasoning blocks pruned
def hollow_middle(messages: list[dict], target_tokens: int, over: float = 1.25) -> list[dict]  # keep start + end, whole
                                                                          # tool exchanges; marker row where removed
def split_tail(messages: list[dict], tail_tokens: int) -> tuple[list[dict], list[dict]]  # cut only before an assistant
                                                                          # row; tool results stay paired
SUMMARY_PROMPT: str        # sections: goal and done-criteria; constraints and rules in force (with when each changed);
                           # every user/supervisor message verbatim; entities and ids touched; pending and waiting items
                           # with their triggers; errors and fixes; current hypothesis; next step (verbatim); variables
                           # worth remembering. States that the kernel keeps running and the ledger keeps everything.
UPDATE_PROMPT: str         # the same with <previous-summary>…</previous-summary> (PA update mode)
def python_state_row(entries: list[dict], *, lost: list[str] = (), max_names: int = 60, max_chars: int = 2_000) -> str
def boundary_marker(n: int, before: int, after: int, window: int) -> str   # names the history API and "state the gap
                                                                          # before relying on missing details"
def rebuild(system: dict, task_msg: dict, summary: str, state_row: str, boundary: str, tail: list[dict]) -> list[dict]
                           # framing: retained messages are authoritative; the summary may lag. No fabricated assistant turn.

@dataclass
class CompactionOutcome:
    ok: bool
    summary: str
    attempts: int
    failure: str | None        # None | "compaction_failed"
    input_tokens: int

class Compactor:
    async def run(self, llm, messages: list[dict], spec: ContextSpec, *, previous_summary: str | None,
                  on_call: Callable[[Any, int, bool | None, int], None],  # (response|exception, attempt, accepted, input_tokens)
                  sample_key: str | None, temperature: float) -> CompactionOutcome
    # ≤ max_compaction_attempts; an overflowing summary request → hollow_middle its input (×1.25 over-removal) and retry;
    # empty, tool-calling or reasoning-only replies are resampled; all attempts fail → CompactionOutcome(ok=False,
    # failure="compaction_failed"), which the loop turns into a clean typed stop (truncation, not error).

def footer(*, used: int, threshold: int, window: int, step: int, turns: int, budgets: dict[str, tuple[int, int]],
           elapsed_s: float | None, source: str) -> str      # "<context_usage>…</context_usage>" + budgets left (H15)
def wrap_up_notice(kind: str, left: int) -> str               # at 0.9 of an episode budget: finish, do not compact
```

`stall.py` (B) — R11, H9. Only signals the policy itself observes (never Φ):

```python
class StallDetector:
    def __init__(self, *, repeat_cells: int = 3, same_call_window: int = 10, same_call_hits: int = 6,
                 static_results_window: int = 100, max_cell_errors: int = 4, escalate_after: int = 3, stop_after: int = 6): ...
    def on_cell(self, code: str, ok: bool) -> None
    def on_world(self, tool: str, args: dict, result_digest: str) -> None
    def check(self) -> StallVerdict | None    # StallVerdict(signature, evidence, action: "notice" | "reflect" | "stop")
    def state_dict(self) -> dict ; def load_state_dict(self, d: dict) -> None
```

Signatures: the same cell `repeat_cells` times; the same `(tool, args)` ≥ `same_call_hits` of the last
`same_call_window`; ≤ 2 distinct World results over the last `static_results_window` World calls; `max_cell_errors`
failed cells in a row. Escalation: notice → after `escalate_after` verdicts in a row a forced reflection turn (the loop
asks the model to state what is wrong and change approach, recorded as a normal turn) → after `stop_after` a scored
stop `stalled`.

### 3.3 `tree.py` (C) — R7, R8, H6

```python
@dataclass(frozen=True)
class TreeLimits:
    max_total_new_tokens: int = 10_000_000   # completion tokens (+ summary calls) across the whole tree
    max_total_turns: int = 20_000            # model turns across the tree (summary calls excluded)
    max_spawns: int = 64
    max_depth: int = 1                       # 0 = no children; ≤ 3
    max_concurrent: int = 8                  # children running at once, tree-wide
    reserve_per_depth: int = 1               # slots held back for deeper levels so nesting cannot deadlock
    child_wall_s: float = 3_600.0
    child_max_turns: int = 2_000
    max_sub_calls: int = 4_096               # rlm.query / batch, tree-wide
    max_sub_tokens: int = 20_000_000
    max_events: int = 10_000                 # child-event inbox per parent
    max_event_bytes: int = 65_536

class BudgetExhausted(Exception):            # .kind: "budget:new_tokens" | "budget:turns" | "budget:spawns" | ...
    kind: str

class TreeSupervisor:
    def __init__(self, limits: TreeLimits): ...
    def charge_turn(self, depth: int) -> None              # raises BudgetExhausted
    def charge_tokens(self, depth: int, completion: int, prompt: int = 0, *, summary: bool = False) -> None
    def charge_sub(self, n: int) -> None ; def charge_sub_tokens(self, n: int) -> None
    def check_spawn(self, depth: int) -> None              # depth+1 ≤ max_depth, spawns left
    @contextlib.asynccontextmanager
    async def slot(self, depth: int)                       # per-depth reserved concurrency (nano-rlm supervisor.py:146-158,439-456)
    def usage(self) -> dict ; def state_dict(self) -> dict ; def load_state_dict(self, d: dict) -> None

class ChildRunner(Protocol):                               # implemented by F's AgentLoop; C tests with fakes
    name: str
    async def run(self) -> dict                            # {"status": done|failed|cancelled|budget, "answer": str, "error": str|None}
    def deliver(self, text: str, kind: str) -> None        # kind: "message" | "steer"
    def cancel(self) -> None
    def progress(self) -> dict                             # {"turns", "tokens", "last": str}

class ChildRegistry:
    def __init__(self, supervisor: TreeSupervisor, factory: Callable[..., ChildRunner], *, depth: int): ...
    async def spawn(self, brief: str, *, name: str | None = None, persistent: bool = True, world: str = "none",
                    max_turns: int | None = None) -> dict          # {"name", "status": "running"}
    async def result(self, name: str, *, wait_s: float = 0.0) -> dict   # running | done (answer ≤ 8,000 chars) | failed | cancelled
    async def send(self, name: str, text: str) -> dict ; async def steer(self, name: str, text: str) -> dict
    async def cancel(self, name: str) -> dict                  # cascades to grandchildren
    def watch(self, name: str, *, every_turns: int | None = None, every_tokens: int | None = None) -> dict
    def events(self) -> list[dict]                             # drained: child.completed (last 4 KiB) | child.failed | child.progress
    def list(self) -> list[dict]
    async def aclose(self) -> None                             # cancels every child, awaits them, closes kernels
    def handle(self, request: dict) -> Awaitable[Any]          # serves kernel host requests "agent.<op>"
```

`world` grants: `"none"` (default, least privilege: the child computes over its brief, files handed to it and
sub-calls) or `"shared"` (World calls go through the root's action queue under one lock: each is a World step attributed
to the child's model call). A child's kernel follows the root's isolation rule. Child usage is charged to the tree, never
to the parent's context meter.

Kernel-side API (`AGENT_API` string, C): `await rlm.agent.spawn(brief, name=None, persistent=True, world="none")`,
`await rlm.agent.result(name, wait_s=0)`, `send`, `steer`, `cancel`, `rlm.agent.watch(...)`, `await
rlm.agent.events()`, `await rlm.agent.list()`.

### 3.4 Kernel snapshots (G) — R12, H11

```python
@dataclass
class Snapshot:
    path: Path              # host-side blob, outside the kernel's reach; the host never unpickles it
    journal_len: int        # cells covered
    digest: str             # namespace digest at snapshot time
    lost: list[str]         # names that could not be serialized (reported to the model after a restore)
    bytes: int

class Kernel:
    async def snapshot(self, dest: Path) -> Snapshot          # dill in the kernel; ≤ 256 MiB total, ≤ 16 MiB per variable
    async def replay(self, journal, *, live=None, strict=True, snapshot: Snapshot | None = None) -> JournalEntry | None
                    # restore the blob, check the digest, replay journal[snapshot.journal_len:]; any failure → full replay
    async def describe(self) -> list[dict]                    # [{name, type, size}] for the python-state row
```

- `dill` (BSD-3) becomes a dependency. The Docker kernel has no network: the launcher pushes the pure-Python dill
  package into the container (as it pushes `repl.py`) and puts it on the kernel's path.
- A snapshot is taken every `snapshot_every` cells (F's config, default 200) and at each compaction. A restore that
  cannot reproduce the digest falls back to full journal replay, then to lenient replay (existing behaviour).
- The recovery notice after any kernel loss: "variables lost: …; never blindly repeat the interrupted cell".

### 3.5 Records (D) — R9, R10, H13

- `model_calls` entries gain (all optional; absent = old behaviour): `agent` (root = `""`, a child = its path
  `"scan"`, `"scan/sub2"`), `depth`, `edge` (`continuation | compaction_attempt | compaction | subagent_call |
  subagent_return | agent_message | subagent_cancel | sub`), `accepted` (compaction attempts), `trainable`, `window`
  (ledger window), `ledger_index`.
- `Segment` gains `agent: str = ""`, `boundary_reason: str | None`, `attempts: int | None`, `failed: bool = False`.
- `Turn` gains `agent: str = ""` and `edge: str | None`. A sequence chain is keyed by `(actor, agent, segment)`.
  Children are their own chains; they share their spawning root turn's advantage (like `kind="sub"` today).
- A rejected compaction attempt is recorded, not trained (`trainable=False`); an accepted one is trained with the
  episode's advantage.
- Weighting is per logical rollout: a test proves a rollout split into 1, 2 or 8 segments (and with children) gives every
  trainable token the same weight (`train/objectives.row_weights`, M12). An over-length row is masked and counted,
  never tail-cut.

### 3.6 The loop (F) — everything above wired

`actors/rlm/agent.py`: `AgentLoop` — one model loop over one kernel, one ledger, one `ContextManager`, one
`StallDetector`; used by the root `RLMPolicy` (with the World bridge) and by every child (`ChildRunner`). Per turn:

1. drain child events and INBOX; inject notices (bounded);
2. `meter.tokens(messages) ≥ spec.threshold()` (or the policy called `rlm.compact()`) → compact;
3. call the model; on overflow (raised or silent) → discard the reply, compact, retry (no turn charged);
4. anchor the meter to the call's usage; charge the tree;
5. run the cell; log the full output to the ledger; put the capped view (8,192 chars, head + tail, with `OUT(i)`) in the
   context; append the footer from 80 % usage;
6. stall check; budget notices at 0.9; stop on a typed kind.

New kernel names: `CONTEXT` (a handle for large inputs: `CONTEXT.info()`, `await CONTEXT.read(offset, n)`,
`await CONTEXT.search(pattern)`, backed by a host file; preview ≤ 500 chars in the task message), `history`, `OUT`,
`rlm.agent`, `rlm.compact(instructions="")`, `rlm.give_up(reason, evidence="")` (stop `agent_gave_up`, recorded;
scoring is the task's), `rlm.budget()` (what is left).

`RLMConfig` (the harness id) gains: `window_tokens`, `seq_len`, `tail_tokens`, `max_compaction_attempts`,
`max_compactions`, `context_mode`, `snapshot_every`, the `TreeLimits` fields, the stall thresholds, `max_turns`
(default derived from the token budget, thousands). Old fields keep working.

Stops (typed): `agent_done`, `agent_gave_up`, `compaction_failed` (truncation), `budget:<kind>` (truncation),
`stalled` (scored), `context_overflow` (only when compaction is off), `policy_error` (masked, infra kinds).

## 4. Validation (E, then the campaign)

- **Step ladder:** the toy backlog world at 100 / 500 / 1,000 / 2,500 / 5,000 World calls, milestone Φ per segment,
  run by scripted RLM models that keep *no hidden memory*: everything they know after a compaction comes from their
  visible context (the summary text, the state row, the tail) or from `history` / variables. Variants: a faithful
  summarizer and a forgetful one (drops the amendment / the awaiting ids). Measured: reward, lost-constraint rate,
  re-reads of `history` / `OUT` after compaction, compactions, tokens per segment, resume cost.
- **Long-context ladder:** oolong-style aggregation, MRCR-style multi-round retrieval and long-document QA (ported from
  research-environments, Apache-2.0), rungs 8k → 4M tokens, delivered as `CONTEXT` and through a World `docs` app, the
  answer submitted through a World tool and graded from World state.
- **Fault injection:** overflow at each attempt, kernel loss mid-cell, crash and resume at every 500 steps, a child
  that hangs, a child that exceeds its budget, a pathological regex, a 50 MB cell output.
- **Real model:** Qwen2.5-0.5B on CPU in the Docker kernel (free). A larger model needs approval.

## 5. Stress-test break points (03-crucible-stress.md) and who closes each

| Break point | Fix | Owner |
|---|---|---|
| B1 prompt 1.42× over budget (chars/4 trigger); the compaction call is the largest prompt | usage-anchored meter; threshold leaves room for the summary call; the summarizer gets a bounded `summary_view` (hollowed to fit `window − summary_max_tokens − reserve`); a pre-send check on *every* call, root, compaction and child | B (meter, view), F (pre-send check) |
| B2 a root overflow is masked (no failure kind); the compaction call has no `try` | an overflow on a root call compacts and retries; with compaction off it is the scored stop `context_overflow` (no failure kind → never masked); every compaction attempt is inside the Compactor's try: provider faults mask with their kind, overflows hollow and retry | B, F |
| B3 resume never trains: the first call after resume is a prefix break → `capture_incomplete` | a resume opens a new segment (`boundary_reason="resume"`) on every agent; the first call stores its whole prompt; test: a crash + resume rollout is train-valid | F (policy), D (record test) |
| B4 the per-cell digest is O(namespace) and misses changes past 100k chars; restores are full replays | snapshots + journal suffix; the per-cell digest is bounded (cheap per-cell fingerprint, full digest only at snapshot points), and clock / random reads inside a cell are made replayable (seeded random already; the cell's clock is recorded per cell, so replay is exact) | G |
| B5 each checkpoint holds the whole episode (16.9 MB per save; tool results 3–4×; `_archive` literals) | the ledger holds every text once; a checkpoint = the current context + counters + ledger offset + the latest snapshot path + the journal *since* that snapshot; `model_calls` stream to `calls.jsonl` and the checkpoint keeps an offset; `_archive` disappears | F (state_dict), G (journal suffix), A (ledger offset/reopen) |
| B6 episode-wide counters end honest episodes (stalls never decay; 2 continuations per episode; `max_turns=500`; 256 sub-calls) | windowed stall detector that tells waiting (a poll with a changing result or a rising clock) from looping; continuations per segment; turns, sub-calls and wall time from `TreeLimits` | B (stall), C (limits), F |
| B7 sub-calls fragment the root chain (trainer forwards 7×, 19.7× with 256 subs) | a sub-call or child turn never breaks the root's chain: chains keyed `(actor, agent, segment)` with subs as their own short sequences; test: forwarded tokens ≈ entered tokens (± subs' own) | D |
| B8 `trajectory_tokens` 23–35 % low; segment `context_before` in estimate units | accounting from engine usage (prompt delta + completion), every number tagged `source`; segments carry measured tokens | F, B |
| B9 the kernel container lives 6 h; the model never sees the deadline | container lifetime derived from the episode's wall budget (+ margin) and renewable; time left in the footer; 0.9 wrap-up notice | G (lifetime), F (footer) |
| B10 killed-between-cells mislabelled; a World tool called by name burns a continuation; unbounded "Live names"; HISTORY keeps capped outputs | exact recovery notice; a World tool called by name gets a precise correction (the equivalent `await world.call(...)` line) without using a continuation; `python_state_row` capped; the ledger keeps full outputs | F, B, A |
| Real model: Qwen2.5-0.5B never ran a cell (called tools by name) | the correction above; a short worked first-cell example in the system prompt; continuation floor | F |
