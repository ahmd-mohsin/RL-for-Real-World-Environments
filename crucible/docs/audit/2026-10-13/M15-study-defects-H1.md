# M15 audit, hunt H1: `crucible label` and the dataset rows

HEAD 076ab65. Scope: `src/crucible/studio/{label,datasets}.py`, what they call (`bench/metrics.load_run`, `signals/pivots`,
`connectors/redact`, the toy and coding `credit` hooks) and `tests/unit/test_studio.py`. Nothing under `src/`, `packs/` or
`tests/` was edited.

Reproducers and outputs are under the scratchpad
`/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m15/h1/`
(written `$H` below). The M14 campaign's run directories are reused as inputs. They are real M10 runs: Qwen2.5-0.5B native and RLM,
the RLM probes, Qwen3 compaction, scripted adversaries, controls, and coding Φ hacks
(`.../scratchpad/m14/validate/runs`, written `$V`).

Drivers:
- `label_all.py`: labels all 40 `$V` runs with `--select all` and tabulates the rows by policy;
- `doctor.py`: a real toy bench run with one exposed episode, one invalid task and one hack flag;
- `unobserved.py`: tool-call argument values that no earlier message showed;
- `dpo_seeds.py`: maps DPO pairs back to their policies and seeds;
- `pivot_probe.py`: a synthetic correction pivot;
- `redpack/`: the toy pack plus fake canaries.

Python: `crucible/.venv/bin/python`, `PYTHONPATH=src:packs/...`, `HF_HUB_OFFLINE=1`.

**Root cause behind V1 to V5.** `from_run` rebuilds every run row from `trajectory.agent.json` (ATIF) and the bench view of the
reward (`load_run`). The runner already writes the authoritative training view in `rollout.json`: `turns` with
`trainable`, `kind` (`turn` / `sub` / `compaction`), `segments`, the token ids, `mask_sample` and `flags`, and the policy
and harness. Label reads none of it.

## Findings

| id | sev | where | defect |
|---|---|---|---|
| V1 | **P0** | `datasets.py:95-112`, `label.py:148-156` | Control, scripted and adversary policies become SFT positives and DPO `chosen`. The oracle's rows emit answer-key values it never observed. |
| V2 | **P0** | `datasets.py:100-102`, `bench/metrics.py:254-255` | Label ignores the validity manifest, the rollout hack flags and the runner's training mask. Exposed (`answer_exposed`), tampered and flagged episodes are emitted. |
| V3 | **P0** | `datasets.py:67-92,95-112` | RLM episodes are rewritten as native tool-call chats the model never produced. Its `python` cells are dropped, and every World call made inside a cell becomes an assistant `tool_call`. Compaction, sub-call and segment structure is lost. |
| V4 | **P0** | `datasets.py:274,287,302` | DPO `chosen` / `rejected` and the KTO `completion` include tool, user, supervisor and customer messages. TRL trains every completion token. The trailing CSAT and supervisor verdict, which define the label, sit in the KTO completion. |
| V5 | P1 | `datasets.py:67-92,110` | Native rows are not what the model saw or emitted. The system prompt is missing. One completion with 3 parallel tool calls is split into 3 interleaved turns. The "same prompt" check of DPO becomes vacuous across harnesses. |
| V6 | P1 | `datasets.py:244-246,302` | `_prov` is spread after `"label": bool(good)`, so pack KTO rows carry `"label": "good"/"bad"` strings. TRL KTO crashes (`sum(dataset["label"])`). A `bool()` cast makes every `bad` row desirable. |
| V7 | P1 | `datasets.py:233-234,298`, `label.py:134-139` | SFT and KTO disagree on which pack traces are positive. Outcome-`bad` and `unknown` traces are SFT positives (reward 1.0). Outcome-`good` traces with reward 0 are KTO positives. |
| V8 | P1 | `datasets.py:249-257` | SFT trains turns already graded wrong. A pivot's wrong turn, a supervisor-corrected reply, keeps mask True and positive Φ weight even though `Labelled.pivots` / `pivots_unpaired` know it. |
| V9 | P1 | `label.py:120,138` | The toy test tickets T1013 to T1024 are held out by the credit hook ("test tickets are held out"), yet they are emitted as SFT/KTO rows: 9 SFT and 11 KTO. |
| V10 | P1 | `datasets.py:235-236` | `--select best` emits the best *failed* episode of an unsolved task. rLLM's semantics, which the docstring claims, are the best passing episode. |
| V11 | P1 | `datasets.py:269-275` | Best-vs-worst does not require `rejected` to fail. With `--threshold 0.5` the pair is 1.0 vs 0.7; with `--margin 0` it is 1.0 vs 1.0. When prompt lengths differ, the worst episode's extra context moves into `rejected`. |
| V12 | P1 | `datasets.py:268-275,295-303` | Empty or stub completions. A no-op `rejected` is `[]` or a scripted "script finished". There are empty KTO completions: 6/12 Qwen finance_ap rows and 54/60 coding PR rows. |
| V13 | P1 | `datasets.py:179-187,282-292` | Correction pivots with `state_shift=True` are emitted, and their `chosen` depends on feedback that is not in the prompt. `chosen` is the whole later turn: a post-hoc adjustment plus unrelated actions, not an action valid at the rejected state. |
| V14 | P1 | `tests/unit/test_studio.py:65,103-106` | Vacuous tests. The gate enshrines 7 oracle-vs-no-op DPO pairs. The KTO label assertion covers only the `label=None` path. The mask check reads one row. |
| V15 | P2 | `datasets.py:200-240` | Filter and curation edges. `n_correct / n` raises `ZeroDivisionError` for a family with no credited trace. A syntax error is `SyntaxError`, not `ValueError`. `--max-per-task 0` means unlimited; `-1` drops the last episode. |
| V16 | P2 | `datasets.py:57-59,244-246,274-275`, `label.py:153` | Provenance. `traj_id` is not unique: 285 rows give 66 ids, and one id spans 4 tasks. DPO rows have no run, episode, policy or model. Identical rows are not deduplicated (285 SFT rows, 77 distinct). Skipped episodes are not counted. Non-`agent` actors are skipped silently. |
| V17 | P2 | `datasets.py:110` | Run rows carry no per-turn Φ: `weights` is all None although `progress.jsonl` has Φ per step. `--mask-negative` silently does nothing on runs. |
| V18 | P2 | `datasets.py:149-154` | Pack-path tool calls carry log-internal keys (`row_id`, `object`) and fabricated tool names and results (`status_change: recorded`). |
| V19 | P2 | `datasets.py:95-112` | The run path applies no redaction. An email, a phone number and a secret-format canary in a World observation reach `sft.jsonl` verbatim. The pack path redacts the same canary. |

Checked and sound:
- **Determinism and idempotent ingest.** Labelling the same pack three times into fresh and reused out dirs gives
  byte-identical `sft`, `dpo`, `kto`, `traces` and `rewards` files, and the store keeps 25 files. Two labels of the same run
  give identical files.
- **Pack-path redaction.** A fake secret in the event text and in an `extra` field becomes `<secret>`, and the email and phone
  are pseudonymized.
- **Masked episodes are skipped** (`rlm_probe/{host,nodocker,noimage}`, coding c4).
- **The filter whitelist.** Calls, attributes, strings, `**`, lambdas and conditional expressions are refused.
- **Pair seeds.** The DPO pairs from different seeds share one World: the oracle trajectory is identical across seeds of a task.
- **RLM sub-calls** (`trainable=False`) never appear in ATIF, so they are never targets. They are not trained correctly
  either (V3).

---

### V1 (P0): controls, scripted policies and adversaries are SFT positives and DPO `chosen`

`from_run` keeps every non-masked episode whatever its policy. `curate(select="correct")` keeps every episode with reward
≥ threshold. `dpo_best_vs_worst` pairs the best against the worst of any policies. Nothing reads `run.json`'s policy list,
the bench's `controls`, or the `adv:` / `declared:` / `scripted:` / `control:` ids.

Reproducer: `crucible build toy --fresh 2`, then `crucible label --run $H/build/runs/bench-toy-build` (default
`--select correct`).

```
sft: 7 rows, all model=scripted:oracle
dpo: 7 best_vs_worst, chosen = scripted:oracle, rejected = scripted:noop ("script finished") or scripted:half
kto: 21 rows: oracle True, half/noop False
```

**The oracle's rows teach unobserved answer-key values** (`unobserved.py`):
- `t1_duplicate`: `issue_refund amount_usd 20.9` with no order lookup;
- `t5_compose`: `43.17`;
- the anchor (`all_tickets@2`): `request_approval 129.99, O-5001` before any read.

Counts by run:
- toy build: 4/7 oracle SFT rows emit such values;
- toy controls: 15/30.

The answer key is literally `hidden_strings()`: "each gold action serialized".

**Adversaries are positives too** (`label_all.py`):
- `$V/scripted/bench-ap`: 156 SFT positives come from adversaries that score 1.0, against 27 from the oracle. They include
  `new:malformed_plus_oracle` 27, `new:double_decide` 27, `new:price_float_compare` 27, `new:price_round_units` 24 and
  `prev:mask_selective` 15. **All 9 DPO `chosen` are adversaries.**
- `$V/coding/bench-coding-c1`: the Φ hacks `adv:new_string_literal` and `adv:new_dead_code` give 24 SFT positives at
  reward 1.0.

**Fix.** `from_run(..., policies=None)` reads `run.json`.
- By default, refuse and require `--policy` when the run holds more than one policy.
- Always drop control ids, `scripted:*`, `adv:*` and `declared:*`, and the pack's `policies` / `adversaries` hook ids.
- Never use a scripted oracle as SFT unless `--allow-scripted`.

**Regression test.** Label the toy bench run: assert no row's `model` / `policy` is in `{scripted:*, control:*, adv:*}`, and
assert the DPO count is 0 (the run has no model policy). This fails now: 7 SFT, 7 DPO.

### V2 (P0): validity, hack flags and the training mask are ignored

`load_run` is a bench view. It turns an exposed attempt, which the runner masked for training (`mask_sample=True`, P16),
into a scored 0 with `exposed=True`, and it carries `flags`. `from_run` checks only `e.masked`. Nothing reads
`<run>/validity.json`.

Reproducer: `doctor.py` on a copy of the toy build run, with three changes:
- t4 oracle: `answer_exposed`;
- t2: invalid (`noop_passes`);
- t3 oracle: `rollout.json` flag `hack/intercepted`.

```
[correct] sft: t2_approval model=scripted:oracle reward=1.0      <- invalid task
[correct] sft: t3_in_transit model=scripted:oracle reward=1.0    <- hack/intercepted
[correct] kto: t4_decoy model=scripted:oracle label=False        <- runner-masked exposed episode
[correct] dpo: t2_approval 1.0 vs 0.0 ; t3_in_transit 1.0 vs 0.0
[all]     sft: t4_decoy model=scripted:oracle reward=0.0         <- the exposed trajectory as an SFT target
```

On a real run, `.../scratchpad/m13h/tb21run/tb21` (the M13 hunt run) with `--select all`, the `adv_shadow` episode (`crucible:grader_tampered`) is
an SFT row.

**Fix.** In `from_run`:
- skip `e.exposed`, any `failure_kind` in `{answer_exposed, grader_tampered, policy_misbehavior}`, and any `flags` matching
  `hack/*`;
- skip `rollout.json` `mask_sample`;
- load `validity.json` (or `--validity`) and drop invalid tasks;
- report each skip by reason in `summary.json`.

**Regression test.** The `doctor.py` fixture: assert no row for t2 or t4, and no t3 oracle row. This fails now.

### V3 (P0): RLM episodes are rewritten into a chat the model never produced

In `$V/rlm_probe/oracle_toy` (reward 1.0, so a default SFT row), the model's turns from `rollout.json` were:

```
agent:c1 turn trainable=True  '<tool_call>{"name": "python", "arguments": {"code": "ACTS = [{'f': 'tickets.request_approval', ...'
agent:c2 turn trainable=True  '<tool_call>{"name": "python", "arguments": {"code": "rlm.finish('done')"}}</tool_call>'
```

The SFT row trains instead:

```
user, assistant[tickets__request_approval], tool, assistant[tickets__issue_refund], tool, assistant[tickets__add_note], ...
```

These are 9 native calls, each conditioned on a tool result the cell had not yet seen when the model wrote it.

`$V/qwen3_compact/c0` has 16 calls: 10 turns, 1 compaction (segment 1), 1 `sub` call (`trainable=False`) and 4 more turns.
It becomes `[user, assistant "done"]` with mask `[False, True]`: a one-word give-up as the target.

The Qwen RLM rows in `$V/llm/bench-ap-qwen` keep no assistant turn at all. They become empty KTO completions (V12).

**Fix.** Build run rows from `rollout.json` turns: the token ids or the policy's own chat messages, as rLLM uses
`chat_completions`.
- Train only `trainable` turns of `kind == "turn"`.
- Emit one row per segment, with the compaction summary as that segment's context.
- Refuse harnesses whose ATIF is not the chat (`rlm:*`) until then.

**Regression test.** Label `rlm_probe/oracle_toy`: assert no assistant message has a World tool name. The row is
built from the `python` turns, or it is refused. This fails now.

### V4 (P0): DPO and KTO completions contain non-policy messages

`dpo_best_vs_worst` takes `chosen = best.messages[p:]` and `rejected = worst.messages[p:]`. `kto_rows` takes
`completion = messages[p:]`. Each is a whole tail of the conversation.

TRL (`trl/data_utils.py:223-240`) templates `prompt + chosen` and trains every token after the prompt. There is no
per-message mask in DPO or KTO.

Counts:
- toy pack KTO completions: 33 assistant, 34 tool and 47 user messages;
- toy run DPO `chosen` + `rejected`: 152 assistant, 120 tool and 27 user messages.

The T1003 KTO completion ends with:

```
user '[supervisor ...] Refund above 50 issued without supervisor approval.'
user '[supervisor ...] Reversed: annual plans refund pro rata after approval.'
user '[customer ...] Confusing, I got two different answers.'      <- the CSAT that defines label "bad"
```

The policy is trained to generate the environment's feedback and the label evidence.

**Fix.**
- Emit step-level pairs: the prompt is the conversation up to a policy turn, and `chosen` / `rejected` / `completion` is that
  single assistant message.
- Or end the completion at the last assistant message and drop trailing non-policy messages.
- Or refuse multi-turn tails unless the target trainer supports a completion mask, and record that mask.

**Regression test.** For every DPO and KTO row: `all(m["role"] == "assistant" for m in chosen + rejected)` and the same for
`completion`. This fails now on toy pack and run.

### V5 (P1): native rows are not the model's context or output

Run `$V/llm/bench-ap-qwen`, sample `7aab3bb6`.

The decoded prompt of the first call starts with `<|im_start|>system\nYou are a careful operations agent. ...` and the
`<tools>` block of 13 schemas. The SFT row starts at the `user` message and has `tools: []`. The missing tools are a known
open item; the missing system prompt is new.

The first completion holds 3 calls in one generation:

```
<tool_call>{"name": "erp__approve_invoice", ...}</tool_call>
<tool_call>{"name": "erp__hold_invoice", ...}</tool_call>
<tool_call>{"name": "erp__reject_invoice", ...}</tool_call>
```

ATIF has 3 agent steps, so the row is assistant / tool / assistant / tool / assistant / tool: calls 2 and 3 are conditioned on
results the model had not seen.

Arguments are re-serialized with `sort_keys`, so the target tokens differ from the emitted ones.

Because the system prompt is gone, native and RLM episodes of one task have "identical prompts", so best-vs-worst would pair
across harnesses whose real contexts differ.

**Fix.** Use the turns, as in V3. Otherwise:
- record and emit the system message and the tool cards from `calls.json` / the policy's messages;
- group ATIF steps by `extra.call_id` into one assistant message with all its `tool_calls`.

**Regression test.** Label `bench-ap-qwen`: row `7aab3bb6` has `messages[0].role == "system"` and one assistant message with 3
`tool_calls`. This fails now.

### V6 (P1): KTO `label` is overwritten by the provenance string

At `datasets.py:302`, `{"label": bool(good), ..., **_prov(x)}`, and `_prov` adds `"label": x.label` for traces. On `label --pack toy`,
all 22 KTO rows have `label` set to `"good"` or `"bad"`, and coding gives 60. The unit test checks only items with `label=None`.

**Fix.** Rename the provenance key to `outcome` (or `trace_label`), and put `"label": bool(good)` last.

**Regression test.** `kto_rows([Labelled(..., reward=1.0, label="bad")])[0]["label"] is False`. Today it returns `"bad"`.

### V7 (P1): SFT and KTO disagree on pack positives

Toy pack:
- T1012 and T1018 have outcome `bad` and reward 1.0, so they are SFT rows (`resolved: True`) and KTO `bad`.
- T1002 and T1013 are `unknown` and are SFT rows.
- T1008 and T1023 are `good` with reward 0.0, so they are KTO `good` and absent from SFT.

The pack's outcome rule (CSAT / reopen) and its gated reward are two definitions. Label uses one for SFT and the other for
KTO, and never reconciles them.

**Fix.** On the pack path, a positive needs `reward >= thr` and `label != "bad"` (and `!= "unknown"` unless `--allow-unknown`).
Use the same predicate for SFT `select=correct`, DPO `chosen` and KTO `True`. Report the disagreements.

**Regression test.** Toy label: no SFT row has `label` in `{bad, unknown}`, and KTO True implies SFT-eligible. This fails now.

### V8 (P1): SFT trains turns known to be wrong

T1016 (`good`, reward 1.0) is an SFT row. The agent's turn "Check your spam folder" has mask True and weight +0.667. The
supervisor's `correction` signal grades that turn. It is counted in `pivots_unpaired`, yet `sft_rows` ignores pivots.

`pivot_probe.py` gives the same result: the refund graded wrong by a correction keeps mask True and weight +0.5.
`--mask-negative` cannot help, because Φ is positive.

**Fix.** `Labelled` records the message indices of graded-wrong turns, paired or not. `sft_rows` sets their mask to False (or
drops the row under `--strict`).

**Regression test.** T1016's first assistant message has mask False. This fails now.

### V9 (P1): held-out test traces are emitted as training data

`packs/_toy/pack.yaml` declares dev T1001 to T1012 and test T1013 to T1024. The credit hook mines on dev only ("test tickets are
held out"), but label emits rows for every trace: 9 of 18 SFT rows and 11 of 22 KTO rows are test tickets. Those tickets
grade M5/M6 and the toy gates.

**Fix.** Add `label.split` / `label.include` to the manifest (or a `split` hook) and emit only the train split, with the held-out
ids in `summary.json`.

**Regression test.** Toy label: no row with `episode` in T1013 to T1024. This fails now.

### V10 (P1): `--select best` emits failed episodes

`curate([reward 0.0, reward 0.2], select="best")` returns the 0.2 episode, and `sft_rows` trains it. In rLLM
(`rllm/eval/curation.py:_ranked_candidates`), `best` ranks passing attempts only, and only `all` takes failures.

**Fix.** For `best`, filter to `reward >= threshold` first.

**Regression test.** `curate([fail, fail], select="best") == []`.

### V11 (P1): best-vs-worst pair semantics

```
dpo_best_vs_worst([1.0, 0.7], threshold=0.5)   -> pair (1.0, 0.7): rejected passes
dpo_best_vs_worst([1.0, 1.0], margin=0.0)      -> pair (1.0, 1.0)
worst = [user, user "extra hint", assistant]   -> rejected = [user "extra hint", assistant]
```

The prompt is `min(prompt_len)`, and the prefix check passes on the shorter prompt.

**Fix.**
- Require `worst.reward < threshold`, and `margin > 0` (refuse `<= 0` at the CLI).
- Require `best.prompt_len() == worst.prompt_len()`, and that both completions start with an assistant message.

**Regression test.** Each of the three cases above returns `[]`.

### V12 (P1): empty and stub completions

KTO rows with `completion == []`:
- `$V/llm/bench-ap-qwen`: 6 of 12 (the RLM rows);
- `bench-ap-qwen-steps60`: 5;
- `bench-toy-qwen`: 3;
- coding pack: 54 of 60 (PR traces: the `author` role is not `agent`).

`dpo_best_vs_worst([pass, noop with no assistant turn])` has `rejected == []`. In the toy build run, every no-op `rejected` is the
scripted stub `"script finished"`, and so is the oracle's last SFT target.

**Fix.**
- Skip rows whose completion or `rejected` has no assistant message, and count them.
- Mark scripted-policy text (`extra.scripted`) and never emit it.

**Regression test.** On the coding and Qwen runs above, no KTO row has an empty completion. This fails now.

### V13 (P1): correction pivots

`pivot_probe.py` builds this sequence: the agent refunds 120.00, the supervisor says "pro rata: 90.00, fix it", and the agent
adjusts the refund to 90.00 and resolves. The emitted pair:

```
prompt  : [customer request]                       ('pro rata' in prompt: False)
rejected: assistant "Refunded the full 120.00" + refund_issued(120.00)
chosen  : assistant refund_adjusted(90.00) + status_change(resolved)
state_shift: True
```

`chosen` has three problems:
- it needs the supervisor's feedback, which the prompt lacks;
- it is an adjustment that makes sense only after the wrong refund (the right action at that state is
  `refund_issued(90.00)`);
- it bundles an unrelated `status_change`.

TRL ignores `state_shift`.

**Fix.**
- Do not emit `state_shift` pairs by default (`--allow-state-shift` to include them).
- Restrict `chosen` to the fix event's own call.
- Restrict `rejected` to the graded call, not the whole merged turn.

**Regression test.** `dpo_pivots([from_trace(probe)]) == []` by default. With the flag, `chosen` has exactly one tool call.

### V14 (P1): vacuous tests

- `test_readme_path_through_the_cli` asserts `best_vs_worst == 7`. Those are oracle-vs-no-op control pairs, so V1 is
  enshrined.
- `test_filter_grammar_and_rows` asserts KTO labels only for reward-only items, so it misses V6.
- The loss-mask check reads `sft.jsonl` line 0 only.
- No test labels a run with masked, exposed, invalid, RLM or multi-call episodes.

**Fix.** Replace these with the regression tests above, and add a fixture run with one episode per category.

### V15 (P2): filter and curation edges

- `compile_filter("n_correct / n > 0.5")` on a task whose rewards are all None raises `ZeroDivisionError` mid-label. This
  happens after ingest; coding PR families are uncredited.
- `compile_filter("avg >")` raises `SyntaxError`, which escapes the `ValueError` contract.
- `curate(max_per_task=0)` keeps everything, and `-1` drops the last episode. rLLM refuses `< 1`.

**Fix.**
- Wrap evaluation (a failing expression filters the task out, or raises one `ValueError` naming the task).
- Convert `SyntaxError` to `ValueError`.
- Use `click.IntRange(min=1)`.

### V16 (P2): provenance and duplicates

- In `$V/scripted/bench-toy` with `--select all`: 285 SFT rows, 66 distinct `traj_id`, 77 distinct (task, traj_id).
  `traj_id` hashes messages only, so no-op trajectories on 4 generated tasks share one id.
- DPO best-vs-worst rows have no run, episode, policy, model or seed. Their `chosen_traj` / `rejected_traj` are ambiguous.
- Pack rows have `run = "file_drop"`.
- `summary.episodes` counts only the rows kept.
- `from_run` drops an episode with no `trajectory.agent.json` without a count. Multi-actor runs (other actor ids) lose every
  other seat.

**Fix.**
- Hash (task, messages, tools).
- Add `chosen_episode`, `rejected_episode`, `policy` and `harness`.
- Deduplicate by (task, traj_id) under `--dedup` (rLLM has it).
- Write `skipped: {masked, exposed, invalid, no_trajectory, ...}`.

### V17 (P2): no dense credit on runs

Every run row has `weights == [None]*n`, although `progress.jsonl` has Φ per World step and `turns[].world_step` maps calls to
steps. `--mask-negative` does nothing on runs, silently.

**Fix.** Weight each assistant message by ΔΦ over its call's World steps, or refuse `--mask-negative` with `--run`.

### V18 (P2): pack tool calls carry log internals

The arguments include `row_id: "T1012-4"` and `object: "ticket:T1012.status"`. The tool name is the event type
(`status_change`, `refund_issued`), with a fabricated reply `"<type>: recorded"` and no `tools` schema. A model is taught to emit
connector row ids.

**Fix.** Drop `row_id` and other connector keys. Map event types to the pack's tool names (a `tools` hook) or emit text turns.

### V19 (P2): the run path does not redact

`$H/redrun`: an email, a phone number and a secret-format fake canary were appended to one World observation of the toy build
run. `label --run` leaves all three in `sft.jsonl`. The same canary through `label --pack $H/redpack`, which declares
`redaction_env: fake.env`, becomes `<secret>`, `<email:..>` and `<phone:..>`. The runner's credential scrub covers provider
credentials only.

**Fix.** Apply `RedactionPolicy` (with `--redaction-env`, or the pack's when the run names one) to every message in `from_run`.

**Regression test.** The `redrun` fixture: no email, phone or canary in any output file.
