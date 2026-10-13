# M15 sweep G1: how trainers consume SFT / preference / KTO data vs what `crucible label` emits

Crucible HEAD 076ab65. Scope: `src/crucible/studio/{label,datasets}.py`. There is no `credit.datasets`: the "credit
datasets" helpers are `studio/datasets.py`. The consumers checked are TRL (72cf37d), prime-rl (f7e8ab6) with
`renderers` 0.1.12.dev19, verl, the MiMo verl fork, NeMo-RL, OpenRLHF, rLLM, SkyRL, verifiers v1, NeMo-Gym, Harbor,
SWE-smith, and the articles in `07-articles`.

Probes and real outputs are in `scratchpad/m15/g1/`. They used the real tokenizers offline (Qwen2.5-0.5B-Instruct,
Qwen3-0.6B, Qwen3-4B-Instruct-2507, gemma-2-2b-it) and the consumers' own code:
- TRL is imported from source; `datasets` is stubbed only for the import.
- prime-rl uses its `renderers` dependency from the uv cache, and its own helpers copied verbatim.
- verl's per-message path is copied verbatim.
- HF `datasets` 5.1.0 / pyarrow 25.0.1 come from `.envs/nemo`.

The label outputs came from the CLI:

| Output | Command | Result |
|---|---|---|
| `label_toy/` | `crucible label --pack toy` | 24 traces → 18 SFT / 0 DPO / 22 KTO |
| `label_toy_neg/` | same with `--mask-negative` | same counts |
| `build_toy/` + `label_run/` | `crucible build toy --fresh 2`, then `label --run` | 21 episodes → 7 SFT / 7 DPO / 21 KTO |
| `label_coding/` | `label --pack coding` | 66 traces → 1 SFT / 60 KTO |

Doctored copies of the run tested the validity and exposure paths: `run_invalid/` and `run_exposed/`.

Severity: **P0** trains the wrong tokens, leaks hidden or held-out material, or turns failed/masked episodes into
positives. **P1** is a format a consumer rejects or silently misreads, or a vacuous test. **P2** is a gap.

## Headline defects

### P0

**D1 [ran] `label --run` writes held-out benchmark items into training data.**
- `build` seals `splits.json`. On the toy build it put t2_approval, t3_in_transit and t4_decoy in `"bench"`.
- `from_run` (`datasets.py:95-112`) never reads splits. All 7 tasks, bench ones included, become SFT rows with the
  oracle's solution, plus DPO pairs and KTO rows (`label_run/sft.jsonl`).
- Fix: `from_run` honours `splits.json`, emitting the `train` split only unless `--include-bench` is given, and writes
  the split seal into the provenance.

**D2 [ran] Every run-sourced SFT row and DPO "chosen" is the scripted oracle, which acts on information it never observed.**
- All 7 SFT rows have `model: scripted:oracle`.
- Its action sequences (`probe: label_run/sft.jsonl`) include:
  - `issue_refund38, add_note38, set_status38`: no `get_ticket` or `get_order` before refunding 20.9 on O-7108.
  - Generated item: `add_note100, set_status100, issue_refund101 …`.
- The trained note says "Checked against the current refund policy (FIN-7) and the order history." The policy never
  checked. SFT teaches acting on hidden state and claiming checks that were not made.
- Every chosen turn ends with the scripted "script finished".
- `from_run` has no policy filter (`datasets.py:100-111`). `curate` keeps anything at or above the threshold
  (`datasets.py:233-234`).
- Harbor refuses this case: oracle has `atif=False` and aborts `traces export` (G1-B ledger). The repo2rlenv article
  (`hf-repo2rlenv…md:63`) says "keep the gold patch … off the box".
- Fix: exclude `scripted:*`, `oracle` and `gold` policies by default. Record `policy_kind` in the provenance.

**D3 [ran] DPO and KTO rows put environment output in the completion, so TRL and OpenRLHF train on it.**
- `dpo_best_vs_worst` and `kto_rows` split at the first assistant turn. The completion is the whole remaining episode
  (`datasets.py:271-274, 301-302`).
- TRL builds the loss over every completion token (`dpo_trainer.py:1054-1101`, labels at `:1228`;
  `kto_trainer.py:1067-1108`).
- Measured with Qwen3-4B-2507 over the TRL-tokenised completion region (`probe_pref2.py`):

| Rows | Assistant tokens | Tool output tokens | User-turn tokens |
|---|---|---|---|
| run DPO chosen | 4,215 | 6,015 | 625 |
| run DPO rejected | 953 | 1,227 | 136 |
| run KTO | 6,322 | 9,574 | 981 |
| toy KTO | 2,121 | 343 | 1,205 |
| coding KTO | 348,117 | 7,614 | 403,014 |

- In run DPO chosen, 61% of the trained tokens are World output or other actors' turns.
- The coding KTO completions include the hidden verifier's gate verdicts, for example
  `[system system] tests_gpu: fail … perf: fail`. These are `grade_gate` events, rendered as user turns by
  `from_trace:157-159`.
- NeMo-RL accepts only single-response chosen and rejected (`binary_preference_dataset.py:79-90`; `dpo.md:60`
  "each completion being a single response"). It masks all but the final message (`collate_fn.py:378-381`).
- Fix: preference and KTO rows are one decision. The prompt is the full history up to an assistant turn, and the
  completion is that single assistant message. Pivots already work this way (`datasets.py:287`).

**D4 [ran] Workflow traces train invented tool calls.**
- `from_trace` names every non-message policy event's tool call after the event type (`e.type`). Its arguments are
  `{object, text, **scalar e.data}` (`datasets.py:149-153`). The tool result is the fabricated string
  `"<type>: recorded"` (`:154`).
- Under TRL `assistant_only_loss`, the coding row trains 107,978 tokens of this (`probe_coding_span.py`):
  - The name is `tool_call`, not `bash`. A sample trained call:
    `{"name": "tool_call", "arguments": {"arguments": "{\"command\": …}", "call_id": "toolu_01Jiqq7…", "step_id": 3, "text": "bash({…})", "tool": "bash"}}`.
  - Each call carries the vendor call id and recorder step id, and repeats the command twice.
  - The final turn trains `{"name": "final_patch", "arguments": {"bytes": 246915, "files": 47, …}}`.
  - The real bash output comes after the fake tool result, as a user turn: `[system system] <returncode>0…`.
- The toy rows train `status_change` with `row_id: "T1012-4"`. No such tool exists; the World's tool is
  `tickets__set_status`.
- Fix: map M3 `tool_call` / `tool_result` events by call id, the way `interop/harbor/atif.py:41-53` already does for
  World traces (`data.function` / `data.tool`, `data.arguments`). Non-tool workflow actions map only through a tool
  vocabulary the pack declares. Otherwise they go in the content and are never shown as tool calls.

**D5 [ran] A runner-masked exposed episode becomes training data.**
- The run was doctored: the t2 oracle record got `failure.kind=crucible:answer_exposed`, `mask_sample=true`,
  `reward=null`.
- `label --run` wrote it as a KTO row (`label=False, reward=0.0`) with the full transcript, which by definition
  contains the leaked answer.
- Cause: `from_run` reads the bench view (`metrics.py:254-255`, where an exposed episode is "scored 0", and the
  comment says "masked for training by the runner"). It does not read the record's `scored_reward()` /
  `mask_sample`.
- Fix: skip `e.exposed`, or use `RunRecord.scored_reward()`. Never use `Episode.reward` for training sets.

**D6 [ran] Episodes labelled bad (and unknown) are SFT positives; KTO uses the opposite rule.**
- SFT curation ranks and selects by reward (`curate`, `datasets.py:232-234`). The toy SFT has 18 rows: 14 good,
  **2 `bad`** (T1012, T1018; reward 1.0) and 2 `unknown`.
- T1012's trace ends with the customer saying "The discount code never worked", and the trained turn is
  "Yes, 30% with a valid student email."
- KTO prefers the label (`datasets.py:298`), so T1008 and T1023 (`good`, reward 0.0) are desirable.
- The two sets contradict each other on the same trace.
- Fix: one positive rule. A positive needs `label != bad` AND reward ≥ threshold, and `unknown` is excluded unless
  `--allow-unknown`. Record the reason a row was kept.

**D7 [ran] `label --run` ignores the validity manifest.**
- `run_invalid/validity.json` marked t1_duplicate as `valid: false` (`noop_passes`). Its rows were still written:
  1 SFT, 1 DPO and 3 KTO.
- The summary does not mention validity.
- The M13 contract ("validity manifest keeps broken / no-op-passing tasks out of every metric") does not reach
  datasets.
- On a task where the no-op passes, the no-op transcript would be an SFT positive.
- Fix: read `validity.json` and drop invalid tasks loudly.

### P1

**D8 [ran] No `sft.jsonl` loads in HF `datasets`.** Every consumer loads it this way: TRL, prime-rl, NeMo-RL, OpenRLHF,
SkyRL and rLLM.
- `load_dataset("json")` raises `ArrowInvalid: Column 11: … Length spanned by list offsets (134) larger than values
  array (132)` on toy, run and coding (`probe_arrow.py`).
- It is the `weights` column (`[null, 0.66, null, …]`) next to the `messages` Json column. Dropping `weights`, or
  writing 0.0 for null, loads (`probe_arrow2.py`).
- DPO and KTO files load.
- The M15 tests never load the files with `datasets`, so "TRL-canonical" is untested.
- Fix: write `weights` as floats with 0.0 off-policy, plus a separate `weight_mask`, or drop `weights` from the default
  file.

**D9 [ran] On toy and coding, the KTO `label` column is the outcome string, not a bool.**
- `kto_rows` sets `label: bool(good)` and then spreads `**_prov(x)`, whose `label: x.label` overwrites it
  (`datasets.py:246, 302`).
- Effects:
  - TRL `kto_trainer.py:1139` `sum(dataset["label"])` raises TypeError.
  - The collator's `torch.tensor` (`:206`) raises ValueError.
  - Arrow types the column as `string`.
  - Any `bool()` coercion makes all 6 `bad` rows desirable.
- The unit test (`test_studio.py:65`) uses `Labelled` with no `label`, so it passes vacuously.
- Fix: rename the provenance key to `outcome`.

**D10 [ran] Tool-call `arguments` is a JSON string** (`datasets.py:84, 153`).
- TRL's documented format is a dict stored as `Json()` (`dataset_formats.md:134-137, 182-200`).
- With the Qwen2.5 template (`tool_call.arguments | tojson`), 18/18 toy and 7/7 run rows render and train a
  double-encoded `"arguments": "{\"object\": …}"`.
- Qwen3 renders strings verbatim. prime-rl (`utils/chat_template.py:25-50`) and SkyRL (`sft_trainer.py:339-377`) parse
  strings first.
- Fix: emit dicts, and write `tools` / arguments the way TRL documents for `datasets<4.7`.

**D11 [ran] TRL and verl ignore `message_loss_mask` and train turns Crucible masked.**
- Probe: the first assistant turn of a run row was set False.
- prime-rl excluded it (`renderers/base.py:1745-1753`).
- TRL `assistant_only_loss` trained `tickets__issue_refund` anyway (`probe_mask.py`). verl masks by role only
  (`multiturn_sft_dataset.py:234-239`).
- So `--mask-negative`, the D4 fix and any per-turn exclusion are effective only on prime-rl.
- `weights` is read by no consumer: TRL, prime-rl, verl, NeMo-RL, OpenRLHF, rLLM and SkyRL were all checked.
- Fix: also emit expanded rows (one row per kept assistant turn: prompt = history, completion = turn). This is the
  multi-harness article's recipe (`hf-space-fineenvs…md:1031-1033`), and every consumer honours it.

**D12 [ran] Gemma-2 refuses every row.**
- On 18/18 toy and 7/7 run rows, TRL and prime-rl both raise `TemplateError: Conversation roles must alternate`.
- Cause: consecutive user turns, since every non-policy event is its own user turn (`datasets.py:157-159`), plus
  `tool` roles.
- verl's sanity assertion (`:425-456`) fails on 3/18 toy rows (consecutive tool messages; ChatML merges them) and on
  7/7 run rows with Qwen3-0.6B (the empty `<think>` block on the last turn).
- Fix: merge consecutive same-role context turns, and add a `--template <tokenizer>` check that renders each row with
  the consumer's code.

**D13 Run-sourced rows carry `tools: []`.**
- Crucible's ATIF writer omits `agent.tool_definitions`, which Harbor's schema has (`harbor …/agent.py:23`).
  `messages_from_atif` drops `reasoning_content` (`datasets.py:70-91`).
- So the model is trained to call tools no system prompt declares. The doc lists this as open, but it is a format
  error, not a gap.
- Fix: write the World's tool cards into the ATIF and copy them into every row.

**D14 [ran] Coding KTO has 54 rows with `completion: []`.**
- These are PR traces: no policy actor, every event a user turn.
- TRL yields 0 completion tokens. Their prompts end with `[maintainer] merged`, which is the label itself.
- Fix: drop rows with no policy turn.

**D15 The `next` hint is false.**
- `label.py:160` prints `python -m crucible.train.launch <job.toml>  # train on sft.jsonl`.
- `train/launch.py` has no dataset path (its job spec is engines, workers and env servers; `:1-23`). No bridge in
  `train/bridges/` reads the label files.

### P2

- **Qwen3 hybrid.** TRL's training template trains `<think>\n\n</think>\n\n` on every assistant turn; prime-rl does
  not. Same row, different tokens. Rows carry no `chat_template_kwargs` / `enable_thinking` and no
  `reasoning_content`.
- **No length policy.** The coding SFT row is 178,375 tokens, above Qwen's 131,072. TRL's default `max_length=1024`
  with `keep_start` would train only the first turns. NeMo-RL masks overlong rows instead (`processors.py:173-180`).
- **Task granularity.** Pack rows use `task = trace.family`, so all 24 toy tickets are one "task". `--max-per-task`,
  the filter stats and best-vs-worst then work per family.
- **No dedup.** `traj_id` exists but is never used. rLLM's `--dedup` exists. There is no validation split by task
  (rLLM `cli/dataset.py:282-299`).
- **Thin provenance.** Pack rows have `model: ""` although the actor handle is known (`anthropic/claude-fable-5.1` in
  coding), `run: "file_drop"`, and no policy, harness, seed, split or validity fields. The multi-harness article marks
  hosted-API outputs "evaluation only" (`:524-528`).
- **KTO balance is unreported.** Toy is 16:6 and run 7:14. TRL prints a weight recommendation (`kto_trainer.py:1139-1160`).
- **`--mask-negative` is untested on real data.** Assistant weights are never negative on toy: Φ drops land on
  non-policy events. `dpo_pivots` has no test, and toy yields 0 pivots. `state_shift` pairs are flagged but kept.
- **No RL prompt rows.** `label` emits none (verl/MiMo `prompt` / `data_source` / `reward_model.ground_truth` /
  `extra_info.instance_json`).
- **System prompts become user turns.** They render as `[system system] …` user turns (`from_trace:157-159`), so
  training context differs from inference.

## Requirement ledger (consumer → Crucible → status)

Consumer citations below are relative to `06-frameworks-and-env-code/`.

| # | Requirement (consumer file:line) | Crucible (file:line) | Status |
|---|---|---|---|
| 1 | SFT `messages` with roles system/user/assistant/tool (TRL `sft_trainer.py:1330-1401`; prime-rl `trainer/sft/data.py:303-334`; verl `multiturn_sft_dataset.py:103`; NeMo-RL `oai_format_dataset.py:125-200`; SkyRL `sft_config.py:188-193`) | `datasets.py:255` | done (but D4, D12) |
| 2 | Assistant `tool_calls` in OpenAI shape, `arguments` a dict (TRL `dataset_formats.md:134-137`; NeMo-RL `sft.md:419-433`) or a string parsed by the consumer (prime-rl `chat_template.py:25-50`, SkyRL `:339-377`) | string, `:84, :153` | **WRONG for Qwen2.5 / TRL** (D10) |
| 3 | `tool` messages with `tool_call_id` (NeMo-RL docs, Qwen3 templates); `name` optional | `:89, :154` | done; content fabricated for traces (D4) |
| 4 | `tools` schemas per row (TRL `:1331`, prime-rl `:329-333`, NeMo-RL `processors.py:168`, SkyRL `:506-508`, verifiers `trace.py:391`, Harbor `tool_definitions`) | `[]` always | **missing** (D13) |
| 5 | Per-message loss selection (prime-rl `message_loss_mask`, `renderers/base.py:1745-1753`; verifiers `graph.py:103` `sampled`) | `:252` | done for prime-rl; ignored by TRL, verl, NeMo-RL, OpenRLHF, rLLM, SkyRL (D11) |
| 6 | Trained tokens are exactly the policy's sampled tokens (all consumers; GLM SAO; multi-harness article `:556-558`) | role == assistant | **WRONG** for traces (D4); SFT masks themselves verified correct on the rendered rows (`probe_sft.py`: 0 non-assistant bodies trained, prime-rl 0 tokens outside assistant) |
| 7 | Per-turn weights (none of the consumers) | `weights` `:255` | inert; **breaks Arrow loading** (D8) |
| 8 | Loadable with `datasets.load_dataset("json")` (all HF-based consumers) | `write_jsonl :306-309` | **SFT fails** (D8); DPO / KTO load |
| 9 | DPO `prompt` / `chosen` / `rejected`, prompt shared and chosen / rejected differing only after it (TRL `dpo_trainer.py:1054-1101`, `common_prefix_length`; OpenRLHF `reward_dataset.py:19-31`) | `:271-274` | exact shared prompt at message level (verified: no TRL prefix warning); **completion spans the episode** (D3) |
| 10 | DPO completion is one response (NeMo-RL `binary_preference_dataset.py:79-90`, `collate_fn.py:378-381`; verl has none) | multi-message | **rejected or misread** (D3) |
| 11 | KTO `prompt` / `completion` / `label: bool` (TRL `kto_trainer.py:206, 1139`) | `:302` | **WRONG** for pack rows (D9); empty completions (D14) |
| 12 | Positives: success only (rLLM `curation.py:323-326`; Harbor `--filter success`; multi-harness "first one the verifier graded correct"); masked / infra rows excluded (NeMo-Gym `mask_sample`; MiMo / StepFun "mask") | `curate :233`, `from_run :101` | **partial**: bad / unknown labels pass (D6); exposed masked rows enter (D5); invalid tasks enter (D7) |
| 13 | No oracle / privileged trajectories (Harbor oracle `atif=False`; repo2rlenv `:63`; karotte `use_hints`) | none | **missing** (D2) |
| 14 | Held-out split respected (rLLM val split by task; M13 splits) | none | **missing** (D1) |
| 15 | Provenance: task, run, model, reward, resolved, traj hash (Harbor `traces_utils.py:18-35`; SWE-smith `collect_trajs.py:57-74`; verifiers `task.hash`) | `_prov :244-246` | partial (P2) |
| 16 | Dedup / max per task / val split (rLLM `curation.py:350-423`; SWE-smith `max_per_inst=3`) | `curate :239` | partial: no dedup or split |
| 17 | Thinking content kept or explicitly stripped (Harbor `<think>`; NeMo-Gym `reasoning_content`; qingkeai `:915`) | dropped | missing (P2) |
| 18 | Conversation ends on assistant (NeMo-RL `:193` assert; SkyRL `:491-497` drops trailing tool and requires an assistant last) | toy rows end on a user turn | **rejected** by NeMo-RL `OaiFormatDataset`; SkyRL pops only `tool`, so toy rows are dropped |
| 19 | Strict templates (Gemma alternation, verl per-message equality) | consecutive user / tool turns | **rejected** (D12) |
| 20 | Length limit / overlong policy (NeMo-RL `processors.py:173-180`; verl `truncation=error`; qingkeai overlong filtering) | none | gap (P2) |

## Probes (all [ran], in `scratchpad/m15/g1/`)

- **`probe_sft.py`**: TRL `tokenize_fn` / `build_labels` (`sft_trainer.py:1330-1443`, training template `:942-943`) and
  prime-rl `_process` (`data.py:303-369`), for the 4 tokenizers × toy, toy-neg and run SFT.
  - TRL: 18/18 and 7/7 rows OK for Qwen. Qwen2.5 args are double-encoded on 100% of rows. Gemma-2 fails 100%.
  - prime-rl: Qwen3 renderer OK. Qwen2.5 raises `role_to_mask is required` (`base.py:1774-1779`): prime-rl's limitation
    for DefaultRenderer models.
  - Trained tokens:

| Model | TRL | prime-rl |
|---|---|---|
| Qwen3-0.6B, toy | 1,996 | 1,856 |
| Qwen3-0.6B, run | 4,675 | 4,128 |
| Qwen3-4B-2507, toy | 1,884 | 1,856 |
| Qwen3-4B-2507, run | 4,215 | 4,128 |

  - The Qwen3-0.6B gap is the empty `<think>` block TRL trains. With Qwen3-4B-2507, TRL and prime-rl produced identical
    `input_ids` on the toy row checked; spans differ only in the trailing `\n` after `<|im_end|>`.
- **`probe_sft_diff.py`, `probe_coding_span.py`, `probe_mask.py`**: span-level evidence for D4, D10 and D11.
- **`probe_pref.py`, `probe_pref2.py`**: TRL DPO and KTO tokenisation with completion-region role attribution (D3, D9,
  D14).
- **`probe_arrow.py`, `probe_arrow2.py`**: HF `datasets` loading (D8).
- **`probe_verl.py`**: verl per-message tokenisation and its sanity check (D12).
- **`run_invalid/`, `run_exposed/`**: doctored copies of the toy run (D5, D7). D1 and D2 come from `label_run/` and
  `build_toy/splits.json`.

## Picks

| Pick | What | Source and licence | Effort |
|---|---|---|---|
| **PORT** | Per-message "sampled by the policy" bit as the only training signal. Replace the role-based mask (`:252`) and drive the D4 fix from it. | verifiers v1 `graph.py:103`, `trace.py:221-227` (MIT) | S |
| **PATTERN** | One-decision preference rows: a single-response DPO / KTO completion. | NeMo-RL `binary_preference_dataset.py`, `collate_fn.py:378-381` (Apache-2.0); NeMo-Gym DPO pairing by task / input hash with `quality_diff` and `mask_sample` exclusion (`offline-training-w-rollouts.mdx:199-224`, Apache-2.0) | M |
| **PATTERN** | Expanded per-turn SFT rows so every consumer trains the right turns. | multi-harness article `:1031-1033`; Harbor `traces export --episodes all` (`traces_utils.py:527-555`, Apache-2.0) | S |
| **PORT** | Curation: dedup by content hash, `best-n` / `shortest`, validation split by task, and the pass rule as reward and outcome. | rLLM `eval/curation.py:323-423`, `cli/dataset.py:282-320` (Apache-2.0; the pyproject classifier says MIT) | S |
| **DEPEND (test-only)** | `renderers` and TRL `data_utils` / `chat_template_utils` in a `crucible label --check <tokenizer>` and a regression test. Each row is rendered through the consumer's code, and a row fails if any trained token belongs to a non-policy message. This catches D3, D4, D10 and D12. | Apache-2.0 | M |
| **PATTERN** | Tool-call normalisation on write: dict arguments, `Json()` features or JSON-string `tools` for old `datasets`, drop trailing tool turns. | SkyRL `sft_trainer.py:339-377, 491-497`; TRL `dataset_formats.md:182-211` (Apache-2.0) | S |
| **PATTERN** | Oracle and privileged exclusion, and a split and validity gate in `from_run`. | Harbor oracle `atif=False`; M13 `splits.json` / `validity.json`, already Crucible's | S |
| **PATTERN** | Tool cards in the ATIF (`agent.tool_definitions`) and `reasoning_content` carried through. | Harbor `models/trajectories/agent.py:23`, `step.py:48` (Apache-2.0) | S |
