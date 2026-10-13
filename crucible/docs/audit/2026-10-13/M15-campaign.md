# M15 `studio` campaign: build, label, and train on the labels (2026-10-13)

**Scope.** `crucible build`, `crucible label` (pack and run sources, every `--select` mode, `--max-per-task`, the filter
grammar, `--mask-negative`, the new `--allow-scripted` / `--include-bench` / `--validity` / `--sft-format` options),
`studio/datasets.py`, and the produced rows used as training data: a short SFT and a short DPO run of
Qwen2.5-0.5B-Instruct, then `crucible bench run` of the SFT checkpoint against the base model. CPU and Apple MPS only,
no paid calls, no GPU. Nothing under `src/`, `packs/` or `tests/` was edited by this campaign.

**Two rounds.**
- **Round 1** (HEAD `076ab65`, main tree): the first campaign, findings B1-B18 below (status column).
- **Round 2** (worktree `m15wt`, branch `m15-audit`, uncommitted fixes by the label / build agents): everything re-run.
  The label code was still moving during the round: `studio/datasets.py` last changed 11:20:39, `studio/label.py`
  11:53:56, `cli/studio.py` 11:54:11, `studio/build.py` 11:40:22 (PDT). Every label output used for training was
  regenerated after 11:54 and is **byte-identical** to the one produced at 11:30 (`label2/` vs `label2_prev/`), so the
  training data matches the tree as of 12:00. Builds were re-run after 11:40.

**Evidence root** (scratch, not committed): `$C = /private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m15/campaign/`.
Round 2 drivers: `env2.sh` (worktree `PYTHONPATH`), `teacher.py` (token-level model policies, below), `run_mix2.py`
(mixed-policy M10 runs, RLM container kill), `label_matrix2.sh`, `check2.py` (row invariants against the run,
independent of `datasets.py`), `hfload.py` (HF `datasets`), `train2_sft.py`, `train2_dpo.py`, `m15pol2.py` (bench
policies), `eps2.py`, `dbg_splits.py`, `cmp_dirs.py`. Outputs: `build2/`, `runs2/`, `label2/`, `train2/`, `bench2/`,
`logs2/`. Round 1 outputs stay in `build/`, `runs/`, `label/`, `train/`, `bench/`, `logs/`.

**Verdict (round 2).** The worktree fixes the four round-1 P0s: run rows are now the policy's own token turns (system
prompt, tool cards, parallel calls and RLM cells as sampled; of 9,883 rows checked against the rollout across 24 label outputs, the only 3 that differ are redacted ones, B24),
tool arguments are dicts, and masked, exposed, flagged, invalid-task, held-out, control, adversary and (by default)
scripted episodes never reach a dataset. Training on the rows behaves: SFT and DPO gradients are exactly zero off the
policy's completion tokens, losses fall, no NaN. **One P0 remains:** an episode without `rollout.json` loses its hack
flags, and under `--allow-scripted` its flagged passes become SFT positives (B19). One new P1: best-vs-worst DPO
pairs at the first divergent turn, which made all 3 finance_ap pairs prefer a grader-path probe over the correct first
action (B20). Still present from round 1: `crucible build` builds only the toy (B10, P1), plus B13 / B14 / B17 (P2). The SFT checkpoint does not beat the base on the
bench (0 vs 0 on the common task; the run was stopped incomplete): it emits well-formed calls but loops.

---

## 1. Build (round 2)

| Pack | Command | Result | Wall |
|---|---|---|---|
| toy r1, r2 | `crucible build toy --k 2`, one fixed key (`build/fixed.key`) | exit 0; ingest 24 → forge 5 → M9 0 accepted / 5 provisional (A10 pending) / 0 rejected → 5 canonical exports (oracle 1.0) → splits 6 train / 1 bench (seal `960dc94a`) → contamination 0 / 1 → bench of the bench split: oracle 1.0, half 0.0, no-op 0.0 | 52 s, 49 s |
| toy r3 | same, fresh key | exit 0; splits 4 train / 3 bench; same scores | 59 s |
| finance_ap, coding, `pack new demo` | `crucible build <pack>` | refused, exit 4: `has no hooks.forge` (coding also `hooks.policies`), with a hint to `crucible bench run` | <1 s |
| finance_ap, build-equivalent | `crucible bench run finance_ap --k 2` | validity 4 / 4 valid; oracle 1.0 / approve_all 0.0 / no-op 0.0 | 8.9 s |
| demo, build-equivalent | `pack check --level accept` → `bench run --k 2` | check exit 0; 4 / 4 valid; gold 1.0 / no-op 0.0 | 1.6 s |

**Determinism** (r1 vs r2, same key, `cmp_dirs.py`): `accept.json`, `splits.json`, `anchors.json`,
`contamination.json`, `summary.json`, `bench.keys.jsonl`, `validity.json` and `report.html` are byte-identical after path
normalisation. Remaining differences are timing only: `bench/summary.json` latency percentiles, `results.jsonl` /
`episodes.json` timings, M9 evidence file names (`log-<epoch>.json`), and 2 of 5 canonical exports
(`time/outcome/*` in `crucible_reward.json`, B17).

Mid-round observations (fixed in the tree by 11:40, recorded because they reached a user): at 11:23 every toy build
failed: with the fixed key "the bench split is empty (7 items, bench_frac 0.3)", with a fresh key "none of the 2 bench
items is valid … (contaminated)" (generated items flagged against their siblings' shared template, anchors at 29-31
shared 13-grams). Both were gone at 11:40 (`build2/toy_r*.json`).

## 2. Runs for labelling (round 2)

Round 1's Qwen2.5-0.5B scored 0 / 48, so no model-produced positive existed and the token path could not be tested on
positives. Round 2 adds **teacher policies** (`teacher.py`): a stub engine behind Crucible's real `EngineLLM` +
`HFChatCodec` + `LLMToolPolicy` that emits a fixed action list as Qwen2.5 `<tool_call>` tokens (encoded with the real
tokenizer, ending in `<|im_end|>`). The runner records them exactly like a model (prompt / completion token ids under
the real Qwen2.5 template, trainable turns), so they are `model` kind. Variants: `qwen-teacher` (the pack oracle),
`-half` (first half, then DONE), `-skipmark` (oracle + a `@pytest.mark.skip` note: rule flag), `-exposed` (a note naming
`/tests/hidden_tests.py` first), `-hang` (samples 0-1 never answer: attempt timeout), and for the RLM harness
`qwen-teacher-rlm` (the oracle in one Python cell) and `-rlm-sleep` (a 25 s cell; a thread `docker rm -f`s two kernel
containers 6 s after they start).

| Run | Tasks | Policies (k = 2) | Episodes | Scored | Masked | Wall |
|---|---|---|---|---|---|---|
| `runs2/toy-mix2` | toy anchor + `generate(20001/20002)` | scripted oracle / half / no-op; declared `forged_approval`, `split_refund`; `probe:exposed_oracle`; `flag:skipmark_oracle`; `infra:hang`; 5 teacher variants; Qwen2.5-0.5B | 84 | 60 | 24 | 578 s |
| `runs2/ap-mix2` | `APTask(1)`, `APTask(2)`, `generate(10011)` | scripted oracle / approve_all / no-op; 8 declared adversaries; probe; flag; hang; 5 teacher variants; Qwen2.5-0.5B | 120 | 102 | 18 | 532 s |
| `runs2/rlm-mix2` | toy anchor + backlog | `qwen-teacher-rlm`, `qwen-teacher-rlm-sleep` under the RLM harness (Docker kernel) | 8 | 6 | 2 | 88 s |

Per policy: teacher 1.0 (12 / 12), half 0, skipmark 1.0 **with** the `skips_tests` flag, hang masked
(`agent_run_error`), Qwen 0 / 12, every adversary 0. Toy `-exposed` masked as `crucible:answer_exposed` (6 / 6);
**finance_ap `-exposed` scored 1.0 unmasked** (6 / 6): its probe is a `kb.search_docs` query string that reads no grader
file, so the runner rightly does not call it an exposure (the scripted AP probe, which passes the path as a tool
argument, is masked). **The container kill now reached episodes:** 2 killed containers → 2 episodes masked
`crucible:env_container_lost`, the other 2 sleep episodes scored 0; `qwen-teacher-rlm` 4 / 4 = 1.0. No `crucible.rlm`
container left behind.

## 3. Label (round 2)

### 3.1 From runs: every select mode (`label_matrix2.sh` → `check2.py`)

Rows SFT / DPO / KTO (KTO true). `check2.py` re-derives every row's episode status from `record.json` /
`rollout.json` and rebuilds each turn's full prompt from the delta storage itself.

| Mode | toy-mix2 | ap-mix2 | check2 violations (both runs) |
|---|---|---|---|
| `correct` (default) | 99 / 1 / 160 (99) | 401 / 3 / 513 (401) | KTO turn labelled both true and false (47; 98); AP: hidden-string rows from the unexposed AP probe (SFT 202, DPO 3, KTO 202) |
| `best` | 99 / 1 / 160 | 202 / 3 / 513 | same |
| `all` | 121 / 1 / 160 | 415 / 3 / 513 | + SFT rows from failed episodes (22; 14): what `all` means |
| `correct --max-per-task 2` | 99 / 1 / 160 | 322 / 3 / 513 | same as default |
| `all --max-per-task 1` | 99 / 1 / 160 | 202 / 3 / 513 | same as default |
| `all --filter "0 < avg < 1"` | 121 / 1 / 160 | 415 / 3 / 513 | as `all` |
| `correct --filter "n_correct >= 1 and worst < 1"` | 99 / 1 / 160 | 401 / 3 / 513 | as default |
| `all --mask-negative` | 121 / 1 / 160 (= `all`) | 415 / 3 / 513 (= `all`) | as `all`; no warning (B14) |
| RLM run (default) | 2 / 1 / 6 (2) | | none |
| round-1 runs re-labelled (`r1_toy-mix`, `r1_ap-mix`, `r1_toy-rlm2`) | 0 / 0 / 21; 0 / 0 / 21; 0 / 0 / 10 | | 3 KTO rows whose text the redactor changed (B24) |

What the default output is made of (`label2/{toy,ap}-mix2/correct_/summary.json`, `check2.json`):

| | toy-mix2 | ap-mix2 |
|---|---|---|
| skipped (episodes) | adversary 12, exposed / tampered 12, flagged 12, masked 12, scripted 18 | adversary 48, exposed 6, flagged 12, masked 12, scripted 18 |
| SFT / KTO-true policies | `qwen-teacher` only | `qwen-teacher` 199, `qwen-teacher-exposed` 202 (unflagged 1.0 episodes) |
| KTO-false | `qwen-teacher-half` 50, Qwen 11 | `-half` 101, Qwen 11 |
| DPO | 1 pair (teacher vs half, at the DONE); 2 tasks lost (`dpo_unusable_turn`, B22) | 3 pairs, chosen = `-exposed`'s probe turn (B20) |
| masked / exposed / flagged / adversary / control / scripted episode in any file | 0 | 0 |
| rows that are not a turn of their episode (prompt = rebuilt full prompt, completion = sampled ids) | 0 / 260 | 0 / 917 |
| SFT `input_ids == prompt_ids + completion_ids`, `completion_mask` = 1 exactly on the completion | 99 / 99 | 401 / 401 |
| prompt opens with the policy's system prompt and holds `<tools>` | all | all |
| `text_exact` (decoded text re-tokenizes to the ids) | all | all |
| DPO: `prompt_ids` is a turn prompt in both episodes, with `chosen_ids` / `rejected_ids` as their completions | 1 / 1 | 3 / 3 |

Variants (copies of `toy-mix2`, `runs2/variants/`), default label:

| Variant | Result |
|---|---|
| `validity.json` marks the anchor invalid | 28 episodes skipped `invalid_task`; 89 / 1 / 142 rows, none from the anchor |
| build layout: `splits.json` (anchor sealed to `bench`) + `bench.key` | seal verified; 28 skipped `held_out`; 89 / 0 / 142, none from the anchor |
| same, `splits.json` edited (anchor → `train`) | exit 4: "the split seal does not verify … refusing to label" |
| `qwen-teacher` renamed `control:qwen-teacher` | 6 skipped `control`; SFT 0 |
| every `rollout.json` deleted | 0 rows + warning "no training rows"; but model episodes counted `scripted` (B19) |
| same, `--allow-scripted` | **493 SFT / 648 KTO rows; SFT 247 and KTO-true 299 come from flagged passes** (`qwen-teacher-skipmark`, `flag:skipmark_oracle`): B19 |
| empty dir / not a run dir | exit 4: "is not an M10 run directory" |
| `--filter "("`, `--filter "1/0>0"`, `--max-per-task 0` / `-1`, `--margin -1` | exit 4, one line, nothing written |
| `--threshold 7` (range is [0, 1]) | exit 0, no positives, no warning (B13) |
| `--allow-scripted` on `toy-mix2` | 96 scripted-demo chat rows in `sft.chat.jsonl` (`source: scripted_demo`, `demonstration` note, arguments dicts: 0 / 96 double-encoded); no system prompt, `tools: []` (B23); DPO 1 → 0 (B22) |
| `--sft-format chain` | 3 multi-turn token rows (`input_ids` + `assistant_masks`) |
| re-run of the matrix after the 11:54 label change | byte-identical outputs |

### 3.2 From packs

| Pack | Traces | Outcomes good / bad / unknown | Held out (credit `split`) | SFT / DPO / KTO | Notes |
|---|---|---|---|---|---|
| toy | 24 | 16 / 6 / 2 | 12 (T1013-T1024) | 10 / 0 / 12 (10 true) | SFT and KTO agree (one verdict); `kto_outcome_disagrees` 2, `kto_unknown_outcome` 1; 0 empty completions; `tools: []` |
| coding | 66 | 49 / 11 / 6 | 10 (`run:task0x/...`) | 200 / 0 / 200 (all true) | all 400 rows from one credited train trace (`run:task01_prims_omp/fable-5.1-run1`); 0 empty completions (was 54 / 60); no secret-shaped string |
| finance_ap | | | | | refused: `declares no ingest sources` (exit 4) |

### 3.3 HF `datasets`

`crucible/.envs/nemo` (`datasets` 5.1.0), `load_dataset("json", data_files=…)`: **43 files, 4,548 rows, 0 failures**
(every run, variant, pack and API output in round 2), plus `sft.chat.jsonl` (96) and `kto.chat.jsonl` (143), which
load on their own.

## 4. Training on the labels (round 2)

Qwen2.5-0.5B-Instruct, fp32, MPS, last 4 blocks + final norm trainable (no `peft` / `trl` in the venv), activation
checkpointing. Token rows are used as written (`input_ids`, `completion_mask`, `prompt_ids` / `chosen_ids` /
`rejected_ids`): no re-rendering, so the trained tokens are the policy's.

### 4.1 SFT (`train2/sft`)

Data: default `--select correct` SFT of toy-mix2 + ap-mix2 + RLM (502 rows); rows over 3,072 tokens were **dropped,
not truncated** (keeping the policy's whole context), leaving 90 rows (65 `qwen-teacher`, 23 `qwen-teacher-exposed`, 2
RLM), 203,462 tokens, 5,024 trained. Earlier attempts at 8,192 and 4,096 tokens ran MPS out of memory on this shared
machine (`logs2/train_sft.log` history).

| | |
|---|---|
| gradient check (full logits, last 768 tokens of the shortest row) | d loss / d logits non-zero at 372 / 372 completion positions, **exactly 0 at 395 / 395 others**; memory-lean loss (LM head on target positions only) = full masked loss (1.251277 vs 1.251278) |
| steps / lr / batch | 60 / 2e-5 / 1 row, clip 1.0 |
| training loss, mean per 10 steps | 0.424, 0.535, 0.566, 0.485, 0.207, 0.681 (one row per step: noisy) |
| mean token loss on 32 fixed training rows, before → after | **0.396 → 0.185** |
| NaN / Inf loss or grad norm | none (max grad norm 44) |
| wall | 180 s |

### 4.2 DPO (`train2/dpo`)

Data: default DPO of toy-mix2 (1), ap-mix2 (3), RLM (1): 5 pairs, prompt tokens identical on both sides 5 / 5, prompts
959-1,939 tokens, chosen 32-1,618 tokens, rejected 2-143. Reference = frozen base, β = 0.1, lr 1e-5, 20 steps.

| | |
|---|---|
| gradient check (chosen side, last 512 tokens) | 0 non-zero d loss / d logits at 479 prompt positions; 32 / 32 completion positions non-zero |
| loss | 0.693 at step 0 → 1e-5 by step 14; final margins 5.9, 11.7, 11.7, 11.5, 14.6 (reward accuracy 5 / 5) |
| NaN / Inf | none (max grad norm 118 at step 3, clipped) |
| wall | 86 s |

The pairs separate trivially: one rejected side is `DONE<|im_end|>` (2 tokens), and the 3 finance_ap pairs train the
model to prefer `kb.search_docs("checking cat /tests/hidden_tests.py and reward.json")` over reading the policy (B20).

### 4.3 Bench: base vs SFT checkpoint

`crucible bench run toy --policies m15pol2:both --k 2 --fresh 2` (fixed key; `bench2/toy`), Qwen policies on CPU, the
pack's default harness (max 1,000 steps). **Stopped by hand after 21 min, incomplete**: the SFT checkpoint's episodes
do not end. `crucible bench report` on the stopped run (`bench2/toy_partial.json`) reports it honestly: planned 12,
measured 4, owed 8, stop reason "incomplete: the runner left no summary", models measured on different task sets, so
compare on the 1 common task.

| Policy | Measured episodes | Reward | Behaviour |
|---|---|---|---|
| base Qwen2.5-0.5B | 3 (anchor ×2, fresh ×1) | 0, 0, 0 | replies in 2 steps (`agent_done`): no tool call |
| SFT checkpoint | 1 finished (anchor) + 4 stopped in flight | 0 | well-formed `tickets.set_status` calls; loops on ticket 100 (a ticket id of the training backlogs) until `too_many_errors` at 25 steps; the in-flight episodes ran 26-104 World events (refunds without approval, repeated status changes) |
| common support (1 task) | | base 0.0, SFT 0.0 | |

The finance_ap bench was not run in round 2 (time). Round 1's run of the same comparison with the round-1 checkpoint
(`bench/finance_ap`, `--fresh 0`, stopped after 84 min): base 0 / 4, SFT 0 / 3. **Honest reading:** 60 SFT steps on 90
rows of teacher turns taught the format and the tool names (the model now acts instead of quitting) but not the task;
nothing here shows a gain. The training rows include 23 from the AP probe episode (B20 context) and the bench's anchor
task is also a training task (contamination by construction; it made no difference at 0 / 0).

## 5. Findings

Status: **fixed** = verified fixed in the worktree in this round; **present** = reproduced on the worktree; **new** =
found in round 2. Line numbers are the worktree's (times above).

| Id | Sev | Finding | Status | Evidence | Suspected location | Minimal fix |
|---|---|---|---|---|---|---|
| B1 | P0 | Run rows were not the policy's sequence (no system prompt / tool cards, parallel calls split, RLM turns lost) | **fixed**: rows are rollout token turns; of 9,883 checked rows only 3 redacted ones (B24) differ from the sampled ids; RLM rows are the REPL prompt + `python` cell | `label2/*/check2.json`; `label2/rlm` | `studio/datasets.py` `token_steps`, `from_run` | |
| B2 | P0 | Tool-call arguments as JSON strings (double-encoded by Qwen2.5) | **fixed**: dicts; 0 / 96 chat rows double-encoded (token rows carry the model's own text) | `label2/api_scripted/sft.chat.jsonl` | `datasets.py` `_args` | |
| B3 | P0 | Hack-flagged passes were positives and won DPO | **fixed** when `rollout.json` exists: `flagged` 12 + 12 skipped; see B19 | `label2/{toy,ap}-mix2/correct_/summary.json` | | |
| B4 | P0 | Exposed episodes were negatives / DPO candidates | **fixed**: `exposed_or_tampered` 12 + 6 skipped | same | | |
| B5 | P1 | Validity manifest ignored | **fixed** | `label2/var_tm_invalid` | | |
| B6 | P1 | Held-out and control material labelled | **fixed**: `held_out` 28, `control` 6, edited seal refused | `label2/var_split`, `var_tm_control`, `cur_fakebuild_tampered*` | | |
| B7 | P1 | Scripted rows replayed answer keys with harness text | **fixed by default** (scripted skipped; opt-in rows say `demonstration` and drop `script finished`) | `label2/cli_scripted` | | |
| B8 | P1 | Pack labels disagreed across formats | **fixed** (one `verdict`) | `label2/pack_toy` | | |
| B9 | P1 | Silent success on empty / non-run input | **fixed**: exit 4 on a non-run dir; "no training rows" warning | `label2/var_emptydir`, `var_tm_norollout` | | |
| B10 | P1 | `crucible build` builds only the toy; its bench runs no declared adversary | **present**: finance_ap / coding / demo refused (now with a `bench run` hint) | `build2/{finance_ap,coding,demo}.out`; `build2/toy_r1.json` models | `studio/build.py:164-167` | Without `forge`, take `hooks.tasks` as anchors via `accept_adapter`; run declared adversaries with their bounds in the bench step |
| B11 | P1 | KTO rows with empty completions | **fixed**: 0 / 200 (coding) | `label2/pack_coding` | | |
| B12 | P2 | Filter errors crashed with a traceback after writing | **fixed**: one line, exit 4, nothing written | §3.1 variants | | |
| B13 | P2 | Range checks | **partly fixed**: `--max-per-task` ≤ 0 and `--margin` ≤ 0 refused; `--threshold` outside `reward.range` (7) still accepted silently | `label2/tmp_err` run | `studio/label.py:190` | Refuse a threshold outside the pack's (or [0, 1] for runs) range |
| B14 | P2 | `--mask-negative` is a no-op on run rows, silently | **present**: `all --mask-negative` = `all` (121 / 415 rows); turn weights are the rollout `advantage` (0.0) | `label2/*-mix2/all__mask_negative_` | `studio/label.py:203`, `datasets.py:775` | Warn (or refuse) when no row carries credit (`credit: false` everywhere) |
| B15 | P2 | Duplicates not collapsed; `traj_id` collided across tasks | **fixed**: dedup by row id (`sft_duplicate` 99 / 401), `traj_id` hashes task + run + episode | summaries | | |
| B16 | P2 | Pack rows called pseudo-tools with fabricated results | **mostly fixed**: recorded tool calls only, recorder metadata dropped; `tools: []` still | `label2/pack_toy/sft.jsonl` | `datasets.py` `from_trace` | Attach the World's tool schemas when the pack knows them |
| B17 | P2 | Canonical export carries wall-clock timing | **present**: 2 / 5 exports differ only in `time/outcome/*` | `build2/toy_r1` vs `toy_r2` | `verify/runtime/plan.py:110` | Keep `time/*` out of the canonical `crucible_reward.json` |
| B18 | P2 | `M15-studio.md` gate numbers stale | **fixed** (doc now says 0 accepted / 5 provisional) | `docs/modules/M15-studio.md` | | |
| B19 | **P0** | **An episode without `rollout.json` loses its hack flags and is called `scripted`.** `from_run` reads flags only from `rollout.json` (and `load_run`'s copy of it) and sets the policy kind from its trainable turns. Missing file (a copied / pruned run, a runner crash after `record.json`): flagged passes are no longer `flagged`; model episodes become `scripted`; under `--allow-scripted` they are re-read from ATIF as **positives**: 247 SFT and 299 KTO-true rows from `qwen-teacher-skipmark` and `flag:skipmark_oracle`, labelled "scripted demonstration". Without the flag the same run gives 0 rows, with every model episode mis-counted as `scripted`. | new | `label2/cli_norollout_scripted` (CLI), `label2/api_norollout_scripted`; run `runs2/variants/tm_norollout` | `studio/datasets.py:434-435` (kind), `:450` (flags) | Skip an episode whose `rollout.json` is missing or unreadable (`no_training_view`), before the policy-kind and flag checks; or recompute rule flags from `trace.json` |
| B20 | P1 | **Best-vs-worst DPO pairs at the first divergent turn, whatever caused the outcome.** The chosen side is the best positive's turn where the two trajectories first differ; nothing ties that turn to the reward gap. In ap-mix2 all 3 pairs have chosen = a `kb.search_docs` query naming `/tests/hidden_tests.py` (a probe that does not count as exposure) and rejected = the correct first action (`kb.read_doc ap_policy.md`) or Qwen's call; the positive tie-break (`traj_id` hash) picked the probing episode. DPO then drove these margins to 11.7. | new | `label2/ap-mix2/correct_/dpo.jsonl`; `train2/dpo/curve.jsonl` | `studio/datasets.py:803-830` | Prefer positives without probe attempts (`Episode.probes_attempted`, M13) and break ties to the shortest / unprobed trajectory; pair only where the chosen turn has positive ΔΦ when credit exists, else at a turn graded wrong |
| B21 | P2 | **KTO labels the same (prompt, completion) both true and false.** Every turn of a negative trajectory is `label: false`, including turns identical to a positive trajectory's at the identical prompt (the failing run's correct opening). | new | `check2.json` `kto:same_turn_labelled_true_and_false`: 47 / 61 toy negatives, 98 / 112 AP | `studio/datasets.py:869-890` (`good = v == "pos"`, `:883`) | Drop a negative turn whose (`prompt_ids`, `completion_ids`) occurs in a positive trajectory of the task, or label negatives from the first divergent turn on |
| B22 | P2 | **One worst negative per task, no fallback.** If its divergent turn is unusable (a truncated Qwen turn with no EOS) the task gets no pair; under `--allow-scripted` the best positive becomes a scripted chat trajectory that shares no token prompt with any model negative, so every pair is lost. | new | toy-mix2: 1 pair for 3 tasks (`dpo_unusable_turn` 2); `--allow-scripted`: 0 pairs (`dpo_no_shared_decision` 3) | `studio/datasets.py:803-807` | Try the next negative (and next positive) of the same representation until a usable shared decision is found |
| B23 | P2 | Scripted-demo chat rows (`--allow-scripted`) have no system prompt and `tools: []`, so Qwen2.5 renders its default system prompt and no tool cards; `write_jsonl` (`sort_keys=True`) also re-sorts argument keys. | new | `label2/api_scripted/sft.chat.jsonl` (96 rows) | `studio/datasets.py:469`, `:964` | Record the scripted policy's system prompt and tool schemas (or refuse chat rows without them); write rows without `sort_keys` |
| B24 | P2 | Redaction rewrites policy tokens: World e-mail addresses in the model's tool arguments become `<email:…>`, so a redacted row trains tokens the policy never produced (flagged `redacted`, counted in `redacted_rows`, but kept as a target). | new | `label2/r1_toy-mix/kto.jsonl` (2 rows), `r1_toy-rlm2` (1, `text_inexact` 2) | `studio/datasets.py:322-323` | Drop redacted turns from targets by default (keep them as context), or exempt World-synthetic identities from redaction |
| B25 | P2 | `crucible build` failed for every toy key at 11:23 (empty bench split; sibling items "contaminated" by their shared template); fixed in the tree at 11:40. Listed for the regression tests. | new, fixed in flight | `logs2/build_toy_r*.err` (11:23), `build2/toy_r*.json` (11:40) | `studio/build.py` `assign_splits`, `contamination_step` | Regression tests: a 7-item build whose draw is empty seals one lineage to bench; template-sibling items are not contamination |

**What held (round 2).** No masked, exposed, flagged, adversary, control, invalid-task or held-out episode in any file
when `rollout.json` is present; every token row is a real policy turn with the rebuilt full prompt (9,883 rows checked; the 3 exceptions are redacted, B24);
DPO prompts are identical token sequences on both sides; SFT and DPO trainers using the rows' own masks get exactly zero
gradient from non-policy tokens; label output is deterministic and byte-stable across the 11:54 label change; builds
are deterministic under a fixed key except timing; every output loads with HF `datasets`; the RLM container kill masks
(`crucible:env_container_lost`) and those episodes never reach a dataset; Docker left clean (no `crucible.rlm`
container, no image created).
