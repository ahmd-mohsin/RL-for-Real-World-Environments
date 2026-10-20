# 09. The Tinker bridge: Tinker trains the student and samples every non-Claude model

Date: 2026-10-20. Code: `src/crucible/train/bridges/tinker/`, `src/crucible/factory/tinker.py`, `scripts/tinker_rl.py`.
Tests: `tests/unit/test_tinker_bridge.py`. Upstream read: the Tinker docs (losses, RL, checkpoints, models and pricing,
rendering, OpenAI-compatible API), the `tinker` SDK 0.34.0 (Apache-2.0) and tinker-cookbook `ad3b5c5` (Apache-2.0).

The user's decisions of 2026-10-20 set the scope:

- the student we build tasks for and post-train is **`Qwen/Qwen3.6-35B-A3B`** (64K, and `:peft:262144` for long
  horizons); Qwen3.5-4B is out of the plan;
- Tinker replaces OpenRouter for everything that is not Claude: the difficulty ladder and the cross-family roles
  (solver, reviewer, hacker, auditor, judges) sample Tinker models through the same bridge;
- larger students follow (Qwen3.5-9B, Qwen3.5/3.6-35B-A3B, maybe Qwen3.8-27B), so nothing here is tied to one model.

## 1. Shape

Crucible keeps the environment, the renderer, the credit and the accounting. Tinker gets token ids, sampling logprobs
and finished per-token advantages, and returns gradients steps and sampler weights.

| module | role |
|---|---|
| `models` | registry: Tinker id, family, context, tool format, thinking switch, stop tokens, the pinned price table |
| `spend` | `SpendMeter`: per-call tokens (cached and uncached prefill, sample, train, checkpoints), a hard cap, `spend.jsonl` receipts |
| `backend` | the narrow service interface; `RealTinker` over the SDK; the infra-vs-task error split with backoff |
| `fake` | `FakeTinker`: the same interface in memory (scripted sampling, the documented losses, checkpoints) |
| `codec` | `TinkerChatCodec`: the model's HF template with exact TITO, the family's tool parser, thinking explicit |
| `sampler` | `TinkerSampler` (engine), `SamplerLLM` (the policy's client), `TinkerChatClient` (a factory role), `make` (the factory backend) |
| `datums` | records -> datums: outcome advantage, honesty, M6 turn credit, one datum per TITO sequence |
| `trainer` | `TinkerTrainer`: groups, filters, update, refresh, resume, cap |
| `factory/tinker.py` | `train`, `benchmark`, `compare`, `check_protocol`, `download_adapter` |

Layering holds (`lint-imports`: 4 contracts kept): `train` never imports `factory` or `bench`; the factory module joins
`make_env`, the trainer and `bench.lh_metrics`.

## 2. Models and prices

`models.MODELS` holds every model the docs' "Models & Pricing" page listed on 2026-10-20 (`PRICES_AS_OF`), with
prefill, cached prefill, sample and train USD per 1M tokens. A `:peft:<n>` variant is its own row (same prefill and
sample price, its own train price). Where the page showed a limited-time 50% discount (the Nemotron rows) the pinned
price is the discounted one and `list_prices` keeps the list price. A model missing from the table is refused: it
could not be capped.

| model (Tinker id) | family | context | prefill / cached / sample / train | tool format |
|---|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` (target) | qwen | 64K | 0.45 / 0.09 / 1.335 / 1.177 | qwen_xml |
| `Qwen/Qwen3.6-35B-A3B:peft:262144` | qwen | 256K | 0.45 / 0.09 / 1.335 / 2.14 | qwen_xml |
| `Qwen/Qwen3.5-9B` | qwen | 64K | 0.30 / 0.06 / 0.807 / 1.463 | qwen_xml |
| `Qwen/Qwen3.8-27B` (+256K, train 7.46) | qwen | 64K | 0.6975 / 0.1395 / 5.595 / 4.103 | qwen_xml |
| `Qwen/Qwen3.5-397B-A17B` (+256K, train 12.00) | qwen | 64K | 1.65 / 0.33 / 7.50 / 6.60 | qwen_xml |
| `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` | deepseek | 256K | 0.60 / 0.12 / 3.60 / 4.98 | dsml |
| `zai-org/GLM-5.3:peft:262144` | glm | 256K | 3.78 / 0.756 / 12.15 / 14.58 | glm |
| `zai-org/GLM-5.3-Flash:peft:262144` | glm | 256K | 0.84 / 0.168 / 2.11 / 5.58 | glm |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (+256K, train 2.32) | nemotron | 64K | 0.57 / 0.114 / 1.44 / 1.276 (discounted) | qwen_xml |
| `moonshotai/Kimi-K2.6` (+128K, train 15.40) | kimi | 32K | 2.205 / 0.441 / 5.49 / 4.84 | text protocol |
| `openai/gpt-oss-120b` (+128K, train 2.33) | openai | 32K | 0.33 / 0.066 / 0.84 / 0.737 | text protocol (harmony) |

The full table (Inkling, Nemotron Ultra/Lightning, Qwen3-8B, gpt-oss-20b, DeepSeek-V3.1) is in `models.py`.
`with_context(name, tokens)` picks the cheapest variant that holds an episode cap and refuses when none does
(Qwen3.5-9B has no 256K variant); `check_context` refuses an episode cap plus one reply that overflows the model.
`spec()` is the factory's `ModelSpec` table (`provider="tinker"`, explicit `family`: the factory's `family_of` would read
`zai-org` as its own family, not `glm`).

## 3. Our renderer, per family

`TinkerChatCodec` is `train.chat.HFChatCodec` on the model's own tokenizer and chat template: exact TITO (turn k+1's
prompt is prompt_k + completion_k (+ EOS) + the template's framing of the new messages; the model's own tokens are never
re-tokenized), driven by `train.world_env.EngineLLM`. What it adds, each found on the first contact:

1. **Tool-call arguments as mappings.** The Qwen3.5/3.6/3.8 templates iterate `arguments|items`; the policy stores a JSON
   string, so every history with a tool call raised `TypeError`. The codec decodes them before rendering.
2. **The family's tool format** (`tool_format`): `qwen_xml` (`<tool_call><function=f><parameter=p>v</parameter>...`:
   Qwen3.5/3.6/3.8 and Nemotron 3), `dsml` (DeepSeek V4.1 `<｜DSML｜ calls>`), `glm` (`<arg_key>`/`<arg_value>`),
   `hermes` (Qwen3). The parsers are ported from tinker-cookbook (Apache-2.0, `THIRD_PARTY.md`). `HFChatCodec.parse` only
   knew Hermes JSON, so no Qwen3.5 tool call would have run. A malformed or cut-off block becomes a call with
   `__unparseable__` arguments: the World answers it, it is never dropped silently.
3. **Turn ends.** GLM-5.3 has no per-message end token (the next role token ends a turn: `<|user|>`, `<|observation|>`,
   `<|endoftext|>`, its `generation_config` eos list): `stop_tokens`. gpt-oss (harmony) closes history messages with
   `<|end|>` but stops sampling at `<|return|>` / `<|call|>`: `turn_end_tokens` frames the TITO suffix without stopping
   sampling, and the parser reads the `final` channel.
4. **Thinking.** `enable_thinking` goes to the template as the model's switch, default off, recorded in every report
   (`codec.settings()`). A template without a switch (GLM-5.3, gpt-oss: always reasoning) gets `reasoning_effort=low`
   when thinking is off (the cookbook's closest non-thinking mode). With reasoning on, the `<think>` block is removed from
   the parsed text; the tokens, which are what trains, keep it.
5. **The text protocol** (`tool_format="text"`: Kimi, gpt-oss, base models): the tools are declared in the system
   message with a Hermes JSON convention, earlier calls render as that JSON, tool results as user turns.

The probe (`scripts/tinker_probe_templates.py`, $0, every model's real tokenizer) renders a history with a tool call,
computes the TITO suffix and parses the template's own rendering of the call back:

| model | round trip | template history behaviour | turn end |
|---|---|---|---|
| Qwen3.6-35B-A3B, Qwen3.5-397B-A17B | ok | interleaved (reasoning dropped at a new user turn) | `<|im_end|>` |
| DeepSeek-V4.1-Flash | ok | preserved | `<｜end▁of▁sentence｜>` |
| GLM-5.3-Flash | ok (after `stop_tokens`; before: the suffix could not be computed) | preserved | `<|user|>` `<|observation|>` `<|endoftext|>` |
| Nemotron-3-Super-120B | ok | interleaved | `<|im_end|>` |
| Kimi-K2.6 (text protocol) | ok | preserved | `<|im_end|>`, `[EOS]` |
| gpt-oss-120b (text protocol) | ok (after `turn_end_tokens`; before: no suffix) | preserved | `<|return|>` `<|call|>` (history `<|end|>`) |

Kimi's template opens `<think>` whatever `enable_thinking` says (its switch has another name): open item.

## 4. The spend meter and the cap

Every call books tokens and dollars from the pinned table in `spend.jsonl` (kind, model, tokens, USD, cumulative USD,
the sampler version, the episode). Prefill is split cached / uncached by the service's own count
(`SampleResponse.prompt_cache_hit_tokens`); a response without it is booked uncached. Train calls book the datums'
input tokens at the model variant's train rate. Checkpoints are counted; storage (USD 0.10 per GB-month) is listed as
not metered.

The cap is hard: before a call the meter holds its worst case (the prompt uncached plus `max_tokens` sampled; a train
call's tokens) and refuses with `TinkerCapReached` when spent + held + this call would pass it. Calls in flight count.
Inside an episode the refusal is `BudgetExceeded` (`crucible:credits_exhausted`: masked, never a 0); the trainer stops
with `stop="cap"` and never runs an optimizer step on a batch whose train call was refused. A meter re-opened on its file
starts from the booked total (one cap across restarts). Each client also keeps its own USD, so a `ModelBook` that sums
its clients does not double count a shared meter.

## 5. Training (`TinkerTrainer`)

One step: groups of `group_size` attempts of one task from `make_env` (native harness, checkpoint sidecar,
`InfraWatch`, honesty, tool limit, two timeouts) run by the `Runner` in train mode -> filters -> datums -> update ->
refresh.

- **Filters**, the slime bridge's rules: `objective_changed` (the task's verifier version differs from the pinned
  objective: drop and stop), `infra_in_group` (any masked member: the whole group dropped and refilled), `stale` (more
  than `max_staleness` optimizer steps behind: dropped, checked on arrival and again when the batch is built),
  `zero_spread` (optional). Every drop is a `drops.jsonl` row; a step that cannot fill its batch within
  `max_groups_per_step` launches stops the run (`too_many_drops`) instead of looping. Attempts are not retried and the
  Runner's breakers are off inside a group (a breaker deferral re-ran a masked member five times in the first smoke).
- **Credit** (`datums`): R is the task's own reward (0.5·Φ_final + 0.5·[all gates]); advantage R − mean over the group's
  live rollouts (Dr.GRPO), the honesty term composed in (`compose_advantage`: a failure never turns positive), M6's
  per-turn credit through `train.credit.credited_records` (the one rule, commit 64a30ba; refused when no live rollout
  recorded progress), one datum per TITO sequence of `common.flatten`, shifted to next-token form (cookbook
  `trajectory_to_data`, PATTERN). Tokens outside the policy's completions carry advantage 0: Tinker's RL losses take no
  mask. `loss_scale`: `sum` (Tinker's) or `token_mean`.
- **Update**: `forward_backward(loss_fn)` with `importance_sampling` (default), `ppo` or `cispo` and `loss_fn_config`
  (PPO's clip thresholds), `optim_step(Adam, lr 4e-5 by default)`, `save_weights_for_sampler` (version = step, the path
  table in `versions.json`), `save_state` every `save_every` steps.
- **Off-policy bound**: `async_groups` extra groups sample while a step trains; `max_staleness` bounds them.
- **Resume**: `config.json` (a different configuration is refused, except `steps` and `cap_usd`), `state.json` (after each
  `save_state`), the training client restored with its optimizer
  (`create_training_client_from_state_with_optimizer`), the sampler on that step's weights.
- **Model and band**: any Tinker id with a price row; the 64K / 256K choice is the id itself and the meter books that
  variant's train price. A run is refused unless every task has a targeting record for this base model with its band
  measured and in band (`check_band`), and unless every task is a training task (V9).

**Equivalence** (the P6 scripted solvers, `test_native_tinker_and_slime_agree_*`): the native run through
`TinkerSampler` (exact TITO over a byte codec, the fake service) and the slime generate path give identical rewards and
masking, and on every call's tokens the Tinker datum carries exactly slime's per-token advantage (`rewards` +
`turn_advantages` of `slime.credit.convert`); the importance-sampling loss is the same number on both sides.

## 6. The benchmark and its protocol

`benchmark(tasks, model=..., checkpoint=..., k=8, seed=..., purpose=...)` writes `bench.lh_metrics.summarize` (pass@1,
pass^k, final Φ, milestones lost, calibration, give-up, hacks, cost per solved task) plus unbiased pass@k and pass^k for
k = 1, 2, 4, .. k, task-clustered bootstrap CIs (tasks resampled, attempts kept together), USD per episode and per
passing episode (the meter attributes every dollar to the episode that spent it), and the arm's settings.

The protocol (the user's target: the student under 30% before post-training, 50-60% after):

1. a **selection** run picks tasks the base model fails;
2. a separate **fresh baseline** with a new seed is the honest pre-score (selected tasks regress upward on re-test);
3. `compare` re-runs the **base model (control)** beside the **trained adapter** at the same settings and seed, pass@1
   over k = 8 per task with task-clustered CIs, and the paired trained − control difference with its CI and MDE
   (`lh_metrics.paired`);
4. `check_protocol` lists every violation: a baseline that reused the selection seed, a base arm with a checkpoint, a
   trained arm without one, arms whose harness, tool cap, context, thinking setting, sampling parameters or task list
   differ.

## 7. Factory roles on Tinker

`sampler.make(model, role=..., thinking=...)` is the factory's sampler backend (`factory.models.register_backend("tinker",
make, final=True)`, wired at merge): a `TinkerChatClient` with `complete` (tools, TITO conversations per
`sample_key`, so agentic roles run as `LLMToolPolicy` sessions), `complete_json` (the reply parsed, checked by an optional
schema: required keys, a pydantic model or a callable; a bad reply is retried with the error fed back), `spend`,
`lock_settings`, `family`. The process meter is `CRUCIBLE_TINKER_CAP_USD` (default 1 USD).

The OpenAI-compatible endpoint (beta) addresses sampler checkpoints only and is documented as a tool to "poke at" a
model during training; JSON roles use the SamplingClient path.

**Errors**: a connection error, a timeout, a 5xx, a 408/409/429 or a Tinker `RequestFailedError` is retried with capped,
jittered backoff (Retry-After honoured), then raised as `LLMError` (`provider_unavailable`: infra, masked). A 402 or
`BillingError` is `BudgetExceeded`. A 400 that says the prompt is too long is `ContextOverflow` (the episode ends
truncated, scored). Any other 4xx is `RequestRejected`.

## 7b. Running it

From `crucible/` (the key: `set -a; . ../.env; set +a`, or `--env-file ../.env`, read for `TINKER_API_KEY` only):

```
# RL post-training on task rows, with this model's targeting records (resumable: rerun the same command)
.venv/bin/python scripts/tinker_rl.py train --rows rows.jsonl --prompts PROMPTS --targeting targeting.json \
    --model Qwen/Qwen3.6-35B-A3B --group 8 --groups-per-step 2 --steps 100 --cap 200 --out data/tinker/run1
# benchmark protocol: selection, fresh baseline, then control vs trained
.venv/bin/python scripts/tinker_rl.py eval --rows sealed.jsonl --model Qwen/Qwen3.6-35B-A3B --k 8 --purpose selection --seed 0 --cap 20 --out data/tinker/sel
.venv/bin/python scripts/tinker_rl.py eval --rows sealed.jsonl --model Qwen/Qwen3.6-35B-A3B --k 8 --purpose baseline --seed 1 --cap 20 --out data/tinker/base
.venv/bin/python scripts/tinker_rl.py compare --rows sealed.jsonl --model Qwen/Qwen3.6-35B-A3B --checkpoint tinker://.../sampler_weights/run1-s000100 \
    --k 8 --seed 2 --baseline data/tinker/base/metrics.json --selection data/tinker/sel/metrics.json --cap 40 --out data/tinker/cmp
# smokes
.venv/bin/python scripts/tinker_rl.py smoke-rl --model Qwen/Qwen3.6-35B-A3B --cap 1.5
.venv/bin/python scripts/tinker_rl.py smoke-tools --model deepseek-ai/DeepSeek-V4.1-Flash:peft:262144 --cap 0.25
```

Tests (no network): `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_tinker_bridge.py` (the live test runs
only with `CRUCIBLE_TINKER_LIVE=1` and the key in the environment; marker `tinker`).

## 8. Live smoke (2026-10-20; cap USD 5; spent USD 0.26)

The task is the P6 toy window (three milestones, m03 requires m02) in a Docker World (`--network none`, image
`crucible-lh-w:parse-6c176c2fa980`) and graded in a fresh grading container; tool limit 16, `max_tokens` 1024, thinking
off, temperature 1.0. Every number is in `runs/tinker_smoke.json` (run dirs, receipts and `tinker://` paths in
`data/tinker/`, gitignored). Containers are named `crucible-lh-world-tinker-*` / `crucible-m17-tinker-*` and removed by
name.

**RL on the target, `Qwen/Qwen3.6-35B-A3B`** (LoRA rank 32, group 4, 2 optimizer steps, `importance_sampling`, lr 4e-5,
eta 0.1; `smoke-rl`): done, no drop.

| step | rewards (group of 4) | datums | train tokens (policy tokens) | loss:sum | train + save s | sampler |
|---|---|---|---|---|---|---|
| 1 | 1.0, 0.0, 0.167, 0.0 | 10 | 22,015 (3,755) | 38.6 | 11.4 | v0 (base) -> `.../sampler_weights/run-s000001` |
| 2 | 0.167, 0.167, 0.0, 1.0 | 11 | 24,462 (4,003) | 157.7 | 19.2 | v1 -> `.../run-s000002` |

- Step 2's tokens were all sampled by step 1's weights (`min_versions` [1]): the refresh works. The MoE path reports
  its router metrics (`e_frac_with_tokens` 0.997). `save_state` checkpoints were written at both steps.
- Spend USD 0.089 for 8 episodes (sample 0.034, train 0.055): **USD 0.011 per trained episode**. 78 model turns:
  latency per turn mean 2.9 s, p50 2.0 s, max 7.7 s; 34 sampled tokens/s per call; 77% of prefill tokens were cache hits.
- 121 tool calls, 0 unparseable, 1 failed (the World's answer to a wrong path).
- The final adapter downloaded: `adapter_model.safetensors` 2.25 GB (rank 32 over the experts) + `adapter_config.json`.
  Storage is billed per GB-month: intermediate sampler weights now expire after 3 days (`sampler_ttl_s`).
- **Sampler and learner agree exactly** (`scripts/tinker_logprob_check.py`, USD 0.005): a forward pass of a fresh LoRA client (adapter at zero) on a base-model rollout
  returned the sampling logprobs bit for bit on all 633 policy tokens, with the sampling logprobs *sent as zeros* (so
  not an echo). `kl_sample_vs_train` was 0.0 at both steps. The importance ratio starts at exactly 1 on Tinker.
- The OpenAI-compatible endpoint (beta) on the trained sampler path answered a chat completion with a parsed tool call
  (`shell {"command": "ls"}`) in 6.2 s, about 3x the token path.

**Tool-call checks, one sample-only episode each** (`smoke-tools`):

| model | outcome | tool calls (unparseable) | turns | USD / episode | latency / turn (mean, max) | tokens/s per call | cached prefill |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-397B-A17B:peft:262144 | pass, Φ 1 | 15 (0) | 12 | 0.0212 | 1.7 s, 2.6 s | 54 | 81% |
| deepseek-ai/DeepSeek-V4.1-Flash:peft:262144 | pass, Φ 1 | 13 (0), parallel DSML calls | 6 | 0.0067 | 2.4 s, 2.5 s | 83 | 74% |
| zai-org/GLM-5.3-Flash:peft:262144 (`reasoning_effort` low) | pass, Φ 1 | 8 (0) | 7 | 0.0036 | 2.4 s, 3.4 s | 29 | 80% |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144 | tool limit, Φ 1/3, R 0.167 | 16 (0) | 17 | 0.0083 | 2.4 s, 3.4 s | 30 | 84% |

Kimi-K2.6 and gpt-oss-120b passed the template probe (§3) through the text protocol but were not run live.

**Concurrency**: bursts of 4, 48 and 256 concurrent short sample calls (Qwen3.6-35B-A3B; 4 and 64 earlier) all succeeded
with no retry; p50 latency 1.6 s at 48 and 2.2 s at 256. No rate limit was hit; the backoff path (§7) has not fired live.

**What broke, and is fixed** (both in shared code, each with a regression test that fails without the fix):

1. `EngineLLM` (one instance serves every concurrent episode of a sampler) read a call's `_rerendered` flag after the
   `gen` await; another episode's call reset it meanwhile, so a re-rendered prompt lost its new-segment flag, the record
   saw a prefix break, and train mode masked the episode (`crucible:capture_incomplete`).
2. `LLMToolPolicy._final_status_turn` ignored the engine's new-segment flag. The tool limit's notes are user turns, so
   on an "interleaved" template (Qwen3.5/3.6) the final STATUS turn always re-renders the history: every
   limit-stopped episode was masked. Before the fixes 3 of 4 episodes per group were masked and the trainer stopped with
   `too_many_drops`; after them, 0 of 8.
3. The Runner's outage breakers deferred and re-ran a masked group member five times past `max_attempts=1`; inside a
   group the breakers are off (the trainer drops and refills the group).

The first smokes ran on Qwen3.5-4B before the user removed it from the plan (USD 0.12 of the 0.26).

## 9. Projected costs (Qwen/Qwen3.6-35B-A3B)

The toy episode is short (10 turns, about 2k prompt tokens per turn). A long-horizon debugging episode is assumed at
60 turns, a context growing to 40k tokens (20k prompt tokens per turn on average), 250 sampled tokens per turn, 85% of
prefill cached (77-84% measured on the toy, rising with longer shared prefixes) and 40k trained tokens (the TITO sequence
of the whole episode):

| per episode | 64K (`Qwen/Qwen3.6-35B-A3B`) | 256K (`:peft:262144`) |
|---|---|---|
| prefill, uncached 0.18M x 0.45 | 0.081 | 0.081 |
| prefill, cached 1.02M x 0.09 | 0.092 | 0.092 |
| sample 15k x 1.335 | 0.020 | 0.020 |
| train 40k x 1.177 / 2.14 | 0.047 | 0.086 |
| **total** | **0.24** | **0.28** |

- **Full RL run, 20 tasks, about 3,000 episodes**: USD 720 (64K) to 840 (256K), plus checkpoint storage: a 2.25 GB
  sampler save per step (about 190 steps at 16 episodes per step) and a `save_state` every `save_every` steps, held 3
  and 14 days: roughly USD 20-30. Doubling the turn count or the context roughly doubles it; the measured toy cost
  (USD 0.011 per trained episode) is the floor.
- **Benchmark pass, 10 tasks x k=8** (80 episodes, sample only, USD 0.19 each): about **USD 15 per arm**; the protocol's
  four arms (selection, fresh baseline, control, trained) about USD 60, the selection arm more if it screens a larger
  pool.
- The 4-step cost chain to watch: prefill dominates (72% of an episode), so the prefix cache share is the lever;
  `spend.jsonl` records it per call.

## 10. Open

- Live: resume from a real `save_state` (tested with the fake only), the backoff on a real outage, Kimi-K2.6 and
  gpt-oss live (their native tool formats are not ported: the text protocol stands in; Kimi opens `<think>` whatever
  `enable_thinking` says).
- `uv.lock` was not regenerated in the worktree (its sources point at local clones that only the main checkout has):
  run `uv lock` on main for the `tinker` extra.
- The registration line `factory.models.register_backend("tinker", sampler.make, final=True)` belongs to the
  factory's owner (wired at merge).
- The smoke's toy task is built from the unit-test fixtures (`tests/unit/test_m17_w_*`); real runs take task rows
  (`--rows`) and a targeting file (`--targeting`).
