# M12 study, hunt H3: defects in the rollout plane, token fidelity and the weight plane

Scope: `C/train/{engine, chat, world_env, envs, rollout, hf_engine, vllm_http, vllm_ext, sglang_http, gateway, workers, launch, weights, weightsync, teachers, vision}.py`, `C/actors/policy.py` (`LLMToolPolicy`, the `EngineLLM` contract), `C/actors/llm.py` (`LLMProvider.tito`), and `C/core/schema/rollout.py` (`crucible.rollout/v1`, `valid_for_training`).

Paths: `C` = `crucible/src/crucible/`. `S` = the session scratchpad `…/54b8300a-…/scratchpad/m12/h3/`. Each reproducer has a `.out` next to it.

Method: a read-only study on HEAD `3b522cb`, using `crucible/.venv` (torch 2.14, transformers 5.19) offline. **[ran]** means a probe reproduced the defect. **[read]** means the code alone establishes it.

Real tokenizers and chat templates used: Qwen3-0.6B (with and without `enable_thinking=False`), Qwen3-4B-Instruct-2507, Qwen2.5-0.5B-Instruct and gemma-2-2b-it. Qwen3-0.6B weights are not cached (tokenizer only), so the KV and engine checks ran on the real Qwen2.5-0.5B-Instruct weights on CPU.

Engine-server behaviour comes from the frameworks in `RL-Environments/06-frameworks-and-env-code`:
- vLLM processed logprobs are renormalized after the top-p mask (OpenEnv `capture/upstream.py:127-135`).
- SGLang `routed_experts` is base64 int32 with `len(tokens) - 1` rows, and a resumed call passes `routed_experts_start_len` (slime `utils/types.py:429-470`, `rollout/sglang_rollout.py:190-238`).
- SGLang's seed key is `sampling_seed` (slime `sglang_rollout.py:409`).
- SkyRL pauses vLLM before every weight sync (`weight_sync/control_plane.py:150`).

No GPU, Docker, network or paid call was used. Nothing under `crucible/src` or `crucible/tests` was edited.

## Findings

| id | sev | file:line | defect | evidence |
|---|---|---|---|---|
| V1 | **P0** | `C/train/world_env.py:77-90` | `EngineLLM` picks the conversation to extend by message-list prefix plus "next message is an assistant". It never checks that the assistant turn is the reply that call produced. One `EngineLLM` (one per `LLMClient` under `LLMProvider.tito`, documented to serve many episodes at once) splices episode A's sampled completion into episode B's next prompt whenever both started from the same messages. That covers every GRPO group and every repeated seed. | [ran] `S/v1_tito_splice.py`: B's turn-2 prompt ends `{"x": 1}</tool_call>tool\nresult for B`. That is A's tool call followed by B's tool result. "B prompt extends B's own sample: False / A's: True", and `prefix_breaks: 0`, so it happens silently. |
| V2 | **P0** | `C/train/vllm_http.py:129-133` | `VLLMServerEngine.load_weights` swaps the server's weights while calls are running. It does not pause, abort or drain. `disaggregated_stream` and the resident group of `elastic` publish this way (`runtime.py:593`, `:735`). vLLM applies `/collective_rpc` between engine steps, so running requests continue on the new weights. The response is still stamped with the version pinned at call start. M12 doc §2 promises "in-flight calls finish on their pinned version". | [ran, simulated server] `S/v5_vllm_publish_inflight.py`: completion `[100×4, 200×6]` (old then new weights) with `stamped version: 0`. |
| V3 | P1 | `C/train/vllm_http.py:136-158` | `_nuclei` rebuilds the top-p candidate set by summing the returned probabilities until they reach `top_p`. Under `--logprobs-mode processed_logprobs`, which the docstring requires, vLLM returns logprobs already renormalized over the nucleus, so the sum reaches `top_p` early and drops true nucleus members. When a high-probability token was sampled, the candidate set is silently wrong and the trainer renormalizes π_θ over it. When the dropped token was sampled, the whole call fails. `test_top_p_candidates_rebuilt_from_top_k_logprobs` is vacuous: its mock returns unnormalized full-vocabulary probabilities. | [ran] `S/v3_nucleus.py`: p = [.60, .28, .05, .04, .03], top_p .9. True nucleus [0, 1, 2], rebuilt [0, 1] for samples 0 and 1, so the step-0 ratio is 1.057 where it should be 1. Sample 2 raises "outside the rebuilt nucleus". |
| V4 | P1 | `C/train/sglang_http.py:81-83` | Routed experts (R3) are parsed as nested per-completion-token lists. Community SGLang returns base64 int32, so `int('A')` raises. Its rows cover prompt plus completion minus one, and a resume needs `routed_experts_start_len`. A nested reply is therefore misaligned: routing row 0 holds prompt position 0. Layers × top-k are also flattened into one list per token. The mock in `test_sglang_engine.py:26` is the wrong format. | [ran] `S/v4_sglang_routes.py`: b64 gives "ValueError: invalid literal for int() … 'A'". Nested gives "completion 3 tokens, routing rows 7". |
| V5 | P1 | `C/train/hf_engine.py:121-129` | When idle, `HFGenerateEngine.load_weights` loads in place with `strict=False` and validates afterwards. A refused state (missing or extra keys) has already overwritten every matching tensor, and the version stays old. Later calls sample from a half-new model stamped with the old version. | [ran] `S/v6_hf_partial_load.py`: the load raises, then "engine.version = 0; tensors overwritten by the rejected load: 7/29". The same seeded call's logprobs change from `[-4.209, …]` to `[-4.088, …]`, still at `v0`. |
| V6 | P1 | `C/train/gateway.py:58-63`, `:148-156` | `GatewayPool._each` iterates `_live()`, which excludes held engines. In the `elastic` topology (hold, train, publish, release), the elastic group is held at every publish, so it never receives weights. It goes on serving v0 after release. | [ran] `S/v7_v8_v9_gateway.py`: "versions after publish+release: [1, 1, 0]". |
| V7 | P1 | `C/train/gateway.py:193-196` | A health check revives only engines marked dead. An engine the launcher restarts between probes is never marked dead. It comes back at version 0 with fresh weights and is never caught up (`HTTPEngine.health` even reads its version 0). It serves the initial policy for the rest of the run. | [ran] same probe: "versions [7, 0] alive [True, True] revived 0". 6 concurrent calls are stamped `[7, 0, 7, 0, 7, 0]`. |
| V8 | **P0** (cond.) | `C/train/gateway.py:170`, `C/train/hf.py:184` | `_last_weights` stores the caller's state dict by reference. `HFPolicy.state_dict()` returns the live parameters, so with `InprocPublisher` the training steps that follow mutate it. A revived engine then serves the current, unpublished weights stamped with the last published version. | [ran] same probe: "revived engine version 1, serves the v1 weights: False, serves the trainer's current (unpublished) weights: True". |
| V9 | P1 | `C/train/gateway.py:139-143`, `C/train/runtime.py:261-279` | The gateway turns every engine exception into `finish="abort"`, and `Runtime._call` re-issues aborts without limit. A deterministic error (a 400, a response that fails to parse such as V4) never ends: the gateway evicts every engine, the health loop revives them, and the cycle repeats. The rollout never masks, and the run stalls until `max_wall_s`. | [ran] `S/v_gateway_livelock.py`: still looping after 3 s with 231 aborts re-issued, 58 evictions and 58 revivals. |
| V10 | P1 | `C/train/runtime.py:73`, `C/train/world_env.py:123-171` | `RuntimeConfig.stop_ids` defaults to `()`, and `world_rollout` never ties it to `codec.eos_ids` (`tito_llm` does). `HFGenerateEngine` and `TorchEngine` stop only on `req.stop_ids`. With the default config the engine samples past `<|im_end|>` into invented user and tool turns. Those turns become policy tokens, are trained, and are parsed for tool calls. | [ran] `S/kv_resume_qwen.py` (Qwen2.5-0.5B): `'Hello, how are you?<|im_end|>\n<|endoftext|>Human: What is …<|im_end|>\n<|im_start|>user\n…'`, finish `length`. |
| V11 | P1 | `C/train/weightsync.py:274-281` | Errors are all-reduced only for receiver reloads. An owner that fails while staging a bucket raises before the broadcast, and every receiver blocks in `broadcast` until the backend timeout (gloo default 30 min; NCCL watchdog). There is no `WeightSyncError`. Causes include a missing key, a dtype or shape drift after `DistributedPublisher` cached the plan, or an H2D OOM. | [ran] `S/ws_run.sh owner_fails` and `dtype_drift` (gloo, 2 ranks, 8 s timeout): rank 0 raises `KeyError 'c'` (or a size mismatch) at 0.0 s. Rank 1 gets "Timed out waiting 8000ms for recv" at 8.0 s. |
| V12 | P1 | `C/actors/llm.py:400-404`, `C/train/world_env.py:243-263` | `tito_llm` captures the `seed` of the client's first `complete()` call for the client's lifetime. After one temperature-0 call (seed 0), every sampled call is sent with seed 0, even though `LLMToolPolicy` passes `seed=None` "so sampled branches differ". The n members of a GRPO group then sample identical completions, and their advantage is 0. | [ran] `S/v_tito_seed_version.py`: "seed sent per call: [0, 0, 0, 0]". |
| V13 | P2 | `C/train/sglang_http.py:67` | The seed goes out as `sampling_params.seed`. SGLang's key is `sampling_seed`, so any seeded call (for example `probe_logprobs_mode(seed=7)`) sends a field SGLang does not accept. `test_sglang_engine.py:41` asserts the wrong name. | [read] |
| V14 | P2 | `C/train/sglang_http.py:84-86` (also `engine.py:355-357`, `hf_engine.py:67-69`) | A resumed call opens a span for its version before it produces a token. If it is aborted again before its first token (possible on SGLang), the next resume adds a second span at the same start. `Turn.spans_valid()` then fails, and the empty version counts toward staleness. | [ran] `S/v_spans_empty_resume.py`: spans `[(0,0),(3,1),(3,2)]`, `spans_valid False`, versions `[0,1,2]`. |
| V15 | P2 | `C/train/world_env.py:91-93` | After an empty reply, `LLMToolPolicy` appends only a NUDGE user message. `EngineLLM` re-renders in the same segment without `new_segment`, and since the list has no assistant turn it does not count a prefix break either. M10's builder marks the call `prefix_break`, so in `mode="train"` the episode is masked as `crucible:capture_incomplete` (blame infra), although every token is exact. | [ran] `S/v_nudge_prefix_break.py`: `prefix_break [False, True]`, `mask_sample True`, `capture_incomplete`. |
| V16 | P2 | `C/train/workers.py:163-177` | `reap()` re-leases a job but does not cancel the old attempt's in-flight `/gen` handlers. The dead attempt's model call keeps holding the engine. When it finishes after `on_reset`, it writes its version and length into the live attempt's pool tag, which corrupts staleness and length accounting. The data itself stays correct (`/result` is refused). | [ran] `S/v_worker_stale_gen.py`: the tag is `{versions: [4], length: 7}` after the reset. |
| V17 | P2 | `C/train/chat.py:39` | `parse_tool_calls` accepts non-dict `arguments`. A JSON-string encoding (the OpenAI style) turns into a pydantic error in `LLMToolPolicy` and is scored 0 as `policy_error`, although the call was valid. A list argument on `send_message` raises `AttributeError`, which is masked with unknown blame. | [ran] the inline probe: `parse_tool_calls(...)` returns `'arguments': '{"x": 1}'`, and `ToolCall` raises `ValidationError`. |
| V18 | P2 | `C/train/chat.py:101-103`, `:139-141` | `HFChatCodec.eos_ids` is only `tokenizer.eos_token_id`, and there is no way to declare an end-of-turn token. Gemma-2 (`<end_of_turn>` 107, eos `<eos>` 1) fails every suffix ("closes assistant turns without an EOS token"), and its template refuses the system message `LLMToolPolicy` always sends. Qwen's alternative stop `<|endoftext|>`, on which vLLM stops via `generation_config`, gets `<|im_end|>` appended after it. The failure is loud, so this is not silent corruption. | [ran] `S/tito_fuzz.py` gemma rows; inline suffix probe. |
| V19 | P2 | `vllm_http.py:57-61,104,111-113`; `sglang_http.py:49-51,102`; `gateway.py:336-338` | `capabilities()` claims more than the engines do: vLLM and SGLang report `pause: ["keep"]` but pause with abort mode, and `HTTPEngine` always reports `top_p_candidates: True`. `VLLMServerEngine.generate` ignores `resume_completion`, so `version_switch="token"` silently runs at call level while `resumed_calls` and `reprefill_tokens` keep counting. `abort_all` posts `/abort_requests`, which no framework in `06-frameworks` calls over vLLM HTTP (unverified; a 404 raises after the cancels, and under the gateway that evicts every engine at each barrier). | [read] |
| V20 | P2 | `C/train/weightsync.py:56-57`, `:234-236`, `:227`, `:323` | `DTYPES` has no fp8 entries, so planning raises `KeyError 'float8_e4m3fn'`. `update(None)` on the coordinator makes receivers commit version −1. Every subgroup update calls `dist.new_group`, so communicators leak. `publish` raises the version before `_send` and burns it when the send fails. | [ran] `S/ws_run.sh fp8` and `no_version`: "committed version -1". |
| V21 | P2 | `C/train/world_env.py:189`, `C/train/runtime.py:338-367` | `world_rollout` builds records in `mode="eval"`, and the M12 runtime never calls `valid_for_training()`. A record with `policy_version None`, a logprob count that does not match the completion, or invalid spans reaches `trainer.step`. The schema calls that check "the gate every trainer bridge applies". | [read] |
| V22 | P2 | `C/train/gateway.py:303-313` | `generate_app` drops `candidates`, so a `top_p < 1` client cannot renormalize. After `max_aborts` it returns the last aborted (partial or empty) result labelled `finish: "length"`. | [read] |
| V23 | P2 | `hf_engine.py:79-81,107-108`; `launch.py:65-71,41-44` | `HFGenerateEngine.abort(call_id)` on a call that is not running leaves the id in `_aborted`, which then kills the next call with that id; the runtime re-issues with the same `call_id`. There is no caller today. `Child.kill` skips `killpg` once the group leader has exited, so grandchildren survive. `_port()` releases the port before the child binds it, so a restart on "the same port" can lose it. | [read] |

## Detail, fixes and regression tests

**V1 (P0): cross-conversation splice in `EngineLLM`.** `_prompt` matches a served entry when `messages[:len(s)] == s` and `messages[len(s)].role == "assistant"`. When two entries have equal `s`, the first one in the table wins (`len(s) > len(best)` is strict). It is popped, so the other episode later takes the remaining entry: A and B swap histories.

The engine sees the spliced prompt, so behaviour logprobs agree with it. The trajectory itself is incoherent, though: B's World executed B's action while B's context and record show A's. The policy's `_compress` then sees a non-extension, and M10's builder marks `prefix_break` (masked in train mode, trained in M12).

*Fix:* store a fingerprint of the reply with each entry, the `(text, tool_calls)` that `complete` returned, or a per-call nonce the policy echoes back. Extend only when `messages[len(s)]` matches it. Otherwise render whole. Simpler and safer: key the table by `sample_key` (`LLMToolPolicy` passes `sample_key=episode_id`, which `EngineLLM.complete` currently swallows in `**_`).

*Test:* the probe as a test. Two conversations with identical openings and different first replies go through one `EngineLLM`. Assert that each turn-2 prompt starts with its own prompt plus completion. It fails now.

**V2 (P0): vLLM publish under running calls.** *Fix:* in `load_weights`, unless the engine is already paused, pause first with `/pause?mode=keep`, the SkyRL protocol (requests freeze in the scheduler). Then load, reset the prefix cache, and `/resume`. Raise the client's version only after the swap. The calls frozen across the swap then span two versions, which needs `version_spans`. vLLM does not report the token boundary, so the clean option is `mode=abort` plus a re-issue under the new version. Alternatively refuse `disaggregated_stream` and `elastic` for `VLLMServerEngine` until there is a call-boundary swap.

*Test:* the mock server from `S/v5_vllm_publish_inflight.py` (tokens marked by the loaded weights). Assert that every token of a response comes from the stamped version, or that the response is `abort`. It fails now.

**V3 (P1): wrong nucleus from processed logprobs.** *Fix:* under processed mode, the nucleus is the set of returned entries with finite logprob (> −9999). Require at least one `-inf` entry among the top-k, or Σ exp ≈ 1, which shows that k covered the nucleus. Otherwise fail with "raise nucleus_k". Never sum to `top_p`.

*Test:* a mock returning renormalized logprobs for p = [.60, .28, .05, .04, .03] at top_p 0.9. Assert that candidates == [0, 1, 2] for every sampled token. It fails now.

**V4 (P1): SGLang R3.** *Fix:* decode base64 int32 and reshape to `(len(prompt)+len(completion)-1, layers, topk)`, taking layers and top-k from the config. Keep the rows of the completion positions, `[len(prompt)-1:]`. On resume, send `routed_experts_start_len` and append. Keep `[layers][k]` per token. Check the row count, as slime does.

*Test:* the mock from `S/v4_sglang_routes.py` with base64. Assert `len(routing) == len(completion)` and that row t equals server row `len(prompt)-1+t`.

**V5 (P1): HF engine partial load.** *Fix:* load into a deep copy always, or check keys and shapes against `state_dict()` before touching the model. Swap `self.model` and `self.version` only after the load succeeds.

*Test:* `S/v6_hf_partial_load.py`. After the refused load, assert that `state_dict()` is bitwise unchanged.

**V6 (P1): held engines skipped.** *Fix:* `_each` for `load_weights`, `pause`, `resume` and `abort_all` must iterate every alive engine, held or not. Only routing should avoid held engines.

*Test:* hold [2], `load_weights(v1)`, release. Assert that every engine is at version 1.

**V7 (P1): silent restart.** *Fix:* in `check_health`, when `_last_weights` is set and the probed engine reports a version different from `_last_weights[1]`, treat it as stale. Evict it and `revive` it, which replays the weights. Better still, have engines report a boot id.

*Test:* the probe. Assert that after `check_health` every live engine is at version 7.

**V8 (P0, conditional): live-reference replay.** *Fix:* keep a detached snapshot in `_last_weights` (clone to CPU), or keep the checkpoint path only (`CheckpointPublisher` is already safe). Alternatively make `HFPolicy.state_dict` detach and clone, as `TorchTrainer.state_dict` does.

*Test:* the probe. Assert that the revived engine's weights equal the v1 snapshot.

**V9 (P1): unbounded re-issue.** *Fix:* tell infrastructure aborts (EngineDown, barrier) apart from errors. Re-raise, or return a distinct `finish="error"`, for non-connection exceptions after `failure_threshold` consecutive failures of the same call. Cap the re-issues in `Runtime._call`; past the cap, mask the rollout with blame `infra`.

*Test:* the probe. Assert that `_call` raises or masks within N attempts.

**V10 (P1): no end-of-turn stop.** *Fix:* let `world_rollout` wrap `gen` to add `codec.eos_ids` to the stop ids (or pass them through `gen`), or refuse an empty `RuntimeConfig.stop_ids` when the rollout function is a chat codec's.

*Test:* `world_rollout` with `HFGenerateEngine(Qwen2.5-0.5B)` under the default `RuntimeConfig`. Assert that no completion contains a token after the first end-of-turn.

**V11 (P1): owner staging failure.** *Fix:* the owner must validate `owned` against the plan before the first collective: same names, dtype and numel. Then wrap staging in try/except and still enter the broadcast with a poisoned buffer, followed by an all-reduced error code (owner rank + 1), so every rank raises `WeightSyncError`. Alternatively all-reduce a "staged ok" flag before each broadcast.

*Test:* `owner_fails` and `dtype_drift` from `S/ws_worker.py` in `_weightsync_worker.py`. Assert that both ranks raise `WeightSyncError` well inside the timeout.

**V12 (P1): seed captured for life.** *Fix:* pass the seed per call. `gen(prompt, n, temp, seed=...)` takes it from each `complete()`, and `tito_llm` stops closing over it.

*Test:* the probe. Assert seeds `[0, None, None, None]`.

**V13–V23 (P2)** each fix in one or two lines:
- V13: send `sampling_seed`.
- V14: open a span for a version only when that version emits a token. Drop zero-length spans before returning, so starts stay strictly increasing.
- V15: flag `new_segment` (and count a prefix break) on every whole re-render that follows a served call.
- V16: cancel `_gens[rid]` in `reap()`, or ignore a `/gen` result whose attempt no longer owns the job before the runtime's `gen` updates the tag.
- V17: `json.loads` string arguments. Any non-dict becomes `{"__unparseable__": …}`.
- V18: add `HFChatCodec(eos_ids=…)` (an end-of-turn override) and a system-folding option.
- V19: report `pause: ["abort"]`; add a `token_resume` capability and have the runtime refuse `version_switch="token"` without it; make `HTTPEngine` query `/capabilities`; drop `/abort_requests` or verify it.
- V20: add the fp8 dtypes; refuse `version None` on the coordinator; cache subgroups by their rank tuple; raise the version only after a successful send.
- V21: call `valid_for_training()` in `Runtime._complete` and mask the failures with the reason.
- V22: return `candidates`, and an explicit error after `max_aborts`.
- V23: clear `_aborted` entries for calls that are not running; kill the process group always; keep the port reserved or pass `--port 0` and read it back.

Each regression test is the matching probe as an assertion.

## Checked and clean

- **Exact TITO on real templates** [ran, `S/tito_fuzz.py`].
  - Coverage: Qwen3-0.6B (thinking on and off), Qwen3-4B-Instruct-2507 and Qwen2.5-0.5B-Instruct, across these scenarios: one tool call, two tool calls in one turn, a `<think>` block before a call, trailing whitespace, unicode and emoji, stop on EOS, truncated (no EOS), stop on `<|endoftext|>`, and an empty completion.
  - In every case the engine received exactly the recorded prompt, and the recorded completion is exactly the sampled ids.
  - Turn k+1 extends prompt + completion token for token.
  - The suffix equals the tail of a full re-render, so the template frames new messages identically in the real and the sentinel context.
  - EOS is appended only when the completion lacks it. The appended EOS sits in the next prompt and is never trained.
  - Qwen3 is detected as `interleaved` and re-renders (new segment) when a user message arrives. The 2507 and 2.5 templates are `preserved`.
  - Tool-call parsing works on all three Qwen tokenizers.
- **KV cache and resume** on the real Qwen2.5-0.5B-Instruct, CPU fp32 [ran, `S/kv_resume_qwen.py`].
  - `HFGenerateEngine`'s KV-cached logprobs equal one full forward within 9.5e-6 at T = 1.0 and T = 0.7.
  - Greedy resumes at k = 1, 3, 5 reproduce the uninterrupted tokens.
  - The resumed token's logprob matches the full forward.
- **Weights and versions in the in-process engines** [read].
  - `TorchEngine` and `HFGenerateEngine` pin `(version, model)` per call.
  - `HFGenerateEngine` loads into a copy while calls are in flight.
  - Candidates and MoE routes carry across in-process resumes.
  - `probe_logprobs_mode` distinguishes processed from raw.
- **Leases** [read]: `/result` and `/gen` from a superseded attempt are refused (409 or ignored), so nothing is trained twice.
- **Weight-sync plan** [read; existing tests]: a tensor larger than the bucket is refused at plan time, views are made contiguous, and bf16, fp16, bool and 0-dim tensors round-trip bitwise.
