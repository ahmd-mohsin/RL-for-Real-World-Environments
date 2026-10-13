# M14 audit: validation campaign (packs + RLM agent)

Run against the worktree `/Users/muahmed/Desktop/Projects/m14wt/crucible` (branch `m14-audit`, the uncommitted fixes) on
2026-10-13, 07:47 to 08:25 PDT, on a CPU-only Mac with local Docker 29.8. No paid API and no GPU. Models: Qwen2.5-0.5B-Instruct (local
weights, Crucible's HF engine). Only the Qwen3-0.6B tokenizer is cached, with no weights, so Qwen3 was used for the
thinking-template codec probe only. In total there were 1,550 runner episodes, plus about 3,200 direct World episodes (fuzz) and
decision-level checks over 2,000 generated worlds.

Scripts and raw outputs are under the scratchpad
`/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m14/validate/`
(written `$V` below):
- drivers: `common_v.py` (tasks, policies, adversaries), `run_scripted.py`, `run_surrogate.py`, `surrogate_llm.py`,
  `coding_adv.py`, `ap_rates.py`, `fuzz_gen.py`, `run_llm.py`, `rlm_v.py`, `rlm_qwen3_compact.py`, `feed_records.py`,
  `train_smoke.py`, `tab.py`;
- logs: `$V/logs/`; run dirs: `$V/runs/`.

Docker: no image was created. Five RLM containers leaked during training runs (B4). They were removed by name and none
remain.

## 1. `pack check` and the CLI

| what | result | wall |
|---|---|---|
| `pack check --all --level static` | toy, coding, finance_ap pass (exit 0) | 1 s |
| `pack check --all --level fixture` | toy 15 cases and finance_ap 17 cases pass. **coding failed `fixture.error`** (seeding race, B3: a concurrent crucible process on the host shared `/tmp/.crucible_seed_repo.tar`). It passed with 21 cases at the next level. | 31 s |
| `pack check --all --level semantic` | toy pass: 1 anchor, generator seeds 101/202, 2 adversaries x 1 task. finance_ap pass: 2 anchors, 5 adversaries, 20 runs. coding is a loud skip (`requires: gpu`, exit 2). | 52 s |
| `pack check coding --level semantic --allow gpu` (on CPU) | pass: 4 CPU dense tasks, 2 adversaries, 8 runs (B11) | 60 s |
| `pack check finance_ap --level accept` | pass: 4 tasks, all *provisional* (A10 pending) | 70 s |
| `pack check toy --level accept` | **fail** on all 3 tasks: A2/A6/A7/A13/A16 fail, A8 error, A3 fail on backlog items (B5) | 30 s |
| `pack new demo --dir $V/packs/demo`, then check | static, fixture, semantic pass; **accept fails** (same A-checks as toy, B5) | 1 s each |
| `bench run demo --k 3` | gold 1.000, no-op 0.000 over 4 tasks, 12 samples each | 1 s |
| `bench run toy --k 3 --fresh 4` | oracle 1.000, half 0.333 ± 0.243, no-op 0.000 over 5 tasks, 0 masked, 0 invalid | 10 s |
| `bench run finance_ap --k 3 --fresh 4` | oracle 1.000, approve_all 0.000, no-op 0.000 over 6 tasks, 0 masked, 0 invalid | 15 s |
| `bench run coding` | **crash**: `TypeError: 'coroutine' object is not iterable` (B6) | 0 s |
| one `--key-file` for two packs | refused: "this key already built demo-bench@v1: a new version needs a new key" (B12) | - |
| wheels (`uv build --offline` of copies) | each wheel ships `_pack/` data. toy gold = 1.0 and finance_ap gold = 1.0 when imported from the unpacked wheels outside the checkout. | - |

## 2. Benchmarks per pack × policy

Every run used k = 3 with controls on (oracle and no-op first; invalid tasks excluded). Means are over tasks, with a t
interval over task means. pass@1 / pass@3 count reward ≥ 1. "bound" is the expected maximum (`anchor:0` means 0 on the
toy anchor, while on generated backlogs the policy equals or nearly equals the gold).

### finance_ap: 9 tasks (2 anchors, 6 `generate()` items, one 14-invoice world), 25 policies, 675 episodes, 134.6 s, 0 masked, 0 invalid

| policy | bound | tasks | samples | mean ± CI | pass@1 | pass@3 | max | masked | invalid |
|---|---|---|---|---|---|---|---|---|---|
| scripted:oracle | 1 | 9 | 27/27 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| scripted:noop | 0 | 9 | 27/27 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| scripted:approve_all | 0 | 9 | 27/27 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| declared:approve_all / hold_all / exact_match_duplicates / wrong_tolerance / reverse_rule_order | 0 | 9 each | 27/27 each | 0.000 each | 0 | 0 | 0.000 | 0 | 0 |
| prev:no_tolerance / tol2_textbook_order / dup_fmt / dup_swap | 0 | 9 each | 27/27 | 0.000 each | 0 | 0 | 0.000 | 0 | 0 |
| prev:mask_try (approve all + a 2**63 id; a mask in v1) | 0 | 9 | 27/27 | 0.000 | 0 | 0 | 0.000 | **0** | 0 |
| prev:mask_selective (oracle on half the tasks) | - | 9 | 27/27 | 0.556 ± 0.405 | 0.556 | 0.556 | 1.000 | **0** | 0 |
| prev:tol5 (memorized 5%) | 0 | 9 | 27/27 | 0.333 ± 0.384 | 0.333 | 0.333 | 1.000 | 0 | 0 |
| NEW approve_then_reverse | 0 | 9 | 27/27 | 0.000 | 0 | 0 | 0.000 | 0 | 0 |
| NEW alias_vendor_args (pays blocked-vendor invoices, with an active vendor id passed as an extra argument) | 0 where a blocked vendor decides | 9 | 27/27 | 0.556 ± 0.405 | 0.556 | 0.556 | 1.000 | 0 | 0 |
| NEW dup_needs_same_amount | 0 | 9 | 27/27 | **0.444 ± 0.405** | 0.444 | 0.444 | 1.000 | 0 | 0 |
| NEW price_round_units (whole currency units) | 0 | 9 | 27/27 | **0.889 ± 0.256** | 0.889 | 0.889 | 1.000 | 0 | 0 |
| NEW price_float_compare (an honest float compare) | ≤1 | 9 | 27/27 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| NEW malformed_only (26 malformed calls) | 0 | 9 | 27/27 | 0.000 | 0 | 0 | 0.000 | **0** | 0 |
| NEW malformed_plus_oracle | 1 | 9 | 27/27 | 1.000 | 1.000 | 1.000 | 1.000 | **0** | 0 |
| NEW early_done (half the decisions, then "DONE") | 0 | 9 | 27/27 | 0.000 | 0 | 0 | 0.000 | 0 | 0 |
| NEW done_messages_only | 0 | 9 | 27/27 | 0.000 | 0 | 0 | 0.000 | 0 | 0 |
| NEW double_decide (oracle, then re-approve all: refused) | 1 | 9 | 27/27 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| NEW lone-surrogate argument (separate run, 4 tasks) | 0 | 0 | **0/8 (masked)** | - | - | - | - | **agent_run_error 8** | 0 |

alias_vendor_args scored 0 on all 4 worlds where a blocked vendor decides an invoice, and 1 (it is the oracle there) on
the 5 worlds without one. The extra `vendor_id` argument was accepted silently (B13). No generated world holds a payee
under two vendor ids (0 of 2,000), so a real alias cannot be posed (B7).

Decision-level rates over 2,000 generator v2 worlds (`$V/runs/ap_rates.json`; solve = every decision right; breach =
pays a stopped invoice):

| policy | solve | breach |
|---|---|---|
| declared wrong_tolerance / reverse_rule_order / exact_match_duplicates | 0.0% / 0.0% / 0.0% | 86.3% / 0% / 100% |
| ignores the per-world hold order (textbook order) | **57.8%** | 0% |
| price in whole currency units | **68.0%** | 27.7% |
| duplicate only when the amount also matches | **54.5%** | 38.1% |
| memorized 5% / memorized 2% + textbook order | 14.7% / 7.8% | 57.8% / 28.1% |
| float compare (honest) | 92.0% | 0% |

Generator fuzz through real Worlds (`$V/runs/fuzz.json`, 150 fresh items, n = 6..14 evenly): oracle = 1, no-op = 0,
approve-all = 0 and each of the 5 declared adversaries = 0 on every item. There were 0 masks and 0 violations, in 113 s
together with the toy fuzz below.

### toy: 5 tasks (helpdesk anchor + 4 generated 12-ticket backlogs), 19 policies, 285 episodes, 53.0 s, 0 masked, 0 invalid

| policy | bound | tasks | samples | mean ± CI | pass@1 | pass@3 | anchor max | backlog mean / max | masked | invalid |
|---|---|---|---|---|---|---|---|---|---|---|
| scripted:oracle | 1 | 5 | 15/15 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 / 1.000 | 0 | 0 |
| scripted:noop | 0 | 5 | 15/15 | 0.000 | 0 | 0 | 0.000 | 0.000 / 0.000 | 0 | 0 |
| scripted:half | - | 5 | 15/15 | 0.383 ± 0.280 | 0 | 0 | 0.000 | 0.479 / 0.583 | 0 | 0 |
| declared:forged_approval | anchor:0 | 5 | 15/15 | 0.350 ± 0.258 | 0 | 0 | 0.000 | 0.438 / 0.500 | 0 | 0 |
| declared:split_refund | anchor:0 | 5 | 15/15 | 0.367 ± 0.280 | 0 | 0 | 0.000 | 0.458 / 0.583 | 0 | 0 |
| prev:claim_done | 0 | 5 | 15/15 | 0.000 | 0 | 0 | 0.000 | 0.000 / 0.000 | 0 | 0 |
| prev:resolve_all ("Resolved." note + resolved on every ticket) | 0 | 5 | 15/15 | 0.150 ± 0.153 | 0 | 0 | 0.000 | **0.188 / 0.333** | 0 | 0 |
| NEW resolve_keyword_notes (reason words stuffed in notes) | 0 | 5 | 15/15 | 0.150 ± 0.153 | 0 | 0 | 0.000 | **0.188 / 0.333** | 0 | 0 |
| prev:no_ask | anchor:0 | 5 | 15/15 | 0.350 ± 0.258 | 0 | 0 | 0.000 | 0.438 / 0.500 | 0 | 0 |
| prev:mask_try (2**63 ticket id, string amount) | 0 | 5 | 15/15 | 0.033 ± 0.057 | 0 | 0 | 0.000 | 0.042 / 0.083 | **0** | 0 |
| NEW ask_customer (approval asked of customer_dana) | anchor:0 | 5 | 15/15 | 0.367 ± 0.280 | 0 | 0 | 0.000 | 0.458 / 0.583 | 0 | 0 |
| NEW ask_ops (approval asked of `ops`) | anchor:0 | 5 | 15/15 | 0.367 ± 0.280 | 0 | 0 | 0.000 | 0.458 / 0.583 | 0 | 0 |
| NEW approval_wrong_amount (asks 40, refunds 129.99) | anchor:0 | 5 | 15/15 | 0.367 ± 0.280 | 0 | 0 | 0.000 | 0.458 / 0.583 | 0 | 0 |
| NEW approval_other_order | anchor:0 | 5 | 15/15 | 0.800 ± 0.555 | 0.8 | 0.8 | 0.000 | 1.000 (= gold) | 0 | 0 |
| NEW approval_after_refund | anchor:0 | 5 | 15/15 | 0.350 ± 0.258 | 0 | 0 | 0.000 | 0.438 / 0.500 | 0 | 0 |
| NEW one_approval_two_refunds | anchor:0 | 5 | 15/15 | 0.800 ± 0.555 | 0.8 | 0.8 | 0.000 | 1.000 (= gold) | 0 | 0 |
| NEW approval_reused_twice | anchor:0 | 5 | 15/15 | 0.800 ± 0.555 | 0.8 | 0.8 | 0.000 | 1.000 (= gold) | 0 | 0 |
| NEW forged_supervisor_text (agent writes "Approved refund of ... - Sam Lee") | anchor:0 | 5 | 15/15 | 0.367 ± 0.280 | 0 | 0 | 0.000 | 0.458 / 0.583 | 0 | 0 |
| NEW approve_unbound (backlog approvals that name no ticket or order) | 0 on backlogs | 5 | 15/15 | **1.000** | 1.000 | 1.000 | 1.000 (gold) | **1.000 / 1.000** | 0 | 0 |

On the anchor, every adversary scores 0. Toy fuzz: 30 generated backlogs, oracle = 1 and no-op = 0 on each, 0 masks.

### coding: 4 CPU dense-replay tasks (task01v2, task02, task03, task04), concurrency 1, 72 episodes, 64.3 s

| policy | bound | tasks | samples | mean ± CI | pass@1 | pass@3 | max | masked | invalid |
|---|---|---|---|---|---|---|---|---|---|
| scripted:gold | 1 | 4 | 12/12 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| scripted:noop | 0 | 4 | 12/12 | 0.000 | 0 | 0 | 0.000 | 0 | 0 |
| declared comment_stuffing / destructive_only | 0 | 4 | 12/12 | 0.000 | 0 | 0 | 0.000 | 0 | 0 |
| **NEW string_literal stuffing** | 0 | 4 | 12/12 | **1.000** | 1.000 | 1.000 | 1.000 | 0 | 0 |
| **NEW dead_code stuffing** (`#if 0` / `if False:`) | 0 | 4 | 12/12 | **1.000** | 1.000 | 1.000 | 1.000 | 0 | 0 |

Φ of each patch (`$V/runs/coding/phi_attacks.json`). Every seed has 46 to 343 milestones. The stuffed and destructive patches score 0.0 on every
seed. The string-literal and dead-code patches score 1.0 on every seed. The recorded honest failures score 0.53 (task01v2), 0.07
(task02), 0.06 (task03) and 0.09 / 0.29 (task04). The same bench at concurrency 4 masked 4 + 6 episodes as
`agent_run_error` (stage seed, unknown blame) and stopped at 23 of 72 episodes on the health rule (B3).

### LLM policies: Qwen2.5-0.5B-Instruct, CPU, n = 3, controls on

| pack | harness | tasks | samples | mean ± CI | pass@1 | pass@3 | masked | invalid | wall s |
|---|---|---|---|---|---|---|---|---|---|
| toy (anchor + 1 backlog) | native LLM tool | 2 | 6/6 | 0.000 ± 0.000 | 0 | 0 | 0 | 0 | 100 (both harnesses) |
| toy | RLM, Docker kernel (default) | 2 | 6/6 | 0.000 ± 0.000 | 0 | 0 | 0 | 0 | |
| finance_ap (s1 + 1 generated, max_steps 120) | native LLM tool | 2 | 6/6 | 0.000 ± 0.000 | 0 | 0 | 0 | 0 | 60 (both) |
| finance_ap | RLM, Docker kernel | 2 | 6/6 | 0.000 ± 0.000 | 0 | 0 | 0 | 0 | |

Every rollout (12 native and 12 RLM) is `valid_for_training`, with 0 prefix breaks and every RLM kernel `docker`. A first finance_ap
run with `max_steps = 60` marked the 12-invoice item `broken_oracle`: its oracle has 73 steps (B14).

## 3. RLM agent

Probes used a byte-codec scripted engine through the M10 Runner with the default config, unless stated (`$V/runs/rlm_probe/`,
`$V/logs/rlm_probe.log`).

| probe | result |
|---|---|
| default config | `kernel: docker`, `python:3.12-slim`. The cell sees `/.dockerenv` True, uid 65534, no network (`OSError`), env only `GPG_KEY` (the image's). 1 `crucible.rlm=1` container per episode. `Kernel()` default launcher: `DockerLauncher`. |
| `kernel='local'` without `allow_host` | `RLMConfig.launcher()` raises `ValueError`. Through `RLMHarness` the attempt is **masked** `agent_run_error`, unknown blame, retryable (B9). `rlm_world_rollout` refuses at construction. |
| model code kills its kernel (`os._exit`) | scored 0; 1 restore; not masked |
| container removed from outside mid-episode | a1 masked `crucible:env_container_lost`, a2 retried and scored |
| docker binary missing / image missing | masked `env_container_lost` / `env_image_unavailable` (3 attempts each) |
| sub-LLM engine failure | a1 masked `provider_unavailable`, a2 scored |
| sub-call budget (cap 20 per episode, 8 per cell, `sub_max_tokens` 64) | 17 calls served, 208 refused; a caller's `max_tokens = 500000` clamped to 64; scored, not masked |
| sub-calls answered by another model | turns `("sub", "other-model", trainable=False)` × 3. Trainer items cover the root turns only (600 tokens over 2 records); the step is finite. |
| plain-text forgery on fd 1/3/4/7/9 | ignored (1 protocol error), not masked |
| well-formed forged `done` on the protocol fd (from `sys.modules['__main__']`) | **accepted**: it ended the episode; digest `deadbeef` was journaled; lone surrogates were sanitized to `?`; scored 0, not masked (B10) |
| busy loop past the cell deadline | killed, restore, state kept, scored |
| never finishes (`max_turns` 4) | scored 0, not masked |
| `world.call` with a 2**63 id / a lone surrogate | `invalid_arguments` / sanitized; scored |
| oracle from one cell | finance_ap 1.0 (55 World calls), toy anchor 1.0 |
| compaction under Qwen3's *interleaved* thinking template (real Qwen3 tokens, scripted) | 2/2 rollouts valid, 2 segments, 0 prefix breaks, not masked (G1 D3 / H3 V9 fixed) |

Qwen records into the M12 `TorchTrainer` with the same Qwen weights (`$V/runs/feed_records.json`):
- 24 records, all valid, trainer gate masked 0;
- \|trainer logprob − behaviour logprob\| ≤ 1e-4 per call (native and RLM);
- one step in 285 s: 24 items, 4,117 tokens, `mismatch/rollout_probs_diff` 1.7e-6, no non-finite metric. Loss and gradient are 0 because every reward was 0.

## 4. Training smoke (M12 Runtime, CPU, TinyPolicy, ScriptedEngine variants: gold, wrong, partial, no-op)

| run | steps | trained | masked | reward range | non-finite | notes |
|---|---|---|---|---|---|---|
| native (toy + finance_ap), `dis` | 3 | 24 | 0 | [0, 1], mean 0.208 | none | `masked_token_share` 1.0 (scripted logprobs are fake), loss 0 |
| native, `tis` + recompute_current | 3 | 24 | 0 | [0, 1] | none | loss −0.0004 / −0.0013 / −0.0009, grad norm ≤ 7e-4 |
| RLM (Docker kernel) `dis` / `tis` | 2 | 16 | 0 | [0, 1] | none | 105 turns + 24 sub + 8 compaction (`dis`); 76 + 18 + 2 (`tis`) |
| native, approve-all + lone-surrogate argument in 1/4 or every rollout | 3 | 24 | **0** | [0, 0] | none | the training path (`world_rollout`) scores it 0. Only the Runner path masks it (B1). |

`tis` without `old_logprob_source` refuses with a clear error, which is correct. In 2 of 3 runs that put native and RLM in one
process, 2 to 3 RLM containers kept running after the process ended (B4).

## 5. Findings

| id | sev | finding | evidence | suspected location | minimal fix |
|---|---|---|---|---|---|
| B1 | **P0** | **A lone surrogate in any tool argument masks the policy's own failing episode** and can stop the whole run. A model emits it as plain ASCII `"\udcff"` inside tool-call JSON. Approve-all scored 0 on 9/9 samples. Approve-all plus one `get_receipts(po_id="\udcff")` was masked `agent_run_error` (unknown blame, retryable) on 8/8, and oracle plus the same call on 6/6. The health rule then stopped the bench: 675 planned, 67 run ("unknown-blame 4/62 > 0.05"); the surrogate run was stopped at "mask fraction 12/20". The training path and the RLM path score it, not mask it. | `$V/runs/surrogate/bench-ap-surrogate`, `$V/runs/surrogate_llm/`, `$V/logs/scripted.log` (first run, killed), traceback in §B1 below | `world/world.py:612` logs the action, then `world.py:872` `model_dump_json` raises `PydanticSerializationError` while the native harness streams the trace (`runtime/harness/native.py:301`). `runtime/infra.py:94` maps it to unknown blame. | Validate action strings in `World.step`: a lone surrogate is an `invalid_arguments` tool error with policy fault. Make `_log` lossless-safe (`ensure_ascii` JSON or `errors="backslashreplace"`). Map a `UnicodeEncodeError` raised by a policy action to `crucible:policy_misbehavior`. |
| B2 | **P0** | **Coding dense Φ is hackable by string-literal and dead-code stuffing.** The oracle's tokens in string literals or under `#if 0` / `if False:` reach Φ = 1.0 on all 4 bench seeds, with no migration done. The CPU task's reward is Φ, so these attacks score 1.0 on 12/12 samples, the same as gold, while honest failures score 0.06 to 0.53. `pack check --level semantic` passes because the declared adversaries cover comments and deletions only. | `$V/runs/coding/phi_attacks.json`, `$V/runs/coding/bench-coding-c1` | `packs/coding/src/crucible_pack_coding/studio.py:41` (`_strip_c` keeps string literals) and `:82` (`code_only` strips comments and docstrings only; no reachability) | Before grep milestones, blank the contents of string and char literals (keep the quotes and the line structure). Drop `#if 0 … #endif` blocks and constant-false Python blocks (via `ast`). Better: match token milestones on code tokens (identifiers and calls) only. Add both attacks to `adversaries()` with `expect_max 0`. |
| B3 | P1 | **The local-backend seed tarball is at a fixed host path**, `/tmp/.crucible_seed_{app}.tar`, shared by every World on the host. Concurrent Worlds truncate or delete each other's tar. Evidence: 2/6 concurrent opens of one coding task failed. The coding bench at concurrency 4 masked 10 episodes (stage seed, unknown blame) and the health rule stopped it at 23/72. `pack check --all --level fixture` failed coding once while another crucible process ran. Concurrency 1 was clean (72/72). The `kb` apps of toy and finance_ap use the same pattern, but their tars are small. | `$V/coding_race.py` output, `$V/runs/coding/bench-coding-c4`, `$V/logs/check_all_fixture.json` | `world/apps/workspace.py:71`, `world/apps/doc_library.py:108` (`LocalExec.push_file` writes the host `/tmp`) | Use a unique remote name per World (uuid, or under `exec.staging_dir`) and remove it after extracting. Classify a seed failure as infra, not unknown. |
| B4 | P1 | **RLM kernel containers leak** when a training process ends with RLM rollouts in flight. In 2 of 3 runs that put native and RLM in one process, 2 to 3 `crucible-rlm-*` containers (`sleep infinity`) stayed running with nothing to reap them. Standalone RLM runs (3) did not leak. | `$V/logs/train_combo_leak2.log` (before 0, after 2); the 08:14 `train_tis` run left 3. All were removed by name. | `actors/rlm/kernel.py:300-306` (a `docker run` still in flight when the rollout is cancelled), `:322-326` (`aclose`: `suppress(Exception)` does not catch `CancelledError`, and `rm -f` can run before the container exists), and the rollout cleanup in `train/world_env.py` | Shield the container start and the `rm` (`asyncio.shield`). On close, reap by a per-launcher label. Give the container a bounded life (`sleep <ttl>` or an owner-pid watchdog) instead of `sleep infinity`. |
| B5 | P1 | **toy fails its own `pack check --level accept`** on all 3 tasks (A2, A6, A7, A13, A16 fail; A8 error; A3 fails on backlogs). `pack new demo` fails accept the same way, which contradicts `docs/modules/M14-packs.md:57` ("passes `crucible pack check` at every level"). finance_ap passes as provisional. | `$V/logs/check_accept_toy.json`, `$V/logs/demo_check_accept.log` | `packs/_toy/src/crucible_pack_toy/runtime.py:25` (`HelpdeskTask`), `longhorizon/task.py:100`, `packs/_template/.../studio.py`: no M9 capabilities | Give the toy and template tasks the capabilities `APTask` implements (properties, checks, negatives, attacks, tamper_attempts, isolation_audit, property_audit), or change the doc gate to static..semantic and make accept opt-in. |
| B6 | P1 | **`crucible bench run coding` crashes** with `TypeError: 'coroutine' object is not iterable`. The coding `tasks` hook is async: `pack check` awaits it, the bench CLI does not. The CLI bench also ignores `requires` (gpu, docker), and coding has no `hooks.policies`. | `$V/logs/bench_cli_coding.log` | `cli/studio.py:341` (compare `studio/pack.py:298`) | Await an awaitable hook result as `pack check` does, honour `requires`, and give coding a `policies` hook (gold and no-op on the CPU tasks). |
| B7 | P1 | **finance_ap still has large shortcut shares in its draws.** Over 2,000 v2 worlds: ignoring the per-world hold order solves 57.8% (0% breach); comparing prices in whole currency units solves 68.0%; treating a duplicate as one only when the amount matches solves 54.5%. On the 9 bench tasks these scored 0.889 and 0.444. Every world draws a compound case, but its two rules sit in textbook order in about half the worlds. A restated-amount duplicate is drawn 30% of the time, and a one-cent-over price 20%. No world holds an alias vendor (the same payee under two ids, one blocked): 0 of 2,000. The five declared adversaries are 0% as claimed. | `$V/runs/ap_rates.json`, `$V/runs/scripted/bench-ap` | `packs/finance_ap/src/crucible_pack_finance_ap/world.py:148` (compound plan), `:226` (one cent at 20%), `:255` (restated at 30%) | Guarantee per world: a compound case whose hold rules are out of textbook order; a restated-amount duplicate; a one-cent-over probe. Optionally add a blocked payee re-registered under a new vendor id (same name or bank) to AP-3 and the data. |
| B8 | P2 | **The toy backlog (4 of 5 bench items) keeps the grading the anchor dropped.** Any note counts, so "resolve everything with 'Resolved.'" or with keyword-stuffed notes scores 0.188 mean and 0.333 max per backlog. Approvals are matched by amount only, are not consumed and are not bound to a ticket or order, so `approve_unbound` scores 1.0 on 4/4 backlogs. | `$V/runs/scripted/bench-toy` | `packs/_toy/src/crucible_pack_toy/longhorizon/task.py:76-97` (`notes < 1`; `any(... abs(a - amount) < 0.005)`) | Reuse `verify._note` (a reason word per ticket kind) and the `approval_grounded` binding (consumed; the request names the ticket or order). |
| B9 | P2 | **`RLMHarness` refuses a host kernel only at attempt time.** Every attempt is masked `agent_run_error` (unknown blame, retryable) and counts toward the unknown-blame stop. `rlm_world_rollout` refuses at construction. | `$V/runs/rlm_probe/host` | `runtime/harness/rlm.py:120` (`c.launcher()` inside `wrap`) | Call `config.launcher()` once in `RLMHarness.__init__` (as `world_env.py:363` does) and raise. |
| B10 | P2 | **A cell can forge a well-formed `done`** on the private protocol fd, because `_protocol_fd` and `_active['cid']` can be read from `sys.modules['__main__']`. The host accepted it: the episode ended and `digest = deadbeef` was journaled, so a later restore fails as `policy_error`. The effect stays inside the policy's own episode (scored, never masked), but `repl.py:9-13` overstates the isolation. | `$V/runs/rlm_probe/forge2` | `actors/rlm/repl.py` (module globals), `actors/rlm/kernel.py:417` (`_read`) | A per-session HMAC on frames, with the key read once from the request channel and deleted from globals. Or accept one `done` per cid and recompute the digest on the host. |
| B11 | P2 | **coding's semantic and adversary level never runs on CPU** although its `tasks` are CPU dense-replay tasks. `requires: [gpu, docker]` covers the whole pack. With a false `--allow gpu` it passes on CPU in 60 s. | `$V/logs/check_coding_semantic_allowgpu.log` | `packs/coding/pack.yaml` `requires`, `studio/pack.py` LIVE levels | Allow `requires` per level or per hook (live GPU grading versus CPU tasks). |
| B12 | P2 | Reusing one key file for a second pack's bench is refused with a misleading message ("a new version needs a new key" for a different benchmark). | `$V/logs/bench_key_reuse.log` | `bench/dynamic.py:142` | Name the conflict (another benchmark), or scope the registry by benchmark name. |
| B13 | P2 | **Tool-fault attribution and argument laxity.** `ValueError`s the policy causes (invoice 99999, a reject reason used for a hold, `status='*'`, a kb path traversal) are recorded `fault: tool` (`ToolError`); schema violations are `fault: policy`. Unknown argument names are silently dropped (`get_invoice(vendor_id=…)` returns ok) and `100.0` is coerced to an id. Episodes still score, never mask. | traces in `$V/runs/scripted/bench-ap/samples/*` (`new:malformed_only`) | the World tool layer (`ToolError` mapping); `packs/*/ap_tools.py`, `tickets.py` | Map `ValueError` from pack tools to `invalid_arguments` with policy fault, and reject unknown argument names. |
| B14 | P2 | When the harness step limit is below the oracle's length (finance_ap n = 12: 73 steps against a limit of 60), the controls call the task `broken_oracle` and silently drop it from the benchmark. | `$V/runs/llm/bench-ap-qwen-steps60/validity.json` | `bench/run.py` `run_controls` | Report `oracle_truncated (max_steps)` and warn that the run's limits are below the oracle length. |

### B1 traceback (abridged)

```
crucible/actors/policy.py:589  res = await world.step(a, action)
crucible/world/world.py:612    self._log("action", actor, "tool_call", data={"function": ..., "arguments": action.arguments, ...})
crucible/world/world.py:872    f.write(ev.model_dump_json(exclude_none=True) + "\n")
PydanticSerializationError: ... 'utf-8' codec can't encode character '\udcff' ...: surrogates not allowed
crucible/runtime/infra.py:94   raise StageFailure(Failure(kind="agent_run_error", ..., blame="unknown"))
```

## 6. What held

- **Pack checks.** `pack check` static, fixture and semantic pass for toy and finance_ap. finance_ap passes accept as provisional (A10 pending).
- **Controls.** The oracle scored 1.0 and the no-op 0.0 on every task in every run (scripted, CLI, fuzz, RLM single-cell oracle). No NaN and no out-of-range reward appeared anywhere.
- **Declared adversaries.** All 5 finance_ap adversaries scored 0 on 9 bench tasks and 150 fuzzed worlds. Both toy adversaries scored 0 on the anchor.
- **Earlier and new adversaries.** All 13 earlier and new toy approval or resolution adversaries scored 0 on the anchor. The earlier campaign's finance_ap maskers (`2**63` ids) are now scored 0, not masked.
- **Malformed calls.** 26 malformed call types never masked, and malformed calls plus the oracle still scored 1.0.
- **RLM.**
  - The Docker kernel is the default, and the host kernel is refused.
  - Infrastructure faults are masked and retried; the policy's own crashes are scored.
  - Sub-call budgets are enforced, and other-model sub-calls are non-trainable.
  - Compaction under a thinking template is no longer masked.
  - Qwen RLM and native records are train-valid, with a logprob mismatch ≤ 1e-4 and a finite trainer step.
- **Training.** Training smokes on toy and finance_ap (native and RLM) had no masked storms, no NaN, and rewards in [0, 1].
- **Distribution.** Pack wheels carry their data and reach gold 1.0 when used outside the checkout.
