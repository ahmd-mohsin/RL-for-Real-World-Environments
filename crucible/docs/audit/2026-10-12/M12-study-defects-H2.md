# M12 study, hunt H2: defects in the control and data planes of `train`

Scope: `C/train/{runtime, scheduler, pool, sim, state, control, control_record, monitors, admission, noise, seats, envpool, envs, sinks}.py`, and how they meet `engine.py` and `weights.py` at publish and abort. Tests read: `test_train_{runtime, sched, pool_scale, resilience, reward_async, state, control_records, multidomain, envpool, seats, monitoring, sinks}.py`.

Spec: `docs/modules/M12-train.md` §1 (planes), §3 (topologies, partial barriers), §4 (staleness, buffer policies, backfill), §9 (checklist status).

Paths: `C` = `crucible/src/crucible/`. `P` = the session scratchpad `m12/h2/` (`/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m12/h2/`). Every reproducer `P/rNN_*.py` has its captured output next to it in `P/rNN_*.out`.

Method: a read-only study on HEAD `3b522cb` with `crucible/.venv` (Python 3.12.9, torch 2.14) on macOS. No GPU, Docker, network or paid call. **[ran]** means reproduced with a probe. **[read]** means established from the code alone.

The drivers run the real `Runtime`, `Scheduler` and `DataPool` (`P/harness.py`):
- `RandEngine` honours pinned versions, cooperative abort, pause and token-level resume with spans;
- `AdvEnv` has a random number of turns, long tails, `EnvInfraError` masks and generic crashes;
- `CheckTrainer` fails a step on a double train, a masked record, or a record whose staleness exceeds τ for its source (τ + 1 under `allow`);
- every run asserts progress and `max_untrained ≤ S·B·n`.

## 0. What holds (fuzz results)

These parts are clean. The findings below are the exceptions.

- **Scheduler and pool, 20,000 seeds** (`P/fuzz_sched.py`, supported configurations). Configurations:
  - n ∈ {1, 2, 4}, B ∈ {1, 2, 4}, τ ∈ {0..3};
  - `wait` or `drop`; `bound_overflow` `batch` or `drop`; `oldest_first` or `greedy`;
  - zero-variance filter, every backfill mode, per-source τ, S ∈ {1, 2, 4}, `min_groups` 1;
  - `publish_every` 1 or 2, masks up to 30 %.

  Result: 0 double trains, 0 masked or evicted members trained, 0 staleness above τ, 0 cap violations, 0 deadlocks.
- **Runtime, 480 seeds** (`P/fuzz_runtime.py … clean`, `P/fuzz2_*`). Every topology, `call` and `token`, all three partial barriers, `max_cuts` σ ∈ {0, 1, 2}, `publish_every` 1 to 3, warmup, infra masks up to 20 %, `n = 1` critic mode, B = 1. Result: 0 issues.
- **Two sources with per-source τ, 160 seeds** (`P/fuzz_multisource.py`; source `a` has τ = 0 or 1). Result: 0 issues.
- **Status transitions** (`P/fuzz_transitions.py`). Nothing leaves `trained` or `evicted`. Masked members do become `trained` (V17).

The core (admission cap, `wait` on group-oldest including finished members, at-bound-first, `bound_overflow=batch|drop`, `drop`, per-source eviction, lazy heap, release) is sound. Every P0 below is outside that core:
- a failure path in the rollout tasks (V1);
- the group stage (V2);
- the three optional selection filters (V4 to V6).

## 1. Checklist ledger (§9 items this hunt can judge)

| # | §9 item, claimed status | Finding |
|---|---|---|
| 1 | state plane, partial ("bitwise resume in sync mode; in-flight rollouts restart") | partial. With `publish_every > 1` the resume re-publishes newer weights under an old version (V12). Game seats are not checkpointed (V20) |
| 17 | group backfill, done | partial. `refill_member` has no bound (V10). Refilled members skip the group stage (V2). The simulator never launches them (V14) |
| 25 | `max_offpolicy_frac`, done | **wrong**. Trains far past τ (8 at τ = 3 in the shipped test's own run) and can deadlock (V4). The test is vacuous (V11) |
| 26, 27, 28 | windowed FIFO rule, Fig. 9, lag ≤ 10, done | **wrong**. Deadlocks whenever fewer than B terminal groups fit in the window (V5) and trains past τ (V6). The lag-bound test runs 0 steps (V11) |
| 40 | async reward deadlines, done | partial. A NaN reward or a sync `reward_fn` loses the rollout (V1) |
| 42 | infra mask vs agent failure, done | partial. A crash outside `EnvInfraError` loses the rollout instead of masking it (V1). A masked record is counted as ok (V8) |
| 51 | per-step monitor set, done | partial. `confirmed_hack_rate` divides audit confirmations by the whole batch (V13) |
| §4 | "`allow` (staleness may reach τ + 1)" | **wrong**. Unbounded (V3) |
| §1 | "a sink that raises is disabled, never fatal" | live (tested). Not covered: a sink that blocks |
| §6 | "control-plane indexes" | live (the fuzz above) |

## 2. Defect candidates

Severity: **P0** = wrong training data or staleness, a double train, a lost rollout, or a silent mask. **P1** = a contract break, a deadlock or a vacuous test. **P2** = smaller.

### Lost rollouts

**V1 [P0][ran] Any exception other than `EnvInfraError` after a rollout starts loses it silently, and the run stalls.**
- Where:
  - `runtime.py:480-525` `_rollout`: the token-env loop catches only `EnvInfraError` (`:517`); `_complete` sits in `try/finally` with no `except`;
  - `runtime.py:445` `asyncio.create_task(reward_fn(rec))` and `:465` `rec.reward = float(reward)` both sit outside `_grade_inner`'s `try`.
- The exception escapes the task. Asyncio prints "Task exception was never retrieved". `_complete` never runs, so:
  - the tag stays `running` or `grading` for good;
  - the rollout is not masked and not in `failures`, `masked_rollouts` or the breaker window;
  - its group never turns terminal and keeps its untrained slots.
- Under `wait` the group ages to τ and blocks every barrier. Under the other modes the cap shrinks with each loss until admission stops.
- Triggers:

  | trigger | probe | stuck tags | failures recorded | result |
  |---|---|---|---|---|
  | an environment bug (`ValueError`) | `P/r01_env_crash_lost.py` | 2 `running` with no task | `{}` | 2 of 8 steps; untrained 16 = cap 16; stop `wall_clock` |
  | a synchronous `reward_fn` | `P/r01b_other_triggers.py` (a) | 8 `grading` | `{}` | 0 steps |
  | an engine `ConnectionError` in the token loop (an HTTP engine's transport error, which is infra) | `P/r01b_other_triggers.py` (b) | 2 `running` | `{}` | 3 of 5 steps |
  | a grader returning NaN: RolloutRecord's finite check raises on assignment | `P/r17_nan_reward.py` | 1 `grading` | `{}` | |

  A NaN in an environment's `Step.reward` fails the same way, at the record's construction.
- The resilience test (`test_a_harness_bug_stops_the_run_with_its_reason`) covers only the `rollout_fn` path, which does catch `Exception` (`:300-303`).
- Fix:
  - In `_rollout`, wrap the whole episode, including `factory()`, record construction and `_complete`'s grading, in `except Exception as e: self._failure(_blame(e), e); await self._complete(rid, None, failed=True)`.
  - In `_grade_inner`, put `create_task(reward_fn(...))` inside the `try`. Validate the reward with `math.isfinite`, and mask a non-finite reward as `grader_nonfinite`.
  - As a backstop, give each rollout task a done-callback: a task that ended with the tag still live masks it and records `harness:<type>`.
- Regression test (fails now): `CopyTask` subclass raising `ValueError` 5 % of steps; assert `rep.stop_reason in ("steps", "harness_share")`, `rep.failures["harness:ValueError"] > 0`, no tag in `LIVE` without a task; same with a NaN `reward_fn` → masked `grader_nonfinite`.

### Wrong training data

**V2 [P0][ran] `refill_member` plus a `group_fn`: the stage result is stale, so refilled members bypass GAR and hack zeroing.**
- `runtime.py:363-368` launches the stage the first time the group turns terminal. At that point the group includes the masked member.
- At the next `select`, `_backfill` evicts the masked member and adds a new one, and the group re-opens (`scheduler.py:146-150`).
- When the group turns terminal again, `gid in self._group_stage` is true, so no new stage is launched.
- At the barrier, `_apply_group_stage` installs the old result. It covers only the original members. The refilled members train on raw rewards, with no `advantage` override and no hack zeroing, while their siblings were rewritten.
- `P/r09_refill_group_stage.py` (n = 4, 20 % infra, the operator sets `reward = 0.123`): 17 of 20 trained groups mix stage and raw members, e.g. `a:a1@1 [('7', 1.0, 0.123), ('8', 1.0, 0.123), ('17', None, 0.0), ('21', None, 1.0)]`.
- A confirmed hack among the refilled members is trained with its full reward.
- Fix:
  - when `_backfill` re-opens a group, cancel and pop its `_group_stage` entry;
  - or key the stage by the member set, and at the barrier re-run it when the members differ.
- Regression test: the r09 driver; assert every trained record of a group carries the stage's mark.

**V4 [P0][ran] `max_offpolicy_frac` skips off-policy groups past the cap without dropping them, so they train past τ or deadlock the run.**
- `scheduler.py:205-210` removes the excess off-policy groups from this barrier's candidates, including groups at their bound. Nothing evicts them, and `wait` checks only live groups. They age without limit.
- `P/r02_r05_scheduler.py` R4a uses the configuration of `test_max_offpolicy_frac_and_ks` (n 4, B 4, τ 3, frac 0.25). It trains at staleness **8**. The same run without the cap stays at 3.
- R4b: when every buffered group is off-policy and the untrained cap is full, `select` returns `need_more` forever and nothing is admitted.
- Scheduler fuzz: 27 non-windowed `wait` seeds over τ, plus deadlocks (`P/fuzz_sched_5000.txt`). Runtime fuzz seed 80 (`colocated_partial`, `token_budget`, frac 0.5) trains at staleness 4 with τ 3 (`P/fuzz_out_1.jsonl`).
- Fix:
  - treat groups at their bound as exempt from the cap;
  - or evict off-policy groups the cap can never admit (staleness ≥ τ) as `stale`, so admission refills them;
  - never block with a full cap and nothing live: fall back to the at-bound groups.
- Regression test: R4a asserts `max_staleness() ≤ τ`. R4b asserts a batch, or a refill, within one barrier.

**V6 [P0][ran] `windowed_fifo` trains past τ.**
- `scheduler.py:195-199` filters candidates to `[head, head + W)` *before* the at-bound-first sort (`:204`) and before `bound_overflow` (`:223`).
- A finished group at its bound but outside the window is neither trained nor dropped, and it ages past τ.
- Scheduler fuzz, windowed only: 263 of 5,000 seeds train above τ, both with W ≥ B (140) and W < B (123). For example, seed 13 (n 2, B 2, τ 1, `wait`) trains at 2.
- Fix:
  - exempt at-bound groups from the window (they are the head's peers in age);
  - or drop them under the `drop` policy.
- Regression test: the fuzz seed as a unit test, with `max trained staleness ≤ τ`.

**V8 [P1][ran] A masked record returned by a `rollout_fn` is pooled as `finished` and counted as ok.**
- The record is masked when it has `mask_sample=True` or `reward=None`. A game whose trained seats all have no outcome is the in-tree case (`seats.split_seats`).
- `runtime.py:355-362`:
  - `pool.finish(..., status="finished")` and `_recent.append("ok")`;
  - `masked_rollouts` is not incremented;
  - the breaker cannot see it;
  - group backfill never applies;
  - the tag reward feeds the zero-variance filter.
- `P/r08_mask_sample_record.py`, with half the records masked:
  - `reward=None`: the step crashes in the monitor (`TypeError '>=' … NoneType`, `monitors.py:80`);
  - a reward kept: 4 steps, breaker window `{'ok': 16}` against a 20 % limit, `masked_rollouts 0`, and `reward/mean` and `pass_rate` computed over the masked records.
- The trainer drops them from the loss (`trainer.py:312`), so they are not trained. They are still listed as trained records and sent to sinks, audits and the noise curriculum.
- Fix: in `_complete`, a record with `mask_sample` or `reward is None` goes the masked way. For seats, the episode is masked when no seat has an outcome.
- Regression test: the r08 driver; assert `masked_rollouts ≈ half` and `stop_reason == "masked_share"`.

### Deadlocks and contract breaks

**V3 [P1][ran] `bound_overflow="allow"` has no bound.**
- At-bound groups beyond B are left in the buffer (`scheduler.py:223-235`). They sort first again, but only B train per step, so with a steady overflow the leftovers age by one each publish.
- R3 (n 1, B 1, τ 1, S 4) trains at staleness 1, 2, 3, 4. The doc (§4) says "may reach τ + 1".
- Fix: at staleness τ + 1, `allow` must fall back to `batch` or `drop` for those groups. Otherwise document it as unbounded.
- Regression test: R3 asserts `max ≤ τ + 1`.

**V5 [P1][ran] `windowed_fifo` deadlocks.**
- `head` is computed once per barrier (`scheduler.py:196-198`), and the window counts submission indexes, holes included.
- If fewer than B terminal groups fit in `[head, head + W)`, the barrier returns `need_more`. With the untrained cap full and nothing live, it does so forever.
- W < B always does this (R2: τ 2, B 4, n 1, the default `fifo_window_frac` 0.3 gives W 3). τ = 2 with B = 4 is the GPU gate's τ.
- With W ≥ B, evicted or trained groups inside the window shrink it below B (fuzz seed 8, W 2, B 2).
- Windowed fuzz: 1,920 of 5,000 seeds deadlock.
- MiniMax's rule advances the head as the head item is consumed. A batch of 8 with W 4 is exactly the article's example.
- Fix:
  - take candidates greedily inside the window, advancing `head` past each selected or no-longer-pending group;
  - define W over pending groups, not raw indexes;
  - refuse W < B at configuration if the static rule is kept.
- Regression test: R2 asserts a batch at v = 0.

**V9 [P1][ran] Dynamic sampling resamples without a bound, and the spin is silent.**
- `scheduler.py:172-178` evicts zero-variance groups, and admission refills them every barrier. There is no attempt cap (DAPO has `max_num_gen_batches`).
- No step is logged, and the eviction counters reach metrics only inside a step (`runtime.py:757`).
- `P/r06_dynamic_sampling_unbounded.py` (a source the policy has solved): 0 steps, 650 prompts consumed, 646 evictions, `failures {}`, 0 metric lines, stop `wall_clock`.
- The runtime fuzz hit the same stall in 21 of 400 seeds (every `filter_zero_variance` run on a constant reward).
- Fix:
  - a `max_resample_rounds` (or an evicted-to-trained ratio) after which the barrier trains what it has, or stops with `stop_reason="zero_variance_exhausted"`;
  - log eviction counters on blocked barriers (a periodic heartbeat metric).
- Regression test: the r06 driver; assert a stop reason other than `wall_clock`, or progress.

**V10 [P1][ran] `refill_member` has no refill cap, so one broken prompt stops the run.**
- `scheduler.py:146-150` refills every masked member at every barrier.
- `P/r16_refill_unbounded.py`: one prompt that fails every attempt after its first, out of 100,000.
  - Default breaker: 0 of 20 steps, stop `masked_share`.
  - Breaker off: 1,779 refills, 2 steps, then `wait_stale` forever. The group's finished member ages, and the group stays live through its refills.
- Fix: `Group.refills` exists. Add `max_refills` (default 1 or 2 × n) and then evict as `drop_group` (`masked_member`).
- Regression test: the r16 driver; assert 20 steps and the bad group evicted.

**V7 [P1][ran] The group stage crashes on every World rollout.**
- `world_rollout` records carry `rollout_id = "<rid>-a1"`: `runtime/record.build` uses `rec.attempt`. Seat records are `"<rid>-a1#seat=<s>"`. The pool keys values by `rid`.
- `_apply_group_stage` does `self.pool.values[r.rollout_id]` (`runtime.py:411-412`), which raises `KeyError`.
- `P/r07_group_fn_world_rollout_id.py` (toy helpdesk, an identity `group_fn`): `RUN CRASHED: KeyError 'helpdesk:h0@0/1-a1'`.
- The hack-defence chain (`detector_group_fn` then GAR) exists for World rollouts, so it cannot run in the runtime.
- Fix: keep a `rid` per member in `_run_group_fn` and write back by position (the output order is already checked), not by `rollout_id`.
- Regression test: r07; 2 steps and the stage's mark on every trained record.

**V11 [P1][ran] Two scheduler tests are vacuous, and §9 marks their items done.**
- `test_train_sched.py:80-82`, the lag ≤ 10 bound (#28): the simulation runs **0 steps** (W 4 < B 8, see V5). `max_staleness()` of an empty run is 0, so it passes.
- `test_train_sched.py:103-106` (#25): the bound `max(1, len(b) // 4) * 4` equals `len(b)`, the batch's rollout count. It holds with the cap removed (R4a), and the run it checks trains at staleness 8 with τ 3 (V4).
- Fix:
  - assert `r.steps == 12` in the first;
  - assert off-policy groups ≤ ⌊frac·B⌋ per batch and `max_staleness() ≤ τ` in the second;
  - set #25, #26 and #28 back to partial.

### Smaller

**V12 [P2][ran] A resume after a non-publishing step aliases a version.**
- With `publish_every > 1`, a checkpoint on a step that did not publish stores `version = v` with weights one step newer (`state.py:52-56`).
- `Runtime.resume` then calls `publisher.restore(v)`, which serves the newer weights as version v (`weights.py:43-49`).
- `P/r12_resume_publish_every.py`: the engine is at version 1 before and after, `same weights under that version: False`.
- Records labelled v now come from two weight sets. Their staleness and any per-version π_old lookup are off by one optimizer step.
- Fix: checkpoint only on publish steps, or publish a new version on resume when `updates` is past the last publish.

**V13 [P2][ran] `confirmed_hack_rate` uses the wrong denominator.**
- `runtime.py:754` adds the audit confirmations finished since the last step to the batch's detector confirmations. `monitors.py:91` divides by `len(batch)`. Audits are an `audit_rate` (5 %) sample, so the rate reads about `audit_rate ×` the true rate.
- `P/r14_hack_rate_denominator.py`: 33 % true hack share, reported mean **1.8 %** (the threshold is 2 %). The alarms that fired were noise spikes.
- Fix: report `audit/fast_pass_full_fail ÷ audit/runs` as the audit rate, and keep the detector rate over the batch.

**V14 [P2][ran] The simulator never launches refilled members.**
- `sim.py:177` starts members only of newly admitted groups. A member added by `_backfill` stays `pending`.
- `P/r10_sim_refill.py`: `pad` runs 15 of 15 steps, `refill_member` 0 of 15. Sweeps that use refill are wrong.
- Fix: launch `pool.pending()` as the runtime does.

**V15 [P2][ran] The simulator is not deterministic across processes.**
- `default_shapes` uses the salted `hash(prompt)` (`sim.py:48`).
- Three `PYTHONHASHSEED`s give different rewards and different zero-variance evictions: 19, 24 and 28.
- Fix: use `zlib.crc32`.

**V16 [P2][ran] `_group_stage` and `_seats` leak for groups evicted at blocked barriers.**
- The "evicted earlier" release loop (`runtime.py:729-730`) does not pop or cancel them.
- `P/r18_stage_leak.py`: 43 entries for released groups after 30 steps.

**V17 [P2][ran] A masked member becomes `trained`.**
- `mark_trained` moves every terminal status to `trained` (`pool.py:221-227`), masked members included. 11 such transitions in 30 runtime seeds (`P/fuzz_transitions.py`).
- Tags then lose the masked-versus-trained distinction for accounting.
- Fix: leave masked members `masked`, or move them to `evicted`.

**V18 [P2][ran] Episodes on env workers skip their cleanup at shutdown.**
- `run()` cancels the rollout tasks, which only requests cancellation on the worker loops, then `EnvWorkerPool.close()` stops those loops (`runtime.py:666`, `:782-784`; `envpool.py:93-97`).
- `P/r19_envpool_shutdown.py`: 12 episodes started and 4 cleanups never ran ("Task was destroyed but it is pending"). In `world_rollout` that is a World never closed and its temp dir left behind.
- Fix: before `close()`, cancel the worker tasks on their own loops and await them with a bounded timeout.

**V19 [P2][ran] `admission.screen` keeps a broken task when the verdict is a dict.**
- `_passed` returns `bool(v)` for any non-tuple verdict (`admission.py:41`), so a non-empty dict passes. A constant `{"reward": 0.0, …}` verdict screens as **stable**.
- Fix: read `v["reward"] >= 1` for mappings.

**V20 [P2][read] Game seats have three gaps.**
- `_seats` is not in the checkpoint (`state.capture`). After a resume, a finished game trains only seat 0 under the plain group id, and the other seats are dropped.
- A source's `reward_fn` is skipped for game rollouts (`runtime.py:348`).
- The zero-variance filter reads only seat 0's reward (the tag).
