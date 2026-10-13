# M13 `bench` framework sweep, group 1: how every local framework evaluates, vs Crucible (2026-10-13)

Read-only, Crucible HEAD `b6d89fc`. Nine probes back every [ran] row. They live in
`scratchpad/m13/g1/p1..p9_*.py` and run from `crucible/` with `PYTHONPATH=src .envs/nemo/bin/python` (scipy 1.18.1,
numpy 2.5.3; crucible.bench imports cleanly there). Framework estimators are ported verbatim into the probes. NeMo-RL's
torch code is ported to numpy, with the same arithmetic.

Roots and pins (`F/` = `RL-Environments/06-frameworks-and-env-code/`, `CR/` = `crucible/src/crucible/`):

| Root | Repo | Commit | Licence |
|---|---|---|---|
| H | `F/harbor-framework-harbor/src/harbor` | a2fafbc | Apache-2.0 |
| VF | `F/primeintellect-ai-verifiers/verifiers/v1` | ae819de | MIT |
| PE / RE | `F/primeintellect-ai-prime-environments`, `F/primeintellect-ai-research-environments` | cf220ee / d4fb80a | Apache-2.0 |
| NG | `F/nvidia-nemo-gym/nemo_gym` | 3ef478d | Apache-2.0 |
| OE | `F/meta-pytorch-openenv/src/openenv` | 49aa302 | BSD-3 |
| AE | `agentenv-framework/src/agent_env` | 041afd7 | Apache-2.0 |
| RL | `F/infra-rllm-org-rllm/rllm` | 4e828c8 | Apache-2.0 |
| SK | `F/infra-novasky-ai-skyrl/skyrl` | 3d0ae22 | Apache-2.0 |
| VE | `F/infra-volcengine-verl/verl` | 5eb4826 | Apache-2.0 |
| NR | `F/infra-nvidia-nemo-rl/nemo_rl` | c9193bc | Apache-2.0 |
| TRL | `F/infra-huggingface-trl/trl` | 72cf37d | Apache-2.0 |
| AR | `F/infra-inclusionai-areal` | 298412a | Apache-2.0 |
| SL | `F/infra-thudm-slime/slime` | c933787 | Apache-2.0 |
| PR | `F/infra-primeintellect-ai-prime-rl/src/prime_rl` | f7e8ab6 | Apache-2.0 |
| ROLL / ORL | `F/infra-alibaba-roll/roll`, `F/infra-openrlhf-openrlhf/openrlhf` | 581046a / dc2a7ad | Apache-2.0 |
| MS / SS | `F/swe-agent-mini-swe-agent/src/minisweagent`, `F/swe-bench-swe-smith/swesmith` | 04d809c / 9b74ac0 | MIT |
| EO | `F/servicenow-enterpriseops-gym` | 271f2c3 | Apache-2.0 |

Severity: **P0** a wrong benchmark number or a silently miscounted sample; **P1** a contract break, a vacuous test, or a
CI that does not cover; **P2** a gap. Verdicts: DEPEND / PORT / PATTERN / nothing (Crucible is equal or ahead).

## 0. Headline defects

| # | Sev | Where (CR) | Defect | Evidence |
|---|---|---|---|---|
| D1 | **P0** | `bench/compare.py:32-36`; CLI `cli/studio.py:293-294` | `compare` keys episodes on `(task, sample_idx)` only. In a run with several policies or harnesses, the dict keeps the last episode per key and drops the rest. Nothing reports this (`unmatched` = 0). `crucible bench compare A B` has no filter by default and no harness filter at all. The module doc says "or two policies in one run". | [ran p2] base = 2 policies × 10 episodes (p1 all pass, p2 all fail), cand 10 → `pairs 10, unmatched 0, mean_delta 1.0, McNemar p 0.002`. p1's 10 passes vanished. One policy × 2 harnesses (20 eps) → `pairs 10, unmatched 0`. |
| D2 | **P0** | `bench/metrics.py:193-195, 204-205, 218-221` | Each k uses its own task set. Every k drops tasks with fewer than k measured samples, so a task that lost a sample to masking leaves pass@k, pass^k and max@k but stays in pass@1. No count of the tasks used is reported. Hard tasks mask more often, so this biases pass@k upward, and pass@k can exceed pass@1 by a large factor. NeMo-Gym (`NG/reward_profile.py:581`) and prime-rl (`PR/orchestrator/metrics.py:383-386`) have the same flaw. Harbor (`H/utils/pass_at_k.py:58-59`) and ROLL/LiveCodeBench (`ROLL/utils/local_code/pass_k_utils.py`, `(total >= k).all()`) keep one task set. | [ran p8] a=[1,1,0,0], b=[0] (3 masked): pass@1 0.25, **pass@2 0.833** (b dropped); rLLM 0.417 (b falls back to any-pass); Harbor reports only k ≤ min n = 1. [ran p1] A 2/2 pass, B 1 fail + 1 masked: pass@1 0.5, pass@2 1.0, pass^2 1.0. [ran p7] NeMo-Gym gives the same numbers: binary pass@k agrees on 2000/2000 random benchmarks. |
| D3 | **P0** | `bench/metrics.py:289` (`mean_reward`), `:242-243` (bootstrap statistic) | The headline mean is weighted by episode, not by task. Under masking, tasks that keep more samples weigh more. Masking concentrates on hard tasks (stale images, harness crashes), so the headline drifts upward. The CI then concentrates on the wrong value. `pass@1` (macro over tasks) and `mean_reward` (micro) disagree on the same binary data. | [ran p1] A 8/8 pass, B 1 measured fail + 7 masked: **mean_reward 0.889, pass@1 0.5**. [ran p3] Hard tasks (p<.3) mask each sample w.p. .6, true mean 0.40: micro bias **+0.072 to +0.083** at every T. At T=200, k=4 the Crucible CI covers **12.6%**; a t-interval on task means covers 91.6%. |
| D4 | **P0** | `bench/metrics.py:83-85, 109, 311-315`; `runtime/journal.py:97-100` | Cost per success leaves out the spend of retried attempts. `load_run` reads only `results.jsonl`, which holds one terminal record per sample. A masked attempt that was retried exists only in `failures.jsonl` and `samples/*/a<N>/record.json`, so its USD, tokens, wall time and GPU seconds disappear. `cost_uncertain_usd` (harness crashes, blank usage) is never added. This breaks the contract "per success = all spend (masked included)" (`metrics.py:20`). | [ran p5] a1 provider outage $2 (5000 tok) retried → a2 pass $1 (1000 tok); a second sample has a harness crash with $3 uncertain. True spend is $6 / 6000 tok; the bench reports **usd 1.0, tokens 1000, per_success 1.0**. |
| D5 | **P1** | `bench/metrics.py:232-245`, `bench/compare.py:53-60` | The percentile bootstrap undercovers at the sizes Crucible actually runs: the toy-live gate has 7 tasks, and the coding recording has 1–8 tasks per model. No task count and no warning come with the CI. | [ran p3] coverage without masking: T=5 **0.833** (k=1) / 0.835 (k=4); T=10 0.908 / 0.905; T=20 0.921 / 0.927; T=50 0.938 / 0.947; T=200 0.946 / 0.956. A t-interval on task means: 0.91–0.95 at every T. scipy BCa on task means: 0.85–0.96. [ran p4] paired bootstrap, true Δ=0.092: T=5 0.689 / 0.854, T=10 0.855 / 0.896, T=50 0.938 / 0.944; under the null, T=5 k=4 0.876. |
| D6 | **P1** | `cli/studio.py:245` (`bench run`), `:273-277` (`bench report`), `studio/build.py:94-95`, `studio/explorer.py:116` | pass@k and pass^k for k>1 cannot be reached from any entry point. Every caller leaves `ks=(1,)`, so `crucible bench run --k 2` draws 2 samples and reports only pass@1, pass^1 and max@1. max@1 equals the mean, and pass^1 equals pass@1. The doc's "**pass^k**" reliability headline never appears. | Both recorded reports carry `"ks": [1]`, and toy-live ran k=2 (`docs/modules/reports/m13_toy_live/native_summary.json`). Harbor reports every eligible k automatically (`H/utils/pass_at_k.py:71-84`). |
| D7 | **P1** | `bench/compare.py:32-49`; `runtime/keys.py:24-25` | Flips pair sample i of the base with sample i of the candidate. The seed hashes the policy (`derive_seed(run_seed, task, policy, harness, idx)`), so two policies share no random numbers at the same index. The pairing is therefore arbitrary: per-sample "regressions" are noise from sample order, yet they are listed first. McNemar stays valid under H0 (p4 null rejects ≤ 3.9%), but the regression list misleads. | [ran p9] per-task rates identical (1/2 each), sample order swapped → **10 "regressions" + 10 "improvements" listed**, net 0, p = 1.0. Seeds: A idx0 1314048554, B idx0 1299686859. NeMo-Gym flips per task on the group mean (`NG/comparison/diff.py:205-359`). |
| D8 | **P1** | `tests/unit/test_bench.py:27-37, 157` | The estimator gate is vacuous for every defect above. Estimators are checked on one task only. No test covers multi-task denominators with masking, micro vs macro weighting, a multi-policy compare (the toy compare pre-filters by policy at `:157`), `load_run` with retries or uncertain cost, CI coverage, or ks>1 from the CLI. | 4 tests; D1–D7 all pass the suite. |
| D9 | P2 | `bench/metrics.py:40-61, 92` | `Episode` has no failure kind, retry count or stop-reason aggregate. A masked episode reports only as a count; `Episode.stop` is loaded and never used. Communication counts only `actor == "agent"`, so P28 peer policies are not counted. | Harbor `exception_stats` and `n_retries` (`H/models/job/result.py:129-169`); verifiers error rate plus per-row error type (`VF/cli/dashboard/eval.py:331-350, 601-622`); slime `truncated_ratio` (`SL/observability/rollout_metrics.py:236-254`); OpenRLHF `truncated_rate` (`ORL/trainer/ppo_trainer.py:82-140`). |
| D10 | P2 | `bench/metrics.py:291` | The only uncertainty reported is one CI, on mean reward. pass@k, pass^k, AUC and cost per success have none, and there is no spread across sample indices. Study row 3 (`docs/research/bench-studio/01-framework-study.md:45`) promised t-CIs, a warning on unequal coverage, and bootstraps for AUC and cost per success. | NeMo-Gym `std_dev/std_err_across_runs` (`NG/reward_profile.py:627-656`), per-repeat t-CI (`:242-309`), across-repeat CI (`:311-353`); verl `std@N` (`VE/trainer/ppo/metric_utils.py:1020`); Harbor's adapter rule requires mean ± SEM (`H/../adapters/ADAPTER_CONTRIBUTING.md:339-349`). |
| D11 | P2 | `bench/metrics.py:319-338` | Results break down only by horizon, never by category or subset (repo, API family, OOD axis, data source). | NeMo-Gym `compute_subset_metrics` (`NG/reward_profile.py:692-725`); verl per `data_source` (`VE/.../metric_utils.py:896-1074`); OpenRLHF per datasource; tau2 per domain (`NG/../resources_servers/tau2/app.py:443-552`). |
| D12 | P2 | `bench/metrics.py:99-102, 311` | Tokens are one integer. There is no split into prompt, cached, completion and reasoning tokens, no latency percentiles, no per-phase wall time, and no turn counts. | Harbor `n_input/n_cache/n_output_tokens, cost_usd` and per-phase `TimingInfo` (`H/models/job/result.py`, `H/models/trial/result.py:71-138`); NeMo-Gym `compute_perf_summary` (`NG/reward_profile.py:1003-1063`); verifiers per-phase timing (`VF/cli/dashboard/eval.py:421-501`). |
| D13 | P2 | `runtime/run.py:151-157` (cross-ref M10) | A policy that runs past `attempt_timeout_s` is masked as a harness fault and retried, never scored. The bench shows it only as "masked", with no kind (D9). | Harbor still runs the verifier after `AgentTimeoutError` (`H/trial/single_step.py:75-87`); a verifiers stage timeout is not an error (`VF/rollout.py:211-215`). |
| D14 | P2 | `bench/compare.py:21-28` | `mcnemar_exact` sums big-integer binomials: 2.9 s at b+c = 10 200. | [ran p2] value matches scipy (0.048788). |

## 1. Metric-by-metric equivalence ([ran] unless noted)

| Crucible metric (CR file:line) | Framework implementations (file:line) | Probe result | Who is right |
|---|---|---|---|
| pass@k `metrics.py:185-195` (Chen et al. 2021, `math.comb`, threshold `reward >= 1.0`) | Unbiased: Harbor `utils/pass_at_k.py:87-95` (product form, binary only, no k=1); NeMo-Gym `reward_profile.py:575-595`; rLLM `eval/results.py:27-44`; prime-rl `orchestrator/utils.py:97-115`; slime `observability/metric_utils.py:14-56` (`== 1`, k = powers of 2); NeMo-RL `evals/eval.py:178-203` (c = **sum of rewards**); ROLL `local_code/pass_k_utils.py:6-46` (×100). Biased any-of-n: SkyRL `generators/utils.py:260` (`r > 0`); NeMo-RL GRPO `algorithms/grpo.py:4461-4492`; AReaL `examples/swe/arena_agent.py:594`; OpenRLHF `ppo_trainer.py:115` (max raw reward); rLLM trainer `trainer/unified_trainer.py:894`. verl: no pass@k. | p1: Crucible vs brute-force subset enumeration on 3000 random groups (ties, continuous): max error **4.4e-10** (rounding to 9 digits). p8, 3 tasks × 4, c = 1, 2, 0: Crucible = rLLM = prime-rl = NeMo-RL = 0.2500 / 0.4444 / 0.6667 at k = 1 / 2 / 4. p6: Harbor = Crucible at k = 2, 4. | The single-task estimator is correct. **The task set across k is wrong (D2).** NeMo-RL is wrong on non-binary rewards: [0.3]×4 gives pass@2 **0.724** (c = 1.2 → `arange(3.8, 5)`); [0.5]×4 gives 0.833. |
| pass^k `metrics.py:198-205` (`C(c,k)/C(n,k)`, τ-bench) | Only prime-rl, `orchestrator/utils.py:114` | p8: equal to 1e-12 (0.0556 at k=2; 0 at k=4) | Both are right (unbiased without replacement). Same D2 denominator drift: prime-rl averages each key over the groups that have it. |
| max@k `metrics.py:208-221` (unbiased E[max of k], order statistics) | verl best@k = bootstrap **with replacement**, 1000 draws, seed 42 reset per uid (`trainer/ppo/metric_utils.py:805-857, 1030-1040`); NeMo-Gym `max(vals[:k])` on non-binary tasks (`reward_profile.py:592`); OpenRLHF passN = max over all n | p1: max error vs enumeration **4.8e-10**. p7: [0.2, 0.4, 0.9], k=2: NeMo-Gym gives **0.4 or 0.9 depending on row order**; Crucible 0.7333 (exact). p8: [1,0], k=n=2: verl best@2 **0.77** (expected 0.75 = 1-(1/2)²) vs 1.0; 3×4 binary: verl best@4 0.540 vs 0.667. | **Crucible.** verl's with-replacement estimator is biased low and is not 1 at k = n when c > 0. NeMo-Gym's is order-biased, and it averages pass@k on binary tasks together with max-of-first-k on continuous tasks into one number (p7: 0.583). |
| maj@k | verl `calc_maj_val` (bootstrap, `metric_utils.py:860-893`); NeMo-RL `cons@k` (all subsets, `evals/eval.py:206-277`); NeMo-Gym `majority@k` (first k, ties averaged, `reward_profile.py:607-625`) | not ported | Crucible has none. Agentic tasks have no extractable answer to vote on: nothing. |
| mean reward `metrics.py:289` (micro over measured) + raw `:290` (masked = 0) | Harbor Mean: None → 0 in the denominator (`metrics/base.py:14-35`); verifiers dual headline: clean mean (raw in parentheses) (`utils/format.py:57-72`); prime_runs pushed `avg_reward` is raw; NeMo-Gym pools over scored rollouts (`reward_profile.py:186-213`); verl and OpenRLHF average per prompt first (macro); rLLM `score` = correct / total (micro). | p1, p3 | The dual headline matches verifiers. **Weighting is wrong under masking (D3).** verl's macro is the right shape. |
| mean reward CI `metrics.py:232-245` (percentile bootstrap over tasks, 1000 draws, seed 0) | NeMo-Gym t-interval over tasks per repeat (`reward_profile.py:226-240, 242-309`) and across repeats (`:311-353`); verl's bootstrap resamples **within** a prompt and reports a std, not a CI on the benchmark mean (`metric_utils.py:805-857`); Harbor, verifiers, rLLM, SkyRL, OpenEnv and EO report no CI. | p3 | Resampling at the task level, with each task's samples kept together, is the right level. It is the only cluster bootstrap in the sweep. The **interval method** undercovers below 20 tasks (D5), and the **statistic** is micro (D3). |
| gate profile `metrics.py:248-258` | EO `benchmark/executor.py:565-612`: `verifier_level_pass_rate` = Σpassed / Σtotal pooled over runs; error runs add 0 verifiers; infra errors in a verifier count as failed (`benchmark/verifier.py:77-152`) | read | Same micro formula. Crucible also adds the macro rate and drops masked episodes: Crucible is equal or ahead. |
| coverage `metrics.py:224-229`, all-masked → coverage only `:272-273` | NeMo-Gym `_coverage_metrics`, all-masked → coverage keys only (`reward_profile.py:839-958, 1111-1239`) | p1: all-masked group → `{'coverage': {... 'fully_masked_tasks': 1}}`, no 0 | Equal. NeMo-Gym also exports `coverage/expected/missing/imputed` (`rollout_collection.py:2919-2926`). |
| McNemar `compare.py:21-28` | none (NeMo-Gym uses an unpaired Welch test on per-repeat means, `comparison/diff.py:107-123`) | p2: vs `scipy.stats.binomtest(b, b+c, .5)` for every b+c ≤ 120, max error **5.0e-10**. p4 null type-I rate: 0.000–0.039 at every T and k (conservative). Power at T=50, k=4, Δ=.09: 0.512. | Correct. Pairing semantics: see D7. |
| paired bootstrap `compare.py:50-60` | none | p4: see D5 | Resampling level is right (tasks, with deltas nested); the interval undercovers at small T. |
| per-success cost `metrics.py:311-315` | Harbor sums cost over trials but **rmtree's retried trial dirs** (`trial/queue.py:194-237`), so retry spend is lost there too; verifiers sums tokens and cost over every completed rollout, errored ones included (`cli/dashboard/eval.py:421-501`); mini-swe-agent tracks instance and global cost (`agents/default.py:150-168`, `models/__init__.py:13-42`) | p5 | Crucible's contract is right; its loader breaks it (D4). |

## 2. Errors, timeouts and infra failures in metrics

| Framework | Infra error in quality metrics | Timeout | Retry kept | Evidence |
|---|---|---|---|---|
| **Crucible** | masked: excluded from quality metrics, counted in coverage, included in `mean_reward_raw` as 0 | `agent_timeout` scored (policy); run-level `attempt_timeout_s` **masked** (D13) | last terminal attempt; **retry spend lost** (D4) | `core/schema/run.py:65-79, 192-194` |
| Harbor | **0, in the denominator** (Mean and pass@k) | agent timeout: the verifier still runs and the real reward counts; verifier timeout → 0 | last attempt; timeouts and reward errors are never retried by default | `metrics/base.py:25,32`, `utils/pass_at_k.py:41-43`, `trial/queue.py:194-237`, `models/job/config.py:291-322` |
| verifiers | dashboard: excluded (raw in parentheses); pushed `avg_reward`: raw | a stage timeout is not an error: kept and scored | last attempt | `utils/format.py:57-72`, `rollout.py:192-215`, `utils/retries.py:120-158` |
| NeMo-Gym | `mask_sample` excluded; collection failures go to a sidecar and are **absent by default** (opt-in `count_failure_classes_as_zero`); judge failures masked | `failure_kind` is diagnostic only | last per key | `reward_profile.py:839-958`, `rollout_collection.py:170-192, 990-1018, 1693-1705`, `judge.py:85-130` |
| prime-rl | the "effective" subset excludes errored traces; dispatch failures are not episodes | truncated traces kept | n/a | `orchestrator/metrics.py:541-546`, `orchestrator.py:897-929` |
| OpenEnv | failed episodes excluded (`num_failed` only) | n/a | n/a | `core/harness/collect.py:266-377` |
| rLLM | a missing episode → 0; errored episodes kept | n/a | n/a | `eval/runner.py:144-186` |
| SWE-smith | timeout → unresolved; harness error → in neither list but in `total` (counts against resolved rate) | timeout = unresolved | `--redo_existing` | `harness/eval.py:52-94, 232-260` |
| mini-swe-agent | exception → exit status + empty patch | `TimeExceeded` → empty submission | n/a | `run/benchmarks/swebench.py:145-177`, `agents/default.py:132-147` |
| EnterpriseOps | error runs count against success rate, not against `verifier_level_pass_rate`; verifier infra errors = failed | n/a | whole sample re-run ≤ 5 | `benchmark/executor.py:514-612`, `evaluate.py:220-241` |
| agentenv | judge not run → score 0; no cross-run mean; `eval run` reads `metadata["score"]`, which nothing writes | n/a | step-level rollback | `task_step/.../rubrics_verifier.py:381-393`, `cli/eval/run.py:155-181` |
| NeMo-RL / slime / SkyRL / AReaL / ROLL / OpenRLHF | errors → 0, or kept with the env reward; slime `reward=None` raises TypeError; AReaL failed → not solved | truncated kept (NeMo-RL math zeroes non-ended) | n/a | per §1 rows |

**Verdict.** Crucible's "masked samples are never samples" is the strictest rule here and the right one. It agrees with
NeMo-Gym's default, verifiers' dashboard, prime-rl's effective subset and OpenEnv. Harbor, rLLM, SWE-smith,
EnterpriseOps, agentenv and most trainers score infra errors as 0. Crucible counts nothing it shouldn't: an exposed
answer is a mask kind (`core/schema/run.py:43`), so `Episode.exposed` episodes are always masked. The
`0.0 if e.exposed` branch in `max_at_k` (`metrics.py:214`) is dead code. Crucible does drop three things it should
report:
- retry spend (D4);
- the kind of each mask (D9);
- the task set each k used (D2).

## 3. Confidence intervals: simulation detail

Setup [ran p3]: per-task p ~ Beta(1.2, 1.8), true task-population mean μ = 0.40, k Bernoulli samples per task.
Crucible's `bootstrap_ci` is called exactly as `summarize_group` calls it (seed 0, 1000 draws). Masked scenario: a task
with p < .3 loses each sample with probability .6 (difficulty-correlated infra failure). 2000 simulated benchmarks per
cell (500 at T=200).

| T | k | mask | Crucible cov | width | micro bias | t on task means cov | scipy BCa (task means) cov |
|---|---|---|---|---|---|---|---|
| 5 | 1 | no | 0.833 | 0.659 | −0.005 | 0.912 | 0.877 |
| 5 | 4 | no | 0.835 | 0.478 | +0.001 | 0.936 | 0.850 |
| 10 | 1 | no | 0.908 | 0.564 | +0.003 | 0.953 | 0.947 |
| 10 | 4 | no | 0.905 | 0.371 | +0.000 | 0.942 | 0.923 |
| 20 | 4 | no | 0.927 | 0.272 | +0.002 | 0.944 | 0.937 |
| 50 | 4 | no | 0.947 | 0.177 | −0.000 | 0.952 | 0.943 |
| 200 | 4 | no | 0.956 | 0.090 | +0.001 | 0.962 | 0.963 |
| 10 | 4 | yes | 0.852 | 0.400 | +0.073 | 0.946 | 0.943 |
| 20 | 4 | yes | 0.820 | 0.295 | +0.077 | 0.948 | 0.937 |
| 200 | 4 | yes | **0.126** | 0.095 | +0.076 | **0.916** | 0.913 |
| 50 / 200 | 1 | yes | 0.816 / 0.560 | | +0.077 / +0.073 | 0.837 / 0.578 | 0.787 / 0.587 |

Reading:
- With k = 1, a masked task is lost whole, so no estimator recovers it (missing not at random). This is what
  `fully_masked_tasks` must warn about.
- With k ≥ 2, the macro statistic is robust, and Crucible's micro statistic is not.

Paired bootstrap and McNemar [ran p4]:
- Setup: base p_t ~ Beta, candidate logit + 0.5 (true Δ = 0.0919), or the null (Δ = 0); 1000 simulations per cell.
- Paired-bootstrap coverage under Δ = 0.092 (k = 1 / 4): T = 5 0.689 / 0.854; T = 10 0.855 / 0.896; T = 20 0.910 / 0.918;
  T = 50 0.938 / 0.944.
- Null: T = 5 0.947 / 0.876; T = 10 0.949 / 0.916; T = 50 0.961 / 0.945.
- McNemar rejection at .05 under the null: ≤ 0.039 everywhere.

## 4. What frameworks report that Crucible does not, and what to export

| Gap | Who has it (file:line) | Matters because | Pick |
|---|---|---|---|
| Error breakdown by kind, retries count | Harbor `exception_stats`, `n_errors`, `n_retries` (`models/job/result.py:129-213`); NeMo-Gym `_failures.jsonl` + `progress/*/masked_pct` (`rollout_collection.py:131-192`); verifiers `err` + per-row type | "47 masked" is not actionable; "40 stale_image, 7 attempt timeout" is | D9 |
| Stop reason / truncation rate | slime `truncated_ratio`; OpenRLHF `truncated_rate`; tau2 termination counts | separates "ran out of steps" from "wrong" | D9 (`Episode.stop` already loaded) |
| Spread across repeats (std, stderr) | NeMo-Gym `std_dev/std_err_across_runs`, `avg_sample_std_dev` (`reward_profile.py:627-689`); verl `std@N`; Harbor adapter SEM rule | seed variance is the first thing a reviewer asks for | D10 |
| Per-category / subset metrics | NeMo-Gym `compute_subset_metrics`; verl per `data_source`; OpenRLHF per datasource | OOD axes (DESIGN §11.2) are categories | D11 |
| Token split, latency percentiles, phase timing, turns | Harbor; NeMo-Gym `perf_summary`; verifiers dashboard; EO `mean_execution_time_ms`, `tool_usage` | cost model, provider comparisons | D12 |
| pass@k at every eligible k, automatically | Harbor `_eligible_k_values` | D6 | D6 |
| Per-task tables in HTML | Harbor viewer per-task summary (`viewer/server.py:1800-1925`); rLLM viewer | the HTML shows only the overall row per model (`bench/report.py:72-85`); per-task data exists only in `summary.json` | P2 |

**Export so Harbor, verifiers and NeMo-Gym users can read a Crucible bench** (PATTERN, M): write next to `summary.json`:
- `harbor_result.json` in JobStats shape. Evals keyed `agent__model__dataset`, with `n_trials`, `n_errors`, `metrics: [{"mean": …}]`, `pass_at_k: {k: …}`, `reward_stats`, `exception_stats` (masked kinds), `n_input/n_output_tokens`, `cost_usd`. Note in a `crucible` block that masked trials are excluded rather than zeroed.
- `<stem>_aggregate_metrics.json` in NeMo-Gym shape: `agent_metrics` / `key_metrics` with `pass@{k}/accuracy` (×100), `mean/reward`, `coverage/measured_rollouts|masked_rollouts|measured_tasks|fully_masked_tasks`, `group_level_metrics` per task, and `repeat_level_metrics` per sample index. `gym eval compare` then works on Crucible runs.
- The verifiers dual headline string (`format_mean`) for the CLI line.

## 5. Corrections to `docs/research/bench-studio/01-framework-study.md`

- **":37 The one framework that gets it right is NeMo-Gym."** Wrong as stated. Errors are also excluded by:
  - verifiers' dashboard (`VF/utils/format.py:57-72`);
  - prime-rl's effective subset (`PR/orchestrator/metrics.py:541-546`);
  - OpenEnv collect (`OE/core/harness/collect.py:266-377`).

  NeMo-Gym's collection failures are absent by default only because they are written to a sidecar. Its per-k task drop
  (`reward_profile.py:581`) is the D2 defect, which Crucible copied.
- **Row 2.** It cites prime-rl and rLLM as the pass@k pattern. rLLM handles n < k differently (falls back to empirical
  any-pass) from prime-rl (drops the group). Neither keeps the task set fixed; Harbor does. The study does not mention
  the difference.
- **Row 3** (t-CIs, a warning on unequal coverage, bootstraps for AUC and cost per success) is not implemented (D10).
- **ERR=0 citations.** These hold:
  - Harbor `utils/pass_at_k.py:41`, `metrics/base.py:25,32`;
  - verl `metric_utils.py:969-971` (v1 trainer pads missing `acc` with 0);
  - SWE-smith, through `total = len(predictions)`;
  - EO `executor.py:514`;
  - NeMo-RL fractional `num_correct` (p8: [0.3]×4 → 0.724).

  The rLLM zero is in `eval/runner.py:144-186` (missing episode → reward 0); `results.py:74-76` counts errors and keeps them in `total`.

## 6. Picks

| Pick | Source (file:line) | Verdict | Licence | Effort | Fixes |
|---|---|---|---|---|---|
| One task set for every k. Report `pass@k_tasks` and refuse (or flag) a k whose set differs from pass@1's; for comparisons, use k ≤ min measured n (Harbor) | `H/utils/pass_at_k.py:58-59`; `ROLL/utils/local_code/pass_k_utils.py` `(total >= k).all()` | PATTERN | Apache-2.0 | S | D2 |
| Join compare on (task, sample_idx) **within** a declared (policy, harness). Refuse a side with more than one, and add `--base-harness` / `--cand-harness` | own | nothing to adopt | — | S | D1 |
| Task-level flips (per-task pass rate, regressions first), with McNemar kept on task-level discordance or on seed-shared samples only | `NG/comparison/diff.py:205-359` | PATTERN | Apache-2.0 | S | D7 |
| Macro headline: mean of per-task means, bootstrap of the same statistic; keep the micro value as `mean_reward_episodes` | verl macro averaging `VE/trainer/ppo/metric_utils.py:1073` | PATTERN | Apache-2.0 | S | D3 |
| t-interval on task means (or BCa), with `n_tasks` beside every CI and a warning below 10 tasks. NeMo-Gym's `_confidence_interval` uses scipy; Crucible's venv has no scipy, so the t quantile needs a small own implementation | `NG/reward_profile.py:226-240` | PORT (logic) | Apache-2.0 | S | D5 |
| Spend from every attempt: read `failures.jsonl` (or `samples/*/a*/record.json`) for superseded attempts, add `cost_uncertain_usd` as its own field, reconcile with `ledger.summary()` | own; NeMo-Gym `_failure_rows_counted_as_zero` keys on last attempt (`NG/rollout_collection.py:1667-1690`) as a pattern for reading the sidecar | own | — | S | D4 |
| Default `ks` from k: `ks = sorted({1, 2, 4, …, k})` in `run_bench` / `report_run`; `--ks` on `bench report` | Harbor `_eligible_k_values` (`H/utils/pass_at_k.py:71-84`) | PATTERN | Apache-2.0 | S | D6 |
| `failure_kind`, `attempts` and `stop` on `Episode`; masked-by-kind and stop-reason tables | Harbor `exception_stats` (`H/models/job/result.py:129-169`) | PATTERN | Apache-2.0 | S | D9 |
| Across-sample-index std and stderr; a per-task sample std | `NG/reward_profile.py:627-689` | PATTERN | Apache-2.0 | S | D10 |
| Subset (axis) metrics | `NG/reward_profile.py:692-725` | PATTERN | Apache-2.0 | S | D11 |
| Perf summary: token split, latency p50/p90/p99, turns, observability coverage | `NG/reward_profile.py:1003-1063` | PATTERN | Apache-2.0 | S | D12 |
| Harbor / NeMo-Gym / verifiers export files (§4) | shapes in `H/models/job/result.py`, `NG/rollout_collection.py:2954-3095`, `VF/utils/format.py:57-72` | PATTERN | Apache-2.0 / MIT | M | §4 |
| McNemar in log space (`math.lgamma`), or a running binomial pmf | own | — | — | S | D14 |
| Regression tests: every D above, each failing on `b6d89fc` | own | — | — | M | D8 |
| **Do not adopt**: verl with-replacement best@k; NeMo-Gym `max(vals[:k])`; NeMo-RL fractional `c`; SkyRL / OpenRLHF any-of-n as pass@k; any error-as-0 convention; NeMo-Gym's unpaired Welch compare | §1 rows | anti-pattern | — | — | — |

Crucible is already correct (keep):
- the per-task estimators for pass@k, pass^k and max@k;
- exact McNemar;
- the task-level cluster bootstrap (resampling level);
- the mask semantics and the all-masked coverage-only rule;
- the dual headline;
- the gate profile, micro and macro.
