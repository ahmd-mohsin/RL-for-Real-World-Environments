# M4 study: signal reliability, outcome labels and the exit gate's statistics

Scope: `signals/reliability.py`, `signals/gate.py`, `signals/taxonomy.py` (priors), coding pack `outcomes.py`, `signals.py` (what the mapper hands reliability), `m4/{gate.py,build_traces.py,labels_dev.json,test_prs.json}`, report `docs/modules/reports/m4_gate_coding_dev.json`, toy pack `signals.py` and `m4/labels_test.json`, and every reader of `Signal.reliability`. Spec: ARCHITECTURE "### M4 `signals`" (exit gate: 50 hand-labeled events per pack; type accuracy ≥ 0.9; attribution precision ≥ 0.9; outcome agreement ≥ 0.95), DESIGN §5.2, DESIGN "circular M4 reliability" (improvement queue), `docs/research/improvements/M03-M05.md` M4 rows. Read-only study, 2026-10-12.

Paths: `REF` = `RL-Environments/06-frameworks-and-env-code/`. `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Every candidate marked **[ran]** was reproduced with `crucible/.venv` (Python 3.12.9) on HEAD `90f2539`. **[read]** means established by reading code only. Probes live in the session scratchpad (`m4study_rel/p1..p7*.py`, builders in `mk.py`). Probes on real traces read *copies* of `crucible/data/m4` and `crucible/data/m3/store` taken into the scratchpad. The M4 gate module was not run. The `data/m4` copy was taken while a gate run may have been rebuilding; 26 of its 28 heads have no `source.fetched_at`, so they are post-M3-audit builds.

## 1. Mechanism table

| Mechanism | Crucible | References |
|---|---|---|
| Reliability of a signal source | Beta(prior + confirmed, prior + contradicted) per **type** only `C/signals/reliability.py:29-43,69-81`; priors per type `C/signals/taxonomy.py:21-38` | No reference estimates reliability of human expert signals. Closest: TRL Robust DPO takes a label-flip probability ε as a loss parameter `REF/infra-huggingface-trl/trl/trainer/dpo_config.py:102-106`, `dpo_trainer.py:803-806,1407-1409`; reward margins per pair `reward_trainer.py:776-778` |
| What counts as evidence | outcome label of the same trace + `meta.acted_on` set by the same mapper rule that chose the type `reliability.py:46-66`; `CP/signals.py:87-93` | SWE-smith reruns gold and drops tests that fail on it, drops instances left with no F2P `REF/swe-bench-swe-smith/swesmith/harness/repair.py:1-24,138-157` |
| Flaky / unscorable evidence | flaky CI → polarity 0 → unscored, but still gets the type mean `CP/signals.py:112-116`, `reliability.py:49,84-86` | SWE-smith removes flaky tests (above). NeMo-Gym routes invalid judge results to a failure sidecar, never scored `REF/nvidia-nemo-gym/nemo_gym/rollout_reverification.py:485-500`, `failure_kinds.py:78-81` |
| Keeping sources apart | one row per type; human, bot and AI-reviewer signals pooled; all families pooled `reliability.py:73` | research-environments keeps per-stage sub-scores (verifier, rubric, taste, validation) apart `REF/primeintellect-ai-research-environments/environments/swe/senior_swe_bench/senior_swe_bench/taskset.py:140-149` |
| Missing calibration | `apply` silently skips types absent from the table `reliability.py:85` | Karotte refuses to register any task lacking a calibration row `REF/preferencemodel-karotte/src/karotte/templates/default/src/environment/suite.py:39-55` |
| Outcome window | merged + 30 d without revert/hotfix → good; needs `source.fetched_at` `CP/outcomes.py:28-31`; connector no longer writes it `CP/connectors/github_pr.py:255-256` | OpenEnv scores a terminal outcome and spreads it back with discounting; a draw is 0.5 `REF/meta-pytorch-openenv/src/openenv/core/rubrics/trajectory.py:138-165`. No reference has a delayed-outcome window |
| Grader self-check | none for the outcome labeler | senior_swe_bench `validate`: apply the gold solution, valid iff the verifier scores 1.0 `taskset.py:151-160` |
| Verdict-set validation | gate takes the first signal at an event, ignores the rest `C/signals/gate.py:56` | verifiers rejects a criterion answered twice and any verdict set that is not exactly the criteria `REF/primeintellect-ai-verifiers/verifiers/v1/judges/rubric.py:156-183` |
| Missing prediction in a metric | dropped from attribution precision/recall `gate.py:64` | Harbor counts a missing reward as 0 `REF/harbor-framework-harbor/src/harbor/metrics/base.py:14-35`; rejects empty/non-finite reward files `verifier/verifier.py:89-120` |
| Errors vs score | errors listed, not counted against coverage | EnterpriseOps reports success rate, verifier-level pass rate and files-with-errors separately `REF/servicenow-enterpriseops-gym/compute_score.py:12-13,34-49` |
| Inter-annotator agreement | none in M4 (one labeler, the builder). Cohen's κ exists in the toy M5 pipeline `TP/m56/pipeline.py:190-195` | none found in any REF (grep for kappa/krippendorff/inter-annotator) |
| Confidence intervals on gate metrics | none `gate.py:77-81` | none found except Harbor pass@k by evals `REF/harbor-framework-harbor/src/harbor/utils/pass_at_k.py:8` |
| Held-out split | `labels_dev.json` only; `CP/m4/test_prs.json` lists 14 PR numbers that no code reads; no `labels_test.json` for coding | research-environments tasksets carry held-out splits (e.g. `environments/code/forth_lang/forth_lang/taskset.py:50-51,164-170`) |
| Weight / config validation | Beta priors unchecked | OpenEnv `WeightedSum` refuses mismatched or non-normalized weights `REF/meta-pytorch-openenv/src/openenv/core/rubrics/containers.py:251-257`; verifiers refuses non-finite or non-positive total weight `rubric.py:105-109` |

## 2. Defect candidates

Severity: H = corrupts labels, reliability or the gate verdict in normal use; M = in a common edge case; L = rare or cosmetic.

### Outcome labels (`CP/outcomes.py`)

**R1 [H][ran] Every merged PR is now labelled `unknown`.** The M3 audit dropped `fetched_at` from PR traces (`github_pr.py:255`). `outcomes.py:28-29` then sets `now = None` and returns `unknown`.
- Input (p4): commit 2026-01-01, `merged` 2026-01-02, no `fetched_at`. Got `unknown | merged less than 30 days ago`, for a PR merged 280 days before `NOW`. The same trace with `fetched_at="2026-10-09T12:00:00+00:00"` gives `good`.
- Input (p1): the 14 labelled PRs from the `data/m4` copy. All 11 hand-`good` PRs come out `unknown`. Outcome agreement is 3/14 = **0.214** (threshold 0.95). Over all 28 stored traces: 25 unknown, 2 bad, 1 good.
- The `label_source` text is also false: the window state is unknown, not "less than 30 days".
- Fix: have the connector record a deterministic window fact that changes once, e.g. `outcome.details["window_closed"] = now >= merged + 30d` plus `window_until`. Alternatively, pass `now` to the labeler through the registry or constructor (`build_traces.NOW` exists). When the fact is missing, use `label_source="observation time unknown"`.

**R2 [M][ran] Naive vs aware time raises.** `fetched_at="2026-10-09T12:00:00"` with `merged.time="2026-01-02T00:00:00Z"` raises `TypeError: can't subtract offset-naive and offset-aware datetimes` (p4). Fix: use `connectors.base.parse_time` (already used by the connector) to get aware UTC.

**R3 [M][ran] `merged.time` None raises.** Input: a `merged` event with `time=None`. Got `TypeError: fromisoformat: argument must be str` (p4). Expected `unknown` with a reason.

**R4 [L][ran] The labeler trusts that the connector bounded the window.** A `post_merge_commit` with `is_revert=True, days_after_merge=199.0` gives `bad | reverted within 30 days of the merge` (p4). `outcomes.py:18-19` never reads `days_after_merge`. Any other connector feeding `coding/pull_request` gets wrong labels. Fix: filter `days_after_merge <= WINDOW`.

**R5 [M][read] Missing post-merge evidence reads as "no revert".**
- `_after_merge` returns `[]` when the merge commit is not in the local clone. It also returns `[]` when the PR wrote no lines (deletion-only), before it even looks for reverts (`github_pr.py:307-308,318-319`).
- `git log --until` reads the local clone, so the window depends on when the clone was fetched, not on `NOW` (`github_pr.py:322`).
- In all three cases a reverted PR is labelled `good`.
- Fix: the connector records `post_merge_checked` and the clone's head time. The labeler returns `unknown` unless the window was fully observed.

**R6 [L][ran] Closed-unmerged is always `bad`.** A superseded or duplicate PR gets `bad | closed without merging` (p4). The hand-label guideline states the same rule, so the gate cannot catch it (see R24). Fix: label it `unknown`, or `bad` only when a rejection review preceded the close.

### Reliability (`C/signals/reliability.py`)

**R7 [H][ran] Reliability is decided by the mapper's own rule and the trace outcome, not by independent evidence.** Truth table from p2 (verdict per outcome: good / bad / unknown):

| signal as the coding mapper emits it | good | bad | unknown |
|---|---|---|---|
| rejection (comment; mapper sets it iff `acted_on=False`, `CP/signals.py:93`) | False | True | None |
| correction (`acted_on=True` by construction) | True | True | True |
| suggestion_accepted | True | True | True |
| revert (`PullRequestOutcome` makes any revert `bad`) | False* | True | True |
| check_result pass, approval | True | False | None |

\* unreachable: a reverted PR is never `good`.

- Comment rejections equal the outcome label. Correction and suggestion_accepted can never be contradicted. The docstring's "unless the trace outcome contradicts them" (`reliability.py:10-11`) is only implemented for `revert` + `good` (`:52-53`).
- Same signals, different outcome labeler: rejection reliability is **0.043** in the committed report (0/21, outcomes `good`). It is **0.75** on the `data/m4` copy (2/0, outcomes mostly `unknown`, R1) (p1). The number tracks the outcome labeler, not the reviewers.
- Fix: the improvements queue item M4 #1. Score each type only on evidence disjoint from its mapping rule: a rejection is confirmed by a later revert, hotfix or failing check on the graded lines; corrections face the 30-day revert window. Re-estimate on traces the labels never saw.

**R8 [M][ran] One-sided verdicts bias the estimate.**
- Review rejections (`acted_on=None`, `CP/signals.py:107`) and failing `check_result` (no `acted_on`) can be confirmed (`bad`) but never contradicted.
- Positive signals on an `unknown` trace can only be contradicted (`:61-66`).
- So negatives drift up, positives drift down. On the copy, approval is 1 confirmed / 2 contradicted / 33 unscored (p1).
- Fix: score a signal only when both verdicts are reachable from the evidence available for it. The connector should compute `acted_on` for review-level rejections (a later commit touching the reviewed files).

**R9 [H][ran] A positive signal is contradicted by any later negative on the same commit, whatever it is about.** `later_neg` (`:61`) matches on graded seq only.
- Input (p3): commit 0; `lint` passes at seq 1; `build-windows` fails at seq 2; PR good. The lint verdict is `False`.
- On the copy (p7), all 60 contradicted `check_result`s are passing checks:
  - 22 were "contradicted" by a *different* CI job failing on the same commit. Example: `pcl#5958` seq 21 `Build (Formatting Check code formatting)` success, then seq 22 `Build (Build GCC Ubuntu 20.04 GCC)` failure.
  - 38 were "contradicted" by a later review rejection grading that commit.
- Result: `check_result` reliability is 0.016 on the copy, 0.766 in the report.
- Fix: for `check_result`, match the same `meta.check`. For review signals, match `target.object`/span overlap. Confirm a pass from a rerun of the same check (SWE-smith).

**R10 [M][ran] Not per family, not per actor class.**
- The docstring says "per type (per family)". `estimate` keys on `s.type` only (`:73`).
- `github_pr.py:197` says AI-reviewer signals "are kept apart from human experts' in reliability". They are not.
- p3: a toy CSAT `check_result` gets reliability 0.3333, computed only from coding CI rows. 20 bot rejections and 5 human ones share one row (0.222).
- On the copy (p7), 50 of the 102 confirmed corrections come from automated actors (`meta.automated=True`).
- Fix: key on `(family, type, actor class)` with a hierarchical Beta that shrinks to the type prior. `apply` refuses a table built for another family (Karotte's refuse-when-missing).

**R11 [M][ran] `apply` writes the type mean onto signals `verdict` cannot score.** Flaky checks (polarity 0), trace-level signals and ungraded signals all receive it. p3: a flaky check gets reliability 0.3333 (the CI mean) although its verdict is `None`. Fix: leave `reliability=None` (or record an `unscored` reason) for signals outside the estimator's domain. Drop flaky checks as SWE-smith does.

**R12 [M][ran] Nothing reads `Signal.reliability`.** Grep over `C` and the packs finds only the schema field (`C/core/schema/signal.py:48`) and the M4 report writer (`CP/m4/gate.py:31`). None of these read it: `credit/`, `studio/datasets.py` (correction pivots → DPO pairs), `workflow/milestones/mine.py` `from_signals`, `CP/forge/derive.py`, `TP/forge/hindsight.py`. DESIGN says M6 weighs evidence by it. Today a 0.043 type and a 0.99 type have equal weight downstream. Fix: wire it (for example ε = 1 − reliability as TRL Robust DPO `label_smoothing` on pivot pairs, milestone weight), or stop claiming it.

**R13 [M][read] Estimated in-sample on 14 traces, with no interval.** `CP/m4/gate.py:26-29` estimates reliability on the labelled traces only, then applies it to the same traces. The report gives a posterior mean with no interval (`m4_gate_coding_dev.json:38-63`). Fix: estimate on the full corpus minus the gate's labelled traces, and report a Beta 90% interval and n per row.

**R14 [L][ran] Falsy seq 0 in `later_neg`.** `(o.source_event or -1)` maps seq 0 to -1 (`:61`). p3: an approval with no source event plus a rejection at seq 0 on the same step gives approval `True`; the order is unknowable but treated as known. Fix: `-1 if x is None else x`, and skip ordering when either side is None.

### Exit gate (`C/signals/gate.py`, `CP/m4/`)

**R15 [H][ran] Labels are keyed by `(trace_id, seq)`, and seqs moved.**
- On the post-M3 traces of the `data/m4` copy (p1, p1b), the dev gate gives type **0.78**, attribution precision **0.44**, outcome **0.214** → `passed=False`.
- Example: the label at `pcl#6199` seq 36 has the note "suggested '/** \brief Constructor */' is in the final file". The event now at seq 36 is a different comment ("`fricp.h` uses `std::sqrt`...").
- The committed report (`passed: true`) describes traces that no longer exist.
- Fix: key labels on the source's stable id (comment id, review id, check-run id, sha) and store a trace content hash. Resolve seqs at load time; refuse on hash or text mismatch.

**R16 [M][ran] A label for a missing trace raises.** `evaluate({...}, {"events":[{"trace_id":"nope",...}]})` → `KeyError: 'nope'` (p5). Fix: report it as a label error (fail closed with a message).

**R17 [M][ran] Several signals at one event: the first wins, the rest are invisible.** Signals `[approval, rejection]` at seq 2:
- labelled `rejection` → type accuracy 0.0;
- labelled `approval` → 1.0 and `passed=True`; the spurious rejection is never seen (p5).

The agent-run mapper emits several `check_result`s at one seq by design (`CP/signals.py:144-159`). Fix: compare the multiset of types per event. Raise on an unlabelled extra, as verifiers does for a criterion answered twice.

**R18 [H][ran] Attribution precision ignores predictions the label says grade nothing.** `gate.py:64` scores only events where both sides grade.
- Event 1: pred [0] = label [0]. Event 2: pred [3], label null. Precision 1.000, `passed=True` (p5).
- Prediction `[]` vs label `[3]` is also skipped.

Fix: count `pred grades, label null` as a precision miss and `label grades, pred empty` as a recall miss. A missing prediction is a 0, as in Harbor.

**R19 [M][ran] Signals on unlabelled events are never scored.** Two spurious rejections on unlabelled events still give `passed=True` (p5). The labelled 50 were chosen by the builder. Fix: label whole traces, or sample events uniformly from all events *and* from emitted signals, and report signal precision over emitted signals.

**R20 [M][ran] Class imbalance hides weak types.**
- `none` is 15/50 coding labels and 61/84 toy labels. A mapper that emits nothing scores 61/84 = **0.726** toy type accuracy (p6).
- The coding labels hold 1 approval and 0 revert, rework or escalation. Revert reliability is never validated.
- Fix: per-class recall or macro-F1, and a minimum number of labels for every type the mapper emits.

**R21 [M][ran] The 50-label minimum is not enforced.** One correct event plus one correct outcome gives `passed=True` (p5). Empty labels give 0.0 and fail, which is fine. Fix: refuse `n_events < 50`, and set a floor on `n_outcomes`.

**R22 [H][ran] No uncertainty on the thresholds** (Wilson, p5).
- 50/50 type → 95% CI [0.929, 1.0].
- 45/50 (0.90, passes) → [0.786, 0.957].
- Attribution 31/31 → [0.890, 1.0], so even a perfect score does not show precision ≥ 0.9.
- Outcomes 14/14 → [0.785, 1.0]; one-sided 95% lower bound 0.838.
- **51** consecutive agreeing outcomes are needed before the lower bound reaches 0.95, and 25 perfect events for 0.90.
- Fix: gate on the one-sided Wilson lower bound, or size the label set from it (≥ 51 outcomes for the 0.95 bar), and report intervals.

**R23 [M][read] The gate is dev-only, and the labels were tuned on it.**
- `labels_dev.json` notes "4 labels revised after re-examining the diffs".
- `CP/m4/test_prs.json` lists 14 PR numbers but nothing reads it, and there is no coding `labels_test.json`.
- `gate.run` defaults to `split="dev"`.
- Fix: label the test PRs before the next mapper change, store a hash of the labels file in the report, and gate on test.

**R24 [M][read] One labeler, who also wrote the mapper, using the mapper's rules.**
- `labeled_by` is "Claude (the builder)".
- The guideline restates `CP/signals.py:3-11` almost verbatim. The outcome rule *is* `PullRequestOutcome`'s definition.
- So agreement measures implementation fidelity, not whether a "correction" was a real defect or a "good" PR was good. There is no second annotator and no κ.
- Fix: a second blinded labeler (a human, or a different model given only the guideline's intent). Report Cohen's κ, reusing `TP/m56/pipeline.py:190-195`, and adjudicate disagreements.

**R25 [L][ran] Outcome agreement does not separate abstention.** A labeler that always says `unknown` matches every hand `unknown` (p5); with R1 that is the whole 0.214. Fix: report coverage (decided / labelled) and agreement on decided labels separately, as EnterpriseOps separates errors from scores.

### Toy pack M4

**R26 [H][ran] The toy labels cannot be scored.**
- `TP/m4/labels_test.json` uses ids `toy/helpdesk_ticket:T10xx` (the M5 family).
- The mapper is registered for `toy/support_ticket` (`TP/__init__.py:20`), and the M3 store holds `toy/support_ticket:` ids.
- `mapper_for("toy/helpdesk_ticket")` returns `None`, so `extract` would raise `LookupError`. No toy M4 runner exists.
- Remapped by case id (p6): type 0.869, precision 0.909 (n=11), outcome **0.083**, `passed=False`.

**R27 [H][ran] There is no toy outcome labeler.** `labeler_for("toy/support_ticket")` returns `None`, and all 24 toy traces are `unknown` (p6). So toy reliability never sees an outcome: rework, escalation, violation, sla_breach and approval are all unscored.

**R28 [M][ran] CSAT is lost.**
- The 24 toy traces hold 12 `csat` events but yield 0 CSAT signals (p6, p6b).
- The stored event data keeps only `row_id`. The log row has `"score": 3`. Texts such as "Quick and clear, thank you." contain no digit for the fallback regex (`TP/signals.py:54-57`).
- Even when mapped, CSAT is a `check_result` with no grades. `verdict` returns `None` and it is never scored, while the labels call it `outcome`.
- ARCHITECTURE lists CSAT as a toy signal.

**R29 [L][ran] No rule for customer replies.** The toy guideline maps customer replies to rejection/approval, but the mapper has no rule for them: 3 rejection→none and 3 approval→none (p6 confusion).

## 3. Ranked features worth porting

| # | Feature | Reference | Mode | Owning Crucible file |
|---|---|---|---|---|
| 1 | Fail-closed verdict sets: exactly the labelled events, no duplicates, no unknown traces | verifiers `verifiers/v1/judges/rubric.py:156-183` | PATTERN | `C/signals/gate.py` `evaluate` (R16, R17) |
| 2 | Missing prediction = 0 in aggregation | Harbor `src/harbor/metrics/base.py:14-35`; `verifier/verifier.py:89-120` | PATTERN | `gate.py:64` attribution (R18) |
| 3 | Drop flaky evidence instead of averaging it in; drop items left with no valid evidence | SWE-smith `swesmith/harness/repair.py:1-24,138-157` | PATTERN | `reliability.py` `apply` / `estimate`; `CP/signals.py:112-116` (R11) |
| 4 | Label-flip probability as a training parameter: consumer for reliability (ε = 1 − r on correction pivots) | TRL `trl/trainer/dpo_config.py:102-106`, `dpo_trainer.py:803-806,1407-1409`; margins `reward_trainer.py:776-778` | DEPEND if M12 uses TRL losses, else PATTERN | `studio/datasets.py` pivots; `credit/` (R12) |
| 5 | Unscorable items go to a failure sidecar with a reason, never into the score | NeMo-Gym `nemo_gym/rollout_reverification.py:485-500`, `failure_kinds.py:78-81` | PATTERN | `reliability.verdict` → reason for `None`; gate coverage (R8, R25) |
| 6 | Refuse to run without a calibration row | Karotte `templates/default/src/environment/suite.py:39-55` | PATTERN | `reliability.apply` per (family, type, actor class) (R10) |
| 7 | Grader self-check on a known answer | research-environments `senior_swe_bench/taskset.py:151-160` | PATTERN | `CP/outcomes.py`: fixture PRs with a known revert, a known hotfix and a known 40-day clean merge (R1-R5) |
| 8 | Per-stage sub-scores kept apart | research-environments `senior_swe_bench/taskset.py:140-149` | PATTERN | reliability rows per source (human reviewer, AI reviewer, CI job) (R10) |
| 9 | Score, coverage and error counts reported separately | EnterpriseOps `compute_score.py:12-13,34-49` | PATTERN | `GateReport` (R25) |
| 10 | A neutral value for an undecided outcome | OpenEnv `src/openenv/core/rubrics/trajectory.py:138-165` (draw = 0.5) | PATTERN | `verdict` on `unknown`; M6 consumers |
| 11 | Validated weights/priors | OpenEnv `rubrics/containers.py:251-257`; verifiers `rubric.py:105-109` | PATTERN | `taxonomy.SignalKind.prior` |

None of the references has inter-annotator agreement, confidence intervals on gate metrics, or a delayed-outcome window. These (R22, R24, R1) are Crucible's to build. κ can reuse `TP/m56/pipeline.py:190-195`.

## 4. Where Crucible is stronger

- **A reliability layer exists.** No reference weighs human expert signals at all. Beta posteriors with per-type priors (`taxonomy.py:24-38`) are the right shape. What is wrong is the evidence fed into them (R7-R10), not the estimator.
- **Delayed outcomes with an explicit `unknown`.** References score at episode end. A 30-day revert/hotfix window that refuses to guess while open (`outcomes.py:1-3`) is unique. It only needs its clock back (R1).
- **Step attribution by blame inside the PR**, with a hand-checked attribution metric and recall reported next to precision. No reference attributes outcomes to steps of a recorded human workflow.
- **Inline flaky detection.** Both conclusions for one check on one sha give polarity 0 (`CP/signals.py:112-114`) without a rerun. SWE-smith needs a gold rerun for the same fact.
- **Honest label provenance.** The labels file records `labeled_by`, `method`, the full guideline, `now` and the revisions made. The report keeps a confusion matrix and every error. The fixes above (stable keys, hash, test split, second labeler) build on that.
- **Pivots.** (wrong, corrected) pairs linked to the signal that resolved them (`C/signals/pivots.py`) have no counterpart beyond NeMo-Gym's model-rollout pivots.
