# M4 study: how raw events become typed signals (coding and toy mappers)

Scope: `CP/signals.py` (`CodingSignalMapper`, `AgentRunSignalMapper`, `intent`), `TP/signals.py` (`ToyTicketSignalMapper`), the facts they consume from `CP/connectors/github_pr.py` and `CP/connectors/ci_logs.py`, `C/signals/taxonomy.py`, `C/core/schema/signal.py`, `C/signals/extract.py`. Spec: ARCHITECTURE §M4 (`ARCHITECTURE.md:519-536`), DESIGN §5.2 (`DESIGN.md:389-401`). Gate labels: `CP/m4/labels_dev.json`, `TP/m4/labels_test.json`. Read-only study, 2026-10-12, HEAD `90f2539`.

Paths: `REF` = `RL-Environments/06-frameworks-and-env-code/`. `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Every candidate marked **[ran]** was reproduced with `crucible/.venv` on HEAD `90f2539`. Probes live in the session scratchpad (`m4study_mapping/p0..p6_*.py`); each input is given below so it can be retyped. "Corpus" means the 28 PCL PR traces of the M4 gate store (`crucible/data/m4`), read from a copy.

Not repeated here (already in `docs/research/improvements/M03-M05.md`): the circular reliability rule (`reliability.py:49-60`, rejection 0/21) and reliability keyed by type only. Several defects below feed that problem; the fixes are separate.

## 0. Corpus facts (probe p0)

| Fact | Count | Bears on |
|---|---|---|
| review comments / roots / roots by a reviewer | 303 / 215 / 212 | M1 |
| non-CI signals emitted by the AI reviewer (`Copilot`, kind `agent`) | 107 of 214 (54 of 110 corrections, 37 of 58 rejections, 16 of 35 suggestion_accepted) | M2 |
| same AI reviewer as two actors: `Copilot` (agent, comments) and `copilot-pull-request-reviewer[bot]` (system, reviews) | 17 traces each | M2 |
| `(trace, check name, sha)` keys seen more than once | 0 of 342 | M5 |
| PRs whose CI covers more than one commit | 0 of 26 (mean 7.5 commits per PR) | M5 |
| rejections whose `changed_by` is unknown (LEFT side) | 3 of 53 | M4 |
| reviews in state `DISMISSED` (dropped) | 4 | M20 |
| `suggestion_accepted` with no change at or next to the line | 1 of 35 (`pcl#6364` ev 4, a false positive, M3) | M3 |

## 1. Mechanism table

| Mechanism | Crucible | SWE-smith (`REF/swe-bench-swe-smith/swesmith/`) | NeMo-Gym (`REF/nvidia-nemo-gym/resources_servers/`) | verifiers (`REF/primeintellect-ai-verifiers/verifiers/v1/`) | Harbor / mini-swe / EnterpriseOps / repo2rlenv / research-envs |
|---|---|---|---|---|---|
| Review-comment intent | 3 regexes, first match wins: praise (whole comment), request (~50 cues), clarify (`?`) `CP/signals.py:29-58`. Code blocks and `>` quotes stripped, inline code kept | none | pivot candidates by LLM judge, N-run majority vote, 6 failure patterns `swe_pivot/README.md:26-27` | closed-choice judge: reason before verdict `judges/rubric.py:24-35`; verdict parse returns `None` on unfinished think or no verdict `utils/score.py:51-81`; structured output with refusal check `judge.py:186-201` | research-envs keeps rubric, taste and verifier sub-scores apart `environments/swe/senior_swe_bench/senior_swe_bench/taskset.py:140-149` |
| Acted-on evidence | connector: blame + line tracking ±1 line `CP/connectors/github_pr.py:273-290`; mapper: `acted = bool(changed_by)` `CP/signals.py:87` (missing = not acted) | n/a | layered checks with a reason code per failure `swe_pivot/app.py:16-26,473-481` | n/a | n/a |
| Suggestion applied | normalized suggestion text is a substring of the final file `github_pr.py:291-297`; mapper does not cross-check `changed_by` `signals.py:90-91` | n/a | argument similarity with thresholds (partial credit 0.4-0.8) `swe_pivot/app.py:400-450,488-500` | n/a | n/a |
| Review state | APPROVED +1, CHANGES_REQUESTED -1, COMMENTED with request -1; graded on the reviewed head commit `signals.py:98-107`; DISMISSED dropped | n/a | n/a | n/a | n/a |
| Bot / AI reviewers | kind from `[bot]` suffix (system) or `type: Bot` (agent), first sight wins `github_pr.py:191-199`; signals carry `meta.automated` `signals.py:74`, read by nothing | n/a | n/a | n/a | repo2rlenv `exclude_authors` for bots `src/repo2rlenv/spec/options.py:119`; Crucible's own `github_reviews.py:177-178` drops bots |
| CI result | one `check_result` per check run; flaky = success and failure for one (name, sha) `signals.py:76-79,108-116`; check runs fetched for the head only, default filter `github_pr.py:160` | flaky tests found by re-running gold; F2P/P2P failures removed, instance dropped if no F2P passes, timed-out instances dropped `harness/repair.py:122-171`; XFAIL counts as pass, missing test as fail `harness/grading.py:95-106` | per-pass verdicts: supported / broken / flaky / **inconclusive** (fewer verdicts than expected) `swemer_v2/aggregate_golden_patch.py:51-64` | n/a | repo2rlenv strips `(retry xN)` from test ids `src/repo2rlenv/pipelines/_pr_runtime_verifier.py:196-199` |
| CI logs (per test) | `ci_logs` parses JUnit/pytest/gtest/ctest, `flaky()` per commit, infra masking `CP/connectors/ci_logs.py:45-84,313-323`; **no mapper for `coding/ci_run`** | as above | as above | n/a | n/a |
| Revert / hotfix after merge | `revert` word + `#N`, merge sha or quoted title `github_pr.py:364-373`; fix-like = any of 11 words anywhere in subject + body (+ merged-in commit bodies) `:40,365-366,383` | n/a | n/a | n/a | repo2rlenv rejects conventional-commit types chore/docs/refactor/style/test/ci/build/perf/revert, then needs `fix:` prefix, `Fixes #N` or a keyword in the subject `pipelines/commit_runtime.py:84-104,465-500`; non-bug PR titles (revert, backport, cherry-pick, bump, release) `pipelines/pr_runtime.py:318-335`; revert PRs `pipelines/pr_diff.py:379-381` |
| Frontier-run outcome | failing `grade_gate` events -> -1 per gate; else `failing_gates`; else reward `>= 1` -> +1 / -1 `signals.py:136-159` | timed-out runs are dropped, not failed `repair.py:122-136` | binary or continuous reward mode `swe_pivot/app.py:488-500` | n/a | Harbor rejects empty, non-numeric and non-finite rewards `src/harbor/verifier/verifier.py:67-122`; EnterpriseOps reports per-verifier pass rate next to all-pass success `benchmark/executor.py:371,565-612`; mini-swe exit statuses (`LimitsExceeded`, `TimeExceeded`, `Submitted`) `src/minisweagent/exceptions.py:1-25` |
| Ticket events | table map + text fallbacks for rule, SLA, CSAT `TP/signals.py:26-61` | n/a | n/a | n/a | EnterpriseOps verifiers: DB state, response check, tool execution; an error is `passed: False` with an `error` field `benchmark/verifier.py:48-193` |
| Signal id | `sha256(trace, type, source seq, extra)` `C/signals/extract.py:32-33`; duplicates raise `:61-65` | n/a | n/a | n/a | n/a |

## 2. Defect candidates

Severity: P0 = corrupts most labels of a kind in normal use; P1 = corrupts a common case or blocks a stated gate; P2 = edge case or cosmetic.

### Coding mapper and the facts it trusts

**M1 [P1][ran] Intent regexes misread negations, questions, praise and non-English.** `intent()` (`CP/signals.py:49-58`). 25 of 29 short comments in probe p1 are wrong. The inputs are hand-picked, so this is not a precision estimate. The dev gate scores 1.0, but its labels come from the same builder who tuned the regexes, all from one repo (`labels_dev.json` `labeled_by`).
- False `request` (becomes a `rejection` or `correction`, both -1): `"No need to change this, it's fine as is."` (cue `need to`), `"Don't worry about it, this is fine."` (`don't`), `"I don't think we need to change this."`, `"This is not a bug, it is the documented behaviour."` (`bug`), `"FYI this was a bug upstream; this PR handles it correctly."`, `"This calls `os.remove` safely, nice."` (inline code keeps `remove`).
- Questions read as requests: `"Should this also handle the empty cloud case?"`, `"Was this missing before?"`, `"Could you explain why this is needed?"` all give `request`. REQUEST runs before CLARIFY, so any cue wins over the `?`. Probe p2 fed the first one, answered by the author with no code change. Got: `rejection` -1 grading c1, `acted_on: False`. The spec says a question answered without change carries no signal (`signals.py:11`, `labels_dev.json` guideline).
- Praise missed (gives `discussion`, so no approval): `"Looks good to me"`, `"LGTM, thanks!"`, `"LGTM 🚀"`, `"**LGTM**"`, `"👍🏻"`, `"+1"`, `"Thanks for fixing this!"`, `"Good catch, I missed that."`. PRAISE must match the whole comment, and its trailing class allows only `[\s!.👍]`.
- Requests missed: imperatives without a listed cue (`"Use size_t here."`, `"This leaks memory when n == 0."`, `"This can overflow for large inputs."`, `"Off by one: ..."`) and every non-English comment (`"Bitte diese Zeile entfernen."`, `"这里应该删除这个变量"`).

Fix: run the regex as a fast path only. Send comments with a negation in front of a cue, a trailing `?`, or no cue to a closed-choice judge (request / clarify / praise / discussion / abstain) built the verifiers way (reason first, `None` on no verdict, M-feature 3). An abstention means no signal. Add an independent held-out label set (ARCHITECTURE already lists held-out labels as pending).

**M2 [P0][ran] Half of the corpus's expert signals come from an AI reviewer and are pooled with humans.** In the corpus, `Copilot` (kind `agent`) produced 54 of 110 corrections, 37 of 58 rejections and 16 of 35 accepted suggestions. The mapper sets `meta.automated` (`signals.py:74`), but no code in `src/` or `packs/` reads it (grep). `reliability.estimate` keys by type only (`reliability.py:69-80`). The connector's docstring says AI reviewers are "kept apart from human experts' in reliability" (`github_pr.py:197`). They are not. The same reviewer is also two actors with two kinds: comments come as `Copilot` (`type: Bot`, so `agent`) and reviews as `copilot-pull-request-reviewer[bot]` (suffix, so `system`) (`github_pr.py:194-197`). Probe p2 shows both producing `rejection` -1. The other review path drops bots outright (`github_reviews.py:177-178`), so the two coding paths disagree. Fix: one canonical identity per AI reviewer (an alias table); key reliability by (type, actor class); report the gate per actor class.

**M3 [P1][ran, real PR] `suggestion_applied` is a substring test over the whole final file.** `github_pr.py:291-297` marks a suggestion applied when its normalized text occurs anywhere in the head file. The mapper then emits `suggestion_accepted` (prior 4:1, the highest-trust type, the source of pivots and DPO pairs) without checking `changed_by` (`signals.py:90-91`). Probe p6 used a fake repo whose commented line never changed:
- a re-indent suggestion `int max = 10;` -> `suggestion_applied: True, changed_by: None`;
- a suggestion `return max;` that already exists elsewhere in the file -> `True`;
- a "This should be:" code block holding `}` -> `True`.

Real case `github:PointCloudLibrary/pcl#6364` ev 4. The reviewer suggests `add_library(${_name} INTERFACE)` for line 267. The final commit `07815a5c` still has `add_library(${_name} INTERFACE ${ARGS_INCLUDES})` at 267, but the suggested text exists at 271 (the else branch). Got `suggestion_accepted` with `line_changed: False, near_changed: False`. This event is not in the 50 gate labels. Fix: require the change to land in the commented span (`line_changed or near_changed`), compare against the span in the final file rather than the whole file, and treat a suggestion equal to the original span as a no-op.

**M4 [P1][ran] An unknown "acted on" is read as "not acted on".** `acted = bool(d.get("changed_by"))` (`signals.py:87`). The connector computes `changed_by` only for RIGHT-side comments whose commit it has (`github_pr.py:273`). A LEFT-side comment (a deleted line), a file-level comment (no line) or a comment on an unrecoverable commit has no key at all. Input (p2): `review_comment` `"Please keep these comments"`, `data={"side":"LEFT"}`. Got `rejection` -1, `acted_on: False`. The corpus has 3 such rejections in `pcl#6430`, all LEFT-side (`"Please keep these comments"`, `"Please also copy these explanation comments to the .cpp file"`). With `acted_on: False` on a merged PR, `verdict()` scores them as contradicted, which deepens the circular reliability. Fix: `acted_on=None` when the key is missing, and emit no rejection (or an `unknown` reason code) until the connector tracks LEFT-side spans.

**M5 [P1][ran on corpus] CI flakiness can never fire on API data, and CI grades only the head commit.** The connector fetches `GET /commits/{head}/check-runs` (`github_pr.py:160`). That endpoint returns the latest run per check by default (`filter=latest`), so reruns never arrive: 0 of 342 `(trace, name, sha)` keys repeat in the corpus. The flaky branch (`signals.py:112-113`) is dead on real data. A check that failed and was rerun to green shows as one `+1`. Only the head sha is fetched, so 0 of 26 PRs have CI on more than one commit, while PRs average 7.5 commits. DESIGN §5.2's "CI result per commit" (`DESIGN.md:393`) is not delivered: every `check_result` grades the last commit, whatever introduced the break. Fix: fetch `filter=all` for every PR commit (or the workflow-runs API with `run_attempt`), and key by (app, name, sha).

**M6 [P1][ran] `ci_logs` traces cannot be extracted.** No mapper is registered for `coding/ci_run` (`CP/__init__.py:26-27`). Input (p2): a JUnit report with one failing test, normalized by `CILogConnector`. Got `extract()` -> `LookupError: no signal_mapper registered for family 'coding/ci_run'`. Running `CodingSignalMapper` on it directly gives `KeyError('name')` (`signals.py:79` expects API-shaped `ci_run` data). So the per-test parsing, infra masking and `flaky()` of `ci_logs.py` never reach M4. Fix: a `CIRunSignalMapper`: per-test `check_result` excluding `flaky()` tests, and the run unscored when `failure_kind == "infra"`. This is the SWE-smith `repair.py` rule the spec cites.

**M7 [P1][ran] A review-level verdict grades the head commit, not the commit it is about.** `signals.py:101-107`. Input (p2): commits c1 (`a.py`), c2 (`README.md`), then `CHANGES_REQUESTED` on c2: "The lock in a.py is taken twice." Got `rejection` -1 grading c2, the README commit. COMMENTED reviews with a request take the same path, and APPROVED grades one commit for a cumulative diff. Fix: a review verdict grades the PR's commits up to the reviewed sha (or none, as a trace-level signal). Grade a single step only when the review's own inline comments point at lines.

**M8 [P1][ran, expression copied] Revert detection flags re-lands and mentions.** `github_pr.py:367-373` (inline; p6 copies it verbatim). For PR #123 titled "Tweak launch bounds for sm_90", all of these give `is_revert=True`:
- `Revert "Revert "Tweak launch bounds for sm_90""` (a re-land);
- `Reapply "... (#123)" after the revert of #130 was itself reverted`;
- `Fix regression from #123; do not revert, adjust bounds instead`;
- `Update docs: #123 is not reverted in 1.15`.

`revert` has prior 3:1, and a revert makes the outcome `bad`. Fix: require `^Revert "` or git's `This reverts commit <sha>` trailer naming a PR commit or the merge. Count `Revert "Revert` pairs, so an even number cancels.

**M9 [P2][ran] `fixlike` matches non-fixes.** `FIXLIKE` (`github_pr.py:40`) on subject + body, plus all merged-in commit bodies for merges (`:365-366`). `refactor: rename helpers (fixes lint warning)`, `docs: explain the bug-report template`, and a release merge whose body has `fix typo in README` are all fix-like. When such a commit touches PR lines, the mapper emits `correction` -1 `hotfix` (p2, ev2). Fix: repo2rlenv's two-stage rule (feature 4).

**M10 [P2][ran] Reviewer replies under an author's root are dropped.** `signals.py:83` skips every reply. Input: the author's root "I moved the guard here, OK?", then the reviewer replies "No, please move it back: it must run before the lock." Got: no signals. Corpus: 1 case (`pcl#6399` ev 16). Fix: when the root is the author's, the first reviewer reply acts as the root.

**M11 [P2][ran] A suggestion with praise words is an approval.** `"Nice!\n```suggestion\nreturn x\n```"`, not applied -> `approval` +1 (`signals.py:92,96`). It should follow the suggestion path.

**M12 [P2][ran] CI conclusion spelling and name collisions.** Conclusions `startup_failure`, `FAILURE`, `SUCCESS` (GraphQL enum case) and `error` (commit-status API) each give `check_result` with polarity 0 and `flaky: False`: a signal with no information (p2). Two workflows (GitHub Actions and Azure) with a job named `build` on one sha, one green and one red, make both `flaky: True`, polarity 0, so the real failure vanishes (`signals.py:78`, keyed by name only). Fix: lowercase, map `startup_failure`/`error` to fail, key by (app, name).

**M13 [P2][ran] Revert with no blamed lines grades every final commit.** `touched or final_commits` (`signals.py:120`). With c1 (code) and c2 (README), a revert with `touches_seq: []` grades `[0, 1]`. For a squash merge, `touches_pr_commits` holds the merge sha, which has no commit event (`github_pr.py:311` vs `:245,251`), so this fallback is the normal path for squash PRs. Fix: grade the commits whose files the revert touched.

### Frontier-run mapper (`AgentRunSignalMapper`)

**M14 [P1][ran] A contaminated green run yields `+1`.** The connector marks a reward of 1.0 reached after an answer-key probe as `unknown`, `contaminated: True` (`CP/connectors/agent_runs.py:87-89`). The mapper reads only `reward` (`signals.py:153-156`). Input: details `{"reward": 1.0, "contaminated": true}`, label `unknown`. Got `check_result` +1 on `gate:*`. Fix: no positive signal when `contaminated`.

**M15 [P2][ran] A fully passing graded run yields nothing.** With `grade_gate` events all passing, `out` is empty and returned (`signals.py:150-151`). The same run without a kept report gives +1 (`:153-156`), and the docstring promises "a passing run yields one `check_result` +1". Fix: emit +1 per passing gate (or one `gate:*`) in the graded branch too.

**M16 [P2][ran] Reward edge cases.**
- Reward `"1.0"` (a string) -> `TypeError` (`signals.py:154`).
- `nan` -> -1.
- 0.5 -> -1, the same as 0.0.
- `failing_gates: ["perf"]` with reward 1.0 -> -1 on `gate:perf` while the outcome is `good`.
- `failing_gates: ["tests","tests"]` -> two identical ids, and `extract()` raises `ValueError: duplicate signal id` (`extract.py:61-65`).
- A `grade_gate` with no name -> `target gate:None` (twice, distinct only by seq).
- `"pass": "false"` -> `bool("false")` is True, so no signal (`signals.py:145`; the connector also uses `bool()`, `agent_runs.py:72`).

Fix: validate like Harbor (finite numeric or unscored); dedupe gate names; keep partial reward as per-gate signals.

### Toy mapper

**M17 [P1][ran] The toy pack's M4 labels cannot be run and fail when forced.** `labels_test.json` (84 events) names traces `toy/helpdesk_ticket:*`. The mapper is registered for `toy/support_ticket` (`TP/__init__.py:20`). Got `extract()` -> `LookupError` for `toy/helpdesk_ticket`. No toy M4 gate script exists. Forcing `mapper=ToyTicketSignalMapper()` and scoring with `signals.gate.evaluate` (p4) gives type accuracy **0.869** (< 0.9), attribution precision 0.909, outcome agreement **0.083** (< 0.95: no toy outcome labeler). Type errors:
- `csat` is labelled `outcome` but mapped to `check_result` (4 cases);
- customer replies labelled `rejection` (3) and `approval` (3) have no mapping at all;
- one CSAT is missed.

ARCHITECTURE's exit gate is "50 hand-labeled events per pack" (`ARCHITECTURE.md:531`), and M8 traps cite toy M4 signals. Fix: register the mapper for `toy/` (prefix) or align families; add customer-reply and `csat -> outcome` rules; add a toy outcome labeler and gate runner.

**M18 [P2][ran] CSAT parsing.**
- Score `"3"` (a string) -> -1, because `"3" != 3` (`TP/signals.py:58`).
- `"4.5"` -> `ValueError` that aborts extraction.
- The scale is assumed 1-5: 6 -> +1, 0 -> -1.
- The text fallback takes the first digit from 1 to 5: `"Took 2 days, but rated 5/5"` -> score 2 -> -1.
- `"10/10 would recommend"` -> no signal.

Fix: numeric coercion, a declared scale, and no text fallback (or a judge).

**M19 [P2][ran] SLA and rule fallbacks.** `sla_breach` without `data.sla` defaults to `first_response_4h`, so `"Resolution SLA (24h) breached."` is mislabelled (`TP/signals.py:48`). `"First response late; ticket was not reopened."` -> `reopen_response_4h`. `rule_of` misses `"Refund above $50 ..."`, `"Refund of 75.00 exceeds the 50 limit ..."` and `"ID verification skipped"` (`:26-27`). It matches `identity_check_before_deletion` on `"... identity check done."`. A queue's first routing `reassigned` is an `escalation` -1. Toy-only data, so P2.

### Shared

**M20 [P2] Dismissed reviews vanish.** GitHub sets a dismissed review's state to `DISMISSED` and loses the original state. The corpus has 4. The mapper drops them (`signals.py:104-107`). A dismissed change request is evidence that the maintainers overruled the request. Keep it as `rejection` with `dismissed: true` (excluded from grading) or as an annotation.

**M21 [P1][ran] Signal ids depend on event seq, so they change on re-ingest.** `signal_id(trace, type, seq)`. Adding one earlier event (`pr_comment "ping"`) moves an approval from `sig_8a0a73751dd1c331` (seq 1) to `sig_3f5bf4cb5e304178` (seq 2) (p3). `TraceStore._carry_signals` remaps `grades`/`source_event` but keeps the old id (`C/connectors/trace_store.py:44-66`), so a carried signal's id no longer equals what extraction computes. M8 traps cite signal ids as provenance (`TP/forge/hindsight.py:117`). Gate labels are keyed by seq too; the gate rebuilds from the on-disk API cache (`CP/m4/gate.py:21-25`), so this bites only when the cache is refreshed (one labelled PR, `pcl#5958`, is still open). Fix: id from the source event's identity (`TraceStore._identity`: row id, or time/type/actor/text) instead of seq.

**M22 [P2] One comment, one signal.** The mapper emits at most one signal per event, so ids never collide inside `CodingSignalMapper` or the toy mapper. A review comment with three requests, or a review body listing several defects, becomes one signal. Splitting needs the `extra` argument of `signal_id`, already used by `AgentRunSignalMapper`.

## 3. Features worth porting (ranked)

| # | Feature | Reference | Mode | Owning Crucible file |
|---|---|---|---|---|
| 1 | Actor-class separation for bot and AI reviewers: one alias table (`Copilot` = `copilot-pull-request-reviewer[bot]`), signals and reliability keyed by (type, class), gate reported per class | repo2rlenv `src/repo2rlenv/spec/options.py:119`; Crucible's own `github_reviews.py:177-178` | PATTERN | `github_pr.py:191-199`, `reliability.py:69-80` |
| 2 | CI verdict classes with an **inconclusive** bucket, from all attempts per (app, name, sha); flaky tests excluded from per-test grading, timed-out runs dropped | NeMo-Gym `swemer_v2/aggregate_golden_patch.py:51-64` (PORT, 14 lines); SWE-smith `harness/repair.py:122-171`, `harness/grading.py:95-106` (PATTERN) | PORT + PATTERN | `signals.py:76-116`, new `coding/ci_run` mapper over `ci_logs.flaky()` |
| 3 | Closed-choice intent judge behind the regex: reason before verdict, abstain on no verdict, structured output with refusal check; the regex stays the gate path | verifiers `judges/rubric.py:24-35`, `utils/score.py:51-81`, `judge.py:186-201` | PATTERN in core (`verifiers` is a forbidden core import, `pyproject.toml:130`); DEPEND from the pack under the `verifiers` extra | `signals.py:49-58` |
| 4 | Conventional-commit-aware fix and revert classifier: reject chore/docs/refactor/style/test/ci/build/perf, need `fix:` / `Fixes #N` / a subject keyword; revert, backport and cherry-pick titles flagged | repo2rlenv `pipelines/commit_runtime.py:84-104,465-500`, `pipelines/pr_runtime.py:318-335`, `pipelines/pr_diff.py:379-381` | PORT | `github_pr.py:40,367-373,383` |
| 5 | Reason codes when no signal is emitted (unknown span, LEFT side, predates PR, suggestion no-op) instead of choosing `rejection` | NeMo-Gym `swe_pivot/app.py:473-481` (`FailureCode`) | PATTERN | `signals.py:82-97` |
| 6 | Reward validation: finite numbers only, else unscored | Harbor `src/harbor/verifier/verifier.py:67-122` | PATTERN (Harbor is a dependency; the check is 20 lines) | `signals.py:152-156` |
| 7 | Partial outcomes as per-gate pass and fail signals, plus an all-pass flag, instead of one sign of the reward | EnterpriseOps `benchmark/executor.py:371,565-612`; research-envs `senior_swe_bench/taskset.py:140-149` | PATTERN | `signals.py:136-159` |
| 8 | Run exit status: `LimitsExceeded`/`TimeExceeded` runs are unknown, not gate failures | mini-swe-agent `src/minisweagent/exceptions.py:13-18`, `agents/default.py:80,133-145` | PATTERN | `AgentRunSignalMapper`, `agent_runs.py` |
| 9 | Suggestion matching by span similarity with a threshold rather than file substring | NeMo-Gym `swe_pivot/app.py:400-450` (`_capped_similarity`, `compute_argument_similarity`) | PATTERN | `github_pr.py:291-297` |
| 10 | Retry suffixes stripped from test ids before per-commit flaky grouping | repo2rlenv `_pr_runtime_verifier.py:196-199` | PATTERN | `ci_logs.py` parsers |

## 4. Where Crucible is stronger

- **Step-level attribution from git.** No reference ties a review comment to the PR commit that wrote the line (blame at the comment's commit), follows it through later pushes and rebases, and recovers superseded commits (`github_pr.py:175-184,273-290`). SWE-smith and NeMo-Gym grade whole instances or single pivot turns.
- **Acted-on evidence from the code, not the text.** A request becomes a `correction` or a `rejection` by what happened to the lines afterwards. NeMo-Gym's pivot judge reads only the transcript. The defects above (M3, M4) concern precision, not the idea.
- **Post-merge history as signal.** Reverts and hotfixes are carried through renames and line shifts for 30 days and blamed back to PR commits (`github_pr.py:302-386`). No reference mines `vcs_history`. repo2rlenv uses commit messages only to filter tasks.
- **One taxonomy across domains.** Tickets and PRs map onto the same 13 types with priors (`taxonomy.py:24-38`). The references keep domain-specific verdicts (EnterpriseOps DB checks, SWE-smith F2P/P2P) with no shared signal layer.
- **Infra failures masked, never scored 0.** `ci_logs` separates runner failures from code failures and refuses "good" for a run cut off without a summary (`ci_logs.py:69-84,286-306`). SWE-smith drops timed-out instances; EnterpriseOps counts verifier errors as failures.
- **Deterministic, idempotent extraction with a duplicate-id guard** (`extract.py:32-33,61-65`), plus a gate with type, attribution and outcome metrics (`C/signals/gate.py`). The id needs a stable key (M21); the guard already catches M16's duplicate.
- **Frontier failures as gate-level negatives.** `AgentRunSignalMapper` turns hidden-verifier gate failures into trap sources. No reference connects a frontier run's failed gate to task synthesis.

## 5. Suggested audit order

M2, M3, M4, M5/M6, M1, M8, M14, M17, M7, M21, then the P2 items. Each [ran] input above fails on HEAD and should become a regression test.
