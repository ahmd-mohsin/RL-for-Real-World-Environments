# 11. Observability: every result, graph and agent trace in Weights & Biases

The user's requirement (2026-10-20): all results and graphs visible in W&B, and agent traces visible the way people
look at agent traces. No W&B key exists yet, so everything is built to work **offline now and upload later**.

Code: `src/crucible/observe/` (the only module that imports `wandb`, `weave` or `wandb_workspaces`; an import-linter
contract enforces it), `scripts/observe.py`, wired into `train/bridges/tinker/trainer.py` (the `sinks` argument),
`train/sinks.py` (`WandbSink` is now the observe gateway), `factory/tinker.py` (`observe=` on `train`, `benchmark`,
`compare`) and `scripts/tinker_rl.py` (on by default, `--no-observe`). Extra: `observe` (`wandb`, `weave`,
`wandb-workspaces`). Tests: `tests/unit/test_observe.py`.

## 1. What lives where

| what | where (W&B) | local, before sync |
|---|---|---|
| task building: funnels, targeting, intent and verifier cards, the trace audit, failure findings, spend | project `crucible-factory`, one run per report or phase, group = lane or report family | `data/observe/runs/crucible-factory/<run>/` |
| RL post-training: per-step dashboards, drops, the trained episodes, the adapter refs | project `crucible-train`, one run per training run, group `objective-<objective id>`, job type `rl` | `data/observe/runs/crucible-train/<run>/` |
| benchmark arms and comparisons | project `crucible-bench`, one run per arm (job type = purpose: selection, baseline, control, trained, smoke) and one per comparison (job type `compare`); every run of one task list shares the group `tasks-<manifest hash>` | `data/observe/runs/crucible-bench/<run>/` |
| agent traces (Weave) | the Weave tab of the project that produced the episode (`crucible-bench` for benchmark episodes, `crucible-train` for trained ones) | `data/observe/weave/<project>/spans.jsonl`, `index.jsonl`, `full/` |

Each local run directory holds the W&B offline run (`wandb/offline-run-*`, what `wandb sync` uploads), `ops.jsonl`
(every logged operation, redacted: the mirror), `media/*.html` (the HTML charts, viewable in a browser) and
`artifacts/` (the artifact files). The mirror means a run logged on a machine without `wandb` is still complete and
can be replayed into W&B later (`scripts/observe.py replay`).

**Naming** (`observe/naming.py`). Run names: `bench-<purpose>-<model>-s<seed>-<id>`, `bench-compare-<model>-s<seed>-<id>`,
`train-<model>-<run dir>`, `factory-<lane>-<phase>-<id>`. Tags `key:value`: `model:qwen3.6-35b-a3b`, `purpose:`, `k:`,
`thinking:`, `harness:`, `lang:`, `loss:`, `rank:`, `group:`, `checkpoint:base|adapter`, `protocol:ok|problems`.
Config: the full settings of the run. A training run records the whole `TrainConfig` (model, LoRA rank, loss and its
config, learning rate, group size, groups per step, staleness, async groups, caps, thinking, temperature, max tokens,
turn η, honesty, seed) plus the objective id, the task manifest hash and list, the harness, tool limit, timeouts, the
Φ checkpoint config. A benchmark arm records `arm_settings` (model, context, k, temperature, max tokens, thinking, tool
limit, timeouts, system prompt, harness, codec), the checkpoint, the purpose, the seed and the manifest hash. The
entity is `WANDB_ENTITY` (or `CRUCIBLE_WANDB_ENTITY`); unset, it is the key's default entity at sync time.

## 2. Training dashboards (`crucible-train`)

`TinkerTrainer(..., sinks=[...])` logs through a `train.sinks.Tracking` (a failing backend is disabled with one
warning, never stops training). After each optimizer step, `trainer.step_metrics`:

| keys | meaning |
|---|---|
| `reward/{mean,std,min,max}`, `pass_rate`, `phi/final_{mean,std,min,max}`, `gates/pass_rate` | the trained batch's outcomes (masked episodes never count) |
| `advantage/sequence/*`, `advantage/token/*`, `advantage/token_abs_mean` | the outcome advantage per sequence and the per-token advantage over the policy's tokens (M6 turn credit included) |
| `train/loss`, `train/<every forward_backward metric>` | Tinker's loss and its metrics |
| `kl/sample_vs_train`, `kl/sample_vs_train_abs_max`, `ratio/{mean,std,min,max}` | the sampler's logprobs against the learner's on the same tokens, and the importance ratio π_train/π_sample |
| `staleness/{max,mean}` | optimizer steps between the weights that sampled a token and the trained step |
| `drops/<reason>`, `drops/total` | cumulative drops: `infra_in_group`, `stale`, `zero_spread`, `objective_changed`, `unused_at_stop:*` (each drop is also a row of `records/drop`) |
| `tokens/{prefill_uncached,prefill_cached,sampled,train,trained}` | tokens booked since the previous step |
| `usd/step`, `usd/cum`, `usd/cap`, `usd/cap_frac` | spend per step and cumulative against the hard cap |
| `sampler/version`, `episodes`, `groups`, `datums`, `time/train_s`, `time/step_s` | |

At the end (`observe.train_views.finish_train`): the trained episodes as Weave traces and the `train/episodes`
table with their links, `train/steps`, and the artifacts: `train-run-*` (config, receipt, steps, drops, spend,
versions, state, summary, rewards; the `tinker://` sampler and state paths in the metadata, since W&B cannot fetch
them) and `adapter-*` (alias `latest`).

## 3. Benchmark reports (`crucible-bench`)

Per arm (`observe.bench.log_benchmark`, called by `factory.tinker.benchmark`): every number of the metrics block as
`bench/*` scalars (pass@1, pass^k, mean final Φ, task-clustered CIs, calibration, give-up, hack rate, USD per passing
episode), `bench/pass_at_k` (native line chart) and `bench/pass_at_k_chart` (HTML), `bench/per_task` (c of n with a
Wilson 95% interval, USD, USD per pass, language/family/skill) and its HTML bar chart with whiskers,
`bench/by_language`, `bench/by_family`, `bench/by_skill`, and `bench/episodes` (one row per episode: reward, passed,
masked and its infra reason, stop, Φ, milestones lost, gates, turns, tool calls, USD, the audit's cause, mode, first
wrong step and quote once audited, and the **trace link**). Artifacts: `bench-manifest-<hash>` (the task list,
settings, task metadata), `bench-report-*` (`metrics.json`, `episodes.jsonl`, `spend.jsonl`), `episodes-*` (every
episode's full text).

Per comparison (`log_compare`, called by `factory.tinker.compare`):

- `compare/headline`: **the pre/post chart**, pass@1 per arm (selection, baseline, control, trained, in that fixed
  color order) with its task-clustered 95% CI, the paired trained − control difference with its CI and MDE in the
  subtitle, and the protocol problems if any. W&B's chart presets have no error bars, so this is inline SVG in an HTML
  panel (`observe/charts.py`); `compare/pass@1` is the same numbers as a native bar chart and `compare/arms` as a table;
- `compare/paired/*` scalars (mean difference, sd, MDE measured and assumed, significant, n tasks), `paired_ci_lo/hi`,
  `compare/paired_per_task`;
- `compare/protocol` (the `check_protocol` problems, or `none`) and `compare/protocol_ok`;
- `compare/per_task_<arm>` tables and `compare/per_task_chart`; `compare/pass_at_k`, `compare/pass_hat_k` per arm;
  `compare/usd_per_passing_episode`; `compare/by_{language,family,skill}` tables and charts per arm;
- the artifact `compare-report-*` with `compare.json` and `report_template.json`.

Language, family and skill come from `factory.tinker.task_meta` (a task's own `language` / `family` / `intent`, else
its lane: a kernel-lane task is CUDA, a window task Python, the family is the repository) or the `meta=` argument.

**The W&B Report** (`observe/report.py`). `LAYOUT` is the panel layout as data, written beside every comparison as
`report_template.json`: the headline media panel, the arms table, the paired difference and protocol scalars, per-task,
pass@k / pass^k and breakdown media panels, the protocol and paired tables, the episodes table, and the training
curves (reward, spend vs cap, drops, loss and KL). `scripts/observe.py report --entity E --group tasks-<hash>` builds
it with the Reports API (`wandb_workspaces.reports.v2`; offline); `--push` saves it once the runs are synced.

## 4. Task-factory views (`crucible-factory`)

`observe/factory_views.py` takes the factory's own dicts, so a live phase and a report on disk are logged the same way:

- `funnel`: per lane, candidates in and passing at each stage in order (`windows.funnel()["total"]` counts or
  `Funnel.summary()` in/pass/reasons), one table `funnel/table` and a bar chart per lane `funnel/<lane>`;
- `targeting`: per task the target model's pass rate and band (in band = the group's rewards have spread), the
  reward range, the cause mix (skill-gap share, labels), the counterfactual, the prerequisite probe (the fairness
  solver), and the S15 verdict, route, owner, findings and flags; `targeting/cause_mix`; `targeting/band_chart`;
- `intent_cards`: `IntentCard` rows (or `BuilderIntent` candidates with their student and mode);
- `verifier_cards`: per family false-pass and false-fail (errors of n, rate, Wilson upper bound), mutant kill,
  alternatives passing, flake bound, κ, confirmed hacks; `verifier_cards/chart`;
- `audit_summary`: facts, first-wrong-step kinds, causes (deterministic and trusted labels), the auditor's model,
  trusted labels and spend; `findings`: `FailureFinding` rows, each attached to its episode's trace (below);
- `spend`: USD per phase or role, the total in the summary.

## 5. Agent traces (Weave)

**What Weave can and cannot do offline (weave 0.53.12, checked in its source).** `weave.init` always logs in to W&B
(`wandb.login`) and asks the trace server for its version before it creates a client; `WEAVE_DISABLED=true` makes
every call a no-op and keeps nothing; the write-ahead log (`WEAVE_ENABLE_WAL`, `~/.weave/wal/<project>`) only exists
behind an authenticated client, so it is a durability feature, not an offline mode. There is no "log now, sync later"
for Weave the way `wandb offline` + `wandb sync` works for runs.

**The workaround.** Crucible writes each episode as **Weave-shaped spans** to `data/observe/weave/<project>/spans.jsonl`
(id, trace id, parent id, op, display name, inputs, output, attributes, start and end time), and `weave_sync` replays
them later through `WeaveClient.create_call(..., _call_id_override=id, started_at=t)` / `finish_call(..., ended_at=t)`.
The call ids are UUIDv7 (Weave's format) made deterministic from the episode key and its start time, so the trace link
written into a W&B table today (`https://wandb.ai/<entity>/<project>/r/call/<id>`, or `weave:<project>/call/<id>` while
no entity is known) opens the trace after the sync. `_call_id_override` is a private keyword: if a later weave drops it,
the sync uses Weave's ids and writes the mapping to `links.json` (`scripts/observe.py links` prints the new URLs).

**One episode** (from the attempt directory the runtime writes: `record.json`, `calls.json`, `trajectory.agent.json`,
`trace.json`, `progress.jsonl`, `rollout.json`, `artifacts/model.patch`):

```
crucible_episode   <task> | <arm> | PASS / FAIL / INFRA
  inputs:  task, prompt, model, arm, seed
  output:  reward, passed, Φ final, stop, masked + infra reason, gates, final message, patch
  attributes.crucible: task, model, policy, arm, run, episode key, reward, gates, gates_passed, stop, masked, infra,
                       Φ0, Φ final, milestones lost, cost, tokens, turns, tool calls, harness, seed
  crucible_turn    turn k (n tools)          one per model call
    inputs:  messages so far (count, the latest tool results or the request), prompt / cached / context tokens
    output:  the model's text, its tool calls, finish, completion tokens,
             phi: {after, delta, gained, lost, credit (the rollout's per-turn advantage, M6)}
    tool_<function>                          one per tool call
      inputs:  name, arguments          output: result, ok, error, Φ after, ΔΦ, milestones gained / lost
  crucible_trace_audit   audit: <cause>      added when the audit runs
    output:  cause, cause source, mode, routed owner, first wrong step (kind, step, turn, files), verbatim quote,
             milestones missed / lost, facts
```

Text is truncated to 2000 characters in the trace view (it says so and names the full-text file); the whole episode
(every message, result, the patch; token ids stripped) is `full/<episode>.json`, logged as the `episodes-*` artifact
of the run that exported it. Timestamps inside an episode are reconstructed from the record's start and the model
call latencies (`attributes.crucible.timing = "reconstructed"`).

**From a failing benchmark row to the auditor's verdict.** The episode key is `<run id>/<sample>/<attempt>`, which
is also the trace audit's bundle id (`traceaudit.discover` / `load_bundle(adir, run=<run id>)`), so a
`FailureFinding.trace` names its episode. `factory_views.findings(run, findings, store=TraceStore(root, project))`
attaches each finding to its episode's trace as a `crucible_trace_audit` child (under the same root call) and logs
the findings table with the trace links; the episode's later rows show the cause, mode, first wrong step and quote.

**Redaction** (`observe/redact.py`): every payload (configs, metrics keys, tables, HTML, artifact files, spans, the
full text) passes the M3 redaction (`connectors.redact.Redactor`: credential formats, private keys, JWTs,
`KEY=value`, cards; e-mails, phones, internal hosts as HMAC pseudonyms under the tenant salt `data/redaction.salt`),
then removal by value of this process's credential environment variables (`WANDB_API_KEY`, `TINKER_API_KEY`,
`OPENROUTER_API_KEY`, ...: a key without a vendor prefix matches no pattern), then `core.secrets.scrub_secrets`. The
values are read from `os.environ`; `.env` is never opened. W&B's own capture is off (console, code, git, system
metadata, the requirements file, system stats). The test plants a vendor-format key in a tool call and its output and
a prefix-less `WANDB_API_KEY` value, and checks that neither appears in any byte written or handed to W&B.

## 6. Commands

```bash
cd crucible
# status: mode, key present (yes/no), runs and spans waiting for upload
.venv/bin/python scripts/observe.py status

# (a) sync, once WANDB_API_KEY is in .env: export it into this shell only, then upload runs and traces
set -a; . ../.env; set +a
export WANDB_ENTITY=<your entity>          # so links resolve; optional
.venv/bin/python scripts/observe.py sync   # wandb sync of every offline run not yet synced, then the Weave spans
# (or by hand: .venv/bin/wandb sync data/observe/runs/*/*/wandb/offline-run-*)

# (b) view an episode trace
.venv/bin/python scripts/observe.py episodes --failing            # keys, arms, rewards, causes, links
.venv/bin/python scripts/observe.py trace <key or unique part> --html /tmp/ep.html   # tree in the terminal + an HTML page
.venv/bin/python scripts/observe.py trace <attempt dir>          # straight from a run directory, nothing exported
# after the sync: open the link from the episodes table, or the project's Weave tab (Traces), filter on
# attributes.crucible.task / .arm / .passed

# (c) backfill existing results (offline; re-running skips what is already in data/observe/backfill.json)
.venv/bin/python scripts/observe.py backfill docs/research/swarm-pipeline/runs data/tinker

# the report, after the sync
.venv/bin/python scripts/observe.py report --entity <entity> --group tasks-<manifest hash> --push
```

Live runs: `scripts/tinker_rl.py train | eval | compare` log to W&B offline by default (`--no-observe` turns it
off); library callers pass `observe=ObserveConfig(...)` (or set `CRUCIBLE_OBSERVE=1`). `CRUCIBLE_WANDB_MODE=online`
logs straight to W&B when the key is set (without it: offline, one warning). `CRUCIBLE_OBSERVE_DIR` moves the root.

## 7. Backfill (2026-10-20)

`scripts/observe.py backfill` over `docs/research/swarm-pipeline/runs` and the Tinker smoke directories
(`data/tinker` of the Tinker-bridge worktree) wrote 36 offline runs to `data/observe` (4.4 MB): factory views for
P2 (verifier cards of every family, the train and held-out funnels), P3 (the trace audit), P4 / P5b / Phase A
retarget (targeting, spend), the kernel lane (funnel, drops, tasks, spend), the held-out benchmark manifest; the
Qwen3.6-35B-A3B RL smoke as a `crucible-train` run (2 steps, its 8 episodes as traces, the adapter path); the five
one-episode tool smokes and the 4B probe as `crucible-bench` runs (5 traces); the probes, template checks and smoke
summaries as `crucible-train` smoke runs. One source is skipped as malformed (`tools-4b/smoke_tools.json`, a truncated
file). No credential-shaped string is in the output. Nothing is uploaded until `sync`.

## 8. Open

- The `scripts/lh_*.py` factory scripts write their reports to disk and are logged by `backfill`; they do not call
  `factory_views` live yet (the functions take the same dicts, so wiring each is one call).
- Weave's `_call_id_override` is private API: the fallback keeps traces uploadable but changes the links (`links.json`).
- Timestamps inside an episode are reconstructed (the runtime records per-call latency, not wall-clock starts).
- The real sync has not run (no key): the first sync should be watched (`status` before and after).
- `weave` depends on `polyfile-weave`, which depends on `cint`, whose wheel installs a top-level `tests` package into
  site-packages. A regular package beats our namespace `tests/` (no `__init__.py`), so every `from tests.unit...`
  import in the suite fails to collect. Installing the extra here was followed by `uv pip uninstall cint` (weave imports
  without it); `uv sync --extra observe` will bring it back until the lock excludes it (for example a
  `[tool.uv] override-dependencies` entry, or a `tests/__init__.py`).
