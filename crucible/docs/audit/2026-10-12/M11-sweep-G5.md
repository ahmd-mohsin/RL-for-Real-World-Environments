# M11 sweep G5: articles and Hugging Face datasets → import coverage, defects and the I0–I10 design for interop

Read-only, 2026-10-12, HEAD `af402b4`. Scope: every article in `RL-Environments/07-articles/` (incl. `prime-agent/`,
`raw/` as duplicates) and every entry in `RL-Environments/08-hf-rl-env-datasets/` (33 dataset folders + `index.md` +
`index.json` = 35 entries), read for M11 `interop` and checked against `crucible/src/crucible/interop/`. Nothing was
edited except this file. No network, GPU, Docker pull, Modal or paid API was used. Local NeMo-Gym servers ran on
127.0.0.1 only.

Short names:
- `ART/` = `RL-Environments/07-articles/`; `DS/` = `RL-Environments/08-hf-rl-env-datasets/`;
- `C/` = `crucible/src/crucible/`; `I/` = `C/interop/`; `NG/` = `RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym/`;
- `PIRE/` = `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-research-environments/environments/`;
- `SP/` = `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m11sweep/g5/`;
- [ran] = reproduced by a probe (§5); [read] = from code or files;
- prior sweep = `docs/audit/2026-10-12/M08-M10-sweep-G5.md` (its G-numbers are cited as "prior G20").

Method: Crucible's M11 code was read in full for the import side (`harbor/{task_io,inplace,multi_step,verify_dir,atif}`,
`nemo_gym/{import_server,client}`, `verifiers/{v1_import,taskset}`, `openenv/import_env`, `mimo/*`, `eog/importer`,
`replay.py`). Samples were loaded through the importers where that was offline and cheap (Q1–Q11). One helper read
all articles for interop requirements (§1). Every file:line below was opened. Quotes are under 15 words.

## Summary

- **Infrastructure failures score 0 in the NeMo-Gym import (P0, [ran]).** A resources server that dies before
  `/verify`, a `/verify` HTTP 500, or a reply without `reward` all give reward 0.0 with status `ok`. The check's
  exception falls into the plan's "candidate crashed" branch (D1).
- **Long final answers are cut before grading (P0, [ran]).** `trace_to_response` sends `Done.reason`, which
  `LLMToolPolicy` caps at 2,000 characters. A correct 2,356-character calendar answer scores 0.0; the same answer short
  scores 1.0 (D2). ATIF export has the same cut (D2b).
- **Separate-verifier Harbor tasks still grade in the agent container (P0, prior G20, [ran]).** `HarborInPlaceTask`
  accepts 8 of 8 local separate-verifier samples. NatureBench's judge key goes into the agent container (D3).
- **Score-gated imports report success on failure (P1, [ran]).** NeMo-Gym and OpenEnv imports write `outcome` 1.0,
  gate `pass` and `check/<gate>` 1.0 for a 0 reward. A1's gold check can therefore never pass on them (D4).
- **The cross-runtime path loses final answers (P1, [ran]).** `replay` of a correct NeMo transcript scores 0.0 against
  1.0 native. The M11 exit gate covers tool-graded tasks only (D6).
- **Patch mode diverges from Harbor on 17 of 28 tasks (P1, [ran]).** It grades with no network where Harbor grades
  public, drops `[verifier].env` and crashes on a multi-key `reward.json` (D8, D9).
- **Coverage:** 4 of 33 dataset folders grade end to end today (TB2.1, termgrade with a local base image, workplace
  assistant, calendar). All 30 Harbor sample dirs parse. **No verifiers
  row dataset (7) and no procedural or RetroEnv set loads.** No dataset-level loader exists (`registry.json`,
  `dataset.toml`, parquet), and no import reaches acceptance (§2).
- **Picks (§4):** DEPEND on Harbor's `DatasetSpec`/`DatasetManifest` for I0; DEPEND on PI's SWE tasksets
  (`PIRE/swe/{r2e_gym,multiswe,scaleswe,swelego,swerebench_v2,openswe}`) through `.envs/verifiers` for row I1/I2;
  PATTERN NeMo-Gym's `aggregate_golden_patch` verdicts for I3; PATTERN OpenEnv's reward-key rule. Build I0–I10 as
  `interop/imports/` adapters that give existing RunTasks the M9 capabilities.

## 0. Prior claims re-checked

| # | Prior claim | Where | Verdict | Evidence |
|---|---|---|---|---|
| 0.1 | In-place mode grades separate-verifier tasks in the agent container | prior G20 | **Still open [ran].** 8/8 separate samples construct; nothing refuses. | Q1; `I/harbor/inplace.py:95-130`, `:211-212` |
| 0.2 | Grading-relevant unknown keys load silently | prior G4 | **Still open [ran].** `agent.continue_until_timeout` is listed, not refused. In-place never calls `load_harbor_task`, so it does not even list it. LHTB fails in place only because it has no `tests/`. | Q1; `I/harbor/task_io.py:105`; `I/harbor/inplace.py:101` |
| 0.3 | A NeMo server can score a grader crash as 0 | prior G12 | **Confirmed and wider [ran].** Beyond the server's own `except: reward=0` (`NG/resources_servers/calendar/app.py:73-76`), Crucible itself turns transport failures into 0 (D1). | Q9 |
| 0.4 | D10: empty ground truth gives a no-op 1.0 | prior P2 | **Confirmed at HEAD [ran].** Validation row 1: oracle 1.0, no-op 1.0; the other 5 rows: oracle 1.0, no-op 0.0. | Q3 |
| 0.5 | No row-taskset adapter | prior G5 | **Still open; closest code found.** PI ships verifiers tasksets for six of the seven row sets with `validate`/`apply_gold_patch`/`solved`, and verifiers v1 has a Docker runtime. NeMo-Gym ships `swe_rebench` and `scale_swe` servers with golden-patch validation. | `PIRE/swe/r2e_gym/r2e_gym/taskset.py:234-254`; `.../verifiers/v1/runtimes/docker`; `NG/resources_servers/swe_rebench/README.md:40-75` |
| 0.6 | "Imports I0–I10 belong in interop" | M09-accept.md:29; ARCHITECTURE.md:624 | **Agreed; nothing built.** No `imports` package, no `import_report.json`, no CLI. | `ls I/` |
| 0.7 | MiMo round trip: "one env imported, run, graded, re-exported byte-identical" | DESIGN.md (M11 part 1) | **True only inside the gate test.** There is no MiMo `RunTask`; grading is a free function that needs the caller to pass the final answer (D14). | `tests/unit/test_interop_gate.py:232`; `I/mimo/verify.py:49` |
| 0.8 | Exit gate: identical files "on task02 and one toy task" | ARCHITECTURE.md (M11) | **Toy only.** No task02 case; no `interop/verl` folder although ARCHITECTURE lists one. | `tests/unit/test_interop_gate.py:33`; `ls I/` |
| 0.9 | Prior P6: termgrade, data-agent, SmolDataEnvs "parse" | prior P1 | **Confirmed; parse ≠ grade.** data-agent masks every grade in place (`ambiguous_reward`) and crashes in patch mode (D8, D10). | Q1, Q2 |

## 1. Interop requirements from the articles, checked against Crucible

One row per requirement. Status is Crucible at HEAD. The helper found no mention in the corpus of `task.toml` keys,
`reward.txt`, canaries, `mask_sample`, −999, NaN rewards or clamping; those rules come from code (research
m11-m12/05), not from the articles.

| # | Requirement | Source (quote) | Status |
|---|---|---|---|
| R1 | A framework tag is a claim that the framework loads the files as they are | `ART/hf-blog-rl-environments.md:41` ("adding a tag does not convert them") | **Missing.** No dataset-level importer; nothing records which tag a dataset carries or checks it (D17). |
| R2 | Harbor datasets are addressed by repo URL plus `name@version` from `registry.json` | hf-blog:88 | **Missing.** Harbor's own `DatasetSpec` parses all 8 local `registry.json` files (Q8); Crucible never reads them. |
| R3 | Run the oracle through the verifier before any model run | hf-blog:57, :68 | In place: `I/harbor/inplace.py:179-188` (OracleAgent). Not wired to M9 for imports (§3). |
| R4 | Several named scores: use the only one, else `reward`, else the caller chooses; never combine | `ART/hf-space-fineenvs-multi-harness-rl.md:704-706` | **Partial.** `scalar_reward` follows it, but `reward` always wins over the caller's `reward_key`, and there is no priority list (`reward_key="correctness,reward"`, :777) (D13). Patch mode crashes instead (D8). |
| R5 | A verifier that never ran gives `reward=None`; a dead sandbox is never a wrong answer | multi-harness:709-710, :765; hf-blog:114-119 | **Broken in two importers:** NeMo grade-time failures score 0 (D1); OpenEnv `reward=None` at termination scores 0 (D5). Met for Harbor in place (`inplace.py:214-217`). |
| R6 | Failed rollouts come back as values, never as exceptions across the trainer boundary | multi-harness:636-639 | **Partial.** `trace_to_record` raises on a `None` or NaN rubric score (D11); `run_vf_eval` raises on any non-zero exit. |
| R7 | Infra noise counted as reward is a poisoned signal | `ART/qingkeai-2026-agentic-rl-infra.md:90` | Broken by D1, D5 and (in patch mode) D9. |
| R8 | Keep sampled token ids and behaviour logprobs; never re-encode text | multi-harness:198-221; qingkeai:480 | Met for verifiers import (`I/verifiers/v1_import.py:54-73`). |
| R9 | A reply cut by the budget is excluded or flagged | multi-harness:583-584 | **Missing for verifiers imports:** `max_turns` traces arrive with `truncated=False` (D11). |
| R10 | Compare each rollout with the harness's own ATIF; any mismatch fails it | multi-harness:712-720 | Met at runtime (`crucible:capture_incomplete`). ATIF export loses long final answers (D2b). |
| R11 | Record harness id and version on every rollout; one harness per group | multi-harness:19-22, :805-806; `ART/pi-verifiers-v1.md:182-191` | Met (`v1_import.py:49` `harness=`). verifiers env version and dataset revision are not recorded (D16). |
| R12 | Sampling parameters sent with a rollout must match the trainer's | multi-harness:798-807 | **Partial.** NeMo rows carry `temperature` and `parallel_tool_calls`; the importer drops them (D18). |
| R13 | Frameworks name the same concept differently; adapters need a translation map | `ART/hf-space-rl-environments-guide.md:139,166-171` | `I/canonical.py` for tools. **Missing for final answers** (D6) and for failure names (D10). |
| R14 | NeMo `/verify` returns 422 on unexpected trajectory shapes | env guide:528 | **Scored 0 today** (any non-2xx is a check crash → 0, D1). |
| R15 | Structured-outputs reward checks the schema only | hf-blog:127 | No "what the grader does not check" field on import (prior R9.17). |
| R16 | Gold must score 1.0, no-op 0.0, only targeted tests count | `ART/hf-repo2rlenv-how-not-to-build-coding-environments.md:10,58,72` | Not run on any import (§3, I1/I2). |
| R17 | Keep the gold patch and hidden tests out of the container until scoring | Repo2RLEnv:63, :77 | Harbor in place stages `/tests` after the agent. **Separate-verifier tasks break it** (D3). |
| R18 | Harness credentials reach the agent only; the verifier's key never reaches the agent | multi-harness:696-700 | **Broken in place for separate-verifier tasks:** NatureBench `JUDGE_API_KEY` is exported into the agent's container at grade time (D3). |
| R19 | Harbor default: an agent-written `reward.json`, a fake `python3`, a test link each scored 1.0 | `ART/pm-karotte-technical-deep-dive.md:247-248` | Harbor in place inherits Harbor's verifier; the World's `grader_paths` intercept covers part of it (`inplace.py:147`). Not tested on imports. |
| R20 | verifiers: the agent overwrote the grading script staged in `/tmp/vf-scripts` | Karotte deep dive:248 | Export `CrucibleTask.run_verifier` runs `bash /tests/test.sh` in the agent runtime (`I/verifiers/taskset.py:35`); no test. |
| R21 | Parity checks cover only what they exercise | `ART/prime-agent/prime-agent-rust.txt:90` | The M11 gate exercises only tool-graded toy tasks (0.8, D6). |
| R22 | Adapters drift; pin versions | env guide:526; pi-verifiers-v1:18,215 | `.envs/*` pins exist; no per-import record of framework versions (D16). |
| R23 | Imported sets should carry their validation evidence (gold-patch subsets, 8 consistent reruns) | hf-blog:170,172; qingkeai:869-870 | No field for publisher evidence (prior G21). |

## 2. Dataset coverage matrix (35 entries)

**On-disk format.** Locally every dataset is a probe snapshot: `README.md`, `meta.json` (HF API metadata, **no commit
sha**), `tree.txt`, often `first_rows.json` (HF datasets-server rows, max 3 per split) and 1–2 `sample_task/` dirs.
The upstream format below comes from `tree.txt`. No parquet file is local, so row sets were read from `first_rows.json`.

"Importer today": the M11 code that loads it and how far it gets. "Grades" means a reward was produced offline in this
sweep or a prior one. Effort: S ≤ 1 day, M a few days, L a week or more.

### 2.1 Harbor and OpenEnv task directories (18)

| Dataset | Format (upstream / local) | Grader carried | Importer today | Gap | Effort |
|---|---|---|---|---|---|
| ai-and/termgrade-environments | Harbor dirs, no version key (428 `task.toml`); `files/tasks.jsonl` (`oracle_verified`); `task_source` split has the same task as columns (`dockerfile`, `test_sh`, `solution_sh`, `test_outputs_py`) | pytest+CTRF → `reward.txt`; `solve.sh`; offline | `task_io` + in-place construct [ran Q1]. Running needs `FROM dsbase-venv312:local`, built from `base_image/`; nothing builds it | base-image prebuild step; a row → dir materializer for `task_source` | S |
| AweAI-Team/BeyondSWE-harbor | Harbor `version="1.0"` dirs; compose `network_mode: host`; gold patch in `tests/instance.json` | F2P+P2P pytest → `reward.txt`; heredoc `solve.sh` (D1 no-op) | parses, in-place constructs [ran Q1]; images on `hub.byted.org` cannot resolve | image remap; I1 dynamic oracle check; negative fixture | S |
| FineEnvs/data-agent-harbor-eval | Harbor 1.2 dirs; `registry.json` 144 (no commit pin, Q8) | `grader.py` → **multi-key `reward.json`** (`correctness`, `submission`, `tool_efficiency`) or `reward.txt`; `[verifier.env] EXPECTED_ANSWER`; no oracle | in place: masks every grade as `ambiguous_reward` unless `reward_key="correctness"`, an unmapped kind (D10). Patch mode: **TypeError** and drops `EXPECTED_ANSWER` [ran Q2] | reward-key rule (D13), verifier env (D9), setup needs HF network + `HF_TOKEN` | S |
| FineEnvs/HF_ML_Tasksmith | Harbor 1.3 dirs; `registry.json` 50, all commit-pinned; `manifest.json` (quality counts) | separate verifier image (`tests/Dockerfile`); `grade.py`; oracle; `no-network` | parses; in-place constructs and **would grade in the agent container** [ran Q1] (D3) | separate-verifier execution | M |
| FineEnvs/MiMo-V2.6-RL-harbor-code | Harbor 1.4; `registry.json` 2,698 (no pin); `manifest.json` sha256 | `test_command.sh` after `git apply test.patch`; **no reward on testbed failure**; no oracle | in place: a missing reward masks (`RewardFileNotFoundError` → `verifier_error`, `C/runtime/infra.py:32`) | oracle from the source `code.parquet` patch; non-root agent (prior N6) | M |
| FineEnvs/openai-math | Harbor 1.4; `registry.json` 369 | separate no-network Lean verifier; no reward on error; oracle fetches proofs | parses; in place would grade in the agent container (D3); Harbor 0.23 < 0.24 floor | separate path; oracle egress | L |
| FineEnvs/repo2rlenv-equivalence-tests | Harbor `version="1.0"`; `registry.json` 100, pinned | pytest → `reward.txt`; `patch.diff` oracle (prior N1: inverted reward) | parses, in-place constructs [ran Q1] | I2 reference-alias probe; negative fixture | S |
| FineEnvs/repo2rlenv-swe-flow | Harbor 1.3; `registry.json` 100, pinned; `manifest.json` | separate verifier; `grade.py` | as Tasksmith (D3) | separate path; carry `quality_status` | M |
| FineEnvs/SmolDataEnvs-harbor-train | Harbor 1.2; 5,000 | `grader.py` → `reward.txt`; `[verifier.env]`; no oracle | in place constructs; patch mode drops `EXPECTED_ANSWER` (D9) | as data-agent | S |
| FrontisAI/NatureBench-Harbor | 95 `.tar.gz` per task + `manifest.jsonl`; `dataset.toml` | separate verifier, GPU compose, LLM judge (`JUDGE_API_KEY`), continuous reward; no oracle | sample dir parses; Harbor's `DatasetManifest` reads `dataset.toml` as **0 tasks** [ran Q8]; in place would export the judge key into the agent container (D3) | archive fetch; compose sidecars; judge pin (I7) | L |
| harborframework/harbor-mix | gated; `tree.txt` + README only | 68 programmatic, 27 judge, 5 Modal | none (no local task) | gating is a user decision | – |
| harborframework/terminal-bench-2.1 | Harbor 1.1; `registry.json` 89 (no pin) | `uvx pytest` → `reward.txt`; **grade-time network** (installs curl and uv) | **grades in place** (oracle 1.0, no-op 0.0: `docs/modules/reports/m13_tb21_inplace.json`). Patch mode runs `--network none`, so `uvx` fails and `test.sh` writes 0 (D9) [read] | I3 grade-offline check | S |
| harborframework/terminal-bench-science | Harbor `schema_version="1.0"`; `registry.json` 70; nested task paths | separate no-egress verifier with ground truth baked in | parses; in place takes the fail-open path (prior N4, D3) | separate path | M |
| IntelligenceLab/Long-Horizon-Terminal-Bench | Harbor 1.1; 46 tasks, **no tests or solutions**; `continue_until_timeout` | none | `load_harbor_task` lists the key, refuses nothing; in place raises `FileNotFoundError` [ran Q1] (D12) | refuse: no verifier | S |
| Lego-X/Lego-RL-2699 | Harbor 1.4 dirs + `task_instances` JSON; `openswe_source` rows (SWE-bench shape) | stdout marker `OPENSWE_EXIT_CODE`; heredoc gold patch | parses; repo not shipped, cannot build | restore repos; PI `PIRE/swe/openswe` taskset is the row path | M |
| Lego-X/Terminal-Lego-15k | Harbor `version="1.0"`, `memory="1G"`; `first_rows` is `text` only | uvx pytest → `reward.txt`; `solve.sh` | parses, in-place constructs [ran Q1] | memory strings; grade-time network | S |
| PrimeIntellect/Terminal-Lego-15k | same + `docker_image="terminal-lego/<id>:latest"`; `prime-data-excluded-tasks.jsonl` (1,224 labels) | same | parses; in place would pull a private `:latest` image; no option to build from the Dockerfile instead (D19) | Dockerfile override; diff vs exclusion labels (I10) | S |
| LiteFold/RetroEnv | OpenEnv; JSONL splits (`tasks/train`: `target_smiles`, `constraints`, `max_depth`); `audit.json` (`passed: false`) | deterministic route verifier, partial credit (env code not local) | none; `I/openenv/import_env.OpenEnvTask` could host it if the package were installed | env package; strip `tasks-private` | M |

### 2.2 Row and JSONL datasets (15)

| Dataset | Format (upstream / local) | Grader carried | Importer today | Gap | Effort |
|---|---|---|---|---|---|
| PrimeIntellect/Multi-SWE-bench | 4 parquet; Multi-SWE schema (`fix_patch`, `test_patch`, `f2p/p2p/s2p/n2p_tests`) | test names + patches; images per language | **none** | row adapter via PI `PIRE/swe/multiswe` (I1/I2) | M |
| PrimeIntellect/Multi-SWE-RL-Verified | 3 parquet + `validation_*_p1/p2` columns + 2 exclusion JSONL | same + publisher gold runs | none | as above + I10 diff vs exclusions | M |
| PrimeIntellect/R2E-Gym-Subset-Verified | 9 parquet (`train`, `dropped`); `docker_image` `namanjain12/*:<sha>` (public) | `parsed_commit_content` gold, `expected_output_json`; PI taskset `solved()` | none. `v1_import` runs a verifiers env by **index** via `vf-eval` only; the PI taskset calls `load_dataset(<HF id>)` (network) | offline rows + `validate`/`apply_gold_patch` hooks (D16) | M |
| PrimeIntellect/Reverse-Text-RL | 1 parquet, `prompt` only | LCS reward in verifiers' `environments/reverse_text` (local, not installed in `.envs/verifiers`) | `v1_import` could run it by index once installed, but its `load_dataset` needs the network | local-rows option | S |
| PrimeIntellect/Scale-SWE-Verified | 2 parquet; `image_url` `aweaiteam/scaleswe:<id>`, `f2p_script`, `pre_commands`; image exclusion list | pytest F2P ∪ P2P | none. NeMo-Gym `scale_swe` server grades these rows, but `NemoGymApp` exposes only row function tools, not a sandbox shell | PI `PIRE/swe/scaleswe` taskset, or an exec App into NeMo's sandbox | M |
| PrimeIntellect/SWE-Lego-Real-Data-Verified | 4 parquet; `image_name` `jierun/...:latest`; **`messages`** (an OpenHands trajectory) | `eval_cmd`, `log_parser` | none | PI `PIRE/swe/swelego`; never render `messages` (I4) | M |
| PrimeIntellect/SWE-rebench-V2-Filtered-Verified | 1 parquet + 3 exclusion files; image rewritten to `prime/primeintellect/*` | `install_config{test_cmd, log_parser}` | none. NeMo-Gym `swe_rebench` grades the public `swerebenchv2/*` images | PI `PIRE/swe/swerebench_v2`; image remap | M |
| nvidia/…workplace_assistant | 2 JSONL; `responses_create_params` (27 tools, `temperature`, `parallel_tool_calls`), `ground_truth` | NG `workplace_assistant` state check | **grades** with `NemoGymTask` [ran Q3]: oracle 1.0 on 6/6; no-op 0.0 on 5/6, **1.0** on validation row 1 (D10 floor) | M9 adapter; sampling params (D18) | S |
| nvidia/…calendar_scheduling | 2 JSONL; multi-turn `input` (all 6 rows have assistant turns), no tools, `exp_cal_state` | NG `calendar` (final-answer graded; its own `except: reward=0`) | **grades** [ran Q3]: short oracle answer 1.0; **the same answer at 2,356 chars 0.0** (D2); turns flattened (D7); server death scored 0 (D1) | D1, D2, D7 | S |
| nvidia/Nemotron-RL-CFBench-v1 | 1 JSONL; `instructions`, `llm_judge`, `messages`; `agent_ref` `turing_vif_simple_agent` | NG `verifif` + LLM judge (default temperature 0.7) | loads as `NemoGymTask` [ran Q11]; `verifif` imports in `.envs/nemo`; grading needs a paid judge (not run) | I7 judge pin; D2; D7 (2/3 rows multi-turn) | S–M |
| nvidia/Nemotron-RL-SysBench-v1 | 1 JSONL; as CFBench, 3/3 rows multi-turn | as CFBench | as CFBench | as CFBench | S–M |
| nvidia/…instruction_following | 1 JSONL; `prompt`, `instruction_id_list`, `kwargs` | NG `instruction_following` | loads [ran Q11]; server deps missing in `.envs/nemo` (`defusedxml`, `verifiable_instructions`) | install deps; long answers hit D2 | S |
| nvidia/…structured_outputs | 2 JSONL; `schema_str`, `schema_type` | NG `structured_outputs` (schema only, D11 floor) | loads [ran Q11]; deps missing (`xmltodict`, `openapi_schema_validator`) | deps; I2 minimal-object floor | S |
| nvidia/Nemotron-RL-ReasoningGym-v1 | 1 JSONL; `question`, `answer`, `metadata` (difficulty knobs) | NG `reasoning_gym` | loads [ran Q11]; `reasoning_gym` package missing | deps; oracle from `answer` column | S |
| reasoning-core/procedural-pile | 100 parquet; `task`, `prompt`, `answer`, `metadata` (may hold a solver trace), `level` | `reasoning_core.score_answer` (not installed) | none; no framework | a plain answer-task adapter; never show `metadata` | S–M |

### 2.3 Index entries (2)

| Entry | Content | Interop use |
|---|---|---|
| `index.md` | 119 HF datasets tagged `rl-environment` (harbor 56, openenv 23, verifiers 22, nemo-gym 4, none 25); 8 blog datasets lack the tag | The tag list is the claim to test (R1). SmolDataEnvs and data-agent are tagged `openenv`; Crucible's OpenEnv importer cannot load them (it hosts `Environment` classes); only the Harbor path can. |
| `index.json` | the same rows as JSON | input for an I0 provenance record (licence, gating, tags) |

**Coverage totals.** Grades today: TB2.1, termgrade (with a local base image), workplace_assistant, calendar. Load
but cannot grade offline or correctly: 13 Harbor sets, 4 NeMo sets. Nothing loads: 7 verifiers row sets, RetroEnv,
procedural-pile, harbor-mix. **None can go through `accept_task`.**

## 3. M11 defects

Severity: **P0** = a silent wrong reward on real data; **P1** = a wrong reward or record on a path we ship, or a
blocked spec item; **P2** = robustness or provenance. Paths are under `C/`.

| # | Sev | Defect | Evidence | Probe | Fix |
|---|---|---|---|---|---|
| D1 | **P0** | **NeMo-Gym import scores infrastructure failures 0.** A server that dies before `/verify`, a `/verify` HTTP 500 and a reply without `reward` all give reward 0.0 with infra `ok`. `nemo_verify` raises (`raise_for_status`, `ReadError`) and the plan's generic branch records a candidate failure. A missing `reward` defaults to 0.0. A `mask_sample` reply masks, but as `check_infra`, so NeMo's own `failure_kind` (`judge_failed`) is lost. | `interop/nemo_gym/import_server.py:150-153`, `:243-253` (`:251` default 0.0); `verify/runtime/plan.py:313` | Q9 [ran]: (a) correct answer, server killed → 0.0 ok; (b) 500 → 0.0 ok; (c) no `reward` → 0.0 ok; (e) mask → `check_infra` | Wrap `/verify` in `try`: any `httpx` error, non-2xx (incl. NeMo's 422) or missing/non-finite `reward` → `InfraError` with kind `verifier_error`/`env_unreachable`. Carry NeMo's `failure_kind` into `infra.failure_kind`. |
| D2 | **P0** | **Final answers are cut to 2,000 characters before grading.** `trace_to_response` takes `e.text` of the `done` event, which is `Done.reason`. `LLMToolPolicy` cuts `reason` at 2,000 characters and keeps the full reply in `Done.answer` (recorded as `data["answer"]`). Every answer-graded NeMo server (calendar, instruction_following, structured_outputs, verifif, reasoning_gym) grades the cut. | `interop/nemo_gym/import_server.py:177-178`; `actors/policy.py:423`; `world/world.py:634` | Q3 [ran]: oracle schedule after a 2,300-char preamble → **0.0**; same schedule short → 1.0 | Use `e.data.get("answer") or e.text`. Add a regression test with a > 2,000-char answer. |
| D2b | P1 | ATIF export has the same cut: the `done` step's `message` is `e.text`. NatureBench's verifier reads ATIF as judge input. | `interop/harbor/atif.py:52-53` | [read] | as D2 |
| D3 | **P0** | **Separate-verifier tasks grade in the agent container** (prior G20, still open). `HarborInPlaceTask` builds Harbor's `Verifier` on the agent's environment for every task. The verifier image (ground truth, `grade.py` deps) is never built. TB-science `qsm` takes its fail-open path. NatureBench's `[verifier.env] JUDGE_API_KEY` is resolved from the host and set in the agent's container. `multi_step.py` already refuses this case. | `interop/harbor/inplace.py:95-130`, `:211-212`; contrast `interop/harbor/multi_step.py:71-73` | Q1 [ran]: 8/8 separate samples (Tasksmith, openai-math ×2, swe-flow ×2, NatureBench, TB-science ×2) construct | Refuse in the constructor (one line, as `multi_step.py:73`) now. Then add a separate-verifier grade: build `tests/Dockerfile`, copy `[[verifier.collect]]` / `[[artifacts]]` out of the agent container, run the verifier there. |
| D4 | P1 | **Score-gated imports report success on failure.** `NemoGymTask` and `OpenEnvTask` return `Verdict(True, {"score": r})` for any reward, so the score gate "passes". `outcome` is 1.0, `check/<gate>` is 1.0, and the flat `reward.json` says `outcome: 1.0` beside `reward: 0.0`. A1's gold check needs an outcome or score gate false at the start state, so **no NeMo or OpenEnv import can ever pass A1**. `credit/properties.py` R1 reads `outcome`. | `interop/nemo_gym/import_server.py:252-255`; `interop/openenv/import_env.py:593-596`; `verify/runtime/plan.py:251`; `accept/checks/core.py:39-42` | Q5 [ran]: wrong calendar answer → reward 0.0, outcome 1.0, gate pass. Q6 [ran]: OpenEnv terminal reward 0 → outcome 1.0 | Pass the gate only at success: `Verdict(r >= success_at)` (default: the reward range's top, 1.0), keep `score` and `normalized` for continuous rewards. Or declare an `outcome` gate plus a separate score channel. |
| D5 | P1 | **OpenEnv: an unscored terminal step is scored 0, a NaN is scored 0.** OpenEnv's contract: `reward=None` means no reward was produced. `episode_reward` returns `None` for it, which the check reads as "never terminated: scored 0". A NaN final reward clamps to 0.0 through `min(1, max(0, nan))` and passes the gate. | `interop/openenv/import_env.py:577-593` | Q6 [ran]: `NoneEnv` → 0.0 "never terminated"; `NanEnv` → 0.0, outcome 1.0 | `done and reward is None` → `InfraError("env_unscored")` (masked). Non-finite → mask. Keep N4 (no terminal step = scored 0). |
| D6 | P1 | **Cross-runtime replay drops the final answer.** `from_world_trace` keeps only `tool_call` and `message` events; `from_responses` keeps only `function_call` items; `replay` scripts no `Done`. A correct NeMo Responses transcript replays to 0.0. The M11 exit gate therefore cannot cover answer-graded tasks (NeMo, verifiers single-turn, MiMo `agent_output`). | `interop/replay.py:84`, `:115-124`, `:140-151` | Q4 [ran]: calendar oracle transcript → 0 canonical actions → reward 0.0 (native 1.0) | Add `Action(kind="final", text=...)`; parse the `done` answer and Responses `message` items; script `Done(reason=text[:2000], answer=text)`. Add a NeMo calendar case to `test_interop_gate.py`. |
| D7 | P1 | **NeMo inputs are flattened into one string.** The system message becomes `[system] …` user text and earlier assistant turns become plain text, so the policy sees a different prompt than NeMo's `simple_agent`, which sends `input` as messages. 11 of 12 sampled CFBench/SysBench/calendar rows have assistant turns. | `interop/nemo_gym/import_server.py:196-206` | Q3, Q11 [ran]: calendar val#2 has 20 messages (10 assistant); instruction is one string containing assistant text | Carry `input` as a message prefix (World instructions as a list of role messages, or `Observation.history`). The policy renders it with its chat template. |
| D8 | P1 | **Patch-mode reward parsing is not Harbor's.** A multi-key `reward.json` without `reward` raises `TypeError` (data-agent). A list raises `AttributeError`. `{"reward": NaN}` is scored NaN with status `ok`. `reward.txt` takes the last line, where Harbor parses the whole file. | `interop/harbor/verify_dir.py:125-132` | Q2 [ran] | Reuse `inplace.scalar_reward` and Harbor's own reward parser. Non-finite or unparseable → masked `verifier_error`. |
| D9 | P1 | **Patch mode does not follow Harbor's verifier phase.** Network comes from `[verifier].network_mode` only, default `no-network`; Harbor resolves `[environment]` / `allow_internet` too. 17 of 28 samples grade with no network where Harbor grades public. TB2.1's `test.sh` then cannot install uv and writes 0 for the oracle. `[verifier].env` is not passed (data-agent/SmolDataEnvs `EXPECTED_ANSWER` → the grader compares against an empty gold). `[verifier].user` is ignored. | `interop/harbor/verify_dir.py:59-66`, `:101-111`; TB2.1 `tests/test.sh` (curl, uv, then `echo 0`) | Q1 [ran] network diff 17/28; Q2 [ran] no `-e` flags | Use `resolve_trial_network_plan(...).verifier_phase`; pass `-e` for the verifier env (resolved from the host only for the verifier) and `--user`. Or refuse non-Crucible tasks, as the in-place docstring says the patch path is wrong for them. |
| D10 | P1 | **M11's own failure names are unmapped.** `ambiguous_reward` and `no_reward` (in place), `env_unreachable`, `env_session_lost`, `env_admission_timeout` (NeMo client), `mask_sample`, `InfraError` all normalize to `agent_run_error` / `unmapped:*`, blame unknown. Since the M10 audit these count toward the 5% unknown-blame stop rule, so a data-agent import stops the run on a task-format fact. | `interop/harbor/inplace.py:85-92`; `interop/nemo_gym/client.py:58-82`; `runtime/infra.py:27-44` | taxonomy probe [ran]: 7 of 7 M11 names → `unmapped` | Add them to `NORMALIZE`: `ambiguous_reward`, `no_reward` → `verifier_error` (blame task); `env_*` → infra; `mask_sample` → keep NeMo's `failure_kind`. Add a test that every literal kind string under `interop/` normalizes. |
| D11 | P1 | **verifiers import loses truncation and raises on bad scores.** A trace stopped by `max_turns` (verifiers' `is_truncated`) becomes `truncated=False`; turns carry no `finish`, so `train/runtime.py:627` cannot see a length cut either. A `None` rubric score raises `TypeError`; NaN raises a pydantic error instead of masking. | `interop/verifiers/v1_import.py:71-80`; verifiers `v1/trace.py:621-633` | Q7 [ran] | `truncated = stop in {max_turns, max_*_tokens, compaction_failed}` or the last call's `finish_reason == "length"`; set `Turn.finish`. Non-finite or `None` score → `mask_sample=True`, kind `verifier_error`. |
| D12 | P1 | **Grading keys are not refused** (prior G4). `continue_until_timeout` loads; in place does not even call `load_harbor_task`. | `interop/harbor/task_io.py:105`; `interop/harbor/inplace.py:101` | Q1 [ran] | Closed list per schema version; `UnsupportedTaskKey` for grading-relevant keys; call it from `HarborInPlaceTask.__init__`. |
| D13 | P2 | **The caller cannot choose a key when `reward` exists**, and there is no priority list. OpenEnv's rule allows `reward_key="correctness,reward"`. | `interop/harbor/inplace.py:86-91` | [read] | `reward_key` first, as a comma list; then `reward`; then the single key. |
| D14 | P2 | **MiMo has no RunTask.** Import gives a `WorldSpec`; grading is `mimo.verify.grade(source, run_root, agent_output=...)`, and no runtime code passes the final answer. Only the gate test drives it. `verify.py` and its `check_code` run in Crucible's own process with the host environment (MiMo runs them in a sidecar). | `interop/mimo/importer.py:57-98`; `interop/mimo/verify.py:49-75` | [read] | A `MimoTask(RunTask)` with `grade` taking the `done` answer (D2 rule); run `verify.py` in a subprocess with a scrubbed env. |
| D15 | P2 | `run_vf_eval` timeout cancels `communicate()` but never kills `vf-eval`; the process and its sandboxes keep running. | `interop/verifiers/v1_import.py:151-153` | [read] | `p.kill()` in a `finally`, process group. |
| D16 | P2 | **verifiers import has no provenance or local rows.** Tasks are a prompt index into `vf-eval`; no dataset revision, env version or verifiers version is recorded. PI tasksets call `load_dataset(<HF id>)`, so an offline or pinned import is impossible. | `interop/verifiers/v1_import.py:165-181`; `PIRE/swe/r2e_gym/r2e_gym/taskset.py:316-318` | [read] | Record `env_id@version`, `verifiers@version`, dataset `repo@sha`; pass a local parquet through the taskset config. |
| D17 | P2 | **No dataset-level loader.** No code reads `registry.json`, `dataset.toml`, `manifest.json(l)`, parquet or `first_rows`. Harbor's own models parse all 8 local `registry.json` files (4 without commit pins). NatureBench's `dataset.toml` parses to 0 tasks because its tasks are archives in `manifest.jsonl`. | `grep` over `I/` (only EOG's `load_dataset`) | Q8 [ran] | I0 `source.py` (§4.2). |
| D18 | P2 | NeMo row sampling parameters (`temperature`, `parallel_tool_calls`) are dropped. | `interop/nemo_gym/import_server.py:192-215` | Q3 [ran] | Put them on the task (`sampling`), recorded per rollout. |
| D19 | P2 | No "build from the Dockerfile" override when `docker_image` names a private or `:latest` image (PI Terminal-Lego). | `interop/harbor/inplace.py:109-110` | [read] | `prefer_dockerfile=True` → drop `docker_image` from the config passed to Harbor's factory. |
| D20 | P2 | Spec drift: no `interop/verl` although ARCHITECTURE lists it; the exit gate has no task02 case. | `ls I/`; `tests/unit/test_interop_gate.py:33` | [read] | Update ARCHITECTURE status, or build both. |

**Checked and not defects:** bad tool arguments to a NeMo tool become a policy-visible error (`ok=false`, Q3). A NeMo
server that dies mid-episode masks the episode as `crucible:tool_crashed` (Q10). MiMo's `score or 0.0` matches MiMo's
own `run_verify.py:125`. A policy error ending the episode is masked by M10 through `policy.failure_kind`
(`actors/policy.py:378-381`), not by the grade.

## 4. Picks and a minimal design for I0–I10

### 4.1 Picks

| # | Mode | Take | Source | Why | Effort |
|---|---|---|---|---|---|
| 1 | **DEPEND** | Harbor's `DatasetSpec` / `RegistryTaskId` and `DatasetManifest` for dataset provenance and task lists | Harbor `models/registry.py:46-66`, `models/dataset/manifest.py:161-207` | Parses every local `registry.json` today (Q8); gives `git_url` + `git_commit_id` per task, which is I0's pin | S |
| 2 | **DEPEND** | PI's SWE tasksets (`r2e_gym`, `multiswe`, `scaleswe`, `swelego`, `swerebench_v2`, `openswe`) in `.envs/verifiers`, on verifiers' Docker runtime, for row I1/I2: `validate()` = oracle, `apply_gold_patch()`, `solved()` = grade | `PIRE/swe/*/`; `.../verifiers/v1/runtimes/docker` | Six of seven row sets have a taskset written by the publisher of the rows; reimplementing graders invites drift (research 05 §0.1 item 12) | M |
| 3 | **PATTERN** | NeMo-Gym's golden-patch aggregation: supported / flaky / broken / inconclusive, with infra (no verdict) kept apart from flaky | `NG/resources_servers/swe_rebench/README.md:58-75`, `aggregate_golden_patch.py` | Exactly I3's verdict vocabulary; avoids PI's `cargo_missing` confusion (prior 0.5) | S |
| 4 | **PATTERN** | OpenEnv's reward rules: only key, else `reward`, else the caller's priority list; never combine; `None` is unscored, never 0 | multi-harness:704-710, :765, :777 | Fixes D5, D13 with one rule shared by all importers | S |
| 5 | **DEPEND** (optional) | NeMo-Gym `swe_rebench` / `scale_swe` servers as a second grader for cross-checking PI's verdicts | `NG/resources_servers/{swe_rebench,scale_swe}` | Two independent graders on one row expose grader drift; needs an exec App into NeMo's sandbox | M |
| 6 | **PATTERN** | multi-harness's ATIF cross-check as an import-time check: a replayed import's turn count and final answer must match the source transcript | multi-harness:712-720 | Would have caught D2 and D6 | S |

Not picked: Karotte (no code to take; its attack list is already an M9 item), OpenForgeRL's text-only samples
(multi-harness:234-237: token ids are required by R8).

### 4.2 Minimal design: `interop/imports/`

The layer rule stands (ARCHITECTURE.md:624): adapters live in `interop` because they import frameworks; M9 stays
framework-free. An adapter **wraps an existing RunTask** and adds the capabilities `accept_task` reads (the surface of
`accept/registry/fixtures.FakeTask`: `grade(solution, run) -> (GradeReport, fingerprint)`, `noop()`, `negatives`,
`solution_of`, `start_fingerprint`, `agent_visible`, `hidden_strings`, `checks` with `gate_kind`, `key_inputs`,
`grade_env`). No parallel checker.

```
interop/imports/
  source.py     I0  locate + pin: registry.json / dataset.toml (Harbor models), parquet / JSONL rows, HF meta;
                    record repo@sha, licence, gating, tags, canary GUIDs, framework versions -> ImportSource
  lint.py       I0, I4, I5 static: refuse grading keys (D12), unresolvable or :latest images, `|| true` on checkout,
                    host secrets in [environment.env], stdout-marker grading, reward inputs under agent-writable paths
  floors.py     I2 per kind: Harbor no-op / empty patch; NeMo empty action list, empty answer, minimal schema object;
                    row sets empty patch -> `negatives` with kind="floor", origin="import" (prior pick 3.1#3)
  harbor.py     AcceptTask over HarborInPlaceTask / HarborMultiStepTask; oracle = OracleAgent; fingerprint = a digest
                    of the task workdir; separate-verifier tasks refused until D3's separate grade exists
  vrows.py      AcceptTask over a PI taskset run in .envs/verifiers through a JSON-RPC subprocess (the NemoGymServer
                    shape): setup / apply_gold_patch / solved / validate per row; fingerprint = `git diff` hash
  nemo.py       AcceptTask over NemoGymTask; oracle = ground_truth replay, or the row's `answer` column; fingerprint =
                    the App journal digest
  openenv.py, mimo.py, eog.py   thin: oracle_actions() where the importer has one
  report.py     I10 import_report.json: verdict counts, floor rate, flake rate by concurrency, infra rate, leak
                    findings, diff vs the publisher's exclusion files (Terminal-Lego, Multi-SWE, Scale-SWE, R2E)
  cli           `crucible import <source> --kind harbor|vrows|nemo|openenv|mimo|eog`
```

| Item | What it checks | Where | Reuses |
|---|---|---|---|
| I0 Provenance and shape | pin, licence, gating, schema, grading keys, images | `source.py`, `lint.py` | Harbor `DatasetSpec`, `TaskConfig`; `task_io` |
| I1 Oracle in the declared environment | oracle ≥ 3 runs; the oracle changes state | adapter `oracle` | M9 A1 |
| I2 Floors | no-op, empty, constant, minimal object; floor rate per dataset | `floors.py` | M9 A2 (`accept/checks/core.py:84`) |
| I3 Flakiness | reruns at two concurrency levels; supported / flaky / broken / inconclusive | adapter + `accept/policy.py` | M9 A4; pick 3 |
| I4 Leak audit | agent-visible surfaces vs hidden strings (gold hunks, `messages`, `reference_*`) | adapter `agent_visible`, `hidden_strings` | M9 A5 |
| I5 Isolation and tamper | separate verifier honoured; grader owns the reward; no host secrets in the agent phase | `lint.py`; D3 | M9 A6/A13 |
| I6 Instruction ↔ test coverage | stated vs graded properties | adapter `properties` | M9 A7 |
| I7 Grader determinism | judge pinned, temperature 0, ≥ 3 agreeing reruns, else soft channel | `nemo.py`, `harbor.py` | M9 A4/A12 |
| I8 Calibration | publisher difficulty as prior only | `report.py` | M9 A10 |
| I9 Contamination | canaries, temporal cut, near-duplicates vs our eval split | `lint.py`, `report.py` | `bench/contamination.py` |
| I10 Dataset report | `import_report.json` + exclusion-list diff | `report.py` | – |

**Order.** Fix D1–D6 and D10 first: today every NeMo and OpenEnv import would fail A1 (D4), answer tasks grade cut
text (D2), and infra noise scores 0 (D1, D5). Then `source.py` + `harbor.py` + `nemo.py` on the five-set test bed
(Terminal-Lego paired, TB2.1, termgrade, workplace_assistant, calendar), then `vrows.py` on R2E-Gym. A publisher
evidence field on `TaskCard` (prior G21) is needed before `report.py` can carry `oracle_verified` or exclusion
reasons.

## 5. Probe index

Scripts and outputs are in `SP/`. Run from `crucible/` with `.venv/bin/python` unless noted. NeMo servers ran from
`crucible/.envs/nemo` on 127.0.0.1. Nothing touched the network, a GPU, Docker or a paid API.

| Probe | What it does | Result |
|---|---|---|
| Q1 `q1_harbor_import.py` → `q1.out` | All 30 local Harbor sample dirs through `load_harbor_task`, `HarborInPlaceTask(...)` and `HarborDirTask(...)`; compares patch-mode network with Harbor's verifier phase | 30/30 load; 28 construct in place (LHTB ×2 raise `FileNotFoundError`); **8/8 separate-verifier tasks accepted**; ignored keys `version`, `environment.memory/storage`, `agent.continue_until_timeout`; **17/28 network mismatches**; 5 tasks' `[verifier].env` dropped by patch mode |
| Q2 `q2_verify_dir_rewards.py` → `q2.out` | `HarborDirTask.grade_patch` on the data-agent sample with a fake `docker` that writes the grader's reward file | multi-key → `TypeError`; list → `AttributeError`; NaN → scored NaN `ok`; no `-e` or `--user` flags; `inplace.scalar_reward` on the same dict → `ambiguous_reward` |
| Q3 `q3_nemo_rows.py` → `q3.out` | NeMo `calendar` and `workplace_assistant` servers; HF sample rows through `NemoGymTask` | calendar: short oracle 1.0, **2,356-char oracle 0.0**, no answer 0.0; 20-message input flattened. workplace: oracle 1.0 ×6, no-op 0.0 ×5, **1.0** on val#1; bad args → policy-visible error; row `temperature` not carried |
| Q4 `q4_replay_answer.py` → `q4.out` | `from_responses` + `replay` on a correct calendar Responses transcript | 0 canonical actions; reward **0.0** (native 1.0) |
| Q5 `q5_nemo_rewardjson.py` → `q5.out` | `reward_json()` of a wrong and a right calendar answer | wrong: reward 0.0, **outcome 1.0, gate pass, `check/nemo_reward` 1.0** |
| Q6 `q6_openenv_reward.py` → `q6.out` (in `.envs/openenv`) | `OpenEnvTask` on three one-step environments ending with reward `None`, NaN, 0 | `None` → 0.0 "never terminated"; NaN → 0.0, outcome 1.0; 0 → outcome 1.0 |
| Q7 `q7_vf_trace.py` → `q7.out` | `trace_to_record` on minimal v1 traces | NaN → pydantic error; `None` → `TypeError`; empty rubric → masked; error → masked `SandboxError`; `max_turns` → `truncated=False` |
| Q8 `q8_dataset_manifests.py` → `q8.out` | All `registry.json`, `dataset.toml`, `manifest.json` through Harbor's models | 8/8 registries parse (Tasksmith, eqv, swe-flow pinned; 4 unpinned); NatureBench `dataset.toml` → 0 tasks |
| Q9 `q9_nemo_infra.py` → `q9.out` | Grade-time failures: server killed, `/verify` 500, no `reward`, `mask_sample` | **0.0 ok, 0.0 ok, 0.0 ok**; mask → `check_infra` (NeMo `judge_failed` lost) |
| Q10 `q10_nemo_midepisode.py` → `q10.out` | Server killed before the first tool call | episode masked `crucible:tool_crashed` (correct) |
| Q11 `q11_nemo_rows_offline.py` → `q11.out` | Every NeMo sample row → `NemoGymTask` without a server; server deps in `.envs/nemo` | 7 sets load; `verifif`, `calendar`, `workplace_assistant` import; `instruction_following`, `structured_outputs`, `reasoning_gym` lack deps; 11 rows with assistant turns flattened |
| taxonomy (inline, `.venv`) | `runtime.infra.normalize` on M11's own kind strings | `ambiguous_reward`, `no_reward`, `env_unreachable`, `env_session_lost`, `env_admission_timeout`, `mask_sample`, `InfraError` → all `unmapped` |
| articles helper | One read of every article for interop requirements (92 items) | Basis of §1 |
