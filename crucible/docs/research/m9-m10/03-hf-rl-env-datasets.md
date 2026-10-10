# M9/M10 research 03: Hugging Face `rl-environment` datasets, and what importing them takes

Date: 2026-10-09. Scope: the datasets named in the HF blog "Welcome RL Environments to the Hub"
(`RL-Environments/07-articles/hf-blog-rl-environments.md`), plus the most-downloaded datasets under the `rl-environment` tag.
The question is how these datasets are validated (M9 `accept`) and how they are run (M10 `runtime`).
Method: HF Hub HTTP API and datasets-server only (no GitHub API). For each dataset I fetched the card, the file tree, the first
rows, and one or two task directories. For four datasets I scanned every task over HTTP range requests (BeyondSWE), or pulled the
published validation logs (Multi-SWE-RL, Scale-SWE, R2E, Terminal-Lego) and the full JSONL (workplace_assistant). About 80 MB in
total went to scratch, and 21 MB is kept on disk. Every claim marked ✓ was reproduced from the downloaded files. Everything else is quoted from the card.

Local artifacts:
- `RL-Environments/08-hf-rl-env-datasets/index.{json,md}`: all 119 tagged datasets (downloads, likes, frameworks, size, license, last modified).
- `RL-Environments/08-hf-rl-env-datasets/<org>__<name>/`: `README.md`, `meta.json` (card data, tree stats, splits, sizes), `tree.txt` (first 3,000 files), `first_rows.json`, `files/` (small root files: registries, manifests, validation logs), `sample_task/` (1–2 task dirs).

## 0. Landscape (✓)

- **119 datasets** carry the tag as of 2026-10-09. Framework tags: harbor 56, openenv 23, verifiers 22, nemo-gym 4; 25 carry no framework tag. FineEnvs (41) and eltociear (20) publish most of them.
- **API quirk:** `?other=rl-environment` (the web URL in the blog) is ignored by `/api/datasets` and returns *every* dataset. `filter=rl-environment` works. An M10 discovery client must use `filter=`.
- **8 of the 12 blog datasets are not tagged yet.** Their tagging PRs are still open: BeyondSWE, NatureBench, harbor-mix, Lego-X/Terminal-Lego-15k, and all four nvidia datasets. The tag is therefore not a complete catalogue. PrimeIntellect re-hosts several of them under its own org, already tagged, and sometimes re-validated (Terminal-Lego).
- Top downloads: `harborframework/terminal-bench-2.1` (163k), `terminal-bench` (105k), `terminal-bench-science` (68k), `PrimeIntellect/Terminal-Lego-15k` (62k), `FineEnvs/HF_ML_Tasksmith` (42k).
- **Only four packaging shapes exist:**
  1. Harbor task directories, either in-repo or as tarballs.
  2. Verifiers "row tasksets": parquet rows plus a taskset package that lives in a GitHub repo, not in the dataset.
  3. NeMo Gym JSONL: `responses_create_params` plus verifier fields plus `agent_ref`, graded by a resources server that lives in the NeMo Gym repo.
  4. OpenEnv / plain data: RetroEnv, procedural-pile.

  Only shape 1 is self-contained (task, environment, grader and oracle in one folder). Shapes 2 and 3 carry data only; the grader is code outside the dataset.

## 1. Dataset table

Legend:
- **Oracle:** the reference solution shipped with each task (Y = all tasks).
- **Validation claim:** what the card says was done to "verify".
- **Runtime:** what you need to run it.
- Sizes are repo bytes.

| Dataset | Format | Grader | Validation claim | Oracle | Size / count | License | Runtime assumed |
|---|---|---|---|---|---|---|---|
| **AweAI-Team/BeyondSWE-harbor** (blog) | Harbor 1.0 dirs | F2P+P2P pytest via `run_tests.py`; Doc2Repo fractional | none on card | Y, but 450/500 are no-ops ✓ (D1) | 0.37 GB / 500 | CC-BY-4.0 | Harbor + Docker; base images on `hub.byted.org` (NXDOMAIN ✓, D2) |
| **FrontisAI/NatureBench-Harbor** (blog) | Harbor 1.4 dirs in `.tar.gz` (up to 82 GB each) | sidecar eval service (best-of-N on hidden test set) → separate verifier + LLM validity judge (`JUDGE_MODEL` gpt-5.5) reads the trajectory | "original protocol and validity judge unchanged" | **none** | **939 GB** / 90 | MIT + per-task third-party | Harbor schema 1.4 (separate verifier env, `[[verifier.collect]]`, artifacts, healthcheck); GPUs (RTX 3090/4090 healthcheck); local base images `naturebench-base:v3` (not on a registry) |
| **harborframework/harbor-mix** (blog) | Harbor dirs (flattened) | 68 programmatic, 27 LLM-judge, 5 Modal | difficulty filter (frontier fails ≥67%) → AI audit (instruction↔verifier) → human panel | 94/100 (GAIA2 has none) | 0.11 GB / 100 | CC-BY-4.0 | Daytona (95) + Modal (5); **gated (auto)**: card and tree only |
| **Lego-X/Terminal-Lego-15k** (blog) | Harbor 1.0 dirs | pytest `test_outputs.py` → reward 0/1 | "Docker round-trip verified" (build, run solve, tests > 0). Pass rate 41.8% of 36,846 candidates | Y | 15k dirs | Apache-2.0 | Harbor/Terminus-2; test.sh fetches uv over the internet at grade time |
| **PrimeIntellect/Terminal-Lego-15k** (tagged mirror) | same + `docker_image=terminal-lego/<id>:latest` (Prime registry) | same | **re-validated**: 1,224 / 15,049 excluded with a per-task reason ✓ (§3) | Y | 13,825 dirs | Apache-2.0 | Prime sandboxes / verifiers `harbor` taskset |
| **harborframework/terminal-bench-2.1** | Harbor 1.1 dirs + `registry.json` (git-repos dataset) | pytest + CTRF → reward.txt | upstream TB review; canary GUIDs | Y (89/89) | 58 MB / 89 | Apache-2.0 | Harbor ≥0.21, Docker; prebuilt `alexgshaw/*` images; `allow_internet=true` |
| **harborframework/terminal-bench-science** | Harbor dirs, tagged per release | mixed pytest / custom | candidate pool, "not curated yet" | Y (solve.sh/solve.py) | 1.1 GB / 70 | Apache-2.0 | Harbor; the card says "private by design", but the repo is **public** ✓ (D7) |
| **IntelligenceLab/Long-Horizon-Terminal-Bench** | Harbor dirs **without tests/ or solution/** | hidden "rebuild-from-artifact" verifiers (not published) | — | none public | 1.2 GB / 46 | Apache-2.0 | **patched Harbor**: `continue_until_timeout` (30/46) re-prompts with verifier feedback |
| **FineEnvs/HF_ML_Tasksmith** | Harbor 1.3 dirs + tarball | trusted `grade.py`: validates the submission contract (regular files, no symlinks, hashed immutables), then runs pytest as unprivileged | "verified" = baseline + oracle + semantic probes + blind Sonnet rollout, with evidence hashes bound to `bundle_hash` | Y | 50 (39 CPU, 6×1 GPU, 5×2 GPU) | mixed | Harbor + Daytona; `no-network` for agent and verifier; separate verifier image; `artifacts` allowlist |
| **FineEnvs/repo2rlenv-swe-flow** | Harbor 1.3 | retained repo tests | quality labels: 98 unverified, 2 needs_repair | Y | 100 | other | Daytona |
| **FineEnvs/repo2rlenv-equivalence-tests** | Harbor 1.0 | pytest of `f` vs `reference_f` on ≥5 inputs | "oracle must score 1.000" | Y (`patch.diff`) | 100 | Apache-2.0 | Docker; Dockerfile clones GitHub, with `|| true` on every step |
| **FineEnvs/MiMo-V2.6-RL-harbor-code** (+cyber/general/terminal/webdev/music) | Harbor 1.4 dirs | reset test files → apply hidden `test.patch` → `test_command.sh`; a testbed failure writes **no reward** | conversion fidelity only; manifest has a sha256 for every task dir | **none** | 2,698 | Apache-2.0 | Harbor + HF Sandbox/Daytona; image pinned by **digest**; setup via healthcheck; `/etc/hosts` leak blocklist; unprivileged agent |
| **FineEnvs/SmolDataEnvs-harbor-train** (+ data-agent eval/test) | Harbor 1.2 dirs | deterministic short-answer grader (exact → numeric → list → math-verify) | "verified": strong agent models reproduced the gold answer | **none** (gold answer only) | 5,000 (+144 eval) | MIT | Harbor/OpenEnv/E2B; data pulled from an HF bucket at start with the user's `HF_TOKEN` passed into the agent env (D9) |
| **FineEnvs/openai-math** | Harbor dirs | Lean Comparator in a fresh sandbox: exact statement, standard axioms only | 369/405 official proofs verified end to end; 36 listed with reasons | Y, but the oracle **fetches proofs from GitHub at run time** | 369 | Apache-2.0 | Harbor ≥0.24 + Daytona; ~9 GB Lean/Mathlib snapshot |
| **Lego-X/Lego-RL-2699** | Harbor dirs + OpenSWE rows | `test_patch` + `eval_body.sh` → `OPENSWE_EXIT_CODE` marker parsed from stdout | selected by policy pass rate ≈1–3 of 4 | Y (gold patch) | 2,699 | AGPL-3.0 | `environment/repo/` **not shipped** (restore script); unpinned conda deps |
| **ai-and/termgrade-environments** | Harbor dirs + `task_source.jsonl` | pytest | `oracle_verified` per task (offline, `--network none`, built image); overlap vs TB2.1 and near-dup check; pass rates for 6 models | Y (1,004/1,004 verified ✓) | 1,004 | Apache-2.0 | local base image `dsbase-venv312:local` |
| **PrimeIntellect/Multi-SWE-RL-Verified** (blog) | verifiers rows (parquet) | `multiswe_v1` taskset: gold patch + F2P/P2P | 2× gold passes, no-edit filter, low-concurrency retest; per-row logs ✓ | gold `fix_patch` | 1.0 GB / 2,232 of 4,703 | CC0 + ByteDance notice | Prime sandboxes; images resolved by taskset code |
| **PrimeIntellect/R2E-Gym-Subset-Verified** (blog) | verifiers rows | apply gold, run `/testbed/run_tests.sh`, compare to `expected_output_json` | 1 pass @ concurrency 200 + 10× retry of failures; drop iff 0/10; **flaky rows kept** | gold in `parsed_commit_content` | 0.94 GB / 4,522 + 56 dropped | none declared | `docker_image` column (`namanjain12/*`, public ✓) |
| **PrimeIntellect/Scale-SWE-Verified** (blog) | verifiers rows (AweAI schema) | F2P script + P2P list | gold failure (2,061), no-op pass (15), stable infra failure (11), bad images (892) removed; logs ✓ | gold `patch` | 0.22 GB / 17,202 of 20,181 | CC-BY-4.0 | `image_url` `aweaiteam/scaleswe:*` (public ✓) |
| **PrimeIntellect/SWE-Lego-Real-Data-Verified** | verifiers rows | F2P/P2P PASSED | 10× retry; drop iff 0/10; flaky kept | gold + **full agent trajectories** (`messages`) | 1.0 GB / 4,323 + 109 dropped | Apache-2.0 | Prime |
| **PrimeIntellect/SWE-rebench-V2-Filtered-Verified** | verifiers rows | F2P/P2P | LLM-judge metadata filters + **leak hygiene** (external URLs, `#1234` refs) + 2 gold passes + no-edit + RL always-fail audit | gold | 0.24 GB / 6,272 of 32,079 | CC-BY-4.0 | images rewritten to `prime/primeintellect/*` |
| **PrimeIntellect/Multi-SWE-bench** | verifiers rows | F2P/P2P | none (eval re-upload) | gold | 2,132 | CC0 + notice | Prime |
| **PrimeIntellect/Reverse-Text-RL** (blog) | verifiers rows (`prompt` only) | in env code (`reverse_text`; the answer is computed, not stored) | — | n/a | 64 KB / 1,000 | MIT | verifiers; prime-rl CI smoke |
| **nvidia/Nemotron-RL-agent-workplace_assistant** (blog) | NeMo JSONL | replay predicted and ground-truth tool calls on fresh DBs; compare all 5 tables (✓ `NG/resources_servers/workplace_assistant/utils.py:133`) | synthetic; none | GT action list | 30 MB / 1,255 + 545 | CC-BY-4.0 | NeMo Gym resources server (in process, no container) |
| **nvidia/…-structured_outputs** (blog) | NeMo JSONL | JSON-schema validation only (no content check; the blog says so) | synthetic | n/a | 96 MB / 9,949 | CC-BY-4.0 | NeMo Gym |
| **nvidia/Nemotron-RL-CFBench-v1** / **SysBench-v1** (blog) | NeMo JSONL (VerifIF) | IFEval-style programmatic checks + `llm_judge` rubric questions | SDG (Nano-30B, Qwen3-235B), GPT-5 filtering | n/a | 50 MB each | CC-BY-4.0 | NeMo Gym `turing_vif` + a judge model |
| nvidia/…-instruction_following, calendar_scheduling, ReasoningGym-v1 | NeMo JSONL | IF checkers / `exp_cal_state` constraint check / reasoning-gym scorers | automated | n/a / answer | 31–78 MB | ODC-BY / CC-BY | NeMo Gym |
| **LiteFold/RetroEnv** | OpenEnv; tasks-public/private parquet | deterministic step check against a frozen reaction library (no hidden answer) + route-similarity bonus | `audit.json`: leakage, adversarial probes, solvability (witness route per task) | witness routes | 0.21 GB / 78.5k tasks | CC-BY-4.0 | OpenEnv |
| reasoning-core/procedural-pile | plain rows (prompt, answer, metadata) | solver-computed answers; scorer per task | "audited" generators | answer | 13.6 GB / 24.8M | CC-BY-4.0 | built as SFT data; tagged openenv/verifiers |

## 2. Per-format notes

### 2.1 Harbor task directories (≈56 tagged repos; the only self-contained shape)

Layout: `task.toml`, `instruction.md`, `environment/` (Dockerfile + build context), `tests/test.sh` (writes `/logs/verifier/reward.txt` or `reward.json`), and optional `solution/solve.sh`.

**Schema drift.** The datasets use `schema_version` **1.0, 1.1, 1.2, 1.3 and 1.4** ✓, and features land without a version gate:
- 1.0 (BeyondSWE, Terminal-Lego): `allow_internet`, `memory_mb`, and a free `[metadata]`.
- 1.1 (TB2.1): `[task]` block, `docker_image`, `mcp_servers`, `[verifier.env]`, `[solution.env]`.
- 1.2 (SmolDataEnvs): `[environment.healthcheck]` used as a **pre-agent setup hook** (data pull).
- 1.3 (Tasksmith): `network_mode = "no-network"` per phase, `user = "learner"|"root"`, `verifier.environment_mode = "separate"`, and `[[artifacts]]` (the allowlist of files copied from agent to verifier).
- 1.4 (NatureBench, MiMo): `gpus`, `gpu_types`, multi-service `docker-compose`, `[[verifier.collect]]` (run a command in a sidecar before grading), and `network_mode = "public"|"allowlist"` with `allowed_hosts` (Lego-RL).

LHTB adds a non-standard `continue_until_timeout` that stock Harbor ignores. An importer must parse all of these, and must **refuse unknown keys** that change grading semantics, rather than drop them silently.

**Where the image comes from varies:**
- a prebuilt `docker_image` on Docker Hub (TB2.1 `alexgshaw/*`; MiMo pins it by digest ✓);
- a Prime-registry rewrite (PI Terminal-Lego);
- a Dockerfile `FROM` a private or local tag (BeyondSWE `hub.byted.org`, NatureBench `naturebench-base:v3`, termgrade `dsbase-venv312:local`);
- a Dockerfile that `git clone`s at build time (equivalence-tests, with `|| true` on every step, so a failed checkout still "builds");
- shipped without its repo (Lego-RL `environment/repo/` must be restored).

**Oracles:** present in TB2.1, Terminal-Lego, Tasksmith, termgrade, openai-math (fetched from GitHub), equivalence-tests, BeyondSWE (broken), harbor-mix (94/100) and Lego-RL. Absent in NatureBench, MiMo (all six domains), SmolDataEnvs/data-agent and LHTB. **Roughly half of the Harbor-format task volume cannot be oracle-checked as shipped.**

**Good patterns worth copying:**
- Tasksmith's trusted grader validates the submission before importing any learner code: regular files only, no symlinks, size bounds, sha256 on immutable assets, chown to root. It runs learner code unprivileged and owns the reward file.
- Tasksmith's evidence (baseline, oracle, probe and rollout `result.json` sha256) is bound to `subject_bundle_hash`, and probes carry *different* bundle hashes, i.e. they are mutated bundles. This matches Crucible's content-hash-bound acceptance.
- MiMo writes **no reward** on testbed failure (reset or patch-apply), so infrastructure errors are not scored as 0.
- MiMo pins images by digest and moves `.git` away (`git-hidden`) during the agent phase.
- RetroEnv publishes probe rewards for degenerate policies.

### 2.2 Verifiers row tasksets (PrimeIntellect)
Each row carries `image_name`/`docker_image`/`image_url`, base commit, `patch` (gold), `test_patch`/`f2p_script`, F2P/P2P lists and the problem statement. The taskset code (`research-environments/environments/swe/*_v1`) turns a row into a sandbox rollout. That code is **not in the dataset**, so the HF repo alone cannot be run or graded. M10 needs per-family row→task adapters, or must import the taskset.

PrimeIntellect's "Verified" is the most rigorous and best-evidenced claim in the survey (§3). Its gold-patch validation is run by the same SolveEnv harness that will later serve rollouts, so it validates the runtime too.

### 2.3 NeMo Gym JSONL
Each row = `responses_create_params` (OpenAI Responses-API input + tools) + verifier-specific fields (`ground_truth`, `schema_str`, `instructions`/`llm_judge`, `exp_cal_state`, `answer`) + `agent_ref` (which NeMo agent server to use). Grading = the resources server's `/verify` in `NG/resources_servers/<name>/app.py`.

- No containers; tools are in-process Python (workplace_assistant has 27 tools over pandas DataFrames ✓).
- No oracle trajectories, except workplace_assistant's ground-truth action list, which doubles as an oracle for its state-diff grader.
- **Hybrid LLM-judge graders** (CFBench, SysBench) make reward depend on a judge model and its version. M9's determinism checks cannot pass them as is; they need a pinned judge and agreement/rerun statistics.

### 2.4 OpenEnv and plain data
RetroEnv is the most self-auditing dataset in the survey:
- `audit.json` has schema, leakage, probe and solvability sections, and its own audit says **`"passed": false`** (one held-out scaffold collision) ✓.
- Probe policies earn real reward: `first_step_only` 0.595, `duplicated_route` 0.48, `false_stock_claim` 0.20, `empty` 0.0, oracle 1.0 ✓.
- It still ships `tasks-private/` (the hidden known routes) and `runs/` (405 files of full model episodes on `test_id`) in the same public repo ✓.

procedural-pile is SFT data with solver answers. Its RL use needs the reasoning-core scorers.

## 3. What "Verified" means concretely (✓ from the published logs)

| Dataset | Oracle runs | Flake handling | No-op / no-edit floor | Infra triage | Evidence shipped |
|---|---|---|---|---|---|
| Multi-SWE-RL-Verified | 2 independent gold passes, both must be 1.0 | 4,703 → 2,498 pass p1 → 2,410 pass p2 (**88 = 3.5% failed on rerun**); 172 more failed a **low-concurrency retest** after a high-concurrency pass flagged them | no-edit run: 11 rows solved with zero edits removed | p1 reasons: test_failed 1,349, none/timeout 734, setup_failed 107, sandbox_error 10, gold_apply_failed 5; C++ 0/449 dropped wholesale | `validation*.jsonl` per row (attempts, sandbox_create_s, gold_apply_s, test_run_s, oom, timeout, stop_condition, test output tail); `validation_*_p1/p2` columns |
| R2E-Gym-Subset-Verified | 1 pass @ concurrency 200, then 10× retry of failures | drop only if 0/10; **≥1/10 kept as "recovers in training"** | none stated | 39/56 drops are aiohttp/tornado network/timing tests; dataset drift | `metadata/filtered_drops.json` + `dropped` split |
| SWE-Lego-Real-Data-Verified | same as R2E | same (flaky kept) | none | — | `filtered_drops.json` + `dropped` split |
| Scale-SWE-Verified | 1 gold pass (c128, t1200) + low-concurrency rerun of infra failures | `stable_infra_failure` = still failing at low concurrency (11) | `noop_pass` 15 removed; 299 `noop_valid_failure` recorded. The no-op run is evidenced for **only 314 rows** | 892 images failed transfer/availability | `scale-swe-validation.jsonl` (2,386 rows), `exclude-images.json` |
| SWE-rebench-V2-F-V | 2 gold passes | pass-2 flaky removed (`flaky_instances.json`) | no-edit removed | language and image blocklists; **RL always-fail audit** (0/16 in every group across GLM-5.2 runs → re-gold twice) | 4 artifact files |
| Terminal-Lego (upstream) | 1 Docker round trip: build, solve, tests | none | **none**: PI later found **234 no-op passes** (1.6%) | none | pass rate only |
| Terminal-Lego (PI re-validation) | gold oracle in Prime sandbox, rerun of failures at c16 | `oracle_validation_flaky` 8 (passed on retry, attempts = 3, then excluded) | no-op run with `debug_command="true"`: 234 excluded | gold_apply_failed 520, test_failed 116, validation_timeout 96, image_build_timeout 242, sandbox_error 4, malformed 1 | `prime-data-excluded-tasks.jsonl` (1,224 rows, per-task reason, image, timings) |
| termgrade | oracle in built image, `--network none` | — | — | — | `tasks.jsonl` `oracle_verified` + checksum; overlap check vs TB2.1 |
| Tasksmith | baseline + oracle + semantic probes + blind rollout | — | baseline control | — | sha256 evidence per task, but the evidence paths are **absolute paths on the author's laptop** (`/Users/adithya/...`), so they cannot be checked |
| SmolDataEnvs / data-agent | **no oracle**; "strong agent models reproduced the gold" | — | — | — | none |
| openai-math | official proof checked by Comparator in-env (369/405) | — | `sorry` stub is the floor (axiom check) | 36 failures listed with reasons | manifest `oracle result` |
| harbor-mix | difficulty filter + AI audit + human panel | — | — | — | paper |
| BeyondSWE, NatureBench, MiMo, LHTB, NeMo sets | none stated | — | — | — | — |

**Takeaways:**
1. "Verified" ranges from "a model once solved it" to "two gold passes, a no-edit floor, a concurrency-controlled retest, and an RL always-fail audit with per-row logs". The word alone carries no information; the evidence file does.
2. **A single upstream oracle pass is not enough.** Re-validating a "Docker-verified" set removed 8.1% (Terminal-Lego). In Multi-SWE, 3.5% of gold passes fail on an identical rerun, and a further ~7% fail only at high concurrency.
3. **Concurrency is a flake source in its own right.** PI runs a separate low-concurrency retest, and its note calls this temporary "until sandbox high-concurrency degradation is fully investigated".
4. Keeping flaky rows "because they recover in training" (R2E, SWE-Lego) contradicts Crucible A4. An importer must either re-measure the flake rate and reject, or carry `flake_rate` into M10 masking.

## 4. Defects visible in the samples (numbered; each is a candidate M9 registry entry)

| # | Dataset | Defect | Evidence |
|---|---|---|---|
| D1 | BeyondSWE-harbor | **450/500 `solve.sh` are indented 8 spaces.** The heredoc terminator `        __SOLUTION__` never matches, so `cat` swallows the rest of the file, the `git apply` never runs, and the script **exits 0 without changing anything** (a silent no-op oracle). Only the 50 Doc2Repo tasks are unaffected. | Reproduced with `bash -x` ✓; full scan by range requests ✓ |
| D2 | BeyondSWE-harbor | 500/500 Dockerfiles are `FROM hub.byted.org/bandai/swe.swe-python-fanzhe-stage{3,4}` (ByteDance-internal, NXDOMAIN) | scan ✓, `host` ✓ |
| D3 | BeyondSWE-harbor | `allow_internet = true` on all 500, plus `network_mode: host` in compose. The leak defence is a `--ak disallowed_tools` list on the Claude Code CLI, i.e. harness-specific, not environmental. `tests/instance.json` carries the gold `patch` (verifier-side, so fine only if `/tests` is staged after the agent). | ✓ |
| D4 | TB2.1 `adaptive-rejection-sampler` | Ships the reference solution **inside the agent image** (`/protected/protected.tar.gz.enc`). The decryption password sits in plain text in the public `solve.sh`, and the agent has internet. | ✓ |
| D5 | TB2.1 / Terminal-Lego | `test.sh` runs `apt-get update` and `curl astral.sh/uv` **at grade time**. The grader needs the network, and its outcome depends on mirrors (a flake source; it breaks in `--network none` verifiers). | ✓ |
| D6 | Terminal-Lego task_00000 | The instruction lists the exact expected output. Tests are substring checks (`'file1.txt' in content`). `document.pdf`/`image.png` are 23-byte placeholders. The canary comment sits above the shebang. | ✓ |
| D7 | terminal-bench-science, terminal-bench-2.1 | The cards say "**This repository is private**" (TB-science: "every task ships its tests/ and solution/ (the answer key) … promote to public only once upstream does"), yet both repos are public and download without auth. The answer keys of a benchmark in preview are therefore public. | unauthenticated `curl` 200 ✓ |
| D8 | repo2rlenv-equivalence-tests | The task places `reference_<f>` in the agent's `task_module.py` and says inspecting it is "the intended solve-path". `f = reference_f` passes. **Degenerate by construction** (reward = 1 for a one-line alias). | ✓ |
| D9 | SmolDataEnvs / data-agent | `[environment.env] HF_TOKEN = "${HF_TOKEN}"` puts the **operator's HF token inside the agent sandbox**, which has `allow_internet = true`. The source notebooks with answers are public on HF (jupyter-agent dataset). The gold answer is in `task.toml [metadata]` and `[verifier.env]` (published). `test.sh` runs `pip install math-verify … || true` (the grader silently loses a tier offline, though the card claims "no network"). data-agent also emits a `submission` reward (1 for any non-empty answer). | ✓ |
| D10 | workplace_assistant | **13.5% of train (170/1,255) and 13% of val (71/545) have an empty `ground_truth`.** The state-diff grader therefore gives reward 1 to a policy that makes no write calls: a **no-op floor of about 13%**. The card also says "1,260 records, 26 tools, 690 tasks", but the data has 1,800 rows and 27 tools. | full JSONL ✓; grader code ✓ |
| D11 | structured_outputs | A schema-valid object with fabricated values gets full reward (acknowledged in the blog). Degenerate content-free solutions are rewarded. | grader code ✓ |
| D12 | Lego-RL-2699 | The verifier parses `OPENSWE_EXIT_CODE=` from **combined test stdout** ("last occurrence wins"), so it is spoofable by test-time prints. Conda deps are unpinned (`pandas>=0.25`). The repo snapshot is not shipped. | ✓ |
| D13 | RetroEnv | The audit failed (`passed: false`) but the dataset was published anyway. `tasks-private/` (hidden routes) and model episodes on `test_id` are in the same public repo. Partial-credit hacks: first_step_only 0.595. | `audit.json` ✓ |
| D14 | NatureBench | No oracle. Unlimited `/evaluate` calls against the hidden test ground truth with best-score tracking (hill-climbing on test), mitigated only by a post-hoc LLM judge. 939 GB of task archives. | ✓ |
| D15 | Tasksmith | Evidence paths point to the author's machine. Cards link to the `HuggingEnvs/…` namespace while the repo is `FineEnvs/…`. | ✓ |
| D16 | SWE-Lego-Real-Data-Verified | Each row carries `messages`, a full successful agent trajectory, next to the task. Any prompt builder that dumps row fields leaks the solution. | features ✓ |
| D17 | MiMo | The `/etc/hosts` blocklist is the only network leak control under `network_mode = "public"`. Direct IPs, proxies and mirrors bypass it. | ✓ |
| D18 | Multi-SWE-RL-Verified | The card names `multi-swe-rl-validation*.jsonl`, but the repo has `validation*.jsonl` (a minor provenance mismatch). | ✓ |

## 5. What M9 should check when importing such datasets

Import is **re-acceptance, not trust**: every imported task runs the Acceptance Contract (DESIGN §9) in Crucible's own runtime, and the publisher's "verified" flag becomes an input feature, not a pass. Below are concrete checks, each mapped to A1–A13; new registry entries are marked **NEW**.

**I0 Provenance and shape** (A9):
- Pin `repo@revision` (HF commit sha) per import.
- Record license, gating and canary GUIDs.
- Parse `task.toml` by `schema_version` and **reject unknown grading-relevant keys** (`continue_until_timeout`, `[[verifier.collect]]` when unsupported).
- Require every image to be resolvable: pull or digest-resolve each `docker_image`/`FROM`, and fail on NXDOMAIN, local-only tags and `:latest` without a digest (D2, NatureBench, termgrade).
- **NEW `build-or-true`:** fail Dockerfiles whose source checkout or install steps end in `|| true` (D12-style silent drift). Diff the built image's checked-out commit against the declared one.

**I1 Oracle in the declared environment** (A1):
- Run `solution/` ≥3 times in the task's own image, network, user and resources. A row taskset applies its gold patch through the row adapter.
- **NEW `oracle-noop`:** the oracle must change the workspace. Assert that `git diff` (or a filesystem hash delta) is non-empty after `solve.sh`, and that `solve.sh` passes `bash -n` and shellcheck SC1039/SC2317 (catches D1's indented heredoc, which exits 0 while doing nothing).
- **NEW `oracle-network`:** run the oracle with the agent-phase network policy. If it needs the network (openai-math fetches proofs, Terminal-Lego/TB `curl uv`), classify it, stage the assets, or reject.
- Tasks with no oracle (MiMo, NatureBench, SmolDataEnvs, LHTB) enter only as **`unverified`/eval-only** unless Crucible's forge synthesises and validates one.

**I2 Floors** (A2):
- No-op (`true`), empty patch, unmodified original.
- **NEW constant-answer and empty-action floors for non-container sets:** the empty tool-call list (D10: about 13% floor), "Not Applicable", and the minimal schema-valid object (D11).
- **NEW `reference-alias` probe:** if any `reference_*` symbol or the oracle file is visible in the agent workspace, try aliasing or copying it (D8, D4).
- Reject the task if any floor scores above 0. For datasets, report the **floor rate**; a set whose floor rate exceeds the training band's lower edge is unusable for RL without filtering.

**I3 Flakiness** (A4):
- Gold reruns ≥3 at **two concurrency levels** (low and production), because PI's data shows concurrency-only failures (172 rows).
- Classify each failure as deterministic vs flaky vs infra (setup, image pull, sandbox start, OOM, timeout), with the same reason vocabulary as PI's logs: `test_failed`, `setup_failed`, `gold_apply_failed`, `sandbox_error`, `timeout`, `oom`.
- Reject flaky tasks (A4 is zero false negatives); do not keep them the R2E way.
- **NEW `grade-time-network`:** grade with the verifier network off. If the reward changes, reject or re-stage (D5, D9 `pip install || true`).

**I4 Leak audit** (A5):
- Scan the **agent-visible** surfaces (built agent image, `instruction.md`, `environment/`, row fields used in the prompt) for hidden strings: gold patch hunks, `expected_output`, `EXPECTED_ANSWER`, test names, `reference_*`, encrypted blobs together with their passwords (D4), and canaries.
- Check that `tests/` and `solution/` are absent from the agent image (Harbor stages `/tests` after the agent; agentenv does not by default; DESIGN §10.1).
- **NEW `credential-in-agent-env`:** fail if `[environment.env]` forwards host secrets (`HF_TOKEN`, API keys) into the agent phase (D9).
- **NEW `public-answer-key`:** flag a dataset whose `tasks-private/`, `solution/` or published rollouts of the *eval* split are public (D7, D13). Such tasks are training-only, never benchmark.
- Problem-statement scrub (SWE-rebench regexes: `#1234`, `gh-1234`, `/pull/N`, external URLs).
- `.git` history scrub (BeyondSWE's own Dockerfile does `reflog expire` + `gc --prune`; verify it, since it is not guaranteed).

**I5 Isolation and tamper** (A6, A13):
- Separate verifier environment, or staging after the agent.
- The grader owns the reward file.
- Learner code runs unprivileged (copy Tasksmith's `validate_submission`).
- **NEW `stdout-marker-spoof`:** if the grader parses a marker from test stdout (D12 `OPENSWE_EXIT_CODE`), run a mutant that prints the marker and exits non-zero. It must score 0. BeyondSWE's `run_tests.py` also prints `<pytest>true</pytest>`, but its `test.sh` grades on the exit code, so it is safe today.
- Network leak controls must be environmental (`no-network`/allowlist), not `/etc/hosts` (D17) or CLI flags (D3).

**I6 Instruction↔test coverage** (A7):
- Flag instructions that state the exact expected output (D6). This is valid but low-information; mark it easy and keep it out of the hard split.
- Flag graded properties that the instruction does not state.

**I7 Grader determinism** (A4/A12):
- LLM-judge graders (harbor-mix 27, NatureBench, CFBench/SysBench `llm_judge`) need a pinned judge model and version and ≥3 judge reruns with agreement ≥ τ. Otherwise they enter as a soft channel only, never as the gate.

**I8 Calibration** (A10):
- Treat publisher difficulty labels as priors only (termgrade's per-model pass rates, Lego-RL's "1–3 of 4", harbor-mix's "≥67% frontier fail"). Re-measure with Crucible's policy, as the termgrade card itself advises.
- Use PI's **RL always-fail audit** idea as a post-import monitor: a task at 0/16 across groups for N steps triggers re-gold-validation.

**I9 Contamination** (A5/§741):
- Canary GUIDs present (TB, Terminal-Lego) → exclude from SFT corpora.
- Temporal decontamination for PR/issue-derived rows (merge dates vs model cutoff).
- Near-duplicate checks across imported sets and against Crucible's eval split (termgrade checks against TB2.1; RetroEnv uses scaffold/Tanimoto; SWE sets share upstream repos across Scale-SWE, BeyondSWE and SWE-Lego).
- Dataset-level held-out overlap: SmolDataEnvs train and data-agent eval are both drawn from the same 471 Kaggle datasets, per their cards. Tasks are held out, but the data files probably are not (not checked task by task).

**I10 Dataset-level report.** Per import, write `import_report.json` with:
- counts per verdict;
- floor rate, flake rate (by concurrency) and infra-failure rate;
- leak findings;
- a diff against the publisher's own exclusion list where one exists (Terminal-Lego, Multi-SWE, Scale-SWE, R2E). Agreement with PI's labels is a direct measure of M9 recall and precision.

## 6. What M10 needs to run them

1. **Harbor task-dir executor covering schema 1.0–1.4:**
   - per-phase `network_mode` (`no-network`/`allowlist`+`allowed_hosts`/`public`) and per-phase `user`;
   - `verifier.environment_mode = "separate"` with `tests/Dockerfile` and `[[artifacts]]` transfer;
   - multi-service compose with sidecars and `[[verifier.collect]]`;
   - `[environment.healthcheck]` as a setup hook with retries;
   - `gpus`/`gpu_types` (ties to the planned `gpu_docker` provider);
   - reading `reward.json` (multi-channel, e.g. data-agent) over `reward.txt`.

   agentenv's verifier runs in the agent container, so the separate-verifier step in DESIGN §10.1 is required for Tasksmith, NatureBench and MiMo-style tasks.
2. **Repo resolution:** fetch Harbor git-repos datasets by full HF URL plus `registry.json` name@version (a bare repo id is not enough, per the blog); handle tarball-packaged tasks (NatureBench, Tasksmith `tasks.tar.gz` preserves exec bits); fetch task-level lazily (NatureBench is 939 GB; one task is up to 82 GB). Never snapshot a whole repo.
3. **Row-taskset adapters** (verifiers/PI families: R2E, SWE-Lego, Scale-SWE/BeyondSWE schema, Multi-SWE, SWE-rebench): build the sandbox from `image_name`/`docker_image`/`image_url`, run `pre_commands`, stage `test_patch`/`f2p_script` after the agent, run the test command, parse F2P/P2P. Image registry remapping is required (PI rewrites to the Prime registry; BeyondSWE images are unreachable).
4. **NeMo Gym bridge:** an in-process resources-server client (`/seed_session`, tool calls, `/verify`) for JSONL rows, honouring `agent_ref` and `responses_create_params` (Responses-API shape, tools with `strict`). No container is needed, but session state must be isolated per rollout. LLM-judge servers need a judge endpoint and its version recorded on the rollout.
5. **Failure taxonomy, borrowed from the logs:** `setup_failed`, `image_build_timeout`, `sandbox_start_timeout`, `gold_apply_failed`, `test_failed`, `timeout`, `oom`, `sandbox_error`, `no_reward_written`. MiMo's "testbed failure writes no reward" must map to **infra-masked**, never 0. OpenEnv's rollout `reward=None` + `error` is the same contract.
6. **Concurrency-aware scheduling:** a measured degradation knob (PI's high- vs low-concurrency discrepancy), per-task sandbox-create, setup and test timings (PI logs `sandbox_create_s` ≈ 40–70 s, test runs up to 3,400 s for OpenBLAS), and timeouts from `task.toml` with a global cap.
7. **Network and secret policy enforced by the runtime:** never forward host env into the agent phase. Run grade-time network off by default, and pre-stage verifier dependencies at import (uv, pytest, math-verify).
8. **Harness-specific extensions are explicit opt-ins:**
   - LHTB `continue_until_timeout` feeds verifier feedback back to the agent. That is a reward channel visible to the policy, so it must be marked and is never allowed in eval.
   - MiMo's unprivileged-agent and blocklist steps.
   - BeyondSWE's `disallowed_tools`.

   Record the harness id and version on every rollout, because pass rates depend on it (Terminus-2 vs Claude Code vs OpenCode).
9. **Trajectory capture compatible with ATIF.** NatureBench's verifier converts Harbor ATIF `trajectory.json` into judge input, so M10's trajectory must be ATIF-exportable for LLM-judge tasks.

## 7. Shortlist: external acceptance and runtime test bed for Crucible

1. **PrimeIntellect/Terminal-Lego-15k + Lego-X/Terminal-Lego-15k (paired).**
   - The upstream claims "Docker-verified". PI's re-validation labels 1,224 tasks with exact reasons: 234 no-op passes, 520 gold-apply failures, 116 test failures, 96 timeouts, 242 build timeouts, 8 flaky.
   - That is a **labelled ground truth for M9**: import ~300 tasks stratified over those reasons plus clean ones, and measure M9's recall and precision against PI's verdicts.
   - The tasks are small CPU Harbor 1.0 tasks (1 CPU, 1 GB), so they are cheap.
   - They exercise D5/D6 (grade-time network, low-information instructions).
2. **PrimeIntellect/R2E-Gym-Subset-Verified** (with Multi-SWE-RL-Verified as the multi-language follow-up).
   - Forces the M10 **row→task adapter** onto public `namanjain12/*` images.
   - The `dropped` split (56, 0/10 gold) gives negatives, and the 10× retry design gives flaky positives that Crucible's A4 must *reject* where R2E kept them.
   - Multi-SWE adds pass-1/pass-2 logs and a concurrency-retest list for the concurrency-flake check.
3. **FineEnvs/HF_ML_Tasksmith** (with **repo2rlenv-equivalence-tests** as a negative control).
   - Harbor 1.3 features: separate verifier, `[[artifacts]]` allowlist, `no-network`, learner/root users, and 11 GPU tasks (1–2 GPUs). It stresses M10's separate-verifier step and the GPU provider on the RTX 4090/5090 nodes.
   - Its bundle-hash-bound evidence is the closest external analogue to Crucible's content-hash acceptance.
   - equivalence-tests (100 tasks, all degenerate per D8) should be rejected 100% by the `reference-alias` probe.
4. **harborframework/terminal-bench-2.1.**
   - The blog's canonical example and the most-used tagged dataset: 89 tasks, all with oracles, prebuilt images.
   - It is the runtime-compatibility smoke test: Crucible's Harbor executor must reproduce `harbor run --agent oracle` rewards.
   - It contains a real leak case (D4) and grade-time network (D5) for A5 and I3.
5. **nvidia/Nemotron-RL-agent-workplace_assistant.**
   - The non-coding, container-free NeMo shape. It keeps M9/M10 domain-agnostic (G0).
   - It has a deterministic state-diff grader with a built-in oracle (the ground-truth action list) and a known **13% no-op floor** (D10) that the A2 floor check must surface.
   - It is 30 MB and runs in process.

Negative controls to keep as fixtures, sampled small:
- **BeyondSWE-harbor:** the D1 no-op oracle and D2 unreachable images; M9 must reject all 500 for the right reason.
- **RetroEnv** `audit.json` probe values: a reference for partial-credit hack rates.
- **SmolDataEnvs:** D9, a credential in the agent env.

## 8. Open items
- harbor-mix is gated (auto-approve). Accepting its terms is a user decision; until then only the card and tree were read.
- I did not run anything in Docker. All runtime claims are from files and code, except the D1 bash reproduction, which ran only the shell script in a scratch directory.
- The PI taskset code (`research-environments`) and the NeMo resources servers are needed to grade row and JSONL sets. They are in `RL-Environments/06-frameworks-and-env-code/` (NeMo was checked for workplace_assistant and structured_outputs).
