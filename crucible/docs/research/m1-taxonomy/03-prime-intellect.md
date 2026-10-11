<!-- M1 taxonomy 03: Prime Intellect verifiers, prime-environments, research-environments, prime-agent. Read-only study by a subagent, 2026-10-12. -->

# Prime Intellect environment families: taxonomy, API reference, and framework requirements

All paths are relative to `ROOT=/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/RL-Environments/06-frameworks-and-env-code`. Short prefixes used below:
- `PE` = `primeintellect-ai-prime-environments/environments`
- `RE` = `primeintellect-ai-research-environments/environments`
- `RP` = `primeintellect-ai-research-environments/packages`
- `VF` = `primeintellect-ai-verifiers/verifiers/v1`
- `PA` = `primeintellect-ai-prime-agent`

## 0. Things to know first

1. **The verifiers checkout has no v0 code.** It is a 1-commit shallow clone (`ae819de`). `primeintellect-ai-verifiers/verifiers/__init__.py:1` says "the stack lives in `verifiers.v1`", and there is no `verifiers/envs`, `rubrics` or `parsers`. The v0 API in §2 is rebuilt from:
   - its actual use in the 109 prime-environments envs, which pin `verifiers>=0.1.2 … 0.1.11`;
   - the prime-environments contributor guide, `primeintellect-ai-prime-environments/AGENTS.md:86-121` (also served as `CLAUDE.md`).
2. **Each repo uses one API version.**
   - prime-environments: 109 envs, all **v0** (`load_environment()` returning `vf.SingleTurnEnv`/`MultiTurnEnv`/…).
   - research-environments: 106 envs in 14 group folders, all **v1** (`Taskset`/`Task`, needs `verifiers>=0.3.1`, `RE/math/aime25/pyproject.toml`).
3. **Licenses.**
   - prime-environments: Apache-2.0, with exceptions: `PE/agentic_misalignment/LICENSE`, and 3 env pyprojects declare MIT.
   - research-environments: Apache-2.0.
   - verifiers: MIT.
   - prime-agent: MIT.
   - Upstream datasets and benchmarks keep their own licenses; I did not check those.
4. **"Single-turn" in v1 is not a task type.** Every v1 task is harness-agnostic.
   - A "single-turn" taskset (e.g. `RE/math/aime25/aime25/taskset.py:26-33`) runs under the default `bash` harness, which is an agent in a sandbox (`VF/utils/loaders.py:153-161`).
   - To get a pure one-shot completion you pin the `null` harness, a tool-less chat loop, sometimes with `max_turns`. Example: `RE/code/humaneval/README.md:23` uses `--env.agent.harness.id null`.
   - So Crucible's v1 importer needs a "null-harness" mode to reproduce single-turn semantics. A v0 importer needs a real single-turn task type.

---

## 1. Family catalog

### Summary table

| # | Family | PE count | RE count | Representative envs |
|---|---|---|---|---|
| F1 | Single-turn verifiable QA / MCQ / math / science (incl. judge-graded QA) | 22 | 19 | PE `mmlu`, `gauss`, `simpleqa_verified_2`; RE `math/aime25`, `knowledge/simpleqa`, `science/gpqa`, `judges/trace_cheating_recall_500` |
| F2 | Single-turn format/constraint following and synthetic puzzles | 8 | 4 | PE `allenai_ifeval`, `synlogic`, `pydantic_adherence`, `arc_agi`; RE `if/ifeval`, `reasoning/unscramble` |
| F3 | LLM-judge open-ended writing / creative / pairwise | 8 | 0 | PE `creative_writing`, `writing_bench`, `llm_writer_negative_style`, `deep_consult` |
| F4 | Safety / alignment / refusal | 8 | 0 | PE `coconot`, `wildjailbreak`, `harmeval`, `wmdp` |
| F5 | Multimodal (vision) | 6 | 4 | PE `synthetic_clocks`, `seeclick`, `vision_sr1`; RE `multimodal/mmmu_pro`, `multimodal/charxiv` |
| F6 | Code generation graded by executing tests | 16 | 7 | PE `humaneval`, `aider_polyglot`, `kernelbench`, `polars_env`; RE `code/livecodebench`, `code/forth_lang`, `code/nl2repobench` |
| F7 | Formal theorem proving (Lean 4) | 2 | 11 | PE `minif2f`, `stepfun_prover`; RE `lean/minif2f`, `lean/putnambench` |
| F8 | Codebase search / repo QA in a sandbox | 6 | 0 (see F9 `swe_atlas_qna`) | PE `triton_codebase_search`, `vllm_code_search`, `ktor` |
| F9 | SWE: repo repair in a container with hidden tests | 1 | 18 | PE `mini_swe_agent_bench`; RE `swe/r2e_gym`, `swe/swebench_verified`, `swe/pi_deepswe`, `swe/multiswe` |
| F10 | Terminal / sandbox agentic tasks (Harbor-format, data science, CTF) | 3 | 10 | PE `dabstep`, `bixbench`, `arc_agi_tool`; RE `terminal/terminal_bench_2`, `terminal/terminal_lego`, `reasoning/uuid_ctf`, `reasoning/prolog` |
| F11 | Search / RAG / deep research | 8 | 10 | PE `browsecomp_plus`, `search_r1_ish`, `webexplorer`, `art_e`; RE `search/browsecomp_plus`, `search/deepdive`, `search/wideseek` |
| F12 | Tool-use / API / MCP / stateful DB worlds / simulated user | 8 | 10 | PE `tau_bench`, `skyrl_sql`, `mcp_universe`, `agent_dojo`; RE `tool_use/general_agent`, `tool_use/tau2_bench`, `tool_use/mcp_atlas`, `tool_use/bfcl_v3` |
| F13 | Browser / GUI / device control | 2 | 0 | PE `web_voyager`, `androidworld` |
| F14 | Multi-turn games / navigation / interactive fiction | 7 | 1 | PE `mastermind`, `hanabi`, `text_quests`, `wiki_race`; RE `reasoning/wikispeedia`; VF examples `environments/wordle`, `environments/kuhn_poker` |
| F15 | Long-context | 1 | 12 | PE `longbench_v2`; RE `long_context/mrcr_v2`, `long_context/oolong_synth`, `long_context/verbatim_copy` |
| F16 | Multi-turn simulated-user conversation (no tools) | 3 | 0 (built-in `user-sim` env) | PE `spiral_bench`, `sycophancy_eval`, `aidanbench` |

Totals: PE = 109, RE = 106.

### Full membership

**PE**
- **F1:** arc, BigBench_BBH, boolq, gauss, hellaswag, legalbench, misguided_attn, mmlu, phybench, piqa, psycho_bench, reward_bench, sad, simple_bench, simpleqa_verified_2, taxcalc_bench, thematic_generalization, truthfulqa, UncertaintyBench, uq, uq_project, winogrande
- **F2:** allenai_ifeval, arc_agi, ascii_tree, enigmata, internbootcamp_env, lisanbench, pydantic_adherence, synlogic
- **F3:** agency_bench, antislop, creative_writing, deep_consult, llm_writer_negative_style, llm_writing_detection, vrcli, writing_bench
- **F4:** agentic_misalignment, coconot, harmeval, jailbreak_bench, medsafetybench, sg_bench, wildjailbreak, wmdp
- **F5:** clockbench, penrose_diagrams, seeclick, synthetic_clocks, vision_sr1, vpct_1
- **F6:** aider_polyglot, autodiff, backend_bench, colf, gptworld, gpu_puzzles, gpu_puzzles_modal, humaneval, kernelbench, llm_training_puzzles, mbpp, polars_env, rust_cargo, scicode, tensor_puzzles, transformerpuzzles
- **F7:** minif2f, stepfun_prover
- **F8:** datasets_codebase_search, ktor, nextjs_codebase_search, torch_ao_codebase_search, triton_codebase_search, vllm_code_search
- **F9:** mini_swe_agent_bench
- **F10:** arc_agi_tool, bixbench, dabstep
- **F11:** art_e, browsecomp_plus, congressional_records, gutenberg_env, hle_web_py, patents_ar_env, search_r1_ish, webexplorer
- **F12:** agent_dojo, agentharm, fh_aviary, github_mcp, mcp_universe, med_agent_bench, skyrl_sql, tau_bench
- **F13:** androidworld, web_voyager
- **F14:** balrog_bench, fruit_box, hanabi, mastermind, nyt_connections, text_quests, wiki_race
- **F15:** longbench_v2
- **F16:** aidanbench, spiral_bench, sycophancy_eval

**RE**
- **F1:** math/{aime24, aime25, aime26, apex_shortlist, arxivmath, i3_math, math_env, math500}; knowledge/{hle, hle_diamond, mmlu_pro, mmmu, simpleqa, simpleqa_verified, triviaqa}; science/{gpqa, frontierscience, i3_science}; judges/trace_cheating_recall_500
- **F2:** if/{ifbench, ifeval}; reasoning/{i3_logic, unscramble}
- **F5:** multimodal/{charxiv, mmk12, mmmu_pro, virl39k}
- **F6:** code/{forth_lang, humaneval, i3_code, livecodebench, nl2repobench, pi_programbench, scicode}
- **F7:** lean/ (all 11)
- **F9:** swe/ (all 18)
- **F10:** terminal/ (all 6); reasoning/{uuid_ctf, prolog, deshuffle_papers}; science/drug_discovery_bench
- **F11:** search/ (all 10)
- **F12:** tool_use/ (all 10)
- **F14:** reasoning/wikispeedia
- **F15:** long_context/ (all 12)

### Mechanisms across RE (grep evidence)
- **22 envs build on `VF/tasksets/harbor`:** all terminal, most swe, plus apex_agents, lab, drug_discovery_bench, pi_programbench, general_agent (corpus download).
- **7 envs define `vf.Toolset`s:** bfcl_v3, browsecomp_plus, enterprise_ops_gym, forth_lang, general_agent, mcp_atlas, wikispeedia.
- **Custom `Env` subclasses:** tau_common, lean_common, the harbor separate-verifier envs, hle_diamond, pi_officeqa_pro_v2.
- **Custom `Harness`:** apex_agents, browsecomp_plus (bundles `NullHarness`), hle_diamond, tau_common.

---

### F1. Single-turn verifiable QA / MCQ / math / science
1. **Family.** PE 22, RE 19. Representatives: PE `mmlu`, `gauss`; RE `math/aime25`, `knowledge/simpleqa`, `judges/trace_cheating_recall_500`. License Apache-2.0 (env code).
2. **State.** None.
3. **Actions.**
   - v0: one completion (`vf.SingleTurnEnv`, `PE/mmlu/mmlu.py:116`).
   - v1: any harness; one-shot only with the `null` harness.
4. **Actors.** Single agent.
5. **Turns.** Single turn.
6. **Observations.** Prompt (system + user); MCQ options formatted in the prompt (`PE/mmlu/mmlu.py:83-103`).
7. **Grader.**
   - Exact match through a custom `Parser`: `PE/mmlu/mmlu.py:12-52, 108-114`.
   - math-verify on the boxed answer: `RE/math/aime25/aime25/taskset.py:27-33` (`vf.verify_boxed_math_answer`).
   - LLM judge: `vf.JudgeRubric` at `PE/gauss/gauss.py:276-311`, with a weighted mix of judge 1.0 and deterministic accuracy 0.3 at `PE/gauss/gauss.py:350,355`.
   - v1 judge class: `vf.Judge` subclass at `RE/knowledge/simpleqa/simpleqa/taskset.py:124-145`.
   - Plugged reference judge: `ReferenceJudgeConfig` in `TaskConfig.judges` (used by `RE/search/deepdive/deepdive/taskset.py:23-33`).
   - Classifier eval: `trace_cheating_recall_500` scores CHEATING/LEGITIMATE label accuracy plus recall metrics, in the `accuracy` reward of its taskset.
8. **Reward density.** Final only.
9. **Infra.** HF datasets pinned by revision (`aime25/taskset.py:37-40`); OpenAI-compatible judge endpoint.
10. **Isolation.** None needed. v1 sets `network_allow: []` on the data (`aime25/taskset.py:16`).
11. **Port.**
    - `VF/judges/reference.py`: reference-answer judge with configurable verdict labels; it raises on unparsable verdicts instead of scoring 0 (lines 28-61, 63-92). MIT.
    - `RE/math/aime25/aime25/taskset.py`: minimal ~60-line taskset template. Apache-2.0.

### F2. Single-turn format/constraint following and synthetic puzzles
1. **Family.** PE 8, RE 4. Representatives: PE `allenai_ifeval`, `ascii_tree`, `synlogic`; RE `if/ifeval`.
2. **State.** None.
3. **Actions.** Single completion. Structured output via `vf.XMLParser(["think","ascii_formatted"], answer_field=...)` (`PE/ascii_tree/ascii_tree.py:22`) or `ThinkParser`.
4. **Actors.** Single.
5. **Turns.** Single turn.
6. **Observations.** Prompt only.
7. **Grader.**
   - Programmatic instruction checkers: `RE/if/ifeval/ifeval/taskset.py:52-63`, all-instructions-followed reward 1.0 plus rate and count as `@vf.metric`.
   - Pydantic schema validation (`PE/pydantic_adherence`).
   - Puzzle verifiers (`synlogic`, `enigmata`, `internbootcamp_env`).
   - Format reward from `parser.get_format_reward_func()`.
8. **Reward density.** Final.
9. **Infra.** Datasets; vendored checker libraries (`RE/if/ifeval/ifeval/utils/*`, ~3.5k lines).
10. **Isolation.** None.
11. **Port.**
    - `RE/if/ifeval/ifeval/utils/`: IFEval verifiers, copied verbatim upstream (Google, Apache-2.0).
    - `PE/pydantic_adherence`: per-sample schema adherence checker. Apache-2.0.

### F3. LLM-judge writing / creative / pairwise
1. **Family.** PE 8. Representatives: `creative_writing`, `llm_writer_negative_style`, `deep_consult`, `antislop`.
2. **State.** None.
3. **Actions.** Single completion.
4. **Actors.** Single policy plus judge model(s).
5. **Turns.** Single turn.
6. **Observations.** Prompt.
7. **Grader.**
   - Multi-judge ensemble with a weighted power-mean of rubric grades (craft 0.6 / elements 0.4): `PE/creative_writing/creative_writing.py:91-131, 164-250`.
   - One `JudgeRubric` per style rule, each weighted `1/len(rules)`, combined with `vf.RubricGroup`: `PE/llm_writer_negative_style/llm_writer_negative_style.py:15-34, 66`.
   - Pairwise report comparison in `deep_consult`.
8. **Reward density.** Final.
9. **Infra.** External judge APIs (default models such as `gpt-5` / `gpt-4.1-mini` hardcoded in `load_environment` args).
10. **Isolation.** None.
11. **Port.**
    - The ensemble and power-mean aggregation in `PE/creative_writing/creative_writing.py`.
    - The per-rule judge `RubricGroup` pattern.
    - Both Apache-2.0.

### F4. Safety / alignment
1. **Family.** PE 8. Representatives: `coconot`, `wildjailbreak`, `wmdp`, `agentic_misalignment`.
2. **State.** None.
3. **Actions.** Single completion.
4. **Actors.** Single plus judge.
5. **Turns.** Single turn.
6. **Observations.** Harmful, adversarial or noncompliance prompts.
7. **Grader.** `vf.JudgeRubric` refusal/compliance judgments (`coconot`, `wildjailbreak`, `wmdp` all import `JudgeRubric`). `agentic_misalignment` uses a custom `Rubric` subclass (class grep).
8. **Reward density.** Final.
9. **Infra.** Judge API, HF datasets.
10. **Isolation.** None.
11. **Port.** `PE/agentic_misalignment` scenario templates. Note: it carries its own LICENSE file, which must be checked before porting.

### F5. Multimodal
1. **Family.** PE 6, RE 4. Representatives: PE `synthetic_clocks`, `seeclick`, `vision_sr1`; RE `multimodal/mmmu_pro`.
2. **State.** None.
3. **Actions.** Single completion; `vision_sr1` is multi-pass (`MultiTurnEnv`).
4. **Actors.** Single.
5. **Turns.** Single or few.
6. **Observations.** Image content parts. v1 builds them as `vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=data_url))` inside `vf.UserMessage` (`RE/multimodal/mmmu_pro/mmmu_pro/taskset.py:155-169`).
7. **Grader.**
   - Exact or numeric match; click-in-bbox (`seeclick`).
   - Compile-and-compare for `penrose_diagrams`, which uses Modal.
   - LLM judge (`mmk12`, `charxiv`).
8. **Reward density.** Final.
9. **Infra.** Images as data URLs; Modal for TikZ rendering.
10. **Isolation.** None, except the Modal render step.
11. **Port.** The v1 content-part prompt construction in `RE/multimodal/mmmu_pro/mmmu_pro/taskset.py:139-169`. Apache-2.0.

### F6. Code generation graded by test execution
1. **Family.** PE 16, RE 7. Representatives: PE `humaneval`, `aider_polyglot`, `kernelbench`, `polars_env`; RE `code/livecodebench`, `code/forth_lang`, `code/nl2repobench`.
2. **State.** Sandbox filesystem.
   - Prime sandbox: `vf.SandboxEnv` at `PE/humaneval/humaneval.py:11`.
   - Modal sandbox: `PE/aider_polyglot/aider_polyglot.py:134` (`modal.Sandbox.create.aio`).
   - GPU (Modal, `PE/kernelbench/kernelbench.py:236`).
3. **Actions.**
   - v0: single completion graded post-hoc (`post_rollout`, `humaneval.py:12-60`).
   - v0 multi-turn with test feedback: `aider_polyglot` uses `max_turns=2` (`aider_polyglot.py:72`).
   - v0 sandbox tool: `polars_env` `execute_code` via `add_tool(..., args_to_skip=["sandbox_id","sandbox_state"])` at `polars_env.py:113`.
   - v1: harness-agnostic; the code is extracted from all assistant messages (`livecodebench/taskset.py:123-126`).
4. **Actors.** Single.
5. **Turns.** 1 to N (`max_turns`).
6. **Observations.** Problem statement, then test or compiler output as user/tool messages.
7. **Grader.**
   - Exit code: `humaneval.py:101-109`. Exit code 124 is an infra error and raises.
   - Hidden tests run as a uv script inside the rollout runtime: `RE/code/livecodebench/livecodebench/taskset.py:120-155`. Reward 1.0 if all pass; pass_rate recorded as a metric.
   - Weighted multi-reward: `backend_bench` (`backend_bench.py:446-452`).
   - Rewards weighted 0.0 used as metrics: `colf` (`colf.py:188-196`).
8. **Reward density.** Final, with test feedback per turn as observations only.
9. **Infra.** `prime_sandboxes` (`AsyncSandboxClient`), Modal, Docker, GPUs; uv-script verifiers.
10. **Isolation.**
    - Untrusted code runs only in the sandbox/runtime, never the eval process (`livecodebench/taskset.py:1-10` docstring).
    - Verifier network routes are allowlisted (`livecodebench/taskset.py:20,145`).
    - `forth_lang`'s toolset owns a separate execution runtime and does hidden-test grading in a fresh runtime (`RE/code/forth_lang/forth_lang/toolset.py:58-97`).
11. **Port.**
    - `RE/code/livecodebench/livecodebench/verify.py` plus the reward at `taskset.py:120-155`: an in-runtime verifier pattern. Apache-2.0.
    - The `forth_lang` toolset's dual-runtime pattern.

### F7. Formal theorem proving (Lean 4)
1. **Family.** PE 2, RE 11. Representatives: PE `minif2f`, `stepfun_prover`; RE `lean/minif2f`, `lean/putnambench`. The 11 RE envs share `RP/lean_common` (Docker image plus checker).
2. **State.** Sandbox filesystem holding a Lean/Mathlib toolchain.
3. **Actions.**
   - PE `minif2f`: multi-turn, compile feedback (`PE/minif2f/minif2f.py:412`, `max_turns=4` at `:526`).
   - `stepfun_prover`: Lean REPL in a container (`PE/stepfun_prover/containerized_lean_state.py:17`).
   - RE: agent edits `/workspace/proof.lean` through any shell harness.
4. **Actors.** Single.
5. **Turns.** Multi-turn.
6. **Observations.** Statement with `sorry` placeholders (`RP/lean_common/lean_common/prompt.py:16-48`); compiler errors.
7. **Grader.**
   - PE: compile reward `vf.Rubric(funcs=[_compile_reward], weights=[1.0])` (`minif2f.py:606`).
   - RE: `LeanTask.proved` `@vf.reward(weight=1.0)` (`RP/lean_common/lean_common/task.py:149`). It refuses to grade anywhere except the fresh verifier runtime prepared by `stage_verifier` (`task.py:145`). `sorry`, `native_decide` and added axioms are rejected (`prompt.py:36-40`).
   - Metrics: compiler_use, proof_growth.
8. **Reward density.** Final.
9. **Infra.** Prime VM sandboxes are mandatory (`RP/lean_common/lean_common/env.py:10-21`); HF / GitHub statement sources pinned by commit (`RE/lean/minif2f/minif2f/taskset.py:13`).
10. **Isolation.** Strongest in the corpus. `LeanEnv(IsolatedVerifierEnv)` collects proof artifacts, destroys the solver box, and grades in a fresh box from the same image (`env.py:24-26`; `VF/envs/isolated_verifier/env.py:1-7, 60-105`).
11. **Port.**
    - `RP/lean_common/` (Apache-2.0): image, checker, statement normalization in `statements.py`.
    - `VF/envs/isolated_verifier/env.py` (MIT): the fresh-box hidden-verifier protocol.

### F8. Codebase search / repo QA in a sandbox
1. **Family.** PE 6. Representatives: `triton_codebase_search`, `vllm_code_search`, `ktor`, `nextjs_codebase_search`.
2. **State.** Read-only cloned repo in a Prime sandbox (`ubuntu:22.04` plus `git clone`, `triton_codebase_search.py:305-341`).
3. **Actions.** A `run_bash_command` tool, with `sandbox_id` injected via `StatefulToolEnv.add_tool(args_to_skip=["sandbox_id"])` and `update_tool_args` (`triton_codebase_search.py:216, 265, 346`).
4. **Actors.** Single plus judge.
5. **Turns.** Multi-turn, `max_turns` default 5; remaining turns are shown in tool output (`:416-419`).
6. **Observations.** Command stdout and a turn countdown.
7. **Grader.** `CodeSearchJudgeRubric(vf.JudgeRubric)`: judge 0.8 plus `efficiency_metric` 0.2 (`:437-443`).
8. **Reward density.** Final.
9. **Infra.** `prime_sandboxes.AsyncSandboxClient`, `CreateSandboxRequest`; judge API.
10. **Isolation.** One sandbox per rollout; deleted in `destroy_sandbox` (`:359-373`).
11. **Port.** `PE/triton_codebase_search/triton_codebase_search.py`: the sandbox lifecycle, injected-arg tool, and judge-plus-efficiency rubric. Apache-2.0.

### F9. SWE: repo repair with hidden tests
1. **Family.** PE 1, RE 18. Representatives: PE `mini_swe_agent_bench`; RE `swe/r2e_gym`, `swe/swebench_verified`, `swe/pi_deepswe`, `swe/multiswe`, `swe/swe_atlas_qna`.
2. **State.** Per-task Docker image with the repo at `/testbed`.
   - RE `r2e_gym` sets `image=row["docker_image"]`, `workdir`, and `resources=vf.TaskResources(cpu=4, memory=4, disk=10)` (`RE/swe/r2e_gym/r2e_gym/taskset.py:330-334`).
   - PE uses `minisweagent` `DockerEnvironment` (`mini_swe_agent_bench.py:282`).
3. **Actions.**
   - Shell commands. PE parses actions from text; the agent submits with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (`:190`).
   - RE: any harness (bash, codex, claude_code, mini_swe_agent, rlm, prime_agent).
4. **Actors.** Single solver. Some envs add a separate verifier box (DeepSWE, swe_sweep) or LLM rubric judges (`senior_swe_bench`).
5. **Turns.** Long horizon, bounded by `step_limit` / `max_turns` and stage timeouts (`TaskTimeout`, `VF/task.py:66-78`).
6. **Observations.** Command output.
7. **Grader.**
   - Hidden tests are archived host-side at setup and removed from the box (`r2e_gym/taskset.py:256-278`), restored at scoring (`:280-300`), and run with `run_tests.sh` to compute `solved` (`:233-237`).
   - Gold-patch `validate()` (`:239-254`).
   - Harbor tasks: `HarborTask.solved` reads `/logs/verifier/reward.json|txt` after `bash /tests/test.sh`. Stale reward files are wiped before staging (`VF/tasksets/harbor/taskset.py:327-347, 356-391`).
   - PE: `swebench.harness.run_evaluation` with rubric weights `[1.0, 0.0]` (`mini_swe_agent_bench.py:139, 197-226`).
8. **Reward density.** Final (binary, or keyed dicts from reward.json).
9. **Infra.** Prime sandboxes, Docker, Modal, E2B, Apptainer runtimes (`VF/runtimes/*`); Harbor Hub datasets; prebuilt public images (`swebench_verified/taskset.py:38-58`).
10. **Isolation.**
    - Hidden test stash and restore.
    - Patch capture in `finalize` before tests are restored (`r2e_gym:228-231`).
    - Separate verifier image (`pi_deepswe/taskset.py:79-150`).
    - Per-task network allow/block lists (`VF/task.py:110-117`).
11. **Port.**
    - `RE/swe/r2e_gym/r2e_gym/taskset.py`: hide/restore tests plus gold validation. Apache-2.0.
    - `VF/tasksets/harbor/taskset.py` (874 lines): the Harbor task.toml / tests / reward.json contract. MIT.

### F10. Terminal / sandbox agentic tasks
1. **Family.** PE 3, RE 10. Representatives: PE `dabstep`, `bixbench`, `arc_agi_tool`; RE `terminal/terminal_bench_2`, `terminal/terminal_lego`, `reasoning/uuid_ctf`, `reasoning/prolog`.
2. **State.** Sandbox filesystem. Harbor images; synthetic incident corpus (`uuid_ctf`); SWI-Prolog starter file (`prolog`); data files (`dabstep`: `StatefulToolEnv`, `run_python`, `dabstep.py:61`).
3. **Actions.** Sandbox bash or python via the harness.
4. **Actors.** Single.
5. **Turns.** Long horizon.
6. **Observations.** Command output.
7. **Grader.**
   - Harbor `tests/test.sh` writes `reward.txt` (`RE/terminal/terminal_lego/terminal_lego/taskset.py` docstring).
   - Answer file read back from the runtime: `/workspace/answer.json` (`uuid_ctf`).
   - Constraint check in Python against generator metadata, so editing facts cannot spoof the result (`prolog` docstring).
8. **Reward density.** Final.
9. **Infra.**
   - `terminal_bench_2` is an 8-line wrapper pinning `dataset="terminal-bench/terminal-bench-2"` (`RE/terminal/terminal_bench_2/terminal_bench_2/taskset.py:14-19`).
   - Harbor registry pinned per task by git commit (`primeintellect-ai-research-environments/registry.json`, `HARBOR.md`).
10. **Isolation.** Container per rollout; optional separate verifier box.
11. **Port.**
    - `RE/reasoning/uuid_ctf`: procedural, fully synthetic task generator.
    - `RE/reasoning/prolog`: 9 generator kinds with constraint verification.
    - Both Apache-2.0.

### F11. Search / RAG / deep research
1. **Family.** PE 8, RE 10. Representatives: PE `search_r1_ish`, `webexplorer`, `gutenberg_env`, `art_e`; RE `search/browsecomp_plus`, `search/deepdive`.
2. **State.** Search index:
   - BM25/E5 FAISS server: `PE/search_r1_ish/retrieval/retrieval_server.py:140-203`.
   - ChromaDB: `PE/gutenberg_env/gutenberg_env.py:114`.
   - SQLite email DB: `PE/art_e/art_e/utils/local_email_db.py`.
   - bm25s shared toolset: `RE/search/browsecomp_plus/browsecomp_plus/servers/search.py:47-60`.
   - Live web: Exa (`search_r1_ish.py:83-96`), Serper and Jina (`PE/webexplorer/webexplorer/tools.py:27,95`).
3. **Actions.**
   - v0: `vf.ToolEnv(tools=[...], max_turns=...)` (`search_r1_ish.py:197-201`; `webexplorer.py:112-118`, `max_turns=20`).
   - v1: MCP `Toolset` shared across rollouts (`SharedToolsetConfig`, `search.py:38`), or "bring your own search" through the harness flag `--env.agent.harness.search true` (`primeintellect-ai-research-environments/skills/evaluation/SKILL.md:185`; `RE/search/deepdive/deepdive/taskset.py:3-9`).
4. **Actors.** Single plus judge.
5. **Turns.** Multi-turn tool loop.
6. **Observations.** Search results, documents.
7. **Grader.**
   - LLM judge against gold: `vf.JudgeRubric` (`search_r1_ish.py:171-195`); `vf.ReferenceJudge` subclass (`RE/search/browsecomp_plus/browsecomp_plus/taskset.py:107-118`).
   - Retrieval recall and search-call metrics (`:120-136`).
   - Tool-count shaping `1 - calls/max_turns` (`PE/art_e/art_e/art_e.py:161-175`).
   - `RubricGroup([ToolRubric, JudgeRubric])` (`PE/congressional_records/congressional_records.py:368,386`).
8. **Reward density.** Final.
9. **Infra.** External search APIs (SERPER/EXA keys), Chroma server, local indices, judge API.
10. **Isolation.** Process-level; the v1 toolset can run on host, colocated, or in its own runtime (`VF/mcp/toolset.py:16-24`).
11. **Port.**
    - `RE/search/browsecomp_plus/browsecomp_plus/servers/search.py`: shared BM25 toolset with recall tracking on rollout state. Apache-2.0.
    - `PE/search_r1_ish/retrieval/retrieval_server.py`: BM25/dense retrieval server.

### F12. Tool-use / API / MCP / stateful DB worlds / simulated user
1. **Family.** PE 8, RE 10. Representatives: PE `tau_bench`, `skyrl_sql`, `mcp_universe`, `agent_dojo`, `med_agent_bench`; RE `tool_use/general_agent`, `tool_use/tau2_bench`, `tool_use/mcp_atlas`, `tool_use/bfcl_v3`, `tool_use/enterprise_ops_gym`.
2. **State.**
   - DB world: SQLite (`PE/skyrl_sql/skyrl_sql.py:104`).
   - JSON `TaskDB` mutated by per-task `tools.py` (`RE/tool_use/general_agent/general_agent/servers/toolset.py:25-70`).
   - tau domain env (`PE/tau_bench/tau_bench_env.py:155-176`).
   - FHIR EHR server over HTTP (`PE/med_agent_bench`).
   - Live MCP services in a container: `mcp_atlas`; enterprise MCP gyms from container images (`RE/tool_use/enterprise_ops_gym/enterprise_ops_gym/taskset.py:41-71`).
3. **Actions.**
   - OpenAI tool calls. v0: `ToolEnv` / `tool_defs` / custom `call_tool` (`PE/github_mcp/github_mcp.py:34,98` against the remote GitHub MCP at `:22-30`).
   - v1: MCP tools served per rollout by `vf.Toolset` (`@vf.tool`, `VF/utils/decorators.py:78-82`).
4. **Actors.**
   - Simulated user:
     - PE tau user LLM: `user_model="gpt-4.1-mini"` (`tau_bench_env.py:47`).
     - RE `TauEnv` with three seats: assistant (trainable), user, judge (untrainable) (`RP/tau_common/tau_common/env.py:8-44`).
     - BFCL scripted user turns replayed through one interaction (`RE/tool_use/bfcl_v3/bfcl_v3/taskset.py:208-222`).
   - Prompt-injection adversary: `PE/agent_dojo/agent_dojo.py:192, 256`.
5. **Turns.** Multi-turn. `max_turns`/`max_steps` (`TauConfig.max_steps`, `RP/tau_common/tau_common/taskset.py:18`); stop on `###STOP###` (`tau_bench_env.py:178-192`).
6. **Observations.** Tool results, user messages.
7. **Grader.**
   - Upstream reward function: `tau_env.calculate_reward()` (`tau_bench_env.py:120-131`); RE reads `trace.info["tau"]["reward_info"]` (`tau_common/taskset.py:35-37`).
   - DB-hash equality with a replayed gold chain, OR `verify(db)` (`general_agent/taskset.py:54-87`).
   - AST / multi-turn checkers (`bfcl_v3/taskset.py:23-24, 93-130`).
   - Claim-coverage LLM judge (`RE/tool_use/mcp_atlas/mcp_atlas/taskset.py:94-98`).
   - Execution-result match (`skyrl_sql.py:119-150`).
   - MCP-Universe evaluators (`PE/mcp_universe/mcp_universe.py:67-104`).
8. **Reward density.** Final. Per-step tool counts appear only as metrics (`tau_common/taskset.py:39-51`).
9. **Infra.**
   - Pinned upstream packages: tau2 by commit tarball (`RE/tool_use/tau2_bench/pyproject.toml:12`).
   - Docker images; Harbor corpus for general_agent (4,417 tasks).
   - User and judge LLM APIs.
10. **Isolation.**
    - Toolset placement (`colocated` / `runtime`) with files shipped inside the task data when sandboxed (`general_agent/taskset.py:39-42, 119-152`).
    - Per-rollout `vf.State` synced between toolset and task over a signed state channel (`VF/mcp/server.py:131-250`).
11. **Port.**
    - `RE/tool_use/general_agent` (Apache-2.0): TaskDB, per-task tools, and gold-chain hash/verify grading. Maps directly onto Crucible Apps plus hidden verifiers.
    - `RP/tau_common` (Apache-2.0): three-seat env with untrainable user and judge.
    - `RE/tool_use/bfcl_v3` checker dispatch.

### F13. Browser / GUI / device
1. **Family.** PE 2. Representatives: `web_voyager` (Selenium Chrome, `PE/web_voyager/web_voyager.py:16-21`), `androidworld` (Android emulators).
2. **State.** Live browser DOM; emulator pool (`PE/androidworld/utils/emulator_manager.py:34-91`).
3. **Actions.** Click/type/scroll tools.
   - `ToolEnv` in web_voyager (`:126-170`).
   - `StatefulToolEnv` with the env injected via `args_to_skip=["env"]` in androidworld (`androidworld.py:145, 204-206`).
4. **Actors.** Single.
5. **Turns.** Multi-turn: `max_turns` 15 (web_voyager), 10 (androidworld).
6. **Observations.** Annotated screenshots plus accessibility tree (`androidworld.py:246-251`; `PE/web_voyager/utils/utils_webarena.py`).
7. **Grader.**
   - Vision LLM judge over screenshots and final answer (`web_voyager.py:36-60`).
   - Task-native `task.is_successful(emulator_env)` (`androidworld.py:93-142`).
8. **Reward density.** Final.
9. **Infra.** Chrome/Selenium; Android SDK emulators (Docker/Modal mentioned in grep); vision judge.
10. **Isolation.** Emulator acquired per rollout and released after evaluation (`androidworld.py:127-138`).
11. **Port.** `PE/androidworld/utils/emulator_manager.py` (EmulatorPool acquire/release). Apache-2.0.

### F14. Multi-turn games / navigation / interactive fiction
1. **Family.** PE 7, RE 1. Representatives: PE `mastermind`, `hanabi`, `text_quests`, `wiki_race`, `balrog_bench`; RE `reasoning/wikispeedia`; VF `environments/wordle` (textarena), `environments/kuhn_poker`.
2. **State.** Game engine:
   - Python dict in `state`: `PE/mastermind/mastermind.py:302-311`.
   - Infocom z-machine with checkpoints: `PE/text_quests/text_quests.py:127, 219-222`.
   - Navigation state in a Toolset: `RE/reasoning/wikispeedia/wikispeedia/servers/wiki.py:18-29, 107-132`.
   - Live Wikipedia API: `PE/wiki_race/wiki_race.py:105`.
3. **Actions.**
   - Text moves parsed by `XMLParser(fields=["think","guess"])` (`mastermind.py:450-454`).
   - Tool calls: `hanabi` `action` tool with injected `game_state` / `player_id` (`hanabi/hanabi/hanabi.py:57, 196-216`); wikispeedia `click_link` / `go_back`.
4. **Actors.**
   - `hanabi`: co-players are the same model, called from `env_response` via `state["client"]` (`hanabi/hanabi/player.py:44`; `hanabi.py:296-330`).
   - v1 `kuhn_poker`: two seats with zero-sum rewards (`primeintellect-ai-verifiers/environments/kuhn_poker/kuhn_poker/taskset.py:87-157`).
   - textarena: engine plays the user (`VF/tasksets/textarena/taskset.py:1-8, 63-84`).
5. **Turns.**
   - `max_turns`; hanabi scales it by number of players (`hanabi.py:52`).
   - `@vf.stop` predicates (`mastermind.py:249-300`; `wikispeedia/taskset.py:45-48`).
   - INFINITE procedural tasksets (`wikispeedia/taskset.py:82`; kuhn `:159-166`).
6. **Observations.** Game feedback text.
7. **Grader.** Shaped multi-reward. `mastermind` uses solved 1.0, speed 0.5, partial feedback 0.3, candidate-space reduction 0.1, format 0.2, with name-based weight overrides (`mastermind.py:325-372, 463-482`). Others:
   - text_quests progress / score / completion / harm weights (`text_quests.py:343-448`).
   - fruit_box total reward normalized by expert (`fruit_box.py:350, 438-439`).
   - wikispeedia `reached_target` 1.0, efficiency rewards at weight 0.0 (`taskset.py:50-71`).
   - kuhn payoff via `trace.record_reward` in the env (`kuhn_poker/taskset.py:147-148`).
8. **Reward density.** All scored once per rollout. Per-step game rewards are accumulated in state (fruit_box `total_reward`, fh_aviary `cumulative_reward` at `fh_aviary.py:357,494`, balrog `episode_return` at `:263`) and summarized at the end.
9. **Infra.** Local engines; textarena; BALROG; Wikipedia API / SNAP graph.
10. **Isolation.** In-process.
11. **Port.**
    - Mastermind candidate-reduction shaping (`mastermind.py:347-372`). Apache-2.0.
    - `VF/tasksets/textarena/taskset.py`: engine-as-user env. MIT.
    - The kuhn_poker multi-seat env.

### F15. Long-context
1. **Family.** PE 1, RE 12. Representatives: PE `longbench_v2`; RE `long_context/mrcr_v2`, `oolong_synth`, `longcot_env`, `verbatim_copy`.
2. **State.** None (PE, prompt-stuffed MCQ), or a sandbox file: the context is uploaded to `/workspace/context.txt` (`RE/long_context/mrcr_v2/mrcr_v2/taskset.py:87-89`; `oolong_synth/taskset.py:102-104`).
3. **Actions.** Completion, or REPL/shell scanning of the context file.
4. **Actors.** Single (+ optional judge).
5. **Turns.** Single (PE / `verbatim_copy`) or multi.
6. **Observations.** Question; the context is read through tools.
7. **Grader.**
   - Reads `/workspace/answer.txt`, falling back to the last message, then applies the official metric: MRCR difflib ratio gated on a hash prefix (`mrcr_v2/taskset.py:91-107`), or Oolong rules with partial numeric credit, or an optional host-side judge.
   - `longcot` verify runs as a uv script in the runtime.
8. **Reward density.** Final.
9. **Infra.** Public GCS CSVs, HF datasets, runtime.
10. **Isolation.** Runtime filesystem.
11. **Port.** The context-file / answer-file protocol in `RE/long_context/oolong_synth`. Apache-2.0.

### F16. Simulated-user conversation (no tools)
1. **Family.** PE 3. Representatives: `spiral_bench`, `sycophancy_eval`, `aidanbench`. v1 has the built-in generic `user-sim` env (`VF/envs/user_sim/env.py`).
2. **State.** Conversation only; injections tracked in state (`PE/spiral_bench/spiral_bench/spiral_bench.py:42-56`).
3. **Actions.** Completions.
4. **Actors.** Policy plus a user LLM.
   - The user LLM is called inside `env_response` with roles flipped (`spiral_bench.py:46-87`).
   - Canned follow-ups: "Are you sure?" with `max_turns=2` (`PE/sycophancy_eval/sycophancy_eval.py:271-305`).
   - v1 `UserSimEnv`: the user is an untrainable `null`-harness agent that opens the conversation and ends it with `###DONE###`; the user's `max_turns` caps the conversation (`VF/envs/user_sim/env.py:38-99`).
5. **Turns.** Long multi-turn (spiral default `max_turns=41`).
6. **Observations.** User replies.
7. **Grader.** Chunked transcript judge plus a final full-transcript judge, aggregated (`spiral_bench.py:109-200`); flip / no-flip rewards (sycophancy).
8. **Reward density.** Final. Note that spiral judges per chunk of turns but sums to one scalar; this is the closest thing to per-turn grading in the corpus.
9. **Infra.** User and judge APIs.
10. **Isolation.** None.
11. **Port.**
    - `VF/envs/user_sim/env.py` (MIT): about 60 lines, a generic two-agent relay.
    - Spiral chunked judging (`PE/spiral_bench/spiral_bench/utils.py:77-94`, `spiral_bench.py:109-200`). Apache-2.0.

---

## 2. verifiers v0 API (reconstructed; source not in tree)

- **Entry point.** A module named after the env exposes `load_environment(**kwargs) -> vf.Environment` (`PE/mmlu/mmlu.py:55`). Optional entry point group `[project.entry-points."verifiers.environments"]`, e.g. `synlogic = "synlogic:load_environment"` (`PE/synlogic/pyproject.toml`).
- **Dataset contract.** HF `Dataset` with `prompt` (messages) or `question`, plus `answer`, `info`, `task`. Passed as `dataset=` / `eval_dataset=` with `system_prompt`, `parser`, `rubric` (`PE/mmlu/mmlu.py:83-116`, `PE/gauss/gauss.py:377`).
- **SingleTurnEnv.** One completion, then rubric (`mmlu.py:116`).
- **MultiTurnEnv.** Hooks (`AGENTS.md:91, 96-97`; never override `rollout()`):
  - `setup_state(state)`;
  - `env_response(messages, state) -> Messages`;
  - stop predicates via `@vf.stop` (newer) or `is_completed` (older);
  - built-in `max_turns_reached` / `prompt_too_long` (`mastermind.py:249-300`);
  - `post_rollout` (`humaneval.py:12`);
  - `state["final_env_response"]` (`autodiff.py:~174`).

  State keys `prompt`, `completion`, `trajectory`, `turn`, `info`, `answer` are framework-owned and treated as read-only (`AGENTS.md:117-120`). Later 0.1.x versions use typed `vf.UserMessage` / `vf.ToolMessage` (`tau_bench_env.py:242, 261`).
- **ToolEnv.** `tools=[python fns]`; schemas come from signatures and docstrings (`search_r1_ish.py:197-201`). Also `tool_defs=[vf.Tool]` (`tau_bench_env.py:66`; `mcp_universe.py:47-64`), overridable `call_tool` and `error_formatter` (`github_mcp.py:58, 98`).
- **StatefulToolEnv.** `add_tool(fn, args_to_skip=[...])` plus `update_tool_args(...)` inject hidden per-rollout arguments such as sandbox ids or game state (`AGENTS.md:93`; `hanabi.py:57, 196`; `triton…py:265, 346`).
- **SandboxEnv (Prime sandboxes).**
  - Constructor: `sandbox_name`, `docker_image`, `start_command` (`autodiff.py:98-105`).
  - Built-in `bash` tool, removable with `remove_tool(self.bash)` (`autodiff.py:~110`); `self.bash(cmd, sandbox_id=, sandbox_state=)` (`polars_env.py:129, 209`).
  - `self.sandbox_client.wait_for_creation` / `execute_command` (`humaneval.py:43-44`); `state["sandbox_id"]`.
  - The sandbox is destroyed after scoring (`humaneval.py:52` comment).
- **MCPEnv.** Referenced in `AGENTS.md:94`. Community envs instead wrap MCP inside `ToolEnv` (`github_mcp`, `mcp_universe`).
- **Rubric.**
  - `vf.Rubric(funcs=[...], weights=[...])` or `add_reward_func(fn, weight=)`; final reward is the weighted sum.
  - `reward_weights` and `get_reward_func_names()` allow overrides (`mastermind.py:463-482`).
  - Reward functions receive kwargs injected by name (`prompt`, `completion`, `answer`, `state`, `task`, `info`, `parser`) plus `Rubric.class_objects` (`AGENTS.md:102-105`).
  - Weight-0 functions act as metrics (`mini_swe_agent_bench.py:139`).
- **JudgeRubric.** `JudgeRubric(judge_client, judge_model, judge_prompt, judge_sampling_args, parser)`. Exposes an injectable `judge` callable (`llm_writer_negative_style.py:17-23`) and `rubric.judge_client` / `judge_model` (`gauss.py:276-311`).
- **RubricGroup.** `RubricGroup(rubrics=[...])` composes rubrics (`llm_writer_negative_style.py:66`; `congressional_records.py:386`).
- **ToolRubric.** `ToolRubric(tools=[...])` gives tool-call count metrics (`art_e.py:166`).
- **Custom rubric subclasses.** e.g. `kernelbench.py:132`, `web_voyager.py:36`.
- **Parsers.**
  - `vf.Parser` subclass with `parse` / `parse_answer` (`mmlu.py:12-52`).
  - `vf.XMLParser(fields, answer_field)` with `get_format_reward_func()` (`mastermind.py:450-474`).
  - `vf.ThinkParser(extract_fn)` (`seeclick.py:100`).
- **Execution model.** Everything runs in-process. The env object owns clients and pools (judge clients, MCP managers, emulator pools). Scoring runs once at the end of the rollout.

## 3. verifiers v1 API (`VF`)

- **TaskData** (`task.py:81-136`; frozen pydantic):
  - `idx`, `id`, `name`, `prompt` (str or Messages or None), `system_prompt`;
  - `image`, `workdir`, `agent_user`, `skills`;
  - `network_allow` / `network_block` (`:110-117`);
  - `artifacts` plus `artifact_max_bytes` (`:119-125`);
  - `timeout: TaskTimeout(setup, agent, finalize, scoring)` (`:66-78`);
  - `resources: TaskResources(cpu, memory, gpu, disk)` (`:51-63`).
- **Task\[Data, State, Config\]** (`task.py:148`):
  - `NEEDS_CONTAINER` (`:149`); `key` / `hash` (`:156-170`); `runtime_env()` (`:181`).
  - Hooks `setup(trace, runtime)` (`:185`), `finalize` (`:188`), `stage_verifier` (`:191`), and `validate(runtime)` for gold checks (`:195`); `defer_scoring` (`:199`).
  - `score()` (`:213-284`): runs metrics, then weighted rewards, then plugged judges. Signatures are injected by parameter name (`task`, `trace`, `runtime`). Runtime-dependent signals are skipped when scoring offline.
  - `hooks()` merges decorated methods with config-plugged functions (`:299-317`).
  - Per-rollout tool servers come from `toolsets(config)` (`:320-330`).
- **Taskset\[Task, Config\]** (`taskset.py:49`):
  - Abstract `load()`, which may be a lazy or infinite generator (`INFINITE`, `:50, 65-67`).
  - Views `include` / `exclude` / `shuffle` / `skip` / `take` / `select` (`:88-163`).
  - Taskset-wide shared `toolsets` (`:180-189`).
- **Decorators** (`utils/decorators.py`):
  - `@vf.tool(name=)` (`:78`);
  - `@vf.stop(priority)`: predicate over Request/Response/Trace (`:89`);
  - `@vf.intercept`: rewrite request/response (`:99`);
  - `@vf.metric`: unweighted (`:109`);
  - `@vf.reward(weight=1.0, priority=0)`: weighted, returns a float or a keyed dict (`:124-131`).
- **TaskConfig** (`configs/task.py:35-63`): `judges` (plugged), `stops`, `metrics`, `rewards` (function import paths, weight overrides).
- **Judge** (`judge.py:118`): override `build_messages` / `parse` and optional structured `schema`. Calls are recorded in `trace.info["judge_calls"]` and `trace.extra_usage`.
  - `JudgeConfig(model, sampling, weight, prompt)` (`configs/judge.py:13-23`).
  - Built-ins `ReferenceJudge` (`judges/reference.py`) and `RubricJudge` (`judges/rubric.py`).
- **Toolset** (`mcp/toolset.py:27-34`, `mcp/server.py:131-358`): an MCP server class.
  - Hooks: `setup` (once), `setup_task(task)` (per rollout), typed `self.state` synced to the trace.
  - Placement: `ToolsetConfig(colocated, runtime, url)` / `SharedToolsetConfig` (`:16-24`).
- **Harness** (`harness.py:40-317`). A harness is the agent program.
  - Flags: `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_TOOL_INTERCEPTION`, `SUPPORTS_RESUME`, `EXECUTES_CODE`, `SUPPORTS_SKILLS`, `NEEDS_CONTAINER` (`:41-65`).
  - Methods: `setup(runtime)`, `session()` (`:195`), `score()` for harness metrics (`:224`), `resume()` (`:240`), `cleanup` (`:290`), abstract `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data)` (`:294-316`).
  - `HarnessSession.turn()` / `steer()` (`:319-397`).
  - Built-ins in `harnesses/`: bash, null, browser_use, claude_code, codex, hermes_agent, kimi_code, mini_swe_agent, openclaw, pi, prime_agent, rlm, terminus_2.
- **Interception contract.**
  - Every model call goes through a localhost OpenAI/Anthropic-compatible proxy that records turns into the trace graph and enforces limits and `@stop`s (`interception/server.py:1-20`).
  - Tool calls can be gated by `POST /tool` (`:434, 494`); this is the one tool gate.
  - Limits `max_turns`, `max_input_tokens`, `max_output_tokens`, `max_total_tokens` (`session.py:87-104`; `configs/agent.py:45-52`).
- **Env** (`env.py:79`):
  - `setup(agents)` (`:152`), `run(task, agents)` (`:157`), `finalize(task, episode)` for cross-agent scoring (`:166`), `complete`, `start` / `stop`.
  - Seats are `AgentConfig` fields carrying `harness`, `runtime` (default `PrimeConfig`), `model`, `sampling`, limits, `timeout`, `retries` (`configs/agent.py:30-56`).
  - Agent API: `agent.run` / `interaction` / `provision` and `trainable` (`agent.py:578, 649, 739, 848`).
  - Built-in envs: `SingleAgentEnv` (`envs/single_agent/env.py:23`), `UserSimEnv`, `BestOfNEnv` (`envs/best_of_n/env.py:26-44`), `IsolatedVerifierEnv`, `AgenticJudgeEnv` / `SharedAgenticJudgeEnv` / `IsolatedAgenticJudgeEnv` (`envs/agentic_judge/env.py:283-362`).
- **Runtime** (`runtimes/base.py:138-436`):
  - Methods: `start`, `stop`, `run`, `run_program`, `open_process`, `run_background`, `prepare_uv_script`, `run_uv_script`, `read`, `write`, `prepare_execution(routes)` (network policy), `expose(port)`.
  - Implementations: subprocess, docker/podman, apptainer, modal, e2b, prime.
  - No snapshot/restore API.
- **Rollout lifecycle** (`rollout.py:255-320, 510-600`):
  1. boot;
  2. `task.setup`, then `harness.setup` (one shared setup deadline);
  3. build toolsets, serve interception and tools;
  4. `prepare_execution` (network policy);
  5. harness session;
  6. `task.finalize`;
  7. collect artifacts;
  8. `task.score` + `harness.score`;
  9. `harness.cleanup`;
  10. teardown.
- **Errors** (`errors.py`):
  - `RolloutError` (`:30`) with subtypes `ProviderError` (`:34`), `HarnessError` (`:50`), `ToolsetError` (`:54`), `EnvError` (`:58`), `SandboxError` (`:64`), `TaskError` (`:68`; raised by task setup/finalize/reward/metric code), `InterceptionError` / `TunnelError` (`:72-77`).
  - `boundary()` attributes untyped exceptions to the boundary they crossed (`:81-95`).
  - `stop_condition` produces names like `"task_error"` (`:98-103`).
  - A failed rollout is recorded as data, not a crash.
- **Trace format** (`trace.py`, `TRACE_VERSION = 1` at `:38`):
  - `Trace` (`:380-433`): `id`, `verifiers` version, `task: TraceTask(type, data, key, hash)` (`:118-128`), `agent: AgentInfo(config, runtime, name, trainable)` (`:107-115`), `tools`.
  - Message graph: `nodes` (MessageNode graph), `calls: [ModelCall(model, sampling, endpoint, finish_reason, usage, time, error, policy)]` (`:153-175`).
  - Rewrites, scores and status: `request_rewrites` / `response_rewrites`; `rewards: {name: Reward(score, weight)}` where `value = score*weight` (`:135-141`); `metrics`; `info`; `extra_usage`; `is_completed`, `ok`, `stop_condition`, `is_timeout`; `errors`; `timing`.
  - `trace.reward` is the sum of weighted values (`:473`).
  - `branches` are root-to-leaf paths, each one training sample, with token ids, masks, logprobs and advantages (`:192-375`).
  - Written to `traces.jsonl` (`task.py:3-5`).
- **Built-in importer tasksets:** `tasksets/harbor`, `nemo_gym`, `openenv`, `textarena`.

## 4. Packaging and versioning

- **v0 (prime-environments).**
  - `environments/<slug>/pyproject.toml` with name, version, description and `tags` (enforced by `primeintellect-ai-prime-environments/tests/test_envs.py:27-46`), `verifiers>=0.1.x`, and a hatch include of the module.
  - Example: `PE/mastermind/pyproject.toml`, version 0.1.2, `verifiers>=0.1.11.dev0`.
  - The env must import as a package named after its directory (`test_envs.py` "import {env_dir.name}"). A README and an `outputs/` folder are required.
  - Workflow: `prime env init`, `uv run vf-install`, `uv run vf-eval -s <slug>` (`AGENTS.md:28-41`). Git deps go in `tool.uv.sources`.
- **v1 (research-environments).**
  - Each env is a package exporting exactly one `Taskset` subclass via `__all__`, optionally plus a `Harness` (becomes the default harness) and/or an `Env` (becomes the default env) (`VF/utils/loaders.py:99-126, 153-181`).
  - Example: `RE/search/browsecomp_plus/browsecomp_plus/__init__.py:1-5`.
  - The taskset id resolves to a module name. Environments Hub ids `owner/name@version` are stripped to `name` (`loaders.py:70-78`).
  - Shared code ships as local packages via `tool.uv.sources` (`RE/tool_use/tau2_bench/pyproject.toml:15-16`). `tags` include `"v1"`.
  - CI installs each env in a fresh venv and runs one capped rollout in Prime sandboxes (`primeintellect-ai-research-environments/tests/test_envs.py`).
  - Dataset revisions are pinned inside the taskset config (`aime25/taskset.py:40`). Harbor tasks are pinned by git commit in `registry.json`.
  - verifiers itself is versioned by git tags (hatch-vcs, `primeintellect-ai-verifiers/pyproject.toml:1-10`).

## 5. prime-agent (agent/harness design, Rust, MIT)

- **Product.** "Self-improving RLM harness" (`PA/README.md`).
  - Exposes the tools `bash`, `edit` and `ipython`, plus a persistent Python REPL kernel API: `rlm.spawn` (subagents), `collect`, `rlm.harness` CRUD, `agent_message`, `compact`, `goal`, `refine` (`PA/AGENTS.md`, "Surface contract").
  - Continual-harness state: prompts, memories, skills and subagent specs, refined by `/refine` with rollback snapshots.
  - Python kernel shim: `PA/prime-agent-runtime/src/rlm/harness.py`.
  - The README warns it is "not a security sandbox", so it must run inside an external sandbox.
- **Crates** (`PA/AGENTS.md` Crates table): pa-types, pa-agent (provider-independent loop), pa-ai (providers), pa-core (session engine, tools, kernel), pa-daemon (supervisor, ACP), pa-tui, pa-cli.
- **Loop** (`PA/crates/pa-agent`):
  - `AgentTool` trait: `name`, `description`, `parameters` JSON schema, async `execute`, `execution_mode` (`src/types.rs:491-513`).
  - `AgentEvent` stream: Turn, Message and ToolExecution start/update/end (`:527-570`).
  - `BeforeToolCallResult{block, reason}`: a tool gate (`:573-577`).
  - `AfterToolCallResult{…, terminate}` (`:581-591`).
  - `AgentLoopConfig` hooks: `should_stop_after_turn` / `before_turn`, steering / follow-up / continuation messages, `before_tool_call` / `after_tool_call`, `tool_execution` mode (`src/agent_loop/mod.rs:92-113`).
- **Integration with verifiers.**
  - `PrimeAgentHarness(ACPHarness)` drives prime-agent over the Agent Client Protocol (ACP). It is pinned to release 0.9.5 / a commit and installed with SHA256-verified tarballs, with MCP and skills support (`VF/harnesses/prime_agent/harness.py:1-60, 97-100`).
  - The ACP runner has a `ToolGate` (`VF/acp/runner.py:42-81`).
  - `PA/examples/catch` is a Gym-style `reset`/`step` env for the Decision API (real-time control).

---

## 6. Requirements an environment framework must meet to express all of these

| # | Primitive | Families needing it |
|---|---|---|
| R1 | **Single-turn prompt → completion → checker task**, with no tools, no sandbox and no harness loop (v0 SingleTurnEnv / v1 null harness with 1 turn). Crucible currently lacks this. | F1, F2, F3, F4, F5, F15 (≈60 envs) |
| R2 | **Dataset-backed task sampling**: lazy, pinned-revision HF/Harbor/git loaders; infinite procedural generators; include/exclude/shuffle/take views; stable task keys | all; INFINITE in F14, F10 (uuid_ctf, prolog) |
| R3 | **Output parsers** (XML/think/boxed/MCQ letter/JSON) plus a format-reward helper | F1, F2, F5, F14 |
| R4 | **Rubric composition**: named reward functions with weights summed to a scalar; weight-0 metrics; config-time weight overrides; keyed multi-score returns; group-of-rubrics | F1, F3, F6, F8, F11, F14 (mastermind/text_quests), F16 |
| R5 | **LLM-judge primitive with pinning**: model, sampling and prompt-file in config; verdict-label parsing that *errors* rather than scoring 0; judge usage recorded separately from policy usage; multi-judge ensembles and structured output | F1, F3, F4, F8, F11, F12 (mcp_atlas), F13, F16, F9 (senior_swe rubric judges) |
| R6 | **Multi-turn env loop with env_response and stop predicates** (`max_turns`, token caps, custom `@stop`, done markers like `###STOP###`) | F6, F7, F12, F14, F16 |
| R7 | **Stateful tool envs**: per-rollout state; hidden injected tool args (sandbox_id, game_state); tools generated from code per task (general_agent `tools.py`); state readable by graders | F8, F10, F12, F13, F14 |
| R8 | **MCP tools**: per-rollout and taskset-shared tool servers; external/remote MCP servers; placement on host, colocated, or own runtime | F11, F12 |
| R9 | **Sandbox lifecycle**: per-task image/workdir/resources (cpu/mem/GPU/disk); create, exec, read/write files, destroy; pools (emulators); multiple providers (Prime, Docker, Modal, E2B, Apptainer, subprocess) | F6, F7, F8, F9, F10, F13, F15 |
| R10 | **Hidden verifiers and isolation**: stash tests away from the agent and restore at scoring; wipe stale reward files; artifact collection; **fresh verifier box** after destroying the solver box; separate verifier images; gold-solution `validate()` | F6 (forth_lang, nl2repobench), F7, F9, F10 |
| R11 | **Harbor task contract importer**: task.toml, instruction.md, tests/test.sh, `/logs/verifier/reward.{json,txt}`, `[verifier].environment_mode=separate`, collect hooks | F9, F10 (22 RE envs) |
| R12 | **Network policy per task** (allow/block lists, verifier-only routes) | F1 (network_allow=[]), F6, F9, F10, F11 |
| R13 | **Stage timeouts** (setup, agent, finalize, scoring) and retries; errors attributed to a boundary and recorded as data | F6, F7, F9, F10, F12 |
| R14 | **Simulated actors**: user LLM (untrainable), scripted users (BFCL replay, canned prompts), adversarial injections, same-model co-players, judges as seats, trainable flags | F12, F14, F16 |
| R15 | **Multi-agent episodes with cross-agent finalize** (zero-sum payoffs, best-of-n, pass@n, agentic judge) | F14 (kuhn, hanabi), F12 (tau three-seat), F9 (agentic judge) |
| R16 | **Per-turn / accumulated reward**: per-step game rewards, progress and harm counters, chunked transcript judging. Currently reduced to one scalar in both APIs. Crucible's dense milestone reward is a superset, but importers must collapse to final scalars for parity | F14 (fruit_box, text_quests, balrog, fh_aviary), F16 (spiral), F12 metrics |
| R17 | **Harness-agnostic tasks plus a pluggable harness**: bash, null, codex, claude_code, mini_swe_agent, rlm, prime_agent; skills install; bring-your-own search; a tool gate on each call | F9, F10, F11, F15, all v1 |
| R18 | **Model-call interception and trace recording**: message graph, branches as training samples, per-call usage, request/response rewrite hooks, trace versioning | all v1 (training) |
| R19 | **Multimodal observations**: image content parts in prompts; screenshots and accessibility trees per turn | F5, F13, F1 (hle_diamond, mmmu) |
| R20 | **Context-as-file protocol**: upload large context into the sandbox and grade an answer file, with fallback to the last message | F15, F10 |
| R21 | **Engine state checkpointing** (text_quests `save_checkpoint`). Neither verifiers API offers runtime snapshot/restore, so Crucible's snapshot/restore exceeds what these require | F14 |
| R22 | **Packaging/registry**: one package per env exporting a loader (v0 `load_environment` / v1 Taskset `__all__`), versioned pyproject, hub ids `owner/name@version`, pinned upstream deps; **a v0 importer is needed for the 109 PE envs** | all |

### Gaps against Crucible
- **R1:** no single-turn task type.
- **R22:** no v0 importer.
- **R3 and R4:** parser plus weighted-rubric composition. Crucible needs v0 Rubric/JudgeRubric/RubricGroup equivalents.
- **R11:** Harbor contract. Its v1 importer covers it only if `HarborTaskset` semantics, including separate-verifier boxes, are supported.
