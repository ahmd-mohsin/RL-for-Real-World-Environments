# 04 A dense, realistic World: enterprise tools, computer use, simulated people and time

*Written 2026-10-29. Not committed.*

**Purpose.** Crucible's World (DESIGN §4.4, P19) already has Apps, Actors, a Clock and Events, SQL state checks with a collateral check, identity and RBAC, a desktop App with one `computer` tool, QEMU VM pools, an MCP-server App, a browser App and grounded simulated actors. This note surveys how other systems build dense, realistic worlds for agents and what Crucible should add, so that RL tasks come out of a World that looks like a real company. It covers:

- enterprise tools (chat, docs, email, calendar, CRM, ticketing);
- computer use (CUA) through Harbor;
- simulated people and time;
- worlds built from a customer's own data.

**How this was checked.**
- Five research passes on 2026-10-29 (enterprise sims; web, OS and MCP envs; CUA infra and app stand-ins; tool and environment synthesis; people, time, data and metrics) opened arXiv abs and HTML pages, GitHub repos and vendor docs.
- All 102 arXiv ids cited below were then re-checked against the arXiv API on 2026-10-29. Each id returned the title given here.
- Full-text reads covered about the first 100k characters of each paper, so details that only appear in late appendices may be missing.

**Legend.**
- **[paper]**: stated in the paper.
- **[repo]**: stated in the repo or docs.
- **[search]**: seen only in a search digest, primary source not opened.
- **[inferred]**: my reading.
- **UNVERIFIED**: not confirmed.
- All summaries are in my own words.

**PDFs downloaded** to `crucible/docs/research/papers/` (git-excluded). ScaleEnv, `2602.06820.pdf`, was already there.

| File | Paper | Why it matters for the World |
|---|---|---|
| `2407.18901.pdf` | AppWorld | Dense DB world, in-memory reset, collateral-damage tests |
| `2506.07982.pdf` | τ²-bench | User with its own tools and DB; compositional task generator |
| `2412.14161.pdf` | TheAgentCompany | Real self-hosted apps plus LLM colleagues; checkpoints |
| `2509.17158.pdf` | ARE (Gaia2 platform) | Event DAG, simulated time, notifications, write-action verifier |
| `2510.25726.pdf` | Toolathlon | Local containers instead of SaaS; state-init scripts |
| `2408.04682.pdf` | ToolSandbox | Stateful tools, milestones and minefields, user knowledge boundary |
| `2509.13311.pdf` | AgentScaler | Tools → domains → DB schema → Python tools at 1,000+ domains |
| `2602.10090.pdf` | Agent World Model | 1,000 SQLite+MCP envs; executable-vs-LLM-simulated ablation |

---

## 0. TL;DR

1. **Three kinds of world exist, and the best systems mix them.**
   - **Real self-hosted software**: TheAgentCompany, WorkArena, SCUBA, WebArena, Toolathlon's local apps.
   - **Executable mocks over a relational DB**: AppWorld, τ/τ², ToolSandbox, EnterpriseOps-Gym, WorkBench, ClawsBench, AWM, AgentScaler, EnvScaler, ScaleEnv.
   - **LLM-simulated environments**: Simia, OccuBench, Kimi K2's tool simulator, DreamGym, Qwen-AgentWorld.
   - Real software gives realism but resets slowly; executable mocks give cheap exact resets and state-diff rewards.
   - LLM simulation gives coverage, at the cost of consistency.
2. **Executable beats simulated wherever the domain is CRUD-like.** AWM's matched ablation (same tasks and tools, GPT-5 as the simulator) lost at every model size: up to 13.4 points on BFCLv3 and 8.9 on τ² pass@1 [paper]. Simulation won once (Simia, OfficeBench RL: 64.7 sim vs 60.8 real) [paper], and an independent paper later found the released Simia-RL checkpoint overfit [paper, 2604.17739].
3. **Reward should be state-based, with a collateral check.**
   - AppWorld (expected vs allowed changes, everything else fails) and EnterpriseOps-Gym (SQL verifiers for goal, integrity, permissions and side effects) set the bar. Crucible already has both, as `SqlAssertion` and `Collateral`.
   - Gaia2 adds causal order and timing to write-action matching.
   - AWM's code-plus-judge reward beat code-only and judge-only.
4. **Simulated users are the weakest part of every environment.**
   - τ² simulator critical-error rates are 13% (airline), 12% (retail) and 6% (telecom) [paper]. A large human study (2603.11245, 451 people) finds LLM users over-cooperative.
   - GAUGE (2609.12191) found that 57.5% of conversations rated "satisfied" had failed the task [paper].
   - Tying users to their own tools and observable state (τ² dual control) is the one fix shown to cut errors.
5. **Time is the least-solved axis.**
   - Gaia2's Time split: GPT-5 (high) scores 0.0 [paper].
   - ARE is the reference design: an event DAG, a clock that pauses during inference, `wait_for_notification` jumping from event to event, and a notification channel separate from tool output.
   - Crucible's step-mode clock plus `on_tool` triggers covers part of this. It lacks notification policies, event DAG dependencies, and a timing tolerance in the verifier.
6. **For CUA, Harbor is the integration point. Harbor treats tools as part of the agent, not the environment.**
   - An environment can offer tools only as MCP servers, skills, APIs or CLIs.
   - Harbor's `computer-1` agent runs Xvfb + XFCE + noVNC + Chromium inside the task sandbox through `exec`, and it has an OSWorld adapter (QEMU in Docker) and adapters for theagentcompany and crmarena.
   - Crucible's `common/desktop` App already uses the same action vocabulary.
7. **For enterprise apps, the priority stack is:**
   - First, native executable SQL Apps (cheap, exact, snapshot-able) for email, calendar, chat, docs, tickets and CRM.
   - Second, a small set of real OSS stand-ins behind MCP for realism and CUA: Mattermost or Rocket.Chat, Plane, Gitea or GitLab, Mailpit or Stalwart, Radicale, Outline, Zammad, Twenty or ERPNext.
   - LLM simulation only for long-tail tools and for people, grounded in a "worldbook".
8. **Customer data enters as schema plus statistics plus documents, never as raw rows in a training World.**
   - Presidio redaction, then relational synthesis (SDV HMA or MOSTLY AI's TabularARGN), with a DP option (AIM for tables, Aug-PE for text).
   - Tasks are then grounded on the synthetic copy (EnterpriseBench's persona → O*NET goal template pipeline; CRMArena's latent variables).
   - Traces rebuild environments: Terminal-Universe replays file operations for 37.3k envs [paper]; Trace2Env compiles a worldbook when the system is gone.
9. **No standard realism metric exists.**
   - Papers report counts (apps, tools, tables, rows, people), per-task density (apps per task, unique APIs, calls), environment bug rates, and transfer to real benchmarks.
   - §7 proposes a "World density card" that Crucible can compute for every World.

---

## 1. Enterprise and workplace simulation environments

### 1.1 Summary table

Kind: **R** = real self-hosted or SaaS software, **M** = executable mock over a DB or files, **L** = LLM-simulated.

| Env (arXiv) | Apps simulated | Kind | Seed / reset | Verifier | Scale | Best reported |
|---|---|---|---|---|---|---|
| TheAgentCompany (2412.14161) | GitLab, ownCloud, Plane, RocketChat; LLM colleagues (Sotopia) | R + L people | docker-compose services with mock data; one Docker image per task with `init.sh` | Weighted checkpoints, mostly Python state checks; LLM judge on 51 tasks as a fallback | 175 tasks | Gemini 2.5 Pro 30.3% full, 39.3% partial [paper] |
| TheMCPCompany (2510.19286) | Same apps as MCP (18,505 tools from REST); real Azure | R | Pre-loaded Docker apps; Terraform per Azure task | TAC checkpoints | 175 + 17 Azure | GPT-5 with retrieval 42.29% [repo/paper] |
| WorkArena (2403.07718) / ++ (2407.05291) | Real ServiceNow developer instances | R | Installer pins instance config; fresh user per task, deleted after; 10 fictitious brands | Hand-coded Playwright oracle + validator over REST/JS APIs | 33 atomic; 682 compositional (L2/L3) | GPT-4o 42.7% L1, 3.0% L2, 0.0% L3; humans 93.9% [paper] |
| CRMArena (2411.02305) / Pro (2505.18878) | Real Salesforce org filled with LLM data | R | Data generated with latent variables, uploaded once | Exact match on IDs, F1, judge for refusals | 1,170 / 4,280 queries; Pro 29k B2B, 55k B2C records | Pro: gemini-2.5-pro 54–58% single-turn, about 30–35% multi-turn [paper] |
| SCUBA (2509.26506) | Real Salesforce sandbox org | R | Config snapshot, diff after each run, revert only what changed | Rule-based checks via Metadata/Tooling APIs + milestones | 300 tasks | GPT-5 browser agent 51.3% [paper] |
| EnterpriseOps-Gym (2603.13594) | Email, Calendar, Teams, Drive, CSM, HR, ITSM, Hybrid | M (Docker MCP per domain) | One SQL snapshot per DB | SQL verifiers: goal, integrity, permissions, side effects | 164 tables, 512 tools, 1,150 tasks (30 infeasible) | Claude Opus 4.5 37.4% [paper] |
| EnterpriseBench, Fujitsu (2510.27287) | Chat, mail, code, CRM, ITSM, forum, HR, policies | M | JSON/DB records, about 46.9k; RBAC levels 9–14 | LLM judge, 1–5 rubric | 500 tasks | best 41.8% [paper] |
| EnterpriseBench Corecraft (2602.16179) | Customer-support org | M | 2,500+ entities, 14 types, 23 tools | Expert rubrics | UNVERIFIED | RL on GLM-4.6: 25.37 → 36.76% held-out; +7.4 τ²-Retail, +6.8 Toolathlon [paper] |
| AppWorld (2407.18901) | 9 apps (Gmail-, Venmo-, Amazon-, Spotify-like, phone, files...) | M (FastAPI + SQLite in-process) | Copy base DB + per-task setup; freezegun time; load < 0.5 s | State-diff unit tests: expected + allowed changes, else collateral damage | 457 APIs, 101 tables, about 370K rows, 106 people, 750 tasks | GPT-4o ReAct TGC 48.8 / 30.2 [paper] |
| τ-bench (2406.12045) / τ² (2506.07982) | Retail, airline, telecom (+τ³ domains in repo) | M + L user | JSON DBs per domain; τ² user has own DB and tools | Final DB = goal DB; assertions; pass^k | τ: 165 tasks; τ²: 2,285 generated, 114 sampled for telecom | τ² telecom: Claude-3.7-Sonnet 49% pass^1 [paper] |
| ToolSandbox (2408.04682) | 34 stateful tools, 11 domains | M + L user | Execution context DBs | Milestone DAG; minefields zero the score | 1,032 scenarios | GPT-4o 73.0 [paper] |
| OfficeBench (2407.19056) | Word, Excel, PDF, .ics calendar, .eml email, OCR, shell | M (Docker + Python libs) | Per-task synthetic files | Exact, fuzzy and execution-based file checks | 300 tasks (112 three-app) | GPT-4o 47.0% (21.4% three-app) [paper] |
| WorkBench (2405.00823) | 5 sandbox DBs (email, calendar...) | M | DB reset | Outcome-centric DB comparison | 26 tools, 690 tasks | GPT-4 43% [paper] |
| ClawsBench (2604.05172) | Mock Gmail, Slack, Calendar, Docs, Drive | M (REST) | Deterministic snapshot/restore; conformance tests against the real APIs | Task checks + separate safety grading | 44 tasks | 39–64%; unsafe actions 7–33% [paper] |
| OccuBench (2604.10866) | 100 scenarios, 10 industries | L | LLM is the environment; fault injection | Judge | 100 scenarios | Finds good agents are not good simulators [paper] |
| EnterpriseClawBench (2606.23654) | Tasks recovered from proprietary workplace sessions | R-derived | Recovered fixtures | Hard rules + rubrics | 852 tasks | 0.663 best [paper]; data not released |
| SWE-Lancer (2502.12115) | Expensify repo | R | One offline image, checkout per task | Playwright E2E tests, triple-verified | 1,488 tasks, $1M | Claude 3.5 Sonnet 26.2% Diamond IC [paper] |
| GDPval (2510.04374) | None; one-shot deliverables | – | Reference files | Blinded expert pairwise comparison; auto-grader agrees 66% vs 71% human-human | 1,320 tasks, 44 occupations | Claude Opus 4.1 wins or ties 47.6% [paper] |

Others found:
- **AgentMercury** (2608.20634) builds a persistent business world first (entities, services, tools, invariants across services) and draws tasks from it: 4,783 environments [paper].
- **EntWorld** (2601.17722) reverse-engineers schemas and logic of OSS CRM/ITSM/ERP apps such as EspoCRM [search].
- **ITBench** (IBM) and **OdysseyBench** were not checked (ids UNVERIFIED).

### 1.2 Mechanisms worth copying

- **Collateral damage as failure** (AppWorld). Grading only the expected rows rewards sloppy agents; the start-to-end DB diff must be fully explained. Crucible has this (`Collateral`).
- **A fresh identity per episode** (WorkArena). A throwaway user account isolates episodes on a shared live service, so no full reset is needed. This is a cheap pattern for real OSS apps such as Mattermost, Plane and Zammad, which all support user creation via API [inferred].
- **Diff-and-revert reset** (SCUBA). Snapshot the config, diff after the run, and revert only what changed. This is faster than a full reset for heavy apps.
- **Per-task init scripts over one base image** (TheAgentCompany, Toolathlon 72 of 108 tasks, OSWorld). This avoids gigabyte per-task snapshots.
- **Latent variables behind generated data** (CRMArena). Hidden traits such as a customer's complaint habit or an agent's skill shape the records but are never uploaded, so the data has real causal structure and ground truth can be computed from the latents.
- **Infeasible and confidential tasks.**
  - EnterpriseOps-Gym has 30 infeasible tasks; the best refusal rate is 53.9%.
  - CRMArena-Pro has 80 confidentiality probes per org; refusal is near 0–2.9%.
  - Both are cheap to add and expose a large, real weakness.
- **Composed oracles** (WorkArena++, τ² generator). Atomic tasks carry init, solution and assertion parts and compose into thousands of checked tasks. τ² checks that assertions fail before the solution and pass after.
- **Checkpoint partial credit** (TheAgentCompany): 0.5·(points/total) + 0.5·full. A useful dense signal, but it differs from AppWorld's all-or-nothing.

### 1.3 Limits the papers state

- Real SaaS (ServiceNow, Salesforce, Notion, Google) cannot be copied locally. Resets are slow and runs drift; Toolathlon reports weak consistency across runs.
- Mocks have no UI (AppWorld, τ). CUA training needs a UI over the same state.
- Task authoring is expensive: 4–6 person-hours per Toolathlon task, 3–5 expert-hours per MCPMark task, 1+2 person-hours per OSWorld task (start state plus checker).
- Main results often assume oracle tool retrieval (EnterpriseOps-Gym).

---

## 2. Web, OS, mobile and MCP environments

| Env (arXiv) | Hosted apps | Seed / reset | Verifier | Scale | Best / notes |
|---|---|---|---|---|---|
| WebArena (2307.13854) | OneStopShop (Magento), Magento admin, Postmill (Reddit), GitLab, Kiwix Wikipedia, OpenStreetMap | Docker stop/rm/run; map is a separate 1 TB EC2 machine | Answer match + URL and state checks (some LLM fuzzy) | 812 tasks | Automatic reset takes about 5 min; tasks share state, so only partial parallelism [repo, AgentLab] |
| WebArena-Verified (no arXiv; ServiceNow repo) | Same, smaller images | Control port `GET /status`, `POST /init` per site | Deterministic; structured answer + captured HAR network trace; no LLM judge | 812 audited; hard subset 137 (paper) vs 258 (repo) | Offline re-scoring from HAR [repo] |
| WebArena-Lite (via 2408.06327) | Same | Same | Human-checked subset | 165 tasks | Common RL eval; others used for training |
| WebArena-Infinity (repo only) | Generated web apps from real product manuals (e.g. Gmail and Linear settings) | Seed data per app | Programmatic verifiers | 19 envs in manifest | Pipeline runs Claude Code with permission checks off [repo] |
| VisualWebArena (2401.13649) | Classifieds, Shopping, Reddit | Per-site reset | exact / must_include / fuzzy / VQA / SSIM | 910 tasks | GPT-4V+SoM 16.37% [paper] |
| WEBSERV (2510.16252) | WebArena-style sites | Incus copy-on-write containers: about 5× faster launch, about 240× less storage, 200+ envs per host | – | – | RL Qwen3-4B 55.5% on a WebArena-Lite variant [paper] |
| REAL (2504.11543) | 11 clones (Airbnb-, Amazon-, mail-, calendar-like...) | State in localStorage; `/clear` resets, `/finish` returns the state diff; fixed data and clock; injected latency and errors | Key-value state checks; LLM rubric for info tasks | 112 tasks | Claude 3.7 Sonnet 41.07% [paper] |
| WebGym (2601.02439) | About 300k tasks on live sites | – | Rubrics | – | RL Qwen3-VL-8B 26.2 → 42.9% OOD [paper] |
| InfiniteWeb (2601.04126), GUI-GENESIS (2602.14093), Recreated websites (2603.10505), WebForge (2604.10988) | Generated sites or apps with verifiers | Generated | Code verifiers | GUI-GENESIS 149 apps | GUI-GENESIS: synthetic 42.28% vs real-app training 40.94% [search] |
| OSWorld (2404.07972) | Ubuntu VM: Chrome, VLC, Thunderbird, VS Code, LibreOffice, GIMP | VM snapshot + per-task file downloads and commands | 134 evaluation functions over files, configs, a11y tree | 369 tasks | Human 72.36% [paper] |
| OSWorld-Verified (blog) | Same | Docker/QEMU 8–16 envs per server; AWS up to 50 parallel; images 50 → 25 GB | About 300 fixes: drift, anti-crawl, dates, tolerances | 361/369 | Full run from 10+ h to minutes [blog] |
| OSWorld 2.0 (2606.29537) | Multi-hour workflows on realistic files and profiles | – | – | 108 workflows | Claude Opus 4.8 20.6% at 500 steps [paper] |
| OSWorld-MCP (2510.24563) | OSWorld + 158 MCP tools | – | Success + tool-invocation rate | – | Best invocation rate about 36% [paper] |
| Windows Agent Arena (2409.08264) | Windows 11 in QEMU/KVM in Docker; 11 programs | Snapshot built by the user (licensing) | OSWorld-style state checks | 154 tasks | Full parallel run about 20 min on Azure [paper] |
| AndroidWorld (2405.14573) | 20 F-Droid apps | Seeded random params; DB clears; **device clock fixed at 2023-10-15 15:34 UTC** | adb reads of SQLite, files, settings | 116 tasks | Human 80.0% [paper] |
| Toolathlon (2510.25726) | 32 apps, 604 tools. Real: Notion, GitHub, Google, Snowflake, HF, W&B. Local: Poste.io, Canvas, WooCommerce, K8s | State-init scripts clear and refill; one container per task | Deterministic script per task, no judge | 108 tasks, about 70 tools per task | 10 workers ran all 108 in about 70 min on 16 CPU / 64 GB; Claude-4.5-Sonnet 38.6% [paper/repo] |
| MCPMark (2509.24002) | Notion (9 templates), GitHub (real org), Postgres (chinook, employees...), Playwright on WebArena sites, filesystem | Template copies; source vs eval hub keys; reset after check | Script per task (about 210 lines) | 127 tasks, 38 start states | gpt-5-medium 52.56% pass@1, 33.86% pass^4 [paper] |
| MCP-Universe (2508.14704) | 11 real servers, 133 tools | Live | 84 evaluators, 48 dynamic | 231 tasks | GPT-5 43.72% [paper] |
| MCP-Bench (2508.20453) | 28 live servers, 250 tools | Live | Rules + o4-mini rubric | 104 tasks | Same model writes and judges tasks [paper] |
| LiveMCPBench (2508.01780) | 70 servers, 527 tools | Docker, keyless | DeepSeek-V3 judge, about 79–81% agreement with humans | 95 tasks | No end-state check [paper] |
| MCP-Atlas (2602.00933) | 36 production servers, 220 tools | Containerized | Claim-level rubric | 1,000 tasks | 63.3% of failures are reasoning errors [paper] |
| MCPWorld (2506.07672) | White-box OSS apps | Containers | Code instrumentation inside the app | 201 tasks | API, GUI and hybrid agents [paper] |
| BrowseComp-Plus (2508.06600) | Fixed corpus, 100,195 docs | Static | gpt-4.1 judge | 830 queries | Fixed corpus removes live-web drift [paper] |

**Lessons for the World.**
- Seed real services from templates or init scripts, then reset by restart or by revert. Prefer local containers to SaaS: Toolathlon says local hosting is what allows many accounts and complex states.
- **Capture what verifiers need at the network layer.** WebArena-Verified's HAR traces let runs be re-scored offline. Crucible's choke point (P25) already records tool calls; for browser Apps, record HAR too.
- **Fix the clock everywhere.** AndroidWorld pins the device clock, REAL fixes timestamps, AppWorld freezes `now`. OSWorld-Verified spent much of its effort on date drift and live-site drift.
- **Shared-state tasks kill parallelism** (WebArena). One episode per instance, or per-episode identities and row scoping.

---

## 3. CUA infrastructure

### 3.1 Harbor (`github.com/harbor-framework/harbor`; old `laude-institute/harbor` redirects; Apache-2.0) [repo]

**Task format.**
- A task is a directory holding `instruction.md`, `task.toml`, `environment/` (Dockerfile, `docker-compose.yaml`, or `Apptainer.def`), `solution/solve.sh` and `tests/test.sh`.
- The test writes `/logs/verifier/reward.txt` or a multi-key `reward.json`.
- Also supported: multi-step tasks (`steps/step-N/`), a separate verifier sandbox, artifacts, network policy, `harbor lint`, and Rewardkit (programmatic plus judge criteria in TOML).

**The `task.toml` `[environment]` table.**
- It holds image, os (Linux or Windows), cpus, memory, storage, gpus, gpu_types, allow_internet, healthcheck, skills_dir and **`[[environment.mcp_servers]]`**.
- Each MCP server entry takes name, transport (stdio, sse or streamable-http), url, headers with `${VAR}` templating, and command and args.
- The docs show an MCP server as a compose sidecar mocking external services. About 30 agents forward MCP config; only claude-code and codex forward HTTP headers.

**Tools belong to the agent.**
- An environment exposes tools only as MCP servers, skills, APIs or CLIs. An outside agent sees only `BaseEnvironment` primitives (exec, file transfer).
- **So a Crucible World that wants Harbor agents to see Slack- or CRM-like tools must ship them as MCP sidecars in the task's compose file** [inferred from the docs].

**Sandbox providers.**
- About 35 providers: docker, podman, apple-container, singularity, daytona, e2b, modal, runloop, gke, ec2, cua-cloud, hyperbrowser, use-computer, prime and others.
- Compose works on 17 of them. Windows runs on docker, daytona, cua-cloud and use-computer.

**CUA.** The built-in `computer-1` agent starts Xvfb + XFCE + x11vnc/noVNC + Chromium inside the task environment and acts through `xdotool` and ImageMagick over `exec`, so it works on any provider.
- Its providers include native Anthropic, Bedrock, Gemini and OpenAI, plus Tinker.
- Coordinate spaces: native pixels, 0–999 normalised, and Anthropic-scaled.
- Reset relaunches Chromium with a clean profile.

**Adapters.**
- About 90, including osworld (upstream qcow2 under QEMU in Docker), theagentcompany, crmarena, tau3-bench, gaia2, swelancer, bfcl and terminal-bench@2.0.
- The registry is Harbor Hub, or a custom `registry.json`.

**RL.**
- Trajectories are saved in ATIF.
- The verified trainer integration is SkyRL's `harbor_generator.py`, which drives Harbor Trials. Harbor has no trainer.

### 3.2 VMs and sandboxes

| System | Mechanism | Snapshot / fork | Speed claims | Notes |
|---|---|---|---|---|
| OSWorld providers | vmware, virtualbox, docker (QEMU/KVM), aws, azure, gcp, aliyun, daytona, modal... | vmware `revertToSnapshot`; docker revert = remove + restart from qcow2; aws relaunch from AMI | No per-task reset time published (UNVERIFIED) | `run_multienv --num_envs N` |
| E2B | Firecracker microVMs resumed from snapshot | Pause keeps FS + memory; one snapshot seeds many sandboxes | No official ms figure; about 1 s resume from a competitor's post (UNVERIFIED) | `e2b-desktop`: Xfce + stream, mouse, keyboard, window APIs |
| Daytona | Containers or VMs | Cold (FS) or hot (FS + memory) snapshots; warm pools; no live fork | README claims < 90 ms start [repo] | **Public repo unmaintained since June 2026**; AGPL |
| Modal | gVisor sandboxes | FS snapshot → Image; memory snapshots alpha: terminate the source, no GPU, same instance type, 7-day TTL [docs] | – | – |
| Runloop | Devboxes, Blueprints | Disk snapshots that fork; suspend keeps disk only | – | – |
| trycua/cua | Cross-OS CUA sandboxes | – | – | MIT; Harbor provider `cua-cloud` |
| WEBSERV | Incus block-level copy-on-write | Fork per episode | 200+ envs per host [paper] | Best published density for web envs |
| Crucible `common/vm` | QEMU/KVM Debian | `savevm`/`loadvm` | About 2.5 s start, about 1.1 s reset on the lab (DESIGN) | Exact RAM + disk snapshot; already ahead of most published numbers |

**Browser sandboxes.**
- Playwright MCP (Microsoft, Apache-2.0); Steel (self-hostable, Apache-2.0); Browserbase (hosted; Stagehand and an MCP server are open).
- BrowserGym's action set has element-id actions (`click(bid)`, `fill`, `select_option`...), coordinate actions, navigation, `send_msg_to_user` and `report_infeasible`.

### 3.3 CUA action spaces

| Source | Actions |
|---|---|
| Anthropic `computer_toolset_20260801` [docs] | screenshot, zoom, left/right/middle/double/triple click, left_click_drag, mouse_move, left_mouse_down/up, cursor_position, scroll, type, key, hold_key, wait |
| OpenAI `computer` [docs] | click, double_click, drag, move, scroll, keypress, type, wait, screenshot; one call carries an ordered `actions` array |
| OSWorld `computer_13` / `pyautogui` | MOVE_TO, CLICK, MOUSE_DOWN/UP, RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN/UP, HOTKEY, WAIT, FAIL, DONE; or raw pyautogui code |
| UI-TARS (2501.12326) | click, left_double, right_single, drag, hotkey, type, scroll, wait, finished; mobile adds long_press, open_app, press_home, press_back |
| BrowserGym | element-id plus coordinate plus navigation (above) |
| Observations | screenshot, accessibility tree, both, or Set-of-Mark (2310.11441) |

Crucible's `computer` tool covers the Anthropic/OSWorld union. Two additions:
- **batched actions per call** (OpenAI style);
- **an accessibility-tree observation** (Daytona exposes one; OSWorld uses it).

`FAIL`/`report_infeasible` should exist as a terminal action, so infeasible tasks can be graded.

### 3.4 CUA RL at scale

| Work (arXiv) | Env scale | Throughput / mechanism | Result |
|---|---|---|---|
| ComputerRL (2508.14040) | "Thousands" of qemu-in-docker Ubuntu VMs over gRPC; concurrency 1024 in its table | API-GUI hybrid: 103 LLM-generated APIs over 6 apps | GLM-4.1V-9B 48.9% OSWorld |
| UI-TARS-2 (2509.02544) | Several thousand cloud VMs (Windows, Ubuntu, Android); VM manager at several thousand QPS | Browser sandbox with time acceleration and checkpointing | OSWorld 47.5, WAA 50.6, AndroidWorld 73.3 |
| ARPO (2505.16282) | 256 parallel OSWorld VMs, 8 rollouts per task | Batch about 19 min at 256 envs; epoch 6 h → 1.2 h | UI-TARS-1.5-7B 23.5 → 29.9% |
| DART-GUI (2509.23866) | 180 Ubuntu containers on K8s | Decoupled actors: 22.6 → 43.6 actions/min; env utilisation 12.2 → 67.7% | OSWorld 42.13% |
| ZeroGUI (2505.23762) | Ubuntu + Android | GPT-4o tasks, VLM-voted rewards | 17.7 → 20.2 OSWorld |
| OpenCUA (2508.09123) | 200+ apps, 3 OSes (data) | – | 45.0% OSWorld-Verified (72B) |

**Determinism.** None of these papers reports reset latency or a determinism metric. Reset is always a fresh VM or container from an image.
- **[inferred]** For RL this means two things:
  - CUA throughput is bound by environment utilisation, so decoupled rollout matters more than raw VM count (DART-GUI).
  - Determinism has to be built in: fixed clock, fixed fonts and resolution, no network, pinned app versions. OSWorld-Verified's fix list (timing, fonts, date drift) is the checklist.

---

## 4. Simulated apps and tools

### 4.1 Open-source stand-ins for enterprise apps

| Role | Stand-in (license) | API | Seeding | MCP server | Used by |
|---|---|---|---|---|---|
| Slack | **Mattermost** (AGPL source, MIT binaries) | REST v4 + WebSocket | `mmctl import` bulk JSONL, `sampledata` | **Official** (Agents plugin `mcpserver/`) | – |
| Slack | **Rocket.Chat** (MIT except `ee/`) | REST + DDP | Data restore | Community only | TheAgentCompany |
| Slack | Zulip (Apache-2.0) | REST | `populate_db`; Slack/Mattermost importers | Community | – |
| Notion | **Outline** (BSL 1.1, not OSI) | RPC POST API | Markdown/Notion/Confluence import | **Built-in** `/mcp` | – |
| Notion | AFFiNE (MIT except backend) | GraphQL | – | Community | – |
| Notion | Docmost (AGPL) | REST | – | Community | – |
| Notion | AppFlowy (AGPL) | – | – | – | **AppFlowy-Cloud archived Sept 2026; weak choice** |
| Email | **Mailpit** (MIT) | REST + SMTP catch-all | Send over SMTP | Community | – |
| Email + calendar + contacts | **Stalwart** (AGPL) | JMAP, IMAP, SMTP, **CalDAV, CardDAV**, WebDAV | Admin API | Community (JMAP) | – |
| Email | GreenMail (Apache-2.0) | In-memory SMTP/IMAP/POP3 | API | – | – |
| Email | Poste.io | SMTP/IMAP | Pre-created accounts | – | Toolathlon |
| Email | MailHog | – | – | – | **Unmaintained** since 2024 |
| Calendar | **Radicale** (GPL-3.0) | CalDAV/CardDAV | **Write .ics/.vcf files** | Generic `caldav-mcp` | – |
| Calendar | Baikal (GPL), Nextcloud Calendar (AGPL) | CalDAV | – | – | – |
| Jira | **Plane** (AGPL) | REST v1 | Data restore | **Official** `plane-mcp-server` (MIT) | TheAgentCompany |
| Jira | OpenProject (GPL) | REST v3 HAL | Demo seeders | Built-in (tier limits UNVERIFIED) | – |
| Jira | Redmine (GPL-2), Taiga (MPL) | REST | `load_default_data`, `sample_data` | Community | – |
| Code host | **Gitea** (MIT) / Forgejo (GPL) | REST v1 | API | **Official** gitea-mcp | – |
| Code host | GitLab CE (MIT) | REST v4 + GraphQL | Backup restore | **Built-in** `/api/v4/mcp` (GA in 19.5) | WebArena, TAC |
| CRM | **Twenty** (AGPL) | GraphQL + REST | API | **Built-in**, alpha | – |
| CRM | SuiteCRM, EspoCRM (AGPL) | JSON:API / REST | – | Tiny community | EntWorld (EspoCRM) [search] |
| ERP | **ERPNext** (GPL) on Frappe (MIT) | REST `/api/resource` | Fixtures, demo | **Official** `frappe/mcp` | – |
| ERP | Odoo Community (LGPL) | XML/JSON-RPC | Demo-data flag | Community (`mcp-odoo`) | – |
| Ticketing | **Zammad** (AGPL) | REST | `auto_wizard.json` | Community | – |
| Ticketing | GLPI (GPL), osTicket (GPL-2), FreeScout (AGPL) | Limited REST | – | Small or none | – |
| Files | Nextcloud (AGPL) | WebDAV, OCS, CalDAV | – | Community | – |
| Files | ownCloud / oCIS (Apache-2.0) | WebDAV + LibreGraph | – | None | TAC (classic ownCloud) |
| Office | ONLYOFFICE (AGPL), Collabora (MPL) | WOPI | – | DocSpace MCP | – |
| Wiki | BookStack (MIT), Wiki.js (AGPL) | REST / GraphQL | – | Community | – |
| AWS | moto (Apache-2.0) | AWS APIs | – | – | **LocalStack OSS archived March 2026** |
| API mocks | Prism, WireMock, Mockoon | OpenAPI mock, record/playback, stateful scenarios | – | – | – |
| Gmail / GCal APIs | **No emulator found** (UNVERIFIED negative) | – | – | – | Use Stalwart or Mailpit + Radicale behind a Gmail-shaped MCP |
| Slack API fakes | slack-mock (dormant), slack-server-mock (small) | – | – | – | Weak; build our own |

**Licensing note [inferred].**
- Crucible ships Worlds as images, so AGPL apps (Mattermost source, Plane, Twenty, Zammad, Stalwart) are fine to run unmodified in a container.
- Modifying them and offering them to customers over a network triggers AGPL source obligations. Keep them unmodified and do all adaptation in an MCP sidecar.
- Outline's BSL limits commercial hosting.

### 4.2 LLM-simulated vs executable tools: the evidence

| Work (arXiv) | Tools | Executed or simulated | Consistency control | Scale | Downstream |
|---|---|---|---|---|---|
| AWM (2602.10090) | LLM-written SQLite schema + seed + Python MCP server | **Executed** | Execute and repair, up to 5 retries (about 87–88% succeed); DB-diff + GPT-5 judge reward | 1,000 envs, 35,062 tools, 10k tasks | 8B: BFCLv3 53.8 → 65.9; τ² 26.4 → 33.5. **GPT-5-simulated env scored lower at every size** [paper] |
| AgentScaler (2509.13311) | 30k+ APIs → tool graph → Louvain → 1,000+ domains → DB schema from tool params → Python tools | Executed | Final DB = gold; exact match for read-only tasks | 1,000+ domains | 30B-A3B: τ² 70.2/60.0/55.3 [paper] |
| ScaleEnv (2602.06820) | Code agent + test agent + debug agent | Executed | Per-column exact, fuzzy or exempt DB match | 16 domains, about 50 tools/env | See note 01 |
| EnvScaler (2601.05808) | Python class per env (state = attributes, tools = methods) | Executed | Test and checking agents; envs < 0.85 dropped | 191 envs, 18.6 tools/env | Qwen3-8B τ 38.2 → 44.8 [paper] |
| EnvFactory (2605.18703) | Verified stateful envs | Executed | Few, well-verified | 85 envs, 842 tools | +15% BFCLv3 [paper/search] |
| Simia (2511.01824) | o4-mini simulates tool outputs **and** judges reward | **Simulated** | Spec anchoring, JSON repair | 90k SFT trajectories from 5k seeds | OfficeBench RL: sim 64.7 vs real 60.8 [paper]; overfitting critique in 2604.17739 |
| Kimi K2 (2507.20534) | 3,000+ real MCP + 20,000+ synthetic tools | Stateful simulator for general tools; **real sandboxes for code** | Rubric judge, rejection sampling | 10k+ concurrent K8s sandboxes for SWE | – |
| StableToolBench (2403.07714) / MirrorAPI (2503.20527) | Cache of real responses + simulator on miss; MirrorAPI trains simulators on 7,000+ APIs | Hybrid | Cache first | 16k APIs | – |
| GTM (2512.04535) | 1.5B universal tool simulator | Simulated | – | 20k tools, 300 domains | Numbers not verified |
| Qwen-AgentWorld (2606.24597) | LMs trained on 10M+ real trajectories to predict next state | Learned simulator | RL with rubric + rule rewards | 7 domains | Sim + real beats real alone (claim) |
| DreamGym (2511.03773) | Reasoning experience model | Simulated | Replay buffer seeded with real data | – | Synthetic only matches about 80k real interactions [search] |
| TRUSTEE (2604.17739) | 8B model simulates everything | Simulated | Adaptive curriculum | – | Beats baselines "in most cases" [paper] |
| OccuBench (2604.10866) | LLM environment simulator | Simulated | Fault injection | 100 scenarios | Good agents ≠ good simulators |

**Trade-off [inferred from the table].**
- **Fidelity.** Executable wins when state matters: the only matched ablation (AWM) favours code. Simulators drift over long horizons, which is why Trace2Env adds persistent state and a worldbook.
- **Coverage and cost to build.** Simulation wins. Executable building fails about 12% of the time (AWM). AgentMercury shows world authoring is itself a skill: fine-tuning on construction traces raised build success from 3.3% to 83.3% [paper].
- **Reward integrity.** When the simulator also judges (Simia-RL), reward hacking and overfitting follow. Never let the component that produces observations also grade.
- **Determinism and cost per step.** Executable: exact, cheap, snapshot-able. Simulated: needs temperature 0 plus a cache to replay at all, and costs one LLM call per tool call.
- **Rule for Crucible.**
  - CRUD systems of record are executable.
  - Long-tail read-only tools may be simulated, behind a cache, grounded on a worldbook, and with any write effects mirrored into a real state store so verifiers never read LLM text.
  - People are LLMs bounded by tools and records.

### 4.3 Generating DB-backed tools from schemas

- **The research recipe** (AWM, AgentScaler, ScaleEnv, EnvScaler):
  1. A scenario or domain.
  2. A schema: SQLite DDL, or Pydantic models derived from tool parameters.
  3. Seed data chosen so that every task is feasible.
  4. Tool code (Python CRUD plus business rules) with execute-and-repair loops and unit tests.
  5. Tasks built solution-first: sample a tool chain, execute it to get the gold end state, then write the instruction.
- **Off-the-shelf generators.**
  - FastMCP's `from_openapi()` makes an MCP proxy from an OpenAPI spec. Its own docs warn that hand-curated servers work better than auto-converted ones for large APIs.
  - openapi-mcp-generator (TypeScript, Zod validation, `x-mcp` endpoint filter).
  - Speakeasy.
  - Google MCP Toolbox for Databases (`list_tables`, `execute_sql`, and parameterised SQL tools in `tools.yaml`).
  - PostgREST and Hasura derive REST and GraphQL from a Postgres schema.
  - Frappe's official MCP turns any Frappe doctype into tools.
- **Recommendation [inferred].** Generate a CRUD layer automatically (PostgREST-style, or SQL templates), then curate a task-facing tool surface on top of it: business verbs such as `approve_invoice` and `escalate_ticket`, with policy checks in code.
  - The generated layer is the State Adapter's and verifier's view.
  - The curated layer is the policy's view.
  - This is the split EnterpriseOps-Gym and τ use (domain tools, not raw SQL).

---

## 5. Simulated people and time

### 5.1 User simulators: what is known

| Evidence (arXiv) | Finding |
|---|---|
| τ²-bench (2506.07982) | Simulator error rates (critical / benign): airline 13/34%, retail 12/28%, telecom 6/10%. The tool-grounded telecom user errs least. Stop tokens `###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###` |
| τ-bench (2406.12045) | pass^k shows unreliability: gpt-4o retail pass^8 < 25% |
| ToolSandbox (2408.04682) | A "knowledge boundary" plus demos cut simulator hallucination from 12.4% to 6.97% |
| Mind the Sim2Real Gap (2603.11245) | 451 humans, 165 tasks, 31 simulators. Simulators are over-cooperative and uniform in style, with no realistic frustration; this inflates agent success. Stronger LLMs are not more faithful users. Defines the User-Sim Index |
| Lost in Simulation (2601.17087) | Agent success moves up to 9 points depending on which LLM plays the user; dialect groups are simulated worst |
| Non-Collaborative Users (2509.23124) | Unavailable requests, off-topic turns, impatience and fragments break tool agents |
| LLMs Get Lost in Multi-Turn (2505.06120) | Revealing the request in shards costs 39% on average |
| SimulatorArena (2510.05444) | Profile-conditioned simulators reach ρ about 0.7 with human ratings |
| MirrorBench (2601.08118) | Scores human-likeness separately from task success, with human-human controls |
| UserBench (2507.22034), UserRL (2509.19736) | Gym envs where preferences are revealed gradually; open simulators can stand in for GPT-4o |
| CRMArena-Pro (2505.18878) | The user holds the domain knowledge and reveals it gradually, so the agent must elicit it |
| **GAUGE (2609.12191)** | A persona user plus an LLM judge as a ranking gate. 57.5% of "satisfied" conversations had failed. Disagreement is under 1% for agents far apart in reward but 31% for close pairs. Advice: calibrate the judge against a verifiable reward first, and keep a judge-free completion bit |

**Judges in general.**
- One Token to Fool LLM-as-a-Judge (2507.08794): "master key" replies get false positives up to 35% on GPT-4o.
- ARE's LLM judge was gamed during RL and needed a sanity check.
- Agent-as-a-Judge (2410.10934) helps on code deliverables.

**Org simulation.**
- TheAgentCompany: 17 Sotopia NPCs with profiles, reached over chat, plus collaboration checkpoints.
- Concordia (2312.03664): a game master LLM turns actions into event statements and can call real APIs for simulated digital apps.
- Generative Agents and SOTOPIA (2310.11667) supply memory/reflection and social scoring.
- AgentSociety and OASIS show scale (10k to 1M agents), but they are social-media sims, not workplaces.

**What follows for Crucible [inferred].**
- **Bound every simulated person by tools and records**, not by prompts alone (τ² dual control). Crucible already has grounded responders and an actor's own tool use.
- **Default simulated colleagues to be less cooperative than an LLM wants to be.** Give each a sampled profile: responsiveness, patience, whether they know the answer, typical delay, and error rate. 2603.11245 shows the default LLM user is "easy mode".
- **Never let a user-sim or satisfaction judge be the reward.** Every reward needs a state-based bit, which is GAUGE's tripwire. Crucible's fidelity module already measures fabrication; add GAUGE-style rank agreement and τ²-style critical-error audits as gates on any World that uses LLM people.

### 5.2 Time

| System | Clock model | Events / async |
|---|---|---|
| **ARE (2509.17158) / Gaia2 (2602.11964)** | "Generation time" mode: the clock pauses during LLM inference, then advances by the measured generation time; an instant mode charges 1 s per action. `get_current_time()` tool | Event DAG (`depends_on(parent, delay_seconds)`); AGENT/ENV/USER events, ConditionCheck, validation and oracle events; time-ordered queue on its own thread; immutable event log. `wait_for_notification(timeout)` jumps from event to event. Notifications are a separate channel injected before each step, with a verbosity policy (LOW user only, MEDIUM email/chat/calendar, HIGH all). Verifier checks causal order and a timing tolerance around oracle times. Noise: tool failures p=0.1, 10 irrelevant events/min |
| AppWorld | freezegun pins `now` to each task's own datetime | None |
| τ-bench airline | The policy text states the current time | None |
| AndroidWorld | Device clock fixed | None |
| Vending-Bench (2502.15840) | Each tool costs simulated time (5 min to 5 h); `wait_for_next_day` | Overnight sales and supplier emails arrive the next morning; deliveries days later; runs of 2,000 messages over hundreds of simulated days |
| TimeArena (2402.05733) | Actions have durations; agents and objects are busy while acting | Parallel tasks compete for resources |
| libfaketime | LD_PRELOAD on time calls: absolute, offset, or accelerated rate | Not for static binaries (e.g. Go built without cgo) [inferred]; can also seed `getrandom` for tests |
| SimPy pattern | `env.now`, `timeout`, `run(until)` | Discrete-event jumping, like ARE |

**Gaps vs Crucible's clock (`world/clock.py`: wall or step mode, `at`/`after`/`every`/`on_tool` triggers, nl/tool/permission actions).**
1. **Notification channel and policy.** Arrivals (mail, chat, calendar reminders, ticket updates) should reach the policy as notifications before the next step, at a configurable verbosity. They should not only sit in an inbox to be polled.
2. **Event DAG dependencies and condition events.** Example: "vendor replies 2 h *after* the clerk's email". Crucible has `on_tool`; add `depends_on` with a delay, and condition checks over state.
3. **A `wait`/`wait_for_notification` tool that jumps virtual time** to the next event, plus `get_current_time`.
4. **Timing in the verifier.** A write must land inside a window relative to its causal parent (Gaia2). This makes deadline tasks ("reply within 1 business day") gradable.
5. **One time source for every App.**
   - Native Apps read the World clock.
   - Real OSS apps in containers need libfaketime or an app-level "now" setting. Otherwise Mattermost and Plane stamp wall-clock times and the snapshot is not deterministic.
   - Desktop/VM Apps need the guest clock set from the World clock at restore.
6. **Business calendars.** Working hours, weekends, holidays and time zones per simulated person. Needed for realistic delays and SLA tasks [inferred; no paper found that does this well].

### 5.3 Realism checks for people and time

- **Simulator error audits**: τ²-style critical vs benign error labelling on a sample of conversations.
- **Rank agreement**: GAUGE-style comparison of a judge's ranking against the verifiable reward.
- **Human-likeness**: MirrorBench metrics, or USI where human data exists.
- **Fabrication rate**: grounded responders (Crucible already has this).
- **Event-timing realism**: compare the inter-arrival distributions of simulated replies with the customer's logs when they exist (§6).

---

## 6. Enterprise data → World

### 6.1 Pipeline components

| Stage | Tool / method | Notes |
|---|---|---|
| PII discovery and redaction | **Microsoft Presidio** (Analyzer: spaCy NER + regex + checksums + context; Anonymizer: redact, mask, hash, encrypt; image and structured modules) | Add custom recognizers for customer IDs |
| Relational synthesis | **SDV HMASynthesizer** (Gaussian copula per table, parent-child recursion; docs say it is tuned for about 5 tables and 1 level of depth); **MOSTLY AI TabularARGN** (2501.12012; DP variant 2508.06647), open SDK | SDV's depth limit matters for ERP schemas with 100+ tables. Synthesise core entity tables statistically and derive the rest by business rules [inferred] |
| DP tables | **AIM** (2201.12677), PrivBayes (SIGMOD 2014) | Workload-aware marginals with error bounds |
| DP text | DP fine-tuning (2210.14348); **Aug-PE** (2403.01749), Private Evolution via API only | For emails, tickets, chats |
| Fidelity scoring | NIST CRC / SDNist (k-marginal score 0–1000, propensity, PCA, privacy metrics) | Use as a gate before a synthetic copy is used |
| LLM enterprise data | EnterpriseBench (2510.27287): employees from a public dataset at a 4:3:2:1 role ratio; GPT-4o writes mail and chat grounded on records; records synchronised across sources; RBAC drafted by an LLM, checked by experts. CRMArena: latent variables. ARE: PersonaHub persona → unstructured text → decoded into each app's schema, with an app-dependency graph for cross-app consistency | Cross-app consistency is the hard part. ARE says complex cross-app dependencies are not yet handled |
| Data-grounded tasks | EnterpriseBench: domain → persona → retrieved context → O*NET goal template → subgoals → validate; experts kept 80%. τ² compositional atoms; WorkArena++ composed oracles | – |

### 6.2 Traces → environments

- **Terminal-Universe (2609.04148).**
  - Replays the recorded file operations of public agent trajectories to rebuild each workspace's start state; a completion agent fills in missing files and dependencies.
  - Re-poses the original task, then widens it (across codebases) and deepens it (multi-round with a simulated user).
  - 37.3k environments; SFT gains +11.9 on Terminal-Bench 2.1 [paper].
- **Trace2Env (2610.06100).** When the real system is gone, it compiles traces into a worldbook (schemas, grounded evidence, induced behaviour rules), and a world-model agent with persistent state plays the system. It beats prompt-only world models on fidelity and long-horizon consistency across 9 environments [paper].
- **EnterpriseClawBench (2606.23654).** Recovers 852 tasks and fixtures from real proprietary workplace sessions; only the method is released.
- **Record and replay** (VCR, Polly, mitmproxy server-replay, WireMock record mode). Exact-match replay breaks on any unseen request, which is why StableToolBench pairs a cache with a simulator.
- **Crucible already has a lot of this.** DESIGN mentions state reconstruction for agent sessions (editor calls and shell effects) and `grounded_on: traces/...` actors.
- **Missing [inferred]:**
  - **Rebuilding a DB-backed App state from API logs**: replay the logged writes against a synthetic base, keyed by entity.
  - **Learning reply-time distributions per role** from timestamps, for the Clock.

### 6.3 Recommended customer-data path [inferred]

1. **Ingest metadata first.** Schemas, API specs (OpenAPI), policy docs, ticket and email samples, and logs or traces with timestamps.
2. **Redact** with Presidio. A human reviews the recognizer config. Raw data never leaves the customer boundary.
3. **Synthesise.**
   - Tables: SDV or TabularARGN on core entities, DP when required.
   - Text: an LLM grounded on the synthetic records (EnterpriseBench), or Aug-PE when text must be DP.
   - Hidden latent traits per person and customer (CRMArena) so that behaviour is causal.
4. **Gate** on SDNist-style fidelity and privacy metrics, plus referential integrity, plus "every task feasible" (AWM).
5. **Build Apps** from the schema: a generated CRUD layer, plus curated business tools, plus policies as code.
6. **Build people** from the org chart and role docs. Reply-time and cooperativeness distributions come from the logs. Each person is grounded on records, with their own tools.
7. **Build tasks** from tickets and traces: Terminal-Universe-style re-posing of real episodes, plus τ²-style composition of atoms. Every task is checked solvable by an oracle and failing for a null agent (Crucible already has `null_agent_score`, `clean_world_solvable`).

---

## 7. Density and realism metrics

What papers report:

| Metric | Example values |
|---|---|
| Apps / tools per World | AppWorld 9 / 457; Gaia2 12 / 101; EnterpriseOps-Gym 8 domains / 512 tools / 164 tables; Toolathlon 32 / 604; TheMCPCompany 18,505 tools |
| Tools per env (synthetic sets) | τ 12.5, τ² 22.7, MCP-Universe 12.1, EnvScaler 18.6, AWM 35.1 (AWM's comparison table) |
| State size | AppWorld 101 tables, 726 columns, about 370K rows, 106 people; CRMArena-Pro 29k/55k records; Gaia2 about 400K raw tokens per universe; EnterpriseBench about 46.9k records |
| Per-task density | AppWorld Test-C: 2.0 apps (max 6), 10.5 unique APIs, 46.8 calls (max 649); EnterpriseOps-Gym about 24.9 tables touched per task; AWM 8.5 steps, 7.1 unique tools; Toolathlon about 70 tools offered per task |
| Cross-app fraction | OfficeBench 93/95/112 tasks over 1/2/3 apps; MCP-Bench 56/30/18 tasks over 1/2/3 servers |
| Horizon | OSWorld 2.0 multi-hour, 500 steps; Vending-Bench 2,000 messages |
| Environment quality | AWM: tasks blocked by env bugs 11.5–14% vs EnvScaler 47–57%; about 4% env errors during RL; LLM-judged feasibility and completeness 1–5 |
| Transfer to real benchmarks | AWM, AgentScaler, ScaleEnv, EnvScaler, Corecraft (+6.8 Toolathlon), GUI-GENESIS (synthetic ≥ real-app training), WildAWM (2610.26828; gains on 8 OOD benchmarks) |
| People realism | τ² simulator error rates; USI; MirrorBench; SimulatorArena ρ; GAUGE rank agreement |
| Reliability | pass^k (τ), pass@k vs pass^k (Toolathlon, MCPMark) |

**Gap.** No paper defines a standard cross-app workflow metric or rates realism against real API-call distributions. Crucible can lead here.

**Proposed "World density card"** [inferred]. Computed per World and per task family and stored with the World's manifest:
- **Breadth:** Apps, tools (policy-visible vs total), tables, rows, documents, simulated people, roles, RBAC rules.
- **Per-task use:** Apps touched, unique tools, calls, tables written, actors contacted, events fired, virtual time elapsed. Report the median and p90.
- **Coupling:** the share of tasks needing 2 or more Apps, information flowing across Apps (a value read in App A and written in B), and the share needing a person's reply.
- **Asynchrony:** the share of tasks with a timed event or a deadline.
- **Hard negatives:** the share of infeasible, confidential and policy-violating requests.
- **Health:** env-error rate in rollouts, oracle solve rate, null-agent rate, and pass^k spread.
- **Realism, when customer data exists:** the distance between the World's tool-call and entity distributions and the customer's logs (e.g. JS divergence over API endpoints and entity types), SDNist k-marginal on synthetic tables, and the gap between simulated and logged reply times.
- **Transfer:** held-out real benchmark deltas after training (Toolathlon, τ², EnterpriseOps-Gym, OSWorld-Verified).

---

## 8. Recommendations for Crucible

### 8.1 World architecture: four App tiers behind one choke point

All four tiers sit behind `World._call_tool` (P25), the State Adapters and the snapshot. Each tier is an App kind with one contract: a tool surface, a State Adapter, `identity(actor)`, `healthcheck`, `reset_hook`, snapshot/restore, and clock injection.

| Tier | What | Fidelity | Reset | Use for |
|---|---|---|---|---|
| **T1 Native executable Apps** | SQLite/Postgres schema + generated CRUD + curated business tools (Python MCP), like AppWorld, EnterpriseOps-Gym, AWM | Exact state, no UI (or a thin generated UI via `website`) | Copy DB or `loadvm`, under 0.5 s | Bulk RL: email, calendar, chat, docs, tickets, CRM, HR, ERP |
| **T2 Real OSS stand-ins** | Mattermost, Plane, Gitea/GitLab, Stalwart or Mailpit + Radicale, Outline, Zammad, Twenty or ERPNext in containers, with official MCP where present | Real behaviour and UI; Harbor/CUA ready | Per-episode identity + row scoping, diff-and-revert (SCUBA), or container CoW (WEBSERV) | Realism checks, CUA tasks, held-out eval, sim-to-real checks of T1 |
| **T3 LLM-simulated tools** | Long-tail read-only APIs, grounded on a worldbook (Trace2Env), cached (StableToolBench); writes mirrored into a T1 store | Approximate | Cache + journal replay | Coverage only; never a verifier input |
| **T4 Desktop / VM / browser** | `common/desktop`, `common/vm`, `browser` over T1/T2 web UIs | Pixels over real state | `loadvm` about 1.1 s; clean browser profile | CUA. The verifier reads T1/T2 state, never pixels |

**Cross-cutting services.**
1. **Org model.** People, roles, reporting lines, time zones, working hours, RBAC, and latent traits (CRMArena), shared by every App. One person has the same identity in chat, mail, CRM and tickets. EnterpriseBench and ARE both stress cross-source consistency.
2. **Clock bus.**
   - One virtual time feeds every App. T1 Apps read it directly; T2 Apps get libfaketime or an app-level now; T4 guests get their clock set at restore.
   - Add ARE-style notifications with a verbosity policy, `wait_for_notification`, `get_current_time`, an event DAG with `depends_on` and delays, and condition events.
3. **People.** Grounded LLM actors with their own tools (τ² dual control), sampled cooperativeness and latency profiles, and audits for fabrication, critical errors and GAUGE rank agreement.
4. **Verifier.**
   - SQL and State-Adapter assertions plus `Collateral` (existing), plus Gaia2-style write matching with causal order and time windows.
   - Infeasible, confidential and policy probes as first-class task types.
   - An LLM judge only for free text, and only with a state bit alongside.
5. **Harbor export.**
   - Export the World as a compose project: T1/T2 Apps as MCP sidecars declared under `[[environment.mcp_servers]]`, with credentials via `${VAR}`.
   - The verifier runs the State-Adapter checks in `tests/test.sh` and writes `reward.json`.
   - CUA tasks run `computer-1` against the desktop App. Harbor sees tools only through MCP, CLI or skills, so this is the only route.

### 8.2 Priority list: what to add first

1. **T1 "office core" Apps, executable, sharing one org model and clock.** Mail exists (`mailbox`).
   - Add **calendar** (events, attendees, free/busy, rooms, time zones, RSVP).
   - Add **chat** (channels, DMs, threads, mentions, reactions; Slack-shaped tools).
   - Add **docs/wiki** (pages, comments, permissions; Notion-shaped).
   - Add **tickets** (queues, SLA timers, assignment, states; Jira/ServiceNow-shaped).
   - These four plus mail cover most of EnterpriseOps-Gym, ClawsBench, OfficeBench and Gaia2 Mobile.
2. **Notification channel + event DAG + `wait_for_notification`** in the Clock (§5.2, items 1–4). Without these, chat and calendar are just databases to poll.
3. **CRM T1 App** (accounts, contacts, opportunities, cases, activities), seeded with latent variables, plus infeasible and confidential probes. Highest enterprise demand; CRMArena-Pro and EnterpriseOps-Gym show large headroom.
4. **The schema → World generator** (AWM recipe):
   - schema and seed generation from a scenario or a customer schema;
   - a generated CRUD layer plus curated tools;
   - execute-and-repair loops;
   - solution-first tasks;
   - the density card (§7).
   - Reuse ScaleEnv findings from note 01.
5. **T2 realism pair: Mattermost (or Rocket.Chat) + Plane**, both with official or ready MCP, both used by TheAgentCompany (and Harbor has its adapter), plus **Gitea** (official MCP, MIT). Use them to check that agents trained on T1 chat and tickets transfer to real software.
6. **Mail/calendar realism: Stalwart** (JMAP + CalDAV in one AGPL box), or **Mailpit + Radicale** (MIT + GPL, file-seeded .ics). Put a Gmail- and GCal-shaped MCP in front, because no Google emulator exists.
7. **CUA path.**
   - Harbor `computer-1` against T4 desktops showing T2 web UIs.
   - Add batched actions, an accessibility-tree observation and a `report_infeasible` terminal action to Crucible's `computer` tool.
   - Record HAR for browser Apps (WebArena-Verified style).
8. **ERP: ERPNext via Frappe's official MCP** (MIT framework), or extend the existing `finance_ap` T1 ERP. Odoo's MCP is community-only.
9. **Ticketing realism: Zammad** (REST, `auto_wizard.json` bootstrap).
10. **Customer-data importer** (§6.3): Presidio, then SDV or TabularARGN, then an SDNist gate, then T1 Apps.
11. **T3 long-tail tool simulator** with a cache and a worldbook, last, and only for read-only coverage.

### 8.3 Integration risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Wall-clock leakage in real apps breaks determinism and snapshot identity | OSWorld-Verified date and timing fixes; AndroidWorld pins the clock | libfaketime or app-level now; set guest clocks at restore; fingerprint timestamps in snapshots |
| Shared state across episodes on live services | WebArena task interference; Toolathlon weak run consistency | Per-episode identities and row scoping (WorkArena), `serial` worlds (exists), CoW containers |
| Slow resets of heavy apps (GitLab boots ≥ 5 min) | WebArena; TheAgentCompany images need 30+ GB | Pre-booted pools with RAM snapshots (as `common/vm`), diff-and-revert, prefer light apps (Gitea over GitLab) |
| LLM simulator hallucination or inconsistency poisons rewards | AWM ablation; TRUSTEE critique of Simia | Simulated tools never feed verifiers; writes mirrored to real state; cache + temperature 0 |
| User-sim "easy mode" inflates scores; judges mis-rank close agents | 2603.11245; GAUGE 31% disagreement on close pairs; τ² error rates | Tool-grounded users, non-cooperative profiles, state-based rewards, GAUGE calibration |
| LLM judges gamed during RL | ARE; One Token to Fool | Judge only free text, alongside a state bit; master-key probes in CI |
| Auto-generated tool surfaces are hard to use | FastMCP docs warning; MCP-Atlas reasoning failures | Curated business-verb tools over a generated CRUD layer |
| Tool overload / retrieval masks ability | EnterpriseOps-Gym oracle retrieval; TheMCPCompany 18k tools | Catalog distractors (exists); report results with and without oracle retrieval |
| Cross-app inconsistency in generated data | ARE notes its limits; EnterpriseBench synchronises records | One org model and an entity registry feeding every App; referential checks in the density card |
| License exposure (AGPL, BSL) | §4.1 | Unmodified containers, adaptation in MCP sidecars; avoid BSL Outline for hosted offers, or use Docmost |
| Upstream churn | Daytona public repo frozen; AppFlowy-Cloud and LocalStack OSS archived; Harbor repo moved | Pin images by digest; prefer apps with active OSS and official MCP |
| Customer data leakage into training Worlds | – | Redaction + synthesis + DP option; SDNist privacy metrics as a gate; raw data stays in the customer boundary |
| Contamination of held-out tasks by real OSS docs or changelogs | Crucible's own changelog-strip work | Same leak probes on T2 Apps' bundled docs and seed data |
| Throughput bound by env utilisation, not VM count | DART-GUI 12% → 68% utilisation | Decoupled rollout workers; async episodes; pools |

---

## 9. Sources

All arXiv ids were checked on 2026-10-29 against the arXiv API, which returned these titles.

**Enterprise and workplace.**
- TheAgentCompany 2412.14161 (github.com/TheAgentCompany/TheAgentCompany)
- TheMCPCompany 2510.19286
- WorkArena 2403.07718, WorkArena++ 2407.05291 (github.com/ServiceNow/WorkArena)
- CRMArena 2411.02305, CRMArena-Pro 2505.18878 (github.com/SalesforceAIResearch/CRMArena)
- SCUBA 2509.26506
- EnterpriseOps-Gym 2603.13594 (github.com/ServiceNow/EnterpriseOps-Gym)
- EnterpriseBench (Fujitsu) 2510.27287 (github.com/ast-fri/EnterpriseBench)
- EnterpriseBench Corecraft 2602.16179
- EnterpriseBench (strategic reasoning) 2609.37658
- AppWorld 2407.18901 (github.com/StonyBrookNLP/appworld)
- τ-bench 2406.12045, τ²-bench 2506.07982 (github.com/sierra-research/tau2-bench)
- ToolSandbox 2408.04682 (github.com/apple/ToolSandbox)
- OfficeBench 2407.19056
- WorkBench 2405.00823
- ClawsBench 2604.05172
- OccuBench 2604.10866
- EnterpriseClawBench 2606.23654
- AgentMercury 2608.20634
- EntWorld 2601.17722
- SWE-Lancer 2502.12115
- GDPval 2510.04374

**Web, OS, MCP.**
- WebArena 2307.13854
- WebArena-Verified (github.com/ServiceNow/webarena-verified; OpenReview 94tlGxmqkN)
- WebArena-Infinity (github.com/web-arena-x/webarena-infinity)
- VisualAgentBench / WebArena-Lite 2408.06327
- VisualWebArena 2401.13649
- WEBSERV 2510.16252
- BrowserGym 2412.05467
- REAL 2504.11543
- Online-Mind2Web 2504.01382
- WebGym 2601.02439
- InfiniteWeb 2601.04126
- GUI-GENESIS 2602.14093
- Recreated websites 2603.10505
- WebForge 2604.10988
- OSWorld 2404.07972 (github.com/xlang-ai/OSWorld)
- OSWorld-Verified (xlang.ai/blog/osworld-verified)
- OSWorld 2.0 2606.29537
- OSWorld-MCP 2510.24563
- Windows Agent Arena 2409.08264
- AndroidWorld 2405.14573
- Toolathlon 2510.25726 (github.com/hkust-nlp/toolathlon)
- MCPMark 2509.24002
- MCP-Universe 2508.14704
- MCP-Bench 2508.20453
- LiveMCPBench 2508.01780
- MCP-Atlas 2602.00933
- MCPWorld 2506.07672
- BrowseComp-Plus 2508.06600

**CUA infrastructure and RL.**
- Harbor (github.com/harbor-framework/harbor; docs.harborframework.com)
- SkyRL Harbor integration (github.com/NovaSky-AI/SkyRL)
- E2B and e2b-desktop (github.com/e2b-dev)
- Daytona (github.com/daytonaio/daytona)
- Modal sandbox snapshots (modal.com/docs)
- Runloop (docs.runloop.ai)
- trycua/cua
- Playwright MCP (github.com/microsoft/playwright-mcp)
- Steel (github.com/steel-dev/steel-browser)
- Anthropic and OpenAI computer-use tool docs
- UI-TARS 2501.12326
- Set-of-Mark 2310.11441
- ComputerRL 2508.14040
- UI-TARS-2 2509.02544
- ARPO 2505.16282
- DART-GUI 2509.23866
- ZeroGUI 2505.23762
- OpenCUA 2508.09123

**Synthesis and simulated tools.**
- AgentScaler 2509.13311
- AWM 2602.10090 (github.com/Snowflake-Labs/agent-world-model)
- ScaleEnv 2602.06820
- EnvScaler 2601.05808
- EnvFactory 2605.18703
- Agent-World 2604.18292
- APIGen-MT 2504.03601
- Simia 2511.01824
- TRUSTEE 2604.17739
- Kimi K2 2507.20534
- StableToolBench 2403.07714
- MirrorAPI 2503.20527
- Qwen-AgentWorld 2606.24597
- DreamGym 2511.03773
- WebDreamer 2411.06559
- UI-Simulator 2510.14969
- AutoEnv 2511.19304
- Environment Tuning 2510.10197
- Terminal-Universe 2609.04148
- Trace2Env 2610.06100
- Agentic Environment Engineering survey 2606.12191
- WildAWM 2610.26828
- FastMCP OpenAPI docs (gofastmcp.com)
- openapi-mcp-generator
- Google MCP Toolbox for Databases
- PostgREST
- Frappe MCP

**People, judges, time.**
- Mind the Sim2Real Gap 2603.11245
- Lost in Simulation 2601.17087
- GAUGE 2609.12191
- Non-Collaborative Users 2509.23124
- SimulatorArena 2510.05444
- MirrorBench 2601.08118
- LLMs Get Lost in Multi-Turn 2505.06120
- UserBench 2507.22034
- UserRL 2509.19736
- Concordia 2312.03664
- SOTOPIA 2310.11667
- One Token to Fool 2507.08794
- Agent-as-a-Judge 2410.10934
- ARE 2509.17158 (github.com/facebookresearch/meta-agents-research-environments)
- Gaia2 2602.11964
- Vending-Bench 2502.15840
- TimeArena 2402.05733
- libfaketime (github.com/wolfcw/libfaketime)

**Data and privacy.**
- AIM 2201.12677
- Aug-PE 2403.01749
- Presidio (github.com/microsoft/presidio)
- SDV (DSAA 2016)
- MOSTLY AI TabularARGN 2501.12012 and 2508.06647; DP text 2210.14348; GTM 2512.04535
- NIST CRC / SDNist (github.com/usnistgov/privacy_collaborative_research_cycle)

**Open-source app stand-ins.**
- Mattermost, Rocket.Chat, Zulip
- Outline, AFFiNE, Docmost, AppFlowy
- Mailpit, Stalwart, GreenMail, MailHog
- Radicale, Baikal
- Plane, OpenProject, Redmine, Taiga
- Gitea, Forgejo, GitLab
- Twenty, SuiteCRM, EspoCRM
- ERPNext/Frappe, Odoo
- Zammad, GLPI
- Nextcloud, oCIS
- ONLYOFFICE, Collabora
- BookStack, Wiki.js
- moto, Prism, WireMock, Mockoon

Repo facts (licenses, MCP status, archive status) were taken from the GitHub API and vendor docs on 2026-10-29.

**Unverified items kept out of the recommendations:**
- E2B resume latency.
- OSWorld per-task reset time.
- The current OSWorld-Verified leader.
- The WebArena-Verified hard-subset size (paper and repo conflict).
- The OpenProject MCP tier limits.
- The DreamGym headline numbers.
- The ITBench and OdysseyBench ids.
