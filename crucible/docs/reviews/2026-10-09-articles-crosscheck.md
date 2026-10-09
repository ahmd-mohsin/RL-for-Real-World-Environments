# Cross-check: three articles against modules M0–M4 and the seed tasks (2026-10-09)

**Sources:**
- Preference Model, "Building RL Environments for Superintelligence" (J. Zhou, 2026-10-08). **PM** below.
- Hugging Face, "How not to build coding environments" (Repo2RLEnv). **R2E** below.
- Prime Intellect, "On the Nature of the Swarm" (K. Dunas, 2026-10-06). **SW** below.

**What was read when:**
- PM and R2E were read on 2026-10-09 and drove the M1/M2 hardening pass (P20–P25, guard, leak scrubber, verified git scrub).
- That pass predates M3 and M4. This review re-checks every claim against all built modules (M0 core, M1 world, M2 actors, M3 connectors, M4 signals so far) **and against our own seed tasks 01–04**, which the earlier pass did not audit.
- SW is new.

**Legend:** ✅ built and tested · 🟡 partial or design-only · ❌ violated or missing · 🔧 fixed in this pass.

## 1. Repo2RLEnv ("How not to build coding environments")

| # | Claim / rule | Crucible (M0–M4) | Seed tasks 01–04 |
|---|---|---|---|
| R1 | Verifier first: gold = 1.0, no-op = 0.0, only targeted tests count | ✅ `world/audit.py` `oracle_passes`, `floor_gap.nop_fails` (World audit). 🟡 **Never run on the seed worlds through Crucible**: the M1 seed gate tested snapshot/restore only | 🟡 Gold = 1 and no-op = 0 are claimed in MIGRATION_LOGs, but only task04's oracle has a saved report. task01/01v2 were validated by running `grade.py` natively, and their images were never built. task04 has no `solution/solve.sh`, so Harbor's `oracle` agent cannot run it. ❌ **task03's grader crashes on ≈ 18% of fresh seeds whatever the submission** (`fresh_cases.py:34`: `nchains∈[12,40)` with 0.48 nm spacing gives a 1.92 nm box, below 2 × the 1.0 nm cutoff), and tasks 01–03 leave `reward.txt` empty on any grader exception |
| R2 | Strip answer pointers from the prompt | ✅ `world/leaks.py` scrubs instructions, actor replies and CI output; audit `no_solution_leakage.static/.dynamic` | ✅ No PR or commit links or fix versions. task04 cites issues #5571/#5995, which have no fix PR |
| R3 | Scrub git history to the base commit, and verify it | ✅ guard `git_scrub` + verification (`rev-list --all` ⊆ base ancestry, `fsck --unreachable` empty); red-team `git log --all` | ✅ Every repo is one `base` commit (task04 removes PCL's `.git`). All 11 recorded runs that probed history saw only `base` |
| R4 | Pre-bake dependencies; offline by default | ✅ `network: no-network` default + egress self-test as the policy uid | ✅ no-network for agent and verifier. ❌ The task01/01v2 recorded runs were *graded inside the agent's Runpod pod, which had internet* (`common/README.md:31`), against their own `task.toml` |
| R5 | If network is needed: an allowlist (never a full block or a denylist), ideally a date-frozen package mirror | ✅ `network: allowlist` enforced by Harbor's docker egress sidecar or agentenv's `NetworkPolicy`. 🔧 The egress self-test now probes 17 code hosts, package registries and mirrors (incl. `files.pythonhosted.org`, `codeload.github.com`, `web.archive.org`) with a **TLS/SNI handshake** (a transparent proxy accepts every TCP connect), skips allowlisted hosts, and checks that each allowed lane is reachable. 🔧 Audit `network.answer_sources` **fails any world whose allowlist opens a code host or public index**, and flags `public` worlds. 🟡 The date-frozen mirror (an index proxy filtering by PEP 700 `upload-time` ≤ task date) is designed, not built | n/a (offline) |
| R6 | Keep the gold patch and hidden tests out of the box; apply at scoring time | ✅ Hidden CI in a separate runner, re-staged per run; grader paths root-0700; `oracle_containment` audit | ✅ No image copies `tests/` or `solution/` (`.dockerignore`). 🟡 At grade time, task03's golden `.npz` files are readable by candidate code |
| R7 | "A prompt instruction is a nudge, not a control" | ✅ P21: every prohibition is an enforced property with a red-team attacker | ❌ The graders enforce bans by **source regex** (task01 CUDA/`std::`, task02 extension bans, task03 ctypes), which macros, aliases, `importlib`, `third_party/` or renamed binaries bypass. In task03 the original engine stays reachable at grade time and can be delegated to |
| R8 | Treat git history, `pip download` and the web as leaks; every green score through them is contamination | 🔧 **New:** `connectors/agent_events.py` tags every tool call that probes git history, package registries, code hosts, grader files, privilege or `/proc` (`data.probes`). It separates **ran** from **exposed**: a `git log --all` that lists only `base` ran but exposed nothing. A World episode in which the policy reached an answer source is **masked** (`crucible:answer_exposed`); agent-run traces whose pass followed an exposure are relabelled `unknown`. Probes are never scored (P12/P20) | ✅ All 11 frontier runs scanned, automatically and by a manual audit: **no exposure**. The 3 passing runs are clean |
| R9 | Temporal decontamination (SWE-rebench) | 🟡 M3 traces keep merge and commit times; the benchmark rule (§11.3) is now explicit in DESIGN. A filter by model cutoff is not built (M10/M14) | n/a |

## 2. Preference Model ("Building RL Environments for Superintelligence")

| # | Claim / rule | Crucible | Seed tasks |
|---|---|---|---|
| PM1 | "If the grader can't check it, it cannot go into the prompt" | ✅ P20 + Acceptance Contract A7 (design). 🟡 No automated instruction↔check coverage yet (M9) | ❌ **Unenforced instruction sentences:** task01 `omp_target_alloc`; task01v2 `tsr::DeviceError` and the `FrameScheduler` thread pool; task02 exact FPS indices, `arg` outputs, "at least the same performance" (bars allow 1.10–1.35× slower); task03 port→original checkpoint loading. Tolerances looser than the text: task01 "only reassociation" vs `epsilon(1e-3)`; v2 "one or two 8-bit levels" vs an uncapped tail; task04 "within floating-point noise" vs the automatic KinFu-LS BIFURCATED fallback |
| PM2 | Grade outcomes, not process | ✅ P12/P20; probes are recorded, never scored (this pass) | 🟡 Process-style probes: task02 counts Triton launches, task03 counts `pallas_call`s; static regex bans (R7) |
| PM3 | A strong AI escalates to root to read the environment | ✅ guard: fresh uid, `no_new_privs`, root-only grader paths, verifier integrity; red-team suite (3/3 on local Docker after this pass) | ❌ **task02's verifier runs candidate code as root** (`setup.py`, `import pcops`). It can rewrite `/tests/hidden/bench_pcops.py`, slow the reference, write `/logs/verifier/reward.txt`, or leave processes running. task03 and task04 drop privileges correctly |
| PM4 | Rewards never come from a channel the agent controls | ✅ P23 for the World (snapshots after quiesce). ✅ CI-as-a-tool is feedback only, never reward (now stated) | ❌ **task01/01v2**: the hidden suite is built by the *candidate's* CMake and judged by doctest's exit code (`DOCTEST_CONFIG_DISABLE` as a PUBLIC definition passes with 0 cases run), and probe/perf rows are parsed from candidate stdout. ❌ **task03**: pytest gates are `rc == 0` only (`if "pytest" in sys.modules: os._exit(0)` passes), and timing runs inside the candidate's process. 🟡 **task04** (the hardened grader): perf can be faked by interposing timers, memoizing repeated identical calls, or caching results in a HOME/TMPDIR that persists across the grading run |
| PM5 | Accept environment awareness; security over realism | ✅ P22 | ✅ |
| PM6 | Deep tasks: reward doing better, not clearing a fixed bar | 🟡 P24 is implemented as fixed perf and quality *bars* (task04 v2/v3). Open-ended, record-relative scoring is now designed (§6.4), not built (M6/M9) | 🟡 Bars only |
| PM7 | Tasks too hard for us are fine if the grader checks the outcome | ✅ §7.1 solution-first construction; design note added to §7 | — |

## 3. Prime Intellect ("On the Nature of the Swarm")

| # | Observation | What it requires of RL environments | Crucible |
|---|---|---|---|
| SW1 | Compaction is a lossy bet; disk offloading preserves detail | Long-horizon worlds must be solvable across compactions; traces must mark compaction | ✅ M3 maps ATIF `is_copied_context` → `copied_context`. ✅ Workspaces persist in `WorldRef`. 🟡 Context overflow ends an episode as truncated and scored (M2); our minimal `LLMToolPolicy` has no compaction (real harnesses run through Harbor, M10) |
| SW2 | Sub-agents and *persistent* sub-agents (state survives, summaries written after the question is known) | Traces keep sub-agent structure; persistent peers carry their own state | ✅ M3: sub-agent steps tagged `data.subagent`. ✅ M2: peers and simulated actors keep memory in `WorldRef` |
| SW3 | Mesh beats tree at scale; peers share a workspace (git, issue tracker, chat) instead of routing through a root | Several policy agents in one World, talking peer-to-peer through shared Apps | ✅ `run_episode` drives any number of policies; `send_message` is peer-to-peer under ACLs; Apps include git repo, mailbox, tickets, CI and website. 🔧 **New test:** two policy agents share a workspace with no orchestrator, and each keeps its own credit |
| SW4 | Communication costs tokens and is lossy | Measure it per episode | 🟡 Designed as a world metric (§11.1), not built |
| SW5 | Credit in a swarm | Each agent's steps are graded by what *it* wrote | 🔧 World traces now record `data.writes` per tool call, so M4's last-writer rule credits the agent (or sub-agent) that edited the file. Previously, World traces carried no writes |
| SW6 | Self-modifying organisation (bitter lesson) | Don't hard-code topology | ✅ World topology is data (ACLs, actors). Task Forge roles now coordinate through a shared task repo and issue tracker (§7.2), not an orchestrator's context |
| SW7 | 10,000-agent results still await verification | Verification is the bottleneck | ✅ PM1/P20: only checkable outcomes are rewarded |

## 4. Fixes made in this pass

1. **Answer-key probes** (`connectors/agent_events.py`): `probes_of`, `exposed`, `annotate_probes`.
   - Applied to ATIF, capture-proxy, OTel and World traces.
   - A World episode with an exposure is masked as `crucible:answer_exposed`; agent-run traces with an exposure are no longer labelled good.
   - Tests: `tests/unit/test_probes.py` (classification, ran vs exposed, a World with and without future history).
2. **Egress self-test** (`world/guard/guard.py`):
   - 17 answer-source hosts;
   - TLS/SNI handshake on port 443;
   - allowlisted hosts excluded from the deny set and checked as lanes.
   - Red-team suite 3/3 on local Docker.
3. **Audit `network.answer_sources`** (`world/audit.py`): fails allowlists that open code hosts or public indexes.
4. **World traces record `writes`** per edit call (M4 attribution for multi-agent and single-agent episodes).
5. **DESIGN.md:**
   - P26 (trust only channels the candidate cannot write) and P27 (probes are recorded, never scored);
   - §5.1 as-built connectors;
   - §8.3 and §9 additions (A1 in the declared environment, A13 grader attack suite);
   - §6.4 open-ended objectives;
   - §7.2 swarm coordination;
   - §11 temporal decontamination and communication cost;
   - Appendix B defects 15–24.

## 5. Not fixed: the seed-task graders (needs a decision and lab-GPU time)

The holes in §1–§2 live in `migration_tasks/task0*/tests/`, not in Crucible. Fixing them changes the task versions and invalidates the recorded calibrations, so each task must be re-validated (gold, no-op, every attack below) on the lab RTX 5090. The prioritized list:

- **P0: graders that can be passed without doing the task.**
  - task01/01v2:
    - build the hidden suite from a grader-owned CMake that consumes the installed library;
    - reject `DOCTEST_*` interface definitions;
    - require exact doctest counts;
    - read probe/perf rows from a grader-owned file with a nonce.
  - task02:
    - `setpriv` to 65534 and `pkill` after each step;
    - time the reference before any candidate code runs;
    - run the benchmark from a root-owned copy;
    - make `/opt/originals` and `/logs/verifier` unwritable.
  - task03:
    - compare dumped outputs against root-only goldens instead of trusting pytest's exit code (at minimum, exact pass counts with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`);
    - put the consumers before the candidate on `PYTHONPATH`;
    - make the original engine unreachable;
    - fix `fresh_cases.py:34` (`nchains ≥ 17` or an explicit box).
  - All graders: any exception writes reward 0 plus `infra_error`.
- **P1: perf integrity.**
  - Time from outside the candidate's process.
  - Fresh inputs per timed repetition (no memoization).
  - A fresh HOME and TMPDIR per candidate invocation (task04 replay).
  - An idle-GPU check, and at least 15% headroom on every bar.
- **P2: prompt↔grader alignment.** Enforce or delete every UNENFORCED sentence; align tolerances with the text; replace regex bans with environment enforcement (the toolkit is absent) or binary inspection.
- **P3: evidence.**
  - Grade task01/01v2 in a fresh container, as `task.toml` declares.
  - Add `task04/solution/solve.sh`.
  - Commit gold and no-op reports.

Each of these becomes a check in the A13 grader attack suite, so every future task is tested against it.
