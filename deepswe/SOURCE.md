# Source of this folder

A vendored copy of Datacurve's **DeepSWE** benchmark: https://github.com/datacurve-ai/deep-swe. The upstream git history was not copied.

- Upstream commit: `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (2026-08-26)
- License: Apache-2.0 for Datacurve's task specs, harness and verifiers (see `LICENSE`). Each task's upstream project keeps its own license (see `PROVENANCE.md`).
- Website / leaderboard: https://deepswe.datacurve.ai/

To refresh it:
```bash
git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git /tmp/deep-swe
rsync -a --delete --exclude .git --exclude SOURCE.md /tmp/deep-swe/ deepswe/
```

## Notes from our analysis (2026-10-04)
- **Size:** 113 tasks in Harbor format, about 90 open-source repos.
- **Repos:** most repos have one task; the most is 5 (pmndrs/koota). Examples: httpx, fastapi, helm, prometheus, drizzle-orm, tomlkit.
- **Languages:** TypeScript 35, Go 34, Python 34, Rust 5, JavaScript 5.
- **Task types:** 106 feature requests, 4 bug fixes, 3 enhancements.
- **Reference solution:** median **6 files, about 612 added lines** (max 1,647).
- **Instruction length:** median about 2.1k characters.
- **Where difficulty comes from:** task scope (multi-file features, long specs), not the repos. Compare our `sandbox/` slugify task: 1 file, 6 added lines.
- **Tasks are original**, written by Datacurve rather than taken from historical PRs, so they can't have leaked into models' training data.
- **Grading:** v1.1 uses a separate verifier container. The agent's commits are extracted as a patch and graded in a pristine container, the same design as our `sandbox/grading/grade.py`.
- **Leaderboard runs** use `mini-swe-agent` via Pier on Modal.

Task layout: `tasks/<id>/{task.toml, instruction.md, environment/Dockerfile, tests/{test.sh,test.patch,grader.py,config.json}, solution/solution.patch}`.
