# coding (a Crucible pack)

The legacy GPU code migration family (DESIGN §12): four seed tasks are real migrations with kit graders in GPU
verifier images:
- task01: OpenMP offload;
- task02: Triton;
- task03: Pallas;
- task04: a PCL CUDA 12 port.

The pack also holds span tasks and traps derived from task04.

```bash
crucible pack check packs/coding --level semantic  # CPU: oracle 1, base 0, comment / string-literal / dead-code stuffing 0, deletion-only 0, honest failures at their Φ
crucible label --pack coding --out label_coding    # recorded frontier runs + PCL pull requests -> traces, Φ credit, SFT/KTO
```

The dense reward Φ reads **code only** (`crucible_pack_coding/codeonly.py`, M14 campaign B2). A token earns credit only
where the compiler or interpreter runs it:
- comments are removed and the contents of string and char literals are blanked (C family by a lexer with raw strings
  and digit separators, Python by `tokenize`, CMake quoted and bracket arguments, TOML strings);
- preprocessor-dead branches are dropped (`#if` on a constant expression; `#ifdef` of a macro the agent introduced and
  never defines);
- constant-false branches (`if (0)`, `if False:`, `if(FALSE)`, `False and x`) and statements after a jump are dropped;
- new functions that nothing references are dropped (Python, C family, CMake);
- a Python or TOML file that does not parse has no code.

Token milestones (a token or call added) earn credit on their own. Created, deleted, removed and located milestones
count only in proportion to the token milestones met, so placeholder files, deletions or touched blocks earn nothing
by themselves. Before the code view, the oracle's tokens in string literals or under `#if 0` reached Φ 1.0 on every
bench seed with nothing migrated. A recorded run graded under an earlier task version is labelled `unknown` and earns
no credit.

`hooks.tasks` holds CPU stand-ins (`coding/dense_replay/<seed>@cpu2`), one per bench seed: the seed's base repository
in a workspace, graded by Φ through a World, with four bounded adversaries (comment, string-literal and dead-code
stuffing, deletion only). They check the dense grader end to end; they are not the seeds' real outcome. Every hook runs
on CPU, so the pack declares no `requires` and its semantic level runs on any host. Live grading (`crucible build`, the
kit graders) needs the lab GPU host, docker and the seeds' verifier images; it is not a pack hook. A manifest cannot yet
say "this hook needs a GPU" (`requires` is pack-wide), so a future GPU `hooks.tasks` would need per-hook requirements in
`PackManifest`. The GPU gates' runners are
`python -m crucible_pack_coding.verify.golden_gpu` and `python -m crucible_pack_coding.forge.forge_gpu`.
