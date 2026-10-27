# Handoff: kernel GPU tasks on the RTX 5090 (2026-10-26)

**State:** task creation and revision are stopped (the lead: 10 training and 5 benchmark tasks are in hand). Nothing of
this run is still running; no containers of ours are left on the 5090.

## Verdicts (`data/targeting/kernel-rtx5090-r1/report.json`)

| task | student (k=8, 300 calls) | frontier | verdict |
|---|---|---|---|
| km-cuda13-hecbench-591f311-4 | 8/8 | 397B passes | too easy |
| km-cuda13-hecbench-ba3a30a-3 | 7/8 | not run | too easy |
| km-cuda12-cudahandbook-b62f9ad-4 | 4/8 at 4096 tokens, 3 of 4 failures cut at the reply limit | 397B 1/2 | not kept (last valid result) |
| km-cuda13-hecbench-8fcc759-4 | not measured | — | dropped: fails S10 (1/60 false negative) |
| kd-bug-llamacpp-1b8b8fb1 (K6) | 1/8 | 397B 0/2, Nemotron-Ultra 0/2 | unfair (no strong model passes) |
| kd-bug-llamacpp-ca21f3a1 | 0/8, no milestone ever fixed | not run | too hard |
| kd-bug-llamacpp-cef66d4a (K6) | 0/8 | not run | too hard |
| kd-bug-llamacpp-efe7d4b1 (K6) | 0/8 | not run | too hard |

**Kept GPU training tasks: 0.** The 35B solves the CUDA 13 migrations and fails the llama.cpp kernel bugs, and no strong model
on Tinker passed the one kernel bug that reached the band.

## Spend

About $171.5 of the $180 share: $168.58 on the phase meter, about $2 for the swarm's K6 writers and checkers, $0.91 for two
plumbing checks. About 25.5 GPU-hours of grading on the 5090. $40 of it bought nothing: $10.6 masked by the kit
ImportError, $29.5 masked by a network outage.

## Framework findings (all fixed on main, each with a test)

1. The episode kit copied only the host gate kit's files (no `kernel_profiles.py`): every grade raised ImportError. The kit
   now adds each runner's modules, and the runner must start before any episode. (`bfcc0b64`)
2. `kernel_profiles.of_spec` read a ggml kdspec's `harness` table as a profile name: TypeError in every grade. (`bfcc0b64`)
3. The kernel targeting preflight grades each task's oracle (full pass) and empty patch (nothing) on the GPU before paid
   episodes, and refuses a report that fails S7k. (`bfcc0b64`, `bffdee4a`)
4. `max_output_tokens` (every reply cut at our `--max-tokens`) counted as a skill failure; it is a limit stop now. (`b4d7f1d1`)
5. The GPU ingest dropped milestones without re-composing the report: 6 records asked for an ungraded requirement. The ingest
   re-authors now. (`e1fb8e0b`)
6. Gate S7k: the report against the graded milestones (two-way map, op and symptom named, a parameter that sets a milestone's
   cases apart). (`bffdee4a`)
7. The swarm's K6 reviser for kernel records and its claim check (every number stated is a case parameter). (`2984b34b`)
8. The location guard: a deterministic probe (location words, a grep that finds a word only in the fault's files) and a
   blind non-builder checker comparing original and revised text. (`92b57062`)
9. The migration intent card (the mine stage raised KeyError on migration windows). (`e209f4ce`)

## Open, not fixed

- **A network drop costs a full attempt clock per episode.** Tool calls over SSH to the GPU host hang until the 9000 s
  attempt clock; they should fail fast and mask at once.
- **No CUDA fairness check (S9b)** and no DeltaAI episode path (`HOSTS["gh200"]` refused); the GH200 gate jobs were last seen
  pending.
- **Unexplained container.** A `crucible-k-gpu-ep-*` grade container was up on the 5090 with no grader process of this run
  behind it. It was left alone.
- **Frontier cost.** Nemotron-3-Ultra is slow and costly as a frontier solver on kernel tasks ($24.32 for 6 episodes, one
  masked).

## Files

- Out dirs: `data/targeting/kernel-rtx5090-r1` (the measurement), `-masked`, `-original`, `-1b8b-original`,
  `-b62f-4096` (set-aside runs); records under `data/kernel_debug/store/train/*/record.json` (with `.bak-*`).
- Revisions: `data/kernel_debug/revisions.jsonl`, `data/kernel_debug/prompts/*.json.bak-k6-*`.
- Docs: `10-kernel-debug-lane.md` §9.6, `13-swarm-self-improvement.md` §13.
