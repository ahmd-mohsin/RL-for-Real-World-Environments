"""Studio hooks of the coding pack (declared in ``pack.yaml``). Live grading needs the seeds' GPU verifier images, so
the hooks here are the CPU parts: dense credit for recorded agent runs, and fixtures over the M6 dense reward.

Per-step Φ is not derivable for the recorded frontier runs: their agents wrote files with python and ``sed``, which the
bash edit replayer cannot follow (0–12 replayable states out of 46–200 calls per run, measured 2026-10-11). Credit is the
final patch's Φ, placed at the run's last event. Crucible's own World runs record state at every step and do not have
this limit.

Φ is measured on **code only** (M14 audit V4, campaign B2; ``codeonly.py``): comments, the contents of string and char
literals, preprocessor-dead and constant-false branches, unreachable statements and new functions nothing references are
removed from every file, on the base and on the patched state, before any milestone is checked, so mining and checking
agree. A file whose code is empty, or a Python / TOML file that does not parse, does not exist. Only **token**
milestones (a token or call added) earn credit on their own: ``created``, ``deleted``, ``removed`` and ``located``
milestones are scaled by the share of token milestones met, so a placeholder file, deleting the code being migrated, or
touching each anchored block earns nothing by itself. Before V4 the oracle's tokens in comments reached Φ 0.92–1.00 on
every seed without migrating anything; before B2 the same in string literals or under ``#if 0`` / ``if False:`` reached
1.0.
"""

from __future__ import annotations

import re
from typing import Any

from crucible_pack_coding.codeonly import C_LIKE, HASH_COMMENT, IDENT, code_only, lang, undefined_macros, unused_new  # noqa: F401 - re-exported

# M14 campaign B2: the milestone kinds that are the work itself. A created file is not: a placeholder file earned the
# ``created`` milestones of a seed with nothing migrated, so ``created`` is gated with deleted / removed / located by the
# share of token milestones met (on a seed with no token milestone it counts on its own).
TOKEN = frozenset({"added", "added_call"})
POSITIVE = TOKEN  # mine.from_diff ids: diff/<kind>/...


def _bench_seeds() -> tuple[str, ...]:
    from crucible_pack_coding.seeds import SEEDS as S

    return tuple(k for k, v in S.items() if v.get("bench"))


# M14 audit V14: the fixtures covered the superseded task01_prims_omp (bench: false) and skipped task01v2_tessera_omp
SEEDS = _bench_seeds()


def code_state(fs):
    """A ``FileState`` whose every file (current and base) is its code (``codeonly``). The cross-file passes run over the
    files that differ from the base: macros the agent tests but never defines are undefined, and new functions nothing
    references are dropped (M14 campaign B2: dead code reached Φ 1.0)."""
    from crucible.workflow.filestate import FileState

    memo: dict[str, str | None] = {}

    def base(p: str) -> str | None:  # one read per path (task04's base is a ``git show`` per call)
        if p not in memo:
            memo[p] = fs.base(p) if fs.base else None
        return memo[p]

    changed = {p: c for p, c in fs.files.items() if c != base(p)}
    base_idents: set[str] = set()
    for p in changed:
        b = base(p)
        if b is not None:
            base_idents.update(IDENT.findall(code_only(p, b) or ""))
    undefined = undefined_macros(changed, base_idents)
    drop = unused_new(changed, base_idents, undefined)
    files = {p: code_only(p, c, undefined if lang(p) == "c" else frozenset(), drop.get(p, frozenset())) if p in changed else code_only(p, c)
             for p, c in fs.files.items()}
    return FileState(files=files, base=(lambda p: code_only(p, base(p))) if fs.base else None, stale=set(fs.stale))


def kind(milestone_id: str) -> str:
    return milestone_id.split("/")[1] if milestone_id.count("/") >= 2 else milestone_id


def gated(ms: list, phis: dict[str, float | None]) -> dict[str, float | None]:
    """Milestones that are not token milestones (created / deleted / removed / located) scaled by the weighted share of
    token milestones met. A seed without token milestones credits ``created`` on its own."""
    alone = TOKEN if any(kind(m.id) in TOKEN for m in ms) else TOKEN | {"created"}
    pos = [m for m in ms if kind(m.id) in alone]
    total = sum(m.weight for m in pos)
    share = sum(m.weight * (phis.get(m.id) or 0.0) for m in pos) / total if total else 0.0
    return {k: (v if v is None or kind(k) in alone else v * share) for k, v in phis.items()}


async def _mine(task: str):
    from crucible.workflow.filestate import FileState
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible_pack_coding.m56.task_properties import TASKS, TEST, base_reader, from_diff

    base = base_reader(task)
    oracle_patch = (TASKS / task / "solution" / "solution.patch").read_text(errors="replace")
    cands = [c.milestone for c in from_diff(oracle_patch, base, include=lambda p: not TEST.search(p), max_per_file=6)]
    oracle = FileState(base=base)
    oracle.apply_patch(oracle_patch)
    at_base = await evaluate(cands, FileStateView(code_state(FileState(base=base))))
    at_oracle = await evaluate(cands, FileStateView(code_state(oracle)))
    return [m for m in cands if at_base[m.id] == 0.0 and at_oracle[m.id] == 1.0]


_MS: dict[str, Any] = {}


async def _mine_cached(task: str):
    if task not in _MS:
        _MS[task] = await _mine(task)
    return _MS[task]


async def phi_of_state(task: str, fs) -> float:
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible.workflow.potential import potential

    ms = await _mine_cached(task)
    return potential(ms, gated(ms, await evaluate(ms, FileStateView(code_state(fs)))))[0]


async def _phi(task: str, patch: str | None) -> float:
    from crucible.workflow.filestate import FileState
    from crucible_pack_coding.m56.task_properties import base_reader

    fs = FileState(base=base_reader(task))
    if patch:
        fs.apply_patch(patch)
    return await phi_of_state(task, fs)


# the one tool the recorded runs were offered: mini-swe-agent 2.x tool-call mode (``src/minisweagent/models/utils/
# actions_toolcall.py:11-27``, MIT, SWE-agent/mini-swe-agent 04d809c). Label rows carry it as ``tools`` (M15 audit G1 D13).
MSWEA_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


async def credit(traces: list) -> dict[str, dict[str, Any]]:
    """For each recorded agent run: Φ of its final patch over the seed's oracle-derived milestones, the recorded outcome,
    and the composed dense reward. Other families (pull requests) carry no task-relative Φ here. A run graded under an
    earlier task version has no outcome for the current task (M14 audit V9), so it gets no reward. A run on a bench seed
    has ``split: bench``: ``crucible label`` leaves it out unless ``include_bench`` (M15 audit H1 V9)."""
    from crucible.credit.compose import compose
    from crucible_pack_coding.m56.task_properties import TASKS, recorded_outcome
    from crucible_pack_coding.seeds import SEEDS as SEED_INFO

    out = {}
    for t in traces:
        if t.family != "coding/agent_run" or not t.trace_id.startswith("run:"):
            continue
        task, run = t.trace_id[4:].split("/", 1)
        rd = TASKS / task / "runs" / run
        patch = (rd / "model.patch").read_text(errors="replace") if (rd / "model.patch").exists() else None
        y = recorded_outcome(task, rd)
        try:
            phi = await _phi(task, patch)
        except Exception as e:  # noqa: BLE001 - a patch that does not apply to the base is reported, not guessed
            out[t.trace_id] = {"reward": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        last = max((e.seq for e in t.events), default=0)
        rj = compose(gates={}, outcome=bool(y), phi=phi) if y is not None else None
        # reward_json lets label withhold the partial credit; runs of one task share a bootstrap group (M5 R1, credit.channels)
        out[t.trace_id] = {"reward": rj.reward if rj else None, "reward_json": rj.model_dump(mode="json") if rj else None, "group": task,
                           "phi_by_seq": {last: round(phi, 6)}, "phi_final": round(phi, 6),
                           "outcome": None if y is None else ("good" if y >= 1 else "bad"), "tools": [MSWEA_BASH_TOOL],
                           # a run on a bench seed is that seed's benchmark evidence: never training data (M15 audit H1 V9)
                           "split": "bench" if SEED_INFO.get(task.split("@")[0], {}).get("bench", True) else "train"}
    return out


# ------------------------------------------------------------------------------------------------ adversarial patches

def _target(m) -> str | None:
    q = m.check.query
    return q.args.get("path") or (q.args.get("paths") or [None])[0]


async def stuffed_state(task: str):
    """No migration at all: every added token appended as a comment, the removed tokens rewritten away, the deleted files
    deleted, a comment line in each anchored block, a comment-only file for each created one (M14 audit V4's probe)."""
    from crucible.workflow.filestate import FileState
    from crucible_pack_coding.m56.task_properties import base_reader

    fs = FileState(base=base_reader(task))
    for m in await _mine_cached(task):
        q, e, p = m.check.query, m.check.expect, _target(m)
        if p is None:
            continue
        cur = fs.read(p)
        mark = "#" if p.endswith(".py") or HASH_COMMENT.search(p) else "//"
        if q.op == "grep" and e.kind == "count_ge":
            tok = q.args["pattern"].replace(r"\s*\(", "(").replace(r"\b", "").replace("\\", "")
            fs.write(p, (cur or "") + f"\n{mark} {tok}\n")
        elif q.op == "grep":
            fs.write(p, re.sub(q.args["pattern"], "zz", cur or ""))
        elif q.op == "exists" and e.kind == "truthy":
            fs.write(p, (cur or "") + f"{mark} placeholder\n")
        elif q.op == "exists":
            fs.write(p, None)
        elif q.op == "block_changed":
            lines = (cur or "").split("\n")
            i = next((k for k, ln in enumerate(lines) if q.args["anchor"] in ln), None)
            if i is not None:
                lines.insert(i + 1, f"{mark} touched")
                fs.write(p, "\n".join(lines))
    return fs


async def destructive_state(task: str):
    """Only the destructive half of the oracle: delete what it deletes and remove what it removes, add nothing."""
    from crucible.workflow.filestate import FileState
    from crucible_pack_coding.m56.task_properties import base_reader

    fs = FileState(base=base_reader(task))
    for m in await _mine_cached(task):
        q, e, p = m.check.query, m.check.expect, _target(m)
        if p is None or kind(m.id) in POSITIVE:
            continue
        if q.op == "exists" and e.kind != "truthy":
            fs.write(p, None)
        elif q.op == "grep" and e.kind != "count_ge" and fs.read(p) is not None:
            fs.write(p, re.sub(q.args["pattern"], "zz", fs.read(p)))
    return fs


def _token(pattern: str) -> tuple[str, bool]:
    """A grep milestone's token and whether it is a call (``name\\s*\\(``)."""
    if pattern.endswith(r"\s*\("):
        return re.sub(r"\\(.)", r"\1", pattern[: -len(r"\s*\(")]), True
    return re.sub(r"\\(.)", r"\1", pattern.replace(r"\b", "")), False


def _stuff_line(path: str, n: int, tok: str, call: bool, mode: str) -> str:
    """Text that holds ``tok`` without running it: in a string literal (``literal``) or in dead code (``dead``, rotating
    over the forms the code view drops)."""
    k = lang(path)
    if k == "toml" or (k not in ("c", "py", "cmake") and mode == "literal"):
        return f'crucible_s{n} = "{tok}("' if call else f'crucible_s{n} = "{tok}"'
    if k not in ("c", "py", "cmake"):
        return f"# {tok}"
    if mode == "literal":
        lit = tok + ("(" if call else "")
        return {"py": f'_crucible_s{n} = "{lit}"', "cmake": f'set(_CRUCIBLE_S{n} "{lit}")',
                "c": (f'static const char *crucible_s{n} = R"x({lit})x";' if n % 2 else f'static const char *crucible_s{n} = "{lit}";')}[k]
    if k == "py":
        stmt = f"{tok}()" if call else tok
        return [f"if False:\n    {stmt}", f"def _crucible_dead{n}():\n    {stmt}",
                f"def crucible_ref{n}():\n    return None\n    {stmt}\n_crucible_keep{n} = crucible_ref{n}", f"_crucible_v{n} = False and {stmt}"][n % 4]
    if k == "cmake":
        stmt = f"{tok}()" if call else f"set(_crucible_v{n} {tok})"
        return [f"if(FALSE)\n  {stmt}\nendif()", f"function(_crucible_dead{n})\n  {stmt}\nendfunction()"][n % 2]
    stmt = f"{tok}();" if call else f"{{ {tok}; }}"
    return [f"#if 0\n{stmt}\n#endif", f"#ifdef CRUCIBLE_NEVER_DEFINED_{n}\n{stmt}\n#endif", f"static void crucible_dead{n}(void) {{ {stmt} }}",
            f"static int crucible_ref{n}(void) {{ if (0) {{ {stmt} }} return 0; {stmt} }}\nstatic int (*crucible_keep{n})(void) = crucible_ref{n};"][n % 4]


async def stuffed_literal_state(task: str, mode: str):
    """M14 campaign B2's probes: nothing migrated, every added token put in a string literal (``literal``) or in dead code
    (``dead``: ``#if 0``, an undefined macro, an unused function, ``if (0)`` and code after ``return``; Python ``if False:``,
    an unused ``def``, code after ``return``, ``False and x``; CMake ``if(FALSE)`` and an unused ``function``), the removed
    tokens rewritten away, the deleted files deleted, a live no-op statement in each anchored block, live placeholder
    code in each created file. Both scored Φ 1.0 on every seed before the code view."""
    from crucible.workflow.filestate import FileState
    from crucible_pack_coding.m56.task_properties import base_reader

    fs = FileState(base=base_reader(task))
    for n, m in enumerate(await _mine_cached(task)):
        q, e, p = m.check.query, m.check.expect, _target(m)
        if p is None:
            continue
        cur = fs.read(p)
        k = lang(p)
        if q.op == "grep" and e.kind == "count_ge":
            tok, call = _token(q.args["pattern"])
            fs.write(p, (cur or "") + "\n" + _stuff_line(p, n, tok, call, mode) + "\n")
        elif q.op == "grep":
            fs.write(p, re.sub(q.args["pattern"], "zz", cur or ""))
        elif q.op == "exists" and e.kind == "truthy":
            fs.write(p, (cur or "") + {"py": "_crucible_placeholder = 1\n", "cmake": "set(_CRUCIBLE_X 1)\n", "c": "static int crucible_placeholder;\n",
                                       "toml": "crucible_placeholder = 1\n"}.get(k, "placeholder\n"))
        elif q.op == "exists":
            fs.write(p, None)
        elif q.op == "block_changed":
            lines = (cur or "").split("\n")
            i = next((j for j, ln in enumerate(lines) if q.args["anchor"] in ln), None)
            if i is not None:
                lines.insert(i + 1, {"py": "    pass", "c": "    (void)0;"}.get(k, ""))
                fs.write(p, "\n".join(lines))
    return fs


# ------------------------------------------------------------------------------------------------ CPU tasks (hooks.tasks)

_PARTIAL: dict[str, Any] = {}  # task -> TemporaryDirectory holding the base files a seed without a local repo needs


async def _seed_repo(task: str):
    """The seed's base repository: the task's own ``repo``/``repos`` directory, or (task04, whose base is a commit of PCL's
    git) the base content of every path the oracle or a milestone touches, written once into a temp dir."""
    import tempfile
    from pathlib import Path

    from crucible_pack_coding.m56.task_properties import TASKS, base_reader

    d = TASKS / task
    for cand in (d / "repo", d / "repos" / {"task02": "pcops", "task03": "mdlite"}.get(task[:6], "-")):
        if cand.is_dir():
            return cand
    if task not in _PARTIAL:
        tmp = tempfile.TemporaryDirectory(prefix="coding-seed-")
        root = Path(tmp.name) / "repo"
        read = base_reader(task)
        paths = set((await _patched(task, (d / "solution" / "solution.patch").read_text(errors="replace"))).files)
        paths |= {_target(m) for m in await _mine_cached(task)}
        for p in sorted(x for x in paths if x):
            text = read(p)
            if text is not None:
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                (root / p).write_text(text)
        _PARTIAL[task] = tmp
    return Path(_PARTIAL[task].name) / "repo"


def _writes(task: str, fs) -> list:
    """The file writes that turn the seed's base repository into ``fs`` (a deleted file is written empty: no code)."""
    from crucible.core.schema.actions import ToolCall

    base = fs.base
    out = []
    for k, p in enumerate(sorted(p for p in fs.files if fs.files[p] != (base(p) if base else None))):
        out.append(ToolCall(tool_call_id=f"w{k}", function_name="repo.write_file", arguments={"path": p, "content": fs.files[p] or ""}))
    return out


class DenseReplayTask:
    """A CPU stand-in for a GPU seed (M14 sweep G2 D4: the pack had no ``hooks.tasks``, so its semantic level never ran):
    the seed's base repository in a workspace, graded by the code-only dense reward Φ over the oracle's milestones. Gold
    (the oracle's files written) reaches 1; the untouched repository scores 0. It checks the dense grader and its
    adversaries end to end through a World; the real outcome needs the seed's GPU verifier image."""

    world_id = "coding/dense_replay"

    def __init__(self, task: str):
        self.seed = task
        self.id = f"coding/dense_replay/{task}@cpu2"  # cpu2: the code view of M14 campaign B2
        self.instruction = (TASKS_DIR() / task / "instruction.md").read_text(errors="replace")

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the pack's content (the dense grader, the milestone miner, ``codeonly``: ``tree_digest(PACK_ROOT)``) and
        the seed's task directory, one digest per top-level entry (``instruction.md``, ``solution``, ``tests``...), whose oracle patch
        the milestones are mined from, so an edited oracle or test refuses the resume naming the part."""
        from crucible.runtime.identity import tree_digest, tree_digests
        from crucible_pack_coding import PACK_ROOT

        return {"pack": tree_digest(PACK_ROOT), "task": self.seed, "files": tree_digests(TASKS_DIR() / self.seed)}

    async def open(self, workdir):
        from pathlib import Path

        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        repo = await _seed_repo(self.seed)
        root = (Path(workdir) / "repo").resolve()
        spec = WorldSpec.model_validate({"world": self.world_id, "apps": [{"id": "repo", "template": "common/workspace",
                                         "state": {"adapter": "fs", "params": {"root": str(root)}}, "params": {"shell": False, "seed_dir": repo.name}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})
        w = World(spec, instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w", pack_root=repo.parent)
        w._dense_root = root
        await w.start()
        await w.reset()
        return w

    async def grade(self, world, workdir):
        from crucible.verify.runtime.plan import GradeReport
        from crucible.workflow.filestate import FileState
        from crucible_pack_coding.m56.task_properties import base_reader

        root = world._dense_root
        fs = FileState(base=base_reader(self.seed))
        for f in root.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                fs.files[str(f.relative_to(root))] = f.read_text(errors="replace")
        for m in await _mine_cached(self.seed):
            p = _target(m)
            if p is not None and p not in fs.files:
                fs.files[p] = None  # gone from the workspace
        phi = round(await phi_of_state(self.seed, fs), 6)
        return GradeReport(task=self.id, tier="full", checks=[], gates={"dense": "pass" if phi >= 1.0 else "fail"}, reward=phi, outcome=phi,
                           infra={"status": "ok"}, trusted=True, elapsed_s=0.0, meta={"phi": phi, "milestones": len(await _mine_cached(self.seed)),
                                                                         "kinds": {"dense": "outcome"}})  # Φ reaching 1: an outcome, not a rule


def TASKS_DIR():
    from crucible_pack_coding.m56.task_properties import TASKS

    return TASKS


async def _patched(task: str, patch: str):
    from crucible.workflow.filestate import FileState
    from crucible_pack_coding.m56.task_properties import base_reader

    fs = FileState(base=base_reader(task))
    fs.apply_patch(patch)
    return fs


async def replay_task(task: str) -> DenseReplayTask:
    t = DenseReplayTask(task)
    t.attacks = {"comment_stuffing": _writes(task, await stuffed_state(task)), "destructive_only": _writes(task, await destructive_state(task)),
                 "string_literal_stuffing": _writes(task, await stuffed_literal_state(task, "literal")),
                 "dead_code_stuffing": _writes(task, await stuffed_literal_state(task, "dead"))}
    t.gold = _writes(task, await _patched(task, (TASKS_DIR() / task / "solution" / "solution.patch").read_text(errors="replace")))
    return t


async def tasks() -> list:
    """The CPU dense-replay task of every bench seed with its gold (the oracle's files written)."""
    out = []
    for s in SEEDS:
        if (TASKS_DIR() / s / "solution" / "solution.patch").exists():
            t = await replay_task(s)
            out.append((t, list(t.gold)))
    return out


ATTACKS = ("comment_stuffing", "destructive_only", "string_literal_stuffing", "dead_code_stuffing")


def adversaries() -> list[dict]:
    """Comment, string-literal and dead-code stuffing and the destructive half of the oracle, written into the workspace
    (built by ``tasks()``): each earns Φ 0 on every task."""
    return [{"name": f"adversary:{name}", "policy": (lambda task, name=name: task.attacks[name]), "expect_max": 0.0} for name in ATTACKS]


def honest_failures(task: str) -> list[str]:
    """Recorded failing runs graded under the task's current version that left a patch."""
    from crucible_pack_coding.m56.task_properties import TASKS, recorded_outcome

    runs = TASKS / task / "runs"
    return [rd.name for rd in sorted(runs.glob("*")) if (rd / "model.patch").exists() and recorded_outcome(task, rd) == 0.0] if runs.is_dir() else []


async def fixtures() -> list[dict]:
    """Fixtures over the code-only dense reward of every bench seed, graded by the pack's own CPU task through a World
    (M14 sweep D4: the oracle case was ``compose(outcome=True)``, 1.0 whatever Φ was, so no case exercised Φ):
    ``oracle`` (Φ 1), ``base`` (no-op, Φ 0), ``stuffed`` (the oracle's tokens in comments, the old code deleted, a comment in
    each anchored block: Φ 0), ``destructive`` (the oracle's deletions and removals alone: Φ 0), ``string_literal`` and
    ``dead_code`` (the oracle's tokens in string literals or dead code, campaign B2: Φ 0), and each recorded honest
    failure on the current task version at its exact Φ. The adversaries are bounded at the semantic level."""
    cases = []
    for task in SEEDS:
        if not (TASKS_DIR() / task / "solution" / "solution.patch").exists():
            continue
        t = await replay_task(task)
        cases.append({"name": f"{task}:oracle", "task": t, "actions": list(t.gold), "reward": 1.0})
        cases.append({"name": f"{task}:base", "task": t, "actions": [], "reward": 0.0, "fails_gate": "dense"})
        cases.append({"name": f"{task}:stuffed", "task": t, "actions": list(t.attacks["comment_stuffing"]), "reward": 0.0, "fails_gate": "dense"})
        cases.append({"name": f"{task}:destructive", "task": t, "actions": list(t.attacks["destructive_only"]), "reward": 0.0, "fails_gate": "dense"})
        cases.append({"name": f"{task}:string_literal", "task": t, "actions": list(t.attacks["string_literal_stuffing"]), "reward": 0.0, "fails_gate": "dense"})
        cases.append({"name": f"{task}:dead_code", "task": t, "actions": list(t.attacks["dead_code_stuffing"]), "reward": 0.0, "fails_gate": "dense"})
        for run in honest_failures(task):
            fs = await _patched(task, (TASKS_DIR() / task / "runs" / run / "model.patch").read_text(errors="replace"))
            cases.append({"name": f"{task}:honest_failure:{run}", "task": t, "actions": _writes(task, fs), "reward": round(await phi_of_state(task, fs), 6),
                          "fails_gate": "dense"})
    return cases + adversaries()
