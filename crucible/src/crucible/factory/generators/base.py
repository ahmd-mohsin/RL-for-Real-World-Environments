"""What every generator shares (swarm-pipeline/00 §1, §2, §4, §15 P5): the ``Generator`` Protocol, the synthetic history
the window machinery runs on, the deterministic gates, authoring in the generator's voice, and the funnel.

**One contract, one machine.** A generator turns typed source records (``factory.sources``) into a *window record*: a
start tree A, an oracle tree B, and units between them. The mined window's machinery then runs unchanged on it
(``windows.validate_window``: the World image at A, the leak probes, fail-to-pass and pass-to-pass with reruns, prefix
ownership, the executed dependency prober, commit-revert mutants), and the task is a ``windows.WindowTask`` subclass, so
it is a ``factory.longtask.LongTask`` and every P0-P4 gate handler reads it as it reads a mined window.

**Synthetic history.** A generator whose units are not consecutive commits of one repository writes them as commits of
its own: ``derived_clone`` (a ``--shared`` clone, so the source clone is never written) or ``init_mono`` (a repository
whose object store borrows several clones'), ``apply_chain`` (trees from a base plus forward or reverse patches, 3-way,
through a private index; a conflict stops the chain and is reported) and ``write_chain`` (one commit per tree under a
private ref ``refs/crucible/<generator>/<id>``). The World never sees this history: it is built from A's tree alone
(``history: cut``), and the leak probes check that.

**Gates** (``deterministic_gates``), cheap first (``00`` §4, "the funnel order is fixed"):
- S6: at least one executed ``requires`` edge (``longtask.check_longtask``, long-horizon on);
- S5: the P2 handler ``verifier2.s5_solvability`` (oracle from the clean World, null agent, fairness lint);
- S8: the P2 handler ``verifier2.s8_ground_truth`` (floors 0 with Phi 0, every mutant fails its milestone, oracle and
  alternatives 1, the ten-competent-people test). Its "two valid alternatives" finding needs P2's paid rewriters and is
  reported apart (``alternatives_pending``), never hidden;
- S3: ``windows_s3.world_s3`` (builds, A's suite runs, leak probes, 60 clean reruns at concurrency 1 and 4, the verifier
  does not move);
- S10: ``flake.measure_flake`` (30 oracle grades at concurrency 1 and 30 at 4, Clopper-Pearson bound <= 5%).

Before a prompt exists the checks are bound to placeholder requirements (``gate_requirements``: one per milestone, plus
the regression requirement), so the check set the gates measure is the one the authored task keeps (the checks depend
only on the requirement id <-> milestone map, which authoring never changes).

**Authoring** (``author_generated``): ``windows.author_window``'s ledger, lint and required-name rules with the
generator's brief and heading (a bug report for rebreak, a release request for a stitched window).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from crucible.factory.lanes.executable import REGRESSION_REQ, REGRESSION_TEXT, call_supported
from crucible.factory.lanes.windows import (
    EPILOGUE,
    WindowLane,
    WindowStore,
    WindowTask,
    author_payload,
    lint_prompt,
    locate,
    unstated_names,
    window_checkset,
    window_ledger,
)
from crucible.factory.lanes.windows_git import RepoGit

GENERATORS = ("rebreak", "stitched_window", "open")
WORLD_PREFIX = "crucible-lh-p5-"  # World, leak-probe and S3 containers of this phase
GRADE_PREFIX = "crucible-p5-"  # grading containers of this phase
GIT_ID = {"GIT_AUTHOR_NAME": "crucible-factory", "GIT_AUTHOR_EMAIL": "factory@crucible.local", "GIT_COMMITTER_NAME": "crucible-factory",
          "GIT_COMMITTER_EMAIL": "factory@crucible.local", "GIT_CONFIG_NOSYSTEM": "1"}


# ---- the Protocol ------------------------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """One unit of generator work: the source records it came from (``factory.sources.SourceRecord`` JSON) and the
    generator's own parameters (shas, tags, chosen fixes)."""

    id: str
    generator: str
    repo: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Generator(Protocol):
    """A task generator (``00`` §2): ``candidates`` from source records, ``build`` one candidate into a stored record
    (whatever stage it reached, with the reason it stopped), ``task`` a stored record as a ``LongTask``."""

    name: str

    async def candidates(self) -> list[Candidate]: ...

    async def build(self, cand: Candidate) -> dict[str, Any]: ...

    def task(self, record_id: str, *, authored: Mapping[str, Any] | None = None) -> Any: ...


# ---- synthetic history -------------------------------------------------------------------------------------------------------
def git(repo: Path, *args: str, input: bytes | None = None, env: Mapping[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, input=input, check=False,
                       env={**os.environ, **GIT_ID, **dict(env or {})})
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r


def out(r: subprocess.CompletedProcess) -> str:
    return r.stdout.decode("utf-8", "replace").strip()


def derived_clone(src: Path, dest: Path) -> Path:
    """A clone of ``src`` that shares its objects (``--shared``: alternates), with every tag; the generator writes its
    synthetic commits here, so ``src`` is never written."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(src), str(dest)], capture_output=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"derived clone of {src}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    return dest


def objects_dir(repo: Path) -> Path:
    return Path(out(git(Path(repo), "rev-parse", "--absolute-git-dir"))) / "objects"


def init_mono(dest: Path, sources: Sequence[Path]) -> Path:
    """A repository whose object store borrows every ``sources`` clone's (alternates, followed transitively), so trees of
    several repositories can be combined under prefixes and committed."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git(dest, "init", "-q")
    alt = dest / ".git" / "objects" / "info" / "alternates"
    alt.parent.mkdir(parents=True, exist_ok=True)
    have = alt.read_text().split() if alt.exists() else []
    want = [str(objects_dir(Path(s))) for s in sources]
    alt.write_text("\n".join(dict.fromkeys([*have, *want])) + "\n")
    return dest


def tree_of(repo: Path, rev: str) -> str:
    return out(git(Path(repo), "rev-parse", f"{rev}^{{tree}}"))


def full_patch(repo: Path, parent: str, sha: str, paths: Sequence[str] = ()) -> bytes:
    """The commit's whole diff (every file, or ``paths``), with full index lines for ``--3way``."""
    return git(Path(repo), "diff", "--full-index", "--binary", parent, sha, "--", *paths).stdout


def apply_chain(repo: Path, base: str, steps: Sequence[tuple[bytes, bool]]) -> tuple[list[str], int | None]:
    """Trees after each step, starting from ``base`` (a commit or tree): ``(patch, reverse)`` applied with ``git apply
    --cached --3way`` through a private index. Returns ``(trees, None)``, or ``(trees so far, index of the step that
    did not apply)``."""
    repo = Path(repo)
    with tempfile.TemporaryDirectory(prefix="p5idx-") as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git(repo, "read-tree", base, env=env)
        trees: list[str] = []
        for i, (patch, rev) in enumerate(steps):
            if patch.strip():
                r = git(repo, "apply", "--cached", "--3way", "--whitespace=nowarn", *(["-R"] if rev else []), "-", input=patch, env=env, check=False)
                if r.returncode != 0:
                    return trees, i
            w = git(repo, "write-tree", env=env, check=False)
            if w.returncode != 0:
                return trees, i
            trees.append(out(w))
        return trees, None


def with_files(repo: Path, tree: str, files: Mapping[str, str | None]) -> str:
    """``tree`` with ``files`` written (text) or removed (None), as a new tree."""
    if not files:
        return tree
    with tempfile.TemporaryDirectory(prefix="p5idx-") as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git(Path(repo), "read-tree", tree, env=env)
        for path, text in files.items():
            if text is None:
                git(Path(repo), "update-index", "--force-remove", "--", path, env=env)
                continue
            blob = out(git(Path(repo), "hash-object", "-w", "--stdin", input=text.encode()))
            git(Path(repo), "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        return out(git(Path(repo), "write-tree", env=env))


def show(repo: Path, rev: str, path: str) -> str | None:
    r = git(Path(repo), "show", f"{rev}:{path}", check=False)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def changed_paths(repo: Path, a: str, b: str) -> list[str]:
    return [p for p in out(git(Path(repo), "diff", "--name-only", "--no-renames", a, b)).splitlines() if p]


def combine_trees(repo: Path, parts: Mapping[str, str]) -> str:
    """One tree holding each ``{prefix: tree-ish}`` under ``prefix/`` (objects reachable from ``repo``)."""
    with tempfile.TemporaryDirectory(prefix="p5idx-") as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        git(Path(repo), "read-tree", "--empty", env=env)
        for prefix, rev in parts.items():
            git(Path(repo), "read-tree", f"--prefix={prefix.strip('/')}/", rev, env=env)
        return out(git(Path(repo), "write-tree", env=env))


def write_chain(repo: Path, trees: Sequence[str], *, messages: Sequence[str], dates: Sequence[str], parent: str | None, ref: str) -> list[str]:
    """One commit per tree, each the child of the one before (the first a child of ``parent``), deterministic
    identity and dates; ``ref`` points at the last."""
    shas: list[str] = []
    prev = parent
    for tree, msg, date in zip(trees, messages, dates, strict=True):
        args = ["commit-tree", tree, *(["-p", prev] if prev else []), "-F", "-"]
        sha = out(git(Path(repo), *args, input=(msg.strip() or "change").encode(), env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}))
        shas.append(sha)
        prev = sha
    if shas:
        git(Path(repo), "update-ref", ref, shas[-1])
    return shas


def commit_meta(repo: Path, sha: str) -> dict[str, str]:
    r = out(git(Path(repo), "log", "-1", "--format=%H%x1f%P%x1f%aI%x1f%s%x1f%b", sha))
    h, p, d, s, b = (r.split("\x1f") + ["", "", "", "", ""])[:5]
    return {"sha": h, "parent": (p.split() or [""])[0], "date": d, "subject": s, "body": b.strip()[:4000]}


def drop_crlf_prose(rec: dict[str, Any], base_dir: Path) -> list[str]:
    """Remove from ``rec["oracle_files"]`` the non-Python files whose start copy has CRLF line endings: the window lane's
    text round trip loses the CRs, so the oracle patch would not apply on them (dateutil's NEWS), and prose is never
    graded. Returns the paths dropped."""
    drop = [p for p in rec.get("oracle_files") or {} if not p.endswith(".py") and (Path(base_dir) / p).is_file()
            and b"\r\n" in (Path(base_dir) / p).read_bytes()]
    for p in drop:
        rec["oracle_files"].pop(p, None)
    return drop


# ---- tasks -------------------------------------------------------------------------------------------------------------------
def gate_requirements(rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Placeholder requirements binding the checks before a prompt exists: one per milestone (same ids an authored
    prompt gets: ``R1..Rn`` in milestone order), plus the regression requirement when the record has regression tests."""
    reqs = [{"id": f"R{i}", "milestone": m["id"], "text": f"(unauthored) {m.get('subject') or m['id']}", "kind": "behaviour"}
            for i, m in enumerate(rec.get("milestones") or [], 1)]
    if set(rec.get("p2p") or []) | set(rec.get("collateral") or []):
        reqs.append({"id": REGRESSION_REQ, "milestone": None, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
    return reqs


class GeneratedTask(WindowTask):
    """A window task a generator built: ``generator`` names it (``targeting.generator_of``), the card's lineage and
    family say so, and ``simplify`` takes the generator's kinds (``targeting.SIMPLIFY_KINDS``)."""

    generator = "generated"

    def oracle_steps(self) -> list[dict[str, Any]]:
        """B as the window's unit steps, without the CRLF prose files ``drop_crlf_prose`` took out of the oracle."""
        from crucible.factory.lanes.windows_git import Unit

        dropped = set((self.window.get(self.generator) or {}).get("crlf_prose_dropped") or [])
        if not dropped or self.window["repo"] not in self.lane.repo_dirs or self.window.get("simplified_from"):
            return super().oracle_steps()
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]]))
        steps = []
        for u in self.window["units"]:
            unit = Unit(**{**u, "files": [p for p in u["files"] if p not in dropped]})
            patch = repo.patch(unit)
            if patch.strip():
                steps.append({"function_name": "apply_unit", "arguments": {"unit": unit.sha, "patch": patch.decode("utf-8", "replace")}})
        return steps

    def card(self, **kw: Any) -> Any:
        c = super().card(**kw)
        src = {k: v for k, v in (self.window.get(self.generator) or {}).items() if k in ("b", "b_ref", "host", "consumer", "fixes", "pair")}
        return c.model_copy(update={"family": f"{self.generator.replace('_', '-')}-{self.window['repo']}",
                                    "lineage": {**c.lineage, "generator": self.generator, "source": src}})


    def simplify_units(self, pre_units: set[str], *, kind: str, root: Path | None = None) -> Any:
        """The task with ``pre_units`` (unit shas) pre-applied to A (source only; the prefix up to the last of them when they
        do not apply alone): the milestones they own are dropped with their requirements and mutants, every other milestone
        keeps its checks with ``requires`` on dropped milestones removed (``windows.WindowTask.simplify``'s rule, for any set
        of units: a stitched window pre-applies the whole host change). A changed task: S8 runs on it again."""
        import json as _json

        from crucible.factory.lanes.executable import sha256_text
        from crucible.factory.lanes.windows import guard_tests, locate, mined_of
        from crucible.factory.lanes.windows_git import Unit

        if not pre_units:
            raise ValueError("nothing to pre-apply")
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]]))
        units = [Unit(**u) for u in self.window["units"]]
        key = sha256_text(self.mined.id, kind, *sorted(pre_units))[:10]
        dest = (Path(root) if root is not None else Path(self.lane.store.root) / "simplified").resolve()
        new_id = f"{self.mined.id}~{kind[:4]}-{key}"
        base = dest / new_id / "base"
        mode = "exact"
        ok, _ = repo.build(self.window["a"], [u for u in units if u.sha in pre_units], base, source_only=True)
        if not ok:
            last = max(i for i, u in enumerate(units) if u.sha in pre_units)
            pre_units = {u.sha for u in units[: last + 1]}
            ok, bad = repo.build(self.window["a"], units[: last + 1], base, source_only=True)
            mode = "prefix"
            if not ok:
                raise RuntimeError(f"the units do not apply at {bad}")
        pre_ms = sorted(m["id"] for m in self.milestones if m["unit"] in pre_units)
        # an observation milestone (no unit of its own: a stitched window's consumer tests) whose prerequisites are all
        # pre-applied already holds at the new start: it goes with them
        pre_ms = sorted({*pre_ms, *(m["id"] for m in self.milestones if m.get("unit") is None and m.get("requires") and set(m["requires"]) <= set(pre_ms))})
        keep = []
        for m in self.milestones:
            if m["id"] in pre_ms:
                continue
            m2 = _json.loads(_json.dumps(m))
            m2["requires"] = [r for r in m2.get("requires") or [] if r not in pre_ms]
            m2["requires_units"] = [u for u in m2.get("requires_units") or [] if u not in pre_units]
            keep.append(m2)
        if not keep:
            raise ValueError("every milestone would be pre-applied: nothing is left to solve")
        keep_ids = {m["id"] for m in keep}

        def now(p: str) -> str | None:
            f = base / p
            return f.read_text(errors="replace") if f.is_file() else None

        rec = _json.loads(_json.dumps(self.window, default=str))
        # the pre-applied milestones' own tests pass at the new start: they guard the regression check (c_regress), so an
        # agent cannot undo an earlier fix for free (it keeps Phi but loses the full pass)
        guards = guard_tests(self.milestones, pre_ms)
        rec.update(p2p=sorted({*(rec.get("p2p") or []), *guards}))
        rec.update(id=new_id, milestones=keep, oracle_files={p: t for p, t in self.window["oracle_files"].items() if now(p) != t},
                   mutants=[x for x in self.window.get("mutants") or [] if x.get("milestone") in keep_ids],
                   dependency_edges=sum(len(m["requires"]) for m in keep),
                   simplified_from={"task": self.task_id, "window": self.mined.id, "kind": kind, "targets": sorted(keep_ids), "pre_applied": pre_ms,
                                    "units": sorted(pre_units), "mode": mode, "guards": guards})
        reqs = [dict(r) for r in self.requirements if r.get("milestone") is None or r.get("milestone") in keep_ids]
        dropped = [r["id"] for r in self.requirements if r.get("milestone") and r.get("milestone") not in keep_ids]
        prompt = "".join(ln for ln in self.instruction.splitlines(keepends=True) if not any(ln.startswith(f"- [{d}] ") for d in dropped))
        spans = {r["id"]: list(loc) for r in reqs if (loc := locate(prompt, r["text"]))}
        t = type(self)(self.lane, mined_of(rec, base), rec, task_id=f"{self.task_id}~{kind[:4]}-{key[:6]}", prompt=prompt, requirements=reqs,
                       spans=spans, checkset=window_checkset(rec, reqs), phi_weight=self.phi_weight, import_tolerant=self.import_tolerant)
        t.world_image, t.workspace_mode = self.world_image, self.workspace_mode
        return t


class GenLane(WindowLane):
    """The window lane for one generator: its store, its clones (synthetic refs live there), its grading sandboxes and
    its task class. ``task(wid)`` is the stored record as a task, unauthored (placeholder requirements) or authored."""

    def __init__(self, store: WindowStore, *, generator: str, task_cls: type[GeneratedTask], repo_dirs: Mapping[str, str], sandboxes: Any,
                 phi_weight: float = 0.5):
        super().__init__(store, repo_dirs=repo_dirs, sandboxes=sandboxes, phi_weight=phi_weight)
        self.generator = generator
        self.task_cls = task_cls
        self.world_prefix = f"{WORLD_PREFIX}world-"
        self._made: dict[str, GeneratedTask] = {}

    def task(self, wid: str, *, authored: Mapping[str, Any] | None = None) -> GeneratedTask:
        key = wid + ("#a" if authored else "")
        if key not in self._made:
            rec = self.store.window(wid)
            mined = self.store.get(wid)
            if authored:
                reqs, spans, prompt = list(authored["requirements"]), dict(authored.get("spans") or {}), str(authored["prompt"])
            else:
                reqs, spans, prompt = gate_requirements(rec), {}, ""
            self._made[key] = self.task_cls(self, mined, rec, task_id=wid, prompt=prompt, requirements=reqs, spans=spans,
                                            checkset=window_checkset(rec, reqs))
        return self._made[key]


# ---- deterministic gates ---------------------------------------------------------------------------------------------------
@dataclass
class GateConfig:
    s3_reruns: int = 60
    s3_concurrency: int = 4
    flake_reruns: int = 30  # per level
    flake_levels: tuple[int, ...] = (1, 4)
    stop_on_fail: bool = True
    gates: tuple[str, ...] = ("S6", "S5", "S8", "S3", "S10")


def executed_edges(task: Any) -> int:
    from crucible.factory.longtask import _executed

    by_id = {str(m.get("id")): m for m in task.milestones}
    return sum(1 for m in task.milestones for r in m.get("requires") or [] if r in by_id and _executed(m, r, by_id))


async def deterministic_gates(task: Any, *, cfg: GateConfig | None = None, s3_sandboxes: Any = None, s3_prefix: str = f"{WORLD_PREFIX}s3-"
                              ) -> dict[str, Any]:
    """S6, S5, S8, S3, S10 on one task, in that order (see the module doc); each gate ``{ok, findings, evidence, seconds}``.
    ``ok`` is every gate run passing; with ``stop_on_fail`` the first failure ends the run (later gates ``not_run``)."""
    from crucible.factory.flake import FlakeConfig, measure_flake
    from crucible.factory.lanes.windows_s3 import world_s3
    from crucible.factory.longtask import check_longtask
    from crucible.factory.verifier2 import P2Gates, s5_solvability, s8_ground_truth

    cfg = cfg or GateConfig()
    rec = task.window
    gates: dict[str, Any] = {}
    services = P2Gates(task=task)
    ctx = SimpleNamespace(services=services)

    async def s6() -> dict[str, Any]:
        n = executed_edges(task)
        free = bool(rec.get("allow_independent"))  # a window built for separate bugs (validate_window allow_independent)
        problems = check_longtask(task, long_horizon=not free, need_prompt=False)
        return {"ok": (n >= 1 or free) and not problems, "findings": problems,
                "evidence": {"executed_edges": n, "milestones": len(task.milestones), "allow_independent": free}}

    async def s5() -> dict[str, Any]:
        v = (await s5_solvability(ctx)).outputs["verdict"]
        return {"ok": v["ok"], "findings": v["findings"], "evidence": v["evidence"]}

    async def s8() -> dict[str, Any]:
        v = (await s8_ground_truth(ctx)).outputs["verdict"]
        alt = [f for f in v["findings"] if "valid alternative(s), need 2" in f]
        rest = [f for f in v["findings"] if f not in alt]
        gt = services.results.get("s8", {}).get("ground_truth", {})
        rows = {k: {kk: r.get(kk) for kk in ("kind", "reward", "phi", "ok")} for k, r in (gt.get("rows") or {}).items()}
        return {"ok": not rest, "findings": rest, "alternatives_pending": alt, "evidence": {**v["evidence"], "rows": rows}}

    async def s3() -> dict[str, Any]:
        r = await world_s3(rec, task.base_dir, rec["world"]["image"], reruns=cfg.s3_reruns, concurrency=cfg.s3_concurrency, sandboxes=s3_sandboxes,
                           name_prefix=s3_prefix)
        return {"ok": r["ok"], "findings": [r["why"]] if r.get("why") else [], "evidence": {"checks": r["checks"], "reruns": r.get("reruns"),
                                                                                            "existing": {k: (r.get("existing_tests") or {}).get(k) for k in ("targets", "counts")},
                                                                                            "leak": (r.get("leak") or {}).get("probes"), "grading": r.get("grading")}}

    async def s10() -> dict[str, Any]:
        patch = task.oracle_patch()
        rep = await measure_flake(lambda i: task.grade_patch(patch, trail=[]), config=FlakeConfig(reruns=cfg.flake_reruns, concurrency=cfg.flake_levels))
        d = rep.to_dict()
        return {"ok": rep.ok, "findings": d["reasons"], "evidence": {k: d[k] for k in ("levels", "n", "false_negatives", "masked", "upper")}}

    fns = {"S6": s6, "S5": s5, "S8": s8, "S3": s3, "S10": s10}
    failed = False
    for g in cfg.gates:
        if failed and cfg.stop_on_fail:
            gates[g] = {"ok": None, "findings": ["not_run: an earlier gate failed"]}
            continue
        t0 = time.monotonic()
        try:
            res = await fns[g]()
        except Exception as e:  # noqa: BLE001 - a crashed gate is a failed gate with its reason, never a pass
            res = {"ok": False, "findings": [f"error: {type(e).__name__}: {e}"[:400]], "evidence": {}}
        res["seconds"] = round(time.monotonic() - t0, 1)
        gates[g] = res
        failed = failed or not res["ok"]
    return {"task": task.task_id, "generator": getattr(task, "generator", ""), "gates": gates,
            "ok": all(gates[g]["ok"] for g in cfg.gates), "passed": [g for g in cfg.gates if gates[g]["ok"]]}


# ---- authoring -----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AuthorStyle:
    """The voice of a generator's request: the author's brief, the heading over the requirement list."""

    role: str
    brief: str
    heading: str


def compose_prompt(opening: str, requirements: Sequence[Mapping[str, str]], closing: str = "", *, heading: str, epilogue: str = EPILOGUE) -> str:
    body = "\n".join(f"- [{r['id']}] {r['text']}" for r in requirements)
    return (f"{opening.strip()}\n\n{heading}\n{body}" + (f"\n\n{closing.strip()}" if closing.strip() else "") + f"\n\n{epilogue}")


async def author_generated(client: Any, rec: Mapping[str, Any], repo: RepoGit, base_dir: Path, *, style: AuthorStyle, findings: Sequence[str] = (),
                           extra: Mapping[str, Any] | None = None, max_tokens: int = 12000, sample_key: str | None = None) -> dict[str, Any]:
    """``windows.author_window`` in the generator's voice: the payload (messages, public names, changelog, required
    names), one requirement per milestone id, lint, the two-way ledger, unstated required names stated by the lane.
    ``extra`` joins the payload (e.g. the two repositories of a stitched window)."""
    # an observation milestone (no unit of its own) is described by the unit it observes
    view = {**rec, "milestones": [{**m, "unit": m.get("unit") or m.get("observes")} for m in rec["milestones"]]}
    payload = {**author_payload(view, repo, base_dir), **dict(extra or {})}
    for k in ("changelog", "readme_head"):  # release numbers in the author's input end up stated in the request
        if isinstance(payload.get(k), str):
            payload[k] = VERSION_RX.sub("<version>", payload[k])
    user = {**payload, **({"findings_to_address": list(findings)} if findings else {})}
    msgs = [{"role": "system", "content": f"ROLE: {style.role}\n{style.brief}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(user, sort_keys=True)}]
    got_raw = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0, **({"sample_key": sample_key} if sample_key else {}))
    problems: list[str] = []
    ids = [m["id"] for m in rec["milestones"]]
    reqs_in = got_raw.get("requirements") if isinstance(got_raw, dict) else None
    got = {str(r.get("id")): " ".join(str(r.get("text") or "").split()) for r in reqs_in or [] if isinstance(r, dict)}
    problems += [f"unknown_requirement_id: {k} is not a change id" for k in got if k not in ids]
    problems += [f"unmapped_milestone: change {k} has no requirement" for k in ids if not got.get(k)]
    requirements = [{"id": f"R{i}", "milestone": mid, "text": got[mid], "kind": "behaviour"} for i, mid in enumerate(ids, 1) if got.get(mid)]
    opening = str((got_raw or {}).get("opening") or "").strip()
    closing = str((got_raw or {}).get("closing") or "").strip()
    if not opening:
        problems.append("author.opening is empty")
    authored = "\n".join([opening, closing, *[r["text"] for r in requirements]])
    problems += lint_prompt(authored)
    # a version number is a fact the grader never checks and the model tends to invent ("releasing version 0.13.2")
    problems += [f"lint version: {m.group(0)!r} (state behaviour, not version numbers)" for m in VERSION_RX.finditer(authored)]
    missing = [f.split("'")[1] for f in unstated_names(payload, authored)]
    if missing:
        closing = (closing + "\n\n" if closing else "") + f"The change must also expose these public names: {', '.join(missing)}."
    regress = bool(set(rec.get("p2p") or []) | set(rec.get("collateral") or []))
    if regress:
        requirements.append({"id": REGRESSION_REQ, "milestone": None, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
    prompt = compose_prompt(opening, [r for r in requirements if r["id"] != REGRESSION_REQ],
                            closing + (("\n\n" if closing else "") + REGRESSION_TEXT if regress else ""), heading=style.heading)
    spans = {r["id"]: list(loc) for r in requirements if (loc := locate(prompt, r["text"]))}
    _, ledger_problems = window_ledger(rec, requirements, spans, prompt)
    problems += ledger_problems
    return {"prompt": prompt, "requirements": requirements, "spans": spans, "checkset": window_checkset(rec, requirements), "findings": problems,
            "names_stated_by_lane": missing, "payload_keys": sorted(payload)}


VERSION_RX = re.compile(r"\b(?:version|release|v)\s*\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+\b", re.IGNORECASE)


# ---- the funnel ------------------------------------------------------------------------------------------------------------------
STAGES = ("S1_source", "S1_executed", "S3_world", "S4_oracle", "S6", "S5", "S8", "S3", "S10")


class Funnel:
    """Candidates -> each state -> accepted, with the reason at every drop (``00`` §9 yield ledger), one JSONL row per
    (candidate, stage) under ``path``; the last row of a (candidate, stage) wins."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def note(self, generator: str, cand: str, stage: str, ok: bool, reason: str = "", **evidence: Any) -> None:
        row = {"t": round(time.time(), 1), "generator": generator, "candidate": cand, "stage": stage, "ok": bool(ok), "reason": reason[:300],
               **({"evidence": evidence} if evidence else {})}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        last: dict[tuple[str, str, str], dict[str, Any]] = {}
        for ln in self.path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                last[(r["generator"], r["candidate"], r["stage"])] = r
        return list(last.values())

    def summary(self) -> dict[str, Any]:
        """Per generator and stage: candidates in, passed, and the drop reasons (first token of each reason)."""
        outp: dict[str, Any] = {}
        for r in self.rows():
            g = outp.setdefault(r["generator"], {})
            s = g.setdefault(r["stage"], {"in": 0, "pass": 0, "reasons": {}})
            s["in"] += 1
            if r["ok"]:
                s["pass"] += 1
            else:
                k = re.split(r"[:(]", r.get("reason") or "fail")[0].strip()[:70] or "fail"
                s["reasons"][k] = s["reasons"].get(k, 0) + 1
        return {g: {st: v[st] for st in [*STAGES, *sorted(set(v) - set(STAGES))] if st in v} for g, v in outp.items()}
