"""M17 first end-to-end long-horizon coding run (report 11): 10 release windows, CPU only, cheap OpenRouter models.

    cd <repo root>   # the OpenRouter key resolves from the nearest .env
    python crucible/scripts/lh_run.py --work <dir> --store <train store> --clones <clones> --out <runs dir> --cap 18 <stage> ...

Stages (each resumable; a JSON journal per window under ``<work>/journal/<window>.json``; every paid call is booked on one
forge ``Ledger`` (``factory.models.booked``) whose spend persists in ``<work>/spend.jsonl``, and every client is guarded so
no call starts once the run's total spend reaches ``--cap``):

- ``select``   the 10 training windows (>= 1 dependency edge, >= 1 valid revert mutant, >= 4 repos, <= 3 per repo,
               milestone counts spread) and why;
- ``prompt``   author (qwen3.7-flash) -> ledger + lint -> realism review (deepseek-v4-flash), <= 3 revisions;
- ``floors``   no-op, start state, A's own source rewritten, ``pass``-stubs of B's new symbols: outcome 0, Phi 0; the
               oracle (B's source diff and B as unit steps): outcome 1, Phi 1; every valid revert mutant fails its milestone;
- ``flake``    oracle grade reruns at concurrency 1 and 4 (``factory.flake.measure_flake``), Clopper-Pearson bound;
- ``hacker``   deepseek-v4-flash as an agent in the World (``WorkspaceHarness``) told to pass the hidden tests without the work;
- ``pilot``    one window x one rung, cost per episode, the extrapolation;
- ``solvers``  the three rungs x ``--attempts`` under ``WorkspaceHarness`` + ``LLMToolPolicy`` with the checkpoint sidecar;
- ``audit``    the auditor (deepseek-v4-flash, ``factory.audit``) on every solver episode, seeded controls mixed in;
- ``horizon``  2-3 milestones of 3 windows as isolated short tasks (A + every earlier unit), qwen x 4, vs in context;
- ``metrics``  ``bench.lh_metrics`` per window x rung, cause mix, the band and a verdict per window -> ``lh_run.json``;
- ``cleanup``  removes this run's ``crucible-lh-*`` containers.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import dataclasses
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded
from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy
from crucible.core.schema.actions import Done
from crucible.factory.models import ModelBook, booked
from crucible.forge.budget import Ledger

# ---- the selection (study 08 §9 step 5, first 10 windows) -------------------------------------------------------------
# Chosen from the 48 training windows at ``mutants_valid`` (report 09): every one has >= 1 executed milestone edge and >= 1
# valid revert mutant; grading takes <= 16 s (each episode grades up to 21 snapshots and the flake gate 60 oracles, so
# pyparsing's 100-190 s and lark's 22-47 s suites were left out); 7 repos, <= 2 per repo; milestones 4 to 10.
SELECTED = (
    "python-semver-w1bafb9f-3f92aa5",
    "python-semver-w41775dd-1bafb9f",
    "parse-wc669f87-35b9ffb",
    "parse-w07c2ff9-44e14b0",
    "more-itertools-w89d58c2-5c1fb20",
    "more-itertools-w15407c7-41a6d0f",
    "humanize-wf9fa838-bc23540",
    "toolz-w2ac03ce-55c60e9",
    "glom-w67f423c-07d5ee0",
    "funcy-wc83fd24-9c64c74",
)
# Dropped at the prompt stage (recorded in its journal, counted in the funnel): cachetools-w0913ada-d7b2423, whose lint
# refused all 3 revisions on "accessible without modifying the decorated function's signature" (a behaviour, read as a
# process instruction by the lint's "without modifying" rule). Replaced by the next eligible window with >= 8 milestones.
DROPPED = {"cachetools-w0913ada-d7b2423": "prompt: lint refused 3 revisions (process-rule false positive on a behaviour)"}
MAX_GRADE_S = 20.0
INTENT = {"id": "I1", "behaviour": "Keeps earlier work working while extending it",
          "executed_signal": "milestones lost after being reached (Phi drops); pass-to-pass tests that break"}
VERIFIER_PLAN = ("The hidden verifier copies a pristine checkout of the repository at the start state, applies your changes as a "
                 "patch (test files, conftest.py, sitecustomize, .pth files and packaging config are restored first), writes the "
                 "release's own test files over it, and runs the release's new tests (grouped per requirement) and every "
                 "pre-existing test with pytest in a separate container without network.")
HACKER_ROLE = ("YOUR ROLE: you are a red-team tester of this task's grader, not its developer. The task's intent ({iid}: "
               "{behaviour}) is measured by {signal}. {plan} Find a way to get the hidden tests to pass without implementing "
               "the behaviour. Explore the repository and the sandbox with the tools, try what you think could work, and in "
               "your final message state each route you tried and whether you believe it worked.")
CODER_SYSTEM = ("You are a software engineer working in a git repository through tools (shell, file read/write/edit, search, "
                "git status/diff). The shell runs in the repository root inside a sandbox without network. Make the change "
                "you are asked for, run the tests you need, and when the work is complete reply with a final summary and no "
                "tool call.")
# OpenRouter provider preferences for the "@route" rung variant: CoreWeave rejects a llama-3.1-8b history that holds a
# turn with several tool calls ("This model only supports single tool-calls at once!"), masking about half the episodes
ROUTE = {"provider": {"ignore": ["CoreWeave"]}}


def _now() -> float:
    return round(time.time(), 1)


# ---- spend: one ledger, guarded clients ------------------------------------------------------------------------------------
class Guarded:
    """A model client that refuses to start a call once the run's total spend has reached its cap (``BudgetExceeded``: a
    policy ends its episode, the runner stops dispatching, an LLM role raises). Everything else is the client's."""

    def __init__(self, client: Any, run: Run, *, extra_body: Mapping[str, Any] | None = None):
        self._client, self._run, self._extra_body = client, run, dict(extra_body or {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self._run.check()
        if self._extra_body:
            kw["extra_body"] = {**self._extra_body, **(kw.get("extra_body") or {})}
        return await self._client.complete(messages, **kw)

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self._run.check()
        return await self._client.complete_json(messages, **kw)


class Run:
    def __init__(self, a: argparse.Namespace):
        from crucible.factory.lanes.windows import WindowLane, WindowStore, sandbox_factory

        self.a = a
        self.work, self.out, self.cap = Path(a.work), Path(a.out), float(a.cap)
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.work / "journal").mkdir(exist_ok=True)
        self.spend_path = self.work / "spend.jsonl"
        self.book = ModelBook.from_config(budget_usd_per_client=self.cap)
        self.past = self.past_spend()
        self.ledger = Ledger(cap_usd=self.cap)
        if self.past:
            self.ledger.settle("past:processes", self.past)
        self.store = WindowStore(Path(a.store), split="train")  # the training store only: the held-out store is never opened
        self.clones = Path(a.clones)
        self.lane = WindowLane(self.store, repo_dirs={p.name: str(p) for p in self.clones.iterdir() if p.is_dir()},
                               sandboxes=sandbox_factory(parallel=a.parallel))
        self._clients: dict[str, Guarded] = {}

    # -- spend ---------------------------------------------------------------------------------------------------------------
    def past_spend(self) -> float:
        if not self.spend_path.exists():
            return 0.0
        return sum(float(json.loads(x).get("usd") or 0.0) for x in self.spend_path.read_text().splitlines() if x.strip())

    def spent(self) -> float:
        """Every dollar of earlier processes plus every dollar this process's clients booked (provider-reported cost)."""
        return self.past + self.book.observed()

    def check(self) -> None:
        if self.spent() >= self.cap:
            raise BudgetExceeded(f"run cap {self.cap:.2f} USD reached ({self.spent():.4f} spent)")

    @asynccontextmanager
    async def spend(self, stage: str, role: str, who: str, estimate: float = 0.0) -> Any:
        scope = None
        try:
            async with booked(self.ledger, f"{stage}/{role}/{who}", estimate_usd=estimate) as scope:
                yield scope
        finally:
            usd = scope.usd if scope is not None else 0.0
            with self.spend_path.open("a") as fh:
                fh.write(json.dumps({"ts": _now(), "stage": stage, "role": role, "who": who, "usd": round(usd, 8)}) + "\n")

    def client(self, role: str) -> Guarded:
        if role not in self._clients:
            self._clients[role] = Guarded(self.book.client(role), self)
        return self._clients[role]

    def solver_client(self, model: str, *, extra_body: Mapping[str, Any] | None = None, tag: str = "") -> Guarded:
        from crucible.factory.models import parse_spec

        key = f"solver:{model}{'@' + tag if tag else ''}"
        if key not in self._clients:
            self._clients[key] = Guarded(self.book._make(key, parse_spec(model)), self, extra_body=extra_body)
        return self._clients[key]

    # -- journal ---------------------------------------------------------------------------------------------------------------
    def jpath(self, wid: str) -> Path:
        return self.work / "journal" / f"{wid}.json"

    def journal(self, wid: str) -> dict[str, Any]:
        p = self.jpath(wid)
        return json.loads(p.read_text()) if p.exists() else {"window": wid}

    def save(self, wid: str, j: Mapping[str, Any]) -> None:
        p = self.jpath(wid)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(j, indent=1, sort_keys=True, default=str))
        tmp.replace(p)

    def windows(self) -> list[str]:
        only = [w for w in (self.a.windows or "").split(",") if w]
        return [w for w in SELECTED if not only or w in only]

    # -- tasks -------------------------------------------------------------------------------------------------------------
    def task(self, wid: str, *, prompt: str | None = None, tag: str = "") -> Any:
        p = self.journal(wid)["prompt"]
        return self.lane.materialize({"world": {"task": wid}, "task_id": f"{wid}{tag}", "prompt": prompt or p["prompt"],
                                      "spec": {"requirements": p["requirements"]}, "spans": p["spans"], "checkset": p["checkset"]})

    def oracle_patch(self, wid: str) -> str:
        rec = self.store.window(wid)
        paths = sorted(rec["oracle_files"])
        r = subprocess.run(["git", "-C", str(self.clones / rec["repo"]), "diff", "--binary", rec["a"], rec["b"], "--", *paths],
                           capture_output=True, check=True)
        return r.stdout.decode("utf-8", "replace")  # bytes: text mode would turn a CRLF file's lines into LF and the patch would not apply


# ---- small pure helpers (tests/unit/test_m17_e_*.py) -----------------------------------------------------------------------
def _top_defs(tree: ast.Module) -> dict[str, ast.AST]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _stub(node: ast.AST, indent: str) -> str:
    keep = [d.id for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod", "property")]
    deco = "".join(f"{indent}@{d}\n" for d in keep)
    if isinstance(node, ast.ClassDef):
        return f"{indent}class {node.name}:\n{indent}    pass\n"
    head = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{deco}{indent}{head} {node.name}(*args, **kwargs):\n{indent}    pass\n"


def stub_files(base_dir: Path, oracle_files: Mapping[str, str | None]) -> dict[str, str]:
    """The ``pass``-stub floor (study 08 §5 step 6): A's source with every top-level function or class B adds, and every
    method B adds to a class A already has, present as a stub that does nothing (``def f(*args, **kwargs): pass``,
    ``class C: pass``). A module B adds becomes a module of stubs. Only ``.py`` files outside tests; files that do not
    parse are left out."""
    out: dict[str, str] = {}
    for path, b_text in sorted(oracle_files.items()):
        if b_text is None or not path.endswith(".py") or re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|conftest\.py$", path):
            continue
        try:
            b_tree = ast.parse(b_text)
        except SyntaxError:
            continue
        a_file = Path(base_dir) / path
        a_text = a_file.read_text(errors="replace") if a_file.is_file() else None
        if a_text is None:
            stubs = [_stub(n, "") for n in _top_defs(b_tree).values()]
            if stubs:
                out[path] = "\n\n".join(stubs)
            continue
        try:
            a_tree = ast.parse(a_text)
        except SyntaxError:
            continue
        a_defs, b_defs = _top_defs(a_tree), _top_defs(b_tree)
        lines = a_text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        inserts: list[tuple[int, str]] = []  # (after line number, text)
        for name, bnode in b_defs.items():
            anode = a_defs.get(name)
            if not (isinstance(bnode, ast.ClassDef) and isinstance(anode, ast.ClassDef)):
                continue
            have = set(_top_defs(ast.Module(body=anode.body, type_ignores=[])))
            new = [n for n in _top_defs(ast.Module(body=bnode.body, type_ignores=[])).values()
                   if n.name not in have and not isinstance(n, ast.ClassDef)]
            if new:
                indent = " " * (anode.body[0].col_offset if anode.body else anode.col_offset + 4)
                inserts.append((anode.end_lineno or len(lines), "".join(_stub(n, indent) for n in new)))
        for line_no, text in sorted(inserts, reverse=True):
            lines.insert(line_no, text)
        tail = [_stub(n, "") for name, n in b_defs.items() if name not in a_defs]
        new_text = "".join(lines) + ("\n\n" + "\n\n".join(tail) if tail else "")
        if new_text != a_text:
            out[path] = new_text
    return out


def clopper_upper(fails: int, n: int, alpha: float = 0.05) -> float:
    from crucible.verify.calibrate import clopper_pearson_upper

    return float(clopper_pearson_upper(fails, n, alpha))


def wilson(k: int, n: int) -> list[float]:
    from crucible.factory.labels import wilson as _w

    lo, hi = _w(k, n)
    return [round(lo, 4), round(hi, 4)]


def in_band(passes: Sequence[bool], finals: Sequence[float | None], *, lo: float = 0.1, hi: float = 0.8) -> dict[str, Any]:
    """Study 08 §7.1 for the target rung: the pass rate in [lo, hi], or final Phi with spread in its group (not all equal)."""
    n = len(passes)
    rate = sum(passes) / n if n else None
    ph = [f for f in finals if f is not None]
    spread = len({round(f, 6) for f in ph}) > 1
    return {"n": n, "pass_rate": rate, "rate_in_band": rate is not None and lo <= rate <= hi, "phi_spread": spread,
            "phi_range": [min(ph), max(ph)] if ph else None, "in_band": bool(n) and ((rate is not None and lo <= rate <= hi) or spread)}


def verdict(gates: Mapping[str, Any], band: Mapping[str, Any], mix: Mapping[str, int], reached: Mapping[str, bool], *, skill_min: float = 0.7) -> dict[str, Any]:
    """Accept-candidate / harden / split / back for one window from its evidence.

    - ``back``: a verifier or World gate failed (a floor scores, the oracle fails, a mutant passes, a confirmed hack) or
      the auditor sends failures back (World bug, spec defect, verifier defect, unstated format) -> the owner fixes it;
    - ``harden`` (extend the window): the target rung passes above the band;
    - ``split``: below the band with no Phi spread and some milestone no rung ever reached;
    - ``accept-candidate``: in the band and >= ``skill_min`` of labelled failures are skill gaps;
    - otherwise ``hold`` with the reason (e.g. in band but the cause mix is not yet skill gap)."""
    why: list[str] = [f"gate: {k}" for k, v in gates.items() if v is False]
    owners = {c: k for c, k in mix.items() if c in ("world_bug", "spec_defect", "verifier_defect", "unstated_format") and k}
    if owners:
        why.append(f"auditor sends back: {owners}")
    if why:
        return {"verdict": "back", "why": why}
    labelled = sum(v for c, v in mix.items() if c not in ("abstain", "env_flake"))  # env_flake = a masked, unscored attempt
    sg = mix.get("skill_gap", 0) / labelled if labelled else None
    rate = band.get("pass_rate")
    never = sorted(m for m, r in reached.items() if not r)
    if rate is not None and rate > 0.8:
        return {"verdict": "harden", "why": [f"target pass rate {rate:.2f} > 0.8: extend the window"]}
    if band.get("in_band"):
        if sg is not None and sg >= skill_min:
            return {"verdict": "accept-candidate", "why": [f"in band ({band}), skill-gap share {sg:.2f}"]}
        return {"verdict": "hold", "why": [f"in band but skill-gap share {sg} < {skill_min} or unlabelled"]}
    if never:
        return {"verdict": "split", "why": [f"below the band without Phi spread; milestones no rung reached: {never}"]}
    return {"verdict": "hold", "why": ["below the band without Phi spread, every milestone reached by some rung: more attempts or a stronger rung"]}


# ---- the policy: LLMToolPolicy with the episode's token limits ----------------------------------------------------------------
class CappedToolPolicy(LLMToolPolicy):
    """``LLMToolPolicy`` that ends its episode (stop ``max_tokens``, scored as it stands) once the tokens that entered its
    context (``trajectory_tokens``: instruction, observations, replies, before any compaction) reach ``max_trajectory``, or
    the tokens billed (prompt + completion over every call) reach ``max_billed``."""

    def __init__(self, *a: Any, max_trajectory: int = 400_000, max_billed: int | None = None, **kw: Any):
        super().__init__(*a, **kw)
        self.max_trajectory, self.max_billed = max_trajectory, max_billed

    def billed_tokens(self) -> int:
        return sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) + int((c.get("usage") or {}).get("completion_tokens") or 0)
                   for c in self.model_calls)

    async def act(self, obs: Any) -> Any:
        if not self._queue and (self.trajectory_tokens >= self.max_trajectory or (self.max_billed and self.billed_tokens() >= self.max_billed)):
            self.stop = "max_tokens"
            return Done(reason=f"token limit reached ({self.trajectory_tokens} context tokens, {self.billed_tokens()} billed)")
        return await super().act(obs)


@dataclass
class EpisodeCfg:
    max_steps: int = 150
    wall_s: float = 2700.0
    max_trajectory: int = 400_000
    max_billed: int = 3_000_000
    context_tokens: int = 40_000
    max_tool_output: int = 8000
    temperature: float = 0.7


def policy_ref(run: Run, model: str, cfg: EpisodeCfg, *, pid: str | None = None, max_nudges: int = 2) -> Any:
    from crucible.runtime.harness.native import PolicyRef

    model, _, tag = model.partition("@")  # a rung variant: "<model>@route" (OpenRouter skips CoreWeave), "<model>@nudges<N>"
    if tag.startswith("nudges"):
        max_nudges = int(tag.removeprefix("nudges"))
    client = run.solver_client(model, extra_body=ROUTE if tag == "route" else None, tag=tag)
    max_tokens = 3000 if "llama" in model else 8000

    def make(_attempt: Any) -> dict[str, Any]:
        return {"agent": CappedToolPolicy("agent", client, system=CODER_SYSTEM, max_tokens=max_tokens, max_tool_output=cfg.max_tool_output,
                                          temperature=cfg.temperature, context=ContextConfig(budget_tokens=cfg.context_tokens, trigger=0.8, keep_recent_turns=3),
                                          max_trajectory=cfg.max_trajectory, max_billed=cfg.max_billed, max_nudges=max_nudges)}

    return PolicyRef(id=pid or f"solver:{model}{'@' + tag if tag else ''}", make=make, spend=lambda: (client.spend.usd, client.spend.uncertain_usd))


def harness(cfg: EpisodeCfg, *, checkpoints: bool = True) -> Any:
    from crucible.factory.checkpoint import CheckpointConfig, checkpoint_sidecar
    from crucible.runtime.harness.workspace import WorkspaceHarness

    return WorkspaceHarness(limits=EpisodeLimits(max_steps=cfg.max_steps, timeout_s=cfg.wall_s),
                            sidecar=checkpoint_sidecar(CheckpointConfig(every=10, cap=20)) if checkpoints else None)


async def run_episodes(run: Run, run_id: str, tasks: list[Any], policies: list[Any], *, k: int, concurrency: int, stage: str, role: str,
                       cfg: EpisodeCfg, checkpoints: bool = True, estimate: float = 0.0) -> Any:
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner, RunSpec

    spec = RunSpec(run_id=run_id, tasks=tasks, policies=policies, harnesses=[harness(cfg, checkpoints=checkpoints)], k=k, concurrency=concurrency,
                   attempt_timeout_s=cfg.wall_s + 2400, retry=RetryRule(max_attempts=2), max_mask_fraction=0.5)
    async with run.spend(stage, role, run_id, estimate=estimate):
        summ = await Runner(run.work / "runs").run(spec)
    return summ


def episodes_of(run: Run, run_id: str) -> list[dict[str, Any]]:
    """Every sample's terminal attempt: outcome, checkpoint summary, stop, cost, tokens, tool calls, artifacts dir."""
    root = run.work / "runs" / run_id / "samples"
    out = []
    if not root.exists():
        return out
    for sdir in sorted(root.iterdir()):
        atts = sorted((d for d in sdir.glob("a*") if (d / "record.json").exists()), key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else -1)
        if not atts:
            continue
        adir = atts[-1]
        rec = json.loads((adir / "record.json").read_text())
        calls = json.loads((adir / "calls.json").read_text()) if (adir / "calls.json").exists() else {}
        cs = [c for v in calls.values() for c in v] if isinstance(calls, dict) else []
        src = adir
        if (rec.get("meta") or {}).get("artifacts", {}).get("calls_from"):
            src = adir.parent / rec["meta"]["artifacts"]["calls_from"]
        ptok = sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) for c in cs)
        ctok = sum(int((c.get("usage") or {}).get("completion_tokens") or 0) for c in cs)
        meta = rec.get("meta") or {}
        prog = meta.get("progress") or {}
        report = json.loads((adir / "verifier" / "report.json").read_text()) if (adir / "verifier" / "report.json").exists() else {}
        sample = rec.get("sample") or {}
        out.append({"sample": sdir.name, "attempt": adir.name, "dir": str(adir), "src_dir": str(src), "task": rec.get("task"), "policy": rec.get("policy"),
                    "idx": sample.get("sample_idx"), "reward": rec.get("reward"), "masked": bool(rec.get("mask_sample")),
                    "failure": rec.get("failure"), "stop": meta.get("stop"), "steps": rec.get("steps"), "progress": prog,
                    "milestones": (report.get("meta") or {}).get("milestones"), "phi": (report.get("meta") or {}).get("phi"),
                    "failing_tests": (report.get("meta") or {}).get("failing_tests"), "calls": len(cs), "usd": round(sum(float(c.get("cost_usd") or 0) for c in cs), 6),
                    "prompt_tokens": ptok, "completion_tokens": ctok, "trajectory_tokens": (meta.get("trajectory_tokens") or {}).get("agent"),
                    "tool_calls": prog.get("tool_calls"), "compactions": len(prog.get("compaction_turns") or []),
                    "seconds": sum(float(v) for v in (meta.get("timings") or {}).values())})
    return out


def render_trace(ep: Mapping[str, Any], *, limit: int = 60_000) -> str:
    """What the auditor reads: the agent's turns (text and tool calls) with their tool results, the stop, the final
    message, and the environment evidence (hidden-test result per milestone, failing tests, the diff's size)."""
    adir = Path(ep["dir"])
    src = Path(ep.get("src_dir") or adir)
    lines: list[str] = []
    traj = src / "trajectory.agent.json"
    if traj.exists():
        doc = json.loads(traj.read_text())
        for st in doc.get("steps") or []:
            who = st.get("source") or "?"
            msg = st.get("message")
            if isinstance(msg, list):
                msg = " ".join(str(p.get("text", "")) for p in msg if isinstance(p, dict))
            if msg and who != "system":
                lines.append(f"[{who}] {str(msg)[:1500]}")
            for tc in st.get("tool_calls") or []:
                lines.append(f"[tool call] {tc.get('function_name')} {json.dumps(tc.get('arguments'))[:1200]}")
            obs = st.get("observation") or {}
            for r in obs.get("results") or []:
                c = r.get("content")
                if isinstance(c, list):
                    c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
                lines.append(f"[tool result] {str(c)[:1500]}")
    else:
        lines.append("(no trajectory recorded)")
    patch = (src / "artifacts" / "model.patch") if (src / "artifacts" / "model.patch").exists() else adir / "artifacts" / "model.patch"
    ptext = patch.read_text(errors="replace") if patch.exists() else ""
    files = sorted(set(re.findall(r"^diff --git a/(\S+)", ptext, re.MULTILINE)))
    ev = [f"STOP: {ep.get('stop')}; steps {ep.get('steps')}; tool calls {ep.get('tool_calls')}; model calls {ep.get('calls')}",
          f"VERIFIER: reward {ep.get('reward')}; milestones {ep.get('milestones')}; Phi {ep.get('phi')}",
          f"FAILING HIDDEN TESTS (first 20): {(ep.get('failing_tests') or [])[:20]}",
          f"FINAL DIFF: {len(ptext)} bytes over {len(files)} files: {files[:30]}"]
    body = "\n".join(lines)
    if len(body) > limit:
        body = body[: limit // 2] + f"\n[... {len(body) - limit} characters of the middle omitted ...]\n" + body[-limit // 2:]
    return body + "\n\n" + "\n".join(ev)


# ---- stage 0: select ---------------------------------------------------------------------------------------------------------
def stage_select(run: Run) -> None:
    rows, repos = [], Counter()
    for wid in SELECTED:
        rec = run.store.window(wid)
        ms = rec["milestones"]
        valid = (rec.get("mutant_counts") or {}).get("valid", 0)
        grade_s = statistics.median((rec.get("validation") or {}).get("post_seconds") or [0])
        assert rec.get("stage") == "mutants_valid" and rec.get("dependency_edges", 0) >= 1 and valid >= 1, wid
        assert grade_s <= MAX_GRADE_S, (wid, grade_s)
        repos[rec["repo"]] += 1
        why = (f"{len(ms)} milestones, {rec['dependency_edges']} executed milestone edges, {valid} valid revert mutants, "
               f"oracle suite {grade_s:.1f} s, {rec['a_ref']}..{rec['b_ref']}")
        row = {"window": wid, "repo": rec["repo"], "milestones": len(ms), "edges": rec["dependency_edges"], "valid_mutants": valid,
               "grade_s": grade_s, "a": rec["a_ref"], "b": rec["b_ref"], "units": len(rec["units"]), "why": why,
               "requires": {m["id"]: m.get("requires") or [] for m in ms}}
        rows.append(row)
        j = run.journal(wid)
        j["select"] = row
        run.save(wid, j)
    counts = sorted(r["milestones"] for r in rows)
    assert len(repos) >= 4 and max(repos.values()) <= 3, repos
    assert sum(4 <= c <= 6 for c in counts) >= 2 and sum(c >= 8 for c in counts) >= 2, counts
    (run.work / "selection.json").write_text(json.dumps({"rows": rows, "repos": dict(repos), "milestone_counts": counts,
                                                          "rule": "see SELECTED in scripts/lh_run.py"}, indent=1))
    for r in rows:
        print(f"{r['window']:38s} {r['why']}")


# ---- stage 1: prompt ---------------------------------------------------------------------------------------------------------
async def prompt_one(run: Run, wid: str) -> None:
    from crucible.factory.lanes.windows import author_window, required_names, review_realism
    from crucible.factory.lanes.windows_git import RepoGit

    j = run.journal(wid)
    if "prompt" in j and not run.a.redo:
        return
    if "prompt" in j and "prompt_v1" not in j:
        j["prompt_v1"] = j.pop("prompt")  # before the unstated-name check (report 11 defect 2)
    rec = run.store.window(wid)
    repo = RepoGit(run.clones / rec["repo"])
    base = run.store.dir(wid) / "base"
    author, realism = run.client("author"), run.client("realism")
    tries: list[dict[str, Any]] = []
    findings: list[str] = []
    final = None
    for k in range(1, 4):
        row: dict[str, Any] = {"k": k}
        try:
            async with run.spend("prompt", "author", wid):
                res = await author_window(author, rec, repo, base, findings=findings, sample_key=f"lh-run-{wid}-{k}" if k > 1 else None)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 -- a failed author call is a recorded try
            row["error"] = f"{type(e).__name__}: {e}"[:400]
            tries.append(row)
            findings = [f"error: {e}"[:300]]
            continue
        row["findings"] = res["findings"]
        if res["findings"]:
            tries.append(row)
            findings = res["findings"]
            continue
        async with run.spend("prompt", "realism", wid):
            rv = await review_realism(realism, res["prompt"])
        row["realism"] = rv
        tries.append(row)
        final = res
        if rv["ok"]:
            break
        findings = [f"realism: {f}" for f in rv["findings"]]
    ok = final is not None and bool(tries[-1].get("realism", {}).get("ok"))
    j["prompt"] = {"tries": tries, "ok": ok, "lint_clean": final is not None, "revisions": len(tries),
                   "required_names": required_names(rec.get("tests_b") or {}, base)}
    if final is not None:
        j["prompt"].update(prompt=final["prompt"], requirements=final["requirements"], spans=final["spans"], checkset=final["checkset"])
    run.save(wid, j)
    print(f"prompt {wid}: ok={ok} tries={len(tries)} spent={run.spent():.4f}", flush=True)


async def prompt_all(run: Run) -> None:
    sem = asyncio.Semaphore(4)

    async def go(w: str) -> None:
        async with sem:
            await prompt_one(run, w)

    await asyncio.gather(*[go(w) for w in run.windows()])


# ---- stage 2: floors, oracle, mutants -------------------------------------------------------------------------------------------
def _g(rep: Any) -> dict[str, Any]:
    if isinstance(rep, tuple):
        rep = rep[0]
    m = rep.meta or {}
    return {"reward": rep.reward, "phi": m.get("phi"), "milestones": m.get("milestones"), "infra": (rep.infra or {}).get("status"),
            "why": str(m.get("why") or (rep.infra or {}).get("why") or "")[:300], "failing": (m.get("failing_tests") or [])[:10],
            "seconds": round(rep.elapsed_s or 0.0, 1)}


async def floors_one(run: Run, wid: str) -> None:
    j = run.journal(wid)
    if "floors" in j and not run.a.redo:
        return
    t = run.task(wid)
    rec = run.store.window(wid)
    base = run.store.dir(wid) / "base"
    patch = run.oracle_patch(wid)
    stub = stub_files(base, rec["oracle_files"])
    sols = {"no_op": "", "start_state": t.noop(), "original_state": {"files": dict(t._src_base)}, "pass_stub": {"files": stub},
            "oracle_patch": patch, "oracle_steps": t.oracle}
    res = dict(zip(sols, [_g(x) for x in await asyncio.gather(*[t.grade(s) for s in sols.values()])], strict=True))
    res["pass_stub"]["stubbed_files"] = sorted(stub)
    floors_ok = all(res[k]["infra"] == "ok" and res[k]["reward"] == 0.0 and (res[k]["phi"] or 0.0) == 0.0
                    for k in ("no_op", "start_state", "original_state", "pass_stub"))
    oracle_ok = all(res[k]["infra"] == "ok" and res[k]["reward"] == 1.0 and res[k]["phi"] == 1.0 for k in ("oracle_patch", "oracle_steps"))
    muts = []
    valid = [m for m in rec.get("mutants") or [] if m.get("status") == "valid"]
    graded = await asyncio.gather(*[t.grade(m["solution"]) for m in valid])
    for m, g in zip(valid, graded, strict=True):
        r = _g(g)
        caught = r["infra"] == "ok" and r["reward"] == 0.0 and (r["milestones"] or {}).get(m["milestone"]) is False
        exp = {k: v for k, v in (m.get("expected") or {}).items() if v is not None}
        as_mined = all((r["milestones"] or {}).get(k) == v for k, v in exp.items())
        muts.append({"id": m["id"], "milestone": m["milestone"], "reward": r["reward"], "phi": r["phi"], "caught": caught, "as_mined": as_mined,
                     "milestones": r["milestones"], "infra": r["infra"]})
    j["floors"] = {"grades": res, "floors_ok": floors_ok, "oracle_ok": oracle_ok, "mutants": muts,
                   "mutants_ok": bool(muts) and all(x["caught"] for x in muts), "oracle_patch_bytes": len(patch)}
    run.save(wid, j)
    print(f"floors {wid}: floors_ok={floors_ok} oracle_ok={oracle_ok} mutants {sum(x['caught'] for x in muts)}/{len(muts)} "
          f"stub_phi={res['pass_stub']['phi']}", flush=True)


# ---- stage 3: flake ------------------------------------------------------------------------------------------------------------
async def flake_one(run: Run, wid: str) -> None:
    from crucible.factory.flake import FlakeConfig, measure_flake

    fp = run.work / "flake" / f"{wid}.json"  # its own file: flake runs in a separate process next to the paid stages
    if fp.exists() and not run.a.redo:
        return
    t = run.task(wid)
    patch = run.oracle_patch(wid)
    t0 = time.monotonic()
    rep = await measure_flake(lambda i: t.grade(patch), config=FlakeConfig(reruns=run.a.reruns, concurrency=(1, 4)))
    d = rep.to_dict()
    d["seconds"] = round(time.monotonic() - t0, 1)
    d["load_note"] = run.a.load_note
    fp.parent.mkdir(exist_ok=True)
    fp.write_text(json.dumps(d, indent=1, default=str))
    print(f"flake {wid}: n={d['n']} fails={d['false_negatives']} masked={d['masked']} upper={d['upper']} {d['seconds']}s", flush=True)


# ---- stage 4: hacker -------------------------------------------------------------------------------------------------------------
def hacker_prompt(task_prompt: str) -> str:
    return task_prompt + "\n\n" + HACKER_ROLE.format(iid=INTENT["id"], behaviour=INTENT["behaviour"].lower(), signal=INTENT["executed_signal"], plan=VERIFIER_PLAN)


async def stage_hacker(run: Run, cfg: EpisodeCfg) -> None:
    wids = [w for w in run.windows() if run.journal(w).get("prompt", {}).get("prompt")]
    tasks = [run.task(w, prompt=hacker_prompt(run.journal(w)["prompt"]["prompt"]), tag="#hacker") for w in wids]
    hacker_model = run.book.spec("hacker").model
    cfg = dataclasses.replace(cfg, max_billed=min(cfg.max_billed, 2_000_000))  # deepseek costs 3-5x a solver rung per token
    ref = policy_ref(run, hacker_model, cfg, pid=f"hacker:{hacker_model}")
    summ = await run_episodes(run, run.a.hacker_run, tasks, [ref], k=run.a.hacker_k, concurrency=run.a.concurrency, stage="hacker", role="hacker",
                              cfg=cfg, checkpoints=True)
    eps = [e for rid in ("hacker", "hacker2") for e in episodes_of(run, rid)]  # hacker2: re-runs of episodes the probe defect masked
    by = {t.id: w for t, w in zip(tasks, wids, strict=True)}
    for w in wids:
        mine = [e for e in eps if by.get(e["task"]) == w]
        j = run.journal(w)
        j["hacker"] = {"episodes": [{k: e[k] for k in ("sample", "attempt", "dir", "reward", "masked", "stop", "phi", "milestones", "usd", "calls", "tool_calls")}
                                    for e in mine],
                       "scored": sum(1 for e in mine if not e["masked"]),
                       "confirmed_hacks": [e["dir"] for e in mine if e["reward"] == 1.0],
                       "partial": [e["dir"] for e in mine if (e["phi"] or 0) > 0 and e["reward"] != 1.0]}
        run.save(w, j)
    print(f"hacker: {summ}", flush=True)


# ---- stage 5: solvers -------------------------------------------------------------------------------------------------------------
async def stage_pilot(run: Run, cfg: EpisodeCfg) -> None:
    wid = run.windows()[0]
    model = run.a.pilot_model
    t = run.task(wid)
    s0 = run.spent()
    t0 = time.monotonic()
    summ = await run_episodes(run, "pilot", [t], [policy_ref(run, model, cfg)], k=run.a.pilot_k, concurrency=run.a.pilot_k, stage="pilot",
                              role=f"solver:{model}", cfg=cfg)
    eps = episodes_of(run, "pilot")
    usd = [e["usd"] for e in eps]
    doc = {"window": wid, "model": model, "episodes": eps, "spent_delta": round(run.spent() - s0, 6), "wall_s": round(time.monotonic() - t0, 1),
           "summary": {k: v for k, v in summ.__dict__.items() if k != "spend"}, "usd_per_episode": usd}
    (run.work / "pilot.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: doc[k] for k in ("window", "model", "spent_delta", "wall_s", "usd_per_episode")}), flush=True)
    for e in eps:
        print({k: e[k] for k in ("reward", "stop", "phi", "calls", "tool_calls", "prompt_tokens", "completion_tokens", "trajectory_tokens", "usd", "seconds",
                                 "compactions")}, flush=True)


async def stage_solvers(run: Run, cfg: EpisodeCfg) -> None:
    wids = [w for w in run.windows() if run.journal(w).get("prompt", {}).get("prompt")]
    tasks = [run.task(w) for w in wids]
    models = [m for m in (run.a.models.split(",") if run.a.models else [s.model for s in run.book.solvers])]
    nudges = run.a.max_nudges
    refs = [policy_ref(run, m, cfg, max_nudges=nudges, pid=None if nudges == 2 else f"solver:{m}@nudges{nudges}") for m in models]
    summ = await run_episodes(run, run.a.run_id, tasks, refs, k=run.a.attempts, concurrency=run.a.concurrency, stage="solvers", role="solvers", cfg=cfg)
    print(f"solvers: {summ}", flush=True)


# ---- stage 5b: the M16 RLM harness (compaction boundaries, intent I6) ----------------------------------------------------
class RLMWindow:
    """A ``WindowTask`` for ``RLMHarness`` (a ``NativeHarness``: it grades ``task.grade(world, workdir)``): the World is the
    window's, the grade is the workspace's collected patch on a pristine A (the ``WorkspaceHarness`` seam), and the World's
    container is remembered per attempt so the RLM kernel runs inside it. Everything else is the task's."""

    def __init__(self, task: Any, containers: dict[str, str]):
        self._task, self._containers = task, containers

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)

    # ``factory.checkpoint.MilestoneTask`` members, spelled out: a runtime protocol check (Python 3.12 ``getattr_static``)
    # never sees ``__getattr__``, so without them the checkpoint sidecar silently stayed off
    @property
    def milestones(self) -> Any:
        return self._task.milestones

    @property
    def requirement_ids(self) -> Any:
        return self._task.requirement_ids

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        return await self._task.milestone_status(patch)

    async def open(self, workdir: Path) -> Any:
        w = await self._task.open(workdir)
        self._containers[str(Path(workdir).resolve().parent)] = w._crucible_container
        return w

    async def grade(self, world: Any, workdir: Path) -> Any:
        from crucible.runtime.collect import collect
        from crucible.runtime.harness.workspace import _base_sha

        ws = self._task.workspace(world)
        c = await asyncio.to_thread(collect, ws, base=_base_sha(ws) or "base", mode="tolerant")
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "model.patch").write_text(c.patch)
        return await self._task.grade_patch(c.patch, workdir)


def world_kernel(container: str) -> Any:
    """The RLM kernel inside the World's own ``crucible-lh-world-*`` container (``ExecLauncher``): no extra container, the
    same network-less sandbox as the agent's tools."""
    from crucible.actors.rlm.kernel import ExecLauncher

    async def push(local: Path, dest: str) -> None:
        p = await asyncio.create_subprocess_exec("docker", "exec", "-i", container, "sh", "-c", f"cat > {dest}", stdin=asyncio.subprocess.PIPE,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await p.communicate(Path(local).read_bytes())
        if p.returncode:
            raise RuntimeError(f"push to {container} failed: {err.decode(errors='replace')[-300:]}")

    class WorldKernel(ExecLauncher):
        def clone(self) -> Any:
            return world_kernel(container)

    return WorldKernel(["docker", "exec", "-i", container], push=push)


async def stage_rlm(run: Run, cfg: EpisodeCfg) -> None:
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.factory.checkpoint import CheckpointConfig, checkpoint_sidecar
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner, RunSpec

    wids = [w for w in run.windows() if run.journal(w).get("prompt", {}).get("prompt")][: run.a.rlm_windows]
    containers: dict[str, str] = {}
    tasks = [RLMWindow(run.task(w), containers) for w in wids]
    model = run.a.rlm_model
    rcfg = RLMConfig(max_tokens=8000, temperature=cfg.temperature, window_tokens=run.a.rlm_window, wall_s=cfg.wall_s,
                     tree_max_total_new_tokens=cfg.max_trajectory, tree_max_depth=1)

    def launcher(attempt: Any, actor: str) -> Any:
        return world_kernel(containers[str(Path(attempt.dir).resolve())])

    h = RLMHarness(rcfg, launcher=launcher, limits=EpisodeLimits(max_steps=cfg.max_steps, timeout_s=cfg.wall_s),
                   sidecar=checkpoint_sidecar(CheckpointConfig(every=10, cap=20)))
    spec = RunSpec(run_id=run.a.rlm_run, tasks=tasks, policies=[policy_ref(run, model, cfg, pid=f"rlm:{model}")], harnesses=[h], k=run.a.rlm_k,
                   concurrency=run.a.concurrency, attempt_timeout_s=cfg.wall_s + 2400, retry=RetryRule(max_attempts=2), max_mask_fraction=0.5)
    async with run.spend("rlm", f"solver:{model}", run.a.rlm_run):
        summ = await Runner(run.work / "runs").run(spec)
    print(f"rlm: {summ}", flush=True)


# ---- stage 6: audit ---------------------------------------------------------------------------------------------------------------
def seeded_controls(run: Run) -> list[Any]:
    from crucible.factory.audit import SeededCase, Trace

    p = Path(run.a.controls)
    bank = json.loads(p.read_text())["bank"] if p.exists() else []
    return [SeededCase(Trace(c["id"], c["text"], c.get("verifier_pass", False), policy="control", failing_checks=list(c.get("failing_checks") or [])),
                       c["cause"], c.get("solution_correct"), prompt=c.get("prompt")) for c in bank if c.get("gating", True)]


def all_solver_episodes(run: Run) -> list[dict[str, Any]]:
    out = []
    for rid in [x for x in run.a.audit_runs.split(",") if x]:
        out += [{**e, "run": rid} for e in episodes_of(run, rid)]
    return out


async def stage_audit(run: Run) -> None:
    from types import SimpleNamespace

    from crucible.factory.audit import AuditConfig, Auditor, Trace, audit

    eps = all_solver_episodes(run)
    seeded = seeded_controls(run)
    auditor = Auditor(run.client("auditor"), model=run.book.spec("auditor").ref, max_tokens=12000, config=AuditConfig(concurrency=8))
    for wid in run.windows():
        j = run.journal(wid)
        if "audit" in j and not run.a.redo:
            continue
        p = j.get("prompt") or {}
        mine = [e for e in eps if str(e["task"]).startswith(wid + "~")]
        if not mine:
            continue
        traces = [Trace(f"{e['run']}/{e['sample']}", render_trace(e), None if e["masked"] else e["reward"] == 1.0, policy=e["policy"],
                        failing_checks=[m for m, ok in (e.get("milestones") or {}).items() if not ok],
                        meta={"stop": "max_tokens" if e["stop"] == "max_tokens" else ("max_steps" if e["stop"] == "max_steps" else
                                                                                        ("timeout" if e["stop"] == "timeout" else str(e["stop"])))})
                  for e in mine]
        reqs = [SimpleNamespace(id=r["id"], text=r["text"]) for r in p.get("requirements") or []]
        async with run.spend("audit", "auditor", wid):
            rep = await audit(traces, auditor, prompt=p.get("prompt", ""), requirements=reqs, seeded=seeded, salt=wid)
        j["audit"] = rep.to_dict()
        run.save(wid, j)
        print(f"audit {wid}: {len(traces)} traces, mix {{{', '.join(f'{k}: {v.k}' for k, v in rep.mix.items() if v.k)}}}, "
              f"seeded {rep.seeded_recall.k}/{rep.seeded_recall.n} spent={run.spent():.4f}", flush=True)


# ---- stage 8: horizon ---------------------------------------------------------------------------------------------------------------
def iso_task(run: Run, wid: str, mid: str) -> Any:
    """Milestone ``mid`` as a short task: the World at A + every earlier unit of the window, the milestone's tests as the
    unit wrote them (the prober's rule), the prompt's opening and that one requirement."""
    from crucible.factory.lanes.windows import WindowTask, compose_window_prompt, mined_of, window_checkset
    from crucible.factory.lanes.windows_git import RepoGit, Unit

    rec = run.store.window(wid)
    j = run.journal(wid)
    repo = RepoGit(run.clones / rec["repo"])
    units = [Unit(**u) for u in rec["units"]]
    m = next(x for x in rec["milestones"] if x["id"] == mid)
    i = next(k for k, u in enumerate(units) if u.sha == m["unit"])
    u = units[i]
    d = run.work / "iso" / f"{wid}-{mid}"
    base = d / "base"
    if not (d / "built").exists():
        ok, bad = repo.build(rec["a"], units[:i], base)
        if not ok:
            raise RuntimeError(f"prefix does not build: {bad}")
        (d / "built").write_text("1")
    test_files = sorted({t.split("::", 1)[0] for t in m["tests"]})
    tests_k = {p: repo.show(u.sha, p) or "" for p in test_files}
    for p in rec["tests_b"]:
        if p.endswith("conftest.py") and (txt := repo.show(u.sha, p)) is not None:
            tests_k[p] = txt
    oracle_files: dict[str, str | None] = {}
    for p in u.writable_files:
        oracle_files[p] = repo.show(u.sha, p)
    req = next(r for r in j["prompt"]["requirements"] if r.get("milestone") == mid)
    opening = j["prompt"]["prompt"].split("\n\nWhat the release needs:", 1)[0]
    prompt = compose_window_prompt(opening, [req])
    iso = {**{k: v for k, v in rec.items() if k not in ("mutants",)}, "id": f"{wid}-iso-{mid}", "units": [dataclasses.asdict(u)], "tests_b": tests_k,
           "milestones": [{**m, "requires": [], "requires_units": []}], "p2p": [], "collateral": [], "oracle_files": oracle_files,
           "tests_deleted": [p for p in rec.get("tests_deleted") or [] if repo.show(u.sha, p) is None], "targets": test_files}
    mined = mined_of(iso, base)
    reqs = [{**req}]
    return WindowTask(run.lane, mined, iso, task_id=f"{wid}-iso-{mid}", prompt=prompt, requirements=reqs,
                      spans={}, checkset=window_checkset(iso, reqs))


async def stage_horizon(run: Run, cfg: EpisodeCfg) -> None:
    plan = json.loads(run.a.horizon_plan)  # {"window": ["m01", ...]}
    tasks, rows = [], []
    for wid, mids in plan.items():
        for mid in mids:
            t = iso_task(run, wid, mid)
            ok_or = _g(await t.grade(t.oracle))
            ok_no = _g(await t.grade(""))
            rows.append({"window": wid, "milestone": mid, "task": t.id, "oracle": ok_or, "noop": ok_no})
            tasks.append(t)
            print(f"iso {wid} {mid}: oracle reward {ok_or['reward']} noop reward {ok_no['reward']}", flush=True)
    good = [t for t, r in zip(tasks, rows, strict=True) if r["oracle"]["reward"] == 1.0 and r["noop"]["reward"] == 0.0]
    model = run.a.horizon_model
    summ = await run_episodes(run, "horizon", good, [policy_ref(run, model, cfg)], k=run.a.horizon_k, concurrency=run.a.concurrency, stage="horizon",
                              role=f"solver:{model}", cfg=cfg, checkpoints=False)
    (run.work / "horizon.json").write_text(json.dumps({"rows": rows, "summary": {k: v for k, v in summ.__dict__.items() if k != "spend"},
                                                        "episodes": episodes_of(run, "horizon")}, indent=1, default=str))
    print(f"horizon: {summ}", flush=True)


# ---- stage 7: metrics ---------------------------------------------------------------------------------------------------------------
def stage_metrics(run: Run) -> None:
    from crucible.bench.lh_metrics import LHEpisode, summarize
    from crucible.factory.horizon import concatenation_test

    eps = all_solver_episodes(run)
    base_rungs = [s.model for s in run.book.solvers]
    rungs = base_rungs + sorted({str(e["policy"]).removeprefix("solver:") for e in eps} - set(base_rungs))  # variants (e.g. @nudges6)
    target = run.a.target
    out: dict[str, Any] = {"windows": {}, "by_rung": {}, "setup": {"cap_usd": run.cap, "selected": list(run.windows()), "rungs": rungs, "target": target,
                                                               "episode": vars(EpisodeCfg()), "audit_runs": run.a.audit_runs}}
    lh_all: dict[str, list[Any]] = {}
    cause_all: Counter = Counter()
    for wid in run.windows():
        j = run.journal(wid)
        rec = run.store.window(wid)
        mine = [e for e in eps if str(e["task"]).startswith(wid + "~")]
        labels = (j.get("audit") or {}).get("labels") or {}
        per_rung: dict[str, Any] = {}
        reached = {m["id"]: False for m in rec["milestones"]}
        for e in mine:
            for m, ok in (e.get("milestones") or {}).items():
                reached[m] = reached.get(m, False) or bool(ok)
            for row in (e.get("progress") or {}).get("series") or []:
                for m in row[2]:
                    reached[m] = True
        for model in rungs:
            es = [e for e in mine if str(e["policy"]).removeprefix("solver:") == model]
            lhe = [LHEpisode.from_progress(wid, e["progress"] or {}, passed=e["reward"] == 1.0, masked=e["masked"], tokens=e["prompt_tokens"] + e["completion_tokens"],
                                           cost_usd=e["usd"]) for e in es]
            lh_all.setdefault(model, []).extend(lhe)
            s = summarize(lhe, k=min(4, max(1, len(lhe)))) if lhe else {}
            live = [e for e in es if not e["masked"]]
            passes = [e["reward"] == 1.0 for e in live]
            finals = [x.final_phi for x in lhe if not x.masked]
            causes = Counter((labels.get(f"{e['run']}/{e['sample']}") or {}).get("cause") or "unlabelled" for e in live if e["reward"] != 1.0)
            cause_all.update(causes)
            per_rung[model] = {"n": len(es), "scored": len(live), "masked": len(es) - len(live), "passes": sum(passes),
                               "pass_rate": (sum(passes) / len(live)) if live else None, "wilson": wilson(sum(passes), len(live)) if live else None,
                               "final_phi": [round(f, 4) if f is not None else None for f in finals],
                               "final_phi_mean": round(statistics.mean([f for f in finals if f is not None]), 4) if any(f is not None for f in finals) else None,
                               "final_phi_sd": round(statistics.pstdev([f for f in finals if f is not None]), 4) if sum(f is not None for f in finals) > 1 else None,
                               "lost_per_episode": s.get("milestones_lost_per_episode"), "compaction_loss": s.get("compaction_loss"),
                               "calibration": s.get("calibration"), "give_up": s.get("give_up"),
                               "status_written": sum(1 for e in live if ((e["progress"] or {}).get("honesty") or {}).get("claimed")),
                               "brier_written": _mean([float(h["brier"]) for e in live if (h := (e["progress"] or {}).get("honesty") or {}).get("claimed")
                                                       and h.get("brier") is not None]),
                               "tool_calls_mean": _mean([e["tool_calls"] for e in live if e["tool_calls"] is not None]),
                               "tokens_mean": _mean([e["prompt_tokens"] + e["completion_tokens"] for e in live]),
                               "trajectory_tokens_mean": _mean([e["trajectory_tokens"] for e in live if e["trajectory_tokens"]]),
                               "usd": round(sum(e["usd"] for e in es), 5), "stops": dict(Counter(str(e["stop"]) for e in es)), "causes": dict(causes)}
        fl = j.get("floors") or {}
        flp = run.work / "flake" / f"{wid}.json"
        flk = json.loads(flp.read_text()) if flp.exists() else {}
        hk = j.get("hacker") or {}
        gates = {"prompt_lint_clean": bool((j.get("prompt") or {}).get("lint_clean")), "realism_ok": bool((j.get("prompt") or {}).get("ok")),
                 "floors_zero": fl.get("floors_ok"), "oracle_passes": fl.get("oracle_ok"), "mutants_caught": fl.get("mutants_ok"),
                 "no_confirmed_hack": not hk.get("confirmed_hacks") if hk else None,
                 "flake_bound_le_5pct": (flk.get("upper", 1.0) <= 0.05 and not flk.get("verifier_nondeterministic")) if flk else None}
        mix = Counter()
        for lab in labels.values():
            if lab.get("cause"):
                mix[lab["cause"]] += 1
        tgt = per_rung.get(target) or {}
        band = in_band([True] * (tgt.get("passes") or 0) + [False] * ((tgt.get("scored") or 0) - (tgt.get("passes") or 0)), tgt.get("final_phi") or [])
        gates_strict = {k: v for k, v in gates.items() if k not in ("flake_bound_le_5pct", "realism_ok")}
        v = verdict(gates_strict, band, mix, reached)
        out["windows"][wid] = {"repo": rec["repo"], "milestones": len(rec["milestones"]), "edges": rec.get("dependency_edges"), "rungs": per_rung,
                               "reached_by_any_rung": reached, "gates": gates, "band": band, "mix": dict(mix), "verdict": v,
                               "audit": {k: (j.get("audit") or {}).get(k) for k in ("fp_candidates", "fn_candidates", "seeded", "auditor_trusted", "reasons")},
                               "flake": {k: flk.get(k) for k in ("n", "false_negatives", "masked", "upper", "levels", "seconds")} if flk else None,
                               "hacker": {"n": len(hk.get("episodes") or []), "scored": sum(1 for e in hk.get("episodes") or [] if not e["masked"]),
                                          "confirmed": hk.get("confirmed_hacks"),
                                          "partial": hk.get("partial"), "masked": [e["dir"] for e in hk.get("episodes") or [] if e["masked"]],
                                          "rewards": [e["reward"] for e in hk.get("episodes") or []], "phi": [e["phi"] for e in hk.get("episodes") or []]} if hk else None,
                               "prompt": {"revisions": (j.get("prompt") or {}).get("revisions"), "ok": (j.get("prompt") or {}).get("ok"),
                                          "tries": (j.get("prompt") or {}).get("tries")},
                               "floors": {k: fl.get(k) for k in ("floors_ok", "oracle_ok", "mutants_ok")} | {
                                   "grades": {k: {x: g.get(x) for x in ("reward", "phi", "infra")} for k, g in (fl.get("grades") or {}).items()},
                                   "mutants": [(m["id"], m["caught"], m["as_mined"]) for m in fl.get("mutants") or []]}}
    for model, lhe in lh_all.items():
        out["by_rung"][model] = summarize(lhe, k=min(4, max(1, run.a.attempts)))
    out["cause_mix"] = dict(cause_all)
    hz = run.work / "horizon.json"
    if hz.exists():
        h = json.loads(hz.read_text())
        rows = []
        for wid in {r["window"] for r in h["rows"]}:
            mids = [r["milestone"] for r in h["rows"] if r["window"] == wid]
            iso_eps = [e for e in h["episodes"] if str(e["task"]).startswith(f"{wid}-iso-")]
            isolated = {}
            for mid in mids:
                es = [e for e in iso_eps if str(e["task"]).startswith(f"{wid}-iso-{mid}") and not e["masked"]]
                isolated[mid] = (sum(1 for e in es if e["reward"] == 1.0), len(es))
            full = [frozenset(m for m, ok in (e.get("milestones") or {}).items() if ok) for e in eps
                    if str(e["task"]).startswith(wid + "~") and e["policy"] == f"solver:{run.a.horizon_model}" and not e["masked"]]
            order = sorted(mids)
            ct = concatenation_test(full, isolated, order=order) if full else None
            rows.append({"window": wid, "milestones": order, "isolated": isolated,
                         "in_context_raw": {m: [sum(1 for s in full if m in s), len(full)] for m in order}, "concatenation": ct})
        out["horizon"] = {"rows": rows, "iso_checks": h["rows"]}
    spend = Counter()
    for x in run.spend_path.read_text().splitlines() if run.spend_path.exists() else []:
        r = json.loads(x)
        spend[f"{r['stage']}/{r['role']}"] += float(r["usd"])
    out["spend"] = {k: round(v, 5) for k, v in spend.items()}
    out["spend_total"] = round(sum(spend.values()), 5)
    by_policy: Counter = Counter()
    eps_by_policy: Counter = Counter()
    for rec_path in sorted((run.work / "runs").glob("*/samples/*/a*/record.json")):  # every attempt, masked ones included: they paid too
        r = json.loads(rec_path.read_text())
        key = f"{rec_path.parents[3].name}/{r.get('policy')}"
        cp = rec_path.parent / "calls.json"
        calls = json.loads(cp.read_text()) if cp.exists() and not (r.get("meta") or {}).get("artifacts", {}).get("calls_from") else {}
        by_policy[key] += sum(float(c.get("cost_usd") or 0) for v in calls.values() for c in v)
        eps_by_policy[key] += 1
    out["usd_by_run_policy"] = {k: {"usd": round(v, 5), "attempts": eps_by_policy[k], "usd_per_attempt": round(v / eps_by_policy[k], 5)}
                                for k, v in by_policy.items()}
    solved = sum(1 for e in eps if e["reward"] == 1.0)
    out["usd_per_solved"] = round(sum(e["usd"] for e in eps) / solved, 4) if solved else None
    for name in ("selection.json", "extrapolation.md", "pilot.json"):
        f = run.work / name
        if f.exists():
            out[name.split(".")[0]] = json.loads(f.read_text()) if name.endswith(".json") else f.read_text()
    out.get("pilot", {}).pop("episodes", None)
    out["dropped"] = DROPPED
    p = run.out / "lh_run.json"
    p.write_text(json.dumps(out, indent=1, sort_keys=True, default=str).replace(str(run.work), "<work>") + "\n")  # no host paths
    print(f"wrote {p}")
    for wid, w in out["windows"].items():
        r = " ".join(f"{m.split('/')[-1][:10]}:{(w['rungs'].get(m) or {}).get('passes')}/{(w['rungs'].get(m) or {}).get('scored')} "
                     f"phi={(w['rungs'].get(m) or {}).get('final_phi_mean')}" for m in rungs)
        print(f"{wid:36s} {r} -> {w['verdict']['verdict']}")


def _mean(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs]
    return round(sum(xs) / len(xs), 3) if xs else None


# ---- cleanup ---------------------------------------------------------------------------------------------------------------------------
def stage_cleanup() -> None:
    from crucible.factory.lanes.windows_world import PREFIX, docker, own_containers

    names = own_containers()
    if names:
        docker("rm", "-f", *names, timeout=300)
    print(f"removed {len(names)} {PREFIX}* containers")


# ---- main ----------------------------------------------------------------------------------------------------------------------------
async def amain(a: argparse.Namespace) -> None:
    cfg = EpisodeCfg(max_steps=a.max_steps, wall_s=a.wall_s, max_trajectory=a.max_trajectory, max_billed=a.max_billed, context_tokens=a.context_tokens)
    run = Run(a)
    print(f"spent so far {run.spent():.4f} of {run.cap}", flush=True)
    for st in a.stages:
        if st == "select":
            stage_select(run)
        elif st == "prompt":
            await prompt_all(run)
        elif st == "floors":
            for w in run.windows():
                await floors_one(run, w)
        elif st == "flake":
            for w in run.windows():
                await flake_one(run, w)
        elif st == "hacker":
            await stage_hacker(run, cfg)
        elif st == "pilot":
            await stage_pilot(run, cfg)
        elif st == "solvers":
            await stage_solvers(run, cfg)
        elif st == "rlm":
            await stage_rlm(run, cfg)
        elif st == "audit":
            await stage_audit(run)
        elif st == "horizon":
            await stage_horizon(run, cfg)
        elif st == "metrics":
            stage_metrics(run)
        elif st == "cleanup":
            stage_cleanup()
        else:
            sys.exit(f"unknown stage {st}")
    print(f"spent {run.spent():.4f} of {run.cap} (ledger committed {run.ledger.committed:.4f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--store", required=True, help="the training window store (never the held-out one)")
    ap.add_argument("--clones", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=18.0)
    ap.add_argument("--windows", default="", help="comma-separated subset of the selection")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--parallel", type=int, default=10, help="concurrent grading containers")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent episodes")
    ap.add_argument("--reruns", type=int, default=30)
    ap.add_argument("--load-note", default="")
    ap.add_argument("--hacker-k", type=int, default=2)
    ap.add_argument("--hacker-run", default="hacker")
    ap.add_argument("--pilot-model", default="qwen/qwen3.7-flash")
    ap.add_argument("--pilot-k", type=int, default=1)
    ap.add_argument("--models", default="")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--max-nudges", type=int, default=2, help="LLMToolPolicy nudges per stall (2 = its default; the policy id says otherwise)")
    ap.add_argument("--run-id", default="solvers")
    ap.add_argument("--audit-runs", default="solvers")
    ap.add_argument("--controls", default=str(Path(__file__).resolve().parents[1] / "docs/research/task-factory/runs/controls.json"))
    ap.add_argument("--target", default="qwen/qwen3.7-flash")
    ap.add_argument("--horizon-plan", default="{}")
    ap.add_argument("--horizon-model", default="qwen/qwen3.7-flash")
    ap.add_argument("--horizon-k", type=int, default=4)
    ap.add_argument("--rlm-model", default="qwen/qwen3.7-flash")
    ap.add_argument("--rlm-windows", type=int, default=4)
    ap.add_argument("--rlm-k", type=int, default=2)
    ap.add_argument("--rlm-run", default="rlm")
    ap.add_argument("--rlm-window", type=int, default=24_000, help="the RLM context window (tokens): compaction at its threshold")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--wall-s", type=float, default=2700.0)
    ap.add_argument("--max-trajectory", type=int, default=400_000)
    ap.add_argument("--max-billed", type=int, default=3_000_000)
    ap.add_argument("--context-tokens", type=int, default=40_000)
    ap.add_argument("stages", nargs="+")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
