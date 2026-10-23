"""P3 (swarm-pipeline/00 §8, §15): the RLM trace-audit loop over stored long-horizon trails, and its seeded exit criterion.

    cd <repo root>   # the OpenRouter key resolves from the nearest .env
    python crucible/scripts/lh_audit.py --out <dir> --cap 5 seeded real report

Stages:
- ``seeded``  the seeded scenario (``factory.traceaudit.SeededScenario``): a lucky pass, an unstated public name, a broken
              World tool, and a fake proposal; grading and every World command run in ``crucible-p3-*`` containers
              (``--image``). With ``--seeded-auditor`` the RLM auditor also labels the seeded traces with the control bank.
- ``real``    every stored trail under ``--runs`` (run 1 and run 2: ``<root>/runs/<run>/samples/<sample>/a*``): facts, first
              wrong step, pass audits for every episode; the RLM auditor (``--auditor``, default deepseek-v4-flash) on a
              stratified sample (``--label-sample``) with the seeded control bank in every batch, cross-family only;
              proposals and their executed acceptance (spec defects against the window record and the prompt the trace
              saw; World bugs reproduced in the window's World image on the oracle state).
- ``report``  the summary of both stages (first-wrong-step distribution, cause mix, proposals with reasons, spend).

Spend: one forge ``Ledger`` capped at ``--cap`` USD; every auditor call is refused once the total (persisted in
``<out>/spend.jsonl`` across processes) reaches it. Containers: ``crucible-p3-*`` only; ``cleanup`` removes this phase's.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from crucible.actors.llm import BudgetExceeded
from crucible.factory.models import ModelBook, booked, parse_spec
from crucible.forge.budget import Ledger

PREFIX = "crucible-p3-"
INTENT = {"id": "I1", "behaviour": "Keeps earlier work working while extending it",
          "executed_signal": "milestones lost after being reached (Phi drops); pass-to-pass tests that break"}


class Spend:
    """One USD ledger for the process plus what earlier processes spent (``spend.jsonl``); refuses calls at the cap."""

    def __init__(self, out: Path, cap: float, book: ModelBook):
        self.path, self.cap, self.book = out / "spend.jsonl", float(cap), book
        self.past = sum(float(json.loads(x).get("usd") or 0) for x in self.path.read_text().splitlines() if x.strip()) if self.path.exists() else 0.0
        self.ledger = Ledger(cap_usd=self.cap)
        if self.past:
            self.ledger.settle("past:processes", self.past)

    def spent(self) -> float:
        return self.past + self.book.observed()

    def check(self) -> None:
        if self.spent() >= self.cap:
            raise BudgetExceeded(f"cap {self.cap:.2f} USD reached ({self.spent():.4f})")

    @contextlib.asynccontextmanager
    async def scope(self, role: str, who: str) -> Any:
        scope = None
        try:
            async with booked(self.ledger, f"{role}/{who}") as scope:
                yield scope
        finally:
            usd = scope.usd if scope is not None else 0.0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps({"ts": round(time.time(), 1), "role": role, "who": who, "usd": round(usd, 8)}) + "\n")

    def by_role(self) -> dict[str, float]:
        out: Counter[str] = Counter()
        if self.path.exists():
            for x in self.path.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    out[r["role"]] += float(r.get("usd") or 0)
        return {k: round(v, 6) for k, v in out.items()}


class Guarded:
    def __init__(self, client: Any, spend: Spend):
        self._client, self._spend = client, spend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self._spend.check()
        return await self._client.complete(messages, **kw)

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self._spend.check()
        return await self._client.complete_json(messages, **kw)


def controls(path: Path) -> list[Any]:
    from crucible.factory.audit import SeededCase, Trace

    bank = json.loads(path.read_text())["bank"] if path.exists() else []
    return [SeededCase(Trace(c["id"], c["text"], c.get("verifier_pass", False), policy="control", failing_checks=list(c.get("failing_checks") or [])),
                       c["cause"], c.get("solution_correct"), prompt=c.get("prompt")) for c in bank if c.get("gating", True)]


def make_auditor(a: argparse.Namespace, spend: Spend, out: Path) -> Any:
    from crucible.factory.audit import AuditConfig
    from crucible.factory.traceaudit import P3Launcher, RLMAuditor

    spec = parse_spec(a.auditor)
    client = Guarded(spend.book._make(f"role:auditor:{spec.ref}", spec), spend)
    return RLMAuditor(client, model=spec.model, workdir=out / "auditor", launcher=P3Launcher, config=AuditConfig(concurrency=a.concurrency),
                      max_steps=a.auditor_steps)


# ---- the windows' World, for World-bug reproductions -----------------------------------------------------------------------
class WindowWorld:
    """A ``WorldPort`` over a mined window: the World image, the start tree (``base``) plus a state, run in a
    ``crucible-p3-world-*`` container (``--network none``) at ``/w``. A command's absolute workspace path (the solver's
    World mounted the workspace at its host path) is rewritten to ``/w``."""

    def __init__(self, image: str, base: Path, *, timeout_s: float = 180.0):
        self.image, self.base, self.timeout_s = image, Path(base), timeout_s

    def _tree(self, dest: Path, state: Any) -> Path:
        shutil.copytree(self.base, dest, symlinks=True)
        for p, txt in ((state or {}).get("files") or {}).items() if isinstance(state, Mapping) else []:
            f = dest / p
            if txt is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(txt)
        return dest

    async def reproduce(self, repro: Mapping[str, Any], *, state: Any, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = repro.get("arguments") or {}
        cmd = str(args.get("command") or "") if isinstance(args, Mapping) else ""
        if not cmd:
            return {"failed": False, "exit": None, "output": "the reproduction names no shell command"}
        from crucible.factory.traceaudit import fault_regex, repro_command

        ws = str(repro.get("workspace") or "")
        if ws:
            cmd = cmd.replace(ws, "/w")
        cmd = repro_command(cmd)
        if fix:
            return {"failed": True, "exit": None, "output": "World fixes for windows are applied by the World expert (not in P3)"}
        with tempfile.TemporaryDirectory(prefix="p3-ww-") as d:
            t = self._tree(Path(d) / "t", state)
            name = f"{PREFIX}world-{uuid.uuid4().hex[:10]}"
            argv = ["docker", "run", "--rm", "--network", "none", "--name", name, "--memory", "2g", "--cpus", "2", "-v", f"{t}:/w", "-w", "/w",
                    "-e", "PYTHONPATH=/w/src:/w", "-e", "HOME=/tmp", self.image, "sh", "-c", cmd]
            p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                o, _ = await asyncio.wait_for(p.communicate(), timeout=self.timeout_s)
                rc, out = p.returncode or 0, o.decode(errors="replace")
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    p.kill()
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)  # noqa: ASYNC221 - cleanup after a timeout
                rc, out = -1, "timeout"
        rx = fault_regex(str(repro.get("signature") or ""))
        same = bool(rx.search(out)) if rx is not None else str(repro.get("observed") or "")[:80] in out
        return {"failed": rc != 0 and same, "exit": rc, "output": out[-1500:], "same_signature": same, "command": cmd}

    async def s3(self, *, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": False, "why": "S3 for a fixed window World is the World expert's state (not run in P3)"}

    def content(self) -> dict[str, Any]:
        return {"image": self.image}


def _sha(text: str) -> str:
    from crucible.factory.versioning import sha

    return sha(text)


class Contexts:
    """Task contexts for the stored trails: per window, the window record of the store the run mined it into, the prompt and
    requirements of the journal whose prompt the trace saw, and the World port."""

    def __init__(self, a: argparse.Namespace):
        from crucible.factory.lanes.windows import WindowStore
        from crucible.factory.traceaudit import TaskContext

        self.TaskContext = TaskContext
        self.stores = [WindowStore(Path(s), split="train") for s in a.stores.split(",") if s]
        self.journals = [Path(j) for j in a.journals.split(",") if j]
        self.cache: dict[tuple[str, str], Any] = {}
        self.worlds: dict[str, Any] = {}

    def _journal(self, wid: str, prompt_sha: str | None) -> tuple[dict[str, Any] | None, int]:
        for i, jd in enumerate(self.journals):
            f = jd / f"{wid}.json"
            if not f.exists():
                continue
            p = (json.loads(f.read_text()).get("prompt") or {})
            if p.get("prompt") and (prompt_sha is None or _sha(p["prompt"]) == prompt_sha):
                return p, i
        return None, -1

    def get(self, wid: str, prompt_sha: str | None = None) -> Any:
        key = (wid, prompt_sha or "")
        if key in self.cache:
            return self.cache[key]
        j, i = self._journal(wid, prompt_sha)
        store = None
        for s in ([self.stores[i]] if 0 <= i < len(self.stores) else []) + self.stores:
            if (s.root / "tasks" / wid / "window.json").exists():
                store = s
                break
        if store is None:
            self.cache[key] = None
            return None
        rec = store.window(wid)
        base = store.dir(wid) / "base"
        img = (rec.get("world") or {}).get("image")
        world = WindowWorld(img, base) if img and base.is_dir() else None
        ctx = self.TaskContext(wid, world=world, rec=rec, prompt=(j or {}).get("prompt", ""), requirements=list((j or {}).get("requirements") or []),
                               oracle_files=sorted(rec.get("oracle_files") or {}), oracle_state={"files": dict(rec.get("oracle_files") or {})},
                               base_dir=base if base.is_dir() else None)
        self.cache[key] = ctx
        return ctx

    def __call__(self, p: Any) -> Any:
        wid = p.task.split("-iso-")[0]
        return self.get(wid, p.artifact.get("prompt_sha") if p.kind in ("spec_defect", "unstated_format") else None)


async def stage_seeded(a: argparse.Namespace, spend: Spend, out: Path) -> dict[str, Any]:
    from crucible.factory.traceaudit import AcceptConfig, LoopConfig, PytestRunner, run_seeded

    root = out / "seeded"
    if root.exists():
        shutil.rmtree(root)
    auditor = make_auditor(a, spend, out) if a.seeded_auditor else None
    cfg = LoopConfig(intent={"id": "I-seeded", "behaviour": "Durations parse and render"}, accept=AcceptConfig(flake_reruns=a.flake_reruns))
    rep = await run_seeded(root, runner=PytestRunner(image=a.image or None), auditor=auditor, seeded_controls=controls(Path(a.controls)) if auditor else (),
                           config=cfg, spend=(lambda who: spend.scope("auditor", f"seeded/{who}")) if auditor else None)
    print(json.dumps({"seeded": {k: {x: v[x] for x in ("expected", "got", "ok", "version")} for k, v in rep["seeded"].items()}, "ok": rep["seeded_ok"],
                      "auditor": rep["auditor"]}, indent=1, default=str), flush=True)
    return rep


def sample_ids(bundles: list[Any], n: int, *, auditor_model: str, seed: int = 7) -> set[str]:
    """A stratified sample of the failing, unmasked, judge-eligible bundles: round robin over (window, family)."""
    from crucible.factory.audit import _family
    from crucible.factory.traceaudit import deterministic_cause, facts

    fam = _family(auditor_model)
    pool: dict[tuple[str, str], list[str]] = {}
    for b in bundles:
        if b.masked or b.family == fam or deterministic_cause(facts(b)) is not None:
            continue
        pool.setdefault((b.window, b.family), []).append(b.id)
    rnd = random.Random(seed)
    for v in pool.values():
        rnd.shuffle(v)
    keys = sorted(pool)
    out: list[str] = []
    while len(out) < n and any(pool[k] for k in keys):
        for k in keys:
            if pool[k] and len(out) < n:
                out.append(pool[k].pop())
    return set(out)


async def stage_real(a: argparse.Namespace, spend: Spend, out: Path) -> dict[str, Any]:
    from crucible.factory.traceaudit import AcceptConfig, LoopConfig, discover, load_bundle, run_loop
    from crucible.factory.versioning import VersionStore

    found = discover([Path(r) for r in a.runs.split(",") if r])
    bundles = []
    for i, (run, d) in enumerate(found):
        root = Path(str(d).split("/runs/")[0]).name
        bundles.append(load_bundle(d, run=f"{root}:{run}"))
        if (i + 1) % 100 == 0:
            print(f"loaded {i + 1}/{len(found)}", flush=True)
    for b in bundles:  # the World path the solver's commands name (rewritten to /w in reproductions)
        b.meta["workspace"] = str(Path(b.dir) / "work" / "ws")
    ctxs = Contexts(a)
    auditor = make_auditor(a, spend, out) if a.label_sample > 0 else None
    ids = sample_ids(bundles, a.label_sample, auditor_model=auditor.model) if auditor else set()
    root = out / "real"
    root.mkdir(parents=True, exist_ok=True)
    for f in ("intent_candidates.jsonl", "human_queue.jsonl"):
        (root / f).unlink(missing_ok=True)
    if not a.keep_versions:  # a fresh analysis: the versions this run accepts start from the published tasks
        (root / "versions.jsonl").unlink(missing_ok=True)
    store = VersionStore(root / "versions.jsonl")
    cfg = LoopConfig(intent=INTENT, batch_size=a.batch_size, accept=AcceptConfig(flake_reruns=a.flake_reruns))
    print(f"bundles {len(bundles)}; labelling {len(ids)}; spent so far {spend.spent():.4f}", flush=True)
    alts, second = fairness_ports(a)
    rep = await run_loop(bundles, out=root, store=store, contexts=ctxs, auditor=auditor, seeded=controls(Path(a.controls)) if auditor else (),
                         config=cfg, label_ids=ids, spend=lambda who: spend.scope("auditor", f"real/{who}"), world_limit=a.world_limit,
                         alts=alts, second=second)
    return rep


def fairness_ports(a: argparse.Namespace) -> tuple[Any, Any]:
    """The acceptance ports from S9b's results (``lh_debug_target.py fairness``): the accepted alternatives and the qualified
    second verifier (it needs the tasks: ``--fairness-debug`` names the debug lane's data dir). None: the fakes."""
    if not a.fairness:
        return None, None
    from crucible.factory.alternatives_gate import LiveAlternatives, LiveSecondVerifier

    second = None
    if a.fairness_debug:
        from lh_debug_target import Tasks

        tasks = Tasks(Path(a.fairness_debug))
        second = LiveSecondVerifier(Path(a.fairness), tasks.task)
    return LiveAlternatives(Path(a.fairness)), second


def summarize(out: Path, spend: Spend) -> dict[str, Any]:
    s: dict[str, Any] = {"spend_usd": round(spend.spent(), 6), "spend_by_role": spend.by_role()}
    for stage in ("seeded", "real"):
        f = out / stage / ("loop/report.json" if stage == "seeded" else "report.json")
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        s[stage] = {"bundles": r["bundles"], "failing": r["failing"], "passed": r["passed"], "masked": r["masked"], "facts_episodes": r.get("facts_episodes"),
                    "first_wrong_step": r["first_wrong_step"], "causes": r["causes"], "auditor": {k: v for k, v in r["auditor"].items() if k != "batches"},
                    "auditor_batches": [{k: b[k] for k in ("n", "trusted")} | {"seeded_ok": sum(1 for x in b["seeded"] if x["ok"]), "seeded_n": len(b["seeded"])}
                                        for b in r["auditor"]["batches"]],
                    "pass_audit": r["pass_audit"], "proposals": r["proposals"]["by_kind_status"],
                    "proposal_reasons": Counter(f"{p['kind']}:{p['status']}:{(p['reasons'] or [''])[0][:90]}" for p in r["proposals"]["rows"]).most_common(20),
                    "versions": len(r["versions"]), "intent_candidates": [{k: c[k] for k in ("signature", "n", "families")} for c in r["intent_candidates"]]}
        if stage == "seeded":
            s[stage]["cases"] = {k: {x: v[x] for x in ("expected", "got", "ok")} for k, v in r["seeded"].items()}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def cleanup() -> None:
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    names = [x for x in r.stdout.split() if x.startswith(PREFIX)]
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)
    print(f"removed {len(names)} {PREFIX}* containers", flush=True)


async def amain(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spend = Spend(out, a.cap, ModelBook.from_config())
    print(f"spent so far {spend.spent():.4f} of {spend.cap}", flush=True)
    for st in a.stages:
        if st == "seeded":
            await stage_seeded(a, spend, out)
        elif st == "real":
            await stage_real(a, spend, out)
        elif st == "report":
            print(json.dumps(summarize(out, spend), indent=1, default=str), flush=True)
        elif st == "cleanup":
            cleanup()
        else:
            sys.exit(f"unknown stage {st}")
    print(f"spent {spend.spent():.4f} of {spend.cap}; by role {spend.by_role()}", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    scratch = "/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--runs", default=f"{scratch}/lhrun,{scratch}/lhrun2", help="run roots holding runs/<run>/samples/...")
    ap.add_argument("--stores", default=f"{scratch}/lhwork/store/train,{scratch}/lhwork/store2/train", help="window stores, in journal order")
    ap.add_argument("--journals", default=f"{scratch}/lhrun/journal,{scratch}/lhrun2/journal", help="prompt journals, in store order")
    ap.add_argument("--controls", default=str(here / "docs/research/task-factory/runs/controls.json"))
    ap.add_argument("--auditor", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--auditor-steps", type=int, default=30)
    ap.add_argument("--label-sample", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--world-limit", type=int, default=24, help="World-bug reproductions executed (each a container run)")
    ap.add_argument("--flake-reruns", type=int, default=30)
    ap.add_argument("--image", default="crucible-lh-base:py312", help="the seeded grader's and World's image ('' = local, trusted code only)")
    ap.add_argument("--seeded-auditor", action="store_true")
    ap.add_argument("--keep-versions", action="store_true", help="real: build on the versions an earlier run accepted")
    ap.add_argument("--fairness", default="", help="real: S9b's results dir (lh_debug_target.py fairness): live alternatives for acceptance")
    ap.add_argument("--fairness-debug", default="", help="real: the debug lane's data dir, for the live second verifier on those tasks")
    ap.add_argument("stages", nargs="+")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
