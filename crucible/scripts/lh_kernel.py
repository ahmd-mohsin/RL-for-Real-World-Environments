#!/usr/bin/env python3
"""Kernel-migration family (swarm-pipeline/06): mine, build windows, run the CPU gates, author prompts, stage the GPU gates.

    lh_kernel.py scan     --store S --clones C [--repos owner/name ...]   clone + mine migration commits -> S/commits.jsonl
    lh_kernel.py windows  --store S --clones C                              windows per commit -> S/windows/*.json, S/drops.jsonl
    lh_kernel.py gates    --store S [--ids ...] [--parallel 3]              S3/S4/S6/S8 compile gates (Docker, crucible-k-* only)
    lh_kernel.py author   --store S --cap 15                                 prompts (author family), ledger, lint
    lh_kernel.py stage-gpu --store S --out B                                 GPU bundle for the lab 5090 / Marlowe
    lh_kernel.py report   --store S --out runs/kernel_funnel.json            funnel, rungs, spend

Spend: one USD ledger (``S/spend.jsonl``) with a hard cap; no key is read here (``crucible.core.config`` resolves it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_gates as kg
from crucible.factory.lanes import kernel_mine as km
from crucible.factory.lanes import kernel_world as kw

ROLES = {"author": "qwen/qwen3.8-flash", "reviewer": "deepseek/deepseek-v4-flash", "second_verifier": "deepseek/deepseek-v4-flash",
         "hacker": "deepseek/deepseek-v4-flash", "realism": "deepseek/deepseek-v4-flash"}
STUDENT = "qwen/qwen-2.5-7b-instruct"  # Qwen2.5-Coder-7B-Instruct is not served on OpenRouter (2026-10-19 model list)
SEED_REPOS = ["ArchaeaSoftware/cudahandbook", "zjin-lcf/HeCBench", "Lai-YT/rodinia_3.1", "Celebrandil/CudaSift", "NVIDIA/cub", "facebookresearch/faiss",
              "RTKConsortium/RTK", "prcvlabs/multiview", "NVIDIA/cuda-samples"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


def windows(store: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((store / "windows").glob("*.json"))]


def tree_of(store: Path, repo: str, sha: str) -> Path:
    return store / "trees" / f"{repo}@{sha[:10]}"


# ---- scan --------------------------------------------------------------------------------------------------------------------
def cmd_scan(a: argparse.Namespace) -> None:
    store, clones = Path(a.store), Path(a.clones)
    rows = []
    for full in a.repos or SEED_REPOS:
        if full in K.SEALED_REPOS:
            print(f"skip {full}: sealed (benchmark seed repository)")
            continue
        name = full.replace("/", "_")
        d = km.clone(full, clones / name)
        cs = km.scan_repo(d, name)
        ok = [c for c in cs if c.licence and any(k in c.kinds for k in K.HARD_KINDS)]
        print(f"{full}: {len(cs)} migration commits, {len(ok)} hard-kind with a permissive licence", flush=True)
        rows += [{**c.to_json(), "full_name": full} for c in cs]
    km.write_jsonl(store / "commits.jsonl", rows)


# ---- windows -----------------------------------------------------------------------------------------------------------------
def cmd_windows(a: argparse.Namespace) -> None:
    store, clones = Path(a.store), Path(a.clones)
    rows = km.read_jsonl(store / "commits.jsonl")
    drops, n = [], 0
    for r in rows:
        if not r.get("licence"):
            drops.append({"commit": r["sha"], "repo": r["repo"], "reason": "licence not permissive or unknown"})
            continue
        if not any(k in r["kinds"] for k in K.HARD_KINDS):
            drops.append({"commit": r["sha"], "repo": r["repo"], "reason": f"deprecation only ({','.join(r['kinds'])}): the start builds"})
            continue
        if "_oversized" in r["kinds"]:
            drops.append({"commit": r["sha"], "repo": r["repo"], "reason": "oversized commit"})
            continue
        repo = clones / r["repo"]
        sub = kw.subtree_paths(repo, r["sha"], r["migrated_files"])
        ta, tb = tree_of(store, r["repo"], r["parent"]), tree_of(store, r["repo"], r["sha"])
        try:
            if not ta.exists():
                kw.export_tree(repo, r["parent"], ta, paths=sub)
            if not tb.exists():
                kw.export_tree(repo, r["sha"], tb, paths=sub)
            ws, dr = K.windows_from_commit(r, ta, tb)
        except Exception as e:  # noqa: BLE001 -- one commit's failure is a drop with its reason
            drops.append({"commit": r["sha"], "repo": r["repo"], "reason": f"export or discovery failed: {type(e).__name__}: {e}"[:300]})
            continue
        drops += [{**d, "repo": r["repo"]} for d in dr]
        for w in ws:
            w["tree_a"] = str(ta.relative_to(store))
            w["source"] = {"commit": r["sha"], "full_name": r.get("full_name") or r["repo"].replace("_", "/", 1), "licence_source": r.get("licence_source")}
            w["message"] = subject_body(clones / r["repo"], r["sha"])
            w["locate"] = K.locate_lines(ta, w)
            jdump(store / "windows" / f"{w['id']}.json", w)
            n += 1
    km.write_jsonl(store / "drops.jsonl", drops)
    print(f"{n} windows, {len(drops)} drops")


def subject_body(repo: Path, sha: str) -> str:
    return km._git(repo, "show", "-s", "--format=%B", sha).strip()[:3000]


# ---- gates -------------------------------------------------------------------------------------------------------------------
def cmd_gates(a: argparse.Namespace) -> None:
    store = Path(a.store)
    todo = [w for w in windows(store) if (not a.ids or w["id"] in a.ids) and (a.redo or "gates" not in w)]
    print(f"{len(todo)} windows to gate", flush=True)

    def one(w: dict[str, Any]) -> str:
        t0 = time.time()
        try:
            kg.compile_gates(w, store / w["tree_a"])
            w["gates"]["seconds"] = round(time.time() - t0)
            w["gates"]["at"] = now()
        except Exception as e:  # noqa: BLE001 -- recorded as an infra failure of the gate run, never a pass
            w["gates"] = {"error": f"{type(e).__name__}: {e}"[:500], "cpu_pass": False, "at": now()}
        jdump(store / "windows" / f"{w['id']}.json", w)
        g = w["gates"]
        return f"{w['id']}: rung {w.get('rung')} cpu_pass {g.get('cpu_pass')} S3 {g.get('S3_needed')} S4 {g.get('S4')} orig {g.get('S3_original')} {g.get('error', '')}"

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for line in ex.map(one, todo):
            print(line, flush=True)
    kw.cleanup()


# ---- author (paid) -------------------------------------------------------------------------------------------------------------
class Spend:
    """One USD ledger for the family run (``spend.jsonl``), a hard cap checked before every call."""

    def __init__(self, store: Path, cap: float):
        from crucible.factory.models import ModelBook

        self.path, self.cap = store / "spend.jsonl", float(cap)
        self.book = ModelBook.from_config(overrides=ROLES, budget_usd_per_client=cap)
        self.past = sum(float(json.loads(x).get("usd") or 0) for x in self.path.read_text().splitlines() if x.strip()) if self.path.exists() else 0.0

    def total(self) -> float:
        return self.past + self.book.observed()

    async def call(self, role: str, who: str, fn: Any) -> Any:
        if self.total() >= self.cap:
            raise RuntimeError(f"cap {self.cap} USD reached ({self.total():.4f})")
        before = self.book.observed()
        try:
            return await fn(self.book.client(role))
        finally:
            usd = self.book.observed() - before
            with self.path.open("a") as f:
                f.write(json.dumps({"ts": now(), "stage": "author", "role": role, "who": who, "model": self.book.spec(role).ref, "usd": round(usd, 8)}) + "\n")


def cmd_author(a: argparse.Namespace) -> None:
    store = Path(a.store)
    sp = Spend(store, a.cap)
    todo = [w for w in windows(store) if (w.get("gates") or {}).get("cpu_pass") and (a.redo or not w.get("prompt"))]
    print(f"{len(todo)} windows to author (spent so far {sp.total():.4f} USD)")

    async def go() -> None:
        for w in todo:
            w["oracle_new_names"] = K.oracle_new_names(w)
            opening, problems = "", ["not authored"]
            for attempt in range(3):
                opening = await sp.call("author", w["id"], lambda c, w=w: K.author_opening(c, w))
                prompt, _spans = K.compose_prompt(opening, w)
                problems = K.lint_prompt(prompt, w)
                if not problems and opening:
                    break
            w["opening"] = opening
            w["prompt"], w["spans"] = K.compose_prompt(opening, w)
            w["requirements"] = K.requirement_texts(w)
            _led, led_problems = K.ledger_of(w)
            w["lint"] = {"prompt": K.lint_prompt(w["prompt"], w), "ledger": led_problems, "attempts": attempt + 1}
            w["kspec"] = K.kspec(w)
            jdump(store / "windows" / f"{w['id']}.json", w)
            print(w["id"], "lint", w["lint"]["prompt"], "ledger", led_problems[:3], f"spent {sp.total():.4f}", flush=True)

    asyncio.run(go())


# ---- S5/S8 through the real grader, compile milestones (CPU) -------------------------------------------------------------------
def cmd_kitcheck(a: argparse.Namespace) -> None:
    """The oracle, the floors and each revert mutant graded by ``kernel_plan`` in the verifier image (root, candidate
    unprivileged, patch applied by the candidate uid), compile milestones only: the oracle passes every compile milestone,
    floors pass none, each revert mutant fails its milestone. Recorded as ``gates['kit_compile']``."""
    from crucible.factory.longtask import status_claim_patch

    store = Path(a.store)
    todo = [w for w in windows(store) if (w.get("gates") or {}).get("cpu_pass") and (a.redo or "kit_compile" not in w["gates"]) and (not a.ids or w["id"] in a.ids)]

    def one(w: dict[str, Any]) -> str:
        be = kw.DockerBackend(store, gpu=False)
        t = K.KernelTask(w, be)
        comp = [m["id"] for m in w["milestones"] if m.get("kind") == "compile"]
        res: dict[str, Any] = {}
        cases = {"oracle": t.oracle_patch(), "no_op": "", "claims_only": status_claim_patch(t)}
        for m in comp:  # a revert mutant: the oracle without this compile milestone's files
            back = {f: K.read_raw(store / w["tree_a"] / f) if (store / w["tree_a"] / f).is_file() else None
                    for f in next(x for x in w["milestones"] if x["id"] == m)["files"]}
            cases[f"revert-{m}"] = K.oracle_patch_of(store / w["tree_a"], {**w["oracle_files"], **back})
        for name, patch in cases.items():
            rep = asyncio.run(t.grade_patch(patch))
            res[name] = rep.meta.get("compile_milestones") or {}
            if (rep.raw or {}).get("infra", {}).get("failure_kind") not in (None, "check_infra"):
                res[f"{name}_infra"] = rep.raw.get("infra")
        ok = (bool(res["oracle"]) and all(res["oracle"].values()) and not any(res["no_op"].values()) and not any(res["claims_only"].values())
              and all(res[f"revert-{m}"].get(m) is False for m in comp))
        w["gates"]["kit_compile"] = {"ok": ok, "cases": res}
        jdump(store / "windows" / f"{w['id']}.json", w)
        return f"{w['id']}: kit_compile ok={ok}"

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for line in ex.map(one, todo):
            print(line, flush=True)


# ---- the student, CPU World (compile milestones only) ---------------------------------------------------------------------------
KERNEL_CODER = ("You are a software engineer working in a git repository through tools (shell, file read/write/edit, search, git "
                "status/diff). The shell runs in the repository root inside a sandbox without network; CUDA 12.8 (nvcc) is installed. Make "
                "the change you are asked for, build what you changed, and when the work is complete reply with a final summary and no "
                "tool call.")
SNAPSHOT = ("set -e; I=$(mktemp); cp .git/index \"$I\" 2>/dev/null || true; GIT_INDEX_FILE=\"$I\" git add -A >/dev/null 2>&1; "
            "GIT_INDEX_FILE=\"$I\" git -c core.quotepath=off diff --cached --binary base; rm -f \"$I\"")


async def episode(sp: Spend, store: Path, task: Any, arch: str, a: argparse.Namespace, tag: str) -> dict[str, Any]:
    """One student attempt in a CPU-only World (the start tree, CUDA 12.8, no GPU, no network), graded on the compile
    milestones by the kernel plan in the verifier image. Run milestones need the GPU: the reward is masked, ``phi_compile``
    is the measurement."""
    import tempfile

    from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.factory.lanes.windows_world import open_world
    from crucible.world.exec import DockerExec

    t0 = time.time()
    row: dict[str, Any] = {"task": task.task_id, "tag": tag, "model": STUDENT, "arch": arch}
    with tempfile.TemporaryDirectory(prefix="ck-ep-") as td:
        world = await open_world(image=kw.BASE_TAG, base_dir=(store / task.window["tree_a"]).resolve(), workdir=Path(td),
                                 instruction=K.instruction(task.prompt, arch), memory="4g", cpus="2", name_prefix="crucible-lh-k-world-")
        before = sp.book.observed()
        try:
            if sp.total() >= sp.cap:
                raise RuntimeError("cap reached")
            pol = LLMToolPolicy("agent", sp.book.client("student"), system=KERNEL_CODER, max_tokens=3000, max_tool_output=12000, temperature=0.7,
                                context=ContextConfig(budget_tokens=20000, trigger=0.8, keep_recent_turns=3), tool_limit=a.tool_limit, limit_warning=5)
            res = await run_episode(world, {"agent": pol}, limits=EpisodeLimits(max_steps=a.tool_limit + 2, timeout_s=a.wall_s))
            row.update(steps=res.steps, stop=res.stop_condition, calls=len(res.model_calls.get("agent") or []))
            if res.masked:  # provider refusal (402 credits, outages), a dead World: infrastructure, never a 0 (V8)
                row.update(masked=True, error=str((res.failures or [{}])[0].get("kind") or "masked"))
            out = await DockerExec(world._crucible_container).run(SNAPSHOT, cwd=str(world._crucible_ws), timeout_s=300)
            patch = out.stdout if out.ok else ""
            row["collect_ok"] = out.ok
        except Exception as e:  # noqa: BLE001 -- infrastructure or provider failure: masked, never a 0 (V8)
            row.update(masked=True, error=f"{type(e).__name__}: {e}"[:300])
            patch = ""
        finally:
            await world.close()
            usd = sp.book.observed() - before
            with sp.path.open("a") as f:
                f.write(json.dumps({"ts": now(), "stage": "target", "role": "student", "who": f"{task.task_id}/{tag}", "model": STUDENT, "usd": round(usd, 8)}) + "\n")
            row["usd"] = round(usd, 6)
    if not row.get("masked"):
        rep = await K.KernelTask(task.window, kw.DockerBackend(store, arch=arch, gpu=False)).grade_patch(patch)
        row.update(patch_bytes=len(patch), phi_compile=rep.meta.get("phi_compile"), compile_milestones=rep.meta.get("compile_milestones"),
                   reward_infra=rep.infra.get("failure_kind"))
        if row["phi_compile"] is None:
            row.update(masked=True, error="compile milestones unmeasured")
    row["seconds"] = round(time.time() - t0)
    return row


def cmd_target(a: argparse.Namespace) -> None:
    """S13/S15 at compile level for the student (``STUDENT``): ``--k`` attempts per arm (the task, and ``simplify('locate')``),
    outcomes in ``S/targets.jsonl``; verdicts with ``targeting.counterfactual`` and ``band`` in the report."""
    store = Path(a.store)
    global ROLES
    ROLES = {**ROLES, "student": STUDENT}
    sp = Spend(store, a.cap)
    ws = [w for w in windows(store) if (w.get("gates") or {}).get("cpu_pass") and w.get("prompt") and (not a.ids or w["id"] in a.ids)]
    done = {(r["task"], r["tag"]) for r in km.read_jsonl(store / "targets.jsonl")} if (store / "targets.jsonl").exists() else set()
    jobs = []
    for w in ws:
        base = K.KernelTask(w)
        for arm in a.arms:
            t = base if arm == "full" else base.simplify(arm)
            jobs += [(t, f"{arm}-{i}") for i in range(a.k) if (t.task_id, f"{arm}-{i}") not in done]
    print(f"{len(jobs)} episodes ({len(ws)} windows), spent so far {sp.total():.4f} USD", flush=True)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(t: Any, tag: str) -> None:
        async with sem:
            if sp.total() >= sp.cap - 0.5:
                return
            row = await episode(sp, store, t, a.arch, a, tag)
            with (store / "targets.jsonl").open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(row["task"], row["tag"], "phi_compile", row.get("phi_compile"), "steps", row.get("steps"), f"${row.get('usd')}", row.get("error", ""), flush=True)

    async def go() -> None:
        await asyncio.gather(*[one(t, tag) for t, tag in jobs])

    asyncio.run(go())


def target_summary(store: Path) -> dict[str, Any]:
    from crucible.factory.targeting import Outcome, band, counterfactual

    rows = km.read_jsonl(store / "targets.jsonl") if (store / "targets.jsonl").exists() else []
    ws = {w["id"]: w for w in windows(store)}
    by: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        wid = r["task"].split("~")[0]
        by.setdefault(wid, {}).setdefault(r["tag"].rsplit("-", 1)[0], []).append(r)
    out = {}
    for wid, arms in sorted(by.items()):
        def outs(rs: list[dict[str, Any]]) -> list[Outcome]:
            return [Outcome(success=bool(r.get("phi_compile") == 1.0), phi_set=float(r.get("phi_compile") or 0.0), phi=float(r.get("phi_compile") or 0.0),
                            reward=float(r.get("phi_compile") or 0.0), masked=bool(r.get("masked")), usd=float(r.get("usd") or 0)) for r in rs]
        full = outs(arms.get("full", []))
        row = {"rung": (ws.get(wid) or {}).get("rung"), "full": [r.get("phi_compile") for r in arms.get("full", [])],
               "band": band([o.phi for o in full if not o.masked])}
        if "locate" in arms:
            row["counterfactual_locate"] = counterfactual(full, outs(arms["locate"]))
        out[wid] = row
    per_rung: dict[str, list[float]] = {}
    for wid, r in out.items():
        per_rung.setdefault(r["rung"] or "?", []).extend(x for x in r["full"] if x is not None)
    return {"windows": out, "per_rung_compile_pass": {k: {"n": len(v), "pass": sum(1 for x in v if x == 1.0), "mean_phi_compile": round(sum(v) / len(v), 4) if v else None}
                                                      for k, v in sorted(per_rung.items())}}


# ---- stage the GPU gates ---------------------------------------------------------------------------------------------------------
RUNNER = r"""#!/usr/bin/env bash
# Kernel family GPU gates (swarm-pipeline/06 §6). Run on the GPU host from this bundle's directory:
#   bash run_gpu.sh [docker|apptainer]
# Shared machine rules: only crucible-k-* containers are created and removed; nothing is pruned; the runner waits for an
# idle GPU (utilization <= 10%, load < 8) before each window's perf-sensitive steps.
set -euo pipefail
MODE=${1:-docker}
ARCH=${CRUCIBLE_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
echo "arch sm_$ARCH on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python3 stage_gpu.py --arch "$ARCH" --mode "$MODE"
"""


def cmd_stage_gpu(a: argparse.Namespace) -> None:
    """A self-contained bundle: per window the start tree, the oracle and every mutant as patches, floors, the grader spec;
    the kit runtime with ``kernel_plan``; the hooks; and a runner that builds the images and originals on the GPU host and
    grades oracle, floors, mutants and 60 oracle reruns (S5, S8, S10) with the kit."""
    import shutil
    import tarfile

    from crucible.factory.longtask import status_claim_patch
    from crucible.verify.stage import bundle

    store, out = Path(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_src = Path(K.__file__).with_name("kernel_plan.py")
    bundle(out / "kit", {"crucible.factory.lanes.kernel_plan": plan_src})
    kw.write_hooks(out / "hooks")
    sel = [w for w in windows(store) if (w.get("gates") or {}).get("cpu_pass") and ((w.get("gates") or {}).get("kit_compile") or {}).get("ok")]
    index = []
    for w in sel:
        d = out / "windows" / w["id"]
        d.mkdir(parents=True, exist_ok=True)
        with tarfile.open(d / "base.tar.gz", "w:gz") as t:
            t.add(store / w["tree_a"], arcname="base")
        (d / "oracle.patch").write_text(w["oracle_patch"])
        jdump(d / "kspec.json", K.kspec(w, legacy_dir="/k/legacy"))
        jdump(d / "legacy.json", {p: {"legacy_arch": s.get("legacy_arch"), "legacy_defines": s.get("legacy_defines") or []}
                                   for p, s in w["programs"].items()})
        floors = {"no_op": "", "claims_only": status_claim_patch(K.KernelTask(w))}
        for name, patch in floors.items():
            (d / f"floor_{name}.patch").write_text(patch)
        for mu in w.get("mutants") or []:
            if mu.get("kind") == "fault" and mu.get("builds"):
                fix = {**w["oracle_files"], mu["file"]: mu["content"]}
                (d / f"mutant_{mu['id']}.patch").write_text(K.oracle_patch_of(store / w["tree_a"], fix))
        index.append({"id": w["id"], "rung": w.get("rung"), "l4_candidate": w.get("l4_candidate"), "programs": sorted(w["programs"])})
    jdump(out / "index.json", index)
    shutil.copy(Path(__file__).with_name("lh_kernel_gpu.py"), out / "stage_gpu.py")
    (out / "Dockerfile.base").write_text(kw.dockerfile(kw.BASE_FROM))
    (out / "Dockerfile.legacy").write_text(kw.dockerfile(kw.LEGACY_FROM))
    (out / "run_gpu.sh").write_text(RUNNER)
    print(f"staged {len(index)} windows in {out}")


# ---- report ------------------------------------------------------------------------------------------------------------------
def cmd_report(a: argparse.Namespace) -> None:
    store = Path(a.store)
    ws = windows(store)
    drops = km.read_jsonl(store / "drops.jsonl") if (store / "drops.jsonl").exists() else []
    commits = km.read_jsonl(store / "commits.jsonl") if (store / "commits.jsonl").exists() else []
    gated = [w for w in ws if "gates" in w]
    passed = [w for w in gated if w["gates"].get("cpu_pass") and (w["gates"].get("kit_compile") or {}).get("ok")]
    reasons: Counter[str] = Counter()
    for w in gated:
        g = w["gates"]
        if g.get("cpu_pass") and (g.get("kit_compile") or {}).get("ok"):
            continue
        if g.get("cpu_pass"):
            reasons["kit grader: oracle/floors/reverts not as required at compile level" if g.get("kit_compile") else "kit check not run"] += 1
            continue
        if g.get("error"):
            reasons["gate run error"] += 1
        elif not g.get("S3_needed"):
            reasons["S3: the start already builds and holds no graded construct"] += 1
        elif not g.get("S4"):
            reasons["S4: the oracle does not build for sm_90/sm_120 with the start's build"] += 1
        elif not g.get("S3_original"):
            reasons["S3: the original does not build with CUDA 11.8"] += 1
        elif g.get("needs") == "static_only":
            reasons["S3: the start builds; the removed construct is only in dead code (static only)"] += 1
        elif not g.get("static", {}).get("graded_kinds"):
            reasons["no graded kind (the oracle keeps a construct)"] += 1
    spend = [json.loads(x) for x in (store / "spend.jsonl").read_text().splitlines() if x.strip()] if (store / "spend.jsonl").exists() else []
    by_role: Counter[str] = Counter()
    for r in spend:
        by_role[r["role"]] += float(r.get("usd") or 0)
    rep = {"at": now(), "commits": len(commits), "repos": sorted({c["repo"] for c in commits}),
           "drops": Counter(d["reason"].split(" (")[0] for d in drops).most_common(),
           "windows": len(ws), "gated": len(gated), "cpu_pass": len(passed), "cpu_fail_reasons": reasons.most_common(),
           "rungs_cpu_pass": Counter(w.get("rung") for w in passed).most_common(), "l4_candidates": sorted(w["id"] for w in passed if w.get("l4_candidate")),
           "edges_executed": sum(1 for w in passed for m in w["milestones"] for v in (m.get("requires_evidence") or {}).values() if v == "executed"),
           "fault_mutants": {"built (GPU pending)": sum(1 for w in passed for f in w["gates"].get("fault_mutants", []) if f["builds"]),
                             "compile_killed": sum(1 for w in passed for f in w["gates"].get("fault_mutants", []) if not f["builds"])},
           "authored": sum(1 for w in passed if w.get("prompt")), "lint_clean": sum(1 for w in passed if w.get("prompt") and not (w.get("lint") or {}).get("prompt")
                                                                               and not (w.get("lint") or {}).get("ledger")),
           "spend_usd": round(sum(by_role.values()), 6), "spend_by_role": {k: round(v, 6) for k, v in by_role.items()},
           "tasks": [{"id": w["id"], "repo": w["repo"], "rung": w.get("rung"), "kinds": w.get("graded_kinds"), "milestones": len(w["milestones"]),
                      "programs": sorted(w["programs"]), "legacy_arch": {p: s.get("legacy_arch") for p, s in w["programs"].items()},
                      "l4_candidate": w.get("l4_candidate"), "authored": bool(w.get("prompt")), "gpu": "pending_gpu"} for w in passed],
           "seeds": K.SEEDS, "student": {"model": STUDENT, "mode": "CPU World, compile milestones only; run milestones pending_gpu",
                                          **target_summary(store)}}
    jdump(Path(a.out), rep)
    print(json.dumps({k: rep[k] for k in ("commits", "windows", "gated", "cpu_pass", "rungs_cpu_pass", "cpu_fail_reasons", "spend_usd")}, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--store", required=True)
    s.add_argument("--clones", required=True)
    s.add_argument("--repos", nargs="*")
    s = sub.add_parser("windows")
    s.add_argument("--store", required=True)
    s.add_argument("--clones", required=True)
    s = sub.add_parser("gates")
    s.add_argument("--store", required=True)
    s.add_argument("--ids", nargs="*")
    s.add_argument("--parallel", type=int, default=3)
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("author")
    s.add_argument("--store", required=True)
    s.add_argument("--cap", type=float, default=15.0)
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("kitcheck")
    s.add_argument("--store", required=True)
    s.add_argument("--ids", nargs="*")
    s.add_argument("--parallel", type=int, default=3)
    s.add_argument("--redo", action="store_true")
    s = sub.add_parser("target")
    s.add_argument("--store", required=True)
    s.add_argument("--cap", type=float, default=15.0)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--arms", nargs="*", default=["full", "locate"])
    s.add_argument("--ids", nargs="*")
    s.add_argument("--arch", default="90")
    s.add_argument("--concurrency", type=int, default=3)
    s.add_argument("--tool-limit", dest="tool_limit", type=int, default=40)
    s.add_argument("--wall-s", dest="wall_s", type=float, default=1800.0)
    s = sub.add_parser("stage-gpu")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("report")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "windows": cmd_windows, "gates": cmd_gates, "author": cmd_author, "kitcheck": cmd_kitcheck, "target": cmd_target, "stage-gpu": cmd_stage_gpu,
     "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
