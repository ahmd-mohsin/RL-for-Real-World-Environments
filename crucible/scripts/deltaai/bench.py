#!/usr/bin/env python
"""A sealed benchmark as one arm on DeltaAI (doc 14 §4; ``factory.benchrun``): the fresh ``baseline`` of the untrained
student now, the ``trained`` arm later, under the very harness and limits the keep rule measured.

Commands (paths are the caller's; held-out material lives only under ``crucible/data/sealed`` and the job's mode-700
directory, never in git):

- ``verify --manifest M --debug D [--targeting T] [--relocated]``: the tasks on disk are the sealed ones (every hash),
  and, with ``--targeting``, the eval harness this script builds (``factory.envs.make_env``: ``WorkspaceHarness`` +
  ``LLMToolPolicy``, eval mode, the lane's ``ENV_BUDGET``) is the one id the keep rule's student measurements used.
- ``bundle --manifest M --debug D --out B [--images]``: the debug data for the sealed tasks only, plus (``--images``) a
  ``docker save`` archive and config side file per World image (arm64, the Mac's), for ``apptainer build``.
- ``dryrun --debug B --out O``: no model. Per task: its World opened exactly as an episode opens it, the lockdown probe
  and plain network calls run inside it (they must fail), the null patch graded 0 and the oracle, applied in the World and
  collected back, graded 1 (``longtask.null_agent_score``, ``clean_world_solvable``), all through the Apptainer shim.
- ``run --manifest M --debug B --out O --backend vllm --vllm-engine E [--purpose baseline]``: k attempts per task at the
  manifest's per-task seed (``factory.tinker.benchmark``), one directory per task under ``O/<purpose>/<task>`` with
  ``rollouts/``, ``episodes.jsonl``, ``metrics.json``, ``lh_episodes.jsonl`` (for ``benchmark.paired_report``) and
  ``done.json`` (a finished task is never re-run). Refused unless the eval harness id is ``--expect-harness``.
- ``report --manifest M --out O [--sacct S]``: ``factory.benchrun.arm_summary`` to ``O/<purpose>/summary.json``; prints the
  aggregate only (no task ids).
- ``stub-serve --port P``: a scripted stand-in for the vLLM server (no GPU, no weights: the tokenizer only) whose one
  reply runs a network probe in the World's shell, so a CPU job drives the real episode path end to end (the harness's path
  guard refuses the probe before the World sees it: the first of the two layers; ``dryrun`` measures the second, the
  World's own network, from inside an episode World).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "Qwen/Qwen3.6-35B-A3B"
NET_MARK = "NETPROBE_42"  # printed by the probe as NETPROBE_$((40+2)): the command text never contains it
NET_PROBE = ("echo NETPROBE_BEGIN; "
             "(timeout 8 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null && echo tcp_1.1.1.1_443=open || echo tcp_1.1.1.1_443=blocked); "
             "(timeout 8 bash -c 'exec 3<>/dev/tcp/140.82.112.3/443' 2>/dev/null && echo tcp_github_443=open || echo tcp_github_443=blocked); "
             "(getent hosts proxy.golang.org >/dev/null 2>&1 && echo dns=resolves || echo dns=blocked); "
             "(command -v curl >/dev/null && (curl -sS -m 8 -o /dev/null https://proxy.golang.org 2>&1 && echo https=open || echo https=blocked) "
             "|| echo https=no_curl); "
             "echo ifaces=$(awk -F: 'NR>2{gsub(/ /,\"\",$1); printf \"%s,\", $1}' /proc/net/dev); "
             "echo NETPROBE_$((40+2))")


def jload(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)


def lane_env_kw() -> dict[str, Any]:
    """The lane's env budget (``generators.debug.ENV_BUDGET``), exactly as the keep rule's ``lh_debug_target.py`` runs it."""
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import Timeouts
    from crucible.factory.generators.debug import ENV_BUDGET as B

    return {"phi_checkpoint": CheckpointConfig(every=B.checkpoint_every, cap=B.checkpoint_cap, concurrency=1), "tool_limit": B.tool_limit,
            "timeouts": Timeouts(agent_s=B.agent_s, rollout_s=B.rollout_s)}


def tasks_of(debug: Path, out: Path | None, *, grade_parallel: int = 2) -> Any:
    import lh_debug_target

    return lh_debug_target.Tasks(Path(debug), grade_parallel=grade_parallel, out=out)


def eval_harness_ids(ts: Any, ids: list[str]) -> dict[str, str]:
    """task -> the harness id of its eval env (``make_env`` in eval mode, ``training`` off, the lane's budget)."""
    from crucible.factory.envs import make_env

    kw = lane_env_kw()
    return {t: make_env(ts.task(t), checkpoint=kw["phi_checkpoint"], tool_limit=kw["tool_limit"], timeouts=kw["timeouts"]).harness.id for t in ids}


# ---- verify / bundle ---------------------------------------------------------------------------------------------------------
def cmd_verify(a: argparse.Namespace) -> None:
    from crucible.factory import benchrun
    from crucible.factory.generators.debug import ENV_BUDGET

    m = benchrun.load_manifest(a.manifest)
    v = benchrun.verify(m, debug=Path(a.debug), sealed=Path(a.sealed) if a.sealed else None, relocated=a.relocated)
    out: dict[str, Any] = {"name": v["name"], "k": v["k"], "tasks": len(v["tasks"]), "hashes_ok": v["ok"], "problems": len(v["problems"])}
    ids = sorted(m["tasks"])
    ts = tasks_of(Path(a.debug), None)
    ev = eval_harness_ids(ts, ids)
    out["eval_harness"] = sorted(set(ev.values()))
    out["offline"] = all(getattr(ts.task(t), "world_network", "none") in (None, "none") for t in ids)
    if a.targeting:
        meas = benchrun.measured_harness(Path(a.targeting), ids, tool_limit=ENV_BUDGET.tool_limit)
        hp = benchrun.harness_problems(ev, meas)
        out["measured_harness"] = {h: len(v) for h, v in meas.items()}
        out["harness_ok"] = not hp
        v["problems"] += hp
        out["measured_sampling"] = benchrun.measured_sampling(Path(a.targeting), ids, tool_limit=ENV_BUDGET.tool_limit)
        if a.write_measured:  # the record an arm carries (run --measured): what the keep rule measured with, beside the arm's own
            jsave(Path(a.write_measured), {"harness": out["measured_harness"], "sampling": out["measured_sampling"]})
    imgs = benchrun.world_images(m, debug=Path(a.debug) if a.relocated else None)
    out["images"] = len(imgs)
    for p in v["problems"]:
        print("PROBLEM", p, file=sys.stderr)
    print(json.dumps(out, indent=1))
    if v["problems"] or not out["offline"]:
        sys.exit(2)


def cmd_bundle(a: argparse.Namespace) -> None:
    import lh_debug

    from crucible.factory import benchrun

    lh_debug.ROOT = Path(a.debug)
    m = benchrun.load_manifest(a.manifest)
    clones = {name: path for name, (_full, path) in lh_debug.clones().items()}
    res = benchrun.bundle(m, debug=Path(a.debug), out=Path(a.out) / "debug", clones=clones, sealed=Path(a.sealed) if a.sealed else None)
    print(json.dumps({"tasks": len(res["tasks"]), "repos": len(res["repos"]), "out": res["out"]}))
    if a.images:
        imgs = benchrun.world_images(m)
        d = Path(a.out) / "images"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for img, row in sorted(imgs.items()):
            ins = json.loads(subprocess.run(["docker", "image", "inspect", img], capture_output=True, text=True, check=True).stdout)[0]
            if ins["Id"] != row["digest"] or ins.get("Architecture") != "arm64":
                raise SystemExit(f"{img}: image id {ins['Id'][:19]} / arch {ins.get('Architecture')} is not the window's {str(row['digest'])[:19]} / arm64")
            base = img.replace("/", "_").replace(":", "_")
            tar = d / f"{base}.tar"
            if not tar.exists():
                subprocess.run(["docker", "save", "-o", str(tar) + ".part", img], check=True)
                os.replace(str(tar) + ".part", tar)
            (d / f"{base}.config.json").write_text(json.dumps(ins.get("Config") or {}))
            rows.append({"image": img, "sif": f"{base}.sif", "digest": row["digest"], "bytes": tar.stat().st_size})
        jsave(d / "images.json", rows)
        print(json.dumps({"images": len(rows), "gb": round(sum(r["bytes"] for r in rows) / 1e9, 2)}))


# ---- dry run (no model) ----------------------------------------------------------------------------------------------------
async def dry_one(ts: Any, tid: str, out: Path) -> dict[str, Any]:
    from crucible.factory.longtask import clean_world_solvable, null_agent_score
    from crucible.world import lockdown
    from crucible.world.exec import DockerExec

    task = ts.task(tid)
    row: dict[str, Any] = {"image": task.world_image or task.window["world"]["image"]}
    wd = out / "worlds" / tid
    wd.mkdir(parents=True, exist_ok=True)
    world = await task.open(wd / "probe")
    try:
        ex = DockerExec(world._crucible_container)
        (Path(task.workspace(world)) / ".lockdown_probe.sh").write_text(lockdown.PROBE)
        r = await ex.run(f"bash .lockdown_probe.sh {os.path.expanduser('~')} 0; rm -f .lockdown_probe.sh", cwd=str(task.workspace(world)), timeout_s=120)
        rep = lockdown.parse(r.stdout)
        row["lockdown_problems"] = lockdown.verdict(rep, home=os.path.expanduser("~"))
        n = await ex.run(NET_PROBE, cwd=str(task.workspace(world)), timeout_s=120)
        row["network"] = dict(x.split("=", 1) for x in n.stdout.splitlines() if "=" in x)
    finally:
        await world.close()
    row["null"] = await null_agent_score(task)
    row["oracle"] = await clean_world_solvable(task, in_world=True, workdir=wd / "oracle")
    row["null_ok"], row["oracle_ok"] = bool(row["null"].get("ok")), bool(row["oracle"].get("ok"))
    return row


async def cmd_dryrun_async(a: argparse.Namespace) -> None:
    from crucible.factory import benchrun

    out = Path(a.out)
    m = benchrun.load_manifest(Path(a.debug) / "manifest.json")
    ts = tasks_of(Path(a.debug), None, grade_parallel=a.grade_parallel)
    ids = [t for t in (a.tasks.split(",") if a.tasks else sorted(m["tasks"])) if t]
    sem = asyncio.Semaphore(a.parallel)

    async def one(t: str) -> tuple[str, Any]:
        async with sem:
            try:
                return t, await dry_one(ts, t, out)
            except Exception as e:  # noqa: BLE001 - one task's failure is recorded
                return t, {"error": f"{type(e).__name__}: {e}"[:600]}

    rows = dict(await asyncio.gather(*[one(t) for t in ids]))
    jsave(out / "dryrun.json", rows)
    net_blocked = all(all(v != "open" and v != "resolves" for k, v in (r.get("network") or {}).items() if k != "ifaces") and r.get("network")
                      for r in rows.values())
    agg = {"tasks": len(rows), "errors": sum(1 for r in rows.values() if "error" in r), "lockdown_clean": sum(1 for r in rows.values() if r.get("lockdown_problems") == []),
           "network_blocked_all": net_blocked, "null_ok": sum(1 for r in rows.values() if r.get("null_ok")),
           "oracle_ok": sum(1 for r in rows.values() if r.get("oracle_ok"))}
    print(json.dumps(agg, indent=1))


# ---- the arm -----------------------------------------------------------------------------------------------------------------
def phase_args(a: argparse.Namespace) -> argparse.Namespace:
    """``lh_debug_target.py``'s own arguments for this arm (the sampler backend, the codec, the defaults it measured with)."""
    import lh_debug_target

    argv = ["student", "--out", str(Path(a.out) / a.purpose), "--debug", a.debug, "--backend", a.backend, "--student", a.student, "--cap", "1",
            "--audit-reserve", "0", "--slots", str(a.slots), "--grade-parallel", str(a.grade_parallel), "--student-concurrency", str(a.concurrency),
            "--preset", "tinker", "--frontier", "none"]
    if a.vllm_engine:
        argv += ["--vllm-engine", a.vllm_engine]
    if a.vllm_url:
        argv += ["--vllm-url", a.vllm_url]
    if a.max_tokens:
        argv += ["--max-tokens", str(a.max_tokens)]
    return lh_debug_target.parse_args(argv)


def measured_note(a: argparse.Namespace, max_tokens: int) -> dict[str, Any]:
    """``measured_with`` (``verify --write-measured``: how the keep rule's measurements sampled) and every way this arm
    differs from it, stated in the arm's metadata (the lead, 2026-10-28: 8192 tokens per reply on our vLLM, to match the
    trainer and the trained arm, while the gate and keep-rule measurements used 4096 on Tinker)."""
    if not a.measured:
        return {}
    meas = jload(Path(a.measured), {}) or {}
    smp = meas.get("sampling") or {}
    diff = []
    if set(smp.get("max_tokens") or {}) - {str(max_tokens)}:
        diff.append(f"max_tokens: this arm {max_tokens}, the keep-rule measurements {sorted(smp.get('max_tokens') or {})}")
    if set(smp.get("backend") or {}) - {a.backend}:
        diff.append(f"backend: this arm {a.backend}, the keep-rule measurements {sorted(smp.get('backend') or {})}")
    return {"measured_with": meas, "differs_from_measured": diff}


async def cmd_run_async(a: argparse.Namespace) -> None:
    import lh_debug_target as T

    from crucible.factory import benchrun
    from crucible.factory.tinker import benchmark

    m = benchrun.load_manifest(a.manifest)
    v = benchrun.verify(m, debug=Path(a.debug), relocated=True)
    if not v["ok"]:
        raise SystemExit("the tasks are not the sealed ones:\n  " + "\n  ".join(v["problems"]))
    ids = [t for t in (a.tasks.split(",") if a.tasks else sorted(m["tasks"])) if t]
    unknown = sorted(set(ids) - set(m["tasks"]))
    if unknown:
        raise SystemExit(f"not in the manifest: {len(unknown)} task(s)")
    ph = T.Phase(phase_args(a))
    ev = eval_harness_ids(ph.tasks, ids)
    bad = sorted(t for t, h in ev.items() if h != a.expect_harness)
    if bad:
        raise SystemExit(f"eval harness {sorted(set(ev.values()))} is not the measured {a.expect_harness} ({len(bad)} task(s))")
    k = int(a.k or m["k"])
    root = Path(a.out) / a.purpose
    jsave(root / "arm.json", {"manifest_sha256": m["manifest_sha256"], "purpose": a.purpose, "k": k, "harness": a.expect_harness, "backend": a.backend,
                              "engine": jload(Path(a.vllm_engine)) if a.vllm_engine else None, "max_tokens": ph.a.max_tokens, "env": T.env_budget(ph.a),
                              "started": time.time(), "tasks": len(ids), **measured_note(a, ph.a.max_tokens)})

    async def job(tid: str) -> dict[str, Any]:
        rd = root / tid
        if (d := jload(rd / "done.json")) and d.get("k") == k:
            return d
        seed = int(m["tasks"][tid]["seed"]) + int(a.seed_offset)
        async with ph.slots.take(min(k, a.concurrency)):
            t0 = time.time()
            print(f"[{a.purpose}] start {len(list(root.glob('*/done.json')))}/{len(ids)} done", flush=True)
            try:
                rep = await benchmark([ph.tasks.task(tid)], model=ph.row(a.student), k=k, run_dir=rd, cap_usd=1.0, backend=ph.backend(),
                                      codec=ph.codec(a.student), max_tokens=ph.a.max_tokens, enable_thinking=False, concurrency=min(k, a.concurrency),
                                      seed=seed, purpose=a.purpose, observe=False, **lane_env_kw())
            except Exception as e:  # noqa: BLE001 - one task's failure is recorded; the arm goes on
                err = {"error": f"{type(e).__name__}: {e}"[:600], "wall_s": round(time.time() - t0, 1)}
                jsave(rd / "error.json", err)
                print(f"[{a.purpose}] a task failed: {err['error'][:200]}", flush=True)
                return err
            benchrun.save_episodes(rd / "lh_episodes.jsonl", rep.pop("episodes"))
            row = {"k": k, "seed": seed, "purpose": a.purpose, "wall_s": round(time.time() - t0, 1), "settings": rep["settings"],
                   "metrics": {x: rep["metrics"].get(x) for x in ("pass@1", "episodes", "masked")}}
            jsave(rd / "done.json", row)
            print(f"[{a.purpose}] a task finished in {row['wall_s']} s", flush=True)
            return row

    await asyncio.gather(*[job(t) for t in ids])


def cmd_report(a: argparse.Namespace) -> None:
    from crucible.factory import benchrun
    from crucible.factory.tinker import policy_id

    m = benchrun.load_manifest(a.manifest)
    root = Path(a.out) / a.purpose
    ids = sorted(t for t in m["tasks"] if (root / t / "lh_episodes.jsonl").exists())
    eps = benchrun.load_episodes([root / t / "lh_episodes.jsonl" for t in ids])
    rows = [json.loads(ln) for t in ids for ln in (root / t / "episodes.jsonl").read_text().splitlines() if ln.strip()]
    gpu = benchrun.gpu_hours(Path(a.sacct).read_text()) if a.sacct else None
    repo = {t: str(r.get("repo") or "") for t, r in m["tasks"].items()}
    s = benchrun.arm_summary(eps, rows, k=int(m["k"]), policy=policy_id(MODEL, None, a.backend), category_of=lambda t: repo.get(t, ""), gpu=gpu)
    s["missing_tasks"] = len(set(m["tasks"]) - set(ids))
    jsave(root / "summary.json", s)
    print(json.dumps({"aggregate": s["aggregate"], "missing_tasks": s["missing_tasks"],
                      "per_task_pass_rates": sorted(round(r["pass_rate"], 3) for r in s["per_task"].values() if r["pass_rate"] is not None)}, indent=1))


# ---- the scripted server (CPU dry run of the episode path) ----------------------------------------------------------------
def cmd_stub_serve(a: argparse.Namespace) -> None:
    import re
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from crucible.train.bridges.tinker.codec import load_tokenizer
    from crucible.train.bridges.vllm.client import vllm_model

    tok = load_tokenizer(vllm_model(a.student))
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def reply(prompt: str) -> str:
        if NET_MARK in prompt or "Refused by the environment" in prompt:  # the probe ran in the World, or the path guard refused it first
            return "The network is unreachable from this workspace. I stop here."
        names = re.findall(r'"name":\s*"([^"]*shell)"', prompt)
        tool = names[0] if names else "shell"
        return f"<tool_call>\n<function={tool}>\n<parameter=command>\n{NET_PROBE}\n</parameter>\n</function>\n</tool_call>"

    class H(BaseHTTPRequestHandler):
        def _send(self, doc: Any) -> None:
            b = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:
            self._send({"data": [{"id": a.student}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            ids = tok.encode(reply(tok.decode(body.get("prompt") or [])), add_special_tokens=False) + [eos]
            self._send({"choices": [{"logprobs": {"tokens": [f"token_id:{i}" for i in ids], "token_logprobs": [0.0] * len(ids)}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(body.get("prompt") or []), "completion_tokens": len(ids), "prompt_tokens_details": {"cached_tokens": 0}}})

        def log_message(self, *x: Any) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--debug", required=True)
    v.add_argument("--sealed", default="")
    v.add_argument("--targeting", default="")
    v.add_argument("--relocated", action="store_true")
    v.add_argument("--write-measured", default="", help="with --targeting: write the measured harness and sampling here (run --measured)")
    b = sub.add_parser("bundle")
    b.add_argument("--manifest", required=True)
    b.add_argument("--debug", required=True)
    b.add_argument("--sealed", default="")
    b.add_argument("--out", required=True)
    b.add_argument("--images", action="store_true")
    d = sub.add_parser("dryrun")
    d.add_argument("--debug", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--tasks", default="")
    d.add_argument("--parallel", type=int, default=4)
    d.add_argument("--grade-parallel", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--debug", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--purpose", default="baseline", choices=("baseline", "trained", "control", "smoke"))
    r.add_argument("--expect-harness", required=True, help="the harness id the keep rule measured with (verify --targeting prints it)")
    r.add_argument("--measured", default="", help="verify --write-measured's file: recorded in arm.json with every difference from this arm")
    r.add_argument("--backend", default="vllm", choices=("vllm",))
    r.add_argument("--vllm-engine", default="")
    r.add_argument("--vllm-url", default="")
    r.add_argument("--student", default=MODEL)
    r.add_argument("--max-tokens", type=int, default=0, help="per reply (default: the vllm backend's, 8192)")
    r.add_argument("--k", type=int, default=0, help="attempts per task (default: the manifest's)")
    r.add_argument("--seed-offset", type=int, default=0, help="0 for the counted arm; a smoke run uses another")
    r.add_argument("--tasks", default="", help="comma list in start order (default: the manifest's)")
    r.add_argument("--concurrency", type=int, default=8, help="episodes at once per task")
    r.add_argument("--slots", type=int, default=24, help="Worlds open at once on the node")
    r.add_argument("--grade-parallel", type=int, default=2)
    p = sub.add_parser("report")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purpose", default="baseline")
    p.add_argument("--backend", default="vllm")
    p.add_argument("--sacct", default="")
    s = sub.add_parser("stub-serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--student", default=MODEL)
    a = ap.parse_args()
    if a.cmd == "verify":
        cmd_verify(a)
    elif a.cmd == "bundle":
        cmd_bundle(a)
    elif a.cmd == "dryrun":
        asyncio.run(cmd_dryrun_async(a))
    elif a.cmd == "run":
        asyncio.run(cmd_run_async(a))
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "stub-serve":
        cmd_stub_serve(a)


if __name__ == "__main__":
    main()
