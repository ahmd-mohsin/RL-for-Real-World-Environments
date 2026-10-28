#!/usr/bin/env python
"""Tinker RL and benchmarking on Crucible tasks (swarm-pipeline/09).

The key: ``TINKER_API_KEY`` in the repo-root ``.env`` (gitignored). Source it into this process's environment and never
print it::

    set -a; . ../.env; set +a; .venv/bin/python scripts/tinker_rl.py <command> ...

Commands (each writes JSON under ``--out``; every paid command takes ``--cap`` USD and stops there):

- ``train --rows ROWS.jsonl --prompts DIR --targeting T.json --model M``: RL post-training on the task rows
  (``factory.envs.read_task_rows``; held-out rows are refused) with ``--steps --group --groups-per-step --lr
  --loss {importance_sampling,ppo,cispo} --eta --max-staleness --async-groups``; resumable: rerun the same command.
- ``eval --rows ROWS.jsonl --prompts DIR --model M [--checkpoint tinker://...] --k 8``: the benchmark pass
  (``factory.tinker.benchmark``: lh_metrics, pass@k, pass^k, task-clustered CIs, USD per passing episode).
- ``compare --rows ROWS.jsonl --checkpoint tinker://... --k 8 [--baseline B.json --selection S.json]``: the control
  (the base model) and the trained adapter at identical settings and seed, the paired trained - control pass@1
  difference with its task-clustered CI, and the protocol check (``factory.tinker.check_protocol``). Use ``eval
  --purpose selection --seed 0`` to choose tasks and ``eval --purpose baseline --seed 1`` for the fresh pre-score.
- ``smoke-rl``: the toy window (three milestones) in a Docker World, ``--model`` (default the target Qwen/Qwen3.6-35B-A3B), group 4,
  2 optimizer steps, sampler refresh, then the final adapter downloaded.
- ``smoke-tools --model M``: one sample-only toy episode (tool calls, $/episode, latency per turn); ``--train-step``
  adds one ``forward_backward`` + ``optim_step`` on a 2-episode group.
- ``probe-concurrency --model M --n 16``: n concurrent tiny sample calls (latency, errors, the service's behaviour
  under a burst).
- ``openai-check --checkpoint tinker://...``: one chat completion through Tinker's OpenAI-compatible endpoint (beta).
- ``download --path tinker://.../sampler_weights/...``: the LoRA adapter.
- ``eval`` / ``compare`` with ``--backend vllm`` (DeltaAI, doc 14): the student from our own vLLM server (``--vllm-url``,
  ``--vllm-engine`` a ``ServeSpec.record()`` JSON, ``--vllm-context``); ``--checkpoint`` is then a PEFT LoRA directory on
  the node, loaded onto the same base. No Tinker key is needed and both arms run on one engine build.
- ``power --arms LABEL=DIR ... --n 10 --k 8 --json OUT``: offline, no key, $0. The A/A noise of existing arms' attempts,
  the MDE of an n x k paired comparison and the tasks needed for MDE <= each ``--target`` (``bench.power.study``).

The toy window is built from the unit-test fixtures (``tests/unit/test_m17_w_*``: a three-commit toy repository), with a
World image that has Python, git and pytest (``--image``, default a refreshed window image); its containers are named
``crucible-lh-world-tinker-*`` / ``crucible-m17-tinker-*`` and removed by name at the end (nothing else is touched).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "crucible-lh-w:parse-6c176c2fa980"
WORLD_PREFIX = "crucible-lh-world-tinker-"
GRADE_PREFIX = "crucible-m17-tinker-"


ENV_KEY = "TINKER_API_KEY"


def load_env_key(path: Path) -> bool:
    """Put ``TINKER_API_KEY`` from a dotenv file into this process's environment (the same effect as ``set -a; . .env``,
    for the one variable): the value is never printed, logged or returned."""
    if os.environ.get(ENV_KEY) or not path.is_file():
        return bool(os.environ.get(ENV_KEY))
    for line in path.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep and k.strip() == ENV_KEY:
            os.environ[ENV_KEY] = v.strip().strip("'\"")
            return True
    return False


def sampler_kw(args) -> dict:
    """``model`` and ``backend`` for ``benchmark`` / ``compare``: the Tinker id (backend default), or on ``--backend vllm``
    our server's model row and a ``VLLMBackend`` carrying the engine record."""
    if getattr(args, "backend", "tinker") != "vllm":
        return {"model": args.model}
    from crucible.train.bridges.vllm.client import VLLMBackend, vllm_model

    eng = json.loads(Path(args.vllm_engine).read_text()) if args.vllm_engine else None
    return {"model": vllm_model(args.model, context=args.vllm_context), "backend": VLLMBackend(args.vllm_url or None, engine=eng)}


def need_key(args=None) -> None:
    if args is not None and getattr(args, "backend", "tinker") == "vllm":
        return  # our own server: no Tinker key
    if not os.environ.get(ENV_KEY):
        sys.exit("TINKER_API_KEY is not in the environment: source the repo .env (set -a; . ../.env; set +a) or pass --env-file")


def backend_args(p) -> None:
    p.add_argument("--backend", choices=("tinker", "vllm"), default="tinker", help="the student's sampler: Tinker, or our own vLLM server (doc 14)")
    p.add_argument("--vllm-url", default="", help="default CRUCIBLE_VLLM_URL, else http://127.0.0.1:8000")
    p.add_argument("--vllm-context", type=int, default=262_144)
    p.add_argument("--vllm-engine", default="", help="a ServeSpec.record() JSON: the engine stamped into each arm's settings")


def cleanup_containers() -> list[str]:
    """Remove only this script's containers, by name prefix."""
    gone = []
    for prefix in (WORLD_PREFIX, GRADE_PREFIX):
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
        for name in r.stdout.split():
            if name.startswith(prefix):
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
                gone.append(name)
    return gone


def toy_task(tmp: Path, image: str):
    """The toy window of the P6 tests, with its World and its grading in Docker containers of ``image``."""
    sys.path.insert(0, str(ROOT / "tests" / "unit"))
    from test_m17_w_miner import SB, built_record  # the toy repository and its record (test fixtures)
    from test_m17_w_task import REQS

    from crucible.factory.lanes.windows import WindowLane, WindowStore, mined_of, probe_dependencies, revert_mutants, sandbox_factory, window_checkset

    tmp.mkdir(parents=True, exist_ok=True)
    rec, repo, base = built_record(tmp)
    asyncio.run(probe_dependencies(rec, repo, SB()))
    rec.update(asyncio.run(revert_mutants(rec, repo, base, SB())))
    digest = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, check=True).stdout.strip()
    rec["world"] = {"image": image, "digest": digest}
    store = WindowStore(tmp / "store")
    d = store.dir(rec["id"])
    d.mkdir(parents=True)
    subprocess.run(["cp", "-R", str(base), str(d / "base")], check=True)
    store.save_window(rec)
    store.save(mined_of(rec, d / "base"))
    lane = WindowLane(store, repo_dirs={"toyrepo": str(repo.dir)}, sandboxes=sandbox_factory(parallel=4, name_prefix=GRADE_PREFIX))
    lane.world_prefix = WORLD_PREFIX
    prompt = ("Release request. The repository is a small Python package `pkg`. Implement these in `pkg/` and commit your work:\n"
              + "\n".join(f"- [{r['id']}] {r['text']}" for r in REQS)
              + "\nThe functions are `pkg.core.alpha`, `pkg.beta.beta` and `pkg.gamma.gamma`.")
    task = lane.materialize({"world": {"task": rec["id"]}, "prompt": prompt, "spec": {"requirements": REQS}, "spans": {}, "checkset": window_checkset(rec, REQS)})
    return task


def _out(args) -> Path:
    p = Path(args.out).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def write(path: Path, doc) -> None:
    """Write a report; the key's value, should anything have echoed it, is replaced (a key-name scrubber would also blank
    fields such as ``tokens_per_solved``)."""
    text = json.dumps(doc, indent=1, sort_keys=True, default=str)
    key = os.environ.get(ENV_KEY)
    if key and len(key) >= 8:
        text = text.replace(key, "<secret>")
    path.write_text(text + "\n")
    print(text[:4000])


def _envs_kw(args):
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import Timeouts

    return {"phi_checkpoint": CheckpointConfig(every=args.ckpt_every, cap=20, concurrency=1), "tool_limit": args.tool_limit,
            "timeouts": Timeouts(agent_s=args.agent_s, rollout_s=args.rollout_s)}


def _observe(args):
    """The observability config (``crucible.observe``): offline W&B runs and local Weave spans, or False."""
    if getattr(args, "no_observe", False):
        return False
    from crucible.observe import ObserveConfig

    return ObserveConfig.from_env()


def _tasks_from_rows(args):
    from crucible.factory.envs import read_task_rows, resolve_ref

    return [resolve_ref(r["args"]["ref"], prompts=Path(args.prompts) if args.prompts else None) for r in read_task_rows(Path(args.rows))]


def cmd_train(args) -> None:
    from crucible.factory.tinker import train

    need_key()
    tasks = _tasks_from_rows(args) if args.rows else [toy_task(Path(tempfile.mkdtemp(prefix="tinker-toy-")), args.image)]
    targeting = json.loads(Path(args.targeting).read_text()) if args.targeting else None
    cfg = train_config(args)
    try:
        out = asyncio.run(train(tasks, config=cfg, run_dir=_out(args), targeting=targeting, observe=_observe(args), **_envs_kw(args)))
    finally:
        cleanup_containers()
    write(_out(args) / "train_report.json", {k: v for k, v in out.items() if k != "steps_log"} | {"steps_log": out["steps_log"]})


def cmd_eval(args) -> None:
    from crucible.factory.tinker import benchmark

    need_key(args)
    tasks = _tasks_from_rows(args) if args.rows else [toy_task(Path(tempfile.mkdtemp(prefix="tinker-toy-")), args.image)]
    try:
        out = asyncio.run(benchmark(tasks, **sampler_kw(args), k=args.k, run_dir=_out(args), cap_usd=args.cap, checkpoint=args.checkpoint,
                                    max_tokens=args.max_tokens, enable_thinking=args.thinking, concurrency=args.concurrency, seed=args.seed,
                                    purpose=args.purpose, observe=_observe(args), **_envs_kw(args)))
        out.pop("episodes", None)
    finally:
        cleanup_containers()
    write(_out(args) / "eval_report.json", out)


def cmd_compare(args) -> None:
    from crucible.factory.tinker import compare

    need_key(args)
    tasks = _tasks_from_rows(args) if args.rows else [toy_task(Path(tempfile.mkdtemp(prefix="tinker-toy-")), args.image)]

    def load(p):
        return json.loads(Path(p).read_text()) if p else None

    try:
        out = asyncio.run(compare(tasks, **sampler_kw(args), checkpoint=args.checkpoint, run_dir=_out(args), cap_usd=args.cap, k=args.k, seed=args.seed,
                                  baseline=load(args.baseline), selection=load(args.selection), max_tokens=args.max_tokens, enable_thinking=args.thinking,
                                  concurrency=args.concurrency, observe=_observe(args),
                                  arm_dirs={a: Path(p).parent for a, p in (("baseline", args.baseline), ("selection", args.selection)) if p},
                                  **_envs_kw(args)))
    finally:
        cleanup_containers()
    write(_out(args) / "compare_report.json", out)


def arm_rows(path: Path) -> list[dict]:
    """An arm's episode rows: ``<path>/episodes.jsonl`` (a ``benchmark`` arm) or every ``<path>/*/episodes.jsonl`` (an S15
    arm: one directory per task)."""
    files = [path / "episodes.jsonl"] if (path / "episodes.jsonl").exists() else sorted(path.glob("*/episodes.jsonl"))
    return [json.loads(x) for f in files for x in f.read_text().splitlines() if x.strip()]


def cmd_power(args) -> None:
    """The offline power study ($0, no key): A/A noise of existing arms, the MDE of an ``--n`` x ``--k`` comparison and the
    tasks needed for MDE <= each ``--target`` (``bench.power.study``)."""
    from crucible.bench.power import study

    sets = {}
    for spec in args.arms:
        label, _, p = spec.partition("=")
        sets[label if p else Path(spec).name] = arm_rows(Path(p or spec))
    out = study(sets, n=args.n, k=args.k, targets=args.target)
    out["inputs"] = {lab: {"path": (s.partition("=")[2] or s), "episodes": len(sets[lab])} for lab, s in zip(sets, args.arms, strict=True)}
    write(Path(args.json), out)
    for lab, per in out["sets"].items():
        for m, blk in per.items():
            nz = blk["noise"]
            g = blk["gates"][f"target={args.target[0]:.2f}|bounded|tau=0.0"]
            print(f"{lab:10} {m:6} tasks={nz['n_tasks']:3} k={nz['k']} mu_c={nz['control_mean']:.3f} sigma2={nz['sigma2']:.4f} "
                  f"AA diff {nz['aa']['diff']:+.3f} (se {nz['aa']['se']:.3f}); {args.n}x{args.k}: MDE {g['mde']:.3f} (A/A only {g['mde_aa']:.3f}) {g['verdict']}")


def _turn_stats(run_dir: Path) -> dict:
    """Tool-call status from the episodes' World traces: calls executed, calls with unparseable arguments, failed calls."""
    from crucible.core.schema.rollout import RolloutRecord

    calls = unparse = failed = turns = 0
    names: dict[str, int] = {}
    for ro_path in run_dir.rglob("rollout.json"):
        ro = RolloutRecord.model_validate_json(ro_path.read_text())
        turns += sum(1 for t in ro.turns if t.completion_token_ids)
    for tr in run_dir.rglob("trace.json"):
        try:
            events = json.loads(tr.read_text()).get("events") or []
        except (OSError, ValueError, AttributeError):
            continue
        for e in events:
            d = e.get("data") or {}
            if e.get("kind") == "action" and e.get("type") == "tool_call":
                calls += 1
                names[d.get("function", "")] = names.get(d.get("function", ""), 0) + 1
                unparse += "__unparseable__" in (d.get("arguments") or {})
            elif e.get("kind") == "observation" and e.get("type") == "tool_result" and not d.get("ok", True):
                failed += 1
    return {"model_turns": turns, "tool_calls": calls, "unparseable_calls": unparse, "failed_calls": failed, "tools": names}


def cmd_smoke_tools(args) -> None:
    from crucible.factory.tinker import benchmark
    from crucible.train.bridges.tinker.trainer import TrainConfig

    need_key()
    out_dir = _out(args)
    task = toy_task(Path(tempfile.mkdtemp(prefix="tinker-toy-")), args.image)
    t0 = time.monotonic()
    rep: dict = {"model": args.model}
    try:
        ev = asyncio.run(benchmark([task], model=args.model, k=1, run_dir=out_dir / "episode", cap_usd=args.cap, max_tokens=args.max_tokens,
                                   enable_thinking=args.thinking, concurrency=1, purpose="smoke", **_envs_kw(args)))
        ev.pop("episodes", None)
        rep.update(episode=ev, turns=_turn_stats(out_dir / "episode"), wall_s=round(time.monotonic() - t0, 1))
        log = [json.loads(x) for x in (out_dir / "episode" / "spend.jsonl").read_text().splitlines() if x.strip()]
        lat = [r["latency_s"] for r in log if r["kind"] == "sample"]
        rep["latency_per_turn_s"] = {"mean": sum(lat) / len(lat), "max": max(lat), "n": len(lat)} if lat else None
        rep["sample_tokens_per_s"] = sum(r["sample"] for r in log if r["kind"] == "sample") / sum(lat) if lat and sum(lat) else None
        rep["cached_prefill_share"] = (sum(r["prefill_cached"] for r in log) / max(1, sum(r["prefill"] + r["prefill_cached"] for r in log)))
        if args.train_step:
            from crucible.factory.tinker import train

            cfg = TrainConfig(model=args.model, group_size=2, steps=1, cap_usd=max(0.0, args.cap - ev["spend"]["usd"]), require_band=False,
                              drop_zero_spread=False, max_tokens=args.max_tokens, enable_thinking=args.thinking, max_groups_per_step=2)
            tr = asyncio.run(train([task], config=cfg, run_dir=out_dir / "train", **_envs_kw(args)))
            rep["train_step"] = {k: tr[k] for k in ("stop", "steps_done", "sampler_path", "state_path", "drops", "spend", "steps_log")}
    finally:
        rep_c = cleanup_containers()
    rep["containers_removed"] = rep_c
    write(out_dir / "smoke_tools.json", rep)


def cmd_smoke_rl(args) -> None:
    from crucible.factory.tinker import download_adapter, train
    from crucible.train.bridges.tinker.trainer import TrainConfig

    need_key()
    out_dir = _out(args)
    task = toy_task(Path(tempfile.mkdtemp(prefix="tinker-toy-")), args.image)
    cfg = TrainConfig(model=args.model, group_size=args.group, steps=args.steps, cap_usd=args.cap, require_band=False, drop_zero_spread=False,
                      max_tokens=args.max_tokens, enable_thinking=args.thinking, learning_rate=args.lr, turn_eta=args.eta, max_groups_per_step=3)
    rep: dict = {"model": args.model}
    try:
        out = asyncio.run(train([task], config=cfg, run_dir=out_dir / "run", **_envs_kw(args)))
        rep.update(train=out, turns=_turn_stats(out_dir / "run" / "rollouts"))
        if out.get("sampler_path"):
            rep["adapter"] = asyncio.run(download_adapter(out["sampler_path"], out_dir / "adapter"))
    finally:
        removed = cleanup_containers()
    rep["containers_removed"] = removed
    write(out_dir / "smoke_rl.json", rep)


def cmd_probe(args) -> None:
    from crucible.train.bridges.tinker.backend import RealTinker
    from crucible.train.bridges.tinker.sampler import make_sampler
    from crucible.train.bridges.tinker.spend import SpendMeter

    need_key()
    out_dir = _out(args)

    async def go():
        meter = SpendMeter(args.cap, path=out_dir / "spend.jsonl")
        backend = RealTinker()
        s = await make_sampler(backend, args.model, meter=meter)
        llm = s.llm()

        async def one(i: int):
            t0 = time.monotonic()
            try:
                r = await llm.complete([{"role": "user", "content": f"Reply with the number {i} and nothing else."}], max_tokens=16, temperature=0.0,
                                       sample_key=f"probe-{i}")
                return {"i": i, "ok": True, "s": round(time.monotonic() - t0, 3), "text": r.text[:40]}
            except Exception as e:  # noqa: BLE001 -- the probe records every failure
                return {"i": i, "ok": False, "s": round(time.monotonic() - t0, 3), "error": f"{type(e).__name__}: {str(e)[:200]}"}

        t0 = time.monotonic()
        res = await asyncio.gather(*(one(i) for i in range(args.n)))
        wall = time.monotonic() - t0
        lat = sorted(r["s"] for r in res if r["ok"])
        return {"model": args.model, "n": args.n, "ok": sum(r["ok"] for r in res), "wall_s": round(wall, 2),
                "latency_s": {"p50": lat[len(lat) // 2], "max": lat[-1], "min": lat[0]} if lat else None, "errors": [r for r in res if not r["ok"]][:10],
                "retry_stats": backend.stats, "spend": meter.summary()}

    write(out_dir / f"probe_{args.n}.json", asyncio.run(go()))


def cmd_openai(args) -> None:
    from openai import OpenAI

    need_key()
    c = OpenAI(base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", api_key=os.environ["TINKER_API_KEY"])
    t0 = time.monotonic()
    out: dict = {"checkpoint": args.checkpoint}
    tools = [{"type": "function", "function": {"name": "shell", "description": "Run a shell command", "parameters": {"type": "object",
              "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    try:
        r = c.chat.completions.create(model=args.checkpoint, messages=[{"role": "user", "content": "List the files here with the shell tool."}], tools=tools,
                                      max_tokens=96, temperature=0.0)
        m = r.choices[0].message
        out.update(ok=True, content=(m.content or "")[:300], tool_calls=[{"name": t.function.name, "arguments": t.function.arguments} for t in m.tool_calls or []],
                   usage=r.usage.model_dump() if r.usage else None, finish=r.choices[0].finish_reason)
    except Exception as e:  # noqa: BLE001
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")
    out["latency_s"] = round(time.monotonic() - t0, 2)
    write(_out(args) / "openai_check.json", out)


def cmd_download(args) -> None:
    from crucible.factory.tinker import download_adapter

    need_key()
    write(_out(args) / "download.json", asyncio.run(download_adapter(args.path, _out(args) / "adapter")))


def add_common(p, *, model="Qwen/Qwen3.6-35B-A3B", cap=1.0) -> None:
    """The flags every command shares (also used by ``delta_rl.py``)."""
    p.add_argument("--model", default=model)
    p.add_argument("--cap", type=float, default=cap, help="hard USD cap for this command")
    p.add_argument("--out", default=str(ROOT / "data" / "tinker" / time.strftime("%Y%m%d-%H%M%S")))
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--max-tokens", type=int, default=None, help="per reply (default 1024; 8192 on --backend vllm, recorded in each arm)")
    p.add_argument("--thinking", action="store_true", help="enable_thinking (default off; recorded)")
    p.add_argument("--tool-limit", type=int, default=16)
    p.add_argument("--agent-s", type=float, default=900.0)
    p.add_argument("--rollout-s", type=float, default=1800.0)
    p.add_argument("--ckpt-every", type=int, default=3)
    p.add_argument("--rows")
    p.add_argument("--prompts")
    p.add_argument("--env-file", default=str(ROOT.parent / ".env"), help="dotenv file holding TINKER_API_KEY (read for that key only)")
    p.add_argument("--no-observe", action="store_true", help="do not log to W&B / Weave (default: offline runs under data/observe; "
                                                             "scripts/observe.py sync uploads them)")


def add_train_flags(p) -> None:
    """The RL flags of ``train`` and ``smoke-rl`` (``delta_rl.py train`` takes the same, unchanged)."""
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--groups-per-step", type=int, default=1)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--loss", default="importance_sampling", choices=["importance_sampling", "ppo", "cispo"])
    p.add_argument("--loss-scale", default="sum", choices=["sum", "token_mean"])
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--max-staleness", type=int, default=0)
    p.add_argument("--async-groups", type=int, default=0)
    p.add_argument("--targeting")
    p.add_argument("--no-band", action="store_true", help="skip the per-model band check (smoke runs only)")
    p.add_argument("--keep-flat", action="store_true", help="train zero-spread groups too")
    p.add_argument("--quarantine-q", type=int, default=2, help="flat groups on this many consecutive steps quarantine a task (0: off)")
    p.add_argument("--quarantine-r", type=int, default=3, help="steps a quarantined task is not sampled, then retried once")
    p.add_argument("--sampler", default="round_robin", choices=["round_robin", "difficulty"],
                   help="difficulty: online p̂ filtering and learnability weights (train.curriculum, swarm-pipeline/12)")
    p.add_argument("--band-lo", type=float, default=0.1, help="difficulty sampler: lower edge of the p̂ band")
    p.add_argument("--band-hi", type=float, default=0.7, help="difficulty sampler: upper edge of the p̂ band")
    p.add_argument("--explore", type=float, default=0.1, help="difficulty sampler: share of picks that re-measure out-of-band tasks")
    p.add_argument("--difficulty-decay", type=float, default=0.9, help="difficulty sampler: per-step decay of old p̂ evidence")


def train_config(args):
    """``TrainConfig`` from the train flags (shared with ``delta_rl.py``)."""
    from crucible.train.bridges.tinker.trainer import TrainConfig

    return TrainConfig(model=args.model, lora_rank=args.rank, group_size=args.group, groups_per_step=args.groups_per_step, steps=args.steps,
                       loss_fn=args.loss, learning_rate=args.lr, turn_eta=args.eta, max_staleness=args.max_staleness, async_groups=args.async_groups,
                       max_tokens=args.max_tokens, enable_thinking=args.thinking, cap_usd=args.cap, require_band=not args.no_band,
                       drop_zero_spread=not args.keep_flat, loss_scale=args.loss_scale, quarantine_q=args.quarantine_q, quarantine_r=args.quarantine_r,
                       sampler=args.sampler, band_lo=args.band_lo, band_hi=args.band_hi, explore=args.explore, difficulty_decay=args.difficulty_decay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = add_common

    for name, fn in (("train", cmd_train), ("smoke-rl", cmd_smoke_rl)):
        p = sub.add_parser(name)
        common(p)
        add_train_flags(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("eval")
    common(p)
    p.add_argument("--checkpoint")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="run seed (a fresh baseline uses one the selection run did not)")
    p.add_argument("--purpose", default="baseline", choices=["selection", "baseline", "control", "trained", "smoke"])
    backend_args(p)
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("compare")
    common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--baseline", help="metrics.json of the fresh baseline run")
    p.add_argument("--selection", help="metrics.json of the selection run")
    backend_args(p)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("power", help="offline A/A noise, MDE and the tasks needed (no key, $0)")
    p.add_argument("--arms", nargs="+", required=True, help="LABEL=DIR: an arm dir with episodes.jsonl, or a dir of <task>/episodes.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--target", type=float, nargs="+", default=[0.20, 0.15])
    p.add_argument("--json", required=True)
    p.add_argument("--env-file", default=str(ROOT.parent / ".env"))
    p.set_defaults(fn=cmd_power)
    p = sub.add_parser("smoke-tools")
    common(p, cap=0.25)
    p.add_argument("--train-step", action="store_true")
    p.set_defaults(fn=cmd_smoke_tools)
    p = sub.add_parser("probe-concurrency")
    common(p, cap=0.05)
    p.add_argument("--n", type=int, default=16)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("openai-check")
    common(p)
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(fn=cmd_openai)
    p = sub.add_parser("download")
    common(p)
    p.add_argument("--path", required=True)
    p.set_defaults(fn=cmd_download)
    args = ap.parse_args()
    if hasattr(args, "max_tokens") and args.max_tokens is None:  # commands without common() (power) have no reply cap
        if getattr(args, "backend", "tinker") == "vllm":
            from crucible.train.bridges.vllm.client import DEFAULT_MAX_TOKENS

            args.max_tokens = DEFAULT_MAX_TOKENS
        else:
            args.max_tokens = 1024
    load_env_key(Path(args.env_file))
    args.fn(args)


if __name__ == "__main__":
    main()
