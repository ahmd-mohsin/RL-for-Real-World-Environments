#!/usr/bin/env python
"""Crucible RL on DeltaAI: prime-rl's GPU trainer and vLLM server under Crucible's loop (swarm-pipeline/14).

Every training flag is ``tinker_rl.py train``'s (shared builders ``add_common`` / ``add_train_flags`` / ``train_config``),
so a run means the same on both backends; the deployment adds its own (``--train-gpus --infer-gpus --ep --cp --seq-len
--kl-coeff --pg-correction --sc-topk --time``). ``--pg-correction`` (default ``sc+mis``: score centering with masked
importance sampling, swarm-pipeline/14 §7) is the off-policy correction of the prime-rl loss; it needs ``--loss
importance_sampling`` (the default), and ``--pg-correction none`` keeps the plain ``--loss`` objective (ppo, cispo). The account is ``bjhg-dtai-gh`` (CIS262200) and nothing else (``deploy.DeltaConfig``).

Commands:

- ``plan --steps 30 --groups-per-step 4 --group 8 [--balance 991]``: GPU-hours per step and for the run (an estimate,
  stated as one), against the account's balance less a benchmark reserve. No cluster access.
- ``render --rows ROWS.jsonl --targeting T.json --run-name NAME --stage DIR [--resume-step S]``: ``trainer.toml``,
  ``inference.toml``, ``job.sbatch``, ``deploy.json`` in ``DIR``; prints the submit command (``deploy.submit_command``:
  ``env -u SBATCH_ACCOUNT ... sbatch --account=bjhg-dtai-gh``). Held-out tasks and tasks without a band are refused.
- ``render-smoke --stage DIR --run-name NAME``: the plumbing smoke on a small model (default ``Qwen/Qwen3-0.6B``, 1 GPU
  trainer + 1 GPU vLLM on the ``test`` partition): install prime-rl at the pin, two RL steps, the trainer killed and
  relaunched from its checkpoint, one more step (``smoke``).
- ``train ...`` (on the compute node, inside the job): the Tinker loop over ``PrimeRLBackend``.
- ``smoke --out OUT [--resume REF]`` (on the compute node): ``train.bridges.primerl.smoke.run``.
- ``topk-bench --report R.json [--ks 0,8,32]`` (on the compute node, next to vLLM): what returning the top-k head costs.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tinker_cli():
    spec = importlib.util.spec_from_file_location("tinker_rl", ROOT / "scripts" / "tinker_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# per model call on DeltaAI (coordinator, 2026-10-26): 4096 cut the replies of ~90 stored episodes and ended GPU migration episodes
MAX_TOKENS = 8192


def add_deploy_flags(p) -> None:
    from crucible.train.bridges.primerl.deploy import DeltaConfig
    from crucible.train.bridges.primerl.loss import CORRECTIONS

    p.set_defaults(max_tokens=MAX_TOKENS)  # a reply cut there is a limit stop (max_output_tokens), never a skill failure
    d = DeltaConfig()
    p.add_argument("--base-model", default=d.base_model, help="the HF repo vLLM and the trainer load")
    p.add_argument("--partition", default=d.partition)
    p.add_argument("--nodes", type=int, default=d.nodes)
    p.add_argument("--train-gpus", type=int, default=d.train_gpus)
    p.add_argument("--infer-gpus", type=int, default=d.infer_gpus)
    p.add_argument("--ep", type=int, default=d.ep)
    p.add_argument("--cp", type=int, default=d.cp)
    p.add_argument("--infer-tp", type=int, default=d.infer_tp)
    p.add_argument("--seq-len", type=int, default=d.seq_len)
    p.add_argument("--kl-coeff", type=float, default=d.kl_coeff, help="sampler-learner anchor β (0: off)")
    p.add_argument("--pg-correction", default=d.pg_correction, choices=list(CORRECTIONS),
                   help="off-policy correction: score centering (sc) alone or with TIS / MIS, TIS / MIS alone, or none (the --loss objective)")
    p.add_argument("--sc-topk", type=int, default=d.sc_topk, help="the sampler's top-k logprobs per token kept for score centering")
    p.add_argument("--time", default=d.time)
    p.add_argument("--workdir", default=d.workdir)
    p.add_argument("--envdir", default=d.envdir, help="prime-rl's checkout, venv, uv cache and CPython (NVMe)")
    p.add_argument("--no-router-replay", action="store_true", help="sample over /v1/completions and let the trainer recompute MoE routing")


def delta_config(args):
    from crucible.train.bridges.primerl.deploy import DeltaConfig

    return DeltaConfig(base_model=args.base_model, partition=args.partition, nodes=args.nodes, train_gpus=args.train_gpus, infer_gpus=args.infer_gpus,
                       ep=args.ep, cp=args.cp, infer_tp=args.infer_tp, seq_len=args.seq_len, lora_rank=args.rank, kl_coeff=args.kl_coeff, time=args.time,
                       workdir=args.workdir, envdir=args.envdir, router_replay=not args.no_router_replay, pg_correction=args.pg_correction, sc_topk=args.sc_topk)


def cmd_plan(args) -> None:
    from crucible.train.bridges.primerl.deploy import plan

    p = plan(steps=args.steps, tasks_per_step=args.groups_per_step, group_size=args.group, d=delta_config(args), balance=args.balance)
    print(json.dumps(asdict(p), indent=1))


# the venv's interpreter: a uv-managed CPython (it ships its headers), never the system python, whose headers DeltaAI does
# not install. Triton compiles a small C launcher (cuda_utils.c, #include <Python.h>) on the GPU node; without the header
# every Triton kernel fails, and flash-linear-attention falls back to CPU (smoke attempt 5, job 3467688)
PYTHON = "3.12"


def uv_env(d) -> str:
    """The environment every uv call on the cluster shares: build and job alike. The prime-rl checkout, its venv, uv's cache
    and the managed CPython sit on ``d.envdir`` (/work/nvme): uv hard-links the venv from its cache, so both must be there.
    On /work/hdd a cold GPU node took 8 min to import torch + vLLM and 13 min to start the trainer (attempt 6)."""
    e = d.envdir
    return (f"export UV_CACHE_DIR={e}/uv-cache UV_PYTHON_INSTALL_DIR={e}/uv-python UV_PYTHON_PREFERENCE=only-managed HF_HOME={d.workdir}/hf_cache "
            f"PATH=$HOME/.local/bin:$PATH UV_FROZEN=1")


# the modules the GPU job imports, imported once at build time: a missing package fails the build, not a GPU job (attempt 6:
# the trainer needs prime_kernels, from prime-rl's kernels extra), and their bytecode is written on the login node
JOB_IMPORTS = ("import sys, prime_rl, verifiers, torch, vllm, triton, transformers; "
               "from prime_rl.trainer.rl.train import train; import prime_rl.entrypoints.trainer, prime_rl.entrypoints.inference, prime_rl.inference.vllm.server; "
               "import fla.ops.gated_delta_rule, fla.modules.conv; "
               "print('python', sys.version.split()[0], 'verifiers', verifiers.__version__, 'torch', torch.__version__, torch.version.cuda, 'vllm', vllm.__version__, 'triton', triton.__version__)")
CRUCIBLE_IMPORTS = "import crucible.train.bridges.primerl.loss, crucible.train.bridges.primerl.backend, crucible.train.bridges.primerl.smoke, crucible.train.bridges.primerl.kernel_check, crucible.train.bridges.primerl.sc_head, crucible.train.bridges.primerl.topk_server; print('crucible bridge ok')"


HAS_HEADERS = ("import os, sys, sysconfig; h = os.path.join(sysconfig.get_paths()['include'], 'Python.h'); "
               "print('Python.h', h, os.path.exists(h)); sys.exit(0 if os.path.exists(h) else 1)")


def build_script(d, *, version: str) -> str:
    """The environment build, run once on the login node (downloads, unpacks and imports only; coordinator, 2026-10-26:
    three install bugs each cost a turn on the one interactive GPU slot). Nothing compiles: the CUDA builds of mamba and
    causal-conv1d are skipped (only nemotron_h uses them), and ``mooncake-transfer-engine`` is not installed (its aarch64
    wheel needs glibc 2.39; DeltaAI has 2.38; it is for KV offload, which this run does not use). The venv is built on a
    uv-managed CPython (``PYTHON``) and the build fails unless its ``Python.h`` exists; a venv left on another interpreter
    is removed and rebuilt from the uv cache. Extras: ``gpu``, ``flash-attn`` and ``kernels`` (prebuilt ``prime_kernels``,
    which the trainer imports unconditionally). Every module the job imports is imported here once (``JOB_IMPORTS``)."""
    w, e = d.workdir, d.envdir
    return f"""#!/bin/bash
set -uo pipefail
{uv_env(d)}
export MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAX_JOBS=4
[ -x $HOME/.local/bin/uv ] || (curl -LsSf https://astral.sh/uv/install.sh | sh)
mkdir -p {e} && cd {e}
[ -d prime-rl ] || git clone --depth 1 --branch {version} https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl && rm -f .crucible-installed-{version}
# the submodules are declared with SSH URLs and the cluster has no GitHub key: HTTPS, full history
[ -f deps/pydantic-config/pyproject.toml ] || {{ rm -rf deps/verifiers deps/renderers deps/pydantic-config .git/modules/deps
  git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init -- deps/verifiers deps/renderers deps/pydantic-config || exit 11; }}
# a uv-managed CPython {PYTHON} with its headers (Triton compiles against Python.h on the GPU node)
nice -n 19 uv python install {PYTHON} || exit 16
PY=$(uv python find {PYTHON}) || exit 16  # only-managed: never the system python
"$PY" -c "{HAS_HEADERS}" || exit 17
[ -x .venv/bin/python ] && ! .venv/bin/python -c "{HAS_HEADERS}" && {{ echo "venv on an interpreter without headers: rebuilt"; rm -rf .venv; }}
# prime-rl's own lockfile, never re-resolved (UV_FROZEN, also for every later uv run): the trainer and vLLM extras only
nice -n 19 uv sync --frozen --python "$PY" --extra gpu --extra flash-attn --extra kernels --no-install-package mooncake-transfer-engine || exit 12
nice -n 19 uv pip install 'pydantic>=2.12' pyyaml tomli-w httpx rich click dill || exit 13
uv run --frozen python -c "{HAS_HEADERS}" || exit 17
nice -n 19 uv run --frozen python -c "{JOB_IMPORTS}" || exit 14
PYTHONPATH={w}/crucible/src nice -n 19 uv run --frozen python -c "{CRUCIBLE_IMPORTS}" || exit 15
touch {e}/prime-rl/.crucible-installed-{version}
echo BUILD_OK
"""


def setup_lines(d, *, version: str) -> str:
    """The GPU job's preamble: the environment ``build_script`` made on the login node, reused; never installed here."""
    from crucible.train.bridges.primerl.deploy import common_env_line

    w, e = d.workdir, d.envdir
    return f"""{uv_env(d)}
[ -f {e}/prime-rl/.crucible-installed-{version} ] || {{ echo "prime-rl environment not built: run build_env.sh on the login node first"; exit 4; }}
cd {e}/prime-rl
export PYTHONPATH={w}/crucible/src  # Crucible's source (rsynced); the trainer imports its loss from here
{common_env_line()}
"""


def passthrough(argv: list[str]) -> list[str]:
    """The ``render`` arguments the in-job ``train`` takes: the staging-only flags (with their values) removed."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("--stage", "--resume-step"):
            skip = True
            continue
        if a.startswith(("--stage=", "--resume-step=")):
            continue
        out.append(a)
    return out


def cmd_render(args) -> None:
    from crucible.train.bridges.primerl.deploy import PRIME_RL_VERSION, refuse_tasks, render, submit_command
    from crucible.train.bridges.primerl.loss import check_combination

    tl = _tinker_cli()
    cfg = tl.train_config(args)
    d = delta_config(args)
    try:
        check_combination(cfg.loss_fn, d.pg_correction)
    except ValueError as e:
        sys.exit(f"refused: {e}")
    if args.rows:
        tasks = tl._tasks_from_rows(args)
        targeting = json.loads(Path(args.targeting).read_text()) if args.targeting else None
        bad = refuse_tasks(tasks, targeting, cfg.model, require_band=cfg.require_band)
        if bad:
            sys.exit("refused: " + "; ".join(bad[:8]))
    out = Path(d.workdir) / "runs" / args.run_name
    cmd = f"uv run python {d.workdir}/crucible/scripts/delta_rl.py train --run-dir {out}/crucible --primerl-out {out} " + " ".join(passthrough(sys.argv[2:]))
    paths = render(cfg, d, Path(args.stage), run_name=args.run_name, crucible_cmd=cmd, resume_step=args.resume_step,
                   setup=setup_lines(d, version=PRIME_RL_VERSION))
    (Path(args.stage) / "build_env.sh").write_text(build_script(d, version=PRIME_RL_VERSION))
    print(json.dumps({k: str(v) for k, v in paths.items()} | {"submit": " ".join(submit_command(out / "job.sbatch", d))}, indent=1))


def cmd_render_smoke(args) -> None:
    import tomli_w

    from crucible.train.bridges.primerl.deploy import PRIME_RL_VERSION, DeltaConfig, inference_toml, submit_command, trainer_command, trainer_toml
    from crucible.train.bridges.tinker.trainer import TrainConfig

    d = DeltaConfig(base_model=args.base_model, partition=args.partition, nodes=1, train_gpus=1, infer_gpus=1, ep=1, cp=1, infer_tp=1, seq_len=4096,
                    lora_rank=16, time=args.time, workdir=args.workdir, envdir=args.envdir, mem_gb_per_gpu=100, router_replay=False,  # a dense model: no experts
                    pg_correction=args.pg_correction, sc_topk=args.sc_topk)
    cfg = TrainConfig(model="Qwen/Qwen3.6-35B-A3B", lora_rank=16, steps=3, learning_rate=args.lr, loss_fn="importance_sampling", require_band=False)
    out = Path(d.workdir) / "runs" / args.run_name
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    t0 = trainer_toml(cfg, d, out)
    t0["model"].pop("ep")
    t0["model"].pop("cp")
    t1 = json.loads(json.dumps(t0))
    t1["resume"] = {"step": 1}  # the checkpoint of step 1 (the second step): the relaunched trainer continues at step 2
    inf = inference_toml(cfg, d)
    (stage / "trainer.toml").write_text(tomli_w.dumps(t0))
    (stage / "trainer_resume.toml").write_text(tomli_w.dumps(t1))
    (stage / "inference.toml").write_text(tomli_w.dumps(inf))
    py = (f"uv run python {d.workdir}/crucible/scripts/delta_rl.py smoke --base-model {d.base_model} --lr {args.lr} --primerl-out {out} "
          f"--sc-topk {d.sc_topk if d.centered else 0}")
    job = f"""#!/bin/bash
#SBATCH --job-name=crucible-rl-smoke
#SBATCH --account={d.account}
#SBATCH --partition={d.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={args.gpus}
#SBATCH --cpus-per-gpu=16
#SBATCH --mem={args.mem_gb}G
#SBATCH --time={d.time}
#SBATCH --output={out}/slurm-%j.out
set -uo pipefail
unset SBATCH_ACCOUNT SLURM_ACCOUNT
mkdir -p {out}
{setup_lines(d, version=PRIME_RL_VERSION)}
echo "=== versions"; uname -m; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uv run python -c "{HAS_HEADERS}" || {{ echo "the venv's python has no headers: rerun build_env.sh"; exit 8; }}
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 > {out}/gpu_mem.csv 2>/dev/null &
MEM=$!
# vLLM, the trainer and the kernel check start together: each import is paid once, in parallel
CUDA_VISIBLE_DEVICES=1 uv run inference @ {out}/inference.toml > {out}/inference.log 2>&1 &
INF=$!
CUDA_VISIBLE_DEVICES=0 {trainer_command(f"{out}/trainer.toml", gpus=1, port="$((29500 + SLURM_JOB_ID % 1000))")} > {out}/trainer.log 2>&1 &
TRN=$!
CUDA_VISIBLE_DEVICES=0 timeout 1500 uv run python -m crucible.train.bridges.primerl.kernel_check > {out}/kernel_check.json 2>{out}/kernel_check.err &
KC=$!
T0=$SECONDS
for i in $(seq 1 240); do curl -sf localhost:{d.port}/health > /dev/null && break; kill -0 $INF 2>/dev/null || {{ echo "vLLM exited"; tail -40 {out}/inference.log; exit 5; }}; kill -0 $TRN 2>/dev/null || {{ echo "trainer exited"; tail -40 {out}/trainer.log; exit 6; }}; sleep 10; done
curl -sf localhost:{d.port}/health > /dev/null || {{ echo "vLLM not healthy after 40 min"; tail -40 {out}/inference.log; exit 7; }}
echo "vLLM healthy after $((SECONDS - T0)) s"
echo "=== phase 1: two steps"
{py} --steps 2 --report {out}/smoke1.json || {{ echo "phase 1 failed"; tail -30 {out}/trainer.log; tail -20 {out}/inference.log; kill $INF $TRN $MEM 2>/dev/null; exit 9; }}
kill $TRN; wait $TRN 2>/dev/null; pkill -u $USER -f prime_rl.trainer.rl.train; sleep 5  # torchrun's workers too: GPU 0 is free for the relaunch
echo "=== phase 2: trainer relaunched from its step-1 checkpoint"
CUDA_VISIBLE_DEVICES=0 {trainer_command(f"{out}/trainer_resume.toml", gpus=1, port="$((30500 + SLURM_JOB_ID % 1000))", tag="resume")} > {out}/trainer_resume.log 2>&1 &
TRN=$!
{py} --steps 1 --resume-from {out}/smoke1.json --report {out}/smoke2.json
RC=$?
[ $RC = 0 ] || {{ tail -30 {out}/trainer_resume.log; tail -20 {out}/inference.log; }}
echo "=== the student's Gated DeltaNet kernels on the GPU, and vLLM's registry (advisory: the 0.6B smoke does not need them)"
wait $KC; echo "kernel_check rc=$?"; cat {out}/kernel_check.json; grep -m3 -i -e error -e "roll back" {out}/kernel_check.err
echo "=== the top-k head's cost on the generate endpoint (advisory)"
uv run python {d.workdir}/crucible/scripts/delta_rl.py topk-bench --base-model {d.base_model} --port {d.port} --report {out}/topk_bench.json || true
kill $INF $TRN $MEM 2>/dev/null || true
echo "=== peak GPU memory (MiB) per GPU"; awk -F', ' '{{g=$2; m=$3+0; if(m>p[g])p[g]=m}} END{{for(g in p) print "gpu", g, p[g]}}' {out}/gpu_mem.csv
echo "=== done rc=$RC"
exit $RC
"""
    (stage / "job.sbatch").write_text(job)
    (stage / "build_env.sh").write_text(build_script(d, version=PRIME_RL_VERSION))
    print(json.dumps({"stage": str(stage), "cluster_out": str(out), "submit": " ".join(submit_command(out / "job.sbatch", d))}, indent=1))


def _backend(args, *, lr: float, loss: str, ranks: int, seq_len: int, router_replay: bool = False, sc_topk: int = 0):
    from crucible.train.bridges.primerl.backend import FileIO, HttpClient, PrimeRLBackend
    from crucible.train.bridges.primerl.routing import RoutingStore
    from crucible.train.bridges.primerl.topk import TopKStore

    return PrimeRLBackend(io=FileIO(Path(args.primerl_out), ranks=ranks, seq_len=seq_len, model=args.base_model),
                          http=HttpClient(f"http://localhost:{args.port}"), base_model=args.base_model, learning_rate=lr, loss_kind=loss,
                          routing=RoutingStore() if router_replay else None, topk=TopKStore(k=sc_topk) if sc_topk > 0 else None)


def cmd_smoke(args) -> None:
    from transformers import AutoTokenizer

    from crucible.train.bridges.primerl.smoke import run

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def encode(text: str) -> list[int]:
        msgs = [{"role": "user", "content": text}]
        try:
            return list(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            return list(tok.apply_chat_template(msgs, add_generation_prompt=True))

    resume = sampler_path = None
    if args.resume_from:
        prev = json.loads(Path(args.resume_from).read_text())
        resume, sampler_path = prev["last_checkpoint"], prev["last_adapter"]
    b = _backend(args, lr=args.lr, loss="importance_sampling", ranks=1, seq_len=4096, sc_topk=args.sc_topk)
    out = asyncio.run(run(b, encode=encode, decode=lambda ids: tok.decode(list(ids), skip_special_tokens=True), model="Qwen/Qwen3.6-35B-A3B",
                          steps=args.steps, learning_rate=args.lr, resume=resume, sampler_path=sampler_path, rank=16))
    Path(args.report).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])


def cmd_topk_bench(args) -> None:
    """The top-k head's cost on the running vLLM server (``backend.topk_cost``): wall time, tokens/s and response bytes
    per token at each k, k = 0 the baseline. Run inside a job, next to the server."""
    from transformers import AutoTokenizer

    from crucible.train.bridges.primerl.backend import HttpClient, topk_cost

    tok = AutoTokenizer.from_pretrained(args.base_model)
    ids = list(tok.apply_chat_template([{"role": "user", "content": "Write a long story about a lighthouse keeper."}], add_generation_prompt=True))
    rows = asyncio.run(topk_cost(HttpClient(f"http://localhost:{args.port}"), args.base_model, ids, ks=[int(k) for k in args.ks.split(",")],
                                 n=args.n, max_tokens=args.max_tokens))
    Path(args.report).write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


def cmd_train(args) -> None:
    from crucible.factory.tinker import train
    from crucible.train.bridges.primerl.backend import local_model

    tl = _tinker_cli()
    cfg = tl.train_config(args)
    d = delta_config(args)
    tasks = tl._tasks_from_rows(args)
    targeting = json.loads(Path(args.targeting).read_text()) if args.targeting else None
    b = _backend(args, lr=cfg.learning_rate, loss=cfg.loss_fn, ranks=d.data_ranks, seq_len=d.seq_len, router_replay=d.router_replay,
                 sc_topk=d.sc_topk if d.centered else 0)
    out = asyncio.run(train(tasks, config=cfg, run_dir=Path(args.run_dir), targeting=targeting, observe=tl._observe(args), backend=b,
                            model=local_model(cfg.model, context=d.seq_len), **tl._envs_kw(args)))
    tl.write(Path(args.run_dir) / "train_report.json", {k: v for k, v in out.items() if k != "steps_log"} | {"steps_log": out["steps_log"]})


def main() -> None:
    tl = _tinker_cli()
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    tl.add_common(p, cap=1e9)
    tl.add_train_flags(p)
    add_deploy_flags(p)
    p.add_argument("--balance", type=float, help="the account's GPU-hours left (accounts | grep bjhg)")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("render")
    tl.add_common(p, cap=1e9)
    tl.add_train_flags(p)
    add_deploy_flags(p)
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", required=True, help="local directory the files are written to (rsync it to the run directory)")
    p.add_argument("--resume-step", type=int)
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("render-smoke")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--partition", default="test")
    p.add_argument("--time", default="01:45:00")
    p.add_argument("--workdir", default="/work/hdd/bjhg/mahmedmohsin/crucible-train")
    p.add_argument("--envdir", default="/work/nvme/bjhg/mahmedmohsin/crucible-train")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gpus", type=int, default=2, help="GPUs requested (the smoke uses 2: trainer + vLLM)")
    p.add_argument("--mem-gb", type=int, default=200, help="memory request; keep it proportional to the GPUs (charging follows the largest share)")
    p.add_argument("--run-name", default="smoke")
    p.add_argument("--stage", required=True)
    p.add_argument("--pg-correction", default="sc+mis", help="the smoke's loss correction (sc+mis exercises the top-k path end to end)")
    p.add_argument("--sc-topk", type=int, default=32)
    p.set_defaults(fn=cmd_render_smoke)
    for name, fn in (("train", cmd_train), ("smoke", cmd_smoke)):
        p = sub.add_parser(name)
        p.add_argument("--primerl-out", required=True, help="the prime-rl trainer's output_dir")
        p.add_argument("--port", type=int, default=8000)
        if name == "train":
            tl.add_common(p, cap=1e9)
            tl.add_train_flags(p)
            add_deploy_flags(p)
            p.add_argument("--run-dir", required=True)
            p.add_argument("--run-name", help="recorded; the run directory is --run-dir")
        else:
            p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--steps", type=int, default=2)
            p.add_argument("--resume-from")
            p.add_argument("--report", required=True)
            p.add_argument("--sc-topk", type=int, default=0, help="keep the sampler's top-k head (score centering); 0: off")
        p.set_defaults(fn=fn)
    p = sub.add_parser("topk-bench")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ks", default="0,8,32")
    p.add_argument("--n", type=int, default=8, help="concurrent calls per k")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--report", required=True)
    p.set_defaults(fn=cmd_topk_bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
