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
  The job derives the trainer's resume step from the loop's ``state.json`` each time it starts (``deploy.launch_toml``),
  so the same job file resubmits safely; ``--resume-step`` only pins it, and a pin that disagrees with the state fails
  the job before any GPU work.
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


# per model call on DeltaAI (coordinator, 2026-10-26): 4096 cut the replies of ~90 stored episodes
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
    p.add_argument("--optimization-dtype", default=d.optimization_dtype, choices=["float32", "bfloat16"],
                   help="the trainer's parameter dtype (bfloat16 halves the frozen base's memory; float32 keeps the LoRA updates exact)")
    p.add_argument("--kl-coeff", type=float, default=d.kl_coeff, help="sampler-learner anchor β (0: off)")
    p.add_argument("--pg-correction", default=d.pg_correction, choices=list(CORRECTIONS),
                   help="off-policy correction: score centering (sc) alone or with TIS / MIS, TIS / MIS alone, or none (the --loss objective)")
    p.add_argument("--sc-topk", type=int, default=d.sc_topk, help="the sampler's top-k logprobs per token kept for score centering")
    p.add_argument("--time", default=d.time)
    p.add_argument("--workdir", default=d.workdir)
    p.add_argument("--envdir", default=d.envdir, help="prime-rl's checkout, venv, uv cache and CPython (NVMe)")
    p.add_argument("--hf-home", default=d.hf_home, help="the HF cache the job reads (default <workdir>/hf_cache)")
    p.add_argument("--no-router-replay", action="store_true", help="sample over /v1/completions and let the trainer recompute MoE routing")


def add_debug_flags(p) -> None:
    p.add_argument("--debug", default="", help="the debug lane's data dir: train on its tasks (--tasks) instead of --rows")
    p.add_argument("--tasks", nargs="+", default=[], help="debug-lane task ids (training split only)")
    p.add_argument("--grade-parallel", type=int, default=2, help="grading containers at once per World image")
    p.add_argument("--ckpt-cap", type=int, default=0, help="Phi checkpoints per episode (0: tinker_rl's 20); the debug lane uses 15")


def delta_config(args):
    from crucible.train.bridges.primerl.deploy import DeltaConfig

    return DeltaConfig(base_model=args.base_model, partition=args.partition, nodes=args.nodes, train_gpus=args.train_gpus, infer_gpus=args.infer_gpus,
                       ep=args.ep, cp=args.cp, infer_tp=args.infer_tp, seq_len=args.seq_len, lora_rank=args.rank, kl_coeff=args.kl_coeff, time=args.time,
                       workdir=args.workdir, envdir=args.envdir, hf_home=args.hf_home, optimization_dtype=args.optimization_dtype, router_replay=not args.no_router_replay, pg_correction=args.pg_correction, sc_topk=args.sc_topk)


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
    return (f"export UV_CACHE_DIR={e}/uv-cache UV_PYTHON_INSTALL_DIR={e}/uv-python UV_PYTHON_PREFERENCE=only-managed HF_HOME={d.hf_cache} "
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


# the inference worker's Apptainer host (handoff-deltaai-inference): the World SIFs and the lane's training data (no
# held-out store on the cluster)
INFER = "/work/hdd/bjhg/mahmedmohsin/crucible-infer"
WORLD_PREFIX_LOCKDOWN = "crucible.world.lockdown"


def world_lines(d, *, out: str, data_src: str, sif_dir: str, world_image: str) -> str:
    """The job's Worlds (doc 14-deltaai §3, the inference worker's smoke.sbatch, reused): the ``docker`` shim first on
    ``PATH`` (Apptainer instances under the lockdown), node-local state, the lane's data copied to node-local disk without
    ACLs (``/work``'s ACLs ride into the graders' temp trees and the grader's ``cp -a`` into the fuse overlay fails, exit 125)
    with the clones' git alternates repointed, and the lockdown probe in the task's World image: no model episode runs
    unless its verdict is clean. ``$J`` (node-local) is the only bind root the shim allows; the loop's run directories live
    there too (``$J/runs``), copied to ``<out>/crucible-runs`` every 5 minutes and at the end."""
    w = d.workdir
    return f"""J=/tmp/crucible-$SLURM_JOB_ID
mkdir -p $J/state $J/ovl $J/cfg $J/cache $J/atmp $J/ws $J/data $J/runs
export CRUCIBLE_SIF_DIR={sif_dir} CRUCIBLE_SHIM_STATE=$J/state CRUCIBLE_SHIM_SCRATCH=$J/ovl CRUCIBLE_SHIM_CONFIGDIR=$J/cfg CRUCIBLE_SHIM_ALLOW=$J
export APPTAINER_CACHEDIR=$J/cache APPTAINER_TMPDIR=$J/atmp TMPDIR=$J CRUCIBLE_SHIM_LOG={out}/shim.log
export PATH={w}/crucible/scripts/deltaai:$PATH  # the docker shim
rsync -a --no-acls --no-xattrs {data_src}/ $J/data/debug/ || exit 21
find $J/data/debug -path "*objects/info/alternates" -print0 | xargs -0 -r sed -i "s|/[^ ]*/data/debug|$J/data/debug|g"
[ -d {out}/crucible-runs ] && rsync -a --no-acls --no-xattrs {out}/crucible-runs/ $J/runs/  # a later job resumes the loop's runs
uv run python -c "from {WORLD_PREFIX_LOCKDOWN} import PROBE; open('$J/ws/probe.sh','w').write(PROBE)" || exit 22
docker run --rm --network none -v "$J/ws:$J/ws" {world_image} bash "$J/ws/probe.sh" "$HOME" 0 > {out}/lockdown.txt 2> {out}/lockdown.err
uv run python -c "
from {WORLD_PREFIX_LOCKDOWN} import parse, verdict
p = verdict(parse(open('{out}/lockdown.txt').read()), home='$HOME')
print('LOCKDOWN', 'OK' if not p else 'FAILED', p)" | tee {out}/lockdown.verdict
grep -q "LOCKDOWN OK" {out}/lockdown.verdict || {{ echo "lockdown failed: no model episodes"; exit 23; }}
sync_back() {{ rsync -a --no-acls --no-xattrs $J/runs/ {out}/crucible-runs/ 2>/dev/null; }}
(while true; do sleep 300; sync_back; done) &
SYNCBACK=$!
"""


def passthrough(argv: list[str], drop: tuple[str, ...] = ("--stage", "--resume-step")) -> list[str]:
    """The ``render`` arguments the in-job ``train`` takes: the staging-only flags (with their values) removed."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in drop:
            skip = True
            continue
        if a.startswith(tuple(f"{f}=" for f in drop)):
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
    t1["resume"] = {"step": 2}  # phase 1 trains prime-rl steps 1 and 2 (backend.FIRST_STEP); the relaunch continues at step 3
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
echo "=== phase 2: trainer relaunched from its step-2 checkpoint"
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


# the smoke's bounds: a 0.6B step takes seconds, so a stalled handshake fails the job in minutes instead of holding the GPUs
# to the time limit (attempt 8 held them 1:55 waiting for a broadcast that never came)
SMOKE_TIMEOUTS = {"step_timeout_s": 900.0, "startup_timeout_s": 1200.0}


# render-verify's own flags (with values): kept out of the in-job ``train`` commands, which set their phase's step budget
VERIFY_ONLY = ("--stage", "--resume-step", "--steps", "--async-groups", "--max-staleness", "--debug", "--sync-steps", "--resume-steps",
               "--async-steps", "--verify-async-groups", "--verify-staleness", "--data-src", "--sif-dir", "--world-image", "--mem-gb",
               "--cpus-per-gpu", "--probe-time", "--keep-last", "--targeting")


def _train_cfg(tl, args, **over):
    ns = argparse.Namespace(**{**vars(args), **over})
    return tl.train_config(ns)


def verify_targeting(path: Path, tasks: list[str], model: str) -> dict:
    """The targeting records of ``tasks`` only, each for ``model``'s base, on the training split and in band (the trainer's
    ``check_band``); anything else is refused before a GPU is asked for."""
    from crucible.train.bridges.tinker.trainer import check_band

    allrec = json.loads(Path(path).read_text())
    rec = {t: allrec[t] for t in tasks if t in allrec}
    held = [t for t, r in rec.items() if str(r.get("split", "train")) != "train"]
    if held:
        raise SystemExit(f"refused: held out (V9): {held}")
    bad = check_band(tasks, model, rec)
    if bad:
        raise SystemExit("refused: " + "; ".join(bad))
    return rec


def cmd_render_verify(args) -> None:
    """The real student's GPU verification (lead, 2026-10-28): on training tasks of the debug lane, one job proves
    (1) score centering with MIS live, (2) MoE routing replay in the synchronous steps, (3) a resume from prime-rl's
    checkpoint that trains a step with a gradient, (4) the asynchronous loop (``--async-groups`` / ``--max-staleness``).

    - ``probe.sbatch`` (2 GPUs): the trainer alone on prime-rl's fake data at ``--seq-len``: the 35B loads, takes two
      steps and writes a checkpoint, with its memory and times, before the 4-GPU job is asked for.
    - ``verify.sbatch`` (4 GPUs): vLLM on GPUs 2-3 and the trainer on 0-1; the phases (``PHASES``, default all):
      ``sync`` (a run of ``--sync-steps``), ``resume`` (the trainer killed and relaunched from the step the loop's
      ``state.json`` names, ``launch_toml``, and the same run raised by ``--resume-steps``), ``async`` (a second run with its
      own trainer: ``--async-steps`` steps at ``--verify-async-groups`` / ``--verify-staleness``; a resume may not change
      those, ``RESUMABLE_CHANGES``, so it is its own run). A watchdog stops a phase whose trainer or vLLM died."""
    import tomli_w

    from crucible.train.bridges.primerl.deploy import PRIME_RL_VERSION, inference_toml, submit_command, trainer_command, trainer_toml

    tl = _tinker_cli()
    d = delta_config(args)
    if d.train_gpus != 2 or d.infer_gpus != 2:
        raise SystemExit("the verification job is one node: 2 trainer GPUs and 2 vLLM GPUs")
    tasks = list(args.tasks)
    if not tasks:
        raise SystemExit("--tasks: the debug-lane training tasks to verify on")
    total = args.sync_steps + args.resume_steps
    cfg_sync = _train_cfg(tl, args, steps=total, async_groups=0, max_staleness=0)
    cfg_async = _train_cfg(tl, args, steps=args.async_steps, async_groups=args.verify_async_groups, max_staleness=args.verify_staleness)
    rec = verify_targeting(Path(args.targeting), tasks, cfg_sync.model)
    out = Path(d.workdir) / "runs" / args.run_name
    stage = Path(args.stage)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "targeting.json").write_text(json.dumps(rec, indent=1))

    def trainer_doc(cfg, sub: str) -> dict:
        doc = trainer_toml(cfg, d, out / sub)
        doc["ckpt"]["keep_last"] = int(args.keep_last)  # the 35B's checkpoint is its full state dict: about 70 GB each
        return doc

    (stage / "trainer_sync.toml").write_text(tomli_w.dumps(trainer_doc(cfg_sync, "sync")))
    (stage / "trainer_async.toml").write_text(tomli_w.dumps(trainer_doc(cfg_async, "async")))
    probe = trainer_doc(_train_cfg(tl, args, steps=2), "probe")
    probe["data"] = {"fake": {"batch_size": d.data_ranks}}  # one seq_len micro batch per data rank; no broadcast in fake mode
    probe["enable_router_replay"] = False  # the fake batch carries no routes
    probe["ckpt"] = {"interval": 1, "keep_last": 1}  # times one 35B checkpoint save
    (stage / "trainer_probe.toml").write_text(tomli_w.dumps(probe))
    (stage / "inference.toml").write_text(tomli_w.dumps(inference_toml(cfg_async, d)))  # max_loras 2: the async phase shares it
    w = d.workdir
    common = " ".join(passthrough(sys.argv[2:], drop=VERIFY_ONLY))
    train = (f"uv run python {w}/crucible/scripts/delta_rl.py train --debug $J/data/debug --targeting {out}/targeting.json {common}")
    cmd_sync = f"{train} --run-dir $J/runs/sync --primerl-out {out}/sync --steps {args.sync_steps}"
    cmd_resume = f"{train} --run-dir $J/runs/sync --primerl-out {out}/sync --steps {total}"
    cmd_async = (f"{train} --run-dir $J/runs/async --primerl-out {out}/async --steps {args.async_steps} --async-groups {args.verify_async_groups} "
                 f"--max-staleness {args.verify_staleness}")
    port = "$((29500 + SLURM_JOB_ID % 1000 + {n}))"

    def trn(cfg: str, n: int, tag: str) -> str:
        return trainer_command(cfg, gpus=d.train_gpus, port=port.format(n=n), tag=tag)

    head = f"""#!/bin/bash
#SBATCH --account={d.account}
#SBATCH --partition={d.partition}
#SBATCH --nodes=1
#SBATCH --time={d.time}
set -uo pipefail
unset SBATCH_ACCOUNT SLURM_ACCOUNT
{setup_lines(d, version=PRIME_RL_VERSION)}export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
T0=$SECONDS
log() {{ echo "[$(date +%H:%M:%S) +$(( (SECONDS - T0) / 60 ))m] $*"; }}
mkdir -p {out}
log "node $(hostname): $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\\n' ' ')"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 > {out}/gpu_mem_$SLURM_JOB_ID.csv 2>/dev/null &
MEM=$!
peak() {{ awk -F', ' '{{g=$2; m=$3+0; if(m>p[g])p[g]=m}} END{{for(g in p) print "gpu", g, p[g], "MiB"}}' {out}/gpu_mem_$SLURM_JOB_ID.csv | sort; }}
"""
    probe_job = head.replace("#SBATCH --nodes=1", f"#SBATCH --nodes=1\n#SBATCH --job-name=crucible-rl-probe\n#SBATCH --gpus-per-node=2\n#SBATCH --cpus-per-gpu=16\n"
                             f"#SBATCH --mem=120G\n#SBATCH --output={out}/probe-%j.out").replace(f"--time={d.time}", f"--time={args.probe_time}") + f"""
log "trainer probe: the 35B on fake data at seq_len {d.seq_len}, ep {d.ep}, cp {d.cp}"
CUDA_VISIBLE_DEVICES=0,1 {trn(f"{out}/trainer_probe.toml", 0, "probe")} > {out}/trainer_probe.log 2>&1
RC=$?
log "probe rc=$RC"
sed "s/\\x1b\\[[0-9;]*m//g" {out}/trainer_probe.log | grep -E "SUCCESS|Step [0-9]|checkpoint|Error|error|Killed|OOM|out of memory" | grep -v _pytree | tail -30
kill $MEM 2>/dev/null; peak
exit $RC
"""
    verify_job = head.replace("#SBATCH --nodes=1", f"#SBATCH --nodes=1\n#SBATCH --job-name=crucible-rl-verify\n#SBATCH --gpus-per-node=4\n"
                              f"#SBATCH --cpus-per-gpu={args.cpus_per_gpu}\n#SBATCH --mem={args.mem_gb}G\n#SBATCH --output={out}/verify-%j.out") + f"""
{world_lines(d, out=str(out), data_src=args.data_src, sif_dir=args.sif_dir, world_image=args.world_image)}
PHASES="${{PHASES:-sync resume async}}"
CUDA_VISIBLE_DEVICES=2,3 uv run inference @ {out}/inference.toml > {out}/inference_$SLURM_JOB_ID.log 2>&1 &
INF=$!
TRN=
stop_trainer() {{
  [ -n "$TRN" ] && {{ kill $TRN 2>/dev/null; wait $TRN 2>/dev/null; }}
  pkill -u $USER -f prime_rl.trainer.rl.train; TRN=
  for i in $(seq 1 60); do [ -z "$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
}}
watch_run() {{  # $1 phase, $2 the loop's command: stops the loop when its trainer or vLLM dies
  log "phase $1: start"
  eval "$2" > {out}/crucible_$1.log 2>&1 &
  CR=$!
  while kill -0 $CR 2>/dev/null; do
    kill -0 $INF 2>/dev/null || {{ log "vLLM died"; kill $CR; tail -30 {out}/inference_$SLURM_JOB_ID.log; }}
    [ -n "$TRN" ] && ! kill -0 $TRN 2>/dev/null && {{ log "trainer died"; kill $CR; sed "s/\\x1b\\[[0-9;]*m//g" $TLOG | grep -v _pytree | tail -40; }}
    sleep 15
  done
  wait $CR; local rc=$?
  sync_back
  log "phase $1: rc=$rc"; tail -5 {out}/crucible_$1.log
  return $rc
}}
finish() {{ stop_trainer; kill $INF $MEM $SYNCBACK 2>/dev/null; sync_back; log "peak GPU memory:"; peak; log "done rc=$1"; exit $1; }}
for i in $(seq 1 240); do curl -sf localhost:{d.port}/health > /dev/null && break; kill -0 $INF 2>/dev/null || {{ log "vLLM exited"; tail -40 {out}/inference_$SLURM_JOB_ID.log; finish 5; }}; sleep 10; done
curl -sf localhost:{d.port}/health > /dev/null || {{ log "vLLM not healthy after 40 min"; finish 7; }}
log "vLLM healthy"
if [[ " $PHASES " == *" sync "* ]]; then
  TLOG={out}/trainer_sync_$SLURM_JOB_ID.log
  uv run python -c "from crucible.train.bridges.primerl.deploy import launch_toml; print('trainer resume step:', launch_toml('{out}/trainer_sync.toml', '$J/runs/sync/state.json', '{out}/trainer_sync_run.toml'))" || finish 12
  CUDA_VISIBLE_DEVICES=0,1 {trn(f"{out}/trainer_sync_run.toml", 0, "sync")} > $TLOG 2>&1 &
  TRN=$!
  watch_run sync "{cmd_sync}" || finish 9
fi
if [[ " $PHASES " == *" resume "* ]]; then
  stop_trainer
  TLOG={out}/trainer_resume_$SLURM_JOB_ID.log
  uv run python -c "from crucible.train.bridges.primerl.deploy import launch_toml; print('trainer resume step:', launch_toml('{out}/trainer_sync.toml', '$J/runs/sync/state.json', '{out}/trainer_sync_run.toml'))" || finish 12
  CUDA_VISIBLE_DEVICES=0,1 {trn(f"{out}/trainer_sync_run.toml", 1, "resume")} > $TLOG 2>&1 &
  TRN=$!
  watch_run resume "{cmd_resume}" || finish 10
fi
if [[ " $PHASES " == *" async "* ]]; then
  stop_trainer
  TLOG={out}/trainer_async_$SLURM_JOB_ID.log
  uv run python -c "from crucible.train.bridges.primerl.deploy import launch_toml; print('trainer resume step:', launch_toml('{out}/trainer_async.toml', '$J/runs/async/state.json', '{out}/trainer_async_run.toml'))" || finish 12
  CUDA_VISIBLE_DEVICES=0,1 {trn(f"{out}/trainer_async_run.toml", 2, "async")} > $TLOG 2>&1 &
  TRN=$!
  watch_run async "{cmd_async}" || finish 11
fi
finish 0
"""
    (stage / "probe.sbatch").write_text(probe_job)
    (stage / "verify.sbatch").write_text(verify_job)
    (stage / "build_env.sh").write_text(build_script(d, version=PRIME_RL_VERSION))
    print(json.dumps({"stage": str(stage), "cluster_out": str(out), "tasks": tasks,
                      "probe": " ".join(submit_command(out / "probe.sbatch", d)), "verify": " ".join(submit_command(out / "verify.sbatch", d))}, indent=1))


def _backend(args, *, lr: float, loss: str, ranks: int, seq_len: int, router_replay: bool = False, sc_topk: int = 0, timeouts: dict[str, float] | None = None,
             temperature: float = 1.0):
    """``temperature``: the loop's sampling temperature, which the trainer scores every token at (``wire``) and which
    score centering's heads must come from (``backend._Sampler``)."""
    from crucible.train.bridges.primerl.backend import FileIO, HttpClient, PrimeRLBackend
    from crucible.train.bridges.primerl.routing import RoutingStore
    from crucible.train.bridges.primerl.topk import TopKStore

    t = timeouts or {}
    io = FileIO(Path(args.primerl_out), ranks=ranks, seq_len=seq_len, model=args.base_model)
    if "step_timeout_s" in t:
        io.send_timeout_s = t["step_timeout_s"]
    return PrimeRLBackend(io=io, http=HttpClient(f"http://localhost:{args.port}"), base_model=args.base_model, learning_rate=lr, loss_kind=loss,
                          routing=RoutingStore() if router_replay else None,
                          topk=TopKStore(k=sc_topk) if sc_topk > 0 else None, temperature=float(temperature), **t)


def smoke_encode(tok, text: str) -> list[int]:
    """The chat prompt's token ids: the template rendered as text, then encoded. ``apply_chat_template(tokenize=True)``
    returns a ``BatchEncoding`` in transformers 5, and ``list()`` of it is its keys (smoke attempt 9, job 3474394)."""
    msgs = [{"role": "user", "content": text}]
    try:
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    except TypeError:
        rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tok.encode(rendered, add_special_tokens=False)]


def cmd_smoke(args) -> None:
    from transformers import AutoTokenizer

    from crucible.train.bridges.primerl.smoke import run

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def encode(text: str) -> list[int]:
        return smoke_encode(tok, text)

    resume = sampler_path = None
    if args.resume_from:
        prev = json.loads(Path(args.resume_from).read_text())
        resume, sampler_path = prev["last_checkpoint"], prev["last_adapter"]
    b = _backend(args, lr=args.lr, loss="importance_sampling", ranks=1, seq_len=4096, sc_topk=args.sc_topk, timeouts=SMOKE_TIMEOUTS)
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


def debug_tasks(debug: Path, ids: list[str], *, grade_parallel: int = 2) -> list:
    """The debug lane's own tasks (``lh_debug_target.Tasks``: ``generators.debug.DebugTask`` under its authored prompt, the
    lane's World and grader), for training. Rows (``--rows``) rebuild a plain window task, which lacks the lane's
    per-language scope and sandboxes. Only training-split tasks: a held-out or benchmark task is refused here (V9), before
    the trainer refuses it again."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import lh_debug_target

    pool = lh_debug_target.Tasks(Path(debug), grade_parallel=grade_parallel)
    unknown = [t for t in ids if t not in pool.rows]
    if unknown:
        raise SystemExit(f"refused: not a gated debug-lane task in {debug}: {unknown}")
    held = [t for t in ids if pool.rows[t]["split"] != "train"]
    if held:
        raise SystemExit(f"refused: held out (V9): {held}")
    return [pool.task(t) for t in ids]


def env_kw(args, tl) -> dict:
    """``tinker_rl._envs_kw`` with the Phi checkpoint's cap from ``--ckpt-cap`` (the debug lane's budget: every 20, cap 15)."""
    from crucible.factory.checkpoint import CheckpointConfig

    kw = tl._envs_kw(args)
    if getattr(args, "ckpt_cap", None):
        kw["phi_checkpoint"] = CheckpointConfig(every=args.ckpt_every, cap=args.ckpt_cap, concurrency=1)
    return kw


def cmd_train(args) -> None:
    from crucible.factory.tinker import train
    from crucible.train.bridges.primerl.backend import local_model

    tl = _tinker_cli()
    cfg = tl.train_config(args)
    d = delta_config(args)
    tasks = debug_tasks(Path(args.debug), args.tasks, grade_parallel=args.grade_parallel) if args.debug else tl._tasks_from_rows(args)
    targeting = json.loads(Path(args.targeting).read_text()) if args.targeting else None
    b = _backend(args, lr=cfg.learning_rate, loss=cfg.loss_fn, ranks=d.data_ranks, seq_len=d.seq_len, router_replay=d.router_replay,
                 sc_topk=d.sc_topk if d.centered else 0, temperature=cfg.temperature)
    try:
        out = asyncio.run(train(tasks, config=cfg, run_dir=Path(args.run_dir), targeting=targeting, observe=tl._observe(args), backend=b,
                                model=local_model(cfg.model, context=d.seq_len), **env_kw(args, tl)))
    finally:
        if args.debug:
            import lh_debug_target

            lh_debug_target.cleanup()  # the lane's Worlds and graders, by their name prefix, and nothing else
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
    p.add_argument("--resume-step", type=int, help="pin the prime-rl resume step (steps count from 1); by default the job derives it from Crucible's state.json")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("render-verify")
    tl.add_common(p, cap=1e9)
    tl.add_train_flags(p)
    add_deploy_flags(p)
    add_debug_flags(p)
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--sync-steps", type=int, default=2)
    p.add_argument("--resume-steps", type=int, default=1, help="steps after the trainer is relaunched from its checkpoint")
    p.add_argument("--async-steps", type=int, default=2)
    p.add_argument("--verify-async-groups", type=int, default=4)
    p.add_argument("--verify-staleness", type=int, default=1)
    p.add_argument("--data-src", default=f"{INFER}/data/debug", help="the lane's training data on the cluster (copied node-local)")
    p.add_argument("--sif-dir", default=f"{INFER}/sif")
    p.add_argument("--world-image", required=True, help="the task's World image (its SIF in --sif-dir): the lockdown probe runs in it")
    p.add_argument("--mem-gb", type=int, default=240)
    p.add_argument("--cpus-per-gpu", type=int, default=32)
    p.add_argument("--probe-time", default="00:40:00")
    p.add_argument("--keep-last", type=int, default=2, help="prime-rl checkpoints kept per run (the 35B's are ~70 GB)")
    p.set_defaults(fn=cmd_render_verify)
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
            add_debug_flags(p)
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
