"""The DeltaAI trainer bridge (``train.bridges.primerl``): the losses, the wire to prime-rl, the backend's handshake, the
deployment, and the Tinker loop running unchanged over it (swarm-pipeline/14). No GPU, no prime-rl: the process boundary
(``IO``, ``Http``) is faked."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.primerl import deploy, smoke, wire
from crucible.train.bridges.primerl.backend import FileIO, PrimeRLBackend, local_model, parse_ckpt, parse_completion
from crucible.train.bridges.primerl.loss import LossOutputsMirror, crucible_loss, token_objective
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.bridges.tinker.fake import FakeTinker
from crucible.train.bridges.tinker.trainer import Group, TinkerTrainer, TrainConfig
from crucible.train.chat import ByteChatCodec

MODEL = "Qwen/Qwen3.6-35B-A3B"
ROOT = Path(__file__).resolve().parents[2]


# ---- losses -------------------------------------------------------------------------------------------------------------------------
def _t(xs, grad=False):
    return torch.tensor(xs, dtype=torch.float64, requires_grad=grad)


def test_the_three_objectives_match_their_formulas() -> None:
    lq, adv, mask = _t([-1.0, -2.0, -0.5, -3.0]), _t([1.0, -2.0, 0.5, 9.0]), torch.tensor([True, True, True, False])
    lp = _t([-0.9, -2.5, -0.5, -1.0], grad=True)
    r = torch.exp(lp.detach() - lq)[:3]
    a = adv[:3]
    is_loss, _ = token_objective(lp, lq, adv, mask, kind="importance_sampling")
    assert math.isclose(is_loss.item(), float(-(r * a).sum()), rel_tol=1e-12)
    ppo, m = token_objective(lp, lq, adv, mask, kind="ppo", ratio_low=0.8, ratio_high=1.2)
    assert math.isclose(ppo.item(), float(-torch.minimum(r * a, r.clamp(0.8, 1.2) * a).sum()), rel_tol=1e-12)
    cispo, _ = token_objective(lp, lq, adv, mask, kind="cispo", ratio_low=0.0, ratio_high=1.05)
    assert math.isclose(cispo.item(), float(-(r.clamp(0.0, 1.05) * a * lp.detach()[:3]).sum()), rel_tol=1e-12)
    assert 0.0 < m["clip/frac"].item() < 1.0  # the masked token (A = 9) is not counted anywhere
    cispo.backward()
    assert lp.grad[3].item() == 0.0  # outside the mask: no gradient
    with pytest.raises(ValueError):
        token_objective(lp, lq, adv, mask, kind="grpo")


def test_on_policy_importance_sampling_is_minus_the_advantage_sum_and_the_anchor_pulls_every_sampled_token() -> None:
    lq = _t([-1.0, -2.0, -0.5])
    lp = _t([-1.0, -2.0, -0.5], grad=True)
    adv, mask = _t([0.0, 0.0, 0.0]), torch.tensor([True, True, True])
    loss, m = token_objective(lp, lq, adv, mask, kind="importance_sampling", kl_coeff=0.1)
    loss.backward()
    assert loss.item() == pytest.approx(0.0) and m["ratio/mean"].item() == pytest.approx(1.0) and m["kl/mismatch"].item() == pytest.approx(0.0)
    assert torch.allclose(lp.grad, torch.full((3,), -0.1, dtype=torch.float64))  # -β ∇log p: raises each sampled token, advantage 0


def test_the_prime_rl_entry_point_reads_loss_inputs_and_weights() -> None:
    inp = SimpleNamespace(trainer_logprobs=_t([-1.0, -1.0], grad=True), inference_logprobs=_t([-1.0, -1.0]), advantages=_t([1.0, 1.0]),
                          loss_mask=torch.tensor([True, True]), loss_weights=_t([0.5, 0.0]), ref_logprobs=None)
    out = crucible_loss(inp, kind="importance_sampling")
    assert isinstance(out, LossOutputsMirror) and out.loss.item() == pytest.approx(-0.5) and "kl/anchor" in out.metrics


# ---- the wire -----------------------------------------------------------------------------------------------------------------------
def _datum(prompt, comp, lps, adv, rid="r0"):
    return smoke.datum(prompt, comp, lps, adv, rollout=rid, segment=0)


def test_a_datum_becomes_one_full_length_sample_and_the_token_scale_cancels_prime_rls_mean() -> None:
    d = _datum([5, 6, 7], [8, 9], [-0.1, -0.2], 0.5)
    s = wire.to_sample(d, temperature=1.0, scale=2.0)
    assert s["token_ids"] == [5, 6, 7, 8, 9] and s["mask"] == [False, False, False, True, True]
    assert s["logprobs"] == [0.0, 0.0, 0.0, -0.1, -0.2] and s["advantages"] == [0.0, 0.0, 0.0, 1.0, 1.0] and len(s["temperatures"]) == 5
    assert s["trace_id"] == "r0" and s["branch_index"] == 0
    ds = [d, _datum([1, 2], [3, 4, 5], [0.0] * 3, -1.0, "r1")]
    assert wire.undo_token_mean(ds) == 5.0
    rows = wire.samples(ds, temperature=0.7)
    # prime-rl divides the rl loss by the 5 trainable tokens: the advantages leave multiplied by 5
    assert rows[1]["advantages"][-1] == -5.0 and rows[0]["temperatures"][0] == 0.7
    grid = wire.simple_pack(rows + rows[:1], ranks=2)
    assert len(grid[0]) == len(grid[1]) == 2 and grid[1][1] == {"dummy": True}
    with pytest.raises(ValueError):
        wire.to_sample(TokenDatum([1, 2], [9, 3], [0, 0], [0, 0], [0, 1]), temperature=1.0)


# ---- the backend over a faked trainer and server ------------------------------------------------------------------------------------
class FakeHttp:
    def __init__(self):
        self.posts = []

    async def post(self, path, body, *, timeout_s):
        self.posts.append((path, body))
        if path == "/v1/completions":
            n = 3
            return {"choices": [{"logprobs": {"tokens": [f"token_id:{100 + i}" for i in range(n)], "token_logprobs": [-0.5] * n}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens_details": {"cached_tokens": 2}}}
        return {}


class FakeIO:
    """The prime-rl trainer: every batch sent is 'trained' and broadcast at the same step; checkpoints every step."""

    def __init__(self, root: Path):
        self.root, self.sent, self.waited = root, [], []

    async def send(self, step, rows):
        self.sent.append((step, [dict(r) for r in rows]))

    async def wait_broadcast(self, step, *, timeout_s):
        self.waited.append(step)
        d = self.root / "broadcasts" / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints" / f"step_{step}").mkdir(parents=True, exist_ok=True)
        return d

    def metrics(self, step):
        return {"primerl/loss/mean": -0.25}

    def checkpoint(self, step):
        d = self.root / "checkpoints" / f"step_{step}"
        return d if d.exists() else None


def _backend(tmp_path, **kw):
    return PrimeRLBackend(io=FakeIO(tmp_path), http=FakeHttp(), base_model=MODEL, learning_rate=4e-5, **kw)


def test_the_backend_handshake_steps_and_resume(tmp_path: Path) -> None:
    b = _backend(tmp_path)

    async def go():
        tr = await b.trainer(MODEL, rank=32)
        assert b.io.waited == [-1] and tr.step == 0  # the startup broadcast (policy v-1) is acknowledged first
        s = await b.sampler(MODEL, None)
        out = await s.sample([1, 2, 3], max_tokens=8, temperature=1.0, stop=[7], seed=3)
        assert out.tokens == [100, 101, 102] and out.cached_prompt_tokens == 2 and b.http.posts[-1][1]["model"] == MODEL
        assert b.http.posts[-1][1]["prompt"] == [1, 2, 3] and b.http.posts[-1][1]["stop_token_ids"] == [7]
        fb = await tr.forward_backward([_datum([1, 2], [3, 4], [-1.0, -1.0], 1.0)], loss_fn="importance_sampling", config=None)
        assert b.io.sent[0][0] == 0 and fb.metrics["primerl/step"] == 0.0 and fb.metrics["primerl/loss/mean"] == -0.25
        assert b.http.posts[-1] == ("/load_lora_adapter", {"lora_name": "crucible-s000000", "lora_path": str(tmp_path / "broadcasts" / "step_0")})
        assert await tr.optim_step(learning_rate=4e-5) == {}
        path = await tr.save_for_sampler("x", ttl_s=None)
        s2 = await b.sampler(MODEL, path)
        await s2.sample([1], max_tokens=2, temperature=0.0, stop=None, seed=None)
        assert b.http.posts[-1][1]["model"] == "crucible-s000000"
        ref = await tr.save_state("x", ttl_s=None)
        assert parse_ckpt(ref)[0] == 0
        with pytest.raises(ValueError, match="launched with loss"):
            await tr.forward_backward([], loss_fn="ppo", config=None)
        with pytest.raises(ValueError, match="launched with lr"):
            await tr.optim_step(learning_rate=1e-3)
        # resume: a new process (the trainer relaunched with resume.step = 0) continues at batch step 1
        b2 = _backend(tmp_path / "r2")
        tr2 = await b2.trainer(MODEL, rank=32, resume=ref)
        assert tr2.step == 1 and b2.io.waited == [0] and b2.http.posts[-1][1]["lora_name"] == "crucible-s000000"
        s3 = await b2.sampler(MODEL, str(tmp_path / "elsewhere" / "step_9"))  # a version this process never loaded
        assert s3.name == "crucible-step_9" and b2.http.posts[-1][0] == "/load_lora_adapter"

    asyncio.run(go())


def test_parse_completion_needs_aligned_tokens_and_logprobs() -> None:
    with pytest.raises(ValueError):
        parse_completion({"choices": [{"logprobs": {"tokens": ["token_id:1"], "token_logprobs": []}}]})
    out = parse_completion({"choices": [{"logprobs": {"tokens": ["token_id:7"], "token_logprobs": [None]}, "finish_reason": "length"}]})
    assert out.tokens == [7] and out.stop_reason == "length" and out.cached_prompt_tokens is None


def test_file_io_answers_the_broadcast_handshake_and_writes_rank_files(tmp_path: Path) -> None:
    io = FileIO(tmp_path, ranks=2, seq_len=64, model=MODEL, poll_s=0.01, simple_pack=True)

    async def trainer_side():
        d = tmp_path / "broadcasts" / "step_3"
        d.mkdir(parents=True)
        (d / ".sender_ready").touch()
        while not (d / ".receiver_ready").exists():
            await asyncio.sleep(0.01)
        (d / ".finished").touch()

    async def go():
        await io.send(3, wire.samples([_datum([1, 2], [3], [-1.0], 1.0)], temperature=1.0))
        got, _ = await asyncio.gather(io.wait_broadcast(3, timeout_s=5), trainer_side())
        return got

    got = asyncio.run(go())
    assert got == tmp_path / "broadcasts" / "step_3"
    ranks = sorted(p.name for p in (tmp_path / "batches" / "step_3").iterdir())
    assert ranks == ["rank_0.json", "rank_1.json"] and json.loads((tmp_path / "batches" / "step_3" / "rank_1.json").read_text()) == [{"dummy": True}]
    with pytest.raises(TimeoutError):
        asyncio.run(io.wait_broadcast(4, timeout_s=0.05))
    (tmp_path / "monitors" / "file").mkdir(parents=True)
    (tmp_path / "monitors" / "file" / "m.jsonl").write_text(json.dumps({"step": 3, "loss/mean": 0.5, "note": "x"}) + "\n")
    assert io.metrics(3) == {"primerl/loss/mean": 0.5} and io.checkpoint(3) is None


# ---- the Tinker loop unchanged over the bridge --------------------------------------------------------------------------------------
def _env(tid: str) -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(task_id=tid, split="train", verifier_version="v1"), run_spec=None, policy_kwargs=dict)


class TokenGroups(TinkerTrainer):
    """Groups of token-complete rollouts from a script (the real update path, ``_step``, runs)."""

    rewards: ClassVar[list[float]] = [0.0, 1.0, 0.5, 0.0]

    async def _group(self, env, gi):
        tid = str(env.task.task_id)
        recs = []
        for j, r in enumerate(self.rewards):
            turns = [{"call_id": f"c{k}", "actor": "policy", "prompt_token_ids": [1, 2, 3] if k == 0 else [4], "prompt_is_delta": k > 0,
                      "completion_token_ids": [10 + j, 20 + k], "logprobs": [-0.3, -0.4], "policy_version": self.state["version"], "advantage": 0.1 * k * r}
                     for k in range(2)]
            recs.append(RolloutRecord.model_validate({"rollout_id": f"{tid}-{gi}-{j}", "sample": f"s{j}", "task": tid, "harness": "h", "policy": "p",
                                                      "reward": r, "progress": {"phi_final": r}, "turns": turns}))
        return Group(gi, env, tid, "v1", f"g{gi}", recs, self.state["version"], 0.0)


def test_the_tinker_loop_trains_the_same_datums_on_prime_rl_as_on_tinker(tmp_path: Path) -> None:
    cfg = TrainConfig(model=MODEL, group_size=4, groups_per_step=1, steps=2, require_band=False, turn_eta=0.1, loss_scale="token_mean")
    fake = FakeTinker()
    t1 = TokenGroups([_env("a")], config=cfg, backend=fake, run_dir=tmp_path / "tinker", codec=ByteChatCodec(), objective_id="o")
    out1 = asyncio.run(t1.run())
    b = _backend(tmp_path / "primerl")
    t2 = TokenGroups([_env("a")], config=cfg, backend=b, run_dir=tmp_path / "run", codec=ByteChatCodec(), objective_id="o",
                     model=local_model(MODEL, context=262_144))
    out2 = asyncio.run(t2.run())
    assert out1["stop"] == out2["stop"] == "done" and out2["steps_done"] == 2
    assert [s for s, _ in b.io.sent] == [0, 1] and out2["spend"]["usd"] == 0.0
    fb = [c for c in fake.calls if c["op"] == "forward_backward"]
    for call, (_, rows) in zip(fb, b.io.sent, strict=True):
        assert rows == wire.samples(call["datums"], temperature=cfg.temperature)  # the same credit, token for token
    assert any(a != 0.0 for r in b.io.sent[0][1] for a in r["advantages"])
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert parse_ckpt(state["state_path"])[0] == 1 and state["sampler_path"].endswith("step_1")


def test_a_reply_cut_at_the_token_limit_is_a_limit_stop_in_the_curriculum_and_still_trains() -> None:
    """A truncated-reply stop (``max_output_tokens``) is a budget stop for the difficulty tracker (out of p̂, never a skill
    failure), and its group still trains on its scored outcome (the limit policy scores it: it is not infrastructure)."""
    from crucible.forge.difficulty import BUDGET_STOPS, DifficultyTracker
    from crucible.train.curriculum import draws_of

    recs = [RolloutRecord.model_validate({"rollout_id": f"r{j}", "sample": f"s{j}", "task": "a", "harness": "h", "policy": "p", "reward": r,
                                          "stop": stop, "progress": {"phi_final": r}})
            for j, (r, stop) in enumerate([(0.0, "max_output_tokens"), (0.0, "max_output_tokens"), (1.0, "agent_done"), (0.5, "agent_done")])]
    draws = draws_of(recs)
    assert "max_output_tokens" in BUDGET_STOPS and all(d.budget for d in draws[:2]) and not any(d.budget for d in draws[2:])
    tr = DifficultyTracker()
    tr.observe("a", 0, draws)
    row = tr.rows()[0]
    assert row["budget"] == 2 and row["p"] == pytest.approx(0.5)  # two measured draws, one pass: the truncated failures are not in p̂
    g = Group(0, _env("a"), "a", "v1", "g0", recs, 0, 0.0)
    assert not g.masked and len(g.rewards()) == 4  # scored, so the group trains


# ---- deployment ---------------------------------------------------------------------------------------------------------------------
def test_the_account_is_cis262200_only_and_shapes_are_checked() -> None:
    with pytest.raises(ValueError, match="bjhg-dtai-gh"):
        deploy.DeltaConfig(account="bfgp-dtai-gh")
    with pytest.raises(ValueError):
        deploy.DeltaConfig(train_gpus=3, infer_gpus=2)
    with pytest.raises(ValueError):
        deploy.DeltaConfig(train_gpus=2, ep=4)
    d = deploy.DeltaConfig()
    assert d.data_ranks == 1 and d.gpus == 4
    cmd = deploy.submit_command(Path("/w/job.sbatch"), d)
    assert cmd[:5] == ["env", "-u", "SBATCH_ACCOUNT", "-u", "SLURM_ACCOUNT"] and "--account=bjhg-dtai-gh" in cmd
    job = deploy.sbatch_script(d, out=Path("/w/runs/x"), run_name="x", crucible_cmd="true")
    assert "#SBATCH --account=bjhg-dtai-gh" in job and "unset SBATCH_ACCOUNT SLURM_ACCOUNT" in job and "bfgp" not in job
    assert "#SBATCH --mem=440G" in job and "CUDA_VISIBLE_DEVICES=2,3 uv run inference" in job


def test_the_trainer_config_uses_crucibles_loss_and_checkpoints_every_step(tmp_path: Path) -> None:
    cfg = TrainConfig(model=MODEL, steps=30, loss_fn="cispo", learning_rate=2e-5, lora_rank=16, require_band=False)
    t = deploy.trainer_toml(cfg, deploy.DeltaConfig(kl_coeff=0.01, pg_correction="none"), Path("/w/runs/x"), resume_step=4)  # cispo: no centering
    assert t["loss"]["import_path"] == deploy.LOSS_IMPORT and t["loss"]["kwargs"]["kind"] == "cispo" and t["loss"]["kwargs"]["kl_coeff"] == 0.01
    assert t["ckpt"]["interval"] == 1 and t["resume"] == {"step": 4} and t["optim"]["lr"] == 2e-5 and t["model"]["lora"]["rank"] == 16
    assert t["rollout_transport"]["type"] == t["weight_broadcast"]["type"] == "filesystem" and t["max_steps"] == 30
    mod, _, fn = deploy.LOSS_IMPORT.rpartition(".")
    assert getattr(importlib.import_module(mod), fn) is crucible_loss
    inf = deploy.inference_toml(cfg, deploy.DeltaConfig())
    assert inf["vllm"]["enable_lora"] and inf["vllm"]["max_lora_rank"] == 16 and inf["vllm"]["max_model_len"] == 262_144
    paths = deploy.render(cfg, deploy.DeltaConfig(pg_correction="none"), tmp_path / "stage", run_name="x", crucible_cmd="true")
    assert all(p.exists() for p in paths.values()) and json.loads(paths["deploy"].read_text())["prime_rl"] == deploy.PRIME_RL_VERSION


def test_the_plan_scales_with_the_run_and_flags_what_does_not_fit() -> None:
    d = deploy.DeltaConfig()
    small = deploy.plan(steps=10, tasks_per_step=4, group_size=8, d=d, balance=991)
    big = deploy.plan(steps=300, tasks_per_step=8, group_size=8, d=d, balance=991)
    assert small.fits and not big.fits and big.sampling_waves > small.sampling_waves and big.gpu_hours_per_step > small.gpu_hours_per_step
    assert small.gpu_hours_total == pytest.approx(10 * small.gpu_hours_per_step, rel=1e-3) and small.concurrent_episodes >= 1
    assert any("estimate" in n for n in small.notes)


def test_held_out_tasks_and_unbanded_tasks_are_refused() -> None:
    tasks = [SimpleNamespace(task_id="a", split="train"), SimpleNamespace(task_id="h", split="heldout")]
    bad = deploy.refuse_tasks(tasks, {"a": {"model": MODEL, "band": {"in_band": True}}}, MODEL)
    assert any("held out" in x for x in bad) and not any(x.startswith("a:") for x in bad)
    assert deploy.refuse_tasks(tasks[:1], None, MODEL)[0].startswith("a: no targeting record")


# ---- the smoke task and the CLI -----------------------------------------------------------------------------------------------------
def test_the_smoke_reward_has_spread_and_its_loop_runs_on_fakes(tmp_path: Path) -> None:
    assert smoke.reward("abc", "cba") == 1.0 and 0.0 < smoke.reward("abc", "cbx") < 1.0
    b = _backend(tmp_path)
    out = asyncio.run(smoke.run(b, encode=lambda s: [ord(c) % 50 for c in s[:12]], decode=lambda ids: "x", model=MODEL, steps=2, prompts=2, group=2,
                                learning_rate=4e-5))
    assert [r["trainer_step"] for r in out["steps"]] == [0, 1] and parse_ckpt(out["last_checkpoint"])[0] == 1


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_delta_rl_takes_every_tinker_train_flag_unchanged() -> None:
    import argparse

    tl = _load("tinker_rl")

    def opts(build):
        p = argparse.ArgumentParser()
        build(p)
        return {a.dest: (a.default, a.option_strings) for a in p._actions if a.dest != "help"}

    tinker = opts(lambda p: (tl.add_common(p), tl.add_train_flags(p)))
    dr = _load("delta_rl")
    delta = opts(lambda p: (tl.add_common(p, cap=1e9), tl.add_train_flags(p), dr.add_deploy_flags(p)))
    assert all(k in delta and delta[k][1] == v[1] for k, v in tinker.items())
    # no USD cap on DeltaAI (GPU hours are the cost); 8192 output tokens per call (4096 cut replies and ended episodes)
    assert {k for k, v in tinker.items() if delta[k][0] != v[0]} == {"cap", "max_tokens"} and delta["max_tokens"][0] == dr.MAX_TOKENS == 8192
    assert dr.passthrough(["--rows", "r", "--stage", "s", "--run-name", "n", "--resume-step=3"]) == ["--rows", "r", "--run-name", "n"]
    cfg = tl.train_config(argparse.Namespace(**{k: v[0] for k, v in tinker.items()}))
    assert isinstance(cfg, TrainConfig) and cfg.loss_fn == "importance_sampling" and statistics.fmean([cfg.band_lo, cfg.band_hi]) == pytest.approx(0.4)


def test_the_build_uses_a_managed_python_with_headers_and_the_job_runs_one_engine_without_a_router(tmp_path: Path) -> None:
    # smoke attempt 5 (job 3467688): the system python had no Python.h, so Triton could not build its launcher and
    # flash-linear-attention fell back to CPU; prime-rl's default vllm-router is in the disagg extra the build skips;
    # prime-rl v0.10.0 has no sdpa attention, and auto picks FA3 on the GH200, which is not installed
    import subprocess
    import sys
    import tomllib

    dr = _load("delta_rl")
    d = deploy.DeltaConfig()
    build = dr.build_script(d, version=deploy.PRIME_RL_VERSION)
    assert "UV_PYTHON_PREFERENCE=only-managed" in build and f"uv python install {dr.PYTHON}" in build and '--python "$PY"' in build
    assert build.index("Python.h") < build.index("uv sync") and build.index(f'-c "{dr.HAS_HEADERS}" || exit 17', build.index("uv sync")) < build.index("touch ")
    assert "rm -rf .venv" in build and "--extra disagg" not in build
    assert "UV_PYTHON_PREFERENCE=only-managed" in dr.setup_lines(d, version=deploy.PRIME_RL_VERSION)
    assert subprocess.run([sys.executable, "-c", dr.HAS_HEADERS], capture_output=True, check=False).returncode == 0  # this python has its headers
    cfg = TrainConfig(model=MODEL, steps=2, require_band=False)
    inf = deploy.inference_toml(cfg, d)
    assert inf["router"] == "None" and inf["vllm"]["data_parallel_size"] == 1
    with pytest.raises(ValueError, match="vllm-router"):
        deploy.inference_toml(cfg, deploy.DeltaConfig(infer_gpus=2, infer_tp=1))
    assert deploy.trainer_toml(cfg, d, Path("/w"))["model"]["attn"] == deploy.ATTN == "flash_attention_2"
    stage = tmp_path / "smoke"
    dr.cmd_render_smoke(SimpleNamespace(base_model="Qwen/Qwen3-0.6B", partition="ghx4-interactive", time="01:55:00", workdir="/w", lr=1e-4, gpus=2, mem_gb=120,
                                        run_name="smoke1", stage=str(stage), pg_correction="sc+mis", sc_topk=32))
    for name in ("trainer.toml", "trainer_resume.toml"):
        assert tomllib.loads((stage / name).read_text())["model"]["attn"] == "flash_attention_2"
    smoke_inf = tomllib.loads((stage / "inference.toml").read_text())
    assert smoke_inf["router"] == "None" and smoke_inf["vllm"]["middleware"] == [deploy.MIDDLEWARE]  # the smoke exercises the top-k path
    job = (stage / "job.sbatch").read_text()
    assert "--sc-topk 32" in job and "topk-bench" in job.split("=== done")[0]
    assert job.index("uv run inference") < job.index("kernel_check") < job.index("uv run trainer") and "exit 8" in job
