"""The asynchronous-RL audit's fixes on the DeltaAI bridge (docs/research/async-rl-audit.md): every sampling call states an
untruncated distribution, the trainer is relaunched exactly where Crucible's loop resumes (stale batches and checkpoints
cleared, a re-offer after an answer answered again, a mismatched startup step refused at once), and the trainer scores at
the sampling temperature. No GPU, no prime-rl: the trainer's side of the filesystem handshake is played by the test."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.train.bridges.primerl import deploy
from crucible.train.bridges.primerl.backend import NEUTRAL_SAMPLING, FileIO, PrimeRLBackend, StartupMismatch
from crucible.train.bridges.primerl.topk import TopKStore
from crucible.train.bridges.tinker.trainer import RunRefused, TinkerTrainer, TrainConfig
from crucible.train.chat import ByteChatCodec

MODEL = "Qwen/Qwen3.6-35B-A3B"
ROOT = Path(__file__).resolve().parents[2]


class Http:
    def __init__(self):
        self.posts = []

    async def post(self, path, body, *, timeout_s):
        self.posts.append((path, body))
        if path == "/v1/completions":
            return {"choices": [{"logprobs": {"tokens": ["token_id:5"], "token_logprobs": [-0.5]}, "finish_reason": "stop"}]}
        if path == "/inference/v1/generate":
            return {"choices": [{"token_ids": [5], "logprobs": {"content": [{"logprob": -0.5}]}, "finish_reason": "stop"}]}
        return {}


# ---- 1. the sampled distribution is the full softmax the trainer scores ---------------------------------------------------------
def test_every_sampling_call_states_an_untruncated_distribution() -> None:
    """vLLM fills omitted sampling fields from the repo's generation_config.json (Qwen3: top_k 20, top_p 0.95), and prime-rl
    returns processed logprobs: an omitted top_k / top_p truncates q while the trainer scores the full softmax."""
    for topk in (None, TopKStore(k=4)):  # /v1/completions, and /inference/v1/generate (routing replay / score centering)
        b = PrimeRLBackend(io=None, http=Http(), base_model=MODEL, learning_rate=4e-5, topk=topk)  # type: ignore[arg-type]

        async def go(b=b):
            s = await b.sampler(MODEL, None)
            await s.sample([1, 2], max_tokens=4, temperature=1.0, stop=None, seed=None)
            return b.http.posts[-1]

        path, body = asyncio.run(go())
        params = body if path == "/v1/completions" else body["sampling_params"]
        assert {k: params.get(k) for k in NEUTRAL_SAMPLING} == NEUTRAL_SAMPLING and NEUTRAL_SAMPLING["top_p"] == 1.0 and NEUTRAL_SAMPLING["top_k"] in (0, -1), path
    cfg = TrainConfig(model=MODEL, require_band=False)
    inf = deploy.inference_toml(cfg, deploy.DeltaConfig())
    assert inf["vllm"]["generation_config"] == "vllm" and inf["vllm"]["max_loras"] == 1
    # an asynchronous run has the previous version's calls in flight when the next adapter loads: two slots
    acfg = TrainConfig(model=MODEL, require_band=False, async_groups=2, max_staleness=1)
    assert deploy.inference_toml(acfg, deploy.DeltaConfig())["vllm"]["max_loras"] == 2


def test_the_prefix_cache_salt_names_the_weight_version() -> None:
    from crucible.train.bridges.tinker.sampler import _EPISODE

    b = PrimeRLBackend(io=None, http=Http(), base_model=MODEL, learning_rate=4e-5, topk=TopKStore(k=2))  # type: ignore[arg-type]
    b.adapters["/o/broadcasts/step_3"] = "crucible-s000003"

    async def go():
        salts = []
        for path in (None, "/o/broadcasts/step_3"):
            s = await b.sampler(MODEL, path)
            tok = _EPISODE.set("ep")
            try:
                await s.sample([1], max_tokens=1, temperature=1.0, stop=None, seed=None)
            finally:
                _EPISODE.reset(tok)
            salts.append(b.http.posts[-1][1]["cache_salt"])
        return salts

    base, v3 = asyncio.run(go())
    assert base != v3 and v3.endswith("crucible-s000003") and base.startswith("crucible-ep-")


def test_two_calls_with_the_same_tokens_keep_their_own_routes(tmp_path: Path) -> None:
    """Two group members (or one task in two groups on different adapter versions) return the same short completion.
    One entry per sha(prompt + completion) let the second call overwrite the first's routes, so the first sequence
    replayed another call's experts, and training it dropped the key the second still needed (coverage 0: no replay)."""
    import base64

    import numpy as np

    from crucible.train.bridges.primerl import smoke
    from crucible.train.bridges.primerl.routing import RoutingStore

    class TwoCalls:
        def __init__(self):
            self.i = 0

        async def post(self, path, body, *, timeout_s):
            if path != "/inference/v1/generate":
                return {}
            self.i += 1
            ex = np.full((len(body["token_ids"]) + 1, 1, 2), self.i, dtype=np.uint8)  # call i routes every token to experts (i, i)
            return {"choices": [{"token_ids": [50, 51], "finish_reason": "stop",
                                 "logprobs": {"content": [{"logprob": -0.1 * self.i}, {"logprob": -0.2 * self.i}]},
                                 "routed_experts": {"data": base64.b64encode(ex.tobytes()).decode(), "shape": list(ex.shape), "dtype": "uint8",
                                                    "start": 0}}]}

    class IO:
        def __init__(self):
            self.sent = []

        async def send(self, step, rows):
            self.sent.append(rows)

        async def wait_broadcast(self, step, *, timeout_s):
            return tmp_path / f"b{step}"

        def metrics(self, step):
            return {}

        def checkpoint(self, step):
            return tmp_path

    store = RoutingStore()
    b = PrimeRLBackend(io=IO(), http=TwoCalls(), base_model=MODEL, learning_rate=4e-5, routing=store, metrics_timeout_s=0.0)

    async def go():
        tr = await b.trainer(MODEL, rank=32)
        s = await b.sampler(MODEL, None)
        outs = [await s.sample([1, 2, 3], max_tokens=2, temperature=1.0, stop=None, seed=None) for _ in range(2)]
        d1, d2 = (smoke.datum([1, 2, 3], o.tokens, o.logprobs, 1.0, rollout=f"r{i}", segment=0) for i, o in enumerate(outs))
        await tr.forward_backward([d1], loss_fn="importance_sampling", config=None)  # the first group trains first
        await tr.forward_backward([d2], loss_fn="importance_sampling", config=None)

    asyncio.run(go())
    (r1,), (r2,) = b.io.sent
    e1 = np.frombuffer(r1["routed_experts"]["data"], dtype=np.uint8).reshape(r1["routed_experts"]["shape"])
    assert r1["routing_coverage"] == 1.0 and set(e1[:-1].ravel()) == {1}  # call 1's own experts, not call 2's
    assert r2["routing_coverage"] == 1.0 and r2["routed_experts"] is not None  # still there after the first group trained
    e2 = np.frombuffer(r2["routed_experts"]["data"], dtype=np.uint8).reshape(r2["routed_experts"]["shape"])
    assert set(e2[:-1].ravel()) == {2} and store.calls == 0 and not store.by_key


# ---- 2. resume: the trainer relaunched where the loop resumes ----------------------------------------------------------------------
def _offer(d: Path) -> None:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / ".sender_ready").touch()


async def _trainer_side(d: Path, *, after: float = 0.05, seen: dict | None = None, out: Path | None = None) -> bool:
    """prime-rl's ``WeightSender.broadcast`` for one step: reset the dir, offer, wait for the answer, write, finish."""
    await asyncio.sleep(after)
    if seen is not None and out is not None:  # what the trainer would read once the handshake completes
        seen["batches"] = sorted(p.name for p in (out / "batches").glob("step_*")) if (out / "batches").exists() else []
    _offer(d)
    for _ in range(300):
        if (d / ".receiver_ready").exists():
            (d / ".finished").touch()
            return True
        await asyncio.sleep(0.01)
    return False


def _ckpt(out: Path, s: int) -> None:
    t = out / "checkpoints" / f"step_{s}" / "trainer"
    t.mkdir(parents=True)
    (t / ".metadata").write_bytes(b"m")


def test_a_resume_clears_the_previous_jobs_future_batches_and_checkpoints(tmp_path: Path) -> None:
    """The previous job sent batch 3 and prime-rl checkpointed step 3, but Crucible's state recorded step 2 (killed in
    between). prime-rl's receiver reads a batch file as soon as it exists, and DCP rewrites a checkpoint in place with
    .metadata last: without the clean-up the relaunched trainer trains the stale batch 3 and the loop counts the stale
    checkpoint of step 3 before the new one is written. prime-rl's own rl launcher does this (clean_future_steps)."""
    out = tmp_path / "o"
    for s in (1, 2, 3):
        _ckpt(out, s)
        (out / "batches" / f"step_{s}").mkdir(parents=True)
        (out / "batches" / f"step_{s}" / "rank_0.bin").write_bytes(b"old")
    io = FileIO(out, ranks=1, seq_len=64, model=MODEL, poll_s=0.01, simple_pack=True)
    b = PrimeRLBackend(io=io, http=Http(), base_model=MODEL, learning_rate=4e-5, startup_timeout_s=5)
    seen: dict = {}

    async def go():
        return await asyncio.gather(b.trainer(MODEL, rank=32, resume=f"primerl-ckpt:2:{out / 'checkpoints' / 'step_2'}"),
                                    _trainer_side(out / "broadcasts" / "step_2", seen=seen, out=out))

    tr, answered = asyncio.run(go())
    assert answered and tr.step == 3
    assert seen["batches"] == ["step_1", "step_2"]  # batch 3 was gone before the trainer could read it
    assert io.checkpoint(3) is None and io.checkpoint(2) is not None  # the stale step-3 checkpoint no longer counts
    assert list((out / "stale").glob("*/checkpoints/step_3/trainer/.metadata"))  # moved aside, never deleted


def test_a_fresh_run_refuses_an_output_directory_with_checkpoints(tmp_path: Path) -> None:
    out = tmp_path / "o"
    _ckpt(out, 4)
    io = FileIO(out, ranks=1, seq_len=64, model=MODEL)
    with pytest.raises(StartupMismatch, match="fresh Crucible run"):
        io.prepare_resume(0)
    assert FileIO(tmp_path / "clean", ranks=1, seq_len=64, model=MODEL).prepare_resume(0) == []


def test_an_answered_stale_offer_is_answered_again_when_the_trainer_re_offers(tmp_path: Path) -> None:
    """A stale unanswered offer (the dead trainer offered step 2; the loop died before answering) is answered, then the
    relaunched trainer deletes the directory and offers step 2 itself. The old wait kept polling for .finished and never
    answered the new offer: both sides waited to their timeouts."""
    io = FileIO(tmp_path, ranks=1, seq_len=64, model=MODEL, poll_s=0.01)
    d = tmp_path / "broadcasts" / "step_2"
    _offer(d)  # stale

    async def go():
        return await asyncio.gather(io.wait_broadcast(2, timeout_s=3), _trainer_side(d, after=0.2))

    got, answered = asyncio.run(go())
    assert got == d and answered and (d / ".finished").exists()


def test_a_trainer_offering_another_startup_step_fails_the_wait_at_once(tmp_path: Path) -> None:
    io = FileIO(tmp_path, ranks=1, seq_len=64, model=MODEL, poll_s=0.01)

    async def go():
        async def wrong():
            await asyncio.sleep(0.05)
            _offer(tmp_path / "broadcasts" / "step_3")  # relaunched at step 3; the loop's state says 2

        return await asyncio.gather(io.wait_startup(2, timeout_s=30), wrong(), return_exceptions=True)

    res = asyncio.run(asyncio.wait_for(go(), 5))
    assert isinstance(res[0], StartupMismatch) and "step 3" in str(res[0])


def test_the_job_derives_the_trainers_resume_step_from_crucibles_state(tmp_path: Path) -> None:
    cfg = TrainConfig(model=MODEL, steps=40, require_band=False)
    base = tmp_path / "trainer.toml"
    import tomli_w

    base.write_text(tomli_w.dumps(deploy.trainer_toml(cfg, deploy.DeltaConfig(), tmp_path)))
    state = tmp_path / "crucible" / "state.json"
    run = tmp_path / "trainer_run.toml"
    assert deploy.launch_toml(base, state, run) is None and "resume" not in tomllib.loads(run.read_text())  # fresh
    state.parent.mkdir()
    state.write_text(json.dumps({"step": 7, "resume": {"step": 7, "state_path": "primerl-ckpt:7:/x"}}))
    assert deploy.launch_toml(base, state, run) == 7 and tomllib.loads(run.read_text())["resume"] == {"step": 7}
    pinned = tmp_path / "pinned.toml"
    pinned.write_text(tomli_w.dumps(deploy.trainer_toml(cfg, deploy.DeltaConfig(), tmp_path, resume_step=5)))
    with pytest.raises(ValueError, match="resumes at step 5"):
        deploy.launch_toml(pinned, state, run)
    job = deploy.sbatch_script(deploy.DeltaConfig(), out=Path("/w/runs/x"), run_name="x", crucible_cmd="true")
    assert job.index("launch_toml('/w/runs/x/trainer.toml', '/w/runs/x/crucible/state.json', '/w/runs/x/trainer_run.toml')") < job.index("torchrun")
    assert "-m prime_rl.trainer.rl.train @ /w/runs/x/trainer_run.toml" in job and "|| exit 12" in job


# ---- 3. the trainer scores at the sampling temperature ----------------------------------------------------------------------------
def test_a_backend_training_at_another_temperature_is_refused(tmp_path: Path) -> None:
    env = SimpleNamespace(task=SimpleNamespace(task_id="a", split="train", verifier_version="v1"), run_spec=None, policy_kwargs=dict)
    b = PrimeRLBackend(io=None, http=Http(), base_model=MODEL, learning_rate=4e-5, temperature=1.0)  # type: ignore[arg-type]
    t = TinkerTrainer([env], config=TrainConfig(model=MODEL, require_band=False, temperature=0.7), backend=b, run_dir=tmp_path,
                      codec=ByteChatCodec(), objective_id="o")
    with pytest.raises(RunRefused, match="temperature"):
        asyncio.run(t.run())
    spec = importlib.util.spec_from_file_location("delta_rl", ROOT / "scripts" / "delta_rl.py")
    dr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dr)
    nb = dr._backend(SimpleNamespace(primerl_out=str(tmp_path), base_model=MODEL, port=8000), lr=1e-4, loss="importance_sampling", ranks=1,
                     seq_len=64, temperature=0.7)
    assert nb.temperature == 0.7
