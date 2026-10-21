"""The Tinker bridge (swarm-pipeline/09) against an in-memory Tinker (``train.bridges.tinker.fake``): no network, no key.

- **Equivalence** with the slime path on the P6 scripted solvers (pass, fail, hack, partial, infra death): the native run
  through ``TinkerSampler`` (exact TITO over a byte codec) and the slime ``generate`` path give the same rewards and
  masking, and the Tinker datums carry, on every call's tokens, exactly slime's per-token advantage (``rewards`` +
  ``turn_advantages`` from ``slime.credit.convert``); the importance-sampling loss is the same number on both sides.
- **The loop**: two optimizer steps with sampler refresh (each group's turns carry the version that sampled them),
  resume from the saved state with its optimizer, whole-group drop and refill on an infrastructure death, the staleness
  filter dropping (not logging) a stale group, the cap stopping the run before any refused call is sent, the band, the
  held-out and the objective refusals.
- **The pieces**: the price table and context variants, the meter's receipts, the per-family tool parsers, the
  Qwen3.5 chat template (from the local HF cache), the infra-vs-task error split, the factory JSON client.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_p6_equiv import HostWindow, scripts
from test_m17_w_task import make_task

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMError, RequestRejected
from crucible.core.schema.rollout import RolloutRecord
from crucible.factory.checkpoint import CheckpointConfig
from crucible.factory.envs import SlimeProvider, Timeouts, make_env, objective_of, register, register_script, reset_provider, slime_rows
from crucible.train.bridges.slime import credit as sc
from crucible.train.bridges.slime.episode import ModelTurn, ScriptedSteps, step_turn
from crucible.train.bridges.slime.generate import generate
from crucible.train.bridges.slime.types import MirrorSample
from crucible.train.bridges.tinker import models as tm
from crucible.train.bridges.tinker.backend import TokenDatum, classify, with_retries
from crucible.train.bridges.tinker.codec import parse_dsml, parse_glm, parse_qwen_xml, split_thinking
from crucible.train.bridges.tinker.datums import build_datums, call_advantages, scale
from crucible.train.bridges.tinker.fake import FakeTinker, ScriptedResponder, fake_loss
from crucible.train.bridges.tinker.sampler import TinkerChatClient, TinkerSampler, check_schema, parse_json_reply
from crucible.train.bridges.tinker.spend import SpendMeter, TinkerCapReached
from crucible.train.bridges.tinker.trainer import Group, RunRefused, TinkerTrainer, TrainConfig, check_band
from crucible.train.chat import ByteChatCodec

ETA = 0.5
MODEL = "Qwen/Qwen3.6-35B-A3B"  # the build-and-train target (2026-10-20); the bridge itself is model-agnostic


# ---- the price table, the meter ---------------------------------------------------------------------------------------------
def test_registry_prices_context_and_family() -> None:
    m = tm.resolve(MODEL)
    assert (m.prices.prefill, m.prices.prefill_cached, m.prices.sample, m.prices.train) == (0.45, 0.09, 1.335, 1.177) and m.context == 65_536
    assert tm.PRICES_AS_OF == "2026-10-20" and m.tool_format == "qwen_xml" and m.family == "qwen" and m.moe
    long = tm.with_context(MODEL, 100_000)
    assert long.name == "Qwen/Qwen3.6-35B-A3B:peft:262144" and long.prices.train == 2.14 and long.prices.sample == 1.335
    assert "Qwen/Qwen3.5-4B" not in tm.MODELS  # removed from the plan (2026-10-20)
    with pytest.raises(ValueError):
        tm.with_context("Qwen/Qwen3.5-9B", 100_000)  # no 256K variant of the 9B on Tinker
    with pytest.raises(ValueError):
        tm.check_context(m, episode_tokens=65_000, max_tokens=1024)
    with pytest.raises(KeyError):
        tm.resolve("Qwen/Qwen9-1T")  # unpriced: never metered at 0
    ds = tm.resolve("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144")
    assert ds.spec() == {"provider": "tinker", "model": ds.name, "family": "deepseek"} and ds.tool_format == "dsml"
    assert tm.resolve("zai-org/GLM-5.3-Flash:peft:262144").family == "glm" and tm.resolve("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16").tool_format == "qwen_xml"
    from crucible.factory.models import parse_spec

    assert parse_spec(ds.spec()).family == "deepseek"  # the factory's ModelSpec keeps the explicit family


def test_meter_cap_holds_in_flight_calls_and_reopens(tmp_path: Path) -> None:
    m = tm.resolve(MODEL)
    meter = SpendMeter(0.01, path=tmp_path / "spend.jsonl")
    with meter.hold(0.006, "a"):
        with pytest.raises(TinkerCapReached), meter.hold(0.006, "b"):  # the first call's reservation counts
            pass
        meter.book("sample", m, prefill=1000, prefill_cached=9000, sample=2000, episode="e1")
    usd = (1000 * 0.45 + 9000 * 0.09 + 2000 * 1.335) / 1e6
    assert meter.spent == pytest.approx(usd) and meter.exhausted  # a refused call stops the run
    assert issubclass(TinkerCapReached, BudgetExceeded)  # in an episode: crucible:credits_exhausted (masked)
    again = SpendMeter(0.01, path=tmp_path / "spend.jsonl")
    assert again.spent == pytest.approx(usd) and not again.exhausted
    row = json.loads((tmp_path / "spend.jsonl").read_text().splitlines()[0])
    assert row["prefill_cached"] == 9000 and row["episode"] == "e1" and row["cum_usd"] == pytest.approx(usd)
    assert again.summary()["prices_as_of"] == tm.PRICES_AS_OF


# ---- parsers, the template, errors ----------------------------------------------------------------------------------------------
def test_family_tool_parsers() -> None:
    text = ("Let me look.\n\n<tool_call>\n<function=repo__shell>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=repo__write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\nx = 1\ny = [1, 2]\n</parameter>\n"
            "</function>\n</tool_call>")
    content, calls = parse_qwen_xml(text)
    assert content == "Let me look." and [c["name"] for c in calls] == ["repo__shell", "repo__write_file"]
    assert calls[0]["arguments"] == {"command": "ls -la"} and calls[1]["arguments"] == {"path": "a.py", "content": "x = 1\ny = [1, 2]"}
    _, bad = parse_qwen_xml("<tool_call>\n<function=f>\nstray<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>")
    assert "__unparseable__" in bad[0]["arguments"]
    _, cut = parse_qwen_xml("ok <tool_call>\n<function=repo__shell>\n<parameter=command>\nls")
    assert cut and "unterminated" in cut[0]["arguments"]["__error__"]  # a cut-off call is surfaced, never dropped
    _, hermes = parse_qwen_xml('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')
    assert hermes[0]["name"] == "f" and hermes[0]["arguments"] == {"a": 1}
    ds = ('Sure.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="repo__shell">\n<｜DSML｜ parameter name="command" string="true">ls</｜DSML｜ parameter>\n'
          '<｜DSML｜ parameter name="timeout" string="false">30</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
    c2, calls2 = parse_dsml(ds)
    assert c2 == "Sure." and calls2[0]["name"] == "repo__shell" and calls2[0]["arguments"] == {"command": "ls", "timeout": 30}
    _, g = parse_glm("<tool_call>repo__shell<arg_key>command</arg_key><arg_value>ls -la</arg_value></tool_call>")
    assert g[0]["name"] == "repo__shell" and g[0]["arguments"] == {"command": "ls -la"}
    assert split_thinking("<think>\nplan\n</think>\n\nanswer") == ("plan", "answer") and split_thinking("plain") == (None, "plain")


def _qwen_tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception:  # noqa: BLE001
        pytest.skip(f"the {MODEL} tokenizer is not in the local Hugging Face cache")


def test_qwen35_template_renders_string_arguments_and_parses_its_own_calls() -> None:
    from crucible.train.bridges.tinker.codec import TinkerChatCodec

    tok = _qwen_tok()
    codec = TinkerChatCodec(tok, tm.resolve(MODEL), enable_thinking=False)
    assert codec.settings()["template_kwargs"] == {"enable_thinking": False} and codec.thinking_behaviour() == "interleaved"
    tools = [{"type": "function", "function": {"name": "repo__shell", "description": "run", "parameters": {"type": "object",
              "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "repo__shell", "arguments": json.dumps({"command": "ls"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]
    full = tok.decode(codec.render(hist, tools))  # the raw template raises on a JSON-string argument (09 §3)
    assert "<function=repo__shell>" in full and full.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # the template's own rendering of a call parses back to the call (what the model is trained to emit)
    turn = full.split("<|im_start|>assistant\n<think>\n\n</think>\n\n")[1].split("<|im_end|>")[0]
    text, calls = codec.parse(tok.encode(turn, add_special_tokens=False) + codec.eos_ids)
    assert text == "" and calls[0]["name"] == "repo__shell" and calls[0]["arguments"] == {"command": "ls"}
    # exact TITO: the suffix after a sampled turn frames the tool result and reopens the assistant with the empty think block
    suf = tok.decode(codec.suffix([{"role": "tool", "tool_call_id": "c1", "content": "out"}], tools))
    assert suf == "\n<|im_start|>user\n<tool_response>\nout\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    on = TinkerChatCodec(tok, tm.resolve(MODEL), enable_thinking=True)
    assert tok.decode(on.render(hist[:2], tools)).endswith("<|im_start|>assistant\n<think>\n")
    assert on.parse(tok.encode("plan it\n</think>\n\ndone", add_special_tokens=False) + on.eos_ids) == ("done", [])


@pytest.mark.parametrize("name", ["deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", "zai-org/GLM-5.3-Flash:peft:262144",
                                  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "openai/gpt-oss-120b:peft:131072", "moonshotai/Kimi-K2.6:peft:131072"])
def test_each_family_frames_tito_and_parses_its_templates_own_call(name: str) -> None:
    """Every family's real template (local HF cache): the TITO suffix is computable (GLM-5.3 and gpt-oss needed their
    turn-end tokens) and our parser reads back the template's own rendering of a tool call (the text protocol for Kimi
    and gpt-oss)."""
    from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    m = tm.resolve(name)
    try:
        tok = load_tokenizer(m)
    except Exception:  # noqa: BLE001
        pytest.skip(f"the {m.base} tokenizer is not in the local Hugging Face cache")
    c = TinkerChatCodec(tok, m, enable_thinking=False)
    tools = [{"type": "function", "function": {"name": "repo__shell", "description": "run", "parameters": {"type": "object",
              "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "repo__shell", "arguments": json.dumps({"command": "ls -la"})}}]}]
    assert c.suffix([{"role": "tool", "tool_call_id": "c1", "content": "out"}], tools)
    pre, full = tok.decode(c.render(hist[:2], tools)), tok.decode(c._apply(hist, tools, gen=False))
    seg = full[full.find(pre[-40:]) + 40:] if pre[-40:] in full else full
    ids = tok.encode(seg, add_special_tokens=False)
    cut = next((i for i, t in enumerate(ids) if t in c.eos_ids or t in c.turn_end_ids), len(ids))
    _, calls = c.parse(ids[:cut] + c.eos_ids[:1])
    assert calls and calls[0]["name"] == "repo__shell" and calls[0]["arguments"] == {"command": "ls -la"}
    if not m.thinking_kw and m.off_kwargs:
        assert c.settings()["template_kwargs"] == dict(m.off_kwargs)  # no on/off switch: the lowest effort, recorded


def test_errors_split_infra_from_the_request() -> None:
    class E(Exception):
        def __init__(self, status: int | None, msg: str = "x"):
            super().__init__(msg)
            self.status_code = status

    class APIConnectionError(Exception):
        pass

    assert classify(E(503)) == classify(E(429)) == classify(APIConnectionError()) == "retry"
    assert classify(E(400, "prompt is too long: maximum context length")) == "overflow" and classify(E(400)) == "reject" and classify(E(402)) == "budget"
    assert classify(KeyError("bug")) == "raise"

    async def go() -> None:
        n = {"i": 0}

        async def flaky():
            n["i"] += 1
            if n["i"] < 3:
                raise E(503)
            return "ok"

        async def nosleep(_s: float) -> None:
            return None

        assert await with_retries("x", flaky, sleep=nosleep) == "ok" and n["i"] == 3
        for exc, kind in ((E(500), LLMError), (E(404), RequestRejected), (E(400, "context length exceeded"), ContextOverflow), (E(402), BudgetExceeded)):
            async def bad(exc=exc):
                raise exc

            with pytest.raises(kind):
                await with_retries("x", bad, attempts=2, sleep=nosleep)
        down = LLMError("x")
        assert getattr(down, "failure_kind", None) == "provider_unavailable"  # infra: masked, never scored

    asyncio.run(go())


def test_json_client_retries_and_checks_the_schema(tmp_path: Path) -> None:
    replies = iter(["not json", '{"verdict": "ok"}', '{"verdict": "ok", "score": 0.5}'])
    codec = ByteChatCodec()

    def responder(_p):
        t = next(replies)
        return list(t.encode()) + [codec.EOS]

    async def go() -> dict:
        meter = SpendMeter(1.0)
        s = await TinkerSampler(FakeTinker(responder=responder), MODEL, meter=meter, codec=codec).start()
        client = TinkerChatClient(s, json_retries=2)
        assert client.family == "qwen" and client.spec()["provider"] == "tinker"
        return await client.complete_json([{"role": "user", "content": "judge"}], schema=["verdict", "score"])

    assert asyncio.run(go()) == {"verdict": "ok", "score": 0.5}
    assert parse_json_reply("```json\n{\"a\": 1}\n```") == {"a": 1}
    with pytest.raises(ValueError):
        check_schema({"a": 1}, lambda d: "b" in d)


# ---- the equivalence run --------------------------------------------------------------------------------------------------------
class ByteScriptedModel:
    """slime's ``ScriptedModel`` emitting the byte-codec ids of exactly the completion the fake Tinker samples on the native
    path (same text + EOS), so both paths train the same number of tokens per turn."""

    def __init__(self, steps):
        self.cursor = ScriptedSteps(steps)
        self.text_of = ScriptedResponder([steps], ByteChatCodec()).text_of

    async def __call__(self, messages, sampling_params) -> ModelTurn:
        c = self.cursor
        while c.i < len(c.steps) and c.steps[c.i][0] == "chaos":
            c.steps[c.i][1]()
            c.i += 1
        c.k += 1
        if c.i >= len(c.steps):
            t, text = ModelTurn(text="done", weight_version="0"), "done"
        else:
            st = c.steps[c.i]
            c.i += 1
            t, text = step_turn(st, c.k), self.text_of(st)
        t.output_ids = list(text.encode()) + [ByteChatCodec.EOS]
        t.output_log_probs = [-0.25] * len(t.output_ids)
        return t


@pytest.fixture(scope="module")
def eq(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tinkereq")
    task, rec, _ = make_task(tmp)
    task.__class__ = HostWindow
    cfg = CheckpointConfig(every=2, cap=20, concurrency=1)
    env = make_env(task, checkpoint=cfg, timeouts=Timeouts(agent_s=300, rollout_s=900))
    sc_ = scripts(rec)
    native: dict[str, RolloutRecord] = {}

    async def run_native() -> None:
        from crucible.runtime.infra import RetryRule
        from crucible.runtime.run import Runner

        for name, steps in sc_.items():
            HostWindow._dead = False
            meter = SpendMeter(1.0)
            s = await TinkerSampler(FakeTinker(responder=ScriptedResponder([steps], ByteChatCodec())), MODEL, meter=meter, codec=ByteChatCodec()).start()
            spec = env.run_spec(run_id=f"t-{name}", policies=[s.policy_ref(pid=f"tinker-{name}", temperature=1.0)], k=1, rollout_mode="train",
                                retry=RetryRule(max_attempts=1))
            await Runner(tmp / "native").run(spec)
            adir = min((tmp / "native" / f"t-{name}" / "samples").glob("*/a*"))
            native[name] = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())

    asyncio.run(run_native())
    reset_provider(SlimeProvider(checkpoint=cfg))
    register(task)
    (row,) = slime_rows([task], agent="scripted")
    args = SimpleNamespace(crucible_agent_timeout_s=300, crucible_rollout_timeout_s=900, crucible_turn_eta=ETA, reward_key=None)
    slime: dict[str, MirrorSample] = {}

    async def run_slime() -> None:
        for i, (name, steps) in enumerate(sc_.items()):
            HostWindow._dead = False
            register_script(name, lambda steps=steps: ByteScriptedModel(steps))
            s = MirrorSample(index=i, group_index=0, prompt=row["prompt"], metadata={**row["metadata"], "script": name})
            (out,) = await generate(args, s, {})
            slime[name] = out

    asyncio.run(run_slime())
    HostWindow._dead = False
    yield SimpleNamespace(task=task, env=env, native=native, slime=slime, args=args, tmp=tmp, rec=rec)
    reset_provider(None)
    shutil.rmtree(tmp, ignore_errors=True)


def test_native_tinker_and_slime_agree_on_rewards_and_masking(eq) -> None:
    got = {n: (None if r.mask_sample else r.reward, None if eq.slime[n].remove_sample else eq.slime[n].reward) for n, r in eq.native.items()}
    assert got == {"pass": (1.0, 1.0), "fail": (0.0, 0.0), "hack": (0.0, 0.0), "partial": (pytest.approx(1 / 6, abs=1e-5), pytest.approx(1 / 6, abs=1e-5)),
                   "infra": (None, None)}
    for n, r in eq.native.items():  # every live rollout is token-complete (exact TITO through the sampler)
        if not r.mask_sample:
            ok, why = r.valid_for_training()
            assert ok, (n, why)
            assert all(t.policy_version == 0 for t in r.turns if t.completion_token_ids)


def test_tinker_datums_carry_slimes_per_token_advantages_and_loss(eq) -> None:
    names = list(eq.native)
    recs = [eq.native[n] for n in names]
    datums = build_datums(recs, ["g"] * len(recs), turn_eta=ETA)
    samples = [eq.slime[n] for n in names]
    data = sc.convert(eq.args, samples)
    by_ro = {d.meta["rollout"]: d for d in datums}
    assert eq.native["infra"].rollout_id not in by_ro  # masked: no datum
    tinker_loss = slime_loss = 0.0
    for i, n in enumerate(names):
        if eq.native[n].mask_sample:
            continue
        d = by_ro[eq.native[n].rollout_id]
        assert d.meta["seq_adv"] == pytest.approx(data["rewards"][i]), n  # outcome advantage with the honesty term composed
        spans = [tuple(t["span"]) for t in eq.slime[n].metadata["crucible"]["turn_credit"]]
        per_call = call_advantages(d)
        assert len(per_call) == len(spans), n
        for (cid, adv), (a, b) in zip(per_call.items(), spans, strict=True):
            want = [data["rewards"][i] + x for x in data["turn_advantages"][i][a:b]]
            assert adv == pytest.approx(want), (n, cid)  # token for token
        tinker_loss += fake_loss(d, d.logprobs, "importance_sampling")
        m = data["loss_masks"][i]
        slime_loss -= sum((data["rewards"][i] + t) * mk for t, mk in zip(data["turn_advantages"][i], m, strict=True))
        assert all(a == 0.0 for a, mk in zip(d.advantages, d.mask, strict=True) if not mk)  # nothing outside the policy's tokens
    assert tinker_loss == pytest.approx(slime_loss) and tinker_loss != 0.0
    tm_ = scale(datums, "token_mean")
    n_tr = sum(d.n_train for d in datums)
    assert sum(fake_loss(d, d.logprobs, "importance_sampling") for d in tm_) == pytest.approx(tinker_loss / n_tr)


# ---- the loop --------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tinkerloop")
    task, rec, _ = make_task(tmp)
    task.__class__ = HostWindow
    return SimpleNamespace(task=task, rec=rec, sc=scripts(rec), tmp=tmp)


def _trainer(toy, run_dir: Path, steps_cycle: list[str], fake: FakeTinker | None = None, **cfg) -> tuple[TinkerTrainer, FakeTinker]:
    env = make_env(toy.task, checkpoint=CheckpointConfig(every=2, cap=20, concurrency=1), timeouts=Timeouts(agent_s=300, rollout_s=900))
    codec = ByteChatCodec()
    fake = fake or FakeTinker()
    fake.responder = ScriptedResponder([toy.sc[n] for n in steps_cycle], codec)
    base = {"model": MODEL, "group_size": 2, "steps": 2, "turn_eta": ETA, "require_band": False, "cap_usd": 1.0}
    t = TinkerTrainer([env], config=TrainConfig(**{**base, **cfg}), backend=fake, run_dir=run_dir, codec=codec, objective_id=objective_of([toy.task]))
    return t, fake


def test_two_steps_refresh_the_sampler_and_resume_with_the_optimizer(toy) -> None:
    HostWindow._dead = False
    rd = toy.tmp / "run-a"
    t, fake = _trainer(toy, rd, ["pass", "fail"])
    out = asyncio.run(t.run())
    assert out["stop"] == "done" and out["steps_done"] == 2
    ops = [c["op"] for c in fake.calls]
    assert ops.count("optim_step") == 2 and ops.count("forward_backward") == 2 and ops.count("save_state") == 2
    paths = [c["path"] for c in fake.calls if c["op"] == "sampler"]
    assert paths[0] is None and paths[1:] == [out["versions"]["1"], out["versions"]["2"]]  # base, then each step's weights
    steps = [json.loads(x) for x in (rd / "steps.jsonl").read_text().splitlines()]
    assert [s["min_versions"] for s in steps] == [[0], [1]]  # step 2 trained on tokens sampled by step 1's weights
    fb = [c for c in fake.calls if c["op"] == "forward_backward"]
    assert all(c["loss_fn"] == "importance_sampling" for c in fb) and fb[0]["loss"] != 0.0
    spend = json.loads((rd / "summary.json").read_text())["spend"]
    assert spend["tokens"]["train"] == sum(c["tokens"] for c in fb) and spend["usd"] > 0
    state = json.loads((rd / "state.json").read_text())
    assert state["resume"]["step"] == 2 and state["resume"]["state_path"].endswith("state-s000002")
    rows = [json.loads(x) for x in (rd / "rewards.jsonl").read_text().splitlines()]
    assert {r["objective_id"] for r in rows} == {objective_of([toy.task])} and len(rows) == 4
    # resume: a new process (a fresh FakeTinker on the same store) continues from the saved state, optimizer included
    t2, fake2 = _trainer(toy, rd, ["pass", "fail"], fake=FakeTinker(store=fake.store), steps=3)
    out2 = asyncio.run(t2.run())
    assert out2["steps_done"] == 3
    tr = next(c for c in fake2.calls if c["op"] == "trainer")
    assert tr["resume"] == state["resume"]["state_path"]
    assert next(c for c in fake2.calls if c["op"] == "optim_step")["step"] == 3  # the optimizer-step count came back
    assert next(c for c in fake2.calls if c["op"] == "sampler")["path"] == out["versions"]["2"]
    assert SpendMeter(1.0, path=rd / "spend.jsonl").spent == pytest.approx(out2["spend"]["usd"], abs=1e-6)  # one meter across restarts
    with pytest.raises(RunRefused):  # a changed configuration is refused on resume
        asyncio.run(_trainer(toy, rd, ["pass"], fake=FakeTinker(store=fake.store), steps=4, learning_rate=1e-3)[0].run())


class FreshWindow(HostWindow):
    """A World opened after the chaos kill is alive again (the kill marks the class; a new container would not be dead)."""

    async def open(self, workdir: Path):
        HostWindow._dead = False
        return await super().open(workdir)


def test_an_infra_death_drops_the_whole_group_and_refills(toy) -> None:
    HostWindow._dead = False
    rd = toy.tmp / "run-infra"
    t, fake = _trainer(toy, rd, ["infra", "pass", "pass", "fail"], steps=1)
    toy.task.__class__ = FreshWindow
    try:
        out = asyncio.run(t.run())
    finally:
        toy.task.__class__ = HostWindow
        HostWindow._dead = False
    assert out["stop"] == "done" and out["drops"] == {"infra_in_group": 1}
    drop = json.loads((rd / "drops.jsonl").read_text().splitlines()[0])
    assert drop["reason"] == "infra_in_group" and "env_container_lost" in (drop["detail"] or "")
    trained = [d for c in fake.calls if c["op"] == "forward_backward" for d in c["datums"]]
    groups = {d.meta["group"] for d in trained}
    assert drop["group"] not in groups and len(groups) == 1  # the partial group never trained; the refill did


def test_zero_spread_groups_are_dropped_and_the_run_stops_when_it_cannot_fill(toy) -> None:
    HostWindow._dead = False
    t, fake = _trainer(toy, toy.tmp / "run-flat", ["fail"], steps=1, max_groups_per_step=2)
    out = asyncio.run(t.run())
    assert out["stop"] == "too_many_drops" and out["drops"] == {"zero_spread": 2} and not any(c["op"] == "optim_step" for c in fake.calls)


def test_a_stale_group_is_dropped_not_trained(toy, tmp_path: Path) -> None:
    t, _ = _trainer(toy, tmp_path / "run-stale", ["pass"], async_groups=1, max_staleness=1)
    t._open()
    rec = RolloutRecord.model_validate({"rollout_id": "r1", "sample": "s", "task": "t", "harness": "h", "policy": "p", "reward": 1.0, "turns": [{"call_id": "agent:c1", "actor": "agent", "completion_token_ids": [1],
                                                                                     "logprobs": [-0.1], "policy_version": 1}]})
    rec2 = rec.model_copy(update={"rollout_id": "r2", "reward": 0.0})
    g = Group(0, None, str(toy.task.task_id), str(toy.task.verifier_version or ""), "g0", [rec, rec2], 1, 0.0)
    t.state["version"] = 2
    assert t._verdict(g) is None  # 1 step behind: inside the bound
    t.state["version"] = 3
    assert t._verdict(g) == "stale"
    t._drop(g, "stale")
    assert json.loads((tmp_path / "run-stale" / "drops.jsonl").read_text())["reason"] == "stale" and t.drops == {"stale": 1}
    with pytest.raises(ValueError):
        TrainConfig(model=MODEL, async_groups=1, max_staleness=0)  # off-policy sampling needs a staleness bound


def test_the_cap_stops_the_run_before_a_refused_call_is_sent(toy) -> None:
    HostWindow._dead = False
    rd = toy.tmp / "run-cap"
    t, fake = _trainer(toy, rd, ["pass", "fail"], cap_usd=0.0005)
    out = asyncio.run(t.run())
    assert out["stop"] == "cap" and out["steps_done"] == 0 and not any(c["op"] in ("forward_backward", "optim_step") for c in fake.calls)
    assert out["spend"]["usd"] <= 0.0005 + 1e-12 and out["spend"]["refused_calls"] >= 1
    assert "infra_in_group" in out["drops"] or any(k.startswith("unused_at_stop") for k in out["drops"])


def test_band_split_and_objective_are_refused(toy, tmp_path: Path) -> None:
    env = make_env(toy.task, timeouts=Timeouts(agent_s=60, rollout_s=120))
    tid = str(toy.task.task_id)
    kw = {"backend": FakeTinker(), "codec": ByteChatCodec(), "objective_id": objective_of([toy.task])}
    with pytest.raises(RunRefused, match="no recorded band"):
        TinkerTrainer([env], config=TrainConfig(model=MODEL), run_dir=tmp_path / "a", **kw)
    assert check_band([tid], "Qwen/Qwen3.6-35B-A3B:peft:262144", {tid: {"model": "Qwen/Qwen3.6-35B-A3B", "band": {"in_band": True}}}) == []
    assert "not Qwen/Qwen3.6-35B-A3B" in check_band([tid], "Qwen/Qwen3.6-35B-A3B", {tid: {"model": "Qwen/Qwen3.5-9B", "band": {"in_band": True}}})[0]
    assert "out of band" in check_band([tid], MODEL, {tid: {"model": MODEL, "band": {"in_band": False, "rewards": [0, 0]}}})[0]
    TinkerTrainer([env], config=TrainConfig(model=MODEL), run_dir=tmp_path / "b", targeting={tid: {"model": MODEL, "band": {"in_band": True}}}, **kw)
    held = SimpleNamespace(task=SimpleNamespace(task_id="h1", verifier_version="v", split="heldout"), run_spec=None, policy_kwargs=dict)
    with pytest.raises(RunRefused, match="held-out"):
        TinkerTrainer([held], config=TrainConfig(model=MODEL, require_band=False), run_dir=tmp_path / "c", **kw)
    t = TinkerTrainer([env], config=TrainConfig(model=MODEL, require_band=False), run_dir=tmp_path / "d", **kw)
    t._open()
    with pytest.raises(RunRefused, match="objective changed"):
        TinkerTrainer([env], config=TrainConfig(model=MODEL, require_band=False), run_dir=tmp_path / "d", **{**kw, "objective_id": "obj-other"})._open()
    g = Group(0, env, tid, "v-new", "g0", [], 0, 0.0)
    assert t._verdict(g) == "objective_changed"


def test_token_datum_alignment() -> None:
    with pytest.raises(ValueError):
        TokenDatum([1, 2], [2, 3], [0.0], [0.0, 0.0], [0, 1])


# ---- the benchmark runner ------------------------------------------------------------------------------------------------------
def test_benchmark_writes_lh_metrics_with_pass_at_k_and_clustered_cis(toy, tmp_path: Path) -> None:
    from crucible.factory.tinker import benchmark

    HostWindow._dead = False
    codec = ByteChatCodec()
    fake = FakeTinker(responder=ScriptedResponder([toy.sc["pass"], toy.sc["fail"], toy.sc["partial"]], codec))
    out = asyncio.run(benchmark([toy.task], model=MODEL, k=3, run_dir=tmp_path / "bench", cap_usd=1.0, backend=fake, codec=codec,
                                timeouts=Timeouts(agent_s=300, rollout_s=900), checkpoint=None, concurrency=1))
    m = out["metrics"]
    assert m["episodes"] == 3 and m["scored"] == 3 and m["pass@1"] == pytest.approx(1 / 3)
    assert m["pass@k"]["3"] == pytest.approx(1.0) and m["pass^k"]["3"] == pytest.approx(0.0) and m["pass^3"] == pytest.approx(0.0)
    assert m["ci"]["pass@1"]["lo"] - 1e-9 <= 1 / 3 <= m["ci"]["pass@1"]["hi"] + 1e-9 and m["ci"]["units"] == 1
    assert m["usd_per_passing_episode"] == pytest.approx(out["spend"]["usd"], abs=1e-6) and out["checkpoint"] is None
    rows = [json.loads(x) for x in (tmp_path / "bench" / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["usd"] > 0 for r in rows)  # every dollar attributed to the episode that spent it
    assert sum(r["usd"] for r in rows) == pytest.approx(out["spend"]["usd"], abs=1e-5)


def test_benchmark_arms_share_one_phase_meter_and_each_reports_its_own_share(toy, tmp_path: Path) -> None:
    """S15 runs the student and the frontier arms under one hard cap: a shared ``meter`` books every arm, each arm's
    ``usd_total`` is what it spent, and the meter refuses the arm that would pass the cap (masked, never a 0)."""
    from crucible.factory.tinker import benchmark

    HostWindow._dead = False
    codec = ByteChatCodec()
    meter = SpendMeter(1.0, path=tmp_path / "phase-spend.jsonl")
    outs = []
    for i, purpose in enumerate(("selection", "fairness")):
        fake = FakeTinker(responder=ScriptedResponder([toy.sc["pass"], toy.sc["fail"]], codec))
        outs.append(asyncio.run(benchmark([toy.task], model=MODEL, k=2, run_dir=tmp_path / f"arm{i}", cap_usd=99.0, backend=fake, codec=codec, meter=meter,
                                          timeouts=Timeouts(agent_s=300, rollout_s=900), checkpoint=None, concurrency=1, seed=i, purpose=purpose)))
    shares = [o["metrics"]["usd_total"] for o in outs]
    assert all(s > 0 for s in shares) and sum(shares) == pytest.approx(meter.spent, abs=1e-5)
    assert not (tmp_path / "arm0" / "spend.jsonl").exists()  # one receipt file for the phase
    assert SpendMeter(1.0, path=tmp_path / "phase-spend.jsonl").spent == pytest.approx(meter.spent, abs=1e-6)  # a restart starts from the total
    fake = FakeTinker(responder=ScriptedResponder([toy.sc["pass"]], codec))
    out = asyncio.run(benchmark([toy.task], model=MODEL, k=1, run_dir=tmp_path / "arm-capped", cap_usd=99.0, backend=fake, codec=codec,
                                meter=SpendMeter(1e-7), timeouts=Timeouts(agent_s=300, rollout_s=900), checkpoint=None, concurrency=1, purpose="fairness"))
    assert out["metrics"]["masked"] == 1 and out["metrics"]["usd_total"] == 0.0


def test_regrade_rebuilds_the_arm_spec_and_readers_keep_the_agents_trail(toy, tmp_path: Path) -> None:
    """After a verifier fix an arm is re-graded from its stored patch and trail (no model call); the re-graded attempt holds
    only the new grade, so the trace audit, the trace export and the benchmark loop read the agent's trail from its source."""
    from crucible.factory.tinker import benchmark, regrade
    from crucible.factory.traceaudit import load_bundle, regrade_source
    from crucible.observe.traces import attempt_dirs, episode_spans, trail_dir

    HostWindow._dead = False
    codec = ByteChatCodec()
    kw = {"model": MODEL, "k": 2, "run_dir": tmp_path / "arm", "codec": codec, "timeouts": Timeouts(agent_s=300, rollout_s=900), "concurrency": 1,
          "seed": 3, "purpose": "fairness"}
    fake = FakeTinker(responder=ScriptedResponder([toy.sc["pass"], toy.sc["partial"]], codec))
    first = asyncio.run(benchmark([toy.task], cap_usd=1.0, backend=fake, checkpoint=None, **kw))
    before = [d for d in attempt_dirs(tmp_path / "arm" / "rollouts")]
    out = asyncio.run(regrade([toy.task], select=lambda r: True, **kw))
    assert out[str(toy.task.task_id)]["regraded"] == 2
    after = attempt_dirs(tmp_path / "arm" / "rollouts")
    assert [d.name for d in after] == ["a2", "a2"] and [d.name for d in before] == ["a1", "a1"]
    for d in after:
        src = regrade_source(d)
        assert src.name == "a1" and trail_dir(d) == src and not (d / "trace.json").exists()
        b = load_bundle(d, run="r")
        assert b.steps and b.id.endswith("/a2") and b.meta["regraded_from"] == "a1" and b.stop != "regraded" and b.reward is not None
        spans, row, _full = episode_spans(d, project="p")
        assert row["episode"].endswith("/a2") and any(s.op.startswith("tool_") for s in spans)
    # the arm re-read: the re-graded attempts count, with the agent attempt's rollout (same outcomes: the verifier did not change)
    again = asyncio.run(benchmark([toy.task], cap_usd=1.0, backend=FakeTinker(responder=ScriptedResponder([], codec)), checkpoint=None, **kw))
    assert again["metrics"]["episodes"] == 2 and again["metrics"]["pass@1"] == pytest.approx(first["metrics"]["pass@1"])
    rows = [json.loads(x) for x in (tmp_path / "arm" / "episodes.jsonl").read_text().splitlines()]
    assert all(r["regraded_from"] == "a1" and r["attempt"].endswith("/a2") and r["milestones"] and r["gates"] for r in rows)  # per-milestone and gates


def test_compare_pairs_control_and_trained_and_checks_the_protocol(toy, tmp_path: Path) -> None:
    from crucible.factory.tinker import check_protocol, compare

    HostWindow._dead = False
    codec = ByteChatCodec()
    path = "tinker://fake-run/sampler_weights/s000001"
    fake = FakeTinker(responder=ScriptedResponder([toy.sc["fail"], toy.sc["fail"], toy.sc["pass"], toy.sc["pass"]], codec), store={"sampler_paths": {path}})
    kw = {"backend": fake, "codec": codec, "timeouts": Timeouts(agent_s=300, rollout_s=900), "concurrency": 1}
    out = asyncio.run(compare([toy.task], model=MODEL, checkpoint=path, run_dir=tmp_path / "cmp", cap_usd=1.0, k=2, seed=7, n_boot=200, **kw))
    assert out["pass@1"] == {"control": 0.0, "trained": 1.0}  # the responder cycles: the control arm got the two fails
    d = out["paired_trained_minus_control"]
    assert d["mean_diff"] == pytest.approx(1.0) and d["n_tasks"] == 1 and out["settings"]["enable_thinking"] is False
    assert out["arms"]["trained"]["checkpoint"] == path and out["arms"]["control"]["checkpoint"] is None and out["arms"]["control"]["seed"] == 7
    assert out["protocol_problems"] == ["no baseline arm"]  # no fresh baseline was given
    a = out["arms"]
    sel = {**a["control"], "purpose": "selection", "seed": 3}
    assert check_protocol({"selection": sel, "baseline": {**a["control"], "seed": 3}, "control": a["control"], "trained": a["trained"]}) == [
        "the fresh baseline reuses the selection run's seed: it is not a fresh re-test"]
    other = {**a["trained"], "settings": {**a["trained"]["settings"], "enable_thinking": True}}
    probs = check_protocol({"baseline": {**a["control"], "seed": 4}, "control": a["control"], "trained": other})
    assert probs == ["arm trained differs in settings: ['enable_thinking']"]
    assert "the control arm is not the base model" in check_protocol({"baseline": a["control"], "control": a["trained"], "trained": a["trained"]})[0]


def test_engine_keeps_each_calls_rerender_flag_under_concurrency() -> None:
    """Regression (Tinker smoke 2026-10-20): one ``EngineLLM`` serves concurrent episodes; a re-render's ``new_segment`` flag
    was read after the ``gen`` await, so another episode's call reset it and the record saw a prefix break (masked)."""
    from crucible.train.engine import GenResult
    from crucible.train.world_env import EngineLLM

    async def gen(prompt, n, temp, **kw):
        await asyncio.sleep(0.05)
        return GenResult(completion_ids=[65, ByteChatCodec.EOS], logprobs=[-0.1, -0.1], version=0, finish="stop")

    async def go():
        eng = EngineLLM(ByteChatCodec(), gen)
        u = {"role": "user", "content": "task"}
        await eng.complete([u], sample_key="A")
        # A continues without its reply (a nudge): rendered whole, a new segment; B's first call runs during A's await
        ra, _ = await asyncio.gather(eng.complete([u, {"role": "user", "content": "nudge"}], sample_key="A"),
                                     eng.complete([{"role": "user", "content": "other"}], sample_key="B"))
        return ra

    assert asyncio.run(go()).tokens.get("new_segment") is True


def test_a_limit_stopped_qwen35_episode_stays_token_complete(toy, tmp_path: Path) -> None:
    """Regression (Tinker smoke 2026-10-20): the tool limit's harness notes are user turns, so the Qwen3.5 template
    ("interleaved") re-renders the history at the final STATUS turn; ``_final_status_turn`` ignored the engine's
    new_segment flag, so every limit-stopped episode was masked capture_incomplete (a prefix break)."""
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner
    from crucible.train.bridges.tinker.codec import TinkerChatCodec

    tok = _qwen_tok()
    codec = TinkerChatCodec(tok, tm.resolve(MODEL), enable_thinking=False)
    call = tok.encode("<tool_call>\n<function=repo__list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False)
    fake = FakeTinker(responder=lambda p: call + codec.eos_ids)
    HostWindow._dead = False
    env = make_env(toy.task, tool_limit=3, limit_warning=2, timeouts=Timeouts(agent_s=300, rollout_s=900), checkpoint=CheckpointConfig(every=2, cap=20, concurrency=1))

    async def go():
        s = await TinkerSampler(fake, MODEL, meter=SpendMeter(1.0), codec=codec).start()
        spec = env.run_spec(run_id="lim", policies=[s.policy_ref(pid="lim", **env.policy_kwargs())], k=1, rollout_mode="train", retry=RetryRule(max_attempts=1),
                            breaker=None)
        await Runner(tmp_path).run(spec)

    asyncio.run(go())
    ro = RolloutRecord.model_validate_json(next(tmp_path.glob("lim/samples/*/a1/rollout.json")).read_text())
    assert not ro.mask_sample, ro.failure_kind
    assert ro.valid_for_training()[0] and max(t.segment for t in ro.turns) >= 1  # the re-rendered final turn opened a segment


def test_factory_backend_make_serves_a_role_with_its_own_spend() -> None:
    from crucible.train.bridges.tinker.sampler import make

    codec = ByteChatCodec()
    meter = SpendMeter(1.0)
    fake = FakeTinker(responder=lambda p: list(b'<tool_call>{"name": "repo__shell", "arguments": {"command": "ls"}}</tool_call>') + [codec.EOS])
    a = make("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", role="hacker", backend=fake, codec=codec, meter=meter)
    b = make("zai-org/GLM-5.3-Flash:peft:262144", role="judge", backend=fake, codec=codec, meter=meter)
    tools = [{"type": "function", "function": {"name": "repo__shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    r = asyncio.run(a.complete([{"role": "user", "content": "go"}], tools=tools, sample_key="s1"))
    assert r.tool_calls[0]["name"] == "repo__shell" and r.cost_usd > 0
    assert a.family == "deepseek" and b.family == "glm" and a.lock_settings()["role"] == "hacker"
    assert a.spend.usd == pytest.approx(meter.spent) and b.spend.usd == 0.0  # a ModelBook summing clients never double counts


# ---- live (never in CI) -------------------------------------------------------------------------------------------------------
@pytest.mark.tinker
@pytest.mark.skipif(os.environ.get("CRUCIBLE_TINKER_LIVE") != "1" or not os.environ.get("TINKER_API_KEY"),
                    reason="live Tinker: set CRUCIBLE_TINKER_LIVE=1 with TINKER_API_KEY in the environment (paid)")
def test_live_one_tool_turn_on_the_target_model(tmp_path: Path) -> None:
    from crucible.train.bridges.tinker.backend import RealTinker
    from crucible.train.bridges.tinker.sampler import make_sampler

    async def go():
        meter = SpendMeter(0.02, path=tmp_path / "spend.jsonl")
        s = await make_sampler(RealTinker(), MODEL, meter=meter)
        tools = [{"type": "function", "function": {"name": "repo__shell", "description": "Run a shell command", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
        r = await s.llm().complete([{"role": "user", "content": "List the files in the current directory with the shell tool."}], tools=tools,
                                   max_tokens=128, temperature=0.0, sample_key="live")
        return r, meter

    r, meter = asyncio.run(go())
    assert r.tool_calls and r.tool_calls[0]["name"] == "repo__shell" and meter.spent < 0.02
