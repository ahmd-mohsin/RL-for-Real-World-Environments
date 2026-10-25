"""DeltaAI inference path (doc 14-deltaai): the vLLM sampler backend, the server's command line, the Apptainer docker
shim's lockdown, and the World lockdown verdict."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from crucible.actors.llm import ContextOverflow, LLMError
from crucible.train.bridges.vllm.client import VLLMBackend, adapter_name, parse_choice, vllm_model
from crucible.train.bridges.vllm.serve import VLLM_SIF, ServeSpec
from crucible.world import lockdown

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.6-35B-A3B"


def shim():
    spec = importlib.util.spec_from_file_location("docker_shim", ROOT / "scripts" / "deltaai" / "docker_shim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the model row and the backend ----------------------------------------------------------------------------------------
def test_vllm_model_is_the_tinker_rows_renderer_at_no_price():
    m = vllm_model(MODEL)
    assert m.provider == "vllm" and m.ref == f"vllm:{MODEL}" and m.name == MODEL and m.context == 262_144
    assert m.tool_format == "qwen_xml" and m.family == "qwen" and m.thinking_kw == "enable_thinking" and m.moe
    assert m.prices.usd(prefill=10**6, sample=10**6) == 0.0
    assert m.spec() == {"provider": "vllm", "model": MODEL, "family": "qwen"}
    with pytest.raises(KeyError):
        vllm_model("nobody/unknown-model")


def test_parse_choice_reads_token_ids_logprobs_finish_and_cache():
    doc = {"choices": [{"logprobs": {"tokens": ["token_id:11", "token_id:42"], "token_logprobs": [-0.5, -0.25]}, "finish_reason": "length"}],
           "usage": {"prompt_tokens_details": {"cached_tokens": 7}}}
    out = parse_choice(doc)
    assert out.tokens == [11, 42] and out.logprobs == [-0.5, -0.25] and out.stop_reason == "length" and out.cached_prompt_tokens == 7
    with pytest.raises(LLMError, match="token_ids"):
        parse_choice({"choices": [{"logprobs": {"tokens": ["hello"], "token_logprobs": [-1.0]}}]})  # text, not ids


def _server(calls):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content or b"{}")
        calls.append((req.url.path, body))
        if req.url.path == "/v1/models":
            names = [MODEL] + [b["lora_name"] for p, b in calls if p == "/v1/load_lora_adapter"]
            return httpx.Response(200, json={"data": [{"id": n} for n in names]})
        if req.url.path == "/v1/load_lora_adapter":
            return httpx.Response(200, text="Success")
        if req.url.path == "/v1/completions":
            if len(body["prompt"]) > 100:
                return httpx.Response(400, json={"error": "This model's maximum context length is 100 tokens"})
            return httpx.Response(200, json={"choices": [{"logprobs": {"tokens": ["token_id:5"], "token_logprobs": [-0.1]}, "finish_reason": "stop"}]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_backend_samples_token_ids_and_loads_an_adapter_once():
    calls: list = []
    be = VLLMBackend("http://x", transport=_server(calls), sampling={"top_p": 1.0})

    async def go():
        base = await be.sampler(MODEL)
        out = await base.sample([1, 2, 3], max_tokens=8, temperature=1.0, stop=[9], seed=3)
        tuned = await be.sampler(MODEL, "/w/adapters/step-4")
        await tuned.sample([1], max_tokens=8, temperature=1.0, stop=None, seed=None)
        again = await be.sampler(MODEL, "/w/adapters/step-4")
        return out, tuned, again

    out, tuned, again = asyncio.run(go())
    assert out.tokens == [5] and out.logprobs == [-0.1]
    comp = [b for p, b in calls if p == "/v1/completions"]
    assert comp[0]["prompt"] == [1, 2, 3] and comp[0]["return_tokens_as_token_ids"] is True and comp[0]["stop_token_ids"] == [9]
    assert comp[0]["seed"] == 3 and comp[0]["model"] == MODEL and comp[0]["skip_special_tokens"] is False and comp[0]["top_p"] == 1.0
    assert comp[1]["model"] == adapter_name("/w/adapters/step-4") == tuned.served == again.served
    assert sum(1 for p, _ in calls if p == "/v1/load_lora_adapter") == 1  # the same path is one adapter


def test_a_prompt_past_the_context_is_a_context_overflow_not_a_retry():
    be = VLLMBackend("http://x", transport=_server([]))

    async def go():
        s = await be.sampler(MODEL)
        await s.sample(list(range(200)), max_tokens=8, temperature=1.0, stop=None, seed=None)

    with pytest.raises(ContextOverflow):
        asyncio.run(go())


def test_the_tinker_engine_side_runs_on_vllm_and_records_the_backend():
    from crucible.factory.student import is_final_backend, student_model_of
    from crucible.factory.tinker import policy_id
    from crucible.train.bridges.tinker.sampler import TinkerSampler
    from crucible.train.bridges.tinker.spend import SpendMeter

    m = vllm_model(MODEL)
    s = TinkerSampler(VLLMBackend("http://x", transport=_server([])), m, meter=SpendMeter(1e12), codec=object())
    r = asyncio.run(s.gen([1, 2], 8, 1.0, seed=1))
    assert r.completion_ids == [5] and s.usd == 0.0
    assert s.settings()["provider"] == "vllm"
    pol = s.policy_ref(pid=policy_id(m.name, None, m.provider))
    assert pol.id == f"solver:vllm:{MODEL}@vllm" and pol.endpoint == f"vllm:{MODEL}"
    assert student_model_of(pol.id) == MODEL and is_final_backend("vllm") and policy_id(MODEL, "/a", "vllm").endswith("@vllm-adapter")


def test_the_factory_registers_vllm_as_a_final_backend():
    from crucible.factory import models

    b = models.backend("vllm")
    assert b.final and b.name == "vllm"


def test_arm_settings_carry_the_backend_and_engine():
    from crucible.factory.tinker import arm_settings

    a = arm_settings(model=MODEL, k=8, temperature=1.0, max_tokens=4096, enable_thinking=False, tool_limit=300, timeouts=None, system=None,
                     harness="h", context=262144, codec=None, backend="vllm", engine=ServeSpec(model="/w", name=MODEL).record())
    assert a["backend"] == "vllm" and a["engine"]["engine"] == "vllm" and a["engine"]["sif_sha256"]


# ---- the server ---------------------------------------------------------------------------------------------------------------
def test_serve_spec_command_line():
    s = ServeSpec(model="/work/w/qwen", name=MODEL, tp=2, seed=7)
    a = s.vllm_argv()
    for flag in ("--served-model-name", "--max-model-len", "--tensor-parallel-size", "--enable-lora", "--generation-config", "--seed",
                 "--enable-prefix-caching"):
        assert flag in a
    assert a[a.index("--max-model-len") + 1] == "262144" and a[a.index("--tensor-parallel-size") + 1] == "2" and a[a.index("--host") + 1] == "127.0.0.1"
    full = s.apptainer_argv(binds=["/work/w/qwen"])
    assert full[:3] == ["apptainer", "exec", "--nv"] and "--cleanenv" in full and VLLM_SIF in full
    assert "/work/w/qwen:/work/w/qwen:ro" in full and "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True" in full and "HF_HUB_OFFLINE=1" in full


# ---- the Apptainer docker shim -------------------------------------------------------------------------------------------------
def test_shim_refuses_networks_privilege_and_binds_outside_the_allowed_roots(tmp_path, monkeypatch):
    d = shim()
    monkeypatch.setenv("CRUCIBLE_SHIM_ALLOW", str(tmp_path))
    with pytest.raises(d.ShimError):
        d.parse_run(["--network", "host", "img", "true"])
    with pytest.raises(d.ShimError):
        d.parse_run(["--privileged", "img"])
    o = d.parse_run(["--rm", "-v", f"{Path.home()}:/h", "img", "true"])
    with pytest.raises(d.ShimError):
        d.common_flags(o, overlay=None)
    o = d.parse_run(["--rm", "-v", "/work/hdd/x:/w:ro", "img", "true"])
    with pytest.raises(d.ShimError):
        d.common_flags(o, overlay=None)


def test_shim_run_is_locked_down(tmp_path, monkeypatch):
    d = shim()
    monkeypatch.setenv("CRUCIBLE_SHIM_ALLOW", str(tmp_path))
    monkeypatch.setenv("CRUCIBLE_SIF_DIR", str(tmp_path / "sif"))
    ws = tmp_path / "ws"
    ws.mkdir()
    o = d.parse_run(["--rm", "--name", "w1", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--memory", "2g", "--cpus", "2",
                     "-v", f"{ws}:/w:ro", "-w", "/w", "-e", "A=1,2", "--runtime=nvidia", "crucible-lh-d:go-x", "go", "test", "./..."])
    assert o["image"] == "crucible-lh-d:go-x" and o["cmd"] == ["go", "test", "./..."] and o["gpus"] and o["name"] == "w1"
    ovl = d.new_overlay("w1") if False else str(tmp_path / "o")
    a = d.run_argv(o, overlay=ovl, env_file=None)
    for f in ("--containall", "--cleanenv", "--no-home", "--fakeroot", "--nv"):
        assert f in a
    assert a[a.index("--network") + 1] == "none" and "--net" in a
    assert f"{ws.resolve()}:/w:ro" in a and f"{Path(ovl) / 'tmp'}:/tmp" in a and a[a.index("--overlay") + 1] == str(Path(ovl) / "ovl")
    assert a[a.index("--pwd") + 1] == "/w" and a[-3:] == ["go", "test", "./..."]
    assert str(d.sif_of("crucible-lh-d:go-x")).endswith("sif/crucible-lh-d_go-x.sif")
    assert "--memory" not in a  # cgroup limits only when the host takes them (CRUCIBLE_SHIM_LIMITS=1)
    monkeypatch.setenv("CRUCIBLE_SHIM_LIMITS", "1")
    assert "--memory" in d.run_argv(o, overlay=ovl, env_file=None)
    u = d.parse_run(["--rm", "--user", "1000:1000", "img", "true"])
    assert "--fakeroot" not in d.run_argv(u, overlay=None, env_file=None)


def test_shim_env_file_keeps_commas_and_inspect_formats(tmp_path, monkeypatch):
    d = shim()
    monkeypatch.setenv("CRUCIBLE_SHIM_STATE", str(tmp_path))
    p = d.write_env_file(["A=1,2", "B=x y"], [])
    assert Path(p).read_text() == 'A="1,2"\nB="x y"\n'
    assert d.fmt_value("{{.State.Running}} {{ .Id }}", {".State.Running": True, ".Id": "sha256:ab"}) == "true sha256:ab"
    assert d.exec_argv("w1", ["ls"], workdir=None, env_file=None, record={"workdir": "/w"})[-4:] == ["--pwd", "/w", "instance://w1", "ls"]


# ---- the lockdown verdict -------------------------------------------------------------------------------------------------------
GOOD = {"net_tcp_1_1_1_1_443": "blocked", "net_tcp_8_8_8_8_53": "blocked", "net_dns": "blocked", "net_ifaces": "lo,", "path:/u/me": "absent",
        "path:/u/me/.ssh": "absent", "path:/u/me/.hf_token": "absent", "path:/u": "absent", "path:/projects": "absent", "path:/work": "absent",
        "path:/sw": "absent", "path:/run/munge": "absent", "env_sensitive": "0", "dev_fabric": "0", "dev_gpu": "0", "gpu_ok": "0",
        "host_procs": "0", "mounts_shared": "0", "mounts_host_trees": "0", "tmp_writable": "yes", "probe_done": "1"}


def test_lockdown_verdict_passes_a_locked_world_and_names_each_leak():
    assert lockdown.verdict(GOOD, home="/u/me") == []
    bad = {**GOOD, "net_tcp_1_1_1_1_443": "open", "path:/u/me/.hf_token": "readable", "path:/run/munge": "exists", "env_sensitive": "2",
           "dev_fabric": "1", "host_procs": "3", "mounts_shared": "1", "net_ifaces": "lo,hsn0,"}
    probs = lockdown.verdict(bad, home="/u/me")
    for word in ("1_1_1_1_443", ".hf_token", "munge", "sensitive", "fabric", "host daemons", "shared filesystem", "hsn0"):
        assert any(word in p for p in probs), word
    assert lockdown.verdict({**GOOD, "dev_gpu": "1"}, home="/u/me") and not lockdown.verdict({**GOOD, "dev_gpu": "1", "gpu_ok": "1"}, home="/u/me")
    assert lockdown.verdict({}, home="/u/me")  # no report: refused
    with pytest.raises(lockdown.LockdownFailed):
        lockdown.require(bad, home="/u/me")
    assert lockdown.parse("a=1\nnoise\nb=x=y\n") == {"a": "1", "b": "x=y"}


def test_the_kernel_host_on_a_deltaai_node_runs_locally_and_grades_in_apptainer_mode():
    from crucible.factory.lanes import kernel_episode as ke

    h = ke.host_of("gh200-node")
    assert h.local and h.ssh_argv("echo hi") == ["bash", "-c", "echo hi"] and h.grade_mode == "apptainer" and h.runtime == "docker"
    assert h.arch == "90" and ke.host_of("gh200").runtime == "apptainer" and ke.host_of("rtx5090").ssh_argv("x")[0] == "ssh"


def test_a_held_out_task_must_have_an_offline_world():
    from types import SimpleNamespace

    from crucible.factory.lanes.windows import WindowTask
    from crucible.factory.tinker import require_offline

    assert WindowTask.world_network == "none"
    require_offline(SimpleNamespace(task_id="h", window={"split": "heldout"}, world_network="none"))
    require_offline(SimpleNamespace(task_id="t", window={"split": "train"}, world_network="bridge"))  # training: not the benchmark
    with pytest.raises(ValueError, match="offline"):
        require_offline(SimpleNamespace(task_id="h", window={"split": "heldout"}, world_network="bridge"))
