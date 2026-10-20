"""The student (factory/student.py; docs/research/swarm-pipeline/08): the registry entry, the provider seam a Tinker sampler
plugs into, the explicit thinking setting, backend metadata from the serve script's descriptor, and the dev-only rule
(MLX numerics and the retired Qwen3-4B stand-in never reach an S15 verdict or a benchmark report). No network, no GPU."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from crucible.factory.models import parse_spec
from crucible.factory.student import (
    STUDENT_MODEL,
    DevOnlyNumerics,
    StudentServer,
    assert_countable,
    is_dev_policy,
    numerics_of,
    probe,
    student_provider,
    student_spec,
    thinking_of,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_student_is_qwen35_4b_with_a_local_provider_and_thinking_off() -> None:
    assert STUDENT_MODEL == "Qwen/Qwen3.5-4B"
    assert student_spec().family == "qwen" == parse_spec("student:Qwen/Qwen3.5-4B").family
    assert parse_spec("student:Qwen/Qwen3.5-4B").provider == "student"
    prov = tomllib.loads((ROOT / "crucible.toml").read_text())["llm_providers"]["student"]
    assert prov["base_url"] == "http://127.0.0.1:8100/v1" and prov["api_key"] == "EMPTY"
    assert thinking_of(prov["request_defaults"]) is False  # an explicit, recorded setting


def test_provider_seam_local_tinker_and_tito() -> None:
    local = student_provider()
    assert thinking_of(local) is False and not local.tito
    tk = student_provider("tinker_openai", base_url="https://tinker.invalid/v1", api_key="env:TINKER_API_KEY", thinking=True)
    assert thinking_of(tk) is True and tk.api_key == "env:TINKER_API_KEY"
    tito = student_provider("tito", policy_version=0)
    assert tito.tito and tito.template_kwargs == {"enable_thinking": False}
    with pytest.raises(ValueError):
        student_provider("tinker_openai", api_key="literal")
    with pytest.raises(ValueError):
        student_provider("other")
    assert thinking_of({}) is None


def test_probe_reads_the_descriptor(tmp_path: Path) -> None:
    (tmp_path / "student-8100.json").write_text(json.dumps({"backend": "mlx", "model": STUDENT_MODEL, "dtype": "bfloat16", "engine_version": "0.32.2",
                                                             "base_url": "http://127.0.0.1:8100/v1", "pid": 1, "tool_parser": "qwen3_coder",
                                                             "thinking": False}))
    s = probe(root=tmp_path, check=False)
    assert s is not None and s.backend == "mlx" and s.dev_only and s.numerics == "dev" and s.thinking is False
    assert s.meta()["numerics"] == "dev" and s.meta()["tool_parser"] == "qwen3_coder"
    none = probe("http://127.0.0.1:8199/v1", root=tmp_path, check=False)
    assert none is not None and none.backend == "unknown" and none.dev_only  # unrecorded numerics never count
    assert StudentServer("vllm", STUDENT_MODEL, "u").numerics == "dev"  # local serving is calibration, not Tinker's sampler
    assert StudentServer("tinker", STUDENT_MODEL, "u").numerics == "final"
    assert StudentServer("tinker", "Qwen/Qwen3-4B-Instruct-2507", "u").dev_only  # the stand-in is dev whatever served it


@pytest.mark.parametrize(("policy", "dev"), [
    ("solver:student:Qwen/Qwen3.5-4B@mlx", True), ("solver:student:Qwen/Qwen3.5-4B@nudges6-mlx", True),
    ("solver:student:Qwen/Qwen3.5-4B@vllm", True), ("solver:student:Qwen/Qwen3.5-4B@tinker", False),
    ("solver:student:Qwen/Qwen3-4B-Instruct-2507@tinker", True), ("solver:qwen/qwen3.7-flash@route", False),
    ("solver:student:Qwen/Qwen3.5-4B@unknown", True), ("", False)])
def test_dev_policy_stamps(policy: str, dev: bool) -> None:
    assert is_dev_policy(policy) is dev and numerics_of(policy) == ("dev" if dev else "final")


def test_dev_numerics_never_count() -> None:
    from crucible.factory.benchmark import report
    from crucible.factory.targeting import TargetEvidence, target_verdict

    with pytest.raises(DevOnlyNumerics):
        assert_countable("solver:student:Qwen/Qwen3.5-4B@mlx", "S15")
    with pytest.raises(DevOnlyNumerics):
        assert_countable("solver:student:Qwen/Qwen3.5-4B@vllm", "S15")
    assert_countable("solver:student:Qwen/Qwen3.5-4B@tinker", "S15")
    v = target_verdict(TargetEvidence(task="t", band={"in_band": True}, numerics="dev"))
    assert v["ok"] is False and v["dev_only"] is True and v["route"] is None and "dev-only" in v["flags"][0]
    assert target_verdict(TargetEvidence(task="t")).get("dev_only") is None
    with pytest.raises(DevOnlyNumerics):
        report([], category_of=str, policy="solver:student:Qwen/Qwen3.5-4B@mlx")
    assert "overall" in report([], category_of=str, policy="solver:student:Qwen/Qwen3.5-4B@tinker")


def test_a_sampler_registers_as_a_student_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Tinker bridge's seam: a registered backend builds an LLMClient-shaped client and decides whether it counts."""
    import crucible.factory.student as st

    monkeypatch.setattr(st, "_BACKENDS", {})
    made = []
    st.register_backend("tinker", lambda model, **kw: made.append((model, kw)) or "client", final=True, doc="TinkerSampler")
    st.register_backend("cpu", lambda model, **kw: "cpu-client")
    assert st.backend("tinker").make(STUDENT_MODEL, thinking=False) == "client" and made == [(STUDENT_MODEL, {"thinking": False})]
    assert st.is_final_backend("tinker") and not st.is_final_backend("cpu") and not st.is_final_backend("vllm")
    assert is_dev_policy("solver:student:Qwen/Qwen3.5-4B@cpu") and not is_dev_policy("solver:student:Qwen/Qwen3.5-4B@tinker")
    assert sorted(st.backends()) == ["cpu", "tinker"]
    with pytest.raises(KeyError):
        st.backend("nope")
    with pytest.raises(ValueError):
        st.register_backend("a-b", lambda m: None)


def test_serve_script_parses_and_documents_both_backends() -> None:
    sh = ROOT / "scripts" / "student_serve.sh"
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    out = subprocess.run(["bash", str(sh), "--help"], capture_output=True, text=True).stdout
    assert "Qwen/Qwen3.5-4B" in out and "--thinking" in out and "DEV ONLY" in out and "--tool-call-parser" in out
    text = sh.read_text()
    assert "VLLM_USE_FLASHINFER_SAMPLER" in text and "--enable-auto-tool-choice" in text and "mlx_lm.server" in text


def _lh_run():
    import importlib.util

    spec = importlib.util.spec_from_file_location("lh_run_sb", ROOT / "scripts" / "lh_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lh_run_sb"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_lh_run_policy_ref_declares_its_config_and_stamps_the_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The policy carries its configuration (the Runner refuses an undeclared PolicyRef, M10 R-4), the student's backend in
    its id, the thinking setting in its config; a non-student rung is unchanged."""
    import types

    from crucible.core.config import Config, LLMProvider

    lh = _lh_run()
    import crucible.factory.student as st

    monkeypatch.setattr(st, "probe", lambda base_url, **kw: StudentServer("mlx", STUDENT_MODEL, base_url, "bfloat16", "0.32.2", "qwen3_coder", 1))
    cfg = Config(llm_providers={"student": student_provider(), "openrouter": LLMProvider(base_url="https://x.invalid/v1", api_key="EMPTY")})

    class Client:
        spend = types.SimpleNamespace(usd=0.0, uncertain_usd=0.0)

        def lock_settings(self) -> dict:
            return {"model": "m"}

    run = types.SimpleNamespace(a=types.SimpleNamespace(routes=False), book=types.SimpleNamespace(config=cfg),
                                solver_client=lambda model, **kw: Client())
    ref = lh.policy_ref(run, "student:Qwen/Qwen3.5-4B", lh.EpisodeCfg())
    assert ref.id == "solver:student:Qwen/Qwen3.5-4B@mlx" and is_dev_policy(ref.id)
    assert ref.config["student"]["backend"] == "mlx" and ref.config["student"]["thinking"] is False and ref.config["max_tokens"] == 4096
    assert ref.identity_config()["config"]["model"] == "student:Qwen/Qwen3.5-4B"
    other = lh.policy_ref(run, "qwen/qwen3.7-flash@route", lh.EpisodeCfg())
    assert other.id == "solver:qwen/qwen3.7-flash@route" and other.config["student"] is None and other.config["max_tokens"] == 8000
    bad = types.SimpleNamespace(a=run.a, book=types.SimpleNamespace(config=Config(llm_providers={"student": LLMProvider(base_url="http://127.0.0.1:8100/v1")})),
                                solver_client=run.solver_client)
    with pytest.raises(SystemExit, match="enable_thinking"):
        lh.policy_ref(bad, "student:Qwen/Qwen3.5-4B", lh.EpisodeCfg())
