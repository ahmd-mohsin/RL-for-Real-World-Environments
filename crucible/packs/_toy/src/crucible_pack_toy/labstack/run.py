"""Inside the lab container: serve a small model with vLLM, point every Crucible model alias at it, run every module's
gate, the unit suite on Linux + CUDA, and the M12 GRPO gate. One JSON report: ``/work/out/labstack.json``.

    python -m crucible_pack_toy.labstack.run --model Qwen/Qwen3-4B-Instruct-2507 [--only m10,m11] [--skip m12]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

OUT = Path("/work/out")
CR = Path("/crucible")
PY = sys.executable


def lab_config(model: str, port: int) -> Path:
    cfg = OUT / "crucible.lab.toml"
    cfg.write_text(f"""data_dir = "/work/data"

[llm_providers.lab]
base_url = "http://127.0.0.1:{port}/v1"
api_key = "EMPTY"
request_defaults = {{ logprobs = true, extra_body = {{ return_token_ids = true }} }}

[models]
sim-default = "lab:{model}"
judge-default = "lab:{model}"
policy-default = "lab:{model}"
smoke-cheap = "lab:{model}"

[prices]
"{model}" = [0.0, 0.0]
""")
    return cfg


def start_vllm(model: str, port: int, mem: float, max_len: int) -> subprocess.Popen:
    log = open(OUT / "vllm_policy.log", "w")  # noqa: SIM115 -- handed to the server process for its lifetime
    cmd = ["vllm", "serve", model, "--port", str(port), "--gpu-memory-utilization", str(mem), "--max-model-len", str(max_len),
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--dtype", "bfloat16"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 1200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log = (OUT / "vllm_policy.log").read_text(errors="replace")
            if any(k in log for k in ("free memory on device", "less than desired GPU memory", "out of memory")):
                print("[labstack] vLLM could not get its memory (shared GPU busy): exit 75, the runner waits again", flush=True)
                sys.exit(75)
            raise RuntimeError("vLLM exited (see vllm_policy.log): " + log[-2000:])
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


def phase(name: str, argv: list[str], timeout: float, env: dict, result_file: Path | None = None) -> dict:
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(CR), check=False)
        rc, out = p.returncode, (p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"TIMEOUT after {timeout}s\n{(e.stdout or b'')[-4000:]!r}"
    res = None
    if result_file is not None and result_file.exists():
        try:
            res = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            res = None
    rec = {"phase": name, "rc": rc, "seconds": round(time.monotonic() - t0, 1), "result": res, "tail": out[-6000:]}
    (OUT / f"phase_{name}.log").write_text(out)
    print(f"[labstack] {name}: rc={rc} {rec['seconds']}s", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mem", type=float, default=0.55)
    ap.add_argument("--max-len", dest="max_len", type=int, default=32768)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    skip = {x for x in a.skip.split(",") if x}
    want = lambda n: (not only or n in only) and n not in skip
    env = {**os.environ, "CRUCIBLE_CONFIG": str(lab_config(a.model, a.port)), "PYTHONUNBUFFERED": "1"}
    report: dict = {"model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "phases": []}

    def save() -> None:
        (OUT / "labstack.json").write_text(json.dumps(report, indent=1, default=str))

    llm_phases = [
        ("probe", [PY, "-m", "crucible_pack_toy.labstack.phases", "probe", str(OUT / "probe.json")], 600, OUT / "probe.json"),
        ("smoke_m1_m9", [PY, "-m", "crucible_pack_toy.smoke", "100"], 3600, CR / "docs/modules/reports/smoke_cheap.json"),
        ("m2_actors", [PY, "-m", "crucible_pack_toy.m2_gate", str(OUT / "m2.json"), "2"], 3600, OUT / "m2.json"),
        ("m7_verify_golden", [PY, "-m", "crucible_pack_toy.verify"], 1800, None),
        ("m8_forge_calibrate", [PY, "-m", "crucible_pack_toy.forge.calibrate", "--model", "policy-default", "--k", "4", "--budget", "100"], 3600, None),
        ("m8_forge_world_synth", [PY, "-m", "crucible_pack_toy.forge.world_synth", "--llm", "policy-default", "--budget", "100", "--out", str(OUT / "forge")], 3600, None),
        ("m9_accept", [PY, "-m", "crucible_pack_toy.accept_gate", "--budget", "100"], 3600, None),
        ("m10_runtime", [PY, "-m", "crucible_pack_toy.labstack.phases", "m10", str(OUT / "m10.json")], 3600, OUT / "m10.json"),
        ("m11_interop_native", [PY, "-m", "crucible_pack_toy.m11.gate", "native", str(OUT / "m11")], 1800, None),
        ("m11_llm_replay", [PY, "-m", "crucible_pack_toy.labstack.phases", "m11", str(OUT / "m11_llm.json")], 1800, OUT / "m11_llm.json"),
        ("longhorizon", [PY, "-m", "crucible_pack_toy.longhorizon.gate", "--n", "40", "--budget", "32000", "--crash", "60"], 3600, None),
    ]
    run_llm = any(want(n) for n, *_ in llm_phases)
    proc = None
    try:
        if run_llm:
            proc = start_vllm(a.model, a.port, a.mem, a.max_len)
            wait_health(a.port, proc)
            for name, argv, timeout, res in llm_phases:
                if want(name):
                    report["phases"].append(phase(name, argv, timeout, env, res))
                    save()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
    if want("unit_suite"):  # Linux + CUDA: the accelerator tests that skip on the laptop run here
        report["phases"].append(phase("unit_suite", [PY, "-m", "pytest", "-q", "tests/unit", "-p", "no:cacheprovider"], 3600, env))
        save()
    if want("m12_grpo"):  # its own vLLM (Qwen3-0.6B, sleep mode) + trainer on the same GPU
        report["phases"].append(phase("m12_grpo", [PY, "-m", "crucible_pack_toy.m12.gate", "--steps", "20", "--out", str(OUT / "m12")], 5400, env,
                                      OUT / "m12" / "m12_gpu_gate.json"))
        save()
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    def verdict(p: dict) -> str:  # a gate's own "pass" field wins over its exit code (the M12 gate exits 0 on a failed gate)
        res = p.get("result")
        if p["rc"] != 0:
            return f"rc={p['rc']}"
        if isinstance(res, dict) and res.get("pass") is False:
            return "fail (gate pass=false)"
        return "ok"

    report["summary"] = {p["phase"]: verdict(p) for p in report["phases"]}
    save()
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
