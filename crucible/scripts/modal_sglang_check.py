"""Live acceptance of the SGLang M12 engine (``crucible.train.sglang_http``) against a real SGLang server on Modal.
Until now it was tested against mock servers only.

    modal run scripts/modal_sglang_check.py            # ~10 GPU minutes on an L4; prints and writes the report

The server and the check run in one container, so nothing is exposed. The checks follow the engine contract:
- **generate:** token ids plus behaviour logprobs, one per completion token, and a `stop` finish at an EOS stop id;
  temperature-0 determinism.
- **abort:** a per-call abort of a long generation returns `abort`.
- **colocated barrier:** `pause` (pause_generation + release_memory_occupation), then `resume` (resume_memory_occupation
  + continue_generation); the same greedy output after waking.
- **weight load:** `load_weights` from a checkpoint directory (the model's own snapshot, as a publisher writes it) stamps
  the new version on the next completion, and greedy output is unchanged (same weights).
"""

import json
from pathlib import Path

import modal

CRUCIBLE = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B"
image = (modal.Image.from_registry("lmsysorg/sglang:latest").entrypoint([])
         .pip_install("huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/crucible/src", "HF_HOME": "/hfcache"})  # the image's own HF cache dir is not empty
         .add_local_dir(str(CRUCIBLE / "src" / "crucible"), "/crucible/src/crucible", ignore=["**/__pycache__"]))
hf_cache = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)
app = modal.App("crucible-sglang-check")

CHECK = r'''
import asyncio, json, subprocess, sys, time
import httpx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from crucible.train.engine import GenRequest
from crucible.train.sglang_http import SGLangServerEngine

MODEL = sys.argv[1]
path = snapshot_download(MODEL)
srv = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", path, "--port", "30000", "--host", "127.0.0.1",
                        "--enable-memory-saver", "--mem-fraction-static", "0.6"], stdout=open("/tmp/sglang.log", "w"), stderr=subprocess.STDOUT)
for _ in range(180):
    try:
        if httpx.get("http://127.0.0.1:30000/health", timeout=5).status_code == 200:
            break
    except Exception:
        pass
    if srv.poll() is not None:
        print(open("/tmp/sglang.log").read()[-4000:]); sys.exit(1)
    time.sleep(5)
tok = AutoTokenizer.from_pretrained(path)
rep = {}
ids = tok.apply_chat_template([{"role": "user", "content": "Name three colours, comma separated."}], add_generation_prompt=True, enable_thinking=False)
ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)  # newer transformers return a BatchEncoding
rep["probe"] = {"prompt_tokens": len(ids), "first": ids[:4]}
eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
async def main():
    eng = SGLangServerEngine("http://127.0.0.1:30000", placement="colocated", model_path=path)
    req = lambda cid, n=64, t=0.0: GenRequest(rid="r", call_id=cid, prompt_ids=list(ids), max_tokens=n, temperature=t, stop_ids=eos)
    a = await eng.generate(req("g1"))
    b = await eng.generate(req("g2"))
    rep["generate"] = {"tokens": len(a.completion_ids), "logprobs": len(a.logprobs), "finish": a.finish, "version": a.version,
                       "text": tok.decode(a.completion_ids, skip_special_tokens=True)[:80], "greedy_deterministic": a.completion_ids == b.completion_ids,
                       "ok": len(a.completion_ids) == len(a.logprobs) > 0 and a.finish == "stop" and all(x <= 0 for x in a.logprobs)}
    long_ids = tok.apply_chat_template([{"role": "user", "content": "Count from 1 to 3000, comma separated, nothing else."}], add_generation_prompt=True, enable_thinking=False)
    long_ids = list(long_ids["input_ids"] if hasattr(long_ids, "keys") else long_ids)
    long = asyncio.create_task(eng.generate(GenRequest(rid="r", call_id="long", prompt_ids=long_ids, max_tokens=8000, temperature=0.0, stop_ids=[])))
    await asyncio.sleep(1.0)
    await eng.abort("long")
    lr = await long
    rep["abort"] = {"finish": lr.finish, "tokens": len(lr.completion_ids), "ok": lr.finish == "abort" and len(lr.completion_ids) < 4000}
    await eng.pause()
    await eng.resume()
    c = await eng.generate(req("g3"))
    rep["barrier"] = {"same_after_wake": c.completion_ids == a.completion_ids, "ok": c.completion_ids == a.completion_ids}
    await eng.pause()
    await eng.load_weights(path, 1)
    await eng.resume()
    d = await eng.generate(req("g4"))
    rep["weight_load"] = {"version": d.version, "same_output": d.completion_ids == a.completion_ids, "ok": d.version == 1 and d.completion_ids == a.completion_ids}

try:
    asyncio.run(main())
except Exception as e:
    rep["error"] = f"{type(e).__name__}: {e}"[:1500]
rep["pass"] = "error" not in rep and all(v.get("ok") for k, v in rep.items() if isinstance(v, dict) and k != "probe")
print("REPORT " + json.dumps(rep))
srv.terminate()
'''


@app.function(image=image, gpu="L4", timeout=1800, volumes={"/hfcache": hf_cache})
def check(model: str = MODEL) -> dict:
    import subprocess
    import sys

    Path("/tmp/check.py").write_text(CHECK)
    r = subprocess.run([sys.executable, "/tmp/check.py", model], capture_output=True, text=True, check=False)
    line = next((x for x in r.stdout.splitlines() if x.startswith("REPORT ")), None)
    if line is None:
        return {"pass": False, "stdout": r.stdout[-3000:], "stderr": r.stderr[-3000:]}
    import importlib.metadata as md

    return {**json.loads(line[7:]), "sglang": md.version("sglang"), "model": model}


@app.local_entrypoint()
def main(model: str = MODEL) -> None:
    rep = check.remote(model)
    out = CRUCIBLE / "docs" / "modules" / "reports" / "f3_sglang_live.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
