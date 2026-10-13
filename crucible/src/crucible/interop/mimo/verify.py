"""MiMo's own verifier on a Crucible World run (DEPEND on each env's ``verify.py``; research m11-m12/05 §3.7).

The World runs on a *copy* of the env (bundle), with ``verify.py`` / ``verifier_meta.json`` / ``run_verify.py`` withheld
from the policy (``importer``). Grading happens like MiMo's separate verifier container: a fresh grading root = the
source env's files (verifier included) overlaid with the run's state (``system/<app>/state.db`` and ``workspace/``), then
the env's own ``verify(env_path, post_state, agent_output)`` — exactly the call ``run_verify.py`` makes, with the same
``post_state`` construction. Outcome mapping (MiMo's masking contract → Crucible):

- ``error`` (verify crashed) → masked ``verifier_error`` / ``verify_crashed``;
- ``reward_error`` (judge unavailable) → masked ``judge_failed``;
- otherwise reward = verify()'s weighted ``score`` (Σw·s/Σw, rounded to 4 by MiMo: recorded as is), each item → a
  ``check/<id>`` channel.

The LLM-judge items need a judge endpoint: ``judge={"key", "model", "url", "api": "chat"}`` becomes MiMo's ``GA_JUDGE_*``
variables in the grading subprocess, which gets no other judge configuration. Without one, MiMo masks.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def post_state(system_dir: Path) -> dict:
    """``run_verify.build_post_state``, pointed at a directory (read-only)."""
    out: dict[str, Any] = {}
    if not system_dir.is_dir():
        return out
    for app in sorted(os.listdir(system_dir)):
        db = system_dir / app / "state.db"
        if not db.is_file():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        out[app] = {t: [dict(r) for r in con.execute(f'SELECT * FROM "{t}" LIMIT 200000')] for t in tables}
        con.close()
    return out


RUNNER = r"""
import contextlib, importlib.util, io, json, sys
root, ps, agent, out, log = sys.argv[1:6]
sys.path.insert(0, root)
sys.modules["synthesis"] = None  # MiMo prefers synthesis.office_gen when it imports: pin the env's self-contained scorer
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        spec = importlib.util.spec_from_file_location("mimo_verify", root + "/verify.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res = mod.verify(env_path=root, ground_truth=None, model_answer={}, pre_state={}, post_state=json.load(open(ps)), agent_output=open(agent).read())
except Exception as e:
    res = {"error": f"{type(e).__name__}: {e}"}
open(log, "w").write(buf.getvalue())
json.dump(res if isinstance(res, dict) else {"error": f"verify() returned {type(res).__name__}, not a dict"}, open(out, "w"), default=str)
"""

# the judge configuration MiMo's verify.py reads (``_judge_chain_sc``: KEY, KEY2, KEY3 and their MODEL/API/URL): only the
# configured judge reaches the grader (M11 audit H3 V6: ``GA_JUDGE_KEY2`` from the environment was used with judge=None)
_JUDGE_PREFIXES = ("GA_JUDGE_", "GRADE_", "VERIFY_")


def grade(source_env: Path, run_root: Path, *, agent_output: str = "", judge: dict | None = None, workdir: Path | None = None,
          timeout_s: float = 1800.0) -> dict:
    """``run_root``: the World's bundled copy of the env (``<workdir>/mimo``). Returns a Crucible reward document.

    ``verify.py`` runs in a subprocess (``python -I``), with an environment holding only the configured judge and the
    ``synthesis`` package blocked (H3 V6, V8, V21: it ran in this process, read any ``GA_JUDGE_*`` variable, let an
    importable ``synthesis`` package decide the grade, and blocked the event loop). A timeout masks."""
    import json
    import math
    import subprocess
    import sys

    own = workdir is None
    g = Path(tempfile.mkdtemp(prefix="mimo-grade-")) if own else Path(workdir)
    g.mkdir(parents=True, exist_ok=True)
    root = g / "env"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(source_env, root, ignore=shutil.ignore_patterns("__pycache__", ".cache"))
    for part in ("system", "workspace"):
        if (Path(run_root) / part).exists():
            shutil.rmtree(root / part, ignore_errors=True)
            shutil.copytree(Path(run_root) / part, root / part)
    (g / "post_state.json").write_text(json.dumps(post_state(root / "system"), default=str))
    (g / "agent_output.txt").write_text(agent_output or "")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_JUDGE_PREFIXES) and k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")}
    if judge:
        env.update({"GA_JUDGE_KEY": judge["key"], "GA_JUDGE_MODEL": judge["model"], "GA_JUDGE_URL": judge.get("url") or "",
                    "GA_JUDGE_API": judge.get("api", "chat"), "GA_JUDGE_PHASE_BUDGET": str(judge.get("budget_s", 180))})
    out, log = g / "verify_result.json", g / "verify.log"
    try:
        subprocess.run([sys.executable, "-I", "-c", RUNNER, str(root), str(g / "post_state.json"), str(g / "agent_output.txt"), str(out), str(log)],
                       cwd=str(root), env=env, capture_output=True, timeout=timeout_s, check=False)
        res = json.loads(out.read_text()) if out.exists() else {"error": "the verifier process wrote no result"}
    except subprocess.TimeoutExpired:
        res = None
    if log.exists():  # MiMo prints judge chain details (a key prefix): scrub every key that was set, and anything key-shaped
        text = log.read_text(errors="replace")
        for k in ([judge["key"]] if judge and judge.get("key") else []):
            text = text.replace(k, "***").replace(k[:12], "***")
        log.write_text(re.sub(r"(sk-[A-Za-z0-9_-]{2})[A-Za-z0-9_.-]+", r"\1***", text))
    if own:
        shutil.rmtree(root, ignore_errors=True)  # the env copy; the log and result stay beside it (H3 V9)
    where = {"grade_dir": str(g)}
    if res is None:
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "verifier_timeout", "reason": f"verify.py exceeded {timeout_s:.0f}s"}, **where}
    if not isinstance(res, dict) or "error" in res:
        err = res.get("error") if isinstance(res, dict) else res
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "verifier_error", "reason": f"verify_crashed: {str(err)[:300]}"}, **where}
    if res.get("reward_error"):
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "judge_failed", "reason": str(res["reward_error"])[:300]}, **where}
    try:
        score = float(res.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = float("nan")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:  # MiMo's runner masks these (G3 D14: 1.5 and -0.2 were scored)
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "verifier_error",
                                                                      "reason": f"verify() score {res.get('score')!r} is not a finite reward in [0, 1]"}, **where}
    detail = res.get("detail") if isinstance(res.get("detail"), dict) else {}
    comps = {f"check/{r.get('id', i)}": float(bool(r.get("passed"))) for i, r in enumerate(res.get("results") or []) if isinstance(r, dict)}
    # success is MiMo's own ``detail.passed`` (H3 V7: ``strict_pass`` does not exist, so every env read as never solved)
    return {"reward": score, "gates": {}, "outcome": float(bool(detail.get("passed"))), "infra": {"status": "ok"}, "components": comps,
            "mimo": {k: detail.get(k) for k in ("passed", "gate_failed", "by_method")}, **where}
