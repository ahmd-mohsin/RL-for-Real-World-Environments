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

The LLM-judge items need a judge endpoint: ``judge={"key", "model", "url", "api": "chat"}`` sets MiMo's ``GA_JUDGE_*``
variables for this process before ``verify.py`` is imported (it reads them at import time). Without one, MiMo masks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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


def grade(source_env: Path, run_root: Path, *, agent_output: str = "", judge: dict | None = None) -> dict:
    """``run_root``: the World's bundled copy of the env (``<workdir>/mimo``). Returns a Crucible reward document."""
    g = Path(tempfile.mkdtemp(prefix="mimo-grade-"))
    shutil.copytree(source_env, g / "env", ignore=shutil.ignore_patterns("__pycache__", ".cache"))
    root = g / "env"
    for part in ("system", "workspace"):
        if (Path(run_root) / part).exists():
            shutil.rmtree(root / part, ignore_errors=True)
            shutil.copytree(Path(run_root) / part, root / part)
    saved = {k: os.environ.get(k) for k in ("GA_JUDGE_KEY", "GA_JUDGE_MODEL", "GA_JUDGE_URL", "GA_JUDGE_API", "GA_JUDGE_PHASE_BUDGET")}
    try:
        if judge:
            os.environ.update({"GA_JUDGE_KEY": judge["key"], "GA_JUDGE_MODEL": judge["model"], "GA_JUDGE_URL": judge.get("url") or "",
                               "GA_JUDGE_API": judge.get("api", "chat"), "GA_JUDGE_PHASE_BUDGET": str(judge.get("budget_s", 180))})
        else:
            for k in ("GA_JUDGE_KEY",):
                os.environ.pop(k, None)
        spec = importlib.util.spec_from_file_location(f"mimo_verify_{abs(hash(str(root)))}", root / "verify.py")
        mod = importlib.util.module_from_spec(spec)
        log = io.StringIO()
        try:
            # MiMo's verify.py prints judge chain details (including a key prefix): capture everything, keep it scrubbed
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                spec.loader.exec_module(mod)
                res = mod.verify(env_path=str(root), ground_truth=None, model_answer={}, pre_state={}, post_state=post_state(root / "system"),
                                 agent_output=agent_output)
        except Exception as e:  # noqa: BLE001 -- run_verify.py's own contract: a crashing verify masks, never 0
            res = {"error": f"{type(e).__name__}: {e}"}
        text = log.getvalue()
        if judge and judge.get("key"):
            text = text.replace(judge["key"], "***").replace(judge["key"][:12], "***")
        (g / "verify.log").write_text(re.sub(r"(sk-[A-Za-z0-9_-]{2})[A-Za-z0-9_.-]+", r"\1***", text))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if "error" in res:
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "verifier_error", "reason": f"verify_crashed: {res['error'][:300]}"}}
    if res.get("reward_error"):
        return {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "judge_failed", "reason": str(res["reward_error"])[:300]}}
    score = float(res.get("score", 0.0) or 0.0)
    comps = {f"check/{r.get('id', i)}": float(bool(r.get("passed"))) for i, r in enumerate(res.get("results") or [])}
    return {"reward": score, "gates": {}, "outcome": float(bool(res.get("strict_pass"))), "infra": {"status": "ok"}, "components": comps,
            "mimo": {k: res.get(k) for k in ("strict_pass", "deterministic_score", "judge_score", "judge_all_passed")}}
