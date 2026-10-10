"""Verifier entry point inside a verifier image (standard library only).

    python3 -I /crucible_verify/run.py --plan PKG.MODULE --out /logs/verifier
        [--base DIR --patch FILE | --candidate DIR] [--param key=value ...] [--tier fast|full] [--require-root]

The bundle directory (``/crucible_verify``) holds this runtime and the task's plan module; ``-I`` keeps the
candidate's ``PYTHONPATH`` and user site out of the grader, so this file puts the bundle on ``sys.path`` itself.

``--base`` + ``--patch``: the candidate is assembled by copying the pristine base and applying the agent's patch **as
the unprivileged uid** in a tree it owns (a patch is untrusted input; applying it is the candidate's act). A patch that
does not apply is the candidate's failure: reward 0, every gate unmeasured, infra ok.

Whatever happens, ``reward.txt`` and ``reward.json`` are written; an exception masks the grade (``grader_exception``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[3]  # <bundle>/crucible/verify/runtime/run.py
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from crucible.verify.runtime.plan import Context, GradeReport, prewrite_zero, write_outputs
from crucible.verify.runtime.trusted import Trusted


def _param(kv: str):
    k, _, v = kv.partition("=")
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def assemble(t: Trusted, base: Path, patch: Path | None, pre: Path | None = None) -> tuple[Path | None, str | None]:
    src = t.work / "candidate_src"
    shutil.copytree(base, src, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    t.give(src)
    git = ["git", "-c", "user.name=v", "-c", "user.email=v@v", "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        r = t.run([*git, *cmd], cwd=src, candidate=True, timeout=1800)
        if r.rc != 0:
            raise RuntimeError(f"cannot stage the base tree: git {cmd[0]}: {r.stderr[-400:]}")
        if cmd[0] == "add" and pre is not None:  # a derived (span) task starts from base + the span's start patch (grader material)
            staged_pre = t.work / "start.patch"
            shutil.copy(pre, staged_pre)
            os.chmod(staged_pre, 0o644)
            r = t.run([*git, "apply", "--whitespace=nowarn", staged_pre], cwd=src, candidate=True, timeout=600)
            if r.rc != 0:
                raise RuntimeError(f"the span start patch does not apply to the base: {r.stderr[-400:]}")
            r = t.run([*git, "add", "-A"], cwd=src, candidate=True, timeout=1800)
    if patch is not None and patch.exists() and patch.stat().st_size > 0:
        staged = t.work / "model.patch"
        shutil.copy(patch, staged)
        os.chmod(staged, 0o644)
        r = t.run([*git, "apply", "--whitespace=nowarn", staged], cwd=src, candidate=True, timeout=600)
        if r.rc != 0:
            return None, f"patch does not apply: {r.stderr[-600:]}"
    shutil.rmtree(src / ".git", ignore_errors=True)
    t.seal(src)
    return src, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--patch", type=Path)
    ap.add_argument("--pre-patch", type=Path, help="derived tasks: applied to the base before the agent's patch")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--tier", default="full")
    ap.add_argument("--require-root", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    prewrite_zero(a.out)
    work = Path(tempfile.mkdtemp(prefix="crucible-grade-"))
    report: GradeReport | None = None
    error = None
    try:
        t = Trusted(work, require_root=a.require_root)
        params = dict(_param(p) for p in a.param)
        cand, why = (a.candidate, None)
        if a.base is not None:
            cand, why = assemble(t, a.base, a.patch, a.pre_patch)
        mod = importlib.import_module(a.plan)
        ctx = Context(t, candidate=cand, params=params, task_dir=Path(params["tests"]) if "tests" in params else None)
        if why is not None:
            p = mod.plan(ctx)
            report = GradeReport(p.task, a.tier, [], {g.id: "skip" for g in p.gates}, 0.0, 0.0,
                                 {"status": "ok", "failure_kind": None, "reason": None}, t.is_root, 0.0, meta={"candidate_error": why})
        else:
            report = mod.plan(ctx).run(ctx, tier=a.tier, on_result=lambda r: print(f"[{r.status.upper()}] {r.gate}/{r.id} {r.seconds:.0f}s", flush=True))
    except Exception as ex:  # noqa: BLE001 -- never leave an empty reward
        error = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-3000:]}"
    finally:
        try:
            if "t" in locals():
                t.kill()
        except Exception:  # noqa: BLE001, S110 -- best effort; the reward is written below regardless
            pass
        rj = write_outputs(report, a.out, error=error)
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"reward": rj["reward"], "gates": rj.get("gates"), "infra": rj.get("infra")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
