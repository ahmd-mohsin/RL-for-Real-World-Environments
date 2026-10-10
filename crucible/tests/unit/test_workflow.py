"""M5 workflow: action kinds, segmentation, file-state reconstruction (editors, shell, unified diffs), the replay
test, and milestone mining / evaluation / filters / matching."""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from crucible.connectors.chat_messages import messages_to_trace
from crucible.core.schema.trace import TraceSource
from crucible.workflow.actions import classify, classify_shell, command_of
from crucible.workflow.filestate import EditError, FileState
from crucible.workflow.milestones.evaluate import FileStateView, evaluate, evaluate_series
from crucible.workflow.milestones.filters import monotone, screen
from crucible.workflow.milestones.match import match
from crucible.workflow.milestones.mine import from_ci, from_diff
from crucible.workflow.replay import EditReplayer
from crucible.workflow.shellops import apply_shell
from crucible.workflow.steps import segment


@pytest.mark.parametrize(("cmd", "kind"), [
    ("cd /app && grep -rn foo src | head", "investigate"),
    ("sed -i 's/a/b/' src/x.cu", "edit"),
    ("cat > src/new.cpp <<'EOF'\nint x > 3;\nEOF", "edit"),
    ("mkdir -p /tmp/probe && cd /tmp/probe && cat <<'EOF' > p1.cpp\nint main(){}\nEOF\nclang++ p1.cpp -o p1 && ./p1", "build"),
    ("cd /tmp/x && cat > t.py <<'EOF'\nprint(1)\nEOF\npython3 t.py", "run"),
    ("cmake -S . -B build && ninja -C build 2>&1 | tail -5", "build"),
    ("python -m pytest tests/test_ops.py -x -q 2>&1 | tail -3", "test"),
    ("cd /w && python -c \"print(3 >= 2)\"", "run"),
    ("rm -rf build && cmake -S . -B build", "build"),
    ("git mv src/a.cu src/a.cpp", "edit"),
    ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A", "submit"),
])
def test_shell_kinds(cmd, kind):
    assert classify_shell(cmd) == kind


def test_tool_kinds_and_truncated_arguments():
    assert classify("str_replace_editor", {"command": "view", "path": "/a"}) == "investigate"
    assert classify("ws.write_file", {"path": "a"}) == "edit" and classify("tickets.issue_refund", {}) == "act"
    assert classify("Read", {}) == "investigate" and classify("send_message", {}) == "communicate" and classify("think", {}) == "other"
    truncated = '{"command": "cd /app && sed -i \\"s/a/b/\\" src/x.py && pytest -q tests/te…'
    assert command_of(truncated).startswith("cd /app && sed -i") and classify("bash", truncated) == "edit"


def _chat_trace():
    root = "/workspace/r__x__1"
    call = lambda i, name, args: {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    res = lambda i, out: {"role": "tool", "tool_call_id": f"c{i}", "content": out}
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"<uploaded_files>\n{root}\n</uploaded_files> fix it"},
            call(1, "str_replace_editor", {"command": "view", "path": f"{root}/src/a.py"}), res(1, "a"),
            call(2, "execute_bash", {"command": f"cd {root} && grep -n helper src/a.py"}), res(2, "1:helper"),
            call(3, "str_replace_editor", {"command": "str_replace", "path": f"{root}/src/a.py", "old_str": "old_helper(x)", "new_str": "new_helper(x)"}), res(3, "ok"),
            call(4, "str_replace_editor", {"command": "create", "path": f"{root}/repro.py", "file_text": "print(1)\n"}), res(4, "ok"),
            call(5, "execute_bash", {"command": "python -m pytest -q tests"}), res(5, "1 passed"),
            call(6, "execute_bash", {"command": f"cd {root} && rm -f repro.py"}), res(6, ""),
            call(7, "finish", {}), res(7, "done")]
    return messages_to_trace(msgs, trace_id="t", source=TraceSource(connector="test", ref="t")), root


def test_segmentation_groups_reads_and_covers_every_event():
    tr, _ = _chat_trace()
    st = segment(tr)
    assert sorted(q for s in st for q in s.seqs) == [e.seq for e in tr.events]
    kinds = [s.kind for s in st if s.kind != "message"]
    assert kinds == ["investigate", "edit", "edit", "test", "edit", "submit"]  # view + grep merged into one investigation
    assert next(s for s in st if s.kind == "investigate").calls == 2


def test_replay_reproduces_final_patch_and_flags_divergence():
    tr, root = _chat_trace()
    base = {"src/a.py": "def f(x):\n    return old_helper(x)\n"}.get
    good_patch = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n def f(x):\n-    return old_helper(x)\n+    return new_helper(x)\n"
    po = lambda p: p[len(root) + 1:] if p.startswith(root + "/") else None
    ok = EditReplayer(base, good_patch, path_of=po, cwd=root).replay(tr)
    assert ok.ok and ok.meta["ok_agent_state"]  # repro.py created then rm'd: replayed through the shell op
    bad = EditReplayer(base, good_patch.replace("new_helper(x)", "other(x)"), path_of=po, cwd=root).replay(tr)
    assert not bad.ok and bad.quarantined and "src/a.py" in bad.reason


def test_unified_diff_round_trips_against_git(tmp_path: Path):
    rng = random.Random(0)
    for trial in range(25):
        d = tmp_path / f"r{trial}"
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        files = {}
        for k in range(3):
            lines = [f"l{i} {rng.random():.3f}" for i in range(rng.randint(0, 30))]
            txt = "\n".join(lines) + ("\n" if lines and rng.random() < 0.8 else "")
            files[f"f{k}.txt"] = txt
            (d / f"f{k}.txt").write_text(txt)
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(d), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "b"], check=True)
        new = {}
        for p, txt in files.items():
            if p == "f2.txt":
                continue
            lines = txt.split("\n")
            for _ in range(rng.randint(0, 5)):
                j = rng.randint(0, max(len(lines) - 1, 0))
                op = rng.choice(["ins", "del", "mod"])
                if op == "ins":
                    lines.insert(j, f"n{rng.random():.3f}")
                elif lines and op == "del":
                    lines.pop(j)
                elif lines:
                    lines[j] = f"m{rng.random():.3f}"
            new[p] = "\n".join(lines).rstrip("\n") if rng.random() < 0.3 else "\n".join(lines)
            (d / p).write_text(new[p])
        (d / "added.txt").write_text("x\n")
        os.remove(d / "f2.txt")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        patch = subprocess.run(["git", "-C", str(d), "diff", "--cached"], capture_output=True, text=True, check=True).stdout
        fs = FileState(base=files.get)
        fs.apply_patch(patch)
        assert all(fs.files.get(p, files.get(p)) == v for p, v in {**new, "added.txt": "x\n"}.items()) and fs.files.get("f2.txt", "X") is None


def test_editor_and_shell_semantics():
    fs = FileState(base={"a.py": "a\nb\nc\n"}.get)
    fs.str_replace_editor({"command": "str_replace", "old_str": "b", "new_str": "B"}, "a.py")
    with pytest.raises(EditError):
        fs.str_replace_editor({"command": "str_replace", "old_str": "zzz", "new_str": "q"}, "a.py")
    fs.str_replace_editor({"command": "insert", "insert_line": 0, "new_str": "top"}, "a.py")
    fs.str_replace_editor({"command": "undo_edit"}, "a.py")
    assert fs.files["a.py"] == "a\nB\nc\n"
    po = lambda p: p[len("/w/"):] if p.startswith("/w/") else None
    eff = apply_shell(fs, "cd /w && cp a.py b.py && git checkout a.py && sed -i 's/x/y/' b.py", cwd="/", path_of=po)
    assert fs.files["b.py"] == "a\nB\nc\n" and fs.files["a.py"] == "a\nb\nc\n" and eff.unreplayable == ["sed -i"]


def test_mining_evaluation_and_filters():
    base = {"m.py": "class A:\n    def f(self, x):\n        y = old_helper(x)\n        return y\n"}.get
    patch = "--- a/m.py\n+++ b/m.py\n@@ -2,3 +2,3 @@ class A:\n     def f(self, x):\n-        y = old_helper(x)\n+        y = new_helper(x, strict=True)\n         return y\n"
    cands = from_diff(patch, base)
    ms = [c.milestone for c in cands]
    kinds = {mid.split("/")[1] for mid in (m.id for m in ms)}
    assert {"located", "removed_call", "added_call"} <= kinds
    s0, s_alt = FileState(base=base), FileState(base=base)
    s_alt.write("m.py", base("m.py").replace("old_helper(x)", "other(x)"))  # a different fix in the right place
    phi0 = asyncio.run(evaluate(ms, FileStateView(s0)))
    phi_alt = asyncio.run(evaluate(ms, FileStateView(s_alt)))
    assert all(v == 0.0 for v in phi0.values())  # non-trivial at base
    located = next(m.id for m in ms if "/located/" in m.id)
    assert phi_alt[located] == 1.0 and phi_alt[next(m.id for m in ms if m.id.startswith("diff/added_call"))] == 0.0
    good = asyncio.run(evaluate_series(ms, [(0, FileStateView(s0)), (1, FileStateView(s_alt))]))
    assert monotone([good], located)[0]
    v = screen(located, base_phi=0.0, end_phi=1.0, series=[good], good=[good])
    assert v.kept, v
    order = match(ms, good)
    assert order.achieved[located] == 1 and not order.regressed


def test_ci_fail_to_pass_candidates():
    runs = [(3, {"t::a": "fail", "t::b": "pass"}), (9, {"t::a": "pass", "t::b": "fail"})]
    c = from_ci(runs)
    assert [x.evidence["test"] for x in c] == ["t::a"]  # b regressed: not a milestone
