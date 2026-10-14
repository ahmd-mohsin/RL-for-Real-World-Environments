"""A long-context item as a World task: the document is served by a ``docs`` app (``docs.info()``, ``docs.read(offset, n)``,
``docs.search(q)``), the answer is submitted through ``answer.submit(text)`` and graded from World state (the last
submission, scored by the family's ported scorer): outcome, not process. The same document is a host file
(``context_path``) a harness can mount as the kernel's ``CONTEXT`` (M16 §3.6); the instruction then names ``CONTEXT``.

The gold answer never enters the World: the forged bundle holds the document, the two tool modules and an empty
answers table; the grade reads the table through the verifier view.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any

import yaml

from crucible.bench.longctx.generators import LongCtxItem, generate

HERE = Path(__file__).resolve().parent
T0 = "2026-06-01T09:00:00Z"
TASK_VERSION = 1

WORLD_HOW = ("The document is {chars:,} characters (about {tokens:,} tokens) long and is too large to print. Read it through the `docs` "
             "tools: `docs.info()` gives its size{parts}, `docs.read(offset, n)` returns n characters (at most 65,536) from a "
             "character offset, `docs.search(q)` finds a substring (case-insensitive, the first 50 hits with their offsets).")
CONTEXT_HOW = ("The document is {chars:,} characters (about {tokens:,} tokens) long and is too large to print. It is the kernel's "
               "`CONTEXT`: `CONTEXT.info()`, `await CONTEXT.read(offset, n)`, `await CONTEXT.search(pattern)`. The same document is also "
               "served by the `docs` tools.")
SUBMIT = "Submit your final answer with the tool `answer.submit(text)`; the last submission counts. {fmt}"


class LongCtxTask:
    """``RunTask`` for the M10 runner over one generated item (``family``, ``seed``, ``tokens``)."""

    horizon = "H1"
    attack_surface = "world"

    def __init__(self, family: str, seed: int = 0, tokens: int = 8_000, *, delivery: str = "world", root: Path | None = None,
                 item: LongCtxItem | None = None):
        if delivery not in ("world", "context"):
            raise ValueError("delivery: world | context")
        self.item = item or generate(family, seed, tokens)
        self.delivery = delivery
        self.root = Path(root or tempfile.mkdtemp(prefix="longctx-pack-"))
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self.root), True) if root is None else None
        self.id = f"{self.item.id}{'-ctx' if delivery == 'context' else ''}@{TASK_VERSION}"
        self.version = TASK_VERSION
        self.world_id = "longctx/docs"
        self.parts = _parts(self.item)
        write_pack(self.item, self.root, self.parts)
        self.instruction = instruction(self.item, delivery, self.parts)

    @property
    def context_path(self) -> Path:
        """The document as a host file, for a harness that mounts it as ``CONTEXT``."""
        return self.root / "data" / "longctx" / "corpus" / "context.txt"

    def context_preview(self, n: int = 500) -> str:
        return self.item.context[:n]

    def world(self, workdir: Path) -> Any:
        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        return World(WorldSpec.load(self.root / "worlds" / "longctx.yaml"), instructions={"agent": self.instruction},
                     store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w", pack_root=self.root)

    async def open(self, workdir: Path) -> Any:
        w = self.world(workdir)
        await w.start()
        await w.reset()
        return w

    async def submitted(self, world: Any) -> str | None:
        from crucible.core.schema.state import Query

        res = (await world.verifier_view().query("answer", Query(op="sql", args={"sql": "SELECT text FROM answers ORDER BY id DESC LIMIT 1",
                                                                                  "max_rows": 1}))).value or {}
        rows = res.get("rows") or []
        return rows[0][0] if rows else None

    async def grade(self, world: Any, workdir: Path) -> Any:
        from crucible.verify.runtime.plan import GradeReport

        ans = await self.submitted(world)
        s = self.item.score(ans)
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=round(float(s["reward"]), 6), outcome=round(float(s["reward"]), 6),
                           infra={"status": "ok"}, trusted=True, elapsed_s=0.0,
                           meta={"family": self.item.family, "tokens": self.item.tokens, "answered": ans is not None, "score": s})

    def oracle_actions(self) -> list[dict]:
        """The gold answer submitted (the grader's positive control)."""
        return [{"kind": "tool_call", "tool_call_id": "gold", "function_name": "answer.submit", "arguments": {"text": self.item.gold_answer()}}]

    def hidden_strings(self) -> list[str]:
        return [self.item.gold_answer()]

    def cleanup(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        else:
            shutil.rmtree(self.root, ignore_errors=True)


def _parts(item: LongCtxItem) -> list[dict[str, Any]]:
    if item.family != "docqa":
        return []
    out = []
    text = item.context
    i = text.find("=== Document: ")
    while i != -1:
        j = text.find("=== Document: ", i + 1)
        name = text[i + len("=== Document: "): text.find(" ===", i)]
        out.append({"name": name, "offset": i, "chars": (j if j != -1 else len(text)) - i})
        i = j
    return out


def instruction(item: LongCtxItem, delivery: str, parts: list[dict[str, Any]]) -> str:
    how = (CONTEXT_HOW if delivery == "context" else WORLD_HOW).format(chars=len(item.context), tokens=item.tokens,
                                                                       parts=" and the offset of each document" if parts else "")
    return f"{item.question}\n\n{how}\n\n{SUBMIT.format(fmt=item.answer_format)}"


def write_pack(item: LongCtxItem, root: Path, parts: list[dict[str, Any]]) -> Path:
    root = Path(root)
    b = root / "data" / "longctx"
    (b / "tools").mkdir(parents=True, exist_ok=True)
    (b / "corpus").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "docs_tools.py", b / "tools" / "docs.py")
    shutil.copyfile(HERE / "answer_tools.py", b / "tools" / "answer.py")
    (b / "corpus" / "context.txt").write_text(item.context)
    (b / "corpus" / "parts.json").write_text(json.dumps(parts))
    (root / "data" / "answers.sql").write_text("CREATE TABLE answers (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, at TEXT NOT NULL);\n")
    (root / "data" / "docs.sql").write_text("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);\n")
    world = {
        "schema_version": "crucible.world/v1", "world": "longctx/docs",
        "description": f"A long-context {item.family} task: one document of about {item.tokens} tokens behind a read-only docs app (seed {item.seed}).",
        "backend": "local", "backend_params": {"bundles": [{"src": "data/longctx", "dest": "longctx"}]},
        "apps": [
            {"id": "docs", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "longctx/db/docs.db"}},
             "params": {"schema_sql": "data/docs.sql", "tools_module": "longctx/tools/docs.py", "transport": "inprocess"}, "description": "the document, read-only"},
            {"id": "answer", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "longctx/db/answers.db"}},
             "params": {"schema_sql": "data/answers.sql", "tools_module": "longctx/tools/answer.py", "transport": "inprocess"}, "description": "answer submission"},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["docs.*", "answer.*"]}],
        "clock": {"mode": "step", "t0": T0, "step_dt_s": 1},
        # one docs.read of 65,536 characters arrives whole (its JSON escapes at most double it); the default 30,000 cut the middle out
        "limits": {"max_tool_output_chars": 140_000},
        "events": [], "policies": [],
    }
    (root / "worlds").mkdir(parents=True, exist_ok=True)
    (root / "worlds" / "longctx.yaml").write_text(yaml.safe_dump(world, sort_keys=False, width=200))
    return root
