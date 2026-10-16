"""Every persisted Crucible format, frozen per version (M0 audit 2026-10-14 R-18; PATTERN: OpenEnv
``tests/test_validation/test_schema_sync.py``, BSD-3-Clause, with one file per version instead of one overwritten file).

``schemas/<name>/<version>.schema.json`` is the format's JSON Schema at that version and ``<version>.example.json`` a full
document of it, as Crucible writes it. Neither is ever edited: ``tests/unit/test_formats.py`` fails when a model's schema
no longer matches the snapshot of its current version, and every example of every version must keep loading. Changing
a format means a new version: an added optional field is a minor bump (``/v1`` to ``/v1.1``; give the field
``exclude_if`` so documents that leave it unset keep their bytes and content hashes), anything else a major bump with an
upcaster (``core.schema.base.register_upcaster``). Then ``uv run --frozen python scripts/freeze_schemas.py`` writes the
new version's files. Unversioned formats (``steps.jsonl`` rows, ``WorldRef``) are frozen as ``unversioned``: their
readers outside Crucible see any change, so changing one needs a version field first.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CRUCIBLE = Path(__file__).resolve().parents[2]
SCHEMAS = CRUCIBLE / "schemas"

_MILESTONE = {"id": "m", "check": {"app": "a", "query": {"op": "exists"}}, "states": ["p1"]}


@dataclass(frozen=True)
class Format:
    pointer: str  # module:Class
    example: dict[str, Any] = field(default_factory=dict)
    name: str = ""  # formats without a SCHEMA string: the folder name
    version: str = ""  # formats without a SCHEMA string: v<N>, or "unversioned"

    def model(self) -> Any:
        mod, _, cls = self.pointer.partition(":")
        return getattr(importlib.import_module(mod), cls)

    def ident(self) -> tuple[str, str]:
        schema = getattr(self.model(), "SCHEMA", "")
        if schema:
            name, _, version = schema.partition("/")
            return name, version
        return self.name, self.version

    def schema_file(self) -> Path:
        name, version = self.ident()
        return SCHEMAS / name / f"{version}.schema.json"

    def example_file(self) -> Path:
        name, version = self.ident()
        return SCHEMAS / name / f"{version}.example.json"


FORMATS: list[Format] = [
    Format("crucible.core.schema.world:WorldSpec", {"world": "t/w", "apps": [{"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql"}}],
                                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}]}),
    Format("crucible.core.schema.trace:Trace", {"trace_id": "t1", "source": {"connector": "c", "ref": "r"},
                                                "events": [{"seq": 0, "kind": "action", "type": "commit", "text": "x"}],
                                                "signals": [{"id": "s", "type": "approval", "polarity": 1, "grades": [0]}]}),
    Format("crucible.core.schema.task:CrucibleTask", {"id": "p/f/n@1", "family": "f", "horizon": "H1",
                                                      "negatives": [{"id": "n", "path": "x.patch", "expected_gate": "g"}]}),
    Format("crucible.core.schema.task:TaskCard", {"id": "p/f/n@1", "family": "f", "horizon": "H1", "forge": {"generator": "g"}, "difficulty": {"traps": 1}}),
    Format("crucible.core.schema.milestone:MilestoneSet", {"family": "f", "milestones": [_MILESTONE]}),
    Format("crucible.core.schema.reward:RewardJson", {"reward": 0.5, "gates": {"g": 1}, "progress": 0.25}),
    Format("crucible.core.schema.run:RunRecord", {"run_id": "r", "task": "t", "task_digest": "sha256:00", "world": "w", "policy": "p",
                                                  "started_at": "2026-01-01T00:00:00Z", "outcome": "masked", "mask_sample": True,
                                                  "failure": {"kind": "agent_timeout", "stage": "agent", "reason": "x", "mask": True},
                                                  "usage": {"input_tokens": 100, "cache_read_tokens": 900, "output_tokens": 50, "calls": 1, "reported": 1},
                                                  "timing": {"seed": 1.0, "agent": 4.0, "model": 2.0}, "attempt": "s-a1", "lock": "sha256:01", "launch": 2}),
    Format("crucible.core.schema.run:RunLock", {"run_id": "r", "scheme": "crucible.compat_identity/v1", "fingerprint": "sha256:00",
                                                "excluded": ["concurrency"], "settings": {"k": 1, "policies.0.settings.temperature": 0.7},
                                                "opaque": ["harnesses.0 (h@1)"], "created_at": "2026-01-01T00:00:00.000000Z",
                                                "changes": [{"at": "2026-01-02T00:00:00.000000Z", "changed": ["k"], "before": "sha256:00", "after": "sha256:01"}]}),
    Format("crucible.core.schema.run:RunLaunch", {"run_id": "r", "n": 2, "started_at": "2026-01-02T00:00:00.000000Z", "lock": "sha256:01",
                                                  "lock_status": "changed", "argv": ["crucible", "bench", "run", "toy"],
                                                  "operational": {"concurrency": 4, "budget_usd": None}, "config": {"data_dir": "data"},
                                                  "build": {"crucible_version": "0.1.0", "source_sha256": "ab", "git_commit": "c27f5ca",
                                                            "git_dirty": False, "lock_sha256": "cd"}, "python": "3.12.14"}),
    Format("crucible.core.schema.rollout:RolloutRecord", {"rollout_id": "s1-a1", "sample": "s1", "task": "t", "harness": "native:llm-tool@1", "policy": "p",
                                                          "turns": [{"call_id": "c1", "actor": "agent", "prompt_token_ids": [1, 2],
                                                                     "completion_token_ids": [3], "logprobs": [-0.1], "policy_version": 0,
                                                                     "sampling": {"temperature": 0.7, "max_tokens": 800, "seed": 0}, "finish": "stop"}],
                                                          "capture": {"engine": "vllm", "logprobs": "processed",
                                                                      "tokenizer": {"token_map": "sha256:00", "render": "sha256:01", "vocab_size": 32},
                                                                      "template_kwargs": {"enable_thinking": False}}}),
    Format("crucible.workflow.graph:WorkflowGraph", {"family": "f", "milestones": [_MILESTONE]}),
    Format("crucible.accept.model:AcceptanceReport", {"task_id": "p/f/n@1", "key_digest": "d",
                                                      "key": {"task_id": "p/f/n@1", "overlay_hash": "o", "kit_hash": "k", "policy_version": "1",
                                                              "registry_version": "1"}}),
    Format("crucible.train.control_record:ControlRecord", {"rollout_id": "s1-a1", "sample": "s1", "task": "t", "harness": "h", "policy": "p",
                                                           "env": "gymnasium:CartPole-v1", "seed": 0, "reset_digest": "r"}),
    Format("crucible.core.schema.pack:PackManifest", {"name": "demo", "version": "0.1.0", "description": "d", "license": "Apache-2.0",
                                                      "package": "crucible_pack_demo", "entry_point": "crucible_pack_demo:register"},
           name="crucible.pack", version="v1"),
    # unversioned: written on their own without a version field
    Format("crucible.core.schema.reward:StepRow", {"step": 1, "phi": 0.5, "delta_phi": 0.1}, name="crucible.step_row", version="unversioned"),
    Format("crucible.core.schema.state:WorldRef", {"world": "w", "step": 1, "apps": {"db": {"app": "db", "adapter": "sql", "fingerprint": "f"}}},
           name="crucible.world_ref", version="unversioned"),
    Format("crucible.core.schema.trace:Record", {"id": "t1#0", "source": "t1", "text": "x"}, name="crucible.record", version="unversioned"),
    Format("crucible.core.schema.task:HarborTaskSummary", {"name": "n", "schema_version": "1.0", "task_hash": "h"},
           name="crucible.harbor_task_summary", version="unversioned"),
]


def render_schema(model: Any) -> str:
    """The format's shape: its JSON Schema without titles and descriptions, so a docstring edit is not a format change."""
    return json.dumps(_shape(model.model_json_schema(by_alias=True)), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _shape(node: Any, names: bool = False) -> Any:
    """``node`` without "title" and "description" annotations; under ``properties`` and ``$defs`` the keys are names
    (a field may be called ``title``), so they are kept and only their values are cleaned."""
    if isinstance(node, dict):
        return {k: _shape(v, names=not names and k in ("properties", "$defs", "patternProperties"))
                for k, v in node.items() if names or k not in ("title", "description")}
    if isinstance(node, list):
        return [_shape(v) for v in node]
    return node


def render_example(fmt: Format) -> str:
    doc = fmt.model().model_validate(fmt.example)
    return json.dumps(doc.model_dump(mode="json", by_alias=True, exclude_none=True), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
