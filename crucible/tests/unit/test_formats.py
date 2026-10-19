"""Persisted formats are frozen per version (M0 audit 2026-10-14 R-18): see tests/unit/_formats.py for the rules."""

from __future__ import annotations

import json
import re

import pytest

from tests.unit._formats import CRUCIBLE, FORMATS, SCHEMAS, Format, read_file, render_read, render_schema

BUMP = ("changed without a version bump: an added optional field is a minor bump (/v1 to /v1.1, with exclude_if), anything "
        "else a major bump with an upcaster; then run `uv run --frozen python scripts/freeze_schemas.py` (tests/unit/_formats.py)")


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: "/".join(f.ident()))
def test_each_format_matches_the_frozen_schema_of_its_version(fmt: Format):
    path = fmt.schema_file()
    assert path.exists(), f"{path.relative_to(CRUCIBLE)} is missing: run `uv run --frozen python scripts/freeze_schemas.py`"
    assert path.read_text() == render_schema(fmt.model()), f"{fmt.pointer} {BUMP}"


def test_every_frozen_example_of_every_version_still_loads():
    by_name = {f.ident()[0]: f for f in FORMATS}
    examples = sorted(SCHEMAS.glob("*/*.example.json"))
    assert len(examples) >= len(FORMATS)
    for path in examples:
        model = by_name[path.parent.name].model()
        doc = model.model_validate(json.loads(path.read_text()))
        assert model.model_validate_json(doc.model_dump_json(by_alias=True)) == doc, path.name


@pytest.mark.parametrize("example", sorted(SCHEMAS.glob("*/*.example.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_example_reads_as_frozen(example):
    """The content id and stored form of every example (an older version's through its upcaster) are frozen (M0 audit
    2026-10-16 R-9): the shape snapshot let an alias rename, a dropped exclude_if, an id-moving field and an edited
    upcaster through."""
    path = read_file(example)
    assert path.exists(), f"{path.relative_to(CRUCIBLE)} is missing: run `uv run --frozen python scripts/freeze_schemas.py`"
    assert path.read_text() == render_read(example), (f"{example.relative_to(CRUCIBLE)} now reads differently (content id or stored "
                                                      f"form): {BUMP}; for an older version, the upcaster changed")


def test_every_versioned_document_and_core_document_is_frozen():
    from crucible.core.schema import ALL_DOCUMENTS

    frozen = {f.pointer.partition(":")[2] for f in FORMATS}
    declared = {m for p in (CRUCIBLE / "src").rglob("*.py")
                for m in re.findall(r"^class (\w+)\((?:VersionedDocument|SchemaKeyed)\)", p.read_text(), re.MULTILINE)}
    assert declared and not declared - frozen, sorted(declared - frozen)
    assert not {c.__name__ for c in ALL_DOCUMENTS.values()} - frozen


def test_an_unbumped_change_is_caught():
    from pydantic import Field

    from crucible.core.schema.reward import RewardJson

    class RewardJson2(RewardJson):  # the in-place change that crucible.rollout/v1 suffered twice
        note: str = Field("", title="RewardJson")

    fmt = next(f for f in FORMATS if f.pointer.endswith(":RewardJson"))
    assert render_schema(RewardJson2) != fmt.schema_file().read_text()


def test_schema_keyed_documents_follow_the_versioned_document_rules():
    from crucible.core.schema.rollout import RolloutRecord

    base = {"rollout_id": "r", "sample": "s", "task": "t", "harness": "h", "policy": "p"}
    major, _, minor = RolloutRecord.SCHEMA.rpartition("/v")[2].partition(".")
    with pytest.raises(ValueError, match="newer minor version"):
        RolloutRecord.model_validate({**base, "schema": f"crucible.rollout/v{major}.{int(minor or 0) + 1}"})
    with pytest.raises(ValueError, match="newer major version"):
        RolloutRecord.model_validate({**base, "schema": "crucible.rollout/v2"})
    assert next(iter(RolloutRecord.model_validate(base).model_dump(by_alias=True))) == "schema"  # the on-disk key and order are kept


def test_a_pack_manifest_from_a_newer_crucible_is_refused_by_name():
    from crucible.core.schema.pack import PackManifest

    fmt = next(f for f in FORMATS if f.pointer.endswith(":PackManifest"))
    with pytest.raises(ValueError, match="newer Crucible"):
        PackManifest.model_validate({**fmt.example, "schema_version": 2})


def test_the_snapshot_ignores_docs_but_keeps_fields_named_like_them():
    from tests.unit._formats import _shape

    schema = {"title": "Doc", "description": "about it", "properties": {"title": {"type": "string", "title": "Title", "description": "x"}},
              "$defs": {"description": {"title": "Description", "properties": {"text": {"type": "string"}}}}}
    assert _shape(schema) == {"properties": {"title": {"type": "string"}}, "$defs": {"description": {"properties": {"text": {"type": "string"}}}}}
