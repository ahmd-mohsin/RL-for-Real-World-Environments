"""Regression tests for the M0 audit (docs/audit/2026-10-12/M00-core.md, defects D1-D11).

Each test reproduces one reproduced defect; property tests cover every core model, not only the ones the audit
happened to find.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
import shutil
import threading
import typing

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import crucible.core.schema as schema_pkg
from crucible.core.config import Config, LLMProvider
from crucible.core.registry import Registry, RegistryError
from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.milestone import MilestoneSet
from crucible.core.schema.reward import RewardJson, StepRow
from crucible.core.schema.run import RunRecord
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store, StoreError


def _all_models() -> list[type[CrucibleModel]]:
    out = []
    for info in pkgutil.iter_modules(schema_pkg.__path__):
        mod = importlib.import_module(f"crucible.core.schema.{info.name}")
        for obj in vars(mod).values():
            if inspect.isclass(obj) and issubclass(obj, CrucibleModel) and obj.__module__ == mod.__name__:
                out.append(obj)
    return out


BASE_WORLD = {
    "world": "t/w",
    "apps": [{"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql"}}],
    "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}],
}


# ------------------------------------------------------------------------------------------------ D1: lossless None
def test_no_model_has_a_nullable_field_with_a_non_null_default():
    """An explicit None must survive canonical JSON; exclude_none drops it, so such fields may not exist."""
    offenders = []
    for cls in _all_models():
        for name, f in cls.model_fields.items():
            nullable = type(None) in typing.get_args(f.annotation)
            if nullable and not f.is_required() and f.default_factory is None and f.default is not None:
                offenders.append(f"{cls.__name__}.{name}")
    assert offenders == []


def test_world_round_trip_keeps_value_and_hash():
    w = WorldSpec.model_validate({**BASE_WORLD, "isolation": {"pids_limit": 0}})
    back = WorldSpec.model_validate_json(w.canonical_json())
    assert back.isolation.pids_limit == w.isolation.pids_limit
    assert back.content_hash() == w.content_hash()


# ------------------------------------------------------------------------------------------------ D2: finite floats
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad):
    with pytest.raises(ValueError):
        RewardJson(reward=bad)


def test_step_phi_is_a_potential_in_unit_interval():
    with pytest.raises(ValueError):
        StepRow(step=0, phi=3.0, delta_phi=0.0)


@settings(max_examples=200, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
def test_canonical_json_is_strict_json_and_stable(x):
    r = RewardJson(reward=x)
    text = r.canonical_json()
    import json

    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert RewardJson.model_validate_json(text).content_hash() == r.content_hash()


# ------------------------------------------------------------------------------------------------ D4: store races
def test_put_file_never_stores_bytes_that_mismatch_their_key(tmp_path, monkeypatch):
    st_ = Store(tmp_path / "s")
    src = tmp_path / "live.bin"
    src.write_bytes(b"A" * 1_000_000)
    orig = shutil.copyfile

    def racing_copy(a, b, *args, **kw):  # the source changes while the store reads it
        src.write_bytes(b"B" * 1_000_000)
        return orig(a, b, *args, **kw)

    monkeypatch.setattr(shutil, "copyfile", racing_copy)
    key = st_.put_file(src)
    monkeypatch.setattr(shutil, "copyfile", orig)
    st_.get_bytes(key)  # raises StoreError if the blob does not match its key


def test_concurrent_put_file_of_one_new_file(tmp_path):
    st_ = Store(tmp_path / "s")
    src = tmp_path / "x.bin"
    src.write_bytes(bytes(range(256)) * 40_000)
    errors: list[BaseException] = []

    def put():
        try:
            st_.put_file(src)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=put) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_corrupt_blob_is_detected(tmp_path):
    st_ = Store(tmp_path / "s")
    key = st_.put_bytes(b"hello")
    p = st_.blob_path(key)
    p.chmod(0o644)
    p.write_bytes(b"hellO")
    with pytest.raises(StoreError):
        st_.get_bytes(key)


# ------------------------------------------------------------------------------------------------ D5: spec validation
@pytest.mark.parametrize(
    "extra",
    [
        {"network": "allowlist"},
        {"network": "no-network", "allowed_hosts": ["api.example.com"]},
        {"clock": {"step_dt_s": -60}},
        {"clock": {"t0": "yesterday"}},
        {"limits": {"tool_timeout_s": -1}},
        {"limits": {"max_tool_output_chars": -5}},
        {"limits": {"tool_quota": {"db.*": -3}}},
        {"clock": {}, "events": [{"id": "e", "when": {"after": "1h"}, "actor": "agent", "action": {"type": "tool", "tool": "ghost.run"}}]},
        {"clock": {}, "events": [{"id": "e", "when": {"after": "1h"}, "actor": "agent", "action": {"type": "permission", "grant": ["ghost.*"]}}]},
    ],
)
def test_world_spec_rejects_mistakes(extra):
    with pytest.raises(ValueError):
        WorldSpec.model_validate({**BASE_WORLD, **extra})


# ------------------------------------------------------------------------------------------------ D6: redaction
def test_redacted_config_never_contains_raw_secrets():
    # api_key and secret headers cannot be literal (R-8); free-form request_defaults still can, and redaction masks them
    c = Config(llm_providers={"x": LLMProvider(base_url="http://h", request_defaults={"extra_headers": {"Authorization": "Bearer tok-REAL"}})})
    text = str(c.redacted())
    assert "tok-REAL" not in text
    ref = Config(llm_providers={"x": LLMProvider(base_url="http://h", api_key="env:MY_KEY")})
    assert "env:MY_KEY" in str(ref.redacted())


# ------------------------------------------------------------------------------------------------ D7: plugin errors
def test_registry_reports_why_a_plugin_is_missing(monkeypatch):
    class EP:
        name = "broken_pack"

        def load(self):
            raise ImportError("No module named 'crucible_pack_broken'")

    monkeypatch.setattr("crucible.core.registry.entry_points", lambda group: [EP()])
    r = Registry()
    with pytest.raises(RegistryError, match="broken_pack"):
        r.get("pack", "broken")


# ------------------------------------------------------------------------------------------------ D8: versions
@pytest.mark.parametrize("cls", [RewardJson, RunRecord])
def test_persisted_formats_are_versioned(cls):
    assert "schema_version" in cls.model_fields


def test_unversioned_reward_document_still_loads_as_v1():
    r = RewardJson.model_validate({"reward": 1.0})
    assert r.schema_version.endswith("/v1")


# ------------------------------------------------------------------------------------------------ D9: version bypass
def test_schema_version_cannot_be_reassigned_to_unknown_major():
    ms = MilestoneSet(family="f", milestones=[])
    with pytest.raises(ValueError):
        ms.schema_version = "crucible.milestones/v9"


# ------------------------------------------------------------------------------------------------ D10: names
def test_store_lists_a_document_under_every_name(tmp_path):
    st_ = Store(tmp_path / "s")
    doc = RewardJson(reward=1.0)
    st_.put_doc("reward", doc, name="a")
    st_.put_doc("reward", doc, name="b")
    assert len(st_.list_docs("reward", name="a")) == 1
    assert len(st_.list_docs("reward", name="b")) == 1


# ------------------------------------------------------------------------------------------------ D11: run states
def test_run_record_rejects_contradictions():
    with pytest.raises(ValueError):
        RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z", outcome="completed", mask_sample=True, reward=0.7)
    with pytest.raises(ValueError):
        RunRecord(run_id="r", world="w", policy="p", started_at="not-a-time")


# ------------------------------------------------------------------------------------------------ every model
@pytest.mark.parametrize("cls", [c for c in _all_models() if issubclass(c, VersionedDocument) and c is not VersionedDocument], ids=lambda c: c.__name__)
def test_versioned_documents_declare_their_schema(cls):
    assert cls.SCHEMA.startswith("crucible.") and "/v" in cls.SCHEMA


# ------------------------------------------------------------------------------------------------ D12: normalization
def test_negative_zero_and_positive_zero_hash_alike():
    assert RewardJson(reward=0.0).content_hash() == RewardJson(reward=-0.0).content_hash()


def test_unicode_normalization_forms_hash_alike():
    from crucible.core.schema.milestone import Milestone, MilestoneCheck
    from crucible.core.schema.state import Query

    def ms(name: str) -> str:
        return MilestoneSet(family=name, milestones=[Milestone(id="m", check=MilestoneCheck(app="a", query=Query(op="exists")))]).content_hash()

    assert ms("café") == ms("café")


# ------------------------------------------------------------------------------------------------ D13: hash drift
GOLDEN = {
    # pinned content hashes: a schema change that alters an existing id must update these on purpose (scheme bump)
    "reward_one": ("RewardJson", {"reward": 1.0}),
    "world_min": ("WorldSpec", BASE_WORLD),
}


def test_hash_scheme_is_declared():
    from crucible.core.ids import HASH_SCHEME

    assert isinstance(HASH_SCHEME, int) and HASH_SCHEME >= 1


def test_golden_content_hashes_do_not_drift():
    import json
    from pathlib import Path

    from crucible.core.schema import reward as reward_mod
    from crucible.core.schema import world as world_mod

    classes = {"RewardJson": reward_mod.RewardJson, "WorldSpec": world_mod.WorldSpec}
    pin = Path(__file__).with_name("golden_hashes.json")
    got = {k: classes[c].model_validate(v).content_hash() for k, (c, v) in GOLDEN.items()}
    want = json.loads(pin.read_text())
    assert got == want, "content hashes drifted: bump HASH_SCHEME and regenerate tests/unit/golden_hashes.json"


# ------------------------------------------------------------------------------------------------ D15: store hygiene
def test_store_uses_wal_and_sweeps_stale_temp_files(tmp_path):
    root = tmp_path / "s"
    st_ = Store(root)
    assert st_._db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    stale = root / "blobs" / "ab" / "tmpdeadwriter.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")
    import os
    import time

    old = time.time() - 7200
    os.utime(stale, (old, old))
    st_.close()
    Store(root)
    assert not stale.exists()


def test_put_repairs_a_corrupt_blob(tmp_path):
    st_ = Store(tmp_path / "s")
    key = st_.put_bytes(b"payload")
    p = st_.blob_path(key)
    p.chmod(0o644)
    p.write_bytes(b"trunc")
    assert st_.put_bytes(b"payload") == key
    assert st_.get_bytes(key) == b"payload"


# ------------------------------------------------------------------------------------------------ D16: ids
@pytest.mark.parametrize("bad", ["../escape", "a/../b", ".", "..", "x\u200by", "", "a" * 300])
def test_ids_used_as_path_segments_are_validated(bad):
    from crucible.core.ids import validate_id

    with pytest.raises(ValueError):
        validate_id(bad)


def test_valid_ids_pass():
    from crucible.core.ids import validate_id

    for ok in ["toy/helpdesk", "task04@v3", "finance_ap-invoices.s1", "café"]:
        assert validate_id(ok)


# ------------------------------------------------------------------------------------------------ R-4: ids as paths
@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "x\0y", "", "café"])
def test_child_path_refuses_unsafe_names(tmp_path, bad):
    from crucible.core.ids import child_path

    with pytest.raises(ValueError):
        child_path(tmp_path, bad)


def test_child_path_refuses_a_symlink_out_of_root(tmp_path):
    from crucible.core.ids import child_path

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "link").symlink_to(tmp_path)
    assert child_path(tmp_path / "runs", "ok-run", "a1") == tmp_path / "runs" / "ok-run" / "a1"
    with pytest.raises(ValueError, match="outside"):
        child_path(tmp_path / "runs", "link")


def test_run_journal_refuses_a_run_id_that_escapes_the_root(tmp_path):
    from crucible.runtime.journal import RunJournal

    with pytest.raises(ValueError):
        RunJournal(tmp_path / "runs", "../escaped", "identity", {})
    assert not (tmp_path / "escaped").exists()
    assert RunJournal(tmp_path / "runs", "bench-toy-bench", "identity", {}).dir == tmp_path / "runs" / "bench-toy-bench"


# ------------------------------------------------------------------------------------------------ D17: pack kinds
def test_packs_can_register_namespaced_failure_kinds():
    from crucible.core.schema.run import Failure, register_failure_kind

    register_failure_kind("finance_ap:erp_down", blame="infra")
    f = Failure(kind="finance_ap:erp_down", stage="agent")
    r = RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z")
    r.mark_failure(f)
    assert r.mask_sample and r.scored_reward() is None
    with pytest.raises(ValueError):
        register_failure_kind("no_namespace", blame="infra")


def test_failure_reason_is_bounded():
    from crucible.core.schema.run import Failure

    assert len(Failure(kind="agent_timeout", stage="agent", reason="x" * 10_000).reason) <= 2000


# ------------------------------------------------------------------------------------------------ R-5: secrets by value
def test_a_resolved_secret_with_no_vendor_shape_is_scrubbed(tmp_path, monkeypatch):
    from crucible.core.config import resolve_secret
    from crucible.core.schema.run import Failure
    from crucible.core.secrets import scrub_json, scrub_secrets

    key = "gw7Kq2PzR8xLm4Tn9Qa-r5test"  # a gateway key no pattern knows
    assert scrub_secrets(f"invalid key {key}") == f"invalid key {key}"  # before it is resolved
    monkeypatch.setenv("R5_GATEWAY_KEY", key)
    assert resolve_secret("env:R5_GATEWAY_KEY", search_from=tmp_path) == key
    assert scrub_secrets(f"401 Unauthorized: invalid key {key}") == "401 Unauthorized: invalid key <secret>"
    rec = Failure(kind="agent_run_error", stage="agent", reason=f"provider said {key}").model_dump_json()
    assert key not in scrub_json(rec)


def test_short_values_are_not_registered_as_secrets():
    from crucible.core.secrets import register_secret, scrub_secrets

    register_secret("abc1234")  # 7 characters: would mask ordinary words
    assert scrub_secrets("abc1234 stays") == "abc1234 stays"


# ------------------------------------------------------------------------------------------------ R-6: private stores
def test_a_new_store_is_private_to_its_owner(tmp_path):
    import os
    import stat

    old = os.umask(0o022)
    try:
        st_ = Store(tmp_path / "store")
        key = st_.put_bytes(b"trace with customer data")
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)
    assert mode(tmp_path / "store") == 0o700
    assert mode(tmp_path / "store" / "index.sqlite") == 0o600
    assert (tmp_path / "store" / ".gitignore").read_text() == "*\n"
    assert st_.get_bytes(key) == b"trace with customer data"


def test_an_existing_store_root_keeps_its_mode(tmp_path):
    import os
    import stat

    (tmp_path / "shared").mkdir(mode=0o755)
    os.chmod(tmp_path / "shared", 0o755)
    Store(tmp_path / "shared")
    assert stat.S_IMODE(os.stat(tmp_path / "shared").st_mode) == 0o755
    assert not (tmp_path / "shared" / ".gitignore").exists()


# ------------------------------------------------------------------------------------------------ R-7: strict loaders
def test_a_duplicate_yaml_key_is_refused_naming_both_lines(tmp_path):
    import yaml

    from crucible.core.serialize import load_yaml

    text = "name: seed\nnetwork: none\napps: {}\nnetwork: public\n"
    with pytest.raises(yaml.YAMLError, match=r"duplicate key 'network'") as e:
        load_yaml(text, source="world.yaml")
    assert 'world.yaml", line 2' in str(e.value) and 'world.yaml", line 4' in str(e.value)
    assert load_yaml("a: {b: 1, c: 2}\nd: [1, 2]\n") == {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    with pytest.raises(yaml.YAMLError, match="duplicate key 'x'"):
        load_yaml("outer:\n  x: 1\n  x: 2\n")  # nested mappings too


def test_a_yaml_merge_is_not_a_duplicate():
    from crucible.core.serialize import load_yaml

    assert load_yaml("base: &b {x: 1, y: 2}\nleaf:\n  <<: *b\n  y: 3\n")["leaf"] == {"x": 1, "y": 3}


def test_json_refuses_duplicate_keys_and_non_finite_numbers():
    from crucible.core.serialize import load_json

    with pytest.raises(ValueError, match="duplicate key 'network'"):
        load_json('{"network": "none", "network": "public"}')
    for bad in ('{"reward": NaN}', '{"reward": Infinity}', "[-Infinity]"):
        with pytest.raises(ValueError, match="not a JSON number"):
            load_json(bad)
    assert load_json('{"a": [1, 2.5, null]}') == {"a": [1, 2.5, None]}


def test_documents_and_pack_manifests_load_strictly(tmp_path):
    import yaml

    from crucible.studio.pack import load_manifest

    class Doc(CrucibleModel):
        name: str
        network: str = "none"

    (tmp_path / "w.yaml").write_text("name: seed\nnetwork: none\nnetwork: public\n")
    (tmp_path / "w.json").write_text('{"name": "seed", "network": "none", "network": "public"}')
    with pytest.raises(yaml.YAMLError):
        Doc.load(tmp_path / "w.yaml")
    with pytest.raises(ValueError, match="duplicate key"):
        Doc.load(tmp_path / "w.json")
    (tmp_path / "ok.json").write_text('{"name": "seed"}')
    assert Doc.load(tmp_path / "ok.json").network == "none"
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack.yaml").write_text("name: demo\nname: other\n")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'name'"):
        load_manifest(pack)


# ------------------------------------------------------------------------------------------------ R-8: references only
@pytest.mark.parametrize("key", ["sk-REALSECRET123", "env:", "Bearer x"])
def test_a_literal_api_key_is_refused(key):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="api_key must be env:NAME"):
        LLMProvider(base_url="http://h", api_key=key)


@pytest.mark.parametrize("key", ["", "EMPTY", "none", "env:OPENROUTER_API_KEY"])
def test_references_and_local_placeholders_load(key):
    assert LLMProvider(base_url="http://127.0.0.1:8000/v1", api_key=key).api_key == key


def test_secret_headers_are_references_resolved_at_use(tmp_path, monkeypatch):
    from pydantic import ValidationError

    from crucible.core.secrets import scrub_secrets

    with pytest.raises(ValidationError, match=r"\['Authorization'\] look secret"):
        LLMProvider(base_url="http://h", headers={"Authorization": "Bearer tok-REAL", "X-Title": "crucible"})
    p = LLMProvider(base_url="http://h", headers={"Authorization": "env:R8_GATEWAY_AUTH", "X-Title": "crucible"})
    monkeypatch.setenv("R8_GATEWAY_AUTH", "Bearer r8-gateway-token-0001")
    assert p.resolved_headers(search_from=tmp_path) == {"Authorization": "Bearer r8-gateway-token-0001", "X-Title": "crucible"}
    assert "r8-gateway-token-0001" not in p.model_dump_json()  # the dump keeps the reference
    assert scrub_secrets("401 for Bearer r8-gateway-token-0001") == "401 for <secret>"  # registered by resolution (R-5)
    assert scrub_secrets("X-Title crucible") == "X-Title crucible"  # literal non-secret headers are not registered


# ------------------------------------------------------------------------------------------------ R-10: writable text
@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=st.sampled_from(["a", "é", "/", "\udcff", "\ud800", "\U0001f600", " "]), max_size=40))
def test_a_validated_failure_always_serializes_and_hashes(text):
    from crucible.core.schema.run import Failure

    f = Failure(kind="agent_run_error", stage="verification", reason=text, detail=text)
    f.model_dump_json().encode("utf-8")
    assert f.content_hash()


def test_a_failure_naming_a_non_utf8_file_reaches_the_journal(tmp_path):
    import os

    from crucible.core.schema.run import Failure
    from crucible.runtime.journal import RunJournal

    name = os.fsdecode(b"report_\xff.txt")  # 'report_\udcff.txt'
    f = Failure(kind="agent_run_error", stage="verification", reason=f"cannot open {name}")
    assert f.reason == "cannot open report_\\udcff.txt"  # escaped as visible text, not dropped
    rec = RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z", attempt="s1-a1", outcome="failed", failure=f)
    RunJournal(tmp_path / "runs", "r", "identity", {}).finish(rec, terminal=True)
    assert "report_\\\\udcff.txt" in (tmp_path / "runs" / "r" / "results.jsonl").read_text()


# ------------------------------------------------------------------------------------------------ R-11: build identity
def _git_repo(root):
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                              capture_output=True, text=True).stdout.strip()

    root.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    return git


def test_a_foreign_repository_never_lends_its_commit(tmp_path):
    from crucible.core.build import _checkout_state

    git = _git_repo(tmp_path / "project")
    (tmp_path / "project" / "README").write_text("someone else's repo\n")
    git("add", "README")
    git("commit", "-qm", "init")
    pkg = tmp_path / "project" / ".venv" / "site-packages" / "crucible"  # installed inside it, never tracked
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    assert _checkout_state(pkg) == (None, None)


def test_dirty_covers_the_package_only(tmp_path):
    from crucible.core.build import _checkout_state

    git = _git_repo(tmp_path / "mono")
    pkg = tmp_path / "mono" / "src" / "crucible"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (tmp_path / "mono" / "other.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "init")
    head = git("rev-parse", "HEAD")
    assert _checkout_state(pkg) == (head, False)
    (tmp_path / "mono" / "other.txt").write_text("b\n")  # elsewhere in the monorepo
    assert _checkout_state(pkg) == (head, False)
    (pkg / "__init__.py").write_text("x = 1\n")
    assert _checkout_state(pkg) == (head, True)


def test_source_digest_follows_content_not_bytecode(tmp_path):
    from crucible.core.build import build_info, source_digest

    (tmp_path / "a.py").write_text("x = 1\n")
    d = source_digest(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    assert source_digest(tmp_path) == d
    (tmp_path / "a.py").write_text("x = 2\n")
    assert source_digest(tmp_path) != d
    assert len(build_info().source_sha256) == 64


# ------------------------------------------------------------------------------------------------ R-12: index versions
_LEGACY_INDEX = """
CREATE TABLE docs (kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT, created_at TEXT NOT NULL, provenance TEXT, size INTEGER, PRIMARY KEY (kind, key));
CREATE INDEX docs_name ON docs(kind, name);
CREATE TABLE tags (name TEXT PRIMARY KEY, key TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE doc_names (kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL, provenance TEXT, PRIMARY KEY (kind, key, name));
CREATE INDEX doc_names_name ON doc_names(kind, name);
INSERT INTO tags VALUES ('latest', 'sha256:00', '2026-01-01');
"""


def _user_version(root):
    import sqlite3

    with sqlite3.connect(root / "index.sqlite") as c:
        return c.execute("PRAGMA user_version").fetchone()[0]


def test_a_new_index_carries_the_current_version(tmp_path):
    from crucible.core.store import INDEX_VERSION

    Store(tmp_path / "s")
    assert _user_version(tmp_path / "s") == INDEX_VERSION >= 1


def test_an_unversioned_index_is_adopted_with_its_rows(tmp_path):
    import sqlite3

    (tmp_path / "s").mkdir()
    with sqlite3.connect(tmp_path / "s" / "index.sqlite") as c:  # an index written before versioning
        c.executescript(_LEGACY_INDEX)
    assert Store(tmp_path / "s").resolve("latest") == "sha256:00"
    assert _user_version(tmp_path / "s") >= 1


def test_an_index_from_a_newer_crucible_is_refused(tmp_path):
    import sqlite3

    from crucible.core.store import INDEX_VERSION

    Store(tmp_path / "s")
    with sqlite3.connect(tmp_path / "s" / "index.sqlite") as c:
        c.execute(f"PRAGMA user_version = {INDEX_VERSION + 1}")
    with pytest.raises(StoreError, match="newer than this Crucible"):
        Store(tmp_path / "s")


def test_a_later_migration_reaches_an_existing_index(tmp_path, monkeypatch):
    import sqlite3

    import crucible.core.store as store_mod

    first = Store(tmp_path / "s")
    key = first.put_bytes(b"x")
    first.tag("latest", key)
    v = store_mod.INDEX_VERSION
    monkeypatch.setattr(store_mod, "_MIGRATIONS", [*store_mod._MIGRATIONS, ("ALTER TABLE tags ADD COLUMN rev INTEGER NOT NULL DEFAULT 0",)])
    monkeypatch.setattr(store_mod, "INDEX_VERSION", v + 1)
    s = Store(tmp_path / "s")
    with sqlite3.connect(tmp_path / "s" / "index.sqlite") as c:
        assert c.execute("SELECT key, rev FROM tags").fetchall() == [(key, 0)]
    assert _user_version(tmp_path / "s") == v + 1 and s.resolve("latest") == key


# ------------------------------------------------------------------------------------------------ R-13: staged plugins
class _FakeDist:
    def __init__(self, name):
        self.name, self.version = name, "1.0"


class _FakeEP:
    def __init__(self, name, fn):
        self.name, self._fn, self.dist = name, fn, _FakeDist(f"pack-{name}")

    def load(self):
        return self._fn


def _half(reg):
    reg.register("pack", "half_a", "m:a")
    raise RuntimeError("bug in register()")


def _alpha(reg):
    reg.register("grader", "shared", "alpha:G")
    reg.register("grader", "alpha_only", "alpha:A")


def _beta(reg):
    reg.register("grader", "shared", "beta:G")


def _plugins(monkeypatch, *eps):
    monkeypatch.setattr("crucible.core.registry.entry_points", lambda group: list(eps))
    reg = Registry()
    reg.load_plugins()
    return reg


def test_a_plugin_that_fails_halfway_registers_nothing(monkeypatch):
    reg = _plugins(monkeypatch, _FakeEP("half", _half))
    assert not reg.has("pack", "half_a") and "bug in register()" in reg.failures["half"]


@pytest.mark.parametrize("order", [("alpha", "beta"), ("beta", "alpha")])
def test_a_contested_name_is_withheld_whatever_the_install_order(monkeypatch, order):
    fns = {"alpha": _alpha, "beta": _beta}
    reg = _plugins(monkeypatch, *(_FakeEP(n, fns[n]) for n in order))
    assert not reg.has("grader", "shared") and reg.has("grader", "alpha_only")
    assert "also claimed by pack-beta==1.0" in reg.failures["alpha"] and "also claimed by pack-alpha==1.0" in reg.failures["beta"]
    assert reg.describe()["grader"]["alpha_only"].endswith("[pack-alpha==1.0]")
    with pytest.raises(RegistryError, match="withheld"):
        reg.get("grader", "shared")


def test_a_plugin_cannot_take_a_built_in_name(monkeypatch):
    reg = Registry()
    reg.register("grader", "exact", "core:Exact")
    monkeypatch.setattr("crucible.core.registry.entry_points", lambda group: [_FakeEP("p", lambda r: r.register("grader", "exact", "p:X", replace=True))])
    reg.load_plugins()
    assert reg.describe()["grader"]["exact"].startswith("core:Exact") and "built in" in reg.failures["p"]
