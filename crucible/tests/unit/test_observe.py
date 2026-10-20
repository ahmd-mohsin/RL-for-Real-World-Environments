"""crucible.observe (swarm-pipeline/11): W&B runs and Weave traces, offline, redacted, through one module. No network.

- **naming**: the three projects, run names, groups, tags, deterministic UUIDv7 call ids and call links;
- **the gateway**: an ``ObsRun`` against a fake ``wandb`` (what reaches W&B: offline mode, console/code/git/meta capture
  off, tables, charts, HTML, artifacts) and its redacted mirror; one real ``wandb`` offline run (sync later);
- **training**: ``factory.tinker.train`` on the fake Tinker logs every step through the trainer's sink (reward, Φ,
  gates, pass rate, advantages, loss, KL / ratio, staleness, drops, tokens, USD vs cap, sampler version), the trained
  episodes as traces, the adapter's ``tinker://`` path as artifact metadata;
- **benchmarks**: ``compare`` logs the headline chart with CIs, the paired difference, the protocol check, per-task and
  per-language tables, pass@k curves; every episode is a trace (episode -> turns -> tool calls, Φ per turn) with a link;
- **audit**: a failure finding attaches to its episode's trace (the bundle id is the episode key) and its table row links it;
- **redaction**: a planted key (a vendor-format key in a tool call and its output, and a prefix-less ``WANDB_API_KEY``
  value) appears nowhere in what is logged, mirrored, exported or handed to W&B;
- **sync**: the Weave replay keeps the call ids and timestamps (checked against the installed weave's signatures);
- **backfill**: the repo's run reports become factory views, once (ledger);
- **one module**: no ``wandb`` / ``weave`` import outside ``crucible.observe``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_p6_equiv import HostWindow, scripts
from test_m17_w_task import make_task

from crucible.factory.checkpoint import CheckpointConfig
from crucible.factory.envs import Timeouts
from crucible.observe import naming, redact
from crucible.observe.run import ObserveConfig, ObsRun, replay
from crucible.observe.traces import TraceStore, attempt_dirs, render_text, uuid7
from crucible.train.bridges.tinker.fake import FakeTinker, ScriptedResponder
from crucible.train.chat import ByteChatCodec

MODEL = "Qwen/Qwen3.6-35B-A3B"
SRC = Path(__file__).resolve().parents[2] / "src" / "crucible"
RUNS = Path(__file__).resolve().parents[2] / "docs" / "research" / "swarm-pipeline" / "runs"
PLANTED_VENDOR = "sk-or-v1-" + "0123456789abcdefPLANTEDKEY0123456789abcdef"
PLANTED_ENV = "wbPLANTEDvalue" + "9f8e7d6c5b4a39281706f5e4d3c2b1a0"  # no vendor prefix: only removal by value catches it


# ---- a fake wandb: what would reach W&B --------------------------------------------------------------------------------------
class FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.calls: list[tuple] = []
        fw = self

        class Settings(dict):
            def __init__(self, **kw):
                super().__init__(kw)

        class Table:
            def __init__(self, columns=None, data=None, allow_mixed_types=False):
                self.columns, self.data = columns, data

        class Html:
            def __init__(self, data, inject=True, data_is_not_path=False):
                self.data = data

        class Artifact:
            def __init__(self, name, type, metadata=None):
                self.name, self.type, self.metadata, self.files = name, type, metadata, {}

            def add_dir(self, d):
                for p in Path(d).iterdir():
                    self.files[p.name] = p.read_bytes().decode(errors="replace")

        class Summary(dict):
            pass

        class Run:
            def __init__(self, kw):
                self.kw, self.summary = kw, Summary()

            def log(self, d, step=None):
                fw.calls.append(("log", {k: (getattr(v, "data", None), getattr(v, "columns", None)) if not isinstance(v, int | float | str) else v
                                         for k, v in d.items()}, step))

            def log_artifact(self, art, aliases=None):
                fw.calls.append(("artifact", art.name, art.type, art.metadata, art.files, aliases))

            def finish(self, exit_code=0):
                fw.calls.append(("finish", exit_code, dict(self.summary)))

        def init(**kw):
            fw.calls.append(("init", kw))
            return Run(kw)

        plot = SimpleNamespace(bar=lambda t, label, value, title="": Table(t.columns, t.data),
                               line_series=lambda xs, ys, keys=None, title="", xname="x": Table(["xs", "ys", "keys"], [[xs, ys, keys]]))
        self.Settings, self.Table, self.Html, self.Artifact, self.init, self.plot = Settings, Table, Html, Artifact, init, plot

    def dump(self) -> str:
        return json.dumps(self.calls, default=str)


@pytest.fixture
def obs(tmp_path, monkeypatch):
    """An observe root in tmp, a fixed redaction salt, a planted credential in the environment, and the fake wandb."""
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "test-salt-observe")
    monkeypatch.setenv("CRUCIBLE_OBSERVE_DIR", str(tmp_path / "observe"))
    monkeypatch.setenv("WANDB_API_KEY", PLANTED_ENV)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("CRUCIBLE_WANDB_ENTITY", raising=False)
    monkeypatch.delenv("CRUCIBLE_WANDB_MODE", raising=False)
    fw = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fw)
    redact.reset_default()
    yield SimpleNamespace(root=tmp_path / "observe", fw=fw, cfg=ObserveConfig(root=tmp_path / "observe"))
    redact.reset_default()


def everything(root: Path, fw: FakeWandb | None = None) -> str:
    """Every byte written under ``root`` plus every payload handed to (fake) W&B."""
    parts = [p.read_bytes().decode(errors="replace") for p in root.rglob("*") if p.is_file()]
    return "\n".join(parts) + (fw.dump() if fw else "")


# ---- naming -------------------------------------------------------------------------------------------------------------------
def test_naming_projects_names_tags_and_links(monkeypatch) -> None:
    monkeypatch.delenv("CRUCIBLE_WANDB_PROJECT_PREFIX", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("CRUCIBLE_WANDB_ENTITY", raising=False)
    assert {k: naming.project(k) for k in ("factory", "train", "bench")} == {"factory": "crucible-factory", "train": "crucible-train", "bench": "crucible-bench"}
    assert naming.short_model("tinker:Qwen/Qwen3.6-35B-A3B:peft:262144") == "qwen3.6-35b-a3b"
    assert naming.bench_name("control", MODEL, 1000, "abcdef123456") == "bench-control-qwen3.6-35b-a3b-s1000-abcdef12"
    assert naming.tags(model=MODEL, purpose="trained", lang=["python", "cuda"], k=None) == ["model:qwen3.6-35b-a3b", "purpose:trained", "lang:python", "lang:cuda"]
    assert naming.manifest_hash(["b", "a"]) == naming.manifest_hash(["a", "b"])
    cid = uuid7(1_792_000_000_000, "x")
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", cid) and cid == uuid7(1_792_000_000_000, "x")
    assert uuid7(1_792_000_000_000, "y") != cid and int(cid.replace("-", "")[:12], 16) == 1_792_000_000_000  # time-ordered like Weave's ids
    assert naming.call_url("crucible-bench", cid) == f"weave:crucible-bench/call/{cid}"  # no entity yet: a placeholder
    monkeypatch.setenv("WANDB_ENTITY", "lab")
    assert naming.call_url("crucible-bench", cid) == f"https://wandb.ai/lab/crucible-bench/r/call/{cid}"
    with pytest.raises(ValueError):
        naming.project("other")


# ---- the gateway ----------------------------------------------------------------------------------------------------------------
def test_obsrun_hands_wandb_only_redacted_offline_payloads_and_mirrors_them(obs) -> None:
    secret_cfg = {"model": MODEL, "api_key": "hunter2hunter2", "note": f"Bearer {PLANTED_VENDOR}", "author_email": "dev@corp.example.com"}
    with ObsRun("crucible-bench", "t1", group="g", tags=["model:x"], config=secret_cfg, cfg=obs.cfg) as run:
        run.log({"a": 1, "nested": {"b": 2.5}, "flag": True, "nan": float("nan"), "text": "x"}, step=3)
        run.table("tab", ["k", "v"], [["row", f"key={PLANTED_ENV}"], ["obj", {"z": 1}]])
        run.bar("bar", ["x", "y"], [0.5, None], title="t")
        run.lines("ln", [1, 2, 4], [[0.1, 0.2, None]], ["pass@k"], title="p")
        run.html("h", f"<p>{PLANTED_VENDOR}</p>")
        run.artifact("art", "report", files={"r.json": {"token": PLANTED_ENV, "ok": 1}}, metadata={"tinker_path": "tinker://abc/sampler_weights/s1"})
        run.record("drop", {"reason": "stale"}, 2)
        run.record("rollout", {"huge": "x" * 10}, 2)  # rollout bodies never become W&B rows
        run.summary({"pass@1": 0.25, "d": {"e": 1}})
    init = next(c for c in obs.fw.calls if c[0] == "init")[1]
    assert init["mode"] == "offline" and init["project"] == "crucible-bench" and init["group"] == "g" and init["job_type"] is None
    s = init["settings"]
    assert s["console"] == "off" and s["save_code"] is False and s["disable_git"] is True and s["x_disable_meta"] is True and s["x_save_requirements"] is False
    assert init["config"]["api_key"] == "<secret>" and "corp.example.com" not in json.dumps(init["config"])
    logs = [c for c in obs.fw.calls if c[0] == "log"]
    assert logs[0][1] == {"a": 1.0, "nested/b": 2.5, "flag": 1.0} and logs[0][2] == 3  # numbers only, flat keys
    keys = {k for c in logs for k in c[1]}
    assert {"tab", "bar", "ln", "h", "records/drop"} <= keys and "records/rollout" not in keys
    art = next(c for c in obs.fw.calls if c[0] == "artifact")
    assert art[3] == {"tinker_path": "tinker://abc/sampler_weights/s1"} and "<secret>" in art[4]["r.json"]
    fin = next(c for c in obs.fw.calls if c[0] == "finish")
    assert fin[1] == 0 and fin[2]["pass@1"] == 0.25 and fin[2]["d/e"] == 1
    ops = [json.loads(x)["op"] for x in (run.dir / "ops.jsonl").read_text().splitlines()]
    assert ops[0] == "init" and ops[-1] == "finish" and {"log", "table", "chart", "html", "artifact", "summary"} <= set(ops)
    assert (run.dir / "media" / "h.html").exists()
    blob = everything(obs.root, obs.fw)
    assert PLANTED_VENDOR not in blob and PLANTED_ENV not in blob and "hunter2hunter2" not in blob


def test_a_mirror_replays_into_a_wandb_run(obs) -> None:
    cfg = ObserveConfig(root=obs.root, use_wandb=False)
    with ObsRun("crucible-factory", "m", cfg=cfg, config={"x": 1}) as run:
        run.log({"a": 1.0}, step=0)
        run.table("t", ["c"], [[1]])
        run.artifact("a", "report", files={"f.json": {"v": 1}})
    assert not obs.fw.calls  # use_wandb=False: the mirror only
    r = replay(run.dir, cfg=ObserveConfig(root=obs.root / "replayed"))
    kinds = [c[0] for c in obs.fw.calls]
    assert kinds[0] == "init" and "artifact" in kinds and kinds[-1] == "finish"
    assert next(c for c in obs.fw.calls if c[0] == "init")[1]["id"] == r.id == run.id


def test_real_wandb_offline_run_is_written_for_a_later_sync(tmp_path, monkeypatch) -> None:
    wandb = pytest.importorskip("wandb")
    if isinstance(wandb, FakeWandb):
        pytest.skip("fake wandb installed")
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "test-salt-observe")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    redact.reset_default()
    with ObsRun("crucible-test", "real", config={"k": 8}, cfg=ObserveConfig(root=tmp_path)) as run:
        run.log({"pass@1": 0.25}, step=0)
        run.table("t", ["task", "pass@1"], [["a", 0.5]])
        run.bar("b", ["a"], [0.5], title="b")
        run.html("h", "<b>x</b>")
        run.artifact("m", "manifest", files={"m.json": {"tasks": ["a"]}})
    assert run.mode == "offline" and len(run.sync_dirs) == 1 and list(run.sync_dirs[0].glob("run-*.wandb"))
    redact.reset_default()


# ---- training, benchmarks, traces (the fake Tinker on the toy window) ---------------------------------------------------------
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("observetoy")
    task, rec, _ = make_task(tmp)
    task.__class__ = HostWindow
    sc = scripts(rec)
    sc["leaky"] = [("write", "notes.txt", f"deploy token {PLANTED_VENDOR} and {PLANTED_ENV}\n"), ("shell", "cat notes.txt"), ("final", "Nothing else.")]
    return SimpleNamespace(task=task, rec=rec, sc=sc, tmp=tmp)


def _kw():
    return {"timeouts": Timeouts(agent_s=300, rollout_s=900), "phi_checkpoint": CheckpointConfig(every=2, cap=20, concurrency=1)}


def test_training_logs_every_step_through_the_sink_and_exports_traces(toy, obs, tmp_path) -> None:
    from crucible.factory.tinker import train
    from crucible.train.bridges.tinker.trainer import TrainConfig

    HostWindow._dead = False
    codec = ByteChatCodec()
    fake = FakeTinker(responder=ScriptedResponder([toy.sc[n] for n in ("pass", "fail", "partial", "leaky")], codec))
    cfg = TrainConfig(model=MODEL, group_size=2, steps=2, turn_eta=0.5, require_band=False, cap_usd=1.0, drop_zero_spread=False)
    out = asyncio.run(train([toy.task], config=cfg, run_dir=tmp_path / "train", backend=fake, codec=codec, observe=obs.cfg, **_kw()))
    assert out["steps_done"] == 2 and out["observe"]["traces"] == 4
    run_dir = Path(out["observe"]["run"])
    ops = [json.loads(x) for x in (run_dir / "ops.jsonl").read_text().splitlines()]
    init = ops[0]
    assert init["meta"]["project"] == "crucible-train" and init["meta"]["job_type"] == "rl" and init["meta"]["group"].startswith("objective-obj-")
    c = init["config"]
    assert c["model"] == MODEL and c["lora_rank"] == 32 and c["loss_fn"] == "importance_sampling" and c["group_size"] == 2 and c["cap_usd"] == 1.0
    assert c["enable_thinking"] is False and c["objective_id"] == out["objective_id"] and c["tasks_manifest_hash"] and c["agent_s"] == 300
    steps = [o for o in ops if o["op"] == "log"]
    assert [o["step"] for o in steps] == [1, 2]
    m = steps[0]["data"]
    for k in ("reward/mean", "reward/std", "phi/final_mean", "gates/pass_rate", "pass_rate", "advantage/sequence/mean", "advantage/token/mean", "train/loss",
              "kl/sample_vs_train", "ratio/mean", "staleness/max", "drops/total", "tokens/sampled", "tokens/train", "tokens/trained", "usd/step", "usd/cum",
              "usd/cap", "usd/cap_frac", "sampler/version", "time/train_s"):
        assert k in m, k
    assert m["usd/cap"] == 1.0 and m["sampler/version"] == 1 and steps[1]["data"]["sampler/version"] == 2
    assert steps[1]["data"]["usd/cum"] == pytest.approx(steps[0]["data"]["usd/cum"] + steps[1]["data"]["usd/step"])
    arts = {o["name"]: o for o in ops if o["op"] == "artifact"}
    adapter = next(a for n, a in arts.items() if n.startswith("adapter"))
    assert adapter["metadata"]["tinker_path"] == out["sampler_path"] and adapter["aliases"] == ["latest"]
    tr = next(a for n, a in arts.items() if n.startswith("train-run"))
    assert {"config.json", "receipt.json", "steps.jsonl", "spend.jsonl", "versions.json"} <= set(tr["files"])
    assert any(o["op"] == "table" and o["key"] == "train/episodes" for o in ops)
    store = TraceStore(obs.root, "crucible-train")
    assert len(store.index()) == 4
    blob = everything(obs.root, obs.fw)
    assert PLANTED_VENDOR not in blob and PLANTED_ENV not in blob  # the leaky episode's tool call and output were redacted
    assert "deploy token <secret>" in blob


@pytest.fixture(scope="module")
def compared(toy, tmp_path_factory):
    """A comparison (control fails, trained passes) logged with the fake wandb; its root and calls."""
    from crucible.factory.tinker import compare

    tmp = tmp_path_factory.mktemp("observecmp")
    mp = pytest.MonkeyPatch()
    mp.setenv("CRUCIBLE_REDACTION_SALT", "test-salt-observe")
    mp.setenv("WANDB_API_KEY", PLANTED_ENV)
    mp.delenv("WANDB_ENTITY", raising=False)
    fw = FakeWandb()
    mp.setitem(sys.modules, "wandb", fw)
    redact.reset_default()
    HostWindow._dead = False
    codec = ByteChatCodec()
    path = "tinker://fake-run/sampler_weights/s000001"
    fake = FakeTinker(responder=ScriptedResponder([toy.sc[n] for n in ("fail", "partial", "pass", "pass")], codec), store={"sampler_paths": {path}})
    cfg = ObserveConfig(root=tmp / "observe")
    try:
        out = asyncio.run(compare([toy.task], model=MODEL, checkpoint=path, run_dir=tmp / "cmp", cap_usd=1.0, k=2, seed=7, n_boot=200, backend=fake,
                                  codec=codec, concurrency=1, observe=cfg, **_kw()))
    finally:  # the fake wandb and the planted key only for the run itself
        mp.undo()
        redact.reset_default()
    return SimpleNamespace(out=out, root=tmp / "observe", fw=fw, cfg=ObserveConfig(root=tmp / "observe", use_wandb=False), run_dir=tmp / "cmp")


def _ops(run: str) -> list[dict]:
    return [json.loads(x) for x in (Path(run) / "ops.jsonl").read_text().splitlines()]


def test_compare_logs_the_headline_paired_difference_protocol_and_breakdowns(compared) -> None:
    out = compared.out
    ops = _ops(out["observe"]["run"])
    init = ops[0]["meta"]
    assert init["project"] == "crucible-bench" and init["job_type"] == "compare" and init["group"] == out["observe"]["group"]
    tables = {o["key"]: o for o in ops if o["op"] == "table"}
    arms = {r[0]: r for r in tables["compare/arms"]["rows"]}
    assert set(arms) == {"control", "trained"} and arms["control"][1] == 0.0 and arms["trained"][1] == 1.0
    assert tables["compare/protocol"]["rows"] == [["no baseline arm"]]
    assert {"compare/per_task_control", "compare/per_task_trained", "compare/by_language", "compare/by_family", "compare/by_skill",
            "compare/paired_per_task"} <= set(tables)
    assert {r[0] for r in tables["compare/by_language"]["rows"]} == {"python"}
    logs = {k: v for o in ops if o["op"] == "log" for k, v in o["data"].items()}
    assert logs["compare/paired/mean_diff"] == pytest.approx(1.0) and logs["compare/protocol_ok"] == 0.0
    html = {o["key"] for o in ops if o["op"] == "html"}
    assert {"compare/headline", "compare/per_task_chart", "compare/pass_at_k_chart", "compare/pass_hat_k_chart"} <= html
    headline = (Path(out["observe"]["run"]) / "media" / "compare__headline.html").read_text()
    assert "control" in headline and "trained" in headline and "PROTOCOL: no baseline arm" in headline and "<svg" in headline
    charts_ = {o["key"]: o for o in ops if o["op"] == "chart"}
    assert charts_["compare/pass@1"]["rows"] == [["control", 0.0], ["trained", 1.0]] and charts_["compare/pass_at_k"]["names"] == ["control", "trained"]
    tpl = json.loads((Path(out["observe"]["run"]) / "report_template.json").read_text())
    logged = set(tables) | html | set(charts_)
    for b in tpl["layout"]:  # every panel of the report template names a key the comparison or an arm logs
        for p in b.get("panels", []):
            for k in p.get("keys", []) + ([p["key"]] if "key" in p else []):
                if k.startswith("compare/") and not k.startswith("compare/by_"):
                    assert k in logged, k
    for arm in ("control", "trained"):  # each arm is its own run in the same group, with its episodes table
        a_ops = _ops(out["arms"][arm]["observe"]["run"])
        assert a_ops[0]["meta"]["group"] == init["group"] and a_ops[0]["meta"]["job_type"] == arm
        ep = next(o for o in a_ops if o["op"] == "table" and o["key"] == "bench/episodes")
        assert len(ep["rows"]) == 2 and all(r[-1].startswith("weave:crucible-bench/call/") for r in ep["rows"])
        assert any(o["op"] == "artifact" and o["type"] == "benchmark-manifest" for o in a_ops)


def test_every_episode_is_a_trace_of_turns_and_tool_calls_with_phi(compared) -> None:
    store = TraceStore(compared.root, "crucible-bench")
    idx = store.index()
    assert len(idx) == 4 and {r["arm"] for r in idx.values()} == {"control", "trained"}
    key, row = next((k, r) for k, r in idx.items() if r["arm"] == "trained")
    spans = store.spans(key)
    root = next(s for s in spans if s.parent_id is None)
    assert root.op == "crucible_episode" and root.id == row["root"] and root.trace_id == root.id
    a = root.attributes["crucible"]
    assert a["task"] == str(compared.out["arms"]["trained"]["tasks"][0]) and a["arm"] == "trained" and a["reward"] == 1.0 and a["passed"] is True
    assert a["masked"] is False and a["stop"] and a["gates_passed"] is True and "gates" in a and a["model"]
    turns = [s for s in spans if s.op == "crucible_turn"]
    tools = [s for s in spans if s.op.startswith("tool_")]
    assert turns and tools and all(t.parent_id == root.id for t in turns) and all(t.parent_id in {x.id for x in turns} for t in tools)
    assert turns[-1].output["phi"]["after"] == pytest.approx(1.0)  # the passing episode ends at Φ = 1
    assert sum(t.output["phi"]["delta"] for t in turns) == pytest.approx(1.0) and any(t.output["phi"]["gained"] for t in turns)
    assert all("arguments" in t.inputs and "content" in t.output for t in tools)
    assert all(s.started_at <= s.ended_at for s in spans) and root.started_at <= min(s.started_at for s in spans)
    assert (store.dir / row["full"]).exists() and "prompt_token_ids" not in (store.dir / row["full"]).read_text()
    txt = render_text(spans)
    assert "PASS" in txt and "turn 1" in txt
    # re-export is idempotent: the same ids, no duplicate spans
    n = len(store.spans())
    store2 = TraceStore(compared.root, "crucible-bench")
    store2.export(Path(row["record_dir"]), arm="trained")
    assert len(store2.spans()) == n


def test_a_failure_finding_attaches_to_its_trace_and_its_row_links_it(compared) -> None:
    from crucible.factory.failures import finding
    from crucible.factory.traceaudit import load_bundle
    from crucible.observe import factory_views as fv

    store = TraceStore(compared.root, "crucible-bench")
    key, row = next((k, r) for k, r in store.index().items() if not r["passed"])
    adir = Path(row["record_dir"])
    run_id = json.loads((adir / "record.json").read_text())["run_id"]
    f = finding(load_bundle(adir, run=run_id))
    assert f.trace == key  # the trace audit's bundle id is the episode key
    with ObsRun("crucible-factory", "audit", cfg=compared.cfg) as run:
        rows = fv.findings(run, [f], store=store)
    assert rows[0][-1] == row["url"] and rows[0][4] == f.mode
    audit = [s for s in store.spans(key) if s.op == "crucible_trace_audit"]
    assert len(audit) == 1 and audit[0].parent_id == row["root"] and audit[0].output["quote"] == f.quote
    assert TraceStore(compared.root, "crucible-bench").index()[key]["audit"]["mode"] == f.mode  # the episodes table shows the verdict


def test_weave_sync_replays_spans_with_their_ids_and_times(compared) -> None:
    from crucible.observe.weave_sync import sync

    weave = pytest.importorskip("weave")
    from weave.trace.weave_client import WeaveClient

    sig_c, sig_f = inspect.signature(WeaveClient.create_call).parameters, inspect.signature(WeaveClient.finish_call).parameters
    assert {"_call_id_override", "started_at", "use_stack", "display_name", "attributes", "parent"} <= set(sig_c) and "ended_at" in sig_f
    assert weave is not None

    class FakeClient:
        def __init__(self):
            self.created, self.finished = [], []

        def create_call(self, op, inputs, parent=None, attributes=None, display_name=None, *, use_stack=True, _call_id_override=None, started_at=None):
            c = SimpleNamespace(id=_call_id_override, trace_id=parent.trace_id if parent else _call_id_override, parent=parent, _children=[])
            self.created.append((op, c, started_at, attributes))
            return c

        def finish_call(self, call, output=None, exception=None, *, op=None, ended_at=None):
            self.finished.append((call.id, ended_at))

    store = TraceStore(compared.root, "crucible-bench")
    spans = store.spans()
    fc = FakeClient()
    res = sync(store, client=fc)
    assert res["uploaded"] == len(spans) and res["ids_kept"] is True
    by_id = {c.id: (op, c) for op, c, _, _ in fc.created}
    assert set(by_id) == {s.id for s in spans}
    for s in spans:
        _, c = by_id[s.id]
        assert (c.parent.id if c.parent else None) == s.parent_id  # parents uploaded first, links kept
    assert sync(store, client=FakeClient())["uploaded"] == 0  # synced.json: nothing twice


def test_compare_artifacts_and_traces_hold_no_planted_secret(compared) -> None:
    blob = everything(compared.root, compared.fw)
    assert PLANTED_ENV not in blob and PLANTED_VENDOR not in blob


# ---- factory views, backfill --------------------------------------------------------------------------------------------------------
def test_factory_views_funnel_targeting_cards_and_spend(obs) -> None:
    from crucible.factory.cards import IntentCard, VerifierCard
    from crucible.factory.targeting import TargetEvidence, band, cause_mix, target_verdict
    from crucible.observe import factory_views as fv

    cfg = ObserveConfig(root=obs.root, use_wandb=False)
    with ObsRun("crucible-factory", "views", cfg=cfg) as run:
        rows = fv.funnel(run, {"window": {"windows": 10, "world_built": 9, "dependent": 3},
                               "swarm": {"S1_source": {"in": 5, "pass": 3, "reasons": {"1 usable request": 2}}, "S3_world": {"in": 3, "pass": 2, "reasons": {}}}})
        b = band([0.0, 0.0, 1.0, 0.25])
        c = cause_mix(["skill_gap", "skill_gap", "skill_gap", "budget"])
        v = target_verdict(TargetEvidence(task="t1", band=b, cause=c))
        trows = fv.targeting(run, {"t1": {"band": b, "cause": c, "verdict": v, "model": MODEL}})
        irows = fv.intent_cards(run, [IntentCard(id="i1", behaviour="keeps work", failure_mode="loses it", evidence=["o1"])])
        vrows = fv.verifier_cards(run, {"window": VerifierCard(verifier_version="v1", fp=(1, 50), fn=(0, 20), mutant_kill=(9, 10))})
        total = fv.spend(run, {"author": 0.25, "student": 1.0})
    assert [r[2] for r in rows] == ["windows", "world_built", "dependent", "S1_source", "S3_world"] and rows[3][5] == 2
    assert trows[0][3] == 0.25 and trows[0][4] is True and trows[0][7] == 0.75 and irows[0][1] == "candidate"
    assert vrows[0][4] == pytest.approx(0.02) and vrows[0][5] > 0.02 and vrows[0][10] == pytest.approx(0.9) and total == 1.25
    ops = _ops(str(run.dir))
    assert {o["key"] for o in ops if o["op"] == "chart"} >= {"funnel/window", "funnel/swarm", "spend/bar"}
    assert {o["key"] for o in ops if o["op"] == "html"} >= {"targeting/band_chart", "verifier_cards/chart"}


def test_backfill_turns_repo_reports_into_views_once(obs, tmp_path) -> None:
    from crucible.observe.backfill import backfill, discover

    src = tmp_path / "runs"
    src.mkdir()
    for n in ("kernel_funnel.json", "p2_report.json", "p4_report.json", "p3_seeded_report.json"):
        shutil.copy(RUNS / n, src / n)
    (src / "gen").mkdir()
    (src / "gen" / "funnel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"generator": "swarm", "candidate": "c1", "stage": "S1_source", "ok": True, "reason": ""},
        {"generator": "swarm", "candidate": "c2", "stage": "S1_source", "ok": False, "reason": "1 usable request(s)"},
        {"generator": "swarm", "candidate": "c1", "stage": "S3_world", "ok": True, "reason": ""}]))
    assert sorted(k for k, _ in discover([src])) == ["funnel", "report", "report", "report", "report"]
    cfg = ObserveConfig(root=obs.root, use_wandb=False)
    rows = backfill([src], cfg=cfg)
    views = {Path(r["source"]).name: r.get("views") for r in rows}
    assert "funnel" in views["kernel_funnel.json"] and "spend" in views["kernel_funnel.json"]
    assert "verifier_cards" in views["p2_report.json"] and "funnel" in views["p2_report.json"]
    assert "targeting" in views["p4_report.json"] and "audit" in views["p3_seeded_report.json"]
    assert not any(r.get("error") for r in rows)
    again = backfill([src], cfg=cfg)
    assert all(r.get("skipped") for r in again)  # the ledger: once


def test_report_template_builds_offline() -> None:
    pytest.importorskip("wandb_workspaces")
    from crucible.observe.report import build

    r = build("lab", group="tasks-abc", model=MODEL)
    assert r.entity == "lab" and r.project == "crucible-bench" and len(r.blocks) >= 5


# ---- one module -----------------------------------------------------------------------------------------------------------------
def test_only_crucible_observe_imports_wandb_or_weave() -> None:
    rx = re.compile(r"^\s*(?:import|from)\s+(?:wandb|weave|wandb_workspaces)\b", re.MULTILINE)
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "observe" not in p.relative_to(SRC).parts[:1] and rx.search(p.read_text())]
    assert offenders == []
    scripts_dir = SRC.parents[1] / "scripts"
    assert [p.name for p in scripts_dir.glob("*.py") if rx.search(p.read_text())] == []


def test_the_trainer_sink_and_m12_wandb_sink_are_the_observe_gateway() -> None:
    from crucible.observe.run import WandbSink
    from crucible.train import sinks

    assert sinks.WandbSink is WandbSink and issubclass(WandbSink, ObsRun)
    from crucible.train.bridges.tinker import trainer

    assert "sinks" in inspect.signature(trainer.TinkerTrainer.__init__).parameters


def test_attempt_dirs_keep_only_the_counted_attempt(tmp_path) -> None:
    for a in ("a1", "a2"):
        d = tmp_path / "r" / "samples" / "s1" / a
        d.mkdir(parents=True)
        (d / "record.json").write_text("{}")
    assert [p.name for p in attempt_dirs(tmp_path)] == ["a2"]
