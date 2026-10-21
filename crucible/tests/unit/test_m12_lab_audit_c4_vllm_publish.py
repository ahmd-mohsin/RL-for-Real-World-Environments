"""M12 lab audit C-4 (the small item): ``VLLMServerEngine.version`` was client-side only, the publishers moved their
version before the engines loaded, and a failed ``v{n}`` directory was never collected.

- ``VLLMServerEngine.health`` reads the workers' version back (``CrucibleWorkerExtension.crucible_version``): a vLLM that
  restarted, or whose tensor-parallel workers disagree, now reports -1 (no published version), so ``GatewayPool`` evicts
  it and replays the publish. A failed read-back, a paused or sleeping engine or an older extension keep the client-side
  version (they never evict); an older extension still loads (the version argument is dropped).
- ``InprocPublisher`` / ``CheckpointPublisher``: ``version`` and ``published_updates`` move only after every engine
  loaded; a failed attempt's number is never handed out again.
- ``CheckpointPublisher``: a failed version's directory is collected by age like any other."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from crucible.train.vllm_http import VLLMServerEngine


def _engine(workers: list, *, rpc_status: int = 200, up: bool = True, old_ext: bool = False, on_version=None, fail: tuple = ()):
    """A mock vLLM whose workers hold ``workers`` (each worker's ``_crucible_version``); a load sets them all.
    ``old_ext``: the extension before C-4 (``crucible_load_weights(path)``, no ``crucible_version``); ``fail``: paths
    that answer 500."""
    log: list[tuple[str, object]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        if not up:
            raise httpx.ConnectError("refused", request=req)
        body = json.loads(req.content) if req.content else {}
        log.append((req.url.path, body.get("method"), *(body.get("args") or [])))
        if req.url.path in fail:
            return httpx.Response(500, json={"detail": "boom"})
        if req.url.path == "/collective_rpc":
            if body["method"] == "crucible_load_weights":
                if old_ext and len(body["args"]) != 1:
                    return httpx.Response(500, json={"detail": "TypeError: takes 2 positional arguments but 3 were given"})
                workers[:] = [body["args"][1] if len(body["args"]) > 1 else None] * len(workers)
                return httpx.Response(200, json={"results": ["loaded"] * len(workers)})
            if body["method"] == "crucible_version":
                if on_version is not None:
                    on_version()
                if old_ext or rpc_status != 200:
                    return httpx.Response(old_ext and 500 or rpc_status, json={"detail": "no method crucible_version"})
                return httpx.Response(200, json={"results": [{"version": v} for v in workers]})
        return httpx.Response(200, json={})

    eng = VLLMServerEngine("http://vllm", "m", placement="split")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return eng, log


def _versions_read(log) -> int:
    return sum(1 for e in log if e[:2] == ("/collective_rpc", "crucible_version"))


async def test_load_sends_the_version_and_health_reads_it_back():
    workers = [None, None]
    eng, log = _engine(workers)
    await eng.load_weights("/ckpt/v3", 3)
    assert workers == [3, 3] and eng.version == 3
    assert ("/collective_rpc", "crucible_load_weights", "/ckpt/v3", 3) in log
    assert await eng.health() and eng.version == 3 and _versions_read(log) == 1


async def test_a_restarted_server_names_no_published_version():
    workers = [None, None]
    eng, _ = _engine(workers)
    await eng.load_weights("/ckpt/v3", 3)
    workers[:] = [None, None]  # the server restarted: nothing loaded since, its base weights
    assert await eng.health() and eng.version == -1


async def test_nothing_loaded_yet_is_the_base_version():
    eng, _ = _engine([None])  # this client never loaded: the server's base weights are version 0
    assert await eng.health() and eng.version == 0


async def test_disagreeing_or_torn_workers_name_no_published_version():
    eng, _ = _engine([3, 4])  # a partial load across tensor-parallel ranks
    eng.version = 4
    assert await eng.health() and eng.version == -1  # healthy, so the gateway evicts it and replays the publish
    eng, _ = _engine([-1, -1])  # a load that raised part-way
    eng.version = 4
    assert await eng.health() and eng.version == -1


async def test_a_failed_read_back_is_not_an_outage():
    eng, _ = _engine([3], rpc_status=503)
    eng.version = 3
    assert await eng.health() and eng.version == 3


async def test_an_older_extension_still_loads_and_keeps_the_client_side_version():
    workers = [None]
    eng, log = _engine(workers, old_ext=True)
    await eng.load_weights("/ckpt/v2", 2)
    assert eng.version == 2 and ("/collective_rpc", "crucible_load_weights", "/ckpt/v2") in log
    await eng.load_weights("/ckpt/v3", 3)  # no second try with the version
    assert [e for e in log if e[1] == "crucible_load_weights"][-1] == ("/collective_rpc", "crucible_load_weights", "/ckpt/v3")
    assert await eng.health() and eng.version == 3 and _versions_read(log) == 0


async def test_a_server_that_is_down_is_unhealthy():
    eng, _ = _engine([1], up=False)
    assert not await eng.health()


async def test_no_read_back_while_loading_paused_asleep_or_after_an_incomplete_load():
    workers = [1]
    eng, log = _engine(workers)
    eng.version = 1
    eng._loads = 1
    assert await eng.health() and eng.version == 1
    eng._loads = 0
    eng._unpaused.clear()  # paused for a barrier
    assert await eng.health() and eng.version == 1
    eng._unpaused.set()
    eng.asleep = {"kv_cache"}
    assert await eng.health() and eng.version == 1
    assert _versions_read(log) == 0
    eng.asleep = set()
    eng, log = _engine([None], fail=("/reset_prefix_cache",))
    with pytest.raises(httpx.HTTPStatusError):
        await eng.load_weights("/ckpt/v2", 2)  # the workers hold v2, the client still names v0
    assert await eng.health() and eng.version == 0 and _versions_read(log) == 0  # behind: the gateway replays v2


async def test_a_load_that_starts_during_the_read_back_wins():
    eng = None

    def start_load():
        eng._loads += 1  # a publish began while the read-back was on the wire

    workers = [7]
    eng, _ = _engine(workers, on_version=start_load)
    eng.version = 9
    assert await eng.health() and eng.version == 9


async def test_the_gateway_replays_the_publish_to_a_restarted_vllm():
    from crucible.train.gateway import GatewayPool

    workers = [None]
    eng, log = _engine(workers)
    gw = GatewayPool([eng])
    await gw.load_weights("/ckpt/v2", 2)
    assert eng.version == 2 and workers == [2]
    workers[:] = [None]  # restarted between probes
    log.clear()
    await gw.check_health()
    assert gw.stats["evictions"] == 1 and gw.stats["revived"] == 1 and gw.alive == [True]
    assert ("/collective_rpc", "crucible_load_weights", "/ckpt/v2", 2) in log and workers == [2] and eng.version == 2


# ---- publishers -------------------------------------------------------------------------------------------------------
class _Trainer:
    def __init__(self):
        self.updates = 0

    def state_dict(self):
        torch = pytest.importorskip("torch")
        return {"w": torch.full((2,), float(self.updates))}

    def reset_optimizer(self):
        pass


class _Engines:
    def __init__(self, fail_at: set[int]):
        self.fail_at, self.loaded = fail_at, []

    async def load_weights(self, state, version):
        await asyncio.sleep(0)
        if version in self.fail_at:
            raise RuntimeError(f"engine failed to load v{version}")
        self.loaded.append(version)


async def test_inproc_publisher_moves_only_after_the_load():
    from crucible.train.weights import InprocPublisher

    tr, eng = _Trainer(), _Engines({2})
    pub = InprocPublisher(tr, eng)
    tr.updates = 1
    assert await pub.publish() == 1 and pub.published_updates == 1
    tr.updates = 2
    with pytest.raises(RuntimeError):
        await pub.publish()
    assert pub.version == 1 and pub.published_updates == 1  # still names what the engines serve
    assert await pub.publish() == 3 and eng.loaded == [1, 3]  # v2's number is never reused
    assert pub.published_updates == 2


async def test_checkpoint_publisher_collects_a_failed_version_by_age(tmp_path: Path):
    from crucible.train.weights import CheckpointPublisher

    tr = _Trainer()
    pub = CheckpointPublisher(tr, _Engines({2}), tmp_path)  # keep=2
    await pub.publish()
    with pytest.raises(RuntimeError):
        await pub.publish()
    assert not pub._loading and pub.version == 1
    await pub.publish()  # v3: v1 goes; keep=2, so the failed v2 is not yet 2 behind
    assert sorted(d.name for d in tmp_path.iterdir()) == ["v2", "v3"]
    await pub.publish()  # v4: the failed v2 goes too (it stayed forever before)
    assert sorted(d.name for d in tmp_path.iterdir()) == ["v3", "v4"] and sorted(pub.deleted) == ["v1", "v2"]


def test_the_two_extension_copies_are_identical():
    root = Path(__file__).resolve().parents[2]
    a = (root / "src/crucible/train/vllm_ext.py").read_bytes().replace(b"\r\n", b"\n")
    b = (root / "packs/_toy/src/crucible_pack_toy/m12/_ext/crucible_vllm_ext.py").read_bytes().replace(b"\r\n", b"\n")
    assert a == b
