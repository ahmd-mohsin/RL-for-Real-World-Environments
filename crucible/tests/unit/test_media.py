"""M1 item 8: images and audio in messages and instructions (``MediaRef``, ``World.put_media`` / ``hydrate``,
``media_parts`` in the LLM policy, multimodal single-turn prompts).

Checked: media reaches the policy with its bytes on live observations only (never in traces or snapshots); a message
between actors carries attachments; vision and audio models get OpenAI content parts and text models a placeholder;
token estimates count media; a single-turn row with an image (chat ``image_url`` or Responses ``input_image``) or
audio (``input_audio``) delivers it to the model.
"""

from __future__ import annotations

import base64
import json

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, approx_tokens, media_parts
from crucible.core.schema.actions import MediaRef, Message
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.inject import FakeSpend
from crucible.runtime.run import Runner, RunSpec
from crucible.runtime.single_turn import load_tasks, single_turn_policy
from crucible.world.world import World

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def _spec():
    return WorldSpec.model_validate({"world": "t/media", "apps": [],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": []}, {"id": "coach", "kind": "policy", "tools": []}]})


async def test_instruction_media_and_message_attachments(tmp_path):
    w = World(_spec(), instructions={"agent": "What is in the picture?"}, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    img = w.put_media("image", "image/png", PNG)
    w.instruction_media["agent"] = [img]
    await w.start()
    try:
        obs = await w.reset()
        (m,) = obs["agent"].instruction_media
        assert base64.b64decode(m.data_b64) == PNG and m.ref == img.ref  # live observation: bytes attached
        assert obs["coach"].instruction_media is None
        clip = w.put_media("audio", "audio/wav", WAV)
        await w.step("coach", Message(to="agent", text="listen", attachments=[clip]))
        r = await w.step("agent", Message(to="coach", text="ok"))
        (inb,) = r.observation.inbox
        assert inb.attachments[0].kind == "audio" and base64.b64decode(inb.attachments[0].data_b64) == WAV
        trace = w.to_trace().model_dump_json()
        snap = (await w.snapshot()).model_dump_json()
        b64 = base64.b64encode(PNG).decode()
        assert b64 not in trace and base64.b64encode(WAV).decode() not in trace and b64 not in snap  # refs only
        assert clip.ref in trace  # the trace still says which clip was sent
    finally:
        await w.close()


async def test_no_media_keeps_snapshots_unchanged(tmp_path):
    w = World(_spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        await w.step("coach", Message(to="agent", text="hi"))
        ref = await w.snapshot()
        assert "attachments" not in json.dumps(ref.inboxes)  # fingerprints of media-free worlds are what they were
    finally:
        await w.close()


def test_media_parts_and_tokens():
    img = MediaRef(kind="image", mime="image/png", ref="k1", data_b64=base64.b64encode(PNG).decode())
    aud = MediaRef(kind="audio", mime="audio/wav", ref="k2", data_b64=base64.b64encode(b"\0" * 64000).decode())
    remote = MediaRef(kind="image", mime="image/*", url="https://example.test/cat.png")
    assert media_parts("q", None, vision=True, audio=True) == "q"
    parts = media_parts("q", [img, aud, remote], vision=True, audio=True)
    assert [p["type"] for p in parts] == ["text", "image_url", "input_audio", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,") and parts[2]["input_audio"]["format"] == "wav"
    assert parts[3]["image_url"]["url"] == "https://example.test/cat.png"
    text_only = media_parts("q", [img, aud], vision=False, audio=False)
    assert [p["text"] for p in text_only[1:]] == ["[image image/png omitted: this model receives text only]", "[audio audio/wav omitted: this model receives text only]"]
    est = approx_tokens([{"role": "user", "content": parts}])
    assert est >= 768 + 50  # an image tile plus two seconds of audio, not the base64 length
    assert MediaRef.model_validate({"kind": "image", "mime": "image/png", "ref": "k", "data_b64": "AAAA"}).model_dump().get("data_b64") is None
    with pytest.raises(ValueError, match="exactly one"):
        MediaRef(kind="image", mime="image/png")


class EchoLLM:
    model = "echo"

    def __init__(self, reply: str):
        self.reply, self.seen, self.spend = reply, [], FakeSpend()

    async def complete(self, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        return LLMResponse(text=self.reply, usage={"prompt_tokens": 10, "completion_tokens": 5}, model=self.model)


async def _run(tmp_path, tasks, pol, rid):
    await Runner(tmp_path / "runs").run(RunSpec(run_id=rid, tasks=tasks, policies=[pol], harnesses=[NativeHarness("native:single", limits=EpisodeLimits(max_steps=1))], k=1))
    return [json.loads(f.read_text()) for f in (tmp_path / "runs" / rid / "samples").glob("*/a1/record.json")]


async def test_multimodal_single_turn_rows(tmp_path):
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    rows = [{"id": "img-chat", "prompt": [{"role": "user", "content": [{"type": "text", "text": "Color of the pixel?"}, {"type": "image_url", "image_url": {"url": data_url}}]}],
             "answer": "red"},
            {"id": "img-resp", "responses_create_params": {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Color?"},
                                                                                                    {"type": "input_image", "image_url": data_url}]}]},
             "expected_answer": "red"},
            {"id": "audio", "prompt": [{"role": "user", "content": [{"type": "text", "text": "Transcribe."},
                                                                    {"type": "input_audio", "input_audio": {"data": base64.b64encode(WAV).decode(), "format": "wav"}}]}],
             "answer": "red"}]
    tasks = load_tasks(rows, dataset="mm")
    assert [len(t.media) for t in tasks] == [1, 1, 1] and tasks[0].instruction == "Color of the pixel?"
    llm = EchoLLM("red")
    recs = await _run(tmp_path, tasks, single_turn_policy(llm, vision=True, audio=True), "mm")
    assert sorted(r["reward"] for r in recs) == [1.0, 1.0, 1.0]
    kinds = sorted(p["type"] for call in llm.seen for p in call[-1]["content"] if isinstance(call[-1]["content"], list) and p["type"] != "text")
    assert kinds == ["image_url", "image_url", "input_audio"]
    blind = EchoLLM("red")
    await _run(tmp_path, tasks[:1], single_turn_policy(blind), "blind")
    assert "omitted" in json.dumps(blind.seen[0][-1]["content"])  # a text model is told an image was there
    for f in (tmp_path / "runs" / "mm").rglob("*.json"):
        assert base64.b64encode(PNG).decode() not in f.read_text(), f  # no media bytes in any record, trace or trajectory

