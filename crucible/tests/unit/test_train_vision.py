"""M12: a vision-language training codec (``chat.HFVisionChatCodec``) and log-prob recomputation from image refs
(``train/vision.py``).

Offline: a byte-level tokenizer with the Qwen2-VL special tokens, the real Qwen2-VL image processor and processor
expansion, a Qwen2-VL chat template, and a tiny random-init ``Qwen2VLForConditionalGeneration`` built from a config (no
download). A two-turn episode where the second turn brings a tool screenshot: the engine (a direct forward with the
processor's pixels) samples; the record keeps the engine's token ids and content-addressed refs only; ``VisionHFPolicy``
rebuilds the pixels from the refs and its log-probs equal the engine's. A real-model test runs when Qwen2-VL is cached.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL.Image")

from crucible.core.prov import utc_now
from crucible.core.schema.rollout import RolloutRecord, full_prompts
from crucible.core.schema.run import RunRecord
from crucible.runtime.record import build
from crucible.train.chat import HFVisionChatCodec, placeholder_runs
from crucible.train.engine import GenResult
from crucible.train.vision import VisionHFPolicy, prompt_media
from crucible.train.world_env import EngineLLM, attach_call_fields

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}"
            "{% else %}{% for c in message['content'] %}{% if c['type'] == 'image' or c['type'] == 'image_url' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def _processor():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLProcessor
    from transformers.processing_utils import ProcessorMixin

    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil as IP
    except ImportError:  # older transformers: one image processor class
        from transformers import Qwen2VLImageProcessor as IP

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    t = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    t.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=SPECIALS)

    class ImageOnlyQwen2VLProcessor(Qwen2VLProcessor):  # Qwen2-VL's processor without its video half (needs torchvision)
        def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kw):
            self.image_token, self.video_token = "<|image_pad|>", "<|video_pad|>"
            self.image_token_id, self.video_token_id = tokenizer.convert_tokens_to_ids(self.image_token), tokenizer.convert_tokens_to_ids(self.video_token)
            ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)

    return ImageOnlyQwen2VLProcessor(image_processor=IP(min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 16), tokenizer=tok, chat_template=TEMPLATE)


def _model(tok):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

    ids = {k: tok.convert_tokens_to_ids(k) for k in SPECIALS}
    cfg = Qwen2VLConfig(text_config={"vocab_size": len(tok), "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                                     "num_key_value_heads": 2, "bos_token_id": ids["<|endoftext|>"], "eos_token_id": ids["<|im_end|>"],
                                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1]}},
                        vision_config={"depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 2, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2},
                        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
                        vision_end_token_id=ids["<|vision_end|>"])
    torch.manual_seed(0)
    return Qwen2VLForConditionalGeneration(cfg).eval()


def _png(w: int, h: int, color) -> tuple[str, bytes, object]:
    im = PIL.new("RGB", (w, h), color)
    for x in range(0, w, 7):  # some structure, so the vision tower's output depends on the pixels
        for y in range(h):
            im.putpixel((x, y), (x * 3 % 256, y * 5 % 256, 90))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}", data, im


def _direct_logprobs(model, ids, comp, pixels, grid, image_token_id):
    seq = torch.tensor([list(ids) + list(comp[:-1])])
    with torch.no_grad():
        logits = model(input_ids=seq, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(seq == image_token_id).long()).logits[0].float()
    lp = torch.log_softmax(logits[len(ids) - 1: len(ids) - 1 + len(comp)], -1)
    return lp.gather(-1, torch.tensor(comp)[:, None])[:, 0]


async def test_tito_across_an_image_turn_and_logprobs_recomputed_from_refs():
    proc = _processor()
    tok = proc.tokenizer
    model = _model(tok)
    codec = HFVisionChatCodec(proc)
    url1, data1, im1 = _png(56, 84, (200, 10, 10))  # grid (1, 6, 4): 6 placeholders after the 2x2 merge
    url2, data2, im2 = _png(84, 84, (10, 200, 10))  # grid (1, 6, 6): 9 placeholders
    originals = {"sha256:" + hashlib.sha256(d).hexdigest(): im for d, im in ((data1, im1), (data2, im2))}
    replies = iter(['<tool_call>{"name": "click", "arguments": {"x": 3}}</tool_call>', "The button turned green."])
    seen: list[list[str]] = []

    async def engine(prompt, n, temp, media=None):  # an engine for a VL model: every image of the prompt, pixels from the originals
        seen.append([m["ref"] for m in media or []])
        enc = proc.image_processor(images=[originals[m["ref"]] for m in media], return_tensors="pt")
        comp = tok.encode(next(replies), add_special_tokens=False) + codec.eos_ids
        lps = _direct_logprobs(model, prompt, comp, enc["pixel_values"], enc["image_grid_thw"], codec.image_token_id)
        return GenResult(comp, lps.tolist(), 0, "stop", None, len(prompt))

    llm = EngineLLM(codec, engine)
    tools = [{"type": "function", "function": {"name": "click", "description": "click", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}]
    msgs = [{"role": "system", "content": "You operate a screen."},
            {"role": "user", "content": [{"type": "text", "text": "Press the red button."}, {"type": "image_url", "image_url": {"url": url1}}]}]
    r1 = await llm.complete(msgs, tools=tools)
    assert r1.tool_calls and r1.tool_calls[0]["name"] == "click"
    p1, m1 = r1.tokens["prompt_token_ids"], r1.tokens["media"]
    assert [(m["grid"], m["length"]) for m in m1] == [([1, 6, 4], 6)]  # placeholders = t*h*w / merge^2
    assert placeholder_runs(p1, codec.image_token_id) == [(m1[0]["offset"], 6)]
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "click", "arguments": '{"x": 3}'}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "clicked"},
             {"role": "user", "content": [{"type": "text", "text": "[images returned by tool call c1]"}, {"type": "image_url", "image_url": {"url": url2}}]}]
    r2 = await llm.complete(msgs, tools=tools)
    p2, m2 = r2.tokens["prompt_token_ids"], r2.tokens["media"]
    assert p2[: len(p1) + len(r1.tokens["completion_token_ids"])] == p1 + r1.tokens["completion_token_ids"]  # exact TITO across the image turn
    assert llm.prefix_breaks == 0 and len(m2) == 1 and m2[0]["length"] == 9 and m2[0]["offset"] > len(p1)
    assert all(t == codec.image_token_id for t in p2[m2[0]["offset"]: m2[0]["offset"] + 9])  # the offset is absolute in the full prompt
    assert seen == [[m1[0]["ref"]], [m1[0]["ref"], m2[0]["ref"]]]  # the engine got every image of the conversation

    calls = [{"call_id": f"agent:c{i + 1}", "segment": 0, "tokens": r.tokens, "actions": []} for i, r in enumerate((r1, r2))]
    run = RunRecord(run_id="v", task="t/screen@1", world="t/screen", policy="engine", started_at=utc_now(), steps=2, outcome="completed", reward=1.0,
                    attempt="v-a1", meta={"harness": "native:world@1"})
    ro = build(run, calls={"agent": calls}, mode="train")
    attach_call_fields(ro, {"agent": calls})
    assert ro.valid_for_training() == (True, "ok") and ro.turns[1].prompt_is_delta and not ro.turns[1].prefix_break
    text = ro.model_dump_json(by_alias=True)
    assert "pixel" not in text and "base64" not in text and json.loads(text)["turns"][1]["media"][0]["ref"].startswith("sha256:")
    ro = RolloutRecord.model_validate_json(text)  # refs only: the record round-trips without pixels
    assert [len(v) for v in prompt_media(ro).values()] == [1, 2]

    pol = VisionHFPolicy.from_codec(model, codec)
    with torch.no_grad():
        lp = pol.record_logprobs(ro)
    for t in ro.turns:  # recomputed from refs == the engine's direct forward
        assert torch.allclose(lp[t.call_id], torch.tensor(t.logprobs), atol=1e-5), t.call_id

    # an independent check: the processor alone on (the prompt's text, both images) gives the same ids and pixels
    full = full_prompts(ro)[ro.turns[1].call_id]
    collapsed = [t for i, t in enumerate(full) if not (t == codec.image_token_id and i > 0 and full[i - 1] == codec.image_token_id)]
    enc = proc(text=[tok.decode(collapsed)], images=[im1, im2], return_tensors="pt")
    assert enc["input_ids"][0].tolist() == full
    direct = _direct_logprobs(model, full, ro.turns[1].completion_token_ids, enc["pixel_values"], enc["image_grid_thw"], codec.image_token_id)
    assert torch.allclose(lp[ro.turns[1].call_id], direct, atol=1e-5)
    with torch.no_grad():  # the pixels matter: other images (the two swapped) give other log-probs
        swapped = [{**m, "ref": o["ref"]} for m, o in zip(prompt_media(ro)[ro.turns[1].call_id], [m2[0], m1[0]], strict=True)]
        with pytest.raises(ValueError, match="grid"):  # a 6x6 image where a 6x4 one was: refused, never silently shifted
            pol.token_logprobs(full, ro.turns[1].completion_token_ids, media=swapped)
        other = pol.token_logprobs(full, ro.turns[1].completion_token_ids)  # placeholders without pixels: a different forward
    assert not torch.allclose(other, direct, atol=1e-4)


def test_placeholder_and_grid_checks_fail_loudly():
    proc = _processor()
    codec = HFVisionChatCodec(proc)
    url, _, _ = _png(56, 84, (1, 2, 3))
    msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    ids = codec.render(msg, None)
    (m,) = codec.last_media
    codec.merge = 1  # a codec that believed the wrong merge size would mis-count the expansion
    with pytest.raises(ValueError, match="needs 24"):
        codec.render(msg, None)
    codec.merge = 2
    with pytest.raises(ValueError, match="remote image URLs"):
        codec.render([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}], None)
    pol = VisionHFPolicy.from_codec(_model(proc.tokenizer), codec)
    assert pol.pixel_inputs(ids, [m])["pixel_values"].shape[0] == 24  # 1*6*4 patches before the merge
    with pytest.raises(ValueError, match="no placeholder run"):
        pol.pixel_inputs(ids, [{**m, "offset": m["offset"] + 1}])
    with pytest.raises(ValueError, match="recorded"):
        pol.pixel_inputs(ids, [{**m, "grid": [1, 4, 6]}])
    with pytest.raises(ValueError, match="no image for"):
        pol.pixel_inputs(ids, [])


def test_real_qwen2_vl_processor_when_cached():
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", local_files_only=True)
    except Exception as e:  # noqa: BLE001 -- offline / not cached / torchvision missing
        pytest.skip(f"Qwen2-VL processor unavailable: {type(e).__name__}")
    codec = HFVisionChatCodec(proc)
    url, _, _ = _png(56, 84, (5, 5, 5))
    p = codec.render([{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": url}}]}], None)
    (m,) = codec.last_media
    assert m["length"] == m["grid"][0] * m["grid"][1] * m["grid"][2] // 4 and placeholder_runs(p, codec.image_token_id) == [(m["offset"], m["length"])]
    suf = codec.suffix([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}], None)
    assert placeholder_runs(suf, codec.image_token_id) == [(codec.last_media[0]["offset"], m["length"])]
