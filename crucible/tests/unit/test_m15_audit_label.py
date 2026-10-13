"""M15 audit regressions for ``crucible label`` and its dataset rows (docs/audit/2026-10-13/M15-study-defects-H1.md V1-V19,
M15-sweep-G1.md D1-D15, M15-sweep-G2.md D6/D7/D14). Every test fails on the M14 commit (076ab65) and passes after.

The run fixture is a synthetic M10 run directory written with the real Qwen2.5-0.5B-Instruct tokenizer (offline): a
model policy that passes and one that fails the same task from the same first prompt, a duplicate episode, a scripted
oracle, a control, an adversary with model turns, an invalid task (``validity.json``), a held-out task (a sealed
``splits.json`` with the build's ``bench.key``), a flagged, an exposed and an infra-masked episode, and an RLM episode
with a trainable and a foreign sub-call and a compaction segment. A World observation carries an e-mail, a phone number
and a fake secret-format canary.

TRL is not installed: the TRL code the rows are checked against is copied verbatim below with its file:line (TRL
72cf37d, ``RL-Environments/06-frameworks-and-env-code/infra-huggingface-trl``). HF ``datasets`` loading runs in the
``.envs/nemo`` interpreter (datasets 5.1.0)."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
from collections import Counter
from itertools import takewhile
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from crucible.studio import datasets as ds
from crucible.studio.label import label

CRUCIBLE = Path(__file__).resolve().parents[2]
NEMO_PY = CRUCIBLE / ".envs" / "nemo" / "bin" / "python"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "You are a careful support agent. Use the tools."
TOOLS = [{"type": "function", "function": {"name": "tickets__get_ticket", "description": "Read a ticket.",
                                           "parameters": {"type": "object", "properties": {"ticket_id": {"type": "integer"}}, "required": ["ticket_id"]}}},
         {"type": "function", "function": {"name": "tickets__issue_refund", "description": "Refund an order.",
                                           "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount_usd": {"type": "number"}},
                                                          "required": ["order_id", "amount_usd"]}}}]
EMAIL, PHONE = "dana.lee@example.org", "+1 415 555 0134"
CANARY = "AKIA" + "FAKECANARY" + "123456"  # a fake secret in AWS-key format: the redactor's rules must remove it


def _tok(name: str = QWEN):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {name} not in the local cache: {e}")


def run_label(**kw):
    return asyncio.run(label(**kw))


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


# ---- TRL, verbatim ------------------------------------------------------------------------------------------------------
def common_prefix_length(ids, other_ids):  # trl/data_utils.py:406-427
    length = 0
    for token, other_token in zip(ids, other_ids, strict=False):
        if token != other_token:
            break
        length += 1
    return length


def _tokenize(processing_class, input, **apply_chat_template_kwargs):  # trl/data_utils.py:367-403 (text-only branch)
    if isinstance(input, list):  # conversational: list of message dicts
        result = processing_class.apply_chat_template(input, tokenize=True, return_dict=True, **apply_chat_template_kwargs)
    else:  # non-conversational: plain text string
        result = processing_class(text=input)
    return result


def is_conversational(example):  # trl/data_utils.py:126-164
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}  # noqa: SIM118 - verbatim
    if example_keys:
        key = example_keys.pop()
        maybe_messages = example[key]
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            if isinstance(maybe_message, dict) and "role" in maybe_message:
                return True
    return False


def apply_chat_template_prompt_completion(example, processing_class, tools=None):  # trl/data_utils.py:167-300, prompt + completion branch
    tools = tools or None
    last_role = example["prompt"][-1]["role"]
    if last_role in ["user", "tool"]:
        add_generation_prompt, continue_final_message = True, False
    elif last_role == "assistant":
        add_generation_prompt, continue_final_message = False, True
    else:
        raise ValueError(f"Invalid role in the last message: {last_role}")
    prompt = processing_class.apply_chat_template(example["prompt"], tools=tools, continue_final_message=continue_final_message, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)
    prompt_completion = processing_class.apply_chat_template(example["prompt"] + example["completion"], tools=tools, tokenize=False)
    prompt = "".join(x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt, prompt_completion, strict=False)))
    return {"prompt": prompt, "completion": prompt_completion[len(prompt):]}


def sft_build_labels(example, mask_columns):  # trl/trainer/sft_trainer.py:1437-1443
    masks = [example[column] for column in mask_columns]
    labels = [token_id if all(bits) else -100 for token_id, *bits in zip(example["input_ids"], *masks, strict=False)]
    return {"labels": labels}


def sft_mask_columns(column_names, completion_only_loss):  # trl/trainer/sft_trainer.py:1428-1433, 962-963
    mask_columns = []
    if completion_only_loss and "completion_mask" in column_names:
        mask_columns.append("completion_mask")
    if "assistant_masks" in column_names:
        mask_columns.append("assistant_masks")
    return mask_columns


def dpo_tokenize_fn(example, processing_class):  # trl/trainer/dpo_trainer.py:1041-1046 (add_eos) + 1054-1103 (tokenize_fn)
    if not is_conversational(example):
        eos_token = processing_class.eos_token
        if not example["chosen"].endswith(eos_token):
            example["chosen"] = example["chosen"] + eos_token
        if not example["rejected"].endswith(eos_token):
            example["rejected"] = example["rejected"] + eos_token
    tools = example.get("tools")
    tools = json.loads(tools) if isinstance(tools, str) else tools
    apply_chat_template_kwargs = {"tools": tools, **example.get("chat_template_kwargs", {})}
    output = {}
    if is_conversational(example):
        prompt_ids = _tokenize(processing_class, example["prompt"], add_generation_prompt=True, **apply_chat_template_kwargs)["input_ids"]
        prompt_chosen_ids = _tokenize(processing_class, example["prompt"] + example["chosen"], **apply_chat_template_kwargs)["input_ids"]
        prompt_rejected_ids = _tokenize(processing_class, example["prompt"] + example["rejected"], **apply_chat_template_kwargs)["input_ids"]
    else:
        prompt_ids = _tokenize(processing_class, example["prompt"])["input_ids"]
        prompt_chosen_ids = _tokenize(processing_class, example["prompt"] + example["chosen"])["input_ids"]
        prompt_rejected_ids = _tokenize(processing_class, example["prompt"] + example["rejected"])["input_ids"]
    prompt_len = min(common_prefix_length(prompt_ids, prompt_chosen_ids), common_prefix_length(prompt_ids, prompt_rejected_ids))
    output["prompt_ids"] = prompt_ids[:prompt_len]
    output["chosen_ids"] = prompt_chosen_ids[prompt_len:]
    output["rejected_ids"] = prompt_rejected_ids[prompt_len:]
    return output


def kto_tokenize_fn(example, processing_class):  # trl/trainer/kto_trainer.py:1056-1062 (add_eos) + 1067-1108 (tokenize_fn)
    if not is_conversational(example) and not example["completion"].endswith(processing_class.eos_token):
        example["completion"] = example["completion"] + processing_class.eos_token
    tools = example.get("tools")
    tools = json.loads(tools) if isinstance(tools, str) else tools
    apply_chat_template_kwargs = {"tools": tools, **example.get("chat_template_kwargs", {})}
    if is_conversational(example):
        prompt_ids = _tokenize(processing_class, example["prompt"], add_generation_prompt=True, **apply_chat_template_kwargs)["input_ids"]
        prompt_completion_ids = _tokenize(processing_class, example["prompt"] + example["completion"], **apply_chat_template_kwargs)["input_ids"]
    else:
        prompt_ids = _tokenize(processing_class, example["prompt"])["input_ids"]
        prompt_completion_ids = _tokenize(processing_class, example["prompt"] + example["completion"])["input_ids"]
    prompt_len = common_prefix_length(prompt_ids, prompt_completion_ids)
    return {"prompt_ids": prompt_ids[:prompt_len], "completion_ids": prompt_completion_ids[prompt_len:]}


# ---- the run fixture ----------------------------------------------------------------------------------------------------
def _enc(tok, text):
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _call(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _resp(*obs):
    return "\n<|im_start|>user\n" + "\n".join(f"<tool_response>\n{o}\n</tool_response>" for o in obs) + "<|im_end|>\n<|im_start|>assistant\n"


def _first(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tools=TOOLS, add_generation_prompt=True, tokenize=False)


USER = "Ticket 38: the customer was charged twice for order O-7108. Refund the duplicate."
OBS1 = json.dumps({"ticket": 38, "email": EMAIL, "phone": PHONE, "note": f"api key {CANARY}"})
OBS2 = json.dumps({"order": "O-7108", "amount": 20.9})
PASS_C1 = _call("tickets__get_ticket", {"ticket_id": 38}) + "\n" + _call("tickets__get_order", {"order_id": "O-7108"}) + "<|im_end|>"  # parallel calls
PASS_C2 = _call("tickets__issue_refund", {"order_id": "O-7108", "amount_usd": 20.9}) + "<|im_end|>"
FAIL_C1 = _call("tickets__issue_refund", {"order_id": "O-7108", "amount_usd": 41.8}) + "<|im_end|>"


def _turns_pass(tok, user=USER):
    return [{"delta": _first(tok, user), "completion": PASS_C1, "advantage": 0.25},
            {"delta": _resp(OBS1, OBS2), "completion": PASS_C2, "advantage": -0.5},  # a negative-credit turn (V17 --mask-negative)
            {"delta": _resp(json.dumps({"refunded": 20.9})), "completion": "Refunded the duplicate charge.<|im_end|>", "advantage": 0.75}]


def _turns_fail(tok):
    return [{"delta": _first(tok, USER), "completion": FAIL_C1, "advantage": 0.0},
            {"delta": _resp(json.dumps({"refunded": 41.8})), "completion": "Done.<|im_end|>", "advantage": 0.0}]


def _turns_rlm(tok):
    p0 = tok.apply_chat_template([{"role": "system", "content": "You write python cells. Tools: python(code)."}, {"role": "user", "content": USER}],
                                 add_generation_prompt=True, tokenize=False)
    return [{"delta": p0, "completion": _call("python", {"code": "r = tickets.issue_refund(order_id='O-7108', amount_usd=20.9)"}) + "<|im_end|>"},
            {"kind": "sub", "delta": "<|im_start|>user\nSummarize: refund issued<|im_end|>\n<|im_start|>assistant\n", "completion": "Refund issued.<|im_end|>"},
            {"kind": "sub", "trainable": False, "model": "other/model"},
            {"kind": "compaction", "segment": 1, "delta": "<|im_start|>user\nSummarize the episode so far.<|im_end|>\n<|im_start|>assistant\n",
                 "completion": "Summary: the refund of 20.9 was issued.<|im_end|>"},
            {"segment": 2, "delta": "<|im_start|>user\nSummary: the refund of 20.9 was issued.<|im_end|>\n<|im_start|>assistant\n",
                 "completion": _call("python", {"code": "rlm.finish('done')"}) + "<|im_end|>"}]


def _atif(model, user, calls, final="", scripted=False):
    steps = [{"step_id": 1, "source": "user", "message": user}]
    for k, (fn, args, obs) in enumerate(calls):
        steps.append({"step_id": len(steps) + 1, "source": "agent", "message": "", "tool_calls": [{"tool_call_id": f"c{k}", "function_name": fn, "arguments": args}],
                      "observation": {"results": [{"source_call_id": f"c{k}", "content": obs}]}, "extra": {"call_id": f"agent:c{k + 1}"}})
    steps.append({"step_id": len(steps) + 1, "source": "agent", "message": "script finished" if scripted else (final or "Done."), "extra": {"done": True}})
    return {"schema_version": "ATIF-v1.8", "session_id": "s", "agent": {"name": "crucible-policy", "version": "0.1.0", **({"model_name": model} if model else {})},
            "steps": steps}


def _write_episode(run: Path, tok, *, task, policy, idx=0, reward=1.0, turns=None, harness="native:llm-tool@1", flags=(), failure=None, masked=False,
                   atif=None):
    import hashlib

    from crucible.core.schema.rollout import RolloutRecord, Segment, Turn
    from crucible.core.schema.run import RunRecord

    sid = hashlib.sha256(json.dumps([task, policy, harness, idx]).encode()).hexdigest()[:16]
    attempt = f"{sid}-a1"
    adir = run / "samples" / sid / "a1"
    adir.mkdir(parents=True)
    rec = {"schema_version": "crucible.run/v1", "run_id": run.name, "task": task, "world": "w", "policy": policy, "seed": idx, "started_at": "2026-10-13T00:00:00Z",
           "ended_at": "2026-10-13T00:01:00Z", "steps": len(turns or []), "outcome": "masked" if masked else ("failed" if failure else "completed"), "failure": failure, "mask_sample": masked,
           "reward": None if masked else reward, "attempt": attempt, "meta": {"harness": harness},
           "sample": {"task": task, "world": "w", "policy": policy, "harness": harness, "sample_idx": idx, "seed": idx, "tier": "full"}}
    rec = json.loads(RunRecord.model_validate(rec).model_dump_json())
    tt, calls, segs = [], [], []
    prev_seg = None
    for k, t in enumerate(turns or []):
        seg, kind = t.get("segment", 0), t.get("kind", "turn")
        cid = f"agent:c{k + 1}"
        if t.get("trainable", True) is False:
            tt.append(Turn(call_id=cid, actor="agent", kind=kind, segment=seg, trainable=False, model=t.get("model")))
            calls.append({"call_id": cid, "kind": kind, "segment": seg, "model": t.get("model"), "trainable": False})
            continue
        p, c = _enc(tok, t["delta"]), _enc(tok, t["completion"])
        delta = kind == "turn" and prev_seg == seg and k > 0
        tt.append(Turn(call_id=cid, actor="agent", kind=kind, segment=seg, world_step=k, prompt_is_delta=delta, prompt_token_ids=p, completion_token_ids=c,
                       logprobs=[-0.1] * len(c), policy_version=0, advantage=t.get("advantage"), finish="stop"))
        calls.append({"call_id": cid, "kind": kind, "segment": seg, "model": QWEN, "step": k, "actions": [f"tool:{cid}"]})
        if kind != "sub":
            prev_seg = seg
        if kind == "compaction":
            segs.append(Segment(index=seg, summary_call=cid))
    ro = RolloutRecord(rollout_id=attempt, sample=sid, task=task, harness=harness, policy=policy, turns=tt, segments=segs, reward=None if masked else reward,
                       mask_sample=masked, flags=list(flags), failure_kind=(failure or {}).get("kind"))
    (adir / "rollout.json").write_text(ro.model_dump_json(by_alias=True))
    (adir / "record.json").write_text(json.dumps(rec))
    (adir / "calls.json").write_text(json.dumps({"agent": calls}))
    (adir / "progress.jsonl").write_text("".join(json.dumps({"step": k, "phi": 0.0, "delta_phi": t.get("advantage") or 0.0, "actor": "agent"}) + "\n"
                                                 for k, t in enumerate(turns or [])))
    model = QWEN if turns else None
    (adir / "trajectory.agent.json").write_text(json.dumps(atif or _atif(model, USER, [("tickets.get_ticket", {"ticket_id": 38}, OBS1),
                                                                                      ("tickets.issue_refund", {"order_id": "O-7108", "amount_usd": 20.9}, "{}")],
                                                                     scripted=not turns)))
    with (run / "results.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


T = {k: f"toy/helpdesk_v2/{k}" for k in ("t1", "t2", "t3", "t4", "t5")}


def make_run(root: Path, *, seal_ok: bool = True) -> Path:
    from crucible.bench import splits
    from crucible.core.schema.run import Failure, RunRecord

    tok = _tok()
    build = root / "build"
    run = build / "runs" / "bench-fixture"
    (run / "samples").mkdir(parents=True)
    (run / "results.jsonl").write_text("")
    pol = "qwen2.5-0.5b"
    (run / "run.json").write_text(json.dumps({"run_id": run.name, "spec": {"run_id": run.name, "tasks": sorted(T.values()), "policies": [pol]}}))
    _write_episode(run, tok, task=T["t1"], policy=pol, idx=0, reward=1.0, turns=_turns_pass(tok))
    _write_episode(run, tok, task=T["t1"], policy=pol, idx=1, reward=0.0, turns=_turns_fail(tok))
    _write_episode(run, tok, task=T["t1"], policy=pol, idx=2, reward=1.0, turns=_turns_pass(tok))  # an identical episode: deduplicated
    _write_episode(run, tok, task=T["t1"], policy="adv:stuffing", reward=1.0, turns=_turns_pass(tok, user=USER + " "))
    demo = _atif(None, USER, [("tickets.get_ticket", {"ticket_id": 38}, OBS1), ("tickets.issue_refund", {"order_id": "O-7108", "amount_usd": 20.9}, "{}")],
                 scripted=True)
    demo["agent"]["tool_definitions"] = [{"type": "function", "function": {**t["function"], "name": t["function"]["name"].replace("__", ".")}} for t in TOOLS]
    demo["steps"][1]["reasoning_content"] = "Read the ticket first."
    _write_episode(run, tok, task=T["t1"], policy="scripted:oracle", reward=1.0, atif=demo)
    _write_episode(run, tok, task=T["t1"], policy="control:noop", reward=0.0)
    _write_episode(run, tok, task=T["t2"], policy=pol, reward=1.0, turns=_turns_pass(tok, user=USER + " (t2)"))
    _write_episode(run, tok, task=T["t3"], policy=pol, reward=1.0, turns=_turns_pass(tok, user=USER + " (t3)"))
    _write_episode(run, tok, task=T["t4"], policy=pol, idx=0, reward=1.0, turns=_turns_pass(tok, user=USER + " (t4a)"),
                   flags=[{"rule": "intercepted:grader", "category": "grader_tampering", "seq": 3}])
    exp = Failure(kind="crucible:answer_exposed", stage="verification", reason="read the answer key")
    _write_episode(run, tok, task=T["t4"], policy=pol, idx=1, turns=_turns_pass(tok, user=USER + " (t4b)"), failure=exp.model_dump(), masked=exp.masks)
    _write_episode(run, tok, task=T["t4"], policy=pol, idx=2, turns=_turns_pass(tok, user=USER + " (t4c)"),
                   failure={"kind": "crucible:capture_incomplete", "stage": "agent"}, masked=True)
    _write_episode(run, tok, task=T["t5"], policy=pol, reward=1.0, turns=_turns_rlm(tok), harness="rlm:fixture@1")
    assert RunRecord  # the records validated above
    (run / "validity.json").write_text(json.dumps({T["t2"]: {"valid": False, "reasons": ["noop_passes"], "source": "controls"},
                                                  **{T[k]: {"valid": True, "reasons": []} for k in ("t1", "t3", "t4", "t5")}}))
    key = secrets.token_hex(32)
    (build / "bench.key").write_text(key)
    sp = {T["t1"]: "train", T["t2"]: "train", T["t3"]: "bench", T["t4"]: "train", T["t5"]: "train"}
    seal = splits.seal(sp, bytes.fromhex(key) + b"|splits", {})
    if not seal_ok:
        sp[T["t3"]] = "train"  # edited after sealing: the held-out task moved into train
    splits.write_splits(build / "splits.json", {"splits": sp, "params": {}, "seal": seal, "counts": {}})
    return run


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("m15label"))


@pytest.fixture(scope="module")
def labelled_run(fixture_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("m15label_out") / "lab"
    s = run_label(out=out, run=fixture_run)
    return out, s


@pytest.fixture(scope="module")
def labelled_toy(tmp_path_factory):
    out = tmp_path_factory.mktemp("m15label_toy") / "lab"
    s = run_label(out=out, pack="toy")
    return out, s


def _all_rows(out: Path) -> list[dict]:
    return [r for f in ("sft", "dpo", "kto") for x in (f"{f}.jsonl", f"{f}.chat.jsonl") if (out / x).exists() for r in rows(out / x)]


def _who(r: dict) -> set[str]:
    return {str(r.get(k)) for k in ("model", "policy", "chosen_policy", "rejected_policy", "chosen_model", "rejected_model") if r.get(k)}


def _trained(r: dict, key: str = "completion") -> list:
    """The trained target(s) of a row, new or old shape: a completion / chosen / rejected, or the masked messages."""
    if "message_loss_mask" in r:
        return [m for m, k in zip(r["messages"], r["message_loss_mask"]) if k]
    v = r.get(key)
    return v if isinstance(v, list) else [v]


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True)


# ---- V1 / D2: controls, scripted policies and adversaries are never data -------------------------------------------------
def test_V1_controls_scripted_adversaries_are_never_data(labelled_run, fixture_run, tmp_path):
    out, s = labelled_run
    rs = _all_rows(out)
    assert rs, "the model's own episodes must still be labelled"
    for r in rs:
        assert not any(w.startswith(("scripted:", "control:", "adv:")) for w in _who(r)), r.get("policy")
    assert all(r.get("policy") == "qwen2.5-0.5b" or r.get("chosen_policy") == "qwen2.5-0.5b" for r in rs)
    assert s["skipped"].get("scripted") == 1 and s["skipped"].get("control") == 1 and s["skipped"].get("adversary") == 1
    # scripted demonstrations only on request, and labelled as such in every row
    s2 = run_label(out=tmp_path / "demo", run=fixture_run, allow_scripted=True)
    demo = [r for r in _all_rows(tmp_path / "demo") if r.get("policy") == "scripted:oracle"]
    assert demo and all(r["source"] == "scripted_demo" and "not model-produced" in r["demonstration"] for r in demo)
    assert s2["skipped"].get("control") == 1 and s2["skipped"].get("adversary") == 1


# ---- V2 / D5 / D7: invalid, flagged, exposed and masked episodes ---------------------------------------------------------
def test_V2_invalid_flagged_exposed_and_masked_episodes_are_never_data(labelled_run):
    out, s = labelled_run
    tasks = Counter(r["task"] for r in _all_rows(out))
    assert T["t2"] not in tasks  # invalid in validity.json
    assert T["t4"] not in tasks  # flagged, exposed (runner-masked) and infra-masked
    sk = s["skipped"]
    assert sk.get("invalid_task") == 1 and sk.get("flagged") == 1 and sk.get("exposed_or_tampered") == 1 and sk.get("masked") == 1
    assert s["invalid_tasks"] == [T["t2"]]


# ---- V3: RLM episodes are the model's own cells ----------------------------------------------------------------------------
def test_V3_rlm_rows_are_the_models_python_cells_with_subcalls_and_compaction_apart(labelled_run):
    out, _ = labelled_run
    rlm = [r for r in rows(out / "sft.jsonl") if r["task"] == T["t5"]]
    assert rlm
    for r in rlm:
        for m in _trained(r):
            assert "tickets__" not in _text(m)  # no re-synthesized native World call: the model wrote python cells
    kinds = Counter(r["call_kind"] for r in rlm)
    assert kinds == {"turn": 2, "sub": 1, "compaction": 1}  # the foreign sub-call (trainable=False) is never a target
    assert any('"name": "python"' in r["completion"] for r in rlm if r["call_kind"] == "turn")


# ---- V4 / D3: one policy turn per preference / KTO completion ----------------------------------------------------------
def test_V4_preference_and_kto_completions_are_one_policy_turn(labelled_run, labelled_toy):
    for out, _ in (labelled_run, labelled_toy):
        dpo, kto = rows(out / "dpo.jsonl"), rows(out / "kto.jsonl")
        for r in dpo:
            for k in ("chosen", "rejected"):
                v = r[k]
                assert isinstance(v, str) or (len(v) == 1 and v[0]["role"] == "assistant"), v
        for r in kto:
            v = r["completion"]
            assert isinstance(v, str) or (len(v) == 1 and v[0]["role"] == "assistant"), v
            assert "<|im_start|>user" not in _text(v) and "[customer" not in _text(v) and "[supervisor" not in _text(v)
    out, _ = labelled_run
    (pair,) = rows(out / "dpo.jsonl")
    tok = _tok()
    assert pair["chosen"] == tok.decode(_enc(tok, PASS_C1)) and pair["rejected"] == tok.decode(_enc(tok, FAIL_C1))


# ---- V5 / D13: system prompt, tool schema and parallel calls as the model saw and emitted them ---------------------------
def test_V5_D13_rows_keep_the_system_prompt_tools_and_parallel_calls(labelled_run):
    out, _ = labelled_run
    first = [r for r in rows(out / "sft.jsonl") if r["task"] == T["t1"] and r.get("step") == 0]
    assert first, "the first turn of the passing episode is an SFT row"
    r = first[0]
    assert r["prompt"].startswith("<|im_start|>system\n" + SYSTEM) and "<tools>" in r["prompt"] and "tickets__issue_refund" in r["prompt"]
    assert r["completion"].count("<tool_call>") == 2  # one completion, two parallel calls


# ---- V6 / D9: the KTO label is a bool --------------------------------------------------------------------------------------
def test_V6_kto_label_is_a_bool_and_the_outcome_is_apart(labelled_toy):
    msgs = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "done"}]
    (r,) = ds.kto_rows([ds.Labelled("t", msgs, 0.0, label="bad")])
    assert r["label"] is False and r["outcome"] == "bad"
    out, _ = labelled_toy
    kto = rows(out / "kto.jsonl")
    assert kto and all(isinstance(r["label"], bool) for r in kto)


# ---- V7 / D6 / G2 D14: one positive rule ---------------------------------------------------------------------------------
def test_V7_one_positive_rule_for_sft_dpo_and_kto(labelled_toy):
    msgs = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "done"}]
    bad_but_rewarded = ds.Labelled("t", msgs, 1.0, label="bad")
    good_but_failed = ds.Labelled("t", [msgs[0], {"role": "assistant", "content": "nope"}], 0.0, label="good")
    unknown = ds.Labelled("t", [msgs[0], {"role": "assistant", "content": "maybe"}], 1.0, label="unknown")
    assert ds.curate([bad_but_rewarded, good_but_failed, unknown], select="correct") == []
    assert ds.kto_rows([bad_but_rewarded, good_but_failed, unknown]) == []
    out, _ = labelled_toy
    sft = rows(out / "sft.jsonl")
    assert sft and not [r for r in sft if r.get("outcome", r.get("label")) in ("bad", "unknown")]
    sft_eps = {r["episode"] for r in sft}
    assert {r["episode"] for r in rows(out / "kto.jsonl") if r["label"] is True} <= sft_eps


# ---- V8: turns graded wrong are never trained ----------------------------------------------------------------------------
def _pivot_trace(*, feedback: bool = True):
    from crucible.core.schema.signal import Signal
    from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

    ev = [TraceEvent(seq=1, kind="message", actor="c", type="ticket_opened", text="Refund my annual plan, I cancelled after 3 months."),
          TraceEvent(seq=2, kind="action", actor="a", type="refund_issued", text="refund 120.00", data={"amount": "120.00", "row_id": "T1-2"}),
          TraceEvent(seq=3, kind="message", actor="a", type="agent_reply", text="Refunded the full 120.00."),
          *([TraceEvent(seq=4, kind="message", actor="s", type="supervisor_note", text="Annual plans refund pro rata: 90.00, not 120.00. Fix it.")] if feedback else []),
          TraceEvent(seq=5, kind="action", actor="a", type="refund_adjusted", text="adjust refund to 90.00", data={"amount": "90.00", "row_id": "T1-5"}),
          TraceEvent(seq=6, kind="action", actor="a", type="status_change", text="resolved", data={"status": "resolved"})]
    return Trace(trace_id="t:1", family="toy/x", source=TraceSource(connector="file_drop", ref="T1"),
                 actors=[TraceActor(id="c", role="customer", kind="human"), TraceActor(id="a", role="agent", kind="human"),
                         TraceActor(id="s", role="supervisor", kind="human")],
                 events=ev, outcome=TraceOutcome(label="good"),
                 signals=[Signal(id="s1", type="correction", polarity=-1, grades=[2], source_event=5, actor="a")])


def test_V8_a_turn_graded_wrong_is_never_an_sft_target():
    x = ds.from_trace(_pivot_trace(), phi_by_seq={1: 0.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.6, 6: 1.0}, reward=1.0)
    trained = [_text(m) for r in ds.sft_rows([x]) for m in _trained(r)]
    assert trained and not [t for t in trained if "120.00" in t]


# ---- V9 / D1: held-out traces and tasks are never emitted ----------------------------------------------------------------
def test_V9_D1_held_out_tickets_and_bench_tasks_are_never_emitted(labelled_toy, labelled_run, fixture_run, tmp_path):
    out, s = labelled_toy
    test_ids = {f"toy/support_ticket:T10{i}" for i in range(13, 25)}
    assert not [r for r in _all_rows(out) if r.get("episode") in test_ids]
    out, s = labelled_run
    assert not [r for r in _all_rows(out) if r["task"] == T["t3"]] and s["skipped"].get("held_out") == 1 and s["splits"]["verified"] is True
    run_label(out=tmp_path / "inc", run=fixture_run, include_bench=True)
    assert [r for r in rows(tmp_path / "inc" / "sft.jsonl") if r["task"] == T["t3"]]
    bad = make_run(tmp_path / "tampered", seal_ok=False)
    with pytest.raises(ValueError, match="seal does not verify"):
        run_label(out=tmp_path / "t", run=bad)


# ---- V10: --select best is the best passing episode ----------------------------------------------------------------------
def test_V10_select_best_is_the_best_passing_episode():
    msgs = [{"role": "user", "content": "q"}]
    fails = [ds.Labelled("t", [*msgs, {"role": "assistant", "content": f"a{r}"}], r) for r in (0.0, 0.2)]
    assert ds.curate(fails, select="best") == []
    ok = ds.Labelled("t", [*msgs, {"role": "assistant", "content": "ok"}], 1.0)
    assert ds.curate([*fails, ok], select="best") == [ok]


# ---- V11: pair semantics ------------------------------------------------------------------------------------------------
def test_V11_best_vs_worst_needs_a_pass_a_fail_a_margin_and_one_prompt():
    u = {"role": "user", "content": "q"}
    a = lambda c: {"role": "assistant", "content": c}
    assert ds.dpo_best_vs_worst([ds.Labelled("t", [u, a("x")], 1.0), ds.Labelled("t", [u, a("y")], 0.7)], threshold=0.5) == []  # rejected passes
    assert ds.dpo_best_vs_worst([ds.Labelled("t", [u, a("x")], 1.0), ds.Labelled("t", [u, {"role": "user", "content": "extra hint"}, a("y")], 0.0)]) == []
    with pytest.raises(ValueError, match="margin"):
        ds.dpo_best_vs_worst([ds.Labelled("t", [u, a("x")], 1.0), ds.Labelled("t", [u, a("y")], 1.0)], margin=0.0)


# ---- V12 / D14: no empty or stub completions -----------------------------------------------------------------------------
def test_V12_D14_no_empty_or_stub_completions(labelled_run, fixture_run, tmp_path):
    u = {"role": "user", "content": "q"}
    no_turn = ds.Labelled("t", [u], 0.0)
    ok = ds.Labelled("t", [u, {"role": "assistant", "content": "x"}], 1.0)
    assert ds.kto_rows([no_turn]) == [] and ds.dpo_best_vs_worst([ok, no_turn]) == []
    run_label(out=tmp_path / "demo", run=fixture_run, allow_scripted=True)
    for out in (labelled_run[0], tmp_path / "demo"):
        for r in _all_rows(out):
            for k in ("completion", "chosen", "rejected"):
                if k in r:
                    t = _text(r[k])
                    assert r[k] and t.strip() and "script finished" not in t
    # a PR-like trace with no policy turn yields nothing (its prompt would end with the label, "merged")
    from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

    pr = Trace(trace_id="pr:1", family="coding/pull_request", source=TraceSource(connector="x", ref="1"),
               actors=[TraceActor(id="au", role="author", kind="human"), TraceActor(id="m", role="maintainer", kind="human")],
               events=[TraceEvent(seq=0, kind="message", actor="au", type="pr_opened", text="fix"), TraceEvent(seq=1, kind="state", actor="m", type="merged", text="merged")],
               outcome=TraceOutcome(label="good"))
    assert ds.kto_rows([ds.from_trace(pr, reward=1.0)]) == [] and ds.sft_rows([ds.from_trace(pr, reward=1.0)]) == []


# ---- V13: correction pivots ---------------------------------------------------------------------------------------------
def test_V13_correction_pivots_need_no_feedback_and_are_one_call_each():
    x = ds.from_trace(_pivot_trace(), reward=1.0)
    assert ds.dpo_pivots([x]) == []  # the fix came after the supervisor's feedback, which the prompt lacks
    (row,) = ds.dpo_pivots([x], allow_state_shift=True)
    assert len(row["chosen"]) == 1 and row["chosen"][0]["content"] == "adjust refund to 90.00" and "resolved" not in _text(row["chosen"])
    assert row["rejected"][0]["content"] == "refund 120.00" and "Refunded the full" not in _text(row["rejected"])
    assert row["state_shift"] is True


# ---- V14: the dataset tests are no longer vacuous (test_studio.py rewritten); a real run through the CLI -----------------
def test_V14_label_run_through_the_cli_writes_only_model_rows(fixture_run, tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    r = CliRunner().invoke(cli, ["label", "--run", str(fixture_run), "--out", str(tmp_path / "cli"), "--json"], catch_exceptions=False)
    s = json.loads(r.output)
    assert r.exit_code == 0 and s["datasets"]["dpo_kinds"]["best_vs_worst"] == 1 and s["datasets"]["sft"] > 0
    assert all(_who(x) <= {"qwen2.5-0.5b", QWEN} for x in _all_rows(tmp_path / "cli"))


# ---- V15: filter and curation edges --------------------------------------------------------------------------------------
def test_V15_filter_and_curation_errors_are_value_errors():
    with pytest.raises(ValueError):
        ds.compile_filter("avg >")
    uncredited = [ds.Labelled("fam", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], None)]
    with pytest.raises(ValueError, match="fam"):
        ds.curate(uncredited, filter="n_correct / n > 0.5")
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_per_task"):
            ds.curate(uncredited, max_per_task=bad)


# ---- V16: provenance, uniqueness and dedup -------------------------------------------------------------------------------
def test_V16_traj_ids_are_unique_rows_carry_provenance_and_duplicates_are_counted(labelled_run):
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert ds.Labelled("t1", msgs, 1.0, episode="e1").traj_id != ds.Labelled("t2", msgs, 1.0, episode="e2").traj_id
    out, s = labelled_run
    (pair,) = rows(out / "dpo.jsonl")
    assert pair["chosen_episode"] != pair["rejected_episode"] and pair["chosen_policy"] == pair["rejected_policy"] == "qwen2.5-0.5b"
    assert pair["chosen_reward"] == 1.0 and pair["rejected_reward"] == 0.0
    sft = rows(out / "sft.jsonl")
    assert len({r["row_id"] for r in sft}) == len(sft) and s["skipped"].get("sft_duplicate", 0) >= 3  # the duplicate episode's 3 turns


# ---- V17: dense credit on run rows ---------------------------------------------------------------------------------------
def test_V17_run_rows_carry_per_turn_credit_and_mask_negative_works(labelled_run, fixture_run, tmp_path):
    out, _ = labelled_run
    t1 = sorted((r for r in rows(out / "sft.jsonl") if r["task"] == T["t1"]), key=lambda r: r["step"])
    assert [r["weight"] for r in t1] == [0.25, -0.5, 0.75] and all(r["credit"] for r in t1)
    run_label(out=tmp_path / "neg", run=fixture_run, mask_negative=True)
    t1n = [r for r in rows(tmp_path / "neg" / "sft.jsonl") if r["task"] == T["t1"]]
    assert sorted(r["weight"] for r in t1n) == [0.25, 0.75]


# ---- V18 / D4 / D10: real tool calls only, arguments as dicts ------------------------------------------------------------
def _coding_trace():
    from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

    ev = [TraceEvent(seq=0, kind="message", actor="system", type="system_message", text="You are an engineer. Use the bash tool.", data={"step_id": 1}),
          TraceEvent(seq=1, kind="message", actor="user", type="user_message", text="Port the library.", data={"step_id": 2}),
          TraceEvent(seq=2, kind="message", actor="agent", type="agent_message", text="I'll look around.", data={"step_id": 3, "reasoning": "I'll look around."}),
          TraceEvent(seq=3, kind="action", actor="agent", type="tool_call", text='bash({"command": "ls"})',
                     data={"arguments": '{"command": "ls"}', "call_id": "toolu_01X", "step_id": 3, "tool": "bash"}),
          TraceEvent(seq=4, kind="observation", actor="system", type="tool_result", text="<returncode>0</returncode>\nREADME.md", reply_to=3, data={"call_id": None, "step_id": 3}),
          TraceEvent(seq=5, kind="message", actor="agent", type="agent_message", text="Done.", data={"step_id": 4}),
          TraceEvent(seq=6, kind="state", actor="agent", type="final_patch", text="47 files, +2256 -4162", data={"bytes": 246915, "files": 47}),
          TraceEvent(seq=7, kind="check", actor="system", type="grade_gate", text="tests_gpu: fail", data={"gate": "tests_gpu", "pass": False})]
    return Trace(trace_id="run:x", family="coding/agent_run", source=TraceSource(connector="x", ref="x"),
                 actors=[TraceActor(id="user", role="user", kind="human"), TraceActor(id="agent", role="agent", kind="agent", handle="vendor/model"),
                         TraceActor(id="system", role="system", kind="system")], events=ev, outcome=TraceOutcome(label="bad"))


def test_V18_D4_D10_trace_rows_carry_only_recorded_tool_calls_with_dict_arguments(labelled_toy):
    x = ds.from_trace(_coding_trace(), reward=0.0)
    assert x.messages[0] == {"role": "system", "content": "You are an engineer. Use the bash tool."}
    calls = [c for m in x.messages for c in m.get("tool_calls") or []]
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in calls] == [("bash", {"command": "ls"})]
    tool_msgs = [m for m in x.messages if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "toolu_01X", "content": "<returncode>0</returncode>\nREADME.md"}]
    blob = json.dumps(x.messages)
    assert "final_patch" not in blob and "47 files" not in blob and "tests_gpu" not in blob and "step_id" not in blob and "recorded" not in blob
    out, _ = labelled_toy
    blob = json.dumps([r.get(k) for r in _all_rows(out) for k in ("prompt", "completion", "chosen", "rejected", "messages")])
    assert "row_id" not in blob and "ticket:T10" not in blob and ": recorded" not in blob and "writes" not in blob
    for r in _all_rows(out):
        for m in [*(r.get("prompt") or []), *_trained(r)]:
            for c in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
                assert isinstance(c["function"]["arguments"], dict) and c["function"]["name"] not in ("status_change", "refund_issued")


# ---- V19: the run path redacts ------------------------------------------------------------------------------------------
def test_V19_run_rows_are_redacted(labelled_run):
    out, s = labelled_run
    for f in out.iterdir():
        if f.is_file():
            t = f.read_text()
            assert EMAIL not in t and PHONE not in t and CANARY not in t, f.name
    assert s["datasets"]["redacted_rows"] > 0


# ---- D8: every file loads with HF datasets -------------------------------------------------------------------------------
LOAD = """
import json, sys
from datasets import load_dataset
out = {}
for p in sys.argv[1:]:
    ds = load_dataset("json", data_files=p, split="train")
    out[p] = {"n": len(ds), "cols": ds.column_names, "first": ds[0]}
print(json.dumps(out, default=str))
"""


def _hf_load(files):
    r = subprocess.run([str(NEMO_PY), "-c", LOAD, *map(str, files)], capture_output=True, text=True, timeout=600, check=False, env={**os.environ, "HF_DATASETS_OFFLINE": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    for p in files:
        assert got[str(p)]["n"] == len(rows(p)), p
    return got


def test_D8_every_output_file_loads_with_hf_datasets(labelled_run, labelled_toy, tmp_path):
    if not NEMO_PY.exists():
        pytest.skip("no .envs/nemo interpreter with HF datasets")
    _hf_load([p for d in (labelled_run[0], labelled_toy[0]) for p in sorted(d.glob("*.jsonl")) if p.stat().st_size])
    run_label(out=tmp_path / "chain", pack="toy", sft_format="chain")  # the multi-turn prime-rl format too
    got = _hf_load([tmp_path / "chain" / "sft.jsonl"])
    assert all(isinstance(w, float) for w in got[str(tmp_path / "chain" / "sft.jsonl")]["first"]["weights"])


# ---- D11 / D12: the rows tokenize as TRL expects, the trained tokens are the policy's; strict templates ------------------
def test_D11_trl_trains_exactly_the_policys_tokens_on_run_rows(labelled_run):
    tok = _tok()
    out, _ = labelled_run
    sft = rows(out / "sft.jsonl")
    for r in sft:  # TRL's pre-tokenized SFT path: labels from completion_mask and assistant_masks
        cols = set(r)
        labels = sft_build_labels(r, sft_mask_columns(cols, completion_only_loss="prompt" in cols and "completion" in cols))["labels"]
        assert [x for x in labels if x != -100] == r["completion_ids"]
        assert sft_build_labels(r, sft_mask_columns(cols, completion_only_loss=False))["labels"] == labels  # assistant_masks alone suffices
    (pair,) = rows(out / "dpo.jsonl")
    got = dpo_tokenize_fn(dict(pair), tok)
    assert pair["text_exact"] and got["prompt_ids"] == pair["prompt_ids"] and got["chosen_ids"] == pair["chosen_ids"] and got["rejected_ids"] == pair["rejected_ids"]
    for r in rows(out / "kto.jsonl"):
        if r["text_exact"]:
            got = kto_tokenize_fn(dict(r), tok)
            assert got["completion_ids"] == r["completion_ids"] and got["prompt_ids"] == r["prompt_ids"]


def test_D11_D12_trace_rows_train_only_the_assistant_turn_under_strict_templates(labelled_toy):
    out, _ = labelled_toy
    conv = [r for f in ("sft", "kto") for r in rows(out / f"{f}.jsonl")]
    assert conv
    for name in (QWEN, "Qwen/Qwen3-0.6B", "google/gemma-2-2b-it"):
        tok = _tok(name)
        for r in conv:
            got = apply_chat_template_prompt_completion(r, tok, tools=r.get("tools"))
            content = r["completion"][0]["content"]
            assert got["completion"].lstrip().startswith(content) or content in got["completion"], (name, got["completion"][:200])
            assert "[customer" not in got["completion"] and "[supervisor" not in got["completion"]
    # verl's per-message tokenization equals the full rendering (multiturn_sft_dataset.py:425-456) on Qwen2.5
    tok = _tok()

    def norm(x):
        return list(x["input_ids"] if hasattr(x, "keys") else x)

    t1 = norm(tok.apply_chat_template([{"role": "user", "content": ""}], tokenize=True))
    t2 = norm(tok.apply_chat_template([{"role": "user", "content": ""}] * 2, tokenize=True))
    sp = t1[: -(len(t2) - len(t1))]
    for r in conv:
        msgs = r["prompt"] + r["completion"]
        ids = []
        for i, m in enumerate(msgs):
            x = norm(tok.apply_chat_template([m], tokenize=True))
            ids += x[len(sp):] if i != 0 and m["role"] != "system" else x
        assert ids == norm(tok.apply_chat_template(msgs, tokenize=True))


def test_D12_template_adaptation_for_gemma(tmp_path):
    x = ds.from_trace(_coding_trace(), reward=1.0)
    tok = _tok("google/gemma-2-2b-it")
    roles = ds.template_roles(tok)
    assert roles == {"system": False, "tool": False}
    adapted = ds.adapt_for_template(x.messages, system=False, tool=False)
    tok.apply_chat_template(adapted, tokenize=False)  # renders: roles alternate, no system or tool role
    assert any('"command": "ls"' in m["content"] for m in adapted if m["role"] == "assistant")
    s = run_label(out=tmp_path / "g", pack="toy", template="google/gemma-2-2b-it")
    assert s["datasets"]["sft"] > 0 and not any(k.endswith("template_refused") for k in s["skipped"])


# ---- G1 D15 / G2 D7 / G2 D6: the next hint, non-run directories, nothing written on refusal --------------------------------
def test_D15_G2D7_G2D6_next_hint_non_run_dirs_and_refusals_write_nothing(labelled_run, fixture_run, tmp_path):
    _, s = labelled_run
    assert not any("crucible.train.launch" in n for n in s["next"]) and any("load_dataset" in n for n in s["next"])
    (tmp_path / "notarun").mkdir()
    with pytest.raises(ValueError, match="not an M10 run"):
        run_label(out=tmp_path / "o1", run=tmp_path / "notarun")
    with pytest.raises(ValueError, match="tokenizer"):
        run_label(out=tmp_path / "o2", run=fixture_run, tokenizer="no-such-org/no-such-model")
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()
    with pytest.raises(ValueError, match="max-per-task"):
        run_label(out=tmp_path / "o3", run=fixture_run, max_per_task=0)
    assert not (tmp_path / "o3").exists()
    out, _ = labelled_run
    assert not list(out.glob(".*.tmp"))


# ---- G2 D6: a re-run with other flags into the same --out is refused before anything is written ---------------------------
def test_G2D6_rerun_with_other_flags_is_refused_and_nothing_is_rewritten(tmp_path):
    out = tmp_path / "lab"
    run_label(out=out, pack="toy")
    before = {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()}
    with pytest.raises(ValueError, match="use a new --out"):
        run_label(out=out, pack="toy", select="all")
    assert {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()} == before
    cfg = json.loads((out / "configs" / "resolved" / "label.json").read_text())
    assert cfg["identity"]["select"] == "correct" and cfg["provenance"]["pack"]["content_sha256"]
    run_label(out=out, pack="toy")  # the same flags resume into the same --out
    before = {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()}
    from click.testing import CliRunner

    from crucible.cli.main import cli

    r = CliRunner().invoke(cli, ["label", "--pack", "toy", "--out", str(out), "--sft-format", "chain"])
    assert r.exit_code == 4 and "use a new --out" in r.output and "sft_format" in r.output  # a user error, nothing rewritten
    assert {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()} == before


def test_cli_label_flags_reach_label(fixture_run, tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    r = CliRunner().invoke(cli, ["label", "--run", str(fixture_run), "--out", str(tmp_path / "c"), "--allow-scripted", "--policy", "scripted:oracle",
                                 "--tokenizer", QWEN, "--json"], catch_exceptions=False)
    s = json.loads(r.output)
    assert r.exit_code == 0 and s["config"]["allow_scripted"] and s["config"]["policies"] == ["scripted:oracle"] and s["config"]["tokenizer"] == QWEN
    assert {x["policy"] for x in _all_rows(tmp_path / "c")} == {"scripted:oracle"} and s["skipped"]["other_policy"] > 0


# ---- D13 / scripted demonstrations: one shape per file, the World's tool cards, reasoning kept -------------------------------
def test_D13_demo_rows_carry_the_tool_schema_and_never_share_a_file_with_token_rows(fixture_run, tmp_path):
    out = tmp_path / "demo"
    s = run_label(out=out, run=fixture_run, allow_scripted=True)
    for f in ("sft", "kto"):  # TRL picks text vs conversational from a dataset's first row: never mixed in one file
        assert all(isinstance(r["prompt"], str) for r in rows(out / f"{f}.jsonl")), f
        chat = rows(out / f"{f}.chat.jsonl")
        assert chat and all(isinstance(r["prompt"], list) and r["source"] == "scripted_demo" for r in chat), f
        assert s["files"][f"{f}_chat"] == str(out / f"{f}.chat.jsonl")
    names = {t["function"]["name"] for t in rows(out / "sft.chat.jsonl")[0]["tools"]}
    assert names == {"tickets__get_ticket", "tickets__issue_refund"}  # the World's tool cards, as the calls name them
    called = {c["function"]["name"] for r in rows(out / "sft.chat.jsonl") for m in r["completion"] for c in m.get("tool_calls") or []}
    assert called and called <= names
    assert any(m.get("reasoning_content") == "Read the ticket first." for r in rows(out / "sft.chat.jsonl") for m in r["completion"])
    if NEMO_PY.exists():
        got = _hf_load([p for p in sorted(out.glob("*.jsonl")) if p.stat().st_size])
        first = got[str(out / "sft.chat.jsonl")]["first"]
        args = [c["function"]["arguments"] for m in first["completion"] for c in m.get("tool_calls") or []]
        assert all(isinstance(a, dict) and None not in a.values() for a in args)  # Json() arguments, no null-filled keys


def test_G2D7_label_run_on_a_non_run_dir_fails_non_zero_through_the_cli(tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    (tmp_path / "notarun").mkdir()
    r = CliRunner().invoke(cli, ["label", "--run", str(tmp_path / "notarun"), "--out", str(tmp_path / "o")])
    assert r.exit_code != 0 and "not an M10 run" in r.output and "next" not in r.output and "train" not in r.output
    assert not (tmp_path / "o").exists()


def test_V9_coding_runs_on_bench_seeds_are_held_out():
    pytest.importorskip("crucible_pack_coding")
    from crucible_pack_coding.studio import credit

    from crucible.core.schema.trace import Trace, TraceEvent, TraceOutcome, TraceSource

    ids = ("task02_pcops_triton/gpt-5.6-sol-run1", "task01_prims_omp/fable-5.1-run1")  # a bench seed; the superseded non-bench seed
    if not all((CRUCIBLE.parent / "migration_tasks" / i.split("/")[0] / "runs" / i.split("/")[1]).is_dir() for i in ids):
        pytest.skip("migration_tasks runs not in this checkout")
    ts = [Trace(trace_id=f"run:{i}", family="coding/agent_run", source=TraceSource(connector="agent_runs", ref=i),
                events=[TraceEvent(seq=0, kind="message", actor="a", type="m", text="x")], outcome=TraceOutcome(label="unknown")) for i in ids]
    got = asyncio.run(credit(ts))
    assert got[f"run:{ids[0]}"]["split"] == "bench" and got[f"run:{ids[1]}"]["split"] == "train"
    assert [t["function"]["name"] for t in got[f"run:{ids[1]}"]["tools"]] == ["bash"]  # the recorded harness's tool card (G1 D13)


def test_G1_D13_runner_writes_tool_definitions_into_atif():
    """The runner's ATIF trajectories carry the actor's tool cards, so label rows from real runs have a real ``tools``."""
    from types import SimpleNamespace

    from crucible.core.schema.actions import ToolSpec
    from crucible.runtime.harness.native import tool_cards

    w = SimpleNamespace(tools_for=lambda a: [ToolSpec(app="erp", name="approve", description="Approve an invoice",
                                                         input_schema={"type": "object", "properties": {"invoice_id": {"type": "integer"}}})])
    cards = tool_cards(w, "agent")
    assert cards == [{"type": "function", "function": {"name": "erp.approve", "description": "Approve an invoice",
                                                       "parameters": {"type": "object", "properties": {"invoice_id": {"type": "integer"}}}}}]


# ---- campaign findings fixed by the lead (B19, B20, B21) ----------------------------------------------------------------
def test_B19_an_episode_without_a_training_view_is_never_data(fixture_run, tmp_path):
    """No ``rollout.json``: flags, mask and trainable turns are unknown, so the episode is skipped even under
    ``allow_scripted`` (it was classed scripted and its flagged passes became positives)."""
    run = tmp_path / "run"
    shutil.copytree(fixture_run, run)
    for f in run.glob("samples/*/a*/rollout.json"):
        f.unlink()
    skipped: Counter = Counter()
    out = ds.from_run(run, allow_scripted=True, skipped=skipped)
    assert out == [] and skipped["no_training_view"] > 0


def test_B20_a_pair_needs_evidence_at_the_turn_it_is_cut():
    u = {"role": "user", "content": "q"}
    a = lambda c: {"role": "assistant", "content": c}
    best, worst = ds.Labelled("t", [u, a("probe the hidden tests")], 1.0), ds.Labelled("t", [u, a("read the policy")], 0.0)
    skipped: Counter = Counter()
    assert ds.dpo_best_vs_worst([best, worst], skipped=skipped) == [] and skipped["dpo_no_evidence_at_turn"] == 1
    credited = ds.Labelled("t", [u, a("x")], 1.0, weights=[None, 0.5])
    flat = ds.Labelled("t", [u, a("y")], 0.0, weights=[None, 0.0])
    assert len(ds.dpo_best_vs_worst([credited, flat])) == 1  # dense credit at the cut separates them


def test_B21_kto_drops_a_turn_labelled_both_ways():
    u = {"role": "user", "content": "q"}
    a = lambda c: {"role": "assistant", "content": c}
    shared = a("look up the ticket")
    win = ds.Labelled("t", [u, shared, {"role": "user", "content": "ok"}, a("refund it")], 1.0)
    lose = ds.Labelled("t", [u, shared, {"role": "user", "content": "ok"}, a("close it")], 0.0)
    skipped: Counter = Counter()
    rows = ds.kto_rows([win, lose], skipped=skipped)
    texts = [(r["completion"][0]["content"] if isinstance(r["completion"], list) else r["completion"], r["label"]) for r in rows]
    assert ("look up the ticket", True) not in texts and ("look up the ticket", False) not in texts
    assert skipped["kto_conflicting_labels"] == 2
