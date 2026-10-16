"""M17 P: the two-way requirement ledger (R1) and the imperative-sentence finder."""

from __future__ import annotations

from crucible.factory import ledger as L
from crucible.factory.cards import CheckRef, Requirement, RequirementLedger
from tests.unit.m17_p_fixtures import PROMPT, card, ledger, obs, span

OBS = {"o1": obs("o1"), "o2": obs("o2", "human_graded_outcome", "human:a"), "ref": obs("ref", "reference_derived"), "sim": obs("sim", "simulator_state")}


def codes(problems):
    return sorted(p.split(":")[0] for p in problems)


def test_sound_ledger_has_no_problems():
    assert ledger().problems(OBS) == []
    assert ledger().problems(OBS, prompt=PROMPT) == []
    assert card().ledger_problems(OBS) == []


def test_unchecked_requirement():
    led = ledger().model_copy(update={"checks": ledger().checks[:1]})
    p = led.problems(OBS)
    assert codes(p) == ["unchecked_requirement"] and "'r2'" in p[0]


def test_check_with_no_requirement_is_unstated():
    led = ledger().model_copy(update={"checks": [*ledger().checks, CheckRef(id="c3", requirement_ids=[], observation_ids=["o1"], gating=True)]})
    p = led.problems(OBS)
    assert codes(p) == ["unstated_check"] and "'c3'" in p[0]


def test_unknown_requirement_and_observation_ids():
    led = ledger().model_copy(update={"checks": [CheckRef(id="c1", requirement_ids=["r1", "rX"], observation_ids=["o1", "oY"], gating=True),
                                                 ledger().checks[1]]})
    p = led.problems(OBS)
    assert codes(p) == ["unknown_observation", "unknown_requirement"]
    assert "'rX'" in " ".join(p) and "'oY'" in " ".join(p)


def test_observations_dict_missing_everything():
    assert codes(ledger().problems({})) == ["ungrounded_gate", "ungrounded_gate", "unknown_observation", "unknown_observation", "unknown_observation"]


def test_gating_check_needs_a_first_class_observation():
    def led(obs_ids, gating=True):
        return RequirementLedger(requirements=[Requirement(id="r1", text="t", prompt_span=(0, 3), kind="k")],
                                 checks=[CheckRef(id="c1", requirement_ids=["r1"], observation_ids=obs_ids, gating=gating)])

    assert codes(led(["ref"]).problems(OBS)) == ["ungrounded_gate"]
    assert codes(led(["sim", "ref"]).problems(OBS)) == ["ungrounded_gate"]
    assert codes(led([]).problems(OBS)) == ["ungrounded_gate"]
    assert led(["ref", "o1"]).problems(OBS) == []  # second-class together with a first-class one
    assert led(["ref"], gating=False).problems(OBS) == []  # a diagnostic check may rest on reference-derived state
    assert led([], gating=False).problems(OBS) == []


def test_span_outside_prompt_and_wrong_span_text():
    # a ledger built without a card, so the span is not rejected at construction
    far = RequirementLedger(requirements=[Requirement(id="r1", text="x y", prompt_span=(0, len(PROMPT) + 5), kind="k")],
                            checks=[CheckRef(id="c1", requirement_ids=["r1"], observation_ids=["o1"], gating=True)])
    assert codes(far.problems(OBS, prompt=PROMPT)) == ["span_out_of_prompt"]
    assert far.problems(OBS) == []  # no prompt, no span checks
    wrong = RequirementLedger(requirements=[Requirement(id="r1", text="Refunds over $50 must be approved by a manager.",
                                                        prompt_span=span("Never pay a duplicate invoice."), kind="k")],
                              checks=[CheckRef(id="c1", requirement_ids=["r1"], observation_ids=["o1"], gating=True)])
    p = wrong.problems(OBS, prompt=PROMPT)
    assert codes(p) == ["span_mismatch"] and "'r1'" in p[0]


def test_span_overlap_accepts_a_paraphrase_and_rejects_a_stranger():
    s = span("Refunds over $50 must be approved by a manager.")
    para = Requirement(id="r", text="manager approval needed for refunds above 50", prompt_span=s, kind="k")
    assert L._span_problems(para, PROMPT) == []  # manager, refunds, 50 of 6 content words: exactly the 50% bar
    weak = Requirement(id="r", text="manager sign-off needed for large payouts", prompt_span=s, kind="k")
    assert codes(L._span_problems(weak, PROMPT)) == ["span_mismatch"]  # only "manager" of 7 words
    tight = Requirement(id="r", text="refunds over 50 approved by manager", prompt_span=s, kind="k")
    assert L._span_problems(tight, PROMPT) == []
    stranger = Requirement(id="r", text="send a welcome email to new customers", prompt_span=s, kind="k")
    assert codes(L._span_problems(stranger, PROMPT)) == ["span_mismatch"]
    assert L.overlap("A  B", "a b c") == 1.0  # verbatim after folding case and whitespace


def test_every_code_is_documented():
    seen = set()
    for led, kw in [(ledger().model_copy(update={"checks": []}), {}), (RequirementLedger(checks=[CheckRef(id="c", gating=True)]), {}),
                    (RequirementLedger(checks=[CheckRef(id="c", requirement_ids=["z"], observation_ids=["q"], gating=True)]), {})]:
        seen |= set(codes(led.problems({}, **kw)))
    assert seen <= set(L.CODES)


# --- candidate sentences -----------------------------------------------------------------------------------------------


def test_sentence_spans_reproduce_text():
    text = "Process the batch. Refunds > $5.50 need e.g. approval! Is it done?\n- first item\n2) second item"
    for a, b in L.sentences(text):
        assert text[a:b] == text[a:b].strip() and text[a:b]
    got = [text[a:b] for a, b in L.sentences(text)]
    assert got == ["Process the batch.", "Refunds > $5.50 need e.g. approval!", "Is it done?", "- first item", "2) second item"]


def test_cues():
    assert "modal:must" in L.cues("Refunds must be approved.")
    assert "modal:never" in L.cues("Never pay twice.")
    assert L.cues("Create a ticket for each failure.") == ("imperative:create",)
    assert "imperative:send" in L.cues("- Then, send the summary to the team.")
    assert "request" in L.cues("Please look at the file")
    assert L.cues("The weather is fine today.") == ()
    assert L.cues("Is the invoice ready?") == ()  # a question is not an order
    assert L.cues("Users sometimes use the old form.") == ()  # 'use' mid-sentence is not imperative


def test_candidates_found_and_unmapped_flagged():
    cands = L.candidates(PROMPT)
    assert [PROMPT[c.start:c.end] for c in cands] == ["Process the invoice batch.", "Refunds over $50 must be approved by a manager.",
                                                       "Never pay a duplicate invoice.", "Reply with a one-line summary."]
    # the ledger fixture covers refunds and duplicates; the batch and the reply format are not recorded
    flagged = [c.text for c in L.unmapped_candidates(PROMPT, ledger().requirements)]
    assert flagged == ["Process the invoice batch.", "Reply with a one-line summary."]
    assert L.unmapped_candidates(PROMPT, []) == cands
    all_reqs = [*ledger().requirements,
                Requirement(id="r3", text="Process the invoice batch.", prompt_span=span("Process the invoice batch."), kind="k"),
                Requirement(id="r4", text="Reply with a one-line summary.", prompt_span=span("Reply with a one-line summary."), kind="k")]
    assert L.unmapped_candidates(PROMPT, all_reqs) == []


def test_a_wide_span_does_not_cover_unrelated_sentences():
    wide = Requirement(id="r", text="Refunds over $50 must be approved by a manager.", prompt_span=(0, len(PROMPT)), kind="k")
    flagged = [c.text for c in L.unmapped_candidates(PROMPT, [wide])]
    assert "Reply with a one-line summary." in flagged and "Never pay a duplicate invoice." in flagged
    assert "Refunds over $50 must be approved by a manager." not in flagged


def test_wording_alone_maps_a_sentence_wherever_the_span_is():
    elsewhere = Requirement(id="r", text="Reply with a one-line summary", prompt_span=(0, 5), kind="k")
    assert "Reply with a one-line summary." not in [c.text for c in L.unmapped_candidates(PROMPT, [elsewhere])]


def test_extraction_is_deterministic():
    assert L.candidates(PROMPT) == L.candidates(PROMPT)
