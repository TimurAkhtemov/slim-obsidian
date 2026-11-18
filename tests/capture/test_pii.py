import pytest

from slim import config
from slim.capture import pii


def _orig(text, a, b):
    norm, idx = pii.normalise(text)
    return text[idx[a]:idx[b]]


def test_digit_words_join():
    assert pii.normalise("one two three four")[0] == "1234"


def test_oh_and_zero_are_zero():
    assert pii.normalise("five oh two")[0] == "502"


def test_double_and_triple():
    assert pii.normalise("double five triple two")[0] == "55222"


def test_tens_and_teens():
    assert pii.normalise("twenty one")[0] == "21"
    assert pii.normalise("fourteen")[0] == "14"


def test_hundred():
    assert pii.normalise("two hundred and five")[0] == "205"


def test_groups_keep_their_separators():
    assert pii.normalise("one two three, four five, six")[0] == "123, 45, 6"


def test_digits_and_words_mix():
    assert pii.normalise("12 three four")[0] == "12 34"


def test_non_numbers_untouched_and_mapped():
    text = "call me at one two three please"
    norm, idx = pii.normalise(text)
    assert norm == "call me at 123 please"
    a = norm.index("123")
    assert _orig(text, a, a + 3) == "one two three"


def test_identity_when_no_numbers():
    text = "**Me:** nothing here"
    norm, idx = pii.normalise(text)
    assert norm == text and idx == list(range(len(text) + 1))


def test_case_and_hyphen():
    assert pii.normalise("Twenty-One")[0] == "21"


def test_tens_then_unit_inside_a_group():
    # chosen reading: "two twenty one" is "2" then "21"
    assert pii.normalise("two twenty one")[0] == "221"


def test_and_only_consumed_inside_hundred():
    assert pii.normalise("one and two")[0] == "1 and 2"
    assert pii.normalise("two hundred and")[0] == "200 and"


def test_offsets_after_group_and_map_length():
    text = "x one two y"
    norm, idx = pii.normalise(text)
    assert norm == "x 12 y" and len(idx) == len(norm) + 1
    a = norm.index("y")
    assert text[idx[a]:] == "y"
    assert idx[len(norm)] == len(text)


def test_dashed_ssn_masked():
    m = pii.mask("my social is 219-09-9999 ok", classes=config.PII_DEFAULT)
    assert m.text == "my social is [SSN] ok" and m.counts == {"ssn": 1}
    assert m.secrets == ["219-09-9999"]


def test_spoken_ssn_with_cue():
    t = "**Them:** social security number two one nine, oh nine, nine nine nine nine."
    m = pii.mask(t, classes=config.PII_DEFAULT)
    assert "[SSN]" in m.text and "nine nine" not in m.text and m.text.startswith("**Them:**")


def test_invalid_ssn_areas_not_masked():
    # only without a cue: a number bound to an SSN cue is masked whatever its range says
    for v in ("000-12-3456", "666-12-3456", "912-12-3456"):
        assert pii.mask(f"ref {v}", classes=config.PII_DEFAULT).counts == {}
        assert pii.mask(f"ssn {v}", classes=config.PII_DEFAULT).text == "ssn [SSN]"


def test_run_together_nine_digits_needs_cue():
    assert pii.mask("order 219099999 shipped", classes=config.PII_DEFAULT).counts == {}
    assert pii.mask("my SSN is 219099999", classes=config.PII_DEFAULT).counts == {"ssn": 1}


def test_card_needs_luhn():
    # a Luhn-failing number bound to a card cue is still a card (a misheard digit), so
    # the Luhn guard is tested without one.
    assert pii.mask("ref 4111 1111 1111 1111", classes=config.PII_DEFAULT).counts == {"card": 1}
    assert pii.mask("ref 4111 1111 1111 1112", classes=config.PII_DEFAULT).counts == {}


def test_spoken_card_is_covered_whole():
    t = "card four one one one one one one one one one one one one one one one then five"
    m = pii.mask(t, classes=config.PII_DEFAULT)
    assert m.text == "card [CARD] then five"


def test_hit_ending_inside_a_spoken_group_is_widened():
    # the SSN pattern ends inside "nine nine nine nine nine": the whole phrase goes
    t = "ssn two one nine oh nine nine nine nine nine"
    m = pii.mask(t, classes=config.PII_DEFAULT)
    assert m.text == "ssn [SSN]"


def test_iban_and_routing_account():
    assert pii.mask("IBAN GB82 WEST 1234 5698 7654 32", classes=config.PII_DEFAULT).counts == {"account": 1}
    m = pii.mask("routing number 011000028 account number 12345678", classes=config.PII_DEFAULT)
    assert m.counts == {"account": 2}


def test_email_written_and_spoken():
    assert pii.mask("write to a.b@example.com", classes=config.PII_DEFAULT).text == "write to [EMAIL]"
    m = pii.mask("it's jane dot doe at example dot com", classes=config.PII_DEFAULT)
    assert m.text == "it's [EMAIL]" and m.secrets == []


def test_address():
    m = pii.mask("I live at two twenty one Baker Street, apartment 4, 90210 now",
                 classes=config.PII_DEFAULT)
    assert m.text == "I live at [ADDRESS] now"


def test_street_without_number_kept():
    assert pii.mask("she lives on Baker Street", classes=config.PII_DEFAULT).counts == {}


def test_phone_off_by_default_on_when_listed():
    t = "call 415 555 0134"
    assert pii.mask(t, classes=config.PII_DEFAULT).counts == {}
    assert pii.mask(t, classes=(*config.PII_DEFAULT, "phone")).text == "call [PHONE]"


def test_ordinary_numbers_survive():
    for t in ("meet at two thirty", "it was four ninety nine", "back in 2019",
              "order number 5512 3349", "call nine one one", "page 123 of 456"):
        assert pii.mask(t, classes=config.PII_DEFAULT).text == t


def test_untouched_text_byte_identical():
    t = "**Me:**  odd  spacing\n\n**Them:** a.b@example.com  end "
    assert pii.mask(t, classes=config.PII_DEFAULT).text == "**Me:**  odd  spacing\n\n**Them:** [EMAIL]  end "


def test_empty_class_list_is_off():
    assert pii.mask("ssn 219-09-9999", classes=()).text == "ssn 219-09-9999"


def test_classes_none_reads_config(monkeypatch):
    monkeypatch.setattr(pii.config, "PII_MASK", ())
    assert pii.mask("ssn 219-09-9999").text == "ssn 219-09-9999"


def test_frontmatter_shape():
    assert pii.frontmatter({"ssn": 1, "email": 2}, ["password", "audio"]) == {
        "pii_masked": ["email:2", "ssn:1"], "pii_by": "pii-1", "pii_partial": ["audio", "password"]}


def test_frontmatter_complete_has_no_partial_and_names_only_known_reasons():
    assert pii.frontmatter({}) == {"pii_masked": [], "pii_by": "pii-1"}
    assert pii.frontmatter({"ssn": 1}, ["audio", "audio", "bogus"]) == {
        "pii_masked": ["ssn:1"], "pii_by": "pii-1", "pii_partial": ["audio"]}


def _same(t):
    assert pii.mask(t, classes=config.PII_DEFAULT).text == t


def test_number_lists_survive():
    for t in ("we got 606, 968, 105, 924, 326, 32 votes", "prices 40 35 69 67 89 66 61 11 16",
              "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen",
              "we had 300, 25, 2000 users", "rooms 101, 22, 3045"):
        _same(t)


def test_spaced_ssn_needs_cue():
    assert pii.mask("my social 219 09 9999 ok", classes=config.PII_DEFAULT).counts == {"ssn": 1}
    assert pii.mask("ssn 219, 09, 9999", classes=config.PII_DEFAULT).counts == {"ssn": 1}
    _same("rooms 219, 09, 9999")


def test_card_shape_or_cue():
    assert pii.mask("4111 1111 1111 1111", classes=config.PII_DEFAULT).counts == {"card": 1}
    assert pii.mask("my visa is 4111 1111 1111 1112", classes=config.PII_DEFAULT).counts == {"card": 1}
    # a card is contiguous or read in groups of 3+ digits, cued or not
    _same("my visa is 41 11 1111 1111 1111")
    _same("ref 91 60 1562 2746 67")


def test_spoken_email_needs_dot_or_cue():
    for t in ("I looked at amazon dot com yesterday", "I met her at Starbucks dot com",
              "check it out at google dot com"):
        _same(t)
    # "me" never forms a local part on its own
    _same("email me at jane dot com")
    m = pii.mask("my email is jane at example dot com", classes=config.PII_DEFAULT)
    assert m.text == "my email is [EMAIL]"


def test_spaced_account():
    m = pii.mask("account number 1234 5678", classes=config.PII_DEFAULT)
    assert m.text == "account number [ACCOUNT]"


CLS = config.PII_DEFAULT


def test_password_found_is_masked():
    m = pii.mask("ok the wifi password is hunter2, all lowercase. Next topic.",
                 classes=CLS, find_password=lambda w: "hunter2")
    assert "hunter2" not in m.text and "[PASSWORD]" in m.text
    assert m.counts == {"password": 1} and m.secrets == ["hunter2"]


def test_model_only_points_never_writes():
    m = pii.mask("the door code is four four one two.", classes=CLS,
                 find_password=lambda w: "9999")
    assert m.text == "the door code is four four one two." and m.counts == {}


def test_no_cue_no_call():
    calls = []
    pii.mask("nothing secret here", classes=CLS, find_password=lambda w: calls.append(w))
    assert calls == []


def test_null_answer_masks_nothing():
    t = "I forgot my password again."
    assert pii.mask(t, classes=CLS, find_password=lambda w: None).text == t


def test_failure_is_partial_not_raised(monkeypatch):
    recorded = []
    monkeypatch.setattr(pii.trace, "record", lambda kind, payload: recorded.append(payload))

    def boom(w):
        raise RuntimeError("server down 4412")
    m = pii.mask("my PIN is 4412 and ssn 219-09-9999", classes=CLS, find_password=boom)
    assert m.partial and m.counts == {"ssn": 1}
    assert recorded == [{"stage": "password", "status": "failed", "error": "RuntimeError"}]


def test_no_finder_skips_pass():
    m = pii.mask("password is hunter2", classes=CLS, find_password=None)
    assert m.text == "password is hunter2" and not m.partial


def test_window_bounds_and_cap():
    long = "x" * 80
    m = pii.mask(f"password is {long}", classes=CLS, find_password=lambda w: long)
    assert m.counts == {}          # over 64 chars: refused


def test_answer_not_in_window_is_ignored():
    t = "the password is hunter2 ok."
    assert pii.mask(t, classes=CLS, find_password=lambda w: "Hunter3").text == t
    # a tidied answer (case, spacing, spoken digits) is located in the window text instead
    assert pii.mask(t, classes=CLS, find_password=lambda w: "Hunter2").text == "the password is [PASSWORD] ok."


def test_every_occurrence_in_window_masked_and_model_never_sees_patterns():
    seen = []

    def finder(w):
        seen.append(w)
        return "hunter2"
    m = pii.mask("ssn 219-09-9999. password hunter2, again hunter2.", classes=CLS, find_password=finder)
    assert m.text.count("[PASSWORD]") == 2 and "hunter2" not in m.text
    assert all("219-09-9999" not in w for w in seen)
    assert m.secrets == ["219-09-9999", "hunter2", "hunter2"]


def test_window_crosses_at_most_one_turn_start():
    seen = []
    pii.mask("**Me:** the password is\n**Them:** sure.\n\n**Me:** secret9 right.", classes=CLS,
             find_password=lambda w: seen.append(w))
    assert seen and all("**Them:**" in w and "secret9" not in w for w in seen)


def test_ask_model_sends_bounded_schema_call(monkeypatch):
    got = {}

    def fake(messages, schema, **kw):
        got.update(messages=messages, schema=schema, kw=kw)
        return {"secret": "hunter2"}, {}
    monkeypatch.setattr(pii.llm, "chat_json", fake)
    assert pii.ask_model("the password is hunter2") == "hunter2"
    # A cold start reloads ~32 GB (llama-server unloads after 10 idle minutes): 60 s was a
    # `password` partial. A 64-character secret plus JSON framing needs more than 64 tokens.
    assert got["kw"]["think"] is False and got["kw"]["num_predict"] == 128
    assert got["kw"]["timeout"] == 180
    assert got["schema"]["properties"]["secret"]["maxLength"] == 64


def test_short_or_function_word_answers_are_rejected():
    t = "the password is a big deal, ab, it is the one. Okay?"
    for ans in ("a", "is", "ab", "the", "it is", "is the", "Okay"):
        m = pii.mask(t, classes=CLS, find_password=lambda w, a=ans: a)
        assert m.text == t and m.counts == {}


def test_answer_before_the_cue_sentence_is_not_masked():
    m = pii.mask("hunter2 was mine. the password is hunter2.", classes=CLS, find_password=lambda w: "hunter2")
    assert m.text == "hunter2 was mine. the password is [PASSWORD]."


def test_placeholder_fragments_rejected():
    t = "ssn 219-09-9999 and the password is blank."
    for ans in ("SSN]", "[SSN]"):
        m = pii.mask(t, classes=CLS, find_password=lambda w, a=ans: a)
        assert m.text == "ssn [SSN] and the password is blank." and m.secrets == ["219-09-9999"]


def test_failure_in_later_window_keeps_earlier_password():
    def finder(w):
        if "hunter2" in w:
            return "hunter2"
        raise RuntimeError("down")
    m = pii.mask("the password is hunter2.\n\nthe PIN is later.", classes=CLS, find_password=finder)
    assert m.partial and m.counts == {"password": 1} and "hunter2" not in m.text


def test_spans_map_secret_to_token_times():
    toks = [(" my", 0.0, 0.2), (" ssn", 0.2, 0.5), (" 219", 0.5, 0.9), ("-09", 0.9, 1.1),
            ("-9999", 1.1, 1.6), (" ok", 1.6, 1.8)]
    assert pii.spans(toks, ["219-09-9999"]) == ([(0.5, 1.6)], 0)


def test_spoken_secret_maps_through_words():
    toks = [(" two", 0.0, 0.3), (" one", 0.3, 0.5), (" nine", 0.5, 0.8)]
    assert pii.spans(toks, ["two one nine"]) == ([(0.0, 0.8)], 0)


def test_unmatched_secret_is_counted():
    assert pii.spans([(" hi", 0, 1)], ["hunter2"]) == ([], 1)


def test_unmatched_secret_does_not_stop_later_ones_and_cache_lists_work():
    toks = [[" a", 0.0, 0.1], [" pw", 0.1, 0.4], [" b", 0.4, 0.6]]
    assert pii.spans(toks, ["nope", "pw"]) == ([(0.1, 0.4)], 1)


def test_every_occurrence_is_muted():
    toks = [(" the", 0, 1), (" password", 1, 2), (" is", 2, 3), (" blue", 3, 4),
            (" sky", 4, 5), (" blue", 5, 6)]
    assert pii.spans(toks, ["blue"]) == ([(3, 4), (5, 6)], 0)
    toks = [(" nine", 0, 1), (" x", 1, 2), (" nine", 2, 3)]
    assert pii.spans(toks, ["nine"]) == ([(0, 1), (2, 3)], 0)


def test_duplicate_secrets_do_not_double_count():
    toks = [(" x", 0, 1), (" y", 1, 2)]
    assert pii.spans(toks, ["x", "x", "zz", "zz"]) == ([(0, 1)], 1)


def test_overlapping_and_touching_spans_merge():
    toks = [(" a", 0, 1), (" b", 1, 2), (" c", 2, 3), (" d", 5, 6)]
    assert pii.spans(toks, ["a b", "b c", "d"]) == ([(0, 3), (5, 6)], 0)
    assert pii.spans(toks, ["a", "b"]) == ([(0, 2)], 0)


# --- what the transcriber writes (synthetic values; Parakeet's rendering first, the
# spoken-word rendering second) ---------------------------------------------------------------

def _m(t, classes=CLS):
    return pii.mask(t, classes=classes)


def _luhn_complete(body: str) -> str:
    """body plus the Luhn check digit."""
    return next(body + d for d in "0123456789" if pii._luhn(body + d))


# normalise

def test_dash_between_number_groups_becomes_a_dash():
    assert pii.normalise("six four four dash zero four dash eight nine zero one")[0] == "644-04-8901"
    assert pii.normalise("644 dash 04 hyphen 8901")[0] == "644-04-8901"
    assert pii.normalise("five minus two")[0] == "5-2"


def test_dash_elsewhere_untouched_and_offsets_kept():
    for t in ("a dash of salt", "five dash", "dash five then", "make a dash for it at nine"):
        norm, idx = pii.normalise(t)
        assert "-" not in norm and len(idx) == len(norm) + 1
    text = "x six four four dash zero four y"
    norm, idx = pii.normalise(text)
    assert norm == "x 644-04 y"
    a = norm.index("644")
    assert _orig(text, a, a + 6) == "six four four dash zero four"
    assert _orig(text, a, a + 3) == "six four four"


# SSN

def test_ssn_as_parakeet_writes_it():
    assert _m("**Them:** Sure. My social is 741 05 7024.").text == "**Them:** Sure. My social is [SSN]."
    assert _m("**Them:** 644 04 8901. That's my social.").text == "**Them:** [SSN]. That's my social."
    assert _m("**Them:** 692 87 1253. That's my social.").counts == {"ssn": 1}
    assert _m("**Me:** Take this down. 493-24-2862.").counts == {"ssn": 1}
    t = "**Me:** Can I get your social security number?\n\n**Them:** Yeah, it's 030809137."
    assert _m(t).text.endswith("it's [SSN].")


def test_ssn_as_words():
    m = _m("**Them:** Sure, my social is seven forty one dash zero five dash seventy twenty four.")
    assert m.text == "**Them:** Sure, my social is [SSN]."
    assert _m("**Them:** six four four dash zero four dash eight nine zero one, that's my social.").text \
        == "**Them:** [SSN], that's my social."
    assert _m("**Them:** six ninety two, eighty seven, twelve fifty three, that's my social.").text \
        == "**Them:** [SSN], that's my social."
    assert _m("**Me:** Take this down, four nine three dash two four dash two eight six two.").counts == {"ssn": 1}


def test_spaced_324_triple_standing_alone_is_an_ssn_without_cue():
    assert _m("rooms 219 09 9999").text == "rooms [SSN]"
    assert _m("rooms 12, 219 09 9999").text == "rooms 12, [SSN]"           # a value ends at a comma
    for t in ("rooms 12 219 09 9999", "rooms 219 09 9999 12",
              "rooms 219 09 9999-1", "rooms 219  09 9999", "rooms 219, 09, 9999", "rooms 000 09 9999"):
        _same(t)


def test_cued_ssn_any_grouping_and_cue_after():
    for t in ("ssn 692 871253", "219-099999 is my social", "219, 09, 9999, that's my social",
              "219099999. That's my social."):
        assert _m(t).counts == {"ssn": 1}, t


def test_garbled_ssn_with_cue_still_masked():
    assert _m("**Me:** Sure. My social is 6858199.").text == "**Me:** Sure. My social is [SSN]."
    t = "**Them:** Can I get your social security number?\n\n**Me:** Yeah, it's 60590647."
    assert _m(t).text.endswith("it's [SSN].")
    # a strong cue ("my social is") holds a garbled number of up to twelve digits
    assert _m("my social is 123456789012 long").text == "my social is [SSN] long"
    for t in ("my social club has 123456 members", "my social is 1234567890123 long"):
        _same(t)


def test_ssn_cue_binding():
    _same("social " + "x" * 60 + " 219, 09, 9999")
    _same("**Them:** 219, 09, 9999.\n\n**Me:** That's my social.")   # after: never across a turn
    _same("219, 09, 9999 and then my social")                          # after: directly only
    _same("social " + "x" * 10 + " 219, 09, 9999")                     # before: filler words only
    assert _m("social, um, well, it's 219, 09, 9999").counts == {"ssn": 1}


# card

def test_card_as_parakeet_writes_it():
    assert _m("**Them:** it on my visa. 4111 7962 7393 8077.").counts == {"card": 1}
    t = "**Them:** What's the card number?\n\n**Me:** Okay. 4242, 7844, 6476, 8778."
    assert _m(t).text.endswith("**Me:** Okay. [CARD].")
    assert _m("**Them:** 371450948481385 is the card.").text == "**Them:** [CARD] is the card."
    assert _m("**Me:** Okay, 3782 332494 74751.").text == "**Me:** Okay, [CARD]."
    assert _m("**Me:** Here you go. 378247629514895.").counts == {"card": 1}


def test_card_as_words():
    for t in ("Put it on my visa, forty one eleven dash seventy nine sixty two dash seventy three "
              "ninety three dash eighty seventy seven.",
              "Here you go, four two four two dash two four eight one dash zero four one seven dash "
              "six five eight six.",
              "What's the card number?\n\n**Me:** Okay, forty two forty two, seventy eight forty four, "
              "sixty four seventy six, eighty seven seventy eight.",
              "My card is three seven eight two, three three two four nine four, seven four seven five one."):
        m = _m(t)
        assert m.counts == {"card": 1} and not any(c.isdigit() for c in m.text), t


def test_card_brand_table():
    good = [("4", 13), ("4", 16), ("4", 19), ("51", 16), ("55", 16), ("2221", 16), ("2720", 16),
            ("34", 15), ("37", 15), ("6011", 16), ("6011", 19), ("644", 16), ("649", 17), ("65", 18),
            ("300", 14), ("305", 14), ("36", 14), ("38", 14), ("39", 14),
            ("3528", 16), ("3589", 19), ("62", 16), ("62", 19)]
    bad = [("4", 15), ("50", 16), ("56", 16), ("2220", 16), ("2721", 16), ("34", 16), ("37", 14),
           ("643", 16), ("6011", 15), ("306", 14), ("36", 16), ("3527", 16), ("3590", 16),
           ("62", 15), ("9", 16), ("1", 16), ("8", 14)]
    for prefix, n in good:
        card = _luhn_complete(prefix + "1" * (n - len(prefix) - 1))
        assert _m(f"ref {card} ok").text == "ref [CARD] ok", (prefix, n)
    for prefix, n in bad:
        card = _luhn_complete(prefix + "1" * (n - len(prefix) - 1))
        _same(f"ref {card} ok")


def test_luhn_without_brand_needs_a_cue_and_a_cue_rescues_a_misheard_digit():
    nobrand = _luhn_complete("9" + "1" * 14)
    _same(f"ref {nobrand}")
    _same(f"my card is {nobrand}")                                       # a bound card needs a brand prefix
    assert _m("my card is 4111 1111 1111 111").counts == {"card": 1}      # 15 digits: no Luhn, no length
    assert _m("card 4111 1111 1111 1112").counts == {"card": 1}             # Luhn fails
    assert _m("4111 1111 1111 1112 is the card").counts == {"card": 1}      # cue after
    assert _m("my visa, 4111, 1111, 1111, 1112").counts == {"card": 1}     # `, ` groups
    _same("my card " + "x" * 60 + " 4111 1111 1111 1112")
    _same("card 4111 1111 111")                                             # 11 digits


def test_comma_cards_need_a_cue():
    # no comma shape is a card without a cue ("The counts were 4187, 1143, 5235, 4423.")
    _same("ref 42, 42, 7844, 6476, 8778")
    _same("ref 4242, 7844, 6476, 8778")
    assert _m("card number 4242, 7844, 6476, 8778").counts == {"card": 1}


def test_bank_strings_and_timestamps_are_not_cards():
    for t in ("ref 91601562274667", "at 20221015123456", "id 5560099292349456 0"):
        _same(t)


# account

def test_iban_glued_to_its_cue_and_spaced():
    assert _m("**Them:** Sure. IBANGB73 HWEC 4532 6581 9231 44.").text == "**Them:** Sure. IBAN[ACCOUNT]."
    assert _m("**Me:** It's GB66GHJW13441818266639.").text == "**Me:** It's [ACCOUNT]."
    assert _m("IBAN GB82 WEST 1234 5698 7654 32 ok").text == "IBAN [ACCOUNT] ok"


def test_spelled_iban_with_cue():
    t = ("**Them:** Sure, IBAN G B seven three, H W E C, four five three two, six five eight one, "
         "nine two three one, four four.")
    assert _m(t).text == "**Them:** Sure, IBAN [ACCOUNT]."
    t = "**Them:** What's the IBAN?\n\n**Me:** It's G B six six G H J W one three four four one eight " \
        "one eight two six six six three nine."
    assert _m(t).text.endswith("**Me:** It's [ACCOUNT].")
    t = ("**Them:** G B one seven L T M X eight eight two zero eight three three two three zero eight "
         "one six nine, that's the IBAN.")
    assert _m(t).text == "**Them:** [ACCOUNT], that's the IBAN."
    _same("**Them:** It's G B six six G H J W one three four four one eight one eight.")  # no cue


def test_cued_account_digits():
    assert _m("**Them:** 80608689949. That's the account number.").text \
        == "**Them:** [ACCOUNT]. That's the account number."
    assert _m("**Them:** eight zero six zero eight six eight nine nine four nine, that's the account "
              "number.").text == "**Them:** [ACCOUNT], that's the account number."
    assert _m("My routing number is 011000028.").text == "My routing number is [ACCOUNT]."
    assert _m("acct no. 1234-5678-900").counts == {"account": 1}
    assert _m("acct no. 1234-5678-90").counts == {"account": 1}   # one short end group, dash-read
    _same("account number " + "x" * 60 + " 865047")
    _same("account number is 12345")                                      # under 6 digits


def test_cued_alphanumeric_account():
    assert _m("Account No. E11248142150, Date").text == "Account No. [ACCOUNT], Date"
    assert _m("Account number: ACCT-911155810.").text == "Account number: [ACCOUNT]."
    assert _m("the BBAN: RBKJ59644698401273.").text == "the BBAN: [ACCOUNT]."
    _same("Your policy number is GPTZ91601562274667.")                     # no cue
    _same("account number is ABCDEFGH12345")                               # 5 digits


def test_routing_needs_a_bound_cue():
    # a routing number needs a cue ("Order 289303696, thanks." was masked when it did not)
    for t in ("**Them:** Here it is. 035 690 281.", "it's 011000028", "Order 289303696, thanks.",
              "It's 015933395.", "Ship to 312973579.", "Scores 245 905 834."):
        _same(t)
    assert _m("My routing number is 035 690 281.").text == "My routing number is [ACCOUNT]."
    t = "**Them:** What's the routing number?\n\n**Them:** Here it is. 035 690 281."
    assert _m(t).text.endswith("Here it is. [ACCOUNT].")
    assert _m("routing number is 035, 690, 281").counts == {"account": 1}


# email

def test_email_as_parakeet_writes_it():
    assert _m("**Them:** email is vanholt at example.org.").text == "**Them:** email is [EMAIL]."
    for t in ("quarrel43 at example.org.", "Quirl94 at example.org.", "quarrel.telvane at example.org.",
              "Zorveth.steran at example.com.", "dev.ops at example.dev."):
        assert _m(f"**Them:** {t}").text == "**Them:** [EMAIL].", t
    assert _m("**Them:** Faxmore.ster and at example.com. That's my email.").text \
        == "**Them:** [EMAIL]. That's my email."


def test_email_as_words():
    assert _m("**Them:** My email is varnholt at example dot org.").text == "**Them:** My email is [EMAIL]."
    assert _m("**Me:** It's quorrel dot telvane at example dot org.").text == "**Me:** It's [EMAIL]."
    assert _m("faxmore sterran at example dot com, that's my email").text == "[EMAIL], that's my email"


def test_websites_are_not_emails():
    for t in ("I ordered it at Amazon.com.", "Can you send me the link? I bought it at Amazon.com.",
              "I bought it at amazon.com, I'll email you the receipt", "Meet me at Starbucks.com",
              "It was only $4.99 at Target.com.", "I bought 3 at amazon.com.",
              "I ordered it at amazon dot com.", "Can you send me the link? I bought it at amazon dot com.",
              "Write to us at example.org", "Contact them at example.org", "Look it up at wiki.org",
              "I'll email you back at example.com", "send it over at example.com",
              "Vandral.corvainatexample.org"):
        _same(t)


def test_email_cue_must_be_direct_or_identifying():
    _same("email: I said I bought six of them at example.com")   # more than 5 tokens between
    _same("I bought it at amazon.com, that's my favourite shop")
    assert _m("email: jane at example.com").text == "email: [EMAIL]"
    _same("jane at example.com is my email")                                # no bare "is"
    assert _m("jane at example.com, that's my email.").text == "[EMAIL], that's my email."
    _same("jane at example.com, email me")                                  # email as a verb
    _same("jane at example.com and " + "x" * 40 + " that's my email")       # beyond 40


# password windows

def _pw(t, answer):
    seen = []

    def finder(w):
        seen.append(w)
        return answer if answer in w else None
    return pii.mask(t, classes=CLS, find_password=finder), seen


def test_password_answered_in_the_next_turn():
    m, _ = _pw("**Them:** What's the password for the Wi Fi?\n\n**Me:** It's F-E-N-N-I-C-K 31.",
               "F-E-N-N-I-C-K 31")
    assert m.text.endswith("**Me:** It's [PASSWORD].") and m.counts == {"password": 1}
    m, _ = _pw("**Me:** What's the password for the Wi-Fi? It's\n\n**Them:** Alvon46.", "Alvon46")
    assert m.text.endswith("**Them:** [PASSWORD].")


def test_password_said_before_the_cue():
    m, _ = _pw("**Me:** Gavril 68 is my password.", "Gavril 68")
    assert m.text == "**Me:** [PASSWORD] is my password."
    m, _ = _pw("**Me:** O M W E L L 46 is my password.", "O M W E L L 46")
    assert m.text == "**Me:** [PASSWORD] is my password."


def test_password_window_is_two_more_sentences():
    m, seen = _pw("The password is coming. One. Two. hunter2 here.", "hunter2")
    assert m.counts == {} and seen == ["The password is coming. One. Two. "]


def test_overlapping_windows_do_not_double_mask():
    m, seen = _pw("The password is hunter2. The PIN is hunter2 too. ok.", "hunter2")
    assert len(seen) == 2
    assert m.text == "The password is [PASSWORD]. The PIN is [PASSWORD] too. ok."
    assert m.counts == {"password": 2} and m.secrets == ["hunter2", "hunter2"]


# phone

PHONE = (*CLS, "phone")


def test_phone_off_is_never_taken_by_another_class():
    for t in ("**Them:** Call me on my cell. 4035550184.", "**Them:** 4345550188. That's my cell.",
              "**Me:** My phone number is 724 555 0311.", "**Them:** Sure. 344 5550101.",
              "**Them:** Okay. 659-555-0158.", "**Them:** 792-3555-0104. That's my cell.",
              "Sure. 7195550128."):
        _same(t)


def test_phone_cue_free_and_cued():
    assert _m("Okay. 659-555-0158.", PHONE).text == "Okay. [PHONE]."
    assert _m("call (415) 555.0134 now", PHONE).text == "call [PHONE] now"
    assert _m("call +1 415 555 0134", PHONE).text == "call [PHONE]"
    for t in ("call 115 555 0134", "call 415 155 0134", "Sure. 7195550128.", "Sure. 344 5550101."):
        assert _m(t, PHONE).counts == {}, t
    assert _m("Call me on my cell. 4035550184.", PHONE).text == "Call me on my cell. [PHONE]."
    assert _m("4345550188. That's my cell.", PHONE).text == "[PHONE]. That's my cell."
    assert _m("my mobile is 344 5550101", PHONE).text == "my mobile is [PHONE]"
    assert _m("my cell is 495, 555, 0133", PHONE).text == "my cell is [PHONE]"   # a bare "number" is no phone cue


# priorities: an exact tie goes to the more harmful leak

def test_exact_tie_goes_to_the_higher_class():
    # bound to an account cue before and an SSN cue after: the same span, ssn wins
    assert pii.mask("account number 219099999, that's my social", classes=CLS).counts == {"ssn": 1}
    assert pii.mask("account number 4111111111111111", classes=CLS).counts == {"card": 1}


# ordinary speech, as Parakeet wrote it and as words

def test_ordinary_speech_survives():
    for t in ("I ordered it at Amazon.com.", "The counts were 194, 34, 67, 34.",
              "The counts were 761, 23, 7372.", "The counts were 488 632473.", "counts were 004 265075.",
              "That was back in 2019.", "That was back in 2008.", "It was only $19.95.", "It was only 499.",
              "It's on page 3.", "It's on page 212.", "I almost called 911.", "Can we do 1015?",
              "The vote was 41 to 30.", "The order number is 865047.", "Okay. 1, 2, 3, go.",
              "Okay. 12345. Go.", "It was only 1995.", "Can we do 230?",
              "The counts were one ninety four, thirty four, sixty seven thirty four.",
              "The counts were seven sixty one, twenty three, seven three seven two.",
              "The counts were four eight eight, sixty three, twenty four seventy three.",
              "The counts were oh oh four, two six, five oh seven five.",
              "The counts were one forty one, seventy three, seven six eight two.",
              "That was back in twenty nineteen.", "It was only nineteen ninety five.",
              "I almost called nine one one.", "call nine one one",
              "The order number is eight six five zero four seven.", "It's on page three."):
        _same(t)


# --- a number is masked on a cue only when bound to it ------------------------------------------
# Every sentence below was masked by the reach rules; each must survive.

MEASURED_ORDINARY = [
    "Credit union members: 3816, 6368, 1424, 8681.",
    "Credit hours were 3, 2, 2, 1, 3, 4, 1, 4, 4, 4, 2, 2.",
    "We dealt 8, 6, 6165, 8, 9772, 1, 52 cards.",
    "Social media followers: 1809, 114, 253.",
    "The social had 556, 63, 9998 people",
    "Routing was 8, 36, 93, 8267, 46, 3 packets",
    "I paid by card, order 974-1707372-9562114.",
    "Visa appointment on 3/12/2027, case 1552630858391413.",
    "The debit went through, confirmation 8841207736.",
    "Your account number is fine, the order was 5512334901.",
    "Sort code questions? Ticket INC3843979.",
    "My social security check came on 2023-08-13.",
    "The account number changed on 2020-09-19.",
    "The Social Security office is open 9 to 5, call (987) 989-3111.",
    "Call (230) 517-1266 about the account number.",
    "Our social media post got 24344906 views.",
    "The routing table has 257 entries and 5693234 lookups.",
    "I left social stuff aside for now. Anyway, call 454-1150.",
    "The counts were 4187, 1143, 5235, 4423.",
    "Order 289303696, thanks.", "It's 015933395.", "Ship to 312973579.", "Scores 245 905 834.",
    "Okay. 4 194 8548 2 8392 7 10.",
    "I bought 4k at bbc.co.uk.", "I bought v2 at bbc.co.uk.", "I bought 2x at amazon.com.",
    "I bought 3pk at Target.com.", "I bought iPhone15 at bestbuy.com.", "I bought 5pm at apple.com.",
    "I bought Model3 at github.io.",
    "I'll email you what I found at amazon.com.", "I'll email you the link I saw at Target.com.",
    "Email me the stuff from yesterday at bestbuy.com.", "email me back at 10.30 at cafe.com",
    "I bought it at bbc.co.uk. Is the email working now?",
    "We sell at bestbuy.com, that is our email list's source.",
    "Check the deals at github.io, that's the email I mentioned.",
    "the scorecard " + "x" * 54 + " 4111 1111 1111 1112",
    "4111 1111 1111 1112 " + "y" * 34 + " cardinals",
    "219 09 999 " + "y" * 32 + " socialite",
    "12,219 09 9999", "Room B219 09 9999",
    "**Them:** Social?\n\n**Me:** Hm.\n\n**Them:** Go on.\n\n**Me:** 1234567",
]


def test_measured_ordinary_speech_survives():
    for t in MEASURED_ORDINARY:
        assert _m(t).text == t, t


def test_only_the_first_run_after_a_cue_is_bound():
    m = _m("my visa is 4111 1111 1111 1111 and by the way order 12345678901234 shipped")
    assert m.text == "my visa is [CARD] and by the way order 12345678901234 shipped"


def test_bound_before_across_one_turn_and_filler():
    for t, want in (
        ("My social is 741 05 7024", "My social is [SSN]"),
        ("**Them:** What's your social?\n\n**Me:** It's 030809137", "**Me:** It's [SSN]"),
        ("Put it on my visa. 4111 7962 7393 8077", "visa. [CARD]"),
        ("**Them:** What's the card number?\n\n**Me:** Okay. 4242, 7844, 6476, 8778", "Okay. [CARD]"),
        ("My routing number is 035 690 281", "is [ACCOUNT]"),
        ("**Me:** What's the routing number?\n\n**Them:** Here it is. 035 690 281", "Here it is. [ACCOUNT]"),
        ("my social is 912 34 5678", "my social is [SSN]"),          # range fails: still masked
    ):
        assert _m(t).text.endswith(want), t


def test_bound_after():
    for t, want in (
        ("644 04 8901. That's my social.", "[SSN]. That's my social."),
        ("371450948481385 is the card", "[CARD] is the card"),
        ("80608689949, that's the account number", "[ACCOUNT], that's the account number"),
        ("4111 1111 1111 1112 and that's my credit card", "[CARD] and that's my credit card"),
    ):
        assert _m(t).text == want, t


def test_bound_shapes_refuse_dates():
    _same("my social is 2023-08-13")
    _same("my social is 3/12/2027")
    _same("account number 2020-09-19")


def test_standing_alone():
    for t in ("ssn 12,219 09 9999", "ssn B219 09 9999", "ssn 219 09 9999,5", "ssn 219 09 9999:30",
              "ref GB82-78345678", "card x4111 1111 1111 1111"):
        _same(t)
    assert _m("(219 09 9999)").text == "([SSN])"
    assert _m('"219-09-9999" ok').text == '"[SSN]" ok'


def test_spaced_324_shape_is_masked_without_cue_today():
    # measurement-gated (pii_detect.SSN_SPACED_CUE_FREE): the shape the transcriber gives a
    # spoken SSN is also an ordinary three-number list. Masked today; the trade-off is known.
    assert _m("we had 300 25 2000 users").text == "we had [SSN] users"


def test_iban_case_insensitive_and_retry_shorter():
    assert _m("iban gb82 west 1234 5698 7654 32 ok").text == "iban [ACCOUNT] ok"
    assert _m("It's BE68 5390 0754 7034 2024.").text == "It's [ACCOUNT] 2024."
    _same("It's GB00 WEST 1234 5698 7654 32.")             # mod-97 fails: no digits taken alone
    _same("IBAN GB00 WEST 1234 5698 7654 32.")


def test_email_name_digits_needs_a_lead():
    for t in ("**Them:** quarrel43 at example.org.", "It's quarrel43 at example.org.",
              "Use quarrel43 at example.org.", "email quarrel43 at example.org", "to: quarrel43 at example.org"):
        assert "[EMAIL]" in _m(t).text, t
    _same("I bought iPhone15 at bestbuy.com.")
    _same("Got the Model3 at example.com.")


def test_email_dotted_parts_need_letters():
    assert _m("Faxmore.ster at example.com").text == "[EMAIL]"
    for t in ("Get v2.1 at example.com", "I bought the U.S at example.com"):
        _same(t)


def test_email_cues_tightened():
    assert _m("The email is jane at example.com").text == "The email is [EMAIL]"
    assert _m("His email address: jane at example.com").text == "His email address: [EMAIL]"
    assert _m("Email: jane at example.com").text == "Email: [EMAIL]"
    _same("Email jane at example.com tomorrow")             # a verb
    _same("jane at example.com, that's the email")           # no "the"
    _same("jane at example.com, that's my email list")        # the phrase must end there
    _same("The email said to look at bestbuy.com.")           # no "is"/":": one word at most
    assert _m("my email jane at example.com").text == "my email [EMAIL]"


def test_password_contractions_rejected():
    for ans in ("It's", "It’s", "that's", "Don't"):
        t = "The password? It's fine. It’s done. That's that. Don't."
        m = pii.mask(t, classes=CLS, find_password=lambda w, a=ans: a)
        assert m.text == t and m.counts == {}, ans


def test_new_street_suffixes():
    for street in ("Trail", "Loop", "Ridge", "Crescent", "Square", "Plaza", "Alley", "Pike",
                   "Turnpike", "Expressway", "Row"):
        assert _m(f"I live at 12 Quorrel {street} now").text == "I live at [ADDRESS] now", street


def test_phone_bound_shapes():
    for t in ("my cell is 7245550311", "my cell is 1 724 555 0311", "my cell is 495, 555, 0133"):
        assert _m(t, PHONE).counts == {"phone": 1}, t
    for t in ("my cell is 72 45 55 03 11", "my number is 123456789", "my number is 123456789012"):
        assert _m(t, PHONE).counts == {}, t


def test_cue_under_a_preposition_directly_before_the_number():
    # the preposition test needs a word between cue and number: "on my account 80608689949" binds
    assert _m("on my account 80608689949").text == "on my account [ACCOUNT]"
    assert _m("on my account: 80608689949").text == "on my account: [ACCOUNT]"
    _same("The phone number on the account, 4155550134.")      # a phone is named before the cue


def test_bound_account_shapes():
    # "was" is not filler; bound accounts are contiguous, 2-5 groups of 3+
    # digits, or the 2-2-2 sort code
    for t in ("Routing was 675, 86, 415, 4, 643 packets per second.",
              "Routing was 8848, 4, 83 packets per second.",
              "Routing was 035 690 281",                  # accepted miss: no cue binds, none is free
              "account number 12 34 56 78", "account number 1234, 56, 7890"):
        _same(t)
    for t, want in (("My routing number is 035 690 281", "My routing number is [ACCOUNT]"),
                    ("routing is 035 690 281", "routing is [ACCOUNT]"),
                    ("sort code 12-34-56", "sort code [ACCOUNT]"),
                    ("sort code 12 34 56", "sort code [ACCOUNT]"),
                    ("account number 1234 5678", "account number [ACCOUNT]"),
                    ("80608689949, was the account number", "[ACCOUNT], was the account number")):
        assert _m(t).text == want, t


# a question in the previous turn binds the first number of the answer

def test_question_binds_the_answer():
    t = "**Them:** Which card do you want to use?\n\n**Me:** Okay. 4242, 7844, 6476, 8778."
    assert _m(t).text == "**Them:** Which card do you want to use?\n\n**Me:** Okay. [CARD]."
    t = "**Me:** What's your social?\n\n**Them:** Sure, it's 030809137."
    assert _m(t).text.endswith("**Them:** Sure, it's [SSN].")
    t = "**Them:** What's the routing number for that?\n\n**Me:** Here it is. 035 690 281."
    assert _m(t).text.endswith("Here it is. [ACCOUNT].")
    t = ("**Them:** Which card do you want to use?\n\n**Me:** Okay, three seven eight two, three three two "
         "four nine four, seven four seven five one.")
    assert _m(t).text.endswith("**Me:** Okay, [CARD].")


def test_question_binding_limits():
    for t in ("**Them:** Did you get the card?\n\n**Me:** Yes. 12 people came.",
              "**Them:** Which card do you want to use?\n\n**Me:** Hm.\n\n**Them:** Okay. 4242, 7844, 6476, 8778.",
              "**Them:** Which card do you want to use?\n\n**Me:** The one ending in 8778, order 12345678901234.",
              "**Them:** Which card do you want? Never mind. What time is it?\n\n**Me:** Okay. 4242, 7844, 6476, 8778.",
              "**Them:** Which card do you want to use.\n\n**Me:** Okay. 4242, 7844, 6476, 8778.",
              "**Them:** Which card do you want to use?\n\n**Me:** Okay. 12, 7844, 6476, 8778."):
        _same(t)


def test_question_with_several_cues_binds_the_class_the_value_fits():
    q = "**Them:** Card, account or social, which do you have?\n\n**Me:** Okay. "
    assert _m(q + "219099999.").counts == {"ssn": 1}
    assert _m(q + "412345678901.").counts == {"card": 1}      # 12 digits: card before account
    assert _m(q + "806086899491.").counts == {"account": 1}   # no brand prefix, not a card
    assert _m(q + "12345").counts == {}


# --- comma pieces, phones, dates, labels, questions ---------------------------------------------

def _q(question, answer):
    return f"**Them:** {question}\n\n**Me:** {answer}"


def _answer(t, classes=CLS):
    return _m(t, classes).text.split("\n")[-1]


def test_a_value_ends_at_the_comma():
    assert _m("My card is 4111 1111 1111 1111, 09 30 is the expiry.").text \
        == "My card is [CARD], 09 30 is the expiry."
    assert _m("Card number 4111 1111 1111 1111, 03/28.").text == "Card number [CARD], 03/28."
    assert _answer(_q("Card number and expiry?", "4111 1111 1111 1111, 02 28.")) == "**Me:** [CARD], 02 28."
    assert _m("My social is 309-75-1370, 81 years old.").text == "My social is [SSN], 81 years old."
    assert _m("It's 351-72-3375, 7 of us share it.").text == "It's [SSN], 7 of us share it."
    assert _answer(_q("Social and date of birth?", "317 41 4878, 04/06/1966.")) == "**Me:** [SSN], 04/06/1966."
    assert _m("My account number is 50266269, 16-89-87 is the sort code.").text \
        == "My account number is [ACCOUNT], [ACCOUNT] is the sort code."
    assert _m("My social is 741 05 7024, 45 Main Street.").text == "My social is [SSN], [ADDRESS]"
    assert _m("card number 4242, 7844, 6476, 8778").text == "card number [CARD]"


def test_phone_numbers_are_never_ssn_or_account():
    for t in (_q("What's the best number for the account?", "Okay, 864-937-8396."),
              _q("What's the phone number on your account?", "Sure. 547-382-3582."),
              _q("Do you have the number for Social Security?", "Yeah, 777-588-2562."),
              _q("Let me pull up your account.", "So 3618077883 is my cell."),
              "It's my account. 573 897 2549.", "My social is 4155550134.",
              "My account number is 1 415 555 0134."):
        _same(t)
    t = _q("What's the best phone number for the account?", "Sure, 4155550134.")
    assert _answer(t, PHONE) == "**Me:** Sure, [PHONE]."
    assert _answer(_q("What's a good number for you?", "Sure. 344 5550101."), PHONE) == "**Me:** Sure. [PHONE]."


def test_after_cue_must_end_its_clause():
    for t in ("888 655 8028 is the Social Security office.", "Call 358 870 9876, the account manager.",
              "We hit 36231709, the social media team said."):
        _same(t)
    for t, want in (("030809137 would be my social.", "[SSN] would be my social."),
                    ("030809137, which is my social.", "[SSN], which is my social."),
                    ("030809137 — that's my social.", "[SSN] — that's my social."),
                    ("030809137, it is my social.", "[SSN], it is my social."),
                    ("12345678 for the account.", "[ACCOUNT] for the account."),
                    ("4000123456789011, which is my card.", "[CARD], which is my card.")):
        assert _m(t).text == want, t


def test_bound_values_are_never_dates_or_zip4():
    for t in (_q("When did you get your social security card?", "So 04.17.2000."),
              _q("When did the social security letter come?", "2016/10/14."),
              _q("What's the zip on the account?", "Yeah, 13946-4891."),
              "My IBAN? Yeah, 39424 6236 is the zip.",
              "My social security is 8-27-2024, that is when it changed.",
              "My account number is 2016 10 14.", "My social is 10 14 2016.", "My social is 3.4.21",
              "My account is 1995 1996 1997.", "My card is 1995 1996 1997 1998 1999."):
        _same(t)


def test_a_number_labelled_as_something_else_is_left():
    for t in ("Your account. Okay so 98919781 is the ticket.", "My social. Well, 467032083 is the claim.",
              "Your Visa. Yes. 48915166171993 is the tracking.",
              "The card number. 51858923398493 is the tracking.",
              "My Amex, yeah, 2007 2008 2009 2010 were the years.",
              "My routing number is yeah, 821 688 dollars short.",
              _q("How much is in the account?", "So 2687213."),
              "My account number is 12345678 points.", "my card is 9999 8888 7777 6666"):
        _same(t)
    assert _m("Your account. 98919781 is my account number.").text == "Your account. [ACCOUNT] is my account number."


def test_a_question_must_ask_for_the_cue_thing():
    for t in (_q("What's the order number on the account?", "So 43562933."),
              _q("What's the confirmation for the card payment?", "4111111111111112"),
              _q("Which years did you use the card?", "Right, 1995 1996 1997."),
              _q("How many followers on social?", "Sure. 55803808.")):
        _same(t)
    assert _answer(_q("What's your social? I need it for the form.", "030809137")) == "**Me:** [SSN]"
    assert _answer(_q("Routing and account number?", "011000028 and 12345678")) \
        == "**Me:** [ACCOUNT] and [ACCOUNT]"
    assert _answer(_q("Can I get your social security number?", "Sure, 030809137.")) == "**Me:** Sure, [SSN]."


def test_cue_with_apostrophe_s():
    for t, want in (("My social's 741057024.", "My social's [SSN]."),
                    ("The card number's 4000 1234 5678 9011.", "The card number's [CARD]."),
                    ("My account number’s 12345678.", "My account number’s [ACCOUNT]."),
                    ("The routing number's 011000028.", "The routing number's [ACCOUNT].")):
        assert _m(t).text == want, t


def test_hash_opens_a_value():
    assert _m("My account number is #12345678.").text == "My account number is #[ACCOUNT]."


def test_filler_phrases_on_the_original_text():
    assert _m("My social is, um, let me see, 030809137.").text == "My social is, um, let me see, [SSN]."
    assert _m("My card number is, hold on, one moment, 4000 1234 5678 9011.").text \
        == "My card number is, hold on, one moment, [CARD]."
    _same("My social is, two weeks ago, 030809137.")


def test_ssn_last_four():
    assert _m("The last four of my social are 8901.").text == "The last four of my social are [SSN]."
    assert _m("last 4 digits of the ssn: 8901").text == "last 4 digits of the ssn: [SSN]"
    _same("The last four of my card are 8901.")
    _same("The last four of my social are 89012.")


def test_pathological_inputs_stay_fast():
    import time
    # 3 s: a linear pass takes a fraction of that; a quadratic one takes minutes
    for t in ("1" * 50_000, "a" * 50_000, "**Me:** 1234567\n\n" * 20_000, "account 1234567 " * 20_000,
              "**Them:** What's your social?\n\n**Me:** It's 1234567.\n\n" * 10_000,
              "**Them:** Okay, and your account number?\n\n**Me:** " + "12, " * 40_000,
              "12, " * 40_000 + "is my account number.", "my account number is " + "1 " * 40_000):
        t0 = time.perf_counter()
        pii.mask(t, classes=CLS)
        assert time.perf_counter() - t0 < 3.0, t[:30]


def test_a_spelled_iban_starts_with_its_country_letters():
    # "1995, 1996, …" after "IBAN" is a list of years, not an IBAN spelled out
    _same("My IBAN, so, 1995, 1996, 1997, 1998, 1999 were the years.")
    t = "My IBAN is G B 82, W E S T, 1234, 5698, 7654, 32, that one."
    assert _m(t).text == "My IBAN is [ACCOUNT], that one."


def test_best_or_good_asks_only_for_a_phone():
    # "best" and "good" ask for a phone only
    t = _q("What's the best phone number for the account?", "Sure, 4155550134.")
    assert _answer(t, PHONE) == "**Me:** Sure, [PHONE]."
    _same(t)                                                  # phone off: left, never [ACCOUNT]
    assert _answer(_q("Is this a good cell to reach you?", "Yes, 4155550134."), PHONE) == "**Me:** Yes, [PHONE]."
    _same(_q("What's the best account to use?", "Okay. 12345678."))


# --- complete values, prepositions, last four, tidied password answers -------------------------

def test_a_bound_value_stops_when_it_is_complete():
    for t, want in (
        ("My account number is 4952859575, 891 dollars was charged.",
         "My account number is [ACCOUNT], 891 dollars was charged."),
        ("The routing number is 011000028, 766 dollars went to the wrong place.",
         "The routing number is [ACCOUNT], 766 dollars went to the wrong place."),
        ("Account number 2363263423, 1992 is the year I opened it.",
         "Account number [ACCOUNT], 1992 is the year I opened it."),
        ("My account number is 12345678, 2024 was when I opened it.",
         "My account number is [ACCOUNT], 2024 was when I opened it."),
        ("My card is 4111 1111 1111 1112, 123", "My card is [CARD], 123"),
        ("My account number is 12345678, 300 people use it.", "My account number is [ACCOUNT], 300 people use it."),
        ("card number 4242, 7844, 6476, 8778", "card number [CARD]"),
        ("card number 3782, 332494, 74751", "card number [CARD]"),
        ("my social is 692, 87, 1253", "my social is [SSN]"),
        ("account number 1234, 5678", "account number [ACCOUNT]"),
        ("account number 123456, 7890", "account number [ACCOUNT]"),        # an account has no set length
        ("My card is 4111 1111 1111 1111, 123", "My card is [CARD], 123"),
        ("my social is 219099999, 12", "my social is [SSN], 12"),            # nine digits is a whole SSN
        ("My account number is 50266269, 16-89-87 is the sort code.",
         "My account number is [ACCOUNT], [ACCOUNT] is the sort code."),
        # too long for one account: the longest leading part that is one is taken
        ("account number 1234, 5678, 9012, 3456, 7890, 1234", "account number [ACCOUNT], 7890, 1234"),
    ):
        assert _m(t).text == want, t
    _same("My account number is 12345678 dollars.")


def test_a_cue_under_a_preposition_is_not_the_subject():
    for t in ("The phone number on the account is 4155550134.", "My cell on my account is 4155550134.",
              "Your claim number on the account is 20417733.", "The policy number for the account is 88213345.",
              "The tracking number they sent to my account is 1Z999AA10123456784.",
              _q("Can I get your employee ID for the account?", "It's 20417733."),
              "My claim account number is 20417733."):
        _same(t)
    assert _m("Put it on my visa, 4111 1111 1111 1112.").counts == {"card": 1}     # cards are exempt
    assert _answer(_q("What's the number on your card?", "5425 2334 3010 9904.")) == "**Me:** [CARD]."
    assert _m("My account number on file is 12345678, can you check it?").counts == {"account": 1}


def test_two_cue_nouns_of_different_classes_bind_in_order():
    t = _q("Can I get a callback number and your account number?", "4155550134 and 12345678.")
    assert _answer(t) == "**Me:** 4155550134 and [ACCOUNT]."
    assert _answer(t, PHONE) == "**Me:** [PHONE] and [ACCOUNT]."


def test_ssn_last_four_asked_as_a_question():
    assert _answer(_q("Can I get the last four of your social to verify?", "Yeah, it's 4471.")) \
        == "**Me:** Yeah, it's [SSN]."
    assert _answer(_q("Last four of your social?", "6789.")) == "**Me:** [SSN]."
    for t in (_q("Last four of your social?", "4471 dollars."), _q("Last four of your social?", "44712."),
              _q("Last four of your card?", "4471."), _q("Last four of your social?", "Hm.\n\n**Them:** 4471.")):
        _same(t)


def test_a_tidied_password_answer_is_located_in_the_window():
    t = "The wifi password is salvorn four six. Thanks."
    assert pii.mask(t, classes=CLS, find_password=lambda w: "salvorn46").text \
        == "The wifi password is [PASSWORD]. Thanks."
    t = "What's the password? It's f e n n i c k three one."
    for ans in ("fennick31", "FENNICK31"):
        m = pii.mask(t, classes=CLS, find_password=lambda w, a=ans: a)
        assert m.text == "What's the password? It's [PASSWORD].", ans
        assert m.secrets == ["f e n n i c k three one"]
    for ans in ("hunter2", "salvorn47", "It's"):
        assert pii.mask("The wifi password is salvorn four six.", classes=CLS,
                        find_password=lambda w, a=ans: a).counts == {}, ans
    t = "**Me:** the password is salvorn\n**Them:** four six."
    assert pii.mask(t, classes=CLS, find_password=lambda w: "salvorn46").counts == {}   # never across a turn


def test_account_with_one_short_edge_group():
    for t in ("the account number is 7731 2209 54.", "Okay so the account number is, um, let me look, 66 2049 1183.",
              "account number 66-2049-1183"):
        assert _m(t).counts == {"account": 1}, t
    for t in ("account number 12 2049 11", "account number 7731 22 2209", "Routing was 675, 86, 415, 4, 643 packets"):
        _same(t)


def test_only_a_dashed_zip4_is_vetoed():
    assert _m("My account number is 12345 6789.").text == "My account number is [ACCOUNT]."
    _same("My account number is 12345-6789.")


def test_money_phrases_are_units():
    for t in ("My IBAN is well, 935 891 in the red.", "My account number is 12345678 overdrawn.",
              "My account is 12345678 over the limit.", "My social is 12345678 in the black."):
        _same(t)


# Realistic two-speaker service-call exchanges (synthetic). `secrets` must be gone, `ordinary`
# must survive; a password secret is named by a stub that returns it when its window holds it.

def _call(*turns):
    return "\n\n".join(f"{'**Me:**' if who == 'M' else '**Them:**'} {say}" for who, say in turns)


_REAL_SECRET = [
 ("s01", "bank", _call(("T", "Can I get the last four of your social to verify?"), ("M", "Yeah, it's 4471.")), ["4471"], []),
 ("s02", "bank", _call(("T", "For verification, what's your social security number?"), ("M", "Sure, it's 512-44-8270.")), ["512-44-8270"], []),
 ("s03", "bank", _call(("T", "And your social?"), ("M", "Um, 512 44 8270.")), ["512 44 8270"], []),
 ("s04", "bank", _call(("M", "Hi, I want to check a deposit. My account number is 004417726315."), ("T", "Thanks, one moment.")), ["004417726315"], []),
 ("s05", "bank", _call(("T", "What's the account number?"), ("M", "Okay, let me find it. It's 3 3 0 1 9 8 4 4 2.")), ["3 3 0 1 9 8 4 4 2"], []),
 ("s06", "bank", _call(("T", "Can you read me the routing number and the account number?"), ("M", "Routing is 021000021, and the account is 483920175.")), ["021000021", "483920175"], []),
 ("s07", "bank", _call(("M", "I'm setting up direct deposit. The routing number is 121000358 and the account number is 7731 2209 54.")), ["121000358", "7731 2209 54"], []),
 ("s08", "online order", _call(("T", "Which card would you like to use?"), ("M", "The Visa. 4532 7153 3790 4122.")), ["4532 7153 3790 4122"], []),
 ("s09", "online order", _call(("T", "Go ahead with the card number when you're ready."), ("M", "4024 0071 5336 0887, expiration 08 27, and the code on the back is 319.")), ["4024 0071 5336 0887"], []),
 ("s10", "insurer", _call(("T", "How would you like to pay the premium?"), ("M", "Credit card. The number is 5425 2334 3010 9903.")), ["5425 2334 3010 9903"], []),
 ("s11", "insurer", _call(("M", "My card number is 3782 822463 10005, it's an Amex."),), ["3782 822463 10005"], []),
 ("s12", "IT help desk", _call(("T", "What's the temporary password I gave you?"), ("M", "It's Bluefin42.")), ["Bluefin42"], []),
 ("s13", "colleague login", _call(("M", "Can you send me the login for the analytics dashboard?"), ("T", "Sure, username is reports, password is Harbor2024!")), ["Harbor2024!"], []),
 ("s14", "colleague login", _call(("T", "The wifi password is greenteapot77."), ("M", "Got it, thanks.")), ["greenteapot77"], []),
 ("s15", "bank", _call(("M", "My social is 078-05-1120, and my date of birth is 03/14/1985.")), ["078-05-1120"], ["03/14/1985"]),
 ("s16", "doctor's office", _call(("T", "Can I have your social security number for the file?"), ("M", "Yeah. 219 09 9999.")), ["219 09 9999"], []),
 ("s17", "bank", _call(("T", "And what's the card number on the account?"), ("M", "5105 1051 0510 5100.")), ["5105 1051 0510 5100"], []),
 ("s18", "landlord", _call(("T", "For the rent, just send it to my account. The sort code is 20-32-06 and the account number is 73849201.")), ["20-32-06", "73849201"], []),
 ("s19", "bank", _call(("T", "Can you confirm your full account number?"), ("M", "Yes, so it's 9876543210.")), ["9876543210"], []),
 ("s20", "online order", _call(("M", "I'll pay with my debit card, 4111 1111 1111 1111, the expiry is 12 26.")), ["4111 1111 1111 1111"], ["12 26"]),
 ("s21", "IT help desk", _call(("T", "I've reset it. Your new password is Tango-Lima-88."), ("M", "Okay.")), ["Tango-Lima-88"], []),
 ("s22", "bank", _call(("T", "What's your SSN?"), ("M", "It's 123456789.")), ["123456789"], []),
 ("s23", "bank", _call(("M", "My account number is 12345678, 300 dollars came out of it yesterday that I didn't authorize.")), ["12345678"], ["300"]),
 ("s24", "insurer", _call(("T", "Can I get the card number?"), ("M", "Yeah, hold on. Okay, 4916 3385 0608 2832.")), ["4916 3385 0608 2832"], []),
 ("s25", "bank", _call(("T", "What's the IBAN?"), ("M", "GB29 NWBK 6016 1331 9268 19.")), ["GB29 NWBK 6016 1331 9268 19"], []),
 ("s26", "bank", _call(("M", "I need to update the direct deposit. New account, 556677889900.")), ["556677889900"], []),
 ("s27", "doctor's office", _call(("T", "Last four of your social?"), ("M", "6789.")), ["6789"], []),
 ("s28", "bank", _call(("M", "Okay so the account number is, um, let me look, 66 2049 1183.")), ["66 2049 1183"], []),
 ("s29", "online order", _call(("T", "And the security code?"), ("M", "It's 742.")), ["742"], []),
 ("s31", "online order", _call(("T", "Go ahead with the card number when you're ready."), ("M", "4024 0071 5336 0885.")), ["4024 0071 5336 0885"], []),
 ("s32", "bank", _call(("T", "What's the number on your card?"), ("M", "5425 2334 3010 9904.")), ["5425 2334 3010 9904"], []),
 ("s33", "bank", _call(("M", "My account number on file is 12345678, can you check it?")), ["12345678"], []),
 ("s34", "bank", _call(("T", "What's the account number for the savings?"), ("M", "It's 40021887.")), ["40021887"], []),
 ("s35", "bank", _call(("T", "Can I get your social, please?"), ("M", "Yes. It's 512 44 8270.")), ["512 44 8270"], []),
 ("s30", "bank", _call(("T", "Please read the card number."), ("M", "Sure, 4532015112830366, expiration 09/29.")), ["4532015112830366"], ["09/29"]),
]

# ---- exchanges with ORDINARY numbers only ----
_REAL_ORDINARY = [
 ("o01", "bank", _call(("T", "Can I get your date of birth for the account?"), ("M", "Sure, 04 12 1979.")), [], ["04 12 1979"]),
 ("o02", "bank", _call(("T", "What's the phone number on the account?"), ("M", "It's 4155550134.")), [], ["4155550134"]),
 ("o03", "bank", _call(("M", "The phone number on my account is 4155550134, but that's my old cell.")), [], ["4155550134"]),
 ("o04", "bank", _call(("T", "Your account balance is 2,450.17 as of today."), ("M", "Okay.")), [], ["2,450.17"]),
 ("o05", "bank", _call(("T", "Your account ending in 4471 was charged 129.99 on the 14th."), ("M", "Right.")), [], ["4471", "129.99"]),
 ("o06", "insurer", _call(("T", "Your claim number on the account is 2024118833."), ("M", "Got it.")), [], ["2024118833"]),
 ("o07", "insurer", _call(("T", "What's your policy number?"), ("M", "It's 44718293.")), [], ["44718293"]),
 ("o08", "insurer", _call(("T", "The policy number for the account is 88210043."), ("M", "Okay, thank you.")), [], ["88210043"]),
 ("o09", "doctor's office", _call(("T", "We have you for Tuesday at 2 30. Can I confirm your date of birth?"), ("M", "Yes, 11/02/1990.")), [], ["2 30", "11/02/1990"]),
 ("o10", "doctor's office", _call(("T", "What's the best number to reach you?"), ("M", "6175550192.")), [], ["6175550192"]),
 ("o11", "doctor's office", _call(("T", "Your copay is 35 dollars and the account is all set."), ("M", "Great.")), [], ["35"]),
 ("o12", "online order", _call(("T", "What's the order number?"), ("M", "It's 113-4827756-2290318.")), [], ["113-4827756-2290318"]),
 ("o13", "online order", _call(("T", "I see the card on file was charged 84.50. Your order number is 30041177."), ("M", "Okay.")), [], ["84.50", "30041177"]),
 ("o14", "online order", _call(("T", "What's the zip code on the card?"), ("M", "02139.")), [], ["02139"]),
 ("o15", "online order", _call(("T", "Can I get the billing zip for the card?"), ("M", "Yeah, 94110-2203.")), [], ["94110-2203"]),
 ("o16", "online order", _call(("M", "The tracking number they sent to my account is 1Z999AA10123456784.")), [], ["1Z999AA10123456784"]),
 ("o17", "IT help desk", _call(("T", "What's your ticket number?"), ("M", "It's 4471023.")), [], ["4471023"]),
 ("o18", "IT help desk", _call(("T", "Your account is locked. I'll call you back at extension 4417."), ("M", "Sure.")), [], ["4417"]),
 ("o19", "IT help desk", _call(("T", "Which account is it?"), ("M", "My work one, the one I've had since 2019.")), [], ["2019"]),
 ("o20", "landlord", _call(("T", "Rent is 1850 a month, due on the 1st. The account stays the same."), ("M", "Okay.")), [], ["1850"]),
 ("o21", "landlord", _call(("M", "I'm at 42 Elm Court, unit 3B. My account with the building is under my wife's name.")), [], []),
 ("o22", "bank", _call(("T", "Your social security deposit of 1,847 came in on 11/03."), ("M", "Good.")), [], ["1,847", "11/03"]),
 ("o23", "bank", _call(("T", "What's the best phone number for the account?"), ("M", "415 555 0134.")), [], ["415 555 0134"]),
 ("o24", "bank", _call(("M", "My cell on the account is 6175550192.")), [], ["6175550192"]),
 ("o25", "insurer", _call(("T", "What's the case number on the account?"), ("M", "It's 5512908.")), [], ["5512908"]),
 ("o26", "bank", _call(("T", "When did you open the account?"), ("M", "Back in 2014, March 2014.")), [], ["2014"]),
 ("o27", "bank", _call(("T", "How much was the card charge?"), ("M", "It was 1,240 dollars, on the 9th.")), [], ["1,240"]),
 ("o28", "doctor's office", _call(("T", "The office number is 6175550100, extension 22. Bring your insurance card."), ("M", "Okay.")), [], ["6175550100"]),
 ("o29", "bank", _call(("T", "Your account number? Actually, never mind, I have it. Your callback number is 6175550192?"), ("M", "Yes.")), [], ["6175550192"]),
 ("o30", "IT help desk", _call(("T", "Can I get your employee ID for the account?"), ("M", "It's 20417733.")), [], ["20417733"]),
 ("o31", "bank", _call(("T", "What's the reference number for the transfer to your account?"), ("M", "It's 77120045.")), [], ["77120045"]),
 ("o32", "bank", _call(("M", "My account was opened 03 2015, and the reference for the dispute is 99012345.")), [], ["99012345"]),
]

# Accepted misses: no question binds them and no shape is free.
_REAL_ACCEPTED = {
    "s05": "read digit by digit: nine one-digit groups are no account shape",
    "s31": "\"Go ahead with the card number\" is not a question, and the number fails Luhn",
}
_REAL_PASSWORDS = ["Bluefin42", "Harbor2024!", "greenteapot77", "Tango-Lima-88", "742"]


@pytest.mark.parametrize("cid,text,secrets,ordinary", [
    pytest.param(cid, text, secrets, ordinary, id=cid,
                 marks=[pytest.mark.xfail(strict=True, reason=_REAL_ACCEPTED[cid])] if cid in _REAL_ACCEPTED else [])
    for cid, _, text, secrets, ordinary in _REAL_SECRET + _REAL_ORDINARY
])
def test_realistic_exchange(cid, text, secrets, ordinary):
    stub = lambda w: next((p for p in _REAL_PASSWORDS if p in w), None)   # noqa: E731
    m = pii.mask(text, classes=CLS, find_password=stub)
    assert [s for s in secrets if s in m.text] == []
    assert [o for o in ordinary if o not in m.text] == []
    if not secrets:
        assert [h for h in pii.find(text, CLS) if h.cls != "address"] == []


def test_a_strong_ssn_cue_masks_a_garbled_number():
    t = "**Me:** The counts were 004\n\n**Them:** 265075.1253. That's my social."
    assert _m(t).text == "**Me:** The counts were 004\n\n**Them:** [SSN]. That's my social."
    assert _m("My social is 68581399.").text == "My social is [SSN]."
    assert _m("My social is 6858.13990.").text == "My social is [SSN]."
    assert _m("My social's um 4471-2.").text == "My social's um [SSN]."
    _same("My social is 4155550134.")              # a phone number, even next to a strong cue
    for t in ("Room 123. That's my social.", "Hello. That's my social.", "1,250.50. That's my social security check.",
              "415-555-0134. That's my social.", "2024-03-15. That's my social.", "12345-6789. That's my social.",
              "My social is 68581399 dollars.", "1234567890123. That's my social."):
        _same(t)


def test_a_glued_email_with_an_identifying_cue():
    t = "**Them:** Faxmore.steranatexample.com, that's my email."
    assert _m(t).text == "**Them:** [EMAIL], that's my email."
    t = "**Them:** Vandrel.corvainat example.org, that's my email."
    assert _m(t).text == "**Them:** [EMAIL], that's my email."
    assert _m("My email is Vandrel.corvainatexample.org.").text == "My email is [EMAIL]."
    _same("I got it from bestbuy.com, that's my email client's fault.")
    _same("Vandral.corvainatexample.org.")                          # no cue: never an email
    # the speaker says this website is their email: masked, by the rule
    assert _m("Check amazon.com, that's my email.").text == "Check [EMAIL], that's my email."


# --- a guard reads the cue's own clause; money is not a social; listed questions ----------------

def test_an_earlier_thing_in_the_sentence_does_not_block_the_cue():
    for t, want in (
        ("My zip code is 10001, and the account number is 48213377.", "account"),
        ("The order number is 5512334, and my routing number is 011000028.", "account"),
        ("My balance is 250 dollars, and the SSN is 219-09-9999.", "ssn"),
        ("My balance is 250 dollars, and my social is 219 09 9999.", "ssn"),
        ("My cell number is 4155550134 and my account number is 48213377.", "account"),
        ("My mobile account number is 48213377.", "account"),
        ("My phone account number is 48213377.", "account"),
        ("I'm calling about my phone bill, my account number is 48213377.", "account"),
        ("I'm calling about my phone bill, the SSN is 219099999.", "ssn"),
        ("My claim number is 7437291, and the routing number is 011000028.", "account"),
    ):
        assert want in _m(t).counts, t
    assert "48213377" not in _m("My cell number is 4155550134 and my account number is 48213377.").text


def test_a_phone_blocks_only_what_it_governs():
    for t in ("The phone number on the account is 4155550134.", "My cell on my account is 4155550134.",
              "Your claim number on the account is 20417733.", "The policy number for the account is 88213345.",
              "The tracking number they sent to my account is 1Z999AA10123456784.",
              _q("Can I get your employee ID for the account?", "It's 20417733."),
              "The phone number on the account, 4155550134.", "A cell for my account: 4155550134."):
        _same(t)


def test_money_is_not_a_social():
    for t in ("My social security is 1,840 a month.", "1,840. That's my social security.",
              "Her social security's 1840.50.", "My social security is 1840 per month.",
              "His social security is 2,310.75 now.", "My social security is 4471."):
        _same(t)
    assert _m("**Me:** The counts were 004\n\n**Them:** 265075.1253. That's my social.").counts == {"ssn": 1}
    assert _m("My social is 68581399.").counts == {"ssn": 1}
    assert _m("My social security is 6858139.").counts == {"ssn": 1}     # seven digits: a garbled SSN


def test_a_listed_question_binds_in_order():
    assert _answer(_q("Can I get your zip code and your account number?", "10001 and 48213377.")) \
        == "**Me:** 10001 and [ACCOUNT]."
    assert _answer(_q("Date of birth and the last four of your social?", "1/2/1980 and 4471.")) \
        == "**Me:** 1/2/1980 and [SSN]."
    assert _answer(_q("Routing and account number?", "011000028 and 12345678")) == "**Me:** [ACCOUNT] and [ACCOUNT]"
    assert _answer(_q("Your zip code, date of birth and account number?", "10001 and 1/2/1980 and 48213377.")) \
        == "**Me:** 10001 and 1/2/1980 and [ACCOUNT]."
    # counts differ: the first cue thing binds the first number
    assert _answer(_q("Can I get your zip code and your account number?", "48213377.")) == "**Me:** [ACCOUNT]."


def test_a_located_password_span_passes_the_answer_filter():
    t = "The password? It is fine. Never mind."
    assert pii.mask(t, classes=CLS, find_password=lambda w: "itis").counts == {}


# --- punctuation and markdown beside a value; lists need named things; overlaps --------------

def test_a_value_followed_by_closers_or_wrapped_in_markdown_is_masked():
    for t, want in (
        ("My social is 123-45-6789...", "My social is [SSN]..."),
        ("My social is 123-45-6789..", "My social is [SSN].."),
        ("My social is 123-45-6789…", "My social is [SSN]…"),
        ("Here 123-45-6789.. ok", "Here [SSN].. ok"),
        ('She said "my social is 123-45-6789."', 'She said "my social is [SSN]."'),
        ("My social (123-45-6789), right", "My social ([SSN]), right"),
        ("My social is 123-45-6789—and that's it", "My social is [SSN]—and that's it"),
        ("My card is 4111 1111 1111 1111...", "My card is [CARD]..."),
        ("- Bob gave his SSN (123-45-6789).", "- Bob gave his SSN ([SSN])."),
        ("- SSN: **123-45-6789**", "- SSN: **[SSN]**"),
        ("- Card number `4111 1111 1111 1111`", "- Card number `[CARD]`"),
        ("- SSN: _123-45-6789_ and ~123-45-6789~", "- SSN: _[SSN]_ and ~[SSN]~"),
        ("<123-45-6789>", "<[SSN]>"),
        ("“123-45-6789”", "“[SSN]”"),
        ("**Me:** 123-45-6789.**Them:** thanks", "**Me:** [SSN].**Them:** thanks"),
        ("My account number is (12345678).", "My account number is ([ACCOUNT])."),
    ):
        assert _m(t).text == want, t


def test_a_value_glued_to_a_letter_or_digit_is_not_alone():
    for t in ("B219 09 9999", "12,219 09 9999", "1,250.50", "v2.1", "123-45-6789.5", "123-45-6789,5",
              "123-45-6789-5", "123-45-6789/5", "123-45-6789:5", "123-45-6789..5", "123-45-6789.x",
              "123-45-6789)x", "123-45-6789—5"):
        _same(t)


def test_a_lead_in_is_not_a_listed_thing():
    for q, a, want in (
        ("Okay, great, and your social?", "219, 09, 9999.", "**Me:** [SSN]."),
        ("Okay, so, your social security number?", "163, 39, 5546.", "**Me:** [SSN]."),
        ("Okay, and your account number?", "4821, 3377.", "**Me:** [ACCOUNT]."),
        ("Okay, and your social?", "219099999, 10001.", "**Me:** [SSN], 10001."),
        ("Great, and your routing number?", "011, 000028", "**Me:** [ACCOUNT]"),
        ("Okay, perfect, and the account number?", "1234, 5678, 9012.", "**Me:** [ACCOUNT]."),
        ("How much was the payment, and what's the account number?", "250 and 48213377.",
         "**Me:** 250 and [ACCOUNT]."),
    ):
        assert _answer(_q(q, a)) == want, (q, a)


def test_a_comma_read_value_is_one_number_of_a_list():
    assert _answer(_q("Can I get the routing number and your zip?", "Sure, 3267, 333943.")) \
        == "**Me:** Sure, [ACCOUNT]."
    assert _answer(_q("Can I get your social security number and your zip?", "6379, 65709.")) == "**Me:** [SSN]."
    # "and" (or "and then") and a sentence end separate the numbers of a listed answer
    assert _answer(_q("Can I get your zip code and your account number?", "10001, and then 48213377.")) \
        == "**Me:** 10001, and then [ACCOUNT]."
    assert _answer(_q("Can I get your zip code and your account number?", "10001. 48213377.")) \
        == "**Me:** 10001. [ACCOUNT]."


def test_a_question_reads_its_whole_sentence_for_another_thing():
    for t in (_q("Can I get the claim number, and then we'll pull up the account?", "7437291."),
              _q("What's the order number and which account was it charged to?", "5512334."),
              _q("Can I get the order number and the account number?", "5512334.")):
        _same(t)
    # same-turn statements still read only the cue's own clause
    assert _m("The order number is 5512334, and my routing number is 011000028.").counts == {"account": 1}
    # a zip, a date or an amount has its own shape: in a question it blocks only in the cue's clause
    for q, a in (("Can I get your zip and your account number?", "Sure, 3744644286."),
                 ("Your zip code and your social?", "Sure, 3299 23294."),
                 ("Your date of birth and your social security number?", "664345765.")):
        assert "[" in _answer(_q(q, a)), q
    _same(_q("What's the zip code on your account?", "02139."))


def test_a_thousands_comma_number_after_a_strong_cue_is_an_ssn():
    assert _m("My social is 219,099,999.").text == "My social is [SSN]."
    assert _m("219,099,999. That's my social.").text == "[SSN]. That's my social."
    for t in ("My social security is 1,840 a month.", "My social is 1,840,000 dollars.",
              "His social security is 2,310.75 now.", "My social is 1,234,567.50."):
        _same(t)


def test_overlapping_hits_mask_their_union_as_the_higher_class():
    m = _m("I live at 219 09 9999 Main Street.")
    assert m.text == "I live at [SSN]" and m.counts == {"ssn": 1}
    assert m.secrets == ["219 09 9999 Main Street."]
    hits = pii.find("I live at 219 09 9999 Main Street.", CLS)
    assert [(h.cls, h.start) for h in hits] == [("ssn", 10)]


def test_an_over_long_bound_value_masks_its_longest_fitting_prefix():
    assert _m("My account number is 4111 1111 1111 1111 2222.").text == "My account number is [ACCOUNT] 2222."
    assert _m("card 4111 1111 1111 1111 9999 Elm Street").text == "card [CARD] [ADDRESS]"
    assert _m("ssn 219 09 9999 12").text == "ssn [SSN] 12"
    # a phone number, a date or a value within its limit is never cut down to fit
    for t in ("It's my account. 573 897 2549.", "My social is 1 415 555 0134.", "My account number is 2016 10 14.",
              "account number 7731 22 2209"):
        _same(t)


def test_the_password_pass_stops_after_the_first_failed_call():
    calls = []

    def boom(w):
        calls.append(w)
        raise TimeoutError
    t = "My PIN is 1234. Ok. Ok. Ok. The passcode is 99. Ok. Ok. Ok. Gate code 7. Ok. Ok. Ok. Door code 8."
    m = pii.mask(t, classes=CLS, find_password=boom)
    assert len(calls) == 1 and m.partial


def test_the_password_pass_keeps_what_it_found_before_a_failure():
    answers = iter(["hunter2"])

    def finder(w):
        return next(answers)                          # StopIteration on the second call
    t = "The password is hunter2. Ok. Ok. Ok. The PIN is 4471."
    m = pii.mask(t, classes=CLS, find_password=finder)
    assert m.partial and m.counts == {"password": 1} and "hunter2" not in m.text


def test_should_stop_is_asked_between_password_windows():
    calls, checks = [], []

    class Stop(Exception):
        pass

    def should_stop():
        checks.append(1)
        if len(calls) >= 1:
            raise Stop
    t = "The password is hunter2. Ok. Ok. Ok. The PIN is 4471."
    with pytest.raises(Stop):
        pii.mask(t, classes=CLS, find_password=lambda w: calls.append(w), should_stop=should_stop)
    assert len(calls) == 1
    # no cue: never asked
    pii.mask("nothing here", classes=CLS, find_password=lambda w: None, should_stop=lambda: checks.append("x"))
    assert "x" not in checks


def test_spans_match_whole_tokens_only():
    toks = [(" the", 0, 1), (" tiger", 1, 2), (" tigers", 2, 3)]
    assert pii.spans(toks, ["tiger"]) == ([(1, 2)], 0)
    assert pii.spans([(" tigers", 0, 1)], ["tiger"]) == ([], 1)
    assert pii.spans([(" room", 0, 1), (" 144712", 1, 2)], ["4471"]) == ([], 1)
    assert pii.spans([(" 44", 0, 1), ("71", 1, 2), ("2", 2, 3)], ["4471"]) == ([], 1)
    assert pii.spans([(" pin", 0, 1), (" 4471.", 1, 2)], ["4471"]) == ([(1, 2)], 0)


def test_no_dead_surface():
    assert not hasattr(pii, "DEFAULT_CLASSES") and not hasattr(pii, "_aba_ok")
