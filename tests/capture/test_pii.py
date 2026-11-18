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
    m = pii.mask("my social is 219-09-9999 ok", classes=pii.DEFAULT_CLASSES)
    assert m.text == "my social is [SSN] ok" and m.counts == {"ssn": 1}
    assert m.secrets == ["219-09-9999"]


def test_spoken_ssn_with_cue():
    t = "**Them:** social security number two one nine, oh nine, nine nine nine nine."
    m = pii.mask(t, classes=pii.DEFAULT_CLASSES)
    assert "[SSN]" in m.text and "nine nine" not in m.text and m.text.startswith("**Them:**")


def test_invalid_ssn_areas_not_masked():
    for v in ("000-12-3456", "666-12-3456", "912-12-3456"):
        assert pii.mask(f"ssn {v}", classes=pii.DEFAULT_CLASSES).counts == {}


def test_run_together_nine_digits_needs_cue():
    assert pii.mask("order 219099999 shipped", classes=pii.DEFAULT_CLASSES).counts == {}
    assert pii.mask("my SSN is 219099999", classes=pii.DEFAULT_CLASSES).counts == {"ssn": 1}


def test_card_needs_luhn():
    # R3b: with a card cue in reach a Luhn-failing run is still a card (a misheard digit), so
    # the Luhn guard is tested without one.
    assert pii.mask("ref 4111 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    assert pii.mask("ref 4111 1111 1111 1112", classes=pii.DEFAULT_CLASSES).counts == {}


def test_spoken_card_is_covered_whole():
    t = "card four one one one one one one one one one one one one one one one then five"
    m = pii.mask(t, classes=pii.DEFAULT_CLASSES)
    assert m.text == "card [CARD] then five"


def test_hit_ending_inside_a_spoken_group_is_widened():
    # the SSN pattern ends inside "nine nine nine nine nine": the whole phrase goes
    t = "ssn two one nine oh nine nine nine nine nine"
    m = pii.mask(t, classes=pii.DEFAULT_CLASSES)
    assert m.text == "ssn [SSN]"


def test_iban_and_routing_account():
    assert pii.mask("IBAN GB82 WEST 1234 5698 7654 32", classes=pii.DEFAULT_CLASSES).counts == {"account": 1}
    m = pii.mask("routing number 011000028 account number 12345678", classes=pii.DEFAULT_CLASSES)
    assert m.counts == {"account": 2}


def test_email_written_and_spoken():
    assert pii.mask("write to a.b@example.com", classes=pii.DEFAULT_CLASSES).text == "write to [EMAIL]"
    m = pii.mask("it's jane dot doe at example dot com", classes=pii.DEFAULT_CLASSES)
    assert m.text == "it's [EMAIL]" and m.secrets == []


def test_address():
    m = pii.mask("I live at two twenty one Baker Street, apartment 4, 90210 now",
                 classes=pii.DEFAULT_CLASSES)
    assert m.text == "I live at [ADDRESS] now"


def test_street_without_number_kept():
    assert pii.mask("she lives on Baker Street", classes=pii.DEFAULT_CLASSES).counts == {}


def test_phone_off_by_default_on_when_listed():
    t = "call 415 555 0134"
    assert pii.mask(t, classes=pii.DEFAULT_CLASSES).counts == {}
    assert pii.mask(t, classes=(*pii.DEFAULT_CLASSES, "phone")).text == "call [PHONE]"


def test_ordinary_numbers_survive():
    for t in ("meet at two thirty", "it was four ninety nine", "back in 2019",
              "order number 5512 3349", "call nine one one", "page 123 of 456"):
        assert pii.mask(t, classes=pii.DEFAULT_CLASSES).text == t


def test_untouched_text_byte_identical():
    t = "**Me:**  odd  spacing\n\n**Them:** a.b@example.com  end "
    assert pii.mask(t, classes=pii.DEFAULT_CLASSES).text == "**Me:**  odd  spacing\n\n**Them:** [EMAIL]  end "


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
    assert pii.mask(t, classes=pii.DEFAULT_CLASSES).text == t


def test_number_lists_survive():
    for t in ("we got 606, 968, 105, 924, 326, 32 votes", "prices 40 35 69 67 89 66 61 11 16",
              "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen",
              "we had 300, 25, 2000 users", "rooms 101, 22, 3045"):
        _same(t)


def test_spaced_ssn_needs_cue():
    assert pii.mask("my social 219 09 9999 ok", classes=pii.DEFAULT_CLASSES).counts == {"ssn": 1}
    assert pii.mask("ssn 219, 09, 9999", classes=pii.DEFAULT_CLASSES).counts == {"ssn": 1}
    _same("rooms 219, 09, 9999")


def test_card_shape_or_cue():
    assert pii.mask("4111 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    assert pii.mask("my visa is 41 11 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    # R3a: Luhn plus a brand is a card in any space grouping; with no brand it needs a cue
    _same("ref 91 60 1562 2746 67")


def test_spoken_email_needs_dot_or_cue():
    for t in ("I looked at amazon dot com yesterday", "I met her at Starbucks dot com",
              "check it out at google dot com"):
        _same(t)
    # R5b: "me" never forms a local part on its own
    _same("email me at jane dot com")
    m = pii.mask("my email is jane at example dot com", classes=pii.DEFAULT_CLASSES)
    assert m.text == "my email is [EMAIL]"


def test_spaced_account():
    m = pii.mask("account number 1234 5678", classes=pii.DEFAULT_CLASSES)
    assert m.text == "account number [ACCOUNT]"


CLS = pii.DEFAULT_CLASSES


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


def test_answer_not_verbatim_in_window_is_ignored():
    t = "the password is hunter2 ok."
    assert pii.mask(t, classes=CLS, find_password=lambda w: "Hunter2").text == t


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


# --- Task 7b: what the transcriber writes (synthetic values; Parakeet's rendering first, the
# spoken-word rendering second) ---------------------------------------------------------------

def _m(t, classes=CLS):
    return pii.mask(t, classes=classes)


def _luhn_complete(body: str) -> str:
    """body plus the Luhn check digit."""
    return next(body + d for d in "0123456789" if pii._luhn(body + d))


def _aba_complete(body: str) -> str:
    """eight digits plus the ABA check digit."""
    return next(body + d for d in "0123456789" if pii._aba_ok(body + d))


# normalise (R7)

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


# SSN (R1, R2)

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
    for t in ("rooms 12 219 09 9999", "rooms 219 09 9999 12", "rooms 12, 219 09 9999",
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
    for t in ("my social club has 123456 members", "my social is 123456789012 long"):
        _same(t)


def test_ssn_cue_reach():
    _same("social " + "x" * 60 + " 219, 09, 9999")
    _same("**Them:** 219, 09, 9999.\n\n**Me:** That's my social.")   # after: never across a turn
    _same("219, 09, 9999 and then " + "x" * 30 + " my social")         # after: beyond 40
    assert _m("social " + "x" * 50 + " 219, 09, 9999").counts == {"ssn": 1}


# card (R3)

def test_card_as_parakeet_writes_it():
    assert _m("**Them:** it on my visa. 4111 7962 7393 8077.").counts == {"card": 1}
    assert _m("**Me:** Okay. 4242, 7844, 6476, 8778.").text == "**Me:** Okay. [CARD]."
    assert _m("**Them:** 371450948481385 is the card.").text == "**Them:** [CARD] is the card."
    assert _m("**Me:** Okay, 3782 332494 74751.").text == "**Me:** Okay, [CARD]."
    assert _m("**Me:** Here you go. 378247629514895.").counts == {"card": 1}


def test_card_as_words():
    for t in ("Put it on my visa, forty one eleven dash seventy nine sixty two dash seventy three "
              "ninety three dash eighty seventy seven.",
              "Here you go, four two four two dash two four eight one dash zero four one seven dash "
              "six five eight six.",
              "Okay, forty two forty two, seventy eight forty four, sixty four seventy six, eighty "
              "seven seventy eight.",
              "Okay, three seven eight two, three three two four nine four, seven four seven five one."):
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
    assert _m(f"my card is {nobrand}").counts == {"card": 1}
    assert _m("card 4111 1111 1111 1112").counts == {"card": 1}             # Luhn fails
    assert _m("4111 1111 1111 1112 is the card").counts == {"card": 1}      # cue after
    assert _m("my visa, 41, 11, 1111, 1111, 1111").counts == {"card": 1}   # any `, ` grouping
    _same("my card " + "x" * 60 + " 4111 1111 1111 1112")
    _same("card 4111 1111 111")                                             # 11 digits


def test_comma_cards_only_in_card_shapes():
    _same("ref 42, 42, 7844, 6476, 8778")
    _same("ref 4242, 7844, 6476, 877, 8")
    assert _m("ref 4242, 7844, 6476, 8778").counts == {"card": 1}


def test_bank_strings_and_timestamps_are_not_cards():
    for t in ("ref 91601562274667", "at 20221015123456", "id 5560099292349456 0"):
        _same(t)


# account (R4)

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
    assert _m("acct no. 1234-5678-90").counts == {"account": 1}
    _same("account number " + "x" * 60 + " 865047")
    _same("account number is 12345")                                      # under 6 digits


def test_cued_alphanumeric_account():
    assert _m("Account No. E11248142150, Date").text == "Account No. [ACCOUNT], Date"
    assert _m("Account number: ACCT-911155810.").text == "Account number: [ACCOUNT]."
    assert _m("the BBAN: RBKJ59644698401273.").text == "the BBAN: [ACCOUNT]."
    _same("Your policy number is GPTZ91601562274667.")                     # no cue
    _same("account number is ABCDEFGH12345")                               # 5 digits


def test_routing_without_cue_needs_checksum_and_fed_prefix():
    assert _m("**Them:** Here it is. 035 690 281.").text == "**Them:** Here it is. [ACCOUNT]."
    assert _m("**Them:** Here it is. 097 684 811.").counts == {"account": 1}
    assert _m("it's 011000028").counts == {"account": 1}
    for p in ("00", "12", "21", "32", "61", "72", "80"):
        r = _aba_complete(p + "123456")
        assert _m(f"it's {r}").counts == {"account": 1}, p
    for p in ("13", "20", "33", "60", "73", "79", "81", "99"):
        _same(f"it's {_aba_complete(p + '123456')}")
    for t in ("it's 035 690 282", "it's 035 690 281 5", "it's 035, 690, 281", "it's 0356 90281"):
        _same(t)
    assert _m("routing number is 035, 690, 281").counts == {"account": 1}


# email (R5)

def test_email_as_parakeet_writes_it():
    assert _m("**Them:** email is vanholt at example.org.").text == "**Them:** email is [EMAIL]."
    for t in ("quarrel43 at example.org.", "Quirl94 at example.org.", "quarrel.telvane at example.org.",
              "Zorveth.steran at example.com.", "dev_42 at example.dev."):
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
    assert _m("jane at example.com is my email").text == "[EMAIL] is my email"
    _same("jane at example.com, email me")                                  # email as a verb
    _same("jane at example.com and " + "x" * 40 + " that's my email")       # beyond 40


# password windows (R6)

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


# phone (R8)

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
    assert _m("the number is 495, 555, 0133", PHONE).text == "the number is [PHONE]"


# priorities (R9)

def test_exact_tie_goes_to_the_higher_class():
    r = _aba_complete("12345678")                      # a valid routing number and SSN shape
    assert pii.mask(f"my ssn is {r}", classes=CLS).counts == {"ssn": 1}
    assert pii.mask(f"it's {r}", classes=CLS).counts == {"account": 1}
    assert pii.mask("account number 4111111111111111", classes=CLS).counts == {"card": 1}


# ordinary speech, as Parakeet wrote it and as words (R11)

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
