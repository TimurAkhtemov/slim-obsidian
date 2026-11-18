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
    assert pii.mask("card 4111 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    assert pii.mask("card 4111 1111 1111 1112", classes=pii.DEFAULT_CLASSES).counts == {}


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
    m = pii.Masked(text="", counts={"ssn": 1, "email": 2}, secrets=[], partial=True)
    assert pii.frontmatter(m) == {"pii_masked": ["email:2", "ssn:1"], "pii_by": "pii-1",
                                  "pii_partial": "true"}


def _same(t):
    assert pii.mask(t, classes=pii.DEFAULT_CLASSES).text == t


def test_number_lists_survive():
    for t in ("we got 606, 968, 105, 924, 326, 32 votes", "prices 40 35 69 67 89 66 61 11 16",
              "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen",
              "we had 300, 25, 2000 users", "rooms 101 22 3045"):
        _same(t)


def test_spaced_ssn_needs_cue():
    assert pii.mask("my social 219 09 9999 ok", classes=pii.DEFAULT_CLASSES).counts == {"ssn": 1}
    assert pii.mask("ssn 219, 09, 9999", classes=pii.DEFAULT_CLASSES).counts == {"ssn": 1}
    _same("rooms 219 09 9999")


def test_card_shape_or_cue():
    assert pii.mask("4111 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    assert pii.mask("my visa is 41 11 1111 1111 1111", classes=pii.DEFAULT_CLASSES).counts == {"card": 1}
    _same("ref 41 11 1111 1111 1111")


def test_spoken_email_needs_dot_or_cue():
    for t in ("I looked at amazon dot com yesterday", "I met her at Starbucks dot com",
              "check it out at google dot com"):
        _same(t)
    m = pii.mask("email me at jane dot com", classes=pii.DEFAULT_CLASSES)
    assert m.counts == {"email": 1}


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


def test_window_does_not_cross_turn_start():
    seen = []
    pii.mask("**Me:** the password is\n**Them:** secret9 right.", classes=CLS,
             find_password=lambda w: seen.append(w))
    assert seen and all("**Them:**" not in w for w in seen)


def test_ask_model_sends_bounded_schema_call(monkeypatch):
    got = {}

    def fake(messages, schema, **kw):
        got.update(messages=messages, schema=schema, kw=kw)
        return {"secret": "hunter2"}, {}
    monkeypatch.setattr(pii.llm, "chat_json", fake)
    assert pii.ask_model("the password is hunter2") == "hunter2"
    assert got["kw"]["think"] is False and got["kw"]["num_predict"] == 64
    assert got["schema"]["properties"]["secret"]["maxLength"] == 64


def test_short_answers_never_touch_other_words():
    t = "the password is a big deal."
    for ans in ("a", "is"):
        m = pii.mask(t, classes=CLS, find_password=lambda w, a=ans: a)
        assert m.text == t.replace(f" {ans} ", " [PASSWORD] ", 1)


def test_answer_before_the_cue_is_not_masked():
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
