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
