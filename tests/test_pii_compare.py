"""The comparison harness: its scoring, the shapes-only arm, and how a classifier's answer is masked."""

from scripts.pii_compare import arms, score
from slim.capture import pii
from slim.capture.pii_detect import Hit

T = "**Them:** Your zip and account number?\n\n**Me:** 10001, 4821, 3377."


def test_scoring_on_a_hand_made_example():
    case = {"id": "x", "text": T, "secrets": [["4821, 3377", "account"]], "ordinary": ["10001"]}
    a = T.index("4821")
    full = [Hit(a, a + len("4821, 3377"), "account")]
    part = [Hit(a, a + 4, "account")]
    wide = [Hit(T.index("10001"), a + len("4821, 3377"), "account")]
    r = score.score_labelled([case], {"x": full})
    assert r["secrets"] == {"masked": 1, "n": 1} and r["ordinary"].get("damaged", 0) == 0
    r = score.score_labelled([case], {"x": part})            # a partial mask is a miss
    assert r["secrets"] == {"masked": 0, "n": 1} and r["leaks"][0]["value"] == "4821, 3377"
    r = score.score_labelled([case], {"x": wide})            # over-wide: masked, and the zip damaged
    assert r["secrets"]["masked"] == 1 and r["ordinary"]["damaged"] == 1
    r = score.score_labelled([{**case, "ambiguous": True}], {"x": []})
    assert r["ambiguous"] == 1 and r["secrets"]["n"] == 0


def test_ordinary_damage_counts_only_secret_class_masks():
    t = "**Them:** Call 415 555 0134 or ship to 10001."
    case = {"id": "o", "family": "f", "text": t}
    phone = [Hit(t.index("415"), t.index("415") + 12, "phone")]
    acct = [Hit(t.index("10001"), t.index("10001") + 5, "account")]
    assert score.score_ordinary([case], {"o": phone})["values"] == {"n": 2, "damaged": 0}
    assert score.score_ordinary([case], {"o": acct})["sentences"] == {"n": 1, "damaged": 1}


SHAPES = [
    ("My number 123-45-6789 here", "ssn"),
    ("then 741 05 7024 ok", "ssn"),
    ("use 4111 1111 1111 1111 please", "card"),
    ("IBAN GB82 WEST 1234 5698 7654 32 thanks", "account"),
    ("write jo.smith@example.com today", "email"),
    ("I live at 42 Elm Street now", "address"),
]
BOUND = [
    "My account number is 48213377.",
    "**Them:** What's your social?\n\n**Me:** 219099999.",
    "The last four of my social is 4471.",
    "**Them:** Your zip and account number?\n\n**Me:** 10001, 48213377.",
    "644048901, that's my social.",
    "My email is vanholt at example.org.",
]


def test_shapes_arm_keeps_every_shape_and_binds_nothing():
    for text, cls in SHAPES:
        assert {h.cls for h in arms.shapes(text)} == {cls}, text
    for text in BOUND:
        assert pii.find(text, arms.CLASSES), text            # the rules bind it
        assert arms.shapes(text) == [], text                  # the shapes alone do not
    assert pii.find(BOUND[0], arms.CLASSES)                   # and the patch is undone


def test_classifier_answers_are_merged_and_checked_by_digit_count():
    cands = arms.candidates(T, [])
    assert [len(c.segments) for c in cands] == [3]            # "10001, 4821, 3377": one run
    hits = arms.merge_labels(cands[0], ["none", "account", "account"])
    assert [(T[h.start:h.end], h.cls) for h in hits] == [("4821, 3377", "account")]
    four = arms.merge_labels(cands[0], ["none", "ssn", "account"])     # four digits can be a last four
    assert [(T[h.start:h.end], h.cls) for h in four] == [("4821", "ssn")]   # "3377" alone is no account
    ssn = arms.merge_labels(cands[0], ["ssn", "ssn", "ssn"])   # 13 digits is no SSN: its leading 9 are
    assert [T[h.start:h.end] for h in ssn] == ["10001, 4821", "3377"]   # then the rest, a last four
    assert arms.merge_labels(cands[0], ["phone", "phone", "phone"]) == []


def test_a_hit_inside_a_segment_leaves_its_other_groups_standing():
    t = "**Me:** card 4111 1111 1111 1111 9999 Elm Street"
    hits = [h for h in arms.shapes(t) if h.cls == "address"]
    assert hits and t[hits[0].start:hits[0].end].startswith(("1111 9999", "9999"))
    cands = arms.candidates(t, hits)
    assert [t[s.start:s.end] for c in cands for s in c.segments][0].startswith("4111 1111 1111")


def test_candidates_skip_what_an_earlier_layer_masked_and_short_numbers():
    t = "**Me:** 123-45-6789, 48213377 and 12."
    hits = arms.rules(t)
    cands = arms.candidates(t, hits)
    assert [t[c.segments[0].start:c.segments[-1].end] for c in cands] == ["48213377"]
