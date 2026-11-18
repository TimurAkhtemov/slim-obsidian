"""Enrichment's schema: what the model may answer is the vocabulary code already holds."""
from slim.capture import enrich

REGISTRY = {
    "beacon": {"name": "Beacon"},
    "school": {"name": "School"},
}


def test_the_project_enum_is_exactly_the_registry_plus_none():
    """Same drift guard as the type vocabulary below: the candidates the model may choose from
    are the registry itself, not a hand-maintained copy of it."""
    assert enrich._schema(REGISTRY)["properties"]["project"]["enum"] == \
        [*sorted(REGISTRY), enrich.NO_PROJECT]


def test_an_invented_project_is_dropped_rather_than_filed():
    """The grammar enum makes the closed set structural, and `registered_project` is the
    redundant half: what survives a grammar that silently degrades."""
    raw = {"title": "Kernels", "type_tag": "lecture", "topics": [], "project": "not-a-project"}
    assert enrich.normalize(raw, "fallback", REGISTRY)["project"] == ""
    assert enrich.normalize({**raw, "project": "School"}, "fallback", REGISTRY)["project"] \
        == "school"


def test_journal_is_not_a_type_the_labeller_can_choose():
    """The journal route is gone, so the label that fed it is too: the grammar enum cannot
    offer it, and one that arrives anyway degrades like any unknown type."""
    from slim.capture import enrich
    assert "journal" not in enrich.TYPE_TAGS
    assert enrich.normalize({"title": "A walk", "type_tag": "journal"}, "x")["type_tag"] == ""
