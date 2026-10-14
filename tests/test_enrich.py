"""Enrichment's schema: what the model may answer is the vocabulary code already holds."""
from slim import enrich, voicetags

REGISTRY = {
    "beacon": {"name": "Beacon"},
    "school": {"name": "School"},
}


def test_the_project_enum_is_exactly_the_registry_plus_none():
    """Same drift guard as the type vocabulary below: the candidates the model may choose from
    are the registry itself, not a hand-maintained copy of it."""
    assert enrich._schema(REGISTRY)["properties"]["project"]["enum"] == \
        [*sorted(REGISTRY), enrich.NO_PROJECT]


def test_the_inferred_and_spoken_vocabularies_are_literally_the_same_list():
    """enrich-v1 claimed this and shipped a different list, which is why the model's inferred
    type sat in frontmatter unusable for two weeks. Pin it so they cannot drift again."""
    assert enrich.TYPE_TAGS == list(voicetags.ROUTING_TYPES)
