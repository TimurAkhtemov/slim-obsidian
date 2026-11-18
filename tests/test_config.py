from slim import config


def test_pii_mask_defaults_without_a_pii_key():
    assert config._pii_mask({}) == config.PII_DEFAULT
    assert "phone" not in config.PII_DEFAULT


def test_pii_mask_empty_list_turns_masking_off():
    assert config._pii_mask({"pii": {"mask": []}}) == ()


def test_pii_mask_custom_list():
    assert config._pii_mask({"pii": {"mask": ["email", "phone"]}}) == ("email", "phone")


def test_pii_skip_defaults_to_nothing():
    assert config._pii_skip({}) == ()
    assert config._pii_skip({"pii": {"mask": []}}) == ()


def test_pii_skip_lists_folder_prefixes():
    assert config._pii_skip({"pii": {"skip": ["Notes/some-folder/", " Journal "]}}) == \
        ("Notes/some-folder", "Journal")
    assert config._pii_skip({"pii": {"skip": "Notes/one"}}) == ("Notes/one",)
