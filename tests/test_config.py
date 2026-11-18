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


def test_pii_mask_names_are_tidied_and_a_bare_string_is_one_class():
    assert config._pii_mask({"pii": {"mask": " SSN "}}) == ("ssn",)
    assert config._pii_mask({"pii": {"mask": ["Email", "PHONE ", "email"]}}) == ("email", "phone")
    assert config._pii_mask({"pii": {"mask": ""}}) == ()


def test_an_unknown_pii_class_is_an_error_naming_it_and_the_allowed_ones():
    import pytest
    for bad, name in (({"pii": {"mask": ["SSN", "Emails"]}}, "emails"), ({"pii": {"mask": "socials"}}, "socials"),
                      ({"pii": {"mask": {"ssn": True}}}, "ssn")):
        with pytest.raises(ValueError) as err:
            config._pii_mask(bad)
        assert name in str(err.value) and "address" in str(err.value), bad


def test_the_allowed_classes_are_the_maskers_classes():
    from slim.capture import pii
    assert set(config.PII_CLASSES) == set(pii.PLACEHOLDER) and set(config.PII_DEFAULT) <= set(config.PII_CLASSES)
