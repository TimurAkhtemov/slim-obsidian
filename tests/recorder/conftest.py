"""Every recorder test starts from the default mask list, whatever the local projects.yaml says;
a test that needs another list sets it."""
import pytest


@pytest.fixture(autouse=True)
def _default_mask_list(monkeypatch):
    from slim import config
    monkeypatch.setattr(config, "PII_MASK", config.PII_DEFAULT)
