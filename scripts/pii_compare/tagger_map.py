"""The tagger's labels as the classes the comparison scores."""

_KIND = {"ssn": "ssn", "credit card": "card", "account number": "account", "bank account": "account",
         "routing number": "account", "phone number": "phone"}


def kind_of(label: str) -> str:
    return _KIND.get(label, "none")
