from jevpoc.masking import contains_pii, mask, mask_text


def test_masks_identifiers_and_names():
    out = mask_text("Maria here, call 917-555-0142 or maria@x.com", ["Maria"])
    assert "917" not in out and "maria@" not in out.lower() and "Maria" not in out
    assert "<PHONE_" in out and "<EMAIL_" in out


def test_tokens_are_stable():
    assert mask_text("212-555-0199") == mask_text("212-555-0199")


def test_drops_identifier_keys_recursively():
    m = mask({"msisdn": "212-555-0199", "inner": {"account_id": "1", "ok": 2}})
    assert m == {"inner": {"ok": 2}}
    assert not contains_pii(m)
