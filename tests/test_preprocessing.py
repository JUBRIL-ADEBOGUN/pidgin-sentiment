from pidgin_senti.preprocessing import normalize_pidgin_social_text


def test_masks_urls_and_mentions() -> None:
    raw = "Abeg @MTNNG check https://example.com/help network no dey"
    cleaned = normalize_pidgin_social_text(raw)
    assert cleaned == "Abeg [USER] check [URL] network no dey"


def test_preserves_emojis_and_pidgin_negation() -> None:
    raw = "Dis app no dey work at all 😭💔"
    cleaned = normalize_pidgin_social_text(raw)
    assert "😭" in cleaned
    assert "no dey work" in cleaned


def test_collapses_extreme_elongations() -> None:
    raw = "Abegggggg help meeeeee"
    cleaned = normalize_pidgin_social_text(raw)
    assert cleaned == "Abegg help mee"


def test_optional_demojize() -> None:
    raw = "E sweet well well 🔥"
    cleaned = normalize_pidgin_social_text(raw, demojize=True)
    assert ":fire:" in cleaned
