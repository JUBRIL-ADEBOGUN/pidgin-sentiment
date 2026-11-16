import re

import emoji
import ftfy

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_pidgin_social_text(text: str, demojize: bool = False) -> str:
    """
    Lightweight, deterministic normalization used identically in training and serving.
    Preserves Pidgin slang, negation, and emojis.
    """
    text = ftfy.fix_text(text)
    text = URL_PATTERN.sub("[URL]", text)
    text = MENTION_PATTERN.sub("[USER]", text)
    text = ELONGATION_PATTERN.sub(r"\1\1", text)

    if demojize:
        text = emoji.demojize(text, delimiters=(" :", ": "))

    return WHITESPACE_PATTERN.sub(" ", text).strip()
