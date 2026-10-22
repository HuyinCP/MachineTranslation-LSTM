"""Tokenization shared by preprocessing and sentence-level inference."""

import re

from underthesea import word_tokenize


def tokenize_en(sentence: str) -> list[str]:
    """Match the English tokenizer used to build the saved vocabulary."""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([.,!?;:])", r" \1 ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip().split()


def tokenize_vi(sentence: str) -> list[str]:
    """Match the Vietnamese word segmentation used during preprocessing."""
    return word_tokenize(sentence.lower().strip())


def join_vi_tokens(tokens: list[str]) -> str:
    """Join segmented Vietnamese tokens for an easily readable sentence."""
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens)).strip()
