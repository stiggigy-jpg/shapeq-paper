"""WikiText-2 → token chunks.

Calibration comes from the train split, perplexity is computed on test: the token sets
are disjoint by construction, which is exactly what the tests verify.
"""
from __future__ import annotations

import numpy as np

_SPLITS: dict[str, object] = {}


def dataset_rows(split: str):
    from datasets import load_dataset
    if split not in _SPLITS:
        _SPLITS[split] = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return _SPLITS[split]


def dataset_text(split: str) -> str:
    return "\n".join(t for t in dataset_rows(split)["text"] if t.strip())


def encode_chunks(tokenizer, text: str, seq_len: int) -> np.ndarray:
    """Tokenize and slice into full chunks; the incomplete tail is dropped."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // seq_len
    if n == 0:
        raise ValueError("text shorter than seq_len")
    return np.asarray(ids[:n * seq_len], dtype=np.int64).reshape(n, seq_len)


def _first_n(chunks: np.ndarray, n: int) -> np.ndarray:
    if chunks.shape[0] < n:
        raise ValueError(f"requested {n} chunks, available {chunks.shape[0]}")
    return chunks[:n]


def calibration_chunks(tokenizer, n: int, seq_len: int, split: str = "train") -> np.ndarray:
    return _first_n(encode_chunks(tokenizer, dataset_text(split), seq_len), n)


def evaluation_chunks(tokenizer, n: int, seq_len: int, split: str = "test") -> np.ndarray:
    return _first_n(encode_chunks(tokenizer, dataset_text(split), seq_len), n)
