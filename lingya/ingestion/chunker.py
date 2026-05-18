from __future__ import annotations

import tiktoken

# cl100k_base is used by most modern models (GPT-4, DeepSeek, etc.)
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", " ", ""]

    chunks: list[str] = []
    _split_recursive(text, separators, 0, chunk_size, chunk_overlap, chunks)
    return chunks


def _split_recursive(
    text: str,
    separators: list[str],
    sep_idx: int,
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[str],
) -> None:
    if count_tokens(text) <= chunk_size:
        if text.strip():
            chunks.append(text.strip())
        return

    if sep_idx >= len(separators):
        # Force-split by token count
        tokens = _ENCODER.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk = _ENCODER.decode(chunk_tokens).strip()
            if chunk:
                chunks.append(chunk)
        return

    separator = separators[sep_idx]
    if separator == "":
        _split_recursive(text, separators, sep_idx + 1, chunk_size, chunk_overlap, chunks)
        return

    splits = text.split(separator)
    current = ""
    for part in splits:
        candidate = current + separator + part if current else part
        if count_tokens(candidate) > chunk_size:
            if current.strip():
                _split_recursive(
                    current, separators, sep_idx + 1, chunk_size, chunk_overlap, chunks
                )
            current = part
        else:
            current = candidate

    if current.strip():
        _split_recursive(
            current, separators, sep_idx + 1, chunk_size, chunk_overlap, chunks
        )
