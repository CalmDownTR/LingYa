from __future__ import annotations

import pytest

from lingya.ingestion.chunker import count_tokens, chunk_text


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_non_empty(self):
        assert count_tokens("hello world") > 0

    def test_chinese_text(self):
        assert count_tokens("你好世界") > 0


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        result = chunk_text("A short sentence.", chunk_size=100)
        assert len(result) == 1
        assert result[0] == "A short sentence."

    def test_long_text_splits_on_separators(self):
        # Generate text that exceeds chunk_size
        paragraph = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. " * 20
        result = chunk_text(paragraph, chunk_size=200)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) > 0

    def test_empty_text(self):
        result = chunk_text("")
        assert result == []

    def test_whitespace_only(self):
        result = chunk_text("   \n\n  ")
        assert result == []

    def test_chinese_text_chunking(self):
        text = "这是第一段。这是第二段。这是第三段。" * 50
        result = chunk_text(text, chunk_size=200)
        assert len(result) > 1
        assert all(isinstance(c, str) and len(c) > 0 for c in result)

    def test_text_exactly_at_chunk_size_boundary(self):
        # Short text under the limit
        text = "hello " * 10
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1

    def test_no_separator_match_force_splits(self):
        # A long continuous string without separators
        text = "abc" * 500
        result = chunk_text(text, chunk_size=100, separators=["\n\n", "\n"])
        assert len(result) > 1
        # Each chunk should be non-empty
        for chunk in result:
            assert len(chunk) > 0

    def test_custom_separators(self):
        text = "part1|part2|part3" * 50
        result = chunk_text(text, chunk_size=100, separators=["|", " "])
        assert len(result) > 1

    def test_chunk_overlap(self):
        text = "sentence one. sentence two. sentence three. " * 30
        result = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(result) > 1
