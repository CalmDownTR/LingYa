from __future__ import annotations

import pytest

from lingya.persona.bucketing import map_formality, map_warmth


class TestMapWarmth:
    @pytest.mark.parametrize(
        "value,expected_keywords",
        [
            (0, "极高的人际边界"),
            (20, "极高的人际边界"),
            (21, "熟人间的安全距离"),
            (35, "熟人间的安全距离"),
            (50, "熟人间的安全距离"),
            (51, "开放和建设性的互动意图"),
            (65, "开放和建设性的互动意图"),
            (80, "开放和建设性的互动意图"),
            (81, "彻底打破人际边界"),
            (100, "彻底打破人际边界"),
        ],
    )
    def test_range_mapping(self, value, expected_keywords):
        result = map_warmth(value)
        assert expected_keywords in result, f"warmth={value}: expected '{expected_keywords}' in result"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            map_warmth(-1)
        with pytest.raises(ValueError):
            map_warmth(101)


class TestMapFormality:
    @pytest.mark.parametrize(
        "value,expected_keywords",
        [
            (0, "网络口语体"),
            (30, "网络口语体"),
            (31, "日常面对面交流"),
            (50, "日常面对面交流"),
            (70, "日常面对面交流"),
            (71, "高度结构化的书面表达"),
            (100, "高度结构化的书面表达"),
        ],
    )
    def test_range_mapping(self, value, expected_keywords):
        result = map_formality(value)
        assert expected_keywords in result, f"formality={value}: expected '{expected_keywords}' in result"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            map_formality(-1)
        with pytest.raises(ValueError):
            map_formality(101)
